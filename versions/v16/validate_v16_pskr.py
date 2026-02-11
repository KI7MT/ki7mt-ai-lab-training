#!/usr/bin/env python3
"""
validate_v16_pskr.py — Validate V16 model against PSK Reporter live data

This script validates the V16 production model against the PSKR fire hose
to measure real-world performance on independent data.

The V16 model outputs per-band Z-scores. To compare with observed PSKR SNR,
we denormalize using V16's NORM_CONSTANTS (wspr scale, since FT8 maps to wspr).

Usage:
    python validate_v16_pskr.py [--limit 100000] [--mode FT8]
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import clickhouse_connect

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ionis_v16.pth")

CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

GATE_INIT_BIAS = -math.log(2.0)

# V16 per-source per-band normalization constants
# FT8 maps to 'wspr' scale (weak signal digital)
NORM_CONSTANTS = {
    102: {'wspr': (-18.04, 6.9), 'rbn': (11.86, 6.23)},   # 160m
    103: {'wspr': (-17.90, 6.9), 'rbn': (16.94, 6.71)},   # 80m
    104: {'wspr': (-17.60, 7.1), 'rbn': (16.91, 7.1)},    # 60m
    105: {'wspr': (-17.34, 6.6), 'rbn': (18.36, 6.6)},    # 40m
    106: {'wspr': (-18.07, 6.5), 'rbn': (16.96, 6.5)},    # 30m
    107: {'wspr': (-17.53, 6.7), 'rbn': (18.47, 6.7)},    # 20m
    108: {'wspr': (-18.35, 7.0), 'rbn': (16.66, 7.0)},    # 17m
    109: {'wspr': (-18.32, 6.6), 'rbn': (16.37, 6.6)},    # 15m
    110: {'wspr': (-18.76, 6.6), 'rbn': (15.07, 6.6)},    # 12m
    111: {'wspr': (-17.86, 6.5), 'rbn': (15.79, 6.5)},    # 10m
}

BAND_TO_HZ = {
    102:  1_836_600,   103:  3_568_600,   104:  5_287_200,
    105:  7_038_600,   106: 10_138_700,   107: 14_097_100,
    108: 18_104_600,   109: 21_094_600,   110: 24_924_600,
    111: 28_124_600,
}

# Mode thresholds (dB) - band considered "open" if predicted SNR >= threshold
THRESHOLDS = {
    'FT8': -20.0,
    'FT4': -20.0,
    'WSPR': -28.0,
}

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


# ── V16 Model Architecture (IonisV12Gate) ─────────────────────────────────────

class MonotonicMLP(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(nn.functional.linear(x, w1, self.fc1.bias))
        return nn.functional.linear(h, w2, self.fc2.bias)


def _gate(x):
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self._init_scaler_heads()

    def _init_scaler_heads(self):
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, GATE_INIT_BIAS)

    def forward(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)
        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)

    def get_sun_effect(self, sfi_normalized):
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty):
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()


# ── Grid Utilities ────────────────────────────────────────────────────────────

def grid4_to_latlon(g):
    """Convert a single 4-char Maidenhead grid to (lat, lon) centroid."""
    s = str(g).strip().rstrip('\x00').upper()
    m = GRID_RE.search(s)
    g4 = m.group(0) if m else 'JJ00'
    lon = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
    lat = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_azimuth(lat1, lon1, lat2, lon2):
    """Calculate initial bearing (azimuth) from point 1 to point 2."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features_batch(df, sfi, kp):
    """Compute features for a batch of PSKR spots."""
    n = len(df)
    X = np.zeros((n, INPUT_DIM), dtype=np.float32)

    # Parse grids
    tx_lats = np.zeros(n, dtype=np.float32)
    tx_lons = np.zeros(n, dtype=np.float32)
    rx_lats = np.zeros(n, dtype=np.float32)
    rx_lons = np.zeros(n, dtype=np.float32)

    for i, row in enumerate(df.itertuples()):
        tx_lats[i], tx_lons[i] = grid4_to_latlon(row.sender_grid)
        rx_lats[i], rx_lons[i] = grid4_to_latlon(row.receiver_grid)

    # Compute path parameters
    distance = haversine_km(tx_lats, tx_lons, rx_lats, rx_lons)
    azimuth = compute_azimuth(tx_lats, tx_lons, rx_lats, rx_lons)
    midpoint_lat = (tx_lats + rx_lats) / 2.0
    midpoint_lon = (tx_lons + rx_lons) / 2.0

    # Extract time
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)

    # Get frequency from band
    band = df['band'].values.astype(np.int32)
    freq_hz = np.array([BAND_TO_HZ.get(b, 14_097_100) for b in band], dtype=np.float32)

    # Build features
    X[:, 0] = distance / 20000.0
    X[:, 1] = np.log10(freq_hz) / 8.0
    X[:, 2] = np.sin(2.0 * np.pi * hour / 24.0)
    X[:, 3] = np.cos(2.0 * np.pi * hour / 24.0)
    X[:, 4] = np.sin(2.0 * np.pi * azimuth / 360.0)
    X[:, 5] = np.cos(2.0 * np.pi * azimuth / 360.0)
    X[:, 6] = np.abs(tx_lats - rx_lats) / 180.0
    X[:, 7] = midpoint_lat / 90.0
    X[:, 8] = np.sin(2.0 * np.pi * month / 12.0)
    X[:, 9] = np.cos(2.0 * np.pi * month / 12.0)
    X[:, 10] = np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)
    X[:, 11] = sfi / 300.0
    X[:, 12] = 1.0 - kp / 9.0

    return X, band


def denormalize_zscore(z_scores, bands, norm_constants):
    """Convert Z-scores to dB using V16's per-band WSPR constants."""
    snr_db = np.zeros_like(z_scores)
    for i, (z, band) in enumerate(zip(z_scores, bands)):
        if band in norm_constants:
            mean, std = norm_constants[band]['wspr']
        else:
            mean, std = -17.8, 6.7  # fallback
        snr_db[i] = z * std + mean
    return snr_db


# ── Validation ────────────────────────────────────────────────────────────────

def validate(limit=100000, mode='FT8'):
    """Run validation against PSKR data."""
    print(f"V16 PSKR Validation")
    print(f"=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Mode: {mode}")
    print(f"Limit: {limit:,}")
    print()

    # Load model
    print(f"Loading V16 model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = IonisV12Gate(dnn_dim=DNN_DIM, sidecar_hidden=8)
    model.load_state_dict(checkpoint['model_state'])  # V16 uses 'model_state' not 'model_state_dict'
    model.to(DEVICE)
    model.eval()
    print(f"  Checkpoint Pearson: {checkpoint.get('val_pearson', 'N/A')}")
    print(f"  Checkpoint RMSE: {checkpoint.get('val_rmse', 'N/A')}")

    # Load norm constants from checkpoint if available
    if 'norm_constants' in checkpoint:
        norm_constants = checkpoint['norm_constants']
        print(f"  Using norm constants from checkpoint")
    else:
        norm_constants = NORM_CONSTANTS
        print(f"  Using default norm constants")
    print()

    # Connect to ClickHouse
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # Get latest solar data (use max date in solar.bronze, which may lag behind)
    solar_query = """
    SELECT
        avg(adjusted_flux) as avg_sfi,
        avg(kp_index) as avg_kp
    FROM solar.bronze
    WHERE date = (SELECT max(date) FROM solar.bronze)
    """
    solar_result = client.query(solar_query)
    if solar_result.result_rows:
        sfi, kp = solar_result.result_rows[0]
        sfi = float(sfi) if sfi and not np.isnan(sfi) else 140.0
        kp = float(kp) if kp and not np.isnan(kp) else 2.0
    else:
        sfi, kp = 140.0, 2.0
    print(f"Solar conditions: SFI={sfi:.1f}, Kp={kp:.1f}")
    print()

    # Query PSKR spots
    print(f"Loading PSKR {mode} spots...")
    pskr_query = f"""
    SELECT
        timestamp,
        sender_grid,
        receiver_grid,
        band,
        snr,
        toHour(timestamp) as hour,
        toMonth(timestamp) as month
    FROM pskr.bronze
    WHERE mode = '{mode}'
      AND sender_grid != ''
      AND receiver_grid != ''
      AND length(sender_grid) >= 4
      AND length(receiver_grid) >= 4
      AND band IN (102, 103, 104, 105, 106, 107, 108, 109, 110, 111)
    ORDER BY rand()
    LIMIT {limit}
    """

    t0 = time.perf_counter()
    df = client.query_df(pskr_query)
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} spots in {elapsed:.1f}s")

    if len(df) == 0:
        print("No PSKR data found!")
        return

    # Get date range
    date_range_query = """
    SELECT
        formatDateTime(min(timestamp), '%Y-%m-%d %H:%M') as min_ts,
        formatDateTime(max(timestamp), '%Y-%m-%d %H:%M') as max_ts
    FROM pskr.bronze
    WHERE mode = 'FT8'
    """
    date_result = client.query(date_range_query)
    if date_result.result_rows:
        min_ts, max_ts = date_result.result_rows[0]
        print(f"  Date range: {min_ts} to {max_ts}")
    print()

    client.close()

    # Engineer features
    print("Engineering features...")
    X, bands = engineer_features_batch(df, sfi, kp)
    observed_snr = df['snr'].values.astype(np.float32)

    # Run predictions
    print("Running predictions...")
    t0 = time.perf_counter()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        z_scores = model(X_tensor).cpu().numpy().flatten()

    elapsed = time.perf_counter() - t0
    print(f"  {len(X):,} predictions in {elapsed:.2f}s ({len(X)/elapsed:.0f}/sec)")

    # Denormalize to dB
    predicted_snr = denormalize_zscore(z_scores, bands, norm_constants)

    # Compute metrics
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Overall metrics
    valid_mask = ~np.isnan(predicted_snr) & ~np.isnan(observed_snr)
    pred_valid = predicted_snr[valid_mask]
    obs_valid = observed_snr[valid_mask]

    pearson = np.corrcoef(pred_valid, obs_valid)[0, 1]
    rmse = np.sqrt(np.mean((pred_valid - obs_valid) ** 2))
    mae = np.mean(np.abs(pred_valid - obs_valid))
    bias = np.mean(pred_valid - obs_valid)

    print(f"\nOverall ({len(pred_valid):,} spots):")
    print(f"  Pearson r:  {pearson:+.4f}")
    print(f"  RMSE:       {rmse:.2f} dB")
    print(f"  MAE:        {mae:.2f} dB")
    print(f"  Bias:       {bias:+.2f} dB")

    # Recall (band-open detection)
    threshold = THRESHOLDS.get(mode, -20.0)
    observed_open = observed_snr >= threshold
    predicted_open = predicted_snr >= threshold

    true_pos = np.sum(observed_open & predicted_open)
    false_neg = np.sum(observed_open & ~predicted_open)
    false_pos = np.sum(~observed_open & predicted_open)
    true_neg = np.sum(~observed_open & ~predicted_open)

    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nRecall Analysis (threshold={threshold} dB):")
    print(f"  Observed open:   {np.sum(observed_open):,} ({100*np.mean(observed_open):.1f}%)")
    print(f"  Predicted open:  {np.sum(predicted_open):,} ({100*np.mean(predicted_open):.1f}%)")
    print(f"  True Positives:  {true_pos:,}")
    print(f"  False Negatives: {false_neg:,}")
    print(f"  Recall:          {100*recall:.2f}%")
    print(f"  Precision:       {100*precision:.2f}%")
    print(f"  F1 Score:        {100*f1:.2f}%")

    # Per-band breakdown
    print(f"\nPer-Band Breakdown:")
    print(f"{'Band':>6s}  {'Count':>8s}  {'Pearson':>8s}  {'RMSE':>6s}  {'Recall':>7s}")
    print("-" * 45)

    band_names = {102: '160m', 103: '80m', 104: '60m', 105: '40m', 106: '30m',
                  107: '20m', 108: '17m', 109: '15m', 110: '12m', 111: '10m'}

    for band_id in sorted(band_names.keys()):
        mask = bands == band_id
        if mask.sum() < 10:
            continue

        b_pred = predicted_snr[mask]
        b_obs = observed_snr[mask]
        b_pearson = np.corrcoef(b_pred, b_obs)[0, 1] if len(b_pred) > 1 else 0
        b_rmse = np.sqrt(np.mean((b_pred - b_obs) ** 2))

        b_obs_open = b_obs >= threshold
        b_pred_open = b_pred >= threshold
        b_recall = np.sum(b_obs_open & b_pred_open) / max(np.sum(b_obs_open), 1)

        print(f"{band_names[band_id]:>6s}  {mask.sum():>8,}  {b_pearson:>+8.4f}  {b_rmse:>6.2f}  {100*b_recall:>6.1f}%")

    print()
    print("=" * 60)

    # Summary
    print("\nSUMMARY:")
    if pearson > 0.20 and recall > 0.75:
        print(f"  PASS: Pearson {pearson:+.4f} > +0.20, Recall {100*recall:.1f}% > 75%")
        print(f"  V16 validates on independent PSKR data.")
    else:
        print(f"  NEEDS REVIEW: Pearson {pearson:+.4f}, Recall {100*recall:.1f}%")
        if pearson <= 0.20:
            print(f"    - Pearson below threshold (+0.20)")
        if recall <= 0.75:
            print(f"    - Recall below threshold (75%)")


def main():
    parser = argparse.ArgumentParser(description='Validate V16 against PSKR data')
    parser.add_argument('--limit', type=int, default=100000, help='Number of spots to validate')
    parser.add_argument('--mode', default='FT8', help='Mode to validate (FT8, FT4, WSPR)')
    args = parser.parse_args()

    validate(limit=args.limit, mode=args.mode)


if __name__ == '__main__':
    main()
