#!/usr/bin/env python3
"""
validate_v20_pskr.py — Validate V20 Golden Master against PSK Reporter live data

Validates the V20 production model against the PSKR fire hose
to measure real-world performance on independent data.

V20 outputs per-band Z-scores. To compare with observed PSKR SNR,
we denormalize using per-band WSPR norm constants from config.

Config-driven: all constants loaded from config_v20.json.
Architecture imported from train_common.py (no re-declaration).

Usage:
    python validate_v20_pskr.py [--limit 100000] [--mode FT8]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import clickhouse_connect

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
sys.path.insert(0, COMMON_DIR)

from train_common import IonisV12Gate, grid4_to_latlon

# ── Load Config ──────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_v20.json")
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL_PATH = os.path.join(SCRIPT_DIR, CONFIG["checkpoint"])
DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
INPUT_DIM = CONFIG["model"]["input_dim"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

CH_HOST = CONFIG["clickhouse"]["host"]
CH_PORT = CONFIG["clickhouse"]["port"]

BAND_TO_HZ = {int(k): v for k, v in CONFIG["band_to_hz"].items()}

# Build per-band WSPR norm constants from config: {band_id: {'wspr': (mean, std)}}
NORM_CONSTANTS = {}
for band_str, sources in CONFIG["norm_constants_per_band"].items():
    band_id = int(band_str)
    NORM_CONSTANTS[band_id] = {
        'wspr': (sources["wspr"]["mean"], sources["wspr"]["std"]),
    }

# Mode thresholds (dB) - band considered "open" if predicted SNR >= threshold
THRESHOLDS = {
    'FT8': -20.0,
    'FT4': -20.0,
    'WSPR': -28.0,
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# V16 reference values for comparison
V16_FT8_RECALL = 98.47
V16_OVERALL_RECALL = 84.14


# ── Local Utilities ──────────────────────────────────────────────────────────

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
    """Convert Z-scores to dB using per-band WSPR constants."""
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
    print(f"V20 Golden Master PSKR Validation")
    print(f"=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Mode: {mode}")
    print(f"Limit: {limit:,}")
    print()

    # Load model
    print(f"Loading V20 model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = IonisV12Gate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"]["gate_init_bias"],
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    print(f"  Checkpoint Pearson: {checkpoint.get('val_pearson', 'N/A')}")
    print(f"  Checkpoint RMSE: {checkpoint.get('val_rmse', 'N/A')}")

    # Use norm constants from checkpoint if available, else from config
    if 'norm_constants' in checkpoint:
        norm_constants = checkpoint['norm_constants']
        print(f"  Using norm constants from checkpoint")
    else:
        norm_constants = NORM_CONSTANTS
        print(f"  Using norm constants from config")
    print()

    # Connect to ClickHouse
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # Get latest solar data
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
    date_range_query = f"""
    SELECT
        formatDateTime(min(timestamp), '%Y-%m-%d %H:%M') as min_ts,
        formatDateTime(max(timestamp), '%Y-%m-%d %H:%M') as max_ts
    FROM pskr.bronze
    WHERE mode = '{mode}'
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

    # Summary with V16 comparison
    print("\nSUMMARY:")
    print(f"  V16 Reference: FT8 recall {V16_FT8_RECALL}%, overall recall {V16_OVERALL_RECALL}%")
    print()

    if pearson > 0.20 and recall > 0.75:
        print(f"  PASS: Pearson {pearson:+.4f} > +0.20, Recall {100*recall:.1f}% > 75%")
        print(f"  V20 validates on independent PSKR data.")
    else:
        print(f"  NEEDS REVIEW: Pearson {pearson:+.4f}, Recall {100*recall:.1f}%")
        if pearson <= 0.20:
            print(f"    - Pearson below threshold (+0.20)")
        if recall <= 0.75:
            print(f"    - Recall below threshold (75%)")

    # V16 comparison
    if mode == 'FT8':
        ft8_delta = 100*recall - V16_FT8_RECALL
        print(f"\n  V20 vs V16 FT8 recall: {ft8_delta:+.2f} pp ({100*recall:.2f}% vs {V16_FT8_RECALL}%)")


def main():
    parser = argparse.ArgumentParser(description='Validate V20 against PSKR data')
    parser.add_argument('--limit', type=int, default=100000, help='Number of spots to validate')
    parser.add_argument('--mode', default='FT8', help='Mode to validate (FT8, FT4, WSPR)')
    args = parser.parse_args()

    validate(limit=args.limit, mode=args.mode)


if __name__ == '__main__':
    main()
