#!/usr/bin/env python3
"""
validate_v16_step_i.py — V16 Contest vs VOACAP comparison on Step I paths

Queries 1M contest paths from validation.step_i_paths, runs V16 inference,
denormalizes to dB, applies mode thresholds, and reports recall.

V16 Contest: WSPR + RBN DXpedition + Contest Anchoring (+10 dB SSB, 0 dB RTTY).

Baselines:
- VOACAP: 75.82%
- IONIS V15: 86.89%

Target: SSB recall 85%+ (V15: 81.01%)
"""

import math
import os
import sys

import clickhouse_connect
import numpy as np
import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v16_contest.pth")

CH_HOST = "10.60.1.1"
CH_PORT = 8123

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Band ID to frequency mapping (for freq_log calculation)
FREQ_MHZ = {
    102: 1.8, 103: 3.5, 104: 5.3, 105: 7.0, 106: 10.1,
    107: 14.0, 108: 18.1, 109: 21.0, 110: 24.9, 111: 28.0,
}

# MHz to band ID
MHZ_TO_BAND = {
    1.8: 102, 3.5: 103, 5.3: 104, 7.0: 105, 10.1: 106,
    14.0: 107, 18.1: 108, 21.0: 109, 24.9: 110, 28.0: 111,
}

# WSPR normalization constants (mean, std) per band
NORM_CONSTANTS = {
    102: (-18.04, 6.9),   # 160m
    103: (-17.90, 6.9),   # 80m
    104: (-17.60, 7.1),   # 60m
    105: (-17.34, 6.6),   # 40m
    106: (-18.07, 6.5),   # 30m
    107: (-17.53, 6.7),   # 20m
    108: (-18.35, 7.0),   # 17m
    109: (-18.32, 6.6),   # 15m
    110: (-18.76, 6.6),   # 12m
    111: (-17.86, 6.5),   # 10m
}

# Mode thresholds (from Step I)
THRESHOLDS = {
    'CW': -22.0,
    'DG': -22.0,  # Digital
    'PH': -21.0,  # Phone/SSB
    'RY': -21.0,  # RTTY
}

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13
GATE_INIT_BIAS = -math.log(2.0)


# ── Model Architecture (must match training) ─────────────────────────────────

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


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz, month, hour_utc, ssn):
    """Build 13 features matching V13 training EXACTLY."""
    n = len(tx_lat)

    # Distance (haversine)
    lat1, lon1 = np.radians(tx_lat), np.radians(tx_lon)
    lat2, lon2 = np.radians(rx_lat), np.radians(rx_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))

    # Azimuth
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth = np.degrees(np.arctan2(y, x)) % 360

    # Midpoint
    midpoint_lat = (tx_lat + rx_lat) / 2.0
    midpoint_lon = (tx_lon + rx_lon) / 2.0

    # Frequency in Hz (training uses Hz, not MHz!)
    freq_hz = freq_mhz * 1_000_000.0

    # SFI from SSN (simplified: SFI ≈ 63 + 0.7*SSN)
    sfi = 63.0 + 0.7 * ssn

    # Kp penalty (assume Kp=2 for contest conditions)
    kp = 2.0
    kp_penalty = np.full(n, 1.0 - kp / 9.0, dtype=np.float32)

    # Stack features EXACTLY matching train_v13_combined.py
    features = np.column_stack([
        distance_km / 20000.0,                              # 0: distance
        np.log10(freq_hz) / 8.0,                            # 1: freq_log (Hz, /8)
        np.sin(2.0 * np.pi * hour_utc / 24.0),              # 2: hour_sin
        np.cos(2.0 * np.pi * hour_utc / 24.0),              # 3: hour_cos
        np.sin(2.0 * np.pi * azimuth / 360.0),              # 4: az_sin
        np.cos(2.0 * np.pi * azimuth / 360.0),              # 5: az_cos
        np.abs(tx_lat - rx_lat) / 180.0,                    # 6: lat_diff (abs!)
        midpoint_lat / 90.0,                                # 7: midpoint_lat
        np.sin(2.0 * np.pi * month / 12.0),                 # 8: season_sin
        np.cos(2.0 * np.pi * month / 12.0),                 # 9: season_cos
        np.cos(2.0 * np.pi * (hour_utc + midpoint_lon / 15.0) / 24.0),  # 10: day_night (continuous!)
        sfi / 300.0,                                        # 11: SFI (for sun sidecar)
        kp_penalty,                                         # 12: kp_penalty (for storm sidecar)
    ])

    return features.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  IONIS V16 Contest — Step I Validation")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    model = IonisV12Gate(dnn_dim=DNN_DIM, sidecar_hidden=8).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    # Architecture print removed
    print(f"  RMSE: {checkpoint.get('val_rmse', 0):.4f}σ")
    print(f"  Pearson: {checkpoint.get('val_pearson', 0):+.4f}")
    print()

    # Query paths from ClickHouse
    print(f"Querying paths from {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    query = """
    SELECT tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz,
           year, month, hour_utc, ssn, mode, threshold
    FROM validation.step_i_paths
    """

    result = client.query(query)
    rows = result.result_rows
    print(f"  Loaded {len(rows):,} paths")
    print()

    # Convert to arrays
    tx_lat = np.array([r[0] for r in rows], dtype=np.float32)
    tx_lon = np.array([r[1] for r in rows], dtype=np.float32)
    rx_lat = np.array([r[2] for r in rows], dtype=np.float32)
    rx_lon = np.array([r[3] for r in rows], dtype=np.float32)
    freq_mhz = np.array([r[4] for r in rows], dtype=np.float32)
    year = np.array([r[5] for r in rows], dtype=np.int32)
    month = np.array([r[6] for r in rows], dtype=np.int32)
    hour_utc = np.array([r[7] for r in rows], dtype=np.int32)
    ssn = np.array([r[8] for r in rows], dtype=np.float32)
    mode = np.array([r[9] for r in rows])
    threshold = np.array([r[10] for r in rows], dtype=np.float32)

    # Map freq_mhz to band IDs
    band_ids = np.array([MHZ_TO_BAND.get(f, 107) for f in freq_mhz], dtype=np.int32)

    # Engineer features
    print("Engineering features...")
    features = engineer_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz,
                                  month, hour_utc, ssn)
    print(f"  Feature matrix: {features.shape}")
    print(f"  Feature ranges (first 5):")
    for i in range(min(5, features.shape[1])):
        print(f"    [{i}]: min={features[:,i].min():.3f}, max={features[:,i].max():.3f}, mean={features[:,i].mean():.3f}")
    print()

    # Run inference in batches
    print("Running V13 inference...")
    batch_size = 50000
    n_samples = len(features)
    predictions_sigma = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = torch.tensor(features[i:end], dtype=torch.float32, device=DEVICE)
            pred = model(batch).cpu().numpy().flatten()
            predictions_sigma[i:end] = pred
            if (i // batch_size) % 5 == 0:
                print(f"  Processed {end:,} / {n_samples:,} ({100*end/n_samples:.1f}%)")

    print(f"  Inference complete")
    print(f"  Predictions (σ): min={predictions_sigma.min():.3f}, max={predictions_sigma.max():.3f}, mean={predictions_sigma.mean():.3f}")
    print()

    # Denormalize per band
    print("Denormalizing predictions to dB...")
    predictions_db = np.zeros_like(predictions_sigma)
    for band_id in NORM_CONSTANTS:
        mask = band_ids == band_id
        if mask.sum() > 0:
            mean, std = NORM_CONSTANTS[band_id]
            predictions_db[mask] = predictions_sigma[mask] * std + mean

    print(f"  Predictions (dB): min={predictions_db.min():.1f}, max={predictions_db.max():.1f}, mean={predictions_db.mean():.1f}")
    print(f"  Thresholds: min={threshold.min():.1f}, max={threshold.max():.1f}")
    print()

    # Apply thresholds
    print("Applying mode thresholds...")
    band_open = predictions_db >= threshold

    # Calculate recall
    total = len(band_open)
    open_count = band_open.sum()
    recall = 100.0 * open_count / total

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Total paths:     {total:,}")
    print(f"  Band open:       {open_count:,}")
    print(f"  V16 Recall:      {recall:.2f}%")
    print()
    print("  Baselines:")
    print(f"    VOACAP:        75.82%")
    print(f"    IONIS V15:     86.89%")
    print()

    delta_voacap = recall - 75.82
    delta_v15 = recall - 86.89
    print(f"  V16 vs VOACAP:   {delta_voacap:+.2f} pp")
    print(f"  V16 vs V15:      {delta_v15:+.2f} pp")
    print()

    # Breakdown by mode
    print("  Recall by Mode:")
    for m in ['CW', 'PH', 'RY', 'DG']:
        mask = mode == m
        if mask.sum() > 0:
            mode_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {m}:  {mode_recall:.2f}% ({mask.sum():,} paths)")
    print()

    # Breakdown by band
    print("  Recall by Band:")
    band_names = {102: '160m', 103: '80m', 105: '40m', 107: '20m', 109: '15m', 111: '10m'}
    for band_id, band_name in sorted(band_names.items()):
        mask = band_ids == band_id
        if mask.sum() > 0:
            band_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {band_name}:  {band_recall:.2f}% ({mask.sum():,} paths)")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
