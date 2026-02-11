#!/usr/bin/env python3
"""
validate_v20.py — IONIS V20 Golden Master Step I Recall Validation

Queries 1M contest paths from validation.step_i_paths, runs V20 inference,
denormalizes Z-scores to dB using per-band WSPR norm constants,
applies mode thresholds, and reports recall.

V20 Golden Master: V16 replication in clean codebase.

Baselines:
- VOACAP:    75.82%
- IONIS V15: 86.89%

Config-driven: all constants loaded from config_v20.json.
Architecture imported from train_common.py (no re-declaration).
"""

import json
import os
import sys
import time

import clickhouse_connect
import numpy as np
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
sys.path.insert(0, COMMON_DIR)

from train_common import IonisV12Gate

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

# Build WSPR norm constants from config (band_id -> (mean, std))
NORM_CONSTANTS = {}
for band_str, sources in CONFIG["norm_constants_per_band"].items():
    band_id = int(band_str)
    NORM_CONSTANTS[band_id] = (sources["wspr"]["mean"], sources["wspr"]["std"])

# MHz to band ID mapping
MHZ_TO_BAND = {
    1.8: 102, 3.5: 103, 5.3: 104, 7.0: 105, 10.1: 106,
    14.0: 107, 18.1: 108, 21.0: 109, 24.9: 110, 28.0: 111,
}

# Mode thresholds (from Step I)
THRESHOLDS = {
    'CW': -22.0,
    'DG': -22.0,
    'PH': -21.0,
    'RY': -21.0,
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz, month, hour_utc, ssn):
    """Build 13 features matching training EXACTLY."""
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

    # Frequency in Hz
    freq_hz = freq_mhz * 1_000_000.0

    # SFI from SSN (simplified: SFI ~ 63 + 0.7*SSN)
    sfi = 63.0 + 0.7 * ssn

    # Kp penalty (assume Kp=2 for contest conditions)
    kp = 2.0
    kp_penalty = np.full(n, 1.0 - kp / 9.0, dtype=np.float32)

    features = np.column_stack([
        distance_km / 20000.0,                                          # 0: distance
        np.log10(freq_hz) / 8.0,                                       # 1: freq_log
        np.sin(2.0 * np.pi * hour_utc / 24.0),                         # 2: hour_sin
        np.cos(2.0 * np.pi * hour_utc / 24.0),                         # 3: hour_cos
        np.sin(2.0 * np.pi * azimuth / 360.0),                         # 4: az_sin
        np.cos(2.0 * np.pi * azimuth / 360.0),                         # 5: az_cos
        np.abs(tx_lat - rx_lat) / 180.0,                               # 6: lat_diff
        midpoint_lat / 90.0,                                           # 7: midpoint_lat
        np.sin(2.0 * np.pi * month / 12.0),                            # 8: season_sin
        np.cos(2.0 * np.pi * month / 12.0),                            # 9: season_cos
        np.cos(2.0 * np.pi * (hour_utc + midpoint_lon / 15.0) / 24.0), # 10: day_night
        sfi / 300.0,                                                   # 11: SFI
        kp_penalty,                                                    # 12: kp_penalty
    ])

    return features.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  IONIS V20 Golden Master -- Step I Validation")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    model = IonisV12Gate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"]["gate_init_bias"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Architecture: {checkpoint.get('architecture', CONFIG['model']['architecture'])}")
    print(f"  RMSE: {checkpoint.get('val_rmse', 0):.4f} sigma")
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

    t0 = time.perf_counter()
    result = client.query(query)
    rows = result.result_rows
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(rows):,} paths in {elapsed:.1f}s")
    client.close()
    print()

    # Convert to arrays
    tx_lat = np.array([r[0] for r in rows], dtype=np.float32)
    tx_lon = np.array([r[1] for r in rows], dtype=np.float32)
    rx_lat = np.array([r[2] for r in rows], dtype=np.float32)
    rx_lon = np.array([r[3] for r in rows], dtype=np.float32)
    freq_mhz = np.array([r[4] for r in rows], dtype=np.float32)
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
    print("Running V20 inference...")
    batch_size = 50000
    n_samples = len(features)
    predictions_sigma = np.zeros(n_samples, dtype=np.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = torch.tensor(features[i:end], dtype=torch.float32, device=DEVICE)
            pred = model(batch).cpu().numpy().flatten()
            predictions_sigma[i:end] = pred
            if (i // batch_size) % 5 == 0:
                print(f"  Processed {end:,} / {n_samples:,} ({100*end/n_samples:.1f}%)")

    elapsed = time.perf_counter() - t0
    print(f"  Inference complete in {elapsed:.1f}s ({n_samples/elapsed:.0f} paths/sec)")
    print(f"  Predictions (sigma): min={predictions_sigma.min():.3f}, max={predictions_sigma.max():.3f}, mean={predictions_sigma.mean():.3f}")
    print()

    # Denormalize per band using WSPR norm constants
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
    print(f"  V20 Recall:      {recall:.2f}%")
    print()
    print("  Baselines:")
    print(f"    VOACAP:        75.82%")
    print(f"    IONIS V15:     86.89%")
    print()

    delta_voacap = recall - 75.82
    delta_v15 = recall - 86.89
    print(f"  V20 vs VOACAP:   {delta_voacap:+.2f} pp")
    print(f"  V20 vs V15:      {delta_v15:+.2f} pp")
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
    band_names = {102: '160m', 103: '80m', 104: '60m', 105: '40m', 106: '30m',
                  107: '20m', 108: '17m', 109: '15m', 110: '12m', 111: '10m'}
    for band_id in sorted(band_names.keys()):
        band_name = band_names[band_id]
        mask = band_ids == band_id
        if mask.sum() > 0:
            band_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {band_name}:  {band_recall:.2f}% ({mask.sum():,} paths)")
    print()

    # PASS/FAIL summary
    print("=" * 70)
    v15_baseline = 86.89
    if recall >= v15_baseline:
        print(f"  PASS: V20 recall {recall:.2f}% >= V15 Diamond baseline ({v15_baseline}%)")
    else:
        print(f"  BELOW V15: V20 recall {recall:.2f}% < V15 Diamond baseline ({v15_baseline}%)")
        if recall >= 75.82:
            print(f"  Still above VOACAP ({75.82}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
