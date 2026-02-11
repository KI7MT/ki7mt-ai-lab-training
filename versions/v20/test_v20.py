#!/usr/bin/env python3
"""
test_v20.py — IONIS V20 Golden Master Sensitivity Analysis

8 sweeps:
  1. SFI sweep (60->300) with dB delta and monotonicity
  2. Kp sweep (0->9) with storm cost in dB
  3. Band comparison (160m->10m) with per-band gate values
  4. Day/night comparison (0->21 UTC)
  5. Distance x latitude matrix (3 paths x 5 distances)
  6. SFI x Kp matrix (5x5 grid)
  7. Storm impact by geography (equatorial/mid-lat/polar)
  8. Physics summary with sidecar effects

Config-driven: all constants loaded from config_v20.json.
Architecture imported from train_common.py (no re-declaration).

V20 outputs in Z-normalized units (sigma). Reports both sigma and approximate dB.
"""

import json
import math
import os
import sys

import numpy as np
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
sys.path.insert(0, COMMON_DIR)

from train_common import IonisV12Gate, _gate_v16

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

BAND_TO_HZ = {int(k): v for k, v in CONFIG["band_to_hz"].items()}

SIGMA_TO_DB = 6.7  # Approximate conversion factor

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Reference Path ───────────────────────────────────────────────────────────

TX_GRID = 'FN31'
RX_GRID = 'JO21'


def grid_to_latlon(grid):
    g = grid.upper()
    lon = (ord(g[0]) - ord('A')) * 20.0 - 180.0 + int(g[2]) * 2.0 + 1.0
    lat = (ord(g[1]) - ord('A')) * 10.0 - 90.0 + int(g[3]) * 1.0 + 0.5
    return lat, lon


TX_LAT, TX_LON = grid_to_latlon(TX_GRID)
RX_LAT, RX_LON = grid_to_latlon(RX_GRID)
REF_DISTANCE = 5900.0
REF_AZIMUTH = 50.0
REF_FREQ_HZ = 14_097_100
REF_HOUR = 12
REF_MONTH = 6
REF_SFI = 150.0
REF_KP = 2.0

# V16 reference values for comparison
V16_KP_STORM = 3.445   # sigma
V16_SFI_BENEFIT = 0.478  # sigma
V16_PEARSON = 0.4873


# ── Feature Builder ──────────────────────────────────────────────────────────

def make_input(distance_km=REF_DISTANCE, freq_hz=REF_FREQ_HZ, hour=REF_HOUR,
               month=REF_MONTH, azimuth=REF_AZIMUTH,
               tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
               ssn=100.0, sfi=REF_SFI, kp=REF_KP):
    distance = distance_km / 20000.0
    freq_log = np.log10(freq_hz) / 8.0
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
    az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
    lat_diff = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    season_sin = np.sin(2.0 * np.pi * month / 12.0)
    season_cos = np.cos(2.0 * np.pi * month / 12.0)
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_h = hour + midpoint_lon / 15.0
    day_night_est = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    sfi_norm = sfi / 300.0
    kp_penalty = 1.0 - kp / 9.0

    return torch.tensor(
        [[distance, freq_log, hour_sin, hour_cos,
          az_sin, az_cos, lat_diff, midpoint_lat,
          season_sin, season_cos, day_night_est,
          sfi_norm, kp_penalty]],
        dtype=torch.float32, device=DEVICE,
    )


def predict(**kwargs):
    inputs = make_input(**kwargs)
    with torch.no_grad():
        return model(inputs).item()


def decompose(x):
    """Decompose forward pass into components."""
    x_deep = x[:, :DNN_DIM]
    x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
    x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
    with torch.no_grad():
        trunk_out = model.trunk(x_deep)
        base_snr = model.base_head(trunk_out)
        sun_logit = model.sun_scaler_head(trunk_out)
        storm_logit = model.storm_scaler_head(trunk_out)
        sun_gate = _gate_v16(sun_logit)
        storm_gate = _gate_v16(storm_logit)
        sun_raw = model.sun_sidecar(x_sfi)
        storm_raw = model.storm_sidecar(x_kp)
        sun_contrib = sun_gate * sun_raw
        storm_contrib = storm_gate * storm_raw
        predicted = base_snr + sun_contrib + storm_contrib
    return {
        'base_snr': base_snr.item(),
        'sun_gate': sun_gate.item(),
        'storm_gate': storm_gate.item(),
        'sun_raw': sun_raw.item(),
        'storm_raw': storm_raw.item(),
        'sun_contrib': sun_contrib.item(),
        'storm_contrib': storm_contrib.item(),
        'predicted': predicted.item(),
    }


# ── Load Model ───────────────────────────────────────────────────────────────

print(f"Loading {MODEL_PATH}...")
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

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total_params:,} params")
print(f"Phase: {checkpoint.get('phase', CONFIG['phase'])}")
print(f"Data sources: {checkpoint.get('data_sources', 'unknown')}")
print(f"Normalization: {checkpoint.get('normalization', CONFIG['data']['normalization'])}")
print(f"RMSE: {checkpoint.get('val_rmse', 0):.4f} sigma, Pearson: {checkpoint.get('val_pearson', 0):+.4f}")

print(f"\n{'='*70}")
print(f"  IONIS V20 Golden Master -- SNR Sensitivity Analysis")
print(f"  Reference path: {TX_GRID} -> {RX_GRID} ({REF_DISTANCE:.0f} km, 20m)")
print(f"  Output units: sigma (Z-normalized) -- multiply by {SIGMA_TO_DB} for approx dB")
print(f"{'='*70}")


# --- 1. SFI Sweep ---
print(f"\n{'='*70}")
print(f"  SFI (F10.7) Sweep ({TX_GRID}->{RX_GRID}, Kp=2)")
print(f"  {'-'*56}")
sfi_snrs = []
for sfi_val in [60, 80, 100, 120, 150, 180, 200, 250, 300]:
    snr = predict(sfi=float(sfi_val), kp=2.0)
    sfi_snrs.append(snr)
    print(f"  SFI {sfi_val:3d}  ->  {snr:+6.3f} sigma  ({snr*SIGMA_TO_DB:+5.1f} dB)")

sfi_delta = sfi_snrs[-1] - sfi_snrs[0]
print(f"\n  SFI 60->300 delta: {sfi_delta:+.3f} sigma ({sfi_delta*SIGMA_TO_DB:+.1f} dB)")
print(f"  V16 reference:     +{V16_SFI_BENEFIT:.3f} sigma")
if sfi_delta > 0:
    print(f"  CORRECT: Higher SFI improves SNR")
else:
    print(f"  WARNING: SFI inversion")


# --- 2. Kp Sweep ---
print(f"\n{'='*70}")
print(f"  Kp Sweep ({TX_GRID}->{RX_GRID}, SFI=150)")
print(f"  {'-'*56}")
kp_snrs = []
for kp_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    snr = predict(sfi=150.0, kp=float(kp_val))
    kp_snrs.append(snr)
    print(f"  Kp {kp_val}  ->  {snr:+6.3f} sigma  ({snr*SIGMA_TO_DB:+5.1f} dB)")

kp_delta = kp_snrs[0] - kp_snrs[-1]
print(f"\n  Kp 0->9 storm cost: {kp_delta:+.3f} sigma ({kp_delta*SIGMA_TO_DB:+.1f} dB)")
print(f"  V16 reference:      +{V16_KP_STORM:.3f} sigma")
if kp_delta > 0:
    print(f"  CORRECT: Higher Kp degrades SNR")
else:
    print(f"  WARNING: Kp inversion")


# --- 3. Band Comparison + Gate Values ---
print(f"\n{'='*70}")
print(f"  Band Comparison + Gate Values ({TX_GRID}->{RX_GRID}, 12 UTC)")
print(f"  {'-'*70}")
bands = [
    ('160m',  BAND_TO_HZ[102]), ('80m',  BAND_TO_HZ[103]),
    ('40m',   BAND_TO_HZ[105]), ('30m',  BAND_TO_HZ[106]),
    ('20m',   BAND_TO_HZ[107]), ('17m',  BAND_TO_HZ[108]),
    ('15m',   BAND_TO_HZ[109]), ('10m',  BAND_TO_HZ[111]),
]

print(f"  {'Band':>5s}  {'MHz':>8s}  {'SNR sig':>7s}  {'~dB':>6s}  "
      f"{'SunG':>6s}  {'StmG':>6s}")
print(f"  {'-'*50}")

for label, hz in bands:
    inp = make_input(freq_hz=hz)
    d = decompose(inp)
    print(f"  {label:>5s}  {hz/1e6:7.3f}  {d['predicted']:+6.3f}  "
          f"{d['predicted']*SIGMA_TO_DB:+5.1f}  "
          f"{d['sun_gate']:6.4f}  {d['storm_gate']:6.4f}")


# --- 4. Day/Night Comparison ---
print(f"\n{'='*70}")
print(f"  Day vs Night ({TX_GRID}->{RX_GRID}, June)")
print(f"  {'-'*56}")
for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
    snr = predict(hour=hour)
    print(f"  {hour:02d}:00 UTC  ->  {snr:+6.3f} sigma  ({snr*SIGMA_TO_DB:+5.1f} dB)")


# --- 5. Distance x Latitude Matrix ---
print(f"\n{'='*70}")
print(f"  Distance x Latitude Matrix (12 UTC, June)")
print(f"  {'-'*56}")

paths = {
    "Equatorial (5N)":  (5.0,  0.0,  5.0,  30.0),
    "Mid-lat (40N)":    (40.0, -73.0, 50.0, 1.0),
    "High-lat (65N)":   (65.0, -20.0, 65.0, 20.0),
}
distances = [1000, 3000, 5000, 8000, 12000]

header = f"  {'Path':<20s}"
for d in distances:
    header += f"  {d:>6d}km"
print(header)
print(f"  {'-'*56}")

for path_name, (tlat, tlon, rlat, rlon) in paths.items():
    row = f"  {path_name:<20s}"
    for d in distances:
        snr = predict(distance_km=d, tx_lat=tlat, tx_lon=tlon,
                      rx_lat=rlat, rx_lon=rlon)
        row += f"  {snr:+6.2f}"
    print(row)


# --- 6. SFI x Kp Matrix ---
print(f"\n{'='*70}")
print(f"  SFI x Kp Matrix ({TX_GRID}->{RX_GRID}, 20m)")
print(f"  Values in sigma (Z-normalized)")
print(f"  {'-'*56}")
sfi_kp_label = 'SFI\\Kp'
kp_header = f"  {sfi_kp_label:<10s}"
for kp_val in [0, 2, 4, 6, 8]:
    kp_header += f"  Kp={kp_val:d}  "
print(kp_header)
print(f"  {'-'*56}")
for sfi_val in [80, 120, 150, 200, 250]:
    row = f"  SFI {sfi_val:<5d}"
    for kp_val in [0, 2, 4, 6, 8]:
        snr = predict(sfi=float(sfi_val), kp=float(kp_val))
        row += f"  {snr:+6.2f} "
    print(row)


# --- 7. Storm Impact by Geography ---
print(f"\n{'='*70}")
print(f"  Storm Impact: Kp=0 vs Kp=9 by Geography")
print(f"  {'-'*60}")

geo_paths = {
    "Equatorial 20m":  dict(tx_lat=5.0, tx_lon=0.0, rx_lat=5.0, rx_lon=30.0,
                            distance_km=3000, freq_hz=14_097_100),
    "Mid-lat 20m":     dict(tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
                            distance_km=REF_DISTANCE, freq_hz=14_097_100),
    "Polar 20m":       dict(tx_lat=65.0, tx_lon=-20.0, rx_lat=65.0, rx_lon=20.0,
                            distance_km=5000, freq_hz=14_097_100),
}

print(f"  {'Path':<18s}  {'Kp0':>8s}  {'Kp9':>8s}  {'Drop sig':>8s}  {'~dB':>6s}")
print(f"  {'-'*55}")

for name, geo in geo_paths.items():
    inp_kp0 = make_input(sfi=150.0, kp=0.0, **geo)
    inp_kp9 = make_input(sfi=150.0, kp=9.0, **geo)
    d0 = decompose(inp_kp0)
    d9 = decompose(inp_kp9)
    drop = d0['predicted'] - d9['predicted']
    print(f"  {name:<18s}  {d0['predicted']:+7.3f}  {d9['predicted']:+7.3f}  "
          f"{drop:+7.3f}  {drop*SIGMA_TO_DB:+5.1f}")


# --- 8. Physics Summary ---
print(f"\n{'='*70}")
print(f"  V20 PHYSICS SUMMARY")
print(f"{'='*70}")

sfi_benefit = model.get_sun_effect(200.0/300.0, DEVICE) - model.get_sun_effect(70.0/300.0, DEVICE)
storm_cost = model.get_storm_effect(1.0, DEVICE) - model.get_storm_effect(0.0, DEVICE)

print(f"\n  Sun Sidecar (SFI 70->200):   {sfi_benefit:+.3f} sigma  ({sfi_benefit*SIGMA_TO_DB:+.1f} dB)")
print(f"  Storm Sidecar (Kp 0->9):     {storm_cost:+.3f} sigma  ({storm_cost*SIGMA_TO_DB:+.1f} dB)")
if sfi_benefit != 0:
    print(f"  Storm/Sun Ratio:            {abs(storm_cost/sfi_benefit):.1f}:1")

print(f"\n  SFI monotonicity: {'CORRECT' if sfi_benefit > 0 else 'INVERTED'}")
print(f"  Kp monotonicity:  {'CORRECT' if storm_cost > 0 else 'INVERTED'}")

print(f"\n  V16 Reference:")
print(f"    SFI benefit:    +{V16_SFI_BENEFIT:.3f} sigma")
print(f"    Kp storm cost:  +{V16_KP_STORM:.3f} sigma")
print(f"    Pearson:        +{V16_PEARSON:.4f}")

# Check against thresholds
kp_min = CONFIG["validation"]["kp_storm_min"]
sfi_min = CONFIG["validation"]["sfi_benefit_min"]
pearson_min = CONFIG["validation"]["pearson_min"]
val_pearson = checkpoint.get('val_pearson', 0)

print(f"\n  V20 vs Thresholds:")
print(f"    Kp storm cost >= {kp_min} sigma:   {'PASS' if storm_cost >= kp_min else 'BELOW'} ({storm_cost:+.3f})")
print(f"    SFI benefit >= {sfi_min} sigma:    {'PASS' if sfi_benefit >= sfi_min else 'BELOW'} ({sfi_benefit:+.3f})")
print(f"    Pearson >= {pearson_min}:          {'PASS' if val_pearson >= pearson_min else 'BELOW'} ({val_pearson:+.4f})")

print(f"\n{'='*70}")
