#!/usr/bin/env python3
"""
test_v13_combined.py — IONIS V13 Combined Sensitivity Analysis

V13 outputs in Z-normalized units (σ). Reports both σ and approximate dB.
Average std ≈ 6.7 dB across sources, so multiply σ by 6.7 for dB estimate.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v13_combined.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

GATE_INIT_BIAS = -math.log(2.0)
SIGMA_TO_DB = 6.7  # Approximate conversion factor

BAND_TO_HZ = {
    102:  1_836_600,   103:  3_568_600,   104:  5_287_200,
    105:  7_038_600,   106: 10_138_700,   107: 14_097_100,
    108: 18_104_600,   109: 21_094_600,   110: 24_924_600,
    111: 28_124_600,
}


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

    def decompose(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            base_snr = self.base_head(trunk_out)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
            sun_gate = _gate(sun_logit)
            storm_gate = _gate(storm_logit)
            sun_raw = self.sun_sidecar(x_sfi)
            storm_raw = self.storm_sidecar(x_kp)
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


def grid_to_latlon(grid):
    g = grid.upper()
    lon = (ord(g[0]) - ord('A')) * 20.0 - 180.0 + int(g[2]) * 2.0 + 1.0
    lat = (ord(g[1]) - ord('A')) * 10.0 - 90.0 + int(g[3]) * 1.0 + 0.5
    return lat, lon


TX_GRID = 'FN31'
RX_GRID = 'JO21'
TX_LAT, TX_LON = grid_to_latlon(TX_GRID)
RX_LAT, RX_LON = grid_to_latlon(RX_GRID)
REF_DISTANCE = 5900.0
REF_AZIMUTH = 50.0
REF_FREQ_HZ = 14_097_100
REF_HOUR = 12
REF_MONTH = 6
REF_SFI = 150.0
REF_KP = 2.0


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


print(f"Loading {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
dnn_dim = checkpoint.get('dnn_dim', DNN_DIM)
sidecar_hidden = checkpoint.get('sidecar_hidden', 8)
model = IonisV12Gate(dnn_dim=dnn_dim, sidecar_hidden=sidecar_hidden).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total_params:,} params")
print(f"Phase: {checkpoint.get('phase', 'unknown')}")
print(f"Data sources: {checkpoint.get('data_sources', 'unknown')}")
print(f"Normalization: {checkpoint.get('normalization', 'unknown')}")
print(f"RMSE: {checkpoint.get('val_rmse', 0):.4f}σ, Pearson: {checkpoint.get('val_pearson', 0):+.4f}")

print(f"\n{'='*70}")
print(f"  IONIS V13 Combined — SNR Sensitivity Analysis")
print(f"  Reference path: {TX_GRID} → {RX_GRID} ({REF_DISTANCE:.0f} km, 20m)")
print(f"  Output units: σ (Z-normalized) — multiply by {SIGMA_TO_DB} for approx dB")
print(f"{'='*70}")


# --- 1. SFI Sweep ---
print(f"\n{'='*70}")
print(f"  SFI (F10.7) Sweep ({TX_GRID}→{RX_GRID}, Kp=2)")
print(f"  {'-'*56}")
sfi_snrs = []
for sfi_val in [60, 80, 100, 120, 150, 180, 200, 250, 300]:
    snr = predict(sfi=float(sfi_val), kp=2.0)
    sfi_snrs.append(snr)
    print(f"  SFI {sfi_val:3d}  →  {snr:+6.3f}σ  ({snr*SIGMA_TO_DB:+5.1f} dB)")

sfi_delta = sfi_snrs[-1] - sfi_snrs[0]
print(f"\n  SFI 60→300 delta: {sfi_delta:+.3f}σ ({sfi_delta*SIGMA_TO_DB:+.1f} dB)")
if sfi_delta > 0:
    print(f"  CORRECT: Higher SFI improves SNR")
else:
    print(f"  WARNING: SFI inversion")


# --- 2. Kp Sweep ---
print(f"\n{'='*70}")
print(f"  Kp Sweep ({TX_GRID}→{RX_GRID}, SFI=150)")
print(f"  {'-'*56}")
kp_snrs = []
for kp_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    snr = predict(sfi=150.0, kp=float(kp_val))
    kp_snrs.append(snr)
    print(f"  Kp {kp_val}  →  {snr:+6.3f}σ  ({snr*SIGMA_TO_DB:+5.1f} dB)")

kp_delta = kp_snrs[0] - kp_snrs[-1]
print(f"\n  Kp 0→9 storm cost: {kp_delta:+.3f}σ ({kp_delta*SIGMA_TO_DB:+.1f} dB)")
if kp_delta > 0:
    print(f"  CORRECT: Higher Kp degrades SNR")
else:
    print(f"  WARNING: Kp inversion")


# --- 3. Band Comparison + Gate Values ---
print(f"\n{'='*70}")
print(f"  Band Comparison + Gate Values ({TX_GRID}→{RX_GRID}, 12 UTC)")
print(f"  {'-'*70}")
bands = [
    ('160m',  1_836_600), ('80m',  3_568_600), ('40m',  7_038_600),
    ('30m', 10_138_700),  ('20m', 14_097_100), ('17m', 18_104_600),
    ('15m', 21_094_600),  ('10m', 28_124_600),
]

print(f"  {'Band':>5s}  {'MHz':>8s}  {'SNR σ':>7s}  {'~dB':>6s}  "
      f"{'SunG':>6s}  {'StmG':>6s}")
print(f"  {'-'*50}")

for label, hz in bands:
    inp = make_input(freq_hz=hz)
    d = model.decompose(inp)
    print(f"  {label:>5s}  {hz/1e6:7.3f}  {d['predicted']:+6.3f}  "
          f"{d['predicted']*SIGMA_TO_DB:+5.1f}  "
          f"{d['sun_gate']:6.4f}  {d['storm_gate']:6.4f}")


# --- 4. Day/Night Comparison ---
print(f"\n{'='*70}")
print(f"  Day vs Night ({TX_GRID}→{RX_GRID}, June)")
print(f"  {'-'*56}")
for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
    snr = predict(hour=hour)
    print(f"  {hour:02d}:00 UTC  →  {snr:+6.3f}σ  ({snr*SIGMA_TO_DB:+5.1f} dB)")


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
print(f"  SFI x Kp Matrix ({TX_GRID}→{RX_GRID}, 20m)")
print(f"  Values in σ (Z-normalized)")
print(f"  {'-'*56}")
kp_header = f"  {'SFI\\Kp':<10s}"
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

print(f"  {'Path':<18s}  {'Kp0':>8s}  {'Kp9':>8s}  {'Drop σ':>8s}  {'~dB':>6s}")
print(f"  {'-'*55}")

for name, geo in geo_paths.items():
    inp_kp0 = make_input(sfi=150.0, kp=0.0, **geo)
    inp_kp9 = make_input(sfi=150.0, kp=9.0, **geo)
    d0 = model.decompose(inp_kp0)
    d9 = model.decompose(inp_kp9)
    drop = d0['predicted'] - d9['predicted']
    print(f"  {name:<18s}  {d0['predicted']:+7.3f}  {d9['predicted']:+7.3f}  "
          f"{drop:+7.3f}  {drop*SIGMA_TO_DB:+5.1f}")


# --- 8. Physics Summary ---
print(f"\n{'='*70}")
print(f"  V13 PHYSICS SUMMARY")
print(f"{'='*70}")

sfi_benefit = model.get_sun_effect(200.0/300.0) - model.get_sun_effect(70.0/300.0)
storm_cost = model.get_storm_effect(1.0) - model.get_storm_effect(0.0)

print(f"\n  Sun Sidecar (SFI 70→200):   {sfi_benefit:+.3f}σ  ({sfi_benefit*SIGMA_TO_DB:+.1f} dB)")
print(f"  Storm Sidecar (Kp 0→9):     {storm_cost:+.3f}σ  ({storm_cost*SIGMA_TO_DB:+.1f} dB)")
print(f"  Storm/Sun Ratio:            {abs(storm_cost/sfi_benefit):.1f}:1")

print(f"\n  SFI monotonicity: {'CORRECT' if sfi_benefit > 0 else 'INVERTED'}")
print(f"  Kp monotonicity:  {'CORRECT' if storm_cost > 0 else 'INVERTED'}")

print(f"\n{'='*70}")
