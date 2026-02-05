#!/usr/bin/env python3
"""
test_v11_final.py — IONIS V11 Gatekeeper Sensitivity Analysis

Loads the V11 final checkpoint and runs the same comprehensive sweeps as V10,
plus gate decomposition to show how the model modulates sidecar outputs
by geography and frequency.

Sweeps:
  1. Solar condition scenarios (SSN co-varied)
  2. Co-varied SSN sweep (SFI = 67 + 0.7*SSN)
  3. SFI sweep (Sun Sidecar test)
  4. Kp sweep (Storm Sidecar test)
  5. Distance x Latitude matrix
  6. SSN impact by latitude
  7. Band comparison + gate values
  8. Day/Night comparison
  9. SFI x Kp matrix
 10. Gate decomposition table (NEW in V11)
"""

import math
import os
import re

import numpy as np
import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v11_final.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

GATE_INIT_BIAS = -math.log(2.0)

BAND_TO_HZ = {
    102:  1_836_600,   103:  3_568_600,   104:  5_287_200,
    105:  7_038_600,   106: 10_138_700,   107: 14_097_100,
    108: 18_104_600,   109: 21_094_600,   110: 24_924_600,
    111: 28_124_600,
}


# ── Model (must match training) ─────────────────────────────────────────────

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


class IonisV11Gate(nn.Module):
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
        """Full forward decomposition: returns dict of all intermediate values."""
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


# ── Grid + Reference Path ───────────────────────────────────────────────────

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


# ── Helper ───────────────────────────────────────────────────────────────────

def make_input(distance_km=REF_DISTANCE, freq_hz=REF_FREQ_HZ, hour=REF_HOUR,
               month=REF_MONTH, azimuth=REF_AZIMUTH,
               tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
               ssn=100.0, sfi=REF_SFI, kp=REF_KP):
    """Build 13-feature vector: 11 trunk + 1 sfi + 1 kp_penalty."""
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


# ── Load Model ───────────────────────────────────────────────────────────────

print(f"Loading {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
dnn_dim = checkpoint.get('dnn_dim', DNN_DIM)
sidecar_hidden = checkpoint.get('sidecar_hidden', 8)
model = IonisV11Gate(dnn_dim=dnn_dim, sidecar_hidden=sidecar_hidden).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total_params:,} params")
print(f"Architecture: {checkpoint.get('architecture', 'unknown')}")
print(f"Features: {checkpoint['features']}")
print(f"Trained on: {checkpoint.get('date_range', 'unknown')}, "
      f"{checkpoint.get('sample_size', 'unknown'):,} rows")
print(f"RMSE: {checkpoint.get('val_rmse', 0):.4f} dB, "
      f"Pearson: {checkpoint.get('val_pearson', 0):+.4f}")
print(f"Gate function: {checkpoint.get('gate_function', 'gate(x) = 0.5 + 1.5*sigmoid(x)')}")


# ═══════════════════════════════════════════════════════════════════════════
#  SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  IONIS V11 Gatekeeper — SNR Sensitivity Analysis")
print(f"  Reference path: {TX_GRID} → {RX_GRID} ({REF_DISTANCE:.0f} km, 20m WSPR)")
print(f"  Trunk: 11 features (no solar — Starvation Protocol)")
print(f"  Gates: [0.5, 2.0] — modulate sidecar outputs per geography")
print(f"  Reference: SFI={REF_SFI}, Kp={REF_KP}")
print(f"{'='*70}")


# --- 1. Solar Condition Scenarios ---
scenarios = {
    "Solar minimum (SSN 10) ": dict(ssn=10),
    "Quiet Sun     (SSN 50) ": dict(ssn=50),
    "Moderate      (SSN 100)": dict(ssn=100),
    "Active        (SSN 150)": dict(ssn=150),
    "Solar maximum (SSN 200)": dict(ssn=200),
}

print(f"\n  Solar Condition Scenarios (12:00 UTC, June, SFI=150 fixed):")
print(f"  {'-'*56}")
results = {}
for name, params in scenarios.items():
    snr = predict(**params)
    results[name] = snr
    print(f"  {name:44s} → {snr:+6.1f} dB")

keys = list(scenarios.keys())
delta = results[keys[-1]] - results[keys[0]]
print(f"\n  SSN 10→200 delta: {delta:+.1f} dB")
if abs(delta) < 0.01:
    print(f"  Note: SSN not a direct feature — effect comes through SFI sidecar")
else:
    print(f"  Note: SSN changes propagation geometry interaction")


# --- 2. Co-Varied SSN Sweep ---
print(f"\n{'='*70}")
print(f"  Co-Varied SSN Sweep (SFI = 67 + 0.7*SSN, Kp from SSN)")
print(f"  {'-'*56}")
ssn_values = np.arange(0, 310, 20)
covar_snrs = []
for ssn_val in ssn_values:
    sfi_val = 67.0 + 0.7 * ssn_val
    kp_val = max(1.0, min(5.0, 1.0 + ssn_val / 100.0))
    snr = predict(ssn=float(ssn_val), sfi=sfi_val, kp=kp_val)
    covar_snrs.append(snr)
    print(f"  SSN {ssn_val:3.0f}  SFI {sfi_val:5.0f}  Kp {kp_val:.1f}  →  {snr:+6.1f} dB")
print(f"\n  SSN 0→300 co-varied delta: {covar_snrs[-1] - covar_snrs[0]:+.1f} dB")
if covar_snrs[-1] > covar_snrs[0]:
    print(f"  CORRECT: Monotonic improvement with realistic solar conditions")
else:
    print(f"  WARNING: Still inverted under co-varied conditions")


# --- 3. SFI Sweep ---
print(f"\n{'='*70}")
print(f"  SFI (F10.7) Sweep ({TX_GRID}→{RX_GRID}, SSN=100, Kp=2)")
print(f"  Tests Sun Sidecar monotonicity + gate modulation")
print(f"  {'-'*56}")
sfi_snrs = []
for sfi_val in [60, 80, 100, 120, 150, 180, 200, 250, 300]:
    snr = predict(ssn=100, sfi=float(sfi_val), kp=2.0)
    sfi_snrs.append(snr)
    print(f"  SFI {sfi_val:3d}  →  {snr:+6.1f} dB")

sfi_delta = sfi_snrs[-1] - sfi_snrs[0]
print(f"\n  SFI 60→300 delta: {sfi_delta:+.1f} dB")
if sfi_delta > 0:
    print(f"  CORRECT: Higher SFI improves SNR (Sun Sidecar working)")
else:
    print(f"  WARNING: SFI inversion persists")


# --- 4. Kp Sweep ---
print(f"\n{'='*70}")
print(f"  Kp Sweep ({TX_GRID}→{RX_GRID}, SSN=100, SFI=150)")
print(f"  Tests Storm Sidecar monotonicity + gate modulation")
print(f"  {'-'*56}")
kp_snrs = []
for kp_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    snr = predict(ssn=100, sfi=150.0, kp=float(kp_val))
    kp_snrs.append(snr)
    print(f"  Kp {kp_val}  →  {snr:+6.1f} dB")

kp_delta = kp_snrs[0] - kp_snrs[-1]
print(f"\n  Kp 0→9 storm cost: {kp_delta:+.1f} dB")
if kp_delta > 0:
    print(f"  CORRECT: Higher Kp degrades SNR (Storm Sidecar working)")
else:
    print(f"  WARNING: Kp inversion persists")


# --- 5. Distance x Latitude Matrix ---
print(f"\n{'='*70}")
print(f"  Distance x Latitude Matrix (SSN=100, 12 UTC, June)")
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
                      rx_lat=rlat, rx_lon=rlon, ssn=100)
        row += f"  {snr:+6.1f}"
    print(row)


# --- 6. SSN Impact by Latitude ---
print(f"\n{'='*70}")
print(f"  SSN Impact by Latitude (5000 km, SSN 10 vs SSN 200)")
print(f"  {'-'*56}")

for path_name, (tlat, tlon, rlat, rlon) in paths.items():
    snr_low = predict(distance_km=5000, tx_lat=tlat, tx_lon=tlon,
                      rx_lat=rlat, rx_lon=rlon, ssn=10)
    snr_high = predict(distance_km=5000, tx_lat=tlat, tx_lon=tlon,
                       rx_lat=rlat, rx_lon=rlon, ssn=200)
    delta = snr_high - snr_low
    print(f"  {path_name:<20s}  SSN 10: {snr_low:+6.1f}  "
          f"SSN 200: {snr_high:+6.1f}  Delta: {delta:+5.1f} dB")


# --- 7. Band Comparison + Gate Values (V11-specific) ---
print(f"\n{'='*70}")
print(f"  Band Comparison + Gate Values ({TX_GRID}→{RX_GRID}, SSN=100, 12 UTC)")
print(f"  {'-'*70}")
bands = [
    ('160m',  1_836_600), ('80m',  3_568_600), ('40m',  7_038_600),
    ('30m', 10_138_700),  ('20m', 14_097_100), ('17m', 18_104_600),
    ('15m', 21_094_600),  ('10m', 28_124_600),
]

print(f"  {'Band':>5s}  {'MHz':>8s}  {'SNR':>7s}  {'Base':>7s}  "
      f"{'SunG':>6s}  {'StmG':>6s}  {'SunC':>7s}  {'StmC':>7s}")
print(f"  {'-'*62}")

for label, hz in bands:
    inp = make_input(freq_hz=hz, ssn=100)
    d = model.decompose(inp)
    print(f"  {label:>5s}  {hz/1e6:7.3f}  {d['predicted']:+6.1f}  "
          f"{d['base_snr']:+6.1f}  "
          f"{d['sun_gate']:6.4f}  {d['storm_gate']:6.4f}  "
          f"{d['sun_contrib']:+6.2f}  {d['storm_contrib']:+6.2f}")


# --- 8. Day/Night Comparison ---
print(f"\n{'='*70}")
print(f"  Day vs Night ({TX_GRID}→{RX_GRID}, SSN=100, June)")
print(f"  {'-'*56}")
for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
    snr = predict(hour=hour, ssn=100)
    print(f"  {hour:02d}:00 UTC  →  {snr:+6.1f} dB")


# --- 9. SFI x Kp Matrix ---
print(f"\n{'='*70}")
print(f"  SFI x Kp Matrix ({TX_GRID}→{RX_GRID}, SSN=100, 20m)")
print(f"  {'-'*56}")
kp_header = f"  {'SFI\\Kp':<10s}"
for kp_val in [0, 2, 4, 6, 8]:
    kp_header += f"  Kp={kp_val:d}  "
print(kp_header)
print(f"  {'-'*56}")
for sfi_val in [80, 120, 150, 200, 250]:
    row = f"  SFI {sfi_val:<5d}"
    for kp_val in [0, 2, 4, 6, 8]:
        snr = predict(ssn=100, sfi=float(sfi_val), kp=float(kp_val))
        row += f"  {snr:+6.1f} "
    print(row)


# --- 10. Gate Decomposition Table (V11-specific) ---
print(f"\n{'='*70}")
print(f"  Gate Decomposition — Path x Condition Matrix")
print(f"  Shows how gates modulate sidecar outputs by geography")
print(f"  {'-'*70}")

geo_paths = {
    "Equatorial 20m":  dict(tx_lat=5.0, tx_lon=0.0, rx_lat=5.0, rx_lon=30.0,
                            distance_km=3000, freq_hz=14_097_100),
    "Mid-lat 20m":     dict(tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
                            distance_km=REF_DISTANCE, freq_hz=14_097_100),
    "Polar 20m":       dict(tx_lat=65.0, tx_lon=-20.0, rx_lat=65.0, rx_lon=20.0,
                            distance_km=5000, freq_hz=14_097_100),
    "Mid-lat 10m":     dict(tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
                            distance_km=REF_DISTANCE, freq_hz=28_124_600),
    "Mid-lat 160m":    dict(tx_lat=TX_LAT, tx_lon=TX_LON, rx_lat=RX_LAT, rx_lon=RX_LON,
                            distance_km=REF_DISTANCE, freq_hz=1_836_600),
}

print(f"  {'Path':<18s}  {'Base':>7s}  {'SunG':>6s}  {'StmG':>6s}  "
      f"{'SunRaw':>7s}  {'StmRaw':>7s}  {'SunC':>7s}  {'StmC':>7s}  {'Total':>7s}")
print(f"  {'-'*78}")

for name, geo in geo_paths.items():
    inp = make_input(ssn=100, sfi=150.0, kp=2.0, **geo)
    d = model.decompose(inp)
    print(f"  {name:<18s}  {d['base_snr']:+6.1f}  {d['sun_gate']:6.4f}  "
          f"{d['storm_gate']:6.4f}  {d['sun_raw']:+6.2f}  {d['storm_raw']:+6.2f}  "
          f"{d['sun_contrib']:+6.2f}  {d['storm_contrib']:+6.2f}  "
          f"{d['predicted']:+6.1f}")

# Storm scenario comparison
print(f"\n  Storm Impact: Kp=0 vs Kp=9 by Geography")
print(f"  {'Path':<18s}  {'Kp0 SNR':>8s}  {'Kp9 SNR':>8s}  {'Drop':>6s}  "
      f"{'StmG@Kp0':>8s}  {'StmG@Kp9':>8s}")
print(f"  {'-'*60}")

for name, geo in geo_paths.items():
    inp_kp0 = make_input(ssn=100, sfi=150.0, kp=0.0, **geo)
    inp_kp9 = make_input(ssn=100, sfi=150.0, kp=9.0, **geo)
    d0 = model.decompose(inp_kp0)
    d9 = model.decompose(inp_kp9)
    drop = d0['predicted'] - d9['predicted']
    print(f"  {name:<18s}  {d0['predicted']:+7.2f}  {d9['predicted']:+7.2f}  "
          f"{drop:+5.2f}  {d0['storm_gate']:8.4f}  {d9['storm_gate']:8.4f}")


print(f"\n{'='*70}")
