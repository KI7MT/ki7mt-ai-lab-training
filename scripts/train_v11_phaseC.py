#!/usr/bin/env python3
"""
train_v11_phaseC.py — IONIS V11 "Gatekeeper" Phase C: Frequency Awakening

Break the 0.5 gate floor with frequency-targeted interventions:
  1. Load Phase B checkpoint (gates differentiate by geography, not frequency)
  2. Frequency-weighted sampling: 3x weight on 160m and 10m bands (extremes)
  3. Targeted LR: trunk/base 1e-6 (preserve geography), scaler heads 5e-5
  4. Reduced variance loss lambda=0.001
  5. Frequency-correlation loss: penalizes same gate at 28 MHz vs 1.8 MHz
  6. "Gate Nudge" at epoch 5 if gates still at 0.5 floor
  7. Train 25 epochs, FREQ-01 + POL-01 audits

Goal: Pearson ≥ +0.2600, 10m sun_scaler > 160m sun_scaler
"""

import os
import re
import math
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(TRAINING_DIR, "data", "training_v6_clean.csv")
PHASE_B_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v11_phaseB.pth")
MODEL_DIR = os.path.join(TRAINING_DIR, "models")
MODEL_FILE = "ionis_v11_phaseC.pth"
PHASE = "Phase C: Frequency Awakening (targeted LR, freq-correlation loss)"

BATCH_SIZE = 8192
EPOCHS = 25
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
LAMBDA_VAR = 0.001
LAMBDA_FREQ = 0.005  # Frequency-correlation loss weight

# Turbo Loader for M3 Ultra
NUM_WORKERS = 12
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Feature layout
DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13
FREQ_LOG_IDX = 1  # freq_log is feature index 1

FEATURES = [
    'distance', 'freq_log', 'hour_sin', 'hour_cos',
    'az_sin', 'az_cos', 'lat_diff', 'midpoint_lat',
    'season_sin', 'season_cos', 'day_night_est',
    'sfi', 'kp_penalty',
]

BAND_TO_HZ = {
    102:  1_836_600,   103:  3_568_600,   104:  5_287_200,
    105:  7_038_600,   106: 10_138_700,   107: 14_097_100,
    108: 18_104_600,   109: 21_094_600,   110: 24_924_600,
    111: 28_124_600,
}

# Normalized freq_log values for the two extremes
FREQ_LOG_160M = np.log10(1_836_600) / 8.0   # ~0.7830
FREQ_LOG_10M = np.log10(28_124_600) / 8.0   # ~0.9311

GATE_INIT_BIAS = -math.log(2.0)

# Frequency-weighted sampling: 3x weight on band extremes (160m, 10m)
BAND_SAMPLING_WEIGHTS = {
    102: 3.0,  # 160m — extreme low
    103: 1.0,  # 80m
    104: 1.0,  # 60m
    105: 1.0,  # 40m
    106: 1.0,  # 30m
    107: 1.0,  # 20m
    108: 1.0,  # 17m
    109: 1.0,  # 15m
    110: 1.0,  # 12m
    111: 3.0,  # 10m — extreme high
}

# Audit constants
BANDS = [
    ('160m',  1_836_600), ('80m',  3_568_600), ('60m',  5_287_200),
    ('40m',  7_038_600),  ('30m', 10_138_700), ('20m', 14_097_100),
    ('17m', 18_104_600),  ('15m', 21_094_600), ('12m', 24_924_600),
    ('10m', 28_124_600),
]
BAND_LABEL_TO_ID = {
    '160m': 102, '80m': 103, '60m': 104, '40m': 105, '30m': 106,
    '20m': 107, '17m': 108, '15m': 109, '12m': 110, '10m': 111,
}
REF_TX_GRID = 'FN31'
REF_RX_GRID = 'JO21'
REF_DISTANCE = 5900.0
REF_AZIMUTH = 50.0
REF_HOUR = 12
REF_MONTH = 6
REF_KP = 2.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v11-phaseC')


# ── Grid Utilities ────────────────────────────────────────────────────────
GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def grid_to_latlon(grid):
    g = grid.upper()
    lon = (ord(g[0]) - ord('A')) * 20.0 - 180.0 + int(g[2]) * 2.0 + 1.0
    lat = (ord(g[1]) - ord('A')) * 10.0 - 90.0 + int(g[3]) * 1.0 + 0.5
    return lat, lon


def grid_to_latlon_series(grids):
    lats = np.zeros(len(grids), dtype=np.float32)
    lons = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        s = str(g).strip().rstrip('\x00')
        m = GRID_RE.search(s)
        g4 = m.group(0).upper() if m else 'JJ00'
        lons[i] = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
        lats[i] = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lats, lons


# ── Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df):
    distance = df['distance'].values.astype(np.float32)
    band = df['band'].values.astype(np.int32)
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)
    azimuth = df['azimuth'].values.astype(np.float32)
    sfi = df['sfi'].values.astype(np.float32)
    midpoint_lat_raw = df['midpoint_lat'].values.astype(np.float32)
    kp_penalty = df['kp_penalty'].values.astype(np.float32)

    tx_lat, tx_lon = grid_to_latlon_series(df['tx_grid'].values)
    rx_lat, rx_lon = grid_to_latlon_series(df['rx_grid'].values)

    freq_hz = np.array([BAND_TO_HZ.get(int(b), 14_097_100) for b in band], dtype=np.float64)

    return np.column_stack([
        distance / 20000.0,
        np.log10(freq_hz.astype(np.float32)) / 8.0,
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * azimuth / 360.0),
        np.cos(2.0 * np.pi * azimuth / 360.0),
        np.abs(tx_lat - rx_lat) / 180.0,
        midpoint_lat_raw / 90.0,
        np.sin(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * (hour + (tx_lon + rx_lon) / 2.0 / 15.0) / 24.0),
        sfi / 300.0,
        kp_penalty,
    ]).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────
class WSPRDataset(Dataset):
    def __init__(self, features, targets, weights):
        self.x = features
        self.y = targets
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


# ── Model ─────────────────────────────────────────────────────────────────
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

        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost

    def forward_with_gates(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)

        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        output = base_snr + sun_gate * sun_boost + storm_gate * storm_boost
        return output, sun_gate, storm_gate

    def get_sun_effect(self, sfi_normalized):
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty):
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        x_deep = x[:, :DNN_DIM]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate(sun_logit), _gate(storm_logit)


# ── Metrics ───────────────────────────────────────────────────────────────
def pearson_r(pred, target):
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Frequency-Correlation Loss ────────────────────────────────────────────
def freq_correlation_loss(sun_gate, freq_log_batch):
    """Penalizes gate if it doesn't correlate with frequency.

    For the sun sidecar, higher frequency should → higher gate (more SFI
    benefit for HF bands). This loss computes the negative Pearson
    correlation between sun_gate and freq_log within the batch.

    L_freq = -λ_freq * corr(sun_gate, freq_log)

    If sun_gate is flat across frequencies, corr ≈ 0 → loss = 0.
    If sun_gate positively correlates with freq → corr > 0 → loss < 0 (reward).
    """
    sg = sun_gate.flatten()
    fl = freq_log_batch.flatten()

    sg_m = sg - sg.mean()
    fl_m = fl - fl.mean()
    num = (sg_m * fl_m).sum()
    den = torch.sqrt((sg_m ** 2).sum() * (fl_m ** 2).sum())

    if den < 1e-8:
        return torch.tensor(0.0, device=sun_gate.device)

    corr = num / den
    return -corr  # Negate: we want to maximize positive correlation


# ── Batch Inference ───────────────────────────────────────────────────────
def batch_predict(model, features, batch_size=16384):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.tensor(features[i:i + batch_size],
                                 dtype=torch.float32, device=DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds).flatten()


def controlled_kp_predict(model, features, kp_penalty_value):
    features_copy = features.copy()
    features_copy[:, KP_PENALTY_IDX] = kp_penalty_value
    return batch_predict(model, features_copy)


def controlled_sfi_predict(model, features, sfi_normalized):
    features_copy = features.copy()
    features_copy[:, SFI_IDX] = sfi_normalized
    return batch_predict(model, features_copy)


# ── FREQ-01 ──────────────────────────────────────────────────────────────
def make_ref_input(freq_hz, sfi, kp=REF_KP):
    tx_lat, tx_lon = grid_to_latlon(REF_TX_GRID)
    rx_lat, rx_lon = grid_to_latlon(REF_RX_GRID)

    distance = REF_DISTANCE / 20000.0
    freq_log = np.log10(freq_hz) / 8.0
    hour_sin = np.sin(2.0 * np.pi * REF_HOUR / 24.0)
    hour_cos = np.cos(2.0 * np.pi * REF_HOUR / 24.0)
    az_sin = np.sin(2.0 * np.pi * REF_AZIMUTH / 360.0)
    az_cos = np.cos(2.0 * np.pi * REF_AZIMUTH / 360.0)
    lat_diff = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    season_sin = np.sin(2.0 * np.pi * REF_MONTH / 12.0)
    season_cos = np.cos(2.0 * np.pi * REF_MONTH / 12.0)
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_h = REF_HOUR + midpoint_lon / 15.0
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


def run_freq01_audit(model):
    log.info("")
    log.info("=" * 75)
    log.info("  FREQ-01 AUDIT: SFI Benefit by Band (V11 Phase C)")
    log.info(f"  Reference Path: {REF_TX_GRID}→{REF_RX_GRID} ({REF_DISTANCE:.0f}km)")
    log.info(f"  Fixed: Hour={REF_HOUR}UTC, Month={REF_MONTH}, Kp={REF_KP}")
    log.info("=" * 75)

    model.eval()
    sfi_sweep = [60, 120, 200, 300]

    hdr = f"  {'Band':>5s}  {'MHz':>8s}"
    for sfi in sfi_sweep:
        hdr += f"  SFI{sfi:>3d}"
    hdr += "   Delta  Gate_Sun Gate_Stm"
    log.info(hdr)
    log.info(f"  {'-' * (len(hdr) - 2)}")

    deltas = []
    sun_gates = []
    storm_gates = []
    for label, freq_hz in BANDS:
        snrs = []
        for sfi in sfi_sweep:
            inp = make_ref_input(freq_hz, sfi)
            with torch.no_grad():
                snr = model(inp).item()
            snrs.append(snr)

        delta = snrs[-1] - snrs[0]
        deltas.append(delta)

        inp_ref = make_ref_input(freq_hz, 150.0)
        sun_g, storm_g = model.get_gates(inp_ref)
        sun_gates.append(sun_g.item())
        storm_gates.append(storm_g.item())

        row = f"  {label:>5s}  {freq_hz/1e6:7.3f}"
        for snr in snrs:
            row += f"  {snr:+6.1f}"
        row += f"  {delta:+5.2f}   {sun_g.item():.4f}   {storm_g.item():.4f}"
        log.info(row)

    spread = max(deltas) - min(deltas)
    is_flat = spread < 0.01
    sun_gate_range = max(sun_gates) - min(sun_gates)
    storm_gate_range = max(storm_gates) - min(storm_gates)

    log.info(f"\n  SFI 60→300 benefit: min={min(deltas):+.4f}, max={max(deltas):+.4f}")
    log.info(f"  Benefit spread: {spread:.4f} dB {'(FLAT)' if is_flat else '(DIFFERENTIATED!)'}")
    log.info(f"  Sun gate range:   {min(sun_gates):.4f} – {max(sun_gates):.4f} "
             f"(span={sun_gate_range:.4f})")
    log.info(f"  Storm gate range: {min(storm_gates):.4f} – {max(storm_gates):.4f} "
             f"(span={storm_gate_range:.4f})")

    sun_160m = sun_gates[0]
    sun_10m = sun_gates[-1]
    log.info(f"\n  KEY TEST: 10m sun_gate ({sun_10m:.4f}) vs 160m sun_gate ({sun_160m:.4f})")
    if sun_10m > sun_160m:
        log.info("  >>> PASS: 28 MHz has higher sun_scaler than 1.8 MHz!")
    elif abs(sun_10m - sun_160m) < 0.001:
        log.info("  FLAT: No differentiation between 10m and 160m yet.")
    else:
        log.info("  INVERTED: 160m has higher sun_scaler — unexpected.")

    return spread, sun_gate_range, storm_gate_range, sun_gates, storm_gates


def run_freq01_real_data_audit(model, df):
    log.info(f"\n  {'─' * 70}")
    log.info("  FREQ-01 Real Data: Per-Band SFI Benefit + Gate Stats")
    log.info(f"  {'─' * 70}")

    log.info(f"\n  {'Band':>5s}  {'Rows':>10s}  {'SNR@60':>8s}  {'SNR@300':>9s}  "
             f"{'Delta':>7s}  {'SunG_μ':>7s}  {'StmG_μ':>7s}")
    log.info(f"  {'-' * 62}")

    real_deltas = {}
    for label, freq_hz in BANDS:
        band_id = BAND_LABEL_TO_ID.get(label)
        if band_id is None:
            continue
        mask = df['band'] == band_id
        df_band = df[mask]
        if len(df_band) < 100:
            continue

        X_band = engineer_features(df_band)
        snr_sfi60 = controlled_sfi_predict(model, X_band, 60.0 / 300.0).mean()
        snr_sfi300 = controlled_sfi_predict(model, X_band, 300.0 / 300.0).mean()
        delta = snr_sfi300 - snr_sfi60
        real_deltas[label] = delta

        X_sample = X_band[:min(5000, len(X_band))]
        X_t = torch.tensor(X_sample, dtype=torch.float32, device=DEVICE)
        sun_g, storm_g = model.get_gates(X_t)

        log.info(f"  {label:>5s}  {len(df_band):>10,}  {snr_sfi60:+7.2f}dB  "
                 f"{snr_sfi300:+7.2f}dB  {delta:+6.2f}dB  "
                 f"{sun_g.mean().item():6.4f}  {storm_g.mean().item():6.4f}")

    if real_deltas:
        vals = list(real_deltas.values())
        real_spread = max(vals) - min(vals)
        log.info(f"\n  Real-data SFI benefit spread: {real_spread:.4f} dB "
                 f"{'(DIFFERENTIATED!)' if real_spread > 0.01 else '(flat)'}")

    return real_deltas


def run_pol01_audit(model, df):
    log.info("")
    log.info("=" * 75)
    log.info("  POL-01 AUDIT: Storm Impact by Latitude (V11 Phase C)")
    log.info("=" * 75)

    model.eval()
    midlat = df['midpoint_lat'].values

    polar_mask = np.abs(midlat) > 60.0
    equatorial_mask = np.abs(midlat) < 20.0
    df_polar = df[polar_mask].copy()
    df_equatorial = df[equatorial_mask].copy()

    log.info(f"  Polar (|lat|>60°):     {len(df_polar):>10,} rows")
    log.info(f"  Equatorial (|lat|<20°): {len(df_equatorial):>10,} rows")

    X_polar = engineer_features(df_polar)
    X_equatorial = engineer_features(df_equatorial)

    log.info(f"\n  {'Kp':>3s}  {'kp_pen':>7s}  {'Polar SNR':>10s}  {'Eq SNR':>10s}  {'P-E':>7s}")
    log.info(f"  {'-' * 42}")

    kp_values = [0, 3, 5, 7, 9]
    polar_snrs = {}
    eq_snrs = {}
    for kp in kp_values:
        kp_penalty = 1.0 - kp / 9.0
        p_snr = controlled_kp_predict(model, X_polar, kp_penalty).mean()
        e_snr = controlled_kp_predict(model, X_equatorial, kp_penalty).mean()
        polar_snrs[kp] = p_snr
        eq_snrs[kp] = e_snr
        log.info(f"  {kp:3d}  {kp_penalty:7.4f}  {p_snr:+9.2f}dB  {e_snr:+9.2f}dB  "
                 f"{p_snr - e_snr:+6.2f}dB")

    polar_drop = polar_snrs[0] - polar_snrs[9]
    eq_drop = eq_snrs[0] - eq_snrs[9]
    geo_amp = polar_drop - eq_drop

    log.info(f"\n  Polar Kp 0→9 drop:      {polar_drop:+.4f} dB")
    log.info(f"  Equatorial Kp 0→9 drop: {eq_drop:+.4f} dB")
    log.info(f"  Geographic amplification: {geo_amp:+.4f} dB")

    # Latitude bins
    log.info(f"\n  {'Lat Bin':>12s}  {'Count':>8s}  {'Kp0→9 Drop':>12s}  "
             f"{'Sun Gate':>9s}  {'Storm Gate':>11s}")
    log.info(f"  {'-' * 60}")

    lat_bins = np.arange(-80, 90, 20)
    for lo in lat_bins:
        hi = lo + 20
        mask = (midlat >= lo) & (midlat < hi)
        count = mask.sum()
        if count < 100:
            continue
        df_bin = df[mask]
        X_bin = engineer_features(df_bin)
        snr_kp0 = controlled_kp_predict(model, X_bin, 1.0).mean()
        snr_kp9 = controlled_kp_predict(model, X_bin, 0.0).mean()
        drop = snr_kp0 - snr_kp9
        X_bin_t = torch.tensor(X_bin[:min(5000, len(X_bin))],
                               dtype=torch.float32, device=DEVICE)
        sun_g, storm_g = model.get_gates(X_bin_t)
        log.info(f"  {lo:+3d}° to {hi:+3d}°  {count:>8,}  {drop:+10.4f} dB  "
                 f"{sun_g.mean().item():8.4f}  {storm_g.mean().item():10.4f}")

    return polar_drop, eq_drop, geo_amp


# ── Gate Nudge ────────────────────────────────────────────────────────────
def gate_nudge(model):
    """If gates are stuck at the 0.5 floor, manually lift them into the
    active sigmoid range by shifting the scaler head output biases.

    The gate function is: gate(x) = 0.5 + 1.5 * sigmoid(x)
    At 0.5 floor: sigmoid(x) ≈ 0 → x << 0 (deep negative saturation)

    Nudge: add +1.0 to the final bias, moving output from ~-4 to ~-3,
    which puts sigmoid into a more responsive range.
    """
    with torch.no_grad():
        for name, head in [('sun_scaler', model.sun_scaler_head),
                           ('storm_scaler', model.storm_scaler_head)]:
            final_bias = head[-1].bias
            old_val = final_bias.item()
            final_bias.add_(1.0)
            new_val = final_bias.item()
            log.info(f"  Gate Nudge: {name} bias {old_val:.3f} → {new_val:.3f}")


def check_gates_at_floor(model):
    """Check if gate values on the reference path are stuck at 0.5."""
    model.eval()
    inp_160m = make_ref_input(1_836_600, 150.0)
    inp_10m = make_ref_input(28_124_600, 150.0)
    sun_160, _ = model.get_gates(inp_160m)
    sun_10, _ = model.get_gates(inp_10m)
    at_floor = (abs(sun_160.item() - 0.5) < 0.01) and (abs(sun_10.item() - 0.5) < 0.01)
    return at_floor, sun_160.item(), sun_10.item()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info(f"IONIS V11 | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Turbo Loader: workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}")
    log.info(f"Targeted LR: trunk/base 1e-6 | scaler heads 5e-5 | sidecars 1e-3")
    log.info(f"Lambda: var={LAMBDA_VAR}, freq={LAMBDA_FREQ}")

    # ── Load Phase B checkpoint ──
    log.info(f"\nLoading Phase B checkpoint: {PHASE_B_PATH}")
    ckpt = torch.load(PHASE_B_PATH, weights_only=False, map_location='cpu')
    log.info(f"  Phase B RMSE: {ckpt.get('val_rmse', '?'):.4f} dB")
    log.info(f"  Phase B Pearson: {ckpt.get('val_pearson', '?'):+.4f}")

    model = IonisV11Gate(dnn_dim=DNN_DIM, sidecar_hidden=8)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())

    # Sidecar constraints
    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total: {total_p:,} | Trainable: {trainable:,}")

    # Pre-training gate state
    at_floor, sun_160, sun_10 = check_gates_at_floor(model)
    log.info(f"\n  Pre-train ref path gates: 160m sun={sun_160:.4f}, 10m sun={sun_10:.4f}")
    log.info(f"  Gates at 0.5 floor: {'YES' if at_floor else 'NO'}")

    # ── Load data ──
    log.info(f"\nLoading {CSV_PATH}...")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    load_sec = time.perf_counter() - t0
    log.info(f"Loaded {len(df):,} rows in {load_sec:.1f}s")

    y_raw = df['snr'].values.astype(np.float32)
    snr_mean = float(y_raw.mean())
    snr_std = float(y_raw.std())
    bands_raw = df['band'].values.astype(np.int32)

    log.info(f"Engineering {INPUT_DIM} features...")
    t0 = time.perf_counter()
    X_np = engineer_features(df)
    y_np = y_raw.reshape(-1, 1)
    w_np = df['sampling_weight'].values.astype(np.float32).reshape(-1, 1)
    feat_sec = time.perf_counter() - t0
    log.info(f"Feature engineering complete in {feat_sec:.1f}s")

    # ── Train/val split ──
    n = len(X_np)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size

    indices = np.random.permutation(n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train, y_train, w_train = X_np[train_idx], y_np[train_idx], w_np[train_idx]
    bands_train = bands_raw[train_idx]

    train_set = WSPRDataset(X_train, y_train, w_train)
    val_set = WSPRDataset(X_np[val_idx], y_np[val_idx], w_np[val_idx])

    # Frequency-weighted sampling: 3x on 160m and 10m
    sample_weights = np.array([BAND_SAMPLING_WEIGHTS.get(int(b), 1.0)
                               for b in bands_train], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=train_size,
        replacement=True,
    )

    extreme_count = np.sum((bands_train == 102) | (bands_train == 111))
    log.info(f"\nFrequency-weighted sampling: 3x on 160m + 10m")
    log.info(f"  Extreme bands (160m+10m): {extreme_count:,} / {train_size:,} "
             f"({100*extreme_count/train_size:.1f}%)")
    log.info(f"  Effective batch composition: ~37.5% extreme, ~62.5% mid-bands")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
    )
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Optimizer: Targeted LR per Gemini spec ──
    optimizer = optim.AdamW([
        {'params': model.trunk.parameters(), 'lr': 1e-6},
        {'params': model.base_head.parameters(), 'lr': 1e-6},
        {'params': model.sun_scaler_head.parameters(), 'lr': 5e-5},
        {'params': model.storm_scaler_head.parameters(), 'lr': 5e-5},
        {'params': [p for p in model.sun_sidecar.parameters() if p.requires_grad], 'lr': 1e-3},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad], 'lr': 1e-3},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion = nn.HuberLoss(reduction='none', delta=1.0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')
    best_pearson = -1.0
    nudge_applied = False

    # ── Training ──
    log.info(f"\nTraining started — Phase C Frequency Awakening ({EPOCHS} epochs)")
    log.info(f"LR: trunk/base 1e-6 | scalers 5e-5 | sidecars 1e-3")
    log.info(f"Loss: Huber + var_loss(λ={LAMBDA_VAR}) + freq_corr_loss(λ={LAMBDA_FREQ})")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  "
           f"{'SunGVar':>8s}  {'FrqCorr':>8s}  "
           f"{'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.perf_counter()

        # Gate Nudge check at epoch 5
        if epoch == 6 and not nudge_applied:
            at_floor, sun_160, sun_10 = check_gates_at_floor(model)
            if at_floor:
                log.info(f"  >>> GATE NUDGE: Gates still at 0.5 floor after 5 epochs")
                gate_nudge(model)
                nudge_applied = True
                at_floor2, sun_160_2, sun_10_2 = check_gates_at_floor(model)
                log.info(f"  Post-nudge: 160m sun={sun_160_2:.4f}, 10m sun={sun_10_2:.4f}")
            else:
                log.info(f"  Gates already off floor (160m={sun_160:.4f}, 10m={sun_10:.4f})"
                         f" — skipping nudge")

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        epoch_sun_var_sum = 0.0
        epoch_freq_corr_sum = 0.0

        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()

            out, sun_gate, storm_gate = model.forward_with_gates(bx)

            # Primary loss
            primary_loss = (criterion(out, by) * bw).mean()

            # Anti-collapse variance loss
            sun_var = sun_gate.var()
            storm_var = storm_gate.var()
            var_loss = -LAMBDA_VAR * (sun_var + storm_var)

            # Frequency-correlation loss: encourage sun_gate ~ freq_log
            freq_log_batch = bx[:, FREQ_LOG_IDX:FREQ_LOG_IDX + 1]
            freq_loss = LAMBDA_FREQ * freq_correlation_loss(sun_gate, freq_log_batch)

            loss = primary_loss + var_loss + freq_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for sidecar in [model.sun_sidecar, model.storm_sidecar]:
                    sidecar.fc1.weight.clamp_(0.5, 2.0)
                    sidecar.fc2.weight.clamp_(0.5, 2.0)

            train_loss_sum += primary_loss.item()
            train_batches += 1
            epoch_sun_var_sum += sun_var.item()
            epoch_freq_corr_sum += freq_loss.item()

        train_loss = train_loss_sum / train_batches
        avg_sun_var = epoch_sun_var_sum / train_batches
        avg_freq_corr = epoch_freq_corr_sum / train_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, by, bw in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by).mean()
                val_loss_sum += loss.item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)
        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        val_pearson = pearson_r(preds_cat, targets_cat)

        sfi_benefit = model.get_sun_effect(200.0 / 300.0) - model.get_sun_effect(70.0 / 300.0)
        storm_cost = model.get_storm_effect(1.0) - model.get_storm_effect(0.0)

        scheduler.step()
        epoch_sec = time.perf_counter() - t_epoch

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pearson = val_pearson
            torch.save({
                'model_state': model.state_dict(),
                'snr_mean': snr_mean,
                'snr_std': snr_std,
                'dnn_dim': DNN_DIM,
                'sidecar_hidden': 8,
                'features': FEATURES,
                'band_to_hz': BAND_TO_HZ,
                'date_range': '2020-01-01 to 2026-02-04',
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'architecture': 'IonisV11Gate (trunk+3heads+2sidecars)',
                'sfi_benefit_dB': sfi_benefit,
                'storm_cost_dB': storm_cost,
                'training_phase': 'C',
                'lambda_var': LAMBDA_VAR,
                'lambda_freq': LAMBDA_FREQ,
                'nudge_applied': nudge_applied,
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.2f}dB  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.1f}  {storm_cost:+4.1f}  "
            f"{avg_sun_var:8.6f}  {avg_freq_corr:+8.5f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        if epoch % 5 == 0 or epoch == 1:
            at_floor, s160, s10 = check_gates_at_floor(model)
            with torch.no_grad():
                sun_b = model.sun_sidecar.fc2.bias.item()
                storm_b = model.storm_sidecar.fc2.bias.item()
            log.info(f"      Sidecar bias: Sun={sun_b:.2f}, Storm={storm_b:.2f}")
            log.info(f"      Ref path gates: 160m sun={s160:.4f}, 10m sun={s10:.4f} "
                     f"{'(FLOOR)' if at_floor else '(off floor)'}")

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Phase C complete. Best RMSE: {best_rmse:.4f} dB, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    phaseB_rmse = ckpt.get('val_rmse', 2.47)
    phaseB_pearson = ckpt.get('val_pearson', 0.256)
    log.info(f"\n  Phase B baseline: RMSE {phaseB_rmse:.4f}, Pearson {phaseB_pearson:+.4f}")
    log.info(f"  Phase C result:   RMSE {best_rmse:.4f}, Pearson {best_pearson:+.4f}")
    log.info(f"  Pearson target ≥ 0.2600: {'PASS' if best_pearson >= 0.26 else 'NOT YET'}")

    # ── Physics verification ──
    log.info("")
    log.info("=== PHYSICS VERIFICATION ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0)
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    kp0 = model.get_storm_effect(1.0)
    kp9 = model.get_storm_effect(0.0)
    log.info(f"Sun Sidecar: SFI 70→200 benefit: {sfi_200 - sfi_70:+.2f} dB")
    log.info(f"Storm Sidecar: Kp 0→9 cost: {kp0 - kp9:+.2f} dB")
    log.info(f"SFI monotonicity: {'CORRECT' if sfi_200 > sfi_70 else 'WARNING'}")
    log.info(f"Kp monotonicity:  {'CORRECT' if kp0 > kp9 else 'WARNING'}")

    # ── Load best checkpoint for audits ──
    log.info("\nLoading best checkpoint for audits...")
    best_ckpt = torch.load(model_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state'])
    model.eval()

    # ── Audits ──
    freq_spread, sun_range, storm_range, sun_gates, storm_gates = run_freq01_audit(model)
    real_deltas = run_freq01_real_data_audit(model, df)
    polar_drop, eq_drop, geo_amp = run_pol01_audit(model, df)

    # ── Final Summary ──
    log.info("")
    log.info("=" * 75)
    log.info("  PHASE C SUMMARY: Frequency Awakening Results")
    log.info("=" * 75)
    log.info(f"  RMSE:     {best_rmse:.4f} dB (Phase B: {phaseB_rmse:.4f})")
    log.info(f"  Pearson:  {best_pearson:+.4f} (Phase B: {phaseB_pearson:+.4f})")
    log.info(f"  Pearson target ≥ 0.2600: {'PASS' if best_pearson >= 0.26 else 'NOT YET'}")
    log.info(f"  SFI benefit: {sfi_200 - sfi_70:+.2f} dB")
    log.info(f"  Storm cost:  {kp0 - kp9:+.2f} dB")
    log.info(f"  FREQ-01 ref path spread: {freq_spread:.4f} dB "
             f"{'(DIFFERENTIATED!)' if freq_spread > 0.01 else '(flat)'}")
    log.info(f"  10m sun_gate ({sun_gates[-1]:.4f}) > 160m sun_gate ({sun_gates[0]:.4f}): "
             f"{'PASS' if sun_gates[-1] > sun_gates[0] else 'NOT YET'}")
    log.info(f"  POL-01 geo amplification: {geo_amp:+.4f} dB "
             f"{'(preserved)' if abs(geo_amp) > 0.01 else '(flat)'}")
    log.info(f"  Gate nudge applied: {'YES' if nudge_applied else 'NO'}")
    log.info("=" * 75)


if __name__ == '__main__':
    main()
