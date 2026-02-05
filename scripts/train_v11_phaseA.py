#!/usr/bin/env python3
"""
train_v11_phaseA.py — IONIS V11 "Gatekeeper" Phase A: V10 Warm-Start

Weight Transfer + Frozen Gates:
  1. Load V10 checkpoint weights into V11 trunk + base_head + sidecars
  2. Scaler heads initialized to gate=1.0 and FROZEN (requires_grad=False)
  3. Train 10 warm-up epochs to confirm V11 recovers V10 baseline
  4. Run FREQ-01 quick check to verify flat SFI benefit (gates are frozen)

Architecture: IonisV11Gate (203,573 total params, 33,026 frozen in scaler heads)
  Trunk:             11 → 512 → 256 (shared, from V10 dnn layers 0-2)
  Base Head:         256 → 128 → 1  (from V10 dnn layers 4-6)
  Sun Scaler Head:   256 → 64 → 1   (FROZEN, gate=1.0)
  Storm Scaler Head: 256 → 64 → 1   (FROZEN, gate=1.0)
  Sun Sidecar:       1 → 8 → 1      (from V10, clamped 0.5-2.0)
  Storm Sidecar:     1 → 8 → 1      (from V10, clamped 0.5-2.0)

Exit criteria: RMSE ≤ 2.50 dB, Pearson ≥ +0.237 (within V10 baseline)
"""

import os
import re
import sys
import math
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(TRAINING_DIR, "data", "training_v6_clean.csv")
V10_MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v10_final.pth")
MODEL_DIR = os.path.join(TRAINING_DIR, "models")
MODEL_FILE = "ionis_v11_phaseA.pth"
PHASE = "Phase A: V10 Warm-Start (scaler heads frozen, gate=1.0)"

BATCH_SIZE = 8192
EPOCHS = 10
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2

# Turbo Loader for M3 Ultra
NUM_WORKERS = 12
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Feature layout (unchanged from V10)
DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

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

# FREQ-01 reference path
BANDS = [
    ('160m',  1_836_600), ('80m',  3_568_600), ('60m',  5_287_200),
    ('40m',  7_038_600),  ('30m', 10_138_700), ('20m', 14_097_100),
    ('17m', 18_104_600),  ('15m', 21_094_600), ('12m', 24_924_600),
    ('10m', 28_124_600),
]
REF_TX_GRID = 'FN31'
REF_RX_GRID = 'JO21'
REF_DISTANCE = 5900.0
REF_AZIMUTH = 50.0
REF_HOUR = 12
REF_MONTH = 6
REF_KP = 2.0

GATE_INIT_BIAS = -math.log(2.0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v11-phaseA')


# --- GRID UTILITIES ---
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


# --- FEATURE ENGINEERING ---
def engineer_features(df):
    """Compute 13 features from raw CSV columns. Identical to V10."""
    n = len(df)
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

    f_distance     = distance / 20000.0
    f_freq_log     = np.log10(freq_hz.astype(np.float32)) / 8.0
    f_hour_sin     = np.sin(2.0 * np.pi * hour / 24.0)
    f_hour_cos     = np.cos(2.0 * np.pi * hour / 24.0)
    f_az_sin       = np.sin(2.0 * np.pi * azimuth / 360.0)
    f_az_cos       = np.cos(2.0 * np.pi * azimuth / 360.0)
    f_lat_diff     = np.abs(tx_lat - rx_lat) / 180.0
    f_midpoint_lat = midpoint_lat_raw / 90.0
    f_season_sin   = np.sin(2.0 * np.pi * month / 12.0)
    f_season_cos   = np.cos(2.0 * np.pi * month / 12.0)
    midpoint_lon   = (tx_lon + rx_lon) / 2.0
    local_solar_h  = hour + midpoint_lon / 15.0
    f_daynight     = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    f_sfi          = sfi / 300.0
    f_kp_penalty   = kp_penalty

    return np.column_stack([
        f_distance, f_freq_log, f_hour_sin, f_hour_cos,
        f_az_sin, f_az_cos, f_lat_diff, f_midpoint_lat,
        f_season_sin, f_season_cos, f_daynight,
        f_sfi, f_kp_penalty,
    ]).astype(np.float32)


# --- DATASET ---
class WSPRDataset(Dataset):
    def __init__(self, features, targets, weights):
        self.x = features
        self.y = targets
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


# --- MODEL (from ionis_v11_gate.py) ---
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


# --- METRICS ---
def pearson_r(pred, target):
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# --- V10 → V11 WEIGHT TRANSFER ---
def transfer_v10_weights(model, v10_state):
    """Map V10 IonisDualMono weights into V11 IonisV11Gate.

    V10 DNN Sequential:           V11 mapping:
      dnn.0 (Linear 11→512)    →  trunk.0
      dnn.1 (Mish)             →  trunk.1  (no weights)
      dnn.2 (Linear 512→256)   →  trunk.2
      dnn.3 (Mish)             →  trunk.3  (no weights)
      dnn.4 (Linear 256→128)   →  base_head.0
      dnn.5 (Mish)             →  base_head.1 (no weights)
      dnn.6 (Linear 128→1)     →  base_head.2
      sun_sidecar.*            →  sun_sidecar.*
      storm_sidecar.*          →  storm_sidecar.*
    """
    weight_map = {
        # Trunk (first 2 linear layers of V10 DNN)
        'dnn.0.weight': 'trunk.0.weight',
        'dnn.0.bias':   'trunk.0.bias',
        'dnn.2.weight': 'trunk.2.weight',
        'dnn.2.bias':   'trunk.2.bias',
        # Base head (last 2 linear layers of V10 DNN)
        'dnn.4.weight': 'base_head.0.weight',
        'dnn.4.bias':   'base_head.0.bias',
        'dnn.6.weight': 'base_head.2.weight',
        'dnn.6.bias':   'base_head.2.bias',
        # Sidecars (direct copy)
        'sun_sidecar.fc1.weight':   'sun_sidecar.fc1.weight',
        'sun_sidecar.fc1.bias':     'sun_sidecar.fc1.bias',
        'sun_sidecar.fc2.weight':   'sun_sidecar.fc2.weight',
        'sun_sidecar.fc2.bias':     'sun_sidecar.fc2.bias',
        'storm_sidecar.fc1.weight': 'storm_sidecar.fc1.weight',
        'storm_sidecar.fc1.bias':   'storm_sidecar.fc1.bias',
        'storm_sidecar.fc2.weight': 'storm_sidecar.fc2.weight',
        'storm_sidecar.fc2.bias':   'storm_sidecar.fc2.bias',
    }

    transferred = 0
    for v10_key, v11_key in weight_map.items():
        if v10_key in v10_state:
            v11_param = model
            for attr in v11_key.split('.'):
                if attr.isdigit():
                    v11_param = v11_param[int(attr)]
                else:
                    v11_param = getattr(v11_param, attr)
            v11_param.data.copy_(v10_state[v10_key])
            transferred += v10_state[v10_key].numel()

    return transferred


# --- FREQ-01 QUICK CHECK ---
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


def freq01_quick_check(model):
    """Run FREQ-01 Test 1: SFI sweep per band on the reference path."""
    log.info("")
    log.info("=" * 65)
    log.info("  FREQ-01 Quick Check: SFI Sweep by Band")
    log.info(f"  Path: {REF_TX_GRID}→{REF_RX_GRID} ({REF_DISTANCE:.0f}km)")
    log.info(f"  Fixed: Hour={REF_HOUR}UTC, Month={REF_MONTH}, Kp={REF_KP}")
    log.info("=" * 65)

    model.eval()
    sfi_sweep = [60, 120, 200, 300]

    hdr = f"  {'Band':>5s}  {'MHz':>8s}"
    for sfi in sfi_sweep:
        hdr += f"  SFI{sfi:>3d}"
    hdr += "   Delta  Gate_Sun Gate_Stm"
    log.info(hdr)
    log.info(f"  {'-' * (len(hdr) - 2)}")

    deltas = []
    for label, freq_hz in BANDS:
        snrs = []
        for sfi in sfi_sweep:
            inp = make_ref_input(freq_hz, sfi)
            with torch.no_grad():
                snr = model(inp).item()
            snrs.append(snr)

        delta = snrs[-1] - snrs[0]
        deltas.append(delta)

        # Gate values for this band
        inp_ref = make_ref_input(freq_hz, 150.0)
        sun_g, storm_g = model.get_gates(inp_ref)

        row = f"  {label:>5s}  {freq_hz/1e6:7.3f}"
        for snr in snrs:
            row += f"  {snr:+6.1f}"
        row += f"  {delta:+5.2f}   {sun_g.item():.4f}   {storm_g.item():.4f}"
        log.info(row)

    spread = max(deltas) - min(deltas)
    is_flat = spread < 0.01
    log.info(f"\n  SFI 60→300 benefit spread: {spread:.4f} dB")
    log.info(f"  Flat line confirmed: {'YES' if is_flat else 'NO'} (threshold < 0.01 dB)")

    if is_flat:
        log.info("  FREQ-01 PASS: Gates frozen at 1.0, flat SFI benefit as expected.")
    else:
        log.info("  FREQ-01 NOTE: Some variation detected — gates may not be perfectly frozen.")

    return is_flat, spread


# --- MAIN ---
def main():
    log.info(f"IONIS V11 | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Turbo Loader: workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}")

    # --- Load V10 checkpoint ---
    log.info(f"\nLoading V10 checkpoint: {V10_MODEL_PATH}")
    v10_ckpt = torch.load(V10_MODEL_PATH, weights_only=False, map_location='cpu')
    v10_state = v10_ckpt['model_state']
    log.info(f"  V10 RMSE: {v10_ckpt.get('val_rmse', '?'):.4f} dB")
    log.info(f"  V10 Pearson: {v10_ckpt.get('val_pearson', '?'):+.4f}")
    log.info(f"  V10 SFI benefit: {v10_ckpt.get('sfi_benefit_dB', '?'):+.2f} dB")
    log.info(f"  V10 Storm cost: {v10_ckpt.get('storm_cost_dB', '?'):+.2f} dB")

    # --- Create V11 model and transfer weights ---
    model = IonisV11Gate(dnn_dim=DNN_DIM, sidecar_hidden=8).to(DEVICE)

    # Transfer V10 weights BEFORE moving to device (load on CPU first)
    model_cpu = IonisV11Gate(dnn_dim=DNN_DIM, sidecar_hidden=8)
    transferred = transfer_v10_weights(model_cpu, v10_state)
    model.load_state_dict(model_cpu.state_dict())
    del model_cpu

    log.info(f"\n  Weight transfer: {transferred:,} parameters mapped from V10")

    # Count params by component
    trunk_p = sum(p.numel() for p in model.trunk.parameters())
    base_p = sum(p.numel() for p in model.base_head.parameters())
    sun_sc_p = sum(p.numel() for p in model.sun_scaler_head.parameters())
    storm_sc_p = sum(p.numel() for p in model.storm_scaler_head.parameters())
    sun_side_p = sum(p.numel() for p in model.sun_sidecar.parameters())
    storm_side_p = sum(p.numel() for p in model.storm_sidecar.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    log.info(f"  Trunk: {trunk_p:,} | Base: {base_p:,} | "
             f"Sun SC head: {sun_sc_p:,} | Storm SC head: {storm_sc_p:,} | "
             f"Sun side: {sun_side_p} | Storm side: {storm_side_p}")
    log.info(f"  Total: {total_p:,} parameters")

    # --- Freeze scaler heads ---
    frozen_count = 0
    for head in [model.sun_scaler_head, model.storm_scaler_head]:
        for p in head.parameters():
            p.requires_grad = False
            frozen_count += p.numel()
    log.info(f"\n  FROZEN: {frozen_count:,} params in scaler heads (gate=1.0)")

    # --- Sidecar constraints (same as V10) ---
    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True
    log.info("  Sidecars: fc1.bias frozen, fc2.bias learnable (relief valve)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable: {trainable:,} / {total_p:,} parameters")

    # --- Verify gate init ---
    x_test = torch.randn(10, INPUT_DIM, device=DEVICE)
    sun_g, storm_g = model.get_gates(x_test)
    log.info(f"\n  Gate check: sun={sun_g.mean():.4f}, storm={storm_g.mean():.4f} (expect 1.0)")

    # --- Pre-training physics check ---
    sfi_high = model.get_sun_effect(200.0 / 300.0)
    sfi_low = model.get_sun_effect(70.0 / 300.0)
    kp_quiet = model.get_storm_effect(1.0)
    kp_storm = model.get_storm_effect(0.0)
    log.info(f"  Pre-train physics: SFI 70→200 = {sfi_high - sfi_low:+.2f} dB, "
             f"Kp 0→9 cost = {kp_quiet - kp_storm:+.2f} dB")

    # --- Load data ---
    log.info(f"\nLoading {CSV_PATH}...")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    load_sec = time.perf_counter() - t0
    log.info(f"Loaded {len(df):,} rows in {load_sec:.1f}s")

    y_raw = df['snr'].values.astype(np.float32)
    snr_mean = float(y_raw.mean())
    snr_std = float(y_raw.std())

    log.info(f"Engineering {INPUT_DIM} features...")
    t0 = time.perf_counter()
    X_np = engineer_features(df)
    y_np = y_raw.reshape(-1, 1)
    w_np = df['sampling_weight'].values.astype(np.float32).reshape(-1, 1)
    feat_sec = time.perf_counter() - t0
    log.info(f"Feature engineering complete in {feat_sec:.1f}s")
    log.info(f"SNR mean: {snr_mean:.1f} dB, std: {snr_std:.1f} dB")

    # --- Train/val split ---
    n = len(X_np)
    dataset = WSPRDataset(X_np, y_np, w_np)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
    )
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # --- Optimizer: Differential LR (same as V10) ---
    # Only unfrozen params are passed to optimizer
    optimizer = optim.AdamW([
        {'params': model.trunk.parameters(), 'lr': 1e-5},
        {'params': model.base_head.parameters(), 'lr': 1e-5},
        # Scaler heads are frozen — not in optimizer
        {'params': [p for p in model.sun_sidecar.parameters() if p.requires_grad], 'lr': 1e-3},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad], 'lr': 1e-3},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.HuberLoss(reduction='none', delta=1.0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')
    best_pearson = -1.0

    # --- Training ---
    log.info(f"\nTraining started — Phase A Warm-Start ({EPOCHS} epochs)")
    log.info("Differential LR: trunk/base 1e-5 | sidecars 1e-3 | scalers FROZEN")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  {'LR':>10s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  {'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.perf_counter()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            per_sample = criterion(out, by)
            loss = (per_sample * bw).mean()
            loss.backward()
            optimizer.step()

            # Post-optimizer weight clamp on sidecars (same as V10)
            with torch.no_grad():
                for sidecar in [model.sun_sidecar, model.storm_sidecar]:
                    sidecar.fc1.weight.clamp_(0.5, 2.0)
                    sidecar.fc2.weight.clamp_(0.5, 2.0)

            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

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

        # Physics check
        sfi_high = model.get_sun_effect(200.0 / 300.0)
        sfi_low = model.get_sun_effect(70.0 / 300.0)
        sfi_benefit = sfi_high - sfi_low

        kp_quiet = model.get_storm_effect(1.0)
        kp_storm = model.get_storm_effect(0.0)
        storm_cost = kp_quiet - kp_storm

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
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
                'v10_baseline_rmse': v10_ckpt.get('val_rmse'),
                'v10_baseline_pearson': v10_ckpt.get('val_pearson'),
                'sfi_benefit_dB': sfi_benefit,
                'storm_cost_dB': storm_cost,
                'scaler_heads_frozen': True,
                'training_phase': 'A',
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.2f}dB  {val_pearson:+7.4f}  "
            f"{lr_now:.2e}  {sfi_benefit:+4.1f}  {storm_cost:+4.1f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        # Sidecar bias check every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                sun_b = model.sun_sidecar.fc2.bias.item()
                storm_b = model.storm_sidecar.fc2.bias.item()
            log.info(f"      Sidecar bias: Sun={sun_b:.2f}, Storm={storm_b:.2f}")

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Phase A complete. Best RMSE: {best_rmse:.2f} dB, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    # --- Exit criteria ---
    v10_rmse = v10_ckpt.get('val_rmse', 2.48)
    v10_pearson = v10_ckpt.get('val_pearson', 0.237)
    rmse_ok = best_rmse <= 2.50
    pearson_ok = best_pearson >= 0.237
    log.info(f"\n  V10 baseline: RMSE {v10_rmse:.4f}, Pearson {v10_pearson:+.4f}")
    log.info(f"  V11 Phase A:  RMSE {best_rmse:.4f}, Pearson {best_pearson:+.4f}")
    log.info(f"  RMSE ≤ 2.50:   {'PASS' if rmse_ok else 'FAIL'}")
    log.info(f"  Pearson ≥ 0.237: {'PASS' if pearson_ok else 'FAIL'}")

    # --- Physics verification ---
    log.info("")
    log.info("=== PHYSICS VERIFICATION ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0)
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    log.info(f"Sun Sidecar: SFI 70→200 benefit: {sfi_200 - sfi_70:+.2f} dB")

    kp0 = model.get_storm_effect(1.0)
    kp9 = model.get_storm_effect(0.0)
    log.info(f"Storm Sidecar: Kp 0→9 cost: {kp0 - kp9:+.2f} dB")

    if sfi_200 > sfi_70:
        log.info("SFI monotonicity: CORRECT")
    else:
        log.info("SFI monotonicity: WARNING — inverted or flat")

    if kp0 > kp9:
        log.info("Kp monotonicity: CORRECT")
    else:
        log.info("Kp monotonicity: WARNING — inverted or flat")

    # --- FREQ-01 Quick Check ---
    freq01_quick_check(model)

    log.info(f"\n{'=' * 65}")
    log.info("  Phase A complete. Ready for Phase B (Gate Awakening).")
    log.info(f"{'=' * 65}")


if __name__ == '__main__':
    main()
