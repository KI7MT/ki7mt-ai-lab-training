#!/usr/bin/env python3
"""
train_v11_final.py — IONIS V11 "Gatekeeper" Final Training

Architecture: IonisV11Gate (203,573 params)
    Trunk:              11 → 512 → 256 (shared representation)
    Base Head:          256 → 128 → 1  (geography/time baseline SNR)
    Sun Scaler Head:    256 → 64  → 1  (geographic SFI modulation)
    Storm Scaler Head:  256 → 64  → 1  (geographic Kp modulation)
    Sun Sidecar:        1 → 8 → 1      (monotonic, weights clamped 0.5-2.0)
    Storm Sidecar:      1 → 8 → 1      (monotonic, weights clamped 0.5-2.0)

    Output = base_snr + gate(sun_logit) * SunSidecar(sfi)
                      + gate(storm_logit) * StormSidecar(kp_penalty)

    gate(x) = 0.5 + 1.5 * sigmoid(x)   →   [0.5, 2.0]

What's new vs V10:
    V10: output = DNN(geo) + SunSidecar(sfi) + StormSidecar(kp)
         → Sidecar outputs are global constants (same storm cost everywhere)

    V11: output = base_snr + sun_gate * SunSidecar(sfi)
                           + storm_gate * StormSidecar(kp)
         → Gates modulate sidecar outputs per-sample based on geography
         → Polar paths can have stronger storm effects (storm_gate > 1)
         → Equatorial paths can have weaker storm effects (storm_gate < 1)
         → High-frequency paths can have stronger SFI benefit (sun_gate > 1)

Key constraints (unchanged from V10):
    - DNN (trunk+heads) has ZERO solar information (Starvation Protocol)
    - Sidecar weights clamped [0.5, 2.0] (prevents collapse AND explosion)
    - Sidecar fc1.bias frozen, fc2.bias learnable (relief valve)
    - Gate function bounded [0.5, 2.0] — can never reverse sidecar direction

Differential LR:
    Trunk + Base Head:  1e-5  (slow — geography learning)
    Scaler Heads:       5e-5  (moderate — gate differentiation)
    Sidecars:           1e-3  (fast — physics calibration)

Anti-collapse regularization:
    L_var = -λ * (Var(sun_gate) + Var(storm_gate))
    Prevents gates from collapsing to a constant (which would reduce V11 to V10)
"""

import math
import os
import re
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(TRAINING_DIR, "data", "training_v6_clean.csv")
MODEL_DIR = os.path.join(TRAINING_DIR, "models")
MODEL_FILE = "ionis_v11_final.pth"
PHASE = "V11 Final: Gatekeeper (trunk+3heads+2sidecars, gate-modulated physics)"

BATCH_SIZE = 8192
EPOCHS = 100
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
LAMBDA_VAR = 0.001  # Anti-collapse variance loss weight

# Turbo Loader for M3 Ultra
NUM_WORKERS = 12
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Feature layout (unchanged from V10 — Starvation Protocol)
DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

FEATURES = [
    # DNN features (11) — geography & time ONLY, zero solar info
    'distance', 'freq_log', 'hour_sin', 'hour_cos',
    'az_sin', 'az_cos', 'lat_diff', 'midpoint_lat',
    'season_sin', 'season_cos', 'day_night_est',
    # Sidecar features (2) — the ONLY solar/storm inputs
    'sfi', 'kp_penalty',
]

BAND_TO_HZ = {
    102:  1_836_600,   103:  3_568_600,   104:  5_287_200,
    105:  7_038_600,   106: 10_138_700,   107: 14_097_100,
    108: 18_104_600,   109: 21_094_600,   110: 24_924_600,
    111: 28_124_600,
}

GATE_INIT_BIAS = -math.log(2.0)  # ≈ -0.693 → gate(x) ≈ 1.0 at init

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v11')


# ── Grid Utilities ───────────────────────────────────────────────────────────

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def grid_to_latlon_series(grids):
    """Convert array of Maidenhead grids → (lat, lon) arrays."""
    lats = np.zeros(len(grids), dtype=np.float32)
    lons = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        s = str(g).strip().rstrip('\x00')
        m = GRID_RE.search(s)
        g4 = m.group(0).upper() if m else 'JJ00'
        lons[i] = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
        lats[i] = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lats, lons


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df):
    """Compute 13 features from raw CSV. Returns (N, 13) float32.

    Features 0-10 feed the trunk (geography & time ONLY).
    Feature 11 (sfi) feeds the Sun Sidecar.
    Feature 12 (kp_penalty) feeds the Storm Sidecar.

    STARVATION PROTOCOL: trunk sees ZERO solar info.
    ALL solar/storm signal MUST flow through the gated sidecars.
    """
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

    freq_hz = np.array([BAND_TO_HZ.get(int(b), 14_097_100) for b in band],
                       dtype=np.float64)

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


# ── Dataset ──────────────────────────────────────────────────────────────────

class WSPRDataset(Dataset):
    def __init__(self, features, targets, weights):
        self.x = features
        self.y = targets
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


# ── Model ────────────────────────────────────────────────────────────────────

class MonotonicMLP(nn.Module):
    """Small MLP with monotonically increasing output.

    Weights forced non-negative via abs() in forward pass,
    then clamped [0.5, 2.0] after each optimizer step.
    """
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
    """Bounded gate: (-inf, +inf) → [0.5, 2.0].

    gate(x) = 0.5 + 1.5 * sigmoid(x)

    At x = -ln(2): sigmoid = 1/3 → gate = 1.0 (V10-equivalent at init).
    """
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV11Gate(nn.Module):
    """IONIS V11 "Gatekeeper" — Multiplicative Interaction Gates.

    output = base_snr + gate(sun_logit) * SunSidecar(sfi)
                      + gate(storm_logit) * StormSidecar(kp_penalty)

    The shared trunk learns a joint representation from 11 geography/time
    features. Three heads decode this into:
      - base_snr:    the geography/time baseline prediction
      - sun_logit:   raw logit → gate → scales sun sidecar output
      - storm_logit: raw logit → gate → scales storm sidecar output

    Gates allow the model to learn that:
      - Polar paths experience stronger storm effects (storm_gate > 1)
      - Equatorial paths are less affected by storms (storm_gate < 1)
      - High-band paths see more SFI benefit (sun_gate > 1)
    """
    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()

        # Shared trunk: geography/time → joint representation
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        # Base head: trunk → base SNR (same role as V10 DNN output)
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )

        # Sun scaler head: trunk → raw logit for sun gate
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # Storm scaler head: trunk → raw logit for storm gate
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # Sidecars: unchanged from V10 (monotonic physics)
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        # Initialize scaler heads so gate ≈ 1.0 at start (V10-equivalent)
        self._init_scaler_heads()

    def _init_scaler_heads(self):
        """Zero-init final weights, bias = -ln(2) → gate ≈ 1.0."""
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
        """Forward pass that also returns gate values for diagnostics."""
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
        """Get raw sun sidecar output for a given normalized SFI."""
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty):
        """Get raw storm sidecar output for a given kp_penalty."""
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        """Return (sun_gate, storm_gate) tensors for diagnostics."""
        x_deep = x[:, :DNN_DIM]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate(sun_logit), _gate(storm_logit)


# ── Metrics ──────────────────────────────────────────────────────────────────

def pearson_r(pred, target):
    """Pearson correlation between two 1-D tensors."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    log.info(f"IONIS V11 | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Turbo Loader: workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}")
    log.info("Architecture: IonisV11Gate (trunk+3heads+2sidecars)")
    log.info("  Trunk:       11 → 512 → 256 (shared geography representation)")
    log.info("  Base Head:   256 → 128 → 1  (baseline SNR)")
    log.info("  Sun Scaler:  256 → 64 → 1   (geographic SFI modulation)")
    log.info("  Storm Scaler:256 → 64 → 1   (geographic Kp modulation)")
    log.info("  Sun Sidecar: 1 → 8 → 1      (monotonic, clamped [0.5, 2.0])")
    log.info("  Storm Sidecar:1 → 8 → 1     (monotonic, clamped [0.5, 2.0])")
    log.info("  Gate: gate(x) = 0.5 + 1.5*sigmoid(x) → [0.5, 2.0]")
    log.info("Differential LR: trunk/base 1e-5 | scalers 5e-5 | sidecars 1e-3")
    log.info(f"Anti-collapse: λ_var = {LAMBDA_VAR}")

    # ── Load Data ──
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

    # Diagnostics
    log.info(f"Dataset: {len(X_np):,} rows x {INPUT_DIM} features")
    log.info(f"SNR range: {y_np.min():.0f} to {y_np.max():.0f} dB")
    log.info(f"SNR mean: {snr_mean:.1f} dB, std: {snr_std:.1f} dB")
    log.info("Feature statistics (normalized):")
    log.info(f"  {'Feature':<20s}  {'Min':>8s}  {'Mean':>8s}  {'Max':>8s}  {'Role':>10s}")
    log.info(f"  {'-' * 62}")
    for i, name in enumerate(FEATURES):
        col = X_np[:, i]
        if i < DNN_DIM:
            role = "Trunk"
        elif i == SFI_IDX:
            role = "Sun SC"
        else:
            role = "Storm SC"
        log.info(f"  {name:<20s}  {col.min():8.4f}  {col.mean():8.4f}  "
                 f"{col.max():8.4f}  {role:>10s}")

    # ── Train/Val Split ──
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

    # ── Model ──
    model = IonisV11Gate(dnn_dim=DNN_DIM, sidecar_hidden=8).to(DEVICE)

    trunk_params = sum(p.numel() for p in model.trunk.parameters())
    base_params = sum(p.numel() for p in model.base_head.parameters())
    sun_sc_params = sum(p.numel() for p in model.sun_scaler_head.parameters())
    storm_sc_params = sum(p.numel() for p in model.storm_scaler_head.parameters())
    sun_side_params = sum(p.numel() for p in model.sun_sidecar.parameters())
    storm_side_params = sum(p.numel() for p in model.storm_sidecar.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    log.info(f"\nParameter Budget:")
    log.info(f"  Trunk (11→512→256):       {trunk_params:>8,}")
    log.info(f"  Base head (256→128→1):    {base_params:>8,}")
    log.info(f"  Sun scaler (256→64→1):    {sun_sc_params:>8,}")
    log.info(f"  Storm scaler (256→64→1):  {storm_sc_params:>8,}")
    log.info(f"  Sun sidecar (1→8→1):      {sun_side_params:>8,}")
    log.info(f"  Storm sidecar (1→8→1):    {storm_side_params:>8,}")
    log.info(f"  {'─' * 38}")
    log.info(f"  Total:                    {total_params:>8,}")

    # ── Defibrillator: Init Sidecars ──
    def wake_up_sidecar(layer):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, 0.8, 1.2)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    log.info("\nApplying Defibrillator to Sidecars...")
    model.sun_sidecar.apply(wake_up_sidecar)
    model.storm_sidecar.apply(wake_up_sidecar)

    # Set fc2.bias to bring sidecar output into target range
    with torch.no_grad():
        model.sun_sidecar.fc2.bias.fill_(-10.0)
        model.storm_sidecar.fc2.bias.fill_(-10.0)

    # CALIBRATED PHYSICS: fc1.bias frozen, fc2.bias learnable (relief valve)
    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Sidecars jump-started:")
    log.info("  fc1.bias frozen | fc2.bias learnable (relief valve)")
    log.info(f"  Trainable: {trainable:,} / {total_params:,}")

    # Initial gate check
    log.info("\nGate initialization check (should be ≈ 1.0):")
    x_test = torch.randn(100, INPUT_DIM, device=DEVICE)
    sun_gate, storm_gate = model.get_gates(x_test)
    log.info(f"  Sun gate:   mean={sun_gate.mean():.4f}, std={sun_gate.std():.6f}")
    log.info(f"  Storm gate: mean={storm_gate.mean():.4f}, std={storm_gate.std():.6f}")

    # Initial sidecar state
    with torch.no_grad():
        sun_fc2_b = model.sun_sidecar.fc2.bias.item()
        storm_fc2_b = model.storm_sidecar.fc2.bias.item()
        sfi_delta = model.get_sun_effect(0.667) - model.get_sun_effect(0.233)
        kp_delta = model.get_storm_effect(1.0) - model.get_storm_effect(0.0)
    log.info(f"  Sidecar bias: Sun={sun_fc2_b:.1f}, Storm={storm_fc2_b:.1f}")
    log.info(f"  Initial SFI delta={sfi_delta:.2f} dB, Kp delta={kp_delta:.2f} dB")

    # ── Optimizer: Differential LR ──
    log.info("\nOptimizer: Differential LR (AdamW)")
    log.info("  Trunk + Base Head: 1e-5  (slow geography)")
    log.info("  Scaler Heads:      5e-5  (moderate gate learning)")
    log.info("  Sidecars:          1e-3  (fast physics calibration)")

    optimizer = optim.AdamW([
        {'params': model.trunk.parameters(), 'lr': 1e-5},
        {'params': model.base_head.parameters(), 'lr': 1e-5},
        {'params': model.sun_scaler_head.parameters(), 'lr': 5e-5},
        {'params': model.storm_scaler_head.parameters(), 'lr': 5e-5},
        {'params': [p for p in model.sun_sidecar.parameters() if p.requires_grad],
         'lr': 1e-3},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad],
         'lr': 1e-3},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.HuberLoss(reduction='none', delta=1.0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')
    best_pearson = -1.0

    # ── Training Loop ──
    log.info(f"\nTraining started ({EPOCHS} epochs)")
    log.info("Physics: SFI 200 benefit | Kp 9 storm cost (dB)")
    log.info("Gates: sun/storm gate mean values from validation")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  "
           f"{'SunG':>6s}  {'StmG':>6s}  "
           f"{'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.perf_counter()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        epoch_sun_var_sum = 0.0

        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()

            out, sun_gate_b, storm_gate_b = model.forward_with_gates(bx)

            # Primary loss (weighted Huber)
            primary_loss = (criterion(out, by) * bw).mean()

            # Anti-collapse: penalize constant gate values
            sun_var = sun_gate_b.var()
            storm_var = storm_gate_b.var()
            var_loss = -LAMBDA_VAR * (sun_var + storm_var)

            loss = primary_loss + var_loss
            loss.backward()
            optimizer.step()

            # Post-optimizer weight clamp on sidecars
            with torch.no_grad():
                for sidecar in [model.sun_sidecar, model.storm_sidecar]:
                    sidecar.fc1.weight.clamp_(0.5, 2.0)
                    sidecar.fc2.weight.clamp_(0.5, 2.0)

            train_loss_sum += primary_loss.item()
            train_batches += 1
            epoch_sun_var_sum += sun_var.item()

        train_loss = train_loss_sum / train_batches

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        val_sun_gates = []
        val_storm_gates = []

        with torch.no_grad():
            for bx, by, bw in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out, sun_g, storm_g = model.forward_with_gates(bx)
                loss = criterion(out, by).mean()
                val_loss_sum += loss.item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())
                val_sun_gates.append(sun_g.cpu())
                val_storm_gates.append(storm_g.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        val_pearson = pearson_r(preds_cat, targets_cat)

        sun_gate_mean = torch.cat(val_sun_gates).mean().item()
        storm_gate_mean = torch.cat(val_storm_gates).mean().item()

        # Physics check
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
                'monotonic_constraint': 'abs(weights) + Softplus, clamp [0.5, 2.0]',
                'gate_function': 'gate(x) = 0.5 + 1.5*sigmoid(x) → [0.5, 2.0]',
                'sfi_benefit_dB': sfi_benefit,
                'storm_cost_dB': storm_cost,
                'sun_gate_mean': sun_gate_mean,
                'storm_gate_mean': storm_gate_mean,
                'lambda_var': LAMBDA_VAR,
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.2f}dB  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.1f}  {storm_cost:+4.1f}  "
            f"{sun_gate_mean:6.4f}  {storm_gate_mean:6.4f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        # Sidecar check every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                sun_fc2_b = model.sun_sidecar.fc2.bias.item()
                storm_fc2_b = model.storm_sidecar.fc2.bias.item()
            sun_gate_all = torch.cat(val_sun_gates)
            storm_gate_all = torch.cat(val_storm_gates)
            log.info(f"      Sidecar bias: Sun={sun_fc2_b:.2f}, Storm={storm_fc2_b:.2f}")
            log.info(f"      Sun gate:  min={sun_gate_all.min():.4f}, "
                     f"max={sun_gate_all.max():.4f}, std={sun_gate_all.std():.4f}")
            log.info(f"      Storm gate: min={storm_gate_all.min():.4f}, "
                     f"max={storm_gate_all.max():.4f}, std={storm_gate_all.std():.4f}")

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f} dB, "
             f"Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    # ── Final Physics Report ──
    log.info("")
    log.info("=== PHYSICS VERIFICATION ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0)
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    kp0 = model.get_storm_effect(1.0)
    kp5 = model.get_storm_effect(1.0 - 5.0 / 9.0)
    kp9 = model.get_storm_effect(0.0)

    log.info(f"Sun Sidecar: SFI 70 → {sfi_70:+.2f} dB, SFI 200 → {sfi_200:+.2f} dB")
    log.info(f"  SFI 70→200 benefit: {sfi_200 - sfi_70:+.2f} dB")
    log.info(f"Storm Sidecar: Kp 0 → {kp0:+.2f} dB, Kp 5 → {kp5:+.2f} dB, "
             f"Kp 9 → {kp9:+.2f} dB")
    log.info(f"  Kp 0→9 storm cost: {kp0 - kp9:+.2f} dB")

    if sfi_200 > sfi_70:
        log.info("SFI monotonicity: CORRECT (higher SFI = stronger signal)")
    else:
        log.info("SFI monotonicity: WARNING — inverted or flat")

    if kp0 > kp9:
        log.info("Kp monotonicity: CORRECT (storms degrade signal)")
    else:
        log.info("Kp monotonicity: WARNING — inverted or flat")

    log.info("")
    log.info("=== GATE SUMMARY ===")
    log.info(f"Sun gate mean:   {sun_gate_mean:.4f}")
    log.info(f"Storm gate mean: {storm_gate_mean:.4f}")
    log.info("Gates ≈ 1.0 means V10-equivalent behavior.")
    log.info("Spread from 1.0 means the model is using geographic modulation.")


if __name__ == '__main__':
    main()
