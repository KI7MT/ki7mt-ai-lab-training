#!/usr/bin/env python3
"""
train_v9_dual_mono.py — IONIS V2: Phase 9 Dual Monotonic MLP Architecture

Physics-Informed ML with two monotonic sidecars:
  - Sun Sidecar: SFI -> monotonically INCREASING output (more sun = stronger signal)
  - Storm Sidecar: kp_penalty -> monotonically INCREASING output (then sign-flipped)
    (kp_penalty = 1 - kp/9, so increasing kp_penalty = less storm = better signal)

The Deep Brain (DNN) handles geometry, time, season, band — but NOT raw sfi or kp_penalty.
This prevents the DNN from learning inverted solar/storm relationships.

Phase 9 changes from Phase 8:
    CHANGED: Dual monotonic MLPs replace single linear sidecar
    CHANGED: SFI removed from DNN, routed through Sun Sidecar
    CHANGED: Monotonic constraint via positive-only weights + Softplus activations
    CHANGED: HuberLoss (robust to outliers) replaces MSE
    CHANGED: Batch size 8192 (was 4096)
    ADDED:   Physics check prints Kp 9 storm cost and SFI 200 benefit each epoch
    KEPT:    IFW sampling weights

Architecture: IonisDualMono
    DNN:          13 -> 512 -> 256 -> 128 -> 1 (geometry, time, season, interactions)
    Sun Sidecar:  1 -> 8 -> 1 (monotonic increasing, learns SFI benefit curve)
    Storm Sidecar: 1 -> 8 -> 1 (monotonic increasing on kp_penalty)
    Output:       DNN(x_deep) + SunSidecar(sfi) + StormSidecar(kp_penalty)

NUCLEAR OPTION (Starvation Protocol): DNN gets ZERO solar information.
The interaction terms have been removed. ALL solar/storm signal MUST flow
through the monotonic sidecars. The DNN learns only geography and time.
"""

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

# --- CONFIGURATION ---
CSV_PATH = "data/training_v6_clean.csv"
BATCH_SIZE = 8192  # Doubled from Phase 8 per Gemini's recommendation
EPOCHS = 20  # HARD FREEZE TEST
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
MODEL_DIR = 'models'
MODEL_FILE = 'ionis_v9_dual_mono.pth'
PHASE = 'Phase B: Geography Reintegration (DNN slow, sidecars clamped)'

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Feature indices after NUCLEAR OPTION (STARVATION PROTOCOL):
# DNN features (0-10): distance, freq_log, hour_sin, hour_cos, az_sin, az_cos,
#                      lat_diff, midpoint_lat, season_sin, season_cos, day_night_est
# Sun Sidecar (11): sfi
# Storm Sidecar (12): kp_penalty
#
# NUCLEAR OPTION: DNN has ZERO solar/storm information.
# The interaction terms (ssn_lat_interact, band_sfi_interact) have been REMOVED.
# ALL solar/storm signal MUST flow through the monotonic sidecars.
# The DNN can only learn geography and time patterns.

DNN_DIM = 11  # NUCLEAR OPTION: removed ssn_lat_interact, band_sfi_interact
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13  # 11 DNN + 1 sfi + 1 kp_penalty (NUCLEAR OPTION)

FEATURES = [
    # DNN features (11) — NUCLEAR OPTION: geography & time ONLY, zero solar info
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v9')


# --- GRID UTILITIES ---
GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def grid_to_latlon_series(grids):
    """Convert pandas Series of Maidenhead grids -> (lat, lon) arrays."""
    lats = np.zeros(len(grids), dtype=np.float32)
    lons = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        s = str(g).strip().rstrip('\x00')
        m = GRID_RE.search(s)
        if m:
            g4 = m.group(0).upper()
        else:
            g4 = 'JJ00'
        lons[i] = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
        lats[i] = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lats, lons


# --- FEATURE ENGINEERING ---
def engineer_features(df, snr_std):
    """Compute 13 features from raw CSV columns. Returns (N, 13) float32.

    Features 0-10 feed the DNN (geography & time ONLY).
    Feature 11 (sfi) feeds the Sun Sidecar.
    Feature 12 (kp_penalty) feeds the Storm Sidecar.

    NUCLEAR OPTION (STARVATION PROTOCOL):
    - DNN has ZERO solar information (no ssn, no sfi, no interaction terms)
    - ALL solar/storm signal MUST flow through the monotonic sidecars
    - The DNN can only learn geography and time patterns
    """
    n = len(df)

    # Raw columns
    distance = df['distance'].values.astype(np.float32)
    band = df['band'].values.astype(np.int32)
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)
    azimuth = df['azimuth'].values.astype(np.float32)
    ssn = df['ssn'].values.astype(np.float32)
    sfi = df['sfi'].values.astype(np.float32)
    midpoint_lat_raw = df['midpoint_lat'].values.astype(np.float32)
    kp_penalty = df['kp_penalty'].values.astype(np.float32)

    # Grid -> lat/lon for lat_diff and day_night_est
    tx_lat, tx_lon = grid_to_latlon_series(df['tx_grid'].values)
    rx_lat, rx_lon = grid_to_latlon_series(df['rx_grid'].values)

    # Frequency from band ID
    freq_hz = np.array([BAND_TO_HZ.get(int(b), 14_097_100) for b in band], dtype=np.float64)

    # Normalized features
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

    # NUCLEAR OPTION: No interaction features for DNN
    # DNN sees ONLY geography & time, ZERO solar information
    # All solar/storm signal MUST flow through the sidecars

    # Sidecar inputs (normalized)
    f_sfi          = sfi / 300.0      # Sun Sidecar input
    f_kp_penalty   = kp_penalty       # Storm Sidecar input (already 1 - kp/9)

    # Features 0-10: DNN (geography & time only), Feature 11: sfi, Feature 12: kp_penalty
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


# --- MONOTONIC MLP ---
class MonotonicMLP(nn.Module):
    """Small MLP with monotonically increasing output.

    Enforced via:
    1. All weights constrained to be non-negative (using torch.abs or clamping)
    2. Softplus activations (smooth, always positive slope)
    3. No bias in final layer (optional, keeps output anchored)

    This allows learning curves/cliffs while guaranteeing monotonicity.
    """
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)  # Bias enables negative output
        self.activation = nn.Softplus()

    def forward(self, x):
        # Apply absolute value to weights to enforce non-negativity
        # No clamp in forward() - we clamp after optimizer step instead
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(torch.nn.functional.linear(x, w1, self.fc1.bias))
        out = torch.nn.functional.linear(h, w2, self.fc2.bias)
        return out


# --- MODEL ---
class IonisDualMono(nn.Module):
    """Dual Monotonic PIML: Deep MLP + Sun Sidecar + Storm Sidecar.

    The DNN processes geometry, time, season, band (14 features).
    The Sun Sidecar processes sfi only — monotonically increasing.
    The Storm Sidecar processes kp_penalty only — monotonically increasing.
    (kp_penalty = 1 - kp/9, so increasing kp_penalty = quieter = better)

    Output = DNN(x_deep) + SunSidecar(sfi) + StormSidecar(kp_penalty)
    """
    def __init__(self, dnn_dim=14, sidecar_hidden=8):
        super().__init__()
        # The Deep Brain: learns complex patterns from 14 features
        self.dnn = nn.Sequential(
            nn.Linear(dnn_dim, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )
        # Sun Sidecar: sfi -> SNR boost (monotonically increasing)
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        # Storm Sidecar: kp_penalty -> SNR boost (monotonically increasing)
        # Higher kp_penalty (quieter) = more boost
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

    def forward(self, x):
        # Split: features 0-13 -> DNN, feature 14 -> sun, feature 15 -> storm
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX+1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX+1]

        base = self.dnn(x_deep)
        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        # PHASE B: Geography Reintegration - DNN explains location, sidecars explain physics
        return base + sun_boost + storm_boost

    def get_sun_effect(self, sfi_normalized):
        """Get the sun sidecar output for a given normalized sfi value."""
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty):
        """Get the storm sidecar output for a given kp_penalty value."""
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()


# --- METRICS ---
def pearson_r(pred, target):
    """Pearson correlation between two 1-D tensors."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# --- TRAINING ---
def main():
    log.info(f"IONIS V2 | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info("Architecture: IonisDualMono (PHASE B: Geography Reintegration)")
    log.info("  DNN: 11 features -> 512 -> 256 -> 128 -> 1 (geography & time)")
    log.info("  Sun Sidecar: sfi -> MonotonicMLP(8) -> SNR boost (clamped 0.5-2.0)")
    log.info("  Storm Sidecar: kp_penalty -> MonotonicMLP(8) -> SNR boost (clamped 0.5-2.0)")
    log.info("  DNN learns geography, sidecars maintain physics")

    # Load data
    log.info(f"Loading {CSV_PATH}...")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    load_sec = time.perf_counter() - t0
    log.info(f"Loaded {len(df):,} rows in {load_sec:.1f}s")

    # Get target stats first (needed for reporting)
    y_raw = df['snr'].values.astype(np.float32)
    snr_mean = float(y_raw.mean())
    snr_std = float(y_raw.std())

    log.info(f"Engineering {INPUT_DIM} features...")
    t0 = time.perf_counter()
    X_np = engineer_features(df, snr_std)
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
            role = "DNN"
        elif i == SFI_IDX:
            role = "Sun SC"
        else:
            role = "Storm SC"
        log.info(f"  {name:<20s}  {col.min():8.4f}  {col.mean():8.4f}  {col.max():8.4f}  {role:>10s}")

    # Train/val split
    n = len(X_np)
    dataset = WSPRDataset(X_np, y_np, w_np)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # Model
    model = IonisDualMono(dnn_dim=DNN_DIM, sidecar_hidden=8).to(DEVICE)
    dnn_params = sum(p.numel() for p in model.dnn.parameters())
    sun_params = sum(p.numel() for p in model.sun_sidecar.parameters())
    storm_params = sum(p.numel() for p in model.storm_sidecar.parameters())
    log.info(f"DNN: {dnn_params:,} params | Sun SC: {sun_params} params | Storm SC: {storm_params} params")
    log.info(f"Total: {dnn_params + sun_params + storm_params:,} parameters")

    # --- THE DEFIBRILLATOR (V2): Weights Positive, Biases Free ---
    def wake_up_sidecar(layer):
        if isinstance(layer, torch.nn.Linear):
            # Force WEIGHTS into clamp range (0.5 to 2.0) - start at 1.0
            torch.nn.init.uniform_(layer.weight, 0.8, 1.2)
            # Allow BIAS to be zero (or learnable negative) - DO NOT force positive
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

    log.info("Applying Defibrillator V2 to Sidecars...")
    model.sun_sidecar.apply(wake_up_sidecar)
    model.storm_sidecar.apply(wake_up_sidecar)

    # Crucial: Shift the "Zero Point" down so sidecars can output negative values
    # Manually set the FINAL layer's bias to -1.0
    with torch.no_grad():
        # Initialize fc2.bias to bring output into target range (~-16 dB mean)
        # Each sidecar contributes half, so start at -10 each
        model.sun_sidecar.fc2.bias.fill_(-10.0)
        model.storm_sidecar.fc2.bias.fill_(-10.0)

    # CALIBRATED PHYSICS: fc1.bias frozen, fc2.bias LEARNABLE (relief valve)
    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True   # Relief valve - can shift output
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True  # Relief valve - can shift output

    log.info("Sidecars are JUMP STARTED (and Centered).")
    log.info("  CALIBRATED: fc1.bias frozen, fc2.bias LEARNABLE (relief valve)")

    # DIAGNOSTIC: Show initial sidecar state
    with torch.no_grad():
        sun_fc1_w = model.sun_sidecar.fc1.weight.abs().mean().item()
        sun_fc2_w = model.sun_sidecar.fc2.weight.abs().mean().item()
        sun_fc2_b = model.sun_sidecar.fc2.bias.item()
        storm_fc1_w = model.storm_sidecar.fc1.weight.abs().mean().item()
        storm_fc2_w = model.storm_sidecar.fc2.weight.abs().mean().item()
        storm_fc2_b = model.storm_sidecar.fc2.bias.item()

        # Test sidecar outputs at extremes
        test_low = torch.tensor([[0.1]], device=DEVICE)
        test_high = torch.tensor([[0.9]], device=DEVICE)
        sun_out_low = model.sun_sidecar(test_low).item()
        sun_out_high = model.sun_sidecar(test_high).item()
        storm_out_low = model.storm_sidecar(test_low).item()
        storm_out_high = model.storm_sidecar(test_high).item()

    log.info(f"  Sun SC init: fc1_w={sun_fc1_w:.3f}, fc2_w={sun_fc2_w:.3f}, fc2_b={sun_fc2_b:.3f}")
    log.info(f"  Sun SC output: low(0.1)={sun_out_low:.3f}, high(0.9)={sun_out_high:.3f}, delta={sun_out_high-sun_out_low:.3f}")
    log.info(f"  Storm SC init: fc1_w={storm_fc1_w:.3f}, fc2_w={storm_fc2_w:.3f}, fc2_b={storm_fc2_b:.3f}")
    log.info(f"  Storm SC output: low(0.1)={storm_out_low:.3f}, high(0.9)={storm_out_high:.3f}, delta={storm_out_high-storm_out_low:.3f}")
    # -----------------------------------------------

    # ---------------------------------------------------------
    # STRATEGY 2: DIFFERENTIAL LEARNING RATES
    # ---------------------------------------------------------
    # DNN learns SLOWLY (to reduce noise without stealing signal)
    # Sidecars learn FAST (to aggressively seize the Global Signal)
    log.info("Initializing Optimizer: PHASE B - Geography Reintegration")
    log.info("  DNN: 1e-5 (slow student) | Sidecars: 1e-3 (maintain physics)")
    optimizer = optim.AdamW([
        # The Deep Brain: Slow student (1e-5) - learns geography without bullying
        {'params': model.dnn.parameters(), 'lr': 1e-5},
        # The Sidecars: Faster (1e-3) to maintain their Physical Truth
        {'params': model.sun_sidecar.parameters(), 'lr': 1e-3},
        {'params': model.storm_sidecar.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-5)
    # ---------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.HuberLoss(reduction='none', delta=1.0)  # Robust to outliers

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')

    # Training
    log.info("Training started (Dual Monotonic Sidecars Active)")
    log.info("Physics check: SFI 200 benefit | Kp 9 storm cost (in dB)")
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

            # POST-OPTIMIZER WEIGHT CLAMP: Keep weights in sane range (0.5 to 2.0)
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

        # Physics check: compute sidecar effects at extreme values
        # SFI 200 (normalized: 200/300 = 0.667) vs SFI 70 (0.233)
        sfi_high = model.get_sun_effect(200.0 / 300.0)
        sfi_low = model.get_sun_effect(70.0 / 300.0)
        sfi_benefit = sfi_high - sfi_low  # Positive = high SFI helps

        # Kp 0 (kp_penalty = 1.0) vs Kp 9 (kp_penalty = 0.0)
        kp_quiet = model.get_storm_effect(1.0)
        kp_storm = model.get_storm_effect(0.0)
        storm_cost = kp_quiet - kp_storm  # Positive = storm hurts

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        epoch_sec = time.perf_counter() - t_epoch

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
                'solar_resolution': '3-hourly Kp, daily SFI/SSN (GFZ Potsdam)',
                'activation': 'Mish (DNN), Softplus (sidecars)',
                'batchnorm': False,
                'sampling': 'IFW (Efraimidis-Spirakis, 2D SSN×lat density)',
                'data_source': 'wspr.gold_v6 (IFW + kp_penalty)',
                'architecture': 'IonisDualMono (DNN 14->512->256->128->1 + 2x MonotonicMLP)',
                'monotonic_constraint': 'abs(weights) + Softplus in sidecars',
                'sfi_benefit_dB': sfi_benefit,
                'storm_cost_dB': storm_cost,
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.2f}dB  {val_pearson:+7.4f}  "
            f"{lr_now:.2e}  {sfi_benefit:+4.1f}  {storm_cost:+4.1f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        # DIAGNOSTIC: Detailed sidecar state after each epoch
        with torch.no_grad():
            sun_fc1_w = model.sun_sidecar.fc1.weight.abs().mean().item()
            sun_fc2_w = model.sun_sidecar.fc2.weight.abs().mean().item()
            sun_fc2_b = model.sun_sidecar.fc2.bias.item()
            storm_fc1_w = model.storm_sidecar.fc1.weight.abs().mean().item()
            storm_fc2_w = model.storm_sidecar.fc2.weight.abs().mean().item()
            storm_fc2_b = model.storm_sidecar.fc2.bias.item()
        log.info(f"      Sun: fc1={sun_fc1_w:.4f} fc2={sun_fc2_w:.4f} b={sun_fc2_b:.3f} | "
                 f"Storm: fc1={storm_fc1_w:.4f} fc2={storm_fc2_w:.4f} b={storm_fc2_b:.3f} | "
                 f"SFI raw: {model.get_sun_effect(0.667):.4f}-{model.get_sun_effect(0.233):.4f}={sfi_benefit:.4f}")

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.2f} dB")
    log.info(f"Checkpoint: {model_path}")

    # Final physics report
    log.info("")
    log.info("=== PHYSICS VERIFICATION ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0)
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    log.info(f"Sun Sidecar: SFI 70 -> {sfi_70:+.2f} dB, SFI 200 -> {sfi_200:+.2f} dB")
    log.info(f"  SFI 70->200 benefit: {sfi_200 - sfi_70:+.2f} dB")

    kp0 = model.get_storm_effect(1.0)  # kp_penalty=1.0 = Kp 0
    kp5 = model.get_storm_effect(1.0 - 5.0/9.0)
    kp9 = model.get_storm_effect(0.0)  # kp_penalty=0.0 = Kp 9
    log.info(f"Storm Sidecar: Kp 0 -> {kp0:+.2f} dB, Kp 5 -> {kp5:+.2f} dB, Kp 9 -> {kp9:+.2f} dB")
    log.info(f"  Kp 0->9 storm cost: {kp0 - kp9:+.2f} dB")

    if sfi_200 > sfi_70:
        log.info("SFI monotonicity: CORRECT (higher SFI = stronger signal)")
    else:
        log.info("SFI monotonicity: WARNING - inverted or flat")

    if kp0 > kp9:
        log.info("Kp monotonicity: CORRECT (storms degrade signal)")
    else:
        log.info("Kp monotonicity: WARNING - inverted or flat")


if __name__ == '__main__':
    main()
