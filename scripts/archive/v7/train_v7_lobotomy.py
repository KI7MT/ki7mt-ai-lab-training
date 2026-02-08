#!/usr/bin/env python3
"""
train_v7_lobotomy.py — IONIS V2: Phase 7 Kp Lobotomy Training

Reads pre-materialized CSV from data/training_v6_clean.csv (exported from
wspr.gold_v6 on 9975WX). Raw kp feature REMOVED — all geomagnetic
logic forced through the monotonic-constrained kp_penalty channel.

Phase 7 changes from Phase 6:
    REMOVED: Raw kp feature (was index 14) — closed the collinearity backdoor
    KEPT:    kp_penalty = 1.0 - kp/9.0 as sole Kp channel (now index 16)
    KEPT:    Monotonic constraint: first-layer weights for kp_penalty clamped >= 0
    KEPT:    IFW sampling weights (applied via per-sample weighted MSE loss)

Architecture: 17 -> 512 -> 256 -> 128 -> 1 (Mish, no BatchNorm)
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
BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
INPUT_DIM = 17
MODEL_DIR = 'models'
MODEL_FILE = 'ionis_v7_lobotomy.pth'
PHASE = 'Phase 7: Kp Lobotomy (17 features, raw kp removed, IFW weighted MSE)'

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# kp_penalty is now index 16 (last feature, 0-indexed) after removing raw kp.
# It represents (1.0 - Kp/9). 1.0 = Quiet, 0.0 = Storm.
# We want Positive Weight on this feature so that Lower Penalty -> Lower SNR.
KP_PENALTY_IDX = 16

FEATURES = [
    'distance', 'freq_log', 'hour_sin', 'hour_cos',
    'ssn', 'az_sin', 'az_cos', 'lat_diff', 'midpoint_lat',
    'season_sin', 'season_cos', 'ssn_lat_interact',
    'day_night_est', 'sfi',
    # 'kp',  <-- REMOVED: collinearity backdoor closed
    'band_sfi_interact',
    'sfi_dist_interact', 'kp_penalty',
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
log = logging.getLogger('ionis-v7')


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
def engineer_features(df):
    """Compute 17 normalized features from raw CSV columns. Returns (N, 17) float32.

    Raw kp is NOT included — only kp_penalty (1.0 - kp/9.0) carries
    geomagnetic information, guarded by the monotonic constraint.
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
    sfi_dist_raw = df['sfi_dist_interact'].values.astype(np.float32)
    kp_penalty = df['kp_penalty'].values.astype(np.float32)

    # Grid -> lat/lon for lat_diff and day_night_est
    tx_lat, tx_lon = grid_to_latlon_series(df['tx_grid'].values)
    rx_lat, rx_lon = grid_to_latlon_series(df['rx_grid'].values)

    # Frequency from band ID
    freq_hz = np.array([BAND_TO_HZ.get(int(b), 14_097_100) for b in band], dtype=np.float64)

    # Normalized features (must match train_v2_core.py conventions)
    f_distance     = distance / 20000.0
    f_freq_log     = np.log10(freq_hz.astype(np.float32)) / 8.0
    f_hour_sin     = np.sin(2.0 * np.pi * hour / 24.0)
    f_hour_cos     = np.cos(2.0 * np.pi * hour / 24.0)
    f_ssn          = ssn / 300.0
    f_az_sin       = np.sin(2.0 * np.pi * azimuth / 360.0)
    f_az_cos       = np.cos(2.0 * np.pi * azimuth / 360.0)
    f_lat_diff     = np.abs(tx_lat - rx_lat) / 180.0
    f_midpoint_lat = midpoint_lat_raw / 90.0
    f_season_sin   = np.sin(2.0 * np.pi * month / 12.0)
    f_season_cos   = np.cos(2.0 * np.pi * month / 12.0)
    f_ssn_lat      = f_ssn * np.abs(f_midpoint_lat)
    midpoint_lon   = (tx_lon + rx_lon) / 2.0
    local_solar_h  = hour + midpoint_lon / 15.0
    f_daynight     = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    f_sfi          = sfi / 300.0
    # f_kp REMOVED — no raw kp feature
    f_band_sfi     = f_sfi * f_freq_log
    f_sfi_dist     = sfi_dist_raw / (300.0 * np.log10(18000.0))
    f_kp_penalty   = kp_penalty  # already 1.0 - kp/9.0

    return np.column_stack([
        f_distance, f_freq_log, f_hour_sin, f_hour_cos, f_ssn,
        f_az_sin, f_az_cos, f_lat_diff, f_midpoint_lat,
        f_season_sin, f_season_cos, f_ssn_lat, f_daynight,
        f_sfi, f_band_sfi, f_sfi_dist, f_kp_penalty,
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


# --- MODEL ---
class IonisMish(nn.Module):
    """Mish MLP: 17 -> 512 -> 256 -> 128 -> 1 (no BatchNorm)."""
    def __init__(self, input_dim=17):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.fc_layers(x)


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
    log.info(f"Monotonic constraint: kp_penalty (idx {KP_PENALTY_IDX}) weights clamped >= 0")
    log.info("Raw kp feature REMOVED — sole Kp channel is kp_penalty")

    # Load and engineer features
    log.info(f"Loading {CSV_PATH}...")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    load_sec = time.perf_counter() - t0
    log.info(f"Loaded {len(df):,} rows in {load_sec:.1f}s")

    log.info("Engineering 17 features (raw kp excluded)...")
    t0 = time.perf_counter()
    X_np = engineer_features(df)
    y_np = df['snr'].values.astype(np.float32).reshape(-1, 1)
    w_np = df['sampling_weight'].values.astype(np.float32).reshape(-1, 1)
    feat_sec = time.perf_counter() - t0
    log.info(f"Feature engineering complete in {feat_sec:.1f}s")

    # Diagnostics
    snr_mean = float(y_np.mean())
    snr_std = float(y_np.std())
    log.info(f"Dataset: {len(X_np):,} rows x {INPUT_DIM} features")
    log.info(f"SNR range: {y_np.min():.0f} to {y_np.max():.0f} dB")
    log.info(f"SNR mean: {snr_mean:.1f} dB, std: {snr_std:.1f} dB")
    log.info("Feature statistics (normalized):")
    log.info(f"  {'Feature':<20s}  {'Min':>8s}  {'Mean':>8s}  {'Max':>8s}")
    log.info(f"  {'-' * 50}")
    for i, name in enumerate(FEATURES):
        col = X_np[:, i]
        log.info(f"  {name:<20s}  {col.min():8.4f}  {col.mean():8.4f}  {col.max():8.4f}")

    # SSN-SNR correlation
    ssn_raw = df['ssn'].values.astype(np.float32)
    corr = np.corrcoef(ssn_raw, y_np.flatten())[0, 1]
    log.info(f"SSN-SNR Pearson correlation: {corr:+.4f}")

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
    model = IonisMish(input_dim=INPUT_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: 17 -> 512 -> 256 -> 128 -> 1  ({params:,} parameters)")
    log.info(f"Activation: Mish (no BatchNorm)")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss(reduction='none')  # per-sample for IFW weighting

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')

    # Training
    log.info("Training started (Monotonic Kp Constraint Active, Raw Kp Removed)")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  {'LR':>10s}  {'Time':>6s}")
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
            per_sample = criterion(out, by)        # (B, 1) per-sample MSE
            loss = (per_sample * bw).mean()         # IFW-weighted mean
            loss.backward()
            optimizer.step()

            # --- MONOTONIC CONSTRAINT ---
            # Clamp first-layer weights for kp_penalty (idx 16) to >= 0.
            # With raw kp removed, this is the ONLY Kp channel — no backdoor.
            with torch.no_grad():
                model.fc_layers[0].weight[:, KP_PENALTY_IDX].clamp_(min=0.0)

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
                loss = criterion(out, by).mean()    # unweighted val metric
                val_loss_sum += loss.item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        val_pearson = pearson_r(preds_cat, targets_cat)

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
                'input_dim': INPUT_DIM,
                'hidden_dims': [512, 256, 128],
                'features': FEATURES,
                'band_to_hz': BAND_TO_HZ,
                'date_range': '2020-01-01 to 2026-02-04',
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'solar_resolution': '3-hourly Kp, daily SFI/SSN (GFZ Potsdam)',
                'activation': 'Mish',
                'batchnorm': False,
                'sampling': 'IFW (Efraimidis-Spirakis, 2D SSN×lat density)',
                'data_source': 'wspr.gold_v6 (IFW + kp_penalty)',
                'monotonic_constraint': 'kp_penalty (idx 16) weights >= 0, raw kp REMOVED',
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.2f}dB  {val_pearson:+7.4f}  "
            f"{lr_now:.2e}  {epoch_sec:5.1f}s{marker}"
        )

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.2f} dB")
    log.info(f"Checkpoint: {model_path}")


if __name__ == '__main__':
    main()
