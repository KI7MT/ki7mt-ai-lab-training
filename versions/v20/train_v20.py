#!/usr/bin/env python3
"""
train_v20.py — IONIS V20 "Golden Master" Replication

V20 = V16 Architecture + V16 Data Recipe + V19 Code Structure

This is a strict replication of V16 using the clean train_common.py module.
The purpose is to prove we can reproduce V16's results in the new codebase.

SUCCESS CRITERIA:
    - Pearson > +0.48 (V16: +0.4873)
    - Kp sidecar > +3.0σ (V16: +3.445σ)
    - SFI sidecar > +0.4σ (V16: +0.478σ)

THE V16 PHYSICS LAWS (non-negotiable):
    1. Architecture: IonisV12Gate (context-aware gates from trunk output)
    2. Loss: HuberLoss(delta=1.0) — robust to synthetic anchors
    3. Regularization: Gate variance loss — forces context sensitivity
    4. Init: Defibrillator — weights uniform(0.8-1.2), fc2.bias=-10
    5. Constraint: Weight clamp [0.5, 2.0] after EVERY step
    6. Data: V16 recipe (WSPR + RBN-DX + Contest, NO RBN-Full)

Checkpoint: versions/v20/ionis_v20.pth
"""

import gc
import json
import math
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import clickhouse_connect

# Add parent to path for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, VERSIONS_DIR)

from common.train_common import (
    IonisV12Gate,
    init_v16_defibrillator,
    clamp_v16_sidecars,
    get_v16_optimizer_groups,
    grid4_to_latlon_arrays,
)

# ── Configuration ─────────────────────────────────────────────────────────────

VERSION = "v20"
PHASE = "V20 Golden Master: V16 Replication"

# ClickHouse
CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

# V16 Data Recipe (DO NOT CHANGE)
WSPR_SAMPLE = 20_000_000       # 20M WSPR signatures
RBN_FULL_SAMPLE = 0            # ZERO RBN Full (V16 law)
RBN_DX_UPSAMPLE = 50           # 91K × 50 = 4.55M
CONTEST_UPSAMPLE = 1           # 6.34M as-is

# Training
BATCH_SIZE = 65536
EPOCHS = 100
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
LAMBDA_VAR = 0.001             # Gate variance weight
HUBER_DELTA = 1.0

# Learning rates (V16 spec)
LR_TRUNK = 1e-5
LR_SCALER = 5e-5
LR_SIDECAR = 1e-3

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model dimensions
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

# V16 per-source per-band normalization constants
NORM_CONSTANTS = {
    102: {'wspr': (-18.04, 6.9), 'rbn': (11.86, 6.23)},
    103: {'wspr': (-17.90, 6.9), 'rbn': (16.94, 6.71)},
    104: {'wspr': (-17.60, 7.1), 'rbn': (16.91, 7.1)},
    105: {'wspr': (-17.34, 6.6), 'rbn': (18.36, 6.6)},
    106: {'wspr': (-18.07, 6.5), 'rbn': (16.96, 6.5)},
    107: {'wspr': (-17.53, 6.7), 'rbn': (18.47, 6.7)},
    108: {'wspr': (-18.35, 7.0), 'rbn': (16.66, 7.0)},
    109: {'wspr': (-18.32, 6.6), 'rbn': (16.37, 6.6)},
    110: {'wspr': (-18.76, 6.6), 'rbn': (15.07, 6.6)},
    111: {'wspr': (-17.86, 6.5), 'rbn': (15.79, 6.5)},
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v20')


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    """Compute 13 features from signature columns (V16 spec)."""
    distance = df['avg_distance'].values.astype(np.float32)
    band = df['band'].values.astype(np.int32)
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)
    azimuth = df['avg_azimuth'].values.astype(np.float32)
    avg_sfi = df['avg_sfi'].values.astype(np.float32)
    avg_kp = df['avg_kp'].values.astype(np.float32)

    tx_lat, tx_lon = grid4_to_latlon_arrays(df['tx_grid_4'].values)
    rx_lat, rx_lon = grid4_to_latlon_arrays(df['rx_grid_4'].values)

    midpoint_lat = (tx_lat + rx_lat) / 2.0
    midpoint_lon = (tx_lon + rx_lon) / 2.0

    freq_hz = np.array([BAND_TO_HZ.get(int(b), 14_097_100) for b in band],
                       dtype=np.float64)

    kp_penalty = (1.0 - avg_kp / 9.0).astype(np.float32)

    return np.column_stack([
        distance / 20000.0,
        np.log10(freq_hz.astype(np.float32)) / 8.0,
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * azimuth / 360.0),
        np.cos(2.0 * np.pi * azimuth / 360.0),
        np.abs(tx_lat - rx_lat) / 180.0,
        midpoint_lat / 90.0,
        np.sin(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0),
        avg_sfi / 300.0,
        kp_penalty,
    ]).astype(np.float32)


def normalize_snr(df, source):
    """Apply per-source per-band Z-score normalization (V16 spec)."""
    snr = df['median_snr'].values.astype(np.float32).copy()
    band = df['band'].values

    for b in NORM_CONSTANTS:
        mask = band == b
        if mask.sum() > 0:
            mean, std = NORM_CONSTANTS[b][source]
            snr[mask] = (snr[mask] - mean) / std

    return snr


# ── Dataset ───────────────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    def __init__(self, features, targets, weights):
        self.x = features
        self.y = targets
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


# ── Metrics ───────────────────────────────────────────────────────────────────

def pearson_r(pred, target):
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_combined_data():
    """Load V16 data recipe: WSPR + RBN DXpedition + Contest (NO RBN Full)."""
    log.info(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # ── WSPR Terrestrial ──
    wspr_count = client.command("SELECT count() FROM wspr.signatures_v2_terrestrial")
    log.info(f"WSPR terrestrial signatures available: {wspr_count:,}")

    wspr_query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM wspr.signatures_v2_terrestrial
    WHERE avg_sfi > 0
    ORDER BY rand()
    LIMIT {WSPR_SAMPLE}
    """

    log.info(f"Loading WSPR terrestrial data (sample {WSPR_SAMPLE:,})...")
    t0 = time.perf_counter()
    wspr_df = client.query_df(wspr_query)
    log.info(f"WSPR: {len(wspr_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # ── RBN DXpedition ──
    rbn_count = client.command("SELECT count() FROM rbn.dxpedition_signatures")
    log.info(f"RBN DXpedition signatures available: {rbn_count:,}")

    rbn_query = """
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM rbn.dxpedition_signatures
    WHERE avg_sfi > 0
    """

    log.info("Loading RBN DXpedition data (all rows)...")
    t0 = time.perf_counter()
    rbn_df = client.query_df(rbn_query)
    log.info(f"RBN DXpedition: {len(rbn_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # ── Contest ──
    contest_count = client.command("SELECT count() FROM contest.signatures")
    log.info(f"Contest signatures available: {contest_count:,}")

    contest_query = """
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM contest.signatures
    WHERE avg_sfi > 0
    """

    log.info("Loading Contest data (all rows)...")
    t0 = time.perf_counter()
    contest_df = client.query_df(contest_query)
    log.info(f"Contest: {len(contest_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # Get date range
    date_query = """
    SELECT
        formatDateTime(min(timestamp), '%Y-%m-%d') as min_date,
        formatDateTime(max(timestamp), '%Y-%m-%d') as max_date
    FROM wspr.bronze
    """
    date_result = client.query(date_query)
    min_date, max_date = date_result.result_rows[0]
    date_range = f"{min_date} to {max_date}"

    client.close()

    # ── Normalize SNR ──
    log.info("Applying per-source per-band Z-score normalization...")
    wspr_snr = normalize_snr(wspr_df, 'wspr')
    rbn_snr = normalize_snr(rbn_df, 'rbn')
    contest_snr = normalize_snr(contest_df, 'wspr')  # Contest uses WSPR constants

    log.info(f"  WSPR normalized:    mean={wspr_snr.mean():.3f}, std={wspr_snr.std():.3f}")
    log.info(f"  RBN normalized:     mean={rbn_snr.mean():.3f}, std={rbn_snr.std():.3f}")
    log.info(f"  Contest normalized: mean={contest_snr.mean():.3f}, std={contest_snr.std():.3f}")

    # ── Engineer Features ──
    log.info("Engineering features...")
    wspr_X = engineer_features(wspr_df)
    rbn_X = engineer_features(rbn_df)
    contest_X = engineer_features(contest_df)

    # ── Prepare Weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w = wspr_w / wspr_w.mean()

    rbn_w = rbn_df['spot_count'].values.astype(np.float32)
    rbn_w = rbn_w / rbn_w.mean()

    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w = contest_w / contest_w.mean()

    # ── Upsample RBN DXpedition ──
    log.info(f"Upsampling RBN DXpedition {RBN_DX_UPSAMPLE}x...")
    rbn_X_up = np.tile(rbn_X, (RBN_DX_UPSAMPLE, 1))
    rbn_snr_up = np.tile(rbn_snr, RBN_DX_UPSAMPLE)
    rbn_w_up = np.tile(rbn_w, RBN_DX_UPSAMPLE)
    log.info(f"  RBN effective rows: {len(rbn_snr_up):,}")

    # ── Upsample Contest ──
    if CONTEST_UPSAMPLE > 1:
        log.info(f"Upsampling Contest {CONTEST_UPSAMPLE}x...")
        contest_X_up = np.tile(contest_X, (CONTEST_UPSAMPLE, 1))
        contest_snr_up = np.tile(contest_snr, CONTEST_UPSAMPLE)
        contest_w_up = np.tile(contest_w, CONTEST_UPSAMPLE)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr
        contest_w_up = contest_w
    log.info(f"  Contest effective rows: {len(contest_snr_up):,}")

    # ── Combine ──
    X_combined = np.vstack([wspr_X, rbn_X_up, contest_X_up])
    y_combined = np.concatenate([wspr_snr, rbn_snr_up, contest_snr_up]).reshape(-1, 1)
    w_combined = np.concatenate([wspr_w, rbn_w_up, contest_w_up]).reshape(-1, 1)

    total = len(X_combined)
    log.info(f"Combined dataset: {total:,} rows")
    log.info(f"  WSPR:    {len(wspr_X):,} ({100*len(wspr_X)/total:.1f}%)")
    log.info(f"  RBN DX:  {len(rbn_snr_up):,} ({100*len(rbn_snr_up)/total:.1f}%)")
    log.info(f"  Contest: {len(contest_snr_up):,} ({100*len(contest_snr_up)/total:.1f}%)")

    # Free memory
    del wspr_df, rbn_df, contest_df
    del wspr_X, rbn_X, contest_X, rbn_X_up, contest_X_up
    gc.collect()

    return X_combined, y_combined, w_combined, date_range


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info(f"IONIS V20 Golden Master | {PHASE}")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE:,}")
    log.info("")
    log.info("V16 PHYSICS LAWS:")
    log.info("  1. Architecture: IonisV12Gate (gates from trunk output)")
    log.info("  2. Loss: HuberLoss(delta=1.0)")
    log.info("  3. Regularization: Gate variance loss")
    log.info("  4. Init: Defibrillator (weights 0.8-1.2, fc2.bias=-10)")
    log.info("  5. Constraint: Weight clamp [0.5, 2.0] every step")
    log.info("  6. Data: V16 recipe (NO RBN Full)")
    log.info("")
    log.info("SUCCESS CRITERIA:")
    log.info("  - Pearson > +0.48")
    log.info("  - Kp sidecar > +3.0σ")
    log.info("  - SFI sidecar > +0.4σ")
    log.info("")

    # ── Load Data ──
    X_np, y_np, w_np, date_range = load_combined_data()

    snr_mean = float(y_np.mean())
    snr_std = float(y_np.std())

    log.info(f"Dataset: {len(X_np):,} rows x {INPUT_DIM} features")
    log.info(f"Normalized SNR: mean={snr_mean:.3f}, std={snr_std:.3f}")
    log.info(f"Source data range: {date_range}")

    # ── Train/Val Split ──
    n = len(X_np)
    dataset = CombinedDataset(X_np, y_np, w_np)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model (V16 Architecture) ──
    model = IonisV12Gate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=8,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisV12Gate ({total_params:,} params)")

    # ── Defibrillator Init (V16 Law #4) ──
    init_v16_defibrillator(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

    # ── 6-Group Optimizer (V16 Law) ──
    param_groups = get_v16_optimizer_groups(
        model,
        trunk_lr=LR_TRUNK,
        scaler_lr=LR_SCALER,
        sidecar_lr=LR_SIDECAR,
    )
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── HuberLoss (V16 Law #2) ──
    criterion = nn.HuberLoss(reduction='none', delta=HUBER_DELTA)

    # ── Checkpoint Path ──
    model_path = os.path.join(SCRIPT_DIR, "ionis_v20.pth")
    best_val_loss = float('inf')
    best_pearson = -1.0
    best_kp_sidecar = 0.0
    best_sfi_sidecar = 0.0

    # ── Training Loop ──
    log.info("")
    log.info(f"Training started ({EPOCHS} epochs)")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  "
           f"{'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.perf_counter()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()

            # Forward with gates (V16 Law #3: Gate variance)
            out, sun_gate, storm_gate = model.forward_with_gates(bx)
            primary_loss = (criterion(out, by) * bw).mean()

            # Gate variance loss (encourages context-sensitivity)
            var_loss = -LAMBDA_VAR * (sun_gate.var() + storm_gate.var())
            loss = primary_loss + var_loss

            loss.backward()
            optimizer.step()

            # ── THE CLAMP (V16 Law #5) ──
            clamp_v16_sidecars(model)

            train_loss_sum += primary_loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bx, by, bw in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out, _, _ = model.forward_with_gates(bx)
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

        # Physics check (normalized units)
        sfi_benefit = model.get_sun_effect(200.0 / 300.0, DEVICE) - \
                      model.get_sun_effect(70.0 / 300.0, DEVICE)
        storm_cost = model.get_storm_effect(1.0, DEVICE) - \
                     model.get_storm_effect(0.0, DEVICE)

        scheduler.step()
        epoch_sec = time.perf_counter() - t_epoch

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pearson = val_pearson
            best_kp_sidecar = storm_cost
            best_sfi_sidecar = sfi_benefit

            torch.save({
                'model_state': model.state_dict(),
                'snr_mean': snr_mean,
                'snr_std': snr_std,
                'dnn_dim': DNN_DIM,
                'sidecar_hidden': 8,
                'features': FEATURES,
                'band_to_hz': BAND_TO_HZ,
                'norm_constants': NORM_CONSTANTS,
                'date_range': date_range,
                'sample_size': n,
                'wspr_sample': WSPR_SAMPLE,
                'rbn_dx_upsample': RBN_DX_UPSAMPLE,
                'contest_upsample': CONTEST_UPSAMPLE,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'version': VERSION,
                'architecture': 'IonisV12Gate',
                'data_sources': [
                    'wspr.signatures_v2_terrestrial',
                    'rbn.dxpedition_signatures',
                    'contest.signatures'
                ],
                'normalization': 'per-source per-band Z-score',
                'v16_physics_laws': [
                    'IonisV12Gate architecture',
                    'HuberLoss(delta=1.0)',
                    'Gate variance loss',
                    'Defibrillator init',
                    'Weight clamp [0.5, 2.0]',
                ],
                'sfi_benefit_normalized': sfi_benefit,
                'storm_cost_normalized': storm_cost,
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.2f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

    # ── Results ──
    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    # ── Physics Report ──
    log.info("")
    log.info("=" * 70)
    log.info("V20 REPLICATION RESULTS")
    log.info("=" * 70)

    # Success criteria
    pearson_pass = best_pearson > 0.48
    kp_pass = best_kp_sidecar > 3.0
    sfi_pass = best_sfi_sidecar > 0.4

    log.info("")
    log.info("SUCCESS CRITERIA:")
    log.info(f"  Pearson > +0.48:   {best_pearson:+.4f}  {'PASS' if pearson_pass else 'FAIL'}")
    log.info(f"  Kp sidecar > +3.0σ: {best_kp_sidecar:+.3f}σ  {'PASS' if kp_pass else 'FAIL'}")
    log.info(f"  SFI sidecar > +0.4σ: {best_sfi_sidecar:+.3f}σ  {'PASS' if sfi_pass else 'FAIL'}")

    if pearson_pass and kp_pass and sfi_pass:
        log.info("")
        log.info(">>> V20 REPLICATION: SUCCESS <<<")
        log.info(">>> V16 architecture reproduced in clean codebase <<<")
        log.info(">>> Ready to lock code and proceed to production <<<")
    else:
        log.info("")
        log.info(">>> V20 REPLICATION: FAILED <<<")
        log.info(">>> Review training logs and compare to V16 <<<")

    # V16 reference
    log.info("")
    log.info("V16 REFERENCE:")
    log.info("  Pearson:     +0.4873")
    log.info("  Kp sidecar:  +3.445σ")
    log.info("  SFI sidecar: +0.478σ")


if __name__ == '__main__':
    main()
