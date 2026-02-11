#!/usr/bin/env python3
"""
train_v20.py — IONIS V20 "Golden Master" Replication

V16 Architecture + V16 Data Recipe + Config-Driven Infrastructure.

All configuration loaded from config_v20.json.
Architecture, features, dataset, and data loading from common/train_common.py.

THE V16 PHYSICS LAWS (non-negotiable):
    1. Architecture: IonisV12Gate (context-aware gates from trunk output)
    2. Loss: HuberLoss(delta=1.0) — robust to synthetic anchors
    3. Regularization: Gate variance loss — forces context sensitivity
    4. Init: Defibrillator — weights uniform(0.8-1.2), fc2.bias=-10
    5. Constraint: Weight clamp [0.5, 2.0] after EVERY step
    6. Data: V16 recipe (WSPR + RBN-DX + Contest, NO RBN-Full)

Success criteria: Pearson > +0.48, Kp > +3.0σ, SFI > +0.4σ
"""

import gc
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import clickhouse_connect

# Add parent for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, VERSIONS_DIR)

from common.train_common import (
    IonisV12Gate,
    SignatureDataset,
    engineer_features,
    init_v16_defibrillator,
    clamp_v16_sidecars,
    get_v16_optimizer_groups,
    load_source_data,
    log_config,
)

# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v20.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Per-Band Normalization (V16 spec) ─────────────────────────────────────────

def normalize_snr_per_band(df, source, norm_constants):
    """Apply per-source per-band Z-score normalization.

    Args:
        df: DataFrame with 'median_snr' and 'band' columns
        source: Which norm source to use ('wspr' or 'rbn')
        norm_constants: Dict of band_str -> {source: {mean, std}}
    """
    snr = df['median_snr'].values.astype(np.float32).copy()
    band = df['band'].values

    for band_str, sources in norm_constants.items():
        b = int(band_str)
        mask = band == b
        if mask.sum() > 0 and source in sources:
            mean = sources[source]['mean']
            std = sources[source]['std']
            snr[mask] = (snr[mask] - mean) / std

    return snr


def pearson_r(pred, target):
    """Pearson correlation coefficient."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training function."""

    log_config(CONFIG, CONFIG_FILE, DEVICE)

    # ── Load Data ──
    ch = CONFIG["clickhouse"]
    client = clickhouse_connect.get_client(host=ch["host"], port=ch["port"])

    wspr_df = load_source_data(client, "wspr.signatures_v2_terrestrial",
                               CONFIG["data"]["wspr_sample"])
    rbn_df = load_source_data(client, "rbn.dxpedition_signatures", None)
    contest_df = load_source_data(client, "contest.signatures", None)

    date_result = client.query("""
        SELECT formatDateTime(min(timestamp), '%Y-%m-%d'),
               formatDateTime(max(timestamp), '%Y-%m-%d')
        FROM wspr.bronze
    """)
    date_range = f"{date_result.result_rows[0][0]} to {date_result.result_rows[0][1]}"
    client.close()

    # ── Per-Band Normalization (V16 spec) ──
    norm_consts = CONFIG["norm_constants_per_band"]
    contest_src = CONFIG["data"]["contest_norm_source"]

    log.info("")
    log.info("=== PER-SOURCE PER-BAND NORMALIZATION (V16 spec) ===")

    wspr_snr = normalize_snr_per_band(wspr_df, 'wspr', norm_consts)
    rbn_snr = normalize_snr_per_band(rbn_df, 'rbn', norm_consts)
    contest_snr = normalize_snr_per_band(contest_df, contest_src, norm_consts)

    log.info(f"  WSPR:    mean={wspr_snr.mean():.3f}, std={wspr_snr.std():.3f}")
    log.info(f"  RBN:     mean={rbn_snr.mean():.3f}, std={rbn_snr.std():.3f}")
    log.info(f"  Contest: mean={contest_snr.mean():.3f}, std={contest_snr.std():.3f}"
             f" (using {contest_src} constants)")

    # ── Features (from train_common) ──
    log.info("")
    log.info("Engineering features...")
    wspr_X = engineer_features(wspr_df, CONFIG)
    rbn_X = engineer_features(rbn_df, CONFIG)
    contest_X = engineer_features(contest_df, CONFIG)

    # ── Weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w /= wspr_w.mean()
    rbn_w = rbn_df['spot_count'].values.astype(np.float32)
    rbn_w /= rbn_w.mean()
    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w /= contest_w.mean()

    # ── Upsample ──
    rbn_dx_up = CONFIG["data"]["rbn_dx_upsample"]
    contest_up = CONFIG["data"]["contest_upsample"]

    log.info(f"Upsampling RBN DXpedition {rbn_dx_up}x...")
    rbn_X_up = np.tile(rbn_X, (rbn_dx_up, 1))
    rbn_snr_up = np.tile(rbn_snr, rbn_dx_up)
    rbn_w_up = np.tile(rbn_w, rbn_dx_up)

    if contest_up > 1:
        log.info(f"Upsampling Contest {contest_up}x...")
        contest_X_up = np.tile(contest_X, (contest_up, 1))
        contest_snr_up = np.tile(contest_snr, contest_up)
        contest_w_up = np.tile(contest_w, contest_up)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr
        contest_w_up = contest_w

    # ── Combine ──
    X = np.vstack([wspr_X, rbn_X_up, contest_X_up])
    y = np.concatenate([wspr_snr, rbn_snr_up, contest_snr_up]).reshape(-1, 1)
    w = np.concatenate([wspr_w, rbn_w_up, contest_w_up]).reshape(-1, 1)

    n = len(X)
    log.info("")
    log.info(f"Combined dataset: {n:,} rows")
    log.info(f"  WSPR:    {len(wspr_X):,} ({100*len(wspr_X)/n:.1f}%)")
    log.info(f"  RBN DX:  {len(rbn_snr_up):,} ({100*len(rbn_snr_up)/n:.1f}%)")
    log.info(f"  Contest: {len(contest_snr_up):,} ({100*len(contest_snr_up)/n:.1f}%)")
    log.info(f"Normalized SNR: mean={y.mean():.3f}, std={y.std():.3f}")
    log.info(f"Source data range: {date_range}")

    del wspr_df, rbn_df, contest_df
    del wspr_X, rbn_X, contest_X, rbn_X_up, contest_X_up
    gc.collect()

    # ── Dataset + Split ──
    dataset = SignatureDataset(X, y, w)
    val_size = int(n * CONFIG["training"]["val_split"])
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model (V16 Law #1) ──
    model = IonisV12Gate(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        sfi_idx=CONFIG["model"]["sfi_idx"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisV12Gate ({total_params:,} params)")

    # ── Defibrillator (V16 Law #4) ──
    init_v16_defibrillator(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

    # ── Optimizer (V16 6-group) ──
    param_groups = get_v16_optimizer_groups(
        model,
        trunk_lr=CONFIG["training"]["trunk_lr"],
        scaler_lr=CONFIG["training"]["scaler_lr"],
        sidecar_lr=CONFIG["training"]["sidecar_lr"],
    )
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG["training"]["weight_decay"])

    epochs = CONFIG["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── HuberLoss (V16 Law #2) ──
    criterion = nn.HuberLoss(reduction='none', delta=CONFIG["training"]["huber_delta"])
    lambda_var = CONFIG["training"]["lambda_var"]

    model_path = os.path.join(SCRIPT_DIR, CONFIG["checkpoint"])
    best_val_loss = float('inf')
    best_pearson = -1.0
    best_kp = 0.0
    best_sfi = 0.0

    # ── Training Loop ──
    log.info("")
    log.info(f"Training started ({epochs} epochs)")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  {'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()

            # V16 Law #3: Gate variance loss
            out, sun_gate, storm_gate = model.forward_with_gates(bx)
            primary_loss = (criterion(out, by) * bw).mean()
            var_loss = -lambda_var * (sun_gate.var() + storm_gate.var())
            loss = primary_loss + var_loss

            loss.backward()
            optimizer.step()

            # V16 Law #5: Weight clamp
            clamp_v16_sidecars(model)

            train_loss_sum += primary_loss.item()
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
                val_loss_sum += criterion(out, by).mean().item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)
        val_pearson = pearson_r(torch.cat(all_preds), torch.cat(all_targets))

        sfi_benefit = (model.get_sun_effect(200.0 / 300.0, DEVICE) -
                       model.get_sun_effect(70.0 / 300.0, DEVICE))
        storm_cost = (model.get_storm_effect(1.0, DEVICE) -
                      model.get_storm_effect(0.0, DEVICE))

        scheduler.step()
        epoch_sec = time.perf_counter() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pearson = val_pearson
            best_kp = storm_cost
            best_sfi = sfi_benefit
            torch.save({
                'model_state': model.state_dict(),
                'config': CONFIG,
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_benefit': sfi_benefit,
                'storm_cost': storm_cost,
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.2f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    log.info("")
    log.info("=" * 70)
    log.info("V20 REPLICATION RESULTS")
    log.info("=" * 70)

    p_min = CONFIG["validation"]["pearson_min"]
    kp_min = CONFIG["validation"]["kp_storm_min"]
    sfi_min = CONFIG["validation"]["sfi_benefit_min"]

    p_pass = best_pearson > p_min
    kp_pass = best_kp > kp_min
    sfi_pass = best_sfi > sfi_min

    log.info("")
    log.info("SUCCESS CRITERIA:")
    log.info(f"  Pearson > +{p_min}:     {best_pearson:+.4f}  {'PASS' if p_pass else 'FAIL'}")
    log.info(f"  Kp sidecar > +{kp_min}σ:  {best_kp:+.3f}σ  {'PASS' if kp_pass else 'FAIL'}")
    log.info(f"  SFI sidecar > +{sfi_min}σ: {best_sfi:+.3f}σ  {'PASS' if sfi_pass else 'FAIL'}")

    if p_pass and kp_pass and sfi_pass:
        log.info("")
        log.info(">>> V20 REPLICATION: SUCCESS <<<")
        log.info(">>> V16 reproduced in clean codebase <<<")
    else:
        log.info("")
        log.info(">>> V20 REPLICATION: FAILED <<<")
        log.info(">>> Review training logs and compare to V16 <<<")

    log.info("")
    log.info("V16 REFERENCE: Pearson +0.4873, Kp +3.445σ, SFI +0.478σ")


if __name__ == '__main__':
    train()
