#!/usr/bin/env python3
"""
train_v19_4.py — IONIS V19.4 Training Script

V16 Recipe + Rosetta Stone normalization:
- 20M WSPR (floor)
- 91K RBN DXpedition x 50 (rare DXCC)
- 6.34M Contest (ceiling)
- Zero RBN Full (kills storm sidecar)
- 20x storm upsample
- Per-source Z-normalization

This is a thin wrapper that:
1. Loads configuration from config_v19_4.json
2. Imports shared code from versions/common/train_common.py
3. Runs the training loop

All model architecture, feature engineering, and data loading
are defined in train_common.py to ensure reproducibility.
"""

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

# Add parent directory to path for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, VERSIONS_DIR)

from common.train_common import (
    IonisModel,
    SignatureDataset,
    load_combined_data,
    log_config,
)

# ── Load Configuration ───────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v19_4.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

# Device selection
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Training Loop ────────────────────────────────────────────────────────────

def train():
    """Main training function."""

    # Log configuration
    log_config(CONFIG, CONFIG_FILE, DEVICE)

    # Load data
    X_np, y_np, w_np, date_range, norm_constants = load_combined_data(CONFIG)

    n = len(X_np)
    log.info(f"Dataset: {n:,} rows x {CONFIG['model']['input_dim']} features")
    log.info(f"Source data range: {date_range}")

    # Create dataset and split
    dataset = SignatureDataset(X_np, y_np, w_np)
    val_split = CONFIG["training"]["val_split"]
    val_size = int(n * val_split)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # DataLoaders
    batch_size = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Create model
    model = IonisModel(CONFIG).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    log.info("")

    # Optimizer with differential learning rates
    trunk_lr = CONFIG["training"]["trunk_lr"]
    sidecar_lr = CONFIG["training"]["sidecar_lr"]
    weight_decay = CONFIG["training"]["weight_decay"]

    trunk_params = list(model.trunk.parameters())
    gate_params = list(model.sun_gate.parameters()) + list(model.storm_gate.parameters())
    sidecar_params = list(model.sun_sidecar.parameters()) + list(model.storm_sidecar.parameters())

    optimizer = optim.AdamW([
        {'params': trunk_params + gate_params, 'lr': trunk_lr},
        {'params': sidecar_params, 'lr': sidecar_lr}
    ], weight_decay=weight_decay)

    epochs = CONFIG["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss(reduction='none')
    lambda_var = CONFIG["training"]["lambda_var"]

    # Checkpoint path
    model_path = os.path.join(SCRIPT_DIR, CONFIG["checkpoint"])

    # Training loop
    print(f"Training started ({epochs} epochs)")
    hdr = " Ep     Train       Val     RMSE   Pearson   SFI+   Kp9-    Time"
    log.info(hdr)
    log.info("-" * len(hdr))

    best_val_loss = float('inf')
    best_pearson = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Training
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0

        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            w_batch = w_batch.to(DEVICE)

            optimizer.zero_grad()

            pred = model(X_batch)
            loss_raw = criterion(pred, y_batch)
            loss_weighted = (loss_raw * w_batch).sum() / w_batch.sum()

            var_loss = -pred.var() * lambda_var
            loss = loss_weighted + var_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss_weighted.item() * w_batch.sum().item()
            train_weight_sum += w_batch.sum().item()

        train_loss = train_loss_sum / train_weight_sum if train_weight_sum > 0 else 0

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                w_batch = w_batch.to(DEVICE)

                pred = model(X_batch)
                loss_raw = criterion(pred, y_batch)
                loss_weighted = (loss_raw * w_batch).sum()

                val_loss_sum += loss_weighted.item()
                val_weight_sum += w_batch.sum().item()

                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss = val_loss_sum / val_weight_sum if val_weight_sum > 0 else 0
        val_rmse = np.sqrt(val_loss)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        val_pearson = float(np.corrcoef(all_preds, all_targets)[0, 1])

        scheduler.step()

        # Physics check
        sfi_benefit = model.get_sun_effect(200.0 / 300.0, DEVICE) - model.get_sun_effect(70.0 / 300.0, DEVICE)
        storm_cost = model.get_storm_effect(1.0, DEVICE) - model.get_storm_effect(0.0, DEVICE)

        epoch_sec = time.perf_counter() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pearson = val_pearson
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CONFIG,
                'norm_constants': norm_constants,
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
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

    # Final report
    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    # Physics verification
    log.info("")
    log.info("=== PHYSICS VERIFICATION ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0, DEVICE)
    sfi_70 = model.get_sun_effect(70.0 / 300.0, DEVICE)
    kp0 = model.get_storm_effect(1.0, DEVICE)
    kp9 = model.get_storm_effect(0.0, DEVICE)

    log.info(f"SFI 70->200 benefit: {sfi_200 - sfi_70:+.3f}σ")
    log.info(f"Kp 0->9 storm cost: {kp0 - kp9:+.3f}σ")

    wspr_std = norm_constants['wspr']['std']
    log.info(f"  (~= {(sfi_200 - sfi_70) * wspr_std:+.1f} dB in WSPR scale)")
    log.info(f"  (~= {(kp0 - kp9) * wspr_std:+.1f} dB storm cost)")

    log.info(f"SFI monotonicity: {'CORRECT' if sfi_200 > sfi_70 else 'INVERTED'}")
    log.info(f"Kp monotonicity: {'CORRECT' if kp0 > kp9 else 'INVERTED'}")

    # Physics assessment
    kp_threshold = CONFIG["validation"]["kp_storm_min"]
    storm_cost = kp0 - kp9
    if storm_cost >= kp_threshold:
        log.info(f"Kp9- sidecar: {storm_cost:+.3f}σ >= {kp_threshold}σ — ALIVE")
    else:
        log.info(f"Kp9- sidecar: {storm_cost:+.3f}σ < {kp_threshold}σ — DEAD")


if __name__ == '__main__':
    train()
