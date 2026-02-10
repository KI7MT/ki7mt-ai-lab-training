#!/usr/bin/env python3
"""
train_v13_combined.py — IONIS V13 Combined Training (WSPR + RBN DXpedition)

V13 adds 91K RBN DXpedition signatures from rare DXCC entities to the
WSPR signature base. Key innovation: per-source per-band Z-score normalization
so the model learns physics, not power levels.

Data sources:
    - wspr.signatures_v1: 93.8M signatures (200mW beacons, SNR ~ -18 dB)
    - rbn.dxpedition_signatures: 91K signatures (1kW+ DXpeditions, SNR ~ +17 dB)

The ~35 dB offset reflects TX power difference, not propagation physics.
Z-score normalization per-source per-band removes this offset while preserving
the relative effects (SFI benefit, Kp cost, path geometry).

Architecture: IonisV12Gate (unchanged — 203,573 params)
Target: Z-normalized median_snr
Weight: spot_count (normalized per source to balance contribution)

Checkpoint: models/ionis_v13_combined.pth
"""

import gc
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

import clickhouse_connect

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(TRAINING_DIR, "models")
MODEL_FILE = "ionis_v13_combined.pth"
PHASE = "V13 Combined: WSPR + RBN DXpedition (Z-normalized)"

# ClickHouse connection (DAC link to 9975WX)
CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

# Sample sizes
WSPR_SAMPLE = 20_000_000    # 20M WSPR signatures
RBN_SAMPLE = None           # All 91K RBN DXpedition signatures (no sampling)

# Upsampling: repeat RBN data to balance contribution
# 20M / 91K ≈ 220x, but we don't want to dominate — use moderate upsampling
RBN_UPSAMPLE_FACTOR = 50    # 91K × 50 = 4.55M effective RBN rows

BATCH_SIZE = 65536
EPOCHS = 100
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
LAMBDA_VAR = 0.001

NUM_WORKERS = 0
PIN_MEMORY = False
PREFETCH_FACTOR = None
PERSISTENT_WORKERS = False

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

# Per-source per-band normalization constants (from 9975 analysis)
# Format: {band: {'wspr': (mean, std), 'rbn': (mean, std)}}
NORM_CONSTANTS = {
    102: {'wspr': (-18.04, 6.9), 'rbn': (11.86, 6.23)},   # 160m
    103: {'wspr': (-17.90, 6.9), 'rbn': (16.94, 6.71)},   # 80m
    104: {'wspr': (-17.60, 7.1), 'rbn': (16.91, 7.1)},    # 60m
    105: {'wspr': (-17.34, 6.6), 'rbn': (18.36, 6.6)},    # 40m
    106: {'wspr': (-18.07, 6.5), 'rbn': (16.96, 6.5)},    # 30m
    107: {'wspr': (-17.53, 6.7), 'rbn': (18.47, 6.7)},    # 20m
    108: {'wspr': (-18.35, 7.0), 'rbn': (16.66, 7.0)},    # 17m
    109: {'wspr': (-18.32, 6.6), 'rbn': (16.37, 6.6)},    # 15m
    110: {'wspr': (-18.76, 6.6), 'rbn': (15.07, 6.6)},    # 12m
    111: {'wspr': (-17.86, 6.5), 'rbn': (15.79, 6.5)},    # 10m
}

GATE_INIT_BIAS = -math.log(2.0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v13')


# ── Grid Utilities ───────────────────────────────────────────────────────────

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def grid4_to_latlon(g):
    """Convert a single 4-char Maidenhead grid to (lat, lon) centroid."""
    s = str(g).strip().rstrip('\x00').upper()
    m = GRID_RE.search(s)
    g4 = m.group(0) if m else 'JJ00'
    lon = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
    lat = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lat, lon


def grid4_to_latlon_arrays(grids):
    """Convert array of 4-char grids to (lat, lon) arrays."""
    lats = np.zeros(len(grids), dtype=np.float32)
    lons = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        lats[i], lons[i] = grid4_to_latlon(g)
    return lats, lons


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df):
    """Compute 13 features from signature columns."""
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
    """Apply per-source per-band Z-score normalization to median_snr."""
    snr = df['median_snr'].values.astype(np.float32).copy()
    band = df['band'].values

    for b in NORM_CONSTANTS:
        mask = band == b
        if mask.sum() > 0:
            mean, std = NORM_CONSTANTS[b][source]
            snr[mask] = (snr[mask] - mean) / std

    return snr


# ── Dataset ──────────────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    def __init__(self, features, targets, weights):
        self.x = features
        self.y = targets
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


# ── Model (unchanged from V12) ───────────────────────────────────────────────

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
        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost, \
               sun_gate, storm_gate

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


# ── Metrics ──────────────────────────────────────────────────────────────────

def pearson_r(pred, target):
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_combined_data():
    """Load and combine WSPR + RBN DXpedition signatures with Z-normalization."""
    log.info(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # ── WSPR Signatures ──
    wspr_count = client.command("SELECT count() FROM wspr.signatures_v1")
    log.info(f"WSPR signatures available: {wspr_count:,}")

    wspr_limit = f" LIMIT {WSPR_SAMPLE}" if WSPR_SAMPLE else ""
    wspr_order = " ORDER BY rand()" if WSPR_SAMPLE else ""

    wspr_query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM wspr.signatures_v1
    WHERE avg_sfi > 0
    {wspr_order}{wspr_limit}
    """

    log.info(f"Loading WSPR data (sample {WSPR_SAMPLE:,})...")
    t0 = time.perf_counter()
    wspr_df = client.query_df(wspr_query)
    log.info(f"WSPR: {len(wspr_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # ── RBN DXpedition Signatures ──
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
    log.info(f"RBN: {len(rbn_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # Get date range from WSPR bronze
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

    # ── Normalize SNR per-source per-band ──
    log.info("Applying per-source per-band Z-score normalization...")
    wspr_snr = normalize_snr(wspr_df, 'wspr')
    rbn_snr = normalize_snr(rbn_df, 'rbn')

    log.info(f"  WSPR normalized SNR: mean={wspr_snr.mean():.3f}, std={wspr_snr.std():.3f}")
    log.info(f"  RBN normalized SNR:  mean={rbn_snr.mean():.3f}, std={rbn_snr.std():.3f}")

    # ── Engineer Features ──
    log.info("Engineering features...")
    wspr_X = engineer_features(wspr_df)
    rbn_X = engineer_features(rbn_df)

    # ── Prepare weights ──
    # Normalize weights within each source to mean=1
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w = wspr_w / wspr_w.mean()

    rbn_w = rbn_df['spot_count'].values.astype(np.float32)
    rbn_w = rbn_w / rbn_w.mean()

    # ── Upsample RBN to balance contribution ──
    log.info(f"Upsampling RBN data {RBN_UPSAMPLE_FACTOR}x...")
    rbn_X_up = np.tile(rbn_X, (RBN_UPSAMPLE_FACTOR, 1))
    rbn_snr_up = np.tile(rbn_snr, RBN_UPSAMPLE_FACTOR)
    rbn_w_up = np.tile(rbn_w, RBN_UPSAMPLE_FACTOR)

    log.info(f"  RBN effective rows: {len(rbn_snr_up):,}")

    # ── Combine ──
    X_combined = np.vstack([wspr_X, rbn_X_up])
    y_combined = np.concatenate([wspr_snr, rbn_snr_up]).reshape(-1, 1)
    w_combined = np.concatenate([wspr_w, rbn_w_up]).reshape(-1, 1)

    log.info(f"Combined dataset: {len(X_combined):,} rows")
    log.info(f"  WSPR contribution: {len(wspr_X):,} ({100*len(wspr_X)/len(X_combined):.1f}%)")
    log.info(f"  RBN contribution:  {len(rbn_snr_up):,} ({100*len(rbn_snr_up)/len(X_combined):.1f}%)")

    # Free DataFrames
    del wspr_df, rbn_df, wspr_X, rbn_X, rbn_X_up
    gc.collect()

    return X_combined, y_combined, w_combined, date_range


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    log.info(f"IONIS V13 | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Data sources: wspr.signatures_v1 + rbn.dxpedition_signatures")
    log.info(f"Normalization: per-source per-band Z-score")
    log.info(f"RBN upsampling: {RBN_UPSAMPLE_FACTOR}x")
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

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model ──
    model = IonisV12Gate(dnn_dim=DNN_DIM, sidecar_hidden=8).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisV12Gate ({total_params:,} params)")

    # ── Defibrillator: Init Sidecars ──
    def wake_up_sidecar(layer):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, 0.8, 1.2)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    model.sun_sidecar.apply(wake_up_sidecar)
    model.storm_sidecar.apply(wake_up_sidecar)

    with torch.no_grad():
        model.sun_sidecar.fc2.bias.fill_(-10.0)
        model.storm_sidecar.fc2.bias.fill_(-10.0)

    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

    # ── Optimizer ──
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

            out, sun_gate_b, storm_gate_b = model.forward_with_gates(bx)
            primary_loss = (criterion(out, by) * bw).mean()
            sun_var = sun_gate_b.var()
            storm_var = storm_gate_b.var()
            var_loss = -LAMBDA_VAR * (sun_var + storm_var)
            loss = primary_loss + var_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for sidecar in [model.sun_sidecar, model.storm_sidecar]:
                    sidecar.fc1.weight.clamp_(0.5, 2.0)
                    sidecar.fc2.weight.clamp_(0.5, 2.0)

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

        # Physics check (in normalized units)
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
                'norm_constants': NORM_CONSTANTS,
                'date_range': date_range,
                'sample_size': n,
                'wspr_sample': WSPR_SAMPLE,
                'rbn_upsample': RBN_UPSAMPLE_FACTOR,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'architecture': 'IonisV12Gate (unchanged)',
                'data_sources': ['wspr.signatures_v1', 'rbn.dxpedition_signatures'],
                'normalization': 'per-source per-band Z-score',
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

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    # ── Physics Report ──
    log.info("")
    log.info("=== PHYSICS VERIFICATION (normalized units) ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0)
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    kp0 = model.get_storm_effect(1.0)
    kp9 = model.get_storm_effect(0.0)

    log.info(f"SFI 70→200 benefit: {sfi_200 - sfi_70:+.3f}σ")
    log.info(f"Kp 0→9 storm cost: {kp0 - kp9:+.3f}σ")

    # Convert to approximate dB (avg std ≈ 6.7 dB)
    avg_std_db = 6.7
    log.info(f"  (≈ {(sfi_200 - sfi_70) * avg_std_db:+.1f} dB SFI benefit)")
    log.info(f"  (≈ {(kp0 - kp9) * avg_std_db:+.1f} dB storm cost)")

    if sfi_200 > sfi_70:
        log.info("SFI monotonicity: CORRECT")
    else:
        log.info("SFI monotonicity: WARNING — inverted")

    if kp0 > kp9:
        log.info("Kp monotonicity: CORRECT")
    else:
        log.info("Kp monotonicity: WARNING — inverted")

    log.info("")
    log.info("=== V13 vs V12 ===")
    log.info(f"V12 (WSPR only):         Pearson +0.3153")
    log.info(f"V13 (WSPR + RBN DXpedition): Pearson {best_pearson:+.4f}")


if __name__ == '__main__':
    main()
