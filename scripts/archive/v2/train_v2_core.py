#!/usr/bin/env python3
"""
train_v2_core.py — IONIS V2 Oracle: Streaming ClickHouse Training Engine

Phase 5.2: Continuous Weighting — 17-feature model with IFW sampling
Reads pre-materialized wspr.training_continuous (10M rows, IFW-weighted)
from ClickHouse (9975WX) directly to PyTorch tensors. No intermediate files.

Phase 5.2 changes from Phase 5:
    DROPPED: prop_quality (collinear SSN-SFI trap)
    ADDED:   sfi (raw, normalized)
    ADDED:   kp (raw, normalized)
    ADDED:   sfi_dist_interact (SFI × log10(distance), pre-computed in CH)
    ADDED:   WeightedRandomSampler using IFW sampling_weight column

Architecture: 17 → 512 → 256 → 128 → 1 (Mish + BatchNorm)
Data source:  wspr.training_continuous (1M rows × 10 bands, IFW-weighted)

Usage:
    python train_v2_core.py
    CH_HOST=10.0.0.1 python train_v2_core.py

Environment:
    CH_HOST  ClickHouse host (default: 10.60.1.1)
    CH_PORT  ClickHouse HTTP port (default: 8123)
"""

import os
import sys
import time
import logging
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, WeightedRandomSampler
import clickhouse_connect


# ═══════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════

CH_HOST = os.environ.get('CH_HOST', '10.60.1.1')
CH_PORT = int(os.environ.get('CH_PORT', '8123'))
CH_CONNECT_TIMEOUT = 30
CH_QUERY_TIMEOUT = 600       # 10 min for large per-band queries
MAX_RETRIES = 3

# ClickHouse settings — lightweight since we read from a pre-materialized table
CH_SETTINGS = {
    'max_block_size': 524_288,                       # 512k rows per streamed block
}

BATCH_SIZE = 4096            # PyTorch training mini-batch
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # AdamW L2 regularization
SAMPLE_SIZE = 10_000_000     # Pilot: 10M rows
VAL_SPLIT = 0.2

INPUT_DIM = 17
HIDDEN_DIMS = [512, 256, 128]   # Mish MLP architecture
DATE_START = '2020-01-01'
DATE_END = '2026-02-04'

MODEL_DIR = 'models'
MODEL_FILE = 'ionis_v2_continuous.pth'
PHASE = 'Phase 5.2: Continuous IFW + Mish MLP (17→512→256→128→1)'

# Band IDs after Clean Slate re-ingest (v2.1.0, ADIF standard)
HF_BANDS = list(range(102, 112))  # 102=160m through 111=10m

BAND_TO_HZ = {
    102:  1_836_600,   # 160m
    103:  3_568_600,   # 80m
    104:  5_287_200,   # 60m
    105:  7_038_600,   # 40m
    106: 10_138_700,   # 30m
    107: 14_097_100,   # 20m
    108: 18_104_600,   # 17m
    109: 21_094_600,   # 15m
    110: 24_924_600,   # 12m
    111: 28_124_600,   # 10m
}

FEATURES = [
    'distance', 'freq_log', 'hour_sin', 'hour_cos', 'ssn',
    'az_sin', 'az_cos', 'lat_diff', 'midpoint_lat',
    'season_sin', 'season_cos',
    'ssn_lat_interact', 'day_night_est',
    'sfi', 'kp',
    'band_sfi_interact', 'sfi_dist_interact',
]

# Device selection: MPS (M3 Ultra) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v2')


# ═══════════════════════════════════════════════════════════════════
# 2. Model Architecture — Mish MLP (17 → 512 → 256 → 128 → 1)
# ═══════════════════════════════════════════════════════════════════

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)).
    Smoother, non-monotonic gradient for better solar cycle learning."""
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class IONIS_V2(nn.Module):
    """Multi-layer perceptron with Mish activation and BatchNorm.

    Architecture: 17 → 512 → 256 → 128 → 1
    Activation:   Mish (x·tanh(softplus(x)))
    Regularization: BatchNorm between each layer
    """
    def __init__(self, input_dim=17, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                Mish(),
            ])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════
# 3. Maidenhead Grid Utilities (matches clickhouse_loader.cpp:27-77)
# ═══════════════════════════════════════════════════════════════════

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def clean_grids(raw_grids):
    """Clean FixedString(8) grids from ClickHouse. Invalid grids → 'JJ00'."""
    out = []
    for g in raw_grids:
        if isinstance(g, (bytes, bytearray)):
            s = g.decode('ascii', errors='ignore').rstrip('\x00')
        else:
            s = str(g).rstrip('\x00')
        m = GRID_RE.search(s)
        out.append(m.group(0).upper() if m else 'JJ00')
    return out


def grid_to_latlon(grids):
    """Vectorized Maidenhead 4-char grid → (lat, lon) arrays."""
    grid4 = np.array(grids, dtype='U4')
    codes = grid4.view('U1').reshape(-1, 4)
    b = codes.view(np.uint32).astype(np.float32)
    lon = (b[:, 0] - ord('A')) * 20.0 - 180.0 + (b[:, 2] - ord('0')) * 2.0 + 1.0
    lat = (b[:, 1] - ord('A')) * 10.0 - 90.0 + (b[:, 3] - ord('0')) * 1.0 + 0.5
    return lat, lon


# ═══════════════════════════════════════════════════════════════════
# 4. Feature Engineering (17 normalized features)
# ═══════════════════════════════════════════════════════════════════

def band_to_hz(band_ids):
    """Map band ID array → frequency in Hz. Unknown bands default to 20m."""
    default = 14_097_100
    return np.array([BAND_TO_HZ.get(int(b), default) for b in band_ids],
                    dtype=np.float64)


def engineer_features(distance, freq_hz, hour, month, azimuth,
                      ssn_arr, sfi_arr, kp_arr, sfi_dist_arr,
                      tx_lat, tx_lon, rx_lat, rx_lon):
    """Compute 17 normalized features from raw columns. Returns (N, 17) float32.

    Features 1-13:  original IONIS V2 (path geometry, solar, interactions)
    Feature 14:     sfi = raw SFI / 300 (replaces prop_quality)
    Feature 15:     kp = raw Kp / 9
    Feature 16:     band_sfi_interact = SFI_norm × freq_log
    Feature 17:     sfi_dist_interact = SFI × log10(distance), normalized
    """
    f_distance     = distance / 20000.0
    f_freq_log     = np.log10(freq_hz.astype(np.float32)) / 8.0
    f_hour_sin     = np.sin(2.0 * np.pi * hour / 24.0)
    f_hour_cos     = np.cos(2.0 * np.pi * hour / 24.0)
    f_ssn          = ssn_arr / 300.0
    f_az_sin       = np.sin(2.0 * np.pi * azimuth / 360.0)
    f_az_cos       = np.cos(2.0 * np.pi * azimuth / 360.0)
    f_lat_diff     = np.abs(tx_lat - rx_lat) / 180.0
    f_midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    f_season_sin   = np.sin(2.0 * np.pi * month / 12.0)
    f_season_cos   = np.cos(2.0 * np.pi * month / 12.0)
    f_ssn_lat      = f_ssn * np.abs(f_midpoint_lat)
    midpoint_lon   = (tx_lon + rx_lon) / 2.0
    local_solar_h  = hour + midpoint_lon / 15.0
    f_daynight     = np.cos(2.0 * np.pi * local_solar_h / 24.0)

    # Phase 5.2: Raw solar features — network learns interactions
    f_sfi          = sfi_arr / 300.0          # F10.7 solar flux, range ~60-300+
    f_kp           = kp_arr / 9.0             # Kp geomagnetic index, range 0-9
    f_band_sfi     = f_sfi * f_freq_log       # SFI effect varies by band
    f_sfi_dist     = sfi_dist_arr / (300.0 * np.log10(18000.0))  # normalized

    return np.column_stack([
        f_distance, f_freq_log, f_hour_sin, f_hour_cos, f_ssn,
        f_az_sin, f_az_cos, f_lat_diff, f_midpoint_lat,
        f_season_sin, f_season_cos, f_ssn_lat, f_daynight,
        f_sfi, f_kp, f_band_sfi, f_sfi_dist,
    ]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 5. Materialized Table Ingestion (Phase 5.2)
# ═══════════════════════════════════════════════════════════════════

def build_continuous_query():
    """Single query to read pre-materialized IFW-weighted training table.

    wspr.training_continuous is populated on the 9975WX with
    Efraimidis-Spirakis weighted sampling against a 2D (SSN, lat)
    density histogram. 10M rows, 1M per band, no discrete SSN bins.
    """
    return """
        SELECT snr, distance, band, hour, month, azimuth,
               tx_grid, rx_grid, ssn, sfi, kp,
               midpoint_lat, sampling_weight, sfi_dist_interact
        FROM wspr.training_continuous
        ORDER BY cityHash64(tx_grid, rx_grid, toString(snr))
    """


def connect_clickhouse():
    """Connect to ClickHouse with exponential-backoff retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = clickhouse_connect.get_client(
                host=CH_HOST,
                port=CH_PORT,
                connect_timeout=CH_CONNECT_TIMEOUT,
                send_receive_timeout=CH_QUERY_TIMEOUT,
            )
            ver = client.server_version
            log.info(f"Connected to ClickHouse {ver} at {CH_HOST}:{CH_PORT}")
            return client
        except Exception as e:
            log.warning(f"Connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise ConnectionError(
                    f"Cannot reach ClickHouse at {CH_HOST}:{CH_PORT} "
                    f"after {MAX_RETRIES} attempts"
                ) from e
            time.sleep(2 ** attempt)


def stream_dataset():
    """Read pre-materialized wspr.training_continuous → (X, y, weights) tensors.

    Single query reads all 10M rows. Feature engineering runs in Python.
    Returns sampling_weight array for WeightedRandomSampler.
    """
    client = connect_clickhouse()

    log.info(f"Reading wspr.training_continuous (pre-materialized, IFW-weighted)")

    t0 = time.perf_counter()
    feat_blocks = []
    tgt_blocks = []
    weight_blocks = []
    band_counts = {}
    ssn_accum = []
    snr_accum = []
    total_rows = 0

    query = build_continuous_query()

    try:
        with client.query_column_block_stream(
            query, settings=CH_SETTINGS,
        ) as stream:
            for block in stream:
                snr        = np.asarray(block[0], dtype=np.float32)
                distance   = np.asarray(block[1], dtype=np.float32)
                band_id    = np.asarray(block[2], dtype=np.int32)
                hour       = np.asarray(block[3], dtype=np.float32)
                month      = np.asarray(block[4], dtype=np.float32)
                azimuth    = np.asarray(block[5], dtype=np.float32)
                raw_tx     = list(block[6])
                raw_rx     = list(block[7])
                ssn_arr    = np.asarray(block[8], dtype=np.float32)
                sfi_arr    = np.asarray(block[9], dtype=np.float32)
                kp_arr     = np.asarray(block[10], dtype=np.float32)
                # block[11] = midpoint_lat (used for verification, not features)
                weights    = np.asarray(block[12], dtype=np.float64)
                sfi_dist   = np.asarray(block[13], dtype=np.float32)

                n = len(snr)
                total_rows += n

                ssn_accum.append(ssn_arr)
                snr_accum.append(snr)
                weight_blocks.append(weights)

                # Count per band
                for b in np.unique(band_id):
                    band_counts[int(b)] = band_counts.get(int(b), 0) + int(np.sum(band_id == b))

                # Feature engineering
                freq_hz = band_to_hz(band_id)
                tx_grids = clean_grids(raw_tx)
                rx_grids = clean_grids(raw_rx)
                tx_lat, tx_lon = grid_to_latlon(tx_grids)
                rx_lat, rx_lon = grid_to_latlon(rx_grids)

                feats = engineer_features(
                    distance, freq_hz, hour, month, azimuth,
                    ssn_arr, sfi_arr, kp_arr, sfi_dist,
                    tx_lat, tx_lon, rx_lat, rx_lon,
                )
                feat_blocks.append(feats)
                tgt_blocks.append(snr)

    except Exception as e:
        log.warning(f"Stream error after {total_rows:,} rows: {e}")
        if total_rows == 0:
            raise

    elapsed = time.perf_counter() - t0
    rps = total_rows / elapsed if elapsed > 0 else 0
    log.info(f"Ingestion complete: {total_rows:,} rows in {elapsed:.1f}s ({rps:,.0f} rows/sec)")

    if total_rows == 0:
        raise RuntimeError("No rows ingested — check wspr.training_continuous")

    X_np = np.concatenate(feat_blocks, axis=0)
    y_np = np.concatenate(tgt_blocks, axis=0)
    ssn_all = np.concatenate(ssn_accum, axis=0)
    snr_all = np.concatenate(snr_accum, axis=0)
    weight_all = np.concatenate(weight_blocks, axis=0)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    return (X, y), band_counts, ssn_all, snr_all, weight_all


# ═══════════════════════════════════════════════════════════════════
# 6. Metrics
# ═══════════════════════════════════════════════════════════════════

def pearson_r(pred, target):
    """Pearson correlation between two 1-D tensors."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# 7. Training Loop
# ═══════════════════════════════════════════════════════════════════

def print_diagnostics(X, y, band_counts, ssn_all, snr_all):
    """Print dataset diagnostics: feature stats, band distribution, correlations."""
    n = len(X)
    snr_mean = y.mean().item()
    snr_std = y.std().item()

    log.info(f"Dataset: {n:,} rows x {INPUT_DIM} features")
    log.info(f"SNR range: {y.min().item():.0f} to {y.max().item():.0f} dB")
    log.info(f"SNR mean: {snr_mean:.1f} dB, std: {snr_std:.1f} dB")

    # Feature statistics
    log.info("Feature statistics (normalized):")
    log.info(f"  {'Feature':<20s}  {'Min':>8s}  {'Mean':>8s}  {'Max':>8s}")
    log.info(f"  {'-' * 50}")
    for i, name in enumerate(FEATURES):
        col = X[:, i]
        log.info(f"  {name:<20s}  {col.min():8.4f}  {col.mean():8.4f}  {col.max():8.4f}")

    # Per-band row counts
    if band_counts:
        log.info("Per-band row counts:")
        for bid in sorted(band_counts):
            hz = BAND_TO_HZ.get(bid, 0)
            label = f"{hz / 1e6:.3f}MHz" if hz else f"band={bid}"
            log.info(f"  Band {bid:3d} ({label:>10s}): {band_counts[bid]:>10,} rows")

    # SSN-SNR correlation diagnostic
    if len(ssn_all) > 0 and len(snr_all) > 0:
        corr = np.corrcoef(ssn_all, snr_all)[0, 1]
        log.info(f"SSN-SNR Pearson correlation: {corr:+.4f}")
        if corr > 0.1:
            log.info("  TARGET MET: correlation > 0.1")
        elif corr > 0:
            log.info("  Positive but below 0.1 target")
        else:
            log.info("  WARNING: Negative correlation")

    return snr_mean, snr_std


def main():
    log.info(f"IONIS V2 Oracle | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")

    # ── Read pre-materialized data from ClickHouse ──
    (X, y), band_counts, ssn_all, snr_all, weight_all = stream_dataset()
    snr_mean, snr_std = print_diagnostics(X, y, band_counts, ssn_all, snr_all)
    n = len(X)

    # ── Train/val split ──
    dataset = TensorDataset(X, y)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # IFW sampling: WeightedRandomSampler uses the density-based weights
    # so rare (SSN, lat) combinations are seen more frequently per epoch.
    train_indices = train_set.indices
    train_weights = torch.tensor(weight_all[train_indices], dtype=torch.float64)
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=train_size,
        replacement=True,
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val (IFW-weighted sampler)")

    # ── Model ──
    model = IONIS_V2(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    arch_str = ' -> '.join(str(d) for d in [INPUT_DIM] + HIDDEN_DIMS + [1])
    log.info(f"Model: {arch_str}  ({params:,} parameters)")
    log.info(f"Activation: Mish | BatchNorm between layers")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')

    # ── Training ──
    log.info("Training started")
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
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

        # Validate (accumulate for Pearson)
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by)
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
                'hidden_dims': HIDDEN_DIMS,
                'features': FEATURES,
                'band_ids': HF_BANDS,
                'band_to_hz': BAND_TO_HZ,
                'date_range': f'{DATE_START} to {DATE_END}',
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'solar_resolution': '3-hourly Kp, daily SFI/SSN (GFZ Potsdam)',
                'activation': 'Mish',
                'sampling': 'IFW (Efraimidis-Spirakis, 2D SSN×lat density)',
                'data_source': 'wspr.training_continuous (IFW-weighted)',
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
