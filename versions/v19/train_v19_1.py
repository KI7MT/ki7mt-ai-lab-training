#!/usr/bin/env python3
"""
train_v19_1.py — IONIS V19.1 "Physics First" (Two-Stage Training)

V19 diagnosed with "Feature Cannibalization" — the DNN trunk absorbed all
gradient signal, starving the physics sidecars (Kp9- dropped to +0.05σ).

V19.1 remediation per Gemini:
    Stage 1 (Epochs 0-5): FREEZE trunk and gates, train ONLY sidecars
        → Force model to explain variance using ONLY Kp and SFI
    Stage 2 (Epochs 6-100): UNFREEZE all
        → DNN learns to correct residuals left by sidecars

Additional fix: Storm upsampling increased from 10x to 20x.

Expected outcome: Kp9- initializes high (~2.0σ) and stays there.

Architecture: IonisV12Gate (unchanged — 203,573 params)
Checkpoint: versions/v19/ionis_v19_1.pth
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
MODEL_FILE = "ionis_v19_1.pth"
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_FILE)
PHASE = "V19.1 Physics First (Two-Stage)"
VERSION = "v19.1-physics-first"

# ClickHouse connection (DAC link to 9975WX)
CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

# Sample sizes
WSPR_SAMPLE = 20_000_000
RBN_FULL_SAMPLE = 20_000_000
RBN_DX_SAMPLE = None
CONTEST_SAMPLE = None

# Upsampling
RBN_DX_UPSAMPLE_FACTOR = 50
CONTEST_UPSAMPLE_FACTOR = 1
STORM_UPSAMPLE_FACTOR = 20  # ← DOUBLED from V19's 10x

# Two-stage training
WARMUP_EPOCHS = 5  # Epochs with trunk frozen

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

GATE_INIT_BIAS = -math.log(2.0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v19.1')


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
    sfi = df['avg_sfi'].values.astype(np.float32)
    kp = df['avg_kp'].values.astype(np.float32)

    tx_lats, tx_lons = grid4_to_latlon_arrays(df['tx_grid_4'].values)
    rx_lats, rx_lons = grid4_to_latlon_arrays(df['rx_grid_4'].values)

    midpoint_lat = (tx_lats + rx_lats) / 2.0
    midpoint_lon = (tx_lons + rx_lons) / 2.0

    freq_hz = np.array([BAND_TO_HZ.get(b, 14_097_100) for b in band], dtype=np.float32)

    X = np.zeros((len(df), INPUT_DIM), dtype=np.float32)

    # Features 0-10: DNN inputs (geography/time)
    X[:, 0] = distance / 20000.0
    X[:, 1] = np.log10(freq_hz) / 8.0
    X[:, 2] = np.sin(2.0 * np.pi * hour / 24.0)
    X[:, 3] = np.cos(2.0 * np.pi * hour / 24.0)
    X[:, 4] = np.sin(2.0 * np.pi * azimuth / 360.0)
    X[:, 5] = np.cos(2.0 * np.pi * azimuth / 360.0)
    X[:, 6] = np.abs(tx_lats - rx_lats) / 180.0
    X[:, 7] = midpoint_lat / 90.0
    X[:, 8] = np.sin(2.0 * np.pi * month / 12.0)
    X[:, 9] = np.cos(2.0 * np.pi * month / 12.0)
    X[:, 10] = np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)

    # Features 11-12: Sidecar inputs (solar physics)
    X[:, 11] = sfi / 300.0
    X[:, 12] = 1.0 - kp / 9.0  # kp_penalty

    return X


# ── Model Architecture ───────────────────────────────────────────────────────

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


class IonisV12Gate(nn.Module):
    def __init__(self, dnn_dim=DNN_DIM, hidden=256, sidecar_hidden=8):
        super().__init__()

        # DNN for geography/time (starved of solar info)
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Physics sidecars (monotonic)
        self.sun_sidecar = MonotonicMLP(sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(sidecar_hidden)

        # Gating networks
        self.sun_gate = nn.Sequential(
            nn.Linear(dnn_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.storm_gate = nn.Sequential(
            nn.Linear(dnn_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self._init_gates()

    def _init_gates(self):
        for gate in [self.sun_gate, self.storm_gate]:
            if hasattr(gate[-1], 'bias') and gate[-1].bias is not None:
                nn.init.constant_(gate[-1].bias, GATE_INIT_BIAS)
                gate[-1].bias.requires_grad = True
            if hasattr(gate[-1], 'weight'):
                nn.init.xavier_normal_(gate[-1].weight, gain=0.1)

    def forward(self, x):
        x_dnn = x[:, :DNN_DIM]
        sfi_in = x[:, SFI_IDX:SFI_IDX+1]
        kp_penalty = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX+1]

        base = self.trunk(x_dnn)
        sun_effect = self.sun_sidecar(sfi_in)
        storm_effect = self.storm_sidecar(kp_penalty)

        sun_g = torch.sigmoid(self.sun_gate(x_dnn)) + 0.5
        storm_g = torch.sigmoid(self.storm_gate(x_dnn)) + 0.5

        return base + sun_g * sun_effect + storm_g * storm_effect

    def forward_with_components(self, x):
        x_dnn = x[:, :DNN_DIM]
        sfi_in = x[:, SFI_IDX:SFI_IDX+1]
        kp_penalty = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX+1]

        base = self.trunk(x_dnn)
        sun_effect = self.sun_sidecar(sfi_in)
        storm_effect = self.storm_sidecar(kp_penalty)

        sun_g = torch.sigmoid(self.sun_gate(x_dnn)) + 0.5
        storm_g = torch.sigmoid(self.storm_gate(x_dnn)) + 0.5

        total = base + sun_g * sun_effect + storm_g * storm_effect
        return total, base, sun_g, storm_g, sun_effect, storm_effect

    def get_sun_effect(self, sfi_normalized):
        with torch.no_grad():
            inp = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
            return self.sun_sidecar(inp).item()

    def get_storm_effect(self, kp_penalty):
        with torch.no_grad():
            inp = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
            return self.storm_sidecar(inp).item()


# ── Dataset ──────────────────────────────────────────────────────────────────

class SignatureDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_combined_data():
    """Load all data sources and apply PER-SOURCE normalization (V19)."""

    log.info(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # ── WSPR Signatures (floor layer) ──
    wspr_count = client.command("SELECT count() FROM wspr.signatures_v2_terrestrial")
    log.info(f"WSPR terrestrial signatures available: {wspr_count:,}")

    wspr_limit = f" LIMIT {WSPR_SAMPLE}" if WSPR_SAMPLE else ""
    wspr_order = " ORDER BY rand()" if WSPR_SAMPLE else ""

    wspr_query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM wspr.signatures_v2_terrestrial
    WHERE avg_sfi > 0
    {wspr_order}{wspr_limit}
    """

    log.info(f"Loading WSPR floor data (sample {WSPR_SAMPLE:,})...")
    t0 = time.perf_counter()
    wspr_df = client.query_df(wspr_query)
    log.info(f"WSPR: {len(wspr_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # ── RBN Full Signatures (middle layer) ──
    rbn_full_count = client.command("SELECT count() FROM rbn.signatures")
    log.info(f"RBN full signatures available: {rbn_full_count:,}")

    rbn_full_limit = f" LIMIT {RBN_FULL_SAMPLE}" if RBN_FULL_SAMPLE else ""
    rbn_full_order = " ORDER BY rand()" if RBN_FULL_SAMPLE else ""

    rbn_full_query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM rbn.signatures
    WHERE avg_sfi > 0
    {rbn_full_order}{rbn_full_limit}
    """

    log.info(f"Loading RBN middle data (sample {RBN_FULL_SAMPLE:,})...")
    t0 = time.perf_counter()
    rbn_full_df = client.query_df(rbn_full_query)
    log.info(f"RBN Full: {len(rbn_full_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # ── RBN DXpedition Signatures (rare paths) ──
    rbn_dx_count = client.command("SELECT count() FROM rbn.dxpedition_signatures")
    log.info(f"RBN DXpedition signatures available: {rbn_dx_count:,}")

    rbn_dx_query = """
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM rbn.dxpedition_signatures
    WHERE avg_sfi > 0
    """

    log.info("Loading RBN DXpedition data (all rows)...")
    t0 = time.perf_counter()
    rbn_dx_df = client.query_df(rbn_dx_query)
    log.info(f"RBN DXpedition: {len(rbn_dx_df):,} rows in {time.perf_counter() - t0:.1f}s")

    # ── Contest Signatures (ceiling layer) ──
    contest_count = client.command("SELECT count() FROM contest.signatures")
    log.info(f"Contest signatures available: {contest_count:,}")

    contest_limit = f" LIMIT {CONTEST_SAMPLE}" if CONTEST_SAMPLE else ""
    contest_order = " ORDER BY rand()" if CONTEST_SAMPLE else ""

    contest_query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM contest.signatures
    WHERE avg_sfi > 0
    {contest_order}{contest_limit}
    """

    log.info("Loading Contest ceiling data (all rows)...")
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

    # ── V19: PER-SOURCE NORMALIZATION ──
    log.info("")
    log.info("=== V19.1 PER-SOURCE NORMALIZATION (Rosetta Stone) ===")

    # Calculate per-source statistics
    wspr_snr_raw = wspr_df['median_snr'].values.astype(np.float32)
    rbn_full_snr_raw = rbn_full_df['median_snr'].values.astype(np.float32)
    rbn_dx_snr_raw = rbn_dx_df['median_snr'].values.astype(np.float32)
    contest_snr_raw = contest_df['median_snr'].values.astype(np.float32)

    # Compute per-source constants
    norm_constants = {
        "wspr": {
            "mean": float(wspr_snr_raw.mean()),
            "std": float(wspr_snr_raw.std())
        },
        "rbn": {
            "mean": float(np.concatenate([rbn_full_snr_raw, rbn_dx_snr_raw]).mean()),
            "std": float(np.concatenate([rbn_full_snr_raw, rbn_dx_snr_raw]).std())
        },
        "contest": {
            "mean": float(contest_snr_raw.mean()),
            "std": float(contest_snr_raw.std())
        }
    }

    log.info("Per-source normalization constants:")
    log.info(f"  WSPR:    mean={norm_constants['wspr']['mean']:.2f} dB, std={norm_constants['wspr']['std']:.2f} dB")
    log.info(f"  RBN:     mean={norm_constants['rbn']['mean']:.2f} dB, std={norm_constants['rbn']['std']:.2f} dB")
    log.info(f"  Contest: mean={norm_constants['contest']['mean']:.2f} dB, std={norm_constants['contest']['std']:.2f} dB")

    # Normalize each source independently to Z-scores
    wspr_snr_z = (wspr_snr_raw - norm_constants['wspr']['mean']) / norm_constants['wspr']['std']
    rbn_full_snr_z = (rbn_full_snr_raw - norm_constants['rbn']['mean']) / norm_constants['rbn']['std']
    rbn_dx_snr_z = (rbn_dx_snr_raw - norm_constants['rbn']['mean']) / norm_constants['rbn']['std']
    contest_snr_z = (contest_snr_raw - norm_constants['contest']['mean']) / norm_constants['contest']['std']

    log.info("")
    log.info("Normalized Z-scores (should all be ~0 mean, ~1 std):")
    log.info(f"  WSPR:    mean={wspr_snr_z.mean():.4f}, std={wspr_snr_z.std():.4f}")
    log.info(f"  RBN:     mean={rbn_full_snr_z.mean():.4f}, std={rbn_full_snr_z.std():.4f}")
    log.info(f"  Contest: mean={contest_snr_z.mean():.4f}, std={contest_snr_z.std():.4f}")

    # ── Engineer Features ──
    log.info("")
    log.info("Engineering features...")
    wspr_X = engineer_features(wspr_df)
    rbn_full_X = engineer_features(rbn_full_df)
    rbn_dx_X = engineer_features(rbn_dx_df)
    contest_X = engineer_features(contest_df)

    # ── Prepare weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w = wspr_w / wspr_w.mean()

    rbn_full_w = rbn_full_df['spot_count'].values.astype(np.float32)
    rbn_full_w = rbn_full_w / rbn_full_w.mean()

    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w = rbn_dx_w / rbn_dx_w.mean()

    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w = contest_w / contest_w.mean()

    # ── Upsample DXpedition ──
    log.info(f"Upsampling RBN DXpedition {RBN_DX_UPSAMPLE_FACTOR}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (RBN_DX_UPSAMPLE_FACTOR, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr_z, RBN_DX_UPSAMPLE_FACTOR)
    rbn_dx_w_up = np.tile(rbn_dx_w, RBN_DX_UPSAMPLE_FACTOR)
    log.info(f"  DXpedition effective rows: {len(rbn_dx_snr_up):,}")

    # ── Upsample Contest if needed ──
    if CONTEST_UPSAMPLE_FACTOR > 1:
        log.info(f"Upsampling Contest data {CONTEST_UPSAMPLE_FACTOR}x...")
        contest_X_up = np.tile(contest_X, (CONTEST_UPSAMPLE_FACTOR, 1))
        contest_snr_up = np.tile(contest_snr_z, CONTEST_UPSAMPLE_FACTOR)
        contest_w_up = np.tile(contest_w, CONTEST_UPSAMPLE_FACTOR)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr_z
        contest_w_up = contest_w
    log.info(f"  Contest effective rows: {len(contest_snr_up):,}")

    # ── Combine all sources (NOW IN Z-SCORE UNITS) ──
    X_combined = np.vstack([wspr_X, rbn_full_X, rbn_dx_X_up, contest_X_up])
    y_combined = np.concatenate([wspr_snr_z, rbn_full_snr_z, rbn_dx_snr_up, contest_snr_up])
    w_combined = np.concatenate([wspr_w, rbn_full_w, rbn_dx_w_up, contest_w_up])

    total_before_storm = len(X_combined)
    log.info("")
    log.info(f"Combined dataset (before storm upsample): {total_before_storm:,} rows")
    log.info(f"  WSPR (floor):       {len(wspr_X):,} ({100*len(wspr_X)/total_before_storm:.1f}%)")
    log.info(f"  RBN Full (middle):  {len(rbn_full_X):,} ({100*len(rbn_full_X)/total_before_storm:.1f}%)")
    log.info(f"  RBN DX (rare):      {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/total_before_storm:.1f}%)")
    log.info(f"  Contest (ceiling):  {len(contest_snr_up):,} ({100*len(contest_snr_up)/total_before_storm:.1f}%)")

    # ── Storm upsample (DOUBLED TO 20x) ──
    storm_threshold = 4.0 / 9.0  # kp >= 5
    storm_mask = X_combined[:, KP_PENALTY_IDX] <= storm_threshold
    storm_count = storm_mask.sum()

    log.info(f"Storm upsample: {storm_count:,} rows with Kp >= 5 ({100*storm_count/total_before_storm:.2f}%)")

    if storm_count > 0 and STORM_UPSAMPLE_FACTOR > 1:
        storm_X = X_combined[storm_mask]
        storm_y = y_combined[storm_mask]
        storm_w = w_combined[storm_mask]

        log.info(f"  Upsampling storm rows {STORM_UPSAMPLE_FACTOR}x (DOUBLED from V19)...")
        X_combined = np.vstack([X_combined] + [storm_X] * STORM_UPSAMPLE_FACTOR)
        y_combined = np.concatenate([y_combined] + [storm_y] * STORM_UPSAMPLE_FACTOR)
        w_combined = np.concatenate([w_combined] + [storm_w] * STORM_UPSAMPLE_FACTOR)

        log.info(f"  After storm upsample: {len(X_combined):,} rows (+{storm_count * STORM_UPSAMPLE_FACTOR:,})")

    # Verify combined Z-scores
    log.info("")
    log.info(f"Combined targets (Z-scores): mean={y_combined.mean():.4f}, std={y_combined.std():.4f}")

    y_combined = y_combined.reshape(-1, 1)
    w_combined = w_combined.reshape(-1, 1)

    # Free DataFrames
    del wspr_df, rbn_full_df, rbn_dx_df, contest_df
    del wspr_X, rbn_full_X, rbn_dx_X, contest_X, rbn_dx_X_up, contest_X_up
    del wspr_snr_raw, rbn_full_snr_raw, rbn_dx_snr_raw, contest_snr_raw
    gc.collect()

    return X_combined, y_combined, w_combined, date_range, norm_constants


# ── Two-Stage Training Helpers ───────────────────────────────────────────────

def freeze_trunk(model):
    """Freeze DNN trunk and gates, train only sidecars."""
    for param in model.trunk.parameters():
        param.requires_grad = False
    for param in model.sun_gate.parameters():
        param.requires_grad = False
    for param in model.storm_gate.parameters():
        param.requires_grad = False
    # Sidecars remain trainable
    for param in model.sun_sidecar.parameters():
        param.requires_grad = True
    for param in model.storm_sidecar.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    log.info(f"IONIS V19.1 Physics First | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Data sources: WSPR + RBN Full + RBN DX ({RBN_DX_UPSAMPLE_FACTOR}x) + Contest")
    log.info(f"Normalization: PER-SOURCE Z-score (Rosetta Stone)")
    log.info(f"Storm upsample: {STORM_UPSAMPLE_FACTOR}x for Kp >= 5 (DOUBLED)")
    log.info(f"Two-stage training: {WARMUP_EPOCHS} warmup epochs (trunk frozen)")
    log.info("")

    # ── Load Data ──
    X_np, y_np, w_np, date_range, norm_constants = load_combined_data()

    n = len(X_np)
    log.info(f"Dataset: {n:,} rows x {INPUT_DIM} features")
    log.info(f"Source data range: {date_range}")

    dataset = SignatureDataset(X_np, y_np, w_np)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
    )

    # ── Model ──
    model = IonisV12Gate(dnn_dim=DNN_DIM, hidden=256, sidecar_hidden=8).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())

    # Start with trunk frozen (Stage 1)
    freeze_trunk(model)
    trainable_params = count_trainable(model)
    log.info(f"Model: IonisV12Gate ({total_params:,} params)")
    log.info(f"Stage 1: Trainable: {trainable_params:,} / {total_params:,} (sidecars only)")
    log.info("")

    # ── Optimizer ──
    # Use all parameters but frozen ones will have no grad
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss(reduction='none')

    # ── Training Loop ──
    print(f"Training started ({EPOCHS} epochs, {WARMUP_EPOCHS} warmup)")
    hdr = " Ep     Train       Val     RMSE   Pearson   SFI+   Kp9-    Time  Stage"
    log.info(hdr)
    log.info("-" * len(hdr))

    best_val_loss = float('inf')
    best_pearson = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()

        # ── Stage Transition ──
        if epoch == WARMUP_EPOCHS + 1:
            log.info("")
            log.info("=== STAGE 2: UNFREEZING TRUNK ===")
            unfreeze_all(model)
            trainable_params = count_trainable(model)
            log.info(f"Stage 2: Trainable: {trainable_params:,} / {total_params:,} (all)")
            # Recreate optimizer with all parameters
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
            # Continue scheduler from where we were
            for _ in range(WARMUP_EPOCHS):
                scheduler.step()
            log.info("")
            log.info(hdr)
            log.info("-" * len(hdr))

        stage = "S1" if epoch <= WARMUP_EPOCHS else "S2"

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

            var_loss = -pred.var() * LAMBDA_VAR
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
        sfi_benefit = model.get_sun_effect(200.0 / 300.0) - model.get_sun_effect(70.0 / 300.0)
        storm_cost = model.get_storm_effect(1.0) - model.get_storm_effect(0.0)

        epoch_sec = time.perf_counter() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pearson = val_pearson
            torch.save({
                'model_state_dict': model.state_dict(),
                'norm_constants': norm_constants,
                'input_dim': INPUT_DIM,
                'dnn_dim': DNN_DIM,
                'sfi_idx': SFI_IDX,
                'kp_idx': KP_PENALTY_IDX,
                'features': FEATURES,
                'band_to_hz': BAND_TO_HZ,
                'date_range': date_range,
                'sample_size': n,
                'wspr_sample': WSPR_SAMPLE,
                'rbn_full_sample': RBN_FULL_SAMPLE,
                'rbn_dx_upsample_factor': RBN_DX_UPSAMPLE_FACTOR,
                'contest_upsample_factor': CONTEST_UPSAMPLE_FACTOR,
                'storm_upsample_factor': STORM_UPSAMPLE_FACTOR,
                'warmup_epochs': WARMUP_EPOCHS,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'version': VERSION,
                'architecture': 'IonisV12Gate',
                'data_sources': [
                    'wspr.signatures_v2_terrestrial',
                    'rbn.signatures',
                    'rbn.dxpedition_signatures',
                    'contest.signatures',
                ],
                'normalization': 'per_source',
                'sfi_benefit_normalized': sfi_benefit,
                'storm_cost_normalized': storm_cost,
            }, MODEL_PATH)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.2f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s  {stage}{marker}"
        )

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {MODEL_PATH}")

    # ── Physics Report ──
    log.info("")
    log.info("=== PHYSICS VERIFICATION (normalized σ units) ===")
    sfi_200 = model.get_sun_effect(200.0 / 300.0)
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    kp0 = model.get_storm_effect(1.0)
    kp9 = model.get_storm_effect(0.0)

    log.info(f"SFI 70→200 benefit: {sfi_200 - sfi_70:+.3f}σ")
    log.info(f"Kp 0→9 storm cost: {kp0 - kp9:+.3f}σ")

    # Convert to dB using typical WSPR std for reference
    wspr_std = norm_constants['wspr']['std']
    log.info(f"  (≈ {(sfi_200 - sfi_70) * wspr_std:+.1f} dB benefit in WSPR scale)")
    log.info(f"  (≈ {(kp0 - kp9) * wspr_std:+.1f} dB storm cost in WSPR scale)")

    if sfi_200 > sfi_70:
        log.info("SFI monotonicity: CORRECT")
    else:
        log.info("SFI monotonicity: WARNING — inverted")

    if kp0 > kp9:
        log.info("Kp monotonicity: CORRECT")
    else:
        log.info("Kp monotonicity: WARNING — inverted")

    # Physics assessment
    log.info("")
    log.info("=== V19.1 PHYSICS ASSESSMENT ===")
    kp_threshold = 0.10  # Red line from Gemini
    if storm_cost >= kp_threshold:
        log.info(f"Kp9- sidecar: {storm_cost:+.3f}σ >= {kp_threshold}σ — ALIVE")
    else:
        log.info(f"Kp9- sidecar: {storm_cost:+.3f}σ < {kp_threshold}σ — DEAD (remediation failed)")

    log.info("")
    log.info("=== V19.1 ROSETTA STONE SUMMARY ===")
    log.info("Per-source normalization constants stored in checkpoint:")
    for source, vals in norm_constants.items():
        log.info(f"  {source}: mean={vals['mean']:.2f}, std={vals['std']:.2f}")
    log.info("")
    log.info("Inference (Rosetta Stone decoder):")
    log.info("  Model outputs Z-score (σ) = signal quality relative to average")
    log.info("  For WSPR/FT8: snr_dB = σ × wspr_std + wspr_mean")
    log.info("  For CW/RTTY:  snr_dB = σ × rbn_std  + rbn_mean")
    log.info("  For SSB:      snr_dB = σ × contest_std + contest_mean")
    log.info("")
    log.info("Run oracle_v19.py to test predictions with Rosetta Stone decoder.")


if __name__ == '__main__':
    main()
