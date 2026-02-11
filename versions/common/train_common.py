"""
train_common.py — Shared training utilities for IONIS models

This module contains all shared code for training:
- Model architectures (IonisModel, MonotonicMLP)
- Feature engineering
- Grid utilities
- Dataset class
- Data loading from ClickHouse

Version-specific training scripts should:
1. Load their config from JSON
2. Import from this module
3. Call the training functions with their config
"""

import gc
import math
import re
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import clickhouse_connect

log = logging.getLogger(__name__)


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

def engineer_features(df, config):
    """
    Compute features from signature columns.

    Args:
        df: DataFrame with signature data
        config: Dict with model config (band_to_hz, input_dim, sfi_idx, kp_penalty_idx)

    Returns:
        np.ndarray of shape (n_rows, input_dim)
    """
    band_to_hz = config["band_to_hz"]
    input_dim = config["model"]["input_dim"]
    sfi_idx = config["model"]["sfi_idx"]
    kp_penalty_idx = config["model"]["kp_penalty_idx"]

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

    # Convert band_to_hz keys to int if they're strings (from JSON)
    if isinstance(list(band_to_hz.keys())[0], str):
        band_to_hz = {int(k): v for k, v in band_to_hz.items()}

    freq_hz = np.array([band_to_hz.get(b, 14_097_100) for b in band], dtype=np.float32)

    X = np.zeros((len(df), input_dim), dtype=np.float32)

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
    X[:, sfi_idx] = sfi / 300.0
    X[:, kp_penalty_idx] = 1.0 - kp / 9.0  # kp_penalty

    return X


# ── Model Architecture ───────────────────────────────────────────────────────

class MonotonicMLP(nn.Module):
    """Monotonically increasing MLP for physics constraints."""

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


class IonisModel(nn.Module):
    """
    Gated dual-sidecar architecture for HF propagation prediction.

    Components:
        - trunk: DNN for geography/time features
        - sun_sidecar: MonotonicMLP for SFI effect
        - storm_sidecar: MonotonicMLP for Kp effect
        - sun_gate, storm_gate: Context-dependent gating

    Args:
        config: Dict with model parameters (dnn_dim, hidden_dim, sidecar_hidden, gate_init_bias)
    """

    def __init__(self, config):
        super().__init__()

        dnn_dim = config["model"]["dnn_dim"]
        hidden = config["model"]["hidden_dim"]
        sidecar_hidden = config["model"]["sidecar_hidden"]
        gate_init_bias = config["model"]["gate_init_bias"]

        # Store config for reference
        self.config = config
        self.dnn_dim = dnn_dim
        self.hidden = hidden
        self.sidecar_hidden = sidecar_hidden

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

        self._init_gates(gate_init_bias)

    def _init_gates(self, gate_init_bias):
        for gate in [self.sun_gate, self.storm_gate]:
            if hasattr(gate[-1], 'bias') and gate[-1].bias is not None:
                nn.init.constant_(gate[-1].bias, gate_init_bias)
                gate[-1].bias.requires_grad = True
            if hasattr(gate[-1], 'weight'):
                nn.init.xavier_normal_(gate[-1].weight, gain=0.1)

    def forward(self, x):
        dnn_dim = self.config["model"]["dnn_dim"]
        sfi_idx = self.config["model"]["sfi_idx"]
        kp_idx = self.config["model"]["kp_penalty_idx"]

        x_dnn = x[:, :dnn_dim]
        sfi_in = x[:, sfi_idx:sfi_idx+1]
        kp_penalty = x[:, kp_idx:kp_idx+1]

        base = self.trunk(x_dnn)
        sun_effect = self.sun_sidecar(sfi_in)
        storm_effect = self.storm_sidecar(kp_penalty)

        sun_g = torch.sigmoid(self.sun_gate(x_dnn)) + 0.5
        storm_g = torch.sigmoid(self.storm_gate(x_dnn)) + 0.5

        return base + sun_g * sun_effect + storm_g * storm_effect

    def forward_with_components(self, x):
        dnn_dim = self.config["model"]["dnn_dim"]
        sfi_idx = self.config["model"]["sfi_idx"]
        kp_idx = self.config["model"]["kp_penalty_idx"]

        x_dnn = x[:, :dnn_dim]
        sfi_in = x[:, sfi_idx:sfi_idx+1]
        kp_penalty = x[:, kp_idx:kp_idx+1]

        base = self.trunk(x_dnn)
        sun_effect = self.sun_sidecar(sfi_in)
        storm_effect = self.storm_sidecar(kp_penalty)

        sun_g = torch.sigmoid(self.sun_gate(x_dnn)) + 0.5
        storm_g = torch.sigmoid(self.storm_gate(x_dnn)) + 0.5

        total = base + sun_g * sun_effect + storm_g * storm_effect
        return total, base, sun_g, storm_g, sun_effect, storm_effect

    def get_sun_effect(self, sfi_normalized, device):
        with torch.no_grad():
            inp = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=device)
            return self.sun_sidecar(inp).item()

    def get_storm_effect(self, kp_penalty, device):
        with torch.no_grad():
            inp = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
            return self.storm_sidecar(inp).item()


# ── V16 Production Model ─────────────────────────────────────────────────────

def _gate_v16(x):
    """V16 gate function: range 0.5 to 2.0"""
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    """
    V16 Production Model — validated at 98.5% FT8 recall on PSKR live data.

    Key differences from IonisModel:
        - Gates from trunk output (256-dim), not raw input (11-dim)
        - Gate range 0.5-2.0 (vs 0.5-1.5)
        - Separate base_head (256→128→1) and scaler_heads (256→64→1)
        - Uses Mish activation, no LayerNorm or Dropout
        - Requires defibrillator init and weight clamping to keep sidecars alive

    Args:
        dnn_dim: Number of geography/time features (default 11)
        sidecar_hidden: Hidden units in MonotonicMLP (default 8)
        sfi_idx: Index of SFI feature in input (default 11)
        kp_penalty_idx: Index of Kp penalty feature in input (default 12)
        gate_init_bias: Initial bias for scaler heads (default -ln(2))
    """

    def __init__(self, dnn_dim=11, sidecar_hidden=8, sfi_idx=11, kp_penalty_idx=12,
                 gate_init_bias=None):
        super().__init__()

        if gate_init_bias is None:
            gate_init_bias = -math.log(2.0)

        self.dnn_dim = dnn_dim
        self.sfi_idx = sfi_idx
        self.kp_penalty_idx = kp_penalty_idx

        # Trunk: geography/time features → 256-dim representation
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        # Base head: trunk → SNR prediction
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )

        # Scaler heads: trunk → gate logits (256-dim input, expressive)
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # Physics sidecars (monotonic)
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        # Initialize scaler heads
        self._init_scaler_heads(gate_init_bias)

    def _init_scaler_heads(self, gate_init_bias):
        """Initialize scaler head biases for balanced gates."""
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, gate_init_bias)

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate_v16(sun_logit)
        storm_gate = _gate_v16(storm_logit)

        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)

    def forward_with_gates(self, x):
        """Forward pass returning gate values for variance loss."""
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate_v16(sun_logit)
        storm_gate = _gate_v16(storm_logit)

        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost, \
               sun_gate, storm_gate

    def get_sun_effect(self, sfi_normalized, device):
        """Get raw sun sidecar output for a given SFI value."""
        with torch.no_grad():
            x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=device)
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty, device):
        """Get raw storm sidecar output for a given Kp penalty value."""
        with torch.no_grad():
            x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        """Get gate values without gradient tracking."""
        x_deep = x[:, :self.dnn_dim]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate_v16(sun_logit), _gate_v16(storm_logit)


def init_v16_defibrillator(model):
    """
    Apply V16 defibrillator initialization to keep sidecars alive.

    CRITICAL: Call this after model creation, before training.

    This init:
        - Sets sidecar weights to uniform(0.8, 1.2) instead of random
        - Sets fc2.bias to -10.0 (strong initial offset)
        - Freezes fc1.bias (prevents collapse)
    """
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

    # Freeze fc1 bias, keep fc2 bias learnable
    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True

    log.info("Defibrillator applied: sidecar weights uniform(0.8-1.2), fc2.bias=-10.0, fc1.bias frozen")


def clamp_v16_sidecars(model):
    """
    Clamp sidecar weights to [0.5, 2.0] to prevent collapse.

    CRITICAL: Call this after every optimizer.step() during training.

    This prevents sidecars from collapsing to zero, which is what
    killed every V19 variant.
    """
    with torch.no_grad():
        for sidecar in [model.sun_sidecar, model.storm_sidecar]:
            sidecar.fc1.weight.clamp_(0.5, 2.0)
            sidecar.fc2.weight.clamp_(0.5, 2.0)


def get_v16_optimizer_groups(model, trunk_lr=1e-5, scaler_lr=5e-5, sidecar_lr=1e-3):
    """
    Get V16's 6-group optimizer configuration.

    Args:
        model: IonisV12Gate model
        trunk_lr: Learning rate for trunk and base_head
        scaler_lr: Learning rate for scaler heads (intermediate)
        sidecar_lr: Learning rate for sidecars (fastest)

    Returns:
        List of parameter groups for AdamW optimizer
    """
    return [
        {'params': model.trunk.parameters(), 'lr': trunk_lr},
        {'params': model.base_head.parameters(), 'lr': trunk_lr},
        {'params': model.sun_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': model.storm_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': [p for p in model.sun_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
    ]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SignatureDataset(Dataset):
    """PyTorch Dataset for signature data."""

    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_source_data(client, table, sample_size=None, where_clause="avg_sfi > 0"):
    """Load signature data from a ClickHouse table."""

    count = client.command(f"SELECT count() FROM {table}")
    log.info(f"{table}: {count:,} rows available")

    limit = f" LIMIT {sample_size}" if sample_size else ""
    order = " ORDER BY rand()" if sample_size else ""

    query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM {table}
    WHERE {where_clause}
    {order}{limit}
    """

    t0 = time.perf_counter()
    df = client.query_df(query)
    elapsed = time.perf_counter() - t0
    log.info(f"{table}: loaded {len(df):,} rows in {elapsed:.1f}s")

    return df


def load_combined_data(config):
    """
    Load all data sources and apply per-source normalization.

    Args:
        config: Dict with data configuration

    Returns:
        X_combined, y_combined, w_combined, date_range, norm_constants
    """
    ch_host = config["clickhouse"]["host"]
    ch_port = config["clickhouse"]["port"]

    wspr_sample = config["data"]["wspr_sample"]
    rbn_full_sample = config["data"]["rbn_full_sample"]
    rbn_dx_upsample = config["data"]["rbn_dx_upsample"]
    contest_upsample = config["data"]["contest_upsample"]
    storm_upsample = config["data"]["storm_upsample"]
    kp_penalty_idx = config["model"]["kp_penalty_idx"]

    log.info(f"Connecting to ClickHouse at {ch_host}:{ch_port}...")
    client = clickhouse_connect.get_client(host=ch_host, port=ch_port)

    # Load each source
    wspr_df = load_source_data(client, "wspr.signatures_v2_terrestrial", wspr_sample)

    # Guard: skip RBN Full when sample == 0 (V16 recipe)
    if rbn_full_sample > 0:
        rbn_full_df = load_source_data(client, "rbn.signatures", rbn_full_sample)
    else:
        log.info("rbn.signatures: skipped (rbn_full_sample=0)")
        rbn_full_df = None

    rbn_dx_df = load_source_data(client, "rbn.dxpedition_signatures", None)
    contest_df = load_source_data(client, "contest.signatures", None)

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

    # ── Per-source normalization ──
    log.info("")
    log.info("=== PER-SOURCE NORMALIZATION (Rosetta Stone) ===")

    wspr_snr_raw = wspr_df['median_snr'].values.astype(np.float32)
    rbn_dx_snr_raw = rbn_dx_df['median_snr'].values.astype(np.float32)
    contest_snr_raw = contest_df['median_snr'].values.astype(np.float32)

    # Handle RBN Full (may be None if rbn_full_sample=0)
    if rbn_full_df is not None:
        rbn_full_snr_raw = rbn_full_df['median_snr'].values.astype(np.float32)
        rbn_all_snr = np.concatenate([rbn_full_snr_raw, rbn_dx_snr_raw])
    else:
        rbn_full_snr_raw = np.array([], dtype=np.float32)
        rbn_all_snr = rbn_dx_snr_raw  # DXpedition only

    norm_constants = {
        "wspr": {
            "mean": float(wspr_snr_raw.mean()),
            "std": float(wspr_snr_raw.std())
        },
        "rbn": {
            "mean": float(rbn_all_snr.mean()),
            "std": float(rbn_all_snr.std())
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

    # Normalize to Z-scores
    wspr_snr_z = (wspr_snr_raw - norm_constants['wspr']['mean']) / norm_constants['wspr']['std']
    rbn_dx_snr_z = (rbn_dx_snr_raw - norm_constants['rbn']['mean']) / norm_constants['rbn']['std']
    contest_snr_z = (contest_snr_raw - norm_constants['contest']['mean']) / norm_constants['contest']['std']

    # RBN Full Z-scores (empty array if skipped)
    if len(rbn_full_snr_raw) > 0:
        rbn_full_snr_z = (rbn_full_snr_raw - norm_constants['rbn']['mean']) / norm_constants['rbn']['std']
    else:
        rbn_full_snr_z = np.array([], dtype=np.float32)

    log.info("")
    log.info("Normalized Z-scores (should all be ~0 mean, ~1 std):")
    log.info(f"  WSPR:    mean={wspr_snr_z.mean():.4f}, std={wspr_snr_z.std():.4f}")
    if len(rbn_full_snr_z) > 0:
        log.info(f"  RBN:     mean={rbn_full_snr_z.mean():.4f}, std={rbn_full_snr_z.std():.4f}")
    else:
        log.info(f"  RBN:     (skipped - DXpedition only)")
    log.info(f"  Contest: mean={contest_snr_z.mean():.4f}, std={contest_snr_z.std():.4f}")

    # ── Engineer Features ──
    log.info("")
    log.info("Engineering features...")
    wspr_X = engineer_features(wspr_df, config)
    rbn_dx_X = engineer_features(rbn_dx_df, config)
    contest_X = engineer_features(contest_df, config)

    # RBN Full features (empty if skipped)
    if rbn_full_df is not None:
        rbn_full_X = engineer_features(rbn_full_df, config)
    else:
        input_dim = config["model"]["input_dim"]
        rbn_full_X = np.zeros((0, input_dim), dtype=np.float32)

    # ── Prepare weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w = wspr_w / wspr_w.mean()

    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w = rbn_dx_w / rbn_dx_w.mean()

    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w = contest_w / contest_w.mean()

    # RBN Full weights (empty if skipped)
    if rbn_full_df is not None:
        rbn_full_w = rbn_full_df['spot_count'].values.astype(np.float32)
        rbn_full_w = rbn_full_w / rbn_full_w.mean()
    else:
        rbn_full_w = np.array([], dtype=np.float32)

    # ── Upsample DXpedition ──
    log.info(f"Upsampling RBN DXpedition {rbn_dx_upsample}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (rbn_dx_upsample, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr_z, rbn_dx_upsample)
    rbn_dx_w_up = np.tile(rbn_dx_w, rbn_dx_upsample)
    log.info(f"  DXpedition effective rows: {len(rbn_dx_snr_up):,}")

    # ── Upsample Contest if needed ──
    if contest_upsample > 1:
        log.info(f"Upsampling Contest data {contest_upsample}x...")
        contest_X_up = np.tile(contest_X, (contest_upsample, 1))
        contest_snr_up = np.tile(contest_snr_z, contest_upsample)
        contest_w_up = np.tile(contest_w, contest_upsample)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr_z
        contest_w_up = contest_w
    log.info(f"  Contest effective rows: {len(contest_snr_up):,}")

    # ── Combine all sources ──
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

    # ── Storm upsample ──
    storm_threshold = 4.0 / 9.0  # kp >= 5
    storm_mask = X_combined[:, kp_penalty_idx] <= storm_threshold
    storm_count = storm_mask.sum()

    log.info(f"Storm upsample: {storm_count:,} rows with Kp >= 5 ({100*storm_count/total_before_storm:.2f}%)")

    if storm_count > 0 and storm_upsample > 1:
        storm_X = X_combined[storm_mask]
        storm_y = y_combined[storm_mask]
        storm_w = w_combined[storm_mask]

        log.info(f"  Upsampling storm rows {storm_upsample}x...")
        X_combined = np.vstack([X_combined] + [storm_X] * storm_upsample)
        y_combined = np.concatenate([y_combined] + [storm_y] * storm_upsample)
        w_combined = np.concatenate([w_combined] + [storm_w] * storm_upsample)

        log.info(f"  After storm upsample: {len(X_combined):,} rows (+{storm_count * storm_upsample:,})")

    log.info("")
    log.info(f"Combined targets (Z-scores): mean={y_combined.mean():.4f}, std={y_combined.std():.4f}")

    y_combined = y_combined.reshape(-1, 1)
    w_combined = w_combined.reshape(-1, 1)

    # Free DataFrames
    del wspr_df, rbn_dx_df, contest_df
    if rbn_full_df is not None:
        del rbn_full_df
    del wspr_X, rbn_full_X, rbn_dx_X, contest_X, rbn_dx_X_up, contest_X_up
    del wspr_snr_raw, rbn_full_snr_raw, rbn_dx_snr_raw, contest_snr_raw
    gc.collect()

    return X_combined, y_combined, w_combined, date_range, norm_constants


# ── Config Logging ───────────────────────────────────────────────────────────

def log_config(config, config_file, device):
    """Log all configuration at the start of training."""

    log.info(f"{'='*70}")
    log.info(f"IONIS {config['version']} | {config['phase']}")
    log.info(f"{'='*70}")
    log.info(f"Config: {config_file}")
    log.info("")

    log.info("=== TRAINING ===")
    log.info(f"  Device: {device}")
    log.info(f"  Epochs: {config['training']['epochs']}")
    log.info(f"  Batch size: {config['training']['batch_size']:,}")
    log.info(f"  Validation split: {config['training']['val_split']:.0%}")
    log.info("")

    log.info("=== DATA ===")
    log.info(f"  WSPR sample: {config['data']['wspr_sample']:,}")
    log.info(f"  RBN Full sample: {config['data']['rbn_full_sample']:,}")
    log.info(f"  RBN DX upsample: {config['data']['rbn_dx_upsample']}x")
    log.info(f"  Contest upsample: {config['data']['contest_upsample']}x")
    log.info(f"  Storm upsample: {config['data']['storm_upsample']}x (Kp >= 5)")
    log.info("")

    log.info("=== LEARNING RATES ===")
    trunk_lr = config['training']['trunk_lr']
    sidecar_lr = config['training']['sidecar_lr']
    log.info(f"  Trunk + Gates: {trunk_lr}")
    log.info(f"  Sidecars: {sidecar_lr} ({sidecar_lr/trunk_lr:.0f}x boost)")
    log.info(f"  Weight decay: {config['training']['weight_decay']}")
    log.info("")

    log.info("=== MODEL ===")
    log.info(f"  Architecture: {config['model']['architecture']}")
    log.info(f"  DNN dim: {config['model']['dnn_dim']}")
    hidden = config['model']['hidden_dim']
    log.info(f"  Hidden: {hidden} (trunk: {hidden*2}→{hidden}→{hidden//2}→1)")
    log.info(f"  Sidecar hidden: {config['model']['sidecar_hidden']}")
    log.info("")
