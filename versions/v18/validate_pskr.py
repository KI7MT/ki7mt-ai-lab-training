#!/usr/bin/env python3
"""
validate_pskr.py — PSK Reporter Live Validation for IONIS V18

Validates IONIS V18 predictions against PSK Reporter observations.
This is the "acid test" — comparing predictions against data the model
has never seen, from an independent source.

V18 Global Normalization Fix:
  - Global raw-dB normalization (not per-source Z-score)
  - Denormalization: snr_dB = prediction_σ × global_std + global_mean
  - Expected recall: 80-95% (not 99% like broken V17)
  - Training: 51M+ rows (WSPR + RBN + Contest + DXpedition)

Usage:
    python validate_pskr.py [--data-dir /path/to/pskr-data] [--sample N]

Requirements:
    - PSK Reporter JSONL.gz files (from pskr-collector)
    - V18 checkpoint (ionis_v18.pth)
    - Solar data from ClickHouse for validation period
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================

# Default paths
DEFAULT_DATA_DIR = "/tmp/pskr-validation"
SCRIPT_DIR = Path(__file__).parent
V18_CHECKPOINT = SCRIPT_DIR / "ionis_v18.pth"

# Also try models/ directory
if not V18_CHECKPOINT.exists():
    V18_CHECKPOINT = SCRIPT_DIR.parent.parent / "models" / "ionis_v18.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model constants (must match training)
DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13
GATE_INIT_BIAS = -math.log(2.0)

# Mode thresholds (dB) - propagation is "available" if predicted SNR >= threshold
# Calibrated for SDR/skimmer sensitivity (V16.1 update based on PSK Reporter validation)
MODE_THRESHOLDS = {
    "WSPR": -28,
    "FT8": -20,
    "FT4": -20,    # V16.1: was -17, aligned with FT8 parity
    "JS8": -21,    # V16.1: was -15, JS8 "Slow" mode sensitivity
    "CW": -18,     # V16.1: was -10, aligned with CW Skimmer sensitivity
    "RTTY": -5,
    "SSB": 5,
    "PSK31": -13,  # V16.1: was -10
    "JT65": -24,   # Added: JT65 is very sensitive
}

# Default threshold for unknown modes
DEFAULT_THRESHOLD = -15

# ADIF band to frequency (MHz) for model input
BAND_TO_FREQ = {
    102: 1.8,    # 160m
    103: 3.5,    # 80m
    104: 5.3,    # 60m
    105: 7.0,    # 40m
    106: 10.1,   # 30m
    107: 14.0,   # 20m
    108: 18.1,   # 17m
    109: 21.0,   # 15m
    110: 24.9,   # 12m
    111: 28.0,   # 10m
}

BAND_NAMES = {
    102: "160m", 103: "80m", 104: "60m", 105: "40m",
    106: "30m", 107: "20m", 108: "17m", 109: "15m",
    110: "12m", 111: "10m"
}

# ============================================================================
# Model Architecture (IonisV12Gate - must match training exactly)
# ============================================================================

class MonotonicMLP(nn.Module):
    """Single-input MLP with forced monotonicity via abs(weights)."""
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
    """Gate function: sigmoid scaled to [0.5, 2.0]"""
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    """
    V18 architecture with gated monotonic sidecars (same as V12-V16).
    Must match training exactly.
    """
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

# ============================================================================
# Grid Utilities
# ============================================================================

def grid_to_latlon(grid: str) -> tuple:
    """Convert Maidenhead grid to lat/lon (center of grid square)."""
    if not grid or len(grid) < 4:
        return None, None

    grid = grid.upper().strip()

    # Validate format
    if not re.match(r'^[A-R]{2}[0-9]{2}', grid[:4]):
        return None, None

    lon = (ord(grid[0]) - ord('A')) * 20 - 180
    lat = (ord(grid[1]) - ord('A')) * 10 - 90
    lon += int(grid[2]) * 2
    lat += int(grid[3]) * 1

    # 6-char precision
    if len(grid) >= 6 and re.match(r'[A-Xa-x]{2}', grid[4:6]):
        lon += (ord(grid[4].upper()) - ord('A')) * (2/24)
        lat += (ord(grid[5].upper()) - ord('A')) * (1/24)
        lon += 1/24  # Center of subsquare
        lat += 1/48
    else:
        lon += 1  # Center of square
        lat += 0.5

    return lat, lon

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km."""
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def compute_azimuth(lat1, lon1, lat2, lon2):
    """Calculate initial bearing (azimuth) from point 1 to point 2 in degrees."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)

    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)

    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360  # Normalize to 0-360


# ============================================================================
# Feature Engineering (must match training exactly)
# ============================================================================

def compute_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz, hour, month, sfi, kp):
    """
    Compute the 13 input features for IonisV12Gate.
    Must match training feature engineering exactly.

    Feature order (from train_v18.py):
    0. distance / 20000.0
    1. log10(freq_hz) / 8.0
    2. sin(2π × hour / 24)
    3. cos(2π × hour / 24)
    4. sin(2π × azimuth / 360)
    5. cos(2π × azimuth / 360)
    6. |tx_lat - rx_lat| / 180.0
    7. midpoint_lat / 90.0
    8. sin(2π × month / 12)
    9. cos(2π × month / 12)
    10. cos(2π × (hour + midpoint_lon/15) / 24)
    11. sfi / 300.0
    12. (1 - kp/9)  -- kp_penalty
    """
    # Distance (km, normalized)
    dist_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)
    dist_n = dist_km / 20000.0

    # Frequency (Hz, log scale, normalized by 8)
    freq_hz = freq_mhz * 1_000_000  # Convert MHz to Hz
    freq_log = math.log10(freq_hz) / 8.0 if freq_hz > 0 else 0

    # Time encoding (cyclical)
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)

    # Azimuth (bearing from TX to RX)
    azimuth = compute_azimuth(tx_lat, tx_lon, rx_lat, rx_lon)
    azimuth_sin = math.sin(2 * math.pi * azimuth / 360.0)
    azimuth_cos = math.cos(2 * math.pi * azimuth / 360.0)

    # Latitude features
    lat_delta = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0

    # Month encoding (cyclical, month is 1-12)
    month_sin = math.sin(2 * math.pi * month / 12.0)
    month_cos = math.cos(2 * math.pi * month / 12.0)

    # Local solar time at midpoint
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_cos = math.cos(2 * math.pi * (hour + midpoint_lon / 15.0) / 24.0)

    # Solar indices (normalized)
    sfi_n = sfi / 300.0
    kp_penalty = 1.0 - kp / 9.0  # CRITICAL: kp_penalty, not kp_n

    return [
        dist_n,           # 0
        freq_log,         # 1
        hour_sin,         # 2
        hour_cos,         # 3
        azimuth_sin,      # 4
        azimuth_cos,      # 5
        lat_delta,        # 6
        midpoint_lat,     # 7
        month_sin,        # 8
        month_cos,        # 9
        local_solar_cos,  # 10
        sfi_n,            # 11
        kp_penalty,       # 12
    ]

# ============================================================================
# Data Loading
# ============================================================================

def load_pskr_spots(data_dir: str, max_files: int = None):
    """Load PSK Reporter spots from JSONL.gz files."""
    pattern = os.path.join(data_dir, "spots-*.jsonl.gz")
    files = sorted(glob(pattern))

    if not files:
        print(f"No files found matching {pattern}")
        return []

    if max_files:
        files = files[:max_files]

    print(f"Loading {len(files)} files from {data_dir}...")

    spots = []
    for filepath in files:
        filename = os.path.basename(filepath)
        file_count = 0
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        spot = json.loads(line.strip())
                        spots.append(spot)
                        file_count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"  {filename}: {file_count:,} spots")
        except EOFError:
            # File is still being written (truncated gzip)
            print(f"  {filename}: {file_count:,} spots (truncated - still streaming)")
        except Exception as e:
            print(f"  {filename}: error reading - {e}")

    print(f"Total: {len(spots):,} spots loaded")
    return spots

def filter_spots_with_grids(spots: list) -> list:
    """Filter to spots where both sender and receiver grids are present."""
    filtered = [
        s for s in spots
        if s.get('sg') and s.get('rg') and len(s['sg']) >= 4 and len(s['rg']) >= 4
    ]
    pct = 100 * len(filtered) / len(spots) if spots else 0
    print(f"Spots with both grids: {len(filtered):,} ({pct:.1f}%)")
    return filtered

# ============================================================================
# Solar Data
# ============================================================================

def get_solar_for_date(date_str: str) -> tuple:
    """
    Get SFI and Kp for a given date from ClickHouse.
    Returns (sfi, kp) tuple.
    """
    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(host='10.60.1.1', port=8123)

        # Use most recent available date (GFZ lags ~1 day)
        query = """
        SELECT
            max(date) as data_date,
            avg(adjusted_flux) as avg_sfi,
            avg(kp_index) as avg_kp
        FROM solar.bronze
        WHERE date = (SELECT max(date) FROM solar.bronze)
        """
        result = client.query(query)
        if result.result_rows:
            data_date, sfi, kp = result.result_rows[0]
            print(f"Solar data from {data_date}: SFI={sfi}, Kp={kp}")
            if sfi is not None and kp is not None and sfi > 0:
                return float(sfi), float(kp)
            else:
                print(f"Invalid solar data, using defaults")
    except Exception as e:
        print(f"Warning: Could not fetch solar data: {e}")

    # Defaults if query fails
    print("Using default solar values: SFI=150, Kp=3")
    return 150.0, 3.0

# ============================================================================
# Validation
# ============================================================================

def validate_v18(spots: list, model: nn.Module, global_mean: float, global_std: float,
                 sfi: float, kp: float, sample_size: int = None):
    """
    Validate V16 predictions against PSK Reporter spots.

    For each spot:
    1. Compute path features
    2. Run V16 prediction (returns normalized σ units)
    3. Convert to dB: pred_db = pred_sigma * global_std + global_mean
    4. Compare to mode threshold
    5. Calculate recall
    """
    if sample_size and len(spots) > sample_size:
        import random
        random.seed(42)
        spots = random.sample(spots, sample_size)
        print(f"Sampled {sample_size:,} spots for validation")

    model.eval()

    # Group results by mode and band
    mode_results = defaultdict(lambda: {"hits": 0, "total": 0})
    band_results = defaultdict(lambda: {"hits": 0, "total": 0})

    total_hits = 0
    total_spots = 0
    skipped = 0

    print(f"\nValidating with SFI={sfi:.0f}, Kp={kp:.1f}...")
    print(f"SNR normalization: mean={global_mean:.2f}, std={global_std:.2f}")

    for i, spot in enumerate(spots):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} spots...")

        # Parse spot
        sg = spot.get('sg', '')
        rg = spot.get('rg', '')
        band = spot.get('band', 0)
        mode = spot.get('mode', 'FT8').upper()
        ts = spot.get('ts', '')

        # Skip invalid bands
        if band not in BAND_TO_FREQ:
            skipped += 1
            continue

        # Get coordinates
        tx_lat, tx_lon = grid_to_latlon(sg)
        rx_lat, rx_lon = grid_to_latlon(rg)
        if tx_lat is None or rx_lat is None:
            skipped += 1
            continue

        # Parse timestamp for hour/month
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            hour = dt.hour
            month = dt.month
        except:
            hour = 12
            month = 2  # February

        # Compute features
        freq_mhz = BAND_TO_FREQ[band]
        features = compute_features(
            tx_lat, tx_lon, rx_lat, rx_lon,
            freq_mhz, hour, month, sfi, kp
        )

        # Run prediction
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=DEVICE)
            pred_sigma = model(x).item()

        # Convert σ to dB using training normalization
        pred_db = pred_sigma * global_std + global_mean

        # Get mode threshold
        threshold = MODE_THRESHOLDS.get(mode, DEFAULT_THRESHOLD)

        # Propagation "available" if predicted SNR >= threshold
        hit = pred_db >= threshold

        # Track results
        mode_results[mode]["total"] += 1
        band_results[band]["total"] += 1
        total_spots += 1

        if hit:
            mode_results[mode]["hits"] += 1
            band_results[band]["hits"] += 1
            total_hits += 1

    # Calculate recall
    overall_recall = 100 * total_hits / total_spots if total_spots > 0 else 0

    print(f"\n{'='*60}")
    print(f"IONIS V18 vs PSK Reporter — Validation Results")
    print(f"{'='*60}")
    print(f"Total spots validated: {total_spots:,}")
    print(f"Skipped (invalid grid/band): {skipped:,}")
    print(f"Overall recall: {overall_recall:.2f}%")

    print(f"\nRecall by Mode:")
    print(f"{'Mode':<10} {'Recall':>10} {'Spots':>12}")
    print(f"{'-'*34}")
    for mode in sorted(mode_results.keys(), key=lambda m: mode_results[m]["total"], reverse=True):
        r = mode_results[mode]
        recall = 100 * r["hits"] / r["total"] if r["total"] > 0 else 0
        print(f"{mode:<10} {recall:>9.2f}% {r['total']:>11,}")

    print(f"\nRecall by Band:")
    print(f"{'Band':<10} {'Recall':>10} {'Spots':>12}")
    print(f"{'-'*34}")
    for band in sorted(band_results.keys()):
        r = band_results[band]
        recall = 100 * r["hits"] / r["total"] if r["total"] > 0 else 0
        band_name = BAND_NAMES.get(band, str(band))
        print(f"{band_name:<10} {recall:>9.2f}% {r['total']:>11,}")

    return {
        "overall_recall": overall_recall,
        "total_spots": total_spots,
        "mode_results": dict(mode_results),
        "band_results": dict(band_results),
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate IONIS V18 against PSK Reporter")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing spots-*.jsonl.gz files")
    parser.add_argument("--sample", type=int, default=100000,
                        help="Number of spots to sample (default: 100000, 0=all)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to load")
    parser.add_argument("--sfi", type=float, default=None,
                        help="Solar flux index (auto-fetch if not specified)")
    parser.add_argument("--kp", type=float, default=None,
                        help="Kp index (auto-fetch if not specified)")
    args = parser.parse_args()

    # Load model
    print(f"Loading V18 checkpoint from {V18_CHECKPOINT}...")
    if not V18_CHECKPOINT.exists():
        print(f"Error: Checkpoint not found at {V18_CHECKPOINT}")
        print("Tried:")
        print(f"  - {SCRIPT_DIR.parent / 'v16' / 'ionis_v16.pth'}")
        print(f"  - {SCRIPT_DIR.parent.parent / 'models' / 'ionis_v16.pth'}")
        sys.exit(1)

    checkpoint = torch.load(V18_CHECKPOINT, map_location=DEVICE, weights_only=False)

    # Get model config from checkpoint
    dnn_dim = checkpoint.get('dnn_dim', DNN_DIM)
    sidecar_hidden = checkpoint.get('sidecar_hidden', 8)
    global_mean = checkpoint.get('global_mean', 0.0)
    global_std = checkpoint.get('global_std', 1.0)

    # Create and load model
    model = IonisV12Gate(dnn_dim=dnn_dim, sidecar_hidden=sidecar_hidden).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Architecture print removed
    print(f"  SNR normalization: mean={global_mean:.2f}, std={global_std:.2f}")
    print(f"  Device: {DEVICE}")

    # Load spots
    spots = load_pskr_spots(args.data_dir, args.max_files)
    if not spots:
        sys.exit(1)

    # Filter to spots with grids
    spots = filter_spots_with_grids(spots)
    if not spots:
        print("No spots with both grids found")
        sys.exit(1)

    # Get solar data
    if args.sfi is not None and args.kp is not None:
        sfi, kp = args.sfi, args.kp
    else:
        sfi, kp = get_solar_for_date("2026-02-10")

    # Validate
    sample_size = args.sample if args.sample > 0 else None
    results = validate_v18(spots, model, global_mean, global_std, sfi, kp, sample_size)

    print(f"\n{'='*60}")
    print("Validation complete.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
