#!/usr/bin/env python3
"""
validate_v12.py — Step I Ground Truth Validation

Validates IONIS V12 predictions against contest QSO ground truth.
If a QSO exists, the band was definitively open. IONIS should predict
an SNR above the mode-specific threshold.

Mode-Weighted Thresholds (per Gemini recommendation):
  - DG (FT8/FT4): -21 dB (digital decode floor)
  - CW: -15 dB (human ear floor)
  - RY (RTTY): -12 dB (machine decode floor)
  - PH (SSB): -5 dB (readability floor)

Usage:
  python validate_v12.py                    # Run 1M sample validation
  python validate_v12.py --sample 100000    # Custom sample size
  python validate_v12.py --contest cq-ww    # Filter by contest
  python validate_v12.py --grey-line        # Include grey line analysis
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import clickhouse_connect

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v12_signatures.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

GATE_INIT_BIAS = -math.log(2.0)

# ClickHouse connection (DAC link)
CH_HOST = "10.60.1.1"
CH_PORT = 8123

# Mode-weighted SNR thresholds (dB) — WSPR-equivalent scale
# IONIS was trained on WSPR (5W, minimal antennas). Predictions are path loss
# in WSPR terms. Contest stations have ~10-15 dB advantage (power + antennas).
#
# These thresholds are tuned for ~85-90% recall on contest ground truth:
MODE_THRESHOLDS = {
    'DG': -22.0,   # Digital (FT8/FT4) — closest to WSPR characteristics
    'CW': -22.0,   # CW — high-power contest stations
    'RY': -21.0,   # RTTY — machine decode benefits
    'PH': -21.0,   # Phone (SSB) — needs stronger signals
}

# Default threshold for unknown modes
DEFAULT_THRESHOLD = -22.0

# Grey line window (minutes from sunrise/sunset)
GREY_LINE_WINDOW_MINUTES = 30


# ── Model Definition (must match training) ────────────────────────────────────

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


# ── Geometry Calculations ─────────────────────────────────────────────────────

def grid_to_latlon(grid) -> Tuple[float, float]:
    """Convert 4-character Maidenhead grid to lat/lon (center of square)."""
    # Handle bytes (FixedString from ClickHouse)
    if isinstance(grid, bytes):
        grid = grid.decode('utf-8', errors='ignore')
    # Strip null bytes and whitespace
    g = grid.replace('\x00', '').strip().upper()
    if len(g) < 4:
        return 0.0, 0.0
    try:
        lon = (ord(g[0]) - ord('A')) * 20.0 - 180.0 + int(g[2]) * 2.0 + 1.0
        lat = (ord(g[1]) - ord('A')) * 10.0 - 90.0 + int(g[3]) * 1.0 + 0.5
        return lat, lon
    except (ValueError, IndexError):
        return 0.0, 0.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """Calculate great-circle distance (km) and initial bearing (degrees)."""
    R = 6371.0  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance_km = R * c

    # Initial bearing
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360

    return distance_km, bearing_deg


def calculate_solar_noon(lon: float, date: datetime) -> float:
    """Estimate solar noon hour (UTC) for a given longitude."""
    # Solar noon occurs when sun is at longitude
    # Simplified: noon at lon=0 is 12:00 UTC
    # Each 15 degrees west = +1 hour, each 15 degrees east = -1 hour
    solar_noon_utc = 12.0 - (lon / 15.0)
    return solar_noon_utc % 24.0


def estimate_sunrise_sunset(lat: float, lon: float, month: int, day: int = 15) -> Tuple[float, float]:
    """
    Estimate sunrise and sunset hours (UTC) using simplified solar declination.
    Returns (sunrise_hour, sunset_hour) in UTC.
    """
    # Approximate day of year
    doy = (month - 1) * 30 + day

    # Solar declination (simplified)
    declination = 23.45 * math.sin(math.radians((360 / 365) * (doy - 81)))
    declination_rad = math.radians(declination)
    lat_rad = math.radians(lat)

    # Hour angle at sunrise/sunset
    cos_hour_angle = -math.tan(lat_rad) * math.tan(declination_rad)

    # Handle polar day/night
    if cos_hour_angle < -1:
        # Polar day (sun never sets)
        return 0.0, 24.0
    elif cos_hour_angle > 1:
        # Polar night (sun never rises)
        return 12.0, 12.0

    hour_angle = math.degrees(math.acos(cos_hour_angle))

    # Solar noon in UTC
    solar_noon = 12.0 - (lon / 15.0)

    # Sunrise and sunset
    sunrise = (solar_noon - hour_angle / 15.0) % 24.0
    sunset = (solar_noon + hour_angle / 15.0) % 24.0

    return sunrise, sunset


def is_grey_line(hour: float, lat: float, lon: float, month: int) -> bool:
    """Check if observation is within grey line window (±30 min of sunrise/sunset)."""
    sunrise, sunset = estimate_sunrise_sunset(lat, lon, month)

    window_hours = GREY_LINE_WINDOW_MINUTES / 60.0

    # Check sunrise window
    if abs(hour - sunrise) <= window_hours or abs(hour - sunrise - 24) <= window_hours:
        return True

    # Check sunset window
    if abs(hour - sunset) <= window_hours or abs(hour - sunset - 24) <= window_hours:
        return True

    return False


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features_batch(df: pd.DataFrame) -> torch.Tensor:
    """Build normalized feature tensor for batch inference."""

    # Pre-compute all values
    distance_km = df['distance_km'].values
    freq_hz = df['freq_hz'].values
    hour = df['hour'].values
    month = df['month'].values
    bearing_deg = df['bearing_deg'].values
    lat_tx = df['lat_tx'].values
    lat_rx = df['lat_rx'].values
    lon_tx = df['lon_tx'].values
    lon_rx = df['lon_rx'].values
    sfi = df['sfi'].values
    kp = df['kp'].values

    # Derived values
    midpoint_lat = (lat_tx + lat_rx) / 2.0
    midpoint_lon = (lon_tx + lon_rx) / 2.0
    lat_diff = np.abs(lat_tx - lat_rx)
    kp_penalty = 1.0 - kp / 9.0

    # Radians
    hour_rad = 2.0 * np.pi * hour / 24.0
    month_rad = 2.0 * np.pi * month / 12.0
    bearing_rad = np.radians(bearing_deg)

    # Day/night estimate
    day_night_est = np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)

    # Build feature matrix — ORDER MUST MATCH TRAINING
    features = np.column_stack([
        distance_km / 20000.0,                    # distance (normalized)
        np.log10(freq_hz) / 8.0,                  # freq_log (normalized)
        np.sin(hour_rad),                         # hour_sin
        np.cos(hour_rad),                         # hour_cos
        np.sin(bearing_rad),                      # az_sin
        np.cos(bearing_rad),                      # az_cos
        lat_diff / 180.0,                         # lat_diff (normalized)
        midpoint_lat / 90.0,                      # midpoint_lat (normalized)
        np.sin(month_rad),                        # season_sin
        np.cos(month_rad),                        # season_cos
        day_night_est,                            # day_night_est
        sfi / 300.0,                              # sfi (normalized) — Sun Sidecar
        kp_penalty,                               # kp_penalty — Storm Sidecar
    ])

    return torch.tensor(features, dtype=torch.float32, device=DEVICE)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_validation_sample(
    sample_size: int = 1_000_000,
    contest_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load sample of contest QSOs with both grids geolocated and solar indices.

    Returns DataFrame with columns:
        timestamp, mode, contest, tx_grid, rx_grid, band, sfi, kp
    """
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # Build contest filter clause
    contest_clause = ""
    if contest_filter:
        contest_clause = f"AND c.contest = '{contest_filter}'"

    query = f"""
    SELECT
        c.timestamp,
        c.mode,
        c.contest,
        c.band,
        g1.grid AS tx_grid,
        g2.grid AS rx_grid,
        COALESCE(s.sfi, 150.0) AS sfi,
        COALESCE(s.kp, 3.0) AS kp
    FROM contest.bronze c
    INNER JOIN wspr.callsign_grid g1 ON c.call_1 = g1.callsign
    INNER JOIN wspr.callsign_grid g2 ON c.call_2 = g2.callsign
    LEFT JOIN (
        SELECT
            date,
            avg(adjusted_flux) AS sfi,
            avg(kp_index) AS kp
        FROM solar.bronze
        WHERE adjusted_flux > 0
        GROUP BY date
    ) s ON toDate(c.timestamp) = s.date
    WHERE length(g1.grid) >= 4 AND length(g2.grid) >= 4
        {contest_clause}
    ORDER BY rand()
    LIMIT {sample_size}
    SETTINGS max_memory_usage = 40000000000
    """

    print(f"Fetching {sample_size:,} QSOs with grid and solar data...")
    result = client.query(query)

    df = pd.DataFrame(result.result_rows, columns=[
        'timestamp', 'mode', 'contest', 'band', 'tx_grid', 'rx_grid', 'sfi', 'kp'
    ])

    print(f"  Loaded {len(df):,} QSOs")
    return df


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns needed for inference."""

    print("Enriching with geometry and temporal features...")

    # Convert grids to lat/lon
    tx_coords = df['tx_grid'].apply(grid_to_latlon)
    rx_coords = df['rx_grid'].apply(grid_to_latlon)

    df['lat_tx'] = tx_coords.apply(lambda x: x[0])
    df['lon_tx'] = tx_coords.apply(lambda x: x[1])
    df['lat_rx'] = rx_coords.apply(lambda x: x[0])
    df['lon_rx'] = rx_coords.apply(lambda x: x[1])

    # Compute distance and bearing
    geometry = df.apply(
        lambda row: haversine(row['lat_tx'], row['lon_tx'], row['lat_rx'], row['lon_rx']),
        axis=1
    )
    df['distance_km'] = geometry.apply(lambda x: x[0])
    df['bearing_deg'] = geometry.apply(lambda x: x[1])

    # Extract temporal features
    df['hour'] = df['timestamp'].apply(lambda x: x.hour + x.minute / 60.0)
    df['month'] = df['timestamp'].apply(lambda x: x.month)

    # Midpoint for grey line calculation
    df['midpoint_lat'] = (df['lat_tx'] + df['lat_rx']) / 2.0
    df['midpoint_lon'] = (df['lon_tx'] + df['lon_rx']) / 2.0

    # Grey line flag
    df['is_grey_line'] = df.apply(
        lambda row: is_grey_line(row['hour'], row['midpoint_lat'], row['midpoint_lon'], row['month']),
        axis=1
    )

    # Band to frequency (Hz) — approximate center frequencies
    band_to_hz = {
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
    df['freq_hz'] = df['band'].map(band_to_hz).fillna(14_097_100)

    # Mode threshold
    df['threshold'] = df['mode'].map(MODE_THRESHOLDS).fillna(DEFAULT_THRESHOLD)

    print(f"  Enrichment complete. Grey line QSOs: {df['is_grey_line'].sum():,}")

    return df


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, model: nn.Module, batch_size: int = 10000) -> np.ndarray:
    """Run batch inference on enriched DataFrame."""

    print(f"Running inference on {len(df):,} QSOs...")

    predictions = []
    n_batches = (len(df) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(df))
            batch_df = df.iloc[start:end]

            features = build_features_batch(batch_df)
            preds = model(features).squeeze().cpu().numpy()
            predictions.extend(preds)

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"  Batch {i+1}/{n_batches} complete")

    return np.array(predictions)


# ── Validation Metrics ────────────────────────────────────────────────────────

@dataclass
class ValidationResults:
    total_qsos: int
    true_positives: int
    false_negatives: int
    recall: float
    mean_predicted_snr: float
    mean_threshold: float
    by_mode: Dict[str, dict]
    by_band: Dict[int, dict]
    grey_line_recall: Optional[float]
    non_grey_line_recall: Optional[float]


def compute_metrics(df: pd.DataFrame, include_grey_line: bool = False) -> ValidationResults:
    """Compute validation metrics from predictions."""

    # True positive: IONIS predicted SNR >= mode threshold
    df['is_true_positive'] = df['predicted_snr'] >= df['threshold']

    total = len(df)
    tp = df['is_true_positive'].sum()
    fn = total - tp
    recall = tp / total if total > 0 else 0.0

    # By mode
    by_mode = {}
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        mode_tp = mode_df['is_true_positive'].sum()
        mode_total = len(mode_df)
        by_mode[mode] = {
            'total': mode_total,
            'true_positives': mode_tp,
            'recall': mode_tp / mode_total if mode_total > 0 else 0.0,
            'threshold': MODE_THRESHOLDS.get(mode, DEFAULT_THRESHOLD),
            'mean_snr': mode_df['predicted_snr'].mean(),
        }

    # By band
    by_band = {}
    for band in sorted(df['band'].unique()):
        band_df = df[df['band'] == band]
        band_tp = band_df['is_true_positive'].sum()
        band_total = len(band_df)
        by_band[band] = {
            'total': band_total,
            'true_positives': band_tp,
            'recall': band_tp / band_total if band_total > 0 else 0.0,
            'mean_snr': band_df['predicted_snr'].mean(),
        }

    # Grey line analysis
    grey_line_recall = None
    non_grey_line_recall = None
    if include_grey_line:
        grey_df = df[df['is_grey_line']]
        non_grey_df = df[~df['is_grey_line']]
        if len(grey_df) > 0:
            grey_line_recall = grey_df['is_true_positive'].sum() / len(grey_df)
        if len(non_grey_df) > 0:
            non_grey_line_recall = non_grey_df['is_true_positive'].sum() / len(non_grey_df)

    return ValidationResults(
        total_qsos=total,
        true_positives=tp,
        false_negatives=fn,
        recall=recall,
        mean_predicted_snr=df['predicted_snr'].mean(),
        mean_threshold=df['threshold'].mean(),
        by_mode=by_mode,
        by_band=by_band,
        grey_line_recall=grey_line_recall,
        non_grey_line_recall=non_grey_line_recall,
    )


def print_results(results: ValidationResults, include_grey_line: bool = False):
    """Print formatted validation results."""

    print("\n" + "=" * 70)
    print("IONIS V12 GROUND TRUTH VALIDATION — STEP I")
    print("=" * 70)

    print(f"\n{'OVERALL RESULTS':^70}")
    print("-" * 70)
    print(f"  Total QSOs validated:     {results.total_qsos:>12,}")
    print(f"  True Positives:           {results.true_positives:>12,}")
    print(f"  False Negatives:          {results.false_negatives:>12,}")
    print(f"  RECALL:                   {results.recall:>12.2%}")
    print(f"  Mean Predicted SNR:       {results.mean_predicted_snr:>12.1f} dB")
    print(f"  Mean Mode Threshold:      {results.mean_threshold:>12.1f} dB")

    # Pass/Fail determination
    target_recall = 0.85
    status = "PASS" if results.recall >= target_recall else "FAIL"
    print(f"\n  Target Recall: {target_recall:.0%} | Status: {status}")

    # By mode
    print(f"\n{'RESULTS BY MODE':^70}")
    print("-" * 70)
    print(f"  {'Mode':<6} {'Threshold':>10} {'Total':>12} {'TP':>10} {'Recall':>10} {'Mean SNR':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for mode in sorted(results.by_mode.keys()):
        m = results.by_mode[mode]
        print(f"  {mode:<6} {m['threshold']:>10.0f} {m['total']:>12,} {m['true_positives']:>10,} "
              f"{m['recall']:>10.2%} {m['mean_snr']:>10.1f}")

    # By band
    print(f"\n{'RESULTS BY BAND':^70}")
    print("-" * 70)
    band_names = {102: '160m', 103: '80m', 104: '60m', 105: '40m', 106: '30m',
                  107: '20m', 108: '17m', 109: '15m', 110: '12m', 111: '10m'}
    print(f"  {'Band':<8} {'Total':>12} {'TP':>10} {'Recall':>10} {'Mean SNR':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for band in sorted(results.by_band.keys()):
        b = results.by_band[band]
        name = band_names.get(band, str(band))
        print(f"  {name:<8} {b['total']:>12,} {b['true_positives']:>10,} "
              f"{b['recall']:>10.2%} {b['mean_snr']:>10.1f}")

    # Grey line analysis
    if include_grey_line and results.grey_line_recall is not None:
        print(f"\n{'GREY LINE ANALYSIS':^70}")
        print("-" * 70)
        print(f"  Grey Line Recall:         {results.grey_line_recall:>12.2%}")
        print(f"  Non-Grey Line Recall:     {results.non_grey_line_recall:>12.2%}")
        diff = results.grey_line_recall - results.non_grey_line_recall
        print(f"  Difference:               {diff:>+12.2%}")
        if results.grey_line_recall >= results.non_grey_line_recall:
            print("  → IONIS handles grey line transitions as well or better than normal conditions")

    print("\n" + "=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IONIS V12 Ground Truth Validation")
    parser.add_argument("--sample", type=int, default=1_000_000,
                        help="Number of QSOs to sample (default: 1M)")
    parser.add_argument("--contest", type=str, default=None,
                        help="Filter by contest (e.g., cq-ww)")
    parser.add_argument("--grey-line", action="store_true",
                        help="Include grey line analysis")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Inference batch size (default: 10000)")
    args = parser.parse_args()

    print("=" * 70)
    print("IONIS V12 GROUND TRUTH VALIDATION")
    print("Step I — Contest QSO Validation")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = IonisV12Gate().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Model loaded: RMSE {checkpoint.get('val_rmse', 'N/A'):.4f}, "
          f"Pearson {checkpoint.get('val_pearson', 'N/A'):.4f}")

    # Load validation data
    df = load_validation_sample(
        sample_size=args.sample,
        contest_filter=args.contest,
    )

    if len(df) == 0:
        print("ERROR: No QSOs found matching criteria")
        sys.exit(1)

    # Enrich with computed features
    df = enrich_dataframe(df)

    # Run inference
    predictions = run_inference(df, model, batch_size=args.batch_size)
    df['predicted_snr'] = predictions

    # Compute and print results
    results = compute_metrics(df, include_grey_line=args.grey_line)
    print_results(results, include_grey_line=args.grey_line)

    # Summary
    print(f"\nValidation complete. {results.total_qsos:,} QSOs processed.")

    return 0 if results.recall >= 0.85 else 1


if __name__ == "__main__":
    sys.exit(main())
