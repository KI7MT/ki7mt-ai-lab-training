#!/usr/bin/env python3
"""
oracle_v12.py — IONIS V12 Propagation Oracle

Production inference interface for IONIS V12 model.
Predicts HF SNR for arbitrary paths with input validation.

Features:
  - Input validation (frequency, distance, sanity checks)
  - Haversine distance/bearing calculation
  - Feature engineering matching training exactly
  - Confidence assessment based on training domain
  - Interactive and batch modes

Usage:
  python oracle_v12.py                    # Interactive mode
  python oracle_v12.py --test             # Run test suite
  python oracle_v12.py --batch paths.csv  # Batch prediction
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np

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

# Band center frequencies (Hz) — matches training
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

# Reverse lookup: MHz to band ID
MHZ_TO_BAND = {
    1.8: 102, 3.5: 103, 5.3: 104, 7.0: 105, 10.1: 106,
    14.0: 107, 18.1: 108, 21.0: 109, 24.9: 110, 28.0: 111,
}


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


# ── Input Validation ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]


def validate_input(
    freq_mhz: float,
    distance_km: float,
    lat_tx: float,
    lon_tx: float,
    lat_rx: float,
    lon_rx: float,
    sfi: float,
    kp: float,
    hour: float,
    month: float,
) -> ValidationResult:
    """Validate inputs against training domain."""
    errors = []
    warnings = []

    # Frequency — HF only (1.8-30 MHz)
    if freq_mhz <= 0:
        errors.append(f"Frequency {freq_mhz:.2f} MHz must be positive")
    elif freq_mhz < 1.8:
        errors.append(f"Frequency {freq_mhz:.2f} MHz below HF band (min 1.8 MHz)")
    elif freq_mhz > 30:
        errors.append(f"Frequency {freq_mhz:.2f} MHz above HF band (max 30 MHz)")

    # VHF/UHF sanity check — EME trap
    if freq_mhz > 50:
        errors.append(f"Frequency {freq_mhz:.2f} MHz is VHF/UHF — IONIS trained on HF only")

    # Distance sanity
    if distance_km < 100:
        warnings.append(f"Distance {distance_km:.0f} km likely ground wave, not ionospheric")
    if distance_km > 20000:
        errors.append(f"Distance {distance_km:.0f} km exceeds max Earth path (~20,000 km)")

    # Latitude bounds
    for name, lat in [("TX", lat_tx), ("RX", lat_rx)]:
        if lat < -90 or lat > 90:
            errors.append(f"{name} latitude {lat:.2f} out of bounds [-90, 90]")

    # Longitude bounds
    for name, lon in [("TX", lon_tx), ("RX", lon_rx)]:
        if lon < -180 or lon > 180:
            errors.append(f"{name} longitude {lon:.2f} out of bounds [-180, 180]")

    # SFI range (training saw ~50-300)
    if sfi < 0:
        errors.append(f"SFI {sfi:.0f} cannot be negative")
    elif sfi < 50:
        warnings.append(f"SFI {sfi:.0f} below typical range (training: 50-300)")
    if sfi > 350:
        warnings.append(f"SFI {sfi:.0f} above typical range (training: 50-300)")

    # Kp range
    if kp < 0 or kp > 9:
        errors.append(f"Kp {kp:.1f} out of bounds [0, 9]")

    # Hour range
    if hour < 0 or hour >= 24:
        errors.append(f"Hour {hour:.1f} out of bounds [0, 24)")

    # Month range
    if month < 1 or month > 12:
        errors.append(f"Month {month:.0f} out of bounds [1, 12]")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ── Geometry Calculations ─────────────────────────────────────────────────────

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


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(
    lat_tx: float,
    lon_tx: float,
    lat_rx: float,
    lon_rx: float,
    freq_mhz: float,
    sfi: float,
    kp: float,
    hour: float,
    month: float,
) -> torch.Tensor:
    """Build normalized feature vector matching training exactly."""

    # Distance and bearing
    distance_km, bearing_deg = haversine(lat_tx, lon_tx, lat_rx, lon_rx)

    # Frequency in Hz
    freq_hz = freq_mhz * 1e6

    # Derived values
    midpoint_lat = (lat_tx + lat_rx) / 2.0
    midpoint_lon = (lon_tx + lon_rx) / 2.0
    lat_diff = abs(lat_tx - lat_rx)

    # kp_penalty: 1 - kp/9 (high Kp = low penalty value = more storm effect)
    kp_penalty = 1.0 - kp / 9.0

    # Hour/season radians
    hour_rad = 2.0 * math.pi * hour / 24.0
    month_rad = 2.0 * math.pi * month / 12.0
    bearing_rad = math.radians(bearing_deg)

    # Day/night estimate (crude solar terminator proxy)
    day_night_est = math.cos(2.0 * math.pi * (hour + midpoint_lon / 15.0) / 24.0)

    # Build feature vector — ORDER MUST MATCH TRAINING
    features = [
        distance_km / 20000.0,                    # distance (normalized)
        math.log10(freq_hz) / 8.0,                # freq_log (normalized)
        math.sin(hour_rad),                       # hour_sin
        math.cos(hour_rad),                       # hour_cos
        math.sin(bearing_rad),                    # az_sin
        math.cos(bearing_rad),                    # az_cos
        lat_diff / 180.0,                         # lat_diff (normalized)
        midpoint_lat / 90.0,                      # midpoint_lat (normalized)
        math.sin(month_rad),                      # season_sin
        math.cos(month_rad),                      # season_cos
        day_night_est,                            # day_night_est
        sfi / 300.0,                              # sfi (normalized) — Sun Sidecar
        kp_penalty,                               # kp_penalty — Storm Sidecar
    ]

    return torch.tensor([features], dtype=torch.float32, device=DEVICE)


# ── Oracle Class ──────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    snr_db: float
    distance_km: float
    bearing_deg: float
    confidence: str
    condition: str
    warnings: List[str]


class IonisOracle:
    """IONIS V12 Propagation Oracle — production inference interface."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = IonisV12Gate().to(DEVICE)
        checkpoint = torch.load(model_path, weights_only=False, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.metadata = {
            'rmse': checkpoint.get('val_rmse', 'N/A'),
            'pearson': checkpoint.get('val_pearson', 'N/A'),
            'architecture': checkpoint.get('architecture', 'IonisV12Gate'),
            'sample_size': checkpoint.get('sample_size', 'N/A'),
        }

    def predict(
        self,
        lat_tx: float,
        lon_tx: float,
        lat_rx: float,
        lon_rx: float,
        freq_mhz: float,
        sfi: float,
        kp: float,
        hour: float,
        month: float = 6,  # Default June
    ) -> Prediction:
        """Predict SNR for a given path and conditions."""

        # Calculate distance first for validation
        distance_km, bearing_deg = haversine(lat_tx, lon_tx, lat_rx, lon_rx)

        # Validate inputs
        validation = validate_input(
            freq_mhz, distance_km, lat_tx, lon_tx, lat_rx, lon_rx,
            sfi, kp, hour, month
        )

        if not validation.valid:
            raise ValueError(f"Invalid input: {'; '.join(validation.errors)}")

        # Build features and predict
        features = build_features(
            lat_tx, lon_tx, lat_rx, lon_rx, freq_mhz, sfi, kp, hour, month
        )

        with torch.no_grad():
            snr_db = self.model(features).item()

        # Assess condition
        if snr_db > -10:
            condition = "EXCELLENT (Voice/SSB)"
        elif snr_db > -15:
            condition = "GOOD (CW/Digital)"
        elif snr_db > -20:
            condition = "FAIR (FT8/FT4)"
        elif snr_db > -28:
            condition = "MARGINAL (WSPR)"
        else:
            condition = "CLOSED"

        # Confidence based on how close to training domain
        if distance_km < 500 or distance_km > 15000:
            confidence = "LOW (distance edge case)"
        elif kp > 7:
            confidence = "MEDIUM (extreme storm)"
        elif sfi > 250 or sfi < 70:
            confidence = "MEDIUM (unusual SFI)"
        else:
            confidence = "HIGH"

        return Prediction(
            snr_db=snr_db,
            distance_km=distance_km,
            bearing_deg=bearing_deg,
            confidence=confidence,
            condition=condition,
            warnings=validation.warnings,
        )


# ── Test Suite ────────────────────────────────────────────────────────────────

# Canonical test paths
TEST_PATHS = [
    # US East Coast to Western Europe (classic 20m path)
    {"name": "W3 → G (20m day)", "tx": (39.14, -77.01), "rx": (51.50, -0.12),
     "freq": 14.0, "sfi": 150, "kp": 2, "hour": 14, "expect_open": True},

    {"name": "W3 → G (20m night)", "tx": (39.14, -77.01), "rx": (51.50, -0.12),
     "freq": 14.0, "sfi": 150, "kp": 2, "hour": 4, "expect_open": True},  # Grey line possible

    # US to Japan (long path)
    {"name": "W6 → JA (20m)", "tx": (34.05, -118.24), "rx": (35.68, 139.69),
     "freq": 14.0, "sfi": 150, "kp": 2, "hour": 16, "expect_open": True},

    # High latitude path (storm sensitive)
    {"name": "OX → OH (polar)", "tx": (64.18, -51.72), "rx": (60.17, 24.94),
     "freq": 14.0, "sfi": 150, "kp": 2, "hour": 12, "expect_open": True},

    {"name": "OX → OH (polar storm)", "tx": (64.18, -51.72), "rx": (60.17, 24.94),
     "freq": 14.0, "sfi": 150, "kp": 8, "hour": 12, "expect_open": True},  # Degraded but not closed

    # Equatorial path (less storm sensitive)
    {"name": "PY → VU (equatorial)", "tx": (-23.55, -46.63), "rx": (12.97, 77.59),
     "freq": 14.0, "sfi": 150, "kp": 2, "hour": 14, "expect_open": True},

    # Short path NVIS
    {"name": "NVIS 80m", "tx": (40.0, -100.0), "rx": (42.0, -98.0),
     "freq": 3.5, "sfi": 100, "kp": 2, "hour": 2, "expect_open": True},

    # 10m opening (high SFI helps but not required for WSPR)
    {"name": "W → EU 10m low SFI", "tx": (39.14, -77.01), "rx": (51.50, -0.12),
     "freq": 28.0, "sfi": 80, "kp": 2, "hour": 14, "expect_open": True},  # Marginal but open

    {"name": "W → EU 10m high SFI", "tx": (39.14, -77.01), "rx": (51.50, -0.12),
     "freq": 28.0, "sfi": 200, "kp": 2, "hour": 14, "expect_open": True},

    # TST-110: Grey line / twilight enhancement
    {"name": "Grey line 18UTC vs 14UTC", "tx": (39.14, -77.01), "rx": (51.50, -0.12),
     "freq": 14.0, "sfi": 150, "kp": 2, "hour": 18, "expect_open": True},
]

# ── Physics Scoring Functions ─────────────────────────────────────────────────

def score_sfi_monotonicity(delta_db: float) -> tuple:
    """Score SFI monotonicity test. Expected: +1 to +4 dB for SFI 70→200."""
    if delta_db >= 3.0:
        return 100, "A"
    elif delta_db >= 2.0:
        return 85, "B"
    elif delta_db >= 1.0:
        return 70, "C"
    elif delta_db > 0:
        return 50, "D"
    else:
        return 0, "F"


def score_kp_storm_cost(cost_db: float) -> tuple:
    """Score Kp storm cost test. Expected: +2 to +6 dB for Kp 0→9."""
    if cost_db >= 4.0:
        return 100, "A"
    elif cost_db >= 3.0:
        return 90, "A"
    elif cost_db >= 2.0:
        return 75, "B"
    elif cost_db >= 1.0:
        return 60, "C"
    elif cost_db > 0:
        return 40, "D"
    else:
        return 0, "F"


def score_dlayer_absorption(delta_db: float) -> tuple:
    """Score D-layer absorption test. Expected: 20m better than 80m at noon."""
    if delta_db >= 3.0:
        return 100, "A"
    elif delta_db >= 1.0:
        return 80, "B"
    elif delta_db >= 0:
        return 60, "C"
    elif delta_db >= -1.0:
        return 40, "D"
    else:
        return 0, "F"


def score_greyline(delta_db: float) -> tuple:
    """Score grey line / twilight test. Expected: 18 UTC >= 14 UTC on E-W path."""
    if delta_db >= 1.0:
        return 100, "A"
    elif delta_db >= 0.5:
        return 85, "B"
    elif delta_db >= 0:
        return 70, "C"
    elif delta_db >= -0.5:
        return 50, "D"
    else:
        return 0, "F"


def grade_to_label(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


# Physics tests
PHYSICS_TESTS = [
    # SFI monotonicity
    {"name": "SFI 70 vs 200", "test": "sfi_monotonic",
     "base": {"tx": (39.14, -77.01), "rx": (51.50, -0.12), "freq": 14.0, "kp": 2, "hour": 14},
     "sfi_low": 70, "sfi_high": 200, "expect_delta_positive": True},

    # Kp monotonicity
    {"name": "Kp 0 vs 9", "test": "kp_monotonic",
     "base": {"tx": (39.14, -77.01), "rx": (51.50, -0.12), "freq": 14.0, "sfi": 150, "hour": 14},
     "kp_low": 0, "kp_high": 9, "expect_delta_negative": True},

    # D-layer absorption (80m worse than 20m at noon)
    {"name": "D-layer 80m vs 20m noon", "test": "freq_comparison",
     "base": {"tx": (39.14, -77.01), "rx": (51.50, -0.12), "sfi": 150, "kp": 2, "hour": 12},
     "freq_low": 3.5, "freq_high": 14.0, "expect_high_better": True},

    # Polar path storm degradation (Kp 2 vs Kp 8 should show >2 dB drop)
    {"name": "Polar storm degradation", "test": "kp_monotonic",
     "base": {"tx": (64.18, -51.72), "rx": (60.17, 24.94), "freq": 14.0, "sfi": 150, "hour": 12},
     "kp_low": 2, "kp_high": 8, "expect_delta_negative": True},

    # 10m SFI sensitivity (should show improvement with higher SFI)
    {"name": "10m SFI sensitivity", "test": "sfi_monotonic",
     "base": {"tx": (39.14, -77.01), "rx": (51.50, -0.12), "freq": 28.0, "kp": 2, "hour": 14},
     "sfi_low": 80, "sfi_high": 200, "expect_delta_positive": True},

    # TST-110: Grey line / twilight spike (18 UTC should be >= 14 UTC on E-W path)
    {"name": "Grey line twilight", "test": "hour_comparison",
     "base": {"tx": (39.14, -77.01), "rx": (51.50, -0.12), "freq": 14.0, "sfi": 150, "kp": 2},
     "hour_early": 14, "hour_late": 18, "expect_late_better": True},
]

# Invalid input tests
INVALID_TESTS = [
    {"name": "VHF frequency (EME trap)", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 144.0, "sfi": 150, "kp": 2, "hour": 14},
     "should_fail": True},

    {"name": "UHF frequency", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 432.0, "sfi": 150, "kp": 2, "hour": 14},
     "should_fail": True},

    {"name": "Invalid latitude", "params": {
        "lat_tx": 95.0, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 14.0, "sfi": 150, "kp": 2, "hour": 14},
     "should_fail": True},

    {"name": "Invalid Kp", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 14.0, "sfi": 150, "kp": 15, "hour": 14},
     "should_fail": True},

    {"name": "Excessive distance", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 39.14, "lon_rx": 103.0,
        "freq_mhz": 14.0, "sfi": 150, "kp": 2, "hour": 14},
     "should_fail": False},  # ~12000 km, valid

    # TST-602: Extremely large values
    {"name": "Extreme SFI (1e10)", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 14.0, "sfi": 1e10, "kp": 2, "hour": 14},
     "should_fail": False},  # Should warn but not crash

    # TST-603: Negative physical values
    {"name": "Negative SFI", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 14.0, "sfi": -100, "kp": 2, "hour": 14},
     "should_fail": True},

    {"name": "Negative frequency", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": -14.0, "sfi": 150, "kp": 2, "hour": 14},
     "should_fail": True},

    {"name": "Negative Kp", "params": {
        "lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
        "freq_mhz": 14.0, "sfi": 150, "kp": -2, "hour": 14},
     "should_fail": True},
]

# Robustness tests (TST-500 series)
ROBUSTNESS_TESTS = [
    # TST-501: Reproducibility — same input, same output
    {"name": "Reproducibility (100x)", "test": "reproducibility",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 150, "kp": 2, "hour": 14},
     "iterations": 100},

    # TST-502: Input perturbation stability
    {"name": "Perturbation stability", "test": "perturbation",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 150, "kp": 2, "hour": 14},
     "perturbation": 0.001, "max_delta_db": 0.5},

    # TST-503: Boundary values
    {"name": "Boundary: SFI=50 (min)", "test": "boundary",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 50, "kp": 2, "hour": 14}},

    {"name": "Boundary: SFI=300 (max)", "test": "boundary",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 300, "kp": 2, "hour": 14}},

    {"name": "Boundary: Kp=0 (quiet)", "test": "boundary",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 150, "kp": 0, "hour": 14}},

    {"name": "Boundary: Kp=9 (storm)", "test": "boundary",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 150, "kp": 9, "hour": 14}},

    # TST-506: Checkpoint integrity (updated 2026-02-08 Platinum Burn)
    {"name": "Checkpoint RMSE", "test": "checkpoint_rmse",
     "expected": 2.0336, "tolerance": 0.01},

    {"name": "Checkpoint Pearson", "test": "checkpoint_pearson",
     "expected": 0.3153, "tolerance": 0.01},

    # TST-701: Geographic bias — compare dense vs sparse regions
    {"name": "Geo bias: EU vs Africa", "test": "geo_bias"},
]

# Regression tests (TST-800 series)
REGRESSION_TESTS = [
    # TST-801: Reference prediction
    {"name": "Reference: W3→G 20m", "test": "reference",
     "params": {"lat_tx": 39.14, "lon_tx": -77.01, "lat_rx": 51.50, "lon_rx": -0.12,
                "freq_mhz": 14.0, "sfi": 150, "kp": 2, "hour": 14},
     "expected_snr": -20.0, "tolerance": 0.5},
]


def run_test_suite():
    """Run comprehensive test suite."""
    print("=" * 70)
    print("  IONIS V12 Oracle Test Suite")
    print("=" * 70)

    try:
        oracle = IonisOracle()
        print(f"Model loaded: {oracle.metadata['architecture']}")
        print(f"RMSE: {oracle.metadata['rmse']:.4f} dB, Pearson: {oracle.metadata['pearson']:+.4f}")
    except Exception as e:
        print(f"CRITICAL: Could not load model — {e}")
        sys.exit(1)

    passed = 0
    failed = 0

    # ── Path Tests ──
    print("\n" + "=" * 70)
    print("  TEST GROUP 1: Canonical Paths")
    print("=" * 70)

    for test in TEST_PATHS:
        try:
            pred = oracle.predict(
                lat_tx=test["tx"][0], lon_tx=test["tx"][1],
                lat_rx=test["rx"][0], lon_rx=test["rx"][1],
                freq_mhz=test["freq"], sfi=test["sfi"], kp=test["kp"], hour=test["hour"],
            )
            is_open = pred.snr_db > -25  # WSPR threshold
            expected = test["expect_open"]
            status = "PASS" if is_open == expected else "FAIL"

            if status == "PASS":
                passed += 1
            else:
                failed += 1

            print(f"  {test['name']:<30} SNR: {pred.snr_db:+6.1f} dB  "
                  f"Open: {is_open}  Expected: {expected}  [{status}]")
        except Exception as e:
            print(f"  {test['name']:<30} ERROR: {e}")
            failed += 1

    # ── Physics Tests (with scoring) ──
    print("\n" + "=" * 70)
    print("  TEST GROUP 2: Physics Constraints (Scored)")
    print("=" * 70)

    physics_scores = []

    for test in PHYSICS_TESTS:
        try:
            base = test["base"]

            if test["test"] == "sfi_monotonic":
                snr_low = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=base["freq"], sfi=test["sfi_low"], kp=base["kp"], hour=base["hour"],
                ).snr_db
                snr_high = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=base["freq"], sfi=test["sfi_high"], kp=base["kp"], hour=base["hour"],
                ).snr_db
                delta = snr_high - snr_low
                score, grade = score_sfi_monotonicity(delta)
                physics_scores.append(score)
                status = "PASS" if delta > 0 else "FAIL"
                print(f"  {test['name']:<30} Delta: {delta:+.1f} dB  "
                      f"Score: {score:3d}/100 ({grade})  [{status}]")

            elif test["test"] == "kp_monotonic":
                snr_low = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=base["freq"], sfi=base["sfi"], kp=test["kp_low"], hour=base["hour"],
                ).snr_db
                snr_high = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=base["freq"], sfi=base["sfi"], kp=test["kp_high"], hour=base["hour"],
                ).snr_db
                cost = snr_low - snr_high  # Cost is positive when storm degrades signal
                score, grade = score_kp_storm_cost(cost)
                physics_scores.append(score)
                status = "PASS" if cost > 0 else "FAIL"
                print(f"  {test['name']:<30} Cost:  {cost:+.1f} dB  "
                      f"Score: {score:3d}/100 ({grade})  [{status}]")

            elif test["test"] == "freq_comparison":
                snr_low = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=test["freq_low"], sfi=base["sfi"], kp=base["kp"], hour=base["hour"],
                ).snr_db
                snr_high = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=test["freq_high"], sfi=base["sfi"], kp=base["kp"], hour=base["hour"],
                ).snr_db
                delta = snr_high - snr_low
                score, grade = score_dlayer_absorption(delta)
                physics_scores.append(score)
                status = "PASS" if delta >= 0 else "FAIL"
                print(f"  {test['name']:<30} Delta: {delta:+.1f} dB  "
                      f"Score: {score:3d}/100 ({grade})  [{status}]")

            elif test["test"] == "hour_comparison":
                # TST-110: Grey line / twilight test
                snr_early = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=base["freq"], sfi=base["sfi"], kp=base["kp"], hour=test["hour_early"],
                ).snr_db
                snr_late = oracle.predict(
                    lat_tx=base["tx"][0], lon_tx=base["tx"][1],
                    lat_rx=base["rx"][0], lon_rx=base["rx"][1],
                    freq_mhz=base["freq"], sfi=base["sfi"], kp=base["kp"], hour=test["hour_late"],
                ).snr_db
                delta = snr_late - snr_early
                score, grade = score_greyline(delta)
                physics_scores.append(score)
                status = "PASS" if delta >= 0 else "FAIL"
                print(f"  {test['name']:<30} Delta: {delta:+.1f} dB  "
                      f"Score: {score:3d}/100 ({grade})  [{status}]")

            if status == "PASS":
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  {test['name']:<30} ERROR: {e}")
            failed += 1
            physics_scores.append(0)

    # Physics score summary
    if physics_scores:
        avg_physics = sum(physics_scores) / len(physics_scores)
        overall_grade = grade_to_label(avg_physics)
        print(f"\n  {'─' * 60}")
        print(f"  PHYSICS SCORE: {avg_physics:.1f}/100 (Grade: {overall_grade})")
        if avg_physics >= 90:
            print(f"  Rating: Production Ready")
        elif avg_physics >= 75:
            print(f"  Rating: Research Quality")
        elif avg_physics >= 60:
            print(f"  Rating: Needs Improvement")
        else:
            print(f"  Rating: Not Recommended")

    # ── Input Validation Tests ──
    print("\n" + "=" * 70)
    print("  TEST GROUP 3: Input Validation (Invalid Input Rejection)")
    print("=" * 70)

    for test in INVALID_TESTS:
        try:
            pred = oracle.predict(**test["params"])
            if test["should_fail"]:
                print(f"  {test['name']:<30} FAIL (should have rejected)")
                failed += 1
            else:
                print(f"  {test['name']:<30} PASS (accepted valid input)")
                passed += 1
        except ValueError as e:
            if test["should_fail"]:
                print(f"  {test['name']:<30} PASS (correctly rejected)")
                passed += 1
            else:
                print(f"  {test['name']:<30} FAIL (wrongly rejected: {e})")
                failed += 1
        except Exception as e:
            print(f"  {test['name']:<30} ERROR: {e}")
            failed += 1

    # ── Extended Tests: Model Robustness ──
    print("\n" + "=" * 70)
    print("  EXTENDED: Model Robustness (TST-500)")
    print("=" * 70)

    for test in ROBUSTNESS_TESTS:
        try:
            if test["test"] == "reproducibility":
                # TST-501: Same input, same output N times
                results = []
                for _ in range(test["iterations"]):
                    pred = oracle.predict(**test["params"])
                    results.append(pred.snr_db)
                variance = np.var(results)
                status = "PASS" if variance == 0 else "FAIL"
                print(f"  {test['name']:<30} Variance: {variance:.6f}  [{status}]")

            elif test["test"] == "perturbation":
                # TST-502: Small changes, small outputs
                base_pred = oracle.predict(**test["params"]).snr_db
                max_delta = 0
                for key in ["sfi", "kp", "hour"]:
                    if key in test["params"]:
                        perturbed = test["params"].copy()
                        perturbed[key] = test["params"][key] * (1 + test["perturbation"])
                        perturbed_pred = oracle.predict(**perturbed).snr_db
                        delta = abs(perturbed_pred - base_pred)
                        max_delta = max(max_delta, delta)
                status = "PASS" if max_delta < test["max_delta_db"] else "FAIL"
                print(f"  {test['name']:<30} Max delta: {max_delta:.4f} dB  [{status}]")

            elif test["test"] == "boundary":
                # TST-503: Boundary values produce finite output
                pred = oracle.predict(**test["params"])
                is_finite = np.isfinite(pred.snr_db) and -50 < pred.snr_db < 30
                status = "PASS" if is_finite else "FAIL"
                print(f"  {test['name']:<30} SNR: {pred.snr_db:+.1f} dB  [{status}]")

            elif test["test"] == "checkpoint_rmse":
                # TST-506: Checkpoint integrity
                actual = oracle.metadata['rmse']
                diff = abs(actual - test["expected"])
                status = "PASS" if diff < test["tolerance"] else "FAIL"
                print(f"  {test['name']:<30} Expected: {test['expected']:.4f}, "
                      f"Actual: {actual:.4f}  [{status}]")

            elif test["test"] == "checkpoint_pearson":
                actual = oracle.metadata['pearson']
                diff = abs(actual - test["expected"])
                status = "PASS" if diff < test["tolerance"] else "FAIL"
                print(f"  {test['name']:<30} Expected: {test['expected']:+.4f}, "
                      f"Actual: {actual:+.4f}  [{status}]")

            elif test["test"] == "geo_bias":
                # TST-701: Compare dense (EU) vs sparse (Africa) regions
                # Both paths ~6000 km, mid-latitude, same conditions
                # EU path: G → DL (London to Berlin)
                snr_eu = oracle.predict(
                    lat_tx=51.50, lon_tx=-0.12,
                    lat_rx=52.52, lon_rx=13.40,
                    freq_mhz=14.0, sfi=150, kp=2, hour=14,
                ).snr_db
                # Africa path: 5H → 9J (Tanzania to Zambia)
                snr_africa = oracle.predict(
                    lat_tx=-6.17, lon_tx=35.74,
                    lat_rx=-15.42, lon_rx=28.28,
                    freq_mhz=14.0, sfi=150, kp=2, hour=14,
                ).snr_db
                bias = abs(snr_eu - snr_africa)
                # Both paths are similar distance/conditions; >5 dB difference suggests bias
                status = "PASS" if bias < 5.0 else "FAIL"
                print(f"  {test['name']:<30} EU: {snr_eu:+.1f} dB, Africa: {snr_africa:+.1f} dB, "
                      f"Bias: {bias:.1f} dB  [{status}]")

            if status == "PASS":
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  {test['name']:<30} ERROR: {e}")
            failed += 1

    # ── Extended Tests: Regression ──
    print("\n" + "=" * 70)
    print("  EXTENDED: Regression Tests (TST-800)")
    print("=" * 70)

    for test in REGRESSION_TESTS:
        try:
            if test["test"] == "reference":
                pred = oracle.predict(**test["params"])
                diff = abs(pred.snr_db - test["expected_snr"])
                status = "PASS" if diff < test["tolerance"] else "FAIL"
                print(f"  {test['name']:<30} Expected: {test['expected_snr']:+.1f} dB, "
                      f"Actual: {pred.snr_db:+.1f} dB, Diff: {diff:.2f} dB  [{status}]")

            if status == "PASS":
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  {test['name']:<30} ERROR: {e}")
            failed += 1

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed == 0:
        print("\n  ALL TESTS PASSED")
        return 0
    else:
        print(f"\n  {failed} TEST(S) FAILED")
        return 1


# ── Interactive Mode ──────────────────────────────────────────────────────────

def interactive_mode():
    """Interactive prediction mode."""
    print("=" * 70)
    print("  IONIS V12 Propagation Oracle — Interactive Mode")
    print("=" * 70)

    try:
        oracle = IonisOracle()
        print(f"Model loaded: {oracle.metadata['architecture']}")
        print(f"RMSE: {oracle.metadata['rmse']:.4f} dB")
        print("\nEnter path parameters (or 'q' to quit, 't' to run tests)")
    except Exception as e:
        print(f"CRITICAL: Could not load model — {e}")
        return

    while True:
        try:
            print("\n" + "-" * 40)
            i = input("TX Lat,Lon (e.g., 39.14,-77.01): ").strip()
            if i.lower() == 'q':
                break
            if i.lower() == 't':
                run_test_suite()
                continue
            lat_tx, lon_tx = map(float, i.split(','))

            i = input("RX Lat,Lon (e.g., 51.50,-0.12): ").strip()
            lat_rx, lon_rx = map(float, i.split(','))

            freq = float(input("Frequency (MHz, e.g., 14.0): ").strip())
            sfi = float(input("Solar Flux Index (SFI, e.g., 150): ").strip())
            kp = float(input("Kp Index (0-9, e.g., 2): ").strip())
            hour = float(input("Hour (UTC 0-23, e.g., 14): ").strip())

            pred = oracle.predict(lat_tx, lon_tx, lat_rx, lon_rx, freq, sfi, kp, hour)

            print(f"\n>>> PREDICTION:")
            print(f"    Path:       {pred.distance_km:.0f} km @ {pred.bearing_deg:.0f}°")
            print(f"    SNR:        {pred.snr_db:+.1f} dB")
            print(f"    Condition:  {pred.condition}")
            print(f"    Confidence: {pred.confidence}")

            if pred.warnings:
                print(f"    Warnings:   {'; '.join(pred.warnings)}")

        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            break

    print("\nOracle offline.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.exit(run_test_suite())
    else:
        interactive_mode()
