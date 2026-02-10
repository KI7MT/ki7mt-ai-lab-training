#!/usr/bin/env python3
"""
oracle_v13.py — IONIS V13 Propagation Oracle

Production inference interface for IONIS V13 Combined model.
Predicts HF SNR for arbitrary paths using the multi-source hybrid model
trained on WSPR + RBN DXpedition signatures.

Key differences from V12:
  - Output is in Z-normalized σ units (training target was Z-score)
  - dB conversion uses per-band WSPR normalization constants
  - Covers 152 rare DXCC entities via RBN DXpedition data

Usage:
  python oracle_v13.py                    # Interactive mode
  python oracle_v13.py --test             # Run test suite
  python oracle_v13.py --batch paths.csv  # Batch prediction
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
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v13_combined.pth")

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

# Per-source per-band normalization constants (from training)
# Format: {band: {'wspr': (mean, std), 'rbn': (mean, std)}}
# Model outputs in σ units; to convert to dB, use: dB = σ * std + mean
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

# Default to 20m for dB conversion when band is ambiguous
DEFAULT_BAND = 107


def sigma_to_db(sigma: float, band: int = DEFAULT_BAND) -> float:
    """Convert Z-score (σ) to dB using WSPR normalization constants."""
    if band in NORM_CONSTANTS:
        mean, std = NORM_CONSTANTS[band]['wspr']
    else:
        mean, std = NORM_CONSTANTS[DEFAULT_BAND]['wspr']
    return sigma * std + mean


def freq_to_band(freq_mhz: float) -> int:
    """Map frequency in MHz to ADIF band ID."""
    # Find closest band
    closest = min(MHZ_TO_BAND.keys(), key=lambda x: abs(x - freq_mhz))
    if abs(closest - freq_mhz) < 2.0:  # Within 2 MHz tolerance
        return MHZ_TO_BAND[closest]
    return DEFAULT_BAND


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
    """Architecture unchanged from V12 — innovation is in data, not structure."""
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
    snr_sigma: float      # Raw model output (Z-score, σ units)
    snr_db: float         # Converted to dB using WSPR normalization
    distance_km: float
    bearing_deg: float
    confidence: str
    condition: str
    band: int             # ADIF band ID used for dB conversion
    warnings: List[str]


class IonisOracle:
    """IONIS V13 Propagation Oracle — production inference interface."""

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
            'version': 'V13 Combined',
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
            snr_sigma = self.model(features).item()

        # Convert σ to dB using band-specific WSPR constants
        band = freq_to_band(freq_mhz)
        snr_db = sigma_to_db(snr_sigma, band)

        # Assess condition (based on dB)
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
            snr_sigma=snr_sigma,
            snr_db=snr_db,
            distance_km=distance_km,
            bearing_deg=bearing_deg,
            confidence=confidence,
            condition=condition,
            band=band,
            warnings=validation.warnings,
        )


# ── Test Suite ────────────────────────────────────────────────────────────────

def run_test_suite():
    """Run V13 physics verification tests."""
    print("=" * 70)
    print("  IONIS V13 Oracle Test Suite")
    print("=" * 70)

    try:
        oracle = IonisOracle()
        print(f"Model loaded: {oracle.metadata['version']}")
        print(f"RMSE: {oracle.metadata['rmse']:.4f}σ, Pearson: {oracle.metadata['pearson']:+.4f}")
    except Exception as e:
        print(f"CRITICAL: Could not load model — {e}")
        sys.exit(1)

    passed = 0
    failed = 0

    # ── Test 1: Storm Sidecar (Kp) ──
    print("\n" + "-" * 60)
    print("  Test 1: Storm Sidecar (Kp 0 vs Kp 9)")
    print("-" * 60)

    snr_quiet = oracle.predict(
        lat_tx=39.14, lon_tx=-77.01, lat_rx=51.50, lon_rx=-0.12,
        freq_mhz=14.0, sfi=150, kp=0, hour=14, month=6
    )
    snr_storm = oracle.predict(
        lat_tx=39.14, lon_tx=-77.01, lat_rx=51.50, lon_rx=-0.12,
        freq_mhz=14.0, sfi=150, kp=9, hour=14, month=6
    )

    storm_cost_sigma = snr_quiet.snr_sigma - snr_storm.snr_sigma
    storm_cost_db = snr_quiet.snr_db - snr_storm.snr_db

    print(f"  Kp 0 (Quiet): {snr_quiet.snr_sigma:+.3f}σ ({snr_quiet.snr_db:+.1f} dB)")
    print(f"  Kp 9 (Storm): {snr_storm.snr_sigma:+.3f}σ ({snr_storm.snr_db:+.1f} dB)")
    print(f"  Storm Cost:   {storm_cost_sigma:+.2f}σ ({storm_cost_db:+.1f} dB)")

    if storm_cost_sigma > 0:
        print("  [PASS] Signal dropped during storm")
        passed += 1
    else:
        print("  [FAIL] Storm should degrade signal")
        failed += 1

    # ── Test 2: Sun Sidecar (SFI) ──
    print("\n" + "-" * 60)
    print("  Test 2: Sun Sidecar (SFI 70 vs SFI 200)")
    print("-" * 60)

    snr_low = oracle.predict(
        lat_tx=39.14, lon_tx=-77.01, lat_rx=51.50, lon_rx=-0.12,
        freq_mhz=14.0, sfi=70, kp=2, hour=14, month=6
    )
    snr_high = oracle.predict(
        lat_tx=39.14, lon_tx=-77.01, lat_rx=51.50, lon_rx=-0.12,
        freq_mhz=14.0, sfi=200, kp=2, hour=14, month=6
    )

    sfi_benefit_sigma = snr_high.snr_sigma - snr_low.snr_sigma
    sfi_benefit_db = snr_high.snr_db - snr_low.snr_db

    print(f"  SFI  70 (Low):  {snr_low.snr_sigma:+.3f}σ ({snr_low.snr_db:+.1f} dB)")
    print(f"  SFI 200 (High): {snr_high.snr_sigma:+.3f}σ ({snr_high.snr_db:+.1f} dB)")
    print(f"  SFI Benefit:    {sfi_benefit_sigma:+.2f}σ ({sfi_benefit_db:+.1f} dB)")

    if sfi_benefit_sigma > 0:
        print("  [PASS] Higher SFI improved signal")
        passed += 1
    else:
        print("  [FAIL] Higher SFI should improve signal")
        failed += 1

    # ── Test 3: Physics Ratio ──
    print("\n" + "-" * 60)
    print("  Test 3: Storm/Sun Ratio")
    print("-" * 60)

    ratio = storm_cost_sigma / sfi_benefit_sigma if sfi_benefit_sigma > 0 else float('inf')
    print(f"  Storm cost / SFI benefit = {ratio:.1f}:1")

    if ratio > 1.5:
        print("  [PASS] Storms hurt more than solar helps (realistic)")
        passed += 1
    else:
        print("  [FAIL] Expected storm impact > solar benefit")
        failed += 1

    # ── Test 4: Canonical Path ──
    print("\n" + "-" * 60)
    print("  Test 4: Canonical Path (W3 → G, 20m, 14 UTC)")
    print("-" * 60)

    pred = oracle.predict(
        lat_tx=39.14, lon_tx=-77.01, lat_rx=51.50, lon_rx=-0.12,
        freq_mhz=14.0, sfi=150, kp=2, hour=14, month=6
    )

    print(f"  Path:       {pred.distance_km:.0f} km @ {pred.bearing_deg:.0f}°")
    print(f"  SNR:        {pred.snr_sigma:+.3f}σ ({pred.snr_db:+.1f} dB)")
    print(f"  Condition:  {pred.condition}")
    print(f"  Confidence: {pred.confidence}")

    # This path should be open (SNR > -28 dB)
    if pred.snr_db > -28:
        print("  [PASS] Band predicted open")
        passed += 1
    else:
        print("  [FAIL] W3→G 20m should be open at 14 UTC")
        failed += 1

    # ── Summary ──
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"  SUMMARY: {passed}/{total} passed")
    print("=" * 70)

    if failed == 0:
        print("  ALL TESTS PASSED")
        return 0
    else:
        print(f"  {failed} TEST(S) FAILED")
        return 1


# ── Interactive Mode ──────────────────────────────────────────────────────────

def interactive_mode():
    """Interactive prediction mode."""
    print("=" * 70)
    print("  IONIS V13 Propagation Oracle — Interactive Mode")
    print("=" * 70)

    try:
        oracle = IonisOracle()
        print(f"Model loaded: {oracle.metadata['version']}")
        print(f"RMSE: {oracle.metadata['rmse']:.4f}σ")
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
            print(f"    SNR:        {pred.snr_sigma:+.3f}σ ({pred.snr_db:+.1f} dB)")
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
