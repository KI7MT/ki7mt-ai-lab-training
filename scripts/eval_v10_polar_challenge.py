#!/usr/bin/env python3
"""
eval_v10_polar_challenge.py — IONIS-V10-POL-01: Polar Challenge

Stress test proving the DNN amplifies the Storm Sidecar's global penalty
at high latitudes. If the model is truly geographically aware, polar paths
should suffer MORE from geomagnetic storms than equatorial paths.

Test logic:
  1. Load training_v6_clean.csv, split into Polar (|lat| > 60°) and
     Equatorial (|lat| < 20°) subsets.
  2. Run inference on actual data for baseline RMSE/Pearson per region.
  3. Controlled Kp sweep: override kp_penalty in each region's data
     while keeping all geographic/temporal features fixed.
  4. Controlled SFI sweep: verify monotonicity holds in both regions.
  5. Generate scatter plot and automated Markdown report.

Success criteria:
  - Polar Kp 9 SNR drop > Equatorial Kp 9 SNR drop (geographic amplification)
  - Polar RMSE < 2.60 dB
  - SFI monotonicity positive in both regions
"""

import os
import sys
import time
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure we can import from the scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)

# --- Paths ---
CSV_PATH = os.path.join(TRAINING_DIR, "data", "training_v6_clean.csv")
MODEL_PATH = os.path.join(TRAINING_DIR, "models", "ionis_v10_final.pth")
DOCS_DIR = os.path.join(os.path.dirname(TRAINING_DIR), "ki7mt-ai-lab-docs")
REPORT_PATH = os.path.join(
    DOCS_DIR, "docs", "validation", "polar_tests", "IONIS-V10-POL-01.md"
)
PLOT_PATH = os.path.join(
    DOCS_DIR, "docs", "validation", "polar_tests", "IONIS-V10-POL-01-kp-drop-by-lat.png"
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Feature configuration (must match train_v10_final.py exactly)
DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13

BAND_TO_HZ = {
    102:  1_836_600,  103:  3_568_600,  104:  5_287_200,
    105:  7_038_600,  106: 10_138_700,  107: 14_097_100,
    108: 18_104_600,  109: 21_094_600,  110: 24_924_600,
    111: 28_124_600,
}


# ── Model Architecture (must match train_v10_final.py) ─────────────────────
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


class IonisDualMono(nn.Module):
    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.dnn_dim = dnn_dim

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        return self.dnn(x_deep) + self.sun_sidecar(x_sfi) + self.storm_sidecar(x_kp)


# ── Feature Engineering (matches train_v10_final.py) ──────────────────────
GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def grid_to_latlon_series(grids):
    lats = np.zeros(len(grids), dtype=np.float32)
    lons = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        s = str(g).strip().rstrip('\x00')
        m = GRID_RE.search(s)
        g4 = m.group(0).upper() if m else 'JJ00'
        lons[i] = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
        lats[i] = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lats, lons


def engineer_features(df):
    """Compute 13 normalized features from raw CSV columns."""
    distance = df['distance'].values.astype(np.float32)
    band = df['band'].values.astype(np.int32)
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)
    azimuth = df['azimuth'].values.astype(np.float32)
    sfi = df['sfi'].values.astype(np.float32)
    kp_penalty = df['kp_penalty'].values.astype(np.float32)

    tx_lat, tx_lon = grid_to_latlon_series(df['tx_grid'].values)
    rx_lat, rx_lon = grid_to_latlon_series(df['rx_grid'].values)

    freq_hz = np.array([BAND_TO_HZ.get(int(b), 14_097_100) for b in band], dtype=np.float64)

    f_distance     = distance / 20000.0
    f_freq_log     = np.log10(freq_hz.astype(np.float32)) / 8.0
    f_hour_sin     = np.sin(2.0 * np.pi * hour / 24.0)
    f_hour_cos     = np.cos(2.0 * np.pi * hour / 24.0)
    f_az_sin       = np.sin(2.0 * np.pi * azimuth / 360.0)
    f_az_cos       = np.cos(2.0 * np.pi * azimuth / 360.0)
    f_lat_diff     = np.abs(tx_lat - rx_lat) / 180.0
    f_midpoint_lat = df['midpoint_lat'].values.astype(np.float32) / 90.0
    f_season_sin   = np.sin(2.0 * np.pi * month / 12.0)
    f_season_cos   = np.cos(2.0 * np.pi * month / 12.0)
    midpoint_lon   = (tx_lon + rx_lon) / 2.0
    local_solar_h  = hour + midpoint_lon / 15.0
    f_daynight     = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    f_sfi          = sfi / 300.0
    f_kp_penalty   = kp_penalty

    return np.column_stack([
        f_distance, f_freq_log, f_hour_sin, f_hour_cos,
        f_az_sin, f_az_cos, f_lat_diff, f_midpoint_lat,
        f_season_sin, f_season_cos, f_daynight,
        f_sfi, f_kp_penalty,
    ]).astype(np.float32)


# ── Metrics ────────────────────────────────────────────────────────────────
def pearson_r(pred, target):
    p, t = pred.flatten(), target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = np.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ── Batch Inference ────────────────────────────────────────────────────────
def batch_predict(model, features, batch_size=16384):
    """Run inference in batches to avoid OOM on large datasets."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.tensor(features[i:i + batch_size],
                                 dtype=torch.float32, device=DEVICE)
            out = model(batch).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds).flatten()


def controlled_kp_predict(model, features, kp_penalty_value):
    """Run inference with kp_penalty overridden to a fixed value."""
    features_copy = features.copy()
    features_copy[:, KP_PENALTY_IDX] = kp_penalty_value
    return batch_predict(model, features_copy)


def controlled_sfi_predict(model, features, sfi_normalized):
    """Run inference with sfi overridden to a fixed value."""
    features_copy = features.copy()
    features_copy[:, SFI_IDX] = sfi_normalized
    return batch_predict(model, features_copy)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'=' * 70}")
    print(f"  IONIS-V10-POL-01: Polar Challenge")
    print(f"  {timestamp}")
    print(f"{'=' * 70}")

    # ── Load model ──
    print(f"\nLoading model: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    dnn_dim = checkpoint.get('dnn_dim', DNN_DIM)
    sidecar_hidden = checkpoint.get('sidecar_hidden', 8)
    model = IonisDualMono(dnn_dim=dnn_dim, sidecar_hidden=sidecar_hidden).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Architecture: {checkpoint.get('architecture', 'IonisDualMono')}")
    print(f"  Features: {checkpoint.get('features', 'unknown')}")

    # ── Load data ──
    print(f"\nLoading data: {CSV_PATH}")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    load_sec = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} rows in {load_sec:.1f}s")

    # ── Split by latitude ──
    midlat = df['midpoint_lat'].values
    polar_mask = np.abs(midlat) > 60.0
    equatorial_mask = np.abs(midlat) < 20.0

    df_polar = df[polar_mask].copy()
    df_equatorial = df[equatorial_mask].copy()
    print(f"\n  Polar subset (|lat| > 60°):     {len(df_polar):>10,} rows")
    print(f"  Equatorial subset (|lat| < 20°): {len(df_equatorial):>10,} rows")

    # ── Engineer features ──
    print("\nEngineering features...")
    t0 = time.perf_counter()
    X_polar = engineer_features(df_polar)
    X_equatorial = engineer_features(df_equatorial)
    y_polar = df_polar['snr'].values.astype(np.float32)
    y_equatorial = df_equatorial['snr'].values.astype(np.float32)
    feat_sec = time.perf_counter() - t0
    print(f"  Feature engineering: {feat_sec:.1f}s")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Baseline RMSE and Pearson per region
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  TEST 1: Baseline Inference (actual data)")
    print(f"{'─' * 70}")

    pred_polar = batch_predict(model, X_polar)
    pred_equatorial = batch_predict(model, X_equatorial)

    rmse_polar = rmse(pred_polar, y_polar)
    rmse_equatorial = rmse(pred_equatorial, y_equatorial)
    pearson_polar = pearson_r(pred_polar, y_polar)
    pearson_equatorial = pearson_r(pred_equatorial, y_equatorial)

    print(f"  {'Region':<14s}  {'RMSE':>8s}  {'Pearson':>8s}  {'Rows':>10s}")
    print(f"  {'-' * 44}")
    print(f"  {'Polar':<14s}  {rmse_polar:7.2f}dB  {pearson_polar:+7.4f}  {len(df_polar):>10,}")
    print(f"  {'Equatorial':<14s}  {rmse_equatorial:7.2f}dB  {pearson_equatorial:+7.4f}  {len(df_equatorial):>10,}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Controlled Kp Sweep — the core Polar Challenge
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  TEST 2: Controlled Kp Sweep (geographic features fixed)")
    print(f"{'─' * 70}")

    kp_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    polar_snrs = []
    equatorial_snrs = []

    print(f"\n  {'Kp':>3s}  {'kp_penalty':>10s}  {'Polar SNR':>10s}  {'Eq. SNR':>10s}  {'Delta':>8s}")
    print(f"  {'-' * 48}")

    for kp in kp_values:
        kp_penalty = 1.0 - kp / 9.0
        p_snr = controlled_kp_predict(model, X_polar, kp_penalty).mean()
        e_snr = controlled_kp_predict(model, X_equatorial, kp_penalty).mean()
        polar_snrs.append(p_snr)
        equatorial_snrs.append(e_snr)
        delta = p_snr - e_snr
        print(f"  {kp:3d}  {kp_penalty:10.4f}  {p_snr:+9.2f}dB  {e_snr:+9.2f}dB  {delta:+7.2f}dB")

    # Storm drop calculations
    polar_drop = polar_snrs[0] - polar_snrs[-1]       # Kp 0 - Kp 9
    equatorial_drop = equatorial_snrs[0] - equatorial_snrs[-1]

    print(f"\n  Polar Kp 0→9 SNR drop:      {polar_drop:+.2f} dB")
    print(f"  Equatorial Kp 0→9 SNR drop:  {equatorial_drop:+.2f} dB")

    geographic_amplification = polar_drop - equatorial_drop
    kp_test_passed = polar_drop > equatorial_drop

    if kp_test_passed:
        print(f"  PASS: Polar drop ({polar_drop:.2f} dB) > Equatorial drop ({equatorial_drop:.2f} dB)")
        print(f"  Geographic amplification: {geographic_amplification:+.2f} dB")
    else:
        print(f"  FAIL: Polar drop ({polar_drop:.2f} dB) <= Equatorial drop ({equatorial_drop:.2f} dB)")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: SFI Monotonicity in both regions
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  TEST 3: SFI Monotonicity Check (both regions)")
    print(f"{'─' * 70}")

    sfi_values = [70, 100, 150, 200, 250, 300]
    polar_sfi_snrs = []
    equatorial_sfi_snrs = []

    print(f"\n  {'SFI':>4s}  {'Polar SNR':>10s}  {'Eq. SNR':>10s}")
    print(f"  {'-' * 30}")

    for sfi in sfi_values:
        sfi_norm = sfi / 300.0
        p_snr = controlled_sfi_predict(model, X_polar, sfi_norm).mean()
        e_snr = controlled_sfi_predict(model, X_equatorial, sfi_norm).mean()
        polar_sfi_snrs.append(p_snr)
        equatorial_sfi_snrs.append(e_snr)
        print(f"  {sfi:4d}  {p_snr:+9.2f}dB  {e_snr:+9.2f}dB")

    sfi_polar_monotonic = all(
        polar_sfi_snrs[i + 1] >= polar_sfi_snrs[i]
        for i in range(len(polar_sfi_snrs) - 1)
    )
    sfi_equatorial_monotonic = all(
        equatorial_sfi_snrs[i + 1] >= equatorial_sfi_snrs[i]
        for i in range(len(equatorial_sfi_snrs) - 1)
    )

    polar_sfi_benefit = polar_sfi_snrs[-1] - polar_sfi_snrs[0]
    equatorial_sfi_benefit = equatorial_sfi_snrs[-1] - equatorial_sfi_snrs[0]

    print(f"\n  Polar SFI 70→300 benefit:      {polar_sfi_benefit:+.2f} dB  "
          f"{'PASS' if sfi_polar_monotonic else 'FAIL'}")
    print(f"  Equatorial SFI 70→300 benefit: {equatorial_sfi_benefit:+.2f} dB  "
          f"{'PASS' if sfi_equatorial_monotonic else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: Latitude-binned Kp drop profile (for scatter plot)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  TEST 4: Latitude-Binned Kp Drop Profile")
    print(f"{'─' * 70}")

    lat_bins = np.arange(-80, 90, 10)
    bin_centers = []
    bin_drops = []
    bin_counts = []

    print(f"\n  {'Lat Bin':>12s}  {'Count':>8s}  {'Kp0→9 Drop':>12s}")
    print(f"  {'-' * 36}")

    for lo in lat_bins:
        hi = lo + 10
        mask = (midlat >= lo) & (midlat < hi)
        count = mask.sum()
        if count < 100:
            continue

        df_bin = df[mask]
        X_bin = engineer_features(df_bin)

        snr_kp0 = controlled_kp_predict(model, X_bin, 1.0).mean()   # Kp 0
        snr_kp9 = controlled_kp_predict(model, X_bin, 0.0).mean()   # Kp 9
        drop = snr_kp0 - snr_kp9

        center = lo + 5
        bin_centers.append(center)
        bin_drops.append(drop)
        bin_counts.append(int(count))
        print(f"  {lo:+3d}° to {hi:+3d}°  {count:>8,}  {drop:+10.2f} dB")

    # ═══════════════════════════════════════════════════════════════════════
    # Generate scatter plot
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  Generating scatter plot...")
    print(f"{'─' * 70}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = np.array(bin_counts) / max(bin_counts) * 200 + 20
        scatter = ax.scatter(bin_centers, bin_drops, s=sizes, c=bin_drops,
                             cmap='RdYlBu_r', edgecolors='black', linewidth=0.5,
                             zorder=5)
        ax.plot(bin_centers, bin_drops, color='gray', linewidth=1, alpha=0.5, zorder=3)

        # Threshold lines
        ax.axhline(y=equatorial_drop, color='blue', linestyle='--', alpha=0.7,
                    label=f'Equatorial baseline ({equatorial_drop:.2f} dB)')
        ax.axhline(y=1.50, color='red', linestyle='--', alpha=0.7,
                    label='Polar target (1.50 dB)')
        ax.axvline(x=-60, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=60, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=-20, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=20, color='gray', linestyle=':', alpha=0.4)

        ax.fill_betweenx([0, max(bin_drops) * 1.2], -90, -60,
                         alpha=0.08, color='blue', label='Polar zone')
        ax.fill_betweenx([0, max(bin_drops) * 1.2], 60, 90,
                         alpha=0.08, color='blue')
        ax.fill_betweenx([0, max(bin_drops) * 1.2], -20, 20,
                         alpha=0.08, color='green', label='Equatorial zone')

        ax.set_xlabel('Midpoint Latitude (°)', fontsize=12)
        ax.set_ylabel('Kp 0→9 SNR Drop (dB)', fontsize=12)
        ax.set_title('IONIS-V10-POL-01: Storm Impact by Latitude\n'
                      '(larger dots = more data)', fontsize=13)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-85, 85)
        ax.set_ylim(0, max(bin_drops) * 1.2)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Kp 0→9 Drop (dB)')

        plt.tight_layout()
        os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
        plt.savefig(PLOT_PATH, dpi=150)
        plt.close()
        print(f"  Saved: {PLOT_PATH}")
        plot_generated = True
    except Exception as e:
        print(f"  WARNING: Plot generation failed: {e}")
        plot_generated = False

    # ═══════════════════════════════════════════════════════════════════════
    # Summary verdicts
    # ═══════════════════════════════════════════════════════════════════════
    rmse_polar_pass = rmse_polar < 2.60
    equatorial_drop_pass = equatorial_drop <= 1.50
    polar_drop_pass = polar_drop > 1.50

    overall_pass = (
        kp_test_passed
        and sfi_polar_monotonic
        and sfi_equatorial_monotonic
        and rmse_polar_pass
    )

    print(f"\n{'=' * 70}")
    print("  IONIS-V10-POL-01 SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Equatorial Kp 0→9 drop:  {equatorial_drop:+.2f} dB  "
          f"{'PASS' if equatorial_drop_pass else 'INFO'} (target ≤1.50 dB)")
    print(f"  Polar Kp 0→9 drop:       {polar_drop:+.2f} dB  "
          f"{'PASS' if polar_drop_pass else 'FAIL'} (target >1.50 dB)")
    print(f"  Geographic amplification: {geographic_amplification:+.2f} dB  "
          f"{'PASS' if kp_test_passed else 'FAIL'}")
    print(f"  Polar RMSE:              {rmse_polar:.2f} dB  "
          f"{'PASS' if rmse_polar_pass else 'FAIL'} (target <2.60 dB)")
    print(f"  SFI Monotonicity (Polar): {'PASS' if sfi_polar_monotonic else 'FAIL'}")
    print(f"  SFI Monotonicity (Eq.):   {'PASS' if sfi_equatorial_monotonic else 'FAIL'}")
    print(f"\n  OVERALL: {'PASSED' if overall_pass else 'FAILED'}")
    print(f"{'=' * 70}")

    # ═══════════════════════════════════════════════════════════════════════
    # Generate Markdown report
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\nWriting report: {REPORT_PATH}")

    # Build latitude bin table
    lat_table_rows = ""
    for center, drop, count in zip(bin_centers, bin_drops, bin_counts):
        lat_table_rows += f"| {center:+3.0f}° | {count:>8,} | {drop:+.2f} dB |\n"

    # Build Kp sweep table
    kp_table_rows = ""
    for kp in kp_values:
        kp_penalty = 1.0 - kp / 9.0
        p_snr = polar_snrs[kp]
        e_snr = equatorial_snrs[kp]
        kp_table_rows += (
            f"| {kp} | {kp_penalty:.4f} | {p_snr:+.2f} dB | {e_snr:+.2f} dB |\n"
        )

    # Build SFI table
    sfi_table_rows = ""
    for i, sfi in enumerate(sfi_values):
        sfi_table_rows += (
            f"| {sfi} | {polar_sfi_snrs[i]:+.2f} dB | {equatorial_sfi_snrs[i]:+.2f} dB |\n"
        )

    plot_section = ""
    if plot_generated:
        plot_section = (
            "![Kp Storm Drop by Latitude](IONIS-V10-POL-01-kp-drop-by-lat.png)\n\n"
            "*Bubble size proportional to sample count per latitude bin. "
            "Blue shading = polar zones (|lat| > 60°), green = equatorial (|lat| < 20°).*"
        )
    else:
        plot_section = "*Plot generation failed — matplotlib not available.*"

    report = f"""# IONIS-V10-POL-01: Polar Challenge

- **Timestamp**: {timestamp}
- **Model Checkpoint**: `models/ionis_v10_final.pth`
- **Test Objective**: Prove the DNN amplifies the Storm Sidecar's global Kp penalty at high latitudes, confirming geographic awareness rather than a flat global average.

## 1. Methodology

- **Data Source**: `training_v6_clean.csv` ({len(df):,} rows total)
- **Polar Subset**: |midpoint_lat| > 60° ({len(df_polar):,} rows)
- **Equatorial Subset**: |midpoint_lat| < 20° ({len(df_equatorial):,} rows)
- **Controlled Inference**: All geographic/temporal features held fixed per row; only `kp_penalty` (or `sfi`) swept across conditions.
- **Metric Focus**: Kp 0→9 SNR drop differential between polar and equatorial regions.

## 2. Physical Verification (Ionis Integrity Check)

- **SFI Monotonicity (Polar)**: **{"PASS" if sfi_polar_monotonic else "FAIL"}** — SFI 70→300: {polar_sfi_benefit:+.2f} dB
- **SFI Monotonicity (Equatorial)**: **{"PASS" if sfi_equatorial_monotonic else "FAIL"}** — SFI 70→300: {equatorial_sfi_benefit:+.2f} dB
- **Kp Geographic Amplification**: **{"PASS" if kp_test_passed else "FAIL"}** — Polar drop ({polar_drop:.2f} dB) {">" if kp_test_passed else "≤"} Equatorial drop ({equatorial_drop:.2f} dB)

## 3. Quantitative Results

### Baseline Performance (Actual Data)

| Region | RMSE | Pearson | Rows |
|--------|------|---------|------|
| Polar (|lat| > 60°) | {rmse_polar:.2f} dB | {pearson_polar:+.4f} | {len(df_polar):,} |
| Equatorial (|lat| < 20°) | {rmse_equatorial:.2f} dB | {pearson_equatorial:+.4f} | {len(df_equatorial):,} |

### Controlled Kp Sweep

| Kp | kp_penalty | Polar Mean SNR | Equatorial Mean SNR |
|----|------------|----------------|---------------------|
{kp_table_rows}
**Polar Kp 0→9 drop: {polar_drop:+.2f} dB** | **Equatorial Kp 0→9 drop: {equatorial_drop:+.2f} dB** | **Geographic amplification: {geographic_amplification:+.2f} dB**

### SFI Sweep (Monotonicity Check)

| SFI | Polar Mean SNR | Equatorial Mean SNR |
|-----|----------------|---------------------|
{sfi_table_rows}
### Kp Storm Drop by Latitude Bin

| Latitude | Samples | Kp 0→9 Drop |
|----------|---------|-------------|
{lat_table_rows}
### Success Criteria

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Equatorial Kp 0→9 drop | ≤1.50 dB | {equatorial_drop:.2f} dB | **{"PASS" if equatorial_drop_pass else "INFO"}** |
| Polar Kp 0→9 drop | >1.50 dB | {polar_drop:.2f} dB | **{"PASS" if polar_drop_pass else "FAIL"}** |
| Polar RMSE | <2.60 dB | {rmse_polar:.2f} dB | **{"PASS" if rmse_polar_pass else "FAIL"}** |
| SFI Monotonicity (Polar) | Required | {"Monotonic" if sfi_polar_monotonic else "Non-monotonic"} | **{"PASS" if sfi_polar_monotonic else "FAIL"}** |
| SFI Monotonicity (Eq.) | Required | {"Monotonic" if sfi_equatorial_monotonic else "Non-monotonic"} | **{"PASS" if sfi_equatorial_monotonic else "FAIL"}** |
| Polar > Equatorial drop | Required | {polar_drop:.2f} > {equatorial_drop:.2f} | **{"PASS" if kp_test_passed else "FAIL"}** |

## 4. Visual Evidence

{plot_section}

## 5. Analysis & Conclusion

**Overall Verdict: {"PASSED" if overall_pass else "FAILED"}**

{"The Polar Challenge confirms that IONIS V10 is not a flat global-average model. The DNN has learned to amplify the Storm Sidecar's Kp penalty at high latitudes, producing a " + f"{geographic_amplification:+.2f} dB deeper storm impact in polar regions compared to equatorial paths. This proves the model has internalized the geographic structure of ionospheric storm effects — polar regions, where the auroral oval intensifies during geomagnetic disturbances, correctly exhibit greater SNR degradation." if kp_test_passed else "The Polar Challenge did not confirm geographic amplification of the storm penalty. Further investigation is needed to determine whether the DNN is treating Kp effects as geographically uniform."}

{"SFI monotonicity is preserved in both regions, confirming the Sun Sidecar's physics constraints remain intact even when tested on polar-only and equatorial-only subsets." if sfi_polar_monotonic and sfi_equatorial_monotonic else "SFI monotonicity was violated in one or both regions — the Sun Sidecar constraints may need investigation."}

---
*Auto-generated by `eval_v10_polar_challenge.py` — IONIS V10 Phase 11 Validation Suite*
"""

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"  Report written: {REPORT_PATH}")


if __name__ == '__main__':
    main()
