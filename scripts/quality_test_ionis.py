#!/usr/bin/env python3
"""
quality_test_ionis.py — Step K: IONIS Quality Test (Pearson + RMSE)

Run IONIS V13 inference on 100K high-confidence signatures and compute
correlation metrics against ground truth median_snr.

Compares to VOACAP results from validation.quality_test_voacap.

Usage:
  python quality_test_ionis.py
"""

import os
import sys
import time
import numpy as np
import torch
import clickhouse_connect

# Import oracle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from oracle_v13 import IonisOracle, build_features, DEVICE, sigma_to_db

# ── Configuration ─────────────────────────────────────────────────────────────

CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

# Band ID to label
BAND_LABELS = {
    102: "160m", 103: "80m", 104: "60m", 105: "40m", 106: "30m",
    107: "20m", 108: "17m", 109: "15m", 110: "12m", 111: "10m",
}


def main():
    print("=" * 70)
    print("  IONIS Quality Test — Step K (Pearson + RMSE)")
    print("=" * 70)

    # Connect to ClickHouse
    print(f"\nConnecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # Load oracle
    print("Loading IONIS V13 oracle...")
    oracle = IonisOracle()
    print(f"Model: {oracle.metadata['version']}, RMSE: {oracle.metadata['rmse']:.4f}σ")

    # Query test paths
    print("\nQuerying validation.quality_test_paths...")
    t0 = time.monotonic()

    query = """
    SELECT
        path_id,
        band,
        tx_lat,
        tx_lon,
        rx_lat,
        rx_lon,
        freq_mhz,
        hour,
        month,
        avg_sfi,
        avg_kp,
        median_snr
    FROM validation.quality_test_paths
    ORDER BY path_id
    """

    result = client.query(query)
    rows = result.result_rows
    print(f"Loaded {len(rows):,} paths in {time.monotonic() - t0:.1f}s")

    # Extract arrays
    path_ids = np.array([r[0] for r in rows], dtype=np.int64)
    bands = np.array([r[1] for r in rows], dtype=np.int32)
    tx_lats = np.array([r[2] for r in rows], dtype=np.float32)
    tx_lons = np.array([r[3] for r in rows], dtype=np.float32)
    rx_lats = np.array([r[4] for r in rows], dtype=np.float32)
    rx_lons = np.array([r[5] for r in rows], dtype=np.float32)
    freq_mhz = np.array([r[6] for r in rows], dtype=np.float32)
    hours = np.array([r[7] for r in rows], dtype=np.float32)
    months = np.array([r[8] for r in rows], dtype=np.float32)
    sfis = np.array([r[9] for r in rows], dtype=np.float32)
    kps = np.array([r[10] for r in rows], dtype=np.float32)
    actual_snr = np.array([r[11] for r in rows], dtype=np.float32)

    n = len(rows)

    # Run IONIS inference (batched for speed)
    print(f"\nRunning IONIS V13 inference on {n:,} paths...")
    t0 = time.monotonic()

    # Build all features at once
    ionis_snr_sigma = np.zeros(n, dtype=np.float32)
    ionis_snr_db = np.zeros(n, dtype=np.float32)

    # Process in batches for memory efficiency
    batch_size = 10000
    oracle.model.eval()

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_features = []

            for j in range(i, end):
                feat = build_features(
                    lat_tx=float(tx_lats[j]),
                    lon_tx=float(tx_lons[j]),
                    lat_rx=float(rx_lats[j]),
                    lon_rx=float(rx_lons[j]),
                    freq_mhz=float(freq_mhz[j]),
                    sfi=float(sfis[j]),
                    kp=float(kps[j]),
                    hour=float(hours[j]),
                    month=float(months[j]),
                )
                batch_features.append(feat)

            # Stack and predict
            batch_tensor = torch.cat(batch_features, dim=0)
            predictions = oracle.model(batch_tensor).cpu().numpy().flatten()

            # Store results
            for j, pred_sigma in enumerate(predictions):
                idx = i + j
                ionis_snr_sigma[idx] = pred_sigma
                ionis_snr_db[idx] = sigma_to_db(pred_sigma, bands[idx])

            if (end % 20000) == 0 or end == n:
                print(f"  Processed {end:,}/{n:,} paths...")

    elapsed = time.monotonic() - t0
    print(f"Inference complete in {elapsed:.1f}s ({n/elapsed:.0f} paths/sec)")

    # ── Compute Overall Metrics ───────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Overall Results")
    print("=" * 70)

    # Pearson correlation
    ionis_pearson = np.corrcoef(ionis_snr_db, actual_snr)[0, 1]

    # RMSE
    ionis_rmse = np.sqrt(np.mean((ionis_snr_db - actual_snr) ** 2))

    # Mean error (bias)
    ionis_bias = np.mean(ionis_snr_db - actual_snr)

    print(f"\n  IONIS V13 (n={n:,}):")
    print(f"    Pearson r:  {ionis_pearson:+.4f}")
    print(f"    RMSE:       {ionis_rmse:.2f} dB")
    print(f"    Bias:       {ionis_bias:+.2f} dB")

    # ── Per-Band Breakdown ────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Per-Band Results (IONIS vs VOACAP)")
    print("=" * 70)

    # Query VOACAP results for comparison
    voacap_query = """
    SELECT
        p.band,
        count(*) as n,
        corr(v.voacap_snr, p.median_snr) as voacap_pearson
    FROM validation.quality_test_paths p
    JOIN validation.quality_test_voacap v ON p.path_id = v.path_id
    WHERE v.voacap_snr > -100 AND v.voacap_snr < 200
    GROUP BY p.band
    ORDER BY p.band
    """
    voacap_result = client.query(voacap_query)
    voacap_by_band = {int(r[0]): {'n': int(r[1]), 'pearson': float(r[2])} for r in voacap_result.result_rows}

    print(f"\n  {'Band':<6} {'N':>7} {'IONIS r':>10} {'VOACAP r':>10} {'Delta':>8} {'Winner':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    band_results = []
    for band_id in sorted(BAND_LABELS.keys()):
        mask = bands == band_id
        if mask.sum() == 0:
            continue

        band_actual = actual_snr[mask]
        band_ionis = ionis_snr_db[mask]
        band_n = mask.sum()

        ionis_r = np.corrcoef(band_ionis, band_actual)[0, 1]

        voacap_data = voacap_by_band.get(band_id, {'n': 0, 'pearson': 0})
        voacap_r = voacap_data['pearson']

        delta = ionis_r - voacap_r
        winner = "IONIS" if ionis_r > voacap_r else "VOACAP"

        band_results.append({
            'band': band_id,
            'label': BAND_LABELS[band_id],
            'n': band_n,
            'ionis_r': ionis_r,
            'voacap_r': voacap_r,
            'delta': delta,
            'winner': winner,
        })

        print(f"  {BAND_LABELS[band_id]:<6} {band_n:>7,} {ionis_r:>+10.4f} {voacap_r:>+10.4f} {delta:>+8.4f} {winner:>8}")

    # Count wins
    ionis_wins = sum(1 for r in band_results if r['winner'] == 'IONIS')
    voacap_wins = len(band_results) - ionis_wins

    print(f"\n  IONIS wins {ionis_wins}/{len(band_results)} bands")

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    # Get overall VOACAP Pearson
    voacap_overall_query = """
    SELECT corr(v.voacap_snr, p.median_snr)
    FROM validation.quality_test_paths p
    JOIN validation.quality_test_voacap v ON p.path_id = v.path_id
    WHERE v.voacap_snr > -100 AND v.voacap_snr < 200
    """
    voacap_overall = client.query(voacap_overall_query).result_rows[0][0]

    print(f"\n  Overall Pearson Correlation:")
    print(f"    IONIS:   {ionis_pearson:+.4f}")
    print(f"    VOACAP:  {voacap_overall:+.4f}")
    print(f"    Delta:   {ionis_pearson - voacap_overall:+.4f}")

    if ionis_pearson > voacap_overall:
        print(f"\n  *** IONIS outperforms VOACAP on correlation ***")
    else:
        print(f"\n  VOACAP has higher correlation")

    # Low-band analysis (where VOACAP is anti-correlated)
    print("\n  Low-Band Analysis (160m-40m where VOACAP is anti-correlated):")
    low_bands = [102, 103, 104, 105]
    low_mask = np.isin(bands, low_bands)
    if low_mask.sum() > 0:
        low_ionis_r = np.corrcoef(ionis_snr_db[low_mask], actual_snr[low_mask])[0, 1]
        low_voacap_query = """
        SELECT corr(v.voacap_snr, p.median_snr)
        FROM validation.quality_test_paths p
        JOIN validation.quality_test_voacap v ON p.path_id = v.path_id
        WHERE v.voacap_snr > -100 AND v.voacap_snr < 200
          AND p.band IN (102, 103, 104, 105)
        """
        low_voacap_r = client.query(low_voacap_query).result_rows[0][0]

        print(f"    IONIS:   {low_ionis_r:+.4f}")
        print(f"    VOACAP:  {low_voacap_r:+.4f}")
        print(f"    Delta:   {low_ionis_r - low_voacap_r:+.4f}")

    print("\n" + "=" * 70)
    print("  Quality Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
