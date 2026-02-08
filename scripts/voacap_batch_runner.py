#!/usr/bin/env python3
"""
VOACAP Batch Runner — Step I Validation
========================================

Runs ~965K unique HF circuits through voacapl and stores results in
ClickHouse (validation.step_i_voacap) for head-to-head comparison
with IONIS V12 predictions.

Workflow:
    1. Read paths from validation.step_i_paths (ClickHouse)
    2. Group by unique circuit (tx, rx, freq, year, month, ssn)
    3. Generate VOACAP input cards (one per circuit, all 24 hours)
    4. Run voacapl in parallel (ProcessPoolExecutor, N worker dirs)
    5. Parse SNR/REL/MUFday from Method 30 output
    6. Join back to original rows by circuit key + hour
    7. INSERT results into validation.step_i_voacap

Usage:
    python voacap_batch_runner.py [--workers 32] [--host 10.60.1.1]
                                  [--sample 1000] [--dry-run]
"""

import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOACAPL_BIN = "/usr/local/bin/voacapl"
ITSHFBC_DIR = Path.home() / "itshfbc"

# Mode-dependent band-open thresholds (must match validate_v12.py)
MODE_THRESHOLDS = {
    "DG": -22.0,
    "CW": -22.0,
    "RY": -21.0,
    "PH": -21.0,
}

# VOACAP card template — single frequency, 24 hours, Method 30
# Antenna: const17.voa (17 dBi omni, ships with voacapl)
# TX power: 0.0100 kW (10W) — set on ANTENNA card, not SYSTEM card
# SYSTEM card: min_angle=0.10°, noise=-145dBW, req_rel=90%, req_snr=73dB (defaults)
# Note: SYSTEM params affect only the REL calculation, not SNR predictions.
# SNR is the raw predicted value; we apply our own thresholds post-hoc.
CARD_TEMPLATE = """\
LINEMAX      55       number of lines-per-page
COEFFS    CCIR
TIME          1   24    1    1
MONTH      {year:4d} {month:.2f}
SUNSPOT    {ssn:.0f}.
LABEL     BATCH                   VOACAP
CIRCUIT   {tx_lat}   {tx_lon}    {rx_lat}    {rx_lon}  S     0
SYSTEM       1. 145. 0.10  90. 73.0 3.00 0.10
FPROB      1.00 1.00 1.00 0.00
ANTENNA       1    1    2   30     0.000[default/const17.voa  ]  0.0    0.0100
ANTENNA       2    2    2   30     0.000[default/const17.voa  ]  0.0    0.0100
FREQUENCY {freq:5.2f} 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
METHOD       30    0
EXECUTE
QUIT
"""

# VOACAP output parsing: offsets from FREQ line
OFFSET_MUFDAY = 6
OFFSET_SNR = 11
OFFSET_REL = 13

# Regex for FREQ lines: starts with hour (float), then freq values, ends with FREQ
FREQ_LINE_RE = re.compile(r"^\s+(\d+\.\d)\s+.*FREQ\s*$")


# ---------------------------------------------------------------------------
# Coordinate formatting
# ---------------------------------------------------------------------------

def fmt_lat(lat: float) -> str:
    """Format latitude for VOACAP CIRCUIT card: '46.50N' or '12.30S'."""
    hemi = "N" if lat >= 0 else "S"
    return f"{abs(lat):5.2f}{hemi}"


def fmt_lon(lon: float) -> str:
    """Format longitude for VOACAP CIRCUIT card: '107.00W' or '12.00E'."""
    hemi = "E" if lon >= 0 else "W"
    return f"{abs(lon):6.2f}{hemi}"


# ---------------------------------------------------------------------------
# Circuit key
# ---------------------------------------------------------------------------

def circuit_key(row: dict) -> tuple:
    """Return the dedup key for a row. Rows sharing a circuit key differ only
    by hour_utc (and mode, but VOACAP doesn't know about mode)."""
    return (
        round(float(row["tx_lat"]), 2),
        round(float(row["tx_lon"]), 2),
        round(float(row["rx_lat"]), 2),
        round(float(row["rx_lon"]), 2),
        round(float(row["freq_mhz"]), 2),
        int(row["year"]),
        int(row["month"]),
        round(float(row["ssn"]), 0),
    )


# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------

def generate_card(key: tuple) -> str:
    """Generate a VOACAP input card from a circuit key."""
    tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz, year, month, ssn = key
    return CARD_TEMPLATE.format(
        year=year,
        month=float(month),
        ssn=ssn,
        tx_lat=fmt_lat(tx_lat),
        tx_lon=fmt_lon(tx_lon),
        rx_lat=fmt_lat(rx_lat),
        rx_lon=fmt_lon(rx_lon),
        freq=freq_mhz,
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_voacap_output(output_text: str) -> dict:
    """Parse VOACAP Method 30 output, return {hour_utc: (snr, rel, mufday)}.

    The output has one block per hour (1.0 through 24.0). Each block starts
    with a FREQ line. SNR, REL, MUFday are at fixed offsets from FREQ.
    We read the SECOND value in each data row (our requested frequency is
    in column 2; column 1 is the FOT).
    """
    lines = output_text.split("\n")
    results = {}

    for i, line in enumerate(lines):
        m = FREQ_LINE_RE.match(line)
        if not m:
            continue

        hour_voacap = int(float(m.group(1)))  # 1-24
        hour_utc = hour_voacap - 1  # convert to 0-23

        # Extract values at known offsets
        snr = _parse_field(lines, i, OFFSET_SNR)
        rel = _parse_field(lines, i, OFFSET_REL)
        mufday = _parse_field(lines, i, OFFSET_MUFDAY)

        results[hour_utc] = (snr, rel, mufday)

    return results


def _parse_field(lines: list, freq_idx: int, offset: int) -> float:
    """Extract the second numeric value from a VOACAP output line at
    freq_idx + offset. Returns -999.0 if the field is '-' or unparseable."""
    idx = freq_idx + offset
    if idx >= len(lines):
        return -999.0
    parts = lines[idx].split()
    if len(parts) < 3:
        return -999.0
    val = parts[1]  # Second value = our requested frequency (first = FOT)
    if val == "-":
        return -999.0
    try:
        return float(val)
    except ValueError:
        return -999.0


# ---------------------------------------------------------------------------
# Worker setup
# ---------------------------------------------------------------------------

def setup_worker_dirs(base_dir: Path, n_workers: int) -> list:
    """Create N worker directories, each a lightweight copy of ~/itshfbc.

    Large directories (coeffs, antennas, geocity, geonatio, geostate) are
    symlinked. Only run/ and database/ are copied (VOACAP writes to run/).
    """
    worker_dirs = []
    base_dir.mkdir(parents=True, exist_ok=True)

    for wid in range(n_workers):
        wdir = base_dir / f"worker_{wid:03d}"
        itshfbc = wdir / "itshfbc"

        if itshfbc.exists():
            shutil.rmtree(itshfbc)
        itshfbc.mkdir(parents=True)

        # Symlink large read-only dirs
        for name in ("coeffs", "antennas", "areadata", "area_inv",
                      "geocity", "geonatio", "geostate"):
            src = ITSHFBC_DIR / name
            if src.exists():
                (itshfbc / name).symlink_to(src.resolve())

        # Copy writable dirs
        for name in ("run", "database"):
            src = ITSHFBC_DIR / name
            if src.exists():
                shutil.copytree(src, itshfbc / name)

        # Remove any stale output files from the copied run/ dir
        for stale in (itshfbc / "run").glob("*.out"):
            stale.unlink()

        worker_dirs.append(itshfbc)

    return worker_dirs


def cleanup_worker_dirs(base_dir: Path):
    """Remove worker directories."""
    if base_dir.exists():
        shutil.rmtree(base_dir)


# ---------------------------------------------------------------------------
# Single circuit runner (called in worker process)
# ---------------------------------------------------------------------------

def run_single_circuit(args: tuple) -> tuple:
    """Run voacapl for a single circuit. Returns (circuit_key, hour_results).

    args = (circuit_key, card_text, worker_dir)
    hour_results = {hour_utc: (snr, rel, mufday)} or None on error.
    """
    key, card_text, worker_dir = args

    input_path = worker_dir / "run" / "voacapx.dat"
    output_path = worker_dir / "run" / "voacapx.out"

    try:
        input_path.write_text(card_text)

        # voacapl args: itshfbc_dir input_file output_file
        # input/output filenames are relative to the run/ subdirectory
        result = subprocess.run(
            [VOACAPL_BIN, str(worker_dir), "voacapx.dat", "voacapx.out"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return (key, None)

        if not output_path.exists():
            return (key, None)

        output_text = output_path.read_text()
        hour_results = parse_voacap_output(output_text)
        return (key, hour_results)

    except Exception:
        return (key, None)


# ---------------------------------------------------------------------------
# ClickHouse I/O
# ---------------------------------------------------------------------------

def load_paths_from_clickhouse(host: str, port: int, sample: int = 0) -> list:
    """Load paths from validation.step_i_paths via clickhouse-client."""
    query = "SELECT * FROM validation.step_i_paths"
    if sample > 0:
        query += f" LIMIT {sample}"
    query += " FORMAT CSVWithNames"

    result = subprocess.run(
        ["clickhouse-client", f"--host={host}", f"--port={port}", f"--query={query}"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"ERROR: clickhouse-client failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    reader = csv.DictReader(io.StringIO(result.stdout))
    return list(reader)


def insert_results_to_clickhouse(
    host: str, port: int, rows: list, batch_size: int = 50000
):
    """INSERT results into validation.step_i_voacap in batches."""
    if not rows:
        return

    columns = [
        "tx_lat", "tx_lon", "rx_lat", "rx_lon", "freq_mhz",
        "year", "month", "hour_utc", "ssn", "mode",
        "voacap_snr", "voacap_rel", "voacap_mufday",
        "threshold", "voacap_band_open",
    ]

    total = len(rows)
    inserted = 0

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        tsv_lines = []
        for r in batch:
            tsv_lines.append("\t".join(str(r[c]) for c in columns))
        tsv_data = "\n".join(tsv_lines) + "\n"

        col_list = ", ".join(columns)
        query = f"INSERT INTO validation.step_i_voacap ({col_list}) FORMAT TSV"

        result = subprocess.run(
            ["clickhouse-client", f"--host={host}", f"--port={port}", f"--query={query}"],
            input=tsv_data, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"ERROR inserting batch at offset {batch_start}: "
                  f"{result.stderr.strip()}", file=sys.stderr)
        else:
            inserted += len(batch)

    print(f"Inserted {inserted:,} / {total:,} rows into validation.step_i_voacap")


# ---------------------------------------------------------------------------
# Scoring & reporting
# ---------------------------------------------------------------------------

def compute_metrics(results: list):
    """Compute and print recall metrics matching validate_v12.py format."""

    total = len(results)
    if total == 0:
        print("No results to score.")
        return

    tp = sum(1 for r in results if r["voacap_band_open"] == 1)
    fn = total - tp
    recall = tp / total * 100 if total > 0 else 0.0

    print("\n" + "=" * 65)
    print("VOACAP Step I Validation Results")
    print("=" * 65)
    print(f"  Total paths:     {total:>12,}")
    print(f"  Band open (TP):  {tp:>12,}  ({recall:.2f}%)")
    print(f"  Band closed (FN):{fn:>12,}  ({100 - recall:.2f}%)")
    print(f"  Overall recall:  {recall:>12.2f}%")

    # Per-mode breakdown
    mode_stats = defaultdict(lambda: {"total": 0, "tp": 0, "snr_sum": 0.0})
    for r in results:
        m = r["mode"]
        mode_stats[m]["total"] += 1
        if r["voacap_band_open"] == 1:
            mode_stats[m]["tp"] += 1
        mode_stats[m]["snr_sum"] += r["voacap_snr"]

    print(f"\n  {'Mode':<6} {'Total':>10} {'TP':>10} {'Recall':>8} {'Mean SNR':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for mode in sorted(mode_stats.keys()):
        s = mode_stats[mode]
        rec = s["tp"] / s["total"] * 100 if s["total"] > 0 else 0.0
        mean_snr = s["snr_sum"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {mode:<6} {s['total']:>10,} {s['tp']:>10,} {rec:>7.2f}% {mean_snr:>9.1f}")

    # Per-band breakdown
    band_map = {1.8: "160m", 3.5: "80m", 7.0: "40m", 14.0: "20m",
                21.0: "15m", 28.0: "10m"}
    band_stats = defaultdict(lambda: {"total": 0, "tp": 0, "snr_sum": 0.0})
    for r in results:
        freq = r["freq_mhz"]
        # Find closest band (handles 1.84, 3.57, 14.1, etc.)
        label = None
        for bfreq, bname in band_map.items():
            if abs(freq - bfreq) < 1.0:
                label = bname
                break
        if label is None:
            label = f"{freq:.1f}MHz"
        band_stats[label]["total"] += 1
        if r["voacap_band_open"] == 1:
            band_stats[label]["tp"] += 1
        band_stats[label]["snr_sum"] += r["voacap_snr"]

    # Sort bands by frequency order
    band_order = ["160m", "80m", "40m", "20m", "15m", "10m"]
    print(f"\n  {'Band':<6} {'Total':>10} {'TP':>10} {'Recall':>8} {'Mean SNR':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for band in band_order:
        if band not in band_stats:
            continue
        s = band_stats[band]
        rec = s["tp"] / s["total"] * 100 if s["total"] > 0 else 0.0
        mean_snr = s["snr_sum"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {band:<6} {s['total']:>10,} {s['tp']:>10,} {rec:>7.2f}% {mean_snr:>9.1f}")
    # Any bands not in the standard list
    for band in sorted(band_stats.keys()):
        if band not in band_order:
            s = band_stats[band]
            rec = s["tp"] / s["total"] * 100 if s["total"] > 0 else 0.0
            mean_snr = s["snr_sum"] / s["total"] if s["total"] > 0 else 0.0
            print(f"  {band:<6} {s['total']:>10,} {s['tp']:>10,} {rec:>7.2f}% {mean_snr:>9.1f}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VOACAP batch runner for Step I validation")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--host", default="localhost",
                        help="ClickHouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=9000,
                        help="ClickHouse native port (default: 9000)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N rows (0 = all, for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate cards and parse but don't INSERT")
    parser.add_argument("--work-dir", default="/tmp/voacap_work",
                        help="Worker directory base (default: /tmp/voacap_work)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)

    # -----------------------------------------------------------------------
    # 1. Load paths from ClickHouse
    # -----------------------------------------------------------------------
    print(f"Loading paths from validation.step_i_paths "
          f"({args.host}:{args.port})...")
    rows = load_paths_from_clickhouse(args.host, args.port, args.sample)
    print(f"  Loaded {len(rows):,} rows")

    if not rows:
        print("No rows to process.")
        return

    # -----------------------------------------------------------------------
    # 2. Group by unique circuit
    # -----------------------------------------------------------------------
    circuits = defaultdict(list)  # circuit_key -> [row_indices]
    for idx, row in enumerate(rows):
        key = circuit_key(row)
        circuits[key].append(idx)

    print(f"  Unique circuits: {len(circuits):,} "
          f"(dedup ratio: {len(rows)/len(circuits):.2f}x)")

    # -----------------------------------------------------------------------
    # 3. Set up worker directories
    # -----------------------------------------------------------------------
    print(f"Setting up {args.workers} worker directories in {work_dir}...")
    worker_dirs = setup_worker_dirs(work_dir, args.workers)

    # -----------------------------------------------------------------------
    # 4. Generate cards and assign to workers
    # -----------------------------------------------------------------------
    circuit_keys = list(circuits.keys())
    tasks = []
    for i, key in enumerate(circuit_keys):
        card = generate_card(key)
        wdir = worker_dirs[i % args.workers]
        tasks.append((key, card, wdir))

    print(f"  Generated {len(tasks):,} VOACAP input cards")

    # -----------------------------------------------------------------------
    # 5. Run voacapl in parallel
    # -----------------------------------------------------------------------
    print(f"Running voacapl with {args.workers} workers...")
    t0 = time.time()

    circuit_results = {}  # key -> {hour: (snr, rel, mufday)}
    errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_single_circuit, t): t[0] for t in tasks}

        for future in as_completed(futures):
            key, hour_data = future.result()
            completed += 1
            if hour_data is None:
                errors += 1
            else:
                circuit_results[key] = hour_data

            if completed % 10000 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"\r  Progress: {completed:,}/{len(tasks):,} "
                      f"({completed/len(tasks)*100:.1f}%) "
                      f"| {rate:.0f} circuits/s "
                      f"| errors: {errors:,}", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s "
          f"({len(tasks)/elapsed:.0f} circuits/s, {errors:,} errors)")

    # -----------------------------------------------------------------------
    # 6. Join results back to original rows
    # -----------------------------------------------------------------------
    print("Joining VOACAP results to original rows...")
    output_rows = []
    missing = 0

    for idx, row in enumerate(rows):
        key = circuit_key(row)
        hour_utc = int(row["hour_utc"])
        mode = row["mode"]
        threshold = MODE_THRESHOLDS.get(mode, -22.0)

        hour_data = circuit_results.get(key)
        if hour_data is None or hour_utc not in hour_data:
            missing += 1
            # Still emit row with sentinel values
            snr, rel, mufday = -999.0, 0.0, 0.0
        else:
            snr, rel, mufday = hour_data[hour_utc]

        band_open = 1 if snr >= threshold else 0

        output_rows.append({
            "tx_lat": round(float(row["tx_lat"]), 2),
            "tx_lon": round(float(row["tx_lon"]), 2),
            "rx_lat": round(float(row["rx_lat"]), 2),
            "rx_lon": round(float(row["rx_lon"]), 2),
            "freq_mhz": round(float(row["freq_mhz"]), 2),
            "year": int(row["year"]),
            "month": int(row["month"]),
            "hour_utc": hour_utc,
            "ssn": round(float(row["ssn"]), 0),
            "mode": mode,
            "voacap_snr": round(snr, 2),
            "voacap_rel": round(rel, 4),
            "voacap_mufday": round(mufday, 4),
            "threshold": threshold,
            "voacap_band_open": band_open,
        })

    print(f"  Joined {len(output_rows):,} rows "
          f"({missing:,} missing VOACAP data)")

    # -----------------------------------------------------------------------
    # 7. Score and report
    # -----------------------------------------------------------------------
    # Attach numeric fields for scoring
    for r in output_rows:
        r["freq_mhz_num"] = r["freq_mhz"]

    score_rows = [{"mode": r["mode"], "freq_mhz": r["freq_mhz"],
                    "voacap_snr": r["voacap_snr"],
                    "voacap_band_open": r["voacap_band_open"]}
                   for r in output_rows if r["voacap_snr"] > -999.0]
    compute_metrics(score_rows)

    # -----------------------------------------------------------------------
    # 8. INSERT into ClickHouse
    # -----------------------------------------------------------------------
    if not args.dry_run:
        print(f"\nInserting results into validation.step_i_voacap...")
        insert_results_to_clickhouse(args.host, args.port, output_rows)
    else:
        print("\n[DRY RUN] Skipping ClickHouse INSERT")

    # -----------------------------------------------------------------------
    # 9. Cleanup
    # -----------------------------------------------------------------------
    print("Cleaning up worker directories...")
    cleanup_worker_dirs(work_dir)
    print("Done.")


if __name__ == "__main__":
    main()
