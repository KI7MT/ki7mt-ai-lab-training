#!/usr/bin/env python3
"""
score_model.py — Batch scorer for IONIS model validation

Generic scoring tool. Takes any model version + config, scores against
any ground truth source (rbn, pskr, contest), writes per-path results
to validation.model_results, and prints summary stats.

This is not version-specific — same tool for V20, V21, V22.

Usage:
    python tools/score_model.py --config versions/v20/config_v20.json --source rbn --profile home_station
    python tools/score_model.py --config versions/v20/config_v20.json --source pskr --profile home_station
    python tools/score_model.py --config versions/v20/config_v20.json --source contest --profile contest_cw

Sources:
    rbn     — rbn.signatures (56.6M aggregated CW/RTTY/PSK31 paths)
    pskr    — pskr.bronze (raw FT8/FT4/CW/etc. spots, solar JOIN per day)
    contest — validation.step_i_paths (1M contest QSO paths with lat/lon)

Station profiles apply a link budget adjustment on top of the model's
WSPR-equivalent prediction:

    adjusted_snr = predicted_snr + 10*log10(P_tx/0.2W) + G_tx + G_rx

The model predicts ionospheric propagation (the hard part). The profile
adds the station-specific power + antenna advantage (the gearbox).
WSJT-X reports all SNR in 2500 Hz reference bandwidth, so no bandwidth
correction is needed — the decode thresholds already account for it.

Profiles: wspr (baseline), qrp_portable, home_station, contest_cw,
          contest_ssb, dxpedition
"""

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import clickhouse_connect
import numpy as np
import pandas as pd
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(REPO_DIR, "versions", "common")
sys.path.insert(0, COMMON_DIR)

from train_common import (
    IonisV12Gate,
    grid4_to_latlon,
    grid4_to_latlon_arrays,
    engineer_features,
)

# ── Constants ────────────────────────────────────────────────────────────────

BAND_NAMES = {
    102: '160m', 103: '80m', 104: '60m', 105: '40m', 106: '30m',
    107: '20m', 108: '17m', 109: '15m', 110: '12m', 111: '10m',
}
HF_BANDS = set(BAND_NAMES.keys())

# Contest mode abbreviations -> ADIF mode names
CONTEST_MODE_MAP = {
    'CW': 'CW',
    'PH': 'SSB',
    'RY': 'RTTY',
    'DG': 'FT8',
}

# Fixed viability thresholds (pre-computed at INSERT time)
VIABILITY = {
    'ft8':  -20.0,
    'cw':   -10.0,
    'rtty':  -5.0,
    'ssb':    5.0,
}

# ── Station Profiles ─────────────────────────────────────────────────────────
# Link budget: advantage_db = 10*log10(P_tx/0.2W) + G_tx + G_rx
#
# WSJT-X reports all SNR in 2500 Hz reference bandwidth (WSPR, FT8, etc.).
# Decode thresholds already account for mode bandwidth. The profile only
# adjusts for power + antenna above the WSPR 200 mW isotropic baseline.
#
# Profiles are intentionally comparable to VOACAP station presets so that
# IONIS vs VOACAP comparisons are apples-to-apples.

PROFILES = {
    # ── Baseline / Reference ──────────────────────────────────────────────
    'wspr': {
        'description': 'WSPR beacon (200 mW, isotropic) — model baseline',
        'tx_power_w': 0.2,
        'tx_gain_dbi': 0.0,
        'rx_gain_dbi': 0.0,
    },
    'wspr_dipole': {
        'description': 'WSPR beacon (200 mW, dipole both ends)',
        'tx_power_w': 0.2,
        'tx_gain_dbi': 2.0,
        'rx_gain_dbi': 2.0,
    },
    'voacap_default': {
        'description': 'VOACAP default (100W, isotropic) — DXLook baseline',
        'tx_power_w': 100.0,
        'tx_gain_dbi': 0.0,
        'rx_gain_dbi': 0.0,
    },

    # ── QRP / Portable ────────────────────────────────────────────────────
    'qrp_milliwatt': {
        'description': 'Milliwatt QRP (1W, random wire) — extreme QRP',
        'tx_power_w': 1.0,
        'tx_gain_dbi': -3.0,
        'rx_gain_dbi': 0.0,
    },
    'qrp_portable': {
        'description': 'QRP portable (5W, EFHW) — SOTA/POTA activator',
        'tx_power_w': 5.0,
        'tx_gain_dbi': -3.0,
        'rx_gain_dbi': 0.0,
    },
    'qrp_home': {
        'description': 'QRP home (5W, dipole both ends)',
        'tx_power_w': 5.0,
        'tx_gain_dbi': 2.0,
        'rx_gain_dbi': 2.0,
    },
    'sota_activator': {
        'description': 'SOTA summit (10W, linked dipole on 6m mast)',
        'tx_power_w': 10.0,
        'tx_gain_dbi': -1.0,
        'rx_gain_dbi': 0.0,
    },
    'pota_activator': {
        'description': 'POTA park (20W, EFHW at 30 ft, chaser has dipole)',
        'tx_power_w': 20.0,
        'tx_gain_dbi': 0.0,
        'rx_gain_dbi': 2.0,
    },

    # ── License-Class Typical ─────────────────────────────────────────────
    'home_vertical': {
        'description': 'Home vertical (100W, ground-mounted vertical)',
        'tx_power_w': 100.0,
        'tx_gain_dbi': 0.0,
        'rx_gain_dbi': 0.0,
    },
    'home_station': {
        'description': 'Typical home station (100W, dipole both ends)',
        'tx_power_w': 100.0,
        'tx_gain_dbi': 2.0,
        'rx_gain_dbi': 2.0,
    },
    'home_beam': {
        'description': 'Home beam (100W, 3-el Yagi TX, dipole RX)',
        'tx_power_w': 100.0,
        'tx_gain_dbi': 8.0,
        'rx_gain_dbi': 2.0,
    },

    # ── Amplified Home Station ────────────────────────────────────────────
    'home_amp_dipole': {
        'description': 'Amplified home (500W, dipole both ends)',
        'tx_power_w': 500.0,
        'tx_gain_dbi': 2.0,
        'rx_gain_dbi': 2.0,
    },
    'home_amp_beam': {
        'description': 'Amplified home beam (500W, tribander both ends)',
        'tx_power_w': 500.0,
        'tx_gain_dbi': 8.0,
        'rx_gain_dbi': 8.0,
    },
    'big_gun': {
        'description': 'Big gun (1.5 kW, stacked Yagis, low-noise RX)',
        'tx_power_w': 1500.0,
        'tx_gain_dbi': 10.0,
        'rx_gain_dbi': 10.0,
    },

    # ── Contest ───────────────────────────────────────────────────────────
    'contest_lp': {
        'description': 'Contest LP (100W, tribander) — ARRL/CQ LP category',
        'tx_power_w': 100.0,
        'tx_gain_dbi': 8.0,
        'rx_gain_dbi': 8.0,
    },
    'contest_cw': {
        'description': 'Contest CW HP (1 kW, tribander both ends)',
        'tx_power_w': 1000.0,
        'tx_gain_dbi': 8.0,
        'rx_gain_dbi': 8.0,
    },
    'contest_ssb': {
        'description': 'Contest SSB HP (1.5 kW, stacked Yagis)',
        'tx_power_w': 1500.0,
        'tx_gain_dbi': 10.0,
        'rx_gain_dbi': 10.0,
    },
    'contest_super': {
        'description': 'Super station (1.5 kW, stacked monobanders, K3LR-class)',
        'tx_power_w': 1500.0,
        'tx_gain_dbi': 14.0,
        'rx_gain_dbi': 14.0,
    },

    # ── DXpedition ────────────────────────────────────────────────────────
    'dxpedition_lite': {
        'description': 'Suitcase DXpedition (100W, vertical on beach)',
        'tx_power_w': 100.0,
        'tx_gain_dbi': 0.0,
        'rx_gain_dbi': 0.0,
    },
    'dxpedition': {
        'description': 'Mid-scale DXpedition (1 kW, vertical + salt water)',
        'tx_power_w': 1000.0,
        'tx_gain_dbi': 3.0,
        'rx_gain_dbi': 2.0,
    },
    'dxpedition_mega': {
        'description': 'Mega DXpedition (1.5 kW, VDA/Yagi, Beverages RX)',
        'tx_power_w': 1500.0,
        'tx_gain_dbi': 8.0,
        'rx_gain_dbi': 6.0,
    },

    # ── Special Operations ────────────────────────────────────────────────
    'maritime_mobile': {
        'description': 'Maritime mobile (100W, backstay antenna, poor ground)',
        'tx_power_w': 100.0,
        'tx_gain_dbi': -2.0,
        'rx_gain_dbi': 0.0,
    },
    'extreme_hf': {
        'description': 'Extreme HF path (1.5 kW, large stacked arrays) — max link budget',
        'tx_power_w': 1500.0,
        'tx_gain_dbi': 16.0,
        'rx_gain_dbi': 16.0,
    },
}


def compute_station_advantage(profile_name):
    """Compute link budget advantage in dB over WSPR baseline."""
    p = PROFILES[profile_name]
    power_db = 10.0 * math.log10(p['tx_power_w'] / 0.2)
    return power_db + p['tx_gain_dbi'] + p['rx_gain_dbi']


INSERT_BATCH = 500_000
INFERENCE_BATCH = 50_000


# ── Utilities ────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_azimuth(lat1, lon1, lat2, lon2):
    """Vectorized initial bearing in degrees [0, 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def latlon_to_grid4(lat, lon):
    """Convert lat/lon to 4-char Maidenhead grid."""
    lon_adj = float(lon) + 180.0
    lat_adj = float(lat) + 90.0
    field_lon = int(np.clip(lon_adj / 20.0, 0, 17))
    field_lat = int(np.clip(lat_adj / 10.0, 0, 17))
    sq_lon = int(np.clip((lon_adj - field_lon * 20.0) / 2.0, 0, 9))
    sq_lat = int(np.clip(lat_adj - field_lat * 10.0, 0, 9))
    return chr(field_lon + ord('A')) + chr(field_lat + ord('A')) + str(sq_lon) + str(sq_lat)


def freq_mhz_to_band(freq):
    """Map frequency in MHz to ADIF band ID."""
    if freq < 2.5:  return 102
    if freq < 4.5:  return 103
    if freq < 6.0:  return 104
    if freq < 8.5:  return 105
    if freq < 12.0: return 106
    if freq < 16.0: return 107
    if freq < 20.0: return 108
    if freq < 23.0: return 109
    if freq < 27.0: return 110
    return 111


# ── Source Loaders ───────────────────────────────────────────────────────────
# Each loader returns (features, meta) where:
#   features: ndarray (n, 13) — model input
#   meta: dict of arrays — tx_grid_4, rx_grid_4, band, hour, month,
#         distance_km, azimuth, actual_snr, actual_mode, avg_sfi, avg_kp

def load_rbn(client, config, limit):
    """
    Load RBN signatures and engineer features.

    RBN signatures have the same schema as WSPR signatures — use
    train_common.engineer_features directly. actual_mode defaults to 'CW'
    (95%+ of RBN skimmer spots are CW).
    """
    limit_clause = f" LIMIT {limit}" if limit else ""
    order_clause = " ORDER BY rand()" if limit else ""

    query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month,
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM rbn.signatures
    WHERE avg_sfi > 0
    {order_clause}{limit_clause}
    """

    print(f"  Loading RBN signatures...")
    t0 = time.perf_counter()
    df = client.query_df(query)
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} rows in {elapsed:.1f}s")

    if len(df) == 0:
        return None, None

    # Strip null bytes from FixedString grids
    for col in ('tx_grid_4', 'rx_grid_4'):
        df[col] = df[col].astype(str).str.rstrip('\x00')

    # Engineer features using train_common (signature schema)
    features = engineer_features(df, config)

    meta = {
        'tx_grid_4':   df['tx_grid_4'].values,
        'rx_grid_4':   df['rx_grid_4'].values,
        'band':        df['band'].values.astype(np.int32),
        'hour':        df['hour'].values.astype(np.uint8),
        'month':       df['month'].values.astype(np.uint8),
        'distance_km': df['avg_distance'].values.astype(np.uint32),
        'azimuth':     df['avg_azimuth'].values.astype(np.uint16),
        'actual_snr':  df['median_snr'].values.astype(np.float32),
        'actual_mode': np.full(len(df), 'CW', dtype=object),
        'avg_sfi':     df['avg_sfi'].values.astype(np.float32),
        'avg_kp':      df['avg_kp'].values.astype(np.float32),
    }

    return features, meta


def load_pskr(client, config, limit):
    """
    Load PSKR spots and engineer features.

    PSKR bronze has raw spots (not aggregated) — compute distance, azimuth,
    and features from grids. Solar data joined per-day from solar.bronze.
    Ground-wave filter: distance > 500 km.
    """
    limit_clause = f" LIMIT {limit}" if limit else ""

    query = f"""
    SELECT
        substring(sender_grid, 1, 4) AS tx_grid,
        substring(receiver_grid, 1, 4) AS rx_grid,
        band,
        snr,
        mode,
        toHour(timestamp) AS hour,
        toMonth(timestamp) AS month,
        toString(toDate(timestamp)) AS spot_date
    FROM pskr.bronze
    WHERE sender_grid != ''
      AND receiver_grid != ''
      AND length(sender_grid) >= 4
      AND length(receiver_grid) >= 4
      AND band IN (102, 103, 104, 105, 106, 107, 108, 109, 110, 111)
    ORDER BY rand()
    {limit_clause}
    """

    print(f"  Loading PSKR spots...")
    t0 = time.perf_counter()
    df = client.query_df(query)
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} raw spots in {elapsed:.1f}s")

    if len(df) == 0:
        return None, None

    # Get solar data per day
    dates = df['spot_date'].unique()
    date_list = "', '".join(str(d)[:10] for d in dates)
    solar_query = f"""
    SELECT
        toString(date) AS date_str,
        avg(adjusted_flux) AS sfi,
        avg(kp_index) AS kp
    FROM solar.bronze
    WHERE toString(date) IN ('{date_list}')
    GROUP BY date_str
    """
    solar_df = client.query_df(solar_query)
    solar_lookup = {}
    for _, row in solar_df.iterrows():
        s = float(row['sfi']) if pd.notna(row['sfi']) else 140.0
        k = float(row['kp']) if pd.notna(row['kp']) else 2.0
        solar_lookup[str(row['date_str'])[:10]] = (s, k)

    print(f"  Solar data: {len(solar_lookup)} days matched")

    # Map solar to spots
    sfi_arr = np.zeros(len(df), dtype=np.float32)
    kp_arr = np.zeros(len(df), dtype=np.float32)
    for i, d in enumerate(df['spot_date'].values):
        sfi_arr[i], kp_arr[i] = solar_lookup.get(str(d)[:10], (140.0, 2.0))

    # Parse grids -> lat/lon
    tx_lats, tx_lons = grid4_to_latlon_arrays(df['tx_grid'].values)
    rx_lats, rx_lons = grid4_to_latlon_arrays(df['rx_grid'].values)

    # Compute path parameters
    distance_km = haversine_km(tx_lats, tx_lons, rx_lats, rx_lons)
    azimuth = compute_azimuth(tx_lats, tx_lons, rx_lats, rx_lons)

    # Ground-wave filter
    mask = distance_km > 500
    kept = mask.sum()
    print(f"  Ground-wave filter: {kept:,} / {len(mask):,} paths > 500 km ({100*kept/len(mask):.1f}%)")

    df = df[mask].reset_index(drop=True)
    tx_lats, tx_lons = tx_lats[mask], tx_lons[mask]
    rx_lats, rx_lons = rx_lats[mask], rx_lons[mask]
    distance_km = distance_km[mask]
    azimuth = azimuth[mask]
    sfi_arr = sfi_arr[mask]
    kp_arr = kp_arr[mask]

    if len(df) == 0:
        return None, None

    # Engineer features
    band = df['band'].values.astype(np.int32)
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)

    band_to_hz = config["band_to_hz"]
    if isinstance(list(band_to_hz.keys())[0], str):
        band_to_hz = {int(k): v for k, v in band_to_hz.items()}
    freq_hz = np.array([band_to_hz.get(b, 14_097_100) for b in band], dtype=np.float32)

    midpoint_lat = (tx_lats + rx_lats) / 2.0
    midpoint_lon = (tx_lons + rx_lons) / 2.0

    input_dim = config["model"]["input_dim"]
    X = np.zeros((len(df), input_dim), dtype=np.float32)
    X[:, 0]  = distance_km / 20000.0
    X[:, 1]  = np.log10(freq_hz) / 8.0
    X[:, 2]  = np.sin(2.0 * np.pi * hour / 24.0)
    X[:, 3]  = np.cos(2.0 * np.pi * hour / 24.0)
    X[:, 4]  = np.sin(2.0 * np.pi * azimuth / 360.0)
    X[:, 5]  = np.cos(2.0 * np.pi * azimuth / 360.0)
    X[:, 6]  = np.abs(tx_lats - rx_lats) / 180.0
    X[:, 7]  = midpoint_lat / 90.0
    X[:, 8]  = np.sin(2.0 * np.pi * month / 12.0)
    X[:, 9]  = np.cos(2.0 * np.pi * month / 12.0)
    X[:, 10] = np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)
    X[:, config["model"]["sfi_idx"]] = sfi_arr / 300.0
    X[:, config["model"]["kp_penalty_idx"]] = 1.0 - kp_arr / 9.0

    meta = {
        'tx_grid_4':   df['tx_grid'].values.astype(str),
        'rx_grid_4':   df['rx_grid'].values.astype(str),
        'band':        band,
        'hour':        df['hour'].values.astype(np.uint8),
        'month':       df['month'].values.astype(np.uint8),
        'distance_km': distance_km.astype(np.uint32),
        'azimuth':     azimuth.astype(np.uint16),
        'actual_snr':  df['snr'].values.astype(np.float32),
        'actual_mode': df['mode'].values.astype(str),
        'avg_sfi':     sfi_arr,
        'avg_kp':      kp_arr,
    }

    return X, meta


def load_contest(client, config, limit):
    """
    Load contest Step I paths and engineer features.

    Uses lat/lon from step_i_paths (no grids). Computes features from raw
    coordinates matching validate_v20.py pattern. SFI derived from SSN:
    SFI = 63 + 0.7 * SSN. Kp assumed 2.0 (quiet contest conditions).

    actual_snr is set to the step_i threshold value (a lower bound on
    actual SNR — every QSO in the log actually happened).
    """
    limit_clause = f" LIMIT {limit}" if limit else ""
    order_clause = " ORDER BY rand()" if limit else ""

    query = f"""
    SELECT tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz,
           year, month, hour_utc, ssn, mode, threshold
    FROM validation.step_i_paths
    {order_clause}{limit_clause}
    """

    print(f"  Loading contest Step I paths...")
    t0 = time.perf_counter()
    result = client.query(query)
    rows = result.result_rows
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(rows):,} paths in {elapsed:.1f}s")

    if not rows:
        return None, None

    tx_lat   = np.array([r[0] for r in rows], dtype=np.float32)
    tx_lon   = np.array([r[1] for r in rows], dtype=np.float32)
    rx_lat   = np.array([r[2] for r in rows], dtype=np.float32)
    rx_lon   = np.array([r[3] for r in rows], dtype=np.float32)
    freq_mhz = np.array([r[4] for r in rows], dtype=np.float32)
    month    = np.array([r[6] for r in rows], dtype=np.int32)
    hour_utc = np.array([r[7] for r in rows], dtype=np.int32)
    ssn      = np.array([r[8] for r in rows], dtype=np.float32)
    modes    = np.array([r[9] for r in rows], dtype=object)
    thresh   = np.array([r[10] for r in rows], dtype=np.float32)

    # Compute path parameters
    distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)
    azimuth = compute_azimuth(tx_lat, tx_lon, rx_lat, rx_lon)
    band_ids = np.array([freq_mhz_to_band(f) for f in freq_mhz], dtype=np.int32)

    # SFI from SSN, Kp assumed 2.0 (contest = quiet conditions)
    sfi = 63.0 + 0.7 * ssn
    kp = 2.0

    # Engineer features (matching validate_v20.py exactly)
    freq_hz = freq_mhz * 1_000_000.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    kp_penalty = np.full(len(rows), 1.0 - kp / 9.0, dtype=np.float32)

    input_dim = config["model"]["input_dim"]
    X = np.zeros((len(rows), input_dim), dtype=np.float32)
    X[:, 0]  = distance_km / 20000.0
    X[:, 1]  = np.log10(freq_hz) / 8.0
    X[:, 2]  = np.sin(2.0 * np.pi * hour_utc / 24.0)
    X[:, 3]  = np.cos(2.0 * np.pi * hour_utc / 24.0)
    X[:, 4]  = np.sin(2.0 * np.pi * azimuth / 360.0)
    X[:, 5]  = np.cos(2.0 * np.pi * azimuth / 360.0)
    X[:, 6]  = np.abs(tx_lat - rx_lat) / 180.0
    X[:, 7]  = midpoint_lat / 90.0
    X[:, 8]  = np.sin(2.0 * np.pi * month / 12.0)
    X[:, 9]  = np.cos(2.0 * np.pi * month / 12.0)
    X[:, 10] = np.cos(2.0 * np.pi * (hour_utc + midpoint_lon / 15.0) / 24.0)
    X[:, config["model"]["sfi_idx"]] = sfi / 300.0
    X[:, config["model"]["kp_penalty_idx"]] = kp_penalty

    # Compute grids from lat/lon for results table
    tx_grids = np.array([latlon_to_grid4(la, lo) for la, lo in zip(tx_lat, tx_lon)])
    rx_grids = np.array([latlon_to_grid4(la, lo) for la, lo in zip(rx_lat, rx_lon)])

    # Map contest modes to ADIF modes
    actual_modes = np.array([CONTEST_MODE_MAP.get(m, m) for m in modes])

    meta = {
        'tx_grid_4':   tx_grids,
        'rx_grid_4':   rx_grids,
        'band':        band_ids,
        'hour':        hour_utc.astype(np.uint8),
        'month':       month.astype(np.uint8),
        'distance_km': distance_km.astype(np.uint32),
        'azimuth':     azimuth.astype(np.uint16),
        'actual_snr':  thresh,   # surrogate: step_i threshold (lower bound)
        'actual_mode': actual_modes,
        'avg_sfi':     sfi.astype(np.float32),
        'avg_kp':      np.full(len(rows), kp, dtype=np.float32),
    }

    return X, meta


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, features, device):
    """Run batched inference, return Z-score predictions."""
    n = len(features)
    predictions = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, INFERENCE_BATCH):
            end = min(i + INFERENCE_BATCH, n)
            batch = torch.tensor(features[i:end], dtype=torch.float32, device=device)
            pred = model(batch).cpu().numpy().flatten()
            predictions[i:end] = pred
            if i > 0 and i % (INFERENCE_BATCH * 20) == 0:
                print(f"    {end:,} / {n:,} ({100 * end / n:.0f}%)")

    return predictions


def denormalize_predictions(z_scores, band_ids, norm_constants):
    """Convert Z-scores to dB using per-band WSPR norm constants."""
    predicted_db = np.zeros_like(z_scores)
    for band_id, (mean, std) in norm_constants.items():
        mask = band_ids == band_id
        count = mask.sum()
        if count > 0:
            predicted_db[mask] = z_scores[mask] * std + mean
    return predicted_db


# ── Scoring & Insert ─────────────────────────────────────────────────────────

def score_and_insert(client, features, meta, model, config, device,
                     norm_constants, mode_thresholds, run_id, run_timestamp,
                     model_version, source, profile_name, advantage_db):
    """Run inference, apply link budget, insert results. Returns recall %."""
    n = len(features)

    # Inference
    print(f"  Running inference on {n:,} paths...")
    t0 = time.perf_counter()
    z_scores = run_inference(model, features, device)
    elapsed = time.perf_counter() - t0
    print(f"  Inference: {elapsed:.1f}s ({n / elapsed:,.0f} paths/sec)")

    # Denormalize Z-scores -> dB (WSPR scale)
    predicted_snr = denormalize_predictions(z_scores, meta['band'], norm_constants)

    # Apply link budget: adjusted = model prediction + station advantage
    adjusted_snr = predicted_snr + advantage_db

    # Viability flags use adjusted SNR (model + gearbox)
    ft8_viable  = (adjusted_snr >= VIABILITY['ft8']).astype(np.uint8)
    cw_viable   = (adjusted_snr >= VIABILITY['cw']).astype(np.uint8)
    rtty_viable = (adjusted_snr >= VIABILITY['rtty']).astype(np.uint8)
    ssb_viable  = (adjusted_snr >= VIABILITY['ssb']).astype(np.uint8)

    # Mode hit uses adjusted SNR: vectorized lookup via pandas Series.map
    actual_modes_series = pd.Series(meta['actual_mode'])
    mode_thresh_arr = actual_modes_series.map(mode_thresholds).fillna(-20.0).values.astype(np.float32)
    mode_hit = (adjusted_snr >= mode_thresh_arr).astype(np.uint8)

    # SNR error (raw model prediction vs actual — no link budget)
    snr_error = predicted_snr - meta['actual_snr']

    # Insert into ClickHouse
    print(f"  Inserting {n:,} results into validation.model_results...")
    column_names = [
        'run_id', 'run_timestamp', 'model_version', 'source', 'profile',
        'tx_grid_4', 'rx_grid_4', 'band', 'hour', 'month',
        'distance_km', 'azimuth',
        'actual_snr', 'actual_mode',
        'predicted_snr', 'snr_error',
        'ft8_viable', 'cw_viable', 'rtty_viable', 'ssb_viable',
        'mode_hit',
        'avg_sfi', 'avg_kp',
    ]

    t0 = time.perf_counter()
    inserted = 0
    for start in range(0, n, INSERT_BATCH):
        end = min(start + INSERT_BATCH, n)
        s = slice(start, end)
        bs = end - start

        data = [
            [run_id] * bs,
            [run_timestamp] * bs,
            [model_version] * bs,
            [source] * bs,
            [profile_name] * bs,
            [str(g)[:4] for g in meta['tx_grid_4'][s]],
            [str(g)[:4] for g in meta['rx_grid_4'][s]],
            meta['band'][s].astype(int).tolist(),
            meta['hour'][s].astype(int).tolist(),
            meta['month'][s].astype(int).tolist(),
            meta['distance_km'][s].astype(int).tolist(),
            meta['azimuth'][s].astype(int).tolist(),
            meta['actual_snr'][s].astype(float).tolist(),
            [str(m) for m in meta['actual_mode'][s]],
            predicted_snr[s].astype(float).tolist(),
            snr_error[s].astype(float).tolist(),
            ft8_viable[s].astype(int).tolist(),
            cw_viable[s].astype(int).tolist(),
            rtty_viable[s].astype(int).tolist(),
            ssb_viable[s].astype(int).tolist(),
            mode_hit[s].astype(int).tolist(),
            meta['avg_sfi'][s].astype(float).tolist(),
            meta['avg_kp'][s].astype(float).tolist(),
        ]

        client.insert('validation.model_results', data,
                       column_names=column_names, column_oriented=True)
        inserted += bs
        if inserted % (INSERT_BATCH * 5) == 0:
            print(f"    {inserted:,} / {n:,}")

    elapsed = time.perf_counter() - t0
    print(f"  Insert: {elapsed:.1f}s ({n / elapsed:,.0f} rows/sec)")

    # Summary
    recall = 100.0 * mode_hit.sum() / n
    rmse = np.sqrt(np.mean(snr_error ** 2))

    print()
    print(f"  Summary for {source} (profile: {profile_name}, +{advantage_db:.1f} dB):")
    print(f"    Total scored:   {n:,}")
    print(f"    Mode hits:      {mode_hit.sum():,}")
    print(f"    Overall recall: {recall:.2f}%")
    print(f"    Model SNR:      min={predicted_snr.min():.1f}, max={predicted_snr.max():.1f}, mean={predicted_snr.mean():.1f} dB")
    print(f"    Adjusted SNR:   min={adjusted_snr.min():.1f}, max={adjusted_snr.max():.1f}, mean={adjusted_snr.mean():.1f} dB")
    print(f"    SNR error:      bias={snr_error.mean():+.2f}, RMSE={rmse:.2f} dB")
    print()

    # Per-band recall
    print(f"    {'Band':>6s}  {'Count':>10s}  {'Recall':>8s}  {'Adj SNR':>10s}  {'RMSE':>8s}")
    print(f"    {'-'*48}")
    for band_id in sorted(BAND_NAMES.keys()):
        mask = meta['band'] == band_id
        if mask.sum() == 0:
            continue
        b_recall = 100.0 * mode_hit[mask].sum() / mask.sum()
        b_mean = adjusted_snr[mask].mean()
        b_rmse = np.sqrt(np.mean(snr_error[mask] ** 2))
        print(f"    {BAND_NAMES[band_id]:>6s}  {mask.sum():>10,}  {b_recall:>7.2f}%  {b_mean:>+9.1f} dB  {b_rmse:>7.2f}")
    print()

    # Per-mode recall
    unique_modes = sorted(set(meta['actual_mode']))
    print(f"    {'Mode':>8s}  {'Count':>10s}  {'Recall':>8s}")
    print(f"    {'-'*30}")
    for m in unique_modes:
        mask = np.array([x == m for x in meta['actual_mode']])
        if mask.sum() == 0:
            continue
        m_recall = 100.0 * mode_hit[mask].sum() / mask.sum()
        print(f"    {m:>8s}  {mask.sum():>10,}  {m_recall:>7.2f}%")

    return recall


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IONIS Batch Scorer — score model against ground truth sources')
    parser.add_argument('--config', required=True,
                        help='Path to model config JSON (e.g., versions/v20/config_v20.json)')
    parser.add_argument('--source', required=True, choices=['rbn', 'pskr', 'contest'],
                        help='Ground truth source to score against')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max rows to score (0 = all)')
    parser.add_argument('--profile', default='wspr',
                        choices=list(PROFILES.keys()),
                        help='Station profile for link budget (default: wspr)')
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)

    with open(config_path) as f:
        config = json.load(f)

    model_version = config["version"]
    checkpoint_path = os.path.join(config_dir, config["checkpoint"])

    # Compute station advantage
    advantage_db = compute_station_advantage(args.profile)
    profile_info = PROFILES[args.profile]

    print("=" * 70)
    print(f"  IONIS Batch Scorer -- {model_version}")
    print("=" * 70)
    print(f"  Config:     {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Source:     {args.source}")
    print(f"  Profile:    {args.profile} (+{advantage_db:.1f} dB)")
    print(f"              {profile_info['tx_power_w']}W, TX {profile_info['tx_gain_dbi']} dBi, RX {profile_info['rx_gain_dbi']} dBi")
    print(f"  Limit:      {args.limit if args.limit else 'all'}")
    print()

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Load model
    print(f"  Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IonisV12Gate(
        dnn_dim=config["model"]["dnn_dim"],
        sidecar_hidden=config["model"]["sidecar_hidden"],
        sfi_idx=config["model"]["sfi_idx"],
        kp_penalty_idx=config["model"]["kp_penalty_idx"],
        gate_init_bias=config["model"].get("gate_init_bias", -math.log(2.0)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {config['model']['architecture']} ({param_count:,} params)")
    print(f"  Checkpoint Pearson: {checkpoint.get('val_pearson', 'N/A')}")
    print()

    # Build per-band WSPR norm constants: {band_id: (mean, std)}
    norm_constants = {}
    for band_str, sources in config["norm_constants_per_band"].items():
        band_id = int(band_str)
        norm_constants[band_id] = (sources["wspr"]["mean"], sources["wspr"]["std"])

    # Connect to ClickHouse
    ch_host = config["clickhouse"]["host"]
    ch_port = config["clickhouse"]["port"]
    print(f"  ClickHouse: {ch_host}:{ch_port}")
    client = clickhouse_connect.get_client(host=ch_host, port=ch_port)

    # Load mode thresholds from validation.mode_thresholds
    threshold_rows = client.query(
        "SELECT mode, threshold_db FROM validation.mode_thresholds"
    ).result_rows
    mode_thresholds = {row[0]: float(row[1]) for row in threshold_rows}
    print(f"  Mode thresholds: {len(mode_thresholds)} modes loaded")
    print()

    # Generate run metadata
    run_id = uuid.uuid4()
    run_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
    limit = args.limit if args.limit > 0 else None

    # Load source data
    print(f"Loading {args.source} data...")
    if args.source == 'rbn':
        features, meta = load_rbn(client, config, limit)
    elif args.source == 'pskr':
        features, meta = load_pskr(client, config, limit)
    elif args.source == 'contest':
        features, meta = load_contest(client, config, limit)

    if features is None:
        print("  No data loaded!")
        client.close()
        return

    # Score and insert
    recall = score_and_insert(
        client, features, meta, model, config, device,
        norm_constants, mode_thresholds, run_id, run_timestamp,
        model_version, args.source, args.profile, advantage_db,
    )

    client.close()

    print("=" * 70)
    print(f"  Run ID:   {run_id}")
    print(f"  Source:   {args.source}")
    print(f"  Profile:  {args.profile} (+{advantage_db:.1f} dB)")
    print(f"  Scored:   {len(features):,} paths")
    print(f"  Recall:   {recall:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
