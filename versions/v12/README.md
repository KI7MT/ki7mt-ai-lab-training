# IONIS V12

**Date:** 2026-02-08
**Status:** Complete (superseded by V13)
**Recall:** Not measured (pre-Step I)

## Summary

V12 introduced aggregated signature training:
- WSPR signatures (20M sample from 93.4M) — V1 filter (had balloon contamination)
- First model trained on aggregated buckets instead of raw spots

Key innovation: Training on median SNR per path×band×hour×month bucket reduces noise.

## Results

| Metric | Value |
|--------|-------|
| RMSE | 2.03 dB |
| Pearson | +0.3153 |
| Physics Score | 74.2/100 |

## Checklist

- [x] Train model: `python train_v12.py`
- [x] Validate physics: `python verify_v12.py`
- [x] Sensitivity tests: `python test_v12.py`
- [x] Oracle test suite: `python oracle_v12.py --test`
- [x] Write REPORT_v12.md

## Files

| File | Purpose |
|------|---------|
| `train_v12.py` | Training script |
| `verify_v12.py` | Physics verification |
| `test_v12.py` | Sensitivity analysis |
| `oracle_v12.py` | Production oracle (35 tests) |
| `ionis_v12.pth` | Model checkpoint |
| `REPORT_v12.md` | Final report |

## Data Sources

| Source | Table | Rows |
|--------|-------|------|
| WSPR | `wspr.signatures_v1` | 93.4M (20M sampled) |
