# IONIS V13

**Date:** 2026-02-08
**Status:** Complete (superseded by V14)
**Recall:** 85.34%

## Summary

V13 combined WSPR signatures with RBN DXpedition synthesis:
- WSPR signatures (93.4M) — V1 filter (had balloon contamination)
- RBN DXpedition (91K × 50x) — 152 rare DXCC entities

Note: V13 used the broken balloon filter. V14+ used the corrected V2 filter.

## Results

| Metric | Value |
|--------|-------|
| Overall Recall | 85.34% |
| Pearson | +0.2865 |
| RMSE | 0.60σ |

## Checklist

- [x] Train model: `python train_v13.py`
- [x] Validate Step I: `python validate_v13.py`
- [x] Verify physics: `python verify_v13.py`
- [x] Sensitivity tests: `python test_v13.py`
- [x] Oracle test suite: `python oracle_v13.py --test`
- [x] Write REPORT_v13.md

## Files

| File | Purpose |
|------|---------|
| `train_v13.py` | Training script |
| `validate_v13.py` | Step I recall validation |
| `verify_v13.py` | Physics verification |
| `test_v13.py` | Sensitivity analysis |
| `oracle_v13.py` | Production oracle |
| `ionis_v13.pth` | Model checkpoint |
| `REPORT_v13.md` | Final report |
