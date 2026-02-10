# IONIS V15

**Date:** 2026-02-09
**Status:** Complete (superseded by V16)
**Recall:** 86.89% (+11.07 pp vs VOACAP)

## Summary

V15 combines clean WSPR signatures with RBN DXpedition data:
- WSPR signatures (93.3M) — balloon-filtered V2
- RBN DXpedition (91K × 50x) — 152 rare DXCC entities

## Results

| Metric | Value |
|--------|-------|
| Overall Recall | 86.89% |
| SSB Recall | 81.01% |
| Pearson | +0.2828 |
| RMSE | 0.601σ |

## Checklist

- [x] Train model: `python train_v15.py`
- [x] Validate Step I: `python validate_v15.py`
- [ ] Verify physics
- [ ] Sensitivity tests
- [ ] Oracle test suite
- [x] Update docs

## Files

| File | Purpose |
|------|---------|
| `train_v15.py` | Training script |
| `validate_v15.py` | Step I recall validation |
| `ionis_v15.pth` | Model checkpoint |

## Data Sources

| Source | Table | Rows |
|--------|-------|------|
| WSPR | `wspr.signatures_v2_terrestrial` | 93.3M |
| RBN | `rbn.dxpedition_signatures` | 91K |
