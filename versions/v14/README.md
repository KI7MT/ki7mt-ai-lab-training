# IONIS V14

**Date:** 2026-02-09
**Status:** Complete (superseded by V15)
**Recall:** 76.00%

## Summary

V14 tested WSPR-only training with corrected balloon filter:
- WSPR signatures (93.3M) — V2 terrestrial filter

This proved that WSPR alone matches but cannot exceed VOACAP.
DXpedition data is required for rare path coverage.

## Results

| Metric | Value |
|--------|-------|
| Overall Recall | 76.00% |
| Pearson | +0.296 |

## Checklist

- [x] Train model: `python train_v14.py`
- [x] Validate Step I: `python validate_v14.py`

## Files

| File | Purpose |
|------|---------|
| `train_v14.py` | Training script |
| `validate_v14.py` | Step I recall validation |
| `ionis_v14.pth` | Model checkpoint |

## Key Learning

WSPR-only training (V14) achieved 76% recall — essentially matching VOACAP (75.82%).
The +10 pp improvement in V15 came from adding RBN DXpedition data.
