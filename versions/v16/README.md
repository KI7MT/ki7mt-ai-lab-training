# IONIS V16

**Date:** 2026-02-11
**Status:** Complete
**Recall:** 96.38% (+20.56 pp vs VOACAP)

## Summary

V16 adds contest log anchoring to teach the model the "ceiling" of propagation:
- WSPR signatures (93.3M) — floor physics at -28 dB
- RBN DXpedition (91K × 50x) — rare path coverage
- Contest anchors (6.34M) — ceiling proof (+10 dB SSB, 0 dB RTTY)

## Results

| Metric | Value |
|--------|-------|
| Overall Recall | 96.38% |
| SSB Recall | 98.40% |
| Pearson | +0.4873 |
| RMSE | 0.860σ |
| Oracle Tests | 35/35 PASS |

## Checklist

- [x] Train model: `python train_v16.py`
- [x] Validate Step I: `python validate_v16.py`
- [x] Verify physics: `python verify_v16.py`
- [x] Sensitivity tests: `python test_v16.py`
- [x] Oracle test suite: `python oracle_v16.py --test`
- [x] Update repo README.md
- [x] Update ki7mt-ai-lab-docs validation pages
- [x] Write REPORT_v16.md

## Files

| File | Purpose |
|------|---------|
| `train_v16.py` | Training script |
| `validate_v16.py` | Step I recall validation |
| `verify_v16.py` | Physics constraint verification |
| `test_v16.py` | Sensitivity analysis |
| `oracle_v16.py` | Production oracle (35 tests) |
| `ionis_v16.pth` | Model checkpoint |
| `REPORT_v16.md` | Final report |

## Reproduce

Prerequisites:
- ClickHouse with `wspr.signatures_v2_terrestrial`, `rbn.dxpedition_signatures`, `contest.signatures`
- Python environment with PyTorch, clickhouse-connect

```bash
cd versions/v16
python train_v16.py          # ~3.5 hours on M3 Ultra
python validate_v16.py       # ~2 minutes
python verify_v16.py         # ~30 seconds
python test_v16.py           # ~1 minute
python oracle_v16.py --test  # ~30 seconds
```

## Data Sources

| Source | Table | Rows | SNR |
|--------|-------|------|-----|
| WSPR | `wspr.signatures_v2_terrestrial` | 93.3M | Machine (-28 dB floor) |
| RBN | `rbn.dxpedition_signatures` | 91K | Machine |
| Contest | `contest.signatures` | 6.34M | Anchored (+10 SSB, 0 RTTY) |
