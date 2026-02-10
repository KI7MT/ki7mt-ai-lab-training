# IONIS V17 — RBN Grid-Enriched

**Date:** 2026-02-10
**Status:** Training Complete, **Blocked** (normalization calibration issue)
**Decision:** V16 remains production

## Summary

V17 added RBN grid-enriched signatures (56.7M rows) to the training pool, creating a 4-source curriculum:
- Floor: WSPR (-28 dB measured)
- Middle: RBN (real machine-measured SNR, median +17 dB)
- Ceiling: Contest (+10/0 dB anchored)
- Rare: DXpedition (152 DXCC, 50x upsampled)

Training completed successfully with correct physics, but **normalization calibration issue** prevents production deployment.

## Training Results

| Metric | V16 | V17 | Delta |
|--------|-----|-----|-------|
| RMSE | 0.860σ | **0.825σ** | -0.035σ (better) |
| Pearson | +0.4873 | +0.3848 | -0.103 (worse) |
| SFI benefit | +0.48σ | +0.48σ | Same |
| Kp storm cost | +3.45σ | +2.54σ | -0.9σ (weaker) |
| Parameters | 203,573 | 203,573 | Same |
| Epochs | 100 | 100 | Same |
| Training time | ~6.5 hrs | ~6.6 hrs | Similar |

## Validation Results

| Test | V16 | V17 | Notes |
|------|-----|-----|-------|
| verify_vXX.py | 4/4 PASS | 4/4 PASS | Physics correct |
| oracle_vXX.py | 35/35 PASS | 35/35 PASS | All tests pass |
| validate_vXX.py | 96.38% | 100%* | *Inflated |
| validate_pskr.py | 84.14% | 99.99%* | *Inflated |

*V17 percentages are inflated due to normalization mismatch — see issue below.

## The Calibration Issue

**Root Cause:** V17 uses global Z-normalization across 4 sources with very different SNR scales:

| Source | SNR Range | V17 Treatment |
|--------|-----------|---------------|
| WSPR | -28 to -10 dB | Z-normalized to ~-1σ |
| RBN | +8 to +30 dB | Z-normalized to ~+1σ |
| Contest | 0 to +10 dB | Z-normalized to ~0σ |
| Mixed | All | mean=0.44σ, std=1.50 |

**Problem:** The checkpoint stores `snr_mean=0.44, snr_std=1.50` which are the Z-normalization parameters of the mixed training data, **not** the denormalization parameters to convert back to real dB.

When we denormalize V17 predictions:
```
pred_db = pred_sigma * 1.50 + 0.44
```

A prediction of 0σ becomes +0.44 dB, which is well above the -20 dB FT8 threshold. Everything looks "open".

**V16 worked because** it used per-band WSPR normalization (mean ~-17 dB per band), so denormalized predictions landed in the right dB range.

## Possible Fixes

### Option 1: Per-Source Denormalization
Store per-source normalization constants in checkpoint:
```python
norm_constants = {
    'wspr': {band: (mean, std) for band in bands},
    'rbn': {band: (mean, std) for band in bands},
    'contest': {band: (mean, std) for band in bands},
}
```
Apply source-appropriate denormalization based on use case.

### Option 2: Predict in σ Units
Accept that V17 outputs relative predictions (σ from mixed-source mean), not absolute dB. Recalibrate thresholds for σ units.

### Option 3: Train with Absolute dB
Don't Z-normalize across sources. Keep SNR in raw dB, let the model learn the full -28 to +30 dB range.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `train_v17.py` | 4-source curriculum training | Complete |
| `ionis_v17.pth` | Checkpoint (blocked) | Saved |
| `oracle_v17.py` | 35-test physics suite | 35/35 PASS |
| `verify_v17.py` | Sidecar verification | 4/4 PASS |
| `validate_v17.py` | Step I recall | 100%* |
| `validate_pskr.py` | PSK Reporter acid test | 99.99%* |

## Training Details

```
Architecture: IonisV12Gate (unchanged from V12-V16)
Parameters: 203,573
Training rows: 51M (20M WSPR + 20M RBN + 4.55M DXpedition×50 + 6.34M Contest)
Epochs: 100
Optimizer: Adam (lr=1e-4)
Scheduler: OneCycleLR
Device: Mac Studio M3 Ultra (MPS)
Time: 6h 35m
```

## Conclusion

V17 training was successful:
- Physics is correct (SFI↑ = SNR↑, Kp↑ = SNR↓)
- All 35 oracle tests pass
- RMSE improved over V16

However, the 4-source global Z-normalization breaks dB calibration. V16 remains the production model until we resolve the denormalization strategy.

## Next Steps

1. Discuss denormalization strategy with team
2. Consider V17.1 with per-source normalization stored in checkpoint
3. Or proceed to V18 with architectural fixes (grey-line, storm sidecar)
