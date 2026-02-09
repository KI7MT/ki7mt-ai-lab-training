# IONIS V10 Final — Model Performance & Physics Card

**Model**: IONIS V2 Phase 10 Final (Geography Reintegration)
**Architecture**: IonisDualMono
**Training Date**: 2026-02-04
**Training Hardware**: Mac Studio M3 Ultra (MPS backend)

---

## Project Status

| Issue | Status |
|-------|--------|
| **Kp Inversion Problem** | **SOLVED** |

The model now correctly predicts that geomagnetic storms (high Kp) degrade HF signal quality, and that higher solar flux (SFI) improves ionospheric propagation. Physics constraints are enforced through monotonic sidecars with locked weights.

---

## Model Specifications

### Architecture: IonisDualMono

```
Input (13 features)
    │
    ├── DNN (11 features) ──► 512 ──► 256 ──► 128 ──► 1 (base SNR)
    │                                                    │
    ├── Sun Sidecar (1 feature: sfi) ──► MonotonicMLP ──►│ (sun boost)
    │                                                    │
    └── Storm Sidecar (1 feature: kp_penalty) ──► MonotonicMLP ──► (storm boost)
                                                         │
                                                    Output = Sum
```

### Parameter Count

| Component | Parameters | Role |
|-----------|------------|------|
| DNN (Deep Network) | 170,497 | Geography & temporal patterns |
| Sun Sidecar | 25 | SFI → SNR boost (monotonic increasing) |
| Storm Sidecar | 25 | kp_penalty → SNR boost (monotonic increasing) |
| **Total** | **170,547** | |

### Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | 10,000,000 WSPR spots |
| Date Range | 2020-01-01 to 2026-02-04 |
| Batch Size | 8,192 |
| Epochs | 100 |
| DNN Learning Rate | 1e-5 (slow student) |
| Sidecar Learning Rate | 1e-3 (maintain physics) |
| Weight Decay | 1e-5 |
| Optimizer | AdamW with differential LR |

---

## Final Performance Metrics

| Metric | Value |
|--------|-------|
| **RMSE** | 2.48 dB |
| **Pearson Correlation** | +0.2395 |
| **Validation Loss** | 6.1486 |
| Training SNR Mean | -16.6 dB |
| Training SNR Std | 8.7 dB |

---

## Physics Anchors

### Monotonicity Verification: PASSED

Both sidecars maintain correct physics directionality:

| Test | Condition | Result | Status |
|------|-----------|--------|--------|
| **Sun Test** | SFI 70 → 200 | **+0.48 dB** benefit | PASS |
| **Storm Test** | Kp 0 → 9 | **+1.12 dB** cost | PASS |

### Sidecar Output Values

| Sidecar | Input | Output |
|---------|-------|--------|
| Sun | SFI 70 (low) | -7.63 dB |
| Sun | SFI 200 (high) | -7.15 dB |
| Storm | Kp 0 (quiet) | -6.75 dB |
| Storm | Kp 5 (moderate) | -7.41 dB |
| Storm | Kp 9 (storm) | -7.87 dB |

**Physics Interpretation**:
- Higher SFI → More ionization → Better propagation → Higher SNR
- Higher Kp → Ionospheric disturbance → Absorption/fading → Lower SNR

---

## The Relief Valve Setting

The convergence was enabled by a carefully tuned sidecar configuration:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Weight Clamp Range** | 0.5 – 2.0 | Prevents collapse AND explosion |
| **fc1.bias** | Frozen | Maintains activation shape |
| **fc2.bias** | Learnable (-10.65) | Relief valve for calibration |
| **Initial fc2.bias** | -10.0 | Defibrillator jump-start |

### Final Sidecar Biases

| Sidecar | Final Bias |
|---------|------------|
| Sun | -10.65 |
| Storm | -10.65 |

The negative bias allows sidecars to output negative dB values while maintaining monotonic increasing behavior with respect to their inputs.

---

## Feature Role Table

### DNN Features (11) — Geography & Time

| Index | Feature | Normalization | Role |
|-------|---------|---------------|------|
| 0 | `distance` | km / 20,000 | Path length |
| 1 | `freq_log` | log10(Hz) / 8.0 | Operating band |
| 2 | `hour_sin` | sin(2π·hour/24) | Time of day |
| 3 | `hour_cos` | cos(2π·hour/24) | Time of day |
| 4 | `az_sin` | sin(2π·az/360) | Azimuth direction |
| 5 | `az_cos` | cos(2π·az/360) | Azimuth direction |
| 6 | `lat_diff` | |Δlat| / 180 | North-south span |
| 7 | `midpoint_lat` | (lat_tx + lat_rx) / 2 / 90 | Path latitude |
| 8 | `season_sin` | sin(2π·month/12) | Seasonal variation |
| 9 | `season_cos` | cos(2π·month/12) | Seasonal variation |
| 10 | `day_night_est` | cos(2π·local_hour/24) | Solar illumination |

### Sidecar Features (2) — Physics Constrained

| Index | Feature | Normalization | Sidecar | Constraint |
|-------|---------|---------------|---------|------------|
| 11 | `sfi` | SFI / 300 | Sun | Monotonic increasing |
| 12 | `kp_penalty` | 1 - Kp/9 | Storm | Monotonic increasing |

**Note**: The DNN receives **zero** direct solar/storm information (Nuclear Option / Starvation Protocol). All physics flows exclusively through the constrained sidecars.

---

## SFI × Kp Sensitivity Matrix

Reference path: FN31 → JO21 (5,900 km, 20m WSPR, SSN=100)

| SFI \ Kp | Kp=0 | Kp=2 | Kp=4 | Kp=6 | Kp=8 |
|----------|------|------|------|------|------|
| **SFI 80** | -21.2 | -21.5 | -21.7 | -22.0 | -22.2 |
| **SFI 120** | -21.1 | -21.3 | -21.6 | -21.8 | -22.1 |
| **SFI 150** | -20.9 | -21.2 | -21.5 | -21.7 | -22.0 |
| **SFI 200** | -20.7 | -21.0 | -21.3 | -21.5 | -21.8 |
| **SFI 250** | -20.6 | -20.8 | -21.1 | -21.3 | -21.6 |

**Reading the Matrix**:
- Move **down** (higher SFI): SNR improves (correct)
- Move **right** (higher Kp): SNR degrades (correct)
- Best conditions: High SFI, Low Kp (bottom-left)
- Worst conditions: Low SFI, High Kp (top-right)

---

## Additional Sensitivity Results

### Kp Sweep (SFI=150 fixed)

| Kp | SNR (dB) |
|----|----------|
| 0 | -20.9 |
| 1 | -21.1 |
| 2 | -21.2 |
| 3 | -21.3 |
| 4 | -21.5 |
| 5 | -21.6 |
| 6 | -21.7 |
| 7 | -21.8 |
| 8 | -22.0 |
| 9 | -22.1 |

**Storm cost (Kp 0→9): +1.1 dB**

### SFI Sweep (Kp=2 fixed)

| SFI | SNR (dB) |
|-----|----------|
| 60 | -21.5 |
| 80 | -21.5 |
| 100 | -21.4 |
| 120 | -21.3 |
| 150 | -21.2 |
| 180 | -21.1 |
| 200 | -21.0 |
| 250 | -20.8 |
| 300 | -20.6 |

**SFI benefit (60→300): +0.9 dB**

---

## Training Convergence

### Epoch Progression (Selected)

| Epoch | RMSE | Pearson | SFI+ | Kp9- |
|-------|------|---------|------|------|
| 1 | 2.51 | +0.106 | +0.6 | +1.5 |
| 10 | 2.49 | +0.197 | +0.5 | +1.5 |
| 25 | 2.48 | +0.226 | +0.5 | +1.1 |
| 50 | 2.48 | +0.235 | +0.5 | +1.1 |
| 75 | 2.48 | +0.238 | +0.5 | +1.1 |
| 100 | 2.48 | +0.240 | +0.5 | +1.1 |

Physics constraints (SFI+ and Kp9-) remained positive throughout training, confirming the monotonic sidecars maintained correct behavior while the DNN learned geography.

---

## Hardware Performance (Turbo Loader)

| Metric | Before Optimization | After Optimization |
|--------|--------------------|--------------------|
| GPU Usage | 22% @ 958 MHz | 88% @ 1238 MHz |
| P-CPU Usage | 49% @ 3622 MHz | 79% @ 3591 MHz |
| Power Draw | 15.55W | 56.47W |
| Epoch Time | ~25s | ~6-8s |

**Turbo Loader Settings**:
- `num_workers=12`
- `prefetch_factor=4`
- `persistent_workers=True`

---

## Files

| File | Location |
|------|----------|
| Training Script | `scripts/train_v10_final.py` |
| Sensitivity Test | `scripts/test_v10_final.py` |
| Physics Verification | `scripts/verify_v10_final.py` |
| Model Checkpoint | `models/ionis_v10_final.pth` |
| Training Log | `logs/v10_final.log` |

---

## Acknowledgments

This model was developed through collaborative debugging between:
- **Claude Opus 4.5** (Anthropic) — Architecture implementation, code execution
- **Gemini** (Google) — Physics diagnosis, constraint design guidance

The Kp Inversion Problem was identified as survivorship bias in WSPR data (only strong signals decoded during storms), solved via Physics-Informed Machine Learning (PIML) with dual monotonic sidecars.

---

*Generated: 2026-02-04*
*IONIS V10 — Ionospheric Neural Inference System*
