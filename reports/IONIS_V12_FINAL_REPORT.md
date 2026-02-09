# IONIS V12 Final — Model Performance & Physics Card

**Model**: IONIS V12 Signatures
**Architecture**: IonisV12Gate
**Training Date**: 2026-02-08
**Training Hardware**: Mac Studio M3 Ultra (MPS backend)
**Roadmap Step**: E — Golden Burn (Aggregated Signatures)

---

## What's New in V12

| Feature | V11 (IonisV11Gate) | V12 (IonisV12Gate) |
|---------|--------------------|--------------------|
| Training Data | 10M raw WSPR spots | 20M aggregated signatures |
| Data Source | `wspr.bronze` | `wspr.signatures_v1` (93.4M buckets) |
| Target | Raw SNR | Median SNR per bucket |
| RMSE | 2.48 dB | **2.03 dB** (-18%) |
| Pearson | +0.2376 | **+0.3153** (+32.7%) |
| SFI benefit | +0.96 dB | +0.79 dB |
| Kp storm cost | +2.84 dB | +1.92 dB |

**Key innovation**: Training on aggregated signatures (median SNR per path×band×hour×month bucket) reduces noise and improves correlation. The model learns stable propagation patterns instead of individual spot noise.

---

## Architecture: IonisV12Gate

Same architecture as V11 — the improvement comes from data, not structure.

```
Input (13 features)
    │
    ├── Trunk (11 features) ──► 512 ──► 256 (shared geography representation)
    │                                    │
    │                     ┌──────────────┼──────────────┐
    │                     │              │              │
    │               Base Head      Sun Scaler     Storm Scaler
    │              256→128→1       256→64→1        256→64→1
    │                  │              │              │
    │              base_snr      sun_logit     storm_logit
    │                  │              │              │
    │                  │          gate(x)        gate(x)
    │                  │         [0.5,2.0]      [0.5,2.0]
    │                  │              │              │
    ├── Sun Sidecar (sfi) ──► MonotonicMLP ──► × sun_gate ──►│
    │                                                         │
    └── Storm Sidecar (kp_penalty) ──► MonotonicMLP ──► × storm_gate ──►│
                                                                        │
                                                  Output = base + sun_contrib + storm_contrib
```

---

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Trunk (11→512→256) | 137,472 |
| Base Head (256→128→1) | 33,025 |
| Sun Scaler Head (256→64→1) | 16,513 |
| Storm Scaler Head (256→64→1) | 16,513 |
| Sun Sidecar (1→8→1) | 25 |
| Storm Sidecar (1→8→1) | 25 |
| **Total** | **203,573** |

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | 20,000,000 aggregated signatures |
| Source Table | `wspr.signatures_v1` |
| Date Range | 2008-03-11 to 2026-02-01 |
| Batch Size | 65,536 |
| Epochs | 100 |
| Trunk + Base Head LR | 1e-5 |
| Scaler Heads LR | 5e-5 |
| Sidecar LR | 1e-3 |
| Optimizer | AdamW with differential LR |
| Scheduler | CosineAnnealingLR |
| Loss | MSE |

---

## Final Performance Metrics

| Metric | V12 Value | V11 Value | Change |
|--------|-----------|-----------|--------|
| **RMSE** | **2.03 dB** | 2.48 dB | -18% |
| **Pearson** | **+0.3153** | +0.2376 | +32.7% |

---

## Physics Verification: ALL TESTS PASSED

### Test 1: Storm Sidecar (Kp)

| Condition | SNR |
|-----------|-----|
| Kp 0 (Quiet) | -17.23 dB |
| Kp 9 (Storm) | -19.15 dB |
| **Storm Cost** | **+1.92 dB** |

**PASS**: Signal dropped during storm (correct physics)

### Test 2: Sun Sidecar (SFI)

| Condition | SNR |
|-----------|-----|
| SFI 70 (Low) | -18.34 dB |
| SFI 200 (High) | -17.55 dB |
| **SFI Benefit** | **+0.79 dB** |

**PASS**: Signal improved with higher SFI (correct physics)

### Test 3: Gate Range

| Gate | Min | Max |
|------|-----|-----|
| Sun | 0.50 | 1.83 |
| Storm | 0.50 | 1.50 |

**PASS**: All gate values within [0.5, 2.0]

### Test 4: Decomposition Math

**PASS**: base + sun_gate × sun_raw + storm_gate × storm_raw = predicted (exact)

---

## Physics Test Grades

| Test | Description | Result | Grade |
|------|-------------|--------|-------|
| TST-201 | SFI 70→200 benefit | +2.1 dB | B |
| TST-202 | Kp 0→9 storm cost | +4.0 dB | A |
| TST-203 | D-layer absorption | +0.0 dB | C |
| TST-204 | Polar storm sensitivity | +2.5 dB | B |
| TST-205 | 10m SFI sensitivity | +2.0 dB | C |
| TST-206 | Grey line twilight | +0.2 dB | C |

**Overall Physics Score**: 74.2/100

---

## Aggregated Signatures

V12 trains on `wspr.signatures_v1` — 93.4M aggregated buckets compressed from 10.8B raw spots (115:1 compression).

**Bucket schema**:
- Group by: `tx_grid_4`, `rx_grid_4`, `band`, `hour`, `month`
- Output: `median_snr`, `spot_count`, `snr_std`, `reliability`
- Minimum: 5 spots per bucket (noise filter)

Training samples 20M rows from this table, learning stable propagation patterns instead of individual spot noise.

---

## Files

| File | Location |
|------|----------|
| Training Script | `scripts/train_v12_signatures.py` |
| Sensitivity Test | `scripts/test_v12_signatures.py` |
| Physics Verification | `scripts/verify_v12_signatures.py` |
| Oracle (35-test suite) | `scripts/oracle_v12.py` |
| Model Checkpoint | `models/ionis_v12_signatures.pth` |

---

## Reproducibility

V12 was validated for full reproducibility on 2026-02-08:
- Fresh bronze pipeline → same results
- Deterministic training with seeded splits
- Checkpoint includes all metadata for reproduction

---

*Generated: 2026-02-08*
*IONIS V12 — Ionospheric Neural Inference System*
*D-to-Z Roadmap: Step E Complete (Aggregated Signatures)*
