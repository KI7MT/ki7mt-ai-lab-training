# Post-V17 Roadmap — DRAFT

> For review by 9975WX and Gemini Pro. Created 2026-02-10 during V17 training.
>
> **Rev 2**: Incorporated 9975WX feedback on Kp sign convention, VOACAP, solar indices, grey-line, NVIS.
>
> **Rev 5**: Gemini Pro architectural review — APPROVED. Added drift alerts, dual CW thresholds, terminator sidecar spec.

## Executive Summary

V16 achieved the D-to-Z goal: outperform VOACAP on standardized tests. V17 adds RBN grid enrichment. This document defines what comes next.

**Key Question**: Is IONIS ready for production, or do we need more training iterations?

---

## Phase 1: V17 Validation (Immediate)

### Success Criteria

| Metric | V16 Baseline | V17 Target | Fail Threshold |
|--------|--------------|------------|----------------|
| Pearson | +0.4873 | ≥ +0.48 | < +0.40 |
| RMSE | 0.860σ | ≤ 0.860σ | > 0.90σ |
| PSK Reporter Recall | 84.14% | ≥ 84% | < 80% |
| Physics (SFI+) | +0.48σ | +0.4 to +0.9σ | < 0 or > 2.0 |
| Physics (Kp9-) | -3.45σ | ≤ -2.0σ | ≤ 0 (inverted) |

**Note on Kp9- sign**: Negative value means storms COST SNR (correct physics). If coefficient hits zero or goes positive, model has lost physical understanding of geomagnetic damping — likely overfitting on high-SFI/high-Kp noise from solar peak. (Gemini)

### Validation Tests

1. **Oracle Test Suite** (35 tests) — Must pass all
2. **PSK Reporter Acid Test** — 100K independent spots with real solar
3. **Step I Recall** — Contest path prediction vs VOACAP
4. **Band-by-Band Analysis** — Check NVIS gap (160m target: > 50%)

### Decision Gate

| V17 Result | Action |
|------------|--------|
| All targets met | → Phase 2 (Production) |
| Pearson < +0.40 | → Investigate RBN data quality, consider rollback to V16 |
| NVIS still < 50% | → V18 with NVIS focus before production |
| Physics inverted | → Debug sidecars, do not deploy |

---

## Phase 2: Production Deployment

### 2A. Model Export Pipeline

| Step | Owner | Deliverable |
|------|-------|-------------|
| Freeze checkpoint | M3 | `ionis_v17_production.pth` |
| ONNX export | M3 | `ionis_v17.onnx` |
| Validate ONNX | M3 | Numerical equivalence test |
| Deploy to 9975WX | 9975 | ONNX Runtime + FastAPI |

### 2B. Prediction API

```
POST /predict
{
  "tx_grid": "FN31",
  "rx_grid": "JO22",
  "band": 107,
  "hour_utc": 14,
  "month": 2,
  "sfi": 150,
  "kp": 2
}
→ {
    "snr_db": 12.3,
    "modes": {
      "WSPR": {"viable": true, "margin_db": 40.3},
      "FT8":  {"viable": true, "margin_db": 32.3},
      "CW":   {
        "viable_human": true,    // -10 dB threshold
        "viable_machine": true,  // -18 dB threshold
        "margin_db": 22.3
      },
      "RTTY": {"viable": true, "margin_db": 17.3},
      "SSB":  {"viable": true, "margin_db": 7.3}
    }
  }
```

**Product Definition**: "Can I work this path right now, on my mode?" One forward pass, six answers.

**CW Dual Threshold (Gemini recommendation)**: Expose both human (-10 dB) and machine (-18 dB) thresholds. Amateur radio is moving toward "cyborg" state — operator listens while skimmer runs in background. API tells user: *"You won't hear this, but your software will decode it."*

Mode thresholds:
- WSPR: -28 dB, FT8: -20 dB, CW: -10 dB (human) / -18 dB (machine), RTTY: -5 dB, SSB: +5 dB

**Stack**: FastAPI + ONNX Runtime on 9975WX (RTX PRO 6000 for batch inference)

### 2C. VOACAP Comparison

**Confirmed**: Run `itshfbc` locally on 9975WX.
- License: GPL (free)
- Already proven: Step K achieved 2,020 circuits/sec
- No API needed — batch predict against same paths IONIS sees

### 2D. Live Validation Loop

| Component | Source | Destination | Cadence |
|-----------|--------|-------------|---------|
| PSK Reporter spots | MQTT collector | `pskr.bronze` | Real-time |
| Solar indices | `wspr.live_conditions` | API | 15-min |
| IONIS predictions | API | `validation.live_predictions` | Per-spot |
| VOACAP predictions | itshfbc | `validation.voacap_predictions` | Hourly batch |
| Comparison metrics | ClickHouse | `validation.daily_summary` | Daily |

**CRITICAL**: Live validation MUST use `wspr.live_conditions` (NOAA SWPC, 15-min updates), NOT `solar.bronze` (GFZ Potsdam, ~1 day lag). This matters during geomagnetic storms when Kp changes rapidly.

**Nowcasting Note (Gemini)**: Real-time solar data has 1-2 hour lag from NOAA. Using live WSPR/PSK spots as proxy for real-time ionospheric density is the only way to achieve true nowcasting.

**Drift Alert (Gemini recommendation)**: Z-Score Drift Alert — if mean error between IONIS and PSK firehose shifts by >2σ over rolling 4-hour window, trigger "Model Divergence" notification on 9975WX.

**Success Metric**: IONIS Pearson ≥ VOACAP Pearson on rolling 7-day window.

---

## Phase 3: V18 Decision Tree

### Option A: PSK Firehose (Real-Time Training)

**Trigger**: V17 production stable for 30 days, want "live state" not just climatology.

| Pros | Cons |
|------|------|
| 26M spots/day, fresh data | Training/inference drift risk |
| FT8 dominates (88%) — mode gap | Requires continuous retraining infra |
| Real solar conditions | Storage: ~10GB/day raw |

**Implementation**:
1. Ingest `pskr.bronze` → `pskr.signatures` (same schema as others)
2. Rolling 90-day training window
3. Weekly model refresh, A/B validation before promotion

### Option B: Power-Level Normalization

**Trigger**: Users ask "what SNR at my power level?"

| Data Source | TX Power | Notes |
|-------------|----------|-------|
| WSPR | 5W fixed | QRP baseline |
| Contest SSB | 100-1500W | High power ceiling |
| RBN | Unknown | Varied, no metadata |

**Challenge**: RBN doesn't carry power level. Would need to:
- Use contest logs only for power modeling
- Or infer power from SNR distribution (risky)

**Recommendation**: Defer unless user demand is clear.

### Option C: Grey-Line Specialization (HIGH VALUE — GEMINI ENDORSED)

**Trigger**: V17 still weak on sunrise/sunset predictions.

**Current**: `day_night_est = cos(2π × (hour + midpoint_lon/15) / 24)` — crude approximation.

**Gemini Architecture**: Don't need a new model — add a **Terminator Distance Sidecar**.
- Grey-line is the "Holy Grail" of HF operating
- Traditional models (VOACAP) struggle because they use hourly averages
- IONIS can leverage sub-second PSK Reporter timestamps to map exact SNR "surge" as terminator passes

**Recommended Features**:
- `dist_to_terminator_tx` — TX station distance to solar terminator (km, signed)
- `dist_to_terminator_rx` — RX station distance to solar terminator (km, signed)
- Model will latently learn the enhancement effect

**Implementation**:
1. Compute terminator position from solar declination + hour
2. Add 2 features per path endpoint
3. Retrain — low risk, high value

**9975WX Note**: Operators actively exploit grey-line on 160m/80m. This is a real opportunity.

### Option D: NVIS Gap Remediation

**Trigger**: 160m recall still < 50% after V17.

**Root Cause Hypotheses**:
1. WSPR 160m is sparse (fewer operators)
2. RBN 160m skimmers concentrated in NA/EU
3. NVIS physics different from skip (need separate model?)

**Remediation Options**:
1. Upsample 160m/80m in training (balance bands)
2. ~~Add explicit NVIS flag (distance < 500km + band < 80m)~~ — **PROBLEMATIC** (see below)
3. Separate NVIS model (V18-NVIS)
4. Grey-line enhancement (Option C) may help indirectly

**9975WX Note on Option 2**: The 500km distance filter is already applied in WSPR signatures to remove ground-wave. The problem is that NVIS paths ARE short distance but ionospheric — they get filtered out along with ground-wave. Distinguishing NVIS from ground-wave requires elevation angle data, which we don't have. This is a fundamental data limitation, not a modeling fix.

---

## Phase 4: Long-Term Vision

### The "Living Model" Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    IONIS Production                      │
├─────────────────────────────────────────────────────────┤
│  Stable Model (V17+)     │  Challenger Model (V18-dev)  │
│  - Serves predictions    │  - Trains on new data        │
│  - Validated daily       │  - A/B tested weekly         │
│  - Rollback available    │  - Promoted if better        │
└─────────────────────────────────────────────────────────┘
         ↑                            ↑
    Live PSK Reporter            Historical + Live
    + Real Solar                 Training Pool
```

### Milestone Targets

| Milestone | Target Date | Criteria |
|-----------|-------------|----------|
| V17 Validated | 2026-02-11 | Pass all validation tests |
| Production API | 2026-02-15 | FastAPI + ONNX serving |
| Live Validation | 2026-02-20 | 7-day rolling comparison |
| V18 Decision | 2026-03-01 | Based on production metrics |
| NVIS Remediation | 2026-03-15 | 160m recall > 60% |

---

## Open Questions

1. ~~**VOACAP Integration**: Do we run itshfbc locally or use an API? License?~~
   **ANSWERED**: Local `itshfbc` on 9975WX. GPL licensed. Already proven at 2,020 circuits/sec in Step K.

2. ~~**Alert System**: Who gets notified if IONIS drifts below VOACAP?~~
   **ANSWERED (Gemini)**: Z-Score Drift Alert. If mean error shifts >2σ over rolling 4-hour window → "Model Divergence" notification on 9975WX.

3. **User Interface**: CLI only? Web dashboard? Ham radio logger integration?

4. ~~**Public Release**: Open source the model? API access for others?~~
   **ANSWERED (9975WX)**: Model weights are small (~800KB). The value is in the data pipeline and training infrastructure, not the checkpoint. Open-sourcing the model is fine; the 13B-row dataset is the moat.

---

## Appendix: Training Data Pool Status

| Source | Rows | Used in V17 | Available |
|--------|------|-------------|-----------|
| WSPR signatures | 93.3M | 20M (21%) | 73.3M |
| RBN signatures | 56.7M | 20M (35%) | 36.7M |
| Contest signatures | 6.3M | 6.3M (100%) | — |
| DXpedition paths | 91K | 91K × 50 | — |
| PSK Reporter | ~26M/day | 0 | Growing |
| **Total Pool** | **156.4M** | **51M** | **110M+** |

**Note**: V17 uses 33% of available non-PSK data. Full-data V17.1 possible if needed.

---

*Draft by Claude-M3, 2026-02-10. Rev 5 incorporates 9975WX and Gemini Pro feedback.*

**Status: APPROVED by Gemini Pro (Chief Architect)**

*"V17 is the Engine of Record. The move to Phase 4 (Living Model Architecture) ensures that IONIS never becomes a static snapshot like the models that preceded it."*
