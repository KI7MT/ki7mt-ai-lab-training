# IONIS Goal Document

## The Vision

Build a propagation prediction tool that improves on the ITS/NTIA reference
model (VOACAP) — not occasionally, not marginally, but consistently and
substantially. No lab physics. No theoretical ionospheric models. Built
entirely on real-world measurements.

The approach is inspired by Critical Dimension Scatterometry (CDS) in
semiconductor metrology. In CDS, you measure a diffraction signature from a
wafer, match it against a library of known signatures, and reconstruct the
physical structure you can't directly see.

We do the same thing with the ionosphere:

1. **Radio spots are the diffraction patterns** — observed signal behavior
   under specific conditions
2. **The aggregated signatures are the library** — 93.4M buckets encoding
   path geometry, solar conditions, and temporal variables
3. **Given current conditions**, match against the library to find what
   historically happened under similar conditions
4. **The matching signatures tell you what propagation will look like**

This is an **inverse problem**: we have the answers (signal observations)
and we reconstruct the question (what is the ionosphere doing right now?).

## Two Complementary Pieces

### The Signature Library (pattern matching)

93.4 million aggregated signatures in `wspr.signatures_v1`, compressed from
10.8B raw WSPR spots. Query by path, band, hour, month, and solar conditions.
Returns median SNR and reliability from historical observations.

**Step G complete**: kNN search layer built. 50-94 ms query latency, 7/7 physics
tests pass. "FN31 to JO21, 20m, 14 UTC, June" returns -21.0 dB median from
847 matching signatures.

### The Neural Model (learned physics)

IONIS V13 Combined (203K params) learned the shape of the physics from data —
which directions things move, how variables relate. SFI up = better. Storms = worse.
Polar paths behave differently from equatorial. It knows the rules.

**Multi-source hybrid**: V13 combines WSPR weak-signal data with RBN DXpedition
high-power data, covering 152 rare DXCC entities that WSPR alone cannot reach.

These two pieces are complementary. The model knows the rules. The signature
library has the receipts. Together, they solve the inverse problem.

## Current Status (Step I Complete)

**What's built and validated:**

| Component | Status |
|-----------|--------|
| WSPR pipeline | 10.8B spots in ClickHouse |
| RBN pipeline | 2.18B spots ingested |
| Contest logs | 195M QSOs from 15 contests |
| Signature library | 93.4M aggregated buckets |
| Signature search | kNN layer, 50-94 ms latency |
| Neural model | V13 Combined, 203K params |
| Ground truth validation | 1M contest paths validated |

**V13 Results:**

| Metric | Value |
|--------|-------|
| RMSE | 0.60σ (~4.0 dB) |
| Pearson | +0.2865 |
| Step I Recall | 85.34% |
| vs Reference | **+9.5 percentage points** |

**What's next (Step J):**
- Unified prediction interface combining neural model + signature search
- Single CLI: path + conditions → prediction + historical evidence
- Confidence weighting based on signature density

## The Goal (Z)

Given a path, band, time, and solar conditions — produce a propagation
prediction that consistently outperforms the reference model.

**Step I demonstrated this**: On 1M validated contest paths, IONIS V13 shows
85.34% band-open recall vs VOACAP's 75.82% — a +9.5 percentage point improvement.

**Remaining criteria for Step Z:**
- [ ] Pearson r (IONIS vs actual) > Pearson r (reference vs actual)
- [ ] RMSE (IONIS) < RMSE (reference)
- [x] Band-open accuracy (IONIS) > Band-open accuracy (reference) ✓
- [ ] IONIS shows improvement on > 90% of test paths
- [x] Results are reproducible and documented ✓

## Our Advantages

The ITS/NTIA reference model (VOACAP) represents foundational ionospheric
science from the 1960s — monthly medians, Chapman layers, carefully calibrated
theoretical models. It remains the standard for good reason.

IONIS builds on that foundation with three complementary advantages:

1. **Real-world signatures** — not theory, 13.2B actual observations (WSPR + RBN + contests)
2. **Ground truth validation** — 195M contest QSOs proving what worked
3. **Continuous learning** — new WSPR spots flowing in every two minutes,
   24x7x365, forever. The system gets smarter every day.

## Data Sources

| Source | Volume | Role |
|--------|--------|------|
| **WSPR** | 10.8B spots | Signal floor baseline |
| **RBN** | 2.18B spots | High-SNR transitions, DXpeditions |
| **Contest Logs** | 195M QSOs | Ground truth validation |
| **Solar Indices** | 76K rows | Gated physics input (SFI, Kp) |
| **Signatures** | 93.4M buckets | Aggregated path×band×hour×month patterns |

## Model Lineage

```
V2 → V6 (monotonic) → V7 (lobotomy) → V8 (sidecar) →
V9 (dual mono) → V10 (final) → V11 (gates) → V12 (signatures) → V13 (combined)
```

Each version added architectural improvements while preserving physics constraints.
V13 is the first multi-source hybrid, bridging WSPR and RBN data.

## Known Risks (Updated)

- ~~Signature search at 4.4B vectors needs efficient indexing~~ → Solved: 93.4M aggregated, 50-94 ms
- WSPR station coverage is uneven — some paths have thin history → Mitigated by RBN DXpedition data
- ~~The two pieces aren't connected~~ → Step J will unify them
- Steps forward might break things behind us — tests at each step prove we haven't regressed
- The path from D to Z is not a straight line. We adjust as we learn.

## Roadmap Status

| Step | Description | Status |
|------|-------------|--------|
| D | Infrastructure complete | ✓ |
| E | Golden Burn (V12 signatures) | ✓ |
| F | Aggregated Signatures | ✓ |
| G | Signature Search Layer | ✓ |
| H | Contest Log Ingest | ✓ |
| I | Ground Truth Validation | ✓ |
| J | Unified Prediction Interface | **Next** |
| Z | Outperform reference on >90% paths | Target |

---

*Status: Step I Complete — V13 validated at +9.5 pp vs reference model*
*Date: 2026-02-09*
