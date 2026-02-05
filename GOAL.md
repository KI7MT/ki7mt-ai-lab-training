# IONIS Goal Document

## The Vision

Build a propagation prediction tool that beats VOACAP — not sometimes, not
marginally, but every time and by a substantial margin. No lab physics.
No theoretical ionospheric models. Built entirely on real-world measurements:
10.8 billion WSPR spots and solar data.

The approach is inspired by Critical Dimension Scatterometry (CDS) in
semiconductor metrology. In CDS, you measure a diffraction signature from a
wafer, match it against a library of known signatures, and reconstruct the
physical structure you can't directly see.

We do the same thing with the ionosphere:

1. **WSPR spots are the diffraction patterns** — observed signal behavior
   under specific conditions
2. **The CUDA embeddings are the signatures** — float4 vectors encoding
   path geometry, solar conditions, and temporal variables
3. **The 4.4 billion vectors in model_features are the signature library**
4. **Given current conditions**, match against the library to find what
   historically happened under similar conditions
5. **The matching signatures tell you what propagation will look like**

This is an **inverse problem**: we have the answers (signal observations)
and we reconstruct the question (what is the ionosphere doing right now?).

## Two Complementary Pieces

### The Signature Library (pattern matching)

The CUDA signature engine already projects WSPR spots into a high-dimensional
vector space. 4.4 billion signatures sitting in ClickHouse. When you have
today's conditions, you search the library for the closest historical matches
and report what happened. No model needed — just "the last 5,000 times
conditions looked like this, here's what 20m to Europe actually did."

This is the core of the thesis. The receipts.

### The Neural Model (learned physics)

IONIS V11 (203K params) learned the shape of the physics from data — which
directions things move, how variables relate. SFI up = better. Storms = worse.
Polar paths behave differently from equatorial. It knows the rules.

The model can serve as a fast first approximation, or help narrow the search
space before signature matching (like physics-based models do in CDS).

These two pieces are complementary. The model knows the rules. The signature
library has the receipts. Together, they solve the inverse problem.

## Where We Are (D)

**What's built and working:**
- Data pipeline: 10.8B WSPR spots in ClickHouse, solar backfill (2000-2026),
  Go ingesters at 24M rows/sec. Solid and complete.
- CUDA signature engine: 4.4B float4 embeddings in wspr.model_features.
  The signature library exists.
- Neural model: IONIS V11, correct physics directions, monotonic sidecars,
  geographic gates. The rules are learned.
- Infrastructure: M3 Ultra (training), 9975WX (data/CUDA), DAC link,
  ClickHouse on dedicated NVMe. The sovereign stack is real.

**What's not working yet:**
- The neural model alone has Pearson +0.24 (explains 6% of variance).
  Not useful as a standalone prediction tool.
- The signature library exists but has no query/matching layer on top.
  The vectors are there but nobody's searching them.
- The two pieces aren't connected. No system takes current conditions,
  searches signatures, and produces a prediction.

## The Goal (Z)

Given a path, band, time, and solar conditions — produce a propagation
prediction that beats VOACAP. Every time. By a substantial margin.

**What "beating VOACAP" looks like:**

Take real paths on real days. Ask VOACAP what it predicts. Ask IONIS what it
predicts. Compare both to what actually happened (the WSPR spots we recorded).
Whoever is closer, more often, wins.

Metrics:
- Pearson correlation with actual median SNR per path/band/hour
- RMSE against actual observations
- Band-open reliability: "will this band be open?" (yes/no accuracy)
- Temporal responsiveness: can it react to today's solar conditions,
  not last month's median?

## Why We Win

VOACAP is locked into 1960s ionosonde theory — monthly medians, Chapman
layers, static models. It cannot adapt. That's not a software problem — the
theory is the ceiling.

We have three advantages VOACAP will never have:

1. **Real-world signatures** — not theory, 10.8B actual observations
2. **Ground truth validation** — millions of contest QSOs proving what worked
3. **Continuous learning** — new WSPR spots flowing in every two minutes,
   24x7x365, forever. The system gets smarter every day while VOACAP stays
   frozen

## Untapped Data: Contest Logs

WSPR tells you the signal floor. Contest logs tell you **what the ionosphere
actually delivered** — proof that two stations communicated on a specific band,
at a specific time, between specific locations.

**Cabrillo logs from CQ contests** (CQ WW, CQ WPX, etc.) are publicly
available, free, going back decades. Each QSO line contains: frequency, mode,
timestamp, both callsigns, and exchange. Map callsigns to grid squares and
you have millions of confirmed contacts with path, band, time, and date.

During major contest weekends, thousands of stations are on every band
simultaneously — while WSPR beacons are running in the background. This gives
us the signature (WSPR) and ground truth (contest QSOs) at the same time.

Additional sources:
- **Reverse Beacon Network (RBN)**: automated CW/RTTY skimmer spots, free API
- **PSK Reporter**: FT8/FT4 spots, massive volume, free
- **Club Log**: millions of confirmed DX contacts with timestamps

Contest logs answer the question WSPR can't: **"Was the band actually usable?"**
Not just detectable at -28 dB, but usable for real communication. This turns
the prediction from "what's the SNR?" into what operators actually care about:
"Can I make a contact on 20m to Europe right now? On what mode?"

## Known Risks

- Signature search at 4.4B vectors needs efficient indexing (ANN/FAISS)
- WSPR station coverage is uneven — some paths have thin history
- We might need more features in the signatures (geomagnetic latitude, MUF)
- Steps forward might break things behind us — tests at each step must
  prove we haven't gone backwards
- The path from D to Z is not a straight line. We might hit walls and
  need to adjust. That's expected.

## What Comes Next

This document goes to Gemini for a physics-informed roadmap: the concrete
steps from D (correct physics, not useful yet) to Z (beats VOACAP decisively),
in plain language, with a pass/fail test at each step.

The steps might change as we learn. The goal doesn't.

---

*Status: Step D — infrastructure complete, physics correct, not yet useful.*
*Date: 2026-02-04*
