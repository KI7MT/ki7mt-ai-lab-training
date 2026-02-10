# KI7MT AI Lab — Training

PyTorch training and validation for the **IONIS** (Ionospheric Neural Inference System) propagation model.

## Current Model

**IONIS V16 Contest** — Curriculum Learning with Contest Anchoring
- 203,573 parameters (IonisV12Gate architecture)
- Trained on WSPR (floor) + RBN DXpedition (rare paths) + Contest (ceiling)
- Curriculum learning: -28 dB floor first, then +10 dB ceiling

| Metric | Value |
|--------|-------|
| Overall Recall | **96.38%** (+20.56 pp vs VOACAP) |
| SSB Recall | **98.40%** |
| Pearson | +0.4873 |
| RMSE | 0.860σ |
| Oracle Tests | 35/35 PASS |

## Repository Structure

```
ki7mt-ai-lab-training/
├── versions/           # Self-contained version folders
│   ├── v12/           # WSPR signatures (baseline)
│   ├── v13/           # + RBN DXpedition
│   ├── v14/           # WSPR-only A/B test
│   ├── v15/           # Clean filter + DXpedition
│   ├── v16/           # + Contest anchoring (current)
│   └── archive/       # v10, v11, and experiments
├── scripts/           # Shared utilities
│   ├── signature_search.py   # kNN search over 93.4M signatures
│   ├── predict.py            # Generic prediction interface
│   ├── voacap_batch_runner.py # VOACAP comparison harness
│   └── coverage_heatmap.py   # Grid coverage visualization
├── models/            # Canonical checkpoints (ionis_vxx.pth)
├── GOAL.md           # Project vision
└── README.md         # This file
```

Each version folder is self-contained:
```
versions/v16/
├── train_v16.py       # Training script
├── validate_v16.py    # Step I recall validation
├── verify_v16.py      # Physics verification
├── test_v16.py        # Sensitivity analysis
├── oracle_v16.py      # Production oracle (35 tests)
├── ionis_v16.pth      # Model checkpoint
├── README.md          # Checklist and summary
└── REPORT_v16.md      # Final report
```

## Architecture

```
IonisV12Gate (203,573 params)
├── Trunk: 11 geography/time features → 512 → 256
├── Base Head: 256 → 128 → 1 (baseline SNR)
├── Sun Scaler Head: 256 → 64 → 1 (geographic gate)
├── Storm Scaler Head: 256 → 64 → 1 (geographic gate)
├── Sun Sidecar: MonotonicMLP (SFI → SNR boost)
└── Storm Sidecar: MonotonicMLP (Kp → SNR penalty)
```

**Key innovation:** Gated monotonic sidecars enforce physics constraints (SFI+, Kp-) while allowing geographic modulation of sensitivity.

## Model Evolution

| Version | Innovation | Recall |
|---------|------------|--------|
| V12 | Aggregated signatures | — |
| V13 | + RBN DXpedition | 85.34% |
| V14 | WSPR-only (A/B test) | 76.00% |
| V15 | Clean balloon filter | 86.89% |
| **V16** | **+ Contest anchoring** | **96.38%** |

## Quick Start

### Prerequisites
- Python 3.10+ with PyTorch 2.x
- ClickHouse access (10.60.1.1 via DAC or 192.168.1.90 LAN)
- Required tables: `wspr.signatures_v2_terrestrial`, `rbn.dxpedition_signatures`, `contest.signatures`

### Reproduce V16
```bash
cd versions/v16
python train_v16.py          # ~3.5 hours on M3 Ultra
python validate_v16.py       # Step I recall (96.38%)
python verify_v16.py         # Physics verification
python test_v16.py           # Sensitivity analysis
python oracle_v16.py --test  # 35/35 tests
```

## Data Sources

| Source | Table | Rows | Purpose |
|--------|-------|------|---------|
| WSPR | `wspr.signatures_v2_terrestrial` | 93.3M | Floor (-28 dB) |
| RBN | `rbn.dxpedition_signatures` | 91K | Rare paths (152 DXCC) |
| Contest | `contest.signatures` | 6.34M | Ceiling (+10 dB SSB) |

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [ki7mt-ai-lab-docs](https://github.com/KI7MT/ki7mt-ai-lab-docs) | Documentation site |
| [ki7mt-ai-lab-apps](https://github.com/KI7MT/ki7mt-ai-lab-apps) | Go data ingesters |
| [ki7mt-ai-lab-core](https://github.com/KI7MT/ki7mt-ai-lab-core) | DDL schemas, SQL |
| [ki7mt-ai-lab-cuda](https://github.com/KI7MT/ki7mt-ai-lab-cuda) | CUDA signature engine |

## License

GPLv3 — See [COPYING](COPYING) for details.

## Author

Greg Beam, KI7MT

---

*Training infrastructure for HF propagation prediction based on real-world observations.*
