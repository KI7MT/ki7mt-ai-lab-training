# KI7MT AI Lab — Training

PyTorch training and validation for the **IONIS** (Ionospheric Neural Inference System) propagation model.

## Current Model

**IONIS V13 Combined** — Multi-Source Hybrid
- 203,573 parameters (IonisV12Gate architecture)
- Trained on 20M WSPR + 91K RBN DXpedition signatures
- Per-source per-band Z-score normalization
- 152 rare DXCC entities covered

| Metric | Value |
|--------|-------|
| RMSE | 0.60σ (~4.0 dB) |
| Pearson | +0.2865 |
| SFI benefit (70→200) | +5.2 dB |
| Kp storm cost (0→9) | +10.4 dB |
| Step I Recall | 85.34% (+9.5 pp vs reference) |

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

## Scripts

### Training
| Script | Purpose |
|--------|---------|
| `train_v13_combined.py` | V13: WSPR + RBN DXpedition (production) |
| `train_v12_signatures.py` | V12: WSPR signatures only |
| `train_v13_1_combined.py` | V13.1: 25x upsampling experiment |

### Validation
| Script | Purpose |
|--------|---------|
| `test_v13_combined.py` | V13 sensitivity analysis |
| `verify_v13_combined.py` | V13 physics verification (4/4 pass) |
| `validate_v13_step_i.py` | Step I: 1M contest path validation |
| `signature_search.py` | kNN search over 93.4M signatures |

### Legacy
| Script | Purpose |
|--------|---------|
| `oracle_v12.py` | V12 inference + 35-test physics suite |
| `test_v12_signatures.py` | V12 sensitivity analysis |
| `verify_v12_signatures.py` | V12 physics verification |

## Models

| Checkpoint | Description |
|------------|-------------|
| `ionis_v13_combined.pth` | **Production** — V13 Multi-Source Hybrid |
| `ionis_v13_1_combined.pth` | Experiment — 25x upsampling (78.75% recall) |
| `ionis_v12_signatures.pth` | V12 — WSPR signatures only |

## Usage

### Prerequisites
- Python 3.10+
- PyTorch 2.x with MPS (Apple Silicon) or CUDA
- ClickHouse access (10.60.1.1 via DAC or 192.168.1.90 LAN)

### Training V13
```bash
cd ki7mt-ai-lab-training
python scripts/train_v13_combined.py
```

### Running Validation
```bash
# Physics verification
python scripts/verify_v13_combined.py

# Sensitivity analysis
python scripts/test_v13_combined.py

# Step I comparison (requires ClickHouse)
python scripts/validate_v13_step_i.py
```

## Data Requirements

Training pulls directly from ClickHouse:
- `wspr.signatures_v1` — 93.4M aggregated WSPR signatures
- `rbn.dxpedition_signatures` — 91K RBN DXpedition signatures
- `solar.bronze` — Solar indices (SFI, Kp, SSN)

## Model Lineage

```
V2 → V6 (monotonic) → V7 (lobotomy) → V8 (sidecar) →
V9 (dual mono) → V10 (final) → V11 (gates) → V12 (signatures) → V13 (combined)
```

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
