# IONIS V20 — Golden Master Replication

**V20 = V16 Architecture + V16 Data Recipe + V19 Code Structure**

## Purpose

V20 is a strict replication of V16 using the clean `train_common.py` module.
Before attempting any improvements (RBN, PSKR, etc.), we must prove we can
reproduce V16's results in the new codebase.

## Success Criteria

| Metric | Target | V16 Reference |
|--------|--------|---------------|
| Pearson | > +0.48 | +0.4873 |
| Kp sidecar | > +3.0σ | +3.445σ |
| SFI sidecar | > +0.4σ | +0.478σ |

## The V16 Physics Laws

These constraints are **non-negotiable**. They are what killed every V19 variant when missing.

1. **Architecture**: IonisV12Gate — gates from trunk output (256-dim), not raw input
2. **Loss**: HuberLoss(delta=1.0) — robust to synthetic contest anchors
3. **Regularization**: Gate variance loss — forces context-dependent behavior
4. **Init**: Defibrillator — weights uniform(0.8-1.2), fc2.bias=-10.0, fc1.bias frozen
5. **Constraint**: Weight clamp [0.5, 2.0] after EVERY optimizer.step()
6. **Data**: V16 recipe — WSPR 20M + RBN DX 91K×50 + Contest 6.34M (NO RBN Full)

## Data Recipe

| Source | Volume | Role |
|--------|--------|------|
| WSPR signatures | 20M | Floor (-28 dB) |
| RBN DXpedition | 91K × 50 = 4.55M | Rare paths (152 DXCC) |
| Contest | 6.34M | Ceiling (SSB +10 dB) |
| RBN Full | **0** | Excluded (V16 law) |

## Files

- `train_v20.py` — Training script
- `ionis_v20.pth` — Checkpoint (after training)

## Running

```bash
cd /Users/gbeam/workspace/ki7mt-ai-lab/ki7mt-ai-lab-training
python versions/v20/train_v20.py
```

## What Happens Next

**If V20 PASSES**: Lock the code, proceed to production (ONNX/API).

**If V20 FAILS**: The refactored `train_common.py` has a bug. Debug before proceeding.

## History

- V17, V18, V19: All failed due to missing V16 constraints
- V19 used IonisModel (gates from raw input, LayerNorm/GELU, no clamping)
- Root cause identified: Sidecar death from missing weight clamp and defibrillator
- IonisModel purged from train_common.py on 2026-02-11
