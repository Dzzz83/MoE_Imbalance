# Project Context — Expert Method for CIFAR-100-LT

## Overview

Long-tail class imbalance on CIFAR-100 with imbalance ratio 100 (IR=100).
Three separate ResNet-32 backbones are trained with **fundamentally different training paradigms** to produce diverse feature representations, then a lightweight dynamic router selects the best expert per test sample.

**Three experts:**
1. **LAL** — Logit-Adjusted Loss (classification-boundary paradigm, Menon et al. ICLR 2021)
2. **PaCo** — Parametric Contrastive Learning (contrastive representation paradigm, Cui et al. ICCV 2021)
3. **Mixup + CE** — Mixup-augmented Cross-Entropy (interpolation-based augmentation paradigm, Zhang et al. ICLR 2018)

**Key constraint:** IR=0.01, ResNet-32 backbone, single-GPU Kaggle workflow (T4 ×2 available).

---

## Data Pipeline — ⚠️ Critical Fix Applied

### Original (Flawed) Protocol — Now Deprecated
```
CIFAR-100 train (50K)
  ├── 5K balanced validation ← held out BEFORE LT (NON-STANDARD — cheating)
  └── 45K base pool
       └── LT subsampling → 9,754 training samples
```
The old protocol held out 50 samples/class **before** applying long-tail subsampling, creating a balanced validation set that doesn't exist in real long-tail scenarios. All experiments and results in `docs/experiments.md`, `docs/problem.md`, and `docs/final-report.md` were measured on this flawed split.

### Standard Protocol — Now Implemented
```
CIFAR-100 train (50K)
  └── LT subsampling (IR=100) → ~10,847 training samples
       ├── ~8,678 training (80%)
       └── ~2,169 validation (20%) ← also long-tailed

CIFAR-100 test (10K) ← final evaluation only
```

**Implementation status:** ✅ Complete.
- `utils/create_lt_split.py` — Creates proper LT train/val split from full 50K
- `data/cifar_lt.py` — Updated with `already_subsampled` and `use_test_set` parameters
- `data/processed/lt_train_indices.npy` — LT training indices (~8,678)
- `data/processed/lt_val_indices.npy` — LT validation indices (~2,169)
- `data/processed/lt_all_indices.npy` — Complete LT set (~10,847)
- Old indices (`base_train_indices.npy`, `balanced_val_indices.npy`) still present but deprecated

**New experts need to be trained on this proper split.** The old checkpoints (`LAL_best.pt`, `PaCo_best.pt`, `Mixup_best.pt`) were trained on the flawed 9,754-sample split and should not be used for final evaluation.

---

## Current Status

### ✅ Completed — Code Infrastructure

| Item | Details |
|------|---------|
| **Data pipeline fix** | `create_lt_split.py`, updated `cifar_lt.py` with proper split support |
| **Unified training script** | `scripts/train.py` — single entry point for all methods (`--method {lal, mixup, paco, ce, balanced_softmax}`) |
| **Unified evaluation script** | `scripts/evaluate.py` — evaluate any expert on train/val/test |
| **Unified benchmark script** | `scripts/benchmark.py` — run all routers, produce comparison table |
| **Unified analysis script** | `scripts/analyze.py` — diversity, root cause, calibration analysis |
| **Shared utilities** | `scripts/utils/data.py` — model loading, data loaders, class groups |
| | `scripts/utils/metrics.py` — BA, per-class acc, group acc, ECE, routing metrics |
| | `scripts/utils/features.py` — logit extraction, 24-d/89-d/92-d features, PCA |
| **OOP Router framework** | `scripts/router/base.py` — abstract `BaseRouter` |
| | `scripts/router/uniform.py` — `UniformRouter` |
| | `scripts/router/confidence.py` — `ConfidenceRouter` (calibrated) |
| | `scripts/router/product.py` — `ProductRouter` (geometric mean) |
| | `scripts/router/correctness.py` — `CorrectnessRouter` (trust meters) |
| | `scripts/router/pairwise.py` — `PairwiseRouter` (tournament) |
| | `scripts/router/cluster.py` — `ClusterRouter` (per-cluster weights) |
| | `scripts/router/gate.py` — `GateRouter` (learned MLP gate) |
| | `scripts/router/tta.py` — `TTARouter` (test-time augmentation) |
| | `scripts/router/selective.py` — `SelectiveRouter` (abstain on low confidence) |
| **Weighted training support** | `scripts/train_lal_weighted.py` — LAL with per-sample loss weighting |
| | `scripts/train_paco_weighted.py` — PaCo with per-sample loss weighting |
| | `data/weighted_dataset.py` — `WeightedDataset` wrapper |
| **Code hardening** | Absolute imports, device placement, checkpoint overwrite fix, gitignore fixes |

### ❌ Remaining — Need New Training Runs

| Item | Status | Details |
|------|--------|---------|
| **Retrain LAL on proper split** | ⏳ Pending | Train on `lt_train_indices.npy`, validate on `lt_val_indices.npy` |
| **Retrain PaCo on proper split** | ⏳ Pending | Same, with 400-epoch PaCo schedule |
| **Retrain Mixup on proper split** | ⏳ Pending | Same, with 200-epoch Mixup schedule |
| **Re-establish baselines on test set** | ⏳ Pending | Individual BA, uniform avg, optimal fixed weights |
| **Re-run routing methods** | ⏳ Pending | 89-d, 92-d, selective, pairwise, etc. |
| **Re-verify root cause problems** | ⏳ Pending | Feature learning gap, all-wrong ceiling, etc. |
| **MoCo v2 expert replacement** | ⏳ Requires GPU | Raises absolute accuracy |
| **SADE-style test-time adaptation** | ⏳ Requires GPU | Potential orthogonal signal |
| **Final evaluation on CIFAR-100 test set** | ⏳ Not needed until method is settled | |

### Old Results (Flawed Split — For Reference Only)

The following results were obtained on the deprecated 5K-balanced-validation split and are **not comparable** to standard CIFAR-100-LT benchmarks:

| Expert | Val BA (old) | Head | Medium | Tail |
|--------|:------------:|:----:|:------:|:----:|
| CE | 39.46% | 68.5% | 37.9% | 12.0% |
| LAL | 43.98% | 62.8% | 41.5% | 27.7% |
| PaCo | 49.28% | 65.3% | 48.9% | 33.1% |
| Mixup | 40.80% | — | — | — |

| Routing Method | BA (old) | vs Uniform |
|:---------------|:--------:|:----------:|
| Uniform avg | 51.12% | — |
| Opt fixed weights | 52.58% | +1.46% |
| 89-d correctness | 52.42% | +1.30% |
| 92-d combined | 52.49% | +1.37% |
| Selective 92-d (best) | **52.70%** | **+1.58%** |
| 392-d hybrid TTA | 53.22% | +2.10% |

---

## Project Structure (Current)

```
expert_method/
├── data/                          # CIFAR-100 dataset + split indices
│   ├── cifar-100-python/          #   raw CIFAR-100 files
│   ├── processed/                 #   split indices
│   │   ├── lt_train_indices.npy       # ~8,678 (new — proper LT train)
│   │   ├── lt_val_indices.npy         # ~2,169 (new — proper LT val)
│   │   ├── lt_all_indices.npy         # ~10,847 (new — complete LT)
│   │   ├── base_train_indices.npy     # 45,000 (OLD — deprecated)
│   │   ├── balanced_val_indices.npy   # 5,000 (OLD — deprecated)
│   │   └── val_targets.npy            # OLD — deprecated
│   ├── cifar_lt.py                #   LongTailCIFAR100 Dataset class
│   ├── weighted_dataset.py        #   WeightedDataset wrapper for boosting
│   └── __init__.py
│
├── losses/                        # Loss function implementations
│   ├── ce_loss.py                 #   Standard Cross-Entropy Loss
│   ├── lal_loss.py                #   Logit-Adjusted Loss (Menon ICLR 2021)
│   ├── paco_loss.py               #   PaCo Loss — parametric contrastive (Cui ICCV 2021)
│   ├── balanced_softmax_loss.py   #   Balanced Softmax Loss (Ren ECCV 2020)
│   └── __init__.py                #   exports CELoss, LALLoss, PaCoLoss
│
├── models/                        # Model architectures
│   ├── resnet32.py                #   ResNet-32 backbone, classifier, PaCoResNet32
│   └── __init__.py
│
├── scripts/                       # Training and diagnostic scripts
│   ├── __init__.py
│   ├── train.py                   #   UNIFIED: --method {lal, mixup, paco, ce, balanced_softmax}
│   ├── evaluate.py                #   UNIFIED: --expert LAL --dataset {train, val, test}
│   ├── benchmark.py               #   UNIFIED: run all routers, produce comparison table
│   ├── analyze.py                 #   UNIFIED: --mode {diversity, root_cause, calibration, all}
│   ├── base_trainer.py            #   BaseTrainer: common loop, metrics, checkpointing
│   ├── train_lal.py               #   LALTrainer — Logit-Adjusted Loss expert
│   ├── train_paco.py              #   PaCoTrainer — PaCo contrastive expert
│   ├── train_mixup.py             #   Mixup+CE trainer
│   ├── train_lal_weighted.py      #   Weighted LAL trainer (boosting)
│   ├── train_paco_weighted.py     #   Weighted PaCo trainer (boosting)
│   ├── train_ce.py                #   CE trainer (legacy)
│   ├── train_balanced_softmax.py  #   Balanced Softmax trainer (legacy)
│   │
│   ├── utils/                     #   Shared utilities
│   │   ├── __init__.py
│   │   ├── data.py                #   Model loading, data loaders, class groups
│   │   ├── metrics.py             #   BA, per-class acc, group acc, ECE, routing metrics
│   │   └── features.py            #   Logit extraction, 24-d/89-d/92-d features, PCA
│   │
│   └── router/                    #   OOP Router framework (9 routers)
│       ├── __init__.py            #   Router registry
│       ├── base.py                #   BaseRouter (abstract)
│       ├── uniform.py             #   UniformRouter — average logits
│       ├── confidence.py          #   ConfidenceRouter — max-softmax + calibration
│       ├── product.py             #   ProductRouter — geometric mean of probs
│       ├── correctness.py         #   CorrectnessRouter — trust meters
│       ├── pairwise.py            #   PairwiseRouter — tournament comparators
│       ├── cluster.py             #   ClusterRouter — per-cluster optimal weights
│       ├── gate.py                #   GateRouter — learned MLP gate
│       ├── tta.py                 #   TTARouter — test-time augmentation
│       └── selective.py           #   SelectiveRouter — abstain on low confidence
│
├── utils/                         # Standalone utilities
│   ├── create_lt_split.py         #   Create proper CIFAR-100-LT train/val split
│   └── __init__.py
│
├── docs/                          # Planning and context documentation
│   ├── AGENTs.md                  #   Agent skill rules for MoE gate routing
│   ├── PLAN.md                    #   Boosting-style adversarial expert training plan
│   ├── research.md                #   Literature survey and critical analysis
│   ├── problem.md                 #   Verified root causes (OLD SPLIT — for reference)
│   ├── experiments.md             #   Failed experiments log (OLD SPLIT — for reference)
│   ├── project-context.md         #   This file
│   ├── final-report.md            #   Stage 1 final report (OLD SPLIT — for reference)
│   ├── novel-routing-ideas-analysis.md  # Novel routing ideas analysis
│   ├── redo-plan.md               #   Plan to redo with proper data split
│   ├── stage0-data-pipeline.md    #   Data pipeline fix plan
│   └── refactor-plan.md           #   Scripts refactor plan (mostly completed)
│
├── checkpoints/                   # Model checkpoints (OLD SPLIT — will be replaced)
│   ├── LAL_best.pt                #   LAL expert (BA=43.98% on old split)
│   ├── LAL_latest.pt              #   LAL expert
│   ├── PaCo_best.pt               #   PaCo expert (BA=49.28% on old split)
│   ├── PaCo_latest.pt             #   PaCo expert
│   ├── Mixup_best.pt              #   Mixup+CE expert (BA=40.80% on old split)
│   └── Mixup_latest.pt            #   Mixup+CE expert
│
├── requirements.txt               # Python dependencies
└── .gitignore                     # Ignores: .venv/, data/cifar-100-python/
```

---

## Key Decisions Made

### Why Mixup+CE replaces CE as the third expert

- CE was too similar to LAL (κ=0.45, r=0.90 per-class accuracy)
- CE was severely overconfident (avg confidence when wrong=0.66, top wrong predictions at p=1.0)
- Focal Loss is identical to CE on CIFAR-100-LT (+0.09% — essentially CE)
- LDAM is marginally different but requires complex two-phase training (DRW) and is still CE-based
- MoCo v2 self-supervised is diverse but narratively overlaps with PaCo (both "contrastive")
- Mixup+CE is from a THIRD paradigm (interpolation-based augmentation), fixes overconfidence (soft labels prevent p=1.0), and is easy to implement (~15 lines)

### PaCo Hyperparameters

After exhaustive audit of the official PaCo codebase:

- **dim=32** (official shell script uses `--moco-dim 32` — was 128 in our code)
- **K=2048** (official uses K=1024 with batch 128; we use K=2048 with batch 256)
- **alpha=0.01** (official CIFAR-100-LT IR=0.01 config — was 0.5 in our code)
- **temperature=0.05** (official `--moco-t 0.05` — was 0.07)
- **LR=0.05, epochs=400, step schedule at [320,360]** (official shell config)
- **CIFAR-100 normalization** (official PaCo code has a bug using CIFAR-10 stats; we corrected it)

### Diversity Threshold

From TSC paper evidence: classification-boundary loss pairs have κ ≈ 0.82. We set κ < 0.80 as the threshold for "sufficient diversity." If κ exceeds 0.80, we intervene with diversity regularization or expert replacement.

---

## Training Hyperparameters

| Hyperparameter | LAL | PaCo | Mixup+CE |
|---------------|:---:|:----:|:------------------:|
| Epochs | 200 | 400 | 200 |
| Batch size | 128 | 256 | 128 |
| Optimizer | SGD (mom=0.9, nest=True) | SGD (mom=0.9, nest=True) | SGD (mom=0.9, nest=True) |
| Weight decay | 5e-4 | 5e-4 | 5e-4 |
| LR | 0.1 | 0.05 | 0.1 |
| LR schedule | Cosine | Step [320,360] | Cosine |
| Warmup | 5 epochs | 10 epochs | 5 epochs |
| Loss | LAL (τ=1.0) | PaCo (α=0.01, t=0.05, K=2048) | CE + Mixup (α=1.0) |

---

## Workflow (Current — Using Proper Split)

1. **Prepare data split**: `python utils/create_lt_split.py` ✅
2. **Train experts** (on proper LT split — **NEED TO RUN**):
   - `python scripts/train.py --method lal --epochs 200 --lr 0.1`
   - `python scripts/train.py --method paco --epochs 400 --lr 0.05`
   - `python scripts/train.py --method mixup --epochs 200 --lr 0.1`
3. **Evaluate experts**: `python scripts/evaluate.py --expert LAL --dataset test`
4. **Diversity analysis**: `python scripts/analyze.py --mode diversity --dataset test`
5. **Routing benchmark**: `python scripts/benchmark.py --dataset test --output results.json`
6. **Root cause analysis**: `python scripts/analyze.py --mode root_cause --dataset test`

---

## What to Do Next

### Immediate (Requires GPU — Kaggle)

1. **Retrain all three experts** on the proper LT split using `scripts/train.py`
2. **Evaluate** on the CIFAR-100 test set using `scripts/evaluate.py`
3. **Run benchmark** using `scripts/benchmark.py --dataset test`

### After New Baselines Are Established

4. **Re-verify root cause problems** — check if feature learning gap, all-wrong ceiling, lone dissenter paradox still hold
5. **Compare with old results** — does the proper split change the routing dynamics?
6. **Proceed with boosting-style approach** (`docs/PLAN.md`) if routing gap persists
7. **Or proceed with novel routing ideas** (`docs/novel-routing-ideas-analysis.md`)
