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

## Current Status

### ✅ Completed

| Item | Details |
|------|---------|
| Data split | 5K balanced val + 45K base train pool (stratified holdout) |
| LAL expert | Trained 200 epochs, BA=43.98%, H=62.8% M=41.5% T=27.7% |
| PaCo expert | Trained 400 epochs, BA=49.28%, H=65.3% M=48.9% T=33.1% |
| Mixup+CE expert | Trained 200 epochs, BA=40.80% |
| PaCo fixes | dim=32, K=2048, alpha=0.01, temp=0.05, step schedule [320,360], CIFAR-100 normalization |
| Diversity analysis | κ(LAL,PaCo)=0.41, κ(LAL,Mixup)=0.45, κ(PaCo,Mixup)=0.44; oracle=62.90% |
| Code hardening | absolute imports, device placement, checkpoint overwrite, gitignore fixes |
| **Routing experiments (v1)** | 9 routing methods tested — none achieved routing contribution ≥1% (see `docs/experiments.md` §2) |
| **Root cause analysis** | 10 verified problems documented in `docs/problem.md` |
| **89-d enriched routing (v2)** | **52.42% BA (+1.30% over uniform)** across 3 seeds — **achieves the +1% target**. See `docs/experiments.md` §6.1. |
| **Augmentation consistency (Round 3)** | Tested — failed success criterion (+0.18% gain < 0.5% threshold). See `docs/experiments.md` §7.1. |
| **Pairwise ranking routing (Round 3)** | LR pairwise comparators (52.10%) and MLP pairwise (50.66%) — both underperform 89-d correctness. See `docs/experiments.md` §7.2–7.3. |
| **92-d combined routing (Round 3)** | **52.49% ± 0.08% BA (+1.37% over uniform)** across 3 seeds — **best routing method**. Ties optimal fixed weights (52.58%, p=0.71). See `docs/experiments.md` §7.4. |
| **Meta-router (9-d)** | 52.40% BA (+1.28%) — below 89-d and 92-d methods. See `docs/experiments.md` §7.5. |
| **TTA-averaged predictions (Round 4)** | Raises absolute BA (53.00%) but hurts routing fraction. See `docs/experiments.md` §8.1. |
| **Gradient sensitivity routing (Round 4)** | Signal exists (r=0.24-0.34) but redundant with 92-d features. Best: 52.52% (log_grad). See `docs/experiments.md` §8.2. |
| **Selective routing (Round 4)** | **52.70% BA (+0.12% over opt fixed)** — first method to beat opt fixed. See `docs/experiments.md` §8.3. |
| **392-d hybrid TTA routing (Round 4)** | **53.22% BA** — highest absolute BA achieved. See `docs/experiments.md` §8.4. |
| **GDDR (Round 5)** | 46.98% BA — gradient directions in 3072-d space are near-orthogonal (mean cos sim ≈ 0.03). **Failed.** See `docs/experiments.md` §9.1. |
| **Cluster routing (Round 5)** | Best variant 52.56% BA (+0.08% vs opt fixed) — essentially tied. Per-cluster weights differ but gain within noise. **Failed to beat global opt.** See `docs/experiments.md` §9.2. |

### ✅ Resolved — Target Achieved

| Item | Details |
|------|---------|
| **Routing contribution ≥1%** | ✅ **Achieved** — multiple methods exceed +1% over uniform. Best: Selective 92-d at **+1.58%** over uniform. |
| **Standard 24-d correctness routing** | 51.70% (+0.58%) — still below target |
| **89-d enriched correctness routing** | 52.42% (+1.30%) — meets target |
| **Optimal fixed weights (reference)** | 52.58% (+1.46%) — selective routing beats this by +0.12% |

### ❌ Remaining Limitations

| Item | Status |
|------|--------|
| **Beat optimal fixed weights by ≥1%** | ❌ **Not achieved** — best method (Selective 92-d, 52.70%) beats opt fixed by only +0.12%. Need 53.58% for +1% margin. |
| **Augmentation consistency routing** | ❌ **Tested and failed** — +0.18% gain, below +0.5% threshold |
| **Gradient alignment routing (GDDR)** | ❌ **Failed** — gradient directions in 3072-d space are near-orthogonal regardless of expert correctness (mean cos sim ≈ 0.03) |
| **Cluster-based adaptive weighting** | ❌ **Failed to beat global opt** — best variant achieves 52.56% BA (+0.08% vs opt), gain within noise |
| **Gradient sensitivity routing** | ❌ **Tested — redundant with 92-d features** — adds only +0.08% |
| **TTA-averaged routing** | ❌ **Tested — hurts routing fraction** — baseline moves with improvement |
| **MoCo v2 expert replacement** | ⏳ Requires GPU (raises absolute accuracy, not routing fraction) |
| **SADE-style test-time adaptation** | ⏳ Requires GPU (potential orthogonal signal) |
| Final evaluation on CIFAR-100 test set | ⏳ Not needed until method is settled |

---

## Project Structure

```
expert_method/
├── data/                          # CIFAR-100 dataset + split indices
│   ├── cifar-100-python/          #   raw CIFAR-100 files
│   ├── processed/                 #   split indices from split_cifar100.py
│   │   ├── balanced_val_indices.npy   # 5,000 indices (50/class)
│   │   ├── base_train_indices.npy     # 45,000 indices (450/class)
│   │   └── val_targets.npy            # labels for verification
│   ├── cifar_lt.py                #   LongTailCIFAR100 Dataset class
│   └── __init__.py
│
├── losses/                        # Loss function implementations
│   ├── ce_loss.py                 #   Standard Cross-Entropy Loss
│   ├── lal_loss.py                #   Logit-Adjusted Loss (Menon ICLR 2021)
│   ├── paco_loss.py               #   PaCo Loss — parametric contrastive (Cui ICCV 2021)
│   └── __init__.py                #   exports CELoss, LALLoss, PaCoLoss
│
├── models/                        # Model architectures
│   ├── resnet32.py                #   ResNet-32 backbone, classifier, PaCoResNet32
│   └── __init__.py
│
├── scripts/                       # Training and diagnostic scripts
│   ├── base_trainer.py            #   BaseTrainer: common loop, metrics, checkpointing
│   ├── train_lal.py               #   LALTrainer — Logit-Adjusted Loss expert
│   ├── train_paco.py              #   PaCoTrainer — PaCo contrastive expert
│   ├── train_mixup.py             #   Mixup+CE trainer
│   ├── mock_test.py               #   Synthetic dry-run for all experts
│   ├── diversity_analysis.py      #   Cohen's κ, oracle, unique contribution
│   ├── debug_routing.py           #   Initial routing diagnostics
│   ├── debug_routing_root_cause.py #   Systematic root cause verification
│   ├── deep_debug_routing.py      #   Deep routing diagnostic
│   ├── eval_router.py             #   MLP router evaluation (5K data)
│   ├── eval_router_bigdata.py     #   MLP router evaluation (45K data)
│   ├── eval_router_v2.py          #   Various input representations
│   ├── gate_routing_diagnostic.py #   NLL-gated mixture on 24-d features
│   ├── gate_routing_3seeds.py     #   3-seed NLL gate evaluation
│   ├── per_class_calibration.py   #   Per-class temperature rescaling
│   ├── verify_routing_hypotheses.py # Hypothesis verification
│   ├── kaggle_root_cause.py      #   Feature learning gap analysis
│   ├── correctness_routing.py     #   Correctness-prediction routing (24-d trust meters)
│   ├── diagnose_loss_oracle.py    #   Loss vs accuracy diagnostic
│   ├── root_cause_analysis.py     #   Full root cause analysis
│   ├── root_cause_light.py        #   Lightweight root cause
│   ├── novel_routing_test.py      #   Round 2: 8 novel routing approaches
│   ├── refined_routing_test.py    #   Refined tests: MLP, product, signal ensemble
│   ├── verify_routing_target.py   #   Rigorous 5-fold CV of enriched features
│   ├── final_routing_push.py      #   Final push: 89-d features, meta-routing
│   ├── final_verify_89d.py        #   Clean verification of 89-d routing
│   ├── multi_seed_89d_verify.py   #   Multi-seed (3) verification of 89-d routing
│   ├── augmentation_consistency_analysis.py  #   Round 3: consistency feasibility study
│   ├── pairwise_routing.py        #   Round 3: pairwise ranking comparators
│   ├── pairwise_mlp_combined.py   #   Round 3: MLP pairwise + 92-d combined
│   ├── multi_seed_92d_verify.py   #   Round 3: multi-seed 92-d verification
│   ├── rotation_routing.py        #   Round 4: rotation prediction routing
│   ├── tta_routing.py             #   Round 4: TTA-averaged predictions routing
│   ├── hybrid_tta_routing.py      #   Round 4: 392-d hybrid (92-d + TTA probs)
│   ├── gradient_routing.py        #   Round 4: gradient sensitivity routing
│   └── selective_hybrid_routing.py #   Round 4: selective routing with threshold
│
├── utils/
│   ├── split_cifar100.py          #   Stratified 50/class validation split
│   └── __init__.py
│
├── docs/                          # Planning and context documentation
│   ├── AGENTs.md                  #   Agent skill rules for MoE gate routing
│   ├── PLAN.md                    #   Approved experimental blueprint
│   ├── research.md                #   Literature survey and critical analysis
│   ├── problem.md                 #   Verified root causes of routing failure (8 problems)
│   ├── experiments.md             #   Failed experiments log with root causes (9 methods)
│   ├── project-context.md         #   This file
│   └── final-report.md            #   Stage 1 final report
│
├── checkpoints/                   # Model checkpoints
│   ├── LAL_best.pt                #   LAL expert (BA=43.98%)
│   ├── LAL_latest.pt              #   LAL expert
│   ├── PaCo_best.pt               #   PaCo expert (BA=49.28%)
│   ├── PaCo_latest.pt             #   PaCo expert
│   ├── Mixup_best.pt              #   Mixup+CE expert (BA=40.80%)
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

After exhaustive audit of the official PaCo codebase, we discovered:

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

| Hyperparameter | LAL | PaCo | Mixup+CE (planned) |
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

## Workflow

1. **Prepare**: `python utils/split_cifar100.py` ✅
2. **Train experts**:
   - `python scripts/train_lal.py --device cuda` ✅ (200 epochs, BA=43.98%)
   - `python scripts/train_paco.py --device cuda --epochs 400` ✅ (400 epochs, BA=49.28%)
   - `python scripts/train_mixup.py --device cuda` ✅ (200 epochs, BA=40.80%)
3. **Diversity analysis**: `python scripts/diversity_analysis.py` ✅
4. **Routing experiments (v1)**: 9 routing methods tested — none achieved routing contribution ≥1% ❌
5. **Root cause analysis**: 10 verified problems documented in `problem.md` ✅
6. **Routing experiments (v2)**: 89-d enriched correctness-prediction routing achieves **52.41% BA (+1.29%)** ✅
7. **Remaining GPU options**: MoCo v2, test-time augmentation consistency, SADE — not yet tested
