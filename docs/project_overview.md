# Project Overview — Expert Method for CIFAR-100-LT

> Single authoritative entry point: project description, architecture, codebase layout, environment constraints, and current status. Read this after `AGENTs.md` to understand what this project is about and where it stands.

---

## 1. Problem Statement

**Long-tail class imbalance** on CIFAR-100 with imbalance ratio **IR=100** (max class count : min class count = 100:1). Three separate ResNet-32 backbones are trained with fundamentally different loss paradigms to produce diverse feature representations, then a lightweight dynamic router selects the best expert per test sample. The target is **≥1% Balanced Accuracy gain over uniform softmax averaging**.

---

## 2. Architecture

### 2.1 Three-Expert Design

| Expert | Paradigm | Loss | Role |
|--------|----------|------|------|
| **LAL** | Logit-adjusted classification | LAL (τ=1.0) | Balanced logits, strongest on tail classes (~23% tail acc) |
| **PaCo** | Supervised contrastive | PaCo (α=0.01, t=0.05, K=1024, dim=32) | Best overall BA, dominates head/med classes (72%/43%) |
| **Mixup** | Interpolation augmentation | CE + Mixup (α=1.0) | Best calibration (ECE=0.093), unique failure mode |

### 2.2 Data Pipeline (Corrected)

**Old (flawed) — now deprecated:** Held out 50 samples/class balanced validation BEFORE long-tail subsampling. Reduced training pool from 50K to 45K, created non-standard balanced validation set.

**Proper — currently in use:** LT subsampling on full 50K → 80/20 train/val split. Both splits follow the long-tailed distribution.

```
CIFAR-100 train (50K)
  └── LT subsampling (IR=100) → ~10,847 samples
       ├── ~8,678 training (80%)   → lt_train_indices.npy
       └── ~2,169 validation (20%) → lt_val_indices.npy

CIFAR-100 test (10K) → final evaluation only
```

### 2.3 Environment

| Component | Specification |
|-----------|---------------|
| **Primary training** | Kaggle T4 GPU (30h quota per session) |
| **Local execution** | CPU-only for analysis, feature extraction, routing evaluation |
| **Dependencies** | `requirements.txt` (torch, torchvision, numpy, scikit-learn, scipy) |

---

## 3. Codebase Structure

```
expert_method/
├── data/                           # Dataset + split indices
│   ├── cifar_lt.py                 #   LongTailCIFAR100 dataset class
│   ├── weighted_dataset.py         #   WeightedDataset wrapper (boosting)
│   └── processed/                  #   Split indices (lt_train, lt_val, lt_all, deprecated old)
│
├── losses/                         # Loss function implementations
│   ├── ce_loss.py                  #   Standard Cross-Entropy
│   ├── lal_loss.py                 #   Logit-Adjusted Loss (Menon ICLR 2021)
│   ├── paco_loss.py                #   PaCo — parametric contrastive (Cui ICCV 2021)
│   └── balanced_softmax_loss.py    #   Balanced Softmax (Ren ECCV 2020)
│
├── models/
│   ├── resnet32.py                 #   ResNet-32 backbone, classifier, PaCoResNet32
│   └── __init__.py
│
├── scripts/
│   ├── train.py                    #   UNIFIED: --method {lal, mixup, paco, ce, balanced_softmax}
│   ├── evaluate.py                 #   UNIFIED: --expert LAL --dataset {train, val, test}
│   ├── benchmark.py                #   UNIFIED: run all routers, produce comparison table
│   ├── analyze.py                  #   UNIFIED: --mode {diversity, root_cause, calibration, all}
│   ├── base_trainer.py             #   Shared training loop, checkpointing, metrics
│   ├── train_lal.py / train_paco.py / train_mixup.py / train_ce.py / train_balanced_softmax.py
│   ├── train_lal_weighted.py / train_paco_weighted.py  # Weighted trainers (boosting)
│   ├── utils/
│   │   ├── data.py                 #   Model loading, data loaders, class groups
│   │   ├── metrics.py              #   BA, per-class acc, group acc, ECE, routing metrics
│   │   └── features.py             #   Logit extraction, 24-d/89-d/92-d features, PCA
│   └── router/                     #   9 OOP routers
│       ├── base.py                 #   BaseRouter (abstract)
│       ├── uniform.py / confidence.py / product.py
│       ├── correctness.py / pairwise.py / cluster.py
│       ├── gate.py / tta.py / selective.py
│       └── __init__.py             #   Router registry
│
├── utils/
│   └── create_lt_split.py          #   Creates proper CIFAR-100-LT train/val split
│
├── checkpoints/                    # Expert .pt files (on proper split)
├── docs/                           # This knowledge base
└── requirements.txt
```

---

## 4. Current Status (Proper Split — Verified)

### 4.1 Per-Expert Performance (CIFAR-100 Test Set)

| Expert | BA | Head | Med | Tail | ECE |
|--------|:--:|:----:|:---:|:----:|:---:|
| **PaCo** ⭐ | **41.17%** | 71.97% | 42.86% | 12.21% | 0.244 |
| LAL | 39.96% | 59.57% | 39.53% | **23.12%** | 0.263 |
| BalancedSoftmax | 39.99% | 59.50% | 40.08% | 22.68% | 0.266 |
| Mixup | 37.52% | 70.13% | 37.97% | 8.26% | **0.093** |
| CE | 36.64% | 66.37% | 37.89% | 9.09% | 0.382 |

### 4.2 Ensemble Baselines

| Method | 3 Experts (LAL+PaCo+Mixup) | 5 Experts |
|:-------|:--------------------------:|:---------:|
| **Uniform avg** | **45.68%** | **46.07%** |
| Product | 45.68% | 46.07% |
| Optimal fixed weights | 44.89% | 46.39% |
| **Oracle** | **55.79%** | **61.82%** |
| All-wrong ceiling | **44.2%** | **38.2%** |

**Critical finding:** On the proper split, uniform averaging is already optimal for 3 experts (opt fixed is *worse* at 44.89%). The proper split is ~7-8% harder across all metrics than the old flawed split.

### 4.3 Diversity (3 experts: LAL + PaCo + Mixup)

| Pair | Cohen's κ | Per-class r | Verdict |
|------|:---------:|:-----------:|---------|
| LAL ↔ PaCo | 0.417 | 0.919 | ✅ Healthy diversity |
| LAL ↔ Mixup | 0.412 | 0.922 | ✅ Healthy diversity |
| PaCo ↔ Mixup | 0.477 | 0.973 | ⚠️ Moderate |

---

## 5. Training Hyperparameters

| Hyperparameter | LAL | PaCo | Mixup |
|:---------------|:---:|:----:|:-----:|
| Epochs | 200 | 400 | 200 |
| Batch size | 128 | 256 | 128 |
| Optimizer | SGD (mom=0.9, nest=True) | SGD (mom=0.9, nest=True) | SGD (mom=0.9, nest=True) |
| Weight decay | 5e-4 | 5e-4 | 5e-4 |
| LR | 0.1 | 0.05 | 0.1 |
| LR schedule | Cosine | Step [320, 360] | Cosine |
| Warmup | 5 epochs | 10 epochs | 5 epochs |
| Loss | LAL (τ=1.0) | PaCo (α=0.01, t=0.05, K=1024) | CE + Mixup (α=1.0) |

---

## 6. Pending Work

- [ ] **Re-run routing benchmark** on proper split (89-d, 92-d, selective, pairwise methods)
- [ ] **Re-verify root cause problems** on proper split (feature learning gap, all-wrong ceiling, etc.)
- [ ] **Proceed with boosting or novel routing ideas** (see `future-directions.md`)
- [ ] **Train MoCo v2 expert** (raises absolute accuracy, does NOT improve routing fraction)

---

## 7. Key Decisions Made

- **Mixup replaced CE** because CE was too similar to LAL (κ=0.45, r=0.90) and severely overconfident (ECE=0.382)
- **PaCo hyperparameters** aligned with official codebase: dim=32, K=1024, α=0.01, t=0.05, step schedule
- **Three paradigms** (logit-adjusted, contrastive, interpolation) ensure maximum feature diversity
- **Diversity threshold:** κ < 0.80 per TSC paper evidence; all pairs satisfy this
