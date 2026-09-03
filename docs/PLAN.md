# Implementation Plan: Boosting-Style Adversarial Expert Training

> **Target problem:** Frozen experts' features encode class identity, not routing relevance (Feature Learning Gap = 19.06%). The three experts fail on the same samples (37.1% all-wrong ceiling). A router cannot learn to pick the best expert because the features don't encode "which expert is best for this sample."
>
> **Proposed solution:** Train experts *sequentially* where each expert is trained with upweighted loss on samples where the *previous* experts are wrong. This creates natural specialization: Expert B learns "samples like the ones A gets wrong"; Expert C learns "the hardest samples both A and B get wrong." Routing-relevant information emerges in the features without needing to extract it post-hoc.
>
> **Novelty:** Unlike AdaBoost (which produces weak learners → single strong classifier), this approach produces separate deep experts kept for ensemble, with error-aware weighting that creates routing-relevant feature spaces. Unlike RIDE (shared backbone + distribution-aware sampling), this uses separate backbones with *dynamic error-aware loss weighting*.

---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Why It Should Work](#2-why-it-should-work)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 0: Pre-Flight Checks](#4-phase-0-pre-flight-checks)
5. [Phase 1: Expert A (LAL — Already Trained)](#5-phase-1-expert-a-lal--already-trained)
6. [Phase 2: Expert B (Error-Aware LAL)](#6-phase-2-expert-b-error-aware-lal)
7. [Phase 3: Expert C (Error-Aware PaCo)](#7-phase-3-expert-c-error-aware-paco)
8. [Phase 4: Routing Evaluation](#8-phase-4-routing-evaluation)
9. [Error Weighting Strategies](#9-error-weighting-strategies)
10. [Files to Create/Modify](#10-files-to-createmodify)
11. [Hyperparameter Sweep](#11-hyperparameter-sweep)
12. [Evaluation Plan](#12-evaluation-plan)
13. [Expected Outcomes](#13-expected-outcomes)
14. [Risk Mitigation](#14-risk-mitigation)
15. [Timeline & GPU Budget](#15-timeline--gpu-budget)
16. [Appendix: WeightedDataset Implementation](#16-appendix-weighteddataset-implementation)

---

## 1. How It Works

### Core Idea

Standard independent training produces experts whose features organize by **class identity** — because that's what the classification loss optimizes for. Features do NOT organize by "which expert is best" because that information is never part of the training objective.

**Critical finding from Phase 0:** All three experts achieve near-100% training accuracy (LAL: 99.72%, Mixup: 97.84%, PaCo: 95.16%). The training set is **memorized**, not learned. This means training-set errors cannot be used as a hardness signal — there are almost none (LAL has only 27 errors on 9,754 samples).

**Revised approach:** Since training errors are unavailable, we use **confidence as a continuous hardness signal**. Even correct predictions vary in confidence, and low-confidence-correct samples are the ones near decision boundaries — the same samples that cause generalization failures.

Furthermore, different experts have different confidence profiles:
- **LAL:** Only 6% of training samples have confidence < 0.9 (overconfident, memorized)
- **PaCo:** 38.5% have confidence < 0.9 (better calibrated)
- **Mixup:** **64% have confidence < 0.9** (soft labels prevent overconfidence)

Mixup's confidence is the richest hardness signal because its soft labels prevent memorization. **Cross-paradigm hardness** — using one expert's confidence to weight another expert's training — creates more diversity than same-paradigm weighting.

**Revised boosting strategy:**

```
Expert A (LAL):     trained on ALL data (standard) — already done
                    ↓
Expert B (LAL):     trained with LOSS × (1 + α · (1 - Mixup_confidence))
                    → features encode "samples Mixup (a different paradigm) finds hard"
                    → cross-paradigm diversity
                    ↓
Expert C (PaCo):    trained with LOSS × (1 + α · [A_val_error] + β · [B_val_error])
                    → weights based on VALIDATION errors of A and B
                    → targets genuine generalization failures
```

### Why This Creates Routing Signal

After boosting training:
- **Expert B's features** are organized not just by class, but by "how similar is this sample to the ones Mixup finds hard?" This is cross-paradigm information — fundamentally different from what LAL alone provides.
- **Expert C's features** are organized by "is this one of the hardest samples?" — directly encoding difficulty based on where both previous experts fail in generalization.
- The router can now learn: "If features look like Mixup-hard patterns → route to B. If features look extremely hard → route to C. Otherwise → use A."

---

## 2. Why It Should Work

### Evidence from the Project's Own Findings

| Finding | How Boosting Addresses It |
|:--------|:--------------------------|
| **Feature Learning Gap (19.06%)** | Features are *forced* to encode error patterns because error-aware weighting shapes the loss landscape. Expert B's backbone learns to distinguish "easy for A" vs "hard for A." |
| **37.1% All-Wrong Ceiling** | Expert C is explicitly trained on samples where A AND B are wrong. This directly attacks the hardest cases, shrinking the all-wrong set. |
| **Lone Dissenter Paradox** | Experts B and C are less likely to be "lone dissenters" because they were trained to be correct where others fail. The correct expert is no longer the least confident — it's the *specialist*. |
| **Different data > different losses** (SADE finding) | Boosting creates *different data distributions per expert* — proven to be more effective than different losses alone. |
| **Product captures +0.82%** | After boosting, experts are more complementary. Product combination will be even more effective because experts fail on different subsets. |

### Theoretical Support

Boosting (AdaBoost, Gradient Boosting) is mathematically grounded in the principle of sequentially correcting errors. While standard boosting produces a single strong classifier from weak learners, our adaptation for deep ensembles preserves the diversity benefit while creating routing-relevant features.

### Concrete Expectation

| Metric | Current Best | Target with Boosting |
|:-------|:------------:|:--------------------:|
| Uniform average BA | 51.12% | **53-54%** |
| Best routing (Selective 92-d) | 52.70% | **54-56%** |
| Oracle (≥1 expert correct) | 62.90% | **66-70%** (Expert C shrinks all-wrong) |
| Pairwise Cohen's κ | 0.41-0.45 | **0.30-0.38** (more diverse) |
| All-wrong ceiling | 37.1% | **28-33%** |

---

## 3. Architecture Overview

### Expert Configuration After Boosting

| Expert | Loss | Weighting Signal | Est. BA | Role |
|:-------|:----:|:----------------:|:-------:|:-----|
| **A (LAL)** | LAL (τ=1.0) | None (standard) | 43.98% (known) | Generalist, good on head classes |
| **B (LAL)** | LAL (τ=1.0) | 1 + α · (1 - Mixup_confidence) | 44-47% | Specialist on samples Mixup finds hard |
| **C (PaCo)** | PaCo (α=0.01, t=0.05) | 1 + α·[A_val_error] + β·[B_val_error] | 47-50% | Specialist on hardest generalization cases |

### Why Cross-Paradigm Weighting?

Expert B uses **Mixup's confidence** (not LAL's correctness) because:
- **LAL has only 27 training errors** (99.72% train acc) — too few to learn from
- **Mixup has 64% low-confidence training samples** — rich, continuous hardness signal
- **Cross-paradigm signal** — Mixup (interpolation-based) finds different samples hard than LAL (logit-adjusted). This creates fundamentally different feature geometry

Expert C uses **validation errors** of A and B (not training errors) because:
- All experts memorize the training set — training errors are near zero
- Validation errors (~56% of 5K for LAL) are **genuine generalization failures**
- Weighting by validation errors forces Expert C to focus on samples that cause real-world mistakes

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Phase 1: Expert A                          │
│  LT Dataset ──→ LALTrainer (standard) ──→ Expert A .pt       │
│                                       (43.98% BA, 99.72% tr)│
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│        Mixup confidence on LT set (already evaluated)        │
│  Mixup_confidence.npy ──→ per-sample confidence ∈ [0,1]     │
│  weight_B = 1.0 + α * (1 - Mixup_confidence)                │
│  → 64% of training samples get weight > 1.0 (rich signal!)  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 2: Expert B                          │
│  LT Dataset + weight_B ──→ LALTrainer (weighted loss)        │
│                         ──→ Expert B .pt                     │
│  Features encode: "samples Mixup (different paradigm)        │
│                   finds hard"                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│   Evaluate A & B on VALIDATION set (5K balanced)             │
│  → genuine generalization errors (not memorized)             │
│  → A_val_errors: ~2,800 samples (56% of 5K)                 │
│  → B_val_errors: ~2,800 samples (estimated)                 │
│  → weight_C per VALIDATION sample                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 3: Expert C                               │
│  LT Dataset (standard) + VAL Set (with error weights)        │
│  ──→ PaCoTrainer (weighted on val set) ──→ Expert C .pt     │
│  Features encode: "hardest generalization cases"             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 4: Routing Evaluation                      │
│  Experts A, B, C ──→ diversity_analysis.py                   │
│                   ──→ correctness_routing.py                  │
│                   ──→ compare vs baseline                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 0: Pre-Flight Checks

Before implementing, verify the current state of the codebase and checkpoints.

### 4.1 Verify Existing Checkpoints

```bash
ls -la checkpoints/
# Expected: LAL_best.pt, LAL_latest.pt, PaCo_best.pt, PaCo_latest.pt,
#           Mixup_best.pt, Mixup_latest.pt
```

### 4.2 Verify Existing Expert Performance

Run `diversity_analysis.py` to confirm baseline numbers:
```
LAL:  43.98% BA
PaCo: 49.28% BA
Mixup: 40.80% BA
Oracle: 62.90%
```

### 4.3 Compute Error Baseline

Run a diagnostic script to understand the error patterns:
- **How many training samples does Expert A get wrong?** (~56% = ~5,500 samples)
- **Per-class breakdown of errors** — which classes does A fail on most?
- **Confidence on error samples** — is A confidently wrong or uncertain?

This informs the weighting strategy (see §9).

### 4.4 Verify Existing Routing Pipeline

Ensure that `correctness_routing.py` and `diversity_analysis.py` still work with the current checkpoints. These will be used in Phase 4 to evaluate the new experts.

---

## 5. Phase 1: Expert A (LAL — Already Trained)

**Status: ✅ Complete.** Expert A (LAL) is already trained at `checkpoints/LAL_best.pt` with 43.98% BA.

**Action required:** None for training. But we need to evaluate Expert A on the training set to compute error weights for Expert B.

### Compute Error Mask from Expert A

We need to run Expert A on the full long-tailed training set (~9,754 samples) and record per-sample correctness. This produces a boolean array `correct_A[i]` for each training sample.

**Implementation:** See `scripts/evaluate_expert.py` in §10.

---

## 6. Phase 2: Expert B (Confidence-Weighted LAL)

### ⚠️ Critical Finding from Phase 0

**All three experts achieve near-100% training accuracy** (LAL: 99.72%, Mixup: 97.84%, PaCo: 95.16%). The training set is **memorized**, not learned. Using training-set errors as a hardness signal is ineffective — LAL has only 27 errors on 9,754 samples.

**Mixup's confidence is a much richer signal:** 64% of Mixup's training samples have confidence < 0.9 (vs only 6% for LAL). Mixup's soft labels prevent overconfidence, making its confidence a reliable indicator of sample difficulty.

### 6.1 Training Objective

Expert B is trained with the same LAL loss as Expert A, but each sample's loss is multiplied by a weight based on **Mixup's confidence** (cross-paradigm weighting):
```
weight_B[i] = 1.0 + α × (1 - Mixup_confidence[i])
```
where `Mixup_confidence[i]` is Mixup's softmax max-probability on sample `i`.

**Effect:**
- Samples Mixup is very confident about (conf ≈ 1.0): weight ≈ 1.0 (standard)
- Samples Mixup is uncertain about (conf ≈ 0.5): weight ≈ 1.0 + 0.5α (moderately upweighted)
- Samples Mixup is very uncertain (conf ≈ 0.2): weight ≈ 1.0 + 0.8α (strongly upweighted)

This is a **continuous weight**, not binary. It provides a graded hardness signal.

**Recommended α range:** [1.0, 2.0, 3.0, 5.0]

### 6.2 Why Cross-Paradigm Weighting?

Using Mixup's confidence (instead of LAL's correctness) is a deliberate design choice:

1. **Signal richness:** 64% of training samples have confidence < 0.9 → ~6,200 samples with elevated weight. This is a rich, dense signal compared to LAL's 27 errors.

2. **Cross-paradigm diversity:** Mixup (interpolation-based augmentation) and LAL (logit-adjusted loss) are fundamentally different training paradigms. Mixup's "hard" samples are based on prediction smoothness, not class frequency. Expert B (LAL-based) learning from Mixup's hardness creates features that encode a *different view* of the data.

3. **Continuous signal:** Confidence is a continuous value (0-1), not binary. This provides finer-grained information about *how* hard each sample is.

4. **Calibration advantage:** Mixup is the best-calibrated expert (ECE=0.0888 vs LAL's 0.2456). Its confidence is more reliable as a hardness indicator.

### 6.3 Architecture

Expert B uses the same `ResNet32` architecture as Expert A. It is trained from scratch (random initialization), not fine-tuned from Expert A.

**Why train from scratch?** Fine-tuning from A would preserve A's feature geometry. We want B to develop *different* features that encode "samples Mixup finds hard" — this requires independent training to create a genuinely different feature space.

### 6.4 Implementation Strategy

**Option A: Weighted Loss (Recommended)**
Modify the loss function to accept per-sample weights. This preserves the original data distribution (all samples seen equally, but loss is weighted). The weight is applied as a multiplier to each sample's loss before reduction.

**Option B: Weighted Sampling**
Use `WeightedRandomSampler` to oversample low-confidence samples. Simpler but changes the data distribution, which can cause training instability.

**Recommendation:** Use Option A (Weighted Loss). It gives finer control and is more stable.

### 6.5 Training Hyperparameters

| Parameter | Value | Notes |
|:----------|:-----:|:------|
| Loss | LAL (τ=1.0) | Same as Expert A |
| Epochs | 200 | Same as Expert A |
| Batch size | 128 | Same |
| LR | 0.1 | Cosine schedule, 5-epoch warmup |
| Weight decay | 5e-4 | Same |
| α (confidence weight) | 2.0 (default), sweep [1.0, 3.0, 5.0] | Start with 2.0 |
| Weighting signal | 1 - Mixup_confidence | Pre-computed from Mixup eval |
| Initialization | Random (from scratch) | Critical for diversity |
| Validation metric | Balanced Accuracy (BA) | Same as standard |

### 6.6 Expected Behavior During Training

- **Early epochs:** Expert B learns head classes (like any LAL model). The confidence weighting has a subtle effect — low-confidence samples contribute slightly more to the loss.
- **Mid training (epochs 50-150):** The weighting becomes more influential. Expert B's features begin to organize around "samples Mixup finds hard." BA on Mixup's low-confidence samples should improve faster than BA on Mixup's high-confidence samples.
- **Late training (epochs 150-200):** Expert B should have higher accuracy on Mixup-hard samples than Expert A does, potentially at a small cost to accuracy on Mixup-easy samples.

**Monitoring:** Log `acc_on_Mixup_hard` (confidence < 0.7) and `acc_on_Mixup_easy` (confidence > 0.9) separately during training. This directly measures whether the cross-paradigm weighting is working.

---

## 7. Phase 3: Expert C (Validation-Error-Weighted PaCo)

### ⚠️ Key Design Decision

Expert C uses **validation-set errors** of A and B (not training-set errors) as the weighting signal. This is because:
- **All experts memorize the training set** — training errors are near zero
- **Validation errors are genuine generalization failures** — Expert A has ~56% error rate on the 5K balanced validation set (~2,800 samples)
- **The validation set is balanced** (50 samples/class) — errors on tail classes are well-represented
- **Using validation errors directly targets the generalization gap** — the exact problem we need to solve

### 7.1 Training Objective

Expert C is trained on a **combined dataset**: the full LT training set (standard, unweighted) PLUS the validation set with error-aware weights:

**Training set:** Standard PaCo training (no weighting) — provides general knowledge
**Validation set:** PaCo training with per-sample weights:
```
weight_C[i] = 1.0 + α × [A_val_error(i)] + β × [B_val_error(i)]
```
where `A_val_error(i) = 1` if Expert A is wrong on validation sample `i`, else `0`.

**Effect:**
- Validation samples both A and B got correct: weight = 1.0 (standard)
- Validation samples one expert got wrong: weight = 1.0 + α (moderately upweighted)
- Validation samples BOTH got wrong: weight = 1.0 + α + β (strongly upweighted)
- Training samples (not in validation set): weight = 1.0 (standard)

**This directly attacks the all-wrong ceiling.** Expert C sees the hardest validation samples multiple times per epoch, forcing it to learn features that generalize where A and B fail.

### 7.2 Why PaCo for Expert C

PaCo has the highest individual accuracy (49.28%) and the most diverse features (κ ≈ 0.41 vs LAL). Using PaCo as the final expert:
- Gives it the most expressive power (contrastive learning)
- Creates features that are maximally different from the two LAL-based experts
- The error weighting pushes it to cover the hardest generalization cases

### 7.3 Architecture

Expert C uses `PaCoResNet32` with the same architecture as the original PaCo expert. It is trained from scratch with the standard PaCo training pipeline (400 epochs, step LR schedule, two-view augmentations).

The validation set is added as an additional dataset during training. Each batch interleaves training samples (unweighted) and validation samples (weighted).

### 7.4 Data Loading Strategy

**Approach: Two-dataset loader with weighted validation set**

```python
# LT training set (unweighted)
train_set = LongTailCIFAR100(..., imbalance_ratio=100.0)

# Validation set (weighted by A and B errors)
val_set = LongTailCIFAR100(..., skip_longtail=True)  # 5K balanced
weighted_val = WeightedDataset(val_set, weight_C)     # per-sample weights

# Combine: each batch draws from both datasets
# Option 1: Concat datasets with different weights
# Option 2: Two separate loaders, interleave batches
```

**Recommendation:** Use two separate DataLoaders. For each batch, draw ~80% from training set (unweighted) and ~20% from validation set (weighted). This ensures the model sees enough training data while being regularly exposed to hard validation samples.

### 7.5 Training Hyperparameters

| Parameter | Value | Notes |
|:----------|:-----:|:------|
| Loss | PaCo (α=0.01, t=0.05, K=2048) | Same as original PaCo |
| Epochs | 400 | Same |
| Batch size | 256 | Same |
| Training/Val ratio per batch | 80% / 20% | 204 train + 52 val per batch |
| LR | 0.05 | Step schedule [320, 360] |
| Weight decay | 5e-4 | Same |
| α, β (error weights) | 1.0, 1.0 (default) | Sweep [0.5, 2.0] |
| Augmentations | View1: AutoAugment+Cutout, View2: MoCo v2 | Same |
| Initialization | Random (from scratch) | Critical for diversity |
| Validation metric | Balanced Accuracy (BA) | On held-out test set |

### 7.6 PaCo-Specific Considerations for Weighted Loss

The PaCo loss combines supervised logits with contrastive similarities. For the validation samples, we need to apply the error weights. Two approaches:

**Option A: Weight the total loss per sample**
```python
loss, aux = paco_loss(features, labels, logits)
weighted_loss = (loss * per_sample_weights).mean()
```
Simple but weight affects both supervised and contrastive components.

**Option B: Weight only the supervised component**
```python
loss, aux = paco_loss(features, labels, logits)
weighted_loss = (aux['ce_loss'] * per_sample_weights).mean() + aux['contrastive_loss']
```
More targeted — contrastive learning unaffected.

**Recommendation:** Start with Option A. If contrastive features collapse, switch to Option B.

---

## 8. Phase 4: Routing Evaluation

After all three experts are trained, evaluate routing performance using the existing infrastructure.

### 8.1 Diversity Check

Run `diversity_analysis.py` to measure:
- Individual BA for each new expert
- Pairwise Cohen's κ (target: < 0.38)
- Oracle accuracy (target: > 65%)
- Unique correct contributions

### 8.2 Routing Methods to Evaluate

Use the existing routing scripts (no modifications needed):

| Method | Script | Current Best | Target After Boosting |
|:-------|:-------|:------------:|:---------------------:|
| Uniform average | `diversity_analysis.py` | 51.12% | **53-54%** |
| Calibrated confidence | `debug_routing.py` | 51.44% | **52.5-53.5%** |
| 89-d correctness | `final_verify_89d.py` | 52.42% | **54-55%** |
| 92-d combined | `multi_seed_92d_verify.py` | 52.49% | **54-55.5%** |
| Selective 92-d | `selective_hybrid_routing.py` | 52.70% | **54-56%** |
| Optimal fixed weights | `gate_routing_diagnostic.py` | 52.58% | **53.5-54.5%** |

### 8.3 Key Metrics

| Metric | Measurement | Success Criterion |
|:-------|:------------|:-----------------:|
| Routing gain over uniform | Compare best routing BA vs uniform BA | > +1.5% |
| Margin over optimal fixed weights | Best routing BA vs opt fixed BA | > +0.5% |
| Oracle improvement | Compare new oracle vs old (62.90%) | > +3% |
| All-wrong ceiling reduction | New all-wrong % vs old (37.1%) | > -5% |
| Cohen's κ reduction | Average κ vs old (0.43) | < 0.38 |

### 8.4 Statistical Validation

- Run 3 seeds for each expert (0, 42, 123)
- Report mean ± std for all metrics
- Verify improvement is consistent across seeds

---

## 9. Error Weighting Strategies

### Strategy 1: Confidence-Based Weighting (Phase 2 — Expert B)

```
weight = 1.0 + α × (1 - Mixup_confidence)
```
- Continuous weight based on Mixup's softmax confidence
- α = 2.0 means a sample with confidence 0.5 gets 2× loss weight
- 64% of training samples have confidence < 0.9 → rich signal

**Pros:** Continuous signal, uses Mixup's superior calibration, cross-paradigm.
**Cons:** Requires pre-computing Mixup confidence on training set.

### Strategy 2: Validation-Error Progressive Weighting (Phase 3 — Expert C)

```
weight = 1.0 + α × [A_val_error] + β × [B_val_error]
```
- Applied to the 5K balanced validation set (not training set)
- Samples where BOTH A and B fail get highest weight
- Directly targets the all-wrong ceiling via generalization failures

**Pros:** Targets genuine generalization gap, uses balanced data.
**Cons:** Validation set is small (5K), risk of overfitting to val set.

### Strategy 3: Combined Confidence + Error (Backup)

```
weight = 1.0 + α × (1 - Mixup_confidence) + β × [LAL_val_error]
```
- Combines cross-paradigm confidence weighting with validation error targeting
- Most comprehensive but most complex

**Pros:** Multiple signals. **Cons:** Many hyperparameters.

### Recommendation

| Phase | Strategy | Parameters |
|:------|:---------|:-----------|
| Phase 2 (Expert B) | **Strategy 1** (Confidence) | α = 2.0 (default), sweep [1.0, 3.0, 5.0] |
| Phase 3 (Expert C) | **Strategy 2** (Val-Error Progressive) | α = 1.0, β = 1.0 (default), sweep [0.5, 2.0] |

---

## 10. Files to Create/Modify

### New Files

#### `data/weighted_dataset.py` — Weighted Dataset Wrapper

```python
"""
Wrapper that adds per-sample weights to an existing dataset.

The wrapped dataset returns (image, target, weight) tuples,
where weight is a scalar that the trainer uses to scale the loss.
"""

class WeightedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, weights: np.ndarray):
        assert len(base_dataset) == len(weights)
        self.base_dataset = base_dataset
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        return image, target, self.weights[idx]
```

#### `scripts/evaluate_expert.py` — Compute Error Masks

Evaluates a trained expert on the full training set and returns per-sample correctness and confidence.

```python
"""
Evaluate a trained expert on the long-tailed training set.
Returns correctness array and confidence array for all training samples.

Usage:
    python scripts/evaluate_expert.py --expert LAL --checkpoint checkpoints/LAL_best.pt
    # Output: saved to checkpoints/LAL_train_correctness.npy, LAL_train_confidence.npy
"""
```

#### `scripts/train_lal_weighted.py` — Weighted LAL Trainer

Inherits from `LALTrainer`, overrides `_compute_loss` to accept per-sample weights from the dataset.

#### `scripts/train_paco_weighted.py` — Weighted PaCo Trainer

Inherits from `PaCoTrainer`, overrides `_compute_loss` to accept per-sample weights.

#### `scripts/run_boosting_pipeline.py` — Full Orchestration Script

Orchestrates the entire pipeline:
1. Evaluate Expert A on training set → save error mask
2. Compute weights for Expert B
3. Train Expert B with weights
4. Evaluate Experts A+B on training set → save error masks
5. Compute weights for Expert C
6. Train Expert C with weights
7. Run diversity analysis
8. Run routing evaluation

### Modified Files

#### `scripts/base_trainer.py` — Support Weighted Loss

Add optional `sample_weights` parameter to `_compute_loss` and `_train_one_epoch`.

**Minimal change:** In `_train_one_epoch`, if the dataloader yields 3 items (image, target, weight), unpack the weight and pass it to `_compute_loss`. If 2 items, pass `weights=None`.

```python
def _train_one_epoch(self, loader):
    ...
    for batch in loader:
        if len(batch) == 3:
            images, targets, weights = batch
            weights = weights.to(self.device)
        else:
            images, targets = batch
            weights = None

        loss, aux = self._compute_loss(images, targets, weights=weights)
        ...
```

**`_compute_loss` signature change:**
```python
def _compute_loss(self, images, targets, weights=None):
    ...
```

Default implementation ignores weights (backward compatible). Subclasses override to apply weights.

---

## 11. Hyperparameter Sweep

### Phase 2: Expert B (α sweep — Mixup confidence weight)

| α | Expected Effect | Est. BA | Note |
|:-:|:----------------|:-------:|:-----|
| 1.0 | Mild (conf=0.5 → 1.5× weight) | 44-45% | Conservative |
| **2.0** | **Moderate (conf=0.5 → 2× weight)** | **44-47%** | **Recommended start** |
| 3.0 | Strong (conf=0.5 → 2.5× weight) | 44-47% | May degrade head accuracy |
| 5.0 | Aggressive (conf=0.5 → 3.5× weight) | 43-47% | Risk of overfitting |

**Sweep strategy:** Start with α = 2.0. If Expert B's BA drops below 44%, reduce to 1.0. If effect seems weak, try 3.0.

### Phase 3: Expert C (α, β sweep — validation error weights)

| α (A_error) | β (B_error) | Expected Effect | Est. BA |
|:-----------:|:-----------:|:----------------|:-------:|
| 0.5 | 0.5 | Mild | 48-50% |
| **1.0** | **1.0** | **Moderate** | **48-51%** |
| 1.0 | 2.0 | Focus more on B's errors | 47-50% |
| 2.0 | 1.0 | Focus more on A's errors | 47-50% |

**Sweep strategy:** Start with α = 1.0, β = 1.0. If Expert C's BA drops below 47%, reduce to 0.5.

### Diagnostic Sweep for Both Phases

Run a quick diagnostic with 50 epochs (instead of full 200/400) to test the effect of different α values. Use the one that looks most promising for the full training.

---

## 12. Evaluation Plan

### 12.1 Does Boosting Create Routing Signal?

**Test:** Compare correctness-prediction AUROC using features from boosting-trained experts vs original experts.

| Expert Set | Avg AUROC (trust meters) | Interpretation |
|:-----------|:------------------------:|:---------------|
| Original (LAL, PaCo, Mixup) | 0.84-0.89 | Baseline |
| Boosting (A, B, C) | **0.90-0.95** | ✅ Routing signal improved |
| Boosting (A, B, C) | 0.84-0.89 | ❌ No improvement |

### 12.2 Does Boosting Shrink the All-Wrong Ceiling?

**Test:** Compute fraction of validation samples where all three experts are wrong.

| Expert Set | All-Wrong % | Interpretation |
|:-----------|:-----------:|:---------------|
| Original | 37.1% | Baseline |
| Boosting | **< 33%** | ✅ All-wrong ceiling reduced |
| Boosting | > 35% | ❌ Minimal improvement |

### 12.3 Does Boosting Improve Routing BA?

**Test:** Run the existing 89-d correctness-prediction routing on the new experts.

| Expert Set | Routing BA | vs Opt Fixed | Interpretation |
|:-----------|:----------:|:------------:|:---------------|
| Original | 52.42% | −0.16% | Baseline |
| Boosting | **> 53.5%** | **> +0.5%** | ✅ Success |
| Boosting | 52.5-53.5% | 0 to +0.5% | ⚠️ Partial |
| Boosting | < 52.5% | < 0% | ❌ Failed |

### 12.4 Ablation: Is Sequential Training Necessary?

**Test:** Compare three variants:
1. **Full boosting** (A → B → C) — the proposed method
2. **No boosting** (independent A, B, C) — train 3 LAL experts independently, all with α=0
3. **Partial boosting** (A → B, C independent) — only one level of boosting

This ablation isolates the effect of sequential error-aware training from the effect of having three experts.

---

## 13. Expected Outcomes

### Best Case (α = 1.0, both phases work well)

| Metric | Before | After | Change |
|:-------|:------:|:-----:|:------:|
| Uniform average BA | 51.12% | **53.5%** | +2.4% |
| Best routing BA | 52.70% | **55.0%** | +2.3% |
| Margin over opt fixed | +0.12% | **+1.0%** | ✅ Target achieved |
| Oracle | 62.90% | **67%** | +4.1% |
| All-wrong ceiling | 37.1% | **30%** | −7.1% |
| Avg Cohen's κ | 0.43 | **0.35** | More diverse |

### Moderate Case (partial improvement)

| Metric | Before | After | Change |
|:-------|:------:|:-----:|:------:|
| Uniform average BA | 51.12% | **52.5%** | +1.4% |
| Best routing BA | 52.70% | **53.5%** | +0.8% |
| Margin over opt fixed | +0.12% | **+0.5%** | Partial |
| Oracle | 62.90% | **65%** | +2.1% |

### Worst Case (boosting doesn't help)

| Metric | Before | After | Change |
|:-------|:------:|:-----:|:------:|
| Uniform average BA | 51.12% | 51.0% | −0.1% |
| Best routing BA | 52.70% | 52.5% | −0.2% |
| Oracle | 62.90% | 62.5% | −0.4% |

**If worst case occurs:** The error weighting is too aggressive or the experts don't have enough capacity. Reduce α and try fine-tuning from original experts instead of training from scratch.

---

## 14. Risk Mitigation

### Risk 1: Expert B Overfits to Mixup's Confidence Signal

**Symptoms:** Expert B's BA drops below 43%, or it achieves high accuracy on Mixup-low-confidence samples but very low accuracy on Mixup-high-confidence samples (catastrophic forgetting of easy samples).

**Mitigation:**
- Reduce α (try 1.0 instead of 2.0) — weaker weighting preserves general performance
- Add the original (unweighted) loss as a regularization term: `L_total = L_weighted + λ × L_unweighted` with λ=0.5
- Use early stopping based on full validation BA (not just Mixup-hard BA)
- **Note:** Mixup's confidence is a continuous signal (not binary), so overfitting risk is lower than with binary error weighting

### Risk 2: Expert C Overfits to the Validation Set

**Symptoms:** Expert C achieves high accuracy on the 5K validation set but doesn't generalize to the test set, or its contrastive features collapse.

**Mitigation:**
- Reduce the validation batch fraction from 20% to 10%
- Reduce α, β to 0.5
- Use Option B for PaCo weighting (weight only supervised component, not contrastive)
- Add strong augmentation to validation samples (same as training augs) to prevent memorization
- Monitor validation BA vs test BA gap

### Risk 3: Error Weights Don't Create Routing Signal

**Symptoms:** Expert B's features still encode class identity. Trust meter AUROC doesn't improve over baseline (0.84-0.89).

**Mitigation:**
- Verify Expert B has different features (check κ with Expert A — target < 0.40)
- If κ is still high, the cross-paradigm weighting isn't working. Try a stronger α or switch to Strategy 3 (Combined Confidence + Error)
- Add an auxiliary correctness-prediction head (Idea B: RAAL) to explicitly force features to encode correctness
- Try a more aggressive weighting strategy (Strategy 2 or 3)
- Add an auxiliary correctness-prediction head (Idea B: RAAL) to explicitly force features to encode correctness

### Risk 4: Expert B and C Are Too Similar to A

**Symptoms:** Cohen's κ between A and B > 0.45 (same as original CE-LAL).

**Mitigation:**
- Train Expert B with a different loss (e.g., LDAM instead of LAL) — this combines boosting with loss diversity
- Use different augmentations for Expert B (stronger augs like PaCo's)
- Train Expert B with a different architecture (e.g., ResNet-44 instead of ResNet-32)

### Risk 5: The Router Still Can't Beat Optimal Fixed Weights

**Symptoms:** Even with better experts, the routing gain over optimal fixed weights is < 0.5%.

**Mitigation:**
- This would confirm that the 69.4% label ambiguity and 3-way comparison problem are irreducible with 3 experts
- Fallback to 2 experts (A + C only) — binary routing is fundamentally easier
- Switch to product combination (which already gives +0.82% without routing)
- Accept the result and document the ceiling

---

## 15. Timeline & GPU Budget

### Phase 0: Pre-Flight (0.5h CPU)

| Task | Time | Hardware |
|:-----|:----:|:---------|
| Verify checkpoints | 5 min | CPU |
| Run diversity analysis | 10 min | CPU |
| Compute error baseline | 15 min | CPU |

### Phase 1: Evaluate Expert A (0.5h GPU)

| Task | Time | Hardware |
|:-----|:----:|:---------|
| Forward pass on full training set | 30 min | T4/Kaggle GPU |

### Phase 2: Train Expert B (1.5h GPU per run)

| Task | Time | Hardware |
|:-----|:----:|:---------|
| Train with α=1.0 (default) | 1.5h | T4 GPU |
| Train with α=2.0 (sweep) | 1.5h | T4 GPU |
| Evaluate Expert B | 30 min | T4 GPU |
| **Total Phase 2** | **3.5h** | |

### Phase 3: Train Expert C (3h GPU per run)

| Task | Time | Hardware |
|:-----|:----:|:---------|
| Evaluate A+B on training set | 30 min | T4 GPU |
| Train with α_A=1.0, α_B=1.0 | 3h | T4 GPU |
| Evaluate Expert C | 30 min | T4 GPU |
| **Total Phase 3** | **4h** | |

### Phase 4: Routing Evaluation (1h CPU)

| Task | Time | Hardware |
|:-----|:----:|:---------|
| Diversity analysis | 15 min | CPU |
| Correctness routing (89-d) | 20 min | CPU |
| Selective routing | 15 min | CPU |
| Report generation | 10 min | CPU |

### Total Budget

| Component | Time |
|:----------|:----:|
| Phase 0 (pre-flight) | 0.5h CPU |
| Phase 1 (evaluate A) | 0.5h GPU |
| Phase 2 (train B + sweep) | 3.5h GPU |
| Phase 3 (train C) | 4h GPU |
| Phase 4 (evaluation) | 1h CPU |
| **Total** | **~9h GPU, ~1.5h CPU** |

Fits comfortably within a single Kaggle session (30h GPU quota).

---

## 16. Appendix: WeightedDataset Implementation

```python
"""
data/weighted_dataset.py — Weighted dataset wrapper for boosting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WeightedDataset(Dataset):
    """
    Wraps a base dataset and adds per-sample loss weights.

    The wrapped dataset returns (image, target, weight) tuples.
    The weight is a float32 scalar that the trainer uses to scale
    the loss for each sample.

    Args:
        base_dataset: Dataset returning (image, target) tuples.
        weights: 1D numpy array of shape (N,) with per-sample weights.
                 Must be non-negative. length must equal len(base_dataset).

    Usage:
        base = LongTailCIFAR100(...)
        weights = np.ones(len(base))  # start uniform
        weights[error_mask] = 2.0     # upweight errors
        weighted = WeightedDataset(base, weights)
        loader = DataLoader(weighted, batch_size=128, shuffle=True)
        # Now loader yields (images, targets, weights)
    """

    def __init__(self, base_dataset: Dataset, weights: np.ndarray):
        assert len(base_dataset) == len(weights), (
            f"Dataset has {len(base_dataset)} samples but weights has {len(weights)}"
        )
        assert np.all(weights >= 0), "Weights must be non-negative"
        self.base_dataset = base_dataset
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        return image, target, self.weights[idx]
```

---

*Plan generated for the Expert Method project (CIFAR-100-LT). The boosting-style approach is designed to create routing-relevant features by making each expert's training error-aware. Implementation is structured in 5 phases over ~10h GPU budget.*
