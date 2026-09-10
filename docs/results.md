# Results — CIFAR-100-LT Proper Split

> Definitive results on the corrected CIFAR-100-LT protocol (LT subsample full 50K → 80/20 train/val → evaluate on CIFAR-100 test set). This is the single source of truth for all accuracy numbers.

---

## 1. Per-Expert Performance (CIFAR-100 Test Set)

**Class group definitions (based on training counts):**
- **Head (30 classes):** ≥100 training samples (range: 102–410)
- **Med (36 classes):** 20–100 training samples (range: 21–96)
- **Tail (34 classes):** <20 training samples (range: 4–18)

| Expert | BA | Head | Med | Tail | ECE | Avg Conf |
|--------|:--:|:----:|:---:|:----:|:---:|:--------:|
| **PaCo** ⭐ | **41.17%** | **71.97%** | **42.86%** | 12.21% | 0.244 | 0.655 |
| **BalancedSoftmax** | 39.99% | 59.50% | 40.08% | **22.68%** | 0.266 | 0.666 |
| **LAL** | 39.96% | 59.57% | 39.53% | **23.12%** | 0.263 | 0.663 |
| **Mixup** | 37.52% | 70.13% | 37.97% | 8.26% | **0.093** | 0.468 |
| **CE** | 36.64% | 66.37% | 37.89% | 9.09% | 0.382 | 0.748 |

**Three clear paradigms:**
1. **PaCo** (contrastive) — best overall BA, dominates Head/Med, weak on Tail
2. **LAL/BS** (logit-adjusted) — sacrifice Head for Tail (+11% Tail over PaCo)
3. **Mixup** (interpolation) — best calibration by far, but weakest accuracy

---

## 2. Diversity Analysis

### 2.1 Correctness Breakdown (3 experts: LAL + PaCo + Mixup)

| Pattern | Count | Percentage |
|---------|:-----:|:----------:|
| All 3 correct | 1,985 | 19.9% |
| Exactly 2 correct | 1,014 | 10.1% |
| Exactly 1 correct | 1,461 | 14.6% |
| All 3 wrong | **4,418** | **44.2%** |
| At least 1 correct | 5,582 | 55.8% |

### 2.2 Correctness Breakdown (5 experts)

| Pattern | Count | Percentage |
|---------|:-----:|:----------:|
| All 5 correct | 1,985 | 19.9% |
| Exactly 4 correct | 948 | 9.5% |
| Exactly 3 correct | 774 | 7.7% |
| Exactly 2 correct | 1,014 | 10.1% |
| Exactly 1 correct | 1,461 | 14.6% |
| All 5 wrong | **3,818** | **38.2%** |

Adding CE and BS reduces the all-wrong ceiling from 44.2% → 38.2% (6 pp).

### 2.3 Pairwise Agreement (Cohen's κ)

| Pair | Agreement | κ | Per-class r |
|------|:---------:|:--:|:-----------:|
| LAL ↔ PaCo | 42.48% | **0.417** | 0.919 |
| LAL ↔ Mixup | 41.94% | **0.412** | 0.922 |
| LAL ↔ CE | 44.55% | **0.438** | 0.917 |
| LAL ↔ BalancedSoftmax | 45.45% | **0.449** | 0.947 |
| Mixup ↔ PaCo | 48.64% | **0.477** | 0.973 |
| Mixup ↔ CE | 48.53% | **0.475** | 0.977 |
| Mixup ↔ BalancedSoftmax | 42.37% | **0.416** | 0.923 |
| PaCo ↔ CE | 47.46% | **0.465** | 0.972 |
| PaCo ↔ BalancedSoftmax | 42.12% | **0.414** | 0.917 |
| CE ↔ BalancedSoftmax | 44.05% | **0.434** | 0.917 |

All κ values between 0.41 and 0.48 — moderate, healthy diversity. No pair exceeds the κ < 0.80 threshold.

### 2.4 Unique Contributions (3 experts)

| Expert | Only This Expert Correct | % of Samples |
|--------|:------------------------:|:------------:|
| LAL | 726 | 7.26% |
| PaCo | 627 | 6.27% |
| Mixup | 395 | 3.95% |
| **Total unique coverage** | **1,748** | **17.48%** |

---

## 3. Ensemble Baselines

### 3.1 3-Expert Set (LAL + PaCo + Mixup)

| Method | BA | Head | Med | Tail | vs Uniform |
|:-------|:--:|:----:|:---:|:----:|:----------:|
| **Uniform avg** | **45.68%** | 74.6% | 48.2% | 17.4% | — |
| Product (geometric) | 45.68% | 74.6% | 48.2% | 17.4% | +0.00% |
| Optimal fixed weights | 44.89% | 74.2% | 46.8% | 17.0% | −0.79% |
| **Oracle** | **55.79%** | — | — | — | **+10.11%** |

**Optimal weights:** [LAL=0.35, PaCo=0.30, Mixup=0.35] — essentially uniform.

### 3.2 5-Expert Set (All)

| Method | BA | Head | Med | Tail | vs Uniform |
|:-------|:--:|:----:|:---:|:----:|:----------:|
| **Uniform avg** | **46.07%** | 74.0% | 48.4% | 18.9% | — |
| Product (geometric) | 46.07% | 74.0% | 48.4% | 18.9% | +0.00% |
| Optimal fixed weights | 46.39% | 73.4% | 48.4% | 20.4% | +0.32% |
| **Oracle** | **61.82%** | — | — | — | **+15.75%** |

**Optimal weights (5 experts):** [LAL=0.20, Mixup=0.15, PaCo=0.30, CE=0.05, BS=0.30]

---

## 4. Comparison: Old Split vs Proper Split

| Metric | Old (Flawed) | Proper | Δ |
|--------|:------------:|:------:|:-:|
| Best expert BA (PaCo) | 49.28% | 41.17% | −8.11% |
| 3-expert uniform avg | 51.12% | 45.68% | −5.44% |
| 3-expert opt fixed | 52.58% | 44.89% | −7.69% |
| 3-expert oracle | ~62.9% | 55.79% | −7.1% |
| All-wrong (3 experts) | ~37.1% | **44.2%** | +7.1% |
| Product gain over uniform | +0.82% | +0.00% | −0.82% |
| Opt fixed gain over uniform | +1.46% | −0.79% | −2.25% |

**The proper split is fundamentally harder.** All-wrong ceiling is 7 pp higher. Product combination adds nothing. Optimal fixed weights don't beat uniform.

---

## 5. Implications for Routing

1. **Routing headroom is extremely limited.** With 44.2% all-wrong, at most 55.8% of samples can benefit from routing. The oracle gap is +10.11% (3 experts), but requires perfect per-sample selection.

2. **Optimal fixed weights don't beat uniform** for 3 experts — unlike the old split where they provided a strong +1.46% baseline. Even a tuned static ensemble can't improve over simple averaging.

3. **PaCo vs LAL/BS creates a natural tradeoff:** PaCo dominates Head/Med (72%/43%) but is weak on Tail (12%). LAL/BS are weaker on Head (60%) but much stronger on Tail (23%). A class-aware or confidence-based router could exploit this.

4. **Routing methods from the old split** (89-d, 92-d, selective, pairwise) were re-evaluated on this split. **None beat uniform averaging.**

5. **CE and BalancedSoftmax add marginal value.** The 5-expert ensemble only gains +0.39% over 3-expert uniform. Optimal weight for CE is only 0.05.

6. **Definitive conclusion: Frozen-expert routing cannot beat uniform averaging on the proper split.** After testing 14 routing methods, the best routing method ties uniform at 45.68%.

---

## 6. Training Details

| Expert | Epochs | LR | Batch | Scheduler | Best Val BA | Notes |
|--------|:------:|:--:|:-----:|:----------:|:-----------:|-------|
| LAL | 200 | 0.1 | 128 | Cosine | ~43% | τ=1.0 |
| BalancedSoftmax | 200 | 0.1 | 128 | Cosine | ~43% | — |
| PaCo | 400 | 0.05 | 256 | Step [320,360] | 43.15% | α=0.01, t=0.05, K=1024, dim=32 |
| Mixup | 200 | 0.1 | 128 | Cosine | ~41% | α=1.0 |
| CE | 200 | 0.1 | 128 | Cosine | ~40% | — |

All trained with SGD (momentum=0.9, nesterov=True, weight_decay=5e-4).

---

## 7. Data Split Summary

- **Full CIFAR-100 train set:** 50,000 samples (500/class)
- **After LT subsampling (IR=100):** 10,847 samples
- **Training set (80%):** 8,677 samples (class counts: 4–410)
- **Validation set (20%):** 2,170 samples (class counts: 1–103)
- **Test set:** 10,000 samples (100/class, balanced)
