# Research Findings — Verified Root Causes & Experiment Catalog

> Consolidated record of what has been tried and why it failed. These findings were established on the original (flawed) data split and should be re-verified on the proper split, but the fundamental structural problems (feature learning gap, all-wrong ceiling, lone dissenter paradox) are likely irreducible with frozen experts regardless of data split.

---

## 1. Verified Root Causes

### 1.1 Feature Learning Gap (19.06%) — 🔴 CRITICAL

**Statement:** The 192-d concatenated backbone features encode **class identity**, not "which expert's decision boundary will be correct for this sample." A learned router on these features cannot reliably predict which expert to select.

**Evidence (corrected after data index bug fix):**

| Metric | Value |
|--------|-------|
| Oracle-weighted routing (perfect soft routing) | 63.04% |
| Learned soft gate (linear: 192-d → 3 weights, NLL) | 43.98% |
| **Feature learning gap** | **19.06%** |

The learned gate collapsed to weights [1.00, 0.00, 0.00] — always picks LAL, achieving exactly LAL's BA (43.98%). A 3-way classifier trained on backbone features to predict "which expert is correct?" achieves only 37.11% BA (vs chance 33.3%) on the 961 samples where exactly one expert is correct.

### 1.2 All-Wrong Ceiling (37.1%) — 🔴 FUNDAMENTAL

**Statement:** 37.1% of validation samples have ALL three experts predicting the wrong class. No routing method can salvage these.

| Scenario | % of Samples | Routing Can Help? |
|----------|:------------:|:-----------------:|
| Exactly 0 experts correct | **37.1%** | ❌ No — all wrong |
| Exactly 1 expert correct | 19.2% | ✅ Clean routing signal |
| Exactly 2 experts correct | 16.2% | ⚠️ Ambiguous target |
| Exactly 3 experts correct | 27.5% | ⚠️ Any works |

**Impact:** Uniform averaging is wrong on 48.9% of samples. Of those, only 11.9% have a correct expert available to rescue them. Best possible BA = 63.04% (matches oracle-weighted).

**Root cause:** All three experts use ResNet-32 backbones, see the same input data, and fail on the same hard samples. Tail classes have very few training examples, so all experts lack sufficient signal.

### 1.3 Lone Dissenter Paradox (83.9%) — 🔴 FUNDAMENTAL

**Statement:** In the 596 samples where routing could help (uniform wrong, a correct expert exists), the correct expert is systematically the **least confident** one. Any confidence-based metric points AWAY from the correct expert.

```
Of 596 savable samples (uniform wrong, correct expert exists):
  Correct expert IS the most confident:      96  (16.1%)
  Correct expert is NOT the most confident: 500  (83.9%)
```

**Root cause:** A "savable" sample is one where two experts are confidently wrong and one is uncertainly correct. The correct expert's uncertainty is the VERY REASON it's correct — it didn't commit to the wrong majority opinion. This is unfixable with frozen experts.

### 1.4 Label Ambiguity (69.4%) — 🔴 CRITICAL

**Statement:** 69.4% of "trainable" samples (those where at least one expert is correct) have **multiple correct experts**. This makes 3-way expert selection fundamentally ambiguous — any training target is arbitrary for the majority of samples.

| # Correct | Samples | % of Total | % of Trainable |
|:---------:|:-------:|:-----------:|:--------------:|
| 0 (all wrong) | 1,855 | 37.1% | — |
| 1 (unambiguous) | 961 | 19.2% | **30.6%** |
| 2 (ambiguous) | 810 | 16.2% | **25.7%** |
| 3 (all correct) | 1,374 | 27.5% | **43.7%** |

**Label noise rate for a 3-way classifier: 69.4%.** A 3-way classifier achieves 50.38% BA on the unambiguous subset but collapses to **26.25%** (below chance) when trained on all samples.

### 1.5 Product Captures Most Signal (+0.82%) — 🟠 HIGH

**Statement:** The product-of-experts combination (geometric mean of probabilities) achieves +0.82% over uniform averaging with NO routing. Adding per-sample trust scores adds only +0.10% to +0.30%.

| Method | BA | vs Uniform | Routing? |
|:-------|:--:|:----------:|:--------:|
| Uniform avg | 51.12% | — | No |
| Uniform product | 51.94% | **+0.82%** | No |
| Trust-weighted product (36-d) | 52.43% | +1.19% | Yes (+0.29%) |

**Why product beats average:** The product gives each expert "veto power" over classes it's sure about. An expert that says "p≈0 for class A" vetoes class A. Averaging only uses positive certainty; product uses both positive and negative certainty.

### 1.6 Expert Miscalibration — 🟠 HIGH (Partially Fixed)

| Expert | Raw Avg Confidence | Actual BA | ECE | Temperature |
|--------|:------------------:|:---------:|:---:|:-----------:|
| LAL | 68.5% | 43.98% | 0.2456 | 1.865 |
| PaCo | 54.2% | 49.28% | 0.0492 | 1.347 |
| Mixup | 45.1% | 40.80% | 0.0433 | 1.261 |

Global temperature scaling helps but LAL's overconfidence is class-dependent. Per-class temperatures could improve further.

### 1.7 The 3-Way Comparison Problem — 🔴 CRITICAL

Trust meters predict individual expert correctness well (AUROC 0.84-0.89) but cannot reliably rank all three experts. Pairwise ranking AUROC peaks at 0.85 for LAL vs Mixup but is only 0.745 for LAL vs PaCo. The 3-way comparison ("which expert is best for THIS sample?") is fundamentally harder than binary prediction ("is THIS expert correct?").

### 1.8 Disagreement Routing Disproved — 🔴 CRITICAL

The dissenter (expert whose top-1 prediction differs from the other two) is correct only **15.8%** of the time in 2-1 splits. Disagreement routing achieves 40.72% BA — far worse than uniform (51.12%). "Dissenter in prediction" ≠ "dissenter in confidence."

---

## 2. Experiment Catalog (25+ Methods, 5 Rounds)

### Round 1 — Basic Routing Methods (All Failed or Marginal)

| Method | BA | vs Uniform | Verdict |
|:-------|:--:|:----------:|---------|
| Confidence routing (raw) | 50.64% | −0.48% | ❌ LAL's miscalibration dominates |
| Confidence routing (calibrated) | 51.44% | +0.32% | ⚠️ Small gain |
| Entropy routing | 50.10% | −1.02% | ❌ Worse than confidence |
| MLP router (192-d features, 5K train) | 49.32% | −1.80% | ❌ Overfits |
| MLP router (192-d features, 45K train) | 49.82% | −1.30% | ❌ 10× data doesn't help |
| 3-way classifier | 26.25% | −24.87% | ❌ Below chance |
| Disagreement routing | 40.72% | −10.40% | ❌ Dissenter is wrong most of the time |
| Gated mixture (24-d features, NLL) | 43.98% | −7.14% | ❌ Collapsed to always pick LAL |
| Trust-weighted product (36-d) | 52.43% | +1.19% | ⚠️ Routing contributes only +0.29% |
| Optimal fixed weights | 52.58% | +1.46% | Reference (not routing) |

### Round 2 — Enriched Feature Sets

| Method | BA | vs Uniform | Verdict |
|:-------|:--:|:----------:|---------|
| 24-d correctness-prediction | 51.70% | +0.58% | ❌ Below target |
| **★ 89-d enriched correctness** | **52.42%** | **+1.30%** | **✅ Meets +1% target** |
| KL-divergence routing | 38.58% | −12.54% | ❌ |
| Energy-based routing | 49.28% | −1.84% | ❌ |
| Pairwise majority voting | 50.84% | −0.28% | ❌ |
| Bayesian prior routing | 52.34% | +1.22% | ❌ Collapsed to fixed weights |
| Temperature-scaled features | 52.00% | +0.88% | ❌ Worse than unscaled |
| MLP trust meters | 52.10% | +0.98% | ❌ No improvement over LR |
| Adaptive threshold routing | 52.42% | +1.30% | ⚠️ Gain from fixed weights fallback |

### Round 3 — Learning-to-Rank & Combined

| Method | BA | vs Uniform | Verdict |
|:-------|:--:|:----------:|---------|
| Augmentation consistency | 51.30% | +0.18% | ❌ Below 0.5% threshold |
| Tournament soft (pairwise LR) | 52.10% | +0.98% | ❌ Underperforms 89-d |
| MLP pairwise comparators | 50.66% | −0.46% | ❌ Overfits |
| **★ 92-d combined (89-d + pairwise)** | **52.49%** | **+1.37%** | **✅ Best routing method** |
| Meta-router (9-d features) | 52.40% | +1.28% | ⚠️ Below 92-d |

### Round 4 — TTA, Gradient Sensitivity & Selective

| Method | BA | vs Uniform | Verdict |
|:-------|:--:|:----------:|---------|
| TTA uniform avg (N=32) | 52.54% | +1.42% | ⚠️ Raises baseline, shrinks routing fraction |
| TTA opt fixed | 53.54% | +2.42% | Reference |
| 60-d TTA routing | 53.00% | +0.46% (vs TTA uni) | ❌ Routing fraction shrinks |
| Gradient sensitivity (95-d log_grad) | 52.52% | +1.40% | ⚠️ Signal redundant with 92-d |
| **★ Selective 92-d (thresh=0.35)** | **52.70%** | **+1.58%** | **✅ Beats opt fixed by +0.12%** |
| 392-d hybrid TTA | 53.22% | +0.68% (vs TTA uni) | ⚠️ −0.32% vs TTA opt |

### Round 5 — GDDR & Cluster Routing (Both Failed)

| Method | BA | Verdict |
|:-------|:--:|---------|
| GDDR (gradient alignment) | 46.98% | ❌ Gradients near-orthogonal in 3072-d |
| Cluster routing (feature clustering) | 52.56% | ❌ Ties global opt fixed (+0.08%, within noise) |
| Cluster routing (agreement-pattern) | 51.76% | ❌ Below global opt fixed |
| Cluster routing (soft clustering) | 52.40% | ❌ Ties global opt fixed |

---

## 3. What Was Conclusively Ruled Out

| Approach | Reason Ruled Out |
|----------|------------------|
| Insufficient training data | 10× data gave +0.50% — not root cause |
| Wrong router architecture | MLP, linear, soft gate, confidence, pairwise — all fail |
| Wrong input representation | Features, logits, probs, confidences — all fail |
| Disagreement routing | Dissenter correct only 15.8% of 2-1 splits |
| Augmentation consistency | +0.18% gain, below 0.5% threshold |
| Pairwise ranking (tournament) | Underperforms independent trust meters |
| MLP pairwise comparators | Overfits on small XOR training sets |
| Gradient alignment (GDDR) | 3072-d directions near-orthogonal (cos sim ≈ 0.03) |
| Cluster-based adaptive weighting | Per-cluster weights differ but gain within noise |
| TTA averaging | Raises absolute BA but hurts routing fraction |

---

## 4. Summary Table: All-Wrong Ceiling & Label Ambiguity

| Metric | Old Split | Proper Split |
|--------|:---------:|:------------:|
| All-wrong (3 experts) | 37.1% | **44.2%** |
| All-wrong (5 experts) | — | **38.2%** |
| Label ambiguity (% of trainable with >1 correct) | 69.4% | — |
| Oracle (3 experts) | 62.9% | 55.79% |

The proper split is substantially harder. All-wrong ceiling is 7 points higher.
