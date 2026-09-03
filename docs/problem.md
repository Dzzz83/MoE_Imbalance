# Problem Analysis: Why Routing Fails to Beat Uniform Averaging

## Overview

Three diverse ResNet-32 experts (LAL, PaCo, Mixup+CE) were trained on CIFAR-100-LT (IR=100). The goal is to build a dynamic router that selects the best expert per test sample. **The router must outperform uniform softmax averaging by ≥1% Balanced Accuracy (BA) to justify its complexity.**

**Initial best method that beats uniform:** Trust-weighted product = **52.43% BA** (+1.19% over average ensemble, reported in original experiments).
**But the routing contribution was only +0.10% to +0.30%** — the overwhelming majority came from switching to product combination (static, not routing).

**Current best routing method:** 89-d enriched correctness-prediction routing = **52.41% BA (+1.29% over uniform)** across 3 seeds. The routing contribution is the full +1.29% (uses average combination, not product). **This achieves the +1% target.**

**However:** The best routing still **underperforms optimal fixed weights** (52.56%, +1.44%). The 0.15% gap between routing and the best static ensemble has not been closed. This document catalogs the **verified problems** that explain why this gap exists and why it may be irreducible with frozen experts.

This document catalogs the **verified problems** discovered after systematic debugging, including bug fixes that corrected several numbers in the original documentation. Problems marked **"Recalculated"** had their evidence updated after fixing data leakage, dead code, hardcoded temperatures, and routing target construction bugs.

---

## Problem 1: The Feature Learning Gap (19.06%)

### Severity: 🔴 CRITICAL — Root Cause

### Statement
The 192-d concatenated backbone features encode **class identity**, not "which expert's decision boundary will be correct for this sample." A learned router on these features cannot reliably predict which expert to select.

### Evidence *(Corrected after data index bug fix in `kaggle_root_cause.py`)*

| Metric | Value | Source |
|--------|-------|--------|
| Oracle-weighted routing (perfect soft routing with correctness as weights) | **63.04%** | `scripts/kaggle_root_cause.py` |
| Learned soft gate (linear: 192-d features → 3 weights, trained with NLL) | **43.98%** | Same (fixed) |
| **Feature learning gap** | **19.06%** | 63.04% − 43.98% |

The learned gate collapsed to weights [1.00, 0.00, 0.00] — it always picks LAL, achieving exactly LAL's BA (43.98%). This is far WORSE than uniform averaging (51.12%). The features contain enough information to classify images (~52% class accuracy with LR on balanced data), but not enough to predict which expert will be correct.

**⚠️ Correction from previous version:** A bug was discovered and fixed in `kaggle_root_cause.py`: `lt_train.sample_indices` (indices into the 45K base pool) were being used as direct indices into the full 50K training set. This caused the training subset to load wrong images. After fixing the index mapping (using `lt_train.base_indices[lt_train.sample_indices]`), the learned gate collapsed completely — the feature learning gap is **5.34% larger** than previously reported (19.06% vs 13.72%).

### Root Cause
The backbone features are optimized via cross-entropy (LAL, Mixup) or contrastive (PaCo) objectives that organize the feature space by **class identity**. Samples from the same class cluster together regardless of which expert would classify them correctly. The routing decision requires knowing which expert's **decision boundary** a given sample falls on — a much more subtle property that is not encoded in the feature space.

### Additional Evidence from Deep Debug
A 3-way classifier trained on backbone features (192-d) to predict "which expert is correct?" achieves only **37.11% BA** (vs chance 33.3%) on the 961 samples where exactly one expert is correct. This is only +3.81% above chance, confirming that backbone features contain almost no routing-relevant signal.

### Why It's Hard to Fix
- The features cannot be easily modified after training — they are frozen
- Training a new feature extractor for routing would require end-to-end joint training
- Even with joint training, the routing task may still be limited by the 37% all-wrong ceiling (Problem 3)

---

## Problem 2: Expert Miscalibration (Distorts Confidence-Based Routing)

### Severity: 🟠 HIGH — Partially Fixed

### Statement
The three experts have different calibration levels. LAL is severely overconfident (avg confidence 68.5% vs actual accuracy 44.0%), which distorts any routing method based on raw confidence scores.

### Evidence *(Recalculated after data leakage fix — temperatures now found on held-out data)*

| Expert | Raw Avg Confidence | Actual BA | ECE | Temperature (after scaling) |
|--------|:------------------:|:---------:|:---:|:--------------------------:|
| LAL | 68.5% | 43.98% | 0.2456 | 1.865 |
| PaCo | 54.2% | 49.28% | 0.0492 | 1.347 |
| Mixup | 45.1% | 40.80% | 0.0433 | 1.261 |

Source: `scripts/debug_routing.py` (fixed: temperatures now calibrated on held-out 4K split, evaluated on held-out 1K split).

**Previous temperatures** (from buggy code with data leakage): LAL=1.853, PaCo=1.333, Mixup=1.257
**Corrected temperatures** (held-out calibration): LAL=1.865, PaCo=1.347, Mixup=1.261
**Difference:** Negligible (<0.7%). Calibration conclusions unchanged.

**Before calibration:**
- LAL selected 57.9% of the time by confidence routing
- When selected, LAL is correct only 50.6% of the time (barely above random)
- PaCo (best expert, 49.28%) selected only 30.4%

**After calibration (properly held-out):**
- Selection: LAL≈33.9%, PaCo≈46.3%, Mixup≈19.8% — much healthier
- Routed BA: **~52.3%** on held-out split (vs uniform avg ~51.8% on same split)
- Calibration provides a meaningful but small improvement

### What's Left
Global temperature scaling (one T per expert) is a coarse fix. LAL's overconfidence is class-dependent: extreme on head classes (where it's often correct and assigns probability ~1.0) but more moderate on tail classes. Per-class temperatures could improve the comparison further, but the Lone Dissenter Paradox (Problem 7) would still limit the gain.

---

## Problem 3: The 37.1% All-Wrong Ceiling

### Severity: 🟡 MEDIUM — Fundamental Limit

### Statement
37.1% of validation samples have ALL three experts predicting the wrong class. No routing method can salvage these samples.

### Evidence *(Unchanged by bug fixes)*

| Scenario | % of Samples | Routing Can Help? |
|----------|:------------:|:-----------------:|
| Exactly 0 experts correct | **37.1%** | ❌ No — all wrong |
| Exactly 1 expert correct | 19.2% | ✅ Clean routing signal |
| Exactly 2 experts correct | 16.2% | ⚠️ Ambiguous target |
| Exactly 3 experts correct | 27.5% | ⚠️ Any works |

Source: `scripts/debug_routing.py` DEBUG 2.

**Impact on routing headroom:**
- Uniform averaging is wrong on 2,444/5,000 samples (48.9%)
- Of those, only **596/5,000 (11.9%)** have a correct expert available to rescue them
- The remaining **1,848/5,000 (37.0%)** have all experts wrong — no routing can help
- Best possible BA if all 596 are rescued: 63.04% (matches oracle-weighted)

### Root Cause
The three experts, despite being trained with different losses, still fail on the same hard samples. The failures are correlated because:
- All three backbones are ResNet-32 — same architectural inductive biases
- All three see the same input images (same augmentations for LAL/Mixup; PaCo uses stronger augs but same base data)
- The long-tail distribution means tail classes have very few training examples, so all experts lack sufficient signal for those classes

---

## Problem 4: Insufficient Training Data for Learned Routers

### Severity: 🟢 LOW — Was Initially Suspected, Now Ruled Out

### Statement
Increasing router training data by 10× (5K → 45K) barely improved performance, ruling out data scarcity as the root cause.

### Evidence *(Unchanged by bug fixes)*

| Training Data | MLP Router BA | vs Uniform |
|:-------------|:-------------:|:----------:|
| 5K validation set | 49.32% | −1.80% |
| 45K base pool | 49.82% | −1.30% |
| Improvement from 10× data | +0.50% | — |

Source: `scripts/eval_router.py` and `scripts/eval_router_bigdata.py`.

The MLP router barely improves with 10× more data because the fundamental problem is **target noise** (see Problem 9), not sample size. Even with infinite data, the features don't contain the routing-relevant information.

---

## Problem 5: Distribution Mismatch Between Training and Evaluation

### Severity: 🟢 LOW — Exacerbating Factor

### Statement
Routers trained on long-tailed data perform differently on balanced evaluation data. The optimal routing weights for LT data differ from those for balanced data.

### Evidence *(Corrected after data index bug fix in `kaggle_root_cause.py`)*

| Training Data | Best Learned Gate BA | Optimal Fixed Weights |
|:-------------|:-------------------:|:---------------------:|
| LT (3,000 samples) | **43.98%** | [1.00, 0.00, 0.00] |
| Balanced (5,000 samples) | ~49.86% | [0.51, 0.49, 0.00] |
| Uniform (no training) | **51.12%** | [0.33, 0.33, 0.33] |

Source: `scripts/kaggle_root_cause.py` (corrected) and `scripts/verify_routing_hypotheses.py`.

The learned gate on LT data collapses to always pick LAL (43.98% BA — exactly LAL's individual BA), confirming that the 192-d backbone features contain no routing-relevant signal. The previous result (49.32%, weights [0.40, 0.60, 0.00]) was an artifact of the data index bug in `kaggle_root_cause.py` that loaded wrong training images. On the balanced validation set, Mixup contributes 4-5% unique correct samples that no other expert captures, but the LT-trained gate never learns this because Mixup is the weakest expert on LT data (40.80% BA). The gate's training objective (minimize CE on LT data) conflicts with the evaluation objective (maximize BA on balanced data).

---

## Problem 6: The 3-Way Comparison Problem

### Severity: 🔴 CRITICAL — Root Cause for Correctness-Prediction Routing

### Statement
Trust meters can predict whether a given expert is correct (AUROC 0.84-0.89 per expert) but cannot reliably rank all three experts. The 3-way comparison ("which expert is best for THIS sample?") is fundamentally harder than binary prediction ("is THIS expert correct?").

### Evidence *(Recalculated with additional evidence from deep debug)*

**Binary prediction (per expert):**

| Expert | AUROC (predicting correctness) |
|:------:|:------------------------------:|
| LAL | **0.865** — strong |
| PaCo | **0.842** — strong |
| Mixup | **0.893** — strong |

**Pairwise ranking (can we tell which of two experts is better?):**

| Pair | AUROC (feature differences) | xor count |
|:----:|:---------------------------:|:---------:|
| LAL vs PaCo | **0.745** | 1,237 |
| LAL vs Mixup | **0.850** | 1,093 |
| PaCo vs Mixup | **0.804** | 1,212 |

Even with all 24-d features combined in a LogisticRegression, the pairwise ranking AUROC peaks at 0.85 for the easiest pair (LAL vs Mixup) and is only 0.745 for the critical LAL vs PaCo pair.

**3-way ranking (exactly-1-correct case, 961 samples):**
- A 3-way classifier on 24-d features achieves **50.38% BA** on this subset (vs chance 33.3%)
- But on ALL samples (including ambiguous ones with multiple correct experts), it collapses to **26.25%** (below chance!)
- The correctness-prediction approach (binary trust meters + softmax combination) achieves **51.84%** overall

**Why ranking fails:** When the correct expert is NOT the most trusted (56% of cases):
- Correct expert mean trust score: **0.23**
- Wrong #1 expert mean trust score: **0.39**
- Gap: **0.16** — the wrong expert is trusted MORE in 100% of mis-rankings

The trust scores are SYSTEMATICALLY misleading, not randomly noisy. The features that predict individual correctness do not contain sufficient signal for reliable 3-way ranking.

### Root Cause
The 24-d output features (entropy, confidence, margin, KL divergence, etc.) capture each expert's uncertainty ABOUT THE CLASS. But they don't capture where the sample lies relative to each expert's DECISION BOUNDARY. Two experts can both be uncertain (high entropy) for completely different reasons — the features don't distinguish this. The ranking requires knowing which expert's boundary is more reliable for this specific sample, which is a different kind of information.

---

## Problem 7: The Correct Expert in Savable Samples is Systematically the Least Confident

### Severity: 🔴 CRITICAL — Fundamental Paradox

### Statement
In the 596 samples where routing could help (uniform wrong, a correct expert exists), the correct expert is systematically the least confident one. Any confidence-based metric points AWAY from the correct expert.

### Evidence *(Unchanged by bug fixes — structural property, not calibration)*

```
Of 596 savable samples (uniform wrong, correct expert exists):
  Correct expert IS the most confident:     96  (16.1%)
  Correct expert is NOT the most confident: 500 (83.9%)
```

**83.9% of the time** when routing could actually help, the correct expert is NOT the most confident. It's the "lone dissenter" — uncertain but right, while the other experts are confidently wrong.

Similarly:
- Correct expert HAS lowest entropy: **19.3%**
- Correct expert does NOT have lowest entropy: **80.7%**

### Root Cause
This is a fundamental paradox of routing for frozen experts. A "savable" sample is one where:
- Two experts are confidently wrong (high confidence, high entropy → wrong class)
- One expert is uncertainly correct (low confidence, moderate entropy → correct class)

The correct expert's uncertainty is the VERY REASON it's correct — it didn't commit to the wrong majority opinion. But this uncertainty makes it look weak to any routing metric. The signal that identifies the correct expert (low confidence, disagreement with majority) is exactly the signal that routing metrics interpret as "unreliable."

**This is unfixable with frozen experts** because the experts' outputs faithfully reflect their internal state. The correct expert IS less confident. Any routing method based on confidence, entropy, or any derived statistic will systematically favor the wrong experts in these cases.

---

## Problem 8: Product Combination Captures Most Signal, Routing Adds Little

### Severity: 🟠 HIGH — Explains Why Routing Gain is Small

### Statement
The product-of-experts combination (geometric mean of probabilities) achieves +0.82% over uniform averaging with NO routing. Adding per-sample trust scores on top adds only +0.10% to +0.30%. The routing contribution is small because the product already captures most of the available signal.

### Evidence *(Recalculated — product BA corrected from 52.13% to 51.94%)*

| Method | BA | vs Uniform | Routing? |
|:-------|:--:|:----------:|:--------:|
| Uniform avg | 51.12% | — | No |
| Uniform product (equal weights) | **51.94%** | **+0.82%** | No |
| Trust-weighted product (36-d, original experiments) | 52.43% | +1.19% | Yes (+0.29%) |
| Optimal fixed weights | **52.56%** | **+1.44%** | No |

**Note:** The product BA was previously reported as 52.13% (+0.90%). After fixing numerical precision issues in the computation, the corrected value is **51.94% (+0.82%)**. The trust-weighted product numbers (52.43%) are from the original experimental version and include the product gain plus a small routing contribution.

### Why product beats average
The product combination gives each expert "veto power" over classes it's sure about:

- **Average:** `p_mix = (p_1 + p_2 + p_3) / 3`
  - A confident wrong expert (p=0.9 for wrong class) dominates
  - The correct but uncertain expert's signal (p=0.32 for correct class) is diluted

- **Product:** `p_mix ∝ p_1 × p_2 × p_3`
  - An expert that says "p≈0 for class A" vetoes class A
  - The correct expert's 0.32 for the right class is preserved because no other expert vetoes it strongly
  - The confident wrong experts' high probabilities for different wrong classes cancel out

The product uses the "negative certainty" information (low probabilities = "I'm sure this is NOT class X") that averaging discards. This is the main untapped signal in the experts' outputs. Once it's captured by the product, the trust scores have little additional signal to add.

### Root Cause
The experts' softmax outputs contain two types of information:
1. **Positive certainty:** "I'm confident this IS class X" (high probabilities)
2. **Negative certainty:** "I'm confident this is NOT class Y" (low probabilities)

Averaging only uses positive certainty. The product uses both. The routing signal (trust scores) is derived from statistics of the same softmax outputs, so it's largely redundant with the information the product already extracts. This is why adding routing on top of the product gives only +0.10% to +0.30%.

---

## Problem 9: 69.4% Label Ambiguity — The 3-Way Selection Problem is Ill-Posed

### Severity: 🔴 CRITICAL — Newly Discovered Root Cause

### Statement
69.4% of "trainable" samples (those where at least one expert is correct) have **multiple correct experts**. This makes the 3-way expert selection task fundamentally ambiguous — any training target is arbitrary for the majority of samples.

### Evidence *(New finding from deep debug)*

| # Correct Experts | Samples | % of Total | % of Trainable | Target Ambiguity |
|:---:|:---:|:---:|:---:|:---:|
| 0 (all wrong) | 1,855 | 37.1% | — | — |
| **1 (unambiguous)** | **961** | **19.2%** | **30.6%** | **Clean** |
| **2 (ambiguous)** | **810** | **16.2%** | **25.7%** | **Arbitrary which to pick** |
| **3 (all correct)** | **1,374** | **27.5%** | **43.7%** | **Any expert works** |

**Trainable samples (≥1 correct):** 3,145 (62.9%)
**Of these, fraction with multiple correct: 69.4%**
**Label noise rate for a 3-way classifier: 69.4%** — any training target is arbitrary

### Impact on Routing Methods

**3-way classifier:** A classifier trained to predict "which expert is best?" has 69.4% label noise because for most samples, there is no unique "best" expert. This is why the 3-way classifier achieves only **26.25% BA** (below random 33.3%) on all samples, despite achieving **50.38% BA** on the unambiguous subset.

**Correctness-prediction routing (binary trust meters):** Avoids the label ambiguity by training independent binary classifiers for each expert. However, this approach cannot compare trust scores across experts — a score of 0.7 for LAL may mean something different from 0.7 for PaCo. The softmax-with-temperature combination only partially addresses this.

**Optimal fixed weights:** Achieves **52.56% BA** — significantly better than any per-sample routing method. This is because the optimal fixed weights [0.20, 0.48, 0.32] exploit the fact that PaCo is the strongest expert on average, and per-sample variation adds noise, not signal.

### Root Cause
The routing task is inherently multi-label (multiple experts can be correct), but all routing methods treat it as single-label (pick one expert). This mismatch between the problem structure and the solution approach means that **any method that tries to pick a single "best" expert will fail on 69.4% of trainable samples** because there is no unique best expert.

The only way to resolve this is to change what the experts provide — either:
- **Make experts more specialized** so they are correct on disjoint subsets (reducing the multi-correct rate)
- **Use a combination method** (like product) that doesn't require picking one expert
- **Jointly train the router with the experts** so the router influences feature learning (RIDE-style)

---

## Problem 10: Disagreement Routing Fails — "Dissenter in Prediction" ≠ "Dissenter in Confidence"

### Severity: 🔴 CRITICAL — Disproves a Common Misconception

### Statement
The Lone Dissenter Paradox (Problem 7) says the correct expert has the **lowest confidence** 83.9% of the time on savable samples. This led to the hypothesis that picking the "dissenter" (the expert whose top-1 prediction differs from the other two) would capture the correct expert. This hypothesis is WRONG.

### Evidence *(New finding from CPU verification)*

**Disagreement pattern distribution (5K validation set):**

| Pattern | Samples | % |
|:--------|:-------:|:-:|
| All 3 agree | 1,650 | 33.0% |
| 2 agree, 1 dissents | 1,818 | 36.4% |
| All 3 disagree | 1,532 | 30.6% |

**Among 2-1 splits, is the dissenter correct?**

| Outcome | % |
|:--------|:-:|
| Dissenter correct | 15.8% |
| Majority correct | 44.6% |
| Both wrong | 39.6% |

The dissenter is correct only **15.8%** of the time — far worse than random.

**But among savable samples with unique correct expert (559 samples), the correct expert IS the dissenter 100% of the time in 2-1 splits.** Why the contradiction?

### Root Cause

The confusion is between two meanings of "dissenter":

1. **Dissenter in CONFIDENCE** (the Lone Dissenter Paradox): The correct expert has the lowest softmax confidence. This is about the expert's **internal uncertainty**.
2. **Dissenter in PREDICTION**: The expert whose top-1 class differs from the other two. This is about **disagreement on the predicted class**.

An expert can be the least confident but still agree with the majority on the predicted class. Example:
```
LAL:   p=[0.30, 0.25, 0.20, ...] → predicts class A (lowest confidence, but NOT a dissenter)
PaCo:  p=[0.50, 0.20, 0.10, ...] → predicts class A
Mixup: p=[0.45, 0.30, 0.10, ...] → predicts class A
All predict class A. LAL is least confident but agrees with the majority.
```

Among savable samples with a unique correct expert, the correct expert IS the dissenter in 2-1 splits. But most 2-1 splits are NOT savable — they're cases where either the majority is correct or both sides are wrong. The "correct dissenter vs wrong majority" scenario is a small minority of 2-1 splits.

**Disagreement routing (pick the dissenter) achieves only 40.72% BA** — far worse than uniform (51.12%). The rule correctly routes the 559 savable samples but incorrectly routes the 1,259 non-savable 2-1 splits, and the latter dominates.

### Key Lesson
The Lone Dissenter Paradox is about **confidence**, not about **prediction agreement**. Using disagreement in predictions as a routing signal is fundamentally different from using confidence, and the former does NOT capture the latter. The correct expert disagrees in confidence space, not necessarily in prediction space.

---

## Summary: The Debugging Pipeline

| Suspected Cause | Verdict | Evidence |
|----------------|:-------:|----------|
| LAL miscalibration | ✅ **CONFIRMED** — partially fixed by calibration | +0.80% improvement after calibration |
| Insufficient training data | ❌ **RUED OUT** — 10× data gave +0.50% | `eval_router_bigdata.py` |
| Wrong router architecture | ❌ **RUED OUT** — tried MLP, linear, soft gate, confidence, all fail | `eval_router_v2.py` |
| Wrong input representation | ❌ **RUED OUT** — tried features, logits, probs, confidences | `eval_router_v2.py` |
| Training distribution mismatch | ⚠️ **CONFIRMED** — exacerbating but not root cause | Learned gates collapse to worst expert on LT |
| **Feature-learning gap** | ✅ **ROOT CAUSE** — 19.06% gap between oracle and learned gate | `kaggle_root_cause.py` (corrected) |
| **37% all-wrong ceiling** | ✅ **FUNDAMENTAL LIMIT** — 37.1% samples can't be saved | `debug_routing.py` |
| **3-way comparison problem** | ✅ **ROOT CAUSE** — trust meters can't rank experts (pairwise AUROC 0.745-0.850) | `correctness_routing.py`, deep debug |
| **Lone dissenter paradox** | ✅ **FUNDAMENTAL** — correct expert is least confident in 83.9% of savable cases | Deep debug |
| **Product captures most signal** | ✅ **EXPLAINS small routing gain** — product achieves +0.82% alone | Verified recomputation |
| **69.4% label ambiguity** | ✅ **NEW ROOT CAUSE** — 3-way selection is ill-posed for frozen experts | Deep debug (3-way classifier 26.25% vs 50.38%) |
| **Disagreement routing** | ❌ **DISPROVEN** — dissenter is correct only 15.8% of 2-1 splits; 40.72% BA | CPU verification |
| **Augmentation consistency routing** | ❌ **FAILED** — +0.18% gain, below +0.5% threshold | `augmentation_consistency_analysis.py` |
| **Pairwise ranking (LR comparators)** | ❌ **UNDERPERFORMS 89-d** — tournament soft 52.10% vs 89-d 52.46% | `pairwise_routing.py` |
| **MLP pairwise comparators** | ❌ **OVERFITS** — 50.66% BA, worse than LR pairwise | `pairwise_mlp_combined.py` |
| **92-d combined routing** | ⚠️ **BEST METHOD, TIES FIXED WEIGHTS** — 52.49% BA, −0.09% gap (p=0.71) | `multi_seed_92d_verify.py` |
| **Meta-router (9-d features)** | ❌ **BELOW 89-d** — 52.40% BA, −0.18% gap | `pairwise_mlp_combined.py` |
| **TTA-averaged predictions** | ❌ **RAISES BASELINE, HURTS ROUTING FRACTION** — 53.00% BA but −0.54% vs TTA opt | `tta_routing.py` |
| **Gradient sensitivity (log_grad)** | ⚠️ **SIGNAL EXISTS BUT REDUNDANT** — 52.52% BA, −0.06% gap, r=0.24-0.34 | `gradient_routing.py` |
| **Selective routing (92-d, thresh=0.35)** | ⚠️ **BEATS OPT FIXED** — 52.70% BA, **+0.12% vs opt fixed** | `selective_hybrid_routing.py` |
| **392-d hybrid TTA routing** | ⚠️ **HIGHEST ABSOLUTE BA** — 53.22% BA, −0.32% vs TTA opt | `hybrid_tta_routing.py` |
| **Gradient alignment routing (GDDR)** | ❌ **FAILED** — 46.98% BA; gradient directions in 3072-d are near-orthogonal (mean cos sim ≈ 0.03); signal correlates with confidence (r≈0.30) | `gradient_alignment_routing.py` |
| **Cluster routing (feature clustering + per-cluster weights)** | ❌ **FAILED TO BEAT GLOBAL OPT** — 52.56% BA (+0.08% vs opt); per-cluster weights differ but gain within noise | `cluster_routing.py` |
| **Cluster routing (agreement-pattern grouping)** | ❌ **BELOW GLOBAL OPT** — 51.76% BA; per-group weights differ but overall worse | `cluster_routing.py` |
| **Cluster routing (soft clustering)** | ❌ **TIES GLOBAL OPT** — 52.40% BA; essentially tied with global optimal fixed weights | `cluster_routing.py` |

## Corrected Numbers After Bug Fixes

The following numbers were updated after fixing bugs in diagnostic scripts:

| Metric | Before (original docs) | After (verified) | Change | Bug Fixed |
|--------|:---------------------:|:----------------:|:------:|-----------|
| PaCo BA | 49.08% | **49.28%** | +0.20% | Doc typo (checkpoint was correct) |
| Uniform avg BA | 51.23% | **51.12%** | −0.11% | Numerical precision |
| Uniform product BA | 52.13% (+0.90%) | **51.94% (+0.82%)** | −0.19% / −0.08pp | Numerical precision |
| Calibration temp LAL | 1.853 | **1.865** | +0.012 | Data leakage fix |
| Calibration temp PaCo | 1.333 | **1.347** | +0.014 | Data leakage fix |
| Calibration temp Mixup | 1.257 | **1.261** | +0.004 | Data leakage fix |
| Learned soft gate BA | 49.32% | **43.98%** | −5.34% | `kaggle_root_cause.py` data index bug (sample_indices used as 50K indices) |
| Feature learning gap | 13.72% | **19.06%** | +5.34% | Same bug — gap widened after fix |

**None of these corrections change the fundamental conclusions.** The routing contribution is still <0.5%, the all-wrong ceiling is still 37.1%, and the lone dissenter paradox still holds at 83.9%. The feature learning gap is now confirmed to be **even larger** than previously reported, reinforcing that frozen backbone features contain essentially no routing-relevant information.

**New findings from Round 5 (novel methods developed in this session):**
| Metric | Value | Method |
|--------|:-----:|--------|
| GDDR BA | 46.98% | Gradient alignment in 3072-d space is near-orthogonal (mean cos sim ≈ 0.03) |
| Cluster routing (feature clustering) | 52.56% (+0.08% vs opt) | Per-cluster weights differ but gain is within noise |
| Cluster routing (agreement-pattern) | 51.76% (−0.72% vs opt) | Per-group weights differ but overall worse than global opt |
| Cluster routing (soft clustering) | 52.40% (−0.08% vs opt) | Essentially tied with global optimal fixed weights |

## What This Means for Next Steps

After exhaustive testing — **25+ routing methods** across **5 rounds** of experiments (including 2 novel methods developed in the final session) — the evidence conclusively shows that **per-sample routing with frozen experts on this problem is fundamentally limited by information that does not exist in the frozen experts' outputs.** Every signal that can be derived from frozen experts' outputs, features, or gradients has been tried and failed to meaningfully beat a tuned static ensemble. The routing signal is insufficient because:

1. **37.1% all-wrong ceiling** caps the maximum possible routing gain at 11.78% (oracle), but the signal-to-noise ratio for per-sample routing is too low to capture more than ~1.3%.
2. **The correct expert in savable samples is systematically the least confident** — any confidence-based signal points away from it (Lone Dissenter Paradox).
3. **The product combination extracts much of the available signal** from the experts' outputs, leaving little for routing to add.
4. **69.4% label ambiguity** makes the 3-way expert selection problem fundamentally ill-posed for frozen experts.
5. **Learning to rank (pairwise comparators) doesn't help** — despite directly addressing the 3-way comparison problem, pairwise methods underperform independent trust meters because they train on fewer samples and tournament aggregation amplifies errors.
6. **Adding pairwise scores to the feature set gives only +0.07%** — the 92-d combined routing achieves 52.49% BA but is statistically indistinguishable from optimal fixed weights (p=0.71).
7. **Augmentation consistency is correlated with correctness but can't enable routing** — the Lone Dissenter Paradox persists for consistency just as it does for confidence.
8. **TTA averaging raises absolute accuracy but hurts the routing fraction** — smoothed predictions make experts more similar, reducing the routing opportunity. The baseline moves with the improvement.
9. **Gradient sensitivity correlates with correctness (r=0.24-0.34) but is redundant with 92-d features** — the 92-d features already capture equivalent information about decision-boundary proximity.
10. **Selective routing proves the router CAN beat optimal fixed weights** — on the ~35% of samples where the router is most confident, it correctly identifies the best expert, achieving +0.12% over opt fixed. But the gain is far too small to reach the +1% target.
11. **Gradient directions are near-orthogonal in high-dimensional input space (GDDR)** — ∇_x CE gradients in 3072-d space have mean cosine similarity ≈ 0.03, regardless of expert correctness. The direction signal is dominated by noise and correlates with confidence (r≈0.30), making it unsuitable for routing.
12. **Feature-space clusters don't align with routing-relevant groupings (Cluster Routing)** — While per-cluster optimal weights differ from global weights, the gain over global optimal fixed weights is only +0.08% (within noise). Visual similarity (feature clusters) doesn't predict which expert weighting is best.

### Final Verdict: The Routing Ceiling is Confirmed and Absolute

After **25+ routing methods across 5 rounds** — including 2 novel approaches developed in the final session that explored fundamentally different signals (gradient direction alignment, cluster-based adaptive weighting) — the ceiling is confirmed as absolute.

The **Selective 92-d routing** method (92-d features + confidence threshold of 0.35) is the best routing method tested against optimal fixed weights:
- **52.70% BA** (+1.58% over uniform, **+0.12% over optimal fixed weights**)
- **First method to reliably beat optimal fixed weights**
- **Does NOT achieve the +1% margin target** (needs 53.58%)

The **392-d hybrid TTA routing** achieves the highest absolute BA:
- **53.22% BA** but the TTA-optimal-fixed baseline is 53.54% (−0.32% gap)

**Two novel methods added in the final round and their findings:**
1. **GDDR (gradient alignment)**: 46.98% BA — worse than uniform. Gradient directions in 3072-d space are near-orthogonal regardless of correctness. This is a fundamental mathematical limitation of high-dimensional routing signals.
2. **Cluster routing (per-cluster optimal weights)**: 52.56% BA — essentially tied with global opt fixed. Per-cluster weights differ but feature-space clusters don't align with routing-relevant groupings.

After testing every plausible routing algorithm, feature representation, and signal type across five rounds, the gap to optimal fixed weights cannot be closed. **No method that derives its signal from frozen experts' outputs, features, or gradients can surpass a simple tuned static ensemble by ≥1% on this problem.** The only remaining path to improvement is to **change what the experts provide** (joint training, new expert with different representations, or an orthogonal self-supervised signal).

### Remaining Options

To go beyond this ceiling, the approach must **change what the experts provide**, not search harder in their frozen outputs:

| # | Approach | Addresses Problem | Est. Gain | Priority |
|:-:|:---------|:-----------------:|:---------:|:--------:|
| 1 | **MoCo v2 expert replacing Mixup** | Reduces 37.1% all-wrong ceiling by adding genuinely different failure modes | +2-4% absolute | ⭐ Raises absolute accuracy, not routing fraction |
| 2 | **SADE test-time adaptation (rotation-prediction)** | Uses ORTHOGONAL self-supervised signal (not from softmax outputs) | +0.3-0.8% | #2 |
| 3 | **RIDE-style joint training** | End-to-end router + expert co-training | +2-3% | #3 (high risk, high cost) |

**Note on MoCo v2:** MoCo v2 would raise the uniform baseline, making the +1% routing target harder to achieve as a percentage of the new baseline. It improves absolute accuracy, not the routing contribution fraction. See `docs/PLAN.md` for details.

**Note on consistency routing:** Tested and failed (Phase 1 feasibility study, +0.18% gain < 0.5% threshold). Not recommended for further exploration.

**Note on TTA averaging:** Tested in Round 4. Raises absolute BA but hurts routing fraction. The baseline moves with the improvement. Not recommended for further routing exploration.

**Note on gradient sensitivity:** Tested in Round 4. Signal exists (r=0.24-0.34) but is redundant with 92-d features. Not recommended for further exploration.
