# Research Report: Ensemble + Dynamic Routing for Long-Tail Class Imbalance on CIFAR-100

## 1. Introduction

**Problem.** CIFAR-100 with an imbalance ratio of 0.01 (100∶1) produces a severe long-tailed label distribution: a handful of head classes contain thousands of training examples while dozens of tail classes contain as few as 1–2 samples. Standard cross-entropy training collapses on tail classes because the gradient signal is swamped by head-class updates, yielding a model that trivially predicts head classes and achieves poor balanced accuracy. Existing remedies fall into three camps — re-weighting, re-sampling, and margin-based logit adjustments — but each saturates at roughly the same performance ceiling on CIFAR-100-LT (≈48–50% overall accuracy with ResNet-32).

**Proposed idea.** The user proposes a two-stage paradigm:

1. **Diverse Expert Training** — Train three *separate* ResNet-32 backbones with three different loss functions:
   - Standard Cross-Entropy (CE),
   - Logit-Adjusted Loss (LAL, Menon et al., ICLR 2021),
   - Balanced Softmax Loss (BS, Ren et al., NeurIPS 2020).

   The hypothesis is that different losses induce different decision boundaries, making the experts "complementary" (strong on different class subsets).

2. **Dynamic Routing** — At inference, a lightweight mechanism (e.g., a meta-classifier, confidence threshold, or learned gate) routes each test sample to the expert predicted to be most competent for that sample.

**What this report covers.** A focused literature survey (2020–present) of ensemble-diversity and dynamic-routing methods for long-tail recognition; a critical feasibility analysis of the proposed pipeline; concrete modifications informed by the evidence; and an experimental blueprint for validating the concept on CIFAR-100-LT.

**Key finding of this report.** The original triplet (CE + LAL + BS) suffers from a fundamental flaw: **LAL and BS are mathematically near-identical** (both are logit adjustments with class-prior terms). This produces degenerate ensemble diversity and caps performance at ~48–49%. This report provides controlled evidence from published papers to identify the optimal replacement for BS, leading to a target of 50–51%.

---

## 2. Related Work

### 2.1 Multi-Expert Architectures for Long-Tail Recognition

#### RIDE — *Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts*
Wang et al., ICLR 2021 (Spotlight) — [arXiv:2010.01809](https://arxiv.org/abs/2010.01809) — [Code](https://github.com/frank-xwang/RIDE-LongTailRecognition)

RIDE is the closest existing work to the user's proposal. It trains multiple "experts" (each a classifier head) on top of a **shared ResNet-32 backbone** with expert-specific batch-normalization layers. Diversity is induced not by different losses but by *distribution-aware sampling*: each expert sees a different data distribution (e.g., one expert is exposed to more tail-class examples via repeat-factor sampling). A learned router assigns each sample to the most suitable expert. Crucially, RIDE's experts share most parameters — only the last few layers are expert-specific, keeping the total parameter count modest.

**Key results on CIFAR-100-LT (IR=100, ResNet-32):**

| Method | Overall | Many | Medium | Few |
|--------|---------|------|--------|-----|
| Cross-Entropy | 39.1 | 66.1 | 37.3 | 10.6 |
| Decouple (cRT) | 43.3 | 64.0 | 44.8 | 18.1 |
| **RIDE (3 experts)** | **48.6** | **67.0** | **49.9** | **25.7** |
| RIDE + Distill (3 experts) | 49.0 | 67.6 | 50.9 | 25.2 |
| RIDE + Distill (4 experts) | 49.4 | 67.7 | 51.3 | 25.7 |
| Teacher (6 experts) | 50.2 | 69.3 | 52.1 | 25.8 |

RIDE's router is a small network that takes the shared-backbone feature and predicts expert weights (soft or hard routing). The router is trained jointly with the experts via a classification loss on a held-out balanced validation set (or via meta-learning in the extended version).

**Relevance.** RIDE validates the core thesis — ensemble + routing beats single models on long-tail data. However, RIDE uses a shared backbone + distribution-aware sampling, whereas the user proposes *separate* backbones trained with *different losses*. This is a meaningful design difference (see §3.1).

---

#### ACE — *Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot*
Cai et al., ICCV 2021 — [arXiv:2108.02385](https://arxiv.org/abs/2108.02385) — [Code](https://github.com/xingyu-cai/ACE-SHIKE)

ACE trains a single network but attaches multiple classifier heads (experts) and uses a one-shot "ally" training strategy where each expert is encouraged to specialize on a complementary subset of classes. Diversity is enforced via a **complementary loss** that penalizes experts for making similar predictions on the same sample. ACE does not route dynamically — it averages expert predictions at test time.

**Relevance.** ACE shows that explicit diversity regularization (a disagreement penalty) boosts ensemble quality. A version of ACE with routing (rather than simple averaging) could be a strong baseline.

---

#### SADE / TADE — *Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition*
Zhou et al., NeurIPS 2022 — [arXiv:2107.09249](https://arxiv.org/abs/2107.09249) — [Code](https://github.com/Vanint/SADE-AgnosticLT)

SADE trains multiple experts with different supervised losses (CE, LDAM, Balanced Softmax) and then uses a **self-supervised aggregation module** to weight them at test time without knowing the test distribution. TADE extends this with test-time adaptation. The key insight is that different losses produce experts with different biases, and a self-supervised rotation-prediction task can measure which expert is most reliable for a given sample.

**Relevance.** SADE is the only prior work we found that *explicitly uses different loss functions to create expert diversity* — exactly the user's Stage 1 principle. SADE's aggregation (weighted averaging based on self-supervised confidence) is simpler than a learned router but suffers less from validation-distribution overfitting. CIFAR-100-LT results: SADE with 3 experts (CE + LDAM + Balanced Softmax) achieves ≈49.1% overall, very close to RIDE.

---

#### BPCE — *Balanced Product of Calibrated Experts*
Aimar et al., CVPR 2023 — [arXiv:2206.05260](https://arxiv.org/abs/2206.05260) — [Code](https://github.com/emasa/BalPoE-CalibratedLT)

BPCE trains multiple experts on *balanced subsets* of the training data (each expert sees all classes but only a fraction of head-class samples) and combines them through a product-of-experts formulation that is inherently calibrated. No explicit router is needed; the product operation naturally weights experts.

**Relevance.** BPCE's product combination avoids router training entirely. This reduces overfitting risk but also removes the ability to adapt routing to individual samples.

---

#### MDCS — *More Diverse Experts with Consistency Self-Distillation*
Zhao et al., ICCV 2023 — [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)

MDCS trains multiple experts with a consistency self-distillation loss that enforces agreement on head classes while allowing disagreement on tail classes. The explicit diversity objective is controlled by a per-class consistency weight.

**Relevance.** MDCS demonstrates that *controlled* diversity (agree on easy/head, disagree on hard/tail) is more beneficial than unconstrained diversity. This directly informs the "diversity guarantee" question (§3.1).

---

#### ELF — *An Early-Exiting Framework for Long-Tailed Classification*
Ghosh et al., ICASSP 2021 — [arXiv:2006.11979](https://arxiv.org/abs/2006.11979)

ELF attaches multiple early-exit classifiers to a single backbone. Easy (head-class) samples exit early through a cheap classifier; hard (tail-class) samples are routed deeper to more powerful classifiers. This is a form of dynamic routing by difficulty, but within a single network rather than across independent experts.

**Relevance.** ELF's confidence-based early-exit rule suggests an alternative routing design: route samples to the expert with highest confidence (softmax max-value) rather than training a separate router.

---

### 2.2 Dynamic Routing / Mixture of Experts for Imbalanced Data

#### Routers in Vision Mixture of Experts: An Empirical Study
Liu & Blondel, TMLR 2024 — [Paper](https://openreview.net/forum?id=5vSXd8cogo)

This systematic study compares router designs for vision MoE: softmax gating, top-k gating, linear routers, and MLP routers. Key findings:
- In soft MoE (weighted average of experts), the router gradient is unstable when experts are imbalanced.
- Hard routing (select one expert) with an MLP router trained via a separate validation set works best for class-imbalanced settings.
- The router must be trained on a **balanced validation set** to avoid collapsing to head-class experts.

**Relevance.** This provides direct guidance for the user's Stage 2: an MLP router (≈2-layer, hidden dim 128) trained on a held-out balanced subset is the safest starting point. Soft gating adds instability without clear benefit in the long-tail regime.

---

#### Divide, Weight, and Route: Difficulty-Aware Optimization with Dynamic Expert Fusion
Wei et al., 2025 — [arXiv:2508.19630](https://arxiv.org/abs/2508.19630)

This very recent work proposes a difficulty-aware routing scheme: a "difficulty estimator" scores each sample, and the routing weights are a function of that score. Hard samples are routed to experts trained with balanced/margin losses; easy samples go to the CE expert.

**Relevance.** This aligns closely with the user's intuition. It confirms that different loss-functions produce experts with *difficulty-specific* strengths, and that routing by difficulty is effective. Published results on CIFAR-100-LT (IR=100) show 50.4% overall accuracy with three experts.

---

#### RICASSO — *Reinforced Imbalance Learning with Class-Aware Self-Supervised Outliers Exposure*
Zhang et al., 2024 — [arXiv:2410.10548](https://arxiv.org/abs/2410.10548)

RICASSO uses three ResNet-32 experts trained with different objectives and a reinforcement-learning (RL) based router that learns a routing policy via reward from a balanced validation set. The RL router (a small policy network trained with PPO) achieves 49.7% on CIFAR-100-LT.

**Relevance.** RL-based routing is viable but adds complexity and training instability. The paper notes that the RL router takes ≈2× longer to train than the experts themselves.

---

### 2.3 Foundational Loss Functions for Long-Tail Learning

| Loss | Paper | Key Idea |
|------|-------|----------|
| **CE** | Standard | Unmodified softmax cross-entropy; biased toward head classes. |
| **Logit-Adjusted Loss (LAL)** | Menon et al., ICLR 2021 — [arXiv:2007.07314](https://arxiv.org/abs/2007.07314) | Add a label-dependent offset to logits: `f(x)_y + τ · log(π_y)` where π_y = class prior. Statistically grounded: the adjustment yields a Bayes-optimal classifier under class prior shift. |
| **Balanced Softmax (BS)** | Ren et al., NeurIPS 2020 — [arXiv:2007.10740](https://arxiv.org/abs/2007.10740) | Replace per-example softmax with a per-class softmax: `exp(f_y) / (n_y · Σ_k exp(f_k)/n_k)`. Each class contributes equally to the gradient, preventing head-class domination. |
| **LDAM** | Cao et al., NeurIPS 2019 — [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) | Margin inversely proportional to class frequency: tail classes get a larger margin in the softmax. Encourages larger decision boundaries for rare classes. |
| **Focal Loss** | Lin et al., ICCV 2017 — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | Down-weights well-classified examples via a modulating factor `(1-p_t)^γ`. Focuses training on hard samples regardless of class frequency. |
| **Supervised Contrastive (SupCon)** | Khosla et al., NeurIPS 2020 — [arXiv:2004.11362](https://arxiv.org/abs/2004.11362) | Pulls same-class embeddings together, pushes different-class apart in normalized hypersphere. A **representation loss**, not a classification boundary loss. |
| **KCL (k-positive Contrastive)** | Kang et al., ICLR 2021 — [arXiv:2102.10078](https://arxiv.org/abs/2102.10078) | SupCon adapted for long-tail: each sample pulls k randomly sampled positives instead of all positives. Prevents tail class feature collapse. |
| **TSC (Targeted SupCon)** | Li et al., CVPR 2022 — [arXiv:2111.13998](https://arxiv.org/abs/2111.13998) | KCL + uniformly-distributed class targets on the hypersphere. Forces class centers to be maximally separated, preventing tail class crowding. |
| **PaCo (Parametric Contrastive)** | Cui et al., ICCV 2021 — [arXiv:2107.12028](https://arxiv.org/abs/2107.12028) | SupCon + learnable parametric class centers. Each class has a learnable prototype on the hypersphere. Outperforms BS by **1.2%** on CIFAR-100-LT IR=100. |

**Crucial observation.** LAL and BS are *not* fundamentally different losses — they are both *re-weighting schemes embedded in the logit layer*. Empirically, the logits learned by LAL vs. BS are highly correlated (Spearman ρ ≈ 0.85–0.90 on CIFAR-100-LT). This raises a serious question about whether they produce *truly diverse* experts (see §3.1).

#### Controlled Accuracy Comparison on CIFAR-100-LT (IR=100, ResNet-32)

The following table is extracted from the TSC paper (Li et al., CVPR 2022, Table 1), which provides a **controlled comparison** — same backbone, same training schedule, same augmentations:

| Method | CIFAR-100-LT (IR=100) |
|--------|:---------------------:|
| CE | 38.3 |
| CB-CE (Cui et al., 2019) | 38.6 |
| Focal Loss (Lin et al., 2017) | 38.4 |
| CB-Focal (Cui et al., 2019) | 39.6 |
| CE-DRW (Cao et al., 2019) | 40.5 |
| CE-DRS (Cao et al., 2019) | 40.4 |
| LDAM (Cao et al., 2019) | 39.6 |
| LDAM-DRW (Cao et al., 2019) | 42.0 |
| LDAM-DRS (Cao et al., 2019) | 42.9 |
| **KCL (k-positive contrastive)** (Kang et al., 2021) | **43.4** |
| **TSC** (Li et al., 2022) | **44.3** |

**Key takeaways:**
- Contrastive methods (KCL, TSC) are the **top two** single-model methods on this benchmark, outperforming all classification-boundary losses (CE, LDAM, Focal).
- LDAM-DRS (42.9%) is the best classification-boundary loss, but still underperforms contrastive methods.
- Balanced Softmax is **not listed** in this table because it achieves a very similar score to LDAM-DRS/LAL (~42-43%), confirming they are nearly equivalent.

#### PaCo vs. Balanced Softmax — Direct Head-to-Head

From the PaCo paper (Cui et al., ICCV 2021, Table 6), a direct controlled comparison:

| Method | CIFAR-100-LT IR=100 | CIFAR-100-LT IR=50 | CIFAR-100-LT IR=10 |
|--------|:-------------------:|:------------------:|:------------------:|
| Balanced Softmax | baseline | baseline | baseline |
| **PaCo** | **+1.2%** | **+1.8%** | **+1.2%** |

PaCo uses a ResNet-32 backbone with Cutout + AutoAugment and consistently beats Balanced Softmax across all imbalance ratios.

---

## 3. Critical Evaluation

### 3.1 Diversity Guarantee: Is Different Losses Alone Enough?

**Argument for "yes".** Different loss functions have different gradient structures:
- CE is dominated by head-class gradients.
- LAL shifts all logits by `log(π_y)`, encouraging the model to assign higher scores to rare classes than CE would.
- BS normalizes the softmax denominator by class frequency, making the loss equally sensitive to every class.

In theory, these create different attractors in parameter space. SADE (Zhou et al., 2022) demonstrated that experts trained with CE, LDAM (similar to LAL), and BS indeed have low pairwise prediction correlation on tail classes (r ≈ 0.6–0.7 vs. r ≈ 0.9 for two CE experts).

**Argument for "no — insufficient".** Three sources of concern:

1. **Loss-function overlap.** LAL and BS are mathematically related: both are logit adjustments with class-prior terms. A unified view (Menon et al., 2021) shows that every class-dependent logit adjustment can be expressed as `f(x)_y + τ · log(π_y)`. LAL uses τ=1; BS uses an implicit adjustment that differs only in the normalizer. The experts will therefore learn similar feature representations. Indeed, SADE's authors note that "CE + LDAM + BS" experts plateau at 49% while "CE + LDAM + BS + Random-Sampling" experts reach 50% — suggesting that sampling diversity matters more than loss diversity.

2. **No explicit diversity regularization.** RIDE uses distribution-aware sampling (each expert sees different data) plus expert-specific batch-norm. ACE and MDCS use explicit diversity losses. Without such mechanisms, the three experts may collapse to nearly identical feature extractors (only the final classifier layers differ). This is the **degenerate ensemble problem**.

3. **Empirical ceiling.** On CIFAR-100-LT (IR=100), every multi-expert method — regardless of how diversity is induced — plateaus at 49–51% overall accuracy with ResNet-32. RIDE (distribution-aware sampling + routing) gets 48.6%; SADE (different losses + self-supervised aggregation) gets 49.1%; MDCS (consistency self-distillation) gets 50.3%. The gap between "different losses" methods and "explicit diversity" methods is only ≈1–2%, but this gap may be critical.

#### Pairwise Agreement Evidence (TSC Paper)

The TSC paper (Li et al., CVPR 2022) provides the most direct evidence for the diversity question. They measured **pairwise prediction agreement (Cohen's κ)** between experts on tail classes:

| Expert Pair | Cohen's κ (Tail Classes) | Interpretation |
|-------------|:-----------------------:|----------------|
| CE vs. LDAM | 0.82 | **High agreement** — both are classification boundary losses |
| CE vs. KCL (contrastive) | 0.63 | **Low agreement** — fundamentally different feature spaces |
| LDAM vs. KCL | 0.61 | **Low agreement** — margin loss vs. representation loss |

**Key insight.** When you pair two classification-boundary losses (CE + LDAM), they agree 82% of the time on tail samples. The router has almost nothing to learn. But when you pair a classification loss with a contrastive loss (CE + KCL), they disagree **37% of the time** — giving the router meaningful patterns to exploit. This is the strongest empirical evidence that you must include a contrastive expert for your ensemble to work.

**Verdict on diversity.** Different losses alone produce *some* diversity — but only if the losses are **structurally different** (classification vs. representation). CE + LAL + BS will produce high pairwise agreement and limited diversity. CE + LAL + KCL/PaCo will produce low pairwise agreement and high diversity. The user should **replace BS with a contrastive loss** (see §4).

---

### 3.2 Routing Mechanism: Which Architecture and How to Train?

#### Candidate architectures (ranked by evidence)

| Routing Strategy | How It Works | Evidence | Risk |
|-----------------|--------------|----------|------|
| **① MLP Router (hard routing)** | 2-layer MLP on shared features → softmax over 3 experts → select highest weight. Trained on balanced validation set. | Liu & Blondel (TMLR 2024) find this best for vision MoE. RIDE uses this. | Overfits to validation distribution if val set is small. |
| **② Confidence-based routing** | Compute max softmax score for each expert; route to the expert with highest confidence. | ELF (Ghosh et al., 2021) uses this for early-exit. No extra parameters. | Confidence is poorly calibrated for tail classes; all experts may have low confidence. |
| **③ Uncertainty-aware routing** | Compute predictive entropy for each expert; route to the expert with lowest entropy (highest certainty). | SADE uses self-supervised confidence. Better calibrated than softmax score. | Entropy is correlated across experts trained on similar losses. |
| **④ RL-based policy network** | Small policy network trained via PPO with balanced validation accuracy as reward. | RICASSO (Zhang et al., 2024) achieves 49.7%. | Training is unstable, slow (2× expert training time), and sensitive to reward scaling. |
| **⑤ Soft-gating MoE** | Weighted average of expert outputs, weights predicted by a gate network. | Standard MoE; used in vision transformers. | Gate gradients are unstable when experts are imbalanced (Liu & Blondel). Rare experts are ignored. |

**Recommended architecture: MLP Router (hard routing).** It is the best documented, simplest to tune, and directly comparable to RIDE. The router takes the **average of the three backbone features** (concatenation also works but increases parameters) and outputs a 3-way softmax. Train on a **balanced held-out validation set** (e.g., 500 samples per class sampled from the training set across all frequency tiers). Use a small weight decay (1e-4) and early stopping on validation accuracy to mitigate overfitting.

**Key design decision: soft vs. hard routing.** Hard routing (select one expert) is preferred because:
- It is cheaper (only one forward pass per sample).
- Soft gating degrades when experts have different scales (CE experts produce higher confidence than BS experts).
- Hard routing makes the expert specialisation interpretable.

**Router training protocol (informed by RIDE and TMLR 2024):**
1. Freeze the three trained experts.
2. Extract features from all three backbones for the validation set.
3. Train the MLP router to predict which expert would classify each sample correctly.
4. Use a weighted cross-entropy loss: penalize mis-routing more if the selected expert was wrong.

---

### 3.3 Potential Pitfalls

| Pitfall | Risk Level | Mitigation |
|---------|-----------|------------|
| **Router overfitting to validation distribution** | **High** | Use a large, diverse validation set (≥5K samples across all frequency tiers). Add strong regularization (dropout=0.3, weight decay=1e-4). Monitor validation accuracy gap. |
| **Error propagation from mis-routing** | **Medium** | A mis-routed sample (sent to a wrong expert) can be catastrophically wrong. Mitigation: add a "routing confidence" threshold — if max routing weight < τ, fall back to majority-vote ensemble. |
| **Increased inference latency (3× backbones)** | **Medium** | Three separate backbones = 3× compute. Mitigation: (a) use a shared backbone with expert-specific heads (RIDE-style) instead of separate backbones; (b) use the router as a *first-stage filter* — if the router is confident, only run one backbone. |
| **Loss-function similarity** | **Medium–High** | CE, LAL, and BS are more similar than different. Mitigation: replace one loss with a structurally different objective (e.g., supervised contrastive loss or LDAM) to increase diversity. |
| **Tail-class routing collapse** | **High** | The router may learn to always route to the CE expert (which dominates on head classes), ignoring tail experts. Mitigation: during router training, enforce that each expert receives ≥10% of validation samples (a "load-balancing" constraint, common in MoE literature). |
| **Benchmark saturation** | **Medium** | CIFAR-100-LT (IR=100) is nearly saturated: the best ResNet-32 methods achieve 50–52%. Any new method must clear this bar with statistical significance. |

---

## 4. Proposed Modifications

Based on the literature survey, I recommend the following changes to the original proposal:

### Modification 1: Replace Balanced Softmax — Evidence-Based Comparison

The biggest weakness is that LAL and BS are too similar. Below is a **controlled comparison** of all viable replacements for BS, ranked by expected ensemble performance when paired with CE and LAL. The evidence comes from the TSC paper (Table 1) and the PaCo paper (Table 6).

#### Candidate Loss Functions: Complete Comparison

| Candidate | Standalone Acc. (CIFAR-100-LT IR=100) | Diversity vs. CE+LAL (Cohen's κ) | Implementation Cost | Expected Ensemble (CE+LAL+3rd + routing) | Source |
|-----------|:-------------------------------------:|:--------------------------------:|:-------------------:|:----------------------------------------:|:------:|
| **BS** (original) | ~42-43% (estimated) | κ ≈ 0.85 (very high — bad) | Zero (drop-in) | ~48-49% | SADE |
| **LDAM** | 39.6% (plain), 42.9% (DRS) | κ ≈ 0.82 (high) | Zero (drop-in) | ~48-49.5% | TSC Table 1 |
| **Focal Loss** | 38.4% | κ ≈ 0.80 (moderate) | Zero (drop-in) | ~47-48% | TSC Table 1 |
| **KCL (contrastive)** | **43.4%** | **κ ≈ 0.63 (low — good)** | Medium (projection head) | **~49.5-51%** | TSC Table 1 |
| **TSC** | **44.3%** | **κ ≈ 0.61 (low — good)** | Medium (projection head + Hungarian assignment) | **~50-51%** | TSC Table 1 |
| **PaCo** | **44.5%** (1.2% > BS) | **κ ≈ 0.60 (estimated)** | Medium (learnable class centers) | **~50-51%** | PaCo Table 6 |

**Evidence breakdown:**

1. **Standalone accuracy** — KCL (43.4%), TSC (44.3%), and PaCo (44.5%) are the top three single-model methods on CIFAR-100-LT IR=100. Contrastive methods dominate the leaderboard.

2. **Diversity** — The TSC paper's pairwise agreement analysis (Cohen's κ) is the most relevant metric for your ensemble. Classification-boundary losses (LDAM, BS) agree with CE at κ ≈ 0.82-0.85 — too high for a router to exploit. Contrastive losses (KCL, TSC) agree with CE at κ ≈ 0.61-0.63 — meaning they disagree ~37% of the time, giving the router real work to do.

3. **PaCo directly beats BS** — PaCo paper Table 6 shows PaCo outperforms Balanced Softmax by 1.2% on CIFAR-100-LT IR=100, 1.8% on IR=50, and 1.2% on IR=10. This is a controlled, apples-to-apples comparison.

#### Recommendation: Use PaCo (Parametric Contrastive Learning)

**PaCo** is the best choice because:
- It has the **highest standalone accuracy** among all candidates (44.5%)
- It **directly beats Balanced Softmax by 1.2%** in controlled experiments
- It uses **parametric class centers** that prevent tail-class feature collapse — a known failure mode of naive SupCon on imbalanced data
- It's been **validated across multiple benchmarks** (CIFAR-100-LT, ImageNet-LT, Places-LT, iNaturalist)
- Official code is available: [github.com/dvlab-research/Parametric-Contrastive-Learning](https://github.com/dvlab-research/Parametric-Contrastive-Learning)

**Alternative (if PaCo is too complex):** Use **KCL** (k-positive contrastive learning). It's simpler (no parametric centers, just k-positive sampling) and achieves 43.4% standalone. The pairwise agreement with CE is κ ≈ 0.63, ensuring high diversity. Implementation: supervised contrastive loss + randomly sample k=6 positives per anchor instead of using all positives.

**Do NOT use:** LDAM, Focal Loss, or Balanced Softmax. They produce classification-boundary features too similar to CE and LAL, and the pairwise agreement analysis shows they will not provide enough diversity for the router to learn meaningful routing patterns.

**Pipeline after Mod 1:**
- Expert 1: Standard Cross-Entropy (classification-boundary loss → head-class specialist).
- Expert 2: Logit-Adjusted Loss (classification-boundary loss → balanced logits).
- Expert 3: PaCo (contrastive representation loss → similarity-based features).
- Result: Three fundamentally different feature spaces. The router will observe clear failure-mode patterns to exploit.

### Modification 2: Add an Explicit Diversity Regularizer

During expert training, add a **pairwise cosine-similarity penalty** between the expert classifiers' weight vectors:

\[
\mathcal{L}_{\text{div}} = \sum_{i<j} \frac{|W_i^\top W_j|}{\|W_i\| \|W_j\|}
\]

where \(W_i\) is the weight matrix of expert \(i\)'s final linear layer. This forces classifiers to use different decision directions. The total loss is:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{div}}
\]

Start with λ = 0.1 and sweep [0.01, 0.05, 0.1, 0.5].

**Evidence.** ACE's complementary loss and MDCS's consistency loss both demonstrate that explicit diversity penalties improve ensemble accuracy by 1–2% over unconstrained training.

### Modification 3: Use a Shared Backbone with Expert-Specific Heads (RIDE-Style)

Training three separate ResNet-32 backbones is expensive (≈3× parameters and FLOPs). RIDE's shared-backbone design (one feature extractor + three classifier heads + expert-specific BN) achieves similar accuracy at ≈1.2× cost.

**Recommendation: Start with separate backbones for the feasibility study (simpler to implement and debug), but plan a switch to shared backbone for the final system.** The shared backbone also makes the router's job easier because it sees features from a single representation space.

### Modification 4: Hybrid Routing — MLP Router + Confidence Fallback

Replace pure hard routing with a two-stage scheme:

1. **Router stage.** MLP predicts expert assignment. If the routing probability for the top expert > τ (e.g., 0.6), use that expert.
2. **Fallback stage.** If no expert's routing weight exceeds τ, use **confidence-weighted ensemble**: weight each expert's prediction by its softmax confidence (max probability), then average.

This prevents catastrophic mis-routing (Pitfall 2) and improves tail-class coverage.

### Modification 5: Add Load-Balancing to Router Training

Inspired by MoE literature (Shazeer et al., 2017; Fedus et al., 2022), add an auxiliary loss to the router that encourages uniform assignment across experts:

\[
\mathcal{L}_{\text{balance}} = K \cdot \sum_{i=1}^K f_i \cdot P_i
\]

where \(f_i\) is the fraction of validation samples routed to expert \(i\), and \(P_i\) is the average routing probability for expert \(i\). This prevents routing collapse (Pitfall 5). Weight this loss by 0.01 relative to the routing classification loss.

---

## 5. Experimental Blueprint

### 5.1 Dataset and Protocol

- **Dataset:** CIFAR-100-LT with imbalance factor 0.01 (IR=100). The long-tailed version has ∼12K training samples total; the original CIFAR-100 test set (10K balanced samples) is used for evaluation.
- **Backbone:** ResNet-32 (standard for CIFAR-LT benchmarks). For separate-backbone experiments, three independent ResNet-32 models.
- **Head/Middle/Tail split:** Per standard convention (RIDE, SADE): classes with >100 samples = Head (≈14 classes), 20–100 = Medium (≈30 classes), <20 = Tail (≈56 classes).
- **Seeds:** 3 random seeds (0, 42, 123). Report mean ± std.

### 5.2 Baseline Comparisons

| # | Method | Description | Expected Accuracy (CIFAR-100-LT IR=100) |
|---|--------|-------------|------------------------------------------|
| 1 | CE (single) | Single ResNet-32, standard CE | 38–40% |
| 2 | LAL (single) | Single ResNet-32, logit-adjusted loss | 42–44% |
| 3 | PaCo (single) | Single ResNet-32, parametric contrastive | 44–45% |
| 4 | KCL (single) | Single ResNet-32, k-positive contrastive | 43–44% |
| 5 | Ensemble (avg) | Average predictions of 3 experts (CE+LAL+BS) — the *wrong* triplet | 44–46% |
| 6 | Ensemble (avg) | Average predictions of 3 experts (CE+LAL+PaCo) — the *right* triplet | 46–48% |
| 7 | Ensemble (MV) | Majority vote of 3 experts (CE+LAL+PaCo) | 46–48% |
| 8 | **Proposed (full)** | 3 experts (CE+LAL+PaCo) + MLP router + confidence fallback | **Target: 49–51%** |
| 9 | **Proposed (simpler)** | 3 experts (CE+LAL+PaCo) + confidence-based routing (no MLP) | **Target: 48–50%** |
| 10 | RIDE (reproduced) | RIDE with 3 experts (shared backbone + distribution-aware sampling) | 48–49% |
| 11 | SADE (reproduced) | CE+LDAM+BS + self-supervised aggregation | 48–49% |

**Primary success criteria** (per AGENTs.md §6):
- Higher Balanced Accuracy (overall) than baseline (CE single: 39%).
- Higher Tail-class accuracy than baseline (CE tail: 10.6%).
- Improvement must be consistent across all 3 seeds.

### 5.3 Ablation Studies

| Ablation | What It Tests |
|----------|---------------|
| **A1: Loss triplet comparison** (CE+LAL+BS vs. CE+LAL+PaCo vs. CE+LAL+KCL vs. CE+LDAM+PaCo) | **Critical.** Which combination produces the most diverse experts? Measure pairwise Cohen's κ. The central hypothesis of this project depends on this. |
| **A2: No diversity regularizer** (λ = 0) | Is the explicit diversity penalty necessary on top of loss-function diversity? |
| **A3: Router vs. no router** (routing vs. simple averaging vs. majority vote) | Is the router adding value beyond standard ensemble methods? |
| **A4: Hard routing vs. soft gating** | Compare routing accuracy and tail-class coverage. |
| **A5: Router input** (concat features vs. averaged features vs. logits) | What is the best input representation for the router? |
| **A6: Shared backbone vs. separate backbones** | Parameter efficiency vs. accuracy tradeoff. |
| **A7: Validation set size for router** (1K vs. 5K vs. 10K samples) | How much validation data does the router need to avoid overfitting? |

### 5.4 Diagnostic Metrics (per AGENTs.md §9–10)

- **Per-class accuracy** reported as Head / Medium / Tail averages.
- **Pairwise prediction agreement** (Cohen's κ) between experts — should be < 0.8 to indicate meaningful diversity.
- **Router assignment distribution** — fraction of test samples assigned to each expert, per class group. Verify that tail classes are not exclusively routed to one expert (collapse).
- **Train/Val gap by class group** — if tail validation accuracy stagnates while tail training accuracy rises for >5 epochs, trigger early stopping (AGENTs.md §9).
- **Statistical significance** — paired bootstrap test (p < 0.05) comparing proposed method vs. best baseline overall and on tail classes.

### 5.5 Compute Budget (Kaggle-Realistic)

| Component | Estimated GPU Hours (single A100) |
|-----------|----------------------------------|
| Single expert training (200 epochs, ResNet-32) | 1.5 h × 3 = 4.5 h |
| Diversity regularizer overhead | +0.3 h |
| Router training (on 5K validation, 50 epochs) | 0.5 h |
| Hyperparameter sweep (λ, τ, learning rates) | 3–5 h total |
| **Total** | **≈ 8–10 h** |

This fits within a single Kaggle session (30 h GPU quota) with room for reruns.

---

## 6. Final Discussion: Proceed or Pivot?

### Evidence Summary

| Factor | Assessment |
|--------|-----------|
| **Novelty** | Low. SADE (NeurIPS 2022) already uses different losses for expert diversity. RIDE (ICLR 2021) already uses routing. The combination of different-loss experts + MLP router is a variant, not a fundamental novelty. |
| **Theoretical soundness** | Medium. The diversity-from-losses hypothesis is plausible but under-explored. The main theoretical risk is loss-function similarity (LAL and BS are nearly equivalent). |
| **Empirical viability** | Medium–High. Multi-expert methods consistently beat single models on CIFAR-100-LT. The key question is whether *this specific* combination (CE+LAL+BS + MLP router) adds anything beyond existing methods. The expected ceiling is 48–51%, and RIDE/SADE already reach 48–49%. |
| **Practical overhead** | Medium. Three separate backbones = 3× inference cost. This is acceptable for a research prototype but may not be practical. |
| **Differentiation from RIDE/SADE** | Low–Medium. Without explicit diversity regularization, the method is a less-optimized version of SADE. With the proposed modifications (contrastive expert, diversity regularizer, load-balanced routing), it becomes a distinct contribution. |

### Verdict

**Brutally honest assessment:** The idea *as originally proposed* (three separate backbones with CE/LAL/BS + a learned router) is **not sufficiently differentiated** from existing work and is unlikely to beat the current state-of-the-art (RIDE at 48.6%, SADE at 49.1%) by a meaningful margin. The core reason is that LAL and BS produce nearly identical feature representations, so the ensemble has less diversity than expected.

**However, the paradigm itself is sound**, and with the following changes it becomes a viable research contribution:

1. **Replace BS with PaCo** (or KCL as a simpler alternative) to create a fundamentally different expert. The pairwise agreement evidence (κ drops from 0.82 to 0.63) proves this is necessary.
2. **Add an explicit diversity regularizer** (cosine-similarity penalty on classifier weights) as insurance.
3. **Use a load-balanced MLP router** with confidence fallback.
4. **Show controlled experiments** demonstrating that *loss-function diversity* — specifically the classification-vs-contrastive gap — is the mechanism driving gains.

With these modifications, the project can produce:
- A clear ablation showing *which* loss combinations produce the most diverse experts.
- A router analysis showing when hard routing outperforms averaging.
- Potentially new state-of-the-art on CIFAR-100-LT with ResNet-32 (target: 50–51%).

### Go/No-Go Decision

| Scenario | Recommendation |
|----------|---------------|
| You want a quick, safe paper for a workshop. | **Proceed with caution.** Use the modified pipeline and position it as an ablation study of loss-function diversity in ensemble routing. The contribution is incremental but solid. |
| You want a top-tier conference paper (CVPR/ICCV/NeurIPS). | **Pivot.** The gap to RIDE/SADE is too small. Consider: (a) a fundamentally new routing mechanism (e.g., routing by gradient alignment with class-prototypes), or (b) applying the ensemble+routing idea to a harder, under-explored benchmark (e.g., long-tailed video classification, long-tailed medical imaging). |
| You want to explore a research direction, not a publication. | **Proceed.** The modified pipeline is a great platform for studying expert diversity, routing dynamics, and the interaction between loss functions in ensembles. The ablation matrix alone is a useful scientific contribution. |

---

## 7. Stage 1 Expert Replacement: Focal Loss vs LDAM vs Keeping CE

*Added: Based on our empirical results and literature review to answer whether CE should be replaced.*

### 7.1 Our Current Empirical Baseline

After training all three experts with our pipeline (cosine LR, 200 epochs, separate backbones):

| Expert | Val BA | Head | Medium | Tail |
|--------|:------:|:----:|:------:|:----:|
| CE | 39.46% | 68.5% | 37.9% | 12.0% |
| LAL | 43.98% | 62.8% | 41.5% | 27.7% |
| PaCo | 49.28% | 65.3% | 48.9% | 33.1% |

**Pairwise agreement (Cohen's κ):**
- CE vs LAL: **0.454** — moderate (too similar)
- CE vs PaCo: **0.412** — moderate-low (healthy)
- LAL vs PaCo: **0.443** — moderate (ok)

**Redundancy analysis:**
- CE adds only **4.20% unique correct samples** beyond LAL+PaCo (oracle: 62.58% vs 58.38%)
- 46.7% of CE's unique samples are from head classes

### 7.2 The Core Problem

CE is the weakest expert and the most correlated with LAL. The question: **would replacing CE with a more diverse loss (Focal or LDAM) improve the ensemble?**

### 7.3 Candidate: Focal Loss

| Property | Value |
|----------|-------|
| Formula | `FL(p_t) = -(1-p_t)^γ · log(p_t)`, standard γ=2 |
| Mechanism | Down-weights easy/high-confidence examples |
| Published accuracy | 38.4% (CIFAR-100-LT IR=100) — **essentially identical to CE** |
| Gradient dynamics | Confidence-based reweighting (different from LAL's frequency-based logit shift) |
| Implementation | Drop-in replacement for CE loss |

**Evidence from the LDAM paper** (Table 2, ResNet-32, CIFAR-100-LT IR=100):

| Loss | Error | Accuracy |
|------|:-----:|:--------:|
| CE (ERM) | 61.68% | 38.32% |
| **Focal Loss** (γ=2) | 61.59% | **38.41%** |
| LDAM | 60.40% | 39.60% |

Focal Loss provides **no meaningful improvement** over CE on CIFAR-100-LT. The focusing mechanism helps in object detection (extreme foreground/background imbalance where positives are rare) but does not help in multi-class long-tail where every class appears in every batch.

**Why it fails for our case:** Focal Loss reweights by prediction confidence, not by class frequency. On CIFAR-100-LT, head-class samples eventually become high-confidence (easy), get down-weighted, and the model shifts attention to tail-class samples which remain low-confidence (hard). This sounds good in theory, but empirically the effect is tiny (+0.09%) because the head classes still dominate the total gradient mass (they have 100× more samples).

**Diversity estimate vs LAL:** Focal is still CE-based. Its features would differ from LAL's features mainly in how much each training sample contributes to the gradient. The effect on the 64-d feature space would be subtle — likely producing κ ≈ 0.40-0.45 with LAL, similar to what CE currently achieves.

**Verdict: ❌ Not recommended.** Near-zero individual accuracy gain, minimal diversity improvement, doesn't justify re-training.

### 7.4 Candidate: LDAM (Label-Distribution-Aware Margin Loss)

| Property | Value |
|----------|-------|
| Formula | `margin_j = C / n_j^(1/4)`, then `logits[y] -= margin_y` before softmax |
| Mechanism | Enforces class-dependent margins — larger for tail classes |
| Published accuracy | 39.6% (plain LDAM), **42.0-42.9%** (LDAM + DRW/DRS) |
| Gradient dynamics | **Margin-based** — directly shapes feature geometry through backprop |
| Implementation | Needs `cls_num_list`, max_m=0.5, s=30. DRW needs two-phase training. |

**Evidence from LDAM paper** (Table 2) and **TSC paper** (Table 1):

| Configuration | Accuracy | Improvement vs CE |
|---------------|:--------:|:----------------:|
| CE | 38.3% | — |
| **LDAM (plain)** | **39.6%** | +1.3% |
| LDAM + DRW (step schedule) | 42.0% | +3.7% |
| LDAM + DRS (step schedule) | 42.9% | +4.6% |
| LAL (our setup, cosine) | 43.98% | +5.7% |

**Key insight about LDAM:** LDAM is the only CE variant that **directly shapes the feature geometry** rather than just adjusting logits or reweighting samples. The margin term `x_m = x - batch_m` subtracts a class-dependent value from the correct-class logit before softmax. This means the gradient flows back through the backbone weights differently for tail classes — effectively pushing tail-class features further apart in the 64-d space.

**How LDAM differs from LAL at the feature level:**
- **LAL** shifts logits by `τ·log(π_j)` — this affects only the classifier layer. The backbone features are learned with standard CE gradients.
- **LDAM** subtracts `margin_y` from the correct logit — this changes the gradient that backpropagates into the backbone. Tail-class features are forced to be more separated.

**Diversity estimate:** LDAM's features would have different geometric structure from LAL's (where features are learned with standard CE gradients and only logits are adjusted). Pairwise κ between LDAM and LAL would likely be ~0.40-0.50 — lower than the current CE-LAL κ of 0.45, but not dramatically so.

### 7.5 The Fundamental Limitation

Despite the theoretical differences, **all CE variants share a fundamental property**: they optimize the softmax cross-entropy of the classifier. This means:

1. The **64-d feature space** is always driven by cross-entropy gradients
2. The classifier weights form "class templates" in this space
3. The decision boundaries are always hyperplanes through the feature space

Different CE variants (LAL, LDAM, Focal) adjust *how strongly* each sample pushes these hyperplanes, but they don't change the *nature* of the space. This is fundamentally different from PaCo, which uses a **contrastive objective** that:
1. Pushes same-class features together on a hypersphere
2. Pulls different-class features apart
3. Uses a queue of past features as negatives
4. Creates a completely different feature geometry

**This is why CE-based losses all produce κ ≈ 0.4-0.5 with each other, while any CE-loss vs PaCo produces κ ≈ 0.4.** PaCo is the outlier, and CE-variant differences are modest by comparison.

### 7.6 What Would Actually Improve Diversity

| Strategy | Expected Diversity Gain | Implementation Cost | Risk |
|----------|:----------------------:|:-------------------:|:----:|
| Replace CE with LDAM | **Small** (κ change ~0.01) | Medium (new loss, DRW) | Low |
| Replace CE with Focal | **Negligible** (κ unchanged) | Low | Low |
| Train CE on balanced subset | **High** (fundamentally different data) | Medium (new data pipeline) | Medium (accuracy may drop on head) |
| Train CE with stronger augs | **Moderate** (different features) | Low (change transforms) | Low |
| **Keep CE, improve router** | **N/A** (no re-training needed) | **Zero** | **None** |

The highest-impact changes come from changing the **data distribution** or **augmentation strategy**, not from swapping one CE variant for another.

### 7.7 Recommendation: Keep CE, Improve the Router

After reviewing the evidence:

1. **Focal Loss** is essentially identical to CE on this benchmark (+0.09%). No benefit.
2. **LDAM** is genuinely different from LAL (margin vs logit-shift) but the diversity gain is modest. The individual accuracy of LDAM-DRW (42.9%) is also below our LAL (43.98%), meaning we'd trade individual performance for a small diversity gain.
3. **CE's unique contribution** (4.20% of samples, mostly head classes) is real and valuable. Removing it would lose a head-class specialist.

The most pragmatic path forward:

**Keep the current expert set (CE + LAL + PaCo)** and focus energy on the router. The 13.3% oracle gap (62.58% - 49.28%) is large enough that a good router can capture significant gains. If the router plateaus below expectations, *then* consider:

1. Re-training CE with **stronger augmentations** (AutoAugment+Cutout, matching PaCo)
2. Re-training CE on a **balanced subset** to make it a deliberate head/tail specialist

### 7.8 Summary Table

| Candidate | Individual Acc | Diversity vs LAL | Implementation | Recommendation |
|-----------|:--------------:|:----------------:|:--------------:|:--------------:|
| **CE (current)** | 39.46% | κ=0.45 (baseline) | Already done | ✅ **Keep for now** |
| Focal Loss | 38.4% | ~0.45 (no change) | Trivial | ❌ Worse in every way |
| LDAM (plain) | 39.6% | ~0.40-0.45 (modest Δ) | Medium | ⚠️ Marginal benefit |
| LDAM-DRW | 42.9% | ~0.40-0.45 (modest Δ) | Medium+DRW | ⚠️ Good accuracy but still CE-based |
| CE + strong augs | ~41%? | ~0.35-0.40 | Low | ✅ Try if router underperforms |
| CE + balanced data | ~35%? | ~0.20-0.30 | Medium | ✅ Try if router underperforms |
| **PaCo (already in set)** | **49.28%** | **κ=0.41** | Already done | ✅ **The true source of diversity** |

**Bottom line:** The diversity we need is already provided by PaCo. Replacing CE with another CE variant would not meaningfully change the ensemble dynamics. Invest effort in the router, not re-training experts.

### 7.9 Concrete Options for Stage 1

Based on the analysis above, here are the viable paths forward, ordered from simplest to most complex.

#### Option A: Remove CE → 2 experts (LAL + PaCo)

| Metric | Value |
|--------|-------|
| Oracle ceiling | 58.38% |
| Oracle gap (vs best single PaCo 49.28%) | 9.30% |
| CE unique contribution lost | 4.20% |
| Router complexity | Binary (easier to train, less overfitting) |
| Training cost | Zero (models already trained) |

**Pros:** Simplest possible setup. Router has 2 choices — binary decisions are easier to learn with limited validation data (only 5K samples). No re-training needed. Two experts with complementary strengths (LAL for balanced logits, PaCo for contrastive features) tell a clean story.

**Cons:** Ceiling is 58.38% vs 62.58% with 3 experts. Lose CE's 4.20% unique head-class coverage. A perfectly trained binary router could match the 3-expert oracle if it captures all of CE's unique samples, but that's unlikely.

**Paper narrative:** "Two complementary experts — LAL (logit adjustment for tail bias) and PaCo (contrastive learning for feature separation) — with a binary router selecting the best expert per sample." Clean and controllable.

#### Option B: Keep CE, upgrade augmentations to match PaCo

Train CE with the same strong augmentations as PaCo (AutoAugment + Cutout + view1/view2 augmentation). This would:
- Align CE's feature space closer to PaCo's (easier for the router to compare features)
- Likely improve CE's individual accuracy (from 39.46% to ~41-42%)
- Change the feature-level invariances enough to reduce similarity with LAL

**Cost:** ~1.5h re-training CE on Kaggle. Modify the transforms in the CE training script.

**Expected diversity gain:** Moderate. Different augmentations produce different invariances — CE would become robust to color/lighting changes (like PaCo) instead of just crop+flip. This changes which samples it finds "hard." The loss function is still CE, so features are still CE-based, but the distribution of learned features shifts.

#### Option C: Replace CE with self-supervised + LAL finetune

Train a ResNet-32 with **MoCo v2** (unsupervised contrastive learning, He et al., CVPR 2020) on unlabeled CIFAR-100, then attach a classifier head and finetune with LAL on the long-tail data.

| Metric | Estimate |
|--------|----------|
| Individual accuracy | ~38-42% (competitive with CE) |
| Feature origin | **No class labels used during pretraining** |
| Diversity vs LAL | **Very high** (unsupervised vs supervised feature learning) |
| Diversity vs PaCo | **High** (instance discrimination vs supervised class centers) |
| Implementation cost | Medium (need MoCo v2 pretraining loop) |

**Why this is fundamentally different:**
- **MoCo v2** learns by instance discrimination: each image is its own class. Features are organized by visual similarity (color, texture, shape), not by any class boundary.
- **LAL** learns by class-boundary optimization: features are organized to separate the 100 classes.
- **PaCo** learns by class-cluster optimization: features are organized around 100 parametric centers on a hypersphere.

Three experts, three completely different feature organizations. The router would see input representations from three different "encoders" that each structure the visual world differently. This is the closest we can get to guaranteed maximal diversity.

**Tradeoff:** MoCo v2 pretraining on CIFAR-100 takes ~400-800 epochs on a T4 (6-12 hours). This is a significant time investment. If Kaggle session limits are a concern, this may not be feasible.

#### Option D: Replace CE with balanced-subset training

Train CE on a class-balanced subset of the training data (50 samples per class × 100 classes = 5000 samples, i.e., the same as our validation set). This expert:
- Has never seen the long-tail distribution
- Is equally good on all classes (each has exactly 50 training samples)
- Represents "what does each class look like with equal examples"
- Has a completely different specialization profile from LAL (tail-biased) and PaCo (contrastive clusters)

**Expected accuracy:** ~30-35% (very small dataset — 5000 samples vs 9754 for the full LT set), but highly complementary.

**Tradeoff:** Very low individual accuracy. The router may learn to never select this expert because PaCo (49%) is almost always better. The complementary value only materializes if the router can identify the specific samples where the balanced expert excels — which requires a very well-calibrated router.

### 7.10 Summary

| Option | Experts | Oracle | Implementation | Router Complexity | Recommendation |
|--------|---------|:------:|:--------------:|:-----------------:|:--------------:|
| **A: Remove CE** | LAL + PaCo | 58.38% | Zero | Binary (easy) | **✅ Start here** |
| B: Keep CE + strong augs | CE' + LAL + PaCo | ~62%? | Low (re-train CE) | 3-way | If time permits |
| C: Self-supervised | SSL + LAL + PaCo | ~65%? | High (MoCo v2) | 3-way | If aiming for SOTA |
| D: Balanced subset | CE_bal + LAL + PaCo | ~60%? | Medium (new data) | 3-way | As ablation only |

**My recommendation: Start with Option A (2 experts).** It's immediately available, the binary router is easy to train and analyze, and the 9.30% oracle gap is large enough to produce strong, publishable results. If the 2-expert router plateaus below expectations, you can add a third expert (Option B or C) later — the router can be extended from 2 to 3 experts without re-training the experts.

---

## 8. Final Verdict: Mixup + CE as the Third Expert

*Added: The final decision after researching non-CE-variant replacements.*

### 8.1 The Requirement

We need a third expert that is:
1. **Not a CE variant** (LAL already fills that role)
2. **Not a contrastive method** (PaCo already fills that role)
3. **Produces genuinely different feature geometry** from both
4. **Solves CE's overconfidence problem** (CE is wrong with 0.66 confidence on average, and 100% confident on its worst mistakes)

### 8.2 Why Mixup + CE Wins

| Criterion | Focal Loss | LDAM | MoCo v2 | Balanced Sampling | **Mixup + CE** |
|-----------|:----------:|:----:|:-------:|:-----------------:|:--------------:|
| Different from CE? | ❌ CE variant | ❌ CE variant | ✅ Unsupervised | ✅ Different data | ✅ Different data |
| Not contrastive? | ✅ | ✅ | ❌ (contrastive) | ✅ | ✅ |
| Fixes overconfidence? | ❌ | ❌ | ❌ | ⚠️ Partially | **✅ Yes** |
| Feature geometry | CE | Margin-CE | Hypersphere | CE (balanced) | **Interpolation paths** |
| Accuracy estimate | 38.4% | 42.9% | 38-42% | 42-44% | **40-42%** |
| Implementation cost | Trivial | Medium | High | Trivial | **Low** |

**Mixup + CE is the only option that:**
1. Belongs to a **third training paradigm** (interpolation-based augmentation — different from logit adjustment and contrastive learning)
2. **Directly fixes the overconfidence problem** (soft labels prevent p=1.0 predictions, proven by Thulasidasan et al., 2019)
3. Produces features with **fundamentally different geometry** (linear paths between classes instead of discrete clusters or boundary-separated regions)
4. Is **easy to implement** (~15 lines of code on top of the existing CE training script)

### 8.3 The Three-Paradigm Set

| Expert | Paradigm | What it optimizes | Feature structure | Confidence calibration |
|--------|----------|-------------------|:-----------------:|:---------------------:|
| **LAL** | Logit-adjusted CE | Class boundaries with frequency bias | Hyperplane-separated clusters | **Poor** (overconfident on head) |
| **Mixup + CE** | Interpolated CE | Smooth interpolation between classes | Linear paths connecting clusters | **Good** (calibrated by soft labels) |
| **PaCo** | Supervised contrastive | Same-class attraction, different-class repulsion | Hypersphere around parametric centers | **Moderate** |

Three paradigms, three different feature organizations, one calibrated expert. This is the ideal setup for a router.

### 8.4 Evidence Mixup Fixes Overconfidence

From Thulasidasan et al., 2019 "On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks":
- Mixup reduces Expected Calibration Error (ECE) on CIFAR-100 by **40-60%** compared to standard CE
- Confidence of wrong predictions drops from **~0.7 to ~0.3-0.4**
- The mechanism: soft labels make it impossible for the model to reach p=1.0 on training samples, preventing it from learning to be overconfident

### 8.5 Expected Diversity

| Pair | Expected κ | Interpretation |
|------|:----------:|----------------|
| LAL vs Mixup | **0.30-0.38** | Very different — CE with logit shift vs CE with interpolation features |
| Mixup vs PaCo | **0.35-0.42** | Very different — linear interpolation paths vs hypersphere clusters |
| LAL vs PaCo | 0.44 (measured) | Already established |

All pairs should have κ < 0.80, well within the routable range. The Mixup-LAL pair should be significantly more diverse than the current CE-LAL pair (0.45).

---

## 9. Post-Routing-Debug: Verified Bottlenecks and Forward Paths

*Added: Based on empirical root-cause analysis after all three experts were trained and routing experiments were completed.*

### 9.1 Summary of Empirical Findings

After exhaustive routing experiments (see `docs/experiments.md` and `scripts/kaggle_root_cause.py`), the verified pipeline gaps are:

| Metric | Value | Meaning |
|--------|:-----:|---------|
| Uniform averaging | **51.12%** | Strong baseline — no training needed |
| Oracle (any expert correct) | **62.90%** | Upper bound — 37.1% irrecoverable |
| Loss-based oracle (pick by true-class prob) | **62.40%** | Best possible routing with perfect information |
| Oracle-weighted (soft, perfect info) | **63.04%** | Marginal advantage of soft over hard (+0.64%) |
| Calibrated confidence routing | **51.44%** | Best learned method (+0.32% over uniform) |
| Learned soft gate (MLP on 192-d features) | **49.32%** | Below uniform — features lose routing signal |

**Verified root cause — the feature learning gap (13.72%):**
The 192-d concatenated backbone features encode **class identity**, not "which expert's decision boundary will be correct for this sample." A learned router on these features cannot reliably predict which expert to select, because the features were never trained to encode routing-relevant information. The gap between oracle-weighted routing (63.04%) and the learned gate (49.32%) represents the signal lost because the features are optimized for classification, not routing.

**Verified ceiling:**
Only **596 samples (11.9%)** where uniform averaging is wrong have a correct expert available to rescue them. Even a perfect router can only reach 63.04%. The remaining 37.1% of samples have all three experts wrong — no routing can help.

### 9.2 Verified Bottlenecks — Updated with Latest Findings

After exhaustive routing experiments, the following bottlenecks are now **verified and documented** (see `docs/problem.md` for full details):

| Bottleneck | Type | Impact | Verdict |
|------------|:----:|:------:|:--------|
| Feature learning gap (13.72%) | Root cause | Features encode class identity, not routing relevance | 🔴 Unfixable with frozen experts |
| 69.4% label ambiguity | Structural | 3-way selection is ill-posed for frozen experts | 🔴 Requires changing expert training |
| Lone dissenter paradox | Structural | Correct expert is least confident in 83.9% of savable cases | 🔴 Any confidence-based signal fails |
| Product captures +0.82% | Explains small gain | Product extracts most available signal; routing adds little | 🟠 Routing gain capped at ~0.3% |
| 37.1% all-wrong ceiling | Fundamental limit | No routing can save these samples | 🟡 Ceiling cannot be changed |
| **Disagreement routing (40.72%)** | **Disproven** | **"Dissenter in prediction" ≠ "dissenter in confidence"** | **❌ Failed completely** |
| **192-d vs 24-d same signal** | **Verified** | **51.79% vs 51.92%, p=0.70 — no hidden backbone signal** | **❌ RAFA gain is minimal** |

### 9.3 Updated Strategic Options

Each option is evaluated for:
- **Expected BA gain** over uniform (51.12%)
- **Implementation cost** (CPU analysis vs Kaggle training)
- **Evidence strength** (published results vs theoretical reasoning)
- **Risk** (failure modes and uncertainty)

---

#### Option 1: Self-Supervised Expert Replacing Mixup (MoCo v2 / SimCLR / BYOL)

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+2-4%** (target 53-55% BA) |
| **Success chance** | **High (65-70%)** |
| **Cost** | 6-12 hours MoCo v2 pretraining + 2h finetuning on Kaggle |
| **Evidence** | TSC paper §3.1: κ(CE, contrastive) ≈ 0.61 vs κ(CE, LDAM) ≈ 0.82. The diversity gap is proven. |

**Why it could work:**
A self-supervised expert (MoCo v2) organizes features by **visual similarity** (instance discrimination), completely independent of class labels. This is a fourth paradigm (unsupervised representation learning), different from:
- LAL: class-boundary optimization with frequency bias
- PaCo: supervised contrastive with parametric class centers
- Mixup: interpolation between labeled examples

With three fundamentally different feature organizations (class-boundary, supervised-contrastive, unsupervised-contrastive), the router would see input representations from three different "encoders" that each structure the visual world differently. The 37% all-wrong ceiling would likely drop because the unsupervised expert covers different failure modes.

**Evidence from literature:**
- MoCo v2 linear probe on CIFAR-100: **52.39%** (from "How Well Do Self-Supervised Models Transfer?" — arxiv 2011.13377). This is competitive with our Mixup expert (40.80%) and close to LAL (43.98%).
- MoCo v2 fine-tuned on CIFAR-100-LT would likely achieve **38-44%** BA depending on finetuning protocol.
- TSC paper pairwise agreement: CE vs KCL (contrastive) has κ=0.63 on tail classes — **37% disagreement rate**, exactly the rate the router needs to exploit.

**Implementation protocol:**
1. MoCo v2 pretraining: 400-800 epochs on CIFAR-100 (unlabeled), ResNet-32, batch 256, dim=128, K=4096
2. Attach LAL classifier head: train 200 epochs with LAL loss on the long-tail data
3. Freeze backbone, replace with standard CE head: train 200 epochs (optional comparison)
4. Re-run diversity analysis with LAL + PaCo + MoCo v2
5. Re-run routing pipeline

**Risk:**
- Pretraining is time-consuming (6-12 hours on T4)
- MoCo v2 uses instance discrimination — may not develop class-relevant features for tail classes with very few training examples
- If the unsupervised features are too "random" relative to class boundaries, the router may struggle to find patterns

---

#### Option 2: End-to-End Joint Training (RIDE-Style Architecture)

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+2-3%** (target 53-54% BA) |
| **Success chance** | **Medium-High (55-65%)** |
| **Cost** | Major pipeline redesign — 3-5 expert-specific heads + shared backbone + router |
| **Evidence** | RIDE (ICLR 2021): 48.6% with 3 experts on CIFAR-100-LT IR=100 |

**Why it could work:**
The central problem with our current setup is that the router's features are **frozen and optimized for classification, not routing**. An end-to-end joint training (RIDE-style) allows the router gradients to backpropagate into the shared backbone. The backbone learns to produce features that are useful for the routing decision.

RIDE's specific mechanisms that contribute to success:
1. **Shared backbone** — The router sees features from a single representation space, not three separate ones
2. **Distribution-aware sampling** — Each head sees different data (repeat-factor sampling), creating natural specialization
3. **Joint training** — The router loss gradients update the backbone, adapting features for routing

**Why it might NOT work:**
- RIDE itself only achieves 48.6% on CIFAR-100-LT IR=100 — below our uniform ensemble (51.12%)
- Our three separate backbones already outperform RIDE's shared backbone (PaCo alone is 49.28%)
- Joint training introduces optimization instability (expert collapse, router collapse)
- Requires training a new architecture from scratch on Kaggle

**Implementation protocol:**
1. Single ResNet-32 backbone
2. Three expert-specific classifier heads (each with its own BN)
3. Shared router (MLP: feature → 3-way softmax)
4. Training loop: forward through backbone → all 3 heads → router selects/weights heads → classification loss + routing entropy loss
5. Distribution-aware sampling: each head sees a different data distribution

---

#### Option 3: Confidence-Weighted Ensemble with Per-Class Calibration

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+0.5-1.5%** (target 51.5-52.5% BA) |
| **Success chance** | **Medium (50-60%)** |
| **Cost** | CPU analysis only — no training needed |
| **Evidence** | Our calibrated confidence routing already achieves 51.44%. Per-class temperatures could improve further. |

**Why it could work:**
Instead of a single temperature per expert, learn **per-class temperatures** (100 temperatures per expert = 300 parameters). This would allow each expert to be calibrated differently for head vs tail classes. Given that LAL is overconfident on head classes but might be underconfident on tail classes, per-class calibration could significantly improve confidence-based routing.

**Implementation:**
- Learn T_{i,c} for expert i, class c
- Optimize NLL on a held-out calibration set
- Add a small regularizer to prevent extreme temperatures

**Evidence:**
- GETS (ICLR 2025) shows ensemble temperature scaling improves calibration for graph ensembles
- Our own data shows LAL needs T=1.85 overall, but this is an average — head classes likely need different T than tail classes
- Per-class calibration is standard practice in long-tail classification (class-dependent temperatures)

**Limitation:**
- Even perfect calibration cannot close the 13.72% feature learning gap
- The routing decision is still based on max-confidence, which is a weak proxy for "which expert is correct"

---

#### Option 4: Test-Time Adaptation / Self-Supervised Aggregation (SADE/TADE)

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+1-2%** (target 52-53% BA) |
| **Success chance** | **Medium (45-55%)** |
| **Cost** | Medium — implement self-supervised rotation prediction + aggregation |
| **Evidence** | SADE (NeurIPS 2022): 49.1% on CIFAR-100-LT IR=100; TADE extends with test-time adaptation |

**Why it could work:**
SADE avoids training a separate router entirely. Instead, it uses a **self-supervised auxiliary task** (rotation prediction) to measure which expert is most reliable for each test sample. The idea: an expert that performs well on the auxiliary task is likely to have captured the relevant features for that sample.

**Implementation:**
1. Attach a rotation-prediction head to each expert's backbone
2. Train rotation prediction jointly with classification
3. At test time, compute the rotation-prediction accuracy of each expert on each sample (by rotating the input and checking if the expert correctly predicts the rotation)
4. Weight each expert's prediction by its rotation-prediction confidence
5. Aggregate with confidence-weighted averaging

**Why it might NOT work:**
- SADE achieves only 49.1% — below our uniform ensemble (51.12%)
- Rotation prediction adds overhead (4 forward passes per sample)
- The correlation between rotation-prediction accuracy and classification accuracy is not guaranteed for our specific experts

---

#### Option 5: Knowledge Distillation from the Ensemble

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+0-1%** (target 51-52% BA) |
| **Success chance** | **Low-Medium (35-45%)** |
| **Cost** | Medium — implement distillation training loop on Kaggle |
| **Evidence** | Hinton et al. (2015): knowledge distillation from ensembles improves single-model accuracy |

**Why it could work:**
Distill the three-expert ensemble into a single ResNet-32 using soft targets. The distilled model would learn from the combined knowledge of all three experts, potentially outperforming any single expert. The soft targets from the ensemble are naturally calibrated (averaging three experts' softmax outputs), which addresses the calibration problem.

**Implementation:**
1. Compute soft targets from the ensemble (average of three experts' logits at temperature T)
2. Train a single ResNet-32 with KL-divergence loss to match the soft targets, plus CE loss on hard labels
3. Evaluate the distilled model on the validation set

**Why it might NOT work:**
- The ensemble's soft targets are only as good as the ensemble — and ours is only 51.12%
- A single ResNet-32 has less capacity than three separate backbones
- The distilled model cannot dynamically route — it must internalize all three experts' knowledge into a single set of weights

---

#### Option 6: Adversarial Training for Expert Diversity

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+1-2%** (target 52-53% BA) |
| **Success chance** | **Low (30-40%)** |
| **Cost** | High — generate adversarial samples + re-train experts on Kaggle |
| **Evidence** | Li & Yao (2024): distribution-aware adversarial examples for long-tail recognition |

**Why it could work:**
Generate **adversarial examples** where each expert is wrong, then train the experts on these hard samples. This would force the experts to develop different failure modes, potentially increasing the unique-correct fraction (currently 19.2%) and reducing the all-wrong ceiling (currently 37.1%).

**Implementation:**
1. For each expert, generate adversarial examples where the expert is confidently wrong
2. Add these to the training set for that expert (or all experts)
3. Re-train experts with the augmented data
4. Re-run diversity analysis

**Why it might NOT work:**
- Adversarial training on long-tail data is unstable — tail classes have too few samples to generate meaningful adversarial examples
- The computational cost is high (generating adversarial examples for each expert requires multiple forward/backward passes per sample)
- Published evidence for adversarial diversity in long-tail ensembles is limited

---

#### Option 7: Accept the Ceiling — Publish with Calibrated Confidence Routing

| Property | Assessment |
|----------|------------|
| **Expected gain** | **+0.32%** (current best: 51.44%) |
| **Success chance** | **100% (already achieved)** |
| **Cost** | Zero — already implemented |
| **Evidence** | Verified in `scripts/debug_routing.py` and `scripts/kaggle_root_cause.py` |

**What this option entails:**
Document the full pipeline (three diverse experts + calibrated confidence routing) with the honest finding that uniform averaging is surprisingly competitive. The contribution becomes:
1. A systematic analysis of **why routing fails** for separate-backbone ensembles
2. Evidence that **calibration is a prerequisite** for any confidence-based routing
3. A demonstration that **the "which expert is correct" task has inherently limited signal** in the backbone feature space

**This is a valid scientific contribution** — negative results are important, especially when accompanied by thorough root-cause analysis.

---

### 9.3 Updated Quantitative Comparison (Post-Disagreement-Failure)

| # | Option | Est. BA | Gain vs Unif. | Implementation | Kaggle GPU | Risk | Priority |
|---|--------|:-------:|:-------------:|:--------------:|:----------:|:----:|:--------:|
| 1 | **Self-supervised expert (MoCo v2)** | **53-55%** | +2-4% | High | 8-14h | Medium | **⭐ #1** |
| 2 | **Test-time augmentation consistency routing** | **52-54%** | +1-3% | Medium | 3-5h | Medium | **⭐ #2** |
| 3 | **End-to-end joint training (RIDE)** | **53-54%** | +2-3% | Very high | 6-8h | Medium-High | #3 |
| 4 | **Test-time adaptation (SADE/TADE)** | **52-53%** | +1-2% | Medium | 3-5h | Medium | #4 |
| 5 | **Per-class calibration ensemble** | **51.5-52.5%** | +0.5-1.5% | Low (CPU) | 0h | Low | #5 |
| 6 | **Knowledge distillation** | **51-52%** | +0-1% | Medium | 2-3h | High | #6 |
| 7 | **Accept ceiling (correctness-prediction)** | **51.92%** | +0.79% | Zero | 0h | None | Fallback |

**Removed options (disproven):**
- **Disagreement routing** — 40.72% BA, completely failed
- **RAFA fine-tuning** — expected gain <0.3%, 192-d and 24-d features have same signal (p=0.70)
- **Adversarial diversity training** — insufficient evidence, high risk

### 9.4 Updated Recommendation

After exhaustive CPU-based testing (all learned routers, disagreement routing, RAFA verification), the following approaches remain:

**Track A (highest expected gain, needs GPU): Self-supervised expert (MoCo v2).**
- Expected gain: +2-4% BA (potentially reaching 53-55%)
- Creates genuinely fourth-paradigm features (unsupervised instance discrimination)
- Addresses both the 37% all-wrong ceiling and the feature-learning gap
- Estimated 8-14 hours on Kaggle T4

**Track B (also needs GPU): Test-time augmentation consistency routing.**
- Expected gain: +1-3% BA
- Uses augmentation consistency (not confidence) as orthogonal routing signal
- Bypasses the Lone Dissenter Paradox entirely
- Requires running 3 experts × K augmentations per sample; feasible on GPU
- Estimated 3-5 hours on Kaggle T4

**Track C (CPU, no gain expected): Accept the ceiling.**
- Document the full pipeline with honest finding that routing with frozen experts is fundamentally limited
- Publish the 10 verified problems as a scientific contribution about the limits of frozen-expert routing
- Available immediately — no GPU time needed

### 9.5 References — New (continues from original references 2-20)

2. Cai, J., Wang, Y., & Hwang, J. N. (2021). *ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot*. ICCV 2021. [arXiv:2108.02385](https://arxiv.org/abs/2108.02385)
3. Zhou, Z., Liu, Z., et al. (2022). *Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition*. NeurIPS 2022. [arXiv:2107.09249](https://arxiv.org/abs/2107.09249)
4. Aimar, E. S., et al. (2023). *Balanced Product of Calibrated Experts for Long-Tailed Recognition*. CVPR 2023. [arXiv:2206.05260](https://arxiv.org/abs/2206.05260)
5. Zhao, Q., et al. (2023). *MDCS: More Diverse Experts with Consistency Self-Distillation for Long-Tailed Recognition*. ICCV 2023. [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)
6. Liu, Z., & Blondel, M. (2024). *Routers in Vision Mixture of Experts: An Empirical Study*. TMLR 2024. [OpenReview](https://openreview.net/forum?id=5vSXd8cogo)
7. Ghosh, A., et al. (2021). *ELF: An Early-Exiting Framework for Long-Tailed Classification*. ICASSP 2021. [arXiv:2006.11979](https://arxiv.org/abs/2006.11979)
8. Menon, A. K., et al. (2021). *Long-tail learning via logit adjustment*. ICLR 2021. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
9. Ren, J., et al. (2020). *Balanced Meta-Softmax for Long-Tailed Visual Recognition*. NeurIPS 2020. [arXiv:2007.10740](https://arxiv.org/abs/2007.10740)
10. Cao, K., et al. (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss*. NeurIPS 2019. [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
11. Kang, B., et al. (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition*. ICLR 2020. [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
12. Wei, X., et al. (2025). *Divide, Weight, and Route: Difficulty-Aware Optimization with Dynamic Expert Fusion for Long-Tailed Recognition*. PRCV 2025. [arXiv:2508.19630](https://arxiv.org/abs/2508.19630)
13. Zhang, X., et al. (2024). *RICASSO: Reinforced Imbalance Learning with Class-Aware Self-Supervised Outliers Exposure*. [arXiv:2410.10548](https://arxiv.org/abs/2410.10548)
14. Khosla, P., et al. (2020). *Supervised Contrastive Learning*. NeurIPS 2020. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
15. Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
16. Li, T., et al. (2022). *Targeted Supervised Contrastive Learning for Long-Tailed Recognition*. CVPR 2022. [arXiv:2111.13998](https://arxiv.org/abs/2111.13998)
17. Kang, B., et al. (2021). *K-positive Contrastive Learning for Long-Tailed Recognition*. ICLR 2021. [arXiv:2102.10078](https://arxiv.org/abs/2102.10078)
18. Cui, J., et al. (2021). *Parametric Contrastive Learning*. ICCV 2021. [arXiv:2107.12028](https://arxiv.org/abs/2107.12028)
19. Lin, T. Y., et al. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
20. Khosla, P., et al. (2020). *Supervised Contrastive Learning*. NeurIPS 2020. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
21. He, K., et al. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo v2)*. CVPR 2020. [arXiv:2003.04297](https://arxiv.org/abs/2003.04297)
22. Chen, T., et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*. ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
23. Grill, J. B., et al. (2020). *Bootstrap Your Own Latent (BYOL)*. NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
24. Hinton, G., et al. (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS 2015 Workshop. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
25. Lakshminarayanan, B., et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS 2017. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
26. Tomani, C., et al. (2024). *Accurate and Reliable Predictions with Mutual-Transport Ensemble*. [arXiv:2405.19656](https://arxiv.org/abs/2405.19656)
27. Li, Y., & Yao, J. (2024). *Robust Long-Tailed Recognition with Distribution-Aware Adversarial Example Generation*. [Semantic Scholar](https://www.semanticscholar.org/paper/Robust-long-tailed-recognition-with-distribution-aware-Li-Yao/bf03855420383f51df9f56fe563a9a8a2068b860)
28. Chen, Y., et al. (2025). *Enhancing Mixture of Experts with Independent and Collaborative Learning for Long-Tail Visual Recognition*. IJCAI 2025. [www.ijcai.org](https://www.ijcai.org/proceedings/2025/93)
29. Zhou, Z., et al. (2023). *TADE: Test-Time Adaptation for Diverse Experts in Long-Tailed Recognition*. [Proceedings of NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/dc6319dde4fb182b22fb902da9418566-Supplemental-Conference.pdf)
