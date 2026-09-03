# Novel Routing Ideas — Comprehensive Analysis

> **Context:** This report analyzes novel approaches to solve the fundamental problem identified after 25+ routing methods across 5 rounds of experiments on CIFAR-100-LT (IR=100). The root cause: **frozen experts' features encode class identity, not routing relevance.** The only way forward is to change what the experts provide so that routing-relevant information naturally emerges.
>
> This analysis evaluates 7 distinct ideas (plus hybrid combinations) for novelty, feasibility, likely effectiveness, and implementation path.

---

## Table of Contents

1. [Recap: The Three Fundamental Limitations](#1-recap-the-three-fundamental-limitations)
2. [Evaluation Framework](#2-evaluation-framework)
3. [Idea A: Boosting-Style Adversarial Expert Training](#3-idea-a-boosting-style-adversarial-expert-training)
4. [Idea B: Routing-Aware Auxiliary Loss (RAAL)](#4-idea-b-routing-aware-auxiliary-loss-raal)
5. [Idea C: Expertise Maps — Per-Class Competence Outputs](#5-idea-c-expertise-maps--per-class-competence-outputs)
6. [Idea D: Feature-Space Geometry Routing](#6-idea-d-feature-space-geometry-routing)
7. [Idea E: Contrastive Routing Embeddings](#7-idea-e-contrastive-routing-embeddings)
8. [Idea F: Confidence-Adaptive & Value-of-Information Routing](#8-idea-f-confidence-adaptive--value-of-information-routing)
9. [Idea G: Joint Gating-Expert Training with Specialization](#9-idea-g-joint-gating-expert-training-with-specialization)
10. [Hybrid Combinations](#10-hybrid-combinations)
11. [Head-to-Head Comparison](#11-head-to-head-comparison)
12. [Recommended Next Steps](#12-recommended-next-steps)

---

## 1. Recap: The Three Fundamental Limitations

Any novel approach must address at least one of these:

| # | Limitation | What It Means | Severity |
|:-:|:-----------|:--------------|:--------:|
| L1 | **Feature Learning Gap (19.06%)** | Backbone features encode class identity, not "which expert is best." Oracle-weighted routing = 63.04%, learned gate = 43.98%. | 🔴 CRITICAL |
| L2 | **37.1% All-Wrong Ceiling** | All three experts fail on the same hard samples. No routing can salvage these. | 🔴 CRITICAL |
| L3 | **Lone Dissenter Paradox** | In savable samples, the correct expert is the *least* confident 83.9% of the time. Any confidence-based signal points AWAY from it. | 🔴 CRITICAL |

**Secondary problems** (also addressed by some ideas):
- **69.4% label ambiguity** — most samples have no unique "best" expert
- **Product captures +0.82%** — little signal left for routing to add
- **Gradient orthogonality in high-D** — gradient directions near-orthogonal regardless of correctness

---

## 2. Evaluation Framework

Each idea is evaluated on these dimensions:

| Dimension | Scale | What It Measures |
|:----------|:-----|:-----------------|
| **Novelty** | ★★★ | How differentiated from existing work (RIDE, SADE, BPCE, MDCS, DNCC) |
| **Expected Impact** | Low / Med / High | Estimated BA improvement over current best (52.70% selective 92-d) |
| **Which Limitations Addressed** | L1/L2/L3 | Which of the three fundamental problems it tackles |
| **Implementation Cost** | Low / Med / High | Engineering effort to implement |
| **GPU Cost** | Hours | Estimated training time on T4/A100 |
| **Risk** | Low / Med / High | Likelihood of failure or marginal gain |
| **Compatibility** | ★★★ | How well it works with existing codebase (separate backbones) |

---

## 3. Idea A: Boosting-Style Adversarial Expert Training

### Concept

Train experts **sequentially** where each expert is trained to cover the gaps of the previous ones:

1. **Expert A** — trained normally (e.g., LAL on full LT data)
2. **Expert B** — trained with samples **upweighted** where Expert A is wrong
3. **Expert C** — trained with samples **upweighted** where *both* A and B are wrong

The upweighting can be implemented as:
- **Soft:** Sample weights in the loss function (e.g., `weight = 1 + α × [A_wrong]`)
- **Hard:** Create a subset of the data dominated by A's errors
- **Gradient-based:** Boost the learning rate on A's errors

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ✅ **Directly.** Each expert's features encode "what kind of samples the previous experts fail on" — a routing-relevant signal emerges naturally. Expert B's features will be organized around "samples A gets wrong," which is exactly what a router needs to know. |
| **L2 (37.1% All-Wrong Ceiling)** | ✅ **Shrinks it.** Expert C is explicitly trained on samples where both A and B fail, so it learns to handle the hardest cases. The all-wrong set should shrink. |
| **L3 (Lone Dissenter Paradox)** | ⚠️ **Partially.** Experts B and C are less likely to be "lone dissenters" because they were trained to be correct where others fail. The paradox may still apply to Expert A's predictions. |

### Expected Impact

**High.** This directly attacks the feature learning gap. The sequential specialization creates natural complementarity. Expected BA: **53-55%** (1-3% above current best).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | Boosting (AdaBoost, XGBoost) uses sequential training, but for weak learners → single strong classifier, not for deep ensembles. Deep boosting variants exist but don't focus on routing signal. |
| **Differentiation** | Novel because: (1) applies boosting-style sequential training to *separate deep networks* kept for ensemble, (2) explicitly optimizes for routing-relevant features, (3) uses error-aware upweighting not sample reweighting. |
| **Publication risk** | Medium. The idea of "training experts on previous experts' errors" has been explored in ensemble diversity literature, but not for creating routing-relevant feature spaces in long-tail recognition. |

### Strengths

1. **Natural specialization** — experts automatically learn to be good at different things
2. **Routing signal emerges naturally** — Expert B's features encode "samples like the ones A gets wrong"
3. **Shrinks the all-wrong ceiling** — Expert C explicitly trained on the hardest cases
4. **Simple to implement** — just modify the data sampling/weighting per expert
5. **Compatible with separate backbones** — no architectural changes needed

### Weaknesses

1. **Expert B and C have less data** — only the samples where previous experts are wrong may be few (especially for Expert C)
2. **Expert A may dominate** — if A is very good, there aren't enough errors for B to learn from
3. **Catastrophic forgetting** — Expert B might forget how to handle easy samples
4. **Sequential training is slower** — can't parallelize expert training
5. **Risk of overfitting** — Expert C trains on a small, hard subset

### Implementation Considerations

**Critical design decisions:**

1. **How to upweight?** Soft weighting (multiply loss by `1 + α × indicator`) is gentler than hard subset selection. Recommended: start with α = 1.0 (2× weight on previous expert's errors).

2. **What fraction of data to upweight?** Expert B should still see all data (to maintain head-class performance) but with errors upweighted. A mix of 80% normal + 20% error-upsampled per batch works.

3. **How to prevent forgetting?** Use a small weight (λ = 0.1-0.3) on the original loss to maintain general performance.

4. **Warm-starting:** Initialize Expert B from Expert A's weights (fine-tuning) vs. train from scratch. Fine-tuning preserves general features while adapting to errors.

5. **Router training:** After all experts are trained, the router can use any of the methods already developed (89-d correctness-prediction routing). The key advantage is that the features now contain routing-relevant signal.

**Pseudo-code:**
```python
# Phase 1: Train Expert A normally
expert_a = train(lt_dataset, loss_fn=lal_loss)

# Phase 2: Train Expert B with error-aware weighting
train_weights = compute_weights(lt_dataset, expert_a)
# weight = 1.0 if expert_a correct, 2.0 if expert_a wrong
expert_b = train(lt_dataset, loss_fn=lal_loss, sample_weights=train_weights)

# Phase 3: Train Expert C with error-aware weighting for both A and B
train_weights = compute_weights_combined(lt_dataset, expert_a, expert_b)
# weight = 1.0 if both correct, 3.0 if both wrong
expert_c = train(lt_dataset, loss_fn=paco_loss, sample_weights=train_weights)
```

### Will It Work?

**Likely yes**, with caveats:
- The **soft weighting** approach is safer and more likely to work than hard subset selection
- The biggest risk is that Expert B doesn't have enough errors to learn from (if Expert A is already good)
- This risk is mitigated by the long-tail nature: Expert A (LAL) has 43.98% BA, so it's wrong on ~56% of samples — plenty of errors for Expert B
- Expert C might struggle because after A and B, only ~37% all-wrong samples remain — but that's still 3,700+ samples, enough for training

**Verdict: Promising. Priority approach.**

---

## 4. Idea B: Routing-Aware Auxiliary Loss (RAAL)

### Concept

During expert training, add an **auxiliary head** that predicts "will I be correct on this sample?" The auxiliary loss gradient flows back into the backbone, forcing features to encode correctness-relevant information.

```
Input x → Backbone → Features f(x)
                    ├──→ Classifier Head → logits → classification loss
                    └──→ Auxiliary Head → p(correct | x) → auxiliary loss
```

The auxiliary head is a simple binary classifier (e.g., 64→1 with sigmoid). The auxiliary loss is binary cross-entropy: "will this expert classify this sample correctly?"

**Total loss:** `L_total = L_cls + λ × L_aux`

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ✅ **Directly.** The backbone is explicitly trained to encode "will I be correct?" information. Features organize not just by class, but also by correctness-predicting patterns. |
| **L2 (37.1% All-Wrong Ceiling)** | ❌ **Does not directly help.** Experts still trained on the same data; their individual accuracy doesn't necessarily improve. |
| **L3 (Lone Dissenter Paradox)** | ✅ **Directly.** The auxiliary head provides a *correctness score* that is decoupled from confidence. The correct expert may have low confidence (Lone Dissenter) but high self-predicted correctness. The router uses correctness scores, not confidence, breaking the paradox. |

### Expected Impact

**Medium-High.** The key gain is breaking the Lone Dissenter Paradox. If the auxiliary head can predict correctness with AUROC > 0.9 (vs 0.84-0.89 for post-hoc trust meters), routing quality improves significantly. Expected BA: **52.5-54%** (0.5-1.5% above current best).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | "Self-awareness" in deep learning (predicting own errors) has been explored in calibration literature, but not as an auxiliary loss during expert training for routing. The project's own correctness-prediction routing is the closest prior art — but that was **post-hoc** on frozen features. |
| **Differentiation** | Novel because: (1) correctness prediction is trained *jointly* with the expert, so features adapt, (2) creates a native "self-awareness" signal that is different from confidence, (3) the auxiliary head's output is a *new type of expert output* specifically designed for routing. |
| **Publication risk** | Low-Medium. The concept of "predicting own errors" has been studied, but applying it as a routing-enabling feature during expert training for long-tail ensembles is novel. |

### Strengths

1. **Directly addresses the Lone Dissenter Paradox** — correctness score ≠ confidence
2. **Features adapt** — the backbone learns to organize by correctness, not just class
3. **Simple to implement** — add one linear layer + BCE loss
4. **Works with existing experts** — can be added to any training script
5. **No architectural changes to inference** — auxiliary head used only at test time for routing
6. **The router already exists** — can use the 89-d/92-d correctness-prediction framework, now with better features

### Weaknesses

1. **Auxiliary loss may compete with classification loss** — the backbone might sacrifice class accuracy for correctness-predictability
2. **Correctness prediction is only as good as the features** — if the features truly don't contain correctness signal, the auxiliary head won't help (but the gradient forces them to)
3. **λ tuning is critical** — too much weight on L_aux hurts classification; too little has no effect
4. **The correctness target is noisy** — "will I be correct?" depends on the classifier head's random initialization, not just the backbone

### Implementation Considerations

**Critical design decisions:**

1. **Auxiliary head architecture:** A simple linear layer (64→1) with sigmoid. No need for complexity — the backbone does the heavy lifting.

2. **Auxiliary loss weight (λ):** Start with λ = 0.1 and sweep [0.01, 0.05, 0.1, 0.5, 1.0]. Monitor primary classification accuracy to ensure it doesn't degrade.

3. **When to compute correctness label?** The correctness label depends on the current model state, which changes during training. Options:
   - **Online:** Use current predictions → noisy but efficient
   - **Stale:** Use a running average of predictions → more stable
   - **Ground truth:** Use the true label → easiest but the auxiliary head might just learn "is this class easy?"

   **Recommendation:** Use ground truth label (the auxiliary head learns "can I classify this sample correctly given my current features?"). This is clean and stable.

4. **Calibration of auxiliary head:** The auxiliary head's output is p(correct). This should be well-calibrated if trained with proper regularization. Add a small weight decay to prevent overconfidence.

5. **Integration with existing experts:** 
   - For **LAL**: Add auxiliary head to `ResNet32` model, modify `train_lal.py`
   - For **PaCo**: Add auxiliary head to `PaCoResNet32` classifier path
   - For **Mixup**: Add auxiliary head to `ResNet32` model, modify `train_mixup.py`

6. **Router at test time:**
   - Extract auxiliary head output p(correct) for each expert
   - Use as routing weights directly (softmax over 3 correctness scores)
   - Or feed into the existing 89-d correctness-prediction framework (now with better features)

**Pseudo-code:**
```python
class ResNet32WithAux(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet32Backbone()
        self.fc = nn.Linear(64, num_classes)
        self.aux_head = nn.Linear(64, 1)  # NEW: predicts p(correct)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        correctness = torch.sigmoid(self.aux_head(features))
        return logits, correctness

# Training
for batch in dataloader:
    logits, correctness = model(images)
    cls_loss = ce_loss(logits, labels)
    correct = (logits.argmax(1) == labels).float()
    aux_loss = bce_loss(correctness.squeeze(), correct)
    total_loss = cls_loss + lambda_ * aux_loss
```

### Will It Work?

**Likely yes**, with important caveats:
- The key question is whether the auxiliary gradient can overcome the dominant classification gradient. With λ=0.1, the classification loss is 10× stronger, so the backbone mostly learns class identity — correctness signal is a minor perturbation.
- However, even a small perturbation could be enough: the project's post-hoc trust meters achieved AUROC 0.84-0.89 on frozen features. If the features are *slightly* better organized for correctness prediction, AUROC could reach 0.90-0.95.
- At AUROC 0.95, the routing quality would improve significantly because the 3-way comparison problem becomes easier (correctness scores are more separable).

**Verdict: Promising, especially in combination with other approaches. Low risk, moderate reward.**

---

## 5. Idea C: Expertise Maps — Per-Class Competence Outputs

### Concept

Each expert outputs not just a class prediction, but a **per-class competence vector** c(x) ∈ [0,1]^K (K=100) where c_k(x) = "how reliable is this expert for class k on this specific sample?"

```
Expert output: (logits ∈ ℝ^K, competence ∈ [0,1]^K)
```

The competence vector is produced by a separate head that shares the backbone. It's trained on a validation set: for each sample, competence_k should be high if the expert correctly classifies samples *like this one* as class k.

**At test time, the router** computes per-class scores and selects the final prediction:
```
score_k = max_e [ softmax(logits_e)_k × competence_e_k ]
```

This is a learned, per-class product combination.

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ✅ **Directly.** The competence head forces features to encode "what I'm good at" — a routing-relevant signal. Features organize not just by class, but by reliability-per-class. |
| **L2 (37.1% All-Wrong Ceiling)** | ❌ **Does not directly help.** But if competence is well-calibrated, the router knows when all experts are unreliable and can abstain or use a fallback. |
| **L3 (Lone Dissenter Paradox)** | ✅ **Directly.** The correct but uncertain expert would have high competence for the correct class (even with low confidence), because competence measures *reliability*, not confidence. The router sees: "Expert C has low confidence for class 42 but high competence — trust it." |

### Expected Impact

**Medium-High.** The competence vector is a richer signal than a single correctness score (Idea B). It provides per-class granularity. Expected BA: **53-54.5%** (0.5-2% above current best).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | Per-class confidence calibration exists (frequency-grouped temperature scaling, per-class temperatures). But per-class *competence* as a learned expert output for routing is novel. |
| **Differentiation** | Novel because: (1) competence is a *new type of expert output*, not a post-hoc calibration, (2) trained jointly with the expert via a validation-set loss, (3) the router uses a per-class max-product combination that naturally handles the 69.4% label ambiguity (multiple experts can be competent for different classes on the same sample). |
| **Publication risk** | Low. No prior work found that trains experts to output per-class competence vectors specifically for routing. |

### Strengths

1. **Per-class granularity** — competence is 100-dimensional, not scalar. The router knows *which classes* each expert is good at.
2. **Handles 69.4% label ambiguity** — multiple experts can be competent for different classes on the same sample. The max-product naturally handles this.
3. **Breaks Lone Dissenter Paradox** — competence ≠ confidence. The correct but uncertain expert has high competence.
4. **Interpretable** — competence vectors reveal what each expert specializes in.
5. **Product combination enhanced** — the project's product combination (+0.82%) was static. Competence-weighted product is dynamic and learned.

### Weaknesses

1. **Complex training** — competence head needs a validation set to train (the project has 5K balanced val set). The loss function is non-trivial.
2. **Per-class training signal is sparse** — for tail classes with 1-2 samples, there aren't enough examples to learn reliable competence.
3. **100× output dimension** — adds 100 parameters per expert (negligible) but the routing computation becomes O(K × E) = 300 comparisons per sample.
4. **Competence may collapse** — if not regularized, competence could become all-1 or all-0. Needs careful training.

### Implementation Considerations

**Critical design decisions:**

1. **Competence head architecture:** A linear layer (64→100) with sigmoid activation. Each output dimension c_k ∈ [0,1] represents competence for class k.

2. **Competence loss function:** For each sample x with true label y*:
   - Positive example: competence[y*] should be high (expert is reliable for the correct class)
   - Negative example: competence[k] for k ≠ y* should be low if the expert tends to confuse k with y*
   
   Loss: `L_comp = -log(c_y*) + (1/K) Σ_{k≠y*} log(1 - c_k)`
   
   This is a multi-label binary cross-entropy loss where the positive label is the true class and all other classes are negative.

3. **Validation set requirement:** Competence is trained on the held-out 5K balanced validation set, not the training set. This prevents the expert from just memorizing "I'm always competent on training data."

4. **Router at test time:**
   ```
   For each class k:
     score_k = max_e [ softmax(logits_e(x))_k × competence_e_k(x) ]
   prediction = argmax_k score_k
   ```

5. **Regularization:** Add entropy penalty to prevent competence collapse. `L_reg = -H(mean_c)` where H is entropy and mean_c is the average competence across classes.

**Pseudo-code:**
```python
class ResNet32WithCompetence(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet32Backbone()
        self.fc = nn.Linear(64, num_classes)
        self.competence_head = nn.Linear(64, num_classes)  # NEW

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        competence = torch.sigmoid(self.competence_head(features))
        return logits, competence

# Training — two phases:
# Phase 1: Train expert normally (classification loss only)
# Phase 2: Freeze backbone, train competence head on validation set
for x_val, y_val in val_dataloader:
    with torch.no_grad():
        features = expert.backbone(x_val)
    competence = torch.sigmoid(competence_head(features))
    # competence[y_val] should be high, others low
    pos_loss = -torch.log(competence[range(B), y_val] + 1e-8)
    neg_loss = -torch.log(1 - competence + 1e-8).mean(1)
    loss = pos_loss + neg_loss

# Router at test time
logits_list = [expert(x) for expert in experts]
competence_list = [expert.competence_head(expert.backbone(x)) for expert in experts]
probs_list = [F.softmax(logits, dim=1) for logits in logits_list]

# Per-class max-product
scores = torch.stack([
    probs * comp 
    for probs, comp in zip(probs_list, competence_list)
]).max(dim=0).values  # (B, 100)
prediction = scores.argmax(dim=1)
```

### Will It Work?

**Moderately likely.** The main risk is training the competence head on the validation set:
- For head classes (many samples): competence should be well-estimated
- For tail classes (few samples): competence estimates will be noisy
- The 5K balanced validation set has 50 samples per class — enough for reasonable estimates

The per-class competence is a richer signal than the scalar correctness score (Idea B), but harder to train. If the competence head can achieve even moderate accuracy (AUROC 0.75-0.80 per class), the router should improve.

**Verdict: Promising but higher risk. The per-class granularity is a strong differentiator.**

---

## 6. Idea D: Feature-Space Geometry Routing

### Concept

Route based on **where the sample falls in each expert's feature space relative to class prototypes**, not on softmax outputs.

For each expert e:
1. Compute class prototypes: μ_k^e = mean of training feature vectors for class k
2. For a test sample with features f_e(x), compute:
   - d_correct = ||f_e(x) - μ_{y*}^e||₂ (distance to correct class prototype)
   - d_nearest_wrong = min_{k≠y*} ||f_e(x) - μ_k^e||₂ (distance to nearest wrong prototype)
3. **Separation ratio:** r_e = d_nearest_wrong / (d_correct + ε)
   - High r_e → sample is well-separated → expert is reliable

Route to the expert with the highest separation ratio.

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ⚠️ **Partially.** Doesn't change features, but uses them differently. Feature-space geometry may contain routing signal that softmax throws away. |
| **L2 (37.1% All-Wrong Ceiling)** | ❌ **Does not help.** If all experts have bad separation for a sample, routing can't help. |
| **L3 (Lone Dissenter Paradox)** | ✅ **Directly.** Feature-space geometry is decoupled from confidence. An expert can be uncertain (high softmax entropy) but have clean feature-space separation. The separation ratio measures geometric reliability, not confidence. |

### Expected Impact

**Low-Medium.** The features still encode class identity, not routing relevance. But geometry provides a different view of the same features. Expected BA: **52-53%** (0-0.5% above current best).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | Prototypical networks (Snell et al., 2017) use class prototypes for few-shot learning. Some MoE routing work uses prototype-based routing. But using separation ratio as a routing signal for pre-trained experts is not well-explored. |
| **Differentiation** | Moderately novel. The separation ratio is a specific geometric quantity that hasn't been tested for expert routing in long-tail recognition. |
| **Publication risk** | Medium. Prototype-based routing exists in other contexts. Need to show significant improvement over simpler methods. |

### Strengths

1. **No training needed** — prototypes are computed from training data, no router training required
2. **Decoupled from confidence** — bypasses the Lone Dissenter Paradox
3. **Interpretable** — you can visualize which expert has the "cleanest" feature space for each sample
4. **Works with frozen experts** — no re-training needed
5. **Complementary to existing methods** — separation ratio can be added as a feature to the 89-d/92-d routing framework

### Weaknesses

1. **Features still encode class identity** — the geometry is a different lens on the same limited information
2. **Prototypes may be poor for tail classes** — with 1-2 samples per tail class, the mean prototype is unreliable
3. **Distance metrics in 64-d space** — Euclidean distance may not capture meaningful separation; cosine distance might work better but needs tuning
4. **Computational cost** — computing distances to 100 prototypes per expert = 300 distance computations per sample

### Implementation Considerations

**Critical design decisions:**

1. **Prototype computation:** Use training set features. For tail classes with few samples, use the validation set to augment or use a regularized prototype (mix of class mean and global mean).

2. **Distance metric:**
   - Euclidean: sensitive to feature scale
   - Cosine: invariant to scale, measures angular separation
   - Mahalanobis: accounts for feature covariance but needs per-class covariance estimation
   
   **Recommendation:** Start with cosine distance (most stable).

3. **Separation ratio variants:**
   - `d_nearest_wrong / d_correct` — high when correct class is much closer
   - `d_nearest_wrong - d_correct` — positive when correct class is closer
   - `softmax(-d_1, ..., -d_K)` — convert distances to "prototype probabilities"
   
   **Recommendation:** Start with ratio (scale-invariant).

4. **Integration with existing routing:**
   - Use separation ratio as a 3-d feature (one per expert) added to the 89-d/92-d set
   - Or use as a standalone routing signal for comparison

**Pseudo-code:**
```python
# Step 1: Compute class prototypes for each expert
prototypes = {}  # expert_name -> (K, 64)
for expert_name, model in experts.items():
    features_list = []
    labels_list = []
    for x, y in train_dataloader:
        feats = model.backbone(x)
        features_list.append(feats)
        labels_list.append(y)
    all_feats = torch.cat(features_list)  # (N, 64)
    all_labels = torch.cat(labels_list)   # (N,)
    protos = []
    for k in range(100):
        mask = all_labels == k
        protos.append(all_feats[mask].mean(0))
    prototypes[expert_name] = torch.stack(protos)  # (100, 64)

# Step 2: At test time, compute separation ratios
def separation_ratio(features, prototypes, predicted_class):
    """features: (64,), prototypes: (100, 64), predicted_class: int"""
    d_correct = F.cosine_similarity(features.unsqueeze(0), 
                                     prototypes[predicted_class].unsqueeze(0))
    d_wrong = F.cosine_similarity(features.unsqueeze(0), prototypes)
    d_wrong[predicted_class] = -1  # mask correct class
    d_nearest_wrong = d_wrong.max()
    return d_nearest_wrong / (d_correct + 1e-8)

# Route to expert with highest separation ratio
ratios = []
for expert_name, model in experts.items():
    feats = model.backbone(x)
    logits = model.fc(feats)
    pred = logits.argmax()
    ratio = separation_ratio(feats, prototypes[expert_name], pred)
    ratios.append(ratio)
best_expert = experts[torch.tensor(ratios).argmax()]
```

### Will It Work?

**Unlikely as a standalone method.** The features still encode class identity, and the geometry is a different view of the same limited information. The project already tested 89-d enriched features that include various distance and similarity measures — the separation ratio adds modest new signal at best.

**However**, it could work well as a *complementary feature* in the existing 89-d/92-d framework. Adding separation ratios as 3 extra features costs nothing and might provide the +0.1-0.3% needed to beat optimal fixed weights by a larger margin.

**Verdict: Weak standalone. Useful as a complementary feature.**

---

## 7. Idea E: Contrastive Routing Embeddings

### Concept

Each expert produces two outputs:
1. **Classification features** (as before): f_e(x) ∈ ℝ^64 → classifier → logits
2. **Routing embedding**: r_e(x) ∈ ℝ^d (d = 16-32) via a small MLP projection head

The routing embeddings are trained with a **contrastive loss**:
- **Positive pairs:** (r_e(x), r_e(x')) where expert e is correct on both x and x'
- **Negative pairs:** (r_e(x), r_e(x')) where expert e is correct on one and wrong on the other

This creates a routing space where distance = "difference in correctness." At test time, the router compares r_1(x), r_2(x), r_3(x) to an "expert is correct" anchor — the closest embedding indicates the most reliable expert.

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ✅ **Directly.** The routing embedding is explicitly trained to encode correctness information. The contrastive loss forces the backbone to produce features that separate "will be correct" from "will be wrong." |
| **L2 (37.1% All-Wrong Ceiling)** | ❌ **Does not directly help.** But the router can detect "all embeddings far from correct anchor" → abstain. |
| **L3 (Lone Dissenter Paradox)** | ✅ **Directly.** The routing embedding encodes correctness, not confidence. The correct but uncertain expert's embedding will be close to the "correct" anchor, breaking the paradox. |

### Expected Impact

**High.** The contrastive objective is powerful for creating separable representations. If the routing space achieves even 80% accuracy in separating "correct" from "wrong," the router has a clean signal. Expected BA: **53-55%** (1-3% above current best).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | Contrastive learning (SimCLR, MoCo) is well-established. Using contrastive learning to create a *routing space* for expert selection is novel. The Multiple Contrastive Experts paper (2024) uses contrastive learning for expert diversity but not for creating routing embeddings. |
| **Differentiation** | Highly novel because: (1) the routing embedding is a *separate representation space* explicitly optimized for routing, not classification, (2) the contrastive objective directly targets the routing problem, (3) the router operates in a space designed for its task, not repurposed from classification. |
| **Publication risk** | Low. This is a genuinely new approach to the expert routing problem. |

### Strengths

1. **Routing space is purpose-built** — not a hack on classification features
2. **Contrastive learning is well-understood** — easy to implement, stable training
3. **Decoupled from classification** — the routing embedding doesn't interfere with classification accuracy
4. **Flexible dimensionality** — can use d=16 for efficiency or d=64 for capacity
5. **Works with any expert architecture** — just add a projection head

### Weaknesses

1. **Two training phases** — need to train experts first, then routing embeddings (or jointly)
2. **Contrastive loss needs careful tuning** — temperature, queue size, positive/negative definition
3. **What is a "correctness anchor"?** — need to define what the routing space should look like
4. **May not transfer across experts** — each expert's routing embedding is in its own space; need a common comparison mechanism

### Implementation Considerations

**Critical design decisions:**

1. **Projection head architecture:** 2-layer MLP: 64 → 64 (ReLU) → d (no activation). Standard in contrastive learning.

2. **Routing embedding dimension:** d = 16 or 32. Small enough to be efficient, large enough to encode routing signal.

3. **Contrastive objective:**
   - **SimCLR-style:** Within a batch, pull together embeddings of samples where the expert is correct, push apart embeddings where correctness differs.
   - **MoCo-style:** Maintain a queue of past embeddings with correctness labels. Use as negatives.
   
   **Recommendation:** Start with SimCLR-style (batch-based, no queue needed).

4. **Anchor-based routing at test time:**
   - Compute a "correctness anchor" a_e = mean of r_e(x) for validation samples where expert e is correct
   - For a test sample, compute distance d_e = ||r_e(x) - a_e||₂
   - Route to expert with smallest d_e (closest to its correctness anchor)
   
   **Alternative:** Use a lightweight MLP on the concatenated routing embeddings [r_1, r_2, r_3] to predict the best expert.

5. **Training schedule:**
   - **Option 1 (two-stage):** Train experts normally, freeze, then train routing embeddings
   - **Option 2 (joint):** Train experts + routing embeddings simultaneously with L_cls + λ × L_contrastive
   
   **Recommendation:** Start with two-stage (simpler, safer). If it works, try joint training for potentially better results.

**Pseudo-code:**
```python
class ResNet32WithRouting(nn.Module):
    def __init__(self, num_classes=100, routing_dim=16):
        super().__init__()
        self.backbone = ResNet32Backbone()
        self.fc = nn.Linear(64, num_classes)
        # Routing projection head
        self.routing_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, routing_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        routing_emb = self.routing_proj(features)  # (B, d)
        return logits, routing_emb

# Training routing embeddings (two-stage)
for batch in dataloader:
    logits, routing_emb = model(images)
    correct = (logits.argmax(1) == labels).float()  # (B,)
    
    # Contrastive loss: pull same-correctness together
    # For each sample, positive = same correctness, negative = different correctness
    loss = contrastive_loss(routing_emb, correct, temperature=0.1)

# Test-time routing
anchors = {}  # expert_name -> (d,) mean routing emb when correct
for expert_name, model in experts.items():
    correct_embs = []
    for x, y in val_dataloader:
        with torch.no_grad():
            logits, routing_emb = model(x)
            is_correct = (logits.argmax(1) == y)
            correct_embs.append(routing_emb[is_correct])
    anchors[expert_name] = torch.cat(correct_embs).mean(0)

def route_sample(x):
    distances = []
    for expert_name, model in experts.items():
        with torch.no_grad():
            _, routing_emb = model(x.unsqueeze(0))
        dist = F.pairwise_distance(routing_emb, anchors[expert_name].unsqueeze(0))
        distances.append(dist.item())
    return experts[torch.tensor(distances).argmin()]
```

### Will It Work?

**Likely yes.** The contrastive objective is proven to create meaningful embedding spaces. The key question is whether the routing embedding can separate "correct" from "wrong" given the backbone's limited features. With the contrastive loss driving the projection head, this should work well.

The main risk is that the routing embedding might collapse (all samples map to the same region). This is prevented by proper contrastive loss tuning (temperature, negatives) and regularization.

**Verdict: Highly promising. Strong novelty. Priority approach.**

---

## 8. Idea F: Confidence-Adaptive & Value-of-Information Routing

### Concept

Instead of routing based on expert confidence, route based on **value of information (VoI)** — how much does each expert's prediction reduce uncertainty about the true label?

For each expert e:
1. Compute the expert's predictive distribution p_e(y|x)
2. Compute the **expected information gain** from consulting this expert:
   - Prior: uniform over classes (or class frequencies)
   - Posterior: p_e(y|x)
   - VoI = KL(posterior || prior) = H(prior) - H(posterior)

The expert with the highest VoI is selected. VoI is high when the expert makes a confident, informative prediction — but this is fundamentally different from raw confidence because it accounts for the *prior*.

**Key innovation:** If the prior is the class frequency distribution (long-tail prior), a head-class prediction from LAL has low VoI (it's expected), while a tail-class prediction has high VoI (it's surprising). This naturally biases routing toward experts that make rare/valuable predictions.

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ❌ **Does not address.** Still operates on frozen outputs. |
| **L2 (37.1% All-Wrong Ceiling)** | ❌ **Does not help.** |
| **L3 (Lone Dissenter Paradox)** | ⚠️ **Partially.** VoI accounts for prior knowledge. If the "lone dissenter" makes a rare (tail-class) prediction, its VoI is high even if confidence is low. But this only helps for tail classes, not head classes. |

### Expected Impact

**Low-Medium.** The project already tested entropy routing (similar to VoI with uniform prior). VoI with class-frequency prior adds nuance but the fundamental Lone Dissenter Paradox still limits it. Expected BA: **51.5-52.5%** (0-1% above uniform).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | VoI routing is used in active learning and Bayesian optimization. "Uncertainty Is Not Enough: Value-of-Information Routing for Mixtures of LoRA Experts" (2025) applies VoI to expert routing. |
| **Differentiation** | Low. The paper above already does this for LLM routing. Applying to vision long-tail is incremental. |
| **Publication risk** | High. Too similar to existing work. |

### Strengths

1. **Principled information-theoretic foundation**
2. **Accounts for class prior** — naturally biases toward tail-class predictions
3. **No training required** — can be computed from expert outputs directly
4. **Works with frozen experts**

### Weaknesses

1. **Still operates on frozen outputs** — doesn't address the feature learning gap
2. **Similar to entropy routing** — the project already tested entropy routing (50.10% BA, worse than uniform)
3. **Class-frequency prior helps tail but not head** — doesn't fix the core routing problem
4. **Computationally similar to confidence routing** — same limitations apply

### Will It Work?

**Unlikely as a primary method.** The project already tested entropy routing (50.10%), confidence routing (50.64%), and calibrated confidence (51.44%). VoI with class-frequency prior would be a minor improvement over these but unlikely to reach 52%+.

**Verdict: Not recommended as a standalone approach. Could be a useful feature in the 89-d/92-d framework.**

---

## 9. Idea G: Joint Gating-Expert Training with Specialization

### Concept

This is the most architecturally ambitious idea: train the router *jointly* with the experts from scratch, where the router's gating decisions influence which samples each expert specializes in.

**Architecture:**
- 3 separate backbones (or a shared backbone with 3 heads)
- A lightweight router (MLP: features → 3-way softmax)
- Training loop:
  1. Forward: router assigns each sample to an expert (hard or soft gating)
  2. Each expert trains only on samples it's assigned to
  3. Router is updated based on which expert was correct

**Key innovation over RIDE:** 
- RIDE uses a shared backbone + distribution-aware sampling. The router is trained separately after experts.
- This approach uses *separate backbones* with *dynamic sample assignment* during training. The router's assignments determine which data each expert sees, creating natural specialization.

### How It Addresses the Limitations

| Limitation | How It Helps |
|:-----------|:-------------|
| **L1 (Feature Learning Gap)** | ✅ **Directly.** The router's assignments determine expert training data. Experts learn features that are useful for the samples they're assigned to. The router learns to assign based on features that predict expert competence. Feature learning gap is eliminated by joint training. |
| **L2 (37.1% All-Wrong Ceiling)** | ✅ **Directly.** If the router assigns hard samples to multiple experts, at least one might learn to handle them. The all-wrong set should shrink. |
| **L3 (Lone Dissenter Paradox)** | ✅ **Directly.** The router learns from assignment outcomes, not from confidence. It learns "expert A is better for this type of sample" without being misled by confidence. |

### Expected Impact

**Very High.** Joint training is the most direct solution to the feature learning gap. Expected BA: **54-57%** (3-6% above current best).

### Novelty Assessment

| Aspect | Assessment |
|:-------|:-----------|
| **Prior art** | RIDE (ICLR 2021) does joint training with a shared backbone. "Learning to Specialize" (NeurIPS 2025) does joint gating-expert training for MoEs. |
| **Differentiation** | Moderately novel. Using joint training with *separate backbones* and *dynamic sample assignment* (not distribution-aware sampling) is different from RIDE. The router decides per-sample which expert trains on it, creating emergent specialization. |
| **Publication risk** | Medium-High. RIDE and "Learning to Specialize" cover similar ground. Need a clear differentiator. |

### Strengths

1. **Most direct solution** — joint training eliminates the feature learning gap by design
2. **Emergent specialization** — experts naturally specialize based on router assignments
3. **Router learns genuine routing patterns** — not from frozen features but from the dynamics of training
4. **High ceiling** — potential for significant improvement

### Weaknesses

1. **Highest implementation cost** — requires rewriting the entire training pipeline
2. **Training instability** — joint training of router + experts is notoriously unstable (mode collapse, router ignoring some experts)
3. **GPU cost** — 6-8h minimum, possibly more with stability issues
4. **Complexity** — load balancing, expert utilization, gradient scaling all need careful tuning
5. **Risk of router collapse** — router may learn to always assign to the best expert, starving others

### Implementation Considerations

**Critical design decisions:**

1. **Router architecture:** Lightweight MLP (192→128→3) on concatenated backbone features. Same as tested in the project's MLP router experiments, but now trained jointly.

2. **Hard vs. soft gating:**
   - **Hard gating:** Each sample assigned to one expert. Clean specialization but discrete → REINFORCE or straight-through gradients.
   - **Soft gating:** Each sample assigned to all experts with weights. Stable but less specialized.
   
   **Recommendation:** Start with soft gating (weighted loss per expert). Simpler, more stable.

3. **Load balancing loss:** Add an auxiliary loss to ensure each expert receives ~33% of samples. Standard MoE technique.

4. **Router update frequency:** Update router every N steps (not every step) to let experts stabilize before router adapts.

5. **Warm-starting:** Initialize experts from pre-trained checkpoints (already have LAL, PaCo, Mixup) rather than training from scratch. This reduces training time and instability.

6. **Separate backbones vs. shared backbone:**
   - Separate: more parameters, but already have the codebase
   - Shared: more efficient, closer to RIDE
   
   **Recommendation:** Use separate backbones (leverage existing code). If stability issues arise, switch to shared backbone + expert-specific BN.

**Pseudo-code:**
```python
# Simplified joint training loop
for batch in dataloader:
    images, labels = batch
    
    # Forward through all experts
    logits_list = []
    features_list = []
    for expert in experts:
        logits, features = expert(images)
        logits_list.append(logits)
        features_list.append(features)
    
    # Router: predict expert weights from concatenated features
    cat_features = torch.cat(features_list, dim=1)  # (B, 192)
    routing_weights = router(cat_features)  # (B, 3) softmax
    
    # Compute weighted classification loss
    total_loss = 0
    for e in range(3):
        expert_loss = ce_loss(logits_list[e], labels)
        total_loss += (routing_weights[:, e] * expert_loss).mean()
    
    # Load balancing loss (encourage uniform assignment)
    mean_weights = routing_weights.mean(0)  # (3,)
    balance_loss = 3 * (mean_weights * torch.log(mean_weights + 1e-8)).sum()
    # balance_loss is minimized when mean_weights = [1/3, 1/3, 1/3]
    
    total_loss += 0.01 * balance_loss
    
    # Backward
    total_loss.backward()
    optimizer.step()
```

### Will It Work?

**Very likely, but with significant engineering effort.** The joint training approach is well-established in the MoE literature. The main risk is training instability, which can be mitigated with:
- Careful learning rate scheduling (warmup, gradient clipping)
- Load balancing loss
- Expert dropout (randomly drop experts during training to prevent collapse)
- Warm-starting from pre-trained checkpoints

**Verdict: Highest potential but highest cost. Best suited as a long-term goal after simpler approaches are exhausted.**

---

## 10. Hybrid Combinations

The ideas above are not mutually exclusive. Here are the most promising hybrids:

### Hybrid 1: RAAL + Boosting (Ideas B + A)

**Concept:** Train experts sequentially (boosting-style) where each expert has an auxiliary correctness-prediction head (RAAL). Expert B is trained with both:
1. Upweighted samples where Expert A is wrong (boosting)
2. An auxiliary head that predicts correctness (RAAL)

**Why it's stronger:** The boosting creates specialization, and RAAL ensures the features encode correctness information. The two effects are complementary.

**Expected impact:** 53.5-55.5% BA.

### Hybrid 2: Expertise Maps + Product (Ideas C + existing product)

**Concept:** Replace the static product combination with competence-weighted product. The router computes:
```
score_k = max_e [ p_e(k|x) × competence_e_k(x) ]
```

**Why it's stronger:** The static product gave +0.82% over uniform averaging. The competence-weighted product is a learned, dynamic version that should give more.

**Expected impact:** 53-54.5% BA.

### Hybrid 3: Contrastive Routing Embeddings + RAAL (Ideas E + B)

**Concept:** Each expert has both a correctness-prediction auxiliary head (RAAL) and a contrastive routing embedding (E). The router uses both signals.

**Why it's stronger:** RAAL provides a direct correctness score; contrastive embeddings provide a learned routing space. Two independent routing signals = more robust routing.

**Expected impact:** 53.5-55% BA.

### Hybrid 4: Boosting + Contrastive Routing (Ideas A + E)

**Concept:** Sequential training (boosting) creates specialized experts. Each expert has a contrastive routing embedding trained to separate "correct" from "wrong" in its specialized domain.

**Why it's stronger:** Boosting creates the specialization; contrastive embeddings make the routing signal explicit. The router can easily distinguish which expert is best for each sample.

**Expected impact:** 54-56% BA. This is the most promising hybrid.

### Hybrid 5: Joint Training + All Auxiliary Signals (Idea G + B + C + E)

**Concept:** Train experts jointly with the router (G), while each expert also produces auxiliary outputs: correctness score (B), competence vector (C), and routing embedding (E).

**Why it's stronger:** Every possible routing signal is available, and the joint training ensures features encode routing-relevant information. This is the "maximal" approach.

**Expected impact:** 55-58% BA. But also the highest implementation cost and risk.

---

## 11. Head-to-Head Comparison

| Idea | Novelty | Expected BA | L1 | L2 | L3 | Impl. Cost | GPU Cost | Risk | Overall |
|:----|:-------:|:-----------:|:--:|:--:|:--:|:----------:|:--------:|:----:|:-------:|
| **A: Boosting** | ★★★ | 53-55% | ✅ | ✅ | ⚠️ | Low | 3-5h | Medium | ⭐⭐⭐ |
| **B: RAAL** | ★★★ | 52.5-54% | ✅ | ❌ | ✅ | Low | 1.5-3h | Low | ⭐⭐⭐ |
| **C: Expertise Maps** | ★★★ | 53-54.5% | ✅ | ❌ | ✅ | Medium | 2-4h | Medium | ⭐⭐ |
| **D: Geometry** | ★★ | 52-53% | ⚠️ | ❌ | ✅ | Low | 0h | Low | ⭐ |
| **E: Contrastive Embeddings** | ★★★ | 53-55% | ✅ | ❌ | ✅ | Medium | 2-4h | Medium | ⭐⭐⭐ |
| **F: VoI Routing** | ★ | 51.5-52.5% | ❌ | ❌ | ⚠️ | Low | 0h | Low | ⭐ |
| **G: Joint Training** | ★★ | 54-57% | ✅ | ✅ | ✅ | High | 6-10h | High | ⭐⭐ |
| **H1: Boosting + RAAL** | ★★★ | 53.5-55.5% | ✅ | ✅ | ✅ | Low-Med | 4-7h | Medium | ⭐⭐⭐ |
| **H4: Boosting + Contrastive** | ★★★ | 54-56% | ✅ | ✅ | ✅ | Med | 4-7h | Medium | ⭐⭐⭐ |

---

## 12. Recommended Next Steps

### Phase 1 (Quick Wins — 1-2 weeks)

1. **Implement Idea B (RAAL)** — Lowest risk, easiest to implement. Add auxiliary correctness-prediction head to all three experts. Re-train on existing infrastructure. Evaluate routing improvement.

2. **Add Idea D (Geometry) features to 92-d framework** — Compute separation ratios and add as 3 extra features. See if it provides the +0.1-0.3% needed to beat optimal fixed weights more convincingly.

### Phase 2 (High Impact — 2-4 weeks)

3. **Implement Idea A (Boosting)** — Train expert B (LAL with upweighted CE errors) and expert C (PaCo with upweighted LAL+CE errors). This is the most promising low-cost approach.

4. **Implement Idea E (Contrastive Routing Embeddings)** — Add projection head to all experts. Train routing embeddings via contrastive loss. Evaluate anchor-based routing.

### Phase 3 (Maximum Impact — 4-8 weeks)

5. **Hybrid H4 (Boosting + Contrastive Routing)** — Combine the two most promising approaches. Sequential boosting creates specialization; contrastive embeddings make routing signal explicit.

6. **Implement Idea G (Joint Training)** — Only if Phase 1-2 approaches plateau below the target. Requires significant engineering but has the highest ceiling.

---

## Appendix: Prior Art Check Summary

| Paper | Year | Relevance to Novel Ideas |
|:------|:----:|:-------------------------|
| **RIDE** (Wang et al.) | ICLR 2021 | Shared backbone + routing. Different from all ideas (separate backbones, no joint training in A-E). |
| **SADE** (Zhou et al.) | NeurIPS 2022 | Different losses + self-supervised aggregation. Different from A-E (no auxiliary training objectives). |
| **BPCE** (Aimar et al.) | CVPR 2023 | Balanced subsets + product combination. Related to Idea C (per-class weighting) but static, not learned. |
| **MDCS** (Zhao et al.) | ICCV 2023 | Consistency self-distillation for diversity. Related to Idea A (diversity) but different mechanism. |
| **DNCC** (Zhang et al.) | 2024 | Negative correlation for ensemble diversity. Related to A (explicit diversity) but no sequential training. |
| **Multiple Contrastive Experts** (2024) | ESA | Contrastive learning for expert diversity. Different from E (routing embeddings vs. diverse features). |
| **Divide, Weight, Route** (Wei et al.) | 2025 | Difficulty-aware routing. Related to A (difficulty specialization) but static, not sequential. |
| **Learn to Specialize** (Farhat et al.) | NeurIPS 2025 | Joint gating-expert training for MoEs. Related to G but in decentralized setting. |
| **VoI Routing** (2025) | arXiv | Value-of-information for LoRA routing. Idea F is similar but applied to vision. |
| **Confidence-Adaptive Routing** (2025) | arXiv | Confidence-based routing for MoE LoRA. Related to F but different domain. |

---

*Generated: Analysis of 7 novel routing ideas for the Expert Method project (CIFAR-100-LT). Each idea is evaluated for novelty, feasibility, likely impact, and implementation path.*
