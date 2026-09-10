# Future Directions — Boosting & Novel Routing Ideas

> Forward-looking plans for when the current frozen-expert routing ceiling is reached. Two major approaches are documented: (1) Boosting-style adversarial expert training (detailed implementation plan), and (2) Alternative novel routing ideas ranked by feasibility and expected impact. The boosting approach is the primary recommended path.

---

## 1. Boosting-Style Adversarial Expert Training

### 1.1 Concept

Train experts **sequentially** where each expert's loss is weighted by previous experts' errors. This creates features that naturally encode routing-relevant information — Expert B's features organize around "samples Expert A gets wrong," which is exactly what a router needs.

**Core idea:** Standard independent training produces features organized by class identity (because that's what classification loss optimizes for). Boosting forces features to encode error patterns.

### 1.2 Key Finding from Phase 0

All three experts achieve **near-100% training accuracy** (LAL: 99.72%, Mixup: 97.84%, PaCo: 95.16%). Training-set errors cannot be used as a hardness signal — there are almost none.

**Revised strategy:** Use **confidence as a continuous hardness signal**. Even correct predictions vary in confidence, and low-confidence-correct samples are near decision boundaries.

| Expert | % Training Samples with Confidence < 0.9 |
|--------|:----------------------------------------:|
| LAL | 6% (overconfident, memorized) |
| PaCo | 38.5% (better calibrated) |
| **Mixup** | **64%** (soft labels prevent overconfidence) |

Mixup's confidence is the richest hardness signal. **Cross-paradigm hardness** — using one expert's confidence to weight another expert's training — creates more diversity than same-paradigm weighting.

### 1.3 Training Pipeline

```
Expert A (LAL):     trained on ALL data (standard) — already done
                    ↓
Expert B (LAL):     trained with LOSS × (1 + α · (1 - Mixup_confidence))
                    → features encode "samples Mixup finds hard"
                    → cross-paradigm diversity
                    ↓
Expert C (PaCo):    trained with LOSS × (1 + α · [A_val_error] + β · [B_val_error])
                    → weights based on VALIDATION errors of A and B
                    → targets genuine generalization failures
```

### 1.4 Why Cross-Paradigm Weighting for Expert B?

- **LAL has only 27 training errors** (99.72% train acc) — too few to learn from
- **Mixup has 64% low-confidence training samples** — rich, continuous hardness signal
- **Cross-paradigm signal:** Mixup (interpolation-based) finds different samples hard than LAL (logit-adjusted). This creates fundamentally different feature geometry.

### 1.5 Why Validation Errors for Expert C?

- All experts memorize the training set — training errors are near zero
- Validation errors (~56% of 5K for LAL) are **genuine generalization failures**
- Weighting by validation errors forces Expert C to focus on samples that cause real-world mistakes

### 1.6 Expert Configuration After Boosting

| Expert | Loss | Weighting Signal | Est. BA | Role |
|:-------|:----:|:----------------:|:-------:|:-----|
| **A (LAL)** | LAL (τ=1.0) | None (standard) | ~40% | Generalist, good on head classes |
| **B (LAL)** | LAL (τ=1.0) | 1 + α · (1 - Mixup_confidence) | 40-44% | Specialist on samples Mixup finds hard |
| **C (PaCo)** | PaCo (α=0.01, t=0.05) | 1 + α·[A_val_error] + β·[B_val_error] | 43-47% | Specialist on hardest generalization cases |

### 1.7 Implementation

**Key files already in place:**
- `data/weighted_dataset.py` — `WeightedDataset` wrapper for per-sample loss weighting
- `scripts/train_lal_weighted.py` — LAL with per-sample loss weighting
- `scripts/train_paco_weighted.py` — PaCo with per-sample loss weighting

**Weighting formulas:**

**Expert B (confidence-weighted LAL):**
```
weight_B[i] = 1.0 + α × (1 - Mixup_confidence[i])
```
- α = 2.0 (default), sweep [1.0, 3.0, 5.0]
- Samples Mixup is very confident about (conf ≈ 1.0): weight ≈ 1.0 (standard)
- Samples Mixup is uncertain about (conf ≈ 0.5): weight ≈ 1.0 + 0.5α

**Expert C (validation-error-weighted PaCo):**
```
weight_C[i] = 1.0 + α × [A_val_error(i)] + β × [B_val_error(i)]
```
- Applied to the 5K balanced validation set (not training set)
- α = 1.0, β = 1.0 (default), sweep [0.5, 2.0]
- Samples where BOTH A and B fail get highest weight → directly attacks all-wrong ceiling

**Training approach:**
- Expert B: Train from scratch (random init) with weighted LAL loss, 200 epochs
- Expert C: Two-dataset loader — 80% from LT training set (unweighted) + 20% from validation set (weighted), 400 epochs

### 1.8 Expected Outcomes

| Metric | Current Best | Target with Boosting |
|:-------|:------------:|:--------------------:|
| Uniform average BA | 45.68% | **48-50%** |
| Best routing BA | 45.68% (ties uniform) | **50-52%** |
| Oracle (≥1 expert correct) | 55.79% | **60-65%** |
| All-wrong ceiling | 44.2% | **35-40%** |

### 1.9 Risk Mitigation

| Risk | Symptom | Mitigation |
|:-----|:--------|:-----------|
| Expert B overfits to Mixup's confidence signal | BA drops below 40% | Reduce α; add unweighted loss regularization |
| Expert C overfits to validation set | High val BA but poor test BA | Reduce val batch fraction to 10%; weight only supervised loss component |
| Error weights don't create routing signal | κ between A and B > 0.45 | Try stronger α; add auxiliary correctness-prediction head (RAAL) |
| Experts too similar to A | κ > 0.45 | Use different loss or stronger augmentations for B |

---

## 2. Alternative Novel Ideas

### 2.1 RAAL (Routing-Aware Auxiliary Loss)

**Concept:** During expert training, add an auxiliary head that predicts "will I be correct on this sample?" The auxiliary loss gradient flows back into the backbone, forcing features to encode correctness-relevant information.

```
Input x → Backbone → Features f(x)
                    ├──→ Classifier Head → logits → classification loss
                    └──→ Auxiliary Head → p(correct | x) → auxiliary loss
```

- Auxiliary head: single linear layer (64→1) with sigmoid
- Loss: `L_total = L_cls + λ × L_aux` (λ = 0.1, sweep [0.01, 0.05, 0.1, 0.5, 1.0])
- **Expected impact:** 52.5-54% BA. Breaks the Lone Dissenter Paradox by providing a correctness score decoupled from confidence.
- **Implementation cost:** Low. Add one linear layer + BCE loss to any training script.
- **Risk:** Low. Auxiliary loss may compete with classification loss, but λ tuning mitigates this.

### 2.2 Contrastive Routing Embeddings

**Concept:** Each expert produces two outputs: (1) classification features as before, (2) a routing embedding r_e(x) ∈ ℝ^d (d=16-32) via a small MLP projection head. The routing embeddings are trained with a contrastive loss where positive pairs share the same correctness status.

- Projection head: 2-layer MLP (64→64→d)
- Contrastive objective: SimCLR-style within a batch
- At test time: compare routing embeddings to a "correctness anchor" — closest embedding = most reliable expert
- **Expected impact:** 53-55% BA. Purpose-built routing space.
- **Implementation cost:** Medium. Requires two-stage training (experts first, then routing embeddings).
- **Risk:** Medium. Contrastive loss needs careful tuning (temperature, positives/negatives).

### 2.3 Expertise Maps (Per-Class Competence Outputs)

**Concept:** Each expert outputs a per-class competence vector c(x) ∈ [0,1]^K (K=100) where c_k(x) = "how reliable is this expert for class k on this specific sample?" Router uses per-class max-product combination.

- Competence head: linear layer (64→100) with sigmoid
- Training: freeze backbone, train competence head on validation set with multi-label BCE
- Router: `score_k = max_e [ softmax(logits_e)_k × competence_e_k(x) ]`
- **Expected impact:** 53-54.5% BA. Per-class granularity handles the 69.4% label ambiguity.
- **Implementation cost:** Medium. Non-trivial loss function; needs validation set for training.
- **Risk:** Medium. Per-class training signal is sparse for tail classes.

### 2.4 Joint Gating-Expert Training with Specialization

**Concept:** Train the router **jointly** with the experts from scratch, where the router's gating decisions influence which samples each expert specializes in. Most architecturally ambitious approach.

- 3 separate backbones + lightweight MLP router (192→128→3)
- Training loop: router assigns each sample to an expert; each expert trains only on its assigned samples
- Load balancing loss to prevent router collapse
- **Expected impact:** 54-57% BA. Most direct solution to the feature learning gap.
- **Implementation cost:** High. Requires rewriting the entire training pipeline.
- **Risk:** High. Joint training is notoriously unstable (mode collapse, router ignoring some experts).
- **GPU cost:** 6-10h minimum.

### 2.5 Feature-Space Geometry Routing

**Concept:** Route based on where the sample falls in each expert's feature space relative to class prototypes. Compute separation ratio = d_nearest_wrong / d_correct.

- No training needed — prototypes computed from training data
- **Expected impact:** 52-53% BA. Weak standalone; useful as a complementary feature in the 89-d/92-d framework.
- **Risk:** Low. Features still encode class identity; geometry is a different lens on the same limited information.

### 2.6 Value-of-Information (VoI) Routing

**Concept:** Route based on expected information gain from consulting each expert: VoI = KL(posterior || prior). Expert with highest VoI is selected.

- No training required; accounts for class prior
- **Expected impact:** 51.5-52.5% BA. Similar to entropy routing (already tested at 50.10%).
- **Risk:** Low. Not recommended as standalone approach.

---

## 3. Hybrid Combinations

| Hybrid | Ideas Combined | Expected BA | Cost |
|:-------|:--------------:|:-----------:|:----:|
| **H1: Boosting + RAAL** | A + B | 53.5-55.5% | 4-7h GPU |
| **H2: Expertise Maps + Product** | C + product | 53-54.5% | 2-4h GPU |
| **H3: Contrastive Embeddings + RAAL** | E + B | 53.5-55% | 4-7h GPU |
| **H4: Boosting + Contrastive Embeddings** | A + E | 54-56% | 4-7h GPU |

---

## 4. Priority Ranking

| # | Approach | Novelty | Est. BA | Risk | Cost | Recommended |
|:-:|:---------|:-------:|:-------:|:----:|:----:|:-----------:|
| 1 | **Boosting** (primary) | ★★★ | 48-52% | Medium | 3-5h GPU | ⭐ First |
| 2 | **RAAL** (quick win) | ★★★ | 47-50% | Low | 1.5-3h GPU | ⭐ Second |
| 3 | **Contrastive Routing Embeddings** | ★★★ | 49-52% | Medium | 2-4h GPU | ⭐ Third |
| 4 | Expertise Maps | ★★★ | 48-51% | Medium | 2-4h GPU | ⭐ Fourth |
| 5 | Joint Training | ★★ | 50-54% | High | 6-10h GPU | ⚠️ Long-term |
| 6 | Geometry Routing | ★★ | 46-48% | Low | 0h GPU | ❌ Complementary only |
| 7 | VoI Routing | ★ | 45-47% | Low | 0h GPU | ❌ Not recommended |
