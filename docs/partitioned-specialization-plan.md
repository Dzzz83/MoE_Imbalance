# DACE: Disagreement-Aware Cascade Experts

> A novel architecture for expert routing on long-tail data. Trains experts **sequentially** so that each expert's features encode "how does my prediction differ from the previous expert?" — a signal computable entirely from training data. No validation leakage.

---

## 1. The Five Problems This Architecture Must Solve

| # | Problem | Evidence | Why Post-Hoc Can't Fix It |
|:-:|:--------|:---------|:--------------------------|
| **P1** | **Feature learning gap (19.22%)** | Oracle=55.82%, learned gate=36.60% | Features encode class identity — no training signal for routing relevance |
| **P2** | **All-wrong ceiling (44.2%)** | All 3 experts wrong on 4,418/10,000 test samples | Independent training on same LT data → same failure modes |
| **P3** | **Lone dissenter paradox** | Correct expert is most confident only 14.4%, least confident 42.2% | Confidence and correctness are inversely correlated in savable samples |
| **P4** | **Label ambiguity (68.7%)** | 68.7% of trainable samples have >1 correct expert | 3-way selection ill-posed when multiple experts are correct |
| **P5** | **Training set memorized (95-99%)** | No training errors to use as signal | Deep networks memorize small LT datasets |

### The "No Cheating" Constraint

The validation set (2,170 samples) may NOT be used during training. All training signals must come from the training set (8,677 samples). The validation set is only used for:
- Final evaluation checkpoint selection (standard early stopping)
- Post-hoc prototype computation for inference (after training is complete)

---

## 2. Key Insight: Cross-Expert Disagreement Is Computable from Training Data

Even though the training set is memorized (95-99% correct), two experts can still **disagree on their prediction distributions** on the same training sample:

```
Training sample: image of class 42 (e.g., "beaver")

Expert A's distribution:  P(class 42) = 0.85,  P(class 7) = 0.05, ...
Expert B's distribution:  P(class 42) = 0.60,  P(class 15) = 0.20, ...

Both experts are correct (top-1 = class 42),
but their probability distributions are very different → DISAGREEMENT
```

**KL divergence** between two experts' probability distributions measures how much they disagree. This is:

- **Computable from training data** — no correctness labels needed
- **Available during training** — compute it on any training batch
- **Routing-relevant** — samples where experts disagree are exactly where routing matters

---

## 3. DACE Architecture: Disagreement-Aware Cascade Experts

### 3.1 Core Concept

Train experts **sequentially** (cascade). Each expert has a **routing head** trained with a **contrastive loss** where the positive/negative signal is: "does my probability distribution agree with the previous expert's distribution?"

```
Expert A (trained first):
  Standard ResNet-32 + classifier → class logits
  No routing head (or routing head unused)

Expert B (trained second, Expert A frozen):
  ResNet-32 + classifier → class logits
              + routing head → 32-d routing embedding
  Contrastive signal: "does B's distribution match A's?"
  
Expert C (trained third, A and B frozen):
  ResNet-32 + classifier → class logits
              + routing head → 32-d routing embedding
  Contrastive signal: "does C's distribution match average of A and B?"
```

### 3.2 Why Cascade Training Instead of Parallel?

| Aspect | Parallel (Old Approach) | Cascade (DACE) |
|:-------|:-----------------------|:---------------|
| Training order | All 3 at once | Sequential: A → B → C |
| Routing signal | None in training (post-hoc) | Built into training via contrastive loss |
| Features encode | Class identity only | Class identity + cross-expert disagreement pattern |
| P1 addressed? | ❌ No | ✅ Yes — gradient from routing loss flows into backbone |

The cascade is the key novelty. Expert B's backbone learns to organize features by "do I agree or disagree with Expert A on this sample?" — which is exactly the information a router needs.

### 3.3 Data Partitioning (Addresses P2)

To reduce the all-wrong ceiling, each expert specializes in a class group:

| Expert | Primary Classes | Loss | Cascade Comparison | KL Signal Rate |
|:-------|:----------------|:-----|:-------------------|:--------------:|
| **A** | Head (30 classes, ≥100 samples) | LAL (τ=1.0) | — (first expert) | — |
| **B** | Med (36 classes, 20-100 samples) | **CE + Mixup (α=1.0)** | KL(A \|\| B) | **73.8%** of training samples |
| **C** | Tail (34 classes, <20 samples) | **PaCo-style contrastive** | KL(avg(A,B) \|\| C) | **72.2%** of training samples |

The 20% "all classes" sampling ensures each expert maintains general feature quality. This is equivalent to class-balanced sampling — a standard technique, not cheating.

**Why this reduces P2:** Before partitioning, every expert trained on the same 8,677 samples. For a tail class with 5 training samples, ALL experts learned from those same 5 samples → same failure mode. After partitioning, Expert C focuses on those 5 samples (training primarily on tail classes), while Expert A sees them only 20% of the time. Expert C becomes a **tail specialist**.

---

## 4. Detailed Architecture

### 4.1 Expert Architecture

```
                   ┌──────────────────────┐
                   │    ResNet-32 Backbone │
                   │      (64-d features) │
                   └──────┬───────────────┘
                          │
                  ┌───────┴────────┐
                  │                │
          ┌───────▼──────┐  ┌─────▼──────┐
          │ Classifier   │  │ Routing    │
          │ Head         │  │ Head (MLP) │
          │ 64 → 100     │  │ 64 → 64    │
          │              │  │   → 32     │
          │ → class      │  │ → routing  │
          │   logits     │  │   embedding│
          └──────────────┘  └────────────┘
```

- **Classifier head:** `nn.Linear(64, 100)` — standard, trained with LAL or CE
- **Routing head:** 2-layer MLP `Linear(64, 64) → ReLU → Linear(64, 32)` — produces 32-d embedding
- Both heads share the same backbone → routing head gradient flows into backbone features

### 4.2 Contrastive Disagreement Loss

For Expert B (trained after Expert A), each batch:

1. Run Expert A (frozen) → logits_a, probs_a = softmax(logits_a)
2. Run Expert B → logits_b, probs_b, embeddings_b
3. Compute per-sample KL divergence: `KL(probs_a || probs_b)` — how much does B disagree with A?
4. Binarize: if `KL > τ` → label = 1 (disagree); if `KL < τ` → label = 0 (agree)
5. Train routing head with supervised contrastive loss using (embedding_b, label)

```
L_total = L_cls + λ · L_contrastive

L_contrastive pulls together embeddings with SAME agreement label,
pushes apart embeddings with DIFFERENT agreement labels.
```

**Why KL divergence works on training data:** On the 95-99% of samples where both experts are correct, they may still assign very different probability distributions (e.g., one is confident at p=0.95, another uncertain at p=0.60). The KL divergence captures this difference. It's a **continuous disagreement signal** that doesn't require correctness labels.

**Temperature τ:** τ = 0.5 (sweep: 0.1, 0.3, 0.5, 1.0). A lower τ means only strong disagreements are labeled "disagree."

**Contrastive loss formulation:**
```
L_contrastive = -log( Σ_{j: label_j = label_i} exp(sim(e_i, e_j)/τ_c)
                    / Σ_{k ≠ i} exp(sim(e_i, e_k)/τ_c) )
```

Where `sim(e_i, e_j) = cosine_similarity(embedding_i, embedding_j)` and `τ_c = 0.5`.

### 4.3 Cascade Training Procedure

```
Step 1: Train Expert A (Head specialist)
─────────────────────────────────────────
  Data: 80% Head classes + 20% all classes (training set only)
  Loss: LAL (τ=1.0)
  No routing head needed
  → Expert A learns Head class boundaries

Step 2: Train Expert B (Med specialist, A frozen) — Mixup
─────────────────────────────────────────────────────────────
  Data: 80% Med classes + 20% all classes (training set only)
  Loss: CE + Mixup (α=1.0) + λ · L_contrastive
  Why Mixup? KL(LAL || Mixup) has the strongest signal (73.8% > 0.1).
  
  In each batch:
    1. Forward Expert A (frozen) → probs_a
    2. Forward Expert B → probs_b, embeddings_b
    3. Compute KL(probs_a_i || probs_b_i) for each sample i
    4. label_i = 1 if KL > τ else 0
    5. L_contrastive = supervised_contrastive_loss(embeddings_b, label_i)
    6. L_total = L_cls + λ * L_contrastive
    7. Backward → updates B's backbone + routing head + classifier
  
  → Expert B's features encode: "how much does my Mixup distribution
     diverge from Expert A's LAL distribution on this sample?"

Step 3: Train Expert C (Tail specialist, A and B frozen) — PaCo
─────────────────────────────────────────────────────────────────
  Data: 80% Tail classes + 20% all classes (training set only)
  Loss: PaCo (α=0.01, t=0.05, K=1024, dim=32) + λ · L_contrastive
  Why PaCo? KL(Mixup || PaCo) has strong signal (72.2% > 0.1),
  and PaCo's contrastive features provide maximal diversity.
  
  In each batch:
    1. Forward Expert A (frozen) → probs_a
    2. Forward Expert B (frozen) → probs_b
    3. Compute avg_probs = (probs_a + probs_b) / 2  (ensemble average)
    4. Forward Expert C → probs_c, embeddings_c
    5. Compute KL(avg_probs || probs_c) for each sample i
    6. label_i = 1 if KL > τ else 0
    7. L_contrastive = supervised_contrastive_loss(embeddings_c, label_i)
    8. L_total = L_cls + λ * L_contrastive
    9. Backward → updates C's backbone + routing head + classifier
  
  → Expert C's features encode: "how much does my PaCo distribution
     diverge from the A+B ensemble average?"
```

---

## 5. Inference Procedure

### 5.1 Pre-compute Divergence Prototypes (Post-hoc, Validation Set Only)

After all three experts are trained, freeze all backbones. Use the **validation set** to compute per-expert prototypes:

```python
for each expert e:
    for each validation sample:
        embedding = expert.routing_head(expert.backbone(val_image))
        kl_div = KL(probs_e || probs_previous_ensemble)
    
    # Two prototypes:
    expert_e.prototype_agree = mean(embedding for samples where KL < τ)
    expert_e.prototype_disagree = mean(embedding for samples where KL > τ)
```

These prototypes are 32-d vectors. They represent "what does this expert's embedding look like when it agrees/disagrees with the previous expert?"

### 5.2 Test-Time Routing

```python
for each test sample:
    # 1. All three experts produce logits + embeddings
    logits_a, emb_a = expert_A(x, return_embedding=True)
    logits_b, emb_b = expert_B(x, return_embedding=True)
    logits_c, emb_c = expert_C(x, return_embedding=True)
    
    # 2. Measure each expert's "disagreement affinity"
    #    How much does this sample look like a "disagree" sample for each expert?
    scores = []
    for e, (emb, proto_agree, proto_disagree) in enumerate(zip(
        [emb_a, emb_b, emb_c],
        [proto_a_agree, proto_b_agree, proto_c_agree],
        [proto_a_disagree, proto_b_disagree, proto_c_disagree]
    )):
        sim_to_disagree = cosine_similarity(emb, proto_disagree)
        sim_to_agree = cosine_similarity(emb, proto_agree)
        scores[e] = sim_to_disagree - sim_to_agree
        # High score = this sample looks like a "disagree" sample for this expert
    
    # 3. If the three embeddings are all similar → experts agree → uniform is safe
    emb_sim_matrix = cosine_similarity([emb_a, emb_b, emb_c])
    avg_agreement = emb_sim_matrix.mean()  # average pairwise similarity
    
    if avg_agreement > threshold_agree:  # threshold = 0.7 (sweep: 0.5, 0.7, 0.9)
        # All experts agree → use uniform averaging
        final_logits = (logits_a + logits_b + logits_c) / 3
    else:
        # Experts disagree → route to the one that looks most "uniquely disagreeing"
        # This is the expert whose embedding most resembles its "disagree" prototype
        selected = argmax(scores)
        final_logits = [logits_a, logits_b, logits_c][selected]
```

### 5.3 Why This Avoids P3 (Lone Dissenter Paradox)

The routing signal is **embedding similarity to disagreement prototypes** — not the expert's softmax confidence. These are completely decoupled:

- An expert can have low confidence (uncertain prediction) but its routing embedding might still cluster near the "agree" prototype (because its distribution pattern is similar to previous experts')
- An expert can have high confidence (certain prediction) but its routing embedding might cluster near the "disagree" prototype (because its distribution pattern is unique)
- The lone dissenter paradox doesn't apply because we're not using confidence at all

### 5.4 Why This Handles P4 (Label Ambiguity)

The contrastive loss was trained on a binary signal (agree/disagree), not a 3-way "which expert is best" signal. When multiple experts are correct:
- They tend to have similar probability distributions → KL divergence is small → all labeled "agree"
- All three routing embeddings are similar → `avg_agreement` is high → fall back to uniform
- Uniform averaging works well when multiple experts are correct (which is 68.7% of trainable samples)

---

## 5.5 Diagnostic Verification: KL Signal Confirmed

Before finalizing this architecture, we verified that the KL divergence signal is measurable and useful:

**Diagnostic result (on existing experts):**

| Expert Pair | KL > 0.1 (Train) | KL > 1.0 (Train) | Top-1 Agreement (Train) |
|:------------|:----------------:|:----------------:|:-----------------------:|
| LAL vs PaCo | 30.7% | 2.4% | 98.9% |
| **LAL vs Mixup** | **73.8%** | **6.0%** | ~97% |
| PaCo vs Mixup | 72.2% | 4.9% | ~97% |

**Key findings:**
- The strongest signal comes from **cross-paradigm pairs** (LAL vs Mixup: 73.8% > 0.1)
- The weakest signal comes from **same-paradigm pairs** (LAL vs PaCo: 30.7% > 0.1)
- This validates the cascade ordering: A (LAL) → B (**Mixup**) → C (**PaCo**) maximizes KL signal at each step
- High KL on test set is predictive: when LAL vs PaCo disagree + high KL, PaCo is correct 24.1% vs LAL's 10.4%

---

## 6. How DACE Differs from Existing Methods

| Aspect | RIDE | SADE | Boosting | MoE | **DACE (Ours)** |
|:-------|:-----|:-----|:---------|:---:|:----------------:|
| **Backbones** | Shared | Shared | Separate | Separate | **Separate** |
| **Training** | Parallel | Parallel | Sequential | Parallel | **Sequential (cascade)** |
| **Routing signal** | Distribution sampling | Self-supervised | Loss weighting | Random/gated | **KL divergence to previous expert** |
| **Features encode** | Distribution-aware | Augmentation-invariant | Error-weighted classes | Task-specific | **Disagreement pattern** |
| **Validation needed?** | No | No | No | No | **No (training only)** |
| **Addresses P1?** | Only shared backbone | No | Partial | If jointly trained | **Yes — contrastive loss gradient** |
| **Addresses P2?** | No | No | Partial | No | **Yes — data partitioning** |

---

## 7. Hyperparameters

| Hyperparameter | Expert A (Head) | Expert B (Med) | Expert C (Tail) |
|:---------------|:---------------:|:--------------:|:---------------:|
| **Loss** | LAL (τ=1.0) | **CE + Mixup (α=1.0)** | **PaCo (α=0.01, t=0.05)** |
| **Cascade compares to** | — (first expert) | KL(A \|\| B) | KL(avg(A,B) \|\| C) |
| **KL signal rate (>0.1)** | — | **73.8%** (diagnosed) | **72.2%** (diagnosed) |
| Epochs | 200 | 200 | 400 (PaCo standard) |
| Batch size | 128 | 128 | 256 (PaCo standard) |
| Optimizer | SGD (mom=0.9, nest=True) | SGD (mom=0.9, nest=True) | SGD (mom=0.9, nest=True) |
| LR | 0.1 (cosine) | 0.1 (cosine) | 0.05 (step schedule) |
| Weight decay | 5e-4 | 5e-4 | 5e-4 |
| λ (routing weight) | — (no routing head) | 0.1 | 0.05 |
| KL divergence τ | — | **0.1** (lower threshold) | 0.1 |
| Contrastive τ_c | — | 0.5 | 0.5 |
| Routing embedding dim | — | 32 | 32 |

### Search Space

| Parameter | Values | Note |
|:----------|:-------|:-----|
| λ (routing weight) | 0.01, 0.05, **0.1**, 0.5 | |
| KL divergence τ | **0.1**, 0.3, 0.5, 1.0 | Lower threshold captures more signal (30-74% of samples) |
| Contrastive τ_c | 0.1, **0.5**, 1.0 | |
| Inference threshold_agree | 0.5, **0.7**, 0.9 | |
| Routing embedding dim | **32**, 64 | |

---

## 8. Expected Outcomes

| Metric | Current (Uniform) | DACE Target | Source of Improvement |
|:-------|:-----------------:|:-----------:|:----------------------|
| **Overall BA** | 45.68% | **50-54%** | Data partitioning (P2) + routing (P1) |
| Head accuracy | 74.6% | 75-78% | Expert A specializes in head classes |
| Med accuracy | 48.2% | 50-55% | Expert B specializes in med classes |
| Tail accuracy | 17.4% | **25-35%** | Expert C specializes in tail classes |
| All-wrong ceiling | 44.2% | **30-35%** | Every class has a dedicated expert |
| Oracle (≥1 correct) | 55.79% | 65-70% | Partitioning increases class coverage |

---

## 9. Files to Create/Modify

| File | Action | Description |
|:-----|:-------|:------------|
| `models/resnet32.py` | **Modify** | Add `RoutingHead` + `ResNet32WithRouting` dual-output model |
| `losses/contrastive_routing_loss.py` | **Create** | Supervised contrastive loss for routing embeddings |
| `losses/kl_divergence.py` | **Create** | KL divergence computation between two experts' probability distributions |
| `data/partitioned_dataset.py` | **Create** | `PartitionedDataset` — class-group-biased sampling (training set only, no validation mixing) |
| `scripts/train_dace_a.py` | **Create** | Train Expert A (Head specialist, standard LAL) |
| `scripts/train_dace_b.py` | **Create** | Train Expert B (Med specialist, cascade + contrastive routing) |
| `scripts/train_dace_c.py` | **Create** | Train Expert C (Tail specialist, cascade + contrastive routing) |
| `scripts/evaluate_dace.py` | **Create** | Evaluate with prototype-based disagreement routing |

---

## 10. GPU Budget

| Task | Est. Time | Notes |
|:-----|:---------:|:------|
| Expert A training (LAL) | 1.5h | 200 epochs, no routing head, standard |
| Expert B training (Mixup) | 2.0h | 200 epochs, cascade forward through A + routing head |
| Expert C training (PaCo) | **4.0h** | 400 epochs, PaCo is slower (2-view + queue + momentum) |
| Evaluation + analysis | 1.0h | Prototype computation, routing, ablations |
| **Total** | **~8.5h** | Single T4 GPU on Kaggle |

---

## 11. Risk Mitigation

| Risk | Symptom | Mitigation |
|:-----|:--------|:-----------|
| **KL divergence is always small (experts always agree on training data)** | All contrastive labels = 0 | Use a lower KL threshold τ; compare top-5 distributions not full 100-class; try JS divergence instead of KL |
| **Routing head collapses** | All embeddings identical | Gradient clipping; L2 regularization on routing head; monitor embedding variance |
| **Classifier head dominates** | Routing head barely changes from init | Increase λ; normalize gradient magnitudes |
| **Data partitioning hurts accuracy** | Expert B's BA drops below 35% | Increase `full_ratio` to 0.3; use stronger logit adjustment |
| **Cascade is too slow** | Training time exceeds Kaggle limit | Reduce epochs for B and C; use smaller routing head |
| **KL divergence is expensive** | Training time doubles | Compute KL only every N batches; use a simpler divergence metric |
| **Inference threshold doesn't transfer** | Routing BA < uniform | Try adaptive threshold per test batch; fall back to uniform if threshold doesn't help |
