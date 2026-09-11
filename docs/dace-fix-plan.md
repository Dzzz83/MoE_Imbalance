# DACE Fix — Implementation Plan

> Derived from `docs/dace-diagnosis-findings.md` (measured) plus one further
> go/no-go experiment (`scripts/diagnose_dace_gonogo.py`). Every design decision
> below is tied to a specific measured defect or deliberately flagged as an
> open bet with a gate that tests it cheaply.

---

## 1. What the evidence forces

| Finding (measured) | Consequence for the design |
|:--|:--|
| Routing score vs correctness AUROC = **0.327** (anti-predictive) | The routing **target** must be re-aimed. Threshold tuning cannot help. |
| Per-expert correctness **is** predictable: confidence 0.81–0.87, embedding probe 0.68–0.70 | A correctness target is learnable — but see the next row. |
| **Yet** argmax-over-predicted-correctness loses 5.59pp (0.3728 vs 0.4287 uniform), capturing **−49.5%** of oracle headroom | Learning the target is **not sufficient**. The **decision rule** (how experts are compared and combined) is a first-class part of the fix. |
| `score_b` mean +0.601/std 0.449 vs `score_c` mean +0.046/std 0.055 | Cross-expert scores must be **calibrated into a common unit** before comparison. |
| Fallback fires on **0.00%** (max sim 0.431 vs threshold 0.7) | Replace the hard threshold with a **continuous** uniform-interpolation whose parameter is tuned on validation. |
| τ=0.1 → 93.6%/97.3% one-sided labels; loss range compressed 0.634 → 0.219 | The KL-threshold design is removed entirely, not re-tuned. |
| Uniform (41.79% val) beats **every** single expert (35.60–37.09%) | Routing must be a **soft reweighting around uniform**, never a hard selector that throws the ensemble away. |
| DACE uniform 39.84% vs documented 45.68% | Reported as a separate finding; not the bar the user chose (bar = DACE's own uniform). |

**The central insight:** uniform averaging is a strong baseline precisely because it
cancels errors. Any router that hard-selects one expert discards that benefit and must
out-predict it. The fix must therefore be able to *degrade gracefully to uniform* when
the signal is weak — and the data says the signal will be weak.

---

## 2. The non-cheating protocol (the backbone of the whole plan)

The diagnosis showed the current code has **no honest source of "which expert is
correct"**. On the training set the experts are memorised (train loss → 0.08), so
correctness labels there are ~100% and useless — this is why the original design reached
for KL-disagreement as a proxy, and that proxy is anti-predictive.

Honest labels require **held-out data that no expert was trained on**. That is the crux.

### Data roles (enforced, not just documented)

| Split | Size | Permitted use |
|:--|:--:|:--|
| `train_core` (80% of lt_train) | ~6,941 | Expert training: backbone + classifier + routing head |
| `routing_dev` (20% of lt_train) | ~1,736 | **Routing-head training only** — source of honest correctness labels |
| `lt_val` | 2,170 | Design/hyperparameter selection, gates, checkpoint selection. **Never trained on.** |
| CIFAR-100 test | 10,000 | **Final reported numbers only.** Touched once per seed. |

`routing_dev` and `train_core` are carved from `lt_train_indices.npy` only. `lt_val` and
the test set are never in any gradient path. This satisfies the plan's constraint that all
training signals come from the 8,677 training samples — the split is *internal* to them.

### Enforcement (not just intent)

- A shared `load_protocol_splits()` helper returns the four index sets plus an
  assertion that pairwise intersections are empty.
- Unit test: assert `train_core ∩ routing_dev = ∅`, both disjoint from `lt_val`, and that
  the test set index set is never passed to any `Trainer`.
- The test set is loaded by exactly one function, called only from the final evaluation
  entry point.

---

## 3. Root cause → fix mapping

| # | Diagnosed defect | Fix | Removes the defect how |
|:-:|:--|:--|:--|
| 1 | Routing target anti-predictive (0.327) | **Re-aim target**: contrastive over `{correct, incorrect}` from honest `routing_dev` labels | The learned quantity becomes the decision-relevant one |
| 2 | Scores incomparable across experts | **Per-expert calibration** (Platt scaling on `routing_dev`) → all experts output a probability | Common unit; `argmax` becomes meaningful |
| 3 | Fallback unreachable (0.7) | **Temperature-softmax weighting**, `w = softmax(log p_e / T)`; `T` tuned on validation | `T→∞` is exactly uniform; graceful degradation replaces a dead hard gate |
| 4 | τ=0.1 mis-calibrated | **Delete τ and KL entirely** | Defect becomes structurally impossible |
| 5 | Experts weaker than originals | Keep partitioning; expose `full_ratio` as a tunable (0.2 → 0.4) | Recovers general accuracy where partitioning over-specialised |
| 6 | Compressed, unreadable loss | **Balanced contrastive metric + routing AUROC logged each epoch** | Collapse becomes visible during training, not at test time |
| 7 | Reporting/diagnostic gaps | Fix `n_uniform`; log score stats, weight distribution, fallback rate | Wrong inferences become impossible |
| 8 | Routing-head gradient dominated (suspected) | Loss-weight audit; report routing-vs-classifier gradient norms | Turns a suspicion into a measured quantity |

### Explicit trade-off you should sign off on

Re-aiming the target requires honest labels, and honest labels only exist for data the
expert did not train on. Therefore the routing head **cannot backprop into the backbone**
without invalidating the labels that train it.

This deliberately gives up the original plan's **P1** claim ("gradient from routing loss
flows into backbone features"). We measured that this mechanism *did* work (head AUROC
0.96 vs random head 0.7346) — it simply learned the wrong thing. Options:

- **(a) Recommended — frozen-backbone meta-router.** Scientifically clean: base experts
  are fixed, the router is a small honest meta-model trained on held-out data. Standard
  stacking. P1 is dropped.
- **(b) Iterative cascade (optional Phase 6).** Round 1 trains the meta-router; round 2
  re-trains experts with the routing head attached and recomputes labels; iterate. Keeps a
  weak form of P1 at roughly 2× cost and real complexity. Attempt only if (a) shows a
  real gain.

I recommend (a) and treating (b) as a stretch experiment.

---

## 4. Phases and gates

Every phase ends in a **gate**. No phase starts until the previous gate passes, so a dead
end is discovered in minutes rather than after three full training runs.

### Phase 0 — Protocol scaffolding and cheap ceiling probes

Build the split helper and the index-disjointness tests. Then, using the **existing**
checkpoints, measure on validation (as a measuring instrument only) the ceiling of three
candidate decision rules:

- (i) calibrated per-expert correctness, soft-weighted (the main proposal);
- (ii) **group-conditional routing** — route by predicted class-group membership, since
  the experts are already specialised by group (A Head 61%, B Med 53%, C Tail 16%);
- (iii) uniform (reference).

> **G0:** if no candidate beats uniform on validation, routing is not the lever — pivot to
> fixing the experts (Phase 5) and report routing as a negative result. Cost: minutes.

### Phase 1 — Experts on `train_core` + honest labels

Train A/B/C on `train_core` (cascade order and partitioning preserved). Collect their
predictions on `routing_dev`.

> **G1:** the experts' accuracy on `routing_dev` must be far below their training accuracy
> (train loss reaches ~0.08). If `routing_dev` accuracy is ~99%, the split leaked and the
> labels are not honest — stop and fix the split.

### Phase 2 — Re-aimed routing head

Freeze expert backbones and classifiers. Train each routing head with the contrastive loss
over `{correct, incorrect}` using `routing_dev`. Report the resulting label balance (it
should be far healthier than the 93.6%/97.3% that killed the previous objective).

> **G2:** routing AUROC for correctness on `routing_dev` (held-out portion) and on
> `lt_val` must exceed **0.60**. Below that, the head is not learning anything usable.

### Phase 3 — Calibrated decision rule

Per-expert Platt scaling on `routing_dev`, then `w = softmax(log p_e / T)` with `T`
selected on `lt_val` over a grid including `T → ∞` (exact uniform). Combine as
`Σ w_e · logits_e`.

> **G3:** routed validation BA **and** Tail accuracy must both exceed the same-pool uniform
> baseline. If not, iterate on the decision rule here — do not spend a test run.

### Phase 4 — Final evaluation (only after G3 passes)

Final experts re-trained on the **full** `lt_train` for the reported configuration (the
meta-router transfers; its calibration is re-fit on `routing_dev` drawn from the same
pool). Evaluate on the test set, **3 seeds {0, 42, 123}**, reporting routed vs uniform on
the identical expert pool.

### Phase 5 — Expert strengthening (conditional, and independent)

Partitioning is confirmed to weaken experts. Expose `full_ratio` (0.2 → 0.4) and, for
Expert C, recognise the overfitting risk that its 80% pool is only **303 unique tail
images** (`AGENTs.md` §9 red flag). Run only if G0/G3 point at expert weakness.

### Phase 6 — Iterative cascade (optional stretch)

Only if Phase 4 shows a genuine routed gain, test whether routing gradients in the
backbone add anything on top (trade-off (b) above).

---

## 5. Files

| File | Action |
|:--|:--|
| `data/protocol_splits.py` | **New** — `load_protocol_splits()` + disjointness assertions |
| `scripts/build_routing_labels.py` | **New** — honest correctness labels from `routing_dev` |
| `scripts/train_dace_router.py` | **New** — frozen-backbone routing-head training |
| `scripts/calibrate_router.py` | **New** — Platt scaling + temperature selection on validation |
| `scripts/evaluate_dace.py` | **Modify** — calibrated soft-weighted inference; fix `n_uniform`; log score/weight/fallback stats |
| `scripts/train_dace_{a,b,c}.py` | **Modify** — `train_core` only; `full_ratio` exposed; routing AUROC logged per epoch |
| `losses/contrastive_routing_loss.py` | **Keep** — unchanged; only the *labels* change |
| `losses/kl_divergence.py` | **Keep** — no longer used by the routing path (τ removed) |
| `tests/test_protocol_splits.py` | **New** — no-leakage tests |
| `tests/test_router_calibration.py` | **New** — calibration and temperature-fallback tests |

---

## 6. Hyperparameters and search space

| Parameter | Value | Basis |
|:--|:--|:--|
| `routing_dev` fraction | 0.20 | 1,736 honest samples; upgrade to 5-fold cross-fitting (~8,677) only if G2 fails |
| Routing loss | contrastive, τ_c = 0.5 | unchanged from current code |
| Routing labels | `{correct, incorrect}` | replaces KL agreement |
| λ (routing weight) | swept {0.05, 0.1, 0.3} | defect #8 — gradient balance unverified |
| Calibration | Platt scaling per expert, fit on `routing_dev` | produces comparable probabilities |
| Temperature `T` | grid {0.1, 0.25, 0.5, 1, 2, 5, ∞} on validation | `T=∞` **is** uniform; guarantees a uniform fallback is always in the search space |
| `full_ratio` | 0.2 (current) vs 0.4 | Phase 5 only |
| Seeds | 0, 42, 123 | `AGENTs.md` §6 |

No KL threshold remains anywhere in the routing path.

---

## 7. Acceptance criteria

Recorded bar (from the settled spec):

- Routed BA **and** Tail accuracy both exceed the **same-pool** uniform baseline.
- Averaged over **≥3 seeds**, improvement consistent across seeds.
- Cascade training and class-group partitioning preserved.

Recomputed per run, because the baseline must be the uniform of the *same* expert pool:
the historical 39.84% BA / 5.68% Tail are reference points only.

Unchanged constraints: validation never trained on; test set touched once per seed;
`imb_factor` fixed across comparisons; no method accepted from a single run.

---

## 8. Risks

| Risk | Symptom | Mitigation |
|:--|:--|:--|
| Correctness still hard to compare across experts (the measured failure) | G2 passes (~0.68) but G3 fails | The gate catches it in minutes. Fall back to group-conditional routing (Phase 0(ii)), which sidesteps expert ranking entirely by exploiting the partitioning we already have |
| `routing_dev` too small (1,736) | Noisy labels, unstable calibration | Upgrade to 5-fold cross-fitting |
| Soft weighting collapses to uniform at every `T` | Routed ≈ uniform exactly | That is a *safe* failure, not a regression — but report it as "no gain", never as success |
| Experts too weak for any gain to be visible | Routed gain within noise | Phase 5 |
| Frozen backbone loses P1 | DACE's stated novelty is reduced | Explicitly flagged in §3; Phase 6 offers the iterative alternative |

---

## 9. What I recommend NOT doing

- **Do not tune τ.** τ is deleted; the defect was structural.
- **Do not tune the 0.7 fallback threshold.** It is replaced by the temperature parameter.
- **Do not re-run the same architecture with only threshold changes** and present any
  change as a fix — the anti-predictive target would remain.
- **Do not use validation to train the routing head**, even though it would be convenient
  and would inflate the numbers. It is the exact boundary the project set.
- **Do not hard-select a single expert.** Uniform beats every expert; hard selection
  discards the ensemble that makes the baseline strong.
- **Do not judge on one seed.**

---

## 10. Immediate next step

Phase 0 is minutes of work and is decisive: it tells us whether *any* routing rule can beat
uniform on this expert pool, before a single training hour is spent. I recommend running it
first and letting its result choose between the routing path (Phases 1–4) and the expert
path (Phase 5).
