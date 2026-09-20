# Problem — Why Routing Fails

> The verified reasons this project's routing mechanisms do not beat uniform
> averaging. Every claim is labelled **CONFIRMED** (measured, with the number
> and where it came from) or **SUSPECTED** (reasoned, not measured).
>
> Related: [`results.md`](results.md) (all numbers) ·
> [`routing_mechanism.md`](../records/routing_mechanism.md) (what was tried) ·
> [`research.md`](research.md) (literature)

---

## Decision summary

| Finding | Status | Consequence |
|:--|:--:|:--|
| Large all-wrong region / limited oracle headroom | **Confirmed** | Routing cannot rescue samples every expert misses. |
| Confidence points away from the correct lone dissenter on savable cases | **Confirmed** | Confidence-based routing is a poor signal here. |
| Measured expert disagreement is largely not paradigm-specific | **Confirmed** | Low agreement alone is not evidence that the pool is usefully routable. |
| DACE routing score was anti-predictive and cross-expert scales were incomparable | **Confirmed** | Better fitting/tuning of that objective does not solve the mechanism. |
| No honest held-out correctness labels under the current full-training-set protocol | **Confirmed by construction** | Only parameter-free routing is currently valid; fitted routing requires cross-fitting or a permanent holdout. |
| Feature-based learned gate failed to approach oracle routing | **Confirmed on old split** | Existing backbone features did not encode reliable "which expert is right" information. |
| Most routable samples have multiple correct experts | **Confirmed** | A single 3/4-way expert label is often ambiguous. |
| Model-initialization seeding bug | **Fixed** | Current runs seed before model construction; historical runs predate this fix. |

Read the detailed sections below before revisiting any ruled-out mechanism.

## 1. All-wrong floor — CONFIRMED

**Statement.** A large fraction of test samples are classified wrongly by
*every* expert. No router can reach them, so the recoverable region is small.

| Pool | All-wrong floor | Oracle ceiling |
|:--|:--:|:--:|
| Current, 4 experts (mean of 3 seeds) | **39.73%** | 60.27% |
| Same pool, seed 78 / 88 / 1034 | 38.91 / 40.26 / 40.02% | 61.09 / 59.74 / 59.98% |
| Earlier 3-expert pool (old split) | 44.2% | 55.79% |

**Implication.** Routing headroom is `oracle − uniform = 60.27 − 46.98 = 13.3
points`, and it would require *perfect* per-sample expert selection to capture.
No mechanism has captured any of it.

## 2. Lone-dissenter paradox — CONFIRMED

**Statement.** On samples where a router could actually help (uniform is wrong
but some expert is right), the correct expert is systematically the **least**
confident one. Any confidence-based signal therefore points away from the answer.

| Measurement (old split, 596 savable samples) | Value |
|:--|:--:|
| Correct expert *is* the most confident | 16.1% |
| Correct expert is **not** the most confident | **83.9%** |

**Mechanism.** A savable sample is one where two experts are confidently wrong
and one is uncertainly right; the hesitation *is* what makes it correct. This is
why confidence-based routing fails and why the ranked rules in
[`routing_mechanism.md`](../records/routing_mechanism.md) §1 land below uniform.

## 3. The pool's measured "diversity" is mostly initialisation noise — CONFIRMED

**Statement.** The disagreement between experts — the thing a router is supposed
to exploit — comes largely from random initialisation, not from the different
training paradigms.

| Evidence | Value |
|:--|:--:|
| LAL and BalancedSoftmax implement the *same objective* (they differ only by a class-independent constant that cancels in softmax) | κ = **0.458 ±0.012** |
| CE ↔ LAL, two different paradigms | κ = 0.438 ±0.006 |
| CE ↔ Mixup, the two *most similar* paradigms | κ = **0.487 ±0.009** |
| Range across all six pairs | 0.42 – 0.49 |

Two experts trained with mathematically identical losses agree **no more** than
two experts with different losses. And the same-objective pair still diverges in
training (loss differs from epoch 1), because the balanced-softmax form adds
`log(10847) ≈ 9.29` to every logit: that cancels exactly in the forward pass but
changes float rounding in the backward pass, and the difference amplifies
chaotically over 85 steps per epoch.

**Implications.**

- Diversity claims based on κ are not evidence of *paradigm* diversity.
- The project's earlier `κ < 0.80 ⇒ routable` criterion is satisfied by every
  pair, yet routing fails — so it has no predictive power here.
- LAL + BalancedSoftmax are effectively one expert trained twice; the pool is
  closer to three distinct experts than four.

## 4. The routing objective was anti-predictive — CONFIRMED (DACE)

**Statement.** The DACE routing score was *inversely* related to whether the
expert was right, so inference selected the expert most likely to be wrong.

| Quantity | AUROC |
|:--|:--:|
| `score_B` vs "B is correct" | **0.327** |
| `score_B` vs "B is wrong" | 0.673 |
| `score_C` vs "C is correct" | 0.318 |

Learning the contrastive label *better* made routing *worse* — this is not a
tuning problem. Contributing implementation defects, also CONFIRMED:

- **Incomparable score scales.** Per-expert score spreads differed ~8×
  (`score_b` mean +0.601/std 0.449 vs `score_c` mean +0.046/std 0.055), so a
  single `argmax` selected on scale, not merit — it picked expert B on **90%**
  of validation samples.
- **Dead fallback.** The cosine-similarity guard fired at threshold 0.7, but the
  maximum observed similarity was 0.431, so it fired on **0.00%** of samples and
  the intended "experts agree → use uniform" safety valve never operated.
- **Weakened experts.** Partitioning each expert onto one class group made them
  weaker than the originals: DACE uniform **39.84** vs original **45.68** on the
  test set. Expert B, the "medium specialist", had **0.0000** tail recall, and
  expert C — trained on 80% tail data — was in fact the *strongest head* expert.
  **No tail specialist existed**, which is why group-conditional routing could
  not improve tail accuracy however the weights were set.

## 5. No honest held-out data — CONFIRMED (by construction)

**Statement.** Under the current protocol no router can be fitted honestly at
all, so only parameter-free rules can be evaluated.

The canonical protocol trains on the **full** 10,847-sample long-tailed set,
with no validation split. Every sample an expert could be scored on is a sample
it was trained on, so its correctness label there is memorisation (final train
accuracy is 96–98%). Fitting on the test set is excluded by definition. Hence:

- `routing_dev`, carved from the same 10,847 samples, is **not** an honest
  label source.
- Fitted routing would require **cross-fitting** (train experts on K−1 folds,
  score the held-out fold) or a permanent `routing_dev` holdout — both cost
  training data, which is the resource the protocol change was made to protect.

## 6. Feature-learning gap — CONFIRMED (old split)

**Statement.** On the retired old split, the tested backbone-feature gate did
not encode enough reliable information about "which expert will be right on
this sample" to approach the oracle. This is evidence against that tested
signal/probe, not a proof that every future learned router or representation
cannot work.

| Measurement | Value |
|:--|:--:|
| Oracle-weighted routing (perfect soft routing) | 63.04% |
| Learned soft gate (192-d → 3 weights, NLL) | 43.98% |
| **Gap** | **19.06%** |

The learned gate collapsed to always choosing one expert. A 3-way classifier
trained to predict "which expert is correct?" reached only 37.11% on the
unambiguous subset.

## 7. Label ambiguity — CONFIRMED

**Statement.** For most samples where routing could act, more than one expert is
correct, so "which expert should I pick?" has no unique answer.

Computed from the current 3-seed correctness counts: of the ~60% of samples
where at least one expert is right, **73.4 – 74.2%** have two or more experts
correct. (Old split: 69.4%.) Any 3-way selection target is arbitrary for those
samples, which is why the 3-way classifier trained on all samples collapsed
below chance.

## 8. Seeding did not cover model initialisation — CONFIRMED, now FIXED

**Statement.** For the first training round, `--seed` controlled the data order
and the training loop but **not** the model's random initial weights, because
the model was constructed before `set_seed` was called. Two runs with the same
seed therefore produced different models.

- **Fixed**: `set_seed(config.seed)` now runs at the start of
  `ConfigDrivenTrainer.__init__`, before `build_model`. Three regression tests
  cover it (`same seed → identical weights`, `same seed → identical history`,
  `different seeds → different weights`).
- **Correction to an earlier claim:** this bug was once inferred from "κ(LAL, BS)
  = 0.46 instead of ≈1.0". That inference was **wrong** — those two experts
  diverge even with correct seeding, for the float-rounding reason in §3. The bug
  was real, but that symptom did not demonstrate it.

## 9. What is *not* the problem (ruled out)

| Candidate explanation | Why it is ruled out |
|:--|:--|
| Too little training data | 10× more data changed the routing gain by +0.50% |
| Tested router mechanisms | linear, MLP, soft gate, clustering, pairwise, threshold — the recorded variants failed under their tested protocols; this does not rule out every future learned router |
| Wrong input representation | features, logits, probabilities, confidences, entropies — all fail |
| Disagreement routing | the dissenting expert is correct only 15.8% of the time in 2-1 splits |
| Gradient alignment (GDDR) | gradients are near-orthogonal in 3072-d (cosine ≈ 0.03) |
| Threshold tuning | the underlying target is anti-predictive, so better calibration makes the wrong decision more confidently |
| A bug in the current router code | the interface has no fitting path, four rules are verified against their mathematical definitions, and a control run reproduces itself exactly on GPU |
