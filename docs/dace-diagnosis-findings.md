# DACE Diagnosis — What Is Broken and Why Routing Collapsed

> Read-only diagnosis. No fixes implemented. Every claim is labelled **CONFIRMED**
> (measured on the real trained checkpoints) or **SUSPECTED** (reasoned, not measured).
> Reproduction:
> `python scripts/diagnose_dace.py` and `python scripts/diagnose_dace_controls.py`.
>
> Measurement split: long-tailed validation (2,170 samples). The 10K balanced test
> set is quoted only where it was already recorded. Validation and test numbers are
> **not** directly comparable (validation is long-tailed, test is balanced).

---

## 0. Correction to an earlier claim

An earlier reading of the training curve concluded that "the routing heads never
trained", because `L_contrastive` moved only 4.8438 → 4.8017 (Expert B) and
4.8475 → 4.8240 (Expert C). **That conclusion was wrong**, and the controls below
falsify it. The heads *did* learn. The training curve simply cannot show it, because
the metric is compressed under a 94–97% one-sided label distribution.

---

## 1. The headline defect: the routing objective is anti-predictive of correctness

**CONFIRMED.** The routing score is *inversely* related to whether the expert is right:

| Quantity | AUROC |
|:--|:--:|
| `score_B` vs "B is correct" | **0.3269** |
| `score_B` vs "B is wrong" | **0.6731** |
| `score_C` vs "C is correct" | **0.3184** |

An AUROC of 0.327 for correctness (0.673 for wrongness) means samples with a **high**
disagreement score are the samples where that expert is **more likely to be wrong**.
Since inference selects `argmax(scores)`, the router systematically selects the expert
most likely to be wrong.

This is a quantified confirmation of the lone-dissenter paradox already recorded in
`research-findings.md` §1.3 ("the correct expert is systematically the least
confident one"), now measured against the DACE routing signal specifically.

**This single defect explains the observed collapse** (routed 33.91% vs uniform 39.84%).
It is not a tuning problem: learning the label *better* makes routing *worse*.

---

## 2. Q1 — Why the contrastive loss looked like it never trained

### 2a. The label is near-constant at τ=0.1 — CONFIRMED

Measured KL distributions on validation:

| Pair | mean | median | p5 | p95 |
|:--|:--:|:--:|:--:|:--:|
| KL(A‖B) | 1.7158 | 1.6199 | 0.0804 | 3.7718 |
| KL(avg(A,B)‖C) | 1.8905 | 1.6351 | 0.1814 | 4.5763 |

τ=0.1 sits near the **5th percentile**, far below the median of ~1.62:

| τ | disagree (B) | disagree (C) |
|:--:|:--:|:--:|
| **0.1 (used)** | **93.6%** | **97.3%** |
| 0.3 | 84.7% | 91.2% |
| 0.5 | 77.5% | 84.3% |
| 1.0 | 64.4% | 69.3% |
| 2.0 | 42.1% | 40.0% |

The `0.1` threshold was carried over from a diagnostic run on the *old* experts
(plan §5.5, which reported 73.8% / 72.2% disagree). On the DACE experts the KL scale
is larger, so the same threshold produces a far more extreme split.

### 2b. The imbalance compresses the loss, hiding the learning — CONFIRMED

Achievable range of `L_contrastive` at batch size 128 (`ln(127) = 4.8442`):

| label split | chance (random) | floor (perfect clustering) | range |
|:--|:--:|:--:|:--:|
| disagree = 93.6% (B, τ=0.1) | 4.9049 | 4.6855 | **0.2193** |
| disagree = 97.3% (C, τ=0.1) | 4.9070 | 4.7794 | **0.1276** |
| balanced 50/50 | 4.9055 | 4.2720 | **0.6336** |

Observed training movement versus the range actually available:

| | observed Δ | available range | fraction of range captured |
|:--|:--:|:--:|:--:|
| DACE_B | 0.0421 | 0.2193 | **19.2%** |
| DACE_C | 0.0235 | 0.1276 | **18.4%** |

So the loss did descend, but the 94–97% imbalance shrinks the observable range ~3–5×,
making a real 19% gain look like "nothing happened" on the printed curve.

### 2c. The heads did learn — CONFIRMED by control

Control: identical frozen backbone features, trained routing head vs a **random**
routing head of the same shape, same probe:

| Probe target | trained head | random head | backbone only (no head) |
|:--|:--:|:--:|:--:|
| agree/disagree, τ=0.1 | **0.9590** | 0.7346 | 0.7985 |
| agree/disagree, τ=1.0 | 0.7923 | 0.6704 | 0.6937 |

The trained head beats a random head by **+0.224 AUROC** and beats the shared backbone
alone by **+0.160**. So the routing head genuinely encodes agree/disagree.

Note the confound this control exposes: the *shared backbone* alone already scores 0.80,
because agree/disagree is partly a function of sample identity/difficulty. Any future
evaluation of the routing head must subtract this baseline, or it will overstate learning.

**Verdict on Q1:** the objective trained; the label was mis-calibrated (τ too low) and the
metric was unreadable under the resulting imbalance. The failure is not "no learning".

---

## 3. Q3 — Are the DACE experts themselves too weak?

**CONFIRMED — yes, and this is independent of routing.**

| Split | Expert A | Expert B | Expert C | Uniform |
|:--|:--:|:--:|:--:|:--:|
| Validation BA | 37.09% | 36.20% | 35.60% | **41.79%** |
| Test BA (recorded) | 36.58% | 33.96% | 33.90% | **39.84%** |

Documented references on the same balanced 10K test set:

| | BA |
|:--|:--:|
| DACE 3-expert uniform | **39.84%** |
| Original 3-expert uniform (LAL+PaCo+Mixup) | **45.68%** |
| Gap | **−5.84 pp** |

Individual DACE experts (33.9–36.6%) are all below every original expert
(PaCo 41.17, LAL 39.96, BS 39.99, Mixup 37.52, CE 36.64). The 80%-on-one-group
partitioning bought target-group accuracy at the cost of general accuracy, and each
expert is near-zero on its non-target groups (B has Tail = 0.0%), which is what
destroyed Tail accuracy under routing (0.12% vs 5.68% uniform).

**Implication:** even a *perfect* routing fix cannot reach 45.68% while the expert pool
sits at 39.84%. The expert side must be addressed too. (Per the recorded decision, the
success bar for DACE is its own uniform of 39.84%, but this gap is the reason the
project-level target of 45.68% is out of reach without stronger experts.)

---

## 4. Q4 — Are score comparability and the fallback fixable by tuning?

### 4a. The three scores are not comparable — CONFIRMED

| Score | mean | std | min | max |
|:--|:--:|:--:|:--:|:--:|
| `score_a` (derived) | −0.3232 | 0.2417 | −0.5048 | +0.7012 |
| `score_b` | **+0.6008** | **0.4493** | −1.2244 | +0.9271 |
| `score_c` | **+0.0457** | **0.0549** | −0.1892 | +0.1164 |

`score_b` and `score_c` live in different embedding spaces with **opposite offsets** and
an **~8× difference in spread**. Comparing them with a single `argmax` is meaningless —
it selects on scale, not on merit. Measured consequence: `argmax` picks B for **90.0%**
of validation samples (A 9.1%, C 0.8%), closely matching the test-run report
(B 95.7%). This is the same collapse mode documented in `research-findings.md` §1.1.

`score_a` is not learned at all — it is defined as `-(score_b + score_c)/2`, so Expert A
can only win under a narrow algebraic condition. Expert A has no routing head.

### 4b. The uniform fallback is dead — CONFIRMED

The fallback fires when `cosine_similarity(emb_b, emb_c) > 0.7`:

| threshold | fraction of samples that would fire |
|:--:|:--:|
| 0.3 | 3.27% |
| 0.5 | **0.00%** |
| 0.7 (used) | **0.00%** |

Measured `emb_sim(B,C)`: mean −0.1913, std 0.1492, **max 0.4308** — the threshold is
never reached, so `use_uniform` is always False and the router commits to a single
expert on every sample. The intended "experts agree → trust uniform" safety valve
never operates. (This also explains why routed BA tracked Expert B alone almost exactly:
33.91% vs 33.96%.)

**Verdict on Q4:** both are *mechanically* fixable with cheap changes (normalise scores
per expert; derive the threshold from the empirical similarity distribution instead of
a hard-coded 0.7). But fixing them is **not sufficient**: with the score anti-predictive
of correctness (§1), a better-calibrated router would simply make the wrong decision
more confidently. These are enabling fixes, not the fix.

---

## 5. Q5 — Headroom and the all-wrong ceiling (validation)

**CONFIRMED.**

| | value |
|:--|:--:|
| all-wrong (0 of 3 correct) | **26.36%** |
| exactly 1 correct | 15.58% |
| exactly 2 correct | 19.40% |
| exactly 3 correct | 38.66% |
| uniform BA | 41.79% |
| oracle BA (≥1 correct exists) | **52.40%** |
| routing headroom | **+10.61 pp** |

Label ambiguity: of the 73.64% of samples where at least one expert is correct,
**78.8%** have two or more correct experts — so a majority of the "trainable" region
remains ambiguous, consistent with the 69.4% figure already recorded.

Headroom exists (+10.61 pp), so a correct router is not blocked by the ceiling.

---

## 6. Secondary defects found

1. **Reporting bug — CONFIRMED.** `evaluate_dace.py` line 477 computes
   `n_uniform = (routing_decisions == -1).sum()`, but `best_expert` only ever contains
   0/1/2, so this is always 0 and is never printed. The printed "Selected Expert"
   percentages are raw `argmax` counts and **ignore whether the uniform fallback fired**,
   so they do not describe the real decision distribution.
2. **No routing diagnostics were logged — CONFIRMED.** Training history records
   `train_loss_contrastive` and `train_kl_agree_ratio`, but never score statistics or
   embedding variance. A collapse was therefore invisible until test time.
3. **Validation split has empty classes — CONFIRMED.** The log shows
   `Tail class (99) count: 0`. Prototypes are estimated on a long-tailed split with
   missing tail classes but applied to a balanced test set — a distribution mismatch in
   the routing signal itself.
4. **Routing-head gradient is tiny relative to the classifier — SUSPECTED.** In synthetic
   runs the routing head receives far smaller gradients than the classifier (e.g. 1.1e-3
   vs 2.3e2 summed). The plan's own risk table anticipates this ("classifier head
   dominates"). Not measured on the real run.

---

## 7. Summary — cause ranking

| # | Defect | Status | Fixable by tuning? |
|:-:|:--|:--|:--|
| 1 | Routing score is **anti-predictive** of correctness (AUROC 0.327) | CONFIRMED | **No** — needs re-aiming |
| 2 | Score scales incomparable across experts → `argmax` picks B 90% | CONFIRMED | Yes (normalisation) |
| 3 | Uniform fallback threshold 0.7 unreachable → never fires | CONFIRMED | Yes (data-derived threshold) |
| 4 | τ=0.1 mis-calibrated → 93.6%/97.3% one-sided labels | CONFIRMED | Yes (raise τ to ~1.0–1.6) |
| 5 | DACE experts weaker than originals (−5.84 pp ensemble) | CONFIRMED | No — needs retraining |
| 6 | Loss metric compressed by imbalance → unreadable curve | CONFIRMED | Yes (balanced metric / AUC) |
| 7 | Reporting + diagnostics gaps | CONFIRMED | Yes |
| 8 | Routing-head gradient dominated by classifier | SUSPECTED | Unknown |

**Bottom line:** the architecture's *training machinery* works — the routing head demonstrably
learns agree/disagree at AUROC 0.96. What fails is that **the thing being learned is
anti-correlated with correctness**, and that the inference-time combination of the
resulting scores is invalid. Fixing thresholds alone cannot repair a signal that points
backwards.

---

## 8. Not done in this round (by design)

No fixes, no retraining, no threshold tuning presented as a fix. The fix design is the
next round, under the recorded bar: beat DACE's own uniform (39.84% BA / 5.68% Tail) on
**both** BA and Tail accuracy, averaged over **≥3 seeds**, preserving cascade training
and class-group partitioning, with the original experts free to be retrained.
