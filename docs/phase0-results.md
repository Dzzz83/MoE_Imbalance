# Phase 0 Results — Gate G0

> Ran before any training, on the existing checkpoints, as the plan requires.
> Validation split only; the 10K test set was never touched.
>
> Reproduce:
> `python scripts/phase0_ceiling_probe.py`
> `python scripts/phase0b_tail_constrained.py`
> `python scripts/phase0c_group_prediction.py`
> `python scripts/phase0d_class_aware.py`
> `python tests/test_protocol_splits.py`

---

## Verdict: G0 FAILS — and the wider search is now closed

**No routing or combination rule beats uniform averaging beyond noise — on either
expert pool.**

The first reading of this gate blamed the DACE expert pool and recommended rebuilding
it. A wider check falsified that reading: the problem is not this pool, it is the task.

`docs/results.md` §5.6 already recorded the same conclusion on the *original, stronger*
pool (LAL+PaCo+Mixup, uniform 45.68%):

> **"Definitive conclusion: Frozen-expert routing cannot beat uniform averaging on the
> proper split. After testing 14 routing methods, the best routing method ties uniform
> at 45.68%."**

Section 6 below adds one genuinely untested idea (per-**class** weights) on **both**
pools. It also fails. Running total: **~18 methods, 2 pools, one conclusion.**

The structural reason is in the project's own numbers: **44.2% of samples are wrong for
all three experts.** No router can reach those, so the recoverable region is small and
the oracle gap (+10.11pp) requires *perfect* per-sample selection.

---

## 1. Protocol scaffolding (built and green)

`data/protocol_splits.py` + `tests/test_protocol_splits.py` (12/12 passing).

| Split | Size | Permitted use |
|:--|:--:|:--|
| `train_core` | 6,977 | Expert training |
| `routing_dev` | 1,700 | Routing-head training only (honest labels) |
| `val` | 2,170 | Gates / design selection; never trained on |
| test | 10,000 | Final numbers only |

Asserted: `train_core ∩ routing_dev = ∅`, both disjoint from `val`, union reconstructs
`all_train` exactly, split is seed-reproducible, all 100 classes present in
`routing_dev`. The tests include two injected-leakage cases that must be *caught*.

*(Recorded caveat: `lt_val` is long-tailed and some tail classes have **zero** samples,
so the Tail column below is noisy and indicative only. Final Tail claims must come from
the balanced 10K test set.)*

---

## 2. Candidate decision rules vs uniform (5 reps, 50/25/25 fit/tune/eval)

Uniform reference = **logit averaging**, matching `scripts/evaluate_dace.py` and the
project convention. Paired gains over uniform on identical splits:

| method | ΔBA | ΔTail | verdict |
|:--|:--:|:--:|:--|
| calibrated correctness | **−0.0043 ± 0.0110** | +0.0059 ± 0.0400 | **worse** |
| group-conditional | **+0.0026 ± 0.0055** | +0.0154 ± 0.0308 | **gain within noise** |
| group ORACLE (true group) | +0.0223 ± 0.0083 | +0.1053 ± 0.0410 | passes both |
| expert ORACLE | +0.1017 ± 0.0136 | +0.1943 ± 0.0377 | passes both (ceiling) |

Absolute BA: uniform 0.4998, group-conditional 0.5024, calibrated 0.4955.

Two methodological corrections were made during this phase, both of which changed the
answer and are recorded for honesty:

1. An earlier run compared candidates against a uniform baseline built by averaging
   **softmax probabilities** while the candidates combined **logits**. That is a
   different operator (0.4829 vs 0.4998 for the same reference) and made the gain look
   roughly 3× larger. The baseline is now logit averaging throughout.
2. An earlier verdict flagged "PASSES BOTH" on a +0.27pp mean gain. The verdict now
   requires the paired gain to exceed its own standard deviation.

**A tail-constrained blend does not rescue this.** `phase0b` sweeps
`w = (1−α)·uniform + α·routed` with α selected under the constraint `Tail ≥ uniform Tail`.
It reports a nominal pass at +0.0027 BA / +0.0154 Tail — but that margin is inside the
split noise, so it is not a real pass.

---

## 3. Why: the bottleneck is group identification, and it fails exactly where it matters

The oracle-group rule passes both criteria comfortably (+0.0223 BA, +0.1053 Tail), so
the *combination rule* is sound. The loss is entirely in knowing the group.

Group prediction from the ensemble probabilities: overall accuracy **0.8157**
(dedicated 3-way classifier: 0.8359). But overall accuracy is misleading — Head
dominates the split:

| true group | recall | predicted as Head | as Med | as Tail |
|:--|:--:|:--:|:--:|:--:|
| Head | **0.879** | — | 0.119 | 0.002 |
| Med | **0.716** | 0.282 | — | 0.002 |
| **Tail** | **0.114** | **0.534** | 0.352 | — |

**Tail recall is 0.114.** The router cannot distinguish a tail sample from a head
sample — precisely the regime where routing would have to act. This is not an
implementation bug: recognising tail samples *is* the long-tail problem the experts
already fail at.

Routing with the competence table (eval half, n=1085):

| variant | BA | Tail |
|:--|:--:|:--:|
| uniform | 0.4389 | 0.0930 |
| PREDICTED group | 0.4380 | 0.0930 |
| ORACLE group | 0.4551 | 0.1628 |
| predicted group, oracle-fixed where wrong | 0.4472 | 0.1395 |

Predicted-group routing reproduces uniform almost exactly — the oracle advantage is
destroyed by group-prediction error.

---

## 4. The deepest finding: the intended specialisation never materialised

Competence table (per-class recall, fitted on half). The partitioning intended
A = Head, B = Med, C = Tail:

| expert | Head | Med | Tail | intended role |
|:--|:--:|:--:|:--:|:--|
| A | 0.5873 | 0.3825 | 0.1319 | Head specialist |
| B | 0.5284 | **0.4906** | **0.0000** | Med specialist |
| C | **0.6248** | 0.3023 | **0.1354** | Tail specialist |

Best expert per group: Head → **C**, Med → **B**, Tail → **C**.

Three conclusions:

1. **Expert C, trained on 80% tail data, is the strongest HEAD expert** and is only
   marginally the best on Tail (0.1354 vs A's 0.1319, inside noise). 80% of C's batch
   came from only **303 unique tail images**, so it overfits them while its 20%
   all-class sampling still gives it general head ability.
2. **Expert A is dominated by C on every group.** Routing between them is wasted work;
   the pool effectively has ~2 distinct experts, not 3.
3. **No tail specialist exists at all.** B is 0.0000; A and C are both ≈0.13. There is
   no expert to route tail samples *to*, which is why group-conditional routing cannot
   gain on tail no matter how the weights are set.

This is the structural reason the recorded bar (BA **and** Tail above uniform) cannot be
met by routing on the current pool.

---

## 5. The control that changes the conclusion: the ORIGINAL pool also fails

The natural question after §4 was "is the pool the problem, or is routing the problem?".
The original experts are still on disk (`LAL`, `PaCo`, `Mixup`), so this is testable
without any training — a free controlled comparison.

| Pool | uniform BA | uniform Tail | class-aware BA | class-aware Tail | verdict |
|:--|:--:|:--:|:--:|:--:|:--|
| DACE (partitioned) | 0.4998 | 0.1894 | 0.5070 (**+0.0072 ± 0.0092**) | 0.2282 (+0.0388 ± 0.0596) | BA gain **within noise** |
| ORIGINAL (LAL+PaCo+Mixup) | 0.5314 | 0.2689 | 0.5285 (**−0.0029 ± 0.0079**) | 0.2784 (+0.0095 ± 0.0190) | **worse on BA** |

(Validation, logit-averaged uniform, 5× 50/25/25 fit/tune/eval, paired gains.)

Two things follow:

1. The stronger, more diverse original pool scores **higher** (0.5314 vs 0.4998) — which
   independently confirms the DACE partitioning costs accuracy (defect #5). But on that
   *better* pool the class-aware rule is **worse**, not better.
2. Therefore **"rebuild the experts and routing will work" is falsified.** The pool was
   not the blocker.

---

## 6. The one untested idea: per-CLASS weights — also fails

`docs/results.md` §5.3 flagged an idea that had been noted but never implemented:

> "PaCo dominates Head/Med (72%/43%) but is weak on Tail (12%). LAL/BS are weaker on Head
> (60%) but much stronger on Tail (23%). **A class-aware ... router could exploit this.**"

Every method previously tested used **scalar** per-expert weights — one number per expert
for all 100 classes. Per-**class** weights `z[c] = Σ_e w[e][c]·logits_e[c]` are different
in a way that matters here: the class index `c` is always known, so **no sample-level
group prediction is required**. That sidesteps the tail-recall-0.114 failure that killed
group-conditional routing in §3.

Estimated with empirical-Bayes shrinkage toward group competence (so rare classes with
0–2 held-out samples are not driven by noise), sharpness `T` and shrinkage `k` selected
on the tune split. `T → ∞` is exactly uniform, so the baseline is inside the search space.

Result: **+0.0072 ± 0.0092 BA on DACE (not significant), −0.0029 ± 0.0079 on ORIGINAL.**

The direction of the Tail effect on DACE (+3.9pp) is consistent with §5.3's prediction,
but its standard deviation (6.0pp) exceeds the effect, so it cannot be claimed.

A divide-by-zero producing NaN weights when `k=0` and a class was absent from the fit
split was found and fixed; re-running with warnings-as-errors confirmed it did not
materially change either result.

---

## 7. Consolidated evidence

| Pool | Method | ΔBA vs uniform |
|:--|:--|:--|
| ORIGINAL | 14 routing methods (`results.md` §5.4, §5.6) | best **ties** uniform |
| ORIGINAL | optimal **scalar** fixed weights | −0.79% |
| ORIGINAL | product / geometric combination | +0.00% |
| ORIGINAL | class-aware per-class weights (§6, new) | **−0.29pp** |
| DACE | calibrated per-expert correctness (§2) | **−0.43pp** |
| DACE | group-conditional competence (§2) | +0.26pp, within noise |
| DACE | tail-constrained blend (§2) | not a real pass |
| DACE | class-aware per-class weights (§6, new) | +0.72pp, within noise |
| both | oracle (upper bound, not achievable) | +10.11pp / +2.23pp |

---

## 8. Decision

**Stop the routing/combination search. Do not spend Kaggle budget on more routing
experiments.**

Rationale:

1. Eighteen methods over two pools have failed. The two independent lines of work
   (this Phase 0 and the earlier 14-method study in `results.md`) agree.
2. The binding constraint is structural, not architectural: a **44.2% all-wrong floor**
   caps what any router can recover, and the accessible headroom is smaller than the
   measurement noise.
3. Each full A+B+C retrain costs ~1.6h on Kaggle plus upload/download, and Phase 5's
   "rebuild the experts" hypothesis has now been directly falsified by the
   original-pool control in §5.

   *(Correction, recorded 2026-09-11: this item previously read "There is **no local
   GPU**". That is no longer true — the workspace machine now has an RTX 3060 Laptop
   (6 GB, sm_86) with a CUDA-enabled torch build (`2.14.0+cu126`). The GPU is for
   verification and testing only; full training still happens on Kaggle, so the
   budget argument above is unchanged. See `docs/specs/gpu-verification.md`.)*

### What I recommend instead

The evidence supports an **analysis / negative-result contribution**, which is a
legitimate and defensible research outcome, and the material for it already exists:

- a decomposition of why disagreement-aware routing fails (anti-predictive KL target,
  AUROC 0.327, §1 of `dace-diagnosis-findings.md`);
- the **oracle decomposition** separating "signal is absent" from "signal is inaccessible";
- the **all-wrong floor** (44.2%) as a hard ceiling;
- the **partitioning ablation** (45.68% → 39.84%) showing the cascade costs accuracy with
  no compensating routing benefit;
- 18 methods over 2 pools, all with paired significance reporting.

The one thread worth a *small* amount of further work is the **class-aware Tail effect**
(+3.9pp on DACE, consistent with the prior in §5.3). It is not yet claimable — it needs a
proper measurement with more repetitions before it could be reported as a real finding,
and it is a combination-rule result, not a routing result.

### Do NOT

- Do not start Phase 4 (3-seed test runs) — G3 cannot pass.
- Do not retrain the experts expecting routing to then succeed; §5 falsifies this.
- Do not tune thresholds to manufacture a pass; §2 already showed the target is
  anti-predictive, so threshold tuning cannot repair it.
- Do not report the class-aware Tail gain as a finding without more repetitions.
