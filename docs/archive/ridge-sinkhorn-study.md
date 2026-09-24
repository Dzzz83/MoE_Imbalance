# Ridge/Sinkhorn Study: Frozen Protocol, Reproduction, and Outcome

> **Status:** Complete through one locked outer-fold-0 evaluation. The
> prespecified expansion gate failed, so no five-fold, three-seed expansion
> was run. Outer fold 0 has been consumed and is unavailable for further method
> selection. The original balanced CIFAR-100 test set was not accessed.

This document combines the frozen study protocol, implementation contracts,
reproduction commands, and completed outcome. The [OOF results](../oof-results.md)
remain the authoritative source for complete metrics, paired intervals,
provenance, and the leakage audit.

## 1. Study question and evidence at protocol freeze

The study asked whether Ridge could predict useful per-image expert weights
and whether Sinkhorn-style allocation could improve those weights. It first
isolated allocation using the existing Task 3F-A Ridge score source, then
tested a classification-aligned residual Ridge target. Only one locked,
batch-independent candidate could advance to the reserved outer fold.

The original full-data expert pool was CE, LAL, BalancedSoftmax, and Mixup,
trained on all 10,847 CIFAR-100-LT images with seeds 78, 88, and 1034. Its
historical balanced-test uniform-logit result was 46.98% BA / 18.76% Tail.
Those test-set measurements are not directly comparable with OOF results.

The completed OOF collection used expert seed 78, outer fold 0, and four
aligned inner folds. Inner folds 1–3 supplied 6,507 router-fitting rows. Inner
fold 0 supplied 2,170 selection rows, and the 2,170 outer-fold-0 rows were
reserved for one locked evaluation. Task 3C had already inspected inner fold 0
descriptively, so it was not an untouched confirmation set.

At protocol freeze, the existing confidence-only contribution Ridge reached
36.5502% BA / 7.3797% Tail on the permitted fitting rows, versus 35.9328% /
7.0094% for uniform logits. It did not beat the fixed-weight references on
both metrics. Full-13 features reduced contribution MSE but worsened the
primary BA/Tail result. The relevant implementation was
[`scripts/task3f_ridge.py`](../../scripts/task3f_ridge.py); its artifacts remain
immutable controls in
[`artifacts/oof/task3f_ridge/`](../../artifacts/oof/task3f_ridge/).

## 2. Frozen scientific decisions

- **Study sequence:** Isolate allocation on frozen Ridge scores, then redesign
  Ridge. Do not run an unrestricted joint search.
- **Inference:** Batch-coupled OT is diagnostic; outer evaluation requires
  batch-independent predictions.
- **Capacity:** Relaxed expert mass is primary. Exact 25/25/25/25 Sinkhorn is a
  control for unequal expert quality.
- **OT prior:** `q = 0.95 * fixed_007 + 0.05 * uniform =`
  `[0.0125, 0.25, 0.4875, 0.25]`.
- **Ensemble:** Weighted original logits with soft all-four routing is primary.
  Probability mixtures, top-2, and hard argmax are diagnostics.
- **Frozen score:** Confidence-only Ridge uses alpha `1000`, class-weight gamma
  `1`, score temperature `2`, and shrinkage `0.75`. Full-13 and training-only
  global scores are diagnostics.
- **Residual anchor:** `fixed_007 = [0, 0.25, 0.50, 0.25]`; its fitting-partition
  result was 36.4403% BA / 14.7290% Tail.
- **Residual target:** Zero-margin, anchor-regularized multiclass hinge-oracle
  weights minus the anchor. Correct anchor predictions have zero residual.
- **Residual features:** Four expert confidences are primary; full-13 is
  secondary. There is no learned route/stay gate.
- **Selection:** Choose the highest BA among joint BA/Tail improvements that
  create a new fixed-reference Pareto point. Break ties by Tail, no-OT, then
  stable configuration ID.
- **Endpoint:** Evaluate one locked outer-fold-0 candidate. Expand to five
  folds and three seeds only if its prespecified gate passes.

The frozen fixed references were `fixed_006`, `fixed_007`, `fixed_010`,
`fixed_011`, uniform logits, uniform probabilities, and uniform without CE.
The mean training no-OT weights were rejected as an OT prior because those
rows already satisfy that marginal, making fitted prices an identity
correction. `fixed_006` was the BA-strong reference at 37.2134% BA / 9.9115%
Tail on the router-fitting partition. `fixed_007` became the residual anchor.
No retrospective reference was promoted.

## 3. Implementation and numerical contracts

All array APIs use fixed expert order and return finite, nonnegative,
row-stochastic weights. The implementation uses separate versioned modules
and artifact schemas; Task 3F-A functions and artifacts are not modified to
reproduce the new method.

### 3.1 Shared array-only components

1. **Relaxed OT:** Given strictly positive row kernels `K`, optimize over
   row-stochastic `P`:

   ```text
   mean_i KL(P_i || K_i) + rho * KL(mean_i P_i || q)
   ```

   Use `rho = {0.1, 1, 10}`, deterministic log-domain updates, convergence and
   residual diagnostics, and fail on non-convergence. At `rho = 0`, return the
   original kernel. Compare the same kernel before and after allocation.
2. **Balanced Sinkhorn control:** Keep every row sum equal to one and enforce
   exact uniform column-average mass `[0.25] * 4`; validate both marginals.
3. **Frozen dual prices:** Learn four expert multipliers on fitting rows only.
   Apply their softmax-normalized product with each future row kernel
   independently. These prices are an OT-derived global bias, not live batch
   coordination. Include a prior/global-bias control.
4. **Oracle target:** For each labeled fitting row and anchor `w0`, solve
   `min_(w in simplex, t >= 0) t + (lambda/2)||w-w0||²`, subject to
   `t >= sum_e w_e (z[e,c] - z[e,y])` for every `c != y`. Use deterministic
   SciPy constrained optimization and check objective, simplex, slack, and
   feasibility residuals. Cache targets by oracle penalty; they do not depend
   on Ridge alpha/gamma or prediction scale.
5. **Residual weight map:** Project `w0 + scale * predicted_residual`
   Euclideanly onto the four-expert simplex. If a residual kernel contains
   exact zero weights, use
   `K = (1 - 1e-6) * w + 1e-6 * uniform` for the matched OT/no-OT
   comparison. Keep unsmoothed projected weights as the primary residual
   no-OT classifier and report smoothing separately.

### 3.2 Frozen-score allocation study

Fit the highlighted existing Ridge on inner folds 1–3 and predict inner-fold-0
selection rows. Compare its fixed kernel under no OT, relaxed OT, strict
balanced OT, and frozen-price routing. A batch-coupled inner-fold-0 result is
transductive: changing selection-batch composition may change its predictions.
Full-13, intercept-only, probability-mixture, top-2, and hard-argmax variants
are diagnostic comparisons, not candidate-selection families.

Frozen-price relaxed OT was eligible to advance only if it improved both BA
and Tail over the identical highlighted-Ridge no-OT kernel on inner fold 0.
The OT gate did not pass. Consequently, only the prespecified strength-1
residual-Ridge-plus-OT diagnostic was run; it had no selection eligibility.
Prices fitted to the original Ridge kernel were not transferred to a changed
kernel.

### 3.3 Residual Ridge study

The primary residual model regresses per-image oracle residuals from the four
confidence features. It reuses Task 3F-A training-only standardization,
multi-output Ridge, and class-weighting conventions where applicable. The
frozen grid was:

| Parameter | Values |
|:--|:--|
| Ridge alpha | `{0.1, 1, 10, 100, 1000}` |
| Fitting-class weighting gamma | `{0, 0.5, 1}` |
| Oracle anchor penalty | `{0.1, 1, 10}` |
| Residual scale | `{0.25, 0.5, 0.75, 1}` |

Fitting-class weights use a cap of `5` and mean-one normalization within
fitting rows. Selection uses weighted original logits. Existing
contribution-target Ridge, anchor-only, uniform, and frozen fixed references
remain controls. Full-13 features and margin-1.0 target sensitivity were
evaluated only at selected hyperparameters as secondary diagnostics.
Probability mixtures, top-2, hard routing, and batch-coupled OT remain
diagnostics.

## 4. Fit, selection, and outer-evaluation boundaries

1. **Development fit:** Only inner-folds-1–3 labels may build oracle targets,
   fit Ridge, standardize features, estimate class weights, or learn dual
   prices. Router inference may use expert logits and derived inference-time
   features, but no label, true class group, or fold ID.
2. **Development selection:** Evaluate predefined eligible candidates once on
   inner fold 0. A candidate must exceed that fold's uniform-logit baseline on
   both BA and Tail and must not be weakly dominated on both metrics by any
   frozen fixed reference. A tie with a reference is not a new Pareto point.
   Select by BA, then Tail, then no-OT, then stable configuration ID. Record
   every candidate and the decision immutably before outer data access.
   Disclose Task 3C's prior descriptive exposure to inner fold 0.
3. **Lock and refit:** After selection, freeze model family, target, grid
   choice, scale, OT strength/prior, and output semantics. Refit only the
   selected router using OOF rows from all four inner folds before outer
   evaluation. Inner-fold-0 labels enter final fitting only after selection.
   Freeze fitted parameters before loading outer evaluation artifacts. If an
   existing prediction artifact contains labels, the enforceable boundary is
   method lock before any outer-label-based analysis or selection; artificial
   prediction-before-label ordering is not required.
4. **Outer evaluation:** Evaluate the one frozen classifier once against
   uniform logits and frozen fixed references. Report BA, canonical
   Head/Medium/Tail macro recall, sample accuracy, paired differences, and
   class/sample hierarchical paired bootstrap intervals using a fixed seed
   and 10,000 replicates. Intervals describe uncertainty; point estimates
   determine the gate. Do not select settings from outer results.
5. **Expansion gate:** Run the full five-outer-fold, three-seed experiment only
   if the single outer candidate improves both BA and Tail over outer uniform
   and creates a new fixed-reference Pareto point. Success across the full
   experiment requires positive BA and Tail deltas for every configured seed
   after pooling its five outer folds. An OOF gain does not authorize use of
   the original balanced test set.

Every OOF prediction must come from an expert that excluded that image during
training. Use `FoldIntegrityValidator`, provenance hashes, final-epoch expert
checkpoints, and canonical `compute_class_groups`; do not duplicate
Head/Medium/Tail thresholds or fit a router on original full-data expert
predictions from their training images. Inner/outer expert training overlap
remains a statistical limitation, not direct row-wise label leakage.

## 5. Verification and artifact record

The implementation verification contract covered:

- OT marginals, non-convergence, extreme-score stability, determinism, no-OT
  equivalence, stronger prior matching as `rho` increases, and batch
  independence of frozen prices;
- oracle feasibility, zero residual for already-correct anchor rows, simplex
  projection, and rejection of invalid shapes, labels, and weights;
- fit/selection/outer role enforcement, no selection-label influence on
  fit-only models or prices, stable expert ordering, unchanged Task 3F-A
  reproduction, and immutable artifact writes;
- a focused CPU synthetic fit-to-weight-to-mixture-to-metric integration,
  checking shapes, finite values, and original-logit combination.

The focused unit, synthetic, integration, and protocol checks were recorded as
passed. No original test dataset was accessed for these checks. A tiny
OOF-pipeline smoke test was needed only if trainer integration changed; this
array-only study did not modify the expert trainer.

Use separate versioned artifacts for allocation, residual Ridge, and locked
outer evaluation. Include source hashes, fold roles, resolved configuration,
solver diagnostics, selected IDs, weights/predictions, baseline and candidate
metrics, and concise summaries. Use `ImmutableArtifactWriter`; do not overwrite
Task 3C/3F artifacts.

The v3 development artifacts are under `artifacts/oof/ridge_sinkhorn_v3/` in
the `allocation_v1`, `residual_v1`, `selection_v1`, `selected_diagnostics_v1`,
and `locked_outer_v1` directories. Earlier local v1 and v2 directories remain
implementation-verification records. Their selection arrays and selected
metrics match v3; v3 adds complete source and lock hashes and is the
outer-evaluation input. The locked outer results and paired predictions are
[`results.json`](../../artifacts/oof/ridge_sinkhorn_v3/outer_evaluation_v1/results.json)
and
[`predictions.npz`](../../artifacts/oof/ridge_sinkhorn_v3/outer_evaluation_v1/predictions.npz).

At implementation time on 2026-09-24, the machine had 12 CPU threads, about
15 GiB RAM, about 338 GiB free disk, and an RTX 3060 Laptop GPU with 6 GiB
VRAM. System Python lacked SciPy, scikit-learn, and PyTorch; `.venv/bin/python`
had them. Recheck resources before execution and prefer CPU for array analysis.
Short local runs do not establish model quality.

## 6. Reproduction and Kaggle commands

### 6.1 Reproduce the saved array analyses

From the repository root, the staged development study and locked outer report
can be validated or reproduced from the saved artifacts with:

```bash
./.venv/bin/python scripts/run_ridge_sinkhorn.py
./.venv/bin/python scripts/run_ridge_sinkhorn_outer.py
```

The second command consumes the already completed outer-fold expert artifacts.
It must not be used to tune or select another method. These commands do not
load the original CIFAR-100 test set. The locked evaluation expects the four
run directories under
`artifacts/oof/ridge_sinkhorn_outer_s78_o0/`.

### 6.2 Outer expert training commands used on Kaggle

The four seed-78 outer-fold-0 jobs below used the canonical training index
data, the same `artifacts/oof` root, and experiment ID
`ridge_sinkhorn_outer_s78_o0`. The 200-epoch recipes came from each config.
Omitting `--inner-fold` selects the reserved outer-evaluation role. All four
jobs have already completed; these commands document that run and would start
training again if invoked.

```bash
python scripts/run_oof.py --config configs/ce.yaml --expert ce --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/lal.yaml --expert logit_adjusted --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/balanced_softmax.yaml --expert balanced_softmax --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/mixup.yaml --expert mixup --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
```

The run directories include final checkpoints, prediction JSON, metadata, and
resolved configs. With all four in the shared artifact root, run the locked
evaluation from the repository root:

```bash
.venv/bin/python scripts/run_ridge_sinkhorn_outer.py
```

Each expert completed epoch 200 and supplied predictions for 2,170 outer
images. Their 8,677 training IDs and 2,170 prediction IDs were disjoint and
covered all 10,847 canonical long-tailed training images.

## 7. Completed results and interpretation

On the inner-fold-0 selection partition, the measured results were:

| Selection-fold method | BA | Tail accuracy |
|:--|--:|--:|
| Uniform original logits | 37.2941% | 10.8333% |
| Frozen `fixed_006` | 37.9474% | 13.0556% |
| Frozen `fixed_007` | 37.2401% | 14.7222% |
| Highlighted contribution Ridge, no OT | 38.0458% | 10.8333% |
| Highlighted Ridge, frozen OT price, `rho=10` | 37.9514% | 14.7222% |
| Highlighted Ridge, prior-only global bias | 38.4867% | 14.7222% |
| **Locked residual Ridge, no OT** | **38.0374%** | **13.0556%** |

None of the prespecified frozen-price strengths improved both BA and Tail
over the same highlighted-Ridge no-OT kernel, so the OT gate failed. The
strength-1 residual-plus-OT result was diagnostic only. The locked candidate
was `residual_p1_g0_a0.1_s1`: confidence-only features, hinge-oracle
penalty 1, Ridge alpha 0.1, class-weight gamma 0, residual scale 1, anchor
`fixed_007`, Euclidean simplex projection, and weighted original logits. Its
selection BA was 0.0901 percentage points above `fixed_006` with equal Tail.
The prior-only global bias control exceeded it on both metrics; the result
does not establish an adaptive-routing benefit beyond a global composition
change.

On the locked outer fold, the results were:

| Outer method | BA | Tail accuracy |
|:--|--:|--:|
| Uniform original logits | 42.7006% | 15.8333% |
| Locked residual Ridge | 44.7035% | 16.9444% |
| Frozen `fixed_007` | 44.7248% | 18.6111% |
| Frozen `fixed_010` | 44.7298% | 23.3333% |

The candidate's paired 95% bootstrap intervals versus uniform included zero.
Although its point estimates exceeded uniform on both metrics, `fixed_007` and
`fixed_010` each exceeded it on both. The candidate was therefore dominated,
the prespecified expansion gate failed, and no five-fold, three-seed training
was run. Outer fold 0 must not be used to choose a replacement method.
