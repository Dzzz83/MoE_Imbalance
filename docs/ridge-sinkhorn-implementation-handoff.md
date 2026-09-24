# Ridge and Sinkhorn implementation handoff

> Agreed research and implementation plan from the 2026-09-24 planning chat.
> **Planning only:** no Ridge/Sinkhorn code, experiment, outer-fold evaluation,
> or test-set evaluation was run in that chat. This document records intent for
> the next implementation chat; results must not be described as established
> until the stated protocol is executed.

## 1. What the next chat is doing

Implement a staged study of whether Ridge can predict useful per-image expert
weights and whether Sinkhorn-style allocation adds value to those weights.
First isolate allocation using the **existing** Task 3F-A Ridge score source;
then investigate a classification-aligned residual Ridge target. Only one
locked, batch-independent candidate may advance to the reserved outer-fold-0
evaluation. The user will run expensive expert training on Kaggle and return
the artifacts. Array analyses, unit tests, synthetic checks, and development
selection may run locally when resources permit.

The next chat should use **GPT-6 Sol, High** as lead planner, integrator, and
reviewer. Delegate bounded, non-overlapping coding tasks to **GPT-6 Luna, Max**
subagents when that model override is available. The lead owns the scientific
protocol, reviews every change, resolves shared-file conflicts, and performs
final verification. Do not start another broad refactor as a prerequisite.

Before coding, the lead should inspect the current implementation and review
the mathematical details below. Record any correction explicitly, preserve the
scientific comparisons, and do not silently expand the candidate search.

## 2. Current evidence and repository entry points

- [Project context](project-context.md), [nested OOF protocol](nested-oof-protocol.md),
  [problem statement](problem.md), [research status](research.md), and
  [OOF results](oof-results.md) are the current sources. The ranked Ridge/OT
  ideas in [the archive](archive/superseded-research-proposals.md) are historical
  hypotheses, not implementation instructions.
- The original full-data expert pool is CE, LAL, BalancedSoftmax, Mixup,
  trained on all 10,847 CIFAR-100-LT images with seeds 78, 88, 1034. The
  historical balanced-test uniform-logit result is 46.98% BA / 18.76% Tail
  averaged across seeds. Those test numbers are **not directly comparable**
  with the OOF development results.
- The completed OOF collection uses expert seed 78, outer fold 0, and four
  aligned inner folds. Inner folds 1–3 contain 6,507 router-fit images; inner
  fold 0 has 2,170 router-selection images; the 2,170 outer-fold-0 images are
  reserved. Inner fold 0 was inspected descriptively in Task 3C, so it is not
  an untouched confirmation set. No outer expert has been trained yet.
- Existing [Ridge code](../scripts/task3f_ridge.py) and
  [tests](../tests/test_task3f_ridge.py) implement multi-output Ridge,
  inference-time confidence/full-13 features, the label-dependent uniform
  true-class log-probability contribution target, class weighting, three-fold
  held-out prediction, softmax score-to-weight conversion, and shrinkage.
  [Task 3F-A artifacts](../artifacts/oof/task3f_ridge/) are immutable controls.
- The current Ridge reaches 36.5502% BA / 7.3797% Tail on the permitted
  6,507-row development partition versus 35.9328% / 7.0094% for uniform.
  It does not beat the strong fixed-weight references on both metrics. The
  highlighted model gives Mixup highest weight on nearly every row. Full-13
  features lower contribution MSE but worsen the primary BA/Tail result.
- Relevant code: [restricted OOF loading and fixed references](../scripts/task3e_fixed.py),
  [fold validation](../data/nested_oof.py),
  [immutable artifact I/O](../scripts/analysis/artifacts.py), and
  [OOF expert runner](../scripts/run_oof.py). The existing restricted analysis
  loader intentionally rejects inner fold 0; selection needs a separate,
  role-checked path rather than weakening that guard.

## 3. Frozen scientific decisions

| Decision | Agreed choice |
|:--|:--|
| Study sequence | Isolate allocation on frozen Ridge scores, then redesign Ridge; avoid a joint unrestricted search. |
| Inference | Study batch-coupled OT diagnostically, but require batch-independent prediction for the candidate evaluated on the outer fold. |
| Capacity | Relaxed expert mass is primary; exact 25/25/25/25 Sinkhorn is a control because all four experts already run and have unequal quality. |
| OT prior | `q = 0.95 * fixed_007 + 0.05 * uniform = [0.0125, 0.25, 0.4875, 0.25]`. The earlier idea of using the mean training no-OT weights was rejected: those same rows already satisfy that marginal, so fitted prices would be an identity correction. |
| Ensemble | Weighted **original logits**, soft all-four routing, primary; weighted probabilities, top-2, and hard argmax diagnostic. |
| Primary frozen score | Existing highlighted confidence-only Ridge: alpha `1000`, gamma `1`, score temperature `2`, shrinkage `0.75`. Matched full-13 and training-only global scores are diagnostics. |
| Residual anchor | `fixed_007 = [0, 0.25, 0.50, 0.25]`; current OOF fit-partition reference is 36.4403% BA / 14.7290% Tail. |
| Residual target | Zero-margin, anchor-regularized multiclass hinge oracle weights minus the anchor. Correct anchor predictions have zero oracle residual. |
| Residual features | Four expert confidences primary; full-13 secondary. No separate learned route/stay gate initially. |
| Selection | One candidate, highest BA among joint BA/Tail improvements that are new fixed-reference Pareto points; Tail, no-OT, then stable ID break exact ties. |
| Endpoint | One locked outer-fold-0 evaluation after development and local verification gates; full five-fold/three-seed work only if its prespecified gate passes. |

The fixed references are the existing `fixed_006`, `fixed_007`, `fixed_010`,
`fixed_011`, uniform logit, uniform probability, and uniform without CE.
`fixed_006` is the BA-strong reference (37.2134% BA / 9.9115% Tail on the
router-fit partition); `fixed_007` is the selected residual anchor. Do not
promote a retrospectively chosen alternative reference.

## 4. Implementation stages and numerical contracts

### A. Shared array-only components

Implement validated NumPy/SciPy APIs with fixed expert order and finite,
nonnegative, row-stochastic weight outputs:

1. **Relaxed OT:** for strictly positive row kernels `K`, optimize over
   row-stochastic `P`:

   `mean_i KL(P_i || K_i) + rho * KL(mean_i P_i || q)`.

   This normalization makes `rho` independent of the batch size. Use
   `rho = {0.1, 1, 10}`, deterministic log-domain updates, explicit
   convergence/residual diagnostics, and failure on non-convergence. At
   `rho = 0`, the output is the original kernel. Compare the same input
   kernel before and after OT.
2. **Balanced Sinkhorn control:** enforce each row sum of one and exact
   uniform column-average mass `[0.25] * 4`; validate both marginals.
3. **Frozen dual prices:** learn four expert multipliers from fit rows only,
   then apply their softmax-normalized product with each future row kernel
   independently. These are an **OT-derived global bias**, not live batch
   coordination. Include an explicit prior/global-bias control so a global
   composition change is not credited to coordination.
4. **Oracle target:** for each labeled fitting row and anchor `w0`, solve
   `min_(w in simplex, t >= 0) t + (lambda/2)||w-w0||²`, subject to
   `t >= sum_e w_e (z[e,c] - z[e,y])` for every `c != y`. Use deterministic
   SciPy constrained optimization and check objective, simplex, slack, and
   feasibility residuals. Cache targets by oracle penalty; they do not depend
   on Ridge alpha/gamma or prediction scale.
5. **Residual weight map:** Euclidean-project `w0 + scale * predicted_residual`
   onto the four-expert simplex. For a residual kernel containing exact zero
   weights, the matched OT/no-OT comparison uses the same strictly positive
   kernel `K = (1 - 1e-6) * w + 1e-6 * uniform`; retain unsmoothed projected
   weights as the primary residual no-OT classifier and report smoothing as a
   separate numerical control.

Do not modify Task 3F-A functions or artifacts to make the new method appear
to reproduce an old result. Use separate versioned modules/artifact schemas.

### B. Frozen-score allocation study

Fit the highlighted existing Ridge on inner folds 1–3 and predict the
selection rows in inner fold 0. Compare its fixed kernel under no OT,
relaxed OT, strict balanced OT, and frozen-price routing. A full inner-fold-0
batch-coupled result is a **transductive diagnostic only**; changing the
selection batch composition may change those predictions. Diagnose matched
full-13 and intercept-only scores, probability mixtures, top-2 projection,
and hard argmax without making them candidate-selection families.

Frozen-price relaxed OT earns eligibility for the residual-Ridge stage only
if it improves **both** BA and Tail over the identical highlighted-Ridge
no-OT kernel on inner fold 0. If it does not, run at most one prespecified
strength-1 residual-Ridge+OT diagnostic, with no selection eligibility. If
it does, carry only the selected `rho` and prior into Stage C; **refit prices
on the residual-Ridge fit-row kernel**. Prices fitted to the old Ridge kernel
are not transferable to a changed kernel.

### C. Residual Ridge study

Regress the per-image oracle residual from the four confidence features.
Reuse the Task 3F-A training-only standardization, multi-output Ridge, and
class-weighting convention where applicable. The primary grid is:

- Ridge alpha: `{0.1, 1, 10, 100, 1000}`;
- fitting-class weighting gamma: `{0, 0.5, 1}` with the existing cap of `5`
  and mean-one normalization within fit rows;
- oracle anchor penalty: `{0.1, 1, 10}`;
- residual scale: `{0.25, 0.5, 0.75, 1}`.

Select using weighted original logits. The existing contribution-target
Ridge, anchor-only, uniform, and frozen fixed references remain controls.
At the selected hyperparameters only, run full-13 features and margin `1.0`
target sensitivity as secondary diagnostics. Keep probability mixtures,
top-2, hard routing, and batch-coupled OT diagnostic.

## 5. Fit, selection, and evaluation boundaries

1. **Development fit:** only inner folds 1–3 labels may build oracle targets,
   fit Ridge, standardize features, estimate class weights, or learn dual
   prices. Inputs to the router at inference are expert logits and derived
   inference-time features; no label, true class group, or fold ID is an
   inference feature.
2. **Development selection:** evaluate predefined eligible candidates once
   on inner fold 0. A candidate qualifies only when both BA and Tail exceed
   that fold's uniform-logit baseline and **none** of the frozen fixed-weight
   references weakly dominates it in both metrics. A tie with a fixed
   reference is not a new Pareto point. Select the highest BA, then highest
   Tail, then no-OT, then lexicographically stable configuration ID. Record
   every candidate and the selection decision immutably before accessing
   outer data. Disclose Task 3C's earlier descriptive inner-fold-0 exposure.
3. **Outer lock and refit:** after selection, freeze model family, target,
   grid choice, scale, OT strength/prior, and output semantics. It is allowed
   to refit that one selected router on OOF rows from **all four** inner folds
   before outer evaluation. This uses inner-fold-0 labels for final fitting
   only after the method is selected. Freeze all fitted parameters before
   loading outer evaluation artifacts. Do not require an artificial “save
   predictions before reading labels” ordering if the existing prediction
   artifact contains labels; the enforceable boundary is method lock before
   **any** outer-label-based analysis or selection.
4. **Kaggle handoff:** after relevant local unit, synthetic, and integration
   checks pass, report their outcomes and give the user exact four
   `scripts/run_oof.py` commands for seed-78 outer fold 0, using
   `configs/{ce,lal,balanced_softmax,mixup}.yaml`, matching expert keys,
   one dedicated experiment ID, `--device cuda --execute-full`, and no
   `--inner-fold`. The user runs those full 200-epoch expert jobs on Kaggle
   and returns artifacts. No full training is authorized in the local chat.
5. **Outer evaluation:** evaluate the one frozen classifier once on the
   reserved outer fold against uniform and frozen fixed references. Report
   BA, canonical Head/Medium/Tail macro recall, sample accuracy, paired
   differences, and class/sample hierarchical paired bootstrap intervals
   using a fixed seed and 10,000 replicates. Intervals describe uncertainty;
   the gate uses point estimates. No new setting is selected from outer
   outcomes.
6. **Expansion:** proceed to the full five-outer-fold, three-seed nested
   experiment only if the single outer candidate improves both BA and Tail
   over the outer uniform baseline **and** creates a new fixed-reference
   Pareto point. The user controls heavy Kaggle training. Final nested-OOF
   success requires positive BA and Tail deltas for every configured seed
   after pooling its five outer folds. The original balanced CIFAR-100 test
   set remains excluded from this method search and is not automatically
   authorized by an OOF gain.

All OOF predictions must come from an expert that excluded their image during
training. Use `FoldIntegrityValidator`, provenance hashes, final-epoch expert
checkpoints, and canonical `compute_class_groups`; never duplicate the
Head/Medium/Tail thresholds or fit a router on original full-data expert
predictions from their own training images. The existing inner/outer expert
training overlap is a stated statistical limitation, not direct row-wise
label leakage.

## 6. Verification, artifacts, and local resources

- Unit tests should cover OT marginals, non-convergence, extreme score
  stability, determinism, no-OT equivalence, monotonically stronger prior
  matching as `rho` increases, batch independence of frozen prices, oracle
  feasibility and zero residual on already-correct anchor rows, projection,
  and invalid shapes/labels/weights.
- Protocol tests should verify fit/selection/outer role enforcement, no
  influence of selection labels on fit-only models/prices, stable expert
  ordering, unchanged Task 3F-A reproduction, and immutable artifact writes.
  Use synthetic fixtures; do not access the original test dataset for checks.
- Run a focused CPU end-to-end synthetic fit → weight → mixture → metric
  integration. Check array shapes, finiteness, and original-logit combination.
  Run only a tiny OOF-pipeline smoke test if trainer integration changes;
  this array-only study should not modify the expert trainer.
- Write separate versioned artifacts for allocation, residual Ridge, and
  locked outer evaluation. Include source hashes, fold roles, full resolved
  configuration, solver diagnostics, selected IDs, weights/predictions,
  baseline and candidate metrics, and concise summaries. Use the existing
  `ImmutableArtifactWriter` rather than overwriting Task 3C/3F artifacts.
- As checked on 2026-09-24, this machine has 12 CPU threads, about 15 GiB
  total RAM (about 6.7 GiB then available), about 338 GiB free disk, and an
  RTX 3060 Laptop GPU with 6 GiB VRAM. The shell's system Python lacks
  SciPy/scikit-learn/PyTorch; `.venv/bin/python` has NumPy, SciPy,
  scikit-learn, and PyTorch. Recheck resources at execution time. Prefer CPU
  for array analysis. Local short runs do not establish model quality.

## 7. Handoff status

This chat produced the plan and this document only. No implementation,
diagnostic rerun, method selection, outer-fold training, Kaggle execution,
full nested experiment, or test evaluation has occurred. The next chat should
begin by reading this file, `AGENTS.md`, the current protocol/evidence docs,
and the relevant code; then implement the staged study with Sol reviewing
bounded Luna coding tasks. Report deviations from this document explicitly.
