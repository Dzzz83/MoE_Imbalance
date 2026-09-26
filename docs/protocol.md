# Data and Evaluation Protocol

This is the canonical description of the project's data roles, evaluation
rules, artifact checks, router interface, and diagnostic terms. Full-data
metrics belong in [results.md](results.md); OOF measurements belong in
[oof-results.md](oof-results.md). Executable fold rules live in
[`data/nested_oof.py`](../data/nested_oof.py) and the canonical class groups
in [`scripts/base_trainer.py`](../scripts/base_trainer.py).

## Dataset and expert pool

The dataset is CIFAR-100-LT at imbalance ratio 100 (imbalance factor 0.01).
The canonical long-tailed training population contains 10,847 images and is
defined by `data/processed/lt_ir100_train_indices.npy`. The four ResNet-32
experts are CE, LAL (`tau = 1`), BalancedSoftmax, and Mixup (`alpha = 1`).
Full-data runs share the fixed training split generated with seed 42.

There is no validation split in the original full-data track. It trains on all
10,847 images and reports final-epoch checkpoints. The balanced 10,000-image
CIFAR-100 test set is for evaluation only. It has been accessed repeatedly in
historical full-data work and is not an ordinary development or router
selection population.

## Canonical class groups and metrics

All Head/Medium/Tail boundaries must use
`scripts.base_trainer.compute_class_groups`; downstream code and reports
must not duplicate the thresholds.

| Group | Training count | Classes |
|:--|:--|--:|
| Head | `n >= 100` | 35 |
| Medium | `20 <= n < 100` | 35 |
| Tail | `n < 20` | 30 |

Balanced Accuracy (BA) is mean per-class recall. Head, Medium, and Tail
accuracy are also macro recall within each group. Sample accuracy is reported
separately and is not interchangeable with these macro metrics.

## Experimental tracks and data roles

The full-data and OOF tracks have different training and evaluation
populations. Their metrics are not directly comparable.

| Track or role | Population | Use and status |
|:--|--:|:--|
| Full-data training | 10,847 | Train each expert; seeds 78, 88, 1034; final epoch |
| Original balanced test set | 10,000 | Historical full-data evaluation only; already accessed |
| Outer-fold-0 expert training | 8,677 | Train the four experts used for the locked outer evaluation |
| Outer-fold-0 evaluation | 2,170 | Evaluated once for the locked candidate; consumed |
| Inner fold 0 | 2,170 | Candidate selection; descriptively inspected in Task 3C |
| Inner folds 1–3 | 6,507 | Primary router fitting and OOF development analyses |

The completed OOF collection uses expert seed 78 and outer fold 0. Five
outer folds are defined by the frozen manifest algorithm
`numpy.default_rng.per_class_balanced_remainder.v1`, with fold-generation
seed 42. Task 3C completed 16 inner expert runs for this seed and outer fold.
Each outer-training population is divided into four inner folds.
For outer fold 0, each inner expert is trained on the outer-training IDs
excluding its prediction fold (about 6,507–6,509 images). The four held-out
prediction folds cover the 8,677 outer-training images once.

| OOF role | Definition |
|:--|:--|
| Inner expert training | Outer-training IDs minus that expert's held-out inner fold |
| Inner OOF prediction | Held-out rows from one inner fold; the producing expert excluded each row |
| Router fitting | Rows from inner folds 1–3 in the staged study |
| Router selection | Inner fold 0 in the staged study |
| Outer evaluation | Outer-fold-0 held-out rows, loaded only after the router was locked |

Fit and selection rows are disjoint, but inner experts share training data
across these roles: a fit-row expert may have trained on selection-fold
images, and vice versa. OOF rows are therefore not statistically independent.
Task 3C also inspected inner-fold-0 labels descriptively before that fold was
used for the frozen candidate selection. Fold 0 is not an untouched
confirmation set.

The locked outer evaluation has consumed outer fold 0. It is unavailable as an
untouched confirmation set and must not be used to select or tune a replacement.
The distinct planned `ridge_sinkhorn_3seed_v1` study below includes fold 0 only
as a predeclared, fold-local component of a retrospective nested-OOF analysis,
after every v1 method lock is complete; this does not restore its independent
validation status. The original balanced test set was not loaded for that OOF
evaluation. The full five-fold, three-seed OOF matrix (300 expert runs) has
not been run.

## Evaluation hygiene and success criteria

- Every OOF prediction must come from an expert whose training membership
  excludes that image.
- Keep expert fitting, model selection, outer evaluation, and original-test
  evaluation in their declared roles. Full-data experts' predictions on their
  own training examples are not honest router supervision.
- Do not fit, tune, choose checkpoints, select subsets, or choose methods from
  the original balanced test results. Existing test readers use the
  append-only access audit at [`test-access-log.md`](test-access-log.md).
- A multi-seed method succeeds only when both BA and Tail accuracy exceed the
  applicable uniform-logit baseline, with the improvement direction
  consistent across configured seeds and valid provenance. For the completed
  Ridge/Sinkhorn study, expansion also required a new point on the frozen
  fixed-reference BA–Tail Pareto frontier.
- Do not generalize the one seed / one outer-fold result to other folds,
  seeds, full-data experts, or a fresh population.

## Planned study: `ridge_sinkhorn_3seed_v1`

This is a new, exploratory, retrospective nested-OOF study with its own frozen
protocol and artifact namespace, described in [PLAN.md](PLAN.md). It is not a
continuation or reproduction of the completed `ridge_sinkhorn_v3` study. That
study's prespecified expansion gate failed: its locked no-OT residual Ridge
candidate was exceeded by `fixed_007` and `fixed_010` on both BA and Tail, and
the tested frozen-price Sinkhorn did not pass its development gate. Those
results and the decision not to run that expansion remain unchanged. The new
study must not be described as satisfying or overturning that gate.

### Frozen population and roles

The v1 training seeds are `78`, `88`, and `1034`; the canonical fold-generation
seed is `42`. Use the existing
`NestedOOFFoldManager` and
`numpy.default_rng.per_class_balanced_remainder.v1` algorithm to form five
outer folds, each with four inner folds within its outer-training population.
For every seed/outer-fold pair, use experts in the fixed order CE, LAL,
BalancedSoftmax, Mixup (`ce`, `logit_adjusted`, `balanced_softmax`, `mixup` in
run metadata), train for 200 epochs, and use final-epoch checkpoints. The full
inventory is 240 inner expert jobs plus 60 outer expert jobs, or 300 jobs
before any exact-compatible historical reuse is counted.

| v1 role | Membership and use |
|:--|:--|
| Outer expert training | The canonical training population excluding that outer fold; predict the excluded outer-fold rows. |
| Inner expert training | The outer-training population excluding that expert's assigned inner prediction fold. |
| Inner router cross-fit | For each held-out inner fold, fit candidate routers on the other three inner folds and predict this fold; pool the four held-out predictions for candidate selection. Validation labels may score candidates but are not passed to router fitting. |
| Final router refit | After inner-only selection, refit the selected router on all four inner OOF folds; learn any frozen Sinkhorn prices from those inner rows only. |
| Outer evaluation | Load the outer predictions only after all 15 seed/outer-fold configurations have valid immutable locks. Evaluate each locked method once; outer labels cannot fit, select, tune, or revise it. |
| Original balanced test set | Out of scope. Do not load it, use its results, or append an access-log entry for v1. |

For each outer fold, the exclusion guarantee is fold-local: the evaluated
images and labels are excluded from that fold's expert training, router fitting,
and selection. It does not make the canonical population historically
untouched. Task 3C descriptively inspected inner-fold-0 labels, and outer fold 0
has already been evaluated for the completed study. All v1 findings must
therefore be labelled retrospective nested-OOF evidence, not independent
confirmation. The original balanced test set has been accessed in historical
full-data work, but is forbidden to this study.

Inner cross-fitting prevents a row's label from directly supervising the
router that predicts that row during candidate selection. It does not make the
inner predictions statistically independent: experts predicting one inner
fold may have trained on images assigned to other inner folds. The five outer
folds and three seeds also share the canonical population. Report these
dependencies with the results; paired bootstrap intervals are conditional on
the trained models and do not measure retraining variability.

### Frozen methods and selection

The primary methods are contribution Ridge; contribution Ridge plus
frozen-price Sinkhorn; residual Ridge around
`fixed_007 = (0, 0.25, 0.50, 0.25)`; that residual Ridge plus frozen-price
Sinkhorn; and selective residual Ridge plus Sinkhorn. The selective method
uses only the row's distance from the anchor:

```text
d_i = ||w_ridge_i - anchor||_1
g_i = min(1, d_i / tau)
w_i = (1 - g_i) * w_ridge_i + g_i * w_sinkhorn_i
```

Freeze the following reduced search space; add no candidates:

| Component | Values |
|:--|:--|
| Contribution Ridge alpha / class-weight gamma | `{0.1, 10, 1000}` / `{0, 1}` |
| Contribution score temperature / shrinkage | `{1, 2}` / `{0.5, 0.75, 1}` |
| Residual oracle penalty / Ridge alpha / class-weight gamma | `{0.1, 1, 10}` / `{0.1, 10, 1000}` / `{0, 1}` |
| Residual scale | `{0.5, 0.75, 1}` |
| Sinkhorn rho / prior | `{0.1, 1, 10}` / uniform and smoothed `fixed_006`, `fixed_007`, `fixed_010`, `fixed_011` |
| Selective gate tau | `{0.1, 0.25, 0.5, 1}` |

Smooth each fixed Sinkhorn prior using `q = 0.95*w + 0.05*uniform`. Preserve
the existing deterministic log-domain relaxed-Sinkhorn objective; fit dual
prices using inner rows only, record convergence diagnostics, and reject
non-convergence. Apply frozen prices independently to each future row, so
inference is batch-independent.

Select contribution-Ridge and residual-Ridge base settings separately from
their four-fold cross-fitted inner predictions. The Sinkhorn variant of each
Ridge must reuse that selected Ridge kernel and search only prior/rho; the
selective variant must reuse the selected residual and Sinkhorn settings and
search only tau. Select by maximizing
`min(Delta BA, Delta Tail)` against inner uniform logits, then break ties by
BA, Tail, stronger regularization, less routing/OT, and stable configuration
ID, in that order. Apply each tie break lexicographically: contribution-base
regularization prefers larger Ridge alpha; residual-base regularization
prefers larger Ridge alpha, then larger oracle penalty. Less routing/OT
prefers smaller contribution shrinkage, then larger contribution temperature,
smaller residual scale, smaller Sinkhorn rho, and larger selective tau, using
only applicable fields. Gamma and prior identity are not regularization
criteria and fall through to stable configuration ID. The prior-only
global-bias control uses the same inner-only selection rule. Freeze this
ordering before creating the study lock. No outer result may compare or
select method families.

Mandatory controls are uniform logits and probabilities; `fixed_006`,
`fixed_007`, `fixed_010`, and `fixed_011`; the residual anchor without
adaptation; and prior-only global bias over the frozen prior grid. Report BA,
Head, Medium, Tail, and sample accuracy as distinct measures, using canonical
`compute_class_groups`. Label-dependent oracle weights are diagnostics, never
methods.

For this planned study, success requires positive pooled BA and Tail deltas
over uniform logits for every seed after concatenating its five outer folds.
An adaptive advantage additionally requires that no fixed mixture dominate
the adaptive method on both BA and Tail. Report fold and seed results, mean and
standard deviation over seeds, paired deltas, and hierarchical paired
bootstrap intervals using 10,000 stratified class-then-sample resamples with
seed `20260924`. These are prespecified criteria, not measured outcomes.

### Locks, artifact integrity, and historical reuse

Before reading any new outer predictions, freeze an immutable study lock that
records the study ID, hash of `PLAN.md`, hash of the resolved study
configuration, source commit, canonical job inventory, and deterministic shard
assignments. Complete and validate all 15 inner-selection locks before loading
outer predictions. Each fold lock records selected settings, fitted
parameters, source/configuration hashes, and fit/selection memberships. The
outer evaluation and reports consume only a validated artifact view of the
locked methods and the complete prediction matrix.

Keep artifacts under `<run_root>/ridge_sinkhorn_3seed_v1/`, where `run_root` is
selected by the runtime profile. Existing
artifacts are immutable: accept an existing run only after checking canonical
job identity, protocol and fold membership, resolved configuration,
final-epoch checkpoint, prediction IDs and order, schema, and content hashes.
A partial or incompatible immutable artifact is an error and must not be
overwritten. Validate every expert run, all 15 fold locks, and every expected
sample before aggregate evaluation or reporting. Hashes and exact memberships
must be retained in the study records.

The only historical reuse candidates are the 16 `task3c_oof` inner runs and
four `ridge_sinkhorn_outer_s78_o0` outer runs for seed 78 / outer fold 0. A
compatibility record must identify each target job and source experiment and
run-relative path, expert, seed, fold role, training- and
prediction-membership hashes, resolved-config hash, final epoch, checkpoint
hash, and prediction hash. Reuse only exact matches; a failed compatibility
row is a missing new job. Historical sources stay immutable and
reference-only under named, read-only reuse roots. Portable bundles contain
native v1 payloads under their exact `OOFArtifactStore` run-relative paths;
historical references carry expected hashes and source identity, not copied
payloads. Relocation validates paths relative to the run directory and
content hashes; machine-specific absolute path strings are provenance only.

Portable bundles use a versioned canonical JSON manifest as the first member
of a deterministic standard-library tar archive. Validate every path, schema,
size, and SHA-256 before import or merge; reject links, traversal, identity
conflicts, and differing bytes for a job ID, and preserve files already
present. Retain the job inventory, shard assignment, source/config hashes, and
membership provenance across sessions. These checks do not authorize test-set
access.

## Router distribution contract

Router APIs distinguish expert contribution weights from class probabilities:

- `predict_proba` returns nonnegative expert-contribution weights, with one
  row per image and one column per expert; each row sums to one.
- `predict_distribution` returns the class distribution used by the same
  classifier as `predict_class`.

Calibration and class-probability metrics must use
`predict_distribution`, never expert weights. Every router's class
distribution must have an argmax consistent with `predict_class`.

## Artifact integrity and provenance

The OOF manifest records canonical sample IDs, fold assignments, roles,
algorithm, and seed. Each prediction record carries its sample ID, label,
fold, expert identity and order, training-membership hash, checkpoint hash,
resolved configuration, and logits. Validation recomputes counts and
memberships from canonical labels and fold IDs, rejects duplicate, missing,
extra, or out-of-role records, and verifies that each predicted sample was
excluded from the producing model's training membership.

Manifests, resolved configurations, and completed predictions are immutable:
an existing artifact is accepted only when its content and static metadata
match; incompatible output fails instead of overwriting evidence. Mutable
`run_metadata.json` state transitions are written atomically. Model,
configuration, array, decision, and source hashes are retained with results.
Generated summaries stay beside their machine-readable artifacts.

## Diagnostic definitions

The reusable array-only implementation is
[`scripts/expert_diagnostics.py`](../scripts/expert_diagnostics.py). It
accepts aligned logits shaped `(N, E, C)` or predictions shaped `(N, E)`;
it does not load data or checkpoints and does not fit router parameters.

- **Agreement and correctness:** agreement patterns count matching top-1
  predictions; correctness counts compare each prediction with its label.
  Exclusive correctness identifies rows where exactly one expert is right.
- **Hard-selection oracle:** uses labels to count rows where at least one
  expert's existing top-1 prediction is correct. It is an unattainable
  label-dependent reference for selecting among those answers, not a router
  and not a bound on soft logit mixtures.
- **Fixed-weight ensemble:** applies one declared expert-weight vector to
  every row, then predicts from weighted logits or weighted probabilities.
  Selecting weights on a development population does not establish
  independent generalization.
- **Soft-mixture feasibility oracle:** Task 3E-B solves a label-dependent
  per-image simplex problem to test whether any convex weighted-logit
  combination gives the true class a positive margin. It measures existence,
  not prediction or inference-time accuracy.
- **Contribution target and score:** the Task 3F-A target measures each
  expert's marginal change to the uniform ensemble's true-class log
  probability. A Ridge score predicts that target from inference-time
  features; it is not itself a classifier, mixture weight, or accuracy.
- **Margin contribution:** Task 3F-E compares true-class and strongest
  incorrect-class margins. It is retrospective and label-dependent; a
  positive contribution need not change the final top-1 class.

These diagnostics can describe complementarity and feasibility. They do not
establish that a future router can predict useful weights.
