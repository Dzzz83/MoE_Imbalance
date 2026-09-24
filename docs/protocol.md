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

The locked outer evaluation has consumed outer fold 0. It cannot be reused to
select, tune, or validate a replacement. The original balanced test set was
not loaded for that OOF evaluation. The full five-fold, three-seed OOF matrix
(300 expert runs) has not been run.

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
