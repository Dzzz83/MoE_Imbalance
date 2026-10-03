# Data and evaluation protocol

This document owns the project's data roles, metric definitions, evaluation
safeguards, and artifact expectations. Detailed outcomes belong in
[results.md](results.md), [oof-results.md](oof-results.md), and the
[completed full-matrix report](rs3-final-report.md). The exact v1 protocol,
candidate grid, tie-breaking order, and freeze rules are in the hash-bound
[PLAN.md](PLAN.md); do not edit that file.

## Dataset and experts

The training population is CIFAR-100-LT at imbalance ratio 100 (factor 0.01): 10,847
canonical training images defined by
`data/processed/lt_ir100_train_indices.npy`. The four ResNet-32 experts are
CE, LAL (`tau = 1`), BalancedSoftmax, and Mixup (`alpha = 1`). The historical
full-data benchmark uses a fixed training split generated with seed 42 and
expert seeds 78, 88, and 1034. It has no validation split and reports final
epoch checkpoints.

The balanced CIFAR-100 test set has 10,000 examples. It was read repeatedly
for historical full-data evaluations and is not an untouched confirmation or
ordinary development population.

## Canonical groups and metrics

Use `scripts.base_trainer.compute_class_groups` for all group boundaries.

| Group | Training examples per class | Number of classes |
|:--|:--|--:|
| Head | `n >= 100` | 35 |
| Medium | `20 <= n < 100` | 35 |
| Tail | `n < 20` | 30 |

Balanced accuracy (BA) is mean recall over all classes. Head, Medium, and Tail
accuracy are macro recall within their groups. Sample accuracy is the
ordinary fraction of correctly classified examples; it is not a macro
measure. Report all values as percentages unless a table says otherwise.

## Evidence populations

| Evidence | Population | Interpretation |
|:--|:--|:--|
| Full-data benchmark | Three seeds on the balanced test set | Historical comparison; the test set influenced previous research decisions. |
| Task 3C–3F development | Seed 78, outer fold 0; inner folds 1–3 (6,507 rows) | Development and diagnostic evidence; not independent test performance. |
| Locked outer evaluation | Seed 78, outer fold 0 (2,170 rows) | One consumed evaluation; cannot select a replacement method. |
| Completed v1 nested OOF | Three seeds, five outer folds each; 300 expert jobs | Retrospective evidence on the shared canonical population, not independent confirmation. |

In the completed v1 study, each seed's five outer folds cover the 10,847
canonical training rows. Within a fold, outer expert training excludes its
held-out fold. Inner expert training excludes its assigned prediction fold.
Candidate selection cross-fits over the four inner folds; final router states
are refit on all inner OOF rows, and outer predictions are loaded only after
the 15 fold configurations are locked.

The canonical fold algorithm is `numpy.default_rng.per_class_balanced_remainder.v1`,
implemented by `NestedOOFFoldManager` with fold-generation seed 42. The
fold-local exclusion rule does not make this population historically
untouched. Task 3C inspected inner-fold-0 labels descriptively, the earlier
seed-78 outer fold 0 was evaluated, and experts producing different inner
folds share training examples. The seeds and folds also share the canonical
population. Report these dependencies; do not treat the 15 seed/fold pairs as
independent replicates or claim transfer to a new population.

## Evaluation safeguards

- Every OOF prediction must come from an expert whose training membership
  excludes that image.
- Keep fitting, candidate selection, outer evaluation, and test evaluation in
  their declared roles. Full-data expert predictions on their training rows
  are not honest router supervision.
- Do not select methods, checkpoints, subsets, features, or settings from
  balanced-test results. Test readers use the append-only
  [access log](test-access-log.md).
- A method passes the v1 performance gate only if both BA and Tail exceed the
  uniform-logit baseline for every seed after pooling that seed's five folds.
  A separate fixed-mixture non-domination check is needed to claim an
  adaptive advantage. See the exact tie rule and bootstrap procedure in
  [PLAN.md](PLAN.md).
- Label-dependent hard-selection and soft-feasibility oracles are diagnostics,
  not inference methods. Lower contribution-target error alone does not show
  improved classification.
- Never generalize one fold, seed, or evidence population to another.

### Test-file access qualification

The v1 evaluation did not load or score original test examples or labels.
Constructing its fold manager with `CIFAR100(train=True)` caused the installed
torchvision integrity check to compute MD5 over both split files. Therefore,
the extracted `cifar-100-python/test` file was read for checksum verification
only; its pickle was not unpickled, and its examples and labels were not
loaded. The study lock's `test_accessed=false`
records that no test examples were evaluated and does not record this
checksum-only file read. Inner diagnostics read saved OOF artifacts and did
not load CIFAR data. The precise audit and scope are in the
[full-matrix report](rs3-final-report.md).

## Artifact and router contracts

OOF manifests define canonical sample IDs, fold assignments, and roles.
Prediction records retain IDs, labels, expert identity/order,
training-membership hashes, checkpoint/configuration hashes, and logits.
Readers reject duplicate, missing, extra, or out-of-role records and verify
fold membership, including that each prediction row was excluded from its
producer's training set, before scoring. Completed
manifests, locks, configurations, and prediction artifacts are immutable:
conflicts fail instead of overwriting existing evidence.

Router contribution weights and class probabilities are different quantities:
`predict_proba` returns nonnegative weights over E experts with each row summing
to one; `predict_distribution` returns probabilities over C classes, whose argmax
must match `predict_class`. Calibration metrics use class probabilities, never
expert weights. See [the routing mechanism record](../records/routing_mechanism.md).

### Diagnostic terms

- **Hard-selection oracle:** label-dependent share of rows where at least one
  expert's existing top-1 prediction is correct; not an inference method
  or a bound on soft mixtures.
- **Soft-mixture feasibility:** a label-dependent per-row simplex test of
  whether a convex logit mixture can give the true class a positive margin;
  it measures existence, not prediction or inference-time accuracy.
- **Contribution target:** each expert's marginal change to the uniform
  mixture's true-class log probability. A Ridge score predicts this target;
  it is not itself a weight or classification score.
- **Margin contribution:** retrospective, label-dependent difference between
  the true-class and strongest incorrect-class margins. A positive value need
  not change the top-1 prediction.

The completed seed-78/outer-fold-0 expansion gate failed, and the later v1
matrix is reported as retrospective evidence. Its completion does not restore
an independent evaluation population or overturn the earlier decision.
