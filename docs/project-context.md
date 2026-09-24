# Project Context — MoE Imbalance

> Current-state index. See [results.md](results.md) for the full-data benchmark,
> [oof-results.md](oof-results.md) for OOF evidence, and
> [nested-oof-protocol.md](nested-oof-protocol.md) for fold and data-role rules.

## Objective and current decision

The project studies whether inference-time combinations of differently trained experts can
improve both Balanced Accuracy (BA) and Tail accuracy over uniform logit averaging on
CIFAR-100-LT at imbalance ratio 100.

There are two distinct experimental tracks. The completed full-data track trains experts on all
10,847 long-tailed training images and reports historical results on the balanced CIFAR-100 test
set. The OOF track uses held-out predictions from folds of the long-tailed training population;
it has one locked seed-78, outer-fold-0 evaluation. Their metrics are not directly comparable.

The locked confidence-only residual Ridge candidate improved both metrics over outer uniform by
point estimate. Its paired intervals include zero, and frozen `fixed_007` and `fixed_010` exceeded
it on both metrics. The prespecified fixed-reference expansion gate failed. The five-fold,
three-seed expansion is not indicated for this candidate, and outer fold 0 cannot be reused for
method selection.

## Canonical protocol

- Dataset: CIFAR-100-LT with imbalance factor 0.01 (IR=100).
- Canonical long-tailed training population: 10,847 images, defined by
  `data/processed/lt_ir100_train_indices.npy`.
- The original full-data training protocol has no validation split and reports final-epoch
  checkpoints.
- Head, Medium, and Tail groups come only from
  `scripts/base_trainer.py::compute_class_groups`.
- The balanced CIFAR-100 test set is evaluation-only. It has been accessed for historical work
  and must not be used for OOF router fitting, method selection, or ordinary development.
- A method must improve both BA and Tail over the applicable uniform-logit baseline, with valid
  provenance and the required consistency. The locked outer study also required a new point on
  the frozen fixed-reference Pareto frontier.

| Group | Canonical training-count rule | Classes |
|:--|:--|--:|
| Head | `n >= 100` | 35 |
| Medium | `20 <= n < 100` | 35 |
| Tail | `n < 20` | 30 |

## Expert pool

All experts use ResNet-32 and the shared 200-epoch schedule. Full-data experiments use seeds 78,
88, and 1034; completed OOF artifacts use seed 78, outer fold 0, and the fixed expert order below.

| Expert | Objective |
|:--|:--|
| CE | Cross-entropy |
| LAL | Logit adjustment, `tau = 1` |
| BalancedSoftmax | Balanced softmax |
| Mixup | Mixup, `alpha = 1` |

## Milestones and current status

| Area | Status |
|:--|:--|
| Full-data four-expert experiments | Complete; historical test results recorded |
| Nested OOF framework and training pipeline | Complete |
| Task 3C aligned OOF collection | Complete: 16 inner expert runs for seed 78, outer fold 0 |
| Tasks 3D–3F analyses | Complete; exploratory or retrospective development evidence |
| Phase 1 correctness hardening | Complete: artifact I/O, fold validation, router contract, regression coverage |
| Ridge/Sinkhorn staged study | Complete through one locked outer evaluation; expansion gate failed |
| Full five-fold, three-seed OOF matrix | Not run: 300 expert runs |
| New specialized experts | Not implemented |

Follow-up OOF domain and CLI extraction and legacy quarantine remain planned. Phase 1 code changes
do not imply that canonical metrics were rerun.

## Current findings

- On the historical full-data test track, uniform logit averaging is stronger than the
  preregistered parameter-free routing rules. This does not establish that all fitted routing
  must fail.
- On the 6,507-image OOF development partition, fixed convex mixtures improved both BA and Tail
  over uniform. The label-dependent soft-mixture oracle shows feasibility headroom, not a
  deployable classifier.
- Confidence-only Ridge reached 36.55% BA / 7.38% Tail on that development partition. The
  matched full 13-feature result was lower at 36.10% / 6.55%, despite lower contribution MSE.
- Frozen-price Sinkhorn failed its development gate. In the selection analysis, a prior-only
  global-bias control also exceeded the selected residual Ridge on both metrics.
- The locked candidate was `residual_p1_g0_a0.1_s1`: confidence-only residual Ridge around
  `fixed_007`, refit on all four inner OOF folds, with no OT.

| Outer-fold-0 method | BA | Tail |
|:--|--:|--:|
| Uniform original logits | 42.7006% | 15.8333% |
| Locked residual Ridge | 44.7035% | 16.9444% |
| `fixed_007` | 44.7248% | 18.6111% |
| `fixed_010` | 44.7298% | 23.3333% |

The candidate's paired 95% bootstrap intervals versus uniform were [−0.4956, +4.5771] BA
points and [−5.0000, +7.7778] Tail points. Both include zero. The single result does not
establish generalization across folds, seeds, full-data experts, or a fresh test population.
Full provenance, references, and metrics are in [oof-results.md](oof-results.md).

## Experimental populations and data-use boundaries

| Population | Role and status |
|:--|:--|
| 10,847 canonical training images | Train full-data experts and define nested folds |
| Balanced CIFAR-100 test set | Historical full-data evaluation only; excluded from OOF development |
| Outer fold 0 training, 8,677 images | OOF expert-training population |
| Inner folds 1–3, 6,507 images | Router fitting and permitted development analyses |
| Inner fold 0, 2,170 images | Selection role; previously inspected descriptively in Task 3C |
| Outer fold 0 evaluation, 2,170 images | Consumed once by the locked study; unavailable for reselection |

Each OOF prediction is produced by an expert that excluded its image during training. Inner
experts share some training data across fit and selection folds, so OOF rows are not statistically
independent. OOF experts also differ from the full-data checkpoints.

## Code and artifact navigation

| Purpose | Source |
|:--|:--|
| Expert recipes | `configs/` |
| Fold membership and validation | `data/nested_oof.py` |
| OOF expert training and prediction collection | `scripts/oof_pipeline.py`, `scripts/run_oof.py` |
| Shared analysis artifact I/O | `scripts/analysis/artifacts.py` |
| Fixed-mixture and Ridge development analyses | `scripts/task3e_fixed.py`, `scripts/task3f_ridge.py` |
| Ridge/Sinkhorn implementation and entry points | `scripts/ridge_sinkhorn_*.py`, `scripts/run_ridge_sinkhorn*.py` |
| Router implementations | `scripts/router/` |
| Aligned Task 3C predictions | `artifacts/oof/task3c_oof/` |
| Ridge/Sinkhorn results | `artifacts/oof/ridge_sinkhorn_v3/` |
| Outer-fold-0 expert runs | `artifacts/oof/ridge_sinkhorn_outer_s78_o0/` |

The router API keeps contribution weights separate from class probabilities. `predict_proba`
returns expert-contribution weights. Evaluation and calibration use `predict_distribution`, which
must represent the same classifier used by `predict_class`.

## Documentation and authoritative records

- [README.md](../README.md): project overview, pipeline, key evidence, and reproduction pointers.
- [results.md](results.md): authoritative original full-data results.
- [oof-results.md](oof-results.md): authoritative OOF development and outer-evaluation results.
- [nested-oof-protocol.md](nested-oof-protocol.md): fold definitions, roles, and leakage rules.
- [problem.md](problem.md): established limitations and unresolved problems.
- [research.md](research.md): reviewed literature and current research direction.
- [ridge-sinkhorn-runbook.md](ridge-sinkhorn-runbook.md): the completed staged study and its
  artifacts, exact protocol, and reproduction instructions.
- [test-access-log.md](test-access-log.md): append-only history of original test-set access.
