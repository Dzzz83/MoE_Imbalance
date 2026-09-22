# Project Context — MoE Imbalance

> Primary current-state index for the repository. The original full-data
> measurements are in [results.md](results.md); completed OOF findings are in
> [oof-results.md](oof-results.md).

## 1. Objective and current decision

The project investigates adaptive combinations of experts for CIFAR-100-LT
classification at imbalance ratio 100. The original research question is
whether per-sample routing can improve both Balanced Accuracy (BA) and Tail
accuracy over uniform logit averaging.

The original full-data experiments remain a legitimate, completed track:
four experts were trained on all 10,847 long-tailed training images with seeds
78, 88 and 1034, and evaluated on the balanced CIFAR-100 test set. Their
historical uniform-logit baseline is **46.98 ± 0.69% BA** and
**18.76 ± 0.86% Tail**.

The current research direction is narrower: determine whether beneficial expert
contributions can be predicted from information available at inference time.
Nested OOF data now supplies held-out development predictions for that
question. Task 3F-A implements and evaluates the frozen Ridge feasibility
study; it is exploratory and not independently validated. Sinkhorn remains a
separate proposed direction.

## 2. Canonical data and metric protocol

- Dataset: CIFAR-100-LT with imbalance factor 0.01 (IR=100).
- Canonical long-tailed training population: 10,847 images from the committed
  data/processed/lt_ir100_train_indices.npy artifact.
- The original full-data track has no validation split and reports final-epoch
  checkpoints.
- Head/Medium/Tail is defined only by
  scripts/base_trainer.py::compute_class_groups: Head n >= 100 (35 classes),
  Medium 20 <= n < 100 (35 classes), Tail n < 20 (30 classes).
- The original balanced CIFAR-100 test set is evaluation-only and has already
  been accessed repeatedly. It must not be used for router fitting, method
  selection, or ordinary OOF development.
- A method succeeds only if it improves both BA and Tail over the applicable
  uniform-logit baseline, with the required provenance and consistency.

## 3. Existing expert pool

| Expert | Configuration |
|:--|:--|
| CE | Cross-entropy |
| LAL | Logit adjustment, tau = 1 |
| BalancedSoftmax | Balanced softmax |
| Mixup | Mixup with alpha = 1 |

The canonical full-data experiments use seeds 78, 88 and 1034. The completed
nested-OOF experiment uses expert-training seed 78, outer fold 0, and the fixed
order CE, LAL, BalancedSoftmax, Mixup.

## 4. Completed milestones

| Stage | Status |
|:--|:--|
| Original four-expert experiments | Complete |
| Evaluation hardening | Complete |
| Expert diagnostic framework | Complete |
| Nested OOF framework | Complete |
| OOF training pipeline | Complete |
| Task 3C: four-expert aligned OOF experiment | Complete |
| Task 3D: exploratory specialization analysis | Complete |
| Task 3E-A: fixed-weight feasibility | Complete |
| Task 3E-B: adaptive soft-mixture oracle | Complete |
| Task 3F-A: predictable Ridge routing feasibility | Complete; exploratory only |
| Sinkhorn routing | Not implemented |
| New specialized experts | Not implemented |
| Full nested-OOF evaluation across all folds and seeds | Not executed |

Task 3D is an interpretation of Task 3C diagnostics, not a separately
executed training or analysis pipeline.

## 5. Current scientific findings

The full-data results establish that uniform logit averaging is stronger than
the pre-registered parameter-free routing rules on the historical three-seed
test track. They do not establish that all fitted routing is impossible.

Task 3C shows complementary correct predictions in the held-out OOF
development population. LAL and BalancedSoftmax provide useful Tail coverage,
Mixup is strongest on Head in this population, and removing experts changes BA
and Tail in different directions. This is evidence of complementarity, not
evidence that a router can predict the useful contribution.

Task 3E-A shows that fixed convex logit weights can improve both BA and Tail
over the OOF uniform baseline on the router-fit partition. Task 3E-B shows
additional theoretical headroom for per-image convex logit mixtures. Both
analyses are label-dependent development results from one seed and one outer
fold; neither is independently validated or comparable directly with the
full-data test numbers.

The main unresolved scientific question is now whether any predictable signal
generalizes and yields paired BA–Tail improvement beyond fixed composition, not
the existence of any beneficial weight. Tail class coverage is still sparse,
and the soft-mixture oracle is an existence diagnostic rather than a deployable
classifier. Task 3F-A evaluated all 600 adaptive configurations on the
6,507-image permitted population. Uniform logit averaging reproduced **35.9328%
BA / 7.0094% Tail**. Sixty adaptive configurations improved both metrics over
that uniform row, but none exceeded the previously identified fixed-weight
references on both metrics. The best adaptive row reached **36.5502% BA / 7.3797%
Tail**. Its learned global-score control did not improve both metrics, so the
study does not establish useful adaptive routing beyond fixed composition.
Full 13-feature models reduced contribution-prediction error relative to the
confidence-only representation on average, without a corresponding
classification advantage.

## 6. Next research decision

Before any independent evaluation or Sinkhorn work, preserve the frozen Task
3F-A definitions and document:

1. the supervised target for useful expert contribution;
2. inference-time features and any logit calibration;
3. uniform logit, probability-average, fixed-weight, and no-OT learned-gate
   baselines;
4. the fit/selection/outer-evaluation procedure; and
5. safeguards against in-sample supervision, test-set selection, expert-pool
   changes, fold-trained/full-data distribution shift.

Task 3F-A is a simple predictive-model feasibility result, not approval of a
final router. Sinkhorn should be considered separately only after a suitability
signal is shown to be learnable and after its global-allocation value is
isolated.

## 7. Experimental populations

| Population | Role |
|:--|:--|
| Original 10,847 images | Train the canonical full-data experts |
| Original balanced CIFAR-100 test set | Historical full-data evaluation |
| Outer fold 0 training, 8,677 images | OOF development population |
| Inner folds 1–3, 6,507 images | Primary router-fitting analysis partition |
| Inner fold 0, 2,170 images | Router-selection partition in the frozen design; previously inspected descriptively |
| Outer fold 0 evaluation, 2,170 images | Reserved for a future frozen evaluation procedure |

OOF predictions are produced by experts that excluded the corresponding image
from their training population. The OOF experts are not the full-data
checkpoints.

## 8. Code and artifact map

- Fold membership and provenance: data/nested_oof.py
- OOF training and prediction collection: scripts/oof_pipeline.py,
  scripts/run_oof.py, scripts/run_task3c.py
- Reusable diagnostics: scripts/expert_diagnostics.py
- Fixed-weight analysis: scripts/task3e_fixed.py and
  scripts/run_task3e_fixed.py
- Soft-mixture feasibility analysis: scripts/task3e_soft.py and
  scripts/run_task3e_soft.py
- Predictable Ridge analysis: scripts/task3f_ridge.py and
  scripts/run_task3f_ridge.py
- Aligned OOF data and Task 3C diagnostics:
  artifacts/oof/task3c_oof/
- Task 3E-A outputs:
  artifacts/oof/task3e_fixed_feasibility/
- Task 3E-B outputs:
  artifacts/oof/task3e_soft_feasibility/
- Task 3F-A outputs:
  artifacts/oof/task3f_ridge/

## 9. Documentation navigation

- [README.md](../README.md) — public overview and reproducibility entry points
- [results.md](results.md) — original full-data three-seed results
- [oof-results.md](oof-results.md) — completed OOF findings
- [problem.md](problem.md) — established limitations and unresolved routing problems
- [research.md](research.md) — literature and current research boundary
- [nested-oof-protocol.md](nested-oof-protocol.md) — frozen fold and leakage protocol
- [expert-diagnostics.md](expert-diagnostics.md) — reusable diagnostic API
- [archive/historical-results.md](archive/historical-results.md) — superseded results
- [archive/bugfix-report.md](archive/bugfix-report.md) — audit history
- [test-access-log.md](test-access-log.md) — append-only test-set access audit
- [../records/routing_mechanism.md](../records/routing_mechanism.md) — complete routing catalogue
- [../records/routing-preregistration.md](../records/routing-preregistration.md) — frozen historical test candidate set
