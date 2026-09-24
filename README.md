# MoE Imbalance — Expert Routing on Long-Tailed CIFAR-100

This repository studies whether adaptive combinations of differently trained
experts can improve both Balanced Accuracy (BA) and Tail accuracy over uniform
logit averaging on CIFAR-100-LT with imbalance ratio 100.

## Current status

The project has two related experimental tracks:

| Track | Completed scope | Interpretation |
|:--|:--|:--|
| Original full-data experiments | Four experts, three seeds, final-epoch checkpoints, historical balanced-test evaluation and parameter-free routing baselines | Uniform logit averaging remains the historical baseline at 46.98 ± 0.69% BA and 18.76 ± 0.86% Tail |
| Nested OOF research | Frozen folds, seed-78/outer-fold-0 collection, expert diagnostics, fixed-weight and soft-mixture feasibility, Task 3F-A Ridge routing, and Tasks 3F-B–3F-F diagnostics | Development evidence of complementarity and partial target predictability; no validated new method |

All 16 Task 3C inner expert runs are complete. Task 3D is an exploratory
analysis of those diagnostics. Tasks 3E-A and 3E-B and the Task 3F-A–3F-F
analyses use only the 6,507-image router-fitting partition from inner folds
1–3. They do not use the reserved outer evaluation population or the original
CIFAR-100 test set.

Sinkhorn routing, new specialized experts, and a full multi-fold/multi-seed
nested evaluation are not implemented or complete. Tasks 3F-A–3F-F are
exploratory development analyses only; they do not authorize independent
evaluation. Documentation consolidation and Phase 1 correctness hardening are
complete for this handoff. The shared analysis artifact layer, fold-integrity
validation, explicit router distribution contract, and regression coverage are
now in the current worktree. Later OOF extraction, analysis-foundation
completion, legacy quarantine, new Ridge/Sinkhorn experiments, and an
independently frozen evaluation remain planned.

## Experimental protocol

The canonical long-tailed training population has 10,847 images, generated at
IR=100 from the committed split artifact
data/processed/lt_ir100_train_indices.npy. The original full-data track has no
validation split and reports final-epoch checkpoints.

Head, Medium and Tail are the canonical 35/35/30 class groups defined by
scripts/base_trainer.py::compute_class_groups:

| Group | Training-count rule | Classes |
|:--|:--|--:|
| Head | n >= 100 | 35 |
| Medium | 20 <= n < 100 | 35 |
| Tail | n < 20 | 30 |

The expert pool is:

| Expert | Objective |
|:--|:--|
| CE | Cross-entropy |
| LAL | Logit adjustment, tau = 1 |
| BalancedSoftmax | Balanced softmax |
| Mixup | Mixup, alpha = 1 |

The full-data runs use seeds 78, 88 and 1034. The completed OOF artifact uses
training seed 78, outer fold 0, fold-generation seed 42, and expert order CE,
LAL, BalancedSoftmax, Mixup.

## OOF findings

On the 6,507-image router-fitting partition, the uniform logit baseline is
35.9328% BA and 7.0094% Tail. Three of the 35 fixed convex logit candidates
improve both values:

| Candidate | Weights (CE, LAL, BalancedSoftmax, Mixup) | BA | Tail |
|:--|:--|--:|--:|
| fixed_006 | (0, 0.25, 0.25, 0.50) | 37.2134% | 9.9115% |
| fixed_007 | (0, 0.25, 0.50, 0.25) | 36.4403% | 14.7290% |
| fixed_010 | (0, 0.50, 0.25, 0.25) | 36.4236% | 14.5055% |

The label-dependent soft-mixture feasibility oracle reaches 55.5569% macro
feasibility and 26.3644% Tail feasibility on the same partition. This is an
existence result, not inference-time accuracy: it uses the true label to solve
an independent convex-logit feasibility problem for each image.

Task 3F-F found that the full 13-feature Ridge representation reduced
contribution MSE in all 15 matched settings (including all 15 Tail-MSE
comparisons), but its predefined primary classification result was lower than
confidence-only Ridge: 36.10% versus 36.55% BA and 6.55% versus 7.38% Tail.
No feature set or router was selected from this comparison.

See [docs/oof-results.md](docs/oof-results.md) for the complete provenance,
diagnostics, limitations and synthesis. These development values are not
directly comparable with the three-seed full-data test values.

## Repository map

~~~text
configs/                         expert run definitions
data/nested_oof.py               fold membership and provenance contract
scripts/oof_pipeline.py          fold training and OOF prediction collection
scripts/run_task3c.py            Task 3C collection entry point
scripts/expert_diagnostics.py    reusable aligned-logit diagnostics
scripts/task3e_fixed.py          Task 3E-A fixed-weight analysis
scripts/task3e_soft.py           Task 3E-B soft-feasibility analysis
scripts/task3f_ridge.py          Task 3F-A predictable Ridge analysis
scripts/run_task3f_ridge.py      Task 3F-A CLI entry point
scripts/task3f_mixup_diagnostics.py
                                  Task 3F-B Mixup-preference diagnostics
scripts/run_task3f_mixup_diagnostics.py
                                  Task 3F-B CLI entry point
scripts/task3f_tail_signal_diagnostics.py
                                  Task 3F-C Tail-signal diagnostics
scripts/run_task3f_tail_signal_diagnostics.py
                                  Task 3F-C CLI entry point
scripts/task3f_combined_signal_diagnostics.py
                                  Task 3F-D combined-signal diagnostics
scripts/run_task3f_combined_signal_diagnostics.py
                                  Task 3F-D CLI entry point
scripts/task3f_target_diagnostics.py
                                  Task 3F-E target diagnostics
scripts/run_task3f_target_diagnostics.py
                                  Task 3F-E CLI entry point
scripts/task3f_feature_comparison.py
                                  Task 3F-F feature comparison
scripts/run_task3f_feature_comparison.py
                                  Task 3F-F CLI entry point
scripts/analysis/artifacts.py    ArtifactReader and ImmutableArtifactWriter
data/nested_oof.py               FoldIntegrityValidator and OOF compatibility facade
scripts/router/                  parameter-free routers and distribution contract
artifacts/oof/task3c_oof/        aligned OOF data and diagnostics
artifacts/oof/task3e_fixed_feasibility/
                                  fixed-weight outputs
artifacts/oof/task3e_soft_feasibility/
                                  soft-oracle outputs
artifacts/oof/task3f_ridge/      Ridge configurations, predictions and report
artifacts/oof/task3f_mixup_diagnostics/
                                  Task 3F-B outputs
artifacts/oof/task3f_tail_signal_diagnostics/
                                  Task 3F-C outputs
artifacts/oof/task3f_combined_signal_diagnostics/
                                  Task 3F-D outputs
artifacts/oof/task3f_target_diagnostics/
                                  Task 3F-E outputs
artifacts/oof/task3f_feature_comparison/
                                  Task 3F-F outputs
checkpoints/                     original full-data checkpoints and reports
records/                         routing catalogue and frozen historical preregistration
~~~

The current router API keeps two probability concepts separate. `predict_proba`
returns expert-contribution weights. `predict_class` chooses a class using the
router's rule, and `predict_distribution` returns the class distribution from
that same classifier: the default hard-selected expert softmax, Uniform's
softmax of mean logits, ProbabilityAverage's mean softmax, or TTA's delegated
distribution. Evaluation and calibration must consume `predict_distribution`.

The shared `ArtifactReader` and `ImmutableArtifactWriter` live in
`scripts/analysis/artifacts.py`; `FoldIntegrityValidator` remains the
validation boundary in `data/nested_oof.py` while the later package extraction
is planned.

## Reproducing the saved analyses

Requirements are Python 3, the project dependencies, the canonical training
data and the committed long-tailed index artifact. The saved OOF artifacts are
already available; reproducing Task 3E-A or Task 3E-B does not require retraining
the experts.

~~~bash
./.venv/bin/python scripts/run_task3e_fixed.py \
    --data-root ./data \
    --oof-directory artifacts/oof/task3c_oof \
    --output-directory artifacts/oof/task3e_fixed_feasibility
~~~

~~~bash
./.venv/bin/python scripts/run_task3e_soft.py \
    --data-root ./data \
    --oof-directory artifacts/oof/task3c_oof \
    --fixed-results artifacts/oof/task3e_fixed_feasibility \
    --output-directory artifacts/oof/task3e_soft_feasibility
~~~

Both commands are analysis-only and validate the canonical data metadata and
OOF inputs. They do not read the original test set.

~~~bash
./.venv/bin/python scripts/run_task3f_ridge.py \
    --data-root ./data \
    --oof-directory artifacts/oof/task3c_oof \
    --output-directory artifacts/oof/task3f_ridge
~~~

Task 3F-A is CPU-only and uses only inner folds 1–3 of the existing aligned
OOF artifact. Its complete configuration table and caveated measurements are
in [docs/oof-results.md](docs/oof-results.md) and
[artifacts/oof/task3f_ridge/summary.md](artifacts/oof/task3f_ridge/summary.md).

~~~bash
./.venv/bin/python scripts/run_task3f_target_diagnostics.py \
    --data-root ./data \
    --oof-directory artifacts/oof/task3c_oof \
    --ridge-directory artifacts/oof/task3f_ridge \
    --output-directory artifacts/oof/task3f_target_diagnostics
~~~

Task 3F-E is read-only and retrospective: it recomputes contribution and
classification-margin diagnostics on the same permitted OOF rows without
refitting Ridge or selecting a new routing target.

The remaining Task 3F diagnostics are also read-only and consume the saved OOF
and Ridge artifacts. Run them in dependency order only when a fresh protected
diagnostic report is needed:

~~~bash
./.venv/bin/python scripts/run_task3f_mixup_diagnostics.py
./.venv/bin/python scripts/run_task3f_tail_signal_diagnostics.py
./.venv/bin/python scripts/run_task3f_combined_signal_diagnostics.py
./.venv/bin/python scripts/run_task3f_feature_comparison.py
~~~

Task 3F-F additionally reads the saved Task 3F-E output. These commands do not
train experts, fit Ridge, or access the reserved outer population or test set.

The original full-data training and historical evaluation commands remain
available:

~~~bash
./.venv/bin/python utils/create_lt_split.py
./.venv/bin/python scripts/train.py --config configs/ce.yaml --seed 78
./.venv/bin/python scripts/check_runs.py --seeds 78 88 1034
./.venv/bin/python scripts/evaluate_experts.py --seeds 78 88 1034
./.venv/bin/python scripts/analyze_subsets.py --seeds 78 88 1034
~~~

The last two commands access the already-examined balanced test set and append
to docs/test-access-log.md. They are historical full-data evaluations, not part
of ordinary router development.

For a cheap implementation check without training a reported model:

~~~bash
./.venv/bin/python scripts/train.py --config configs/ce.yaml \
    --epochs 1 --max-batches 2
~~~

## Documentation

- [docs/project-context.md](docs/project-context.md) — current objective, status, populations and next decision
- [docs/results.md](docs/results.md) — original full-data three-seed results
- [docs/oof-results.md](docs/oof-results.md) — completed OOF findings
- [docs/problem.md](docs/problem.md) — established limitations and unresolved problems
- [docs/research.md](docs/research.md) — literature and current research direction
- [docs/nested-oof-protocol.md](docs/nested-oof-protocol.md) — frozen fold and non-cheating rules
- [docs/expert-diagnostics.md](docs/expert-diagnostics.md) — reusable diagnostic framework
- [docs/code-audit-report.md](docs/code-audit-report.md) — verified defects, fixes and result impact
- [docs/refactor-plan.md](docs/refactor-plan.md) — phased refactor, extraction and quarantine plan
- [docs/archive/historical-results.md](docs/archive/historical-results.md) — superseded protocols and numbers
- [docs/archive/bugfix-report.md](docs/archive/bugfix-report.md) — audit history
- [docs/test-access-log.md](docs/test-access-log.md) — append-only original test-set access log

The complete routing inventory is in
[records/routing_mechanism.md](records/routing_mechanism.md), and the frozen
historical test candidate set is in
[records/routing-preregistration.md](records/routing-preregistration.md).
