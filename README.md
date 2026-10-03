# Expert Routing on Long-Tailed CIFAR-100

## Contents

- [Overview](#overview)
- [Problem and approach](#problem-and-approach)
- [Metrics](#metrics)
- [Current results](#current-results)
- [Uncertainty and evidence limits](#uncertainty-and-evidence-limits)
- [Quick start](#quick-start)
- [Reading map](#reading-map)
- [Key references](#key-references)

## Overview

This project asks whether a router can combine four long-tail CIFAR-100
classifiers to improve rare-class recognition. Each classifier is a ResNet-32
trained with a different objective: cross-entropy (CE), logit-adjusted loss
(LAL), BalancedSoftmax (BS), or Mixup.

The completed `ridge_sinkhorn_3seed_v1` study contains 300 expert jobs and
covers three training seeds. It found no all-seed adaptive improvement over
uniform logits. Its results are retrospective nested-OOF evidence on a shared
population, not independent confirmation.

## Problem and approach

CIFAR-100-LT has 10,847 canonical training images at imbalance ratio 100
(imbalance factor 0.01). Frequent classes dominate the sample count while
rare classes have few training examples, so overall accuracy can hide poor
Tail recognition. The routing question is whether image-level features can
identify useful expert weights without using the true label at prediction time.

The study uses nested out-of-fold (OOF) predictions so a router is trained and
selected on predictions from experts that excluded those images:

```text
3 expert-training seeds × 5 outer folds
  each outer fold: 4 inner folds for router cross-fitting and selection
  lock all 15 seed/fold configurations before outer evaluation
  pool 5 held-out folds within each seed, then compare the 3 seed results
```

The splits prevent row leakage, but prior development inspected inner-fold-0
labels and seed-78 / outer-fold-0 results; all seeds reuse the same population.
See [evidence limits](#uncertainty-and-evidence-limits).

Each image has four expert-logit vectors. A uniform logit mean is the reference;
fixed convex weights provide comparisons. A regularized linear Ridge regressor
predicts expert contributions or residual adjustments from inference-time
features. The project adapts entropy-regularized transport to relaxed allocation:
prices are fitted internally and held fixed at inference, and the selective
variant gates a residual adjustment.

Adaptive methods are contribution Ridge and residual Ridge, each with
Sinkhorn, plus a selective residual Sinkhorn variant. Settings and freeze
rules are in the [study plan](docs/PLAN.md) and [final report](docs/rs3-final-report.md).

## Metrics

- **Balanced accuracy (BA):** mean recall over all 100 classes.
- **Head / Medium (Middle) / Tail accuracy:** macro recall within groups with at least
  100, 20–99, and fewer than 20 training examples (35 / 35 / 30 classes).
- **Sample accuracy:** ordinary fraction of correctly classified examples.
- **All-seed gate:** an adaptive method must exceed uniform logits on both BA
  and Tail for each of the three seeds.
- **Fixed-mixture non-domination:** a separate check; a fixed mixture with BA
  and Tail both greater than or equal to an adaptive result disqualifies it,
  so ties count.

## Current results

**Table 1.** Completed `ridge_sinkhorn_3seed_v1`: 3 seeds × 5 outer folds,
with 10,847 pooled rows per seed. Values are percentages, mean ± population SD
(`ddof=0`) across three pooled seed-level results. This is retrospective
nested-OOF evidence on the shared canonical population; the seeds are not
independent population replicates.

| Method | Balanced accuracy | Head | Medium | Tail |
|:--|--:|--:|--:|--:|
| Uniform logits | 42.410 ± 0.315 | 67.663 ± 0.225 | 40.584 ± 0.687 | 15.077 ± 0.634 |
| Uniform probabilities | 41.556 ± 0.345 | 66.967 ± 0.330 | 39.481 ± 0.325 | 14.329 ± 0.871 |
| Fixed 006 | 43.062 ± 0.405 | 68.116 ± 0.232 | 41.783 ± 0.466 | 15.323 ± 0.785 |
| Fixed 007 | 42.251 ± 0.955 | 64.047 ± 0.714 | 40.357 ± 1.133 | 19.033 ± 1.233 |
| Fixed 010 | 42.566 ± 0.663 | 64.256 ± 0.315 | 40.606 ± 0.963 | 19.548 ± 1.671 |
| Fixed 011 | 41.802 ± 0.839 | 60.404 ± 0.645 | 39.184 ± 0.679 | 23.153 ± 1.845 |
| Contribution Ridge | 42.245 ± 0.131 | 68.766 ± 0.074 | 40.372 ± 0.235 | 13.489 ± 0.245 |
| Contribution Ridge + Sinkhorn | 42.719 ± 0.366 | 67.634 ± 0.866 | 41.313 ± 0.703 | 15.293 ± 0.324 |
| Residual Ridge | 42.509 ± 0.466 | 66.522 ± 0.437 | 40.964 ± 0.513 | 16.296 ± 0.633 |
| Residual Ridge + Sinkhorn | 42.861 ± 0.706 | 66.824 ± 0.729 | 41.600 ± 0.920 | 16.377 ± 0.711 |
| Selective residual Ridge + Sinkhorn | 42.832 ± 0.673 | 66.777 ± 0.674 | 41.489 ± 0.869 | 16.463 ± 0.830 |

The adaptive display names map in row order to `contribution_ridge`,
`contribution_ridge_sinkhorn`, `residual_ridge`,
`residual_ridge_sinkhorn`, and
`selective_residual_ridge_sinkhorn`. Fixed weights use expert order
(`CE, LAL, BalancedSoftmax, Mixup`): 006 = (0, .25, .25, .50), 007 = (0,
.25, .50, .25), 010 = (0, .50, .25, .25), and 011 = (0, .50, .50, 0).

**Table 2.** Paired changes from uniform logits in percentage points, shown as
mean [95% interval]. The intervals are paired class-then-sample bootstrap
intervals, conditional on fitted models and not retraining uncertainty. Change
estimates use full-precision results before rounding. Each adaptive method
must exceed uniform logits on both BA and Tail for each seed.

| Adaptive method | ΔBA [95% interval] | ΔTail [95% interval] | All-seed gate |
|:--|:--|:--|:--:|
| Contribution Ridge | −0.165 [−0.852, +0.357] | −1.589 [−3.674, −0.253] | Fail |
| Contribution Ridge + Sinkhorn | +0.310 [−0.354, +0.935] | +0.216 [−1.669, +1.883] | Fail |
| Residual Ridge | +0.099 [−0.681, +0.858] | +1.219 [−0.959, +3.270] | Fail |
| Residual Ridge + Sinkhorn | +0.452 [−0.276, +1.162] | +1.300 [−0.686, +3.156] | Fail |
| Selective residual Ridge + Sinkhorn | +0.422 [−0.308, +1.141] | +1.385 [−0.627, +3.250] | Fail |

All five BA intervals include zero. Fixed-mixture non-domination is a separate
assessment; ties with a fixed comparator count as domination. See the
[full report](docs/rs3-final-report.md) for per-seed gates, non-domination
flags, and the complete uncertainty procedure. Values are from the
[canonical results JSON](.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/results.json).

## Uncertainty and evidence limits

Intervals use 10,000 paired, class-then-sample bootstrap replicates with seed
`20260924`. They are conditional on fitted models and omit retraining
variability. Outer BA spread uses population SD (`ddof=0`); inner summaries
aggregate five folds within seed, then use sample SD across three seed means
(`ddof=1`). Inner values are fractions, unlike outer percentages.

Earlier development inspected inner-fold-0 labels, and seed 78 / outer fold 0
was already evaluated in the separate `ridge_sinkhorn_v3` study. The full
matrix is retrospective and does not provide a new-population confirmation.

The v1 evaluation did not load or score test examples or labels. Constructing
`CIFAR100(train=True)` made torchvision read `cifar-100-python/test` for MD5
verification; its pickle was not unpickled, and examples and labels were not
loaded. `test_accessed=false` records no test examples scored, not this
checksum-only read. See the [full audit](docs/rs3-final-report.md).

The hard-selection oracle is label-dependent top-1 opportunity, not an
inference method or a bound on soft mixtures. Soft-mixture feasibility tests
per-row existence of a convex-logit positive true-class margin using the label;
it is not prediction or inference-time accuracy. See the
[protocol glossary](docs/protocol.md#diagnostic-terms).

## Quick start

These checks validate the saved study configuration and local inputs without
starting training or evaluation:

```bash
python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile configs/profiles/local-smoke.yaml study validate

python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile configs/profiles/local-smoke.yaml study doctor
```

The matrix is complete. For exact saved-artifact commands, evaluator identity,
and output locations, use [reproduction](docs/reproduction.md). Do not launch
training merely to reproduce the saved report.

## Reading map

| Need | Start here |
|:--|:--|
| Data roles, metric definitions, evaluation safeguards | [Protocol](docs/protocol.md) |
| Frozen study rules and settings | [Hash-bound plan](docs/PLAN.md) |
| New study sessions and recovery | [Experiment workflow](docs/experiment-workflow.md) |
| Saved study commands and evaluator contract | [Reproduction](docs/reproduction.md) |
| Three-seed current results and audit | [Full-matrix report](docs/rs3-final-report.md) |
| Earlier OOF development and locked fold | [OOF results](docs/oof-results.md) |
| Historical balanced-test benchmark | [Full-data results](docs/results.md) |
| Inner diagnostics and pinned inputs | [Diagnostics](docs/diagnostics.md) |
| Remaining research questions | [Research notes](docs/research.md) |
| Complete document map and local evidence paths | [Documentation index](docs/README.md) |

## Key references

These primary sources give context for methods used here; project-specific
choices and results are documented above.

- **OOF stacking:** Wolpert (1992), [Stacked generalization, *Neural
  Networks*](https://www.sciencedirect.com/science/article/pii/S0893608005800231).
  It motivates training a second-level learner on held-out base-model outputs.
- **Ridge:** Hoerl and Kennard (1970), [Ridge Regression, *Technometrics*](https://doi.org/10.1080/00401706.1970.10488634).
  It motivates the regularized linear router used to predict expert contributions.
- **Sinkhorn:** Cuturi (2013), [Sinkhorn Distances, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).
  It is a foundational entropic solver; this project adapts it to relaxed,
  frozen-price inference.
- **Expert diversity:** Wang et al. (2021), [RIDE, ICLR](https://openreview.net/pdf?id=D9I3drBz4UC).
  It studies diverse experts for long-tailed recognition, motivating the
  broader idea of combining specialists.
- **Logit-adjusted loss:** Menon et al. (2021), [Long-tail learning via logit adjustment, ICLR](https://openreview.net/pdf?id=37nvvqkCo5).
  It motivates one of the four expert training objectives.
- **BalancedSoftmax:** Ren et al. (2020), [Balanced Meta-Softmax, NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html).
  It motivates a second long-tail expert objective.
- **Mixup:** Zhang et al. (2018), [mixup, ICLR](https://openreview.net/pdf?id=r1Ddp1-Rb).
  It motivates the interpolation-based expert recipe.
