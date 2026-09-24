# Expert Routing on Long-Tailed CIFAR-100

## Abstract

This project studies whether predictions from differently trained classifiers can be combined to
improve recognition of rare classes in CIFAR-100-LT at imbalance ratio 100. It compares uniform
logit averaging with fixed expert mixtures and routers that use inference-time information. The
repository contains two distinct tracks: a completed full-data benchmark evaluated on the balanced
CIFAR-100 test set, and a nested out-of-fold (OOF) research track with one locked outer-fold
evaluation. Their metrics come from different populations and are not directly comparable.

**Current conclusion.** The locked residual Ridge router improved over outer-fold uniform logits by
point estimate, but paired intervals include zero and two frozen fixed-weight references exceeded
it on both Balanced Accuracy and Tail accuracy. Its prespecified expansion gate failed. The result
does not establish a generally useful adaptive router.

## Problem Statement & Objectives

CIFAR-100-LT contains far fewer training examples for some classes than others. A classifier that
performs well on frequent classes can miss rare classes, so this project measures both Balanced
Accuracy (BA) and Tail accuracy. The expert pool uses different training objectives to create
potentially complementary predictions.

The main question is whether an inference-time rule can combine those experts well enough to
improve **both BA and Tail accuracy over uniform logit averaging**. For the nested study, a method
must also pass its frozen evaluation gates against fixed-weight references. The balanced CIFAR-100
test set is reserved for the historical full-data benchmark; it is not used to fit or select OOF
routers.

The full-data and OOF tracks answer different questions. The first reports the historical
three-seed test benchmark. The second uses held-out predictions from folds of the long-tailed
training set to study routing and reports one locked outer-fold result. OOF values must not be
compared numerically with full-data test values.

## Pipeline Architecture

1. **Fix the data and metric protocol.** The canonical CIFAR-100-LT training population contains
   10,847 images at IR=100. Head, Medium, and Tail use the shared class-count implementation in
   `scripts/base_trainer.py::compute_class_groups`: 35 Head classes (`n >= 100`), 35 Medium
   classes (`20 <= n < 100`), and 30 Tail classes (`n < 20`).

2. **Train four experts.** Each expert uses ResNet-32 and the shared 200-epoch training schedule,
   with no validation split and a final-epoch checkpoint. The objectives are cross-entropy (CE),
   logit adjustment (LAL, `tau = 1`), BalancedSoftmax, and Mixup (`alpha = 1`). The full-data track
   uses seeds 78, 88, and 1034. The completed OOF collection uses seed 78.

3. **Measure the historical full-data baseline.** Train each expert on all 10,847 images, then
   evaluate the saved final checkpoints on the balanced CIFAR-100 test set. Uniform logit
   averaging is the reference for historical parameter-free routing comparisons. This test set
   has already been examined; it is not a router-development set.

4. **Build held-out OOF predictions.** For outer fold 0, 8,677 images form the outer training
   population and 2,170 form the held-out outer population. The outer training portion is split
   into four inner folds. For each expert and inner fold, train on about 6,507 images and predict
   about 2,170 held-out images. This gives 16 inner expert runs and row-wise held-out predictions.
   Each predicted image is excluded from the expert that produced its prediction.

5. **Test fixed mixtures and router feasibility.** The OOF development analyses use inner folds
   1–3 (6,507 images) for fitting and diagnostics. Fixed convex mixtures combine the original
   expert logits with constant weights. Ridge models test whether expert contributions can be
   predicted from inference-time features. A separate soft-mixture oracle uses each image's true
   label to find feasible weights; it measures existence headroom and is not an inference-time
   classifier.

6. **Test allocation and lock one candidate.** The staged Ridge/Sinkhorn study fits on inner
   folds 1–3 and selects on inner fold 0 (2,170 images). It first tests Sinkhorn-style allocation
   on frozen Ridge scores, then tests residual Ridge around `fixed_007`, whose weights are
   `(CE, LAL, BalancedSoftmax, Mixup) = (0, 0.25, 0.50, 0.25)`. The selected router predicts
   per-image expert weights from the four expert confidences and combines the original logits.
   The selected family and settings are locked before the outer artifacts are loaded, then refit
   on all four inner OOF folds. Task 3C had previously inspected inner fold 0 descriptively, so it
   is not an untouched confirmation set.

7. **Evaluate the lock once.** Four outer experts are trained on the 8,677 outer training images
   and predict the 2,170 held-out images. The locked router is evaluated against uniform logits
   and frozen references using the canonical class groups and paired bootstrap intervals. The
   outer fold has been consumed and cannot be used to select a replacement method. The original
   balanced test set was not accessed for this OOF evaluation.

## Key Findings & Contributions

- **Historical benchmark:** Uniform logit averaging is the full-data reference, with 46.98 ± 0.69%
  BA and 18.76 ± 0.86% Tail accuracy across the three configured seeds. It was stronger than the
  preregistered parameter-free routing rules on that track.
- **OOF complementarity:** Several fixed convex mixtures improve both metrics over the OOF
  uniform baseline on the 6,507-image development partition. This shows that expert composition
  matters; it does not show that a router can predict useful per-image weights.
- **Oracle headroom:** A label-dependent soft-mixture feasibility analysis found feasible weights
  for 55.56% of images by macro feasibility and 26.36% of Tail images. Because it uses true labels,
  this is an existence diagnostic rather than deployable performance.
- **Features and targets:** Full 13-feature Ridge lowered contribution-prediction error compared
  with confidence-only Ridge, but its primary classification result was lower: 36.10% versus
  36.55% BA, and 6.55% versus 7.38% Tail. Improved target prediction did not translate into better
  classification in this comparison.
- **Sinkhorn and residual Ridge:** Frozen-price Sinkhorn failed its development gate. The locked
  outer candidate was confidence-only residual Ridge without OT. It beat outer uniform by point
  estimate but was exceeded by `fixed_007` and `fixed_010` on both target metrics.
- **Reproducible evidence:** The OOF pipeline records fold membership, run configurations,
  prediction provenance, selection decisions, and immutable result artifacts. The nested protocol
  separates router fitting, selection, and the one locked outer evaluation.

## Results & Evidence

The tables below report each experiment on its own evaluation population. Full-data test results
and OOF measurements must be interpreted separately.

### Full-data test benchmark

| Track | Seeds | Method | Balanced Accuracy | Tail accuracy |
|:--|:--|:--|--:|--:|
| Full-data test | 78, 88, 1034 | Uniform logits | 46.98 ± 0.69% | 18.76 ± 0.86% |

See [docs/results.md](docs/results.md) for the complete historical expert and routing tables.

### OOF development partition

These results use the 6,507-image partition from inner folds 1–3. Fixed-weight rows are
development results, not independent test performance. `fixed_006`, `fixed_007`, and `fixed_010`
use `(CE, LAL, BalancedSoftmax, Mixup)` weights shown below.

| Method | Weights | BA | Tail |
|:--|:--|--:|--:|
| Uniform logit average | `(0.25, 0.25, 0.25, 0.25)` | 35.9328% | 7.0094% |
| `fixed_006` | `(0, 0.25, 0.25, 0.50)` | 37.2134% | 9.9115% |
| `fixed_007` | `(0, 0.25, 0.50, 0.25)` | 36.4403% | 14.7290% |
| `fixed_010` | `(0, 0.50, 0.25, 0.25)` | 36.4236% | 14.5055% |
| Confidence-only Ridge (Task 3F) | Per-image predicted weights | 36.5502% | 7.3797% |
| Full 13-feature Ridge (Task 3F-F) | Per-image predicted weights | 36.10% | 6.55% |

### Locked outer-fold-0 evaluation

This single evaluation covers 2,170 images from the long-tailed training population. Head,
Medium, and Tail columns are macro recall within each canonical group; sample accuracy is included
for context.

| Method | BA | Head | Medium | Tail | Sample accuracy |
|:--|--:|--:|--:|--:|--:|
| Uniform original logits | 42.7006% | 67.3885% | 41.0419% | 15.8333% | 63.9171% |
| Locked residual Ridge, no OT | 44.7035% | 68.4018% | 44.7986% | 16.9444% | 65.1613% |
| `fixed_006` | 44.6348% | 68.1301% | 44.6361% | 17.2222% | 65.1613% |
| `fixed_007` | 44.7248% | 65.3856% | 46.4473% | 18.6111% | 62.7650% |
| `fixed_010` | 44.7298% | 64.1379% | 43.6616% | 23.3333% | 61.8433% |

| Candidate difference from uniform | Point estimate | Paired 95% interval |
|:--|--:|:--|
| BA | +2.0028 percentage points | −0.4956 to +4.5771 points |
| Tail accuracy | +1.1111 percentage points | −5.0000 to +7.7778 points |

Intervals use 10,000 class/sample hierarchical paired bootstrap replicates with seed 20260924.
Both intervals include zero. The candidate also fails the fixed-reference Pareto gate: `fixed_007`
and `fixed_010` each exceed it on both BA and Tail. The full five-fold, three-seed experiment was
not run.

### Inner-fold-0 selection diagnostics

The selection fold was used only after fitting on inner folds 1–3. Its earlier descriptive
exposure in Task 3C is documented in the protocol and results report.

| Selection-fold method | BA | Tail |
|:--|--:|--:|
| Uniform original logits | 37.2941% | 10.8333% |
| Locked residual Ridge, no OT | 38.0374% | 13.0556% |
| Prior-only global-bias control | 38.4867% | 14.7222% |
| Frozen-price OT, `rho = 10` diagnostic | 37.9514% | 14.7222% |

The prior-only control exceeded the selected candidate on both metrics. No frozen-price OT
strength improved both metrics over its matched Ridge kernel, so no Sinkhorn configuration
advanced to the outer evaluation.

## Limitations

- The locked evaluation covers one expert-training seed and one outer fold. It does not establish
  generalization across seeds, folds, final full-data experts, or a fresh test population.
- The selection fold had prior descriptive exposure, and OOF expert models share training
  dependence across router-fit and selection rows. These are documented design limitations.
- The outer candidate's paired intervals include zero, and fixed references dominate its point
  estimates. No adaptive method has passed the project's frozen success gates.
- The full five-fold, three-seed nested matrix (300 expert runs) and new specialized experts remain
  unexecuted. This candidate did not meet the prespecified gate for that expansion.
- The original balanced test set has been accessed repeatedly for the historical full-data work.
  It is not an ordinary router-development resource.

## Reproducibility

The saved staged-study artifacts can be checked and the locked outer report reproduced from the
existing outer artifacts with:

```bash
./.venv/bin/python scripts/run_ridge_sinkhorn.py
./.venv/bin/python scripts/run_ridge_sinkhorn_outer.py
```

The outer fold has already been consumed; its results must not be used to tune another method.
Neither command loads the original balanced CIFAR-100 test set. The exact outer-expert run
commands and artifact requirements are in the
[Ridge/Sinkhorn runbook](docs/ridge-sinkhorn-runbook.md).

## Documentation Map

- [Project context](docs/project-context.md): current status, protocol, and repository map.
- [Full-data results](docs/results.md): historical three-seed test results and baselines.
- [OOF results](docs/oof-results.md): development analyses, outer metrics, intervals, and audit.
- [Nested OOF protocol](docs/nested-oof-protocol.md): folds, data roles, and leakage constraints.
- [Problem statement](docs/problem.md): established limitations and unresolved questions.
- [Research notes](docs/research.md): reviewed literature and research direction.
- [Ridge/Sinkhorn runbook](docs/ridge-sinkhorn-runbook.md): completed study and run artifacts.
- [Test access log](docs/test-access-log.md): append-only history of original test-set access.
