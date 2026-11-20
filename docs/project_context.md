# Project context

Use this map to locate the code and understand how the components fit together.

## Project overview

This project studies long-tailed image classification on CIFAR-100. It uses a
plain ResNet-32 and a common evaluator to compare ordinary cross-entropy with
two independent uses of optimal transport (OT). **Reweighting** changes how
much each real training example contributes to the classifier's loss.
**Synthesis** generates extra training features and fits the classifier on
real and synthetic features. Both can start from the same saved baseline for
each seed.

The current research focus is **prototype OT reweighting**, which keeps each
class's total weight fixed and changes weights within that class. It is
distinct from the learned full-pool OT method and from synthesis.

Preparation retains 10,847 images at imbalance factor 0.01 and data seed 42.
The development study uses split seed 43: 8,716 fitting images and 2,131
development images. Development images are excluded from baseline training
and fitted method state. The official 10,000-image test set is reserved for
final evaluation after the recipe is frozen.

Stages A and B are complete: 27 assignment continuations and 15 simple-control
continuations, respectively. They have not established the planned 0.5-point
practical improvement over class balancing. See the [research handoff](HANDOFF.md)
for findings and pinned results.

The [controlled-bias diagnostic](PLAN.md#controlled-bias-diagnostic-the-next-experiment)
completed nine converged fresh-head fits on 34 eligible Head classes and
stopped at Gate 1: 0.619 points of damage and 0.064 points of recovery.
The conditional OT comparisons did not run. Its classifier-only findings
do not establish whether OT is useful in other training settings.

The next [two Stage C experiments](PLAN.md#stage-c-two-bounded-follow-up-experiments)
are **implemented; scientific training has not run**. C2 uses feature-space SMOTE and 30 fresh
CPU classifier fits to separate synthesis, duplication, simple weighting and
OT assignment effects. C1 uses 18 GPU continuations to compare OT and its
controls while backbone parameters update with BN running buffers frozen.
They share audit baselines and split 43 but use separate matched comparisons.
The user chose feature synthesis first; actual DeepSMOTE image generation
is deferred. Confirmation on split 44 is conditional on useful results.

## Project structure

| Location | Responsibility |
|---|---|
| [`pipeline/data.py`](../pipeline/data.py) | Builds and loads the retained CIFAR-100-LT pool and defines training and deterministic evaluation transforms. |
| [`pipeline/model.py`](../pipeline/model.py) | Defines CIFAR ResNet-32, with a 64-dimensional backbone output and a linear classifier for the 100 classes. |
| [`pipeline/train.py`](../pipeline/train.py) and [`pipeline/runtime.py`](../pipeline/runtime.py) | Validate configs, select methods, manage seeds and devices, and save training history and checkpoints. |
| [`pipeline/methods/`](../pipeline/methods/) | Contains the baseline, classifier continuation, OT reweighting, prototype weighting, SAVA, and synthesis implementations. |
| [`pipeline/extract_features.py`](../pipeline/extract_features.py) and [`pipeline/train_classifier.py`](../pipeline/train_classifier.py) | Cache frozen features in retained row order and train a classifier on features, with support for synthesis continuations. |
| [`pipeline/evaluate.py`](../pipeline/evaluate.py) | Loads a saved model and reports overall and balanced accuracy, per-class recall, and Head/Medium/Tail recall. |
| [`pipeline/run_comparison.py`](../pipeline/run_comparison.py) | Runs the synthesis controls and collects their metrics. |
| [`pipeline/run_ablation.py`](../pipeline/run_ablation.py), [`ablation_data.py`](../pipeline/ablation_data.py), and [`ablation_weights.py`](../pipeline/ablation_weights.py) | Run the fitting/development split, schedule pilot, controlled weighting comparisons, and optional final evaluation. |
| [`pipeline/ablation_inputs.py`](../pipeline/ablation_inputs.py) | Validate pinned audit reports, split metadata, baseline checkpoints, and fitting/development feature bundles. |
| [`pipeline/run_weight_study.py`](../pipeline/run_weight_study.py) | Run Stage A and dispatch Stage B; write completion receipts and reuse compatible completed continuations. |
| [`pipeline/ablation_references.py`](../pipeline/ablation_references.py) | Validate Stage A parent reports, receipts, weights, shared implementation, and training environment before Stage B reuse. |
| [`pipeline/simple_control_weights.py`](../pipeline/simple_control_weights.py) | Replay prototype geometry and construct Stage B's five native or matched-spectrum controls. |
| [`pipeline/run_simple_controls.py`](../pipeline/run_simple_controls.py) | Run Stage B and report paired contrasts, active/fallback contributions, and sensitivity to one-image development classes. |
| [`pipeline/controlled_bias_data.py`](../pipeline/controlled_bias_data.py) and [`controlled_bias_weights.py`](../pipeline/controlled_bias_weights.py) | Validate the pinned seed-0 controller, select reference/unbiased/biased fitting rows, and construct known, reference-based and shuffled weights with explicit information boundaries. |
| [`pipeline/controlled_bias_head.py`](../pipeline/controlled_bias_head.py), [`controlled_bias_reporting.py`](../pipeline/controlled_bias_reporting.py), and [`run_controlled_bias.py`](../pipeline/run_controlled_bias.py) | Fit fresh regularized K-class CPU heads, verify convergence, enforce the two gates, report paired results and restore compatible completed fits. |
| [`pipeline/feature_smote.py`](../pipeline/feature_smote.py) and [`followup_weights.py`](../pipeline/followup_weights.py) | Generate raw-coordinate SMOTE and paired anchor copies; build fixed real-only-center OT/simple controls and exact class/origin shuffles. |
| [`pipeline/image_reweighting.py`](../pipeline/image_reweighting.py) | Update image backbone/head parameters with matched augmentation streams, fixed sample weights, and frozen BN running buffers. |
| [`pipeline/followup_images.py`](../pipeline/followup_images.py) | Validate exact retained/fitting/development pixel and label hashes, audit split metadata and cache row order independently of NPZ container serialization. |
| [`pipeline/run_ot_followups.py`](../pipeline/run_ot_followups.py) and [`followup_reporting.py`](../pipeline/followup_reporting.py) | Run the separate 18-fit C1 / 30-fit C2 matrices, verify/preserve artifacts, and report 100-class paired contrasts and screening decisions. |
| [`scripts/kaggle_prototype_comparison.py`](../scripts/kaggle_prototype_comparison.py) and [`scripts/kaggle_ablation_study.py`](../scripts/kaggle_ablation_study.py) | Run full-pool method comparisons, development ablations, and weight audits on Kaggle. |
| [`scripts/check_reweighting_transport.py`](../scripts/check_reweighting_transport.py) and [`scripts/report_results.py`](../scripts/report_results.py) | Check resource use for full-pool OT and regenerate descriptive tables and charts from the archive. |
| [`configs/`](../configs/) and [`tests/`](../tests/) | JSON experiment recipes and automated checks using artificial data. |
| [`results/`](../results/README.md) | Historical metrics and source provenance. [`results/paper/`](../results/paper/README.md) contains the derived tables and plots. |
| [`pyproject.toml`](../pyproject.toml) | Package dependencies, Python requirement, and pytest configuration. |
| `data/` and `outputs/` | Local training data, feature caches, checkpoints, and reports. Both are ignored by Git. |

## How the experiment components work

- **Data preparation.** [`data.py`](../pipeline/data.py) saves the subsampled
  training images, labels, class counts, and preparation settings together.
  [`ablation_data.py`](../pipeline/ablation_data.py) derives the fitting and
  development subsets and records their positions in the retained pool.
- **Baseline.** [`baseline.py`](../pipeline/methods/baseline.py) trains the full
  model with cross-entropy and SGD. The warm-up/multistep and cosine recipes
  are defined in separate configuration files.
- **Full-pool OT reweighting.** [`reweighting.py`](../pipeline/methods/reweighting.py)
  learns example weights against balanced class representatives from the
  retained features. The transport objective updates the example scores.
  Weighted cross-entropy updates the classifier. Equal transport mass at the
  representatives does not guarantee equal total example weight per class.
- **Fixed weighting controls.** [`finetuning.py`](../pipeline/methods/finetuning.py)
  restores a baseline and runs a matched continuation. Uniform gives examples
  equal weight. Class-balanced gives classes equal total weight and divides
  it evenly within each class. Cluster-balanced divides a class's weight
  equally across its two clusters. [Prototype Sinkhorn](../pipeline/methods/prototype_weights.py)
  uses distances to two prototypes to assign weights within a class, keeping
  its total fixed. It falls back to class-balanced weighting when it cannot
  form two valid clusters. [SAVA](../pipeline/methods/sava_weights.py) uses
  its example scores to distribute the same fixed total within each class.
  The development ablation tests **prototype Sinkhorn**, a distinct method
  from the learned full-pool OT weights.
- **Cached-feature studies.** The Stage A/B runners validate the complete input
  seed set, construct fixed weights, and call the shared
  `train_classifier_from_features` loop in
  [`reweighting.py`](../pipeline/methods/reweighting.py). Each continuation
  restores the same full baseline for its seed and freezes backbone parameters
  and batch-normalization state. Classifier training uses raw feature
  coordinates; normalized features are used for prototype geometry.
  Assignment ties and shuffles use independent saved random generators.
  Final checkpoints retain the full 100-class image model.
- **Controlled-bias diagnostic.** The separate runner reserves reference rows,
  undersamples a recorded controller subgroup, and freezes all masks/weights
  before fitting. Ordinary geometry builders receive retained/reference inputs;
  privileged sampling probabilities and controller groups are kept separate.
  Fresh heads use raw features, CPU float64 full-batch L-BFGS and a final gradient
  check. Nine Gate-1 fits establish damage and recovery; only a verified pass
  permits the other 18. The head and predictions are diagnostic artifacts with
  an explicit original-label map, separate from image-model checkpoints.
- **OT synthesis.** [`synthesis.py`](../pipeline/methods/synthesis.py) works
  with 64-dimensional features. The guided recipe fits a one-versus-rest
  linear guide, gives harder examples more transport mass, and forms
  synthetic features as weighted combinations of real features in the same
  class. Its target is 500 real plus synthetic features per class. It uses
  scaled coordinates internally and saves the generated vectors in the
  classifier's raw feature coordinates. The comparison includes real-only,
  random duplication, uniform OT, loss-shifted OT, and full guided OT.
- **Evaluation and reporting.** The evaluator loads a saved checkpoint and
  uses deterministic preprocessing. Balanced accuracy is the mean of
  per-class recalls. Head/Medium/Tail groups use retained counts of real
  images: Head `>=100`, Medium `20–99`, Tail `<20`. The reporting script
  turns archived metrics into tables and plots.

## Execution and saved state

For cached-feature studies, the local pipeline commands read validated local
inputs and perform training and evaluation. The Kaggle wrappers handle source
checkout, GitHub access, artifact restoration, and incremental publication.

For Stages A/B, the study saves resolved settings, input/source/environment
identities, diagnostics, per-run histories and metrics, and completion receipts.
Receipts hash the completed run's files. Reuse checks verify both inputs and
outputs; interrupted continuations restart from their source baseline.
Study summaries require all expected runs. JSON reports live on results
branches; checkpoints, features, weights, and geometry use checksum-linked
release assets. Raw images and binaries stay out of Git history.

The ablation Kaggle script defaults to `STAGES = ("synthetic_reweighting",)`:
30 CPU fits, automatically restored audit inputs, no CIFAR attachment, and
the separate `ablation-synthetic-reweighting-results` branch. C1 uses
`STAGES = ("ot_contribution",)`, a GPU and CIFAR training images, publishing to
`ablation-ot-contribution-results`. Both require Internet and `GITHUB_TOKEN`;
`FOLLOWUP_PREFLIGHT_ONLY` validates without scientific fitting or publication.
Source/config/environment/input identities and saved preparation are checked
on recovery. Completed fits are reused, interrupted attempts preserved and
restarted, and JSON/binary assets published separately. See
[exact settings](PLAN.md#implemented-stage-c-commands-and-kaggle-delivery).
Historical `controlled_bias` and `simple_controls` modes remain available.
The diagnostic's nine completed Gate-1 fits and stop are unchanged.

## Documentation map

| Document | Purpose |
|---|---|
| [Research handoff](HANDOFF.md) | Completed findings, result locations, and the next implementation task |
| [Research plan](PLAN.md) | Authoritative experiment protocols, decision gates, and run budgets |
| [Stage A guide](weight_assignment.md) | Reproduce the completed assignment study and recover published runs |
| [Stage B guide](weight_simple_controls.md) | Reproduce the completed simple-control study and restore its parent references |
| [Method integration](method_integration.md) | Add methods through the shared training/checkpoint interfaces |
| [Reweighting guide](reweighting.md) | Existing method configs, commands, losses, and diagnostics |
| [Running both OT paths](ot_methods.md) | Matched full-pool workflows, synthesis resume, and historical recipe differences |
| [Guo](guo_ot_reweighting.md) and [OTOS](otos_synthetic_samples.md) notes | Source-paper details and boundaries of the local adaptations |

Keep protocols in PLAN.md, research status in HANDOFF.md, and the code map here.
The method and stage guides document implemented workflows; planned interfaces
must be identified as planned.
