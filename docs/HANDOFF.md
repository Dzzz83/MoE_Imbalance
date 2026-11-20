# Research handoff

Work is on `ablation_study`. Read [project context](project_context.md) for
the code map, [AGENTS.md](../AGENTS.md) for working rules, and [PLAN.md](PLAN.md)
for the protocol and commands.

## Completed experiments

- The [initial development study](https://github.com/Dzzz83/MoE_Imbalance/blob/ablation-study-results/reports/study_summary.json)
  selected classifier learning rate `0.002`. Class balancing scored **38.708%**
  balanced accuracy. Prototype OT scored **38.702%**. Half-strength and
  shuffled OT were also close to class balancing.
- The [stronger OT study](https://github.com/Dzzz83/MoE_Imbalance/blob/ablation-strength-results/reports/study_summary.json)
  used rebuilt baselines and their own matched controls. Cubing the weights
  and normalizing within each class preserved their ranking and widened the
  spread. Stronger OT scored **38.075%** versus **38.063%** for its within-class
  shuffle. It lost on two of three seeds and reached only 4–7 of the 30 Tail
  classes per seed. These runs did not establish an advantage from OT's
  particular assignments.
- The [assignment study](https://github.com/Dzzz83/MoE_Imbalance/blob/0d4cf3e90be73513ddbe5c731698610b0285773c/reports/study_summary.json)
  completed all 27 continuations for 40 epochs. Class-balanced/original OT/
  strong OT/easy/boundary/hard scored **38.464/38.708/38.503/38.777/38.075/38.032%**
  balanced accuracy. Original OT gained 0.24 percentage points versus
  class-balanced, with gains of about 0.01/0.07/0.65 points across seeds.
  Easy weighting beat boundary and hard in every seed but original OT in only
  one; it remains exploratory. Boundary failed its primary target, and no method
  gained the planned 0.5 points over class balancing on average.
- The [Stage B simple-control study](https://github.com/Dzzz83/MoE_Imbalance/blob/451349aa463cd5b7ec37ea3a97fd99b8bf50a5d2/reports/study_summary.json)
  completed all 15 new continuations for 40 epochs. In order, cluster balancing,
  prototype distance, easy rank, cluster-ranked strong weights, and
  distance-ranked strong weights scored
  **38.287/38.521/38.731/38.152/38.533%**. No new condition cleared the
  0.5-point target over class balancing. Easy-rank's +0.267-point mean includes
  +0.333 points from one newly correct one-image class in seed 1. Distance-based
  assignment of the exact strong OT values differed from strong OT by only
  +0.031 points on average. These are narrow findings about this prototype
  formulation, not a verdict on all OT reweighting.

Compare methods within a study and within each training seed. The initial and
stronger studies used different fitted checkpoints; the assignment study uses
the rebuilt audit's exact baselines. The development split is fixed and
excluded from baseline training, prototype fitting, and classifier training.
Keep the official test set for final evaluation after the recipe is frozen.

## Completed audit and next experiment

The [rebuilt audit](https://github.com/Dzzz83/MoE_Imbalance/blob/51c7829d505143bd556ff1f3038b6559751e3428/reports/weight_audit.json)
is published. Across seeds 0/1/2, the highest-weight quarter was about
14.5/16.9/15.0 confidence percentile points above its class average among
OT-active fitting classes. This supports the easy-example preference hypothesis;
it does not establish that this preference caused the weak development results.

Follow the [staged investigation in PLAN.md](PLAN.md#next-investigation-assignment-transport-value-and-confirmation).
The [project context](project_context.md#project-structure) maps the implemented
components; the stage guides document commands and recovery.
Stage A is complete on the exact rebuilt audit baselines. Analysis checked
all 27 completion receipts and frozen backbones, reproduced saved metrics from
checkpoint predictions, and verified the 18 stronger-weight reassignments and
shuffles preserve their class weight multisets. Boundary and hard selected
almost identical top-quarter examples (about 99.7% overlap). Tail development
scores are fragile: 15 Tail classes have one held-out image, and one extra
correct prediction in such a class contributes 0.33 points to a three-seed mean.
Boundary's -0.39-point difference versus class-balanced decomposes into +0.14
points from original OT-active classes and -0.53 from fallback classes, as
contributions to overall balanced accuracy. Unchanged weights in a fallback
class do not guarantee unchanged predictions when other classes are reweighted.

Stage B analysis verified the published archives and all 15 completed runs,
reproduced accuracy/recall from saved heads on CPU, checked frozen backbones
and weight invariants, and recomputed the paired contrasts. Its source is
`b4cea1c8b68eb7bd872f1b5500e8e4d9038ecf69`. The pinned input, parent, and
archive identities are recorded in PLAN.md and the study's reports.

The [controlled-bias diagnostic](PLAN.md#controlled-bias-diagnostic-the-next-experiment)
completed **9/9 converged fits** and stopped at Gate 1. Results commit
[f2de29bb74ffbcefe1820668156a29cf6186cff2](https://github.com/Dzzz83/MoE_Imbalance/blob/f2de29bb74ffbcefe1820668156a29cf6186cff2/reports/study_summary.json)
uses source d5416a487239436d5c3fab252fbe9a4d02524862. Before removal,
after removal and known correction scored **67.070/66.452/66.516%** on the
34-class task. The introduced damage was 0.619 points and recovery was
0.064 points, below the 1.0/0.5-point gates. Do not force the 18 conditional
OT fits to run. These results concern a fresh classifier on existing features,
not image training from the biased data or a verdict on OT.

Analysis verified 35 published JSON hashes, nine completion records, saved
convergence diagnostics, class-count accuracy/recall and the gate decision.
It did not independently reproduce binary-head predictions. Under-sampled
groups gained 0.557 points of overall balanced-accuracy contribution with
correction, offset by -0.493 from other groups. Macro prediction loss improved
in every seed; this does not replace the failed primary accuracy gate.

The user requested both [Stage C experiments](PLAN.md#stage-c-two-bounded-follow-up-experiments).
They are **implemented, with scientific training still to run on Kaggle**:

- **C2 first:** feature-space SMOTE plus reweighting; 30 fresh CPU head fits
  covering real-only, anchor duplication, interpolation, simple weighting and
  origin-preserving OT shuffles. Use the same real-only prototype geometry.
  The user chose feature-space SMOTE before considering DeepSMOTE images.
- **C1:** 18 GPU image continuations comparing class balancing, original
  prototype OT, easy rank and three OT shuffles while backbone parameters
  update and BN running buffers stay frozen.
- Share pinned audit baselines and input handling, not scores across different
  training protocols. Both use the original 100-class development task and
  fixed budgets; either proceeds regardless of the other's scientific outcome.
- Confirm only a promising route on split 44 with three fresh fitting-only
  baselines. DeepSMOTE, full-pool final evaluation, parameter searches and
  learned full-pool OT remain outside this implementation scope.

Both matrices use [`run_ot_followups.py`](../pipeline/run_ot_followups.py), with
separate frozen configs. Deterministic banks, fixed real-only geometry weights,
100-class reporting, image updates, recovery and incremental publication are
implemented. The next step is Kaggle training, C2 first, then C1. Read PLAN.md
for exact settings, decision criteria and conditional budgets.

Local verification passed both complete matrices on artificial data. The full
CPU suite passed 378 tests with one optional skip; affected checks were rerun
after review fixes. Both pinned-input preflights passed with zero scientific
fits. C2 reproduced 2,726 added rows and 11,442 total rows per representation.
C1 reconstructed exact pixels/labels and replayed all six fitting/development
feature populations against the published caches (rtol 1e-4, atol 1e-5;
maximum absolute difference 1.72e-5 across all seeds). No official test rows
were loaded and no scientific continuation/head was trained locally.

The local and historical compressed NPZ file hashes differ. Image validation
therefore pins canonical pixel/label array hashes derived from the verified
official training archive and the audit's unchanged preparation code, along
with exact split metadata and cache row labels. Local NPZ files keep their own
file-hash receipts on reuse. Tests reject changed pixels while allowing an
equivalent container serialization. CUDA execution and actual Kaggle access
remain unverified until the user runs the notebooks.

## Kaggle and GitHub

The [Kaggle ablation script](../scripts/kaggle_ablation_study.py) now defaults to
`STAGES = ("synthetic_reweighting",)`: 30 C2 CPU head fits, no CIFAR attachment.
Use a fresh notebook with Internet and the existing `GITHUB_TOKEN` secret.
It restores the pinned audit inputs and publishes to
`ablation-synthetic-reweighting-results`. For C1, use another fresh notebook,
`STAGES = ("ot_contribution",)`, a GPU and CIFAR-100 training images (or
`ALLOW_DOWNLOAD = True`); its 18 continuations publish to
`ablation-ot-contribution-results`. Both use `FOLLOWUP_PREFLIGHT_ONLY = True`
for checks without fitting/publication. See
[Stage C delivery](PLAN.md#implemented-stage-c-commands-and-kaggle-delivery).

The completed diagnostic remains available as `STAGES = ("controlled_bias",)`,
with its own CPU preparation, result branch and `CONTROLLED_BIAS_PREFLIGHT_ONLY`
setting. Its source/results identities above remain historical references.

Use `STAGES = ("simple_controls",)` to reproduce completed Stage B on a Tesla T4.
It restores the exact audit inputs and pinned Stage A references and publishes
to `ablation-simple-controls-results`. Set `SIMPLE_CONTROLS_PREFLIGHT_ONLY = True`
for its input/reference/geometry checks only.
The [Stage B guide](weight_simple_controls.md) documents the fixed environment,
reports and recovery. Stage A remains available explicitly with
`STAGES = ("assignment",)`.

Stage B results and complete assets are published at the pinned results commit
above. Keep its historical scores separate from the new diagnostic's fresh
34-class heads. Controlled-bias results are pinned above; the new Stage C
studies never force its failed gate open or reuse its scores as controls.

The completed audit results branch is `ablation-rebuilt-audit-results`. Look for
`reports/weight_audit.json` and `reports/weight_artifacts.json`. The latter
links to a GitHub release containing the exact baseline checkpoints and the
fitting/development feature caches. The inputs are pinned in PLAN.md to results
commit `51c7829d505143bd556ff1f3038b6559751e3428`; verify the downloaded archive
checksum before reuse. The local preflight verified the archive bytes and all
three seeds: 8,716 fitting and 2,131 development examples per seed, with
75/70/73 OT-active classes. Stages A/B completed on a Tesla T4 with
Python 3.12, PyTorch 2.10.0+cu128, and NumPy 2.0.2. Stage B checks the same
training environment before reusing those references.
Audit rebuilding is still available explicitly through
`STAGES = ("rebuild_audit",)`.
See [the Kaggle guide](PLAN.md#kaggle-and-github-results) for other runner modes.

The user runs full training on Kaggle. Publish results and reusable inputs to
GitHub so the next step does not depend on finding files from an old notebook.
Explain results plainly and discuss experimental changes before implementing
them. Local tests use artificial data and check the workflow rather than
research performance.
