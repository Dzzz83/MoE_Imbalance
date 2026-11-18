# OT reweighting ablation study

Branch: `ablation_study`.

This is an experiment plan, not a report of completed research runs. The staged
runner is implemented in `pipeline/run_ablation.py`. Full-data training has not
started.

## Goal

Understand why prototype Sinkhorn weighting adds little beyond ordinary
class-balanced weighting in the archived experiments. Test three questions:

1. Does the classifier train enough for different weights to matter?
2. Does OT assign larger weights to useful examples?
3. Does reducing the strength of OT weighting help?

Start with the prototype Sinkhorn method implemented in
`pipeline/methods/prototype_weights.py`. SAVA, learned-score `ot_finetuning`,
synthesis, and backbone fine-tuning are outside the first study.

## Data split

- Start with the existing retained CIFAR-100 pool: 10,847 images, preparation
  seed 42, imbalance factor 0.01. Do not use discarded training images.
- Make one fixed, stratified split using split seed 43, separate from the
  preparation and training seeds.
- For each class with `n` retained images, hold out `max(1, floor(0.2 * n))`
  development images. Keep the rest for fitting. The current pool has at least
  five images per class, so a five-image class keeps four for fitting and one
  for development.
- Save fitting/development positions in retained-array order, the split seed,
  and per-class counts. Check that the sets are disjoint and cover the pool.
- Reuse exactly this split for every condition and training seed.

Development images must not enter baseline pretraining, feature-based fitting,
prototype construction, OT references, or classifier training. Extract their
features only for evaluation, using evaluation mode and deterministic transforms.
Existing full-pool baselines cannot be used for this development study because
they have already trained on the held-out images.

Use actual fitting counts to compute weights and save training metadata. Keep
reporting groups fixed using the original retained counts: Head >=100, Medium
20–99, Tail <20. Save those original counts separately for development reporting;
do not substitute them for actual training counts in checkpoints.

## Common training setup

- Train a new baseline on the fitting subset for each training seed: 0, 1, 2.
  Use the existing 200-epoch warm-up/multistep baseline recipe.
- For each seed, start every continuation from the same final baseline,
  including its existing classifier. Start a fresh classifier optimizer.
- Freeze the backbone, including batch-normalization state, and cache
  deterministic raw fitting features in retained-position order.
- Keep batch order, batch size, optimizer settings, and epoch count identical
  across weighting conditions. Include the final partial batch.
- Give weight shuffling a separate random generator so it cannot affect
  clustering, model initialization, or classifier batch order.
- Save the final scheduled model. Do not select an epoch by test performance.

## Stage 1: classifier schedule pilot

Use the seed-0 fitting baseline and class-balanced weights. Compare constant
classifier learning rates `0.00002`, `0.0002`, `0.002`, and `0.02`.

Keep the existing continuation settings otherwise: 40 epochs, batch size 128,
SGD momentum 0.9, weight decay 0.0005, and Nesterov enabled. Record training
behavior and development metrics each epoch.

Choose the rate with the highest final-epoch development balanced accuracy;
break an exact tie using the smaller rate. A nonfinite run is a failed run,
not a candidate. Large classifier movement alone is not evidence of useful
learning. If learning remains unresolved, investigate before the main study.
Record any additions to the pilot and the reason for them.

Freeze the chosen schedule before the main comparison. This chooses a schedule
using class-balanced training; conclusions about OT are conditional on that
schedule. Check a neighboring rate later if an important conclusion appears
dependent on optimization. Do not tune a separate schedule for each condition
in the first comparison.

## Stage 2: weighting experiments

Let `C = 100`, `n_c` be the fitting count for class `c`, and `w_cb[i] =
1 / (C * n_c)`. All four main conditions give each class total mass `1 / C`.

| Condition | Assignment | Question |
|---|---|---|
| Class-balanced | Equal weights within each class: `w_cb` | Main reference |
| OT | Original prototype Sinkhorn weights: `w_ot` | Does OT add value? |
| Shuffled OT | Permute `w_ot` within each class | Does matching weights to particular examples help? |
| Half-strength OT | `0.5 * w_cb + 0.5 * w_ot` | Is full OT weighting too strong? |

Compute OT weights once per baseline seed. Derive shuffled and half-strength
weights from that same vector; do not rebuild prototypes for each condition.
Keep weights fixed throughout each continuation. Use the existing globally
weighted cross-entropy, without renormalizing weights inside each batch.

Keep the current prototype defaults: two clusters, minimum cluster size 5,
entropy `epsilon = 0.1`, row KL strength 0.2, iteration limit 200, and tolerance
0.001. Leave the current class-balanced fallback rule unchanged.

Use three shuffles per training seed, with shuffle seeds 0, 1, 2. Also run a
uniform continuation as context and evaluate the starting baseline. Uniform
continuation gives equal weight to images, not equal total weight to classes.

This makes seven continuations per training seed: class-balanced, OT, three
shuffled OT runs, half-strength OT, and uniform. The main comparison has 21
continuations and three fitting-subset baselines. The pilot is additional;
reuse its selected seed-0 class-balanced run if all settings match exactly.

## Measurements

### Prediction quality

- Primary: development balanced accuracy, the mean recall across all classes.
- Secondary: macro Head, Medium, and Tail recall, plus per-class recall changes.
- Record overall accuracy, but do not use it to choose settings: the proportional
  development split is imbalanced.
- Record a common unweighted cross-entropy on development images. Weighted
  training losses optimize different objectives and are not directly comparable.
- Keep JSON scores as fractions and displayed accuracy as percentages; report
  differences in percentage points.

A tail class may have only one development image. Interpret group averages and
matched-run patterns; do not draw strong conclusions from one tiny class.

### Weights and training behavior

- Class weight totals and finite, positive weights.
- Per-class weight multipliers relative to class-balanced weighting, including
  minimum, maximum, and spread.
- Effective sample size per class: `(sum(w) ** 2) / sum(w ** 2)`. Also report
  its fraction of the fitting count; equal within-class weights give fraction 1.
- Classifier parameter movement from its starting state and training loss.
- Sinkhorn convergence, column mass errors, and cluster sizes.
- Whether each class actually used OT or fell back to class-balanced weights,
  with a reason for fallback and counts by Head/Medium/Tail group.

For each baseline seed, fix OT-active and fallback class membership using its
original OT solve. Add separate macro recall comparisons for these subsets,
including their class counts. An empty subset has no score. These are extra
diagnostics, not replacements for the primary metric or matched class groups.

## Analysis and interpretation

Compare conditions within each matched training seed, then summarize the paired
differences across seeds 0, 1, 2 using mean and sample SD. For shuffled OT, first
average the three shuffle results within a training seed; also show their
variation separately. Do not count shuffles as extra independent training seeds.

| Observation | Interpretation to investigate |
|---|---|
| OT weights are nearly class-balanced | OT makes little change; inspect regularization and fallback frequency |
| OT and shuffled OT are similar | No clear evidence that OT's assignment to specific examples helps |
| Shuffled OT is worse than class-balanced | Uneven weights can hurt; OT needs useful assignment to offset this |
| Half-strength OT beats OT and class-balanced | A useful signal may be weakened by excessive emphasis |
| Many tail classes fall back | The study tests little actual OT weighting on those classes |
| Conclusions change with learning rate | Investigate optimization sensitivity before attributing results to OT |

These patterns suggest explanations; none alone proves a cause. Three training
seeds measure training variability, not variability from choosing a different
data split. Consider a second fixed split only as a later robustness check.

## Final evaluation

Use development results to choose and document the next method or recipe. Keep
a record of all attempted settings, including failed and negative experiments.
Freeze the recipe before evaluating new final results on the official test set.

Then train the selected method and its planned controls on the full retained
pool using matched seeds. Compatible existing full-pool baselines may be reused.
Recompute features and weights using all retained training examples. Evaluate
the final scheduled checkpoints on the original balanced 10,000-image test set.
On this set, overall and balanced accuracy are equal; report Head/Medium/Tail
recall too, using original retained counts.

Earlier test results already informed this research direction. Disclose that
history rather than describe the benchmark as never previously inspected. Do
not use new test results to select the best epoch or tune the final recipe.

## Implementation and execution checklist

- [x] Implement saving and validation of the fixed split and per-class counts.
- [x] Support fitting-only baselines and development evaluation without changing
  the existing full-pool experiment protocol.
- [x] Add fixed weight transformations and OT-active/fallback diagnostics.
- [x] Verify shuffle preserves each class's weight multiset and total mass;
  verify half-strength weights preserve mass and recover the expected midpoint.
- [x] Verify row/weight alignment, isolated shuffle randomness, frozen backbone
  state, and exclusion of development images from all fitting stages.
- [x] Reuse relevant existing tests and add focused checks for new behavior;
  perform a small CPU integration check before GPU experiments.
- [ ] Run the schedule pilot, record results, and freeze main-study settings.
- [ ] Run the matched-seed ablations and summarize paired differences.
- [ ] Choose one targeted follow-up based on the observed diagnostics.
- [ ] Freeze the final recipe and run the full-pool test comparison.

Use separate output folders under `outputs/ablation_study/`; do not overwrite
archived results or existing checkpoints. Save resolved configurations, split
information, source checkpoint identity, weights, diagnostics, histories, and
metrics. Do not add datasets or checkpoints to version control. Reuse the
existing method functions and classifier runner where practical; no run registry
or new trainer framework is planned.

## Running the study

Run from the repository root with the configured Python environment active.
Prepare the retained pool first using the README instructions. Settings are in
`configs/ablation_study.json`; defaults match this plan. Epoch/seed overrides are
available for artificial-data checks, not evidence about full-data performance.

```bash
python -m pipeline.run_ablation prepare
python -m pipeline.run_ablation pilot --device cuda
python -m pipeline.run_ablation study --device cuda
```

The pilot chooses the learning rate from its final-epoch development scores.
Inspect `pilot_results.json`, `selected_schedule.json`, `study_results.json`,
and `study_summary.json` under `outputs/ablation_study/`. Summary scores are
fractions; paired differences are also fractions (multiply by 100 for percentage
points). The summary includes baseline and uniform context alongside the four
main conditions. Full per-epoch development metrics are in continuation histories.
OT-active/fallback coverage and per-class effective sample sizes are in each
run's `weight_diagnostics.json`. Raw features and original OT weights are saved
once per baseline seed in `caches/seedN/features_weights.pt`.

After development is complete and the common recipe is settled:

```bash
python -m pipeline.run_ablation freeze
python -m pipeline.run_ablation final --device cuda --data-root data/raw
```

`freeze` records the selected schedule and development result hashes. The first
implementation carries all planned conditions into the final full-pool comparison;
it does not automatically select an OT winner. `final` trains full-pool baselines
and continuations, then evaluates final checkpoints on the official test set.
Add `--download` only to explicitly allow downloading that test set. Final
results and summaries are saved under `final/`. Development stages never load
the official test set.

All stages accept `--config` and `--output-dir`. Use the same values throughout
one study. Completed runs are reused only when saved inputs and output hashes
match. Source-file hashes prevent mixing changed implementations in one study.
Interrupted runs are preserved and rejected; there is no epoch resume. Choose
a new study folder rather than overwrite them. Nonfinite pilot runs are recorded
as failed and excluded from selection; other errors stop execution.

Optional `final_baselines` maps training seed strings to existing full-pool
baseline checkpoints, for example `{"0": "outputs/baseline_seed0/final.pt"}`.
Set this before starting the study. Recipe, seed, counts, and preparation
settings must match. As with existing continuation commands, metadata alone
cannot prove image identity: only supply checkpoints trained on the exact
retained pool. Their file hashes are recorded in downstream cache inputs.

### Kaggle and GitHub results

Use `scripts/kaggle_ablation_study.py` as a notebook cell. Commit and push the
implementation on `ablation_study` first: Kaggle clones published source and
cannot see uncommitted local changes. Enable a GPU, Internet, and the Kaggle
secret `GITHUB_TOKEN` with write access to `Dzzz83/MoE_Imbalance`. Attach CIFAR
and set `RAW_CIFAR_ROOT` to its directory, or explicitly set
`ALLOW_DOWNLOAD = True`.

The script defaults to `STAGES = ("prepare", "pilot", "study")`, one GPU, seeds
0/1/2, and two data-loader workers. It stops after development. Results are
pushed after completed baselines, pilot runs, and ablation runs to the separate
`ablation-study-results` branch. No GitHub operation is performed by the local
implementation tests.

On the first run, `SOURCE_REF = "ablation_study"` is resolved to an exact commit
and recorded in the results manifest. Later sessions reuse that commit, even
if the source branch has moved. Alternatively set `SOURCE_COMMIT` to a full
published hash before the first run. Changing the implementation requires a new
study and results branch: change `WORK_ROOT`, `RESULTS_ROOT`, and
`RESULTS_BRANCH` together. Existing results are not force-pushed over.

GitHub receives JSON reports under `reports/`: split positions, settings,
histories, weight diagnostics, metrics, summaries, and provenance. It does not
receive images, cached features, or model checkpoints. Git authentication uses
a temporary askpass helper; the token is passed only to authenticated Git
subprocesses and is not embedded in URLs or saved reports.

Preserve `/kaggle/working/ot-ablation/` in Kaggle outputs. It contains `study/`
and `data/`, including the prepared pool, caches, and checkpoints needed for
reuse. For another session, attach those saved artifacts as a Kaggle dataset
and set `RESTORE_ROOT` to the directory containing `study/` and `data/`. GitHub
reports alone cannot restore training. The script rejects starting a fresh
local study over an existing results branch without those artifacts.

Interrupted runs have no epoch resume. By default, the script moves their
partial folders to `ot-ablation/interruptions/` and restarts those runs from
their starting model, preserving completed runs. Set
`RESTART_INTERRUPTED_RUNS = False` to stop instead. Keep settings and artifact
paths stable when restoring a study.

After reviewing development results, rerun with `STAGES = ("freeze", "final")`
in the same workspace, or restore it first. This explicitly starts the planned
full-pool training and final test evaluation. Existing historical full-pool
baselines can be supplied through `CONFIG_OVERRIDES["final_baselines"]` before
starting the study; they cannot replace the fitting-only development baselines.

### Stronger OT follow-up

The first study found little difference between class-balanced, OT, and shuffled
OT weights. To test whether the original weight differences were too small,
cube the saved OT weights within each class and normalize them back to the same
class total. Cubing preserves the example ranking and leaves equal weights
equal, including in the fallback classes.

Reuse the saved fitting baselines, feature caches, development split, and
selected classifier schedule. Train one classifier with stronger OT and three
with the stronger weights shuffled within their classes, for each of the
three training seeds. This is 12 new classifier runs. The original OT and
class-balanced results are the references. Use development balanced accuracy
and the paired comparison to the mean of the shuffles for each seed. Review
weight spread and Head/Medium/Tail recall alongside the primary score.

```bash
python -m pipeline.run_ablation strength --output-dir outputs/ablation_study --device cuda
```

On Kaggle, paste the updated runner into a new notebook session and set
`STAGES = ("strength",)`. Use the completed development workspace from the
first run, or attach its saved output and point `RESTORE_ROOT` to the directory
containing `study/`. This stage uses the saved study settings directly and can
run without `data/train.npz`. The updated source must be
published on `ablation_study`. The runner saves the follow-up in
`study/strength/` and pushes its JSON reports to `ablation-strength-results`.
Keep the updated workspace to resume a partially published follow-up.

Implementation checks use artificial images and do not download CIFAR:

```bash
python -m pytest tests/test_ablation.py tests/test_prototype_weights.py tests/test_finetuning.py
python -m pytest tests/test_kaggle_ablation_study.py
```

CPU verification included a full-suite run (304 tests passed), followed by all
19 ablation tests after fixing pilot folder names for closely spaced learning
rates. The artificial-data integration check exercises all five stages,
including mocked final test data. After adding Kaggle publishing, all 19
ablation tests passed again, and all 9 Kaggle-script tests passed, including
local Git clone/push checks. No real GitHub push or Kaggle GPU run was performed;
full CIFAR training and CUDA behavior remain unverified.
