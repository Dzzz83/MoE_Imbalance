# OT reweighting ablation study

Branch: `ablation_study`.

This guide records the experiment protocol and commands. The runner is
implemented in `pipeline/run_ablation.py`, with Stages A/B in
`pipeline/run_weight_study.py`. The next
[controlled-bias diagnostic](#controlled-bias-diagnostic-the-next-experiment)
is a detailed protocol awaiting implementation. See the
[research handoff](HANDOFF.md) for the earlier studies.

## Goal

Decide whether prototype Sinkhorn reweighting merits further research, identify
why it adds little beyond class balancing, and establish which changes deserve
confirmation. The original development study tested three questions:

1. Does the classifier train enough for different weights to matter?
2. Does OT assign larger weights to useful examples?
3. Does reducing the strength of OT weighting help?

Continue with the prototype Sinkhorn method implemented in
`pipeline/methods/prototype_weights.py`. SAVA, learned-score `ot_finetuning`,
and synthesis are outside this study. Backbone fine-tuning is a conditional
later experiment. Conclusions apply to the tested prototype method and setting;
they do not establish whether every form of OT reweighting is useful.

After Stages A/B, first establish whether a known within-class sampling bias
creates useful reweighting headroom, then compare OT with an informed simple
correction. This controlled diagnostic keeps the retained pool and frozen
features, but uses a separate Head-class task and fresh linear classifiers.
Its explicitly documented data, solver, and reporting rules below override
the historical continuation setup only for that diagnostic.

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

These are the historical image-training and continuation rules. The controlled
diagnostic below uses already published backbones and a deterministic fresh-head
fit; it does not train replacement baselines or reuse historical head scores.

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

## Study status

The pilot, main development ablation, stronger OT follow-up, and rebuilt weight
audit are complete. The audit found that the highest-weight quarter had a mean
confidence percentile about 14.5, 16.9, and 15.0 percentage points above the
class average for seeds 0, 1, and 2. This describes fitting examples; it does
not establish that favoring them caused the weak development results.
See the [research handoff](HANDOFF.md#completed-experiments) for source reports.

Stage A is complete: all 27 assignment continuations ran for 40 epochs on Kaggle.
The [pinned results](https://github.com/Dzzz83/MoE_Imbalance/blob/0d4cf3e90be73513ddbe5c731698610b0285773c/reports/study_summary.json)
give class-balanced/original OT/strong OT/easy/boundary/hard means of
38.464/38.708/38.503/38.777/38.075/38.032% balanced accuracy. Boundary weighting
failed its primary practical target. Easy weighting beat boundary and hard
weighting in all three seeds, but beat original OT in only one; it remains an
exploratory candidate. No tested condition improved on class balancing by the
planned 0.5 percentage points on average.

Stage B is complete: all 15 new continuations ran for 40 epochs. The
[pinned results](https://github.com/Dzzz83/MoE_Imbalance/blob/451349aa463cd5b7ec37ea3a97fd99b8bf50a5d2/reports/study_results.json)
give cluster-balanced/prototype-distance/easy-rank/cluster-rank-strong/
distance-rank-strong means of 38.287/38.521/38.731/38.152/38.533%.
No new condition cleared the 0.5-point improvement target over class balancing.
Easy-rank's +0.267-point mean includes +0.333 points contributed by one newly
correct one-image development class in seed 1. These are exploratory findings,
not a verdict on all OT reweighting.

Stage B used source `b4cea1c8b68eb7bd872f1b5500e8e4d9038ecf69`. Analysis
verified the release archives, all 15 frozen backbones and weight invariants,
reproduced final accuracy/recall from saved heads, and recomputed the published
paired contrasts. The [Stage A](weight_assignment.md) and
[Stage B](weight_simple_controls.md) guides document those completed stages.

The next authorized design is the controlled-bias diagnostic below: first
nine fresh-head fits to check damage and recoverable headroom, then at most
18 further fits to test the target and transport. The diagnostic is implemented;
the scientific fits have not run. Final confirmation and official test
evaluation remain separate.

Use separate output folders under `outputs/ablation_study/`; do not overwrite
archived results or existing checkpoints. Save resolved configurations, split
information, source checkpoint identity, weights, diagnostics, histories, and
metrics. Do not add datasets or checkpoints to version control. Reuse the
existing method functions and classifier runner where practical; no run registry
or new trainer framework is planned.

## Next investigation: assignment, transport value, and confirmation

See [project context](project_context.md#project-structure) for component
responsibilities, and the [Stage A](weight_assignment.md) and
[Stage B](weight_simple_controls.md) guides for implemented commands and
recovery. The protocol and implementation checklist for the next controlled
diagnostic and its implemented CPU/Kaggle commands are recorded below.

### Questions and order

1. **Assignment:** does emphasizing uncertain examples help when the weight
   distribution is unchanged?
2. **Transport value:** can simple rules explain or match the OT result?
3. **Controlled headroom and target:** can a known correction repair introduced
   sampling bias, and does reference-informed OT improve beyond an informed
   simple rule and its own shuffled weights?
4. **Remaining limits, conditional:** is a specific limitation in optimization,
   Tail coverage, or the frozen representation worth a separate experiment?
5. **Confirmation:** does a frozen candidate improve on matched controls on a
   second development split?

Run one stage, publish its results, and discuss the evidence before selecting
the next conditional experiment. Record negative and failed runs. Avoid a large
parameter search across all these factors.

### Inputs and fixed protocol

The shared input identity applies to the controlled diagnostic too. The
40-epoch continuation settings in this section apply to Stages A/B; the
controlled diagnostic specifies its fresh-head objective and solver separately.

Reuse the exact rebuilt audit inputs from results commit
`51c7829d505143bd556ff1f3038b6559751e3428` on
[`ablation-rebuilt-audit-results`](https://github.com/Dzzz83/MoE_Imbalance/tree/51c7829d505143bd556ff1f3038b6559751e3428).
Its `reports/weight_artifacts.json` links to the checkpoint/cache archive with
SHA-256 `e157e664288bd849a0dc5d6fb5eed2f61269a3452327d9a1407145a02331aeb2`.
Verify the downloaded bytes against that hash, then validate each cache against
the audit's seed-specific hashes and its recorded baseline checkpoint identity.
Reuse the published split and original retained class counts. Recreate prepared
image subsets only if the current training interface requires them; verify their
hashes against the cache inputs before training. Missing or incompatible inputs
must stop the run rather than silently train replacement baselines.

These are newly rebuilt audit baselines, distinct from both earlier classifier
studies. Stage A reran every comparison on these inputs; do not use continuations
from the older studies as matched controls. Stage B may reuse Stage A's exact
continuations after the parent compatibility checks below.

- Training seeds 0, 1, 2; preparation seed 42; development split seed 43.
- Constant classifier learning rate 0.002; 40 epochs; batch size 128; SGD
  momentum 0.9, weight decay 0.0005, and Nesterov enabled.
- Restore the full baseline for every condition and start a fresh optimizer.
  Freeze backbone parameters and batch-normalization state.
- Reuse raw deterministic fitting/development features. Use identical seeded
  batch order, globally weighted cross-entropy, and the final partial batch.
- Keep weights fixed during training. Use final-epoch development metrics;
  per-epoch measurements are diagnostics, not checkpoint selection.
- Preserve original OT-active/fallback class membership in the assignment
  experiment. Fallback classes keep equal weights in every condition.
- Development labels are for evaluation only. Fitting labels and baseline
  predictions may define assignment rules. No official test access occurs.

### Stage A: change which examples receive the weights

For each fitting example, compute the baseline margin once:
`m[i] = logit[i, true_class] - max(logit[i, other_classes])`.
Positive margins favor the correct label; negative margins indicate mistakes.
These are in-sample scores from the baseline, not independent measures of
image quality, label correctness, or distance to a geometric decision boundary.

Derive stronger weights from the saved original OT vector using the existing
exponent 3, restoring each class total to 1/100. For every reassignment below,
sort this same stronger weight vector within each class and assign the largest
weights to the highest-priority examples. Preserve the complete multiset of
weights, including repeated values. This preserves class mass, spread, and
effective sample size exactly apart from floating-point tolerance.

| Condition | Priority for the largest weights | Runs per baseline seed |
|---|---|---|
| Class-balanced | Equal weight to every example in a class | 1 |
| Original OT | Saved original assignment and strength | 1 |
| Strong OT | Original assignment with exponent 3 | 1 |
| Shuffled strong OT | Random within-class assignment; shuffle seeds 0, 1, 2 | 3 |
| Easy strong | Larger signed margin `m` first | 1 |
| Boundary strong | Smaller absolute margin `abs(m)` first | 1 |
| Hard strong | Smaller signed margin `m` first | 1 |

This is nine continuations per seed, **27 runs total**, with no new baseline
training. Evaluate the starting baseline once per seed as context. Half-strength
and uniform continuations are already historical diagnostics and are not added
to this stage.

Break exact score ties with a saved seeded permutation independent of the
training and shuffle generators. Use the same tie order for all ranked rules
within a baseline seed; save its seed and ordering convention. Equal-weight
fallbacks must remain unchanged. Reject nonfinite scores or invalid weights.

The primary hypothesis is that boundary assignment improves on strong OT.
Report its paired differences against strong OT, class-balanced, and the
within-seed mean of shuffled strong OT. Easy and hard assignments are secondary
mechanism checks. If either is selected after seeing results, label it exploratory
and require new confirmation rather than treating it as a prespecified winner.

| Result | Interpretation and next step |
|---|---|
| Boundary beats strong OT, shuffled, and class-balanced | Evidence that uncertainty-based assignment is promising; test simple alternatives in Stage B |
| Boundary beats strong OT but not class-balanced | Assignment may repair a disadvantage without improving the useful reference |
| Easy matches OT while boundary/hard improve | Supports the easy-example preference explanation; does not isolate every property of OT |
| Hard hurts while boundary helps | Consistent with useful uncertain examples differing from extreme errors; does not prove label noise |
| All assignments are similar | No demonstrated assignment benefit at this strength and schedule; inspect whether training changed meaningfully |
| Original OT beats strong variants | Revisit weight strength before claiming assignment is the main limitation |

A boundary or hard win establishes a reweighting direction. Because these
conditions reuse OT's weight values, it does not by itself establish an advantage
from solving transport.

### Stage B: determine whether transport adds useful information

#### Purpose and scope

Determine whether prototype OT offers useful information beyond simple cluster,
distance, or confidence rules. Stage A supplied only weak positive signals for
original OT and easy weighting. This stage is a bounded exploratory comparison
authorized after that result; it does not change the 0.5-point practical target
or treat easy weighting as a confirmed improvement.

Add five conditions for seeds 0/1/2: **15 new classifier continuations**, with
no new baseline training, CIFAR download, learning-rate pilot, or official test
evaluation. Keep split 43, the frozen backbone/BN, batch order, optimizer,
40 epochs, learning rate 0.002, and final-epoch selection unchanged. Every class
still receives total mass 1/100. Keep the original active/fallback mask fixed;
the 25/30/27 fallback classes for seeds 0/1/2 remain uniformly weighted.

#### Pinned inputs and existing references

Reuse the original audit archive and checksums in the common protocol. Pin the
Stage A parent to results commit `0d4cf3e90be73513ddbe5c731698610b0285773c` on
`ablation-assignment-results`, produced with source commit
`a792b81ef737e8fba54033ffb4cc72a92c6b2e2a`. Its `reports/study_assets.json`
links to the complete continuation archive with SHA-256
`23f217c28b72e82b91428a9024ec4cc890cd510470de4fe5d59e01ed67d51194`.

Verify report/asset hashes, completion receipts, seed-specific baseline/cache
identities, and resolved training settings before reusing references. Reuse
class-balanced, original OT, strong OT, the three strong OT shuffles per seed,
and easy-strong results; retain boundary/hard and the starting baseline as
context. These are matched references on the same audit baselines, not the
older development/strength studies.

Keep the shared loss, optimizer, loader, evaluator, and model implementation
compatible with the parent. Use the parent training environment (Python 3.12,
PyTorch 2.10.0+cu128, NumPy 2.0.2, Tesla T4, two CPU threads, and the recorded
backend flags). If a training-relevant implementation or environment change
invalidates reuse, stop and document a reference-rerun budget before starting;
do not silently combine the old scores with new training conditions.

#### Five new conditions

Normalize fitting features and prototype centers as in the existing method.
Replay the original two-prototype clustering with baseline seed and minimum
cluster size 5. Require exact agreement in active/fallback status and cluster
sizes, and agreement between replayed and cached original OT weights with
`rtol=1e-5`, `atol=1e-8`. Stop on mismatch. Save centers and assignments once
per seed and reuse this geometry for every cluster/distance condition.

Let `d[i]` be minimum cosine distance to either center, and `s[i]` the size of
the cluster assigned to example `i`. The native rules below calculate weights
without transport; all normalization is within class.

| Condition | Rule | What it tests |
|---|---|---|
| `cluster_balanced` | Give each cluster half the class mass: `w[i] = 1 / (200 * s[i])` | Whether simple balancing of the two groups is sufficient |
| `prototype_distance` | Normalize `exp(-d[i] / 0.1)` to class mass 1/100 | Whether closeness to a center is sufficient |
| `easy_rank` | Normalize `exp(log(4) * r[i])`, using the baseline correct-class margin's scaled midrank `r[i]` | Whether easy-example weighting works without OT-derived values |
| `cluster_rank_strong` | Reassign the saved stronger OT weight multiset using priority `1 / s[i]` | Cluster-size assignment at exactly the same strength as strong OT |
| `distance_rank_strong` | Reassign that same multiset using priority `-d[i]` | Center-distance assignment at exactly the same strength as strong OT |

For `easy_rank`, use the fitting margins saved by Stage A. Equal margins share
their average rank; scale nonconstant midranks using
`r = (rank - min_rank) / (max_rank - min_rank)`. An all-tied class is uniform.
The factor 4 and distance temperature 0.1 are fixed from the earlier design;
do not tune either on these outcomes. Choosing easy rather than boundary
priority follows Stage A and is explicitly exploratory.

For both stronger-rank conditions, use the parent's saved stronger weight vector
and tie permutation. Larger priority receives larger weight; exact priority
ties use the same saved order. Verify exact within-class sorted-weight equality
against strong OT and report how many classes have tied/constant priorities.
Do not renormalize a permuted vector. The native controls need not match OT's
weight spread; report that difference rather than treating them as a pure
assignment experiment. Use stable float64 normalization, cast once to the
reference dtype, and reject nonfinite/nonpositive weights.

#### Comparisons and measurements

Primary outcome remains final development balanced accuracy. Prespecify all
three original-OT-minus-native-control contrasts: against `cluster_balanced`,
`prototype_distance`, and `easy_rank`. Also report each new condition versus
class-balanced so a weak reference cannot make a condition appear useful.

Secondary mechanism contrasts are strong OT versus each stronger-rank control
and versus its existing shuffle mean; each stronger-rank control versus that
shuffle mean; and easy-strong versus `easy_rank`. The last is a complete-recipe
comparison because the weight spectra differ, not an isolated assignment test.
Do not use original OT versus a shuffled stronger vector to claim an assignment
advantage: those weight distributions differ.

Show all three paired seed differences, mean and sample SD. Average parent
shuffles within seed before comparisons. Report Head/Medium/Tail recall,
original active/fallback recall, per-class support/correct counts, common
development cross-entropy, and weight/assignment diagnostics. Break down overall
changes into active/fallback contributions within each seed. Highlight any gain
driven by one-image development classes; do not remove those classes or select
a new metric after seeing results. Verify the intended priority, fixed class
mass, and stronger-rank multisets before interpreting training outcomes.

#### Decision rules and limits

Keep 0.5 percentage points as the practical target, with positive paired
differences on at least two of three seeds for a promising contrast. These
are research-budget rules, not significance or equivalence tests. Report every
prespecified comparison, including negative and inconclusive ones.

- A native rule meeting that target versus class-balanced and original OT
  merits consideration for a frozen, separate-split confirmation. Selection
  among the new rules remains exploratory.
- OT outperforming native controls would support the complete tested OT recipe.
  Native controls also change weight spread, so that result alone cannot isolate
  a benefit from transport's assignment. Use the matched stronger-rank and
  shuffled comparisons for assignment claims. The parent's original OT gain of
  only 0.24 points versus class-balanced still falls short of the practical
  target; do not promote it automatically because a simple control performs poorly.
- If OT shows no consistent practical advantage over the simpler recipes and
  no candidate clears the usefulness target, deprioritize the current prototype
  OT path or record the result as unresolved. Small observed differences alone
  do not prove methods equivalent or rule out other OT formulations.

Review these outcomes before selecting any Stage C route. Do not launch a
parameter sweep, backbone experiment, second split, or final test evaluation
automatically. This stage does not add native-rule shuffles: it tests complete
non-OT recipes. A later assignment claim for a different native weight spectrum
requires its own matched shuffles and a separately recorded budget.

#### Implementation and Kaggle delivery

1. Add `configs/weight_simple_controls.json` and `simple_controls` dispatch to
   the cached study runner. Reuse the shared classifier loop and parent-receipt
   validation; preserve completed Stage A outputs and scientific identities.
2. Add deterministic geometry replay, the five weight rules, their invariants,
   and parent-linked paired reporting. Verify artificial cases with unequal and
   equal clusters, tied distances/margins, fallbacks, invalid inputs, missing
   parent runs, and changed source/settings. Reuse frozen-state, partial-batch,
   checkpoint, and interruption checks where behavior is unchanged.
3. Add explicit Kaggle input/reference restoration and incremental publication
   to `ablation-simple-controls-results`, using a fresh work folder such as
   `/kaggle/working/ot-simple-controls/`. Pin the new implementation at first
   execution. Publish resolved settings, geometry/weight diagnostics, histories,
   metrics, comparisons, and provenance as JSON; store reusable geometry, weights,
   and checkpoints in checksum-linked release assets. Reference original inputs
   and parent assets instead of duplicating their binaries in Git.
4. Perform a focused self-review and affected local tests, publish the source,
   and provide `STAGES = ("simple_controls",)` notebook settings. This is now
   the script default; the completed Stage A remains available explicitly as
   `STAGES = ("assignment",)`.
5. The user runs the 15 continuations on Kaggle. Read the complete published
   results and discuss the decision before implementing the next experiment.

### Controlled-bias diagnostic: the next experiment

#### Purpose, status, and limits

Status: **implemented; scientific fits not run**. This is the next experiment after
Stage B. It has two gates and at most **27 fresh linear-classifier fits**.
It uses cached features and can run on CPU. Historical Stage C routes below
remain deferred until this diagnostic is reviewed.

The questions are:

1. Does deliberately removing examples from one within-class feature subgroup
   damage development performance?
2. On exactly the same remaining examples, does the known sampling correction
   recover a useful part of that damage?
3. If so, does a reference-informed transport target improve on the current
   equal-prototype target, a simple reference-based correction, and matched
   shuffled OT weights?

The experiment concerns a controlled sampling intervention on well-populated
classes, conditional on existing representations. It does not establish that
the original long-tail data have this particular bias, test Tail-class coverage,
or reproduce the full learned-weight method in
[Guo et al.](guo_ot_reweighting.md). The new method below changes prototype
target masses; it does not implement their full meta-set weight-learning algorithm.

The backbones were pretrained on the original fitting pool, including examples
later reserved as references or removed from this diagnostic's classifier input.
They have not trained on development images. Therefore this tests
**classifier reweighting with a fixed representation**, not end-to-end learning
from a biased image dataset. Starting fresh heads avoids carrying the previously
trained classifier's knowledge of the removed examples into the comparison.

#### Fixed inputs and reproducibility

- Reuse the audit commit, archive checksum, three baseline/cache hashes, and
  split 43 from the shared input protocol above. No new CIFAR data or baselines.
- Pin the controller geometry to Stage B results commit
  `451349aa463cd5b7ec37ea3a97fd99b8bf50a5d2`. Its
  `reports/study_assets.json` links to the complete archive with SHA-256
  `c57f28b986321607270da729e2184ae6c9eab2d4808a642b49ea8b3338aac662`.
  Validate that archive and extract the saved `diagnostics/seed0/geometry.pt`;
  verify its hash against the pinned `diagnostics/seed0/controls.json`.
- Use feature/backbone seeds `0, 1, 2`. These are three fitted representations
  of the same images, not independent datasets or three sampling interventions.
- Fix `reference_seed=144`, `retention_seed=145`, and
  `biased_group_seed=146`. Give each role its own CPU generator.
  Shuffle seeds are `0, 1, 2`, also independent of preparation.
- Use one common set of reference, retained, and removed fitting-row positions
  across all conditions and all feature seeds. Verify identical label order
  and split identity before applying the positions to another seed's cache.
- Freeze the resolved protocol, preparation arrays, weight vectors, source
  hashes, Python/PyTorch/NumPy versions, CPU thread count, and numerical settings
  before the first classifier fit. Resume only compatible completed fits.

The diagnostic computes new head scores for every control. It can pin a new
homogeneous CPU environment; it does not need to reproduce Stage A/B's CUDA
environment or combine their head scores with its own.

#### Eligible classes and the controller's groups

1. Candidate classes are the 35 original Head classes, defined by retained
   count `>=100`. Use original counts, not post-removal counts.
2. Use the pinned seed-0 geometry on the full original fitting pool to define
   two **controller groups**, indexed `0, 1`, within each candidate class.
   Preserve the saved assignment order and center labels.
3. Require valid original clustering and at least **30 original fitting
   examples in each controller group**. This is a support rule, not a rule
   based on classifier performance.
4. Published seed-0 cluster-size metadata gives **34 eligible classes,
   original IDs 0 through 33**; class 34 has group sizes 23/59 and is excluded.
   Preparation must verify the criterion and this expected list against the
   pinned inputs. A disagreement is an input/geometry error, not permission
   to choose another subset.
5. Map the ascending eligible original IDs to `0..K-1`, where `K=34`,
   for a separate `64 -> K` linear head. Retain the inverse label map.
   Exclude other classes from this diagnostic's fitting, references, and scoring.

The original development split supplies at least 21 development images per
eligible class. A controller subgroup can still have little or no development
support; report its actual support and use null for an undefined subgroup score.
Do not change eligibility using development subgroup counts, errors, or scores.

Published counts imply 6,992 eligible fitting rows and 1,730 development rows.
The reference rule below reserves 1,372 fitting rows, leaving 5,620 rows in U.
The local preparation preflight verified these counts against the pinned arrays.
The fixed intervention retains **3,485 rows in T** and removes 2,135 from U.
Preparation verifies these inputs without fitting a scientific classifier.

Controller groups define the intervention and its evaluation diagnostics.
They are feature clusters, not established semantic categories. They are
privileged information for the known-correction control. Ordinary weighting
methods must fit their own geometry from their allowed inputs.

#### Reference, unbiased fitting, and biased fitting sets

Let `F` contain the eligible original fitting rows, and let `E` contain
their original development rows. Within each class, process controller group
0 then group 1 in ascending original fitting-row order.

1. In each controller group of size `n_g`, select
   `r_g = floor(0.2 * n_g)` rows uniformly without replacement using the
   reference generator. Their union is the reference set `R`.
2. Define `U = F minus R`. This is the **unbiased fitting context**:
   the classifier training set before introducing the sampling intervention.
   Here “unbiased” means before this introduced intervention, not a claim
   that the original dataset or representation is unbiased.
3. For each class, draw which controller group is `B` using one
   `randint(2)` from the biased-group generator, in ascending class order.
   The other group is `A`. Do not choose the more harmful direction.
4. Keep every row in `U_A`. From `U_B`, keep
   `k_B = ceil(len(U_B) / 4)` rows uniformly without replacement using the
   retention generator. Their union is biased fitting set `T`.
5. Record the actual inclusion probabilities:
   `p_i = 1` for group A and `p_i = k_B / len(U_B)` for group B.
   The latter is approximately 0.25, not necessarily exactly 0.25.

The support rule guarantees at least six reference rows per controller group,
at least 24 rows left in each `U` group, and at least six retained rows
from the downsampled group. Re-fitted method clusters can nevertheless differ
from controller groups or fail to form valid clusters; record that behavior.

Require `R`, `T`, and removed rows `U minus T` to be disjoint and cover
`F`. Require `F` and `E` to be disjoint. Save positions in original
fitting-cache coordinates, actual counts, probabilities, and RNG settings.
Selection uses fitting data only. No development features, labels, predictions,
or margins enter selection, reference fitting, or weight construction.

The reference **distribution** gives each eligible class mass `1/K`.
Reference counts may differ across classes; within a class each reference row
has equal mass. Its within-class subgroup mixture is observed, not forced
to 50/50. The finite reference set only estimates the desired mixture in `U`.

#### Information available to each condition

| Condition | Classifier fitting rows | Extra information allowed for weights |
|---|---|---|
| `unbiased_cb` | `U` | Class counts in U |
| `biased_cb` | `T` | Class counts in T |
| `known_correction` | `T` | Controller's actual inclusion probabilities and U class counts |
| `equal_target_ot` | `T` | Geometry learned from T; equal prototype quotas |
| `reference_target_ot` | `T` | Geometry learned from T plus reference features/labels R |
| `reference_cluster_ratio` | `T` | The same geometry and reference information as reference OT |
| `shuffled_reference_ot` | `T` | The reference OT vector, permuted within class |

`R` never contributes classifier-training cross-entropy, including in the
unbiased context. Development data are evaluation-only. Removed-row features,
controller centers/assignments, and true sampling probabilities are not passed
to the ordinary weighting builders. The unbiased context and known correction
are privileged diagnostic references, not fair competitors for an algorithm's
information efficiency or guaranteed performance upper bounds.

The representations themselves have the pretraining exposure stated above.
Do not describe the reference set as held out from backbone pretraining.

#### Fixed weight rules and method geometry

Every condition has total class mass `1/K` and total weight one. Let
`m_c = len(U_c)` and `t_c = len(T_c)`.

- `unbiased_cb`: `w_i = 1 / (K * m_c)` on U.
- `biased_cb`: `w_i = 1 / (K * t_c)` on T.
- `known_correction`: `w_i = 1 / (K * m_c * p_i)` on T.

For the known correction, fixed-size sampling makes
`sum(i in T_c, 1/p_i) = m_c` exactly in real arithmetic. Thus its class
mass is fixed, not a random normalization. For a fixed classifier, its
weighted empirical loss is unbiased over the prescribed sampling procedure
for the class-balanced loss on U. This statement does not guarantee an
unbiased fitted classifier, better development accuracy, or recovery of
information in removed examples.

Example: if U contains 100 A and 100 B examples, retaining 100 A and 25 B
gives relative weights 1 and 4. Normalize with the formula above so each
class still has mass `1/K`.

For each feature seed, fit one shared two-prototype geometry **using T only**:
L2-normalize raw features, use the existing spherical clustering procedure,
minimum cluster size 5, and a dedicated clustering generator seeded with that
feature seed. Process eligible classes in ascending order. Reuse this geometry
for equal-target OT, reference-target OT, and the simple reference rule.
Do not seed its centers from controller geometry or from removed examples.
Use T and R features from that same backbone seed for every method geometry
and reference assignment; centers from one backbone must not be applied to
another backbone's feature coordinates.

On a valid class, L2-normalize reference features and assign them to their
closest method center by cosine distance; break exact distance ties with the
lower center index.
Let `r_j` be the reference counts assigned to method cluster j, and define:

```text
q_j = (r_j + 0.5) / (len(R_c) + 1)
a_i = 1 / t_c
C_ij = max(0, 1 - normalized_feature_i dot normalized_center_j)
```

The fixed half-count smoothing keeps both target masses positive even when
no reference row is assigned to one method cluster. Use the same smoothed q
for OT and the simple rule. Save raw and smoothed reference proportions.

| Condition | Rule on a valid class |
|---|---|
| `equal_target_ot` | Existing relaxed-row Sinkhorn rule with prototype masses (0.5, 0.5) |
| `reference_target_ot` | Identical rule and geometry, changing only prototype masses to q |
| `reference_cluster_ratio` | `w_i = q[assigned_cluster_i] / (K * size(assigned_cluster_i))` |
| `shuffled_reference_ot` | Permute reference OT's exact saved values within class, separately for shuffle seeds 0/1/2 |

Use entropy `epsilon=0.1`, row KL strength `0.2`, uniform row reference a,
maximum 200 Sinkhorn iterations, and residual tolerance `0.001`. In log space,
replace the uniform log-column mass with `log(q_j)`; check relative column
error `max(abs(column_sum_j / q_j - 1)) <= 0.001`. Require the existing
column-scale update residual to meet the same tolerance. Use the transport plan's
row sums divided by K as weights. **No exponent-3 transformation.**

For a clustering fallback, all three geometry-based rules use `biased_cb`
for that class. Shuffling its constant vector changes nothing. Keep a common
active/fallback mask across these methods for each seed; it need not equal
the historical mask, because T differs from the historical fitting pool.
Do not drop fallback classes from the primary metric. Solver nonconvergence
or nonfinite values are technical failures, not silent fallbacks.

Keep feature normalization, spherical clustering, and cosine-cost construction
in the original cache dtype, matching the existing geometry procedure. Use
float64 for reference proportions, weight calculations, Sinkhorn, and head
optimization; preserve the original cached tensors. Require total and class
masses with `rtol=1e-10, atol=1e-12`, and reject nonpositive/nonfinite weights.
Save each vector once. Verify exact sorted-weight equality for every shuffle;
never renormalize a permuted vector. Require unchanged fallback vectors. The
simple rule must give each valid method cluster total mass `q_j/K`; OT's soft
column quotas alone do not guarantee the same mass in the hard-assigned groups.

The reference-OT versus equal-OT contrast tests target specification with
extra reference information. Reference OT versus the simple rule compares
complete recipes given the same information. Its own shuffles isolate
assignment at its weight spectrum. Do not use historical strong OT shuffles
as controls for this new spectrum or claim that equal native means prove
equivalence.

#### Fresh-head objective and optimization

Use a separate K-class linear classifier on the **raw 64-dimensional features**.
Initialize every weight and bias to zero for every condition. Freeze all
features and backbone state; do not restore the historical fc head. Use a
deterministic full-batch fit so optimization and batch-order differences are
less likely to obscure the weighting comparison.

```text
J(W, b) = sum_i w_i * CE(W x_i + b, y_i)
          + (lambda / 2) * (sum(W**2) + sum(b**2))
lambda = 0.0005
```

Fit with CPU float64 PyTorch L-BFGS: learning-rate parameter 1, history size
100, strong-Wolfe line search, at most 1,000 iterations / 1,250 objective
optimization evaluations, gradient tolerance `1e-7`, and change tolerance `1e-12`.
The final objective/gradient verification is one additional evaluation.
Use two CPU threads. These settings are implemented in the separate diagnostic
runner. No learning-rate, regularization, or initialization search.

Evaluate the full objective and gradient again at the returned solution.
Require finite state and gradient infinity norm `<=1e-6`; reaching the
iteration/evaluation limit without this check passing is not convergence.
Log objective, weighted CE, regularizer, iterations, evaluations, gradient norm,
and parameter norm. Numerical failure blocks a complete gate decision; do not
silently exclude a failed seed or choose its best intermediate head.

The objective is convex with positive L2 regularization on weights and bias.
This reduces uncertainty about fitting a linear head, but does not establish
generalization or that a larger/nonlinear model has no further headroom.
Fresh-head scores are not directly comparable to the historical warm-start
40-epoch SGD continuations.

#### Gate 1: establish damage and recoverable headroom

Run `unbiased_cb`, `biased_cb`, and `known_correction` for feature seeds
0/1/2: **nine fits**. Let BA denote eligible-class development macro recall:

```text
D_s = BA(unbiased_cb, s) - BA(biased_cb, s)
R_s = BA(known_correction, s) - BA(biased_cb, s)
```

Proceed to Gate 2 only when all nine fits and input checks are valid and:

- mean D is at least `0.01` (1.0 percentage point);
- mean R is at least `0.005` (0.5 percentage points);
- D is positive on at least two feature seeds, and R is positive on at least two;
- mean R / mean D is at least `0.25`.

These are prespecified research-budget criteria, not significance tests,
power guarantees, or known population effects. The context uses more training
examples, so D includes sample-size loss as well as mixture distortion.
R compares different weights on exactly the same examples and is the key
evidence of useful correction headroom. Ratios can exceed one; do not cap them
or call the unbiased-context score an upper bound.
Inclusive gate and practical-gain comparisons allow `1e-12` absolute numerical
roundoff so subtraction rounding cannot flip an exact threshold boundary.

If the gate fails, publish all nine results and stop. Distinguish
`insufficient_damage`, `insufficient_recovery`, and `technical_failure`;
retain all applicable reasons. A CE benefit without the prespecified accuracy
benefit is a secondary observation, not an automatic gate override.
Do not try another removal rate, subgroup direction, mask, class subset,
regularizer, or evaluation metric to obtain a pass. A redesigned diagnostic
requires a separate hypothesis and recorded budget; failure does not reject OT
or all possible reweighting.

#### Gate 2: test the target and the transport contribution

If Gate 1 passes, add equal-target OT, reference-target OT, the simple
reference-cluster rule, and three reference-OT shuffles for each feature seed:
**18 additional fits**, for **27 total**. Reuse the nine Gate-1 heads exactly.
Verify the persisted gate receipt before admitting these runs; any changed
input, weight rule, solver, source, or gate criterion invalidates reuse.
Passing the gate permits these 18 already planned fits under the frozen
protocol; it does not permit changing the experiment after seeing Gate 1.

Primary transport comparisons:

1. Reference-target OT minus reference-cluster ratio.
2. Reference-target OT minus the within-feature-seed mean of its three shuffles.

Required usefulness/target comparisons:

- Each of equal-target OT, reference-target OT, and reference-cluster ratio
  minus biased class balancing.
- Reference-target OT minus equal-target OT.
- Each unshuffled geometry method minus the known correction, to show the
  remaining gap to the diagnostic reference.
- Mean shuffled reference OT minus biased class balancing.

For every contrast, show all three paired differences, their mean and sample
SD. Average shuffles within feature seed first and report their individual
values and spread separately. Use the 0.5-point practical threshold and
positive differences on at least two feature seeds for a promising contrast.
Do not count shuffles, classes, development images, or optimizer iterations
as extra independent feature-seed replications.

| Outcome after a valid Gate 1 | Research decision |
|---|---|
| Reference OT clears the target against biased class balancing, the simple rule, and its own shuffle mean | A specific OT-based capability merits a separately designed natural-data follow-up |
| Reference OT beats equal-target OT by the target, but does not beat the informed simple rule or its own shuffles | Target information may help; transport-specific value remains unestablished |
| A simple correction is useful and OT does not improve beyond it | Prefer investigating the simple correction for this mechanism |
| Known correction helps but geometry methods do not | Investigate target estimation, geometry, row regularization, or fallback coverage using the saved diagnostics; no automatic parameter sweep |
| Small, mixed, or imprecise differences | Record unresolved; stop this bounded experiment |

Record every applicable outcome, including cases where more than one row
applies. A natural-data follow-up also requires a plausible connection between
the introduced bias and the actual task. Success here does not demonstrate a
Tail-class benefit, justify using extra reference labels without accounting
for them, or establish that all OT formulations are promising.

#### Measurements and interpretation

- Primary: `diagnostic_balanced_accuracy`, the unweighted mean recall across
  the fixed 34 eligible classes in E. This is not 100-class balanced accuracy
  or Head recall over all 35 original Head classes.
- Report accuracy, per-class recall/support/correct counts, common unweighted
  development CE, and class-macro development CE. Only the primary metric
  drives the gates; no checkpoint selection uses development scores.
- Report controller-group recall/support/correct counts for retained-all and
  downsampled groups, plus weighted training mass in each. In the evaluation
  reporter only, normalize seed-0 development features and assign them once
  to the fixed seed-0 controller centers, breaking exact ties with the lower
  center index. Reuse these group labels by development-row position across
  all feature seeds. Never apply seed-0 centers to seed-1/2 coordinates.
  These development assignments never enter preparation or a weight builder.
- Compare each weighted controller-group fraction with its desired fraction
  in U. Report per-class absolute mass error and its macro average. For a fixed
  retained sample, the known correction restores the U group fractions
  exactly, while individual removed examples remain unavailable.
- Report reference-versus-U mixture errors, confusion between controller and
  re-fitted method groups, OT hard-group mass errors, soft column errors,
  active/fallback masks and reasons, and each rule's weight spread and ESS.
  Method-cluster IDs can be swapped. Save the raw 2-by-2 T-row contingency;
  for reporting only, also show the label permutation maximizing agreement
  with controller groups. Never feed that privileged alignment back into
  geometry, reference targets, or weights.
- Break development changes into per-class and controller-group contributions.
  Show support and largest contributors; do not redefine the primary score
  after identifying sensitive classes.
- Save convergence diagnostics, finite-value checks, full weight vectors,
  fitting/reference/removed positions, final development predictions, and
  hashes. Preserve nulls for unsupported subgroup metrics.

The three feature seeds share images, the reference selection, removal mask,
and development split. Their SD describes sensitivity to these representations,
not uncertainty over all data draws. This is an exploratory diagnostic on a
previously inspected development split. It is neither an independent
confirmation nor a general test of whether OT can ever help.

#### Implementation, verification, and delivery checklist

1. **Preparation without classifier fitting.** Add a diagnostic config and
   a small explicit cached-feature command. Verify published hashes and
   positions, extract controller geometry, freeze eligibility, generate R/U/T
   and probabilities, and build the method geometry/weights. Produce a
   preparation receipt before reading any development outcome.
2. **Information boundaries.** Give builders only their allowed T/R arrays.
   Keep controller/known-correction metadata separate from ordinary builders.
   Preserve retained arrays as read-only and save row positions using the
   existing cache order; add no sample registry or fold framework.
3. **Transport extension.** Extend the existing relaxed Sinkhorn primitive
   with explicit optional prototype masses while preserving its current
   default uniform-column behavior. Keep historical A/B config and receipt
   identities intact. Reuse clustering and solver code, not a duplicate OT
   implementation. Do not apply the current hard-coded 100-class validators
   to the K-class diagnostic weights.
4. **Fresh-head fitting.** Implement the explicit full-batch objective and
   convergence checks. Save a diagnostic head with weights/bias, eligible
   original-label map, objective/config, input hashes, and feature seed.
   Do not label the K-output head as an ordinary 100-class image checkpoint
   or feed it to the existing image evaluator.
5. **Gate receipts and recovery.** Save all nine Gate-1 identities and results,
   computed D/R, thresholds and pass/fail reasons. Run Gate 2 only after
   verifying that receipt. Interrupted solves restart from zero; completed
   compatible solves are reused with file/hash verification.
6. **Focused checks.** Cover exact-size sampling and probabilities; split
   disjointness/order; fixed eligibility across feature seeds; probability
   correction and class-mass invariants; non-50/50 target mixtures; zero
   reference assignment counts and smoothing; shared method geometry;
   seed-specific feature/center compatibility and common seed-0 development
   group assignments across all representations;
   invalid clustering and solver failures; exact shuffle multisets; reference
   exclusion from classifier loss; development exclusion from fitted state;
   fresh K-class label mapping; convex-solver gradient checks; incomplete,
   failed, borderline, and passing gates; and completed-run recovery.
   On a tiny enumerable population, verify that averaging the known-weight
   loss over all fixed-size subsets equals the U class-balanced loss for a
   fixed classifier. Do not claim this proves fitted-head generalization.
7. **Reports and artifacts.** Produce resolved config, provenance, eligibility,
   sampling and geometry reports, weight diagnostics, per-fit optimization
   history, metrics, paired contrasts, gate decisions, and a readable report.
   Publish JSON to a separate `ablation-controlled-bias-results` branch and
   save binary heads, arrays, geometry and weights in checksum-linked release
   assets. Link the original inputs rather than republishing their binaries.
8. **Execution boundary.** First run a technical preflight and artificial-data
   integration checks, then the nine CPU feature fits. Gate 2 is conditional,
   not an additional automatic experiment after failure. Use a fresh local
   folder such as `outputs/controlled_bias/`. If Kaggle delivery is needed,
   add a dedicated `controlled_bias` mode with its own
   `/kaggle/working/ot-controlled-bias/` workspace and source pin; keep the
   current `simple_controls` mode available. These interfaces are implemented.
9. **Review before expansion.** Perform one focused implementation self-review,
   run affected checks, publish the implementation, and report the Gate-1
   outcome. After any Gate-2 results, discuss the research decision before
   new sampling interventions, natural-data confirmation, or backbone training.

The scientific budget is nine planned fit slots for Gate 1 and 18 conditional
slots for Gate 2. Technical artificial-data tests are separate. Record failed
attempts as well as successful fits; a rerun after a genuine code correction
must invalidate incompatible results and preserve their provenance. No
unrecorded scientific settings or additional data masks belong to this budget.

#### Implemented commands and Kaggle delivery

The separate runner is [`run_controlled_bias.py`](../pipeline/run_controlled_bias.py),
with fixed settings in [`controlled_bias.json`](../configs/controlled_bias.json).
Local commands require the validated audit input layout used by Stage A and a
controller folder containing the pinned `diagnostics/seed0/geometry.pt` and
`controls.json`. The Kaggle wrapper downloads and verifies these automatically.

```bash
python -m pipeline.run_controlled_bias preflight --inputs outputs/audit_inputs --controller outputs/controlled_controller --output outputs/controlled_bias
python -m pipeline.run_controlled_bias run --inputs outputs/audit_inputs --controller outputs/controlled_controller --output outputs/controlled_bias
```

Preflight freezes all preparation files and weights without fitting heads.
The same output folder can then be used by `run` in the same environment and
source checkout. Changing source, settings, inputs, weights or environment
requires a fresh folder. Completed heads are hash-verified before reuse;
interrupted attempts are preserved under `attempts/` and restart from zero.

Paste [`kaggle_ablation_study.py`](../scripts/kaggle_ablation_study.py) into a
fresh Kaggle notebook. Enable Internet and the repository-write `GITHUB_TOKEN`
secret, including GitHub release uploads. A GPU and CIFAR attachment are not
required for this CPU-only mode. The defaults are:

```python
STAGES = ("controlled_bias",)
CONTROLLED_BIAS_PREFLIGHT_ONLY = False
SOURCE_COMMIT = None
RESTORE_ROOT = None
```

The first execution resolves and pins the published source; subsequent sessions
use that source commit. `CONTROLLED_BIAS_PREFLIGHT_ONLY = True` performs only
preparation and writes no result reports to GitHub. With the default false,
the runner fits nine heads, saves/verifies `gate_one.json`, stops on failure,
and adds the 18 prespecified fits only on a pass. It never invokes official-test
evaluation or later Stage C/D experiments.

Workspace: `/kaggle/working/ot-controlled-bias/`. JSON reports go to
`ablation-controlled-bias-results`; binary preparation, geometry, weights,
heads and predictions go to checksum-linked release archives. Read
`reports/readable_report.json` first, then `study_summary.json`,
`study_status.json`, `gate_one.json`, and `preparation/sampling.json`.
Each head has optimization history, a final gradient check, metrics, predictions,
and a completion receipt. `head.pt` is a 34-output diagnostic artifact, not an
ordinary image-model checkpoint. A complete Gate-1 stop is a valid bounded
study outcome, not permission to try another intervention.

The local pinned-input preflight verifies the 34 classes, sampling arrays,
all three method geometries and all 27 planned weight vectors. Artificial-data
tests exercise both gates, convergence failures and restoration. The real
scientific fits and Kaggle publication remain unrun until execution.

### Stage C: investigate one remaining limitation

These routes are deferred pending the controlled-bias diagnostic and its
results discussion. Choose at most one justified route, then record its
hypothesis, exact settings, comparisons, and additional run count before running.

| Evidence | Focused follow-up |
|---|---|
| Curves suggest incomplete or unstable optimization, or a strength-dependent result | Compare class-balanced, strong OT, three shuffled strong OT conditions, and the candidate at common neighboring rates 0.0002 and 0.02; reuse 0.002 as reference (at most 36 additional continuations, before any candidate-specific shuffles) |
| A benefit appears in OT-active classes while Tail mostly falls back | Test minimum cluster size 2 versus 5, holding other OT settings fixed; recompute coverage and compare each policy with its own matched shuffles and class-balanced reference |
| Assignment is ineffective despite substantial classifier learning | Consider whether representation learning is needed; define an image-based experiment comparing class-balanced, OT, shuffled OT, and the candidate under one common backbone fine-tuning schedule |

Lowering minimum cluster size cannot activate every tiny class and may produce
unstable prototypes. Report changed coverage and fixed original subsets as well
as aggregate accuracy. The backbone route requires images and changes the
training setting; cached features cannot establish its result. Neither route is
an automatic response to an unfavorable score. If the evidence offers no
specific next hypothesis, stop and record the limitation rather than expand
the search indefinitely.

Every comparison must use the same optimization and backbone-update protocol.
When a candidate changes the within-class weight multiset or active-class mask,
add three shuffles of that candidate's own weights per seed for assignment
claims. Shuffles of the old strong OT vector cannot control a different weight
distribution or coverage policy. Record these additions in the stage budget;
deduplicate conditions that are identical.

### Measurements and decision rules

Use the common measurements above. In addition, save the baseline fitting
margins, final weight vectors, ranking/tie seeds, within-class margin/weight
rank associations, and top-quarter confidence summaries with tie handling.
Report how much weight goes to baseline mistakes and near-zero margins, using
within-class absolute-margin rank to describe uncertainty without choosing a
new score threshold from development outcomes. Record weight spread and effective
sample size alongside those summaries. Check that the intended reassignment
actually occurred before interpreting training outcomes.

Primary outcome is final development balanced accuracy. Show paired seed
differences, their mean and sample SD, the three individual values, and
Head/Medium/Tail and original OT-active/fallback changes. Average shuffle scores
within each baseline seed first; show shuffle variation separately. Three seeds
do not establish equivalence or supply strong evidence from a tiny mean gain.
Subset gains and training diagnostics cannot replace the primary outcome.

Use **0.5 percentage points of balanced accuracy** as an initial practical
target for deciding whether a candidate merits confirmation. This is a research
budget rule, not a significance threshold or a detectable-effect guarantee.
Advance a prespecified candidate when its mean paired gain meets this target
against class-balanced, strong OT, and mean shuffled strong OT, with positive
differences on at least two of three seeds for each comparison. Report all
contrasts, including unfavorable ones. Treat a smaller or inconsistent effect
as unresolved; pursue it only with a specific mechanism and a bounded follow-up.
Record any revision to this target before running the affected experiment.

### Stage D: confirmation and research decision

Freeze one candidate's full recipe and its required controls before generating
new results. Use a second fixed development split with seed 44 and matched
training seeds 0, 1, 2. Train three fresh fitting-only baselines; the seed-43
checkpoints have seen some of the new development images and cannot be reused.
Run class-balanced, original OT, strong OT, three shuffled strong OT conditions,
the candidate, and the strongest native simple control identified in Stage B.
Carry the frozen optimization and backbone-update protocol into every control.
For a candidate with a different weight multiset or active-class mask, include
three candidate-specific shuffles per seed as described in Stage C. Deduplicate
identical conditions. The unchanged Stage A/B protocol needs at most
**27 continuations plus three baselines**; candidate-specific shuffles can add
nine continuations. If Stage B was skipped, document the chosen simple control
before confirmation. Freeze any follow-up schedule too.

This second split checks sensitivity to the held-out partition within the same
retained pool; it is not an independent dataset replication. Apply the same
practical target against distinct reference methods, including the native simple
control and candidate-specific shuffle mean when required. Omit self-comparisons
if the candidate is itself a reference condition. Report every planned paired
contrast. Do not tune on this split. A failed confirmation remains a negative or
inconclusive finding; returning to development starts a new exploratory cycle.

- **Continue OT research:** an OT-based candidate shows a useful, consistent
  gain over class-balanced, shuffled, and native simple controls in confirmation.
  Name the supported mechanism and next targeted improvement.
- **Continue reweighting with a simpler method:** a non-OT rule improves on
  class balancing while OT has no demonstrated extra value.
- **Stop or deprioritize current prototype OT:** no reproducible practical gain
  emerges after the planned comparisons and any justified bounded follow-up.
  State the tested architecture, data regime, coverage, and training scope.
- **Unresolved:** effects vary across seeds/splits or uncertainty remains large.
  Do not describe a small or inconsistent difference as proof of no effect.

Only after a frozen recipe survives development confirmation should the final
evaluation protocol above be used. The current `final` runner carries the old
condition set; extend it to the frozen candidate and controls before using it
for this investigation. Do not start official test evaluation automatically.

### Implementation and execution checklist

Stage A implementation, publication, Kaggle execution, and result analysis are
complete. Its runner, input checks, weight invariants, frozen-state checks,
checkpoint/report publication, and completed-run recovery are implemented and
locally tested. The [assignment guide](weight_assignment.md) documents that
completed stage. Stage B's five controls, pinned-reference checks, geometry
replay, paired reports and Kaggle restoration/publication are implemented;
the [Stage B guide](weight_simple_controls.md) describes execution. Its Kaggle
training and read-only result verification are complete, as recorded in the
study status above. The controlled-bias diagnostic is implemented with a local
pinned-input preflight; scientific fitting and Kaggle execution remain pending.

Stages A and B completed 42 cached-feature continuations on the existing audit
baselines. The new diagnostic adds nine fresh-head fits first and at most
18 after its gate passes, with no new backbone training. Later Stage C routes
need another recorded budget; Stage D adds fresh baseline training. Nothing
in this plan automatically launches those later experiments.

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

To run the original development study, set `STAGES = ("prepare", "pilot", "study")`.
It uses one GPU, seeds 0/1/2, and two data-loader workers. It publishes results
after each completed run to `ablation-study-results` and stops after development.
No GitHub operation is performed by the local implementation tests.

On the first run, `SOURCE_REF = "ablation_study"` is resolved to an exact commit
and recorded in the results manifest. Later sessions reuse that commit, even
if the source branch has moved. Alternatively set `SOURCE_COMMIT` to a full
published hash before the first run. Changing the implementation requires a new
study and results branch: change `WORK_ROOT`, `RESULTS_ROOT`, and
`RESULTS_BRANCH` together. Existing results are not force-pushed over.

GitHub receives JSON reports under `reports/`: split positions, settings,
histories, weight diagnostics, metrics, summaries, and provenance. The stronger
OT runner also publishes its audit and stores the matching baseline checkpoints
and feature caches in a GitHub release. The results branch links to the archive
and records its checksum. Raw dataset images are kept out of publication.
Authentication uses the Kaggle `GITHUB_TOKEN` secret for Git and the GitHub
release API. The token is never saved in URLs or reports.

Preserve `/kaggle/working/ot-ablation/` in Kaggle outputs. It contains `study/`
and `data/`, including the prepared pool, caches, and checkpoints needed for
reuse. For another session, attach those saved artifacts as a Kaggle dataset
and set `RESTORE_ROOT` to the directory containing `study/` and `data/`. The
release archive is for the weight audit and reuse of the model inputs. Resuming
the entire training study still requires its saved workspace. The script rejects starting a fresh
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

On Kaggle, paste `scripts/kaggle_ablation_study.py` into a new notebook session.
Set `STAGES = ("strength",)`. Attach the saved output
from the successful development run if available. The runner searches `/kaggle/input` for
the study and checks that it has the split, baselines, and feature caches
before cloning the repo. Full training workspaces remain separate from the
published report and weight audit archive.

If there is no complete saved workspace, the runner rebuilds the three fitting
baselines and their classifier comparisons using CIFAR-100, the published
development split, and the original selected learning rate. It then runs
stronger OT against the rebuilt controls. This takes longer than reusing the
saved models. Attach CIFAR-100 for the rebuild or set `ALLOW_DOWNLOAD = True`.
The rebuild is recorded in the follow-up reports. Set
`REBUILD_MISSING_STUDY = False` to require saved models.

If more than one complete study is attached, set `RESTORE_ROOT` to the one you
want:

```python
RESTORE_ROOT = Path("/kaggle/input/<attached-output-path>/ot-ablation")
```

Use the saved workspace or the study folder itself. The follow-up reads the
saved study settings and can run without `data/train.npz`. If the completed
workspace is already under `WORK_ROOT`, it uses that copy.
Use source from `ablation_study` that contains the stronger OT stage. The
follow-up checks out code in `ot-strength-code`, saves runs in
`study/strength/`, and publishes JSON to `ablation-strength-results`. Keep the
updated workspace to resume a partially published follow-up.

### Audit which examples OT favors

The Kaggle runner now defaults to `STAGES = ("controlled_bias",)` for the new
diagnostic. Use `STAGES = ("simple_controls",)` to reproduce Stage B.
Set `STAGES = ("rebuild_audit",)` only to reproduce the completed audit when its
inputs are missing. With CIFAR-100 attached, this audit mode
recreates the published fitting split and trains only the three fitting
baselines. It pushes the audit report, baseline logs, and split metadata to
`ablation-rebuilt-audit-results` and links to the GitHub release containing
the rebuilt checkpoints and feature caches. This is an audit of the rebuilt
baselines. The previous stronger OT results remain on their own branch.

Stronger OT training publishes the weight audit automatically. The standalone
`STAGES = ("audit",)` mode downloads the inputs from GitHub for verification
and scores the cached fitting examples on CPU. Runs completed before this
publication change can be backfilled by running the audit once where the
matching files are available. The script uploads them and publishes the
report to `ablation-weight-audit-results`.

Within each OT-active class, compare the top quarter by OT weight with the
whole class. The main report gives their difference in average confidence
percentile. Confidence is the correct-class score minus the strongest rival
score. The report also gives the difference in baseline accuracy, by class
and training seed. No additional training is needed.

Implementation checks use artificial images and do not download CIFAR:

```bash
python -m pytest tests/test_ablation.py tests/test_prototype_weights.py tests/test_finetuning.py
python -m pytest tests/test_kaggle_ablation_study.py
```

The local checks use artificial images, simulated release uploads, and local
Git repositories. Consult the published results for real Kaggle runs and their
provenance.
