# Running both OT methods

The integration branch supports two independent paths from a saved baseline:
reweight retained real examples, or generate extra 64-dimensional features and
train on real plus synthetic features. It does not combine their losses.
Both use the same prepared pool, ResNet-32, and image evaluator. Test data never
enters feature extraction, weighting, guide fitting, or synthesis.

## Branch integration

`integration/ot-methods` merges these source commits:

- Reweighting: `3111f7c5b0ef0e9f7277834eda056c9edccb1f38`.
- Synthesis (`hieu`): `30ac14ca302ab0e6698a88fe97ee371bef04cbb1`.

Generated outputs are excluded from this branch's file tree. The source branches
and their existing results remain available; the merge history still includes
the synthesis branch's historical artifacts.

## Matched baseline checkpoints

Prepare the pool once with data seed `42`. For each training seed `N` in
`0, 1, 2`, train `configs/baseline_matched_seedN.json` into
`outputs/baseline_seedN`. These configs use the reweighting branch's existing
200-epoch warm-up and multistep recipe. They preserve its defaults.

Point each reweighting continuation's `pretrained_checkpoint` at that seed's
`final.pt`, and set its training seed to `N`. See the
[reweighting commands](reweighting.md#matched-baseline-comparison) and
[prototype and SAVA configurations](reweighting.md#prototype-weighting-comparison).

For synthesis, extract features from that **same checkpoint**, then run the
matching generation and classifier configs. For seed 0:

```bash
python -m pipeline.train --config configs/baseline_matched_seed0.json --output-dir outputs/baseline_seed0 --device cuda
python -m pipeline.extract_features --checkpoint outputs/baseline_seed0/final.pt --output outputs/baseline_seed0/train_features.pt --device cuda
python -m pipeline.methods.synthesis --config configs/synthesis_matched_seed0.json --device cuda
python -m pipeline.train_classifier --config configs/classifier_synthesis_matched_seed0.json --device cuda
python -m pipeline.evaluate --checkpoint outputs/classifier_synthesis_matched_seed0/final.pt --output outputs/classifier_synthesis_matched_seed0/metrics.json --device cuda
```

Use the corresponding `seed1` or `seed2` configs and paths for the other seeds.
Do not overwrite existing baseline folders; use a consistent new set of paths
if they already contain another recipe.

Feature caches record input hashes, labels in retained order, and raw feature
coordinates. Synthesis works in scaled coordinates and converts generated
features back before saving. Classifier training uses raw coordinates and keeps
the entire backbone, including batch-normalization buffers, frozen. All final
checkpoints retain **real** class counts for Head/Medium/Tail evaluation.

Add `--resume` to synthesis generation to reuse completed classes, and to
`pipeline.train_classifier` to resume after a saved epoch. Configuration and
input checks reject incompatible resumes. Reweighting continuations restart
from their source baseline if interrupted.

## Historical recipes and comparisons

The original `hieu` baseline used cosine scheduling and weight decay `0.0005`.
It remains selectable with `configs/baseline_cosine.json`; the imported
`baseline_seed1.json` and `baseline_seed2.json` also retain that historical recipe.
The default baseline uses weight decay `0.0002`, five warm-up epochs, and
multistep decay. Matching seed numbers alone does not make those baselines equal.

Synthesis classifier training retains LR `0.01`, cosine scheduling, and ordinary
momentum SGD. Reweighting retains LR `0.00002`, constant scheduling, and Nesterov
SGD. Both continue the loaded classifier for 40 epochs. A common baseline
removes the starting-checkpoint difference; these continuation recipes and the
number of training examples still differ. Existing results must be identified
by their original recipes. No new accuracy claim follows from this merge.

`pipeline.run_comparison` retains synthesis's real-only, random-duplication,
uniform-OT, loss-shifted-OT, and full-method controls. Imported comparison
configs still refer to their historical artifact paths; update paths consistently
before using them with a new experiment.

## Kaggle

Paste `scripts/kaggle_prototype_comparison.py` into a GPU notebook with Internet,
the CIFAR dataset, and the `GITHUB_TOKEN` secret. Despite its historical filename,
the runner now supports `EXPERIMENT_PATH = "reweighting"`, `"synthesis"`, or
`"both"` (default). It pins the integration implementation commit and defaults to
training seeds `(0, 1, 2)` with preparation seed `42`.

The reweighting path runs uniform, class-balanced, cluster-balanced, Sinkhorn,
and SAVA continuations. The synthesis path uses the full `synthesis_final.json`
recipe, then its classifier continuation. Both reuse each seed's baseline.
Set `TRAINING_SEEDS = (0,)` for a GPU pilot before a full run. CPU tests verify
integration behavior, not GPU runtime, resource use, or method quality.

Results go to the separate `ot-methods-results` branch, using
`/kaggle/working/ot-methods-*` directories. Earlier
`prototype-weighting-results` results are preserved. To reuse earlier compatible
baselines, attach their checkpoints and set `BASELINE_CHECKPOINTS` by seed.
Attached checkpoints must match the default multistep recipe and retained pool.

Completed reweighting runs and metrics are reused. Synthesis resumes completed
classes and classifier epochs within the working results directory. Results
are pushed after each completed stage; an interruption before that push can
lose the stage's local progress when the Kaggle session is discarded. Use one
active notebook per results branch and keep its settings fixed across sessions.
Combined metrics are written to `comparison_ot_seedN.json`.

The original SAVA-only runner is still available at commit `3111f7c` on
`feat/ot-reweighting` for continuing those existing experiments unchanged.
