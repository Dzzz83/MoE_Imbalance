# OT reweighting

This method pretrains ResNet-32, freezes its backbone, and learns example weights
by transporting the retained training population to equally weighted class
representatives. Only the existing linear classifier trains in Stage 2. Final
image inference uses the ordinary model; no cache or OT objects are required.

This is an adaptation of Guo et al., not an exact reproduction or a claim of
improved accuracy. All fitted features and references come from the retained
training images. No meta split, test supervision, or test-based model selection
is used.

## Run on the server

Run from the repository root with the project's environment active. Prepare the
training data as described in the README. First check one full-size transport
update on synthetic features; this does not read CIFAR or train images:

```bash
python -m scripts.check_reweighting_transport --config configs/reweighting.json --device cuda
```

This reports time, process peak RSS (Linux), CUDA peak allocated memory, and
marginal errors. GPU allocation includes the synthetic OT problem, not the image
model or other processes. It is a resource check, not research validation.

Train with a new output folder:

```bash
python -m pipeline.train --config configs/reweighting.json --output-dir outputs/reweighting_seed0 --device cuda
python -m pipeline.evaluate --checkpoint outputs/reweighting_seed0/final.pt --output outputs/reweighting_seed0/metrics.json --device cuda
```

The evaluator requires the original test set locally; add `--download` only if
you want it to fetch that dataset. Use `--device cpu` for CPU execution. Full-data
training belongs on the server. To change the training seed, edit `seed` in the
config; keep the preparation seed fixed for comparisons.

## Matched baseline comparison

Compare three final models to distinguish the effect of OT from extra classifier
training:

| Run | Configuration | Training |
|---|---|---|
| Cross-entropy baseline | `configs/baseline.json` | 200 epochs of ordinary full-model training |
| Uniform continuation | `configs/uniform_finetuning.json` | Load the baseline, freeze its backbone, continue its classifier for 40 epochs with uniform example weights |
| OT continuation | `configs/ot_finetuning.json` | Load the same baseline and use the same continuation with learned OT weights |

The two continuation configs point to `outputs/baseline_seed0/final.pt`. Neither
repeats pretraining nor resets the classifier. They use the same deterministic
raw feature cache, shuffled row order, batch size, SGD settings, and number of
epochs. Both start a fresh classifier optimizer. Uniform continuation bypasses
OT entirely; its loss is ordinary mean cross-entropy. OT settings remain the
existing defaults, including 40 score updates in total, to isolate weighting
without changing other settings.

Run once with new output folders; skip the first command if that baseline
checkpoint already exists and uses the intended retained pool and recipe:

```bash
python -m pipeline.train --config configs/baseline.json --output-dir outputs/baseline_seed0 --device cuda
python -m pipeline.train --config configs/uniform_finetuning.json --output-dir outputs/uniform_finetuning_seed0 --device cuda
python -m pipeline.train --config configs/ot_finetuning.json --output-dir outputs/ot_finetuning_seed0 --device cuda
python -m pipeline.evaluate --checkpoint outputs/baseline_seed0/final.pt --output outputs/baseline_seed0/metrics.json --device cuda
python -m pipeline.evaluate --checkpoint outputs/uniform_finetuning_seed0/final.pt --output outputs/uniform_finetuning_seed0/metrics.json --device cuda
python -m pipeline.evaluate --checkpoint outputs/ot_finetuning_seed0/final.pt --output outputs/ot_finetuning_seed0/metrics.json --device cuda
```

The main comparison is OT versus uniform continuation. Compare each with the
baseline to measure the total effect of Stage 2. Report overall accuracy and
Head/Medium/Tail recall for all three. Keep `train_data`, the baseline checkpoint,
the continuation seed, and classifier settings identical between the two
continuations. For additional training seeds, first train a baseline for that
seed, then update both continuation configs' `seed` and `pretrained_checkpoint`
and choose fresh output folders. Keep preparation settings fixed. Several
matched seeds are needed to assess small differences reliably.

Checkpoint loading requires `method: "baseline"` and matching real class counts
and preparation settings. These checks validate metadata, not image contents;
reuse the exact prepared pool. The source checkpoint is read without being
modified. Its path is recorded in the resolved config. This is a new Stage 2
execution, not optimizer resume. An existing `reweighting` final checkpoint
cannot recover the model before Stage 2; it is rejected as a baseline source.

## Defaults and their sources

| Component | Defaults | Source and rationale |
|---|---|---|
| Pretraining | 200 epochs, batch 128, SGD LR 0.1, momentum 0.9, weight decay 0.0002, no Nesterov; five warm-up epochs, LR multiplied by 0.01 after epochs 160 and 180 | [LDAM authors' CIFAR recipe](https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py); identical to our baseline |
| Classifier | 40 epochs, SGD LR 0.00002, momentum 0.9, weight decay 0.0005 | [OT paper §6.1](https://proceedings.neurips.cc/paper_files/paper/2022/file/a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf) |
| Classifier Nesterov and schedule | Nesterov enabled, constant LR | [Released image code](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT/blob/01f35a8c08610c4f09e19e346150af3a3428174e/OT_train.py) enables Nesterov and does not schedule Stage-2 LR |
| Classifier batch size | 128 | Shared baseline batch size retained for this cached-feature adaptation; the OT paper reports 16 |
| Score optimizer | Adam LR 0.001, betas (0.9, 0.999), numerical epsilon 1e-8, weight decay 0, constant LR | [PyTorch defaults](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html); an agreed adaptation, not the paper's SGD weight optimizer |
| Transport | Entropy coefficient 0.1, 200 iterations | OT paper experimental setup |
| Cost | `0.5 * cosine_distance + 0.01 * label_mismatch` | [Released combined-cost code](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT/blob/01f35a8c08610c4f09e19e346150af3a3428174e/Sinkhorn_distance_fl.py); label term is half the mean squared distance between 100-class one-hot labels |
| Convergence threshold | Maximum relative row and column errors <= 0.001 | Engineering tolerance of 0.1% per target mass, not a paper hyperparameter |

The small classifier LR is a paper-based starting value, not a calibrated match
to our changed loss scale and batch size. The earlier proposed LR 0.01 with
cosine decay had no established source and is not the default. Adam's numerical
epsilon and Sinkhorn's entropy coefficient are separate settings. With one score
update per epoch, score weights may stay close to uniform; diagnostics make this
visible. These defaults have not been tuned or validated by full-data training.

## Training behavior

1. Pretrain the whole model with the baseline. Keep its final epoch.
2. Disable backbone gradients, clear old gradients, and keep every backbone
   module in evaluation mode, including batch normalization.
3. Cache raw features using deterministic conversion and normalization, in
   retained-array order. There are no fresh crops or flips in Stage 2.
4. Average raw features within each class, then normalize the 100 means.
   Separately normalize example features for OT; preserve raw classifier inputs.
   References include their contributing examples and remain fixed. A zero vector
   stays zero, giving a zero dot product and feature distance one. Missing classes
   are rejected before pretraining.
5. Give each reference mass 1/100. Each retained position has a zero-initialized
   score; global softmax starts with equal example weights. Label mismatch uses
   known training labels, not predictions.
6. Before each classifier epoch, perform `score_updates` global updates (default
   one) with one persistent Adam optimizer. Differentiate through the fixed-budget
   log-domain Sinkhorn solve. Minimize `sum(transport * cost)`; entropy belongs
   only to the inner solve. Costs and references stay detached.
7. Recompute and detach weights, then train only `model.fc` on uniformly shuffled
   cached rows. Carry retained positions with features and labels. A batch uses:

   ```text
   mean(N * global_weight[position] * cross_entropy)
   ```

   Here N is the entire retained population size. Uniform weights reproduce
   ordinary mean cross-entropy. Never renormalize within a batch. Include the
   final partial batch. Cached-feature loading uses zero worker processes;
   shared `num_workers` still controls image pretraining and feature extraction.
8. Save the final scheduled model through the shared runner. Evaluation is a
   separate command and never influences training.

The fixed backbone makes the OT objective independent of classifier changes.
Interleaving updates is our chosen schedule, not classifier feedback into OT.
The released code instead normalizes OT weights within batches, uses a balanced
meta population, recomputes image features, and allows train-mode BN updates.

## Configuration and diagnostics

`configs/reweighting.json` exposes separate `method_options.pretraining` and
`method_options.classifier` objects. Pretraining accepts baseline options. The
classifier accepts `epochs`, `learning_rate`, `momentum`, `weight_decay`,
`nesterov`, and `scheduler: "constant"`. Other method options are
`score_learning_rate`, `score_updates`, `epsilon`, `sinkhorn_iterations`,
`marginal_tolerance`, `feature_cost_scale`, and `label_mismatch_cost`.
Unknown or invalid values fail before training. Defaults are saved in the run
configuration; input configuration objects are not modified.

The checkpoint-based methods `uniform_finetuning` and `ot_finetuning` require
`method_options.pretrained_checkpoint` and accept the same `classifier` options.
`ot_finetuning` also accepts the OT/score options above. Neither accepts
`pretraining`; the source baseline's saved configuration supplies the completed
epoch count. Uniform fine-tuning rejects OT options instead of ignoring them.

Row errors compare each plan row sum to its example weight; column errors compare
each column sum to 1/100. Absolute errors and per-target relative errors are
reported. Exceeding `marginal_tolerance` stops before that optimizer step;
increase `sinkhorn_iterations` and use a fresh output folder. No final checkpoint
is saved after a failed method. Nonfinite values and zero weights caused by
softmax underflow also fail explicitly, without silently clamping masses.

The runner saves `config.json`, `history.json`, and `final.pt`. History epochs
are globally one-based: pretraining 1–200, reweighting 201–240 by default.
Reweighting records also contain `stage_epoch`, weighted `loss`, classifier and
score learning rates, OT cost, iteration count, marginal errors, maximum weight,
and effective sample size. ESS is computed in float64 as
`sum(weights)**2 / sum(weights**2)` to account for rounding in total mass; it is
equivalent to `1 / sum(weights**2)` for normalized weights.

`ot_cost_before_update` and marginal errors describe the final solve before its
score optimizer step. Classifier loss and weight statistics use the resulting
post-update weights. `reweighting_diagnostics.json` records class weight masses
in class-index order after each completed epoch. Real retained counts still
define evaluation's Head/Medium/Tail groups. Evaluation adds `metrics.json`.

Checkpoint-based runs save only their continuation history, with global epochs
following the source baseline (201–240 for a 200-epoch baseline) and
`stage_epoch` starting at one. Uniform records use `stage: "uniform_finetuning"`,
include weight maximum and ESS, and omit OT/score measurements. Its
`uniform_finetuning_diagnostics.json` records class weight masses, which equal
real class proportions rather than balanced class masses. OT continuation uses
the existing `stage: "reweighting"` and `reweighting_diagnostics.json` format.

## Verification and experiment limits

```bash
python -m pytest
python -m pytest tests/test_pipeline.py::test_reweighting_cli_checkpoint_evaluation_and_reproducibility
```

Artificial-data tests cover loss and gradient equivalence under uniform weights,
shuffle alignment, final partial batches, backbone preservation, transport
marginals, finite-difference score gradients, convergence rejection, reproducible
training, and checkpoint evaluation without training data. CPU tests and the
synthetic resource check do not establish CUDA memory use or research accuracy.

Tests also cover matching the complete source state, identical continuation
shuffle orders, frozen backbone buffers, source-checkpoint preservation,
incompatible source rejection, and identical final models when OT is forced to
return uniform weights. These establish comparison correctness, not an accuracy
improvement. Do not select or tune a checkpoint using test scores.
