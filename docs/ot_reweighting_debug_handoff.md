# Handoff for debugging OT reweighting

The implementation and matched baseline comparison are complete. The current task
is to understand the code and use the saved run artifacts to investigate why OT
produced no measurable gain in the first matched comparison.

The user prefers simple, concise explanations that connect the code to the method.
Discuss one part at a time and agree on a change before editing the training code
or changing defaults. The user runs full training on their GPU server.

## Where to pick up

- Local repo is `/home/dzzz83/Documents/code/expert_method`.
- Server checkout is `/mnt/hdd2/phatht/phat/MoE_Imbalance`.
- Use branch `feat/ot-reweighting` at commit `63de7a2`, already pushed to GitHub.
  Commit `1c1467e` added OT. Commit `63de7a2` added the matched continuations.
  Keep future work off `master`.
- [The method guide](reweighting.md) documents configs, server commands, and
  diagnostics. The older local `reweighting_handoff.md` predates implementation.
- The last full local suite reported 200 passing tests. The user has since
  completed all three comparison runs and evaluations on their CUDA server.

## What the user has covered

The user has read the training runner, the prepared dataset loader, the baseline
SGD loop, and the wrapper that restores the baseline checkpoint for fine-tuning.
They have also seen a high-level explanation of the shared classifier loop and
the distinction between Sinkhorn finding a transport plan and Adam learning new
example weights. They want to understand the implementation well enough to
diagnose the result themselves.

| File | Responsibility |
|---|---|
| [pipeline/train.py](../pipeline/train.py) | Resolves config, loads prepared data, invokes the selected method, and saves the final checkpoint |
| [pipeline/data.py](../pipeline/data.py) | Loads the retained images, labels, class counts, and preparation settings |
| [pipeline/methods/baseline.py](../pipeline/methods/baseline.py) | Trains the backbone and classifier with ordinary cross-entropy |
| [pipeline/methods/finetuning.py](../pipeline/methods/finetuning.py) | Restores the same complete baseline state for both continuations and validates dataset metadata |
| [pipeline/methods/reweighting.py](../pipeline/methods/reweighting.py) | Caches frozen features, learns OT weights when enabled, and trains the classifier |

## Matched seed-0 result

All runs use the same CIFAR-100-LT training pool of 10,847 images with imbalance
factor 0.01 and preparation seed 42. Evaluation uses the original balanced
10,000-image test set. Both continuations load `outputs/baseline_seed0/final.pt`
and use the same frozen backbone, initial classifier, seeded batch order, and
classifier settings.

| Server output folder | Accuracy | Balanced accuracy | Head recall | Medium recall | Tail recall |
|---|---:|---:|---:|---:|---:|
| `outputs/baseline_seed0` | 38.41% | 38.41% | 64.97% | 36.54% | 9.60% |
| `outputs/uniform_finetuning_seed0` | 38.40% | 38.40% | 64.97% | 36.54% | 9.57% |
| `outputs/ot_finetuning_seed0` | 38.39% | 38.39% | 64.94% | 36.54% | 9.57% |

Uniform versus baseline measures the effect of adding uniform fine-tuning. OT
versus uniform isolates the learned weights. OT had one fewer correct
prediction than uniform and the same tail recall. This single seed shows no gain under the
current settings. A NumPy deprecation warning appeared during evaluation and
all three metric files were written successfully.

## How the implementation works

1. The baseline trains the full ResNet-32 for 200 epochs. Both continuations
   restore its backbone, batch normalization buffers, and classifier. They
   freeze the backbone and cache one 64-dimensional feature per retained image
   using deterministic preprocessing.
2. Uniform fine-tuning sets every example weight to `1/N`. OT builds one
   representative per class by averaging raw features and then normalizing.
   It uses a fixed cost of `0.5 * cosine_distance + 0.01 * label_mismatch`.
3. In `_update_example_scores`, a global softmax turns learnable scores into
   example weights. Sinkhorn computes a plan targeting those weights as row sums
   and `1/100` as each column sum. PyTorch backpropagates `sum(plan * cost)`
   through the plan, and Adam updates the scores. The default is one Adam step
   per classifier epoch at learning rate 0.001, for 40 updates total. Sinkhorn
   uses epsilon 0.1, 200 iterations, and a relative marginal error tolerance of
   0.001.
4. In `_train_classifier_epoch`, both variants shuffle the raw cached features
   with their original row indices. The classifier loss is
   `mean(N * detached_example_weight * cross_entropy)`. SGD updates only the
   existing classifier for 40 epochs at a constant learning rate of 0.00002.
   Classification loss updates classifier parameters. Transport cost updates
   example scores.

Equal column sums constrain how much mass each representative receives. The
total weight of examples with a given original label is a separate quantity.
Examples can send mass across labels, so a balanced reference can coexist with
training weights that remain close to the original class proportions.

## Evidence still needed

An earlier standalone run in `outputs/reweighting_seed0` reached 37.26% accuracy.
Its final effective sample size was 10,830.69 out of 10,847 and its maximum weight
was 3.88% above uniform. Its OT cost fell about 0.27% and its marginal errors
were well below tolerance. These measurements belong to that earlier run. The
matched OT continuation's weight and classifier diagnostics are still needed.

1. Read the first and last Stage 2 records in
   `outputs/ot_finetuning_seed0/history.json`. Compare effective sample size
   with 10,847 and maximum weight with `1/10847`. Check the OT cost and marginal
   errors separately. Small marginal errors establish that the plan met its
   constraints for the current weights.
2. Read the final `class_weight_masses` in that run's
   `reweighting_diagnostics.json`. Compare each class with its uniform share
   `class_counts[c] / N` and sum by Head, Medium, and Tail using the evaluator's
   cutoffs of at least 100, 20–99, and below 20 training examples.
3. Compare `fc.weight` and `fc.bias` in the three checkpoints. Report the
   parameter changes from baseline to each continuation and between the two
   continuations. Compare how far the classifier moved with how much the
   weights changed before considering another experiment.

Use the saved artifacts for these checks. Keep the test set for final evaluation
and keep hyperparameter decisions separate from test scores.

## Suggested prompt for the next chat

> Read this handoff and continue helping me debug OT reweighting one part at a
> time. Keep the explanation simple and tie it to the code. Start with how
> `_update_example_scores` changes the weights, then help me inspect the matched
> OT run's saved diagnostics. Discuss any proposed changes with me first.
