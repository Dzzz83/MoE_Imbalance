# Shared pipeline implementation plan

Status: shared components and the baseline are implemented in this repository.
The interfaces and commands below describe the implementation; full CIFAR-100
training and the research methods remain separate work.
This plan replaces the archived version where it conflicts with the decisions
recorded here.

## Goal

Build shared data preparation, ResNet-32 creation, training entry point, and
evaluation so contributors can implement different training methods against the
same interface. Start with an ordinary cross-entropy baseline as the working
example. Add reweighting and synthetic-sample methods afterward.

Share the inputs and evaluation procedure. Each method owns its training
algorithm, including stages, losses, optimizers, sampling, augmentation changes,
feature extraction, freezing, and synthetic examples.

## Experimental protocol

- Start with CIFAR-100's original 50,000 training images.
- Apply exponential imbalance factor `0.01` once. Keep the resulting 10,847
  images as the common training pool, with 500 images in the largest class and
  5 in the smallest. Discarded images are outside the method inputs.
- The retained pool is what we previously called the development set. There is
  no additional shared training/validation split.
- Use the original balanced 10,000-image CIFAR-100 test set for final evaluation.
  Repeated evaluation is allowed. Test examples and labels must not enter
  training, prototype construction, synthesis, or other fitted method state.
- Use the same plain ResNet-32 architecture and label order for every method.
- Evaluate the final model after its configured training schedule. The baseline
  does not select a best checkpoint using test accuracy.
- Keep the data preparation seed fixed when changing the training seed. Use the
  same training seeds across methods when comparing results.

The data and model are shared; a trained first-stage checkpoint is not required.
No OOF folds, sample-ID system, hashes, run registry, test-access gate, or
automatic experiment matrix is part of this implementation. Ridge and the two
research algorithms are separate work after the shared foundation is ready.

## Architecture

```text
CIFAR-100 training images -> prepare once -> retained training data
                                                   |
Fresh ResNet-32 + configuration --------------------+
                                                   |
                                     selected training function
                                  baseline / reweighting / synthesis
                                                   |
                                           trained ResNet-32
                                                   |
                                      runner saves final checkpoint
                                                   |
CIFAR-100 test images ---------------------> shared evaluator
                                                   |
                                                metrics
```

## Shared data preparation

Use the original class order, labels `0..99`. Accept an imbalance factor `f` in
`(0, 1]`, default `0.01`. For class `c`, retain:

```text
max(1, floor(500 * f ** (c / 99)))
```

Use preparation seed `42` by default. Adapt the archived sampling logic: one
NumPy `default_rng(seed)`, sample within each class without replacement in
ascending class order, then sort the selected source positions. This preserves
the previous subset when the source order and settings match. The positions
are an implementation detail, not identifiers carried through the pipeline.

Preparation writes `data/prepared/train.npz`, containing only:

| Field | Meaning |
|---|---|
| `images` | Retained images, `uint8`, shape `(N, 32, 32, 3)` |
| `labels` | Original class labels, `int64`, shape `(N,)` |
| `class_counts` | Real retained counts, `int64`, shape `(100,)` |
| `imbalance_factor` | Preparation setting, default `0.01` |
| `data_seed` | Preparation seed, default `42` |

Load this file for training; do not subsample again in each method. Preparation
is an explicit command and refuses to replace an existing file. To prepare a
different population, choose another output path. Keep raw/prepared data and
training outputs out of Git.

### Data interface

`TrainingData` is a small data record with `images`, `labels`, and `class_counts`
as described above, plus a `data_settings` dictionary containing
`imbalance_factor` and `data_seed`. It has no training behavior. The runner
saves those settings with the checkpoint. Loading checks image/label shapes,
label range, and that the saved class counts match the labels.

```python
load_training_data(path) -> TrainingData
make_dataset(data, transform) -> torch.utils.data.Dataset
```

A dataset item is `(image, label)`: a floating image tensor of shape
`(3, 32, 32)` and an integer label. Methods create ordinary PyTorch data loaders
with their desired batching, shuffling, and sampling. All views use the same
retained arrays; methods must not modify those shared arrays in place.

Provide two common transform factories, `training_transform()` and
`evaluation_transform()`:

- Training: random crop to 32 pixels with padding 4, random horizontal flip,
  tensor conversion, and normalization.
- Evaluation and deterministic training-feature extraction: tensor conversion
  and the same normalization, without random augmentation.

Use fixed CIFAR-100 channel mean `(0.5071, 0.4867, 0.4408)` and standard deviation
`(0.2675, 0.2565, 0.2761)`. Methods can supply another training transform and
record that choice in their configuration.

The evaluator loads the original test split separately. Synthetic images or
features belong to the method's own datasets; they do not change the shared
real-data counts used for evaluation groups.

## Shared model interface

Provide `create_model() -> ResNet32`, using the plain CIFAR ResNet-32 with
Option-A shortcuts. Review and adapt the plain backbone and classifier from the
archive; leave the routing and contrastive model variants there.

```python
model(images)           # logits: (batch, 100)
model.backbone(images)  # features: (batch, 64)
model.fc(features)      # logits: (batch, 100)
```

The runner seeds Python, NumPy, and PyTorch, then constructs a fresh model before
invoking the method. Equal training seeds therefore give methods the same
initial weights. Seed data-loader randomness as well. Reproducibility is for
the same software and hardware environment; do not promise identical results
across devices.

## Method integration interface

Each method module exposes one function:

```python
def train(model, train_data, config, output_dir, device) -> list[dict]:
    ...
```

| Argument / result | Contract |
|---|---|
| `model` | Fresh shared ResNet-32, already on the selected device |
| `train_data` | The retained `TrainingData`; no test data |
| `config` | Loaded configuration dictionary, including method settings |
| `output_dir` | Folder for this execution and optional method-specific files |
| `device` | Selected PyTorch device |
| Return value | JSON-compatible list of training-history records |

The method trains the supplied model in place. It may train a temporary copy,
but must copy its complete final `state_dict` back into the supplied model before
returning, including batch-normalization statistics.
Its final model must support ordinary image-to-logit inference through the
shared architecture. Training-only prototypes, OT solvers, SVMs, synthetic
features, or other auxiliary objects are not needed to evaluate that model.

History records contain `epoch` (one-based within a stage) and `loss` (the mean
training objective for that epoch). Multistage methods also include a `stage`
name. Extra scalar training measurements are allowed. The shared evaluator
computes test metrics separately.

### Responsibilities

| Component | Responsibility |
|---|---|
| Preparation | Retain images once and save the common training pool |
| Training runner | Read config/data, select method/device, seed, create model, invoke method, save standard outputs |
| Method | Construct loaders and own all training stages and auxiliary state |
| Evaluator | Restore the plain model and compute common test metrics |

Each method also exposes `resolve_options(options) -> dict` to apply its defaults
and reject unsupported settings before training. The runner passes that resolved
configuration to `train` and saves it with the checkpoint.

Select methods with a small explicit mapping of names to pairs of training and
option-resolving functions. Begin with `baseline`; add entries when other methods
exist. No plugin discovery, inherited trainer framework, or placeholder research
implementations are needed.

## Configuration and outputs

Use one JSON configuration file per training recipe. For example:

```json
{
  "method": "baseline",
  "train_data": "data/prepared/train.npz",
  "seed": 0,
  "batch_size": 128,
  "num_workers": 4,
  "method_options": {
    "epochs": 200,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "scheduler": "cosine"
  }
}
```

These are starting baseline settings, not fixed settings for every method.
The baseline uses cross-entropy, SGD, and cosine learning-rate decay to zero
over its configured epochs. Method-specific settings stay in `method_options`;
multistage methods can define their stages there. Reject unsupported settings
with a clear error rather than silently ignoring them.

The caller chooses a readable output folder, such as `outputs/baseline_seed0`.
The runner refuses an existing folder to avoid accidental overwrites. Resume
support can be added later when needed.

Each successful execution saves:

```text
outputs/baseline_seed0/
  config.json       # actual settings used, including any supplied defaults
  history.json      # records returned by the method
  final.pt          # standard checkpoint saved by the runner
  metrics.json      # written when evaluation is requested
```

`final.pt` contains `model_state_dict`, `config`, `class_counts`, and
`data_settings` (`imbalance_factor` and `data_seed`). The evaluator can therefore
restore a model and define class groups without needing the training images or
importing the method implementation. Store tensors so loading onto CPU or GPU
is supported through `map_location`.

## Shared evaluator

Expose two reusable operations and a command-line entry point:

```python
predict(model, loader, device) -> (predictions, labels)
compute_metrics(predictions, labels, class_counts) -> dict
```

`predict` uses evaluation mode, no gradient updates, deterministic preprocessing,
and all examples, including the final partial batch. It returns predicted class
labels and true labels in matching order.

Report metrics as fractions in JSON and percentages in terminal output:

- Overall accuracy: correct predictions divided by test examples.
- Per-class recall: correct predictions for each class divided by that class's
  test examples.
- Balanced accuracy: mean recall across the 100 classes.
- Head/Medium/Tail recall: mean per-class recall within each group.

Group membership uses real retained training counts: Head `>=100`, Medium
`20..99`, Tail `<20`. Synthetic counts do not affect membership. Include test
support per class in the metric output. A missing class has recall `null`;
balanced accuracy or a group mean is `null` if any of its required classes is
missing. An empty group also has a `null` mean. The full CIFAR-100 test set
contains all classes, so normal evaluation produces complete metrics.

Evaluation may be run repeatedly on a saved checkpoint. It does not update the
checkpoint. A simple results table can compare saved metric files later; it is
not required to connect a method to the pipeline.

## Minimal file layout and commands

```text
shared_pipeline_plan.md
README.md
pyproject.toml             # package metadata and required dependencies
.gitignore
pipeline/
  __init__.py
  data.py                 # preparation, data record, datasets, transforms
  model.py                # plain ResNet-32
  runtime.py              # seeding and checkpoint helpers
  train.py                # training runner and method mapping
  evaluate.py             # inference, metrics, evaluation command
  methods/
    __init__.py
    baseline.py
configs/
  baseline.json
tests/
  test_data.py
  test_model.py
  test_evaluation.py
  test_pipeline.py
```

Use PyTorch, torchvision, and NumPy, with pytest for verification. Keep the
package self-contained; archived files are references, not runtime imports.

Target commands from the repository root:

```bash
python -m pipeline.data --data-root data/raw --output data/prepared/train.npz --imbalance-factor 0.01 --seed 42 --download
python -m pipeline.train --config configs/baseline.json --output-dir outputs/baseline_seed0 --device cuda
python -m pipeline.evaluate --checkpoint outputs/baseline_seed0/final.pt --data-root data/raw --output outputs/baseline_seed0/metrics.json --device cuda --download
```

`--download` explicitly allows fetching CIFAR-100 when absent. Without it,
commands use existing local data or explain what is missing. Training never
downloads or loads the test set. CPU execution is supported with `--device cpu`;
requesting an unavailable GPU produces a clear error.

## Implementation sequence and acceptance checks

1. **Preparation and data loading.** Implement retention, prepared-file loading,
   dataset views, and transforms. Check expected counts, selection without
   replacement, repeatability with the same preparation seed, and file
   round-tripping. Use artificial images and labels for focused tests.
2. **Plain ResNet-32.** Adapt only the reusable model pieces. Check feature and
   logit shapes, finite forward/backward computation, and identical initial
   weights for equal seeds.
3. **Runner and baseline.** Implement configuration, explicit method selection,
   device/seeding helpers, the ordinary training loop, history, and checkpoint
   saving. Check unsupported method/settings, existing output folders, and a
   tiny CPU training execution through the public method interface.
4. **Evaluator.** Implement inference, checkpoint restoration, and metrics. Use
   hand-calculated unequal-class examples to distinguish overall from balanced
   accuracy and group recall. Cover missing classes and partial batches. Check
   that loading a checkpoint reproduces the trained model's predictions.
5. **Contributor handoff.** Write setup instructions and the three working
   commands in the README, plus a short example of adding another method. Run
   the baseline-to-checkpoint-to-evaluation path on artificial data. Verify
   training receives only retained training data and evaluation leaves model
   parameters and batch-normalization statistics unchanged.

The shared foundation is ready when these interfaces work together and another
contributor can implement a method by adding its function, config, and mapping
entry. Full CIFAR-100 training and research-method performance checks are later
work; a tiny integration check establishes plumbing, not model quality.

## Archive references

Inspect reusable code with `git show` from
`archive/study-analysis-2026-11-05`:

- `utils/create_lt_split.py`: retention formula and seeded sampling.
- `models/resnet32.py`: plain backbone, residual blocks, and classifier only.
- `scripts/base_trainer.py`: previous class-group thresholds. Its group accuracy
  calculation is sample-weighted; implement the macro recall defined above.

The method research notes remain in that archive. Their unresolved algorithm
details do not block shared implementation. Do not restore the old repository's
training framework or experiment-management machinery along with useful code.
