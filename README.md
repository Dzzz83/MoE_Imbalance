# Shared CIFAR-100-LT pipeline

## Project overview

Shared data preparation, ResNet-32, training entry point, and evaluation for
comparing different training methods. Ordinary cross-entropy training is the
first working method. OT reweighting and synthetic-sample methods are planned
extensions; their algorithms are not implemented yet. The implementation plan is in
[shared_pipeline_plan.md](shared_pipeline_plan.md).

Each method receives the same retained training data and a fresh model, owns
its training algorithm, and produces a model evaluated through the shared
checkpoint and metric interfaces.

### Setup

Use Python 3.11 or newer and install a compatible PyTorch/torchvision pair for
your CPU or CUDA environment. Then, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

### Prepare, train, evaluate

```bash
python -m pipeline.data --data-root data/raw --output data/prepared/train.npz --imbalance-factor 0.01 --seed 42 --download
python -m pipeline.train --config configs/baseline.json --output-dir outputs/baseline_seed0 --device cuda
python -m pipeline.evaluate --checkpoint outputs/baseline_seed0/final.pt --data-root data/raw --output outputs/baseline_seed0/metrics.json --device cuda --download
```

Use `--device cpu` for CPU execution. The default is CPU; requesting CUDA when
unavailable fails with an explanation. `--download` allows fetching the raw
CIFAR-100 files if absent. Without it, preparation and evaluation require local
data. Training only reads the prepared file. Paths in configs are relative to
the working directory, so run commands from the repository root.

Preparation selects 10,847 of the original 50,000 training images, using fixed
data seed 42 and imbalance factor 0.01. Every method uses this same saved pool;
changing the training seed does not change the pool. There is no additional
shared validation split. Dataset items are `(image, label)` with labels `0..99`.

The original 10,000-image test set is the final evaluation set. Repeated
evaluation is allowed. Test data must not supply training updates, prototypes,
synthetic examples, or other fitted state. The runner saves the final scheduled
model, independently of its test score.

Edit `configs/baseline.json` for training settings. The baseline uses
cross-entropy, SGD, and cosine learning-rate decay. An output folder contains
`config.json` (including resolved defaults), `history.json`, and `final.pt`.
Evaluation adds `metrics.json`. Preparation and training refuse to overwrite
existing data/output paths; choose another path for another execution. Running
evaluation again can replace its metric file without changing the checkpoint.
Data and outputs are ignored by Git.

Metrics include overall accuracy, balanced accuracy, per-class recall, and
macro Head/Medium/Tail recall. Groups use real training counts: Head `>=100`,
Medium `20..99`, Tail `<20`. JSON scores are fractions; terminal scores are
percentages. Missing test classes produce `null` recalls and `null` means for
groups that require those classes. Full test evaluation covers all 100 classes.

## Project structure

```text
pipeline/
  data.py                 # Retention, prepared data, datasets, and transforms
  model.py                # Plain CIFAR ResNet-32
  runtime.py              # Seeds, devices, JSON, and checkpoint helpers
  train.py                # Shared runner and explicit method mapping
  evaluate.py             # Test inference and common metrics
  methods/
    baseline.py           # Cross-entropy training; integration example
configs/
  baseline.json           # Baseline training settings
tests/
  conftest.py             # Artificial images and CPU test setup
  test_data.py            # Preparation, retention, and data validation
  test_model.py           # Model shapes, initialization, and gradients
  test_evaluation.py      # Metrics and inference behavior
  test_pipeline.py        # Training-to-evaluation integration and CLI checks
AGENTS.md                 # Project-specific contributor guidance
shared_pipeline_plan.md   # Interfaces, responsibilities, and implementation plan
pyproject.toml            # Package dependencies and pytest configuration
```

Preparation creates `data/` and executions create `outputs/`; both are ignored
by Git. Training methods belong in `pipeline/methods/`, with their settings in
`configs/`. Data preparation, checkpoint saving, and evaluation stay shared.

### Add a training method

Create `pipeline/methods/your_method.py` with two small functions:

```python
def resolve_options(options) -> dict:
    # Apply your defaults and reject invalid or unknown method options.
    ...


def train(model, train_data, config, output_dir, device) -> list[dict]:
    # Train the supplied model in place; return epoch/loss history.
    ...
```

Import the module in `pipeline/train.py` and add one mapping entry:

```python
METHODS = {
    "baseline": (baseline.train, baseline.resolve_options),
    "your_method": (your_method.train, your_method.resolve_options),
}
```

Add a JSON config with `method: "your_method"`, a prepared-data path, shared
loader/seed settings, and your `method_options`. See the baseline for a complete
example. The runner resolves options before training and saves the resulting
configuration. Each history record has a one-based `epoch` and mean `loss`;
multistage methods also include `stage`. Additional scalar measurements are
allowed. Use `output_dir` for optional method-specific files.

`train_data` contains raw retained `uint8` images `(N, 32, 32, 3)`, `int64`
labels `(N,)`, real `class_counts` `(100,)`, and `data_settings`. Build views
using `pipeline.data.make_dataset` with `training_transform()` or
`evaluation_transform()`; custom transforms receive PIL images. Create ordinary
PyTorch loaders for your batching/sampling. Use a seeded `torch.Generator` and
`pipeline.runtime.seed_worker` as shown in the baseline. Treat the retained
arrays as read-only and keep synthetic examples in method-owned datasets.

The model exposes `model(images)` for 100 logits, `model.backbone(images)` for
64 features, and `model.fc(features)` for 100 logits. Your method owns its
stages, loss, optimizers, freezing, synthesis, and any temporary state. If you
train a copy, restore its complete final `state_dict` into the supplied model,
including batch-normalization buffers. Final inference must use the plain
ResNet-32 without training-only objects. The runner handles checkpoint saving;
the evaluator does not import method implementations.

Checkpoints contain `model_state_dict`, `config`, real `class_counts` as a CPU
tensor, and `data_settings`. `pipeline.runtime.load_checkpoint(path, device)`
restores them onto CPU or CUDA. The evaluator needs no training images.

### Verify

```bash
python -m pytest
```

Tests use artificial images and labels. They check retention, model shapes and
gradients, training reproducibility, configuration errors, checkpoint loading,
and baseline-to-evaluation integration without downloading CIFAR-100. For a
quick integration check alone:

```bash
python -m pytest tests/test_pipeline.py::test_baseline_cli_checkpoint_and_evaluation
```

These checks verify the shared plumbing. Full CIFAR-100 training, CUDA execution,
and research-method performance require separate experiments. Reproducibility
checks apply to the same software/hardware environment.

## Results

## References

1. Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
   [CIFAR-10 and CIFAR-100 datasets](https://cave.cs.toronto.edu/kriz/cifar.html).
   Dataset source.
2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
   [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
   ResNet architecture reference, 2015 preprint.
3. Dandan Guo et al.
   [Learning to Re-weight Examples with Optimal Transport for Imbalanced Classification](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a39a9aceda771cded859ae7560530e09-Abstract-Conference.html).
   NeurIPS 2022. Reference for the planned reweighting method.
4. Yuguang Yan et al.
   [Oversampling for Imbalanced Data via Optimal Transport](https://tanmingkui.github.io/files/publications/Oversampling.pdf).
   AAAI 2019. Reference for the planned synthetic-sample adaptation.
