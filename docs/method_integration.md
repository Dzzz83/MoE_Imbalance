# Adding a training method

Run commands from the repository root.

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

## Verification

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
