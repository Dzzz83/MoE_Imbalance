# Shared CIFAR-100-LT pipeline

## Project overview

Compare training methods for imbalanced image classification using the same
**data, ResNet-32 model, and evaluation**.

| Method | What it does | Status |
|---|---|---|
| Baseline | Ordinary cross-entropy training | Implemented |
| OT reweighting | Learns how much each real example contributes | Planned |
| OT synthesis | Creates extra features for underrepresented classes | Planned |

All methods use **10,847 retained training images** from CIFAR-100 at imbalance
factor **0.01**. The original **10,000 test images** are used for evaluation.
Repeated evaluation is allowed; test data never enters training.

### Quick start

Use Python 3.11+ and run these commands from the repository root.

**1. Install**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

**2. Prepare the training data**

```bash
python -m pipeline.data --download
```

**3. Train the baseline**

```bash
python -m pipeline.train \
  --config configs/baseline.json \
  --output-dir outputs/baseline_seed0 \
  --device cuda
```

**4. Evaluate**

```bash
python -m pipeline.evaluate \
  --checkpoint outputs/baseline_seed0/final.pt \
  --output outputs/baseline_seed0/metrics.json \
  --device cuda
```

- **No GPU?** Replace `--device cuda` with `--device cpu`.
- **Change training settings:** edit [configs/baseline.json](configs/baseline.json).
- **Outputs:** configuration, training history, final checkpoint, and metrics.
- **Another training run:** choose a new output folder to avoid overwriting results.

Evaluation reports overall accuracy, balanced accuracy, and per-class and
Head/Medium/Tail recall.

### Extract features for OT synthesis

After baseline pretraining, freeze the backbone and cache the retained training
features using deterministic preprocessing:

```bash
python -m pipeline.extract_features \
  --checkpoint outputs/baseline_seed0/final.pt \
  --train-data data/prepared/train.npz \
  --output outputs/baseline_seed0/train_features.pt \
  --device cuda
```

The cache contains raw features `(10847, 64)`, aligned labels, real class counts,
data settings, the synthesis coordinate scale, and input file SHA-256 hashes.
Load it with `torch.load(path, map_location="cpu", weights_only=True)`.
Extraction does not modify the pretrained checkpoint or use test images.
Existing caches are not overwritten. Loading the checkpoint in a new process
requires freezing the backbone again before any further classifier training.

## Project structure

| Location | Purpose |
|---|---|
| `pipeline/data.py` | Data preparation and image transforms |
| `pipeline/model.py` | ResNet-32 |
| `pipeline/train.py` | Training entry point |
| `pipeline/evaluate.py` | Evaluation and metrics |
| `pipeline/runtime.py` | Seeds, devices, and checkpoint helpers |
| `pipeline/methods/` | Individual training algorithms |
| `configs/` | Training settings |
| `tests/` | Automated checks using artificial data |
| `docs/` | Method integration guide and paper notes |

**For contributors:** [Adding a method](docs/method_integration.md) ·
[Pipeline design](shared_pipeline_plan.md)

Run the checks with `python -m pytest`. They require no dataset download.

## Results

## References

- [CIFAR-100 dataset](https://cave.cs.toronto.edu/kriz/cifar.html).
- He et al. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Guo et al. [Learning to Re-weight Examples with Optimal Transport for Imbalanced Classification](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a39a9aceda771cded859ae7560530e09-Abstract-Conference.html). NeurIPS 2022.
- Yan et al. [Oversampling for Imbalanced Data via Optimal Transport](https://tanmingkui.github.io/files/publications/Oversampling.pdf). AAAI 2019.
