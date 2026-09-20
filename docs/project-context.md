# Project Context — MoE Imbalance (CIFAR-100-LT)

> Compact current-state index for this repository. Read `AGENTS.md` first. Open deeper docs only when the task requires them.
>
> Current numbers: [`results.md`](results.md) · verified routing failures: [`problem.md`](problem.md) · literature: [`research.md`](research.md) · historical results: [`archive/historical-results.md`](archive/historical-results.md)

## 1. Project question and status

**Question:** on CIFAR-100-LT (IR=100), can per-sample routing among differently trained experts beat simple uniform logit averaging?

**Success criterion:** improve both Balanced Accuracy (BA) and Tail accuracy over uniform averaging, consistently across the configured seeds.

**Current expert pool:** CE, LAL, BalancedSoftmax, Mixup; each trained for seeds `{78, 88, 1034}`.

**Current status:** uniform logit averaging remains unbeaten by the four pre-registered parameter-free routing rules. The measured reasons are summarized in [`problem.md`](problem.md).

## 2. Canonical protocol

- CIFAR-100 train is exponentially subsampled with `imb_factor=0.01` (IR=100), producing **10,847** training images.
- All 10,847 LT samples are used for training. **There is no validation split.**
- The balanced CIFAR-100 test set (10,000 images) is the **only evaluation set**.
- Reported models are **final-epoch** checkpoints; no checkpoint selection is performed.
- Canonical LT index artifact: `data/processed/lt_ir100_train_indices.npy` (deterministic, seed 42).
- Head/Medium/Tail split is immutable: Head `n >= 100` (35 classes), Medium `20 <= n < 100` (35), Tail `n < 20` (30). The implementation source of truth is `scripts/base_trainer.py::compute_class_groups`.
- Routers in the active registry are parameter-free; fitted routing is excluded under this protocol because there is no honest held-out correctness-label source.

## 3. Active code map

- `configs/*.yaml` — run definitions and expert-specific settings.
- `data/lt_datamodule.py` — active training-data loader.
- `data/protocol_splits.py` — canonical split loader and leakage guards.
- `models/resnet32.py` — backbone/classifier.
- `scripts/train.py` — training entry point.
- `scripts/trainers.py` — expert trainers/registry.
- `scripts/base_trainer.py` — shared training loop, seeding, schedule, numerical guards, class-group definition.
- `scripts/evaluation.py` — BA, Head/Med/Tail, ECE, checkpoint loading, headroom and run-health utilities.
- `scripts/evaluate_experts.py` — current expert + routing evaluation; reads/logs test access.
- `scripts/analyze_subsets.py` — ensemble-size analysis; reads/logs test access.
- `scripts/check_runs.py` — training-run health checks; does not read test data.
- `scripts/router/` — active parameter-free rules: Uniform, Probability, Confidence, TTA.
- `tests/` — protocol, routing, numerical, seeding, and regression checks.

Retired DACE/boosting/PaCo training scripts remain for historical traceability but are not part of the active trainer/evaluation path. Consult historical docs only when investigating them.

## 4. Training/evaluation workflow

1. **Protocol integrity:** `python tests/test_protocol_splits.py`
2. **Train an expert:** `python scripts/train.py --config configs/ce.yaml --seed 78`
3. **Lightweight dry run:** `python scripts/train.py --config configs/ce.yaml --max-batches 2 --epochs 1`
4. **Check completed runs:** `python scripts/check_runs.py --seeds 78 88 1034`
5. **Evaluate current experts/routing:** `python scripts/evaluate_experts.py --seeds 78 88 1034`
6. **Analyze ensemble size:** `python scripts/analyze_subsets.py --seeds 78 88 1034`
7. **Full tests:** `for f in tests/test_*.py; do python "$f"; done`

The two evaluation commands read the test set and append to `docs/test-access-log.md`.

## 5. Current result snapshot

| Method | BA (3 seeds) | Tail |
|:--|--:|--:|
| Best single expert (LAL) | 42.43 ±1.53 | 23.49 ±1.71 |
| **Uniform logit average** | **46.98 ±0.69** | 18.76 ±0.86 |
| Probability average | 45.95 ±0.55 | 18.70 ±0.50 |
| Confidence router | 44.27 ±0.46 | 18.69 ±0.36 |
| TTA router | 44.15 ±0.68 | **19.38 ±0.97** |

For all authoritative numbers, uncertainty, per-seed results, headroom, and ensemble-size analysis, use [`results.md`](results.md).

## 6. Environment and reproducibility

- Reported training environment: Kaggle T4 GPU.
- Local RTX 3060 Laptop/CPU environment is for verification, not the source of reported training metrics.
- Shared recipe: ResNet-32, 200 epochs, SGD, momentum 0.9, weight decay `2e-4`, no Nesterov, LR 0.1 with 5-epoch warmup and decays after epochs 160/180, batch 128.
- `set_seed` runs before model construction and disables TF32 for reproducibility with the reported T4 path.

Exact run settings are stored in `configs/*.yaml` and next to checkpoints as resolved config files.

## 7. Documentation routing

- Need current metrics or training setup → [`results.md`](results.md)
- Need to know why routing failed / what is ruled out → [`problem.md`](problem.md)
- Need prior literature or novelty context → [`research.md`](research.md)
- Need superseded Protocol A/B numbers → [`archive/historical-results.md`](archive/historical-results.md)
- Need provenance of prior bug fixes → [`archive/bugfix-report.md`](archive/bugfix-report.md)
- Need test-set access audit → [`test-access-log.md`](test-access-log.md)
