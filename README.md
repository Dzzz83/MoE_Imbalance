# MoE Imbalance — Expert Routing on Long-Tailed CIFAR-100

Can **routing between differently-trained experts** beat simply **averaging** them
on a long-tailed dataset (CIFAR-100, imbalance ratio 100:1)?

This repository contains the full pipeline — data protocol, four expert models,
three seeds each, and every routing and ensembling mechanism tested against them.

## Headline result

> **No routing mechanism beats uniform averaging.** Across 20+ mechanisms and
> three independent seed runs, plain uniform averaging of the four experts
> reaches **46.98% balanced accuracy**, while the best per-sample routing rule
> reaches **44.27%** — worse, consistently, in every seed.

| | Balanced accuracy | Tail accuracy |
|:--|:--:|:--:|
| Best single expert (LAL) | 42.43 ±1.53 | 23.49 ±1.71 |
| **Uniform averaging (baseline)** | **46.98 ±0.69** | 18.76 ±0.86 |
| Best per-sample routing rule (Confidence) | 44.27 ±0.46 | 18.69 ±0.36 |

*3 seeds {78, 88, 1034}, balanced 10,000-image test set, mean ± std.*

The result is not a tuning failure. **39.73%** of test images are classified
wrongly by *every* expert — no router can reach them — and the remaining 13
points of headroom require per-sample expertise that the pool does not contain.
See [`records/routing_mechanism.md`](records/routing_mechanism.md) for every
mechanism tried and [`records/routing-preregistration.md`](records/routing-preregistration.md)
for the frozen evaluation protocol.

---

## Contents

- [Overview](#overview)
- [Data protocol](#data-protocol)
- [Experts](#experts)
- [Training parameters](#training-parameters)
- [Results](#results)
- [Why routing fails](#why-routing-fails)
- [Repository structure](#repository-structure)
- [Reproducing](#reproducing)
- [Reproducibility notes](#reproducibility-notes)
- [References](#references)

## Overview

The premise is the standard one for mixture-of-experts on imbalanced data:
different training objectives produce experts that are strong on *different*
samples, so a router that picks the right expert per sample should beat any
fixed combination. The literature (RIDE, SADE, BalPoE) reports gains of this
kind on CIFAR-100-LT.

This project tests that premise carefully on a single fixed, standard protocol:

1. **Four experts**, each a ResNet-32 trained on the identical long-tailed
   training set, differing only in loss / augmentation paradigm.
2. **Three seeds** per expert, so every number carries a mean and a standard
   deviation.
3. **Only parameter-free combination rules**, because the protocol has no
   validation split — any rule that fitted parameters would be fitting on the
   test set.
4. **A frozen candidate set and an access log**, so rules could not be selected
   after seeing test results.

The honest outcome is a **negative result with a measured explanation**, plus a
reusable pipeline: a corrected data protocol, verified training code, and an
evaluation harness that reports headroom explicitly.

## Data protocol

```
CIFAR-100 train (50,000, balanced)
  └── exponential subsampling, imbalance factor 0.01  (IR = 100)
        n_i = 500 · 100^(−i/99)        →  10,847 images
        head class 500, tail class 5
        └── ALL 10,847 are the training set
              data/processed/lt_ir100_train_indices.npy   (committed, 87 KB)

CIFAR-100 test (10,000, balanced)   →   the ONLY evaluation set
```

| Property | Value |
|:--|:--|
| Imbalance factor (`imb_factor`) | **0.01** |
| Imbalance ratio (IR) | **100 : 1** |
| Training images | **10,847** |
| Test images | 10,000 (balanced, 100 per class) |
| Validation split | **none** |
| Split seed | 42 (one fixed split, shared by every experiment) |

**Class groups** (immutable, defined by training counts):

| Group | Criterion | Classes |
|:--|:--|:--:|
| Head | ≥ 100 training samples | 30 |
| Medium | 20 – 100 | 36 |
| Tail | < 20 | 34 |

There is deliberately **no validation set**: training uses the full long-tailed
set and the **final-epoch model is the reported model**. Splitting off 20% of an
already tiny 10,847-image set is both non-standard and, as measured here, leaves
some tail classes with zero samples.

The index artifact is committed and deterministic — regenerating it with
`python utils/create_lt_split.py` reproduces the identical array.

## Experts

Four ResNet-32 models, identical in every respect except the objective:

| Expert | Paradigm | Loss | Why this one |
|:--|:--|:--|:--|
| **CE** | classification boundary | cross-entropy | the ERM baseline |
| **LAL** | logit adjustment | logit-adjusted, τ=1.0 | balances head/tail logits |
| **BalancedSoftmax** | logit adjustment | `+log(n_y)` | complementary re-weighting variant |
| **Mixup** | interpolation | CE + mixup, α=1.0 | best-calibrated expert; a third, non-CE paradigm |

*Note:* with τ = 1.0, LAL and Balanced Softmax implement the **same objective**
(they differ by a class-independent constant that cancels in the softmax). Both
are kept for comparability with earlier results, but they contribute no
paradigm diversity — see [Why routing fails](#why-routing-fails).

## Training parameters

Identical for all four experts (verified against the config each run wrote):

| Setting | Value |
|:--|:--|
| Backbone | ResNet-32, 100 classes |
| Optimiser | SGD, momentum 0.9, weight decay 2e-4, **nesterov = False** |
| Learning rate | 0.1 |
| Warmup | linear over 5 epochs |
| LR decay | ×0.01 after epoch 160; ×0.0001 after epoch 180 |
| Epochs | 200 |
| Batch size | 128 |
| Loss — CE | cross-entropy |
| Loss — LAL | logit-adjusted, **τ = 1.0** |
| Loss — BalancedSoftmax | `+log(n_y)` before softmax |
| Loss — Mixup | CE + mixup, **α = 1.0** |
| Seeds | 78, 88, 1034 |
| Checkpoints | final epoch only |
| Hardware | Kaggle T4 GPU, ≈20 min per run |

The schedule (200 epochs, lr 0.1, no nesterov, decay after epochs 160 and 180) is
the reference CIFAR-LT recipe from LDAM-DRW `cifar_train.py`. Decay milestones are
epochs **160** and **180** — the last epochs at the full rate, matching
`decay_epochs` in `configs/*.yaml` — so the reduced rate first **applies** at
epochs 161 and 181.

**Source of truth.** [`configs/*.yaml`](configs) defines each run, and every run
also writes the config it resolved to as
`checkpoints/<expert>_seed<N>_config.yaml`. The parameters behind any number
below are therefore on disk in machine-readable form.

**Deviations from the official repos** (recorded deliberately):

| Expert | Official recipe | Used here | Reason |
|:--|:--|:--|:--|
| LAL | 1419 epochs, own step schedule | shared 200-epoch schedule | 1419 is ≈7× the other experts' budget; would make the ensemble comparison unfair |
| BalancedSoftmax | 13,000 iterations, batch 512, base_lr 0.05 | shared 200-epoch schedule | the official recipe is iteration-based with no IR=100 epoch equivalent |
| Mixup | α = 1.0 | α = 1.0 | matches the original paper and LOB Mixup's CIFAR-100-LT setting |

## Results

All numbers: balanced 10,000-image test set, 3 seeds, mean ± std. BA = balanced
accuracy (mean per-class recall). ECE = expected calibration error (15 bins).

### Per-expert performance

| Expert | BA | Head | Medium | Tail | ECE |
|:--|:--:|:--:|:--:|:--:|:--:|
| **LAL** | **42.43 ±1.53** | 59.54 ±1.14 | 41.56 ±1.88 | **23.49 ±1.71** | 28.96 ±1.07 |
| BalancedSoftmax | 41.34 ±0.78 | 58.88 ±1.32 | 39.87 ±0.91 | 22.60 ±0.39 | 29.49 ±0.56 |
| Mixup | 38.69 ±0.61 | **69.40 ±0.69** | 36.28 ±0.82 | 5.69 ±0.34 | **9.43 ±0.66** |
| CE | 37.69 ±0.73 | 65.15 ±1.35 | 35.10 ±0.57 | 8.67 ±0.42 | 37.89 ±0.57 |

The paradigms behave as designed: the logit-adjusted experts own the tail, Mixup
is the best-calibrated by a factor of three and the best on head classes but the
weakest on tail, and plain CE is weakest overall and the most overconfident.
These sit in the published range for this protocol (CE ≈38.3, LAL/BS ≈42–43).

### Combination rules

| Rule | Type | BA | Tail |
|:--|:--|:--:|:--:|
| **Uniform** | mean of expert logits | **46.98 ±0.69** | 18.76 ±0.86 |
| Probability | mean of expert softmax probabilities | 45.95 ±0.55 | 18.70 ±0.50 |
| Confidence | pick the single most-confident expert | 44.27 ±0.46 | 18.69 ±0.36 |
| TTA | confidence over 10 augmented views | 44.15 ±0.68 | 19.38 ±0.97 |

None beats uniform on **both** BA and tail accuracy, which is the pre-registered
success condition. Confidence and TTA are *significantly worse* on BA (the paired
difference exceeds its own standard deviation and is negative in all three
seeds). Reporting both logit and probability averaging matters here because the
two are known to differ on imbalanced data (Buchanan et al., 2023), and they do:
−1.03 points.

### Routing headroom

| Seed | All-wrong floor | Oracle ceiling |
|:--:|:--:|:--:|
| 78 | 38.91 | 61.09 |
| 88 | 40.26 | 59.74 |
| 1034 | 40.02 | 59.98 |
| **mean** | **39.73** | **60.27** |

*All-wrong floor* = fraction of images **no** expert classifies correctly —
unreachable by any router. *Oracle ceiling* = fraction where at least one expert
is right. The accessible headroom is `60.27 − 46.98 = 13.3` points, and it would
require perfect per-sample expert selection.

### How many experts?

Uniform averaging over **every subset** of the four experts:

| Experts | Subsets | Mean BA | Best BA |
|:--:|:--:|:--:|:--:|
| 1 | 4 | 40.04 ±0.53 | 42.77 ±1.01 |
| 2 | 6 | 44.19 ±0.50 | 46.44 ±1.03 |
| 3 | 4 | 45.97 ±0.57 | 47.59 ±0.91 |
| 4 | 1 | **46.98 ±0.69** | 46.98 ±0.69 |

Each additional expert still helps, with shrinking increments (+4.15, +1.79,
+1.01 points), all larger than seed noise. **Use the "Mean" column**: the "Best"
column picks the winning subset *on the test set*, which is selection-on-test —
the best 3-expert subset (47.59) looks better than all four (46.98), but the
unbiased mean for 3 experts (45.97) is worse.

## Why routing fails

Three measured reasons, in order of importance:

1. **A high all-wrong floor (39.73%).** Nearly 40% of images are wrong for all
   four experts. Four ResNet-32s on the same data fail on the same hard images,
   so there is little for a router to arbitrate.

2. **The routing signal does not exist where it is needed.** Measured on savable
   samples — where averaging is wrong but some expert is right — the correct
   expert is the *least* confident one **83.9%** of the time. A confidence-based
   rule therefore points away from the answer, which is exactly what the
   Confidence and TTA rows show. In an earlier cascade architecture the routing
   score was outright anti-predictive of correctness (AUROC **0.327**).

3. **The pool's diversity is mostly initialisation noise.** κ between experts
   spans only 0.42–0.49, and LAL ↔ BalancedSoftmax — *the same objective,
   trained twice* — sit at κ = 0.458, no higher than any genuinely different
   pair. Disagreement between experts therefore carries little
   paradigm-specific signal to exploit.

Also relevant: the protocol has **no honest held-out data**, so no router can be
fitted at all; only parameter-free rules can be evaluated. Every fitted
mechanism tried earlier (trust meters, pairwise comparators, clustering,
learned gates, threshold tuning) was removed for this reason, with its results
recorded in [`records/routing_mechanism.md`](records/routing_mechanism.md).

## Repository structure

```
├── configs/                  one YAML per expert — the only place a run is defined
├── data/
│   ├── cifar_lt.py           LongTailCIFAR100 dataset
│   ├── lt_datamodule.py      training loader; the only reader of the split artifact
│   ├── protocol_splits.py    canonical artifact loader + no-leakage guards
│   ├── mixup.py              Mixup augmentation
│   └── tta.py                augmented views for test-time augmentation
├── losses/                   ce, lal, balanced_softmax (+ retired paco/contrastive/kl)
├── models/resnet32.py        ResNet-32 backbone
├── scripts/
│   ├── train.py              config-driven training entry point
│   ├── trainers.py           the four expert trainers + registry
│   ├── config.py             strict typed config schema
│   ├── base_trainer.py       training loop, LR schedule, NaN/Inf guards
│   ├── evaluation.py         metrics, ExpertPool, HeadroomAnalyzer, RunHealthChecker
│   ├── evaluate_experts.py   per-expert + routing evaluation (reads test set)
│   ├── analyze_subsets.py    ensemble-size analysis (reads test set)
│   ├── check_runs.py         training-run health check (no test-set access)
│   ├── subsets.py            subset-analysis library
│   └── router/               parameter-free routing rules only
├── records/                  routing mechanisms tried + frozen pre-registration
├── tests/                    17 self-contained suites
├── utils/create_lt_split.py  generator for the committed split artifact
└── checkpoints/              trained models: <Expert>_seed<N>_final.pt + history + config
```

## Reproducing

**Requirements:** Python 3, `pip install -r requirements.txt`. Data downloads
automatically via torchvision (or place `cifar-100-python/` in `./data`).

```bash
# 1. Data — the split artifact is committed; regenerate only if needed
python utils/create_lt_split.py
python tests/test_protocol_splits.py          # verify integrity (fast, no GPU)

# 2. Train an expert (≈20 min per run on a T4 GPU)
python scripts/train.py --config configs/ce.yaml --seed 78

# 3. Verify the runs behaved correctly (no test-set access)
python scripts/check_runs.py --seeds 78 88 1034

# 4. Evaluate (READS THE TEST SET — see notes below)
python scripts/evaluate_experts.py --seeds 78 88 1034
python scripts/analyze_subsets.py  --seeds 78 88 1034
```

Dry run (2 batches, CPU-safe) to check the code without a GPU:

```bash
python scripts/train.py --config configs/ce.yaml --epochs 1 --max-batches 2
```

Run the whole test suite:

```bash
for f in tests/test_*.py; do python "$f"; done
```

## Reproducibility notes

- **Seeds cover initialisation.** `set_seed` runs *before* the model is
  constructed, so `--seed N` fully determines a run; two runs with the same seed
  produce identical weights, loss history and data order (there are regression
  tests for this).
- **cuDNN is pinned and TF32 is disabled.** Measured CPU/GPU logits diverge by
  1.2e-01 with TF32 on versus 2.8e-04 off. Kaggle's T4 (Turing) has no TF32, so
  leaving it on would make local verification unrepresentative of reported runs.
- **The test set is read only by two entry points**, each of which appends to a
  local access log, so any read is auditable.
- **No parameter is fitted on held-out data** — the router interface has no
  `train()`/`fit()` method at all, and a test enforces that no router module
  references held-out labels.
- **GPU tests skip cleanly** on a CPU-only machine, so the suite is green
  anywhere.

### Not included

Parts of an earlier, retired line of work remain in the tree for provenance:
`scripts/train_dace_*.py`, `train_paco*.py`, `train_lal_weighted.py`,
`evaluate_dace.py`, `diagnose_dace*.py` and their losses/datasets. They target an
older trainer API and **do not run** against the current one. All other tests
pass regardless.

## References

Key papers behind the design and the baselines:

1. Wang, X. et al. (2021). *RIDE: Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts*. ICLR. [arXiv:2010.01809](https://arxiv.org/abs/2010.01809)
2. Zhou, Z. et al. (2022). *SADE: Self-Supervised Aggregation of Diverse Experts*. NeurIPS. [arXiv:2107.09249](https://arxiv.org/abs/2107.09249)
3. Aimar, E. S. et al. (2023). *Balanced Product of Calibrated Experts*. CVPR. [arXiv:2206.05260](https://arxiv.org/abs/2206.05260)
4. Menon, A. K. et al. (2021). *Long-tail learning via logit adjustment*. ICLR. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
5. Ren, J. et al. (2020). *Balanced Meta-Softmax for Long-Tailed Visual Recognition*. NeurIPS. [arXiv:2007.10740](https://arxiv.org/abs/2007.10740)
6. Cao, K. et al. (2019). *LDAM: Label-Distribution-Aware Margin Loss*. NeurIPS. [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
7. Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
8. Buchanan, E. K. et al. (2023). *The Effects of Ensembling on Long-Tailed Data*. NeurIPS Heavy Tails Workshop. [code](https://github.com/ekellbuch/longtail_ensembles)
9. Zhang, S. et al. (2021). *Label-Occurrence-Balanced Mixup for Long-Tailed Recognition*. [arXiv:2110.04964](https://arxiv.org/abs/2110.04964)

## Records

- [`records/routing_mechanism.md`](records/routing_mechanism.md) — every routing
  and combination mechanism tried, with its measured result and why it was
  removed where applicable.
- [`records/routing-preregistration.md`](records/routing-preregistration.md) —
  the frozen candidate set, metrics and decision rule, including the dated
  amendments made after results were seen.
