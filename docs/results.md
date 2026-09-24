# Results — Original Full-Data CIFAR-100-LT Experiments

> This document is the authoritative record of the original full-data track:
> four experts trained on all 10,847 long-tailed samples with seeds {78, 88,
> 1034}, followed by the historical balanced CIFAR-100 test evaluation. It is
> not a catalog of every experiment in the repository. The completed nested-OOF
> development results are in [`oof-results.md`](oof-results.md).
>
> Superseded protocols are in [`archive/historical-results.md`](archive/historical-results.md)
> and are not comparable. The original test set has already influenced earlier
> research decisions; its measurements are historical results, not untouched
> confirmation of the OOF findings.
>
> Reproduce: `python scripts/evaluate_experts.py --seeds 78 88 1034` ·
> `python scripts/analyze_subsets.py --seeds 78 88 1034` ·
> `python scripts/check_runs.py --seeds 78 88 1034`
>
> Context: [`project-context.md`](project-context.md) ·
> [`problem.md`](problem.md) · [`oof-results.md`](oof-results.md) ·
> [`routing_mechanism.md`](../records/routing_mechanism.md)

**Audit status (2026-09-23).** The historical TTA BA/Tail row and its derived
comparison values are retained for provenance but are **invalid**: the verified
earlier `data/tta.py` implementation used incorrect padding for normalized
tensors and batch-shared crop semantics. Those values must not support a
conclusion pending a separately authorized reevaluation. Historical routing
ECE/average-confidence fields for **Uniform, Confidence and TTA** are also
invalid because the evaluator used a distribution that could differ from the
classifier used by `predict_class`. Probability routing calibration and all
per-expert calibration fields are unaffected. No replacement values are
reported here.

---

## 1. Experimental setup

**Data.** CIFAR-100 with **imbalance factor 0.01** (IR=100): an exponential
per-class subsample, `n_i = n_max · IR^(−i/(C−1))` with `n_max = 500`, `C = 100`,
giving **10,847** training images (head class 500, tail class 5). The balanced
CIFAR-100 **test set (10,000)** is the historical evaluation population for
this full-data track; the repository also contains a separate OOF development
population documented in [`oof-results.md`](oof-results.md).

**No validation split.** Training uses the full long-tailed set and the
**final-epoch model is the reported model**; no checkpoint is selected. The
split is fixed at seed 42 and shared by every experiment.

**Training parameters.** Identical for all four experts except the loss:

| Setting | Value |
|:--|:--|
| Backbone | ResNet-32, 100 classes |
| Optimiser | SGD, momentum 0.9, weight decay 2e-4, **nesterov=False** |
| Learning rate | 0.1 |
| Warmup | linear over 5 epochs |
| LR decay | ×0.01 after epoch 160; ×0.0001 after epoch 180 |
| Epochs | 200 |
| Batch size | 128 |
| Loss — CE | cross-entropy |
| Loss — LAL | logit-adjusted, **τ = 1.0** |
| Loss — BalancedSoftmax | `+log(n_y)` before the softmax |
| Loss — Mixup | CE + mixup, **α = 1.0** |
| Seeds | 78, 88, 1034 (one fixed data split) |
| Checkpoints | final epoch only |
| Hardware | Kaggle T4 GPU, ≈20 min per run |

The schedule (200 epochs, lr 0.1, no nesterov, decay after epochs 160 and 180) is
the reference CIFAR-LT recipe from LDAM-DRW `cifar_train.py`. Decay milestones are
epochs **160** and **180** — the last epochs at the full rate, matching
`decay_epochs` in `configs/*.yaml` — so the reduced rate first **applies** at
epochs 161 and 181.

**Source of truth.** `configs/*.yaml` defines each run; every run also writes its
resolved config next to its checkpoint as `<expert>_seed<N>_config.yaml`, so the
exact parameters behind any number in this document are on disk.

**Recorded deviations from the official repositories.**

| Expert | Official recipe | Used here | Why |
|:--|:--|:--|:--|
| LAL | 1419 epochs, its own step schedule | shared 200-epoch schedule, τ=1.0 | 1419 epochs is ≈7× the other experts' budget and would make the ensemble comparison unfair |
| BalancedSoftmax | 13,000 iterations, batch 512, base_lr 0.05 (iteration-based, no IR=100 epoch config) | shared 200-epoch schedule | no epoch-based configuration exists for this benchmark |
| Mixup | α=1.0 (official default) | α=1.0 | matches both the original paper and LOB Mixup's setting for CIFAR-100-LT |

**Metric definitions.** BA = mean per-class recall. Head/Medium/Tail = mean
recall over classes with ≥100 / 20–100 / <20 training samples (immutable split).
ECE = |accuracy − confidence| averaged over 15 equal-width confidence bins.

## 2. Per-expert performance (test set, 3 seeds)

Four experts, all ResNet-32, trained on the same 10,847 samples with the same
schedule; only the loss/augmentation differs.

| Expert | Paradigm | BA | Head | Medium | Tail | ECE |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| **LAL** | logit-adjusted (τ=1.0) | **42.43 ±1.53** | 59.54 ±1.14 | 41.56 ±1.88 | **23.49 ±1.71** | 28.96 ±1.07 |
| BalancedSoftmax | balanced softmax | 41.34 ±0.78 | 58.88 ±1.32 | 39.87 ±0.91 | 22.60 ±0.39 | 29.49 ±0.56 |
| Mixup | interpolation (α=1.0) | 38.69 ±0.61 | **69.40 ±0.69** | 36.28 ±0.82 | 5.69 ±0.34 | **9.43 ±0.66** |
| CE | cross-entropy (ERM) | 37.69 ±0.73 | 65.15 ±1.35 | 35.10 ±0.57 | 8.67 ±0.42 | 37.89 ±0.57 |

*All values are percentages. BA = balanced accuracy (mean per-class recall).
Head = 35 classes with ≥100 training samples, Medium = 35 classes with
20 ≤ n < 100, Tail = 30 classes with <20 (one class has exactly 20 training
samples and is Medium). Per-expert ECE values in this table are unaffected by
the router-distribution mismatch.*

### Reading the table

- **LAL is the strongest overall and the strongest on tail**; BalancedSoftmax
  is close behind on both.
- **Mixup is the best-calibrated by a factor of three** (ECE 9.43) and the best
  on head classes — but by far the weakest on tail (5.69). It is the only
  expert from an interpolation paradigm, and was selected for calibration and
  paradigm diversity, not tail accuracy.
- **CE is weakest overall and the most overconfident.**
- Published anchors for this protocol (ResNet-32, IR=100): CE ≈38.3, LAL/BS
  ≈42–43. The measured values sit in that range.

## 3. Ensembling baselines (test set, 3 seeds)

| Method | BA | Tail | vs Uniform |
|:--|:--:|:--:|:--:|
| **Uniform (mean of logits)** | **46.98 ±0.69** | 18.76 ±0.86 | — |
| Probability (mean of softmax) | 45.95 ±0.55 | 18.70 ±0.50 | −1.03 |
| Confidence (most confident expert) | 44.27 ±0.46 | 18.69 ±0.36 | −2.72 |
| TTA (confidence over 10 augmented views) | **INVALID — 44.15 ±0.68** | **INVALID — 19.38 ±0.97** | **INVALID — −2.83** |

*Averaging softmax probabilities rather than logits is the conventional
ensemble baseline; both are parameter-free. They differ because the three
logit-adjusted experts carry ~2.3× Mixup's logit magnitude (measured logit std:
CE 4.48, LAL 4.57, BalancedSoftmax 4.57, Mixup 1.98), so a logit average
implicitly down-weights Mixup.*

**Paired against Uniform, per seed:**

| Rule | ΔBA per seed | mean ΔBA | std | verdict |
|:--|:--|:--:|:--:|:--|
| Probability | −1.19, −1.24, −0.66 | −1.03 | 0.32 | worse, consistent in 3/3 seeds |
| Confidence | −3.11, −2.98, −2.06 | −2.72 | 0.57 | **significantly worse** |
| TTA | **INVALID — −2.87, −3.18, −2.45** | **INVALID — −2.83** | **not applicable** | **withdrawn pending authorized reevaluation** |

The valid historical comparisons still show Uniform above Probability and
Confidence on both reported aggregate BA and Tail. The TTA comparison is
withdrawn, so the former blanket statement that every rule failed the
pre-registered BA-and-Tail criterion must not include TTA. The historical
routing ECE/confidence fields for Uniform, Confidence and TTA require
recomputation under the current distribution contract; Probability routing and
per-expert calibration remain valid.

## 4. Routing headroom (test set, per seed)

| Seed | All-wrong floor | Oracle ceiling |
|:--:|:--:|:--:|
| 78 | 38.91 | 61.09 |
| 88 | 40.26 | 59.74 |
| 1034 | 40.02 | 59.98 |
| **mean** | **39.73** | **60.27** |

*All-wrong floor = fraction of samples where **no** expert's existing top-1
prediction is right — unreachable by a hard-selection router. This is not a
limit on adaptive convex logit mixtures: [Task 3E-B](oof-results.md#5-task-3e-b--adaptive-soft-mixture-oracle)
found 321 images correctable by a convex mixture even though every expert's
individual top-1 prediction was wrong. Oracle = fraction where at least one
expert is right.*

Distribution of correct experts per sample (k experts correct):

| Seed | 0 | 1 | 2 | 3 | 4 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 78 | 3891 | 1623 | 1156 | 1005 | 2325 |
| 88 | 4026 | 1541 | 1142 | 1144 | 2147 |
| 1034 | 4002 | 1562 | 1107 | 1141 | 2188 |

**Pairwise Cohen's κ** (agreement between experts, mean over seeds):

| Pair | κ | | Pair | κ |
|:--|:--:|:-:|:--|:--:|
| CE ↔ Mixup | **0.487 ±0.009** | | LAL ↔ Mixup | 0.428 ±0.009 |
| LAL ↔ BalancedSoftmax | 0.458 ±0.012 | | BalancedSoftmax ↔ Mixup | 0.420 ±0.013 |
| CE ↔ LAL | 0.438 ±0.006 | | CE ↔ BalancedSoftmax | 0.433 ±0.018 |

All pairs lie in a narrow 0.42–0.49 band, and κ never exceeds 0.50 — the
"κ < 0.80 ⇒ routable" heuristic used earlier is satisfied by every pair, yet
routing still fails. LAL and BalancedSoftmax implement the *same objective* and
still show κ ≈ 0.46, i.e. no more agreement than unrelated pairs; see
[`problem.md`](problem.md) §3.

## 5. Ensemble size (test set, 3 seeds)

Uniform logit averaging over **every subset** of the four experts:

| Experts | Subsets | Mean BA | Best BA | Best subset |
|:--:|:--:|:--:|:--:|:--|
| 1 | 4 | 40.04 ±0.53 | 42.77 ±1.01 | LAL |
| 2 | 6 | 44.19 ±0.50 | 46.44 ±1.03 | LAL + BalancedSoftmax |
| 3 | 4 | 45.97 ±0.57 | 47.59 ±0.91 | LAL + BalancedSoftmax + Mixup |
| 4 | 1 | **46.98 ±0.69** | 46.98 ±0.69 | all four |

Under probability averaging the same curve is 40.04 / 43.21 / 44.79 / 45.95 —
logit averaging is better at every size.

- Each additional expert still helps, with shrinking increments: **+4.15**,
  **+1.79**, **+1.01** points. All exceed seed noise (±0.5–0.7), so four
  experts are justified over three, and three over two.
- **Do not quote the "Best BA" column as a result.** Choosing the best subset on
  the test set is selection-on-test. The best 3-expert subset (47.59) exceeds all
  four (46.98), but the unbiased mean for 3 experts is 45.97 — *below* four.
  `scripts/analyze_subsets.py` labels this automatically.

## 6. Training runs

All 12 runs were healthy (`scripts/check_runs.py`): each trained for 200
epochs, used the correct LR decay at the 160/180 milestones, had monotone loss,
and saved a final checkpoint.

| Expert | Loss @10 | Loss @160 | Loss @200 | Final train acc |
|:--|:--:|:--:|:--:|:--:|
| CE | 2.404 | 0.338 | 0.079 | 98.5% |
| LAL | 2.283 | 0.316 | 0.068 | 96.9% |
| BalancedSoftmax | 2.236 | 0.354 | 0.077 | 96.6% |
| Mixup | 3.265 | 2.051 | 1.698 | — |

*Seed-78 values, representative. Train accuracy of 96–98% is the documented
memorisation of a 10,847-sample training set; with no validation split it is
expected and cannot be used for early stopping. Mixup reports no train accuracy
because its logits come from mixed inputs. Its higher loss is the mixup
objective, not a failure to converge.*

---

## 7. Relationship to the OOF track

- The full-data checkpoints remain valid and available for the original
  three-seed comparisons. Their predictions on their own training images are
  not honest supervision for a fitted router.
- The nested-OOF pipeline was introduced to generate held-out predictions on a
  separate development population without using the original test set. Its
  completed seed-78 findings and the one locked outer-fold-0 Ridge evaluation
  are recorded in [`oof-results.md`](oof-results.md). The outer candidate used
  no Sinkhorn and failed the frozen fixed-reference Pareto expansion gate.
- The OOF fixed-weight and soft-mixture results are not directly comparable to
  the three-seed full-data test results: they use different training
  populations, a single seed, and OOF populations rather than the original
  balanced test set.
