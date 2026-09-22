# Task 3F-A — Ridge-Based Predictable Routing Feasibility

Exploratory three-fold cross-validation of multi-output Ridge contribution prediction on the permitted OOF development population.

## Protocol and safeguards

- Source commit: `7de775b77bc3a48db5827e3b8dc60d66bdae7701`
- Population: 6507 images; outer fold 0; inner folds [1, 2, 3]
- Router folds: train on (2,3), (1,3), and (1,2); validate on 1, 2, and 3 respectively.
- Inner fold 0, the reserved outer evaluation population, and the original CIFAR-100 test set were not used.
- Original OOF logits, expert order, checkpoints, and training pipeline were not modified.
- Results are exploratory development measurements; no independent validation claim is made.

## Uniform baseline reproduction

Verification match: **True** at tolerance `1e-12`.

| Method | Accuracy | BA | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| uniform_logit | 59.4283% | 35.9328% | 61.9661% | 34.6909% | 7.0094% |
| uniform_probability | 58.5370% | 35.7778% | 60.9649% | 33.6690% | 8.8532% |
| fixed_006 | 60.1660% | 37.2134% | 62.8900% | 34.9384% | 9.9115% |
| fixed_007 | 55.8322% | 36.4403% | 58.1737% | 33.3167% | 14.7290% |
| fixed_010 | 55.9398% | 36.4236% | 58.4063% | 33.2279% | 14.5055% |
| fixed_011 | 52.3436% | 35.4372% | 54.4225% | 32.4557% | 16.7660% |
| uniform_without_ce | 57.6610% | 37.1860% | 60.1762% | 34.2020% | 13.8454% |

## Ridge results

- Adaptive configurations evaluated: **600**; fitted Ridge models: **90**.
- Configurations improving both BA and Tail over uniform logit: **60**.
- Global score-control configurations evaluated: **60**.

| Highlight | Configuration | BA | Tail | Held-out contribution MSE | Global-control MSE |
|---|---|---:|---:|---:|---:|
| Best BA | `adaptive_confidence_only_alpha1000_gamma1_temperature2_lambda0p75` | 36.5502% | 7.3797% | 4.01741 | 4.24451 |
| Best Tail | `adaptive_confidence_only_alpha0p1_gamma0_temperature2_lambda0p5` | 36.3700% | 7.3797% | 3.98914 | 4.23275 |
| Best BA+Tail sum among joint improvements | `adaptive_confidence_only_alpha1000_gamma1_temperature2_lambda0p75` | 36.5502% | 7.3797% | 4.01741 | 4.24451 |

## Interpretation

All 600 adaptive rows, all 60 global-control rows, all 90 fitted-model diagnostics, fold-level metrics, and weight diagnostics are in `ridge_results.json`.
The global control uses the same training folds, targets, class weighting, temperatures, and shrinkage values but receives no image-dependent features.
Any apparent development improvement remains subject to the documented Tail sparsity, overlapping OOF expert-training populations, prior development exposure, and untested generalization to outer/full-data experts.
