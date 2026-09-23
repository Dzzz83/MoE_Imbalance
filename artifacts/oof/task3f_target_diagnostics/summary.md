# Task 3F-E — Supervised contribution target diagnostics

Retrospective diagnostics on the frozen outer-fold-0 / inner-folds-1–3 OOF development population. No router or expert was retrained.

## Protocol and safeguards

- Population: 6507 images; Head/Medium/Tail = 5294 / 1030 / 183
- Only Task 3C OOF logits from inner folds 1–3 were loaded.
- Inner fold 0, the reserved outer-evaluation population, the original CIFAR-100 test set, and original full-data expert predictions were excluded.
- Labels appear only in retrospective target, margin, group, and outcome calculations.
- The saved Task 3F-A Ridge scores and highlighted weights were read-only inputs.

## Investigation A — target and Ridge score distributions

| Group | Expert | Target mean | Target median | Target positive | Target highest | Ridge mean | Ridge highest |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Head | CE | -0.1069 | 0.0020 | 63.28% | 37.46% | -0.4505 | 0.08% |
| Head | LAL | -0.5005 | -0.0004 | 45.90% | 19.83% | -0.3536 | 0.59% |
| Head | BalancedSoftmax | -0.4610 | -0.0005 | 45.37% | 19.12% | -0.2598 | 0.96% |
| Head | Mixup | 1.0683 | -0.0001 | 46.24% | 23.59% | 1.0639 | 98.38% |
| Medium | CE | -0.7554 | -0.1057 | 42.04% | 19.81% | -0.4583 | 0.00% |
| Medium | LAL | -0.4366 | -0.0044 | 47.77% | 23.01% | -0.4882 | 0.19% |
| Medium | BalancedSoftmax | -0.1552 | 0.0008 | 52.04% | 25.15% | -0.3587 | 0.29% |
| Medium | Mixup | 1.3472 | 0.4832 | 60.19% | 32.04% | 1.3052 | 99.51% |
| Tail | CE | -2.0159 | -2.3042 | 20.22% | 8.74% | -0.4605 | 0.00% |
| Tail | LAL | 0.8306 | 0.9976 | 68.85% | 38.80% | -0.4899 | 0.55% |
| Tail | BalancedSoftmax | 0.4873 | 0.2101 | 56.83% | 27.32% | -0.4569 | 0.00% |
| Tail | Mixup | 0.6980 | 0.4643 | 55.74% | 25.14% | 1.4073 | 99.45% |

| Group | Target/Ridge top-expert agreement | Score MSE | Score MAE | Pearson |
|:--|--:|--:|--:|--:|
| Head | 23.35% | 3.6384 | 1.2028 | 0.3715 |
| Medium | 32.14% | 5.5298 | 1.6855 | 0.3691 |
| Tail | 25.68% | 6.4697 | 2.0392 | 0.2395 |

### Tail cases where a rebalanced target exceeds Mixup

| Expert | Actual target higher than Mixup | Ridge still predicts Mixup highest | Ridge pairwise rebalanced score higher |
|:--|--:|--:|--:|
| LAL | 103 (56.28%) | 102 (99.03%) | 1 (0.97%) |
| BalancedSoftmax | 94 (51.37%) | 94 (100.00%) | 0 (0.00%) |

## Investigation B — fixed convex perturbations

Positive-target rates below are conditional on a positive target. Correction means an incorrect uniform prediction becomes correct; deterioration means a correct uniform prediction becomes incorrect.

| Expert | ε | Group | Δ true log p | Δ true margin | Correction | Deterioration | Positive target → correction | Positive target → deterioration |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| CE | 0.10 | Head | -0.0204 | 0.0607 | 2.48% | 1.49% | 1.34% | 0.09% |
| CE | 0.10 | Medium | -0.0904 | -0.1097 | 0.91% | 4.01% | 1.39% | 0.46% |
| CE | 0.10 | Tail | -0.2155 | -0.2373 | 0.59% | 7.69% | 2.70% | 0.00% |
| CE | 0.25 | Head | -0.0849 | 0.0838 | 5.57% | 4.05% | 3.01% | 0.45% |
| CE | 0.25 | Medium | -0.2789 | -0.3359 | 2.59% | 10.16% | 3.93% | 0.69% |
| CE | 0.25 | Tail | -0.5889 | -0.6464 | 0.59% | 30.77% | 2.70% | 0.00% |
| CE | 0.50 | Head | -0.2736 | -0.0259 | 9.26% | 7.79% | 5.01% | 0.93% |
| CE | 0.50 | Medium | -0.7201 | -0.8630 | 4.12% | 17.11% | 6.24% | 2.08% |
| CE | 0.50 | Tail | -1.3338 | -1.4593 | 1.76% | 61.54% | 8.11% | 0.00% |
| CE | 1.00 | Head | -0.9081 | -0.6543 | 11.14% | 16.38% | 6.03% | 3.46% |
| CE | 1.00 | Medium | -2.0056 | -2.3404 | 5.03% | 38.24% | 7.62% | 7.39% |
| CE | 1.00 | Tail | -3.1825 | -3.3499 | 2.94% | 76.92% | 13.51% | 2.70% |
| LAL | 0.10 | Head | -0.0597 | -0.0556 | 2.21% | 1.84% | 1.48% | 0.04% |
| LAL | 0.10 | Medium | -0.0583 | -0.0488 | 1.37% | 4.28% | 1.83% | 0.41% |
| LAL | 0.10 | Tail | 0.0677 | 0.0840 | 1.76% | 0.00% | 2.38% | 0.00% |
| LAL | 0.25 | Head | -0.1849 | -0.2055 | 3.80% | 4.97% | 2.76% | 0.41% |
| LAL | 0.25 | Medium | -0.1989 | -0.1931 | 3.35% | 9.36% | 4.47% | 0.81% |
| LAL | 0.25 | Tail | 0.1152 | 0.1481 | 2.94% | 7.69% | 3.97% | 0.79% |
| LAL | 0.50 | Head | -0.4863 | -0.6153 | 6.34% | 11.03% | 4.73% | 1.15% |
| LAL | 0.50 | Medium | -0.5634 | -0.6007 | 5.03% | 17.65% | 6.71% | 1.42% |
| LAL | 0.50 | Tail | 0.0642 | 0.1193 | 5.88% | 15.38% | 7.94% | 0.79% |
| LAL | 1.00 | Head | -1.4229 | -1.9130 | 7.06% | 24.25% | 5.27% | 4.03% |
| LAL | 1.00 | Medium | -1.7090 | -1.8837 | 7.01% | 35.56% | 9.35% | 4.67% |
| LAL | 1.00 | Tail | -0.4557 | -0.3651 | 7.06% | 23.08% | 9.52% | 1.59% |
| BalancedSoftmax | 0.10 | Head | -0.0554 | -0.0631 | 1.54% | 1.72% | 1.12% | 0.08% |
| BalancedSoftmax | 0.10 | Medium | -0.0282 | -0.0072 | 0.46% | 4.28% | 0.56% | 0.00% |
| BalancedSoftmax | 0.10 | Tail | 0.0342 | 0.0403 | 0.59% | 0.00% | 0.96% | 0.00% |
| BalancedSoftmax | 0.25 | Head | -0.1727 | -0.2207 | 4.13% | 5.34% | 3.08% | 0.21% |
| BalancedSoftmax | 0.25 | Medium | -0.1156 | -0.0783 | 2.59% | 7.75% | 3.17% | 0.19% |
| BalancedSoftmax | 0.25 | Tail | 0.0320 | 0.0615 | 2.35% | 0.00% | 3.85% | 0.00% |
| BalancedSoftmax | 0.50 | Head | -0.4584 | -0.6302 | 6.67% | 11.15% | 5.04% | 0.62% |
| BalancedSoftmax | 0.50 | Medium | -0.3687 | -0.3312 | 4.88% | 13.90% | 5.97% | 0.56% |
| BalancedSoftmax | 0.50 | Tail | -0.1104 | -0.0551 | 5.29% | 15.38% | 8.65% | 0.96% |
| BalancedSoftmax | 1.00 | Head | -1.3568 | -1.9088 | 7.44% | 24.22% | 5.62% | 3.25% |
| BalancedSoftmax | 1.00 | Medium | -1.2154 | -1.2548 | 7.93% | 29.95% | 9.70% | 4.29% |
| BalancedSoftmax | 1.00 | Tail | -0.8200 | -0.8285 | 6.47% | 30.77% | 10.58% | 2.88% |
| Mixup | 0.10 | Head | 0.1023 | -0.0029 | 2.92% | 0.20% | 2.12% | 0.04% |
| Mixup | 0.10 | Medium | 0.1280 | 0.1087 | 1.22% | 2.14% | 1.29% | 0.00% |
| Mixup | 0.10 | Tail | 0.0618 | 0.0787 | 0.00% | 0.00% | 0.00% | 0.00% |
| Mixup | 0.25 | Head | 0.2368 | -0.0391 | 7.06% | 0.92% | 5.19% | 0.04% |
| Mixup | 0.25 | Medium | 0.2921 | 0.2433 | 3.20% | 5.61% | 3.23% | 0.16% |
| Mixup | 0.25 | Tail | 0.1218 | 0.1672 | 0.00% | 7.69% | 0.00% | 0.00% |
| Mixup | 0.50 | Head | 0.3989 | -0.1950 | 15.60% | 2.50% | 11.52% | 0.16% |
| Mixup | 0.50 | Medium | 0.4729 | 0.3822 | 5.34% | 11.23% | 5.48% | 0.32% |
| Mixup | 0.50 | Tail | 0.1150 | 0.2148 | 1.18% | 46.15% | 1.96% | 0.00% |
| Mixup | 1.00 | Head | 0.2578 | -1.1494 | 26.85% | 12.70% | 19.89% | 2.45% |
| Mixup | 1.00 | Medium | 0.2151 | 0.1030 | 9.15% | 39.30% | 9.68% | 1.94% |
| Mixup | 1.00 | Tail | -0.5432 | -0.1379 | 0.59% | 84.62% | 0.98% | 0.00% |

## Investigation C — classification-margin contribution

The margin target uses the uniform ensemble's strongest incorrect competitor. Final perturbed margins recompute the strongest incorrect class, so a positive local contribution is not a guarantee of correction.

| Group | Target top / Margin top agreement | Target Mixup highest | Margin Mixup highest | Both Mixup highest |
|:--|--:|--:|--:|--:|
| Head | 83.38% | 23.59% | 24.44% | 21.16% |
| Medium | 82.43% | 32.04% | 32.04% | 28.16% |
| Tail | 83.61% | 25.14% | 26.78% | 22.40% |

## Investigation D — rebalanced-expert opportunities and losses

These groups are label-dependent retrospective masks. They were not used as inference-time features or router-training targets.

| Event | Group | n | Correcting target positive | Target > Mixup | Ridge pairwise order correct | Ridge Mixup highest weight |
|:--|:--|--:|--:|--:|--:|--:|
| LAL correcting | All | 235 | 98.30% | 86.81% | 13.62% | 99.15% |
| LAL correcting | Head | 164 | 97.56% | 83.54% | 17.07% | 98.78% |
| LAL correcting | Medium | 56 | 100.00% | 92.86% | 7.14% | 100.00% |
| LAL correcting | Tail | 15 | 100.00% | 100.00% | 0.00% | 100.00% |
| BalancedSoftmax correcting | All | 233 | 99.57% | 86.27% | 13.30% | 99.57% |
| BalancedSoftmax correcting | Head | 164 | 99.39% | 83.54% | 15.85% | 99.39% |
| BalancedSoftmax correcting | Medium | 56 | 100.00% | 91.07% | 8.93% | 100.00% |
| BalancedSoftmax correcting | Tail | 13 | 100.00% | 100.00% | 0.00% | 100.00% |
| LAL damaging | All | 980 | 12.55% | 18.78% | 79.49% | 97.45% |
| LAL damaging | Head | 844 | 11.61% | 16.23% | 81.87% | 97.16% |
| LAL damaging | Medium | 133 | 17.29% | 33.08% | 66.17% | 99.25% |
| LAL damaging | Tail | 3 | 66.67% | 100.00% | 0.00% | 100.00% |
| BalancedSoftmax damaging | All | 959 | 10.84% | 15.64% | 81.96% | 97.18% |
| BalancedSoftmax damaging | Head | 843 | 9.25% | 12.93% | 84.34% | 96.92% |
| BalancedSoftmax damaging | Medium | 112 | 20.54% | 33.04% | 66.96% | 99.11% |
| BalancedSoftmax damaging | Tail | 4 | 75.00% | 100.00% | 0.00% | 100.00% |

## Interpretation limits

A target is a local true-class log-probability derivative, whereas classification outcomes depend on the full competing-class geometry. These measurements can show target preference, prediction error, and classification mismatch on this development population, but they do not establish that a different target or router would improve an independent evaluation.
