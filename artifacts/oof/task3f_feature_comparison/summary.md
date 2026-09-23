# Task 3F-F — Ridge Feature Comparison on Tail Images

Exploratory, read-only comparison of the existing Task 3F-A confidence-only and full 13-feature Ridge results.

## Scope and safeguards

- Population: **6507** OOF rows from outer fold 0, inner folds 1–3; Head/Medium/Tail counts are **Head=5294, Medium=1030, Tail=183**.
- Saved held-out scores, predictions, weights, and model diagnostics were loaded; Ridge was not refit and experts were not retrained.
- Inner fold 0, the reserved outer-evaluation population, the original CIFAR-100 test set, and oracle weights were not used.
- Labels and canonical groups appear only in retrospective target and outcome diagnostics.

## Answers to the research questions

1. **Contribution prediction:** across the 15 matched alpha/gamma settings, the full-feature model had lower pooled mean-expert MSE in **15/15** settings and lower Tail MSE in **15/15**. The mean full-minus-confidence pooled MSE was **-0.044230**; Tail was **-0.141768**.
2. **Tail ranking:** on the primary configuration's **183** Tail rows, actual LAL > Mixup occurred **103** times; confidence-only/full ranked LAL higher **1/4** times. For BalancedSoftmax, the corresponding counts were **94** actual opportunities and **0/4** correct rankings. Across both ranking directions, LAL pairwise accuracy was **44.26%** versus **45.90%**, and BalancedSoftmax was **48.63%** versus **50.82%**.
3. **Classification:** primary Balanced Accuracy changed from **36.55%** to **36.10%** (-0.4459 pp); Tail changed from **7.38%** to **6.55%** (-0.8333 pp). Full-feature gains/losses on Tail were **0 / 1**.
4. **Mixup preference:** primary Tail mean Mixup weight changed from **0.4049** to **0.4065**; its highest-weight fraction changed from **99.45%** to **96.17%**.
5. **Training-data interpretation:** the saved/reconstructed training-versus-held-out diagnostics show full-feature lower training MSE in **45/45** matched model fits and lower held-out MSE in **45/45**. This is compatible with feature benefit and possible overfitting, but the sparse Tail population does not identify insufficient training data as the cause.
   At the predefined primary setting, the fold-level Tail-accuracy differences were **fold 1: +0.0000 pp, fold 2: +0.0000 pp, fold 3: -3.3333 pp**, showing that the pooled Tail decrease was concentrated in fold 3 rather than uniform across all three folds.

These are verified measurements on one development population. Lower contribution-target error is not evidence of better classification, and no new router or final configuration is selected here.

## Primary matched classification comparison

| Method | Accuracy | BA | Head | Medium | Tail | Correct Tail images |
|---|---:|---:|---:|---:|---:|---:|
| uniform_logit | 59.43% | 35.93% | 61.97% | 34.69% | 7.01% | 13 |
| global_control | 60.63% | 36.33% | 63.18% | 34.61% | 7.01% | 13 |
| uniform_probability | 58.54% | 35.78% | 60.96% | 33.67% | 8.85% | 15 |
| fixed_006 | 60.17% | 37.21% | 62.89% | 34.94% | 9.91% | 19 |
| fixed_007 | 55.83% | 36.44% | 58.17% | 33.32% | 14.73% | 25 |
| fixed_010 | 55.94% | 36.42% | 58.41% | 33.23% | 14.51% | 25 |
| fixed_011 | 52.34% | 35.44% | 54.42% | 32.46% | 16.77% | 28 |
| uniform_without_ce | 57.66% | 37.19% | 60.18% | 34.20% | 13.85% | 24 |
| confidence_only | 60.73% | 36.55% | 63.27% | 34.84% | 7.38% | 14 |
| full_13 | 60.73% | 36.10% | 63.28% | 34.27% | 6.55% | 13 |

## Grid-level pattern

The complete matched grid contains **300** alpha/gamma/temperature/shrinkage pairs. Full features improved BA in **114**, Tail in **22**, and produced identical top-1 predictions in **60** pairs.
The mean full-minus-confidence Tail accuracy difference across the grid was **-0.1936 pp**; the mean Mixup-weight difference was **-0.005441**.

## Limitations

The Tail group has only 183 rows, with shared OOF training dependence across folds and one expert seed/outer fold. The target is label-dependent and retrospective; the inference-time feature sets themselves do not use labels. These results cannot distinguish feature insufficiency from insufficient Tail training data, and they do not establish independent or full-data-expert performance.
