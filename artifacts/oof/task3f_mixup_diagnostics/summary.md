# Task 3F-B — Ridge Mixup-preference diagnostics

This is a retrospective, development-only diagnosis on the frozen 6,507 outer-fold-0 / inner-folds-1–3 rows. It does not establish an independent router result and does not use inner fold 0, the reserved outer evaluation population, or the CIFAR-100 test set.

## Executive findings

1. **Why Mixup is favored.** The saved confidence-only Ridge has Mixup intercepts 1.1289, 1.1009, 1.1053 across folds 1–3, exceeding the other-expert intercept mean by 1.5053, 1.4679, 1.4737. Its signed mean held-out feature terms are 0.0134, -0.0071, -0.0063 (mean absolute terms 0.4168, 0.4029, 0.4417), so features modulate individual images but largely cancel in the pooled mean. Its mean held-out predicted scores are 1.1424, 1.0938, 1.0990; the corresponding actual target means are 1.0841, 1.1240, 1.0980. Mixup is the actual-target leader in 3/3 folds and the predicted-score leader in 3/3. The full coefficients, scalers, and per-fold target comparisons are in `diagnostic_results.json`.
2. **Head versus Tail.** Head has 5294 rows and Tail has 183 rows. Mixup's mean weight is 0.3646 on Head versus 0.4049 on Tail; its highest-weight fractions are 98.38% and 99.45%, respectively. Tail mean actual Mixup target is 0.6980 versus predicted 1.4073.
3. **Tail changes.** Relative to uniform logit averaging, Ridge gains 1 Tail rows and loses 0; both-correct is 13 and both-wrong is 169. 1 Tail class have a non-zero net correct-count change. Head accuracy changes by +0.0130 and Medium accuracy by +0.0015.
4. **Mixup sensitivity.** The predefined factors are evaluated without refitting. Factors with Tail accuracy above factor 1.0 are [0.25, 0.0]; factor 1.0 has BA 0.3655 and Tail 0.0738; factor 0.0 changes these to BA 0.3484 and Tail 0.0835. This table is not used to select a new factor.
5. **Image dependence.** Compared with the corresponding global control, non-trivial image-dependent variation is present: 1.0000 of rows differ beyond 1.0e-06, with mean absolute weight difference 0.0310. Thus the result can contain image-dependent variation while still having a strong global Mixup preference; these observations do not establish causality.

## Metric comparison

| Method | BA | Head | Medium | Tail |
|:--|--:|--:|--:|--:|
| uniform_logit | 0.3593 | 0.6197 | 0.3469 | 0.0701 |
| ridge_highlighted | 0.3655 | 0.6327 | 0.3484 | 0.0738 |
| global_control | 0.3633 | 0.6318 | 0.3461 | 0.0701 |
| fixed_006 | 0.3721 | 0.6289 | 0.3494 | 0.0991 |
| fixed_007 | 0.3644 | 0.5817 | 0.3332 | 0.1473 |
| fixed_010 | 0.3642 | 0.5841 | 0.3323 | 0.1451 |
| fixed_011 | 0.3544 | 0.5442 | 0.3246 | 0.1677 |
| uniform_without_ce | 0.3719 | 0.6018 | 0.3420 | 0.1385 |
| expert_CE | 0.2850 | 0.5489 | 0.2370 | 0.0332 |
| expert_LAL | 0.3069 | 0.4923 | 0.2745 | 0.1285 |
| expert_BalancedSoftmax | 0.3114 | 0.4955 | 0.2978 | 0.1127 |
| expert_Mixup | 0.3107 | 0.6177 | 0.2587 | 0.0132 |

## Sensitivity table

| Mixup factor | BA | Head | Medium | Tail | Same predictions as factor 1 |
|--:|--:|--:|--:|--:|:--:|
| 1.00 | 0.3655 | 0.6327 | 0.3484 | 0.0738 | yes |
| 0.75 | 0.3622 | 0.6275 | 0.3442 | 0.0738 | — |
| 0.50 | 0.3598 | 0.6197 | 0.3456 | 0.0731 | — |
| 0.25 | 0.3547 | 0.6064 | 0.3412 | 0.0768 | — |
| 0.00 | 0.3484 | 0.5873 | 0.3365 | 0.0835 | — |

## Interpretation limits

The contribution targets and class-group reports use labels only after the saved Ridge weights and predictions were loaded. The sensitivity factors are fixed before evaluation and do not constitute a new selected router. A higher Tail score after reducing Mixup would be evidence of a development trade-off, not proof that Mixup caused the Ridge Tail weakness. All findings remain exploratory until a frozen independent evaluation is run.
