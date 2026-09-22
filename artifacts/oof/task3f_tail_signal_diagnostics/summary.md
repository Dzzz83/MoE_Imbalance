# Task 3F-C — Tail-specific routing-signal diagnostics

This is a retrospective diagnostic on the frozen 6,507-image outer-fold-0 / inner-folds-1–3 population. Signals are formed from expert OOF predictions and confidences only; labels and true class groups are used afterward for evaluation. No router was refit.

## Findings

1. **Confidence.** On the 183 true Tail rows, mean/median raw confidence is CE 0.6836/0.7149, LAL 0.6187/0.6154, BalancedSoftmax 0.6504/0.6467, and Mixup 0.3871/0.3597. Mixup's incorrect-Tail mean/median confidence is 0.3882/0.3601; on rows where LAL or BalancedSoftmax is correct and Mixup is wrong, the rebalanced-minus-Mixup confidence margins average +0.2871 (LAL, n=20) and +0.2570 (BalancedSoftmax, n=18). This is a retrospective association suggesting limited signal, not a correctness guarantee.
2. **Disagreement.** LAL and BalancedSoftmax agree while Mixup disagrees on 666 rows overall, including 24 Tail rows; their shared prediction is correct on 211 of the overall cases (0.3168). Within this pattern, Tail contributes 24 rows and its shared prediction is correct on 0.2917; Tail all-four agreement is 0.0929; disagreement is more common on Tail than Head, but the shared-prediction pattern is only weakly informative at this sample size; the full Head/Medium/Tail pattern table is in the JSON artifact.
3. **Predicted class groups.** Tail-class prediction rates are LAL 0.1406, BalancedSoftmax 0.1332, and Mixup 0.0040. Their retrospective Tail recalls are 0.3333, 0.2951, and 0.0273; predicted-Tail precision is 0.0667, 0.0623, and 0.1923, with Tail miss rates 0.6667, 0.7049, and 0.9727. This evaluates a prediction-only signal with useful recall differences but low precision for the rebalanced experts, and does not make the true group an input feature.
4. **Ridge weights.** When LAL and BalancedSoftmax agree on a predicted Tail class (132 rows), their mean saved weights are 0.1929 and 0.2054, while Mixup receives 0.4016. In the subset where Mixup also disagrees (126 rows), its mean weight is 0.4009. When Mixup disagrees with both rebalanced experts (2552 rows), its mean weight is 0.4110 versus the overall 0.3702. For prediction-only group conditioning, Mixup weight is 0.3557 when LAL predicts Head and 0.4064 when LAL predicts Tail; the corresponding BalancedSoftmax values are 0.3558 and 0.4058. On true Tail rows where LAL / BalancedSoftmax are correct and Mixup is wrong (n=20 / 18), Ridge still assigns Mixup mean weights 0.4052 / 0.4019. These are conditioned associations, not a new routing rule.

## Confidence correctness splits on Tail rows

| Expert | Correct count | Incorrect count | Mean confidence when correct | Mean confidence when incorrect |
|:--|--:|--:|--:|--:|
| CE | 8 | 175 | 0.7223 | 0.6818 |
| LAL | 22 | 161 | 0.6915 | 0.6087 |
| BalancedSoftmax | 20 | 163 | 0.6611 | 0.6491 |
| Mixup | 3 | 180 | 0.3260 | 0.3882 |

## Limitations

Only 183 Tail rows are available, so subgroup percentages can be unstable, especially for multi-condition disagreement patterns. Raw softmax confidence scales may differ across the four training objectives and are not assumed calibrated or directly comparable. All class-group and correctness results are retrospective. These associations do not establish a routing improvement, justify selecting a new feature, or support independent validation.
