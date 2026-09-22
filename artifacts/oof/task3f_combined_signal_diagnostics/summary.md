# Task 3F-D — Combined Tail-signal diagnostics

This is a retrospective diagnostic on 6,507 held-out OOF images from outer fold 0 / inner folds 1–3. Signals A–D were constructed from expert predictions, raw maximum-softmax confidence, and canonical class frequencies only. True labels and true Head/Medium/Tail groups were used afterward for evaluation; no router was refit.

## Population

Head: 5,294; Medium: 1,030; Tail: 183; Tail prevalence: 183 / 6507 (2.81%).

## All predefined masks

Percentages retain their count in the same cell or adjacent count columns.

| ID | Required | Selected n | Head / Medium / Tail n | False positives n | Tail precision | Tail recall | Rebalanced opportunity n / fraction | Mixup mean weight | Mixup highest n / fraction |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| A | A | 3103 | 2686 / 371 / 46 | 3057 | 46 (1.48%) | 46 (25.14%) | 211 (6.80%) | 0.3409 | 3065 (98.78%) |
| B | B | 2552 | 1866 / 576 / 110 | 2442 | 110 (4.31%) | 110 (60.11%) | 554 (21.71%) | 0.4110 | 2543 (99.65%) |
| C | C | 291 | 200 / 62 / 29 | 262 | 29 (9.97%) | 29 (15.85%) | 13 (4.47%) | 0.4102 | 290 (99.66%) |
| D | D | 4690 | 3812 / 756 / 122 | 4568 | 122 (2.60%) | 122 (66.67%) | 583 (12.43%) | 0.3778 | 4688 (99.96%) |
| AB | A+B | 666 | 474 / 168 / 24 | 642 | 24 (3.60%) | 24 (13.11%) | 211 (31.68%) | 0.3991 | 664 (99.70%) |
| AC | A+C | 132 | 91 / 28 / 13 | 119 | 13 (9.85%) | 13 (7.10%) | 7 (5.30%) | 0.4016 | 131 (99.24%) |
| AD | A+D | 2496 | 2163 / 298 / 35 | 2461 | 35 (1.40%) | 35 (19.13%) | 191 (7.65%) | 0.3447 | 2494 (99.92%) |
| BC | B+C | 285 | 197 / 61 / 27 | 258 | 27 (9.47%) | 27 (14.75%) | 13 (4.56%) | 0.4101 | 284 (99.65%) |
| BD | B+D | 1858 | 1341 / 438 / 79 | 1779 | 79 (4.25%) | 79 (43.17%) | 463 (24.92%) | 0.4231 | 1858 (100.00%) |
| CD | C+D | 190 | 125 / 42 / 23 | 167 | 23 (12.11%) | 23 (12.57%) | 10 (5.26%) | 0.4291 | 190 (100.00%) |
| ABC | A+B+C | 126 | 88 / 27 / 11 | 115 | 11 (8.73%) | 11 (6.01%) | 7 (5.56%) | 0.4009 | 125 (99.21%) |
| ABD | A+B+D | 528 | 370 / 139 / 19 | 509 | 19 (3.60%) | 19 (10.38%) | 191 (36.17%) | 0.4101 | 528 (100.00%) |
| ACD | A+C+D | 87 | 58 / 19 / 10 | 77 | 10 (11.49%) | 10 (5.46%) | 5 (5.75%) | 0.4225 | 87 (100.00%) |
| BCD | B+C+D | 184 | 122 / 41 / 21 | 163 | 21 (11.41%) | 21 (11.48%) | 10 (5.43%) | 0.4294 | 184 (100.00%) |
| ABCD | A+B+C+D | 81 | 55 / 18 / 8 | 73 | 8 (9.88%) | 8 (4.37%) | 5 (6.17%) | 0.4228 | 81 (100.00%) |

## Findings

1. **Tail identification.** 3 of the 11 predefined conjunctions have higher Tail precision than every one of their non-empty constituent masks on this population. Conjunctions are subsets by construction, so any precision increase is accompanied by lost selected images and no increase in Tail recall. The table reports the counts needed to interpret each percentage; it does not identify a winning combination.
2. **Useful rebalanced predictions.** The retrospective opportunity is the union of cases where LAL or BalancedSoftmax is correct while Mixup is wrong. 3 of the 11 conjunctions have a higher opportunity fraction than every constituent on this development population. Opportunity counts and fractions are reported overall and by true group in diagnostic_results.json; these labels were not used to form any mask.
3. **Frozen Ridge response.** The overall saved Ridge mean Mixup weight is 0.3702 (n=6507); 2 of the 15 masks have a lower selected-subgroup Mixup mean. This is a conditioned association, not an appropriate-response test or a new routing rule. Raw confidence scales are not assumed calibrated across experts.
4. **Cost of conjunctions.** Requiring multiple conditions progressively shrinks the selected population. For example, ABCD is compared with A, B, C and D in the JSON comparison table, including its excluded-image counts, Tail precision/recall deltas and opportunity-rate deltas. Very small selected groups can produce unstable percentages, especially for the 183 actual Tail images.
5. **Feature-representation implication.** Non-equivalent masks indicate that the frozen signals contain distinct combinatorial information, but this retrospective result does not justify selecting a new feature representation or fitting a new router. Any such proposal requires a new frozen protocol and independent evaluation.

## Ridge reference

| Population | n | CE mean | LAL mean | BalancedSoftmax mean | Mixup mean | Mixup highest n / fraction |
|:--|--:|--:|--:|--:|--:|--:|
| Overall | 6507 | 0.2029 | 0.2099 | 0.2170 | 0.3702 | 6415 (98.59%) |

## Limitations

This is exploratory development-data evidence from one seed and one outer fold. The population has only 183 actual Tail images. Raw cross-expert confidence comparisons may be affected by calibration differences. Correctness outcomes and true-group composition are retrospective labels, and no combination was selected as a routing rule. The reserved inner fold 0, reserved outer-evaluation population, original CIFAR-100 test set, and full-data expert predictions were not used.
