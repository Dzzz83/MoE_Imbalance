# Task 3E-B — Adaptive Soft-Mixture Oracle Feasibility Study

This is a label-dependent, exploratory feasibility diagnostic. It solves an independent maximum-margin LP for each permitted OOF image; it is not an inference-time router and is not an independently validated generalization result.

## Provenance and restrictions

- Source commit: `0fdb46c7a849b23c14ee472b4536e36120d38b5c`
- Dataset: CIFAR-100-LT, IR=100; canonical population=10847
- Analyzed population: 6507 samples, outer fold 0, inner folds [1, 2, 3]
- Expert order: `CE, LAL, BalancedSoftmax, Mixup`; logits calculated in `float64` from original `float32` values
- Inner fold 0, the reserved outer-evaluation population, and the CIFAR-100 test set were excluded from new calculations.

## Numerical verification

- SciPy `1.18.1`, `linprog(method='highs')`; margin tolerance `1e-06`.
- Status counts: `{'correctable': 5063, 'not_strictly_correctable': 1444, 'numerically_ambiguous': 0, 'solver_failure': 0}`.
- Strict correctability lower bound: **77.8085%** (5063/6507). Potential upper bound if all unresolved cases were feasible: **77.8085%**.
- Uniform-positive-margin invariant violations: `[]`; positive-individual-expert violations: `[]`.

## Reference results

- Uniform restricted-data metrics: ordinary=59.4283%, BA=35.9328%, Head=61.9661%, Medium=34.6909%, Tail=7.0094%
- Hard-selection oracle restricted-data metrics: ordinary=72.8754%, BA=49.5011%, Head=75.4469%, Medium=48.6691%, Tail=20.2015%
- Uniform reproduction verified: **True**; hard-selection reproduction verified: **True**.

| Reference | Weights | BA | Tail | Fixed correct & soft-feasible | Fixed incorrect & soft-feasible |
|---|---|---:|---:|---:|---:|
| Uniform | (0.25, 0.25, 0.25, 0.25) | 35.9328% | 7.0094% | n/a | n/a |
| fixed_020 | (0.25, 0.25, 0.25, 0.25) | 35.9328% | 7.0094% | 3867 | 1196 |
| fixed_006 | (0.00, 0.25, 0.25, 0.50) | 37.2134% | 9.9115% | 3915 | 1148 |
| fixed_007 | (0.00, 0.25, 0.50, 0.25) | 36.4403% | 14.7290% | 3633 | 1430 |
| fixed_010 | (0.00, 0.50, 0.25, 0.25) | 36.4236% | 14.5055% | 3640 | 1423 |
| fixed_011 | (0.00, 0.50, 0.50, 0.00) | 35.4372% | 16.7660% | 3406 | 1657 |
| Historical three-expert reference (not E-A grid) | (0, 1/3, 1/3, 1/3) | 37.1900% approx. | 13.8500% approx. | n/a | n/a |

## Soft-oracle feasibility and headroom

- Soft-oracle strict correctability: **77.8085%**; macro BA feasibility **55.5569%**, Head **80.2013%**, Medium **55.9347%**, Tail **26.3644%**.
- All experts wrong but soft-correctable: **321** (4.9331%); its macro class coverage is **6.0559%** and Tail event coverage is **6.1628%**.
- Uniform wrong but soft-correctable: 1196 (18.3802%).
- These are existence measurements using true labels. They do not estimate the accuracy of Ridge, Sinkhorn, or another learned router.

## Tail-class feasibility

| Class | N | Uniform | Any expert | Soft | Newly soft | Negative optimum | Ambiguous | Solver failure |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 12 | 1 | 2 | 2 | 0 | 10 | 0 | 0 |
| 71 | 11 | 1 | 4 | 5 | 1 | 6 | 0 | 0 |
| 72 | 10 | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| 73 | 9 | 1 | 1 | 2 | 1 | 7 | 0 | 0 |
| 74 | 9 | 0 | 1 | 1 | 0 | 8 | 0 | 0 |
| 75 | 9 | 1 | 3 | 5 | 2 | 4 | 0 | 0 |
| 76 | 9 | 1 | 5 | 7 | 2 | 2 | 0 | 0 |
| 77 | 8 | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| 78 | 7 | 0 | 3 | 3 | 0 | 4 | 0 | 0 |
| 79 | 8 | 2 | 2 | 2 | 0 | 6 | 0 | 0 |
| 80 | 7 | 0 | 1 | 2 | 1 | 5 | 0 | 0 |
| 81 | 6 | 0 | 0 | 1 | 1 | 5 | 0 | 0 |
| 82 | 7 | 3 | 5 | 6 | 1 | 1 | 0 | 0 |
| 83 | 6 | 0 | 1 | 1 | 0 | 5 | 0 | 0 |
| 84 | 6 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| 85 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| 86 | 6 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| 87 | 5 | 0 | 1 | 1 | 0 | 4 | 0 | 0 |
| 88 | 4 | 0 | 1 | 2 | 1 | 2 | 0 | 0 |
| 89 | 5 | 0 | 3 | 3 | 0 | 2 | 0 | 0 |
| 90 | 4 | 0 | 0 | 1 | 1 | 3 | 0 | 0 |
| 91 | 4 | 1 | 0 | 1 | 1 | 3 | 0 | 0 |
| 92 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| 93 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| 94 | 3 | 2 | 3 | 3 | 0 | 0 | 0 | 0 |
| 95 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| 96 | 3 | 0 | 2 | 2 | 0 | 1 | 0 | 0 |
| 97 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| 98 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| 99 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |

Tail counts are development-population counts; classes with very few permitted images are not reliable population-level estimates. The soft Tail macro coverage should therefore be interpreted alongside the per-class denominators above.

## Limitations and implications

- The LP sees the true class and returns label-dependent oracle weights. This establishes convex-mixture feasibility only; it does not establish that an inference-time model can predict those weights.
- The strict result counts only verified margins above the frozen `1e-6` threshold. Ambiguous and solver-failure cases remain unresolved rather than being treated as correctable or uncorrectable; the potential upper bound is intentionally non-definitive.
- No post-hoc tolerance sensitivity analysis was used to replace the frozen result; the reported statuses use the predeclared `1e-6` margin band.
- Earlier Task 3C diagnostics did inspect inner fold 0 descriptively, so that partition cannot be described as untouched for all prior research decisions even though E-B excludes it completely.
- The result cannot establish generalization to other folds, seeds, the reserved outer evaluation population, or the balanced CIFAR-100 test set.
- Task 3E-B does not implement or authorize Ridge, Sinkhorn, adaptive routing, new experts, or a final expert-weight selection. Any next experiment must use a separate, pre-registered protocol.
