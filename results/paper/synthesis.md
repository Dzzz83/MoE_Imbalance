# Synthesis results

Mean ± sample SD across training seeds 0, 1, 2 (n = 3). Higher is better.

| Method | Accuracy (%) ↑ | Head (%) ↑ | Medium (%) ↑ | Tail (%) ↑ | Delta accuracy (pp) ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline | 38.66 ± 1.04 | **65.70 ± 1.45** | 36.64 ± 1.55 | 9.46 ± 0.12 | — |
| Real-only continuation | 38.67 ± 0.92 | 65.69 ± 1.30 | 36.47 ± 1.55 | 9.73 ± 0.21 | +0.02 ± 0.13 |
| Random duplication | **41.02 ± 0.85** | 64.26 ± 0.99 | **39.66 ± 1.24** | 15.49 ± 0.28 | **+2.36 ± 0.27** |
| Uniform OT | 40.66 ± 1.15 | 64.34 ± 1.24 | 38.41 ± 1.59 | **15.64 ± 0.56** | +2.00 ± 0.11 |
| Loss-shifted OT | 40.65 ± 1.13 | 64.35 ± 1.22 | 38.37 ± 1.58 | **15.64 ± 0.56** | +1.99 ± 0.10 |
| Full guided OT | 40.54 ± 1.06 | 64.50 ± 1.11 | 38.06 ± 1.67 | 15.47 ± 0.35 | +1.88 ± 0.05 |

Head: retained class count ≥100; Medium: 20–99; Tail: <20.

Delta accuracy is calculated per seed before computing its mean and SD. Bold marks the highest unrounded mean in each column (including ties); it does not indicate statistical significance.

Baseline recipe: cosine. Compare methods within this table; the other experiment family used a different baseline and classifier recipe.
