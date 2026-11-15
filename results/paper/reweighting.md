# Reweighting results

Mean ± sample SD across training seeds 0, 1, 2 (n = 3). Higher is better.

| Method | Accuracy (%) ↑ | Head (%) ↑ | Medium (%) ↑ | Tail (%) ↑ | Delta accuracy (pp) ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline | 38.64 ± 0.33 | **65.87 ± 0.47** | 37.01 ± 0.45 | 8.78 ± 0.26 | — |
| Uniform continuation | 38.68 ± 0.34 | **65.87 ± 0.49** | 37.10 ± 0.41 | 8.79 ± 0.28 | +0.04 ± 0.01 |
| Class-balanced | **38.99 ± 0.33** | 65.75 ± 0.34 | **37.65 ± 0.49** | **9.34 ± 0.32** | **+0.35 ± 0.03** |
| Cluster-balanced | 38.97 ± 0.33 | 65.72 ± 0.34 | 37.60 ± 0.48 | **9.34 ± 0.32** | +0.33 ± 0.05 |
| Prototype Sinkhorn | 38.98 ± 0.33 | 65.76 ± 0.33 | 37.61 ± 0.48 | **9.34 ± 0.32** | +0.34 ± 0.03 |
| SAVA + class balance | 38.93 ± 0.35 | 65.78 ± 0.34 | 37.50 ± 0.54 | 9.29 ± 0.32 | +0.29 ± 0.03 |

Head: retained class count ≥100; Medium: 20–99; Tail: <20.

Delta accuracy is calculated per seed before computing its mean and SD. Bold marks the highest unrounded mean in each column (including ties); it does not indicate statistical significance.

Baseline recipe: warm-up/multistep. Compare methods within this table; the other experiment family used a different baseline and classifier recipe.
