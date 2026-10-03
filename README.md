# Expert Routing on Long-Tailed CIFAR-100

This project studies whether a router can combine four CIFAR-100-LT ResNet-32
experts (CE, LAL, BalancedSoftmax, and Mixup) to improve rare-class
recognition. The completed `ridge_sinkhorn_3seed_v1` study compares uniform and
fixed mixtures with adaptive Ridge and Sinkhorn routers using nested
out-of-fold predictions.

## Results

No adaptive method improved on uniform logits in both balanced accuracy (BA)
and Tail accuracy for every seed. All five paired BA 95% intervals include
zero.

The study covers 3 seeds and 5 outer folds, with 10,847 pooled rows per seed.
Values are percentages, shown as mean ± population SD (`ddof=0`) across the
three pooled seed-level results. BA is mean recall over 100 classes. Head,
Medium, and Tail are macro recall for classes with at least 100, 20–99, and
fewer than 20 training examples, respectively (35 / 35 / 30 classes).

| Method | BA | Head | Medium | Tail |
|:--|--:|--:|--:|--:|
| Uniform logits | 42.410&nbsp;±&nbsp;0.315 | 67.663&nbsp;±&nbsp;0.225 | 40.584&nbsp;±&nbsp;0.687 | 15.077&nbsp;±&nbsp;0.634 |
| Uniform probabilities | 41.556&nbsp;±&nbsp;0.345 | 66.967&nbsp;±&nbsp;0.330 | 39.481&nbsp;±&nbsp;0.325 | 14.329&nbsp;±&nbsp;0.871 |
| Fixed 006 | 43.062&nbsp;±&nbsp;0.405 | 68.116&nbsp;±&nbsp;0.232 | 41.783&nbsp;±&nbsp;0.466 | 15.323&nbsp;±&nbsp;0.785 |
| Fixed 007 | 42.251&nbsp;±&nbsp;0.955 | 64.047&nbsp;±&nbsp;0.714 | 40.357&nbsp;±&nbsp;1.133 | 19.033&nbsp;±&nbsp;1.233 |
| Fixed 010 | 42.566&nbsp;±&nbsp;0.663 | 64.256&nbsp;±&nbsp;0.315 | 40.606&nbsp;±&nbsp;0.963 | 19.548&nbsp;±&nbsp;1.671 |
| Fixed 011 | 41.802&nbsp;±&nbsp;0.839 | 60.404&nbsp;±&nbsp;0.645 | 39.184&nbsp;±&nbsp;0.679 | 23.153&nbsp;±&nbsp;1.845 |
| Contribution Ridge | 42.245&nbsp;±&nbsp;0.131 | 68.766&nbsp;±&nbsp;0.074 | 40.372&nbsp;±&nbsp;0.235 | 13.489&nbsp;±&nbsp;0.245 |
| Contribution Ridge + Sinkhorn | 42.719&nbsp;±&nbsp;0.366 | 67.634&nbsp;±&nbsp;0.866 | 41.313&nbsp;±&nbsp;0.703 | 15.293&nbsp;±&nbsp;0.324 |
| Residual Ridge | 42.509&nbsp;±&nbsp;0.466 | 66.522&nbsp;±&nbsp;0.437 | 40.964&nbsp;±&nbsp;0.513 | 16.296&nbsp;±&nbsp;0.633 |
| Residual Ridge + Sinkhorn | 42.861&nbsp;±&nbsp;0.706 | 66.824&nbsp;±&nbsp;0.729 | 41.600&nbsp;±&nbsp;0.920 | 16.377&nbsp;±&nbsp;0.711 |
| Selective residual Ridge + Sinkhorn | 42.832&nbsp;±&nbsp;0.673 | 66.777&nbsp;±&nbsp;0.674 | 41.489&nbsp;±&nbsp;0.869 | 16.463&nbsp;±&nbsp;0.830 |

<details>
<summary>Paired changes from uniform logits</summary>

Changes are percentage points, shown as mean [95% interval]. The intervals
are paired class-then-sample bootstrap intervals; estimates use full-precision
results before rounding.

| Adaptive method | ΔBA (pp) | ΔTail (pp) | All-seed gate |
|:--|:--|:--|:--:|
| Contribution Ridge | −0.165&nbsp;[−0.852,&nbsp;+0.357] | −1.589&nbsp;[−3.674,&nbsp;−0.253] | Fail |
| Contribution Ridge + Sinkhorn | +0.310&nbsp;[−0.354,&nbsp;+0.935] | +0.216&nbsp;[−1.669,&nbsp;+1.883] | Fail |
| Residual Ridge | +0.099&nbsp;[−0.681,&nbsp;+0.858] | +1.219&nbsp;[−0.959,&nbsp;+3.270] | Fail |
| Residual Ridge + Sinkhorn | +0.452&nbsp;[−0.276,&nbsp;+1.162] | +1.300&nbsp;[−0.686,&nbsp;+3.156] | Fail |
| Selective residual Ridge + Sinkhorn | +0.422&nbsp;[−0.308,&nbsp;+1.141] | +1.385&nbsp;[−0.627,&nbsp;+3.250] | Fail |

</details>

### Evidence limits

- These retrospective nested-OOF results use a shared population, and prior
  development inspected inner-fold-0 labels and seed-78 / outer-fold-0 results.
- Bootstrap intervals are conditional on fitted models and omit retraining
  variability.
- Test examples and labels were not loaded or scored; the test file was read
  only for checksum verification.

See the [full report and audit](docs/rs3-final-report.md), [protocol](docs/protocol.md),
and [diagnostic glossary](docs/protocol.md#diagnostic-terms) for details.

## Quick start

These commands validate the saved study configuration and local inputs without
starting training or evaluation:

```bash
python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile configs/profiles/local-smoke.yaml study validate

python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile configs/profiles/local-smoke.yaml study doctor
```

The matrix is complete. See [reproduction](docs/reproduction.md) for saved
commands, evaluator identity, and output locations; do not launch training to
reproduce the report. Source results are in the
[canonical results JSON](.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/results.json).

## Project docs

| Need | Guide |
|:--|:--|
| Frozen study rules and settings | [Study plan](docs/PLAN.md) |
| New study sessions and recovery | [Experiment workflow](docs/experiment-workflow.md) |
| Earlier OOF development and locked fold | [OOF results](docs/oof-results.md) |
| Historical balanced-test benchmark | [Full-data results](docs/results.md) |
| Inner diagnostics and pinned inputs | [Diagnostics](docs/diagnostics.md) |
| Remaining research questions | [Research notes](docs/research.md) |
| Complete document map and local evidence paths | [Documentation index](docs/README.md) |

## References

- Wolpert (1992), [Stacked generalization, *Neural Networks*](https://www.sciencedirect.com/science/article/pii/S0893608005800231).
- Hoerl and Kennard (1970), [Ridge Regression, *Technometrics*](https://doi.org/10.1080/00401706.1970.10488634).
- Cuturi (2013), [Sinkhorn Distances, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).
- Wang et al. (2021), [RIDE, ICLR](https://openreview.net/pdf?id=D9I3drBz4UC).
- Menon et al. (2021), [Long-tail learning via logit adjustment, ICLR](https://openreview.net/pdf?id=37nvvqkCo5).
- Ren et al. (2020), [Balanced Meta-Softmax, NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html).
- Zhang et al. (2018), [mixup, ICLR](https://openreview.net/pdf?id=r1Ddp1-Rb).
