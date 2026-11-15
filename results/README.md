# Saved evaluation results

These are copies of completed experiments from the source branches, imported
into `integration/ot-methods`. They are not new evaluations of the merged code.
Only evaluation reports and experiment configurations are included.

## Reweighting

Source: [prototype-weighting-results](https://github.com/Dzzz83/MoE_Imbalance/tree/f39296984fc5b03ca844ac5578a387238efdc5a7),
commit `f39296984fc5b03ca844ac5578a387238efdc5a7`.

The runs cover baseline, uniform, class-balanced, cluster-balanced, prototype
Sinkhorn, and SAVA-inspired weighting for training seeds 0, 1, and 2.

| Seed | Original four-candidate comparison | SAVA comparison |
|---|---|---|
| 0 | [comparison.json](reweighting/comparison.json) | [comparison_sava_seed0.json](reweighting/comparison_sava_seed0.json) |
| 1 | [comparison_seed1.json](reweighting/comparison_seed1.json) | [comparison_sava_seed1.json](reweighting/comparison_sava_seed1.json) |
| 2 | [comparison_seed2.json](reweighting/comparison_seed2.json) | [comparison_sava_seed2.json](reweighting/comparison_sava_seed2.json) |

Individual metrics and resolved training configurations are under
`reweighting/runs/<method>_seed<N>/`. The `experiment*.json` files record the
source implementation and experiment settings. Here `sinkhorn` denotes prototype
weighting; the source results do not contain a separate `ot_finetuning` run.

## Synthesis

Source: [hieu](https://github.com/Dzzz83/MoE_Imbalance/tree/30ac14ca302ab0e6698a88fe97ee371bef04cbb1),
commit `30ac14ca302ab0e6698a88fe97ee371bef04cbb1`.

Each comparison includes baseline, real-only classifier continuation, random
duplication, uniform OT, loss-shifted OT, and full guided OT synthesis.

| Seed | Comparison | Per-class recall |
|---|---|---|
| 0 | [comparison.csv](synthesis/comparison_5000_seed0/comparison.csv) | [per_class_recall.csv](synthesis/comparison_5000_seed0/per_class_recall.csv) |
| 1 | [comparison.csv](synthesis/comparison_5000_seed1/comparison.csv) | [per_class_recall.csv](synthesis/comparison_5000_seed1/per_class_recall.csv) |
| 2 | [comparison.csv](synthesis/comparison_5000_seed2/comparison.csv) | [per_class_recall.csv](synthesis/comparison_5000_seed2/per_class_recall.csv) |

The `_5000` directories use a generation iteration limit of 5,000, with a target
of 500 examples per class. Each directory also contains the original comparison
JSON, individual `<method>_metrics.json` files, and `experiment_config.json`.
Baseline recipes are under `synthesis/baseline_seed<N>/config.json`.

The earlier [seed-0 comparison](synthesis/comparison_seed0/comparison.csv) is
preserved separately. It is an additional historical result, not another seed
to pool with the three-seed comparison above.

## Interpretation and provenance

Reweighting used a warm-up/multistep baseline; synthesis used the historical
cosine baseline. Their classifier schedules also differ. These results are not
a comparison from matched baseline checkpoints across the two methods. See
[the integration guide](../docs/ot_methods.md#historical-recipes-and-comparisons).

Metric JSON values are fractions. Synthesis CSV scores are percentages and
differences are percentage points. Source checkpoint paths inside reports refer
to the original experiment locations; checkpoints are not included here.

[manifest.json](manifest.json) records the source commit, original path, and Git
blob hash for every imported file. All 91 imported files were verified against
their source blob hashes and retain their original contents and line endings.
The `ot-methods-results` branch had no saved results available at import time.
