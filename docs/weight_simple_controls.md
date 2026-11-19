# Run Stage B on Kaggle

This guide reproduces completed Stage B: **15 classifier continuations**,
five controls across seeds
0/1/2, using the exact fitting/development features and frozen baselines from
Stage A. The controls are cluster balancing, prototype distance, easy-margin
rank, and two reassignments of the stronger OT weights. See
[PLAN.md](PLAN.md)
for the hypotheses, fixed rules, comparisons, and decision limits.

All conditions use split 43, 40 epochs at learning rate 0.002, the same frozen
backbone/BN, and class mass 1/100. Original fallback classes stay uniform.
Replay the original two-cluster geometry and require identical active masks and
cluster sizes, plus original weights within `rtol=1e-5, atol=1e-8`.
Let `s[i]` be assigned cluster size and `d[i]` the minimum cosine distance
to a normalized center.

| Condition | Fixed rule within each active class |
|---|---|
| `cluster_balanced` | `1 / (200 * s[i])` |
| `prototype_distance` | Normalize `exp(-d[i] / 0.1)` to class mass 1/100 |
| `easy_rank` | Normalize `exp(log(4) * r[i])`, where `r` is the saved margin's midrank scaled to [0,1]; all ties give uniform weights |
| `cluster_rank_strong` | Assign the parent's exact stronger weight multiset using priority `1 / s[i]` |
| `distance_rank_strong` | Assign that multiset using priority `-d[i]` |

Larger priority receives larger weight; rank controls reuse the parent's saved
tie order. Native normalization uses float64 before casting to the parent dtype.
No temperature, factor, learning-rate or coverage search is added.

## Notebook setup

1. Create a fresh Kaggle notebook with **Tesla T4**, Internet enabled, and the
   existing `GITHUB_TOKEN` secret enabled. The token needs repository write and
   release upload access to `Dzzz83/MoE_Imbalance`.
2. Paste the current
   [Kaggle script](../scripts/kaggle_ablation_study.py) from `ablation_study`
   into a cell, explicitly set `STAGES = ("simple_controls",)`, and run it.
   The script now defaults to the separate controlled-bias diagnostic.
   Use these Stage B settings:

   ```python
   STAGES = ("simple_controls",)
   SIMPLE_CONTROLS_PREFLIGHT_ONLY = False
   SOURCE_COMMIT = None
   RESTORE_ROOT = None
   ```

The first execution pins the published source commit; later sessions reuse
that pin. No CIFAR attachment is needed. The script restores the original
audit inputs and the complete Stage A references automatically from their
pinned GitHub commits and checksum-linked release archives.

Keep Python 3.12, PyTorch `2.10.0+cu128`, NumPy `2.0.2`, CUDA 12.8, and
Tesla T4 to match Stage A. The script sets two CPU threads and the recorded
backend flags, then checks compatibility before training. A mismatch stops
execution and lists the differing values. Restore the matching environment;
changing the environment requires a separately recorded reference-rerun budget.
Do not bypass this check to combine incompatible old and new results.

For an input-only check, set `SIMPLE_CONTROLS_PREFLIGHT_ONLY = True`.
This validates all parent reports and 27 completion receipts, saved scores and
tie permutations, baseline/cache identities, all three geometry replays, and
the 15 new weight vectors. It needs no GPU and does not train or publish reports.
The training environment is checked when the full run starts.

## Outputs and recovery

The new workspace is `/kaggle/working/ot-simple-controls/`. Stage A's workspace
and results branch remain separate. New reports go to
`ablation-simple-controls-results` after every completed continuation:

- `reports/study_status.json`: status and completed count out of 15.
- `reports/study_results.json`: 15 new results plus 30 reused parent records,
  explicitly labelled by origin.
- `reports/study_summary.json`: all prespecified paired comparisons, three
  seed differences, means and sample SD, active/fallback contributions, and
  contributions from classes with only one development image.
- `reports/parent.json` and `reports/inputs.json`: exact parent and input
  identities, original release links, and hashes.
- `reports/diagnostics/seedN/controls.json`: geometry and weight hashes,
  replay checks, and tied/constant priority counts.
- `reports/runs/seedN/CONDITION/`: resolved settings, starting metrics,
  per-epoch histories, final metrics, weight/assignment diagnostics and receipts.
- `reports/study_assets.json`: checksum-linked release archives containing
  the new geometry, weights, and checkpoints. Original inputs and parent
  binaries are referenced rather than republished.

Save the workspace as Kaggle outputs when possible. Rerunning reuses verified
completed continuations and retries failed publication. A fresh session restores
published completed runs automatically. Interrupted runs restart from the same
baseline; partial folders are preserved under `interruptions/`.

Keep the source pin, settings and environment unchanged when continuing.
Corrupted or incompatible reports, inputs, geometry or completed outputs stop
execution. Full Stage A continuations and baselines are never retrained in
this mode.

## Local commands and interpretation

With restored `inputs/` and a parent directory containing `manifest.json`
and `reports/`:

```bash
python -m pipeline.run_weight_study validate --config configs/weight_simple_controls.json --inputs inputs/ot_audit --parent inputs/assignment_parent
python -m pipeline.run_weight_study run --config configs/weight_simple_controls.json --inputs inputs/ot_audit --parent inputs/assignment_parent --output-dir outputs/weight_simple_controls --device cuda
python -m pytest tests/test_simple_controls.py tests/test_weight_study.py tests/test_kaggle_ablation_study.py tests/test_prototype_weights.py
```

The local runner performs no downloads or GitHub writes. Training requires a
matching parent environment. Artificial-data tests establish workflow and
numeric correctness; they do not establish a research improvement.

After training, review all three original-OT-minus-native comparisons and
each new condition versus class balancing. The two stronger-rank controls
isolate assignment at the parent's fixed weight spectrum; native rules change
both assignment and spread. Easy rank is exploratory because its priority was
chosen after Stage A. The 0.5 percentage-point target with positive differences
on at least two seeds is a research-budget rule, not a significance test.
No later experiment or official test evaluation runs automatically.

Stage B completed all 15 continuations for 40 epochs on Kaggle. Its reports and
complete assets are pinned to results commit
`451349aa463cd5b7ec37ea3a97fd99b8bf50a5d2`, produced with source
`b4cea1c8b68eb7bd872f1b5500e8e4d9038ecf69`. Read-only analysis verified
the archive hashes, frozen backbone states, weight invariants, final accuracy/
recall from saved heads on CPU, and the published paired contrasts.
See the [research handoff](HANDOFF.md) for findings and the
[controlled-bias protocol](PLAN.md#controlled-bias-diagnostic-the-next-experiment)
for the next planned experiment. That diagnostic is not implemented by this runner.
