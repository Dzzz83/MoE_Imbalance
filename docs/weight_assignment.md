# Run the OT assignment study on Kaggle

This guide reproduces completed Stage A of [the research plan](PLAN.md):
27 classifier continuations on the three published audit baselines. It compares
class-balanced, original OT, strong OT, three strong-weight shuffles, and
easy/boundary/hard assignments of the same stronger weights. Baselines are
restored from GitHub; no CIFAR attachment or baseline training is needed.

## Notebook setup

1. Create a fresh Kaggle notebook with a GPU accelerator and Internet enabled.
2. Enable the existing `GITHUB_TOKEN` secret with write access to
   `Dzzz83/MoE_Imbalance`, including release asset upload access.
3. Paste the published `scripts/kaggle_ablation_study.py` into one cell and run
   it after explicitly setting `STAGES = ("assignment",)`. The script now
   defaults to the [controlled-bias diagnostic](PLAN.md#implemented-commands-and-kaggle-delivery). Leave
   `ASSIGNMENT_CONFIG_OVERRIDES = {}` and `ASSIGNMENT_PREFLIGHT_ONLY = False`.

Use the new implementation from `ablation_study`; older copies of this script
default to rebuilding the audit. Kaggle checks out the published source commit
and records it. Uncommitted local changes are unavailable to the notebook.

To verify access and inputs before training, set
`ASSIGNMENT_PREFLIGHT_ONLY = True`. This validates all three checkpoints,
feature caches, pinned report hashes, and split metadata without training.
Set it back to `False` for the full run. This check can run without a GPU.

The script downloads only the published baseline/feature archive and source.
It never loads the official test set. Keep the default 40 epochs, learning rate
0.002, training seeds 0/1/2, and shuffle seeds 0/1/2 for the research comparison.

## Outputs and recovery

Results are published incrementally to `ablation-assignment-results`:

- `reports/study_status.json`: running, complete, or failed locally.
- `reports/study_results.json`: available baseline and continuation metrics.
- `reports/study_summary.json`: paired results and the prespecified boundary
  comparisons; incomplete runs cannot produce a complete decision.
- `reports/runs/seedN/.../weight_diagnostics.json`: weight spread, effective
  sample size, and which fitting examples receive emphasis.
- `reports/study_assets.json`: checksum-linked continuation archives in releases.
- `reports/inputs.json`: exact original audit inputs and provenance.

Keep `/kaggle/working/ot-weight-assignment/` in saved notebook outputs when
possible. Publication follows each completed continuation. If an upload/push
fails, the local completed run is retained; rerunning retries publication
without retraining it. A failure before successful publication may only be
visible in the notebook logs and saved local status until publication recovers.

A fresh session can restore published completed runs from the release archives
automatically. Interrupted continuations restart from their baseline; there is
no optimizer/epoch resume. By default their partial folders are preserved under
`interruptions/`. Completed continuations are reused only after input and output
hash verification. Corrupted or incompatible inputs stop execution.

`RESTORE_ROOT`, if set for this stage, selects an attached verified input folder
with `audit/`, `baselines/`, and `caches/`, or a workspace containing `inputs/`.
Published continuation restoration remains automatic. To use an attached full
workspace instead, copy its complete `study/` to the configured `WORK_ROOT`
before running. Keep settings and the source commit unchanged when resuming.

Changing the source, scientific settings, or training environment requires a
new work folder and results branch. The source pin on an existing results
branch prevents accidentally mixing implementations. Do not switch back to
audit rebuilding to bypass an input or compatibility failure.

## Local commands and verification

With a verified input directory and the project's environment active:

```bash
python -m pipeline.run_weight_study validate --inputs inputs/ot_audit
python -m pipeline.run_weight_study run --inputs inputs/ot_audit --output-dir outputs/weight_assignment --device cuda
python -m pytest tests/test_weight_study.py tests/test_ablation.py tests/test_kaggle_ablation_study.py tests/test_reweighting.py tests/test_finetuning.py tests/test_prototype_weights.py
```

The local runner performs no GitHub operations and does not fetch images.
Artificial-data tests check the workflow and numeric invariants; real Kaggle
and CUDA results are separate verification. After the study finishes, review
the paired boundary-versus-OT, boundary-versus-shuffled, and
boundary-versus-class-balanced differences before selecting Stage B or a
conditional follow-up. This runner does not automatically run later stages,
select a final method, or evaluate the official test set.

Stage A completed all 27 continuations for 40 epochs on Kaggle. Its reports and
assets are pinned to results commit
`0d4cf3e90be73513ddbe5c731698610b0285773c`, produced with source
`a792b81ef737e8fba54033ffb4cc72a92c6b2e2a`. Analysis verified completion
receipts, frozen backbones, cached-feature predictions, and stronger-weight
reassignment invariants. See the [research handoff](HANDOFF.md) for findings
and [PLAN.md](PLAN.md) for the next diagnostic, which is not implemented by
these commands.
