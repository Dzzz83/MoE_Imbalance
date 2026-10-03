# Reproducing saved studies

This guide points to existing inputs and safe reproduction commands. Canonical
data roles and metric definitions are in [protocol.md](protocol.md). New study
sessions belong in [experiment-workflow.md](experiment-workflow.md); the
read-only diagnostic procedure belongs in [diagnostics.md](diagnostics.md).

## Historical full-data benchmark

The full-data benchmark used the four recipes in
`configs/experts/`, seeds 78, 88, and 1034, and final-epoch checkpoints.
To retrain one expert/seed:

```bash
./.venv/bin/python scripts/train.py --config configs/experts/ce.yaml --seed 78
```

The historical test evaluation and aggregate analyses use:

```bash
./.venv/bin/python scripts/evaluate_experts.py --seeds 78 88 1034
./.venv/bin/python scripts/analyze_subsets.py --seeds 78 88 1034
./.venv/bin/python scripts/check_runs.py --seeds 78 88 1034
```

Evaluation reads the balanced test examples and appends to the
[access log](test-access-log.md). The test set has already informed historical
decisions; do not use these commands for method selection or router
development. Existing results and known invalid historical rows are described
in [results.md](results.md).

## Saved Task 3E–3F analyses

These CPU array analyses consume saved OOF artifacts. Run Task 3E-A before
3E-B, then Task 3F-A before diagnostics B–F:

```bash
./.venv/bin/python scripts/run_task3e_fixed.py
./.venv/bin/python scripts/run_task3e_soft.py
./.venv/bin/python scripts/run_task3f_ridge.py
./.venv/bin/python scripts/run_task3f_mixup_diagnostics.py
./.venv/bin/python scripts/run_task3f_tail_signal_diagnostics.py
./.venv/bin/python scripts/run_task3f_combined_signal_diagnostics.py
./.venv/bin/python scripts/run_task3f_target_diagnostics.py
./.venv/bin/python scripts/run_task3f_feature_comparison.py
```

| Analysis | Saved evidence |
|:--|:--|
| Task 3C aligned OOF input | `artifacts/oof/task3c_oof/` |
| Task 3E-A fixed mixtures | `artifacts/oof/task3e_fixed_feasibility/` |
| Task 3E-B soft feasibility | `artifacts/oof/task3e_soft_feasibility/` |
| Task 3F-A Ridge | `artifacts/oof/task3f_ridge/` |
| Task 3F-B–F diagnostics | `artifacts/oof/task3f_*/`, `artifacts/oof/task3f_feature_comparison/` |

Existing evidence is immutable. Use a new output directory for changed
analysis code or arguments; do not overwrite saved results.

## Completed Ridge/Sinkhorn evidence

The seed-78 / outer-fold-0 study artifacts are under
`artifacts/oof/ridge_sinkhorn_v3/`; its locked outer results are in
`outer_evaluation_v1/results.json`. The four original outer expert runs are in
`artifacts/oof/ridge_sinkhorn_outer_s78_o0/`. Outer fold 0 is consumed and
cannot select a replacement method. To regenerate its saved analyses:

```bash
./.venv/bin/python scripts/run_ridge_sinkhorn.py
./.venv/bin/python scripts/run_ridge_sinkhorn_outer.py
```

The second command reads the consumed outer results and cannot validate or tune
a replacement. Neither command loads original test examples. Full context is
in the [OOF results record](oof-results.md).

The completed `ridge_sinkhorn_3seed_v1` report and validated artifacts are
summarized in [rs3-final-report.md](rs3-final-report.md).

### Evaluation ID and provenance

Use a fresh, single-use `--evaluation-id` to evaluate frozen expert artifacts
from a separate evaluator checkout. The frozen study keeps its training
commit; the run records that identity alongside the evaluator commit. Without
an evaluation ID, the CLI requires the evaluator checkout to match the
training commit.

Each run is written under
`<run_root>/<study_id>/evaluation_runs/<evaluation-id>/`. Evaluation
publishes fold JSON and NPZ files under `outputs/`; reporting reads only a
completed evaluation with unchanged sidecars, then publishes under `report/`.
Stage records hash-link provenance, all 300 expert artifacts, fold
memberships, settings, environments, and published outputs. IDs cannot be
reused or overwritten. This option changes provenance and output placement;
it does not change scoring, selection, aggregation, or bootstrap settings.

The exact formal rerun
used evaluator checkout
`/mnt/hdd2/phatht/phat/MoE_Imbalance-evaluation-provenance` at commit
`64401e8e4cf076d618948185a2d008ef650d68c2` and this interpreter and runtime
profile:

```bash
cd /mnt/hdd2/phatht/phat/MoE_Imbalance-evaluation-provenance
RS3_EVALUATION_ID=rs3-reproduction-v1
PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/venv/bin/python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/server-gpu2.yaml \
  study evaluate --evaluation-id "$RS3_EVALUATION_ID"
PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/venv/bin/python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/server-gpu2.yaml \
  study report --evaluation-id "$RS3_EVALUATION_ID"
```

The frozen training identity remained
`e7357a7c5028c87f49739d0b390a5e4e6a258fba`; no training or lock creation
was performed.

The separate inner-diagnostics reproduction needs a pinned native result
snapshot and the audited historical reuse root. The
[diagnostics guide](diagnostics.md) records the exact snapshot commit
`5993d26eead9575160886d6130dce744ee9e02e6`, the required 16 historical
inputs, validation steps, and immutable output rules. A clone of that snapshot
alone is insufficient.
