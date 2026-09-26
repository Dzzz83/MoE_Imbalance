# Scripts directory

## Canonical interface

For new experiment work, use `python -m expert_method` with a study YAML and
runtime-profile YAML. The package CLI is the stable human-facing interface
for validation, planning, freezing, training sessions, status, bundle
management, locking, evaluation, and reporting. Start with
[`docs/experiment-workflow.md`](../docs/experiment-workflow.md).

## Compatibility shims

These preserve historical import or command paths while forwarding to the
package implementation:

- `oof_pipeline.py` → `expert_method.oof.pipeline`
- `ridge_sinkhorn_matrix.py` → `expert_method.ridge_sinkhorn.matrix`
- `ridge_sinkhorn_3seed_study.py` → `expert_method.ridge_sinkhorn.three_seed_study`
- `run_ridge_sinkhorn_matrix.py` and `run_ridge_sinkhorn_3seed.py` →
  deprecated command adapters over the same package services

Keep these paths compatible for saved notebooks and older automation. They
are not the preferred entry points for new runs.

## Historical research utilities

The remaining files are retained in place as utilities for completed or
diagnostic workflows. They are grouped by purpose rather than being alternate
front doors to the planned matrix study:

- **Training and evaluation:** `train.py`, `train_lal_weighted.py`,
  `evaluate*.py`, `check_runs.py`, and `run_oof.py` support historical
  full-data or single-run workflows.
- **Task 3C–3F analysis:** `run_task3*`, `task3*`, `ridge_sinkhorn_*`,
  `analyze.py`, and `analyze_subsets.py` reproduce or inspect earlier OOF,
  routing, and diagnostic analyses.
- **DACE experiments:** `train_dace_*.py`, `evaluate_dace.py`, and
  `diagnose_dace*.py` serve the separate DACE research branch.
- **Shared utilities:** `config.py`, `base_trainer.py`, `trainers.py`,
  `evaluation.py`, `subsets.py`, and `expert_diagnostics.py` are existing
  implementation dependencies used by those workflows.

Historical outputs and protocol notes remain useful evidence, but choose the
documented package CLI for a new or resumed `ridge_sinkhorn_3seed_v1` run.
