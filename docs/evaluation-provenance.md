# Separate evaluation provenance

Use `--evaluation-id` when evaluating frozen expert artifacts from a clean evaluator checkout whose commit differs from the original training commit. The frozen study remains tied to its recorded training commit; each evaluation run records that training identity separately from the evaluator commit.

```bash
python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile configs/profiles/local-smoke.yaml \
  study evaluate --evaluation-id ridge3-fixed-evaluator-v1

python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile configs/profiles/local-smoke.yaml \
  study report --evaluation-id ridge3-fixed-evaluator-v1
```

The run is written below `<profile.run_root>/<study_id>/evaluation_runs/<evaluation-id>/`. `evaluate` publishes its fold JSON/NPZ files under `outputs/`; `report` reads only those completed outputs and publishes report files under `report/`. Each stage has immutable `*.start.json` and `*.complete.json` records. A failed stage may have a `*.failed.json` record, and its staged files are removed. Evaluation IDs are single-use and existing runs are never overwritten.

Before scoring and again before publishing, the workflow validates the frozen training provenance and study settings, fold and job manifests, runtime profile, locks, reuse decisions, evaluator checkout, and exact native or historical artifact bytes. The completion record includes hashes for all 300 expert artifacts, fold memberships, the Python executable and working directory, runtime package versions, and every published output. Reporting requires a successful matching evaluation record, its unchanged provenance sidecars, and all 30 hash-verified fold output files. Output is kept separate from the existing `study_analysis/` results.

Without `--evaluation-id`, existing CLI behavior is retained, including the requirement that the current checkout match the frozen training commit. This option changes provenance and output placement only; it does not change scoring, method selection, aggregation, or bootstrap settings.
