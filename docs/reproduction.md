# Reproduction Commands and Artifacts

This page collects entry points and artifact locations for the completed
full-data and OOF work. Canonical data roles and safeguards are in
[protocol.md](protocol.md); headline measurements are in
[results.md](results.md) and [oof-results.md](oof-results.md).

## Full-data benchmark

Training uses the four recipes in `configs/experts/`, seeds 78, 88, and 1034, and
final-epoch checkpoints. The original full-data protocol has no validation
split. To rebuild a run, use:

```bash
./.venv/bin/python scripts/train.py --config configs/experts/ce.yaml --seed 78
```

Repeat for `lal.yaml`, `balanced_softmax.yaml`, and `mixup.yaml`, and for
each configured seed. Full training is expensive; the existing checkpoint
names are `checkpoints/{expert}_seed{seed}_final.pt`, with resolved configs
beside them.

The historical test evaluation and analyses were run with:

```bash
./.venv/bin/python scripts/evaluate_experts.py --seeds 78 88 1034
./.venv/bin/python scripts/analyze_subsets.py --seeds 78 88 1034
./.venv/bin/python scripts/check_runs.py --seeds 78 88 1034
```

`evaluate_experts.py` reads the balanced CIFAR-100 test set and appends an
entry to [test-access-log.md](test-access-log.md). The test set has already
influenced historical decisions; do not run this evaluation for method
selection or router development.

## Task 3E and Task 3F analyses

These CPU array analyses use the existing aligned OOF artifact and default to
the recorded output directories. Run Task 3E-A before 3E-B, then Task 3F-A
before diagnostics B–F:

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

| Analysis | Primary artifacts |
|:--|:--|
| Task 3C aligned OOF input | `artifacts/oof/task3c_oof/` |
| Task 3E-A fixed weights | `artifacts/oof/task3e_fixed_feasibility/` |
| Task 3E-B soft feasibility | `artifacts/oof/task3e_soft_feasibility/` |
| Task 3F-A Ridge fits and scores | `artifacts/oof/task3f_ridge/` |
| Task 3F-B through 3F-F diagnostics | `artifacts/oof/task3f_*_diagnostics/`, `task3f_feature_comparison/` |

The existing artifacts are immutable evidence. A rerun accepts matching
content and static metadata; incompatible outputs fail rather than overwrite
them. For a changed implementation, use a new output directory and retain the
existing artifacts.

## Ridge/Sinkhorn staged study and locked evaluation

The staged CPU development analysis and report from saved outer artifacts use:

```bash
./.venv/bin/python scripts/run_ridge_sinkhorn.py
./.venv/bin/python scripts/run_ridge_sinkhorn_outer.py
```

The second command reads the already consumed outer-fold-0 evaluation
artifacts. It must not be used to tune or select a replacement. These
commands do not load the original balanced test set.

| Artifact | Location |
|:--|:--|
| Staged development and lock | `artifacts/oof/ridge_sinkhorn_v3/` |
| Locked outer result | `artifacts/oof/ridge_sinkhorn_v3/outer_evaluation_v1/results.json` |
| Outer paired predictions | `artifacts/oof/ridge_sinkhorn_v3/outer_evaluation_v1/predictions.npz` |
| Four outer expert runs | `artifacts/oof/ridge_sinkhorn_outer_s78_o0/` |
| Detailed frozen study record | [archive/ridge-sinkhorn-study.md](archive/ridge-sinkhorn-study.md) |

## Historical Kaggle outer jobs

The following commands record the four seed-78, outer-fold-0 expert jobs that
were completed on Kaggle. Running them again starts full training for the
already consumed outer fold; these commands document provenance and are not a
path to a new method comparison.

```bash
python scripts/run_oof.py --config configs/experts/ce.yaml --expert ce --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/experts/lal.yaml --expert logit_adjusted --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/experts/balanced_softmax.yaml --expert balanced_softmax --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/experts/mixup.yaml --expert mixup --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
```

Each expert trained on 8,677 outer-training IDs and predicted on the disjoint
2,170 outer-evaluation IDs. Outer fold 0 is consumed and cannot select,
validate, or tune another method. The original balanced test set was not read
by these OOF jobs or by the locked outer evaluator.

## Planned three-seed Ridge/Sinkhorn matrix

`ridge_sinkhorn_3seed_v1` is a separately declared exploratory study, not a
reproduction of the completed `ridge_sinkhorn_v3` study. Its expansion gate
remains failed, and the larger nested-OOF matrix is retrospective evidence;
it does not establish independent confirmation. The canonical operator guide
is [experiment-workflow.md](experiment-workflow.md), including Kaggle bundle
recovery, lock gates, artifact layout, and failure debugging.

Use the package CLI with the scientific study and runtime profile rather than
repeating protocol parameters on each command:

```bash
# In each Kaggle session, keep edits outside the frozen Git checkout.
export PROFILE=/kaggle/working/rs3-profile.yaml
cp configs/profiles/kaggle.yaml "$PROFILE"
# Edit this external copy for the current reuse choice, shard, and bundles.

python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study validate
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study doctor
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study plan --stage inner
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study freeze
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study session --stage inner --execute-full
```

Never edit the tracked `configs/profiles/kaggle.yaml`; the clean checkout is
required for freezing and must remain on the exact commit in every session.
Set the external profile's `shard_index` to the desired shard (0–3), attach
prior cumulative bundles and list them in its `bundle_inputs`. Runtime paths,
device, shard, job limit, and bundle settings may change without changing the
freeze identity. The logical names under `reuse_roots` are frozen: keep those
names unchanged, while their mounted paths may vary if they still validate
against the audit. Choose reuse names before the first freeze. After inner
completion, run explicit `study lock`, export its lock bundle, then restore
inner bundles and locks before planning or running outer shards. Finish with
explicit `study evaluate` and `study report`. The old
`scripts/run_ridge_sinkhorn_matrix.py` and
`scripts/run_ridge_sinkhorn_3seed.py` command forms are compatibility wrappers
only. No command in this flow reads the original balanced CIFAR-100 test set.

## Provenance

Detailed development decisions, exact fit/selection boundaries, command
history, and completed-study evidence are preserved in the
[Ridge/Sinkhorn study archive](archive/ridge-sinkhorn-study.md). Full-data
test reads are audited separately in [test-access-log.md](test-access-log.md).
