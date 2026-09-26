# Running the Ridge/Sinkhorn study

Use the package CLI for new work: `python -m expert_method`. The scientific
study and the machine-specific runtime profile are separate YAML files:

- [`configs/studies/ridge_sinkhorn_3seed_v1.yaml`](../configs/studies/ridge_sinkhorn_3seed_v1.yaml)
  owns the frozen folds, seeds, metrics, search grids, priors, and references to
  expert recipes. Its resolved scientific contents determine the study hash.
- [`configs/profiles/kaggle.yaml`](../configs/profiles/kaggle.yaml) is the
  committed Kaggle template; copy it outside the checkout before editing.
- [`configs/profiles/local-smoke.yaml`](../configs/profiles/local-smoke.yaml)
  is the checked-in local smoke profile. Runtime profiles own paths, device,
  shard, bundle locations, and the per-session job limit.
  These runtime values do not change the freeze identity. The sorted logical
  reuse-root names do: keep those names fixed after the first freeze, though
  their mounted paths may change if they resolve the same audited artifacts.
- [`configs/experts/`](../configs/experts/) owns the four human-edited training
  recipes. The study references these files; there are no copies in the study.

The supported full inventory is 300 expert jobs: 240 inner-fold jobs and 60
outer-fold jobs. Planning, validation, doctor, status, bundle operations, and
array-only analysis do not train experts. None of these commands reads the
original CIFAR-100 test set. The outer evaluation consumes only the frozen OOF
outer-fold artifacts.

## Before the first freeze

Choose whether to mount historical reuse before the first freeze. The Kaggle
template points at `/kaggle/input/rs3-reuse`; if that dataset is unavailable,
remove the `rs3` entry before freezing. The freeze records source, plan,
scientific config, fold/job inventory, reuse decisions, and the logical reuse
root names (not their machine-specific paths). A different set of reuse names
afterward requires a new study identity, not editing the frozen record.

Start from a real, clean Git checkout. Commit both the implementation and
`docs/PLAN.md`, and retain the exact commit ID for every later session:

```bash
cd /kaggle/working/expert_method
git checkout <FROZEN_COMMIT>
# Never edit the tracked Kaggle profile: freeze requires a clean checkout.
export PROFILE=/kaggle/working/rs3-profile.yaml
cp configs/profiles/kaggle.yaml "$PROFILE"
# Edit this external copy for the reuse choice before the first freeze.
git status --short
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study validate
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study doctor
```

`git status --short` must be empty for `study freeze`. Doctor checks installed
dependencies, Git identity/cleanliness, configured paths, disk, and requested
CUDA availability. It is read-only and does not load the dataset. Validation
also only reads YAML and expert recipes.

## Plan and freeze

`study plan` is a read-only inventory/reuse preview. It does not freeze the
study, create artifact directories, or train jobs. Review the inventory and
reuse audit before the first freeze:

```bash
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study plan --stage inner
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study freeze
```

Freeze is explicit and requires a clean committed checkout with a valid Git
`HEAD`. It creates the immutable study freeze and the fold/job/reuse manifests
under the profile's run root. Outer planning and execution are gated on all 15
method locks; do not plan outer artifacts before those locks exist.

## Kaggle shards and session loop

At the start of every Kaggle session, copy the tracked template to
`/kaggle/working/rs3-profile.yaml` and edit only that external copy. The
provided template points to canonical CIFAR-100-LT at
`/kaggle/input/cifar100`, requests CUDA, uses four stable shards, and caps one
session at eight jobs. Set `PROFILE` to the external copy, then edit its
`shard_index` to `0`, `1`, `2`, or `3` for that session; keep
`shard_count: 4` and `max_jobs: 8`. `data_root`, `run_root`, named reuse-root
paths, device, shard, job limit, and bundle input/output settings may vary
without changing freeze identity. The logical reuse-root names may not. Do not
change the study YAML to select hardware or shards. Use the same committed
source and study YAML in every session.

The command blocks below assume `PROFILE` points to the current session's
external profile. Repeat the copy/edit step in a fresh Kaggle session; never
edit `configs/profiles/kaggle.yaml` in the Git checkout.

For each shard, preview it before executing. A bounded `study run` advances up
to the configured number of jobs but does not restore or export bundles. The
recommended `study session` restores the bundle paths listed in the profile,
validates the frozen state, runs at most `max_jobs`, revalidates completed
jobs, and exports a cumulative bundle. Full 200-epoch execution requires the
explicit `--execute-full` confirmation:

```bash
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study plan --stage inner
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study session --stage inner --execute-full
```

Attach prior cumulative bundles as Kaggle inputs and list their paths (globs
are accepted) in the profile's `bundle_inputs`. The same restore step also
restores the lock bundle when included. To restore explicitly instead:

```bash
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" bundle restore \
  /kaggle/input/rs3-inner-bundles/inner-*.tar
```

`study session` never freezes, locks, evaluates, or reports implicitly. Export
the cumulative bundle to a durable Kaggle output/dataset or download it before
the session ends; `/kaggle/working` is ephemeral. Import/restore is idempotent
and validates bundle manifests and contents before writing. Historical reuse
references remain in their read-only roots and must be mounted for every
operation that validates them.

## Lock, outer work, and final report

When every inner shard reports zero missing and invalid jobs, create the full
set of 15 method locks and export them separately:

```bash
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study lock
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" bundle export --kind locks
```

Persist this lock bundle with the inner bundles. Before each outer session,
restore all inner bundles and the lock bundle, then any previous outer bundle.
Set the external profile's `shard_index` for the desired shard and run
`study plan --stage outer`; the lock-completeness gate runs before outer artifact or reuse
inspection. Continue with `study session --stage outer --execute-full` until
all outer shards are complete.

In a final clean session, restore all inner bundles, the lock bundle, and all
outer bundles, then run the explicit analysis stages:

```bash
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study evaluate
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study report
```

`evaluate` and `report` require the complete validated matrix and lock set.
They use only the frozen OOF predictions and do not read the original test
set. Outputs are under the profile run root in
`ridge_sinkhorn_3seed_v1/study_analysis/`.

## Status and artifacts

Inspect progress at any time with:

```bash
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study status
```

The run root separates immutable canonical jobs from retained attempts:

```text
<run_root>/ridge_sinkhorn_3seed_v1/
├── fold_manifest.json, job_manifest.json, reuse manifests
├── manifests/                                # immutable freeze records
├── expert_CE/seed_78/outer_0/inner_0/        # example canonical job payload
├── attempts/<stage>/<expert>/seed_*/outer_*/<inner_*/outer_eval>/<attempt-id>/
│   ├── attempt.json                         # status, phase, failure/environment
│   ├── execution.log                        # terminal tee and traceback
│   ├── resolved_config.json
│   ├── metrics.jsonl                        # flushed epoch-by-epoch
│   └── expert_CE/.../                       # partial checkpoint/prediction work
└── study_analysis/                           # locks, evaluations, reports
```

Attempts are never treated as completed jobs or included in matrix bundles.
Only a completely validated attempt payload is atomically promoted into its
canonical job directory; an existing valid canonical job is immutable.
Retries get new attempt IDs. `study status` reports attempt failures/running
separately from invalid canonical artifacts and includes attempt locations.

For debugging, inspect `attempt.json` for the job ID, status, and failure
phase; then read `execution.log`, `resolved_config.json`, and any retained
checkpoint or `metrics.jsonl`. Fix the cause in code/config only through a new
clean commit if it changes the frozen experiment, and preserve failed attempt
directories for provenance. Never copy partial attempt files into canonical
job directories.

The old `scripts/run_ridge_sinkhorn_matrix.py` and
`scripts/run_ridge_sinkhorn_3seed.py` commands remain compatibility wrappers;
new sessions should use this module CLI and its study/profile files.
