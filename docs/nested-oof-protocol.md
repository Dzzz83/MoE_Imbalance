# Task 3A — nested OOF protocol

The executable fold and provenance contract is implemented in
[`data/nested_oof.py`](../data/nested_oof.py).  It is infrastructure only: it
does not load images, train experts, fit Ridge/Sinkhorn routers, or access the
balanced CIFAR-100 test set.

## Frozen population and fold map

`NestedOOFFoldManager.from_canonical_training_data()` obtains indices and
labels through [`data.protocol_splits`](../data/protocol_splits.py).  The
canonical artifact is `data/processed/lt_ir100_train_indices.npy`; the
constructor requires exactly 10,847 samples and records the SHA-256 of the
artifact bytes.  The default fold seed is `42` and the manifest records both
the seed and the algorithm:

`numpy.default_rng.per_class_balanced_remainder.v1`

The algorithm sorts canonical sample IDs, shuffles each class with a seeded
NumPy generator, assigns each class's quotient to all folds, and assigns its
remainder to the currently smallest fold buckets.  Thus assignments are
deterministic, class-stratified, and close to equal in total size.  Fold IDs
are zero-based.

For each outer fold, the manager stores the outer expert-training and held-out
indices, then creates four inner folds inside the outer training population.
Inner fold `0` is reserved for router hyperparameter selection; inner folds
`1–3` are the router-fit population.  All four inner held-out prediction sets
together cover the outer training population exactly once.

The current canonical population has 35 Head, 35 Medium, and 30 Tail classes
under the single source of truth
`scripts.base_trainer.compute_class_groups` (Head `n >= 100`, Medium
`20 <= n < 100`, Tail `n < 20`).  The rarest classes are 96–99, each with
five images.  Every outer held-out fold contains one image from each of those
classes; its outer training count is four.  Every inner held-out partition
contains one and its inner expert-training partition contains three.  The
router-fit and router-selection counts for each rare class are three and one,
respectively.

## Data-role contract

| Role | Population allowed in one outer iteration | Purpose |
|---|---|---|
| Outer expert training | Outer training indices | Train the separate expert used for that outer evaluation only |
| Outer evaluation | Outer held-out indices | Evaluate the frozen outer router choice; never used for development |
| Inner expert training | Outer training minus one inner prediction fold | Generate honest held-out predictions for that inner fold |
| Inner prediction / OOF | One inner held-out fold | Rows available for router development |
| Router fit | OOF rows from inner folds 1–3 | Fit a future router on labels/features/logits |
| Router selection | OOF rows from inner fold 0 | Select future router hyperparameters, disjoint from router fit |

The manager validates all set relationships, duplicate/missing samples,
class counts, and class coverage.  An outer evaluation index cannot appear in
any inner or router-development set.  `OOFPredictionRecord` additionally
requires the exact hash of the expert-training membership and rejects a row
whose sample is in that membership.  `inner_fold_id=None` is reserved for a
future outer-expert prediction row; such a row must belong to the outer
evaluation population and must carry the outer training-membership hash.

`FoldManifest` is versioned and round-trippable as JSON.  It records the
canonical indices and labels, canonical artifact hash, seed, algorithm,
experiment configuration, every outer/inner training and held-out membership,
class-count vector, and router-development membership.  `OOFPredictionArtifact`
records a fixed expert ordering and class count, sample/label and fold IDs,
expert identity and seed, training-membership hash, checkpoint path/hash,
resolved configuration/hash, logits, and optional features.  Serialization is
available through `to_json()`/`from_json()`; no persistent artifact is created
by tests.  Any approved development manifest should be placed under the
dedicated `artifacts/oof/` root, never under the canonical processed-data or
checkpoint paths.

## Separation that is—and is not—established

The router-fit and router-selection rows are disjoint, so a future Ridge
implementation cannot use the same OOF rows for fitting and hyperparameter
selection if it follows the manifest contract.  The code intentionally does
not implement Ridge, model selection, or Sinkhorn.

The rows are not statistically independent merely because their sample IDs are
disjoint.  Inner experts are trained on overlapping outer-training
populations: an expert producing a router-fit row may have trained on images
in the router-selection rows, and vice versa.  The cross-fitted prediction is
honest for its own sample, but the fitted models share data and initialization
protocol.  In addition, inner experts see about 60% of the canonical
population, outer experts about 80%, and eventual full-data experts 100%, so
the distribution shift from OOF models to final experts remains a future
experimental question.

No claim of router quality, expert specialization, convergence, or test-set
improvement follows from this infrastructure or its synthetic tests.

## Task 3B execution protocol

The executable development path is `scripts/oof_pipeline.py`, exposed through
`scripts/run_oof.py`.  It is intentionally not called by `scripts/train.py` or
the active evaluation entry points.  A run must name one expert, training seed,
outer fold, and (for an inner run) inner fold.  `OOFRunSpec.resolve()` obtains
the immutable memberships from `NestedOOFFoldManager`; a run cannot request the
full canonical population implicitly.

The order of operations is:

1. Resolve and validate the fold context and its exact class-count vector.
2. Record the frozen manifest, resolved configuration, membership lists, and
   membership hash in a dedicated run directory.
3. Build the training dataset only from the declared expert-training IDs.
4. Build the existing registry trainer with those fold-specific counts.
5. Save and validate a final checkpoint inside that run directory.
6. Build a deterministic, non-augmented loader only from the declared held-out
   IDs and collect logits with original sample IDs.
7. Validate the complete single-run artifact, including order, labels, fold
   IDs, expert identity, checkpoint hash, and resolved-configuration hash.

The artifact layout is:

```
artifacts/oof/<experiment_id>/
  fold_manifest.json
  expert_<name>/seed_<seed>/outer_<id>/<inner_id-or-outer_eval>/
    resolved_config.json
    run_metadata.json
    execution.log
    training_history.json
    checkpoints/<Expert>_seed<seed>_final.pt
    predictions.json
```

Existing files are accepted only when their content and static metadata match;
incompatible files are rejected rather than overwritten.  A completed run can
be recovered by loading its validated checkpoint and collecting the missing
prediction artifact.  An unrecorded checkpoint is treated as ambiguous and is
not silently reused.  The canonical `checkpoints/` directory is rejected as an
artifact root.

## Router-development recommendation

The current five-by-four arrangement is defensible for an outer-fold estimate
of a *pre-specified complete procedure*: all expert and router development for
outer fold `j` is contained in its outer training population, and the outer
evaluation population is not exposed until the procedure is frozen.  It
prevents direct in-sample OOF supervision and prevents outer labels from
entering training, router fitting, or hyperparameter selection.

It does not make router-fit and router-selection measurements statistically
independent.  An expert producing an OOF row for router fitting may have
trained on router-selection images, because each inner expert excludes only
its own prediction fold.  The reverse dependency also exists.  This is shared
model-training dependence, not direct label leakage into the held-out row, and
it must be reported as a limitation rather than described as independence.

Stricter alternatives were considered:

| Alternative | Benefit | Cost and limitation |
|---|---|---|
| Current five-by-four cross-fitting | Honest row-wise OOF predictions; outer evaluation remains untouched; 300 planned fold-expert runs for four experts and three seeds | Fit/selection-generating experts share training images; inner models train on ~60% while outer models train on ~80% |
| Separate expert pools for router-fit and router-selection | Removes the direct cross-role model-training overlap if each pool excludes the other role | Requires an additional cross-fitting pool (roughly doubles inner expert runs, from 240 to ~480 before outer runs); still estimates a procedure involving smaller inner models |
| Permanently separate development population | Strongest development/evaluation separation | Sacrifices training data, changes the canonical protocol, and requires a new frozen manifest and approval |
| Additional nested folds for every router hyperparameter | More orthodox selection nesting | Computationally disproportionate for this project and increases variance in already small tail classes |

The recommended protocol is therefore the existing manifest plus a written
freeze point: choose the router feature definition, Ridge/Sinkhorn search
space, regularization grid, and selection rule before reading any outer-fold
labels or results.  Fit on inner folds `1–3`, select only on inner fold `0`,
then train the separate outer expert on outer training data and evaluate it on
outer evaluation data.  Aggregate outer results only after all outer-fold
decisions are fixed.  Do not claim complete independence of the inner models.

This recommendation avoids changing the frozen Task 3A manifest.  It is a
protocol recommendation, not evidence that a router improves performance.

## Pilot boundary and workload

The reproducible pilot command is documented in the Task 3B handoff and in the
`run_oof.py` module docstring.  Stage A uses a separate experiment ID, one
training batch, and one epoch; it is a smoke test only.  Stage B is one full
200-epoch CE run for seed 78, outer fold 0, inner fold 0.  The wrapper requires
`--execute-full` for a non-smoke 200-epoch run, so preparing the command does
not launch it.

From the repository root on Kaggle, the commands are:

```bash
cd /kaggle/working/MoE_Imbalance

# Stage A: short, non-reportable smoke test
python scripts/run_oof.py \
  --config configs/ce.yaml --expert ce --seed 78 \
  --outer-fold 0 --inner-fold 0 \
  --experiment-id task3b_stageA_smoke \
  --artifact-root ./artifacts/oof \
  --data-root ./data \
  --device cuda --epochs 1 --max-batches 2

# Stage B: prepare/run only after the smoke test and explicit approval
python scripts/run_oof.py \
  --config configs/ce.yaml --expert ce --seed 78 \
  --outer-fold 0 --inner-fold 0 \
  --experiment-id task3b_pilot_ce_s78_o0_i0 \
  --artifact-root ./artifacts/oof \
  --data-root ./data \
  --device cuda --epochs 200 --execute-full
```

Before Stage B, record the assigned accelerator with
`nvidia-smi`.  During it, capture utilization and memory with
`nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > /kaggle/working/gpu_usage.csv`
and run the command under `/usr/bin/time -v`.  Afterward, record
`du -sh ./artifacts/oof/task3b_pilot_ce_s78_o0_i0` and copy the
artifact directory to persistent Kaggle output.  These measurements are not
substituted by the local RTX 3060 smoke test.

The planned full matrix is 5 outer folds × (4 inner experts + 1 outer expert) ×
4 expert identities × 3 seeds = 300 training runs: 240 inner experts and 60
outer experts.  It trains on 2,082,624 sample instances per epoch across the
matrix, or 416,524,800 sample-epochs at 200 epochs.  It produces 520,656 inner
prediction rows and 130,164 outer prediction rows across the 12 expert/seed
combinations, before serialization overhead.  Pilot wall time, peak VRAM,
utilization, and artifact sizes must be measured on Kaggle; local smoke tests
do not establish those resources.

At the time of this task, Kaggle's notebook documentation states a 12-hour
CPU/GPU notebook session and 20 GB of auto-saved `/kaggle/working` disk.  It
also lists P100 and T4 notebook accelerator variants; accelerator availability
and weekly GPU quota are account- and demand-dependent, so the pilot must
record the actual assigned GPU and quota before scaling the matrix.  The
estimated full artifact volume is below that disk ceiling, but it should still
be sharded by experiment/session and copied to a persistent Kaggle Dataset
output rather than relying on scratch space.  See the [Kaggle notebook
technical specifications](https://www.kaggle.com/docs/notebooks) and [GPU usage
guidance](https://www.kaggle.com/docs/efficient-gpu-usage).
