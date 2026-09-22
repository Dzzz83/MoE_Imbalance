# Nested OOF Protocol

This document is the authoritative fold, membership and leakage contract for
OOF development. The executable implementation is
[data/nested_oof.py](../data/nested_oof.py). The protocol does not use the
balanced CIFAR-100 test set.

## Status

| Stage | Status |
|:--|:--|
| Task 3A: frozen fold and provenance infrastructure | Complete |
| Task 3B: OOF training pipeline | Complete |
| Task 3C: seed-78, outer-fold-0 collection | Complete |
| Task 3E-A/B: restricted exploratory analysis | Complete |
| Outer-fold expert training and evaluation | Not executed |
| Full 300-run nested experiment | Not executed |

Task 3C completed all 16 inner expert-training runs for the four experts at
training seed 78. The existing CE pilot was reused without retraining; the
remaining 15 runs completed successfully. The completed collection covers
outer fold 0 only.

## Frozen population and fold map

The canonical population is the 10,847-image index artifact
data/processed/lt_ir100_train_indices.npy with its canonical training labels.
The default fold-generation seed is 42. The manifest records the algorithm
name:

numpy.default_rng.per_class_balanced_remainder.v1

The algorithm sorts canonical sample IDs, shuffles each class with a seeded
NumPy generator, assigns the class quotient to all folds, and assigns each
remainder to the currently smallest fold bucket. This makes assignments
deterministic, class-stratified and close to equal in total size. Fold IDs are
zero-based.

There are five outer folds. For each outer fold, the manager stores the outer
expert-training and held-out indices, then creates four inner folds inside the
outer training population. Inner fold 0 is reserved for router hyperparameter
selection; inner folds 1–3 form the router-fit population. The four inner
held-out prediction sets cover the outer training population exactly once.

For outer fold 0:

- outer expert training has 8,677 images;
- outer evaluation has 2,170 images and is reserved;
- inner fold 0 has 2,170 prediction images;
- inner folds 1–3 have 6,507 prediction images in total; and
- each inner expert trains on the outer training set minus its prediction fold,
  approximately 6,507–6,509 images.

The canonical class groups are obtained from
scripts/base_trainer.py::compute_class_groups: Head n >= 100 (35 classes),
Medium 20 <= n < 100 (35), Tail n < 20 (30). The rarest classes 96–99 have
five canonical images. Five outer folds give each one held-out image per
outer fold; the outer training count is four. The inner development roles
therefore provide three router-fit images and one router-selection image for
each of those classes.

## Data-role contract

| Role | Population in one outer iteration | Purpose |
|:--|:--|:--|
| Outer expert training | Outer training indices | Train the separate expert used for that outer evaluation |
| Outer evaluation | Outer held-out indices | Evaluate the frozen router choice; never use for development |
| Inner expert training | Outer training minus one inner prediction fold | Generate held-out predictions |
| Inner prediction / OOF | One inner held-out fold | Provide router-development rows |
| Router fit | OOF rows from inner folds 1–3 | Fit a future router |
| Router selection | OOF rows from inner fold 0 | Select future hyperparameters |

The manager validates set relationships, duplicate and missing samples, class
counts and class coverage. An OOF row also carries the exact hash of the
expert-training membership and is rejected when its sample is in that
membership. Expert order is fixed and recorded. Provenance includes sample
index, label, fold ID, expert identity, seed, membership hash, checkpoint
hash, resolved configuration and logits.

## Non-cheating rules

Every future analysis must preserve all of these rules:

1. Every OOF prediction must come from an expert that excluded the
   corresponding image from training.
2. Router-fitting data is restricted to the permitted development population.
3. Router fitting and model-selection roles must be separated.
4. The reserved outer-evaluation population remains excluded from router
   development.
5. The original CIFAR-100 test set must not be used for repeated method
   selection.
6. Final-epoch checkpoints are used without test-based checkpoint selection.
7. Predictions from the original full-data experts on their own training
   images must not be used as honest OOF supervision.

The original full-data protocol itself remains valid for its completed
three-seed experiment. OOF training is a separate artifact-producing protocol,
not a replacement for those checkpoints.

## Inner fold 0 exposure

Task 3C diagnostics included descriptive analysis of inner-fold-0 labels.
Consequently, inner fold 0 cannot be characterized as completely untouched for
research choices influenced by those diagnostics. Tasks 3E-A and 3E-B excluded
inner fold 0 from every new metric calculation. A future fitted-router
experiment must document its fitting and selection procedure and acknowledge
this earlier exploratory exposure.

## Separation and dependence

The fit and selection rows are disjoint. This prevents direct reuse of the
same OOF rows for fitting and hyperparameter selection when the manifest
contract is followed.

The design does not make these measurements statistically independent. An
expert producing a router-fit row may have trained on router-selection images,
because each inner expert excludes only its own prediction fold; the reverse
dependency also exists. This is shared model-training dependence, not direct
label leakage into the held-out row, and must be reported as a limitation.

Inner experts train on about 60% of the canonical population, outer experts
would train on about 80%, and eventual full-data experts train on 100%.
Generalization from fold-trained experts to the full-data pool remains a
future question.

## OOF execution and artifacts

The execution path is scripts/oof_pipeline.py through scripts/run_oof.py and
scripts/run_task3c.py. A run names one expert, training seed, outer fold and
inner fold. The pipeline:

1. resolves and validates the frozen fold context;
2. records the manifest, configuration, memberships and hashes;
3. trains only on the declared expert-training IDs;
4. saves a final-epoch checkpoint in the dedicated run directory;
5. collects deterministic, non-augmented held-out logits with sample IDs; and
6. validates order, labels, fold IDs, checkpoint hash and configuration hash.

The reusable run layout is:

~~~text
artifacts/oof/<experiment_id>/
  fold_manifest.json
  expert_<name>/seed_<seed>/outer_<id>/<inner_id-or-outer_eval>/
    resolved_config.json
    run_metadata.json
    execution.log
    training_history.json
    checkpoints/<Expert>_seed<seed>_final.pt
    predictions.json
~~~

The completed analysis layout additionally contains:

~~~text
artifacts/oof/task3c_oof/
  fold_manifest.json
  batch_manifest.json
  aligned_oof.npz
  aligned_oof_metadata.json
  diagnostics.json
artifacts/oof/task3e_fixed_feasibility/
  experiment_config.json
  fixed_weight_results.json
  summary.md
artifacts/oof/task3e_soft_feasibility/
  experiment_config.json
  soft_oracle_results.json
  summary.md
~~~

Existing files are accepted only when their content and static metadata match;
incompatible files are rejected rather than overwritten. The canonical
checkpoints directory is not an OOF artifact root.

## Methodological limitations

The current design provides honest row-wise OOF predictions for each held-out
image, but it does not remove shared training dependence between fit and
selection models. It also does not establish performance across outer folds,
seeds or the final full-data experts.

The full planned matrix is five outer folds × (four inner experts plus one
outer expert) × four expert identities × three seeds = 300 training runs:
240 inner and 60 outer. It has not been executed. No current OOF document may
claim outer-fold accuracy, test-set improvement, or validated router quality.

For completed collection details and analysis numbers, see
[oof-results.md](oof-results.md). For the reusable diagnostic definitions, see
[expert-diagnostics.md](expert-diagnostics.md).
