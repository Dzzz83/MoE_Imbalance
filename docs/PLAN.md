# Three-Seed Ridge–Sinkhorn Study Plan

## Objective and completion criteria

Implement and run a reproducible, nested OOF study of five Ridge/Sinkhorn
methods on CIFAR-100-LT. This is a **new exploratory study**, not an expansion,
reproduction, or continuation of the completed frozen Ridge/Sinkhorn study.
That study's prespecified expansion gate failed after its locked candidate was
dominated by fixed references, and that outcome remains unchanged. The new
study asks a different estimand: how each precommitted algorithm performs when
its complete inner-selection procedure is repeated across five outer folds and
three expert-training seeds.

The new protocol deliberately changes the earlier study by using four-fold
inner cross-fitting instead of fit folds 1--3 plus selection fold 0, maximin
BA/Tail selection instead of the old Pareto gate, reduced grids and multiple
frozen priors, and a new selective allocator. Freeze these deviations, their
rationale, and the new study ID before examining any newly generated outputs.
Do not revise the old study record or describe the new run as satisfying its
failed expansion gate.

Use training seeds `78`, `88`, and `1034`, five outer folds, four inner folds,
and the four experts CE, LAL, BalancedSoftmax, and Mixup. Train experts for 200
epochs and use final-epoch checkpoints. The original balanced CIFAR-100 test
set is out of scope.

Complete the study only when all 300 expert-run references are validated,
all 15 seed/outer-fold router configurations are locked before outer
evaluation, and every method has exactly one held-out prediction per
canonical sample and seed. Recompute results from saved predictions and
publish reports only after the complete artifact matrix passes integrity
checks. Label the work as retrospective nested-OOF evidence: this population
and some of its inner-fold history influenced earlier development.

Before launching full jobs, update [protocol.md](protocol.md) with a distinct
planned-study entry that preserves the old failed-gate result and declares the
new exploratory protocol. Follow its data roles, canonical fold algorithm,
metric definitions, class groups, artifact integrity, and test-access rules.
Hash this plan, the resolved study configuration, and the source commit into a
study lock before reading new outer outputs. Reuse the current Ridge/Sinkhorn
and Task 3F implementations and immutable artifacts as historical evidence;
do not rewrite or overwrite them. Current measured constraints and ruled-out
approaches are summarized in [research.md](research.md), and OOF results
belong in [oof-results.md](oof-results.md).

## Architecture and code quality

Keep the implementation non-redundant: extend or compose the existing
`OOFPipeline`, `OOFArtifactStore`, and `NestedOOFFoldManager` contracts for
expert execution, immutable run storage, and canonical fold membership. Do
not create a second trainer, fold manager, run-artifact format, or parallel
integrity implementation. Reuse the current Ridge/Sinkhorn numerical
functions, Task 3F feature and Ridge implementations where their inputs match,
canonical metric and class-group helpers, and immutable writers. Add only the
study-specific orchestration, schemas, and adapters needed for this protocol;
keep stateless numerical work in pure functions.

Represent the five methods through a compositional method registry: shared
router templates (contribution Ridge and residual Ridge), allocator templates
(identity, frozen-price Sinkhorn, and selective blend), and immutable
`MethodSpec` entries wire those parts and their frozen search spaces together.
Contribution Ridge + Sinkhorn must reference the same contribution-router
template and selected-kernel record; the residual variants likewise share one
residual template. Do not copy fit, prediction, prior, metric, serialization,
or validation logic into per-method implementations. A registry entry may
define method identity and composition only.

Core immutable types:

- `StudyConfig`: frozen seeds, folds, expert order, grids, priors, metric and
  bootstrap rules, and schema version.
- `ExpertRunSpec`: identity of an expert/seed/outer/inner training job.
- `RouterTrainingDataset`: aligned logits, labels, sample IDs, and provenance
  for an explicitly validated inner-training role.
- `RouterValidationDataset`: aligned logits, labels, sample IDs, and provenance
  for inner metric calculation; its labels are never passed to router fitting.
- `InferenceBatch`: logits and sample IDs only. It cannot carry labels, true
  class groups, or fold IDs.
- `MethodSpec`: method identity, router target, allocator, and hyperparameters.
- `FoldLock`: selected configuration, fitted parameter and source hashes, and
  training memberships.
- `StudyResult`: fold, seed, aggregate, and paired-comparison results.

Define narrow `RouterStrategy` / `FittedRouter` and
`AllocationStrategy` / `FittedAllocator` interfaces. `RouterStrategy.fit`
accepts only `RouterTrainingDataset`; `FittedRouter.predict_weights` and every
allocator transform accept only `InferenceBatch` or a weight matrix, never a
labeled dataset. Implement the contribution Ridge and residual Ridge routers,
plus identity, frozen-price Sinkhorn, and selective Sinkhorn allocators. Keep
orchestration in focused components: `OOFMatrixPlanner` (job inventory and
reuse), `InnerCVSelector` (cross-fitting and selection), `FoldStudyRunner`
(refit and locking), `StudyEvaluator` (outer predictions), and
`MarkdownReportBuilder` (reports derived from results). The study repository
must adapt/combine the existing immutable `OOFArtifactStore` and writers;
create a separate `StudyArtifactRepository` only for study-level artifacts
that the existing store does not represent. Inject configuration, strategies,
and repositories explicitly.

Use type annotations and public API docstrings; prefer descriptive names,
small methods, frozen configuration objects, and domain-specific errors. Keep
one source of truth for grids, priors, expert order, metrics, and class groups.
Maintain the dependency direction `CLI → orchestration → strategies and
utilities / artifact repository`; numerical code must not depend on the CLI,
artifact paths, or reporting. Avoid new dependencies unless existing project
libraries cannot meet a requirement.

## Methods and frozen search space

Evaluate these five primary methods:

1. **Contribution Ridge:** predict each expert's local true-class
   log-probability contribution from four inference-time confidence features;
   transform scores into soft expert weights.
2. **Contribution Ridge + Sinkhorn:** apply frozen-price relaxed Sinkhorn to
   the same selected Ridge kernel.
3. **Residual Ridge:** predict classification-aligned hinge-oracle residuals
   around `fixed_007 = (0, 0.25, 0.50, 0.25)`, then project onto the simplex.
4. **Residual Ridge + Sinkhorn:** apply frozen-price relaxed Sinkhorn to the
   same selected residual-Ridge kernel.
5. **Selective Residual Ridge + Sinkhorn:** blend per-row residual Ridge and
   Sinkhorn weights using only that row's Ridge distance from the anchor:

   ```text
   d_i = ||w_ridge_i - anchor||_1
   g_i = min(1, d_i / tau)
   w_i = (1 - g_i) * w_ridge_i + g_i * w_sinkhorn_i
   ```

Use these reduced grids, with no additional search:

| Component | Values |
|:--|:--|
| Contribution Ridge alpha | `{0.1, 10, 1000}` |
| Contribution Ridge gamma | `{0, 1}` |
| Contribution score temperature | `{1, 2}` |
| Contribution shrinkage | `{0.5, 0.75, 1}` |
| Residual oracle penalty | `{0.1, 1, 10}` |
| Residual Ridge alpha | `{0.1, 10, 1000}` |
| Residual Ridge gamma | `{0, 1}` |
| Residual scale | `{0.5, 0.75, 1}` |
| Sinkhorn rho | `{0.1, 1, 10}` |
| Sinkhorn priors | uniform and smoothed `fixed_006`, `fixed_007`, `fixed_010`, `fixed_011` |
| Selective gate tau | `{0.1, 0.25, 0.5, 1}` |

Smooth each fixed prior as `q = 0.95*w + 0.05*uniform`. Preserve the
existing relaxed-Sinkhorn objective and deterministic log-domain implementation;
record convergence diagnostics and fail on non-convergence. Fit dual prices
on inner rows only, then apply the frozen prices independently to each future
row. Do not use live batch-coupled inference.

Mandatory controls are uniform logits and probabilities; `fixed_006`,
`fixed_007`, `fixed_010`, and `fixed_011`; residual anchor without adaptation;
and prior-only global bias using the same frozen prior grid. Use canonical
`compute_class_groups`; report BA, Head, Medium, Tail, and sample accuracy as
distinct metrics. Do not treat label-dependent oracle weights as a method.

## Nested fitting and selection protocol

For each of the 15 seed/outer-fold pairs:

1. Load or train the four experts' held-out predictions for each of four inner
   folds. Each producing expert must exclude that prediction fold from its
   training membership.
2. Cross-fit each router: fit on three inner folds and predict the fourth;
   pool the four held-out predictions for candidate selection.
3. Select contribution-Ridge and residual-Ridge base settings separately.
   Contribution Ridge + Sinkhorn must reuse the selected contribution kernel
   and search only prior/rho; residual Ridge + Sinkhorn must reuse the selected
   residual kernel and search only prior/rho; the selective method must reuse
   that residual and Sinkhorn configuration and search only `tau`. Fit every
   candidate's dual prices on the three router-training folds before applying
   them to the held-out inner fold. This matched construction isolates what
   allocation and selection add to each Ridge kernel.
4. Compare candidates using `Delta BA` and `Delta Tail` against inner uniform
   logits, maximizing `min(Delta BA, Delta Tail)`. Resolve ties by BA, Tail,
   stronger regularization, less routing/OT, then stable configuration ID.
   Select the prior-only control by the same inner-only rule. Do not use
   outer-fold results to compare or select method families.
5. Refit each selected router using all inner OOF rows. Fit any Sinkhorn
   prices from these inner weights only.
6. Write and validate immutable locks for all five methods, including
   fit/selection memberships, selected settings, parameters, source and
   configuration hashes.
7. Only after all locks are valid, load the corresponding outer predictions
   and evaluate each locked method once. Do not use outer labels to fit,
   select, tune, or revise any setting.

The outer exclusion guarantee is fold-local: for a given evaluation fold, its
rows and labels are excluded from that fold's expert training, router fitting,
and selection. It is not a claim that the canonical population is historically
untouched. In addition, inner router folds are not statistically independent:
although each prediction comes from an expert that excluded that row, experts
producing one inner fold may have trained on images assigned to other inner
folds. Four-fold router cross-fitting prevents direct row-wise target leakage;
it does not remove this shared-training dependence. Preserve both limitations
in artifacts and final reporting.

Predictions and weights must use inference-time information only. Assert that
weights are finite, nonnegative, shaped `(N, 4)`, and row-stochastic. Test
that predictions are invariant to evaluation batch composition and that
changing outer labels cannot affect any lock or prediction. Do not use
full-data expert predictions on their own training examples as router
supervision. Do not read the original test set or select checkpoints,
methods, hyperparameters, subsets, or reports from its results. Record the
existing historical test-access status from the protocol; do not append to
the test-access log because this study must not access that set.

## Execution and artifacts

The normative operator interface is the config-driven package CLI,
`python -m expert_method`, with the study YAML and a separate runtime-profile
YAML. Use `study validate`, `study doctor`, read-only `study plan`, explicit
`study freeze`, bounded `study run` or `study session`, `study status`,
`bundle restore`/`bundle export`, and explicit `study lock`, `study evaluate`,
and `study report`. Full training requires `--execute-full`; runtime limits and
shards come from the selected profile. The historical
`scripts/run_ridge_sinkhorn_matrix.py` and `scripts/run_ridge_sinkhorn_3seed.py`
commands remain compatibility wrappers only. The engine modules formerly
under `scripts/` likewise re-export their `expert_method` package locations.
See [experiment-workflow.md](experiment-workflow.md) for the operator guide.

The package workflow calls `OOFPipeline`/`OOFArtifactStore` with the existing
`NestedOOFFoldManager` for each job. The study YAML pins the study identity;
the runtime profile supplies data/run roots, read-only reuse roots, device,
shard index/count, and maximum jobs. Job IDs include study, stage, expert,
training seed, outer fold, and inner fold (or an explicit outer marker).
Shards use the unsigned big-endian integer from the first eight bytes of
`SHA-256(job_id)`, modulo `shard_count`; assignment is independent of machine,
session, and completion order. A bounded run/session takes the first missing
jobs in stable ID order, no more than the profile limit. Read-only `study
plan` reports validated/reused/missing/invalid counts, estimated GPU-hours,
estimated artifact bytes, and available output-disk space.

Before any skip or reuse decision, validate a candidate run's canonical job
identity, protocol/fold membership, resolved config, final-epoch checkpoint,
prediction sample IDs/order, schema, and content hashes. A valid existing
artifact is reused; a missing artifact is runnable; a partial or incompatible
immutable artifact is a hard error and is never overwritten. CUDA preflight
must run before training and fail clearly if `--device cuda` is requested but
CUDA is unavailable or cannot allocate a small probe tensor; it must not
silently fall back to CPU. Planning and validation remain usable without CUDA.

Before claiming the 20 historical reuses, generate a compatibility table for
all 16 `task3c_oof` inner runs and four `ridge_sinkhorn_outer_s78_o0` outer
runs. For each target job ID, record the source experiment ID and run-relative
path, expert, seed, fold role, training and prediction membership hashes,
resolved-config hash, final epoch, checkpoint hash, and prediction hash. Reuse
only exact matches; any failed row becomes a missing new job rather than an
assumed reuse. Reference compatible sources in place and keep their trees and
metadata immutable; do not copy them into or relabel them as native v1 runs.

Kaggle sessions may end between any jobs. Define a portable artifact bundle
contract with a versioned manifest listing study/config/source hashes, stable
job IDs, each run's immutable relative paths, schemas, byte sizes, and SHA-256
hashes. The portable identity is the stable job ID, run-relative artifact path,
and content hash. Existing machine-specific absolute path strings are
informational provenance only: resolve the actual checkpoint and prediction
from the relocated run directory and validate their hashes. Never reject a
relocated run solely because its historical absolute-path string differs, and
never rewrite an immutable source artifact to change that string.

Use the Python standard library to create a deterministic tar bundle whose
archive paths and metadata are normalized and whose first member is the
canonical JSON manifest; do not add an archive dependency.
`bundle export --stage inner|outer` packages only validated complete jobs in
the selected profile shard. `bundle restore PATH...` validates manifest and
payload bytes before importing or merging cumulative bundles idempotently.
Import regular files only, reject absolute or parent-traversing paths and
links, and require every extracted path to remain inside the destination.
Restore must reject identity conflicts or differing bytes for the same job
ID and preserve existing files. Bundles contain no credentials or transient
Kaggle paths.

Bundle native v1 payloads under the exact `OOFArtifactStore` run-relative path
recorded by their manifest, and restore that same namespace below the writable
artifact root. Historical reuse entries are reference-only: record their
source experiment ID, source run-relative path, expected hashes, and a logical
reuse-root name, but do not include or import their payloads as v1 runs. The
operator must mount/provide the corresponding immutable reuse roots in every
session that plans, locks, evaluates, or reports; unresolved or mismatched
references are hard errors. A `StudyArtifactView` resolves native payloads
from the writable root and historical references from the named read-only
roots without copying or rewriting either. This validated view is the sole
logical input to later lock/evaluate/report stages.

The canonical study/profile sequence for new sessions is:

```bash
# At the start of each Kaggle session; edit only this external copy.
export PROFILE=/kaggle/working/rs3-profile.yaml
cp configs/profiles/kaggle.yaml "$PROFILE"
# Set this copy's shard_index and bundle_inputs for the current session.

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
# After all inner jobs are complete, persist the 15 locks separately:
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study lock
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" bundle export --kind locks
# Later outer sessions restore inner bundles and locks listed by the profile:
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" bundle restore
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study plan --stage outer
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study session --stage outer --execute-full
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study evaluate
python -m expert_method --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile "$PROFILE" study report
```

Never edit the tracked Kaggle profile: `study freeze` requires a clean checkout,
and each session must keep the exact frozen commit. The freeze identity includes
logical reuse-root names, not runtime profile paths. Keep those names fixed;
data/run/reuse paths, device, shard, job limit, and bundle paths may vary as
needed between sessions, subject to reuse validation. The two
`scripts/run_ridge_sinkhorn_*.py` commands retain their historical
flags as compatibility adapters only. Kaggle `/working` is session-local, so
every completed bundle must be exported to a durable Kaggle output/dataset or
downloaded before the session ends; later sessions restore it before planning
or running missing jobs. Full setup and bundle details are in
`docs/experiment-workflow.md`.

The normative array-analysis interface is `study lock`, `study evaluate`,
and `study report` on `python -m expert_method`. The old
`scripts/run_ridge_sinkhorn_3seed.py --stage lock|evaluate|report` command is
kept as a compatibility wrapper using the same services, artifact root, and
repeatable reuse roots. The latter two stages must refuse to run until their complete
prerequisite matrix exists. Before submitting jobs, emit a complete/missing/
reused manifest and verify each reused artifact against canonical metadata and
hashes. Never overwrite an incompatible immutable artifact. Preserve the
frozen job inventory and shard assignment in the study lock and bundle
manifests so work can be audited across sessions.

Execution order:

1. Record the protocol deviation in `docs/protocol.md`; freeze the study ID,
   this plan's hash, resolved configuration, source commit, and job manifest.
2. Audit the 16 existing seed-78/outer-fold-0 inner runs and reuse only the
   exact-compatible subset.
3. Train `240 - validated_inner_reuse_count` inner expert runs.
4. Lock methods for all 15 seed/outer-fold combinations.
5. Audit the four existing seed-78/outer-fold-0 outer runs and reuse only the
   exact-compatible subset.
6. Train `60 - validated_outer_reuse_count` outer expert runs.
7. Validate all 300 run references, all 15 locks, all predictions, and all
   hashes; then evaluate the complete matrix in one read-only pass.
8. Generate machine-readable results and Markdown from the same
   `StudyResult` object.

Store versioned outputs under `<run_root>/ridge_sinkhorn_3seed_v1/`, where
`run_root` is selected by the runtime profile. They include the frozen
configuration and canonical hash; job manifest; per-fold selection records
and locks; selected settings
and Sinkhorn diagnostics; predictions and weights in stable method/sample
order; per-fold, per-seed, and aggregate results; and source, config,
checkpoint, membership, and array hashes. Include environment and package
versions. Keep completed artifacts immutable and generated summaries beside
their machine-readable sources.

For each seed, concatenate its five outer folds so every canonical sample
appears exactly once. Report fold and seed metrics, mean and standard
deviation across three seeds, paired deltas against uniform, matched
Ridge-only methods, and fixed controls, and hierarchical paired bootstrap
intervals using 10,000 stratified class-then-sample paired resamples and seed
`20260924`, matching the existing outer-evaluation contract. These intervals
measure paired prediction uncertainty conditional on the trained models; they
do not capture retraining variability or dependence caused by overlapping
inner expert-training populations. Report selected hyperparameters for every
fold and Sinkhorn convergence and mean expert-mass diagnostics. Treat positive
BA and Tail deltas over uniform for every seed as a necessary success
criterion. An adaptive advantage also requires that fixed mixtures do not
dominate the method on both metrics. Do not publish partial matrices as
conclusions, mix OOF metrics into the full-data results table, or imply that
retrospective OOF results are independent confirmation.

After complete immutable results exist, update `README.md`,
`docs/protocol.md`, `docs/oof-results.md`, `docs/research.md`, and
`docs/reproduction.md`, and add a detailed archived study record. Keep
`docs/results.md` limited to its existing full-data role.

## Verification and gates

Before GPU execution, run focused tests for configuration validation, job
enumeration, maximin selection and tie-breaking, priors, target construction,
simplex projection, frozen prices, Sinkhorn convergence failure, selective
gate endpoints, and deterministic serialization/reload. Add a synthetic
end-to-end test covering planning, cross-fitting, selection, locking, outer
prediction, aggregation, report generation, and resume/idempotency. Run a
short integration smoke run for each expert configuration. Verify outputs
are finite and batch-independent; changing outer labels cannot change locked
artifacts. Do not evaluate the original test set for implementation checks.

Run full expert training on Kaggle or the designated training environment,
not as a substitute local full run. Accept each job only when its final-epoch
checkpoint, resolved configuration, finite metrics, and exact training and
prediction membership validate. Stop aggregation on any missing job, lock,
hash, convergence record, duplicate, or sample. Recompute all report metrics
from saved predictions before generating Markdown, and confirm the original
test-access log is unchanged. If all 20 historical runs validate, the 280
missing runs are estimated at about 93 GPU-hours from the project's historical
rate; the worst case of 300 new runs is about 100 GPU-hours. Ridge fitting,
Sinkhorn fitting, aggregation, and reporting are CPU array work.

Final completion requires:

- 300 validated expert-run references and 15 immutable fold locks.
- One complete held-out prediction per canonical sample for every method and
  seed, with all integrity and no-leakage checks passing.
- Reproducible metrics and reports regenerated from immutable artifacts.
- Required documentation updates and an explicit statement of retrospective
  OOF limitations and per-seed success criteria.
