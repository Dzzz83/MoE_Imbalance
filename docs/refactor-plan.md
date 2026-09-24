# Codebase Refactor Plan

Status: planning and review document for the current worktree (2026-09-23).
The verified defect inventory is in [code-audit-report.md](code-audit-report.md).
That report links back here for the migration context.

Phase 1 correctness hardening is complete in the shared worktree. The later
OOF/application extraction, analysis-foundation completion and legacy
quarantine remain planned migration stages; this document does not claim a
canonical metric rerun.

This plan is deliberately conservative. The repository contains completed
scientific tracks and immutable OOF/analysis artifacts; a cleaner package
boundary is useful only if it preserves those protocols, artifact schemas and
the distinction between development and evaluation populations.

## Audited scope

The three audit passes covered the following code and documentation surfaces:

- Nested OOF domain and execution: `data/nested_oof.py`,
  `data/oof_datamodule.py`, `scripts/oof_pipeline.py`, `scripts/run_oof.py`,
  `scripts/run_task3c.py`, `scripts/task3c_oof.py`, and
  `scripts/expert_diagnostics.py`.
- Inference and routing: `data/tta.py`, `scripts/router/`,
  `scripts/evaluate_experts.py`, and `scripts/utils/features.py`.
- Training and legacy evaluation boundaries: `scripts/base_trainer.py`,
  `scripts/evaluate_dace.py`, `scripts/train_dace_*.py`,
  `scripts/train_paco*.py`, and `models/resnet32.py`.
- Task 3F diagnostics and persistence: `scripts/task3f_*.py`, tests for those
  modules, and the shared analysis-artifact foundation currently present in
  `scripts/analysis/`.
- Protocol and handoff documentation: `docs/project-context.md`,
  `docs/nested-oof-protocol.md`, `docs/expert-diagnostics.md`,
  `docs/results.md`, `docs/oof-results.md`, and the historical audit record.

Out of scope for this slice are a new routing method, a schema migration,
full retraining, a full OOF matrix, and any new test-set evaluation. The OOF
metadata atomicity defect is now fixed in the current worktree; broader OOF
application extraction and interruption integration remain later phases.

## Design principles

1. Preserve the scientific contract before improving the shape of the code.
   Existing results, fold roles, class-group definitions, provenance fields and
   artifact bytes are compatibility surfaces.
2. Make roles explicit. Outer expert training, inner prediction, router fit,
   router selection, outer evaluation and historical test evaluation must be
   represented as different domain concepts, not inferred from path names.
3. Keep pure validation and numerical operations independent of datasets,
   checkpoints, CUDA and the test set wherever possible. Inject loaders,
   models, clocks and writers at application boundaries.
4. Fail closed on malformed metadata, non-finite values, invalid class IDs,
   accidental test access and incompatible existing artifacts.
5. Keep CLI modules thin. Argument parsing should create a typed request and
   delegate to an application service; it should not own fold construction,
   training policy, serialization or report formatting.
6. Prefer small value objects and cohesive services over a large inheritance
   hierarchy. Existing public imports and JSON/NPZ schemas remain facades while
   internals are moved behind tested interfaces.
7. Reuse the canonical Head/Medium/Tail implementation and the shared
   write-once artifact primitives. Do not copy threshold or serialization
   logic into another task.
8. Separate measured facts from hypotheses. A local test or synthetic run can
   establish a contract, but cannot establish BA, Tail accuracy or generality.

## Current smells and risks

| Smell | Evidence | Refactor risk if left in place |
|:--|:--|:--|
| Large mixed-responsibility modules | `data/nested_oof.py`, `scripts/oof_pipeline.py` and `scripts/task3c_oof.py` each combine domain objects, validation, persistence and orchestration | New callers can bypass invariants or duplicate serialization logic |
| Validation spread across constructors, managers and CLIs | Fold counts, router roles, provenance and artifact completeness were not all checked at one boundary | A restored artifact can disagree with its declared fold roles |
| Mutable execution state beside immutable scientific records | OOF run status and checkpoint updates share code with frozen fold definitions | Recovery code can accidentally rewrite a completed scientific artifact |
| CLI/application coupling | `run_oof.py`, `run_task3c.py` and Task 3F runners perform parsing, planning, execution and report output in one path | Tests need training-stack imports and small changes have broad blast radius |
| Ambiguous router API | `predict_proba` means expert-routing weights, while calibration needs class probabilities | ECE can be computed from a distribution that is not the classifier's output |
| Legacy API drift | DACE callers still pass validation loaders and `class_counts` to the current no-validation `BaseTrainer` | Retired commands fail at runtime or silently imply a forbidden protocol |
| Repeated artifact I/O code | Several Task 3F modules previously implemented their own JSON/NPZ/hash/write-once helpers | Persistence behavior diverges and atomicity is easy to miss |
| Mutable versus immutable artifact lifecycles | OOF metadata is mutable while manifest/config/predictions are immutable; these paths need different writer semantics | Treating every JSON file as write-once or freely mutable can break recovery or provenance |
| Test and training dependencies at collection time | Some focused tests import Torch/Torchvision at module import | A missing optional dependency obscures which contracts were actually tested |

## Target package and CLI architecture

The following is a target layout, not a claim that all of these files already
exist. During migration, the current import paths remain compatibility facades.

```text
data/
  protocol_splits.py             canonical population and labels
  nested_oof.py                 compatibility facade for public OOF types
  nested_oof/
    models.py                   immutable fold/run value objects
    planning.py                 deterministic outer/inner fold construction
    validation.py               FoldIntegrityValidator and artifact checks
    artifacts.py                manifest/prediction serialization

scripts/
  analysis/
    artifacts.py                shared read/hash/write-once infrastructure
    validation.py               array and weight contracts (future extraction)
    combination.py              logit/probability combination (future extraction)
  oof/
    planning.py                 run matrix and role resolution
    execution.py                train/collect one declared run
    store.py                    atomic mutable run state and immutable outputs
    reports.py                   diagnostics and summaries
  router/
    base.py                     explicit routing-weight/class-distribution API
    uniform.py, probability.py,
    confidence.py, tta.py       parameter-free router implementations
  cli/
    run_oof.py, run_task3c.py   thin request adapters (future extraction)
```

The target has four layers:

- Domain: frozen fold, role, checkpoint and prediction records with no file or
  training-stack dependency.
- Validation: canonical-label class counts, exact membership derivation,
  provenance and schema checks. `FoldIntegrityValidator` is the first slice of
  this boundary.
- Infrastructure: loaders, checkpoint adapters, atomic writers and access-log
  capabilities.
- Application/CLI: a small service that resolves a request, executes the
  declared work and returns a report. It does not redefine domain invariants.

The router contract should reserve `predict_proba` for expert-contribution
weights and expose `predict_distribution` for the class distribution produced
by the same classifier as `predict_class`. Existing callers can be adapted
through a compatibility alias before any deprecation.

## Phased migration, ownership and dependencies

| Phase | Slice and owner | Depends on | Exit gate |
|:--|:--|:--|:--|
| 0. Characterize | Parent/reviewer: freeze protocol, public APIs, artifact schemas and result-status vocabulary | Current executable code and focused tests | Contract inventory is reviewed; no training or test evaluation is needed |
| 1. Correctness hardening | Luna Max coders: fix verified numerical, access-control, Task 3F and OOF integrity defects; add minimized regressions | Phase 0 contracts | **Complete in current worktree:** focused synthetic tests pass; malformed artifacts fail closed; no canonical rerun claimed |
| 2. OOF domain extraction | OOF owner: split fold models, validators and serializers behind `data.nested_oof` facade | Phase 1 validator tests and current manifest round trips | Old imports and JSON/NPZ schemas remain byte/field compatible; manager and manifest agree |
| 3. OOF application and CLI | Pipeline owner: extract plan/execute/store/report services and thin `run_*` adapters | Phase 2 domain API; atomic writer | Dry-run and one synthetic run cover recovery, idempotency and role separation |
| 4. Analysis foundation completion | Analysis owner: finish shared array/combination validation migration without changing metric definitions | Phase 1 tests and Stage 0 characterization | All migrated tasks preserve error taxonomy, tolerances, hashes and output bytes |
| 5. Legacy quarantine | Parent plus legacy owner: isolate DACE/PaCo commands, document stale protocol, block accidental imports | Inventory of active callers and Phase 3 import graph | Active path has no legacy dependency; quarantine smoke/import check is green |
| 6. Authorized reevaluation | Research owner, only after a written protocol decision | All prior gates, frozen method/criteria, access grant | Explicitly authorized run, access log, provenance and report; no retrospective selection |

Ownership names are role names rather than a claim about a particular branch.
The parent agent remains the reviewer and integration owner; coders should keep
each slice small enough to review independently.

## Protocol invariants

- The canonical dataset is CIFAR-100-LT IR=100 with 10,847 training samples;
  the canonical index artifact and labels are the source of truth.
- The active expert pool is CE, LAL, BalancedSoftmax and Mixup, with seeds 78,
  88 and 1034 for the original full-data track.
- The original full-data track has no validation split; final-epoch
  checkpoints are reported. DACE/PaCo validation-oriented code is not an
  alternative active protocol.
- The balanced CIFAR-100 test set is evaluation-only. It cannot fit, tune,
  select, or diagnose a router. Protected reads require a consumed
  `TestAccessGrant` and an append-only access record.
- Head/Medium/Tail boundaries come only from
  `scripts.base_trainer.compute_class_groups`.
- A method succeeds only when both BA and Tail improve over the applicable
  uniform-logit baseline consistently across the configured seeds.
- Each OOF prediction comes from an expert that excluded its sample from
  training. Router fit uses declared inner folds 1–3; selection uses inner fold
  0. Outer fold 0 has been consumed by one locked Ridge/Sinkhorn evaluation and
  cannot be used for method reselection; other outer folds retain their
  evaluation role.
- OOF class-count tuples, membership hashes, router role IDs and derived sample
  memberships are recomputed from canonical labels and exact fold IDs.
- Complete inner-OOF artifacts contain exactly the expected sample/fold/expert
  key set; valid outer-evaluation records are not additional rows in that
  artifact.
- Existing completed artifacts are write-once and incompatible reruns are
  rejected. Mutable run metadata is updated atomically; execution-log append
  semantics remain separate.

## Verification gates

1. Static: inspect the diff, run `python -m py_compile` on touched modules and
   `git diff --check`.
2. Contract/unit: run focused synthetic tests for the changed domain or
   numerical component. These must not load the balanced test set or train a
   model.
3. Artifact: round-trip manifests and predictions, verify hashes, exact role
   membership and write-once/idempotent behavior. Include interruption tests
   for mutable OOF metadata and preserve append-only log semantics.
4. Integration: when a trainer or loader changes, use the repository's small
   CPU dry run and verify shapes, finite losses/logits, gradients and device
   placement. Dependency availability is a verification prerequisite, not a
   scientific result.
5. Canonical experiment: only an explicitly authorized Kaggle/full run can
   establish BA, Head/Medium/Tail, convergence or a method improvement. A
   local unit test, synthetic fixture or compilation check never establishes a
   result.

Parent integration evidence for the current worktree is: the initial full
suite had **380 passed with 4 collection warnings**. Final verification passed
`compileall`; the focused OOF suites completed with **43 passed in 9.63s**, and
the full `pytest` suite completed with **410 passed in 26.39s with no warnings**.
These checks establish software/test status only; no full OOF/training/test
metric rerun is claimed here.

## Legacy quarantine strategy

Legacy DACE and PaCo scripts are retained as historical source until their
provenance and replacement decision are recorded. The quarantine should:

- classify each command as active, historical, or retired in a small manifest;
- move retired implementations under a clearly named `legacy/` namespace (or
  leave a compatibility shim that raises a documented error), without changing
  their historical files in place;
- add a `legacy/README.md` explaining stale data splits, validation semantics,
  checkpoint APIs and why those commands cannot produce current-protocol
  results;
- prevent active packages from importing legacy modules and add a lightweight
  import/reference check in CI; and
- preserve old checkpoints and reports as historical evidence, never silently
  relabeling their metrics as current results.

Quarantine is preferable to silently adapting DACE to the current no-validation
protocol: such an adaptation would be a new experiment and would need its own
design, tests and authorization.

## Completed Phase 0 and Phase 1 slices

The following are present in the current worktree. “Present” means code and
focused regression coverage were added; it does not mean a canonical metric
rerun occurred.

Phase 0 characterization and audit:

- The active protocol, OOF role map, metric boundaries, test-access policy and
  artifact schemas were traced to executable owners.
- The three audit passes produced only evidence-backed defect rows; speculative
  refactors remain in this plan.
- Existing saved OOF manifest validation was checked without loading the test
  set or running training.

Phase 1 correctness/hardening:

- `FoldIntegrityValidator` recomputes outer/inner/router class counts from
  canonical labels, derives router fit/selection memberships from declared
  inner-fold IDs, and `OOFPredictionArtifact.validate(require_complete=True)`
  compares the full key set.
- TTA uses normalized black padding, per-sample crop coordinates and
  device-safe seeded operations; router class distributions now match the
  classifier used for `predict_class`.
- Legacy test readers consume their one-shot access grants; utility classes are
  excluded from pytest's test-class discovery heuristic.
- Energy computation uses a shifted log-sum-exp; PaCo queue insertion handles
  arbitrary batch sizes and device placement.
- Task 3F weight reports reject missing classes, and prediction diagnostics
  reject fractional or out-of-range class IDs. Soft-mixture oracle tolerances
  must be finite and positive.
- The shared immutable analysis-artifact writer is present and several Task 3F
  callers use it. `OOFArtifactStore` delegates immutable
  manifest/config/prediction writes to that writer and uses
  `AtomicMetadataWriter` for mutable status transitions; execution-log append
  behavior is unchanged.

For the defect-by-defect status and result accounting, see
[code-audit-report.md](code-audit-report.md).
