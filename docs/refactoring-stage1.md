# Refactoring Stage 1 — Shared Analysis Foundations

Stage 1 extracts small, stateless foundations while preserving the existing
Task 3E/3F scientific contracts and artifact bytes.

## Shared modules

- `scripts/analysis/validation.py` validates real numeric arrays,
  integer-valued vectors, and row-stochastic non-negative expert weights. Shape
  requirements, finite-value policy, and weight-sum tolerance are explicit.
- `scripts/analysis/combination.py` provides stable final-axis softmax,
  weighted original-logit combination, and weighted per-expert-probability
  combination. Logit and probability ensembles remain separate operations.
- `scripts/analysis/artifacts.py` provides JSON/NPZ loading, canonical JSON and
  NumPy serialization, SHA-256/hash verification, repository provenance, and
  write-once text/JSON/NPZ persistence. Existing sorted, indented JSON with a
  trailing newline is retained.

## Migrated callers

The reusable softmax and weight validation are used by
`scripts/expert_diagnostics.py`; its task-specific metric and label-dependent
logic remains local. `scripts/task3e_fixed.py`, `scripts/task3e_soft.py`, and
`scripts/task3f_ridge.py` use shared artifact loading, hashing, Git provenance,
and immutable persistence. Task 3F-A retains its public combination helpers as
compatibility wrappers around the shared weighted-logit/probability functions.

The OOF execution store remains separate because its run metadata is mutable
during recovery. Specialized Task 3F-B–F diagnostic loaders and validators also
remain local until their differing error, tolerance, and provenance contracts
can be migrated in controlled follow-up work.

## Compatibility guarantees

The shared primitives do not access labels, folds, datasets, checkpoints, or
evaluation populations. Callers continue to enforce their own scientific
requirements, including expert order, restricted OOF partitions, supervised
target construction, metric definitions, and source-artifact schemas. Stage 1
does not consolidate metrics or introduce a router interface or routing method.

Synthetic regression tests cover independent weighted formulas, extreme-logit
softmax behavior, validation boundaries, JSON/NPZ round trips, SHA-256
verification, canonical serialization, Git provenance, and write-once
incompatible-rerun rejection.
