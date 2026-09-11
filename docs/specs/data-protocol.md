# Doublecheck spec

## Goal
Replace the two-way train/val data protocol with a single canonical long-tailed training artifact (10,847 indices, IR=100, seed 42), so experts train on the full LT set and are evaluated only on the balanced 10K CIFAR-100 test set.

## Scope
IN SCOPE: utils/create_lt_split.py (regenerate as a single-artifact generator), data/protocol_splits.py (drop the val role, expose the canonical loader), tests for both. OUT OF SCOPE this round: expert training loops (early-stop removal, epoch/checkpoint policy), the 4-expert set, routing mechanism analysis, the OOP refactor, and any docs/markdown update.

## Acceptance criteria
1. data/processed/lt_ir100_train_indices.npy exists with exactly 10,847 int64 indices, sorted, unique. 2. Its per-class counts match n_i = 500 * 100^(-i/99) exactly: head 500, tail 5, IR 100, all 100 classes present. 3. No validation artifact is produced or loadable: lt_val_indices.npy does not exist and load_protocol_splits() returns no 'val' key. 4. The generator is deterministic: two runs with seed 42 give an identical index array, and that array equals the pre-existing lt_all_indices.npy content. 5. The honest-label carve (train_core/routing_dev) remains disjoint and reconstructs the canonical training set exactly. 6. The test suite runs with real passes shown.

## Failure modes
Missing or unreadable CIFAR-100 source data: raise explicitly, never substitute assumed counts. Regenerated set differs from the old lt_all_indices.npy: stop and report rather than overwrite, because that would mean the generator is non-deterministic or the formula changed. Any class ending with 0 samples: fail loudly (the LT profile must cover all 100 classes). Old artifact files: delete only after the new artifact is verified, and only the three superseded LT files. A test run that fails to execute (import error, missing dependency) counts as neither red nor green evidence.

## Priorities
Correctness and reproducibility over speed. The canonical artifact must be regenerable from a committed script alone, since data/processed/ is gitignored and deletions are not recoverable. Prefer an explicit, self-describing artifact name over backward-compatible names, even at the cost of updating call sites later.

## Non-goals
Do NOT keep or emulate any validation split. Do NOT delete legacy old-protocol artifacts (balanced_val_indices, base_train_indices, val_targets) or the stale val caches this round. Do NOT touch training scripts or implement early-stop removal yet. Do NOT tune the imbalance factor away from 0.01. Do NOT delete the DACE/phase0 diagnostic scripts.
