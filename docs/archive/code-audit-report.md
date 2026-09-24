# Verified Code Audit Report

Status: consolidated review of the three audit passes, 2026-09-23. The
refactor sequencing and ownership plan is in
[refactor-plan.md](refactor-plan.md); this report links back to that plan.

Phase 1 correctness hardening is complete in the current worktree. OOF
extraction, analysis-foundation completion and legacy quarantine remain planned
and are not represented as completed fixes below.

This inventory includes verified defects only. “Fixed in current worktree”
means the implementation and a focused regression test or direct static check
are present in the shared worktree. “In progress” means a partial foundation or
an implementation awaiting the owning slice. “Planned” means intentionally not
changed yet. No row below claims a training or evaluation rerun.

## Result-status boundary

No canonical current OOF or full-data BA/Tail number was recomputed or changed
by these audit slices. The OOF fixes tighten acceptance of manifests and
prediction artifacts; they do not rewrite a valid saved artifact. The current
worktree's software verification status is recorded below; these checks do not
establish a scientific metric.

The historical TTA row, including its BA and Tail values, is invalid pending an
explicitly authorized reevaluation. The historical routing calibration fields
for Uniform, Confidence and TTA are invalid. Probability calibration and
expert calibration are unaffected by the router-probability bug. No replacement
TTA or calibration values are reported here.

## Defect inventory

| ID | Verified defect and evidence | Current status | Affected outputs | Canonical current OOF/full-data BA or Tail result |
|:--|:--|:--|:--|:--|
| A1 | **TTA padding, device and crop semantics.** `data/tta.py:55-111` previously padded already-normalized tensors with zero, sampled one crop position for the whole batch, and could mix generator/index devices. Training uses black PIL padding and independent per-sample crops. | **Fixed in current worktree.** `AugmentationViews` now uses normalized black padding, per-sample coordinates, seeded CPU draws and device-local tensors; focused tests were added. Reevaluation is planned only after explicit authorization. | TTA logits, TTA predictions and TTA-derived calibration are affected. The historical TTA row (including BA/Tail) must not be used. | OOF and non-TTA expert/uniform results are not changed by this code. The old full-data TTA BA/Tail is invalid; no new canonical value exists. |
| A2 | **Router probability/ECE classifier mismatch.** `scripts/evaluate_experts.py:167` used `predict_proba` routing weights as class probabilities, although `predict_class` for Uniform, Probability and TTA can be a different classifier. The router contract is now split in `scripts/router/base.py:77-108`. | **Fixed in current worktree.** `predict_distribution` is implemented and the evaluator uses it; router-contract regressions cover argmax agreement. Historical calibration fields still need an authorized recomputation. | Routing ECE/confidence fields for Uniform, Confidence and TTA are invalid. Probability calibration and per-expert calibration are unaffected. BA/Tail predictions are not the affected field. | No canonical OOF/full-data BA or Tail change; no calibration rerun is claimed. |
| A3 | **Stale DACE `BaseTrainer` API.** Current `scripts/base_trainer.py:435` accepts `train(train_loader)` only, while legacy DACE A/B/C callers still use `train_loader, val_loader, class_counts` (for example `scripts/train_dace_a.py:81-89` and `scripts/train_dace_c.py:203-210`) and reference removed validation-era behavior. | **Planned / legacy quarantine.** The commands are not repointed to the active no-validation protocol because that would create a new experiment. | DACE command execution and any DACE-specific historical checkpoint/report are affected. | DACE is outside the active expert pool and current OOF matrix; no canonical current OOF/full-data BA or Tail result changes. |
| A4 | **Unconsumed `TestAccessGrant`.** Protected legacy readers requested authorization and discarded the returned one-shot capability. `scripts/utils/data.py:251-259` and `scripts/evaluate_dace.py:372-376` now consume it. | **Fixed in current worktree.** A synthetic access-log test verifies consumption. | Test-set audit integrity and fail-closed authorization are affected; no numeric metric is directly changed. | No canonical result change and no test-set rerun. |
| A5 | **Energy-score overflow.** `scripts/utils/features.py:193-213` previously evaluated `exp(logits / temperature)` directly, so large finite logits could produce `inf`. | **Fixed in current worktree.** Shifted log-sum-exp plus finite-input and positive-temperature checks are covered by focused tests. | Energy features and any downstream diagnostic score can change from `inf`/invalid to the mathematically correct finite value on extreme inputs. | No canonical OOF/full-data BA or Tail result was recomputed or changed. |
| A6 | **PaCo queue overflow/device handling.** `models/resnet32.py:_dequeue_and_enqueue` previously assumed a batch smaller than the queue and did not robustly handle device/shape combinations. | **Fixed in current worktree.** Circular insertion validates shapes, handles empty and batch-at-least-queue cases, and aligns devices. PaCo remains a legacy line. | PaCo training can fail or corrupt queue state for oversized batches; affected outputs are PaCo checkpoints and historical diagnostics only. | PaCo is not in the current four-expert pool; no canonical current OOF/full-data BA or Tail result changes. |
| A7 | **OOF count, router-role and extra-record validation.** `data/nested_oof.py:193-347,606-779,1063-1158` now recomputes outer/inner/router class counts from canonical labels, derives router fit/selection sample tuples from declared inner-fold IDs, and rejects inconsistent metadata. `:1544-1634` now compares the complete key set, including extras. | **Fixed in current worktree.** `FoldIntegrityValidator` and minimized regressions cover stale counts, swapped router memberships and valid outer rows smuggled into a complete inner artifact. | Malformed/tampered manifests and OOF artifacts are rejected instead of being analyzed; valid artifacts are not rewritten. | No current OOF BA/Tail number changes; no OOF metrics rerun. A saved artifact that fails the stricter contract must be quarantined, not silently repaired. |
| A8 | **Non-atomic OOF artifact writes.** `scripts/oof_pipeline.py:416` originally wrote initial metadata directly; `:816-834` originally updated metadata with non-atomic `Path.write_text`, so an interruption could leave a truncated mutable run record. | **Fixed in current worktree.** `AtomicMetadataWriter` stages `run_metadata.json` in the destination directory, flushes and fsyncs it, then calls `os.replace`; immutable manifest/config/prediction writes delegate to `ImmutableArtifactWriter`. Focused failure and successful-transition regressions are present. | OOF run metadata, resumability and recovery provenance are protected from partial replacement. Execution-log append semantics were intentionally unchanged. | No canonical current OOF/full-data BA or Tail result changes; no OOF metric rerun is claimed. |
| A9 | **Task 3F missing-class weight report.** `scripts/task3f_ridge.py:281-319` previously used `bincount` without rejecting absent classes up to the maximum label, allowing an incomplete report to look valid. | **Fixed in current worktree.** Labels, weights, positivity, alignment and missing classes are validated before serialization. | Malformed Task 3F weight summaries are rejected. Existing valid summaries are not rewritten. | No canonical current OOF/full-data BA or Tail result was recomputed or changed. |
| A10 | **Fractional/out-of-range predictions.** Task 3F diagnostic paths previously cast arbitrary numeric predictions to integer IDs. `scripts/task3f_mixup_diagnostics.py:79-107`, `scripts/task3f_tail_signal_diagnostics.py:85-107` and the combined diagnostics now check integer-valued and in-range predictions before casting. | **Fixed in current worktree.** Focused regression tests reject fractional and out-of-range IDs. | Invalid diagnostic inputs no longer silently change class membership, gain/loss counts or Head/Medium/Tail reports. | No canonical current OOF/full-data BA or Tail result changes; no diagnostic rerun is claimed. |
| A11 | **NaN oracle tolerances.** `scripts/task3e_soft.py:192-220` previously tested only `tolerance <= 0`; `NaN` bypassed that comparison and could poison LP feasibility/verification decisions. | **Fixed in current worktree.** All margin, feasibility and verification tolerances must be finite positive numbers; regressions cover NaN. | Task 3E-B may now fail closed on invalid configuration instead of producing an unreliable status or report. | Existing canonical OOF results are not recomputed or relabeled; no BA/Tail change is claimed. |
| A12 | **Pytest collection warnings.** Public utility classes named `TestAccessError`, `TestAccessGrant` and `TestAccessLog` can match pytest's test-class discovery convention even though they are not tests. `scripts/utils/test_access.py:20-54` now sets `__test__ = False` on those classes. | **Fixed in current worktree.** Collection intent is explicit; parent integration reports no collection warnings after the fix. | Test discovery noise and possible collection warnings; no runtime or scientific output. | No canonical OOF/full-data BA or Tail result changes. |

## Verification evidence and limitations

Focused OOF verification in the shared worktree included the minimized
regressions for derived counts, router-role membership and extra outer records;
the targeted nested-OOF subset passed (`12 passed, 3 deselected`). The focused
OOF suites (`tests/test_oof_pipeline.py` and `tests/test_task3c_oof.py`) passed
**43 tests in 9.63s**. Parent integration evidence is **380 passed with 4
collection warnings** initially; final verification passed `compileall`, and
the full `pytest` suite completed with **410 passed in 26.39s with no warnings**.
Static compilation of touched Python files and `git diff --check` passed. No
dataset, checkpoint training, balanced test-set read or metric rerun was
performed for this audit.

A full canonical OOF or test evaluation requires separate authorization and
must report fresh provenance rather than overwrite the historical TTA/calibration
fields.

For migration order, ownership, legacy quarantine and verification gates, see
[refactor-plan.md](refactor-plan.md).
