# Doublecheck spec

## Goal
Replace the val-dependent router framework with a parameter-free-only one, and prepare the OOP evaluation code needed to verify the Kaggle runs and measure routing headroom without touching the test set before pre-registration.

## Scope
IN SCOPE: scripts/router/ (refactor interface, delete the five fitted routers, de-calibrate ConfidenceRouter), the val-dependent phase0*/diagnose*/benchmark callers, new OOP modules for run-health checking, test-set expert evaluation with an access log, and routing headroom analysis, plus docs/routing-preregistration.md and docs/routing-results-record.md. OUT OF SCOPE: training code, configs, data protocol, models, losses, and any actual test-set evaluation run.

## Acceptance criteria
1. The router interface exposes no fitting method: BaseRouter has no train(), and no file under scripts/router/ references val_labels or val_logits (enforced by a static test). 2. The registry contains exactly Uniform, Product, Confidence and TTA; the five fitted routers are gone from disk. 3. ConfidenceRouter has no calibration path (no scipy temperature fit, no temperatures attribute). 4. Every registered router produces predictions from logits alone, verified on synthetic logits with no fitting step. 5. A run-health checker reports, per checkpoint, whether the loss descended, the LR hit the 160/180 milestones, the final checkpoint exists and no non-finite loss occurred. 6. A test-set evaluation harness reports per-expert BA/Head/Med/Tail/ECE and writes an access-log entry (command, timestamp, git hash) on every read. 7. A headroom analysis reports the per-pool all-wrong floor, oracle BA and pairwise Cohen's kappa. 8. docs/routing-preregistration.md exists and freezes the candidate rules, metrics and decision rule. 9. docs/routing-results-record.md records the deleted mechanisms and their measured results. 10. The full suite runs green.

## Failure modes
Deleting a mechanism whose results are not yet recorded: consolidate the results into the markdown record BEFORE removing the code. A routing module still imported elsewhere after deletion breaks evaluation: update every importer and prove it with an import smoke test. The test-access log must append, never truncate, and must not fail a run if the log is unwritable. The harness must never write test-set predictions into a checkpoint directory that training reads. A test that cannot execute counts as neither red nor green evidence.

## Priorities
Honesty of the protocol over method breadth: a smaller set of provably parameter-free rules beats a larger set that cannot be justified without held-out labels. Structural prevention over documentation: the interface and static checks must make val-fitting impossible, not merely discouraged. Cheap, repeatable verification over exhaustive tooling.

## Non-goals
Do NOT fit or tune anything on the test set. Do NOT evaluate the test set now - only prepare the code. Do NOT add new routing mechanisms. Do NOT change the training recipe, configs or data protocol. Do NOT delete the parameter-free routers or the DACE documentation. Do NOT retrain anything.
