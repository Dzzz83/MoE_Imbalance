# Doublecheck spec

## Goal
With the local GPU driver recovered, the CUDA path is verified end-to-end (CUDA suite, one full-epoch training run on the real long-tailed set, and one logged test-set evaluation on GPU), and docs/bugfix-report.md records the measured GPU results and whether they reproduce the published numbers.

## Scope
IN: run tests/test_gpu.py on the recovered GPU; one non-reportable full-epoch training run of one expert over the real 10,847-sample set on CUDA; one logged `scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda --tta-augs 10` read of the frozen test set; compare its per-seed and 3-seed aggregates against docs/results.md; update docs/bugfix-report.md (GPU sections, test-suite table, access-log note). OUT: editing any test expectation, seed or threshold; retraining or replacing the 12 checkpoints in checkpoints/; running scripts/analyze_subsets.py (a separate logged read, not requested); committing docs/ (gitignored); a full 200-epoch local training run.

## Acceptance criteria
1) tests/test_gpu.py reports 13 passed, 0 skipped on the local RTX 3060, including CPU/GPU agreement. 2) The full suite of 17 files still exits 0, with no test changed to get there. 3) A 1-epoch CUDA run over all 10,847 samples completes with finite loss and gradients and writes its checkpoint outside the repository. 4) evaluate_experts.py completes on GPU and its output reproduces the published values (Uniform BA 46.98 / Tail 18.76; Probability 45.95; Confidence 44.27; TTA 44.15 / Tail 19.38; per-expert BA 42.43 / 41.34 / 38.69 / 37.69; all-wrong 39.73; oracle 60.27) to the same rounding, with any delta reported explicitly. 5) docs/test-access-log.md gains exactly one new appended row for this read. 6) docs/bugfix-report.md states the measured GPU results, the reproduction verdict, and no longer claims the GPU path is unverified.

## Failure modes
If the driver wedges or a CUDA check fails: capture the evidence, change no test, seed or threshold, and report as-is. If the re-run numbers differ from the published ones: report the exact deltas and treat them as a finding; never adjust code, thresholds or presentation to make them match. If the GPU is busy or VRAM is short (desktop already holds ~1.2 GB of 6.1 GB): report it rather than silently changing batch size; a --batch-size deviation may only be used as an explicitly disclosed last resort. If some seeds/experts lack checkpoints: report which are missing instead of substituting runs. If the evaluation fails partway: the access-log entry stays (append-only) and the partial result is reported as partial.

## Priorities
Protocol integrity first (exactly one logged read, append-only log, pre-registered rules unchanged), then reproducing the published numbers, then speed. Breadth is optional: the subset curve can be dropped without affecting this contract. Local GPU time is verification budget, not a training resource.

## Non-goals
Do not retrain or overwrite the reported experts; do not run analyze_subsets.py; do not modify any test file; do not edit README.md or any tracked file for this task; do not launch a full local training sweep; do not "improve" the routing rules or the evaluation harness while verifying them.
