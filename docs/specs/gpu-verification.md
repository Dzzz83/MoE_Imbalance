# Doublecheck spec

## Goal
Make the local CUDA path verifiable and correctly documented: seeds pin cuDNN for reproducibility, a GPU verification test suite exercises the real training path for all four experts, a reportable run on the local GPU warns loudly, and the stale 'this machine has no GPU' claims are corrected.

## Scope
IN SCOPE: scripts/base_trainer.py (set_seed cuDNN pinning), tests/test_gpu.py (new verification suite), scripts/train.py (advisory local-GPU warning for reportable runs), and correcting the stale 'no local GPU / CUDA path unexercised' statements in docs/specs/*.md and doublecheck-spec.md. OUT OF SCOPE: configs/*.yaml contents, the training recipe, the data protocol, model or loss code, and any actual Kaggle execution.

## Acceptance criteria
1. set_seed also sets torch.backends.cudnn.deterministic=True and benchmark=False, asserted by a test. 2. tests/test_gpu.py verifies the CUDA path and passes on this machine: CUDA visible with a CUDA torch build, config device 'auto' resolves to cuda, identical seeds give identical initial weights, all four experts complete a training step on the GPU with finite on-device gradients, the config's real batch size (128) fits in VRAM, the NaN/Inf guard fires on the GPU path, and CPU/GPU logits agree within 1e-3. 3. The GPU tests skip cleanly and exit 0 on a CPU-only machine, so the suite stays green anywhere. 4. A reportable run (no --max-batches) that resolves to a local GPU prints a loud warning, and a dry run does not. 5. No repository document still claims this machine lacks a GPU or that the CUDA path is unexercised.

## Failure modes
No CUDA device present: GPU tests skip with a clear reason and the suite still exits 0 - they must never fail on a CPU-only machine. VRAM exhaustion: report peak allocation against total VRAM rather than dying opaquely. cuDNN pinning must not alter the CPU path or break existing tests. The local-GPU warning must be advisory only: it must never block or delay a run, since the user may legitimately want a short local run. If CPU and GPU logits disagree beyond tolerance, fail loudly rather than accepting a silently different model. A test that cannot execute counts as neither red nor green evidence.

## Priorities
Reproducibility over Kaggle speed: the accepted cost of pinning cuDNN is roughly 5-15% on the 12-run sweep, because a non-reproducible 3-seed comparison is worthless. Verification must be cheap, repeatable and self-skipping rather than exhaustive. When determinism and throughput conflict, choose determinism and record the cost.

## Non-goals
Do NOT run the 12-run training sweep locally. Do NOT change the published recipe, the configs, or the data protocol. Do NOT add mixed precision (AMP) or a separate per-device batch size. Do NOT add a standalone verification CLI. Do NOT hard-block runs (the user chose a warning). Do NOT install a different torch build, since the installed one is already CUDA-enabled.
