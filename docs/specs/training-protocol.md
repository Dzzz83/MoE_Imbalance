# Doublecheck spec

## Goal
Rewrite the expert training loop so CE/LA/BS/Mixup train on the full 10,847-sample long-tailed training set under the published CIFAR-LT recipe (200 epochs, lr 0.1, SGD, warmup 5, x0.01 decay at 160 and x1e-4 at 180), with no validation split, no early stopping, and the last epoch plus every 20th epoch from 160 saved per seed.

## Scope
IN SCOPE: scripts/base_trainer.py (schedule, loop, checkpoint policy, guards), scripts/train_ce.py, scripts/train_lal.py, scripts/train_balanced_softmax.py, scripts/train_mixup.py (data source, seed, no val loader). OUT OF SCOPE: evaluation, routing mechanisms, docs updates, the OOP refactor, PaCo, and any Kaggle execution.

## Acceptance criteria
1. LR follows the reference CIFAR-LT schedule exactly: base_lr for epochs 1-160, x0.01 for 161-180, x0.0001 for 181-200, linear warmup over the first 5 epochs. Verified by asserting lr at epochs 1, 5, 6, 160, 161, 180, 181, 200. 2. No validation set exists anywhere in the training path: the training entry point accepts only a train loader, nothing computes val_ba, and no checkpoint is selected by validation. 3. Checkpoints are written every 20 epochs from epoch 160 and at the final epoch, named with the expert name, seed, and epoch so three seeds cannot clobber each other. 4. The final-epoch checkpoint is the reported model. 5. Optimiser matches the reference: SGD momentum 0.9, weight_decay 2e-4, nesterov=False. 6. NaN/Inf in logits or loss raises immediately. 7. A CPU synthetic dry-run for each of CE/LA/BS/Mixup completes 1 forward + 1 backward on a tiny dummy batch and populates .grad for every trainable parameter. 8. Seeds are applied via torch.manual_seed and numpy.random.seed.

## Failure modes
Non-finite loss or logits: raise rather than continuing, because a silently diverged 2-hour GPU run wastes the whole budget. Missing canonical artifact: raise the existing ProtocolError from the protocol module, never fall back to an old filename. OOM or crash mid-run: the last completed checkpoint must remain loadable and self-describing (expert, seed, epoch). A test that cannot execute (import error, missing dependency) counts as neither red nor green evidence. If the LA/BS duplication turns out to make one of them unusable, report it rather than silently dropping or silently keeping a redundant expert.

## Priorities
Fidelity to the published recipe over convenience, and fail-fast over silent divergence. A run that cannot be reproduced from a seed is worthless, so seeding is mandatory. Diagnostic checkpoints matter only for inspection; correctness of the final-epoch model is what counts. When fidelity and convenience conflict, choose fidelity and record the deviation.

## Non-goals
Do NOT reintroduce any validation set or validation-based selection. Do NOT select checkpoints on test accuracy. Do NOT train yet - this round only makes the code correct and dry-run verified. Do NOT add PaCo back. Do NOT change the loss formulas (tau=1.0 for LA, alpha=1.0 for Mixup) or the 0.01 imbalance factor. Do NOT refactor the whole project to OOP yet; only the training path.
