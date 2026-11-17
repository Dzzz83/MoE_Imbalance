# Handoff: implement OT reweighting in the shared pipeline

## Next task

Implementation is complete. The remaining operational task is to run the
synthetic transport resource check on the server, then train and evaluate there.
See [the current method guide](reweighting.md) and `configs/reweighting.json` for
the implemented behavior and all defaults. Synthesis and Ridge remain out of scope.

Implementation status (2026-11-08): the registered reweighting method now runs
baseline pretraining, deterministic feature caching with a fully frozen backbone,
fixed equal-mass class references, global softmax score learning through
log-domain Sinkhorn, and weighted classifier epochs. Configuration validation,
scalar history, class weight diagnostics, and checkpoint/evaluation integration
are implemented. The method guide is authoritative for the current recipe.

Pretraining follows the LDAM reference recipe. Classifier defaults follow the
OT paper: 40 epochs, SGD LR 0.00002, momentum 0.9, weight decay 0.0005; Nesterov
and constant LR follow the released image code. Our batch size remains 128.
Scores use persistent Adam at LR 0.001 with standard betas/epsilon and no weight
decay. One global update precedes each classifier epoch by default. Costs use
0.5 cosine distance plus 0.01 label mismatch. Sinkhorn uses epsilon 0.1 and 200
iterations, rejecting relative marginal errors above 0.001 before a score step.

Verification: the complete artificial-data CPU suite passed (179 tests). After
a numerical reporting refinement to ESS, its new regression check and the
affected integration check both passed. A synthetic 10,847-by-100 forward/backward
score update completed locally in about 2.35 seconds with 2,410 MiB peak process
RSS. Maximum relative row/column errors were about 1.03e-6 and 5.59e-7. CUDA
resource use and research performance remain unverified. No full-data training
was performed.

Historical context: the user is separately handling the authors-code
reproduction on their server. It does not block this adaptation. Full-data
experiments belong on that server; only lightweight verification runs locally.
The discussion and reproduction record below are retained for context; use the
current method guide instead of superseded proposals when running the code.

## Read first

1. The root `AGENTS.md` and applicable global instructions.
2. [README](../README.md): short project overview and commands.
3. [Method integration guide](method_integration.md): current method contract.
4. [Shared pipeline plan](../shared_pipeline_plan.md): data/model/evaluation rules.
5. [Guo paper note](guo_ot_reweighting.md): original method and adaptation context.
6. `pipeline/train.py`, `pipeline/methods/baseline.py`, `pipeline/data.py`,
   `pipeline/model.py`, `pipeline/runtime.py`, and `tests/test_pipeline.py`.
7. For source comparison, the separate reproduction checkout and its
   `REPRODUCTION.md` described below. Do not import that checkout at runtime.

The restored paper note has historical relative links to other archived notes.
Those files are not necessarily present beside it. Use this handoff and current
code for the active interface. If needed, inspect the earlier direction with
`git show archive/study-analysis-2026-11-05:reweighting.md`.

## User priorities and established project decisions

- Keep the code simple and readable. No OOF folds, sample-ID infrastructure,
  hashes, run registries, plugin frameworks, or inherited trainer hierarchy.
- Use descriptive names. The user explicitly rejected vague `resolved` names
  and asked for `default_config`, `training_config`, `method_options`, and
  `prepare_method_options` in the relevant configuration code.
- Explain one topic at a time. The user has been reading remotely on a phone;
  include code snippets when explaining code rather than relying on file links.
- Discuss and implement the adaptation in small steps. The user asked for a
  simpler explanation of backbone versus classifier before the reproduction
  detour. Components were subsequently discussed and implemented one at a time;
  the user then authorized researching and completing the remaining integration.
- One full reproduction run is sufficient for the user's initial comparison.
  Do not start a five-seed sweep or full training on the local machine.
- Keep the README short. Detailed implementation guidance belongs in `docs/`.
- CIFAR-100's 50,000 training examples are subsampled once at imbalance factor
  0.01, with preparation seed 42, to 10,847 retained examples.
- All methods use the same retained pool. Discarded examples are unavailable as
  method inputs. No additional shared validation split is required.
- The original 10,000-image test set is the final evaluation set. Repeated
  evaluation is allowed. It must not supply training or fitted reference state.
- Both research methods use plain ResNet-32. A shared trained first-stage model
  and frozen backbone are not general requirements; freezing is an accepted
  choice for this particular first reweighting implementation.
- The accepted reference direction is one mean training feature per class, with
  equal mass per class, constructed from retained examples only.
- Save the final scheduled model; evaluation is separate from method training.

## Workspace and Git state at handoff

Repository: `/home/dzzz83/Documents/code/expert_method`.
Remote: `https://github.com/Dzzz83/MoE_Imbalance.git`.
Branch: `master`, committed HEAD `04d2594`. The local branch and HEAD were
rechecked on 2026-11-07; synchronization with GitHub was not rechecked then.

There are local, uncommitted naming changes in:

- `pipeline/train.py`
- `pipeline/methods/baseline.py`
- `docs/method_integration.md`
- `shared_pipeline_plan.md`

The root `AGENTS.md` is local and intentionally ignored by Git. Its interface
description was also updated. This handoff remains an untracked local file.
Preserve these changes; do not reset to the remote's older function names.
Recheck status before editing in case the user has made further changes.

## Current method interface

Each method provides:

```python
prepare_method_options(options) -> dict
train(model, train_data, config, output_dir, device) -> list[dict]
```

`pipeline.train.METHODS` currently contains:

```python
METHODS = {
    "baseline": (baseline.train, baseline.prepare_method_options),
    "reweighting": (reweighting.train, reweighting.prepare_method_options),
}
```

The runner validates shared settings, applies method defaults, loads prepared
data, seeds randomness, creates a model on the requested device, invokes the
method, and saves `config.json`, `history.json`, and `final.pt`.

The method trains the supplied model in place. History records contain `epoch`
and mean `loss`; multistage methods also include `stage`. Additional scalar
measurements are allowed. If training a copy, restore its complete state,
including batch-normalization buffers, into the supplied model before returning.

`TrainingData` contains raw uint8 images `(N, 32, 32, 3)`, int64 labels `(N,)`,
real class counts `(100,)`, and `data_settings`. `make_dataset(data, transform)`
yields an image tensor and integer label. Custom transforms receive PIL images.

The model exposes:

```python
model(images)           # (B, 100) logits
model.backbone(images)  # (B, 64) features
model.fc(features)      # (B, 100) logits
```

The checkpoint contains `model_state_dict`, resolved `config`, real class counts
as a CPU tensor, and preparation settings. The evaluator needs no method-specific
objects or training images. Keep this property.

## Research source and original/adapted distinction

Paper: Guo et al., *Learning to Re-weight Examples with Optimal Transport for
Imbalanced Classification*, NeurIPS 2022.

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf)
- [Official code](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT)
- [Image training script](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT/blob/main/OT_train.py)

Read Sections 4-6 and Algorithm 1 before implementing the mathematics. The paper
learns example masses by matching a balanced reference distribution through OT.
The inner transport solve is entropy-regularized; the outer weight objective is
the transport cost evaluated at that solution. Differentiate through the solve.

Its general algorithm includes provisional model updates when features change;
the frozen-feature case simplifies this dependency. CIFAR experiments use
200 initial epochs and 40 second-stage epochs, with balanced meta examples.
Our retained-training feature means replace that reference population.

The released image script also contains a supervised meta-data loss, mode
changes, and best-test-checkpoint selection. Do not copy those behaviors into
our pipeline. This is an adaptation, not an exact reproduction.

## Separate reproduction setup and current status

The user first wanted to check whether the authors' code could produce a
comparable CIFAR-100 result before adapting it. Source inspection found no
Stage-1 training script or pretrained checkpoint, a placeholder checkpoint path,
and conflicts between paper settings and released defaults. Repository history,
issues, releases, pull requests, and the supplement did not supply those missing
pieces. The user authorized reconstructing pretraining and preparing one run.

- Local checkout: `/home/dzzz83/Documents/code/guo-ot-reweighting`.
- Upstream commit: `01f35a8c08610c4f09e19e346150af3a3428174e`.
- Fork: `https://github.com/Dzzz83/reweight-imbalance-classification-with-OT`.
- Branch: `codex/reproduce-cifar100-ot`.
- Pushed commit: `306602f1d6d5204025d4c0b128f2cef74bedf8f2`.
- [Reproduction notes](https://github.com/Dzzz83/reweight-imbalance-classification-with-OT/blob/codex/reproduce-cifar100-ot/REPRODUCTION.md)
  document exact behavior, assumptions, outputs, and commands.

That checkout adds `pretrain.py`, a one-run `reproduce.py`, explicit Stage-2
inputs, split checks, diagnostics, checkpoints, and focused tests. It preserves
the authors' custom ResNet-32 and released Stage-2 computations. It is separate
from this project; no reweighting implementation was added here.

Important source findings to keep distinct from our proposed adaptation:

| Item | Reproduction checkout | Proposed shared-pipeline adaptation |
|---|---|---|
| Training population | 10,626 imbalanced images after reserving 1,000 meta images | Existing 10,847 retained images, no additional meta split |
| References | Mean augmented images per meta class, then feature extraction | Mean deterministic retained-training features per class |
| Stage-1 schedule | Paper's literal LR multipliers of 0.01 at one-based epochs 160 and 180 | LDAM reference: five warm-up epochs, then multiply by 0.01 after completed epochs 160 and 180 |
| OT cost | Half cosine distance plus half mean squared one-hot-label distance; mismatch contributes 0.01 for 100 classes | Same default scales: 0.5 cosine distance plus 0.01 label mismatch; configurable |
| Weight learning | Per-batch softmax for OT, raw weights for classification; effective-number initialization | Global softmax weights initialized uniformly |
| Classification loss | Summed raw-weighted CE plus 10 times mean supervised meta CE | Mean of N times detached global weight times CE |
| Stage-2 backbone | Classifier-only optimization but train-mode BN updates remain | Frozen backbone and BN in evaluation mode |
| Checkpoint selection | Both final and best-test checkpoints saved | Final scheduled model only |

The reproduction's paper-settings recipe uses 200 CE epochs followed by 40 OT
epochs, batch size 16, classifier LR 0.00002, weight LR 0.001, epsilon 0.1, and
at most 200 Sinkhorn iterations. The released defaults instead run 90 Stage-2
epochs with weight LR 0.01. The alternate recipe is available but should not be
run automatically. The paper target at CIFAR-100 IF 100 is 51.46% error, or
48.54% accuracy averaged over five runs; one comparable run is enough for the
user's initial check. No numerical comparability threshold was agreed.

Pretraining batch size/SGD details and Nesterov use are documented reconstruction
assumptions; this is not an exact software/protocol reproduction. The released
Sinkhorn stopping threshold is preserved and marginal errors are logged. Unused
backbone gradients are disabled for memory savings while keeping BN behavior;
a CPU epoch comparison verified matching upstream model/weight updates.

Local full training was stopped during downloading, before training began, when
the user clarified that experiments belong on their server. Partial download/run
artifacts were removed. Do not restart a local experiment.

Server setup reported by the user:

- Checkout: `/mnt/hdd2/phatht/phat/test_reproduce/reweight-imbalance-classification-with-OT`.
- Python 3.13.11 in an active `.venv`, PyTorch 2.14.0+cu126, torchvision 0.29.0.
- NVIDIA driver 590.48.01; `CUDA_VISIBLE_DEVICES=1` successfully exposes an RTX
  A4000 as logical CUDA device 0. Other processes already occupy GPU memory.
- All 12 focused reproduction tests passed on the server in 14.30 seconds.
  Three warnings concerned test AST compatibility and upstream tensor copying;
  none were failures. This validates plumbing, not research performance.
- The full-run command failed because `outputs/cifar100_if100_seed42` already
  existed. The user was given a fresh path; successful training start and
  completion have not been reported. Do not claim it is running or reproduced.

Latest suggested server command, inside tmux with `.venv` active:

```bash
CUDA_VISIBLE_DEVICES=1 python -u reproduce.py --download --device cuda --seed 42 --recipe paper --output-dir outputs/cifar100_if100_seed42_run2
```

After completion, `result.json` in that output directory contains pretrained,
final, and best-test accuracy as fractions. `pretrain/` and `ot/` contain
configuration, split indices, history, and final checkpoints; `ot/` also saves
`best.pt` and `metrics.json`. Preserve existing output folders. The file
`requirements-reproduction.txt` pins the authors' legacy stack and is not the
modern environment used for these checks.

## Proposed first implementation from the previous chat

This sequence records the adaptation implemented after the component-by-component
discussion. Defaults are source-based starting points, not experimentally tuned
or verified claims of optimal performance. See the method guide for current
configuration names and execution behavior.

1. Pretrain the complete model using the accepted LDAM reference recipe: 200
   epochs, SGD, learning rate 0.1, momentum 0.9, no Nesterov, weight decay
   0.0002, batch size 128, five warm-up epochs, and 100-fold learning-rate drops
   after epochs 160 and 180. Reuse baseline training with a separate stage
   configuration.
2. Freeze the backbone and keep it in evaluation mode, including batch norm.
3. Extract raw features in retained-array order using deterministic transforms.
4. Calculate a raw mean feature for every class, then normalize each mean for OT.
   Separately normalize example features. Safeguard zero norms. Keep raw
   features for classifier training so final image inference remains consistent.
5. Give all 100 reference means target mass 1/100. Keep them fixed. References
   include their contributing examples; leave-one-out references are not planned.
6. Build cost `C[i,c] = feature_cost_scale * (1 - cosine(feature_i, mean_c)) +
   label_mismatch_cost * indicator(label_i != c)`, using accepted defaults
   `feature_cost_scale=0.5` and `label_mismatch_cost=0.01` from the released code.
7. Maintain one learnable score per retained-array position, initialized to zero.
   Global weights are `softmax(scores)`; positions stay method-local.
8. Solve OT over the entire retained pool versus the 100 references. Row masses
   are learned example weights; column masses are 1/100. Keep costs detached,
   but differentiate through log-domain Sinkhorn into the scores.
9. Minimize outer loss `sum(T * C)`, not an outer objective silently augmented
   with entropy. The entropy term belongs to the inner transport problem.
10. Before each Stage-2 classifier epoch, perform one global score update by
    default, with configurable `score_updates`. This schedule is accepted.
    Recompute and detach weights for classifier training.
11. Train only the existing `model.fc` on cached raw features for 40 epochs.
    Uniformly shuffled batches use `mean(N * weight_i * cross_entropy_i)`.
    Uniform weights must reproduce ordinary mean cross-entropy. Do not renormalize
    weights inside individual batches. Carry local positions with shuffled rows.

Accepted Stage-2 score optimizer defaults: Adam with learning rate 0.001, betas
(0.9, 0.999), numerical epsilon 1e-8, no weight decay, and no scheduler.
Accepted transport settings are entropy coefficient 0.1 and 200 Sinkhorn
iterations (superseding the earlier 100-iteration proposal). Classifier defaults:
SGD learning rate 0.00002, momentum 0.9, weight decay 0.0005, Nesterov enabled,
constant learning rate, batch size 128. This supersedes the earlier unsourced
classifier LR 0.01 and cosine proposal. Relative marginal errors must not exceed
the configurable engineering tolerance 0.001 before a score step.
Expose these as method settings and reject invalid/unknown values.

Full-population updates are an adaptation of the original stochastic algorithm.
They require about 1.08 million transport entries, but the differentiated graph
uses considerably more memory. Measure one update before a full run. Check
marginal errors rather than assuming an iteration limit guarantees convergence.
If memory requires another strategy, explain and validate its mathematical
effect instead of silently switching to batch-normalized weights.

## Implementation tasks and checks

The implementation and artificial-data checks below are complete. The final
baseline/continuation comparison remains future server experiment work.

- Add `pipeline/methods/reweighting.py`, `configs/reweighting.json`, and focused
  tests. Add its function pair to `METHODS`; preserve baseline behavior.
- Keep helpers within the method initially. Share a transport helper only when
  another implemented method actually needs the same operation.
- Add stage names to copied baseline history without mutating caller settings.
- Record weighted loss, learning rate, OT cost, marginal errors, maximum weight,
  effective sample size `1 / sum(weights**2)`, and class weight masses.
  Use a separate diagnostics file for arrays or other non-scalar summaries.
- Test feature/reference alignment, zero norms, positive normalized masses,
  both transport marginals, and shuffle-to-weight alignment.
- Verify score gradients against finite differences on a tiny float64 problem.
  Check that a score update changes weights on an asymmetric example.
- Check uniform-weight equivalence to cross-entropy; the backbone and its buffers
  must remain unchanged during Stage 2 while classifier parameters change.
- Run a tiny CPU method-to-checkpoint-to-evaluator integration check. Full
  training is not needed to establish the interface or mathematical correctness.
- Compare later against the baseline and a matched 40-epoch frozen-backbone
  classifier continuation using uniform weights. Extra training is a confounder.

## Environment and verification history

The system Python did not have PyTorch. Earlier checks used the preserved
environment interpreter:

```text
/home/dzzz83/Documents/code/expert_method-local-archive-2026-11-05/.venv/bin/python
```

This interpreter was rechecked: Python 3.14.8, PyTorch 2.14.0+cu126, torchvision
0.29.0+cu126, NumPy 2.5.2, and pytest were available. CUDA appeared unavailable
inside the sandbox but worked outside it on an RTX 3060 Laptop GPU. Do not
interpret sandbox GPU invisibility as proof the host has no GPU. Recheck the
environment when necessary; keep local checks lightweight. The README documents
normal fresh-environment setup.

Historical verification before this implementation: all 15 tests in
`tests/test_pipeline.py` passed after the naming changes. The current full-suite
and follow-up results are recorded at the top of this handoff. Separately, all 12 tests
in the reproduction checkout passed locally with CPU/CUDA and on the server as
reported above. No completed full-data training result has been reported.
The shared-pipeline method is now implemented and verified with artificial-data
CPU checks as described at the top. Do not claim accuracy improvements or a
completed reproduction from these checks.

## Suggested opening request for the new chat

Read `docs/reweighting.md` and inspect the implemented method before proposing
changes. Implementation is complete; server resource checks and full-data
experiments remain. Preserve local naming changes and existing run directories.
The authors-code reproduction remains separate. Do not introduce its meta split,
supervised meta loss, raw classification weights, or best-test selection into
this adaptation. Do not run full-data training locally.
