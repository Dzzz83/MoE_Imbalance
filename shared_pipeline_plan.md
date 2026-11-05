# Shared pipeline implementation plan

Status: proposed implementation plan. No pipeline code or experiments have been
changed. This is a new training-method comparison, separate from the existing
expert-routing studies. The reweighting decisions are in [reweighting.md](reweighting.md).

## Goal

Let two contributors implement reweighting and synthetic-sample methods
independently, then compare their complete methods using the same retained
dataset, ResNet-32 architecture, and evaluation procedure.

Share data preparation, model creation, run conventions, and evaluation. Each
method owns its training algorithm. Shared architecture does not require the
same loss, optimizer, number of stages, frozen parameters, or synthetic-sample
representation. These differences must be recorded in the comparison.

## Architecture

```text
Shared preparation -> saved retained-data split
                              |
Shared model factory -> fresh ResNet-32 for each run
                              |
                   +----------+-----------+
                   |          |           |
                Ordinary     OT         Synthetic
                training  reweighting    samples
                   |          |           |
                   +----------+-----------+
                              |
                Standard model checkpoint
                              |
               Shared evaluation and comparison
```

The ordinary-training branch establishes a working baseline and demonstrates
the integration interface before either research method is implemented.

## Shared data preparation

1. Use the same retained CIFAR-100-LT population for all methods. Start from the
   existing retained index list; do not silently regenerate a different subset
   or apply long-tail subsampling twice. Creating a different population would
   be a separate, explicitly configured experiment.
2. After retention, exclude discarded images from method inputs. Materialize
   only permitted images, labels, and original sample IDs into prepared files.
   Trainers consume these files rather than constructing a full CIFAR dataset.
3. Make one fixed development split within the retained population if needed
   for tuning. Save membership once; both contributors use the same files.
   Stage-1 training, prototypes, feature synthesis, and any fitted transforms
   use only the training portion. Development data scores choices; it does not
   become an extra reference set for one branch.
4. Provide shared default training transforms and deterministic development /
   evaluation transforms. Method-specific augmentation remains inside its
   training branch and is recorded in its configuration.
5. Save class names, real training counts, split settings, and membership IDs.
   Synthetic examples do not change the real counts used to define class groups.

**Data-role decision before real preparation:** a simple candidate is one
stratified training/development split of retained data, without OOF. The exact
fraction and final evaluation population remain undecided. Very small classes
must retain training examples and have explicit evaluation coverage rules.

The original CIFAR test set has historical exposure; it cannot be described as
untouched. It must not be used for development. Before execution, add the new
study's agreed roles to [docs/protocol.md](docs/protocol.md) without changing the
frozen old studies. Do not load any final evaluation data during this planning
or shared-component verification work.

### Data interface

- Dataset item: `(image, label, sample_id)`.
- Transformed image: floating tensor shaped `(3, 32, 32)`.
- Label: integer class index in the shared class order.
- Sample ID: original CIFAR training index, stable across shuffling and views.
- Training and development data are supplied separately. Training plugins do
  not receive a final-evaluation loader.
- Synthetic features and their labels are internal method state; do not insert
  synthetic rows into the shared original-image dataset or reuse original IDs.

## Shared model creation

Reuse the plain `ResNet32` from `models/resnet32.py`. Its existing interface is
already sufficient:

```text
model(images)         -> logits, shape (batch, 100)
model.backbone(images)-> features, shape (batch, 64)
model.fc(features)    -> logits, shape (batch, 100)
```

For each matched run seed, initialize a fresh model identically before handing
it to the selected method. Isolate model initialization randomness from data
loading and method randomness. Do not reuse a mutable model across methods.

Sharing a trained Stage-1 checkpoint is optional, allowed only if both methods
declare the same Stage-1 training recipe and data. A fresh architecture is the
required shared component. Freezing and feature caching are method decisions.

## Method integration interface

Use a small explicit function interface, not a framework of inherited trainers:

```text
train_method(model, train_data, dev_data, config, run_dir)
    -> final_checkpoint_path
```

The runner creates the data objects, model, seed settings, and run directory.
The method creates its loaders as needed, owns all training stages, and uses
shared checkpoint/logging helpers. Shared development scoring uses the same
metric definitions as final evaluation, with no final-evaluation data access.

The checkpoint must contain a complete plain ResNet-32 `state_dict`, the method
name, seed, and resolved configuration. Evaluating it must not require Ridge,
OT, reference prototypes, an SVM, or synthetic training features.

This interface is the target for the planned ResNet-32 adaptation of OTOS, not
an interface for reproducing the original binary SVM experiment. The synthetic
branch must still define its multiclass synthesis and SVM-guidance adaptation
before its algorithm is implemented. That does not block shared components.

### Ownership boundaries

| Area | Owner | Responsibility |
|---|---|---|
| Shared components | One designated integrator | Data preparation, model factory, runner, evaluator, integration example |
| Reweighting branch | Contributor A | Prototypes, OT cost, weight optimization, weighted training, later Ridge extension |
| Synthetic branch | Contributor B | Multiclass adaptation, generation guidance, OT synthesis, classifier training, later Ridge extension |

Names A/B do not assign a particular branch to either teammate yet. Both
contributors implement against the ordinary-training example. Method changes
stay in their own modules and configs; proposed interface changes go through
the integrator before either branch depends on them.

## Shared evaluator and fair comparison

- Restore a fresh plain model from its checkpoint and run image inference in
  evaluation mode with deterministic preprocessing and no gradient updates.
- Calculate sample accuracy, per-class recall, balanced accuracy, and macro
  Head/Medium/Tail recall. Use `compute_class_groups` for group membership and
  average the per-class recalls within each group.
- Require a declared policy for missing evaluation classes. Do not silently
  report an all-class metric when some classes have no evaluation examples.
- Use one common checkpoint-selection rule. A proposed simple default is the
  final checkpoint after a declared schedule. If development-based selection
  is chosen instead, declare one shared score and tie-break rule first.
- Keep split membership fixed across training seeds. Pair method comparisons
  by seed and initialization; report per-seed results and aggregate variation.
- Give methods comparable tuning budgets. Record stage epochs, optimizer
  updates, real and synthetic examples processed, and total training time.
  Equal architecture or equal epochs alone does not imply equal compute.
- Final results do not feed back into references, generation, hyperparameters,
  or checkpoint selection. Use the existing test-access entry point if the
  agreed final population requires access to the original CIFAR test set.

The comparison first answers which complete training method works better.
Additional controls are needed to attribute gains to OT, synthesis, or Ridge.
Ridge stays outside the first shared-pipeline implementation.

## Minimal file layout and outputs

Proposed new package, keeping current code available until archival is agreed:

```text
rebalancing/
  data.py                 # preparation, prepared datasets, transforms
  model.py                # plain ResNet-32 factory
  runtime.py              # seeding, checkpoint and run-record helpers
  train.py                # selects method and invokes its training function
  evaluate.py             # predictions, metrics, comparison report
  methods/
    baseline.py
    reweighting.py
    synthesis.py
configs/rebalancing/
tests/rebalancing/
```

Keep imports from the old project limited to reviewed reusable pieces. No OOF
pipeline, expert router, artifact-hash system, or automatic experiment matrix
is needed in this new path. Keep ordinary configuration and membership records
so teammates can reproduce a run without rebuilding the old infrastructure.

Prepared data: training/development image arrays, labels and sample IDs, plus
one split-description JSON file. Each run saves a resolved config, training
history, final model checkpoint, and run summary. Final evaluation adds
predictions with sample IDs, metric JSON, and a comparison table. Fail on an
existing run directory unless explicitly resuming a compatible run; never
silently overwrite another contributor's result.

## Implementation sequence

1. **Agree the contract and data roles.** Choose split settings, evaluation
   population, checkpoint policy, and matched seeds; update the protocol for
   this separate study. Preserve the current repository and uncommitted work.
2. **Build preparation and model factory.** Reuse retained IDs and plain
   ResNet-32; add the explicit image/label/ID contract and deterministic views.
3. **Build the ordinary-training example and run helpers.** Establish the
   training function, checkpoint format, and resolved config before delegation.
4. **Build the evaluator.** Verify metrics and checkpoint restoration before
   either branch produces results. Supply a concise contributor guide with one
   local synthetic smoke example and required input/output shapes.
5. **Verify integration.** Both contributors run the same ordinary-training
   example, load its output with the evaluator, and confirm interface agreement.
6. **Implement branches independently.** Each owns its algorithm, settings, and
   focused tests. First integrate OT-only methods; decide Ridge targets later.

Archive/branch migration is a separate setup action before replacing current
entry points. An archive branch alone will not preserve uncommitted changes;
those need a deliberate preserved snapshot. This plan does not perform Git
operations, move files, or alter concurrent work.

## Verification before method handoff

- Data: no duplicate IDs, disjoint roles, no discarded examples, expected
  counts, stable IDs under shuffling, and identical preparation for both methods.
- Model: expected feature/logit shapes, matching initialization for matched
  seeds, and finite CPU forward/backward results.
- Metrics: hand-calculated examples with unequal class counts to distinguish
  macro from sample accuracy, plus declared missing-class behavior.
- Boundary: trainers cannot obtain final evaluation data through their inputs;
  creating the runner/model does not load it implicitly.
- Integration: a tiny CPU synthetic baseline run saves a checkpoint, restores
  it into a fresh model, and reproduces predictions through the evaluator.

No full training is needed for the shared-component handoff. Later real-data
integration and canonical training follow the project's verification levels;
smoke checks establish integration, not model quality.

## Reuse findings from the current workspace

- `models/resnet32.py`: plain ResNet-32 has separate backbone and linear head.
- `utils/create_lt_split.py`: reusable retention logic, but its existing command
  describes the old full-data protocol; do not run it blindly for a new split.
- `data/cifar_lt.py`: useful transform constants, but its full-pool fallback and
  optional test loading should not become method-facing behavior.
- `data/lt_datamodule.py`: explicitly has no development split, so it is not a
  drop-in implementation of the proposed data roles.
- `scripts/base_trainer.py`: reuse canonical class-group membership logic.
  Existing `group_accuracies` computes sample accuracy within a group, so it
  must not be used unchanged for the required macro group recall. Do not copy
  that behavior into the new evaluator or fix unrelated historical paths here.

These are code-inspection findings, not tests of the proposed pipeline. No
data, test evaluation, or ML training was run while preparing this plan.
