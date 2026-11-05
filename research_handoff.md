# Handoff: shared pipeline for OT reweighting and synthetic samples

Workspace: `/home/dzzz83/Documents/code/expert_method`
Current branch at handoff: `dev/study-analysis` (not `main`).

## Where the conversation stands

The user wants to investigate Sinkhorn + Ridge for reducing the effect of
class imbalance and improving classification on a fixed retained dataset.
They have two directions: training-example reweighting and synthetic-sample
creation. One method will be implemented by a teammate.

The current task is to establish a simple shared foundation so each contributor
can implement their own training method and plug it into the same experiment.
Planning is complete enough to discuss the remaining shared decisions and begin
implementation. Neither research method nor the new pipeline has been built.

## Read these files first

1. [shared_pipeline_plan.md](shared_pipeline_plan.md): proposed architecture,
   interfaces, ownership, implementation order, and verification.
2. [reweighting.md](reweighting.md): agreed reweighting direction and open details.
3. [guo_ot_reweighting.md](guo_ot_reweighting.md) and
   [otos_synthetic_samples.md](otos_synthetic_samples.md): original-paper reading
   notes, reported results, code availability, and adaptation limitations.
4. Applicable `AGENTS.md` instructions. Read `docs/protocol.md` before changing
   experiments or data roles; use `docs/research.md` before proposing mechanisms.

These new documents are plans, not executable configuration or a replacement
for the current protocol. The user approved the shared-pipeline concept; exact
split settings, method details, and all proposed defaults remain unconfirmed.

## User priorities and corrections to preserve

- Explain one decision at a time in plain language. Use small examples.
- Avoid more literature lists or repeatedly changing recommendations. The user
  was frustrated by research that did not lead to an implementable plan.
- Create imbalance first. Only retained examples may supply learning or method
  inputs afterward; discarded images must remain unavailable.
- Shared preparation, model creation, and evaluation are for fair comparison
  and easy teammate integration. They do not require identical training loops.
- Both methods use the same ResNet-32 architecture. A shared trained Stage-1
  checkpoint and a frozen feature extractor are optional, not agreed mandates.
- The user wants a simpler pipeline without the existing OOF and artifact-hash
  infrastructure. Basic split membership, configs, and run records are still
  needed. Do not reintroduce the old routing study as the new experiment.
- The user previously requested eventual archival of the old main branch and
  selective reuse. No archival, branch switch, deletion, or code migration has
  occurred. Preserve uncommitted work before any future migration.

## Agreed reweighting starting point

Use Guo et al., *Learning to Re-weight Examples with Optimal Transport for
Imbalanced Classification*, NeurIPS 2022.

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf)
- [Official code](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT)

The paper includes ResNet-32/CIFAR-100-LT and reports a 200-epoch first stage
followed by a 40-epoch second stage. These are paper settings, not locked new
experiment settings. Stage 1 trains normally. Stage 2 learns example weights
with OT and trains with weighted losses. The authors' image implementation
optimizes only the final classifier parameters in Stage 2.

The user's accepted initial reference design is one mean feature vector per
class, calculated from retained training examples, with equal reference mass
per class. This differs from the paper's balanced meta set of ten images per
class taken from the original training pool. Describe our version as an
adaptation, not an exact reproduction.

Feature/label costs, normalization, persistent weights, batch-normalization
behavior, fixed versus refreshed prototypes, and self-exclusion remain open.
Sinkhorn calculates transport for current masses; a separate update learns
example weights. Class counts do not change. Ridge's inputs, target, and place
in the method remain undecided. No improvement or novelty has been established.

## Synthetic direction: candidate and limitations

The narrowed reading list uses Yan et al., *Oversampling for Imbalanced Data
via Optimal Transport* (OTOS), AAAI 2019.

- [Paper](https://tanmingkui.github.io/files/publications/Oversampling.pdf)
- [Author publication page](https://tanmingkui.github.io/publication/Oversampling)

OTOS creates synthetic vectors using transport-guided combinations of minority
examples. Its original formulation is binary and uses an initial SVM's loss
information to guide synthesis, then trains an SVM with real and synthetic
vectors. The paper reports six benchmark and four medical-image datasets.
Medical images use ImageNet-pretrained ResNet-152 features; there is no reported
ResNet-32/CIFAR-100 experiment.

A verified public author repository was not found in the paper, author page,
or targeted GitHub searches. This is not proof that none exists. Consequently,
OTOS is an algorithm reference, not a verified ready-to-run CIFAR-100 baseline.
Adapting its binary formulation and SVM guidance to 100 classes and a ResNet
classifier remains necessary. The user has not finalized those decisions.

Do not imply that the synthetic branch is already specified or that its
adaptation can reproduce the paper's original results.

### Papers set aside

- **OTmix:** released official `cifar_train.py` was inspected. It builds its
  validation loader from `CIFAR100(train=False)`, returns a label-based confusion
  matrix, feeds it into later OTmix pairing, and selects checkpoints on the same
  split. This is test leakage in that code path. We did not run it or establish
  which exact code generated every paper result. Do not claim deliberate
  cheating. The paper also describes training-derived confusion variants, so
  training-only confusion is not a newly invented contribution.
- **GMDC (2026):** publisher previews confirm OT-based synthetic features and
  CIFAR100-LT experiments. Backbone, numerical results, solver, code availability,
  and evaluation hygiene could not be verified from accessible material. It is
  not a verified reproduction candidate. No new recommendation was made.

Neither a leakage problem nor missing code erases prior art. Adding Ridge alone
does not establish novelty. The agreed sequence is OT-only baselines first,
then decide a meaningful Ridge target and test its added value.

## Shared architecture and teammate contract

```text
Shared retained-data preparation
               |
Shared ResNet-32 creation, matched initialization per seed
               |
      +--------+-----------+
      |        |           |
  Ordinary  Reweighting  Synthetic samples
  baseline  contributor  contributor
      |        |           |
      +--------+-----------+
               |
Common model checkpoint -> shared evaluator and comparison
```

Proposed method function:

```text
train_method(model, train_data, dev_data, config, run_dir)
    -> final_checkpoint_path
```

Prepared image items include image, label, and stable original sample ID.
Training methods do not receive final-evaluation data. Each method owns its
training stages, losses, weights, synthetic features, and later Ridge logic.
Both output a complete plain ResNet-32 checkpoint for ordinary image inference.

One designated integrator builds shared components and a working ordinary
training example first. Contributors then work in separate method modules and
configs against that contract. No actual person-to-method assignment has been
made. Full proposed layout and acceptance checks are in the shared plan.

## First decisions in the new chat

Start with the data roles, one decision at a time:

1. Choose development membership/fraction and final evaluation population. A
   single fixed stratified split within retained data is proposed; no OOF is
   required. Do not silently assume discarded images can supply validation.
2. Agree checkpoint selection and matched seeds. Final-epoch selection is a
   proposed simple default, not yet locked.
3. Record the new study's protocol separately from frozen historical studies,
   then build preparation, model factory, ordinary training example, evaluator,
   and contributor guide. Synthetic algorithm design can continue separately.

The original balanced CIFAR test set has historical exposure and is not an
untouched confirmation set. Existing outer fold 0 is also consumed. Do not use
either to tune this new method. Consult the current protocol before choosing
new data roles; do not silently inherit its expensive router OOF workflow.

## Reviewed code worth reusing selectively

- `models/resnet32.py`: plain `ResNet32` exposes `.backbone` (64-dimensional
  features), `.fc` (100-class linear classifier), and image-to-logit `forward`.
- `data/processed/lt_ir100_train_indices.npy`: existing retained population,
  documented as 10,847 images at IR100. Not loaded in this chat.
- `utils/create_lt_split.py`: retention logic, but its CLI describes the old
  full-data protocol; avoid silently replacing the existing population.
- `data/cifar_lt.py`: transform constants; full-pool fallback and optional test
  loading should not enter the new method-facing data interface.
- `data/lt_datamodule.py`: current implementation has no development split.
- `scripts/base_trainer.py`: canonical `compute_class_groups` function. Its
  `group_accuracies` computes sample accuracy within groups, not macro recall;
  the new evaluator must average per-class recalls instead. Do not duplicate
  thresholds or change unrelated historical evaluators during this task.

## Workspace state and verification

Created in this chat, still untracked at handoff:

- `reweighting.md`
- `shared_pipeline_plan.md`
- `guo_ot_reweighting.md`
- `otos_synthetic_samples.md`
- `research_handoff.md` (this file)

Pre-existing/concurrent changes were not edited:

```text
 M scripts/README.md
 M scripts/base_trainer.py
 M scripts/train.py
 M tests/unit/test_training_config.py
?? tests/regression/test_training_resume.py
?? tests/regression/test_training_smoke.py
```

The planning documents and paper notes received focused same-agent self-review.
This handoff was checked against them and the conversation. No new pipeline code,
tests, training runs, dataset reads, test evaluation, commits, or branch
operations were performed. Documentation whitespace checks reported no issues;
implementation and performance remain unverified.
