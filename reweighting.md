# OT reweighting: agreed direction

Status: research and design decisions from our discussion. This method has not
been implemented or evaluated. This document does not change the existing
[data and evaluation protocol](docs/protocol.md) or the frozen routing studies.

## Objective and scope

Investigate whether Sinkhorn-based optimal transport (OT), eventually combined
with Ridge regression, can reduce the effect of class imbalance and improve
classification on a fixed retained dataset.

Reweighting changes each training example's influence through its loss weight.
It does not create images or change the retained class counts. The intended
outcome is better balanced accuracy and rare-class recall; improvement has not
been established.

This branch will eventually be compared with a synthetic-sample branch through
shared data inputs and a shared evaluator. Each branch can own its training
procedure. We have not finalized the synthetic method or the shared interfaces.

## Selected starting paper

Guo et al., **Learning to Re-weight Examples with Optimal Transport for
Imbalanced Classification**, NeurIPS 2022.

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf)
- [Official code](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT)

The paper includes ResNet-32 experiments on CIFAR-100-LT. Its reported CIFAR
schedule is 200 epochs of ordinary training followed by 40 epochs of weighted
training. Those are paper settings, not a finalized configuration for our study.

Stage 1 trains the full model on imbalanced data with ordinary cross-entropy.
Stage 2 continues from that model, repeatedly learning example weights through
OT and training with weighted classification losses. The authors' image code
optimizes only the final linear classifier parameters during Stage 2.

Sinkhorn computes a transport plan for the current example weights. A separate
optimization step updates those weights to reduce the OT discrepancy. Running
Sinkhorn with fixed source masses does not itself learn new example weights.

## Fixed-data constraint

Create the imbalance first. Only retained images and labels may supply
subsequent learning, reference construction, or other method inputs. Discarded
images remain unavailable. Evaluation data and labels must not influence
training, weight learning, or method selection.

Guo's original setting obtains a balanced reference of ten labeled images per
class from the original training pool. Our retained-data-only setting changes
that assumption. Results from our version must therefore be identified as an
adaptation, rather than an exact reproduction of the published experiment.

## Agreed initial reference design

Use **one average feature vector per class**, calculated from retained training
examples using the Stage-1 feature extractor. Give each class representative
equal total importance in the reference distribution.

For example, 100 retained cats produce one cat representative and five retained
dogs produce one dog representative. Both representatives receive equal
reference importance; all 105 original images remain available for training.

This is a feature average, not an average of image pixels. It provides a compact
reference without requiring additional reference images or permanently
reserving scarce tail examples.

The intended OT comparison uses both feature similarity and label agreement,
following the paper's combined-cost idea. Exact costs and settings remain open.

## Training and evaluation flow

```text
Retained imbalanced training data
               |
       Stage 1: train full model
               |
       Build class feature prototypes
               |
       Stage 2: repeat over training batches
         - calculate OT costs against the reference
         - use Sinkhorn and update example weights
         - train the classifier with weighted losses
               |
       Trained model -> shared evaluator
```

At inference, only the trained model is needed. OT, example weights, and the
reference are training components.

## Questions still open

- **Reference details:** feature normalization and augmentation; whether to
  keep prototypes fixed; whether to exclude an image from its own reference
  average. Fixed prototypes were suggested as a simple first version, but
  these choices have not been locked.
- **Stage-2 behavior:** exactly which parameters and batch-normalization
  statistics remain fixed, and which image augmentations continue.
- **Weight learning:** initialization, persistence by image ID, normalization,
  update rule, loss scaling, and protection against weight concentration.
- **Evaluation plan:** development/evaluation roles within the fixed-data
  constraint, matched baselines, seeds, and training budgets. The shared
  evaluator is not yet designed.
- **Ridge:** its inputs, supervised target, and position relative to OT. No
  Ridge role has been selected. First establish the OT-only baseline.

Prototypes estimated from very few images can be inaccurate. Favoring examples
near a class average could also suppress useful unusual examples. These are
limitations to investigate, not measured failures of our proposed version.

## Immediate next step

Finish understanding the synthetic branch's inputs, training process, and
outputs, then design the common pipeline. Do not add implementation machinery
or claim novelty merely because Ridge is included. Repository archival and
replacement of the existing pipeline have not been performed.
