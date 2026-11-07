# Reading note: Guo's OT reweighting

Paper: **Learning to Re-weight Examples with Optimal Transport for Imbalanced
Classification**, Dandan Guo et al., NeurIPS 2022.

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf)
- [Official repository](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT)
- [Image training implementation](https://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT/blob/main/OT_train.py)

## Original method

Learn an influence weight for each training example, keeping images and class
counts unchanged. Train the full model normally first, then alternate weight
learning and weighted classification training.

The reference is a small balanced labeled meta set: ten images per class in
the image experiments. OT compares training and reference distributions using
features and labels. The paper also studies prototype references.

Sinkhorn solves the entropically regularized transport problem for current
weights. A separate gradient update changes the weights through that solution.
The paper considers direct weights and a weight network; its image experiments
use direct weights. See Sections 4–5 and Algorithm 1.

## Reported CIFAR experiment

ResNet-32 on CIFAR-100-LT; Stage 1 is 200 epochs, Stage 2 is 40 epochs. Table 1
reports averages over five experiments. At imbalance factor 100, its weight-vector
method has **51.46% top-1 error**, equivalent to **48.54% accuracy**. This is a
paper result, not a result reproduced here. See Section 6.1 and Table 1.

## What the released image code does

The Stage-2 optimizer targets the final linear classifier's weight and bias.
The code maintains weights by training-example ID, normalizes batch weights
with softmax, and uses a per-example cross-entropy loss. The README expects a
separately pretrained backbone and asks users to adjust paths.

These observations are from source inspection, not execution. Repository
defaults must be reconciled with paper settings before claiming reproduction.
This note does not certify the entire implementation's evaluation hygiene.

## Our intended adaptation

The implemented adaptation is documented in [reweighting.md](reweighting.md).
The main change is to build equal-mass class representatives
from retained training features instead of using additional balanced reference
images from the original pool.

An illustrative reference: 100 cats and five dogs become one mean cat feature
and one mean dog feature, with equal reference importance. This does not create
more dog observations or establish that the resulting weights improve learning.

## Implementation boundary

```text
Shared retained training images + fresh ResNet-32
                      |
       Ordinary training -> class references
                      |
       Weight optimization + weighted training
                      |
       Complete model checkpoint -> shared evaluator
```

The method guide records the implemented cost, fixed reference construction,
global weight normalization, and fully frozen backbone. Focused tests cover
weight-to-image alignment and loss scaling.

Ridge's inputs, supervision target, and position in this process remain open.
Its inclusion is not evidence of novelty or improvement. Start with OT alone.

Status: adaptation implemented and verified with artificial-data CPU tests;
research performance remains unverified. Integration plan:
[shared_pipeline_plan.md](../shared_pipeline_plan.md).
