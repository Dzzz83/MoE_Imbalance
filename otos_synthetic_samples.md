# Reading note: OTOS synthetic samples

Paper: **Oversampling for Imbalanced Data via Optimal Transport**, Yuguang Yan
et al., AAAI 2019, pages 5605–5612.

- [Paper](https://tanmingkui.github.io/files/publications/Oversampling.pdf)
- [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/4503)
- [Author publication page](https://tanmingkui.github.io/publication/Oversampling)

## Original method

OTOS addresses binary imbalance: a minority positive class and a majority
negative class. It synthesizes minority vectors, then trains a classifier on
real and synthetic data.

1. Train an initial linear SVM on the real vectors.
2. Use minority hinge losses to emphasize difficult examples.
3. Draw random vectors and optimize transport toward minority examples.
4. Use exponentiated-gradient updates with Sinkhorn-based KL projection.
5. Form synthetic vectors as transport-weighted combinations of minority vectors.
6. Train an SVM on the augmented data.

A classifier-based regularizer also guides synthesis. See Equations 9–12 and
Algorithm 1. This is more than a single ordinary Sinkhorn solve.

## Reported experiments

Six benchmark and four medical-image datasets; linear SVM classification.
Medical images use ImageNet-pretrained ResNet-152 features. There is no reported
ResNet-32/CIFAR-100 experiment.

Example from Table 4, German credit; means shown as percentages:

| Method | Minority recall | F1 | G-mean |
|---|---:|---:|---:|
| SVM | 48.93 | 55.59 | 65.60 |
| SMOTE + SVM | 71.77 | 60.25 | 71.46 |
| OTOS + SVM | 75.90 | 61.56 | 72.55 |

These are published results, not local measurements.

## Source-code status

No verified public author repository was found in the paper, author publication
page, or targeted GitHub searches during this discussion. This is not proof
that none exists. There is no verified author implementation available to us
for a code-to-code comparison. The paper provides equations and pseudocode.

We have not reproduced its results or audited its evaluation procedure. Source
availability and reported results are separate questions.

## What using it in our project would mean

This is an algorithm reference, not a ready-to-run CIFAR-100 baseline. Our
ResNet-32 adaptation would need explicit answers to:

- How are minority/majority roles defined across 100 classes?
- Does an auxiliary SVM remain, or what replaces its synthesis guidance?
- Which classes receive synthetic features, and how many?
- When are features generated, and what remains fixed afterward?

An illustrative synthetic vector could be a weighted combination of three
retained dog feature vectors. That vector has no corresponding new photograph;
it enters the classifier after the feature extractor. Reusing information this
way does not provide new independent dog observations.

## Candidate pipeline, not an agreed algorithm

```text
Shared retained images -> train ResNet-32
                              |
                   Extract training features
                              |
                 Adapt OTOS to synthesize features
                              |
                    Train final classifier
                              |
           Complete ResNet-32 -> shared image evaluator
```

This requires a ResNet classifier adaptation rather than the original final
SVM. Freezing, feature caching, Stage-1 checkpoint sharing, and Ridge's role are
not mandatory shared-pipeline decisions and remain open. All method inputs
must respect the retained-data constraint.

Status: synthetic-direction candidate needing adaptation and implementation
from the paper. See [shared_pipeline_plan.md](shared_pipeline_plan.md) for the
common contract; no synthesis or training has been implemented.
