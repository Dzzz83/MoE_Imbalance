# Expert Diagnostics

This document describes the reusable array-only diagnostic framework. It is
implementation documentation, not an experimental results report. The
implementation is [scripts/expert_diagnostics.py](../scripts/expert_diagnostics.py);
completed OOF measurements are in [oof-results.md](oof-results.md).

The module accepts aligned logits or predictions and does not load a dataset,
checkpoint or test split. It does not fit router parameters.

## Inputs and validation

The primary input is an aligned logits array with shape (N, E, C), or a
prediction array with shape (N, E). Optional labels have shape (N,), class
counts define the canonical groups, and expert names define the fixed expert
order.

Validation rejects nonfinite values, invalid class indices, missing labels when
class counts are requested, duplicate or misordered expert names, shape
mismatches, prediction/logit disagreements, negative weights and weight rows
that do not sum to one.

Head/Medium/Tail boundaries come from
scripts/base_trainer.py::compute_class_groups: Head n >= 100, Medium
20 <= n < 100, Tail n < 20. BA and each group metric are macro means of class
recall. Sample accuracy is reported separately.

## Definitions

For expert e, the top-1 prediction is
\( \hat y_{ne} = \operatorname{argmax}_c z_{nec} \), and correctness is
\( I[\hat y_{ne}=y_n] \). Confidence diagnostics use the expert's maximum
softmax probability. Confidence ties use the first supplied expert.

For row-stochastic nonnegative weights \(w_{ne}\), the two ensemble semantics
are distinct:

- Weighted logits predict
  \( \operatorname{argmax}_c \sum_e w_{ne}z_{nec} \).
- Weighted probabilities predict
  \( \operatorname{argmax}_c \sum_e w_{ne}\operatorname{softmax}(z_e)_{nc} \).

The uniform logit and uniform probability definitions are checked against the
active router implementations.

## Public reports

- agreement_patterns() reports prediction partitions and fractions. A unique
  dissenter is defined only for the (E−1)-1 pattern.
- correctness_diagnostics() reports the number of correct experts, exactly-one
  and multiple correctness, all-wrong/all-correct events, global
  top-confidence correctness and the rank of the highest-confidence correct
  expert.
- complementarity() reports per-expert correctness/BA, pairwise joint
  correctness and joint error, exclusive correctness, per-class overlap,
  macro Head/Medium/Tail overlap and agreement patterns.
- ensemble_contribution() reports paired BA and Tail changes after removing
  each expert from uniform logit averaging. It requires class counts so Tail
  is not silently replaced by sample accuracy.
- hard_routing_headroom() reports the all-wrong fraction, at-least-one-correct
  fraction, hard-selection oracle sample accuracy and macro metrics.
- evaluate_soft_mixture(weights, combination=...) evaluates caller-supplied
  row-stochastic weights for weighted logits or probabilities. It does not
  optimize weights.

## Three different oracle/ensemble concepts

### Hard-selection oracle

The hard-selection oracle uses the true label only to ask whether at least one
expert's top-1 prediction is correct. It measures the best possible hard
selection among the supplied predictions. It is label-dependent and is not an
inference-time router.

### Fixed-weight logit ensemble

A fixed-weight ensemble applies one weight vector to every image and then
argmaxes the weighted logits. Task 3E-A evaluates such vectors on the OOF
development population. It is an actual classifier evaluation when its weights
are supplied in advance, but selecting the best vector on one development
partition does not establish independent generalization.

### Adaptive soft-mixture feasibility oracle

The Task 3E-B implementation
[scripts/task3e_soft.py](../scripts/task3e_soft.py) solves a separate LP for
each image to ask whether any convex logit combination can give the true class
a positive margin. It uses the true label, so it measures existence of a
correcting combination rather than trained-router accuracy. Its results and
numerical tolerances are documented in [oof-results.md](oof-results.md) and
[soft_oracle_results.json](../artifacts/oof/task3e_soft_feasibility/soft_oracle_results.json).

These three concepts must not be conflated: hard selection chooses an existing
top-1 answer, fixed weights are a specified classifier, and adaptive
soft-mixture feasibility is a label-dependent existence test.

## Corrected diagnostic defects

- The former lone-dissenter logic assumed three experts. The reusable report
  now distinguishes 4-0, 3-1, 2-2, 2-1-1 and 1-1-1-1 prediction partitions.
- The former unique_best quantity mixed confidence and correctness. Reports now
  separate correctness counts, exactly-one correctness, multiple correctness,
  no correctness, global confidence correctness and confidence rank.
- Retired diagnose_dace entry points are identified as three-expert,
  legacy-split postmortems. Their findings are not presented as proof that all
  learned routing is infeasible.
- New Head/Medium/Tail quantities use mean class recall. Existing active
  evaluation metrics and historical numbers were not rewritten.

The diagnostic framework supplies definitions and checks; it does not establish
that a future Ridge, Sinkhorn or other router can predict useful weights.
