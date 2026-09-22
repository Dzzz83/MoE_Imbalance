# Problem — Routing Limitations and Open Questions

This document separates established historical findings from conclusions revised
by the nested-OOF work. Detailed OOF measurements belong in
[oof-results.md](oof-results.md); original full-data measurements belong in
[results.md](results.md).

## 1. Established historical findings

These findings are valid for the population and protocol under which they were
measured.

### Original full-data test track

- The four-expert pool has a historical all-wrong floor of 39.73% across the
  three full-data seeds, leaving a 60.27% hard top-1 oracle ceiling. This is a
  statement about the original balanced test population and hard selection
  among the full-data expert predictions.
- The pre-registered parameter-free confidence, probability-average and TTA
  rules did not improve both BA and Tail over historical uniform logit
  averaging. See [results.md](results.md) and the complete catalogue in
  [records/routing_mechanism.md](../records/routing_mechanism.md).
- Full-data predicted-label agreement was in a narrow kappa range of
  approximately 0.42–0.49. The pair LAL/BalancedSoftmax, despite implementing
  the same objective at tau = 1, was not more alike than the other measured
  pairs. This establishes disagreement under that track; it does not by itself
  identify useful specialization.

### Earlier split and DACE findings

- On the earlier split, among 596 samples considered savable by the tested
  confidence rule, the correct expert was not the most confident 83.9% of the
  time. This supports the failure of that confidence signal, not a universal
  impossibility theorem.
- The historical DACE routing score was anti-predictive: AUROC was 0.327 for
  correctness and 0.673 for wrongness. Its score scales differed by roughly
  eight times across experts, the argmax selected expert B on about 90% of
  samples, and its cosine fallback fired on 0.00% of samples. The DACE
  three-expert uniform result was 39.84 versus 45.68 for the corresponding
  original pool.
- An earlier fitted gate on the old split reached 43.98% against a
  63.04% oracle-weighted reference and collapsed to one expert. This is
  evidence against that feature representation and fitting setup, not proof
  that every learned router fails.
- The model-initialization seeding defect was fixed. Its audit history is in
  [archive/bugfix-report.md](archive/bugfix-report.md).

## 2. Findings revised by the OOF work

### Held-out supervision is now available for a designated development population

The statement that no honest held-out data exists is no longer an accurate
description of the whole repository. The nested-OOF pipeline provides
row-wise held-out predictions for the 8,677-image outer-fold-0 development
population. Tasks 3E-A and 3E-B use only the 6,507-image inner-folds-1–3
router-fitting partition.

The narrower statement remains true: predictions from the original full-data
experts on their own training images are not honest router supervision. The
reserved outer-fold evaluation population and the original test set remain
excluded from the completed analyses. Inner fold 0 was inspected descriptively
during Task 3C, so it cannot be described as completely untouched for every
research choice.

### All-experts-wrong is not a universal soft-mixture upper bound

If every existing expert predicts the wrong class at top 1, hard selection
among those top-1 predictions cannot correct the image. Task 3E-B nevertheless
found 321 such OOF images for which a convex combination of the original logits
can make the true class win. The hard-selection statement must therefore not be
extended to arbitrary convex logit mixtures.

The soft result is label-dependent feasibility, not inference-time accuracy.
It demonstrates possible headroom, not that a router can find the weights.

### Disagreement is not enough to establish useful complementarity

Historical agreement measurements alone cannot justify the claim that
disagreement is only initialization noise or that complementary expertise is
absent. Task 3C documents complementary correct predictions, including
distinct Tail and exclusive-correctness contributions from LAL and
BalancedSoftmax. The unresolved issue is whether those contributions can be
predicted from inference-time information.

### Fixed composition and adaptive prediction are different questions

Task 3E-A found fixed convex logit weights that improve both BA and Tail on the
router-fitting partition. Task 3E-B found further soft-mixture feasibility.
Neither result is an independently validated adaptive router, a full-data test
result, or evidence that a particular Ridge formulation will generalize. Task
3F-A subsequently found lower held-out contribution-target MSE than its
training-only global control, but no paired BA–Tail advantage over the fixed
references on the same development population.

## 3. Current unresolved problems

The next research decision must address:

- generalizing any predictable expert-weight signal beyond the current OOF
  development population without access to the true label;
- improving weak Tail recognition and measuring Tail uncertainty honestly;
- generalizing from fold-trained experts to the eventual full-data experts;
- preventing in-sample router overfitting and separating fitting from
  hyperparameter selection;
- demonstrating value beyond strong fixed-weight and uniform baselines; and
- obtaining independent evidence across additional outer folds, seeds or
  datasets.

Before independent evaluation or any new router variant, the frozen Task 3F-A
target, inference-time features, regularization, baselines, selection
procedure and non-cheating safeguards must be preserved. Sinkhorn is a
separate allocation mechanism and should not be treated as a source of
predictive signal.

## 4. What the current evidence does not establish

- No completed experiment validates a learned router on the reserved outer
  population or on a fresh test population.
- No completed experiment validates Ridge on the reserved outer population or
  a fresh test population; Sinkhorn, a new specialist, and the full nested
  300-run matrix remain unexecuted.
- Zero observed coverage for a rare Tail class is not proof that the class is
  intrinsically unlearnable.
- OOF development improvements cannot be compared directly with the original
  three-seed full-data test baseline because their training populations, seed
  counts and evaluation roles differ.
