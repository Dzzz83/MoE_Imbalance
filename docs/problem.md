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
description of the whole repository. The nested-OOF pipeline provides row-wise
held-out predictions for the 8,677-image outer-fold-0 development population.
Tasks 3E-A/B and 3F-A–3F-F use only the 6,507-image inner-folds-1–3
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

The soft result is label-dependent feasibility, not inference-time accuracy. It
demonstrates possible headroom, not that a router can find the weights.

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

## 3. Current research problems

### Problem A — Expert complementarity is not equivalent to predictable routing

The OOF experts correctly classify some images that the other experts miss, and
fixed composition changes BA and Tail. This establishes complementarity in the
observed predictions, not a usable inference-time signal for choosing weights.
Tasks 3F-C and 3F-D found retrospective confidence and disagreement patterns,
but no feature mask was promoted to a router. See [oof-results.md](oof-results.md)
Sections 2, 8 and 9.

### Problem B — The initial Ridge router learns a strong global preference

The highlighted Ridge models assign Mixup a learned intercept of approximately
1.10–1.13. Mixup receives the highest saved weight on about 98.6% of rows, with
mean weight about 37% overall and about 40.5% on actual Tail rows. Image-
dependent terms are nonzero and change individual scores, but they do not
remove the pooled preference. Reducing the saved Mixup weight raises the
retrospective Tail metric at some fixed factors while lowering BA; this is a
development sensitivity result, not a replacement router or a causal
explanation. See [oof-results.md](oof-results.md) §7.

### Problem C — Tail contribution prediction is weak

The actual supervised target can favor rebalanced experts on Tail rows: LAL's
mean target is 0.8306 versus Mixup's 0.6980, and LAL exceeds Mixup on 103 Tail
rows. The saved Ridge ranks Mixup higher on 102 of those 103 rows. Task 3F-F
shows that adding features lowers contribution MSE in every matched Tail-MSE
comparison, but the primary classification result is worse for the full
13-feature model. The evidence therefore supports weak Tail ranking by the
current predictor, while not identifying whether target design, features or
training data is the dominant cause. See [oof-results.md](oof-results.md)
Sections 10 and 11.

### Problem D — Prediction error and classification performance are different objectives

The Task 3F-E target measures a local change in the true class's log
probability, not the final top-1 decision. At ε = 0.5, among positive-target
Tail cases, the tested perturbation increased true-class log probability in
106/126 LAL cases (84.13%), 88/104 BalancedSoftmax cases (84.62%), and 86/102
Mixup cases (84.31%). However, only 10/126 LAL cases (7.94%), 9/104
BalancedSoftmax cases (8.65%), and 2/102 Mixup cases (1.96%) corrected a
previously incorrect uniform classification. A positive local margin
contribution also does not guarantee a correction because the strongest
incorrect competitor can change. Task 3F-F adds the corresponding model-level
warning: lower contribution MSE did not produce better BA or Tail.

### Problem E — Tail supervision is limited

The permitted development population contains only 183 actual Tail images over
30 classes. Individual Tail classes have very few samples, and 14 Tail classes
had no correct top-1 prediction from any existing expert on this partition.
Small changes in a handful of predictions can materially change macro Tail
accuracy. The current evidence does not establish whether insufficient Tail
training data is the main cause of the Ridge behavior, and zero observed
coverage is not proof that a class is intrinsically unlearnable.

### Problem F — Fixed-weight baselines remain important

Several predefined fixed convex logit ensembles improve both development BA and
Tail over uniform: fixed_006, fixed_007 and fixed_010. Fixed_011 gives the
highest observed Tail value on the grid while trading away BA. Future adaptive
methods must be compared with these strong fixed-weight controls, not only with
uniform averaging. These are development-population comparisons, not selected
independent results; the complete grid is in [oof-results.md](oof-results.md)
§4.

### Problem G — Evaluation and distribution-shift limitations

The completed evidence remains limited by development-data reuse, overlapping
OOF expert-training populations, one completed expert seed and outer fold,
prior descriptive exposure of inner fold 0, and the shift from fold-trained
experts to eventual full-data experts. The original balanced test set has also
been accessed repeatedly during historical research. The reserved outer
population has not been used for method selection or validation. These are
protocol limitations and unresolved generalization questions, not evidence
that every future router will fail.

## 4. Roadmap constraints

Documentation consolidation is complete for this handoff. The next stages are
codebase audit/refactoring, new Ridge experiments, Sinkhorn experiments,
Ridge + Sinkhorn experiments, and independent evaluation. New Ridge and Sinkhorn
approaches have not been implemented, their exact protocols have not been
frozen, and no future stage should be described as complete.

Before any new method or independent evaluation, preserve the frozen Task 3F-A
target, inference-time feature definitions, calibration choices, uniform/
probability/fixed-weight/no-OT controls, fit-versus-selection roles and
non-cheating safeguards. Sinkhorn is an allocation mechanism, not a source of
predictive signal.

## 5. What the current evidence does not establish

- No completed experiment validates a learned router on the reserved outer
  population or on a fresh test population.
- No completed experiment validates Ridge on the reserved outer population or
  a fresh test population; Sinkhorn, a new specialist and the full nested
  300-run matrix remain unexecuted.
- No diagnostic target, margin measure, feature representation or retrospective
  subgroup is an approved inference-time routing rule.
- OOF development improvements cannot be compared directly with the original
  three-seed full-data test baseline because their training populations, seed
  counts and evaluation roles differ.
