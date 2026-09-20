# Expert diagnostics and routing feasibility

This document records the Task 2 diagnostic framework.  It is an implementation
and design note, not a new CIFAR-100-LT experiment.  No real test samples,
cached test predictions, expert retraining, router fitting, Ridge, or Sinkhorn
were used for this task.

## Corrected defects

- The old lone-dissenter loop was written for three experts.  On a four-expert
  `2-1-1` prediction partition it found one agreeing pair and then silently
  treated the first of two dissenters as *the* dissenter.  The reusable
  diagnostic now identifies arbitrary prediction partitions and defines a
  unique dissenter only for the `(E-1)-1` pattern.  For four experts it reports
  `4-0`, `3-1`, `2-2`, `2-1-1`, and `1-1-1-1` separately.
- The old `unique_best` quantity was a confidence event among correct experts,
  but its name and report could be read as “only one expert is correct.”  The
  new report separates the number of correct experts, exactly-one correctness,
  multiple correctness, no correctness, global top-confidence correctness,
  and the rank of the highest-confidence correct expert.  Every fraction
  carries its denominator and conditioning event.
- The three `diagnose_dace*.py` entry points are now explicitly marked and
  guarded as retired three-expert/legacy-split postmortems.  Their negative
  findings are limited to the tested DACE signal and probe; they no longer
  present those findings as proof that all learned routing is infeasible.
- New Head/Medium/Tail quantities use mean class recall within each group.  The
  existing active evaluation metrics and historical numbers were not rewritten.

## Reusable implementation

[`scripts/expert_diagnostics.py`](../scripts/expert_diagnostics.py) exposes the
array-only `ExpertDiagnostics` class.  It accepts aligned `(N, E, C)` logits or
`(N, E)` predictions, optional `(N,)` labels, optional per-class training
counts, and optional expert names.  It never loads a dataset or checkpoint and
never fits parameters.

The public reports are:

- `agreement_patterns()` — partition counts/fractions and, only for `(E-1)-1`,
  correctness of the agreeing group and unique dissenting expert.
- `correctness_diagnostics()` — correctness-count distribution, confidence
  ambiguity, and confidence ranks.  Global confidence ties use the first
  supplied expert column; confidence ranks are conditioned on at least one
  correct expert.
- `complementarity()` — per-expert correctness/BA, pairwise joint correctness
  and joint error, exclusive correctness, per-class overlap, macro H/M/T
  overlap, and agreement patterns.  Prediction disagreement is reported but
  is not treated as useful specialization.
- `ensemble_contribution()` — paired BA and Tail changes after removing each
  expert from uniform logit averaging.  The method requires class counts so
  Tail is not silently replaced by sample accuracy.
- `hard_routing_headroom()` — all-wrong fraction, at-least-one-correct
  fraction, hard-selection oracle sample accuracy, oracle BA, and optional
  macro H/M/T metrics.  This is a label-dependent hard-selection oracle, not
  an inference-time method and not an upper bound on arbitrary soft mixtures.
- `evaluate_soft_mixture(weights, combination=...)` — evaluates caller-
  supplied row-stochastic nonnegative weights for either weighted logits or
  weighted probabilities.  It does not optimize the weights.  Uniform weights
  are checked against the existing Uniform and Probability router definitions.

Input checks reject nonfinite logits/counts/weights, invalid class indices,
missing class labels when class counts are supplied, duplicate or misordered
expert names, shape mismatches, prediction/logit disagreements, negative
weights, and rows whose weights do not sum to one.

The canonical group boundaries are obtained from
`scripts.base_trainer.compute_class_groups`: Head `n >= 100`, Medium
`20 <= n < 100`, Tail `n < 20`.  BA and each nonempty group metric are macro
means of class recalls.  Sample-weighted accuracy is reported separately.

## Evidence assessment for the current pool

The following is an interpretation of the verified historical summaries in
[`docs/results.md`](results.md), [`docs/problem.md`](problem.md), and
[`records/routing_mechanism.md`](../records/routing_mechanism.md).  The numbers
were not regenerated or selected during this task.

Established:

- The four experts have materially different aggregate profiles: LAL is the
  strongest overall/tail expert in the recorded summary, Mixup is strongest on
  Head and best calibrated, and CE is weaker overall and overconfident.  These
  are performance differences, not proof of predictable per-sample routing.
- Uniform logit averaging is the current recorded baseline and remains ahead
  of the frozen parameter-free alternatives under the pre-registered BA + Tail
  criterion.  Adding all four experts improves the unbiased mean ensemble-size
  curve relative to smaller subsets; this establishes useful aggregate
  combination, not which expert should be selected per sample.
- Historical current-pool summaries report pairwise predicted-label kappa in a
  narrow `0.42–0.49` range.  This establishes disagreement, but disagreement
  alone is not complementary correctness.
- Historical correctness-count summaries report a substantial all-wrong
  region and that most savable samples have multiple correct experts.  Thus a
  single “correct expert” target is often ambiguous.  Exact per-class
  exclusive-correctness and joint-error tables still require supplied,
  label-authorized predictions; the new module provides those calculations.

Not established:

- The current protocol has no honest held-out correctness labels.  The full-
  data expert predictions on the training set cannot be used to fit or validate
  a router.  The retired DACE feature/probe result is evidence about one old
  split and one signal, not a universal impossibility theorem.
- No current experiment establishes predictable per-sample specialization,
  feasible soft weights, or a Ridge/Sinkhorn gain.  Synthetic tests verify
  definitions and input contracts only; they do not demonstrate learnability.
- The all-wrong fraction is a ceiling for hard selection on the supplied
  expert predictions, but it is not a ceiling for soft logit or probability
  combinations: a soft combination can select a class that no individual
  expert argmax selected.

Training-only metadata can provide class counts, canonical groups, sample
indices, expert identities, seeds, configurations, and checkpoint provenance.
It cannot provide unbiased correctness, complementarity, or router-target
labels.  Those require out-of-fold predictions or a separately authorized
development set.

## Proposed non-cheating OOF protocol

This is design only.  It must be approved and pre-registered before any fold
training or router fitting.

1. **Freeze the population and fold map.** Use only the canonical
   `lt_ir100_train_indices.npy` population and its training labels.  Construct
   a deterministic, class-stratified `K=5` fold assignment from sample indices
   with a separately recorded fold seed.  The canonical tail classes have five
   samples, so five folds give each such class one held-out sample per fold;
   never use a global shuffle that can starve a rare class.  If a future
   population has fewer than five samples in a class, reduce `K` or use a
   pre-specified repeated/grouped design and disclose that the class cannot
   support five independent folds.

2. **Train fold experts without membership leakage.** For every fold, expert
   identity (CE, LAL, BalancedSoftmax, Mixup, and any future specialist), and
   configured seed, train on exactly the other `K-1` folds.  Keep the canonical
   recipe otherwise fixed: no test data, no test labels, no checkpoint choice
   from test results, and no in-sample prediction for the held-out fold.
   The final-epoch fold checkpoint is the source of the held-out prediction.

3. **Record provenance with every OOF row.** Persist, in a versioned artifact,
   at least `sample_index`, class label, `fold_id`, expert identity, model seed,
   fold-training-index hash, resolved configuration, checkpoint hash, and the
   prediction/logits/features produced for that row.  Validate that the held-
   out index is absent from the corresponding training membership before
   accepting the row.  Keep expert column order fixed and record it in the
   artifact schema.  Never mix predictions from different sample orderings.

4. **Separate router fitting from hyperparameter selection.** Treat the five
   expert folds as outer router folds.  For each outer fold, fit Ridge (or a
   future candidate) on OOF rows from the other four folds and select any
   regularization, feature, temperature, or Sinkhorn parameters without using
   the outer-fold labels.  Evaluate the frozen choice on the outer fold and
   aggregate the five outer results.  Rare-class estimates will be noisy and
   must be reported with class support; do not tune separately for a tail class
   with one observation per outer fold.  After the protocol and hyperparameters
   are frozen, fit the final router on all OOF rows using only the selected
   settings.

5. **Distinguish fold-trained from full-data experts.** OOF predictions come
   from experts trained on 80% of the data, while the eventual inference pool
   will normally be trained on all 10,847 samples.  Do not treat their logits as
   exchangeable without measuring the shift.  Use OOF data to fit any output
   calibration/router mapping, then apply that frozen mapping to the full-data
   experts.  Do not fit a correction on full-data in-sample predictions.

6. **Compare existing and future experts fairly.** Use the same fold IDs,
   sample membership, seeds, training budget, provenance schema, and OOF
   router splits for the current four experts and every proposed specialist.
   Report per-expert macro BA/H/M/T, pairwise correctness overlap, all-wrong
   and at-least-one headroom, uniform logit/probability baselines, and the
   supplied soft-mixture result on the OOF outer folds.  A specialist that is
   only better in-sample or only on a selected fold is not established as
   complementary.

7. **Freeze before the final test read.** Select the expert pool, router
   features, regularization, temperature, transport variant, and reporting
   rules using only OOF development results.  Then train the final full-data
   experts, apply the frozen router once to the balanced test set through the
   authorized evaluation entry point, and report the pre-specified BA and Tail
   comparisons.  Do not inspect test labels to choose a router, subset,
   checkpoint, threshold, or method, and do not create test-derived artifacts
   for later development.

This protocol can answer whether a suitability signal generalizes from
held-out training samples to a final full-data expert pool.  It cannot by
itself guarantee a gain: Sinkhorn can only reorganize a score matrix, and
Ridge can only exploit information present in the OOF features/logits.

## Recommended next steps and limitations

First keep the new module as the analysis seam and run it only on synthetic or
approved OOF arrays.  Next approve the fold map and OOF compute budget, collect
OOF predictions for the existing pool, and estimate whether soft mixtures have
decision-relevant headroom before designing new specialists.  Only after that
evidence should a separate plan approve Ridge, and only a later plan should
approve Sinkhorn implementation.

The framework does not solve finite-sample uncertainty for five-example tail
classes, distribution shift between fold-trained and full-data experts, or
feature leakage in a future pipeline.  Those remain experimental questions.

## Task 3A implementation

The proposed protocol is now executable as a membership and provenance layer in
[`data/nested_oof.py`](../data/nested_oof.py); its role definitions,
canonical fold sizes, and remaining cross-fitted-model dependencies are
documented in [`nested-oof-protocol.md`](nested-oof-protocol.md).  This task
does not train experts, fit routers, or read the real test set.
