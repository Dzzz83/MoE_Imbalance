# Refactoring Stage 0 — Frozen Contracts

This record accompanies the characterization tests. It describes current
behavior; it does not declare every behavior scientifically correct and does
not introduce a new routing interface.

## Numerical definitions

| Area | Current contract |
|:--|:--|
| Active feature softmax | `scripts/utils/features.softmax` subtracts the final-axis maximum, preserves shape, and normalizes each final-axis slice. It does not validate non-finite inputs; the non-finite result propagates. Task 3F-A validates logits before using its own stable-softmax path. |
| Uniform ensemble | `UniformRouter` and the OOF analyses use the arithmetic mean of the original expert logits, followed by `argmax`. |
| Probability ensemble | `ProbabilityAverageRouter` uses the mean of per-expert softmax probabilities, followed by `argmax`. This is not equivalent to logit averaging. |
| Fitted Ridge combination | `combine_weighted_logits` combines original logits with row-wise convex weights. `_weighted_probability_predictions` instead combines per-expert probabilities. |
| Ties and sentinel | NumPy's first `argmax` wins ties. Uniform and probability routers return expert index `0` as their historical sentinel while reporting their actual combined prediction and uniform usage separately. |
| Task 3F-A target | The marginal true-class log-probability contribution is `(z_e,y - z̄_y) - Σ_c p̄_c (z_e,c - z̄_c)`, with labels used only while constructing the supervised target. Contributions sum to zero across experts. |

The target, feature shape/order, standardization, Ridge solver, score-to-weight
transform, and weighted-logit prediction contracts are covered by the existing
synthetic `test_task3f_ridge.py` and `test_task3f_target_diagnostics.py` tests.

## Metric differences intentionally preserved

The repository currently has multiple metric seams:

- `scripts/base_trainer.py` returns `(balanced_accuracy, per_class)` and its
  group accuracy is sample-weighted; an absent group returns `0.0`.
- `scripts/evaluation.py` and `scripts/subsets.py` compute macro recall over
  classes present in the evaluated labels. Their H/M/T group values are
  sample-weighted and absent groups return `0.0`.
- `scripts/utils/metrics.py` omits absent classes unless `num_classes` is
  supplied; with `num_classes`, absent classes contribute zero recall. Its
  group values are sample-weighted.
- `scripts/expert_diagnostics.py` and `scripts/task3f_ridge.py` report macro
  class recall within H/M/T. They require every class needed for a requested
  group to be represented, except that an empty group is reported as `None` by
  the diagnostics module.

`tests/test_stage0_contracts.py` uses unequal class counts, one-sample classes,
absent classes, all-correct predictions, and all-wrong predictions to keep
these differences visible until a later metric-consolidation decision.

## Provenance and persistence coverage

The existing synthetic tests cover JSON round trips, NPZ array preservation,
array/source hashes, schema and expert-order checks, exact sample-ID alignment,
duplicate/missing rows, inner-fold restrictions, training-membership exclusion,
checkpoint/config provenance, and immutable artifact reuse. The OOF execution
metadata is intentionally mutable for status/checkpoint/prediction updates;
completed prediction/config/manifest artifacts are write-once and incompatible
reruns are rejected.

The restricted Task 3F analysis loader accepts inner folds 1–3 and rejects
inner fold 0, while the outer-evaluation population is rejected by the existing
synthetic validation tests. No new Stage 0 test reads those populations.

## Checkpoints and test-access capability

Synthetic checkpoint tests cover expert identity, architecture/state shape,
seed, final epoch, missing metadata, missing files, and canonical expert order.
The modern evaluation loader requires and consumes a single-use
`TestAccessGrant` before constructing the protected loader. The legacy
`scripts/utils/data.create_cifar_loader("test")` path currently calls
`authorize` but discards the returned grant; the Stage 0 test records this
behavior using a synthetic dataset and fake access log. This is a documented
security-refactoring issue, not a Stage 0 repair.

## Full-precision development references

The regression test reads the existing machine-readable summaries without
rerunning analysis or loading OOF arrays:

| Reference | Balanced Accuracy | Tail accuracy |
|:--|--:|--:|
| Task 3E-A uniform (`fixed_020`) | 0.35932756391263865 | 0.07009379509379508 |
| Task 3E-A `fixed_006` | 0.3721341147209944 | 0.09911495911495911 |
| Task 3E-A `fixed_007` | 0.36440310027522765 | 0.14728956228956228 |
| Task 3F-A highlighted Ridge | 0.3655023874404003 | 0.07379749879749879 |

These are development-population references only. They are not independent
validation results and do not establish a method improvement.
