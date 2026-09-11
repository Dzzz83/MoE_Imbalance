# Routing Pre-Registration — Frozen Candidate Set

> **Status: FROZEN.** Written and committed *before* the first test-set
> evaluation of the current expert pool. The candidate set, metrics and decision
> rule below must not be edited after any test-set number has been seen;
> changing them afterwards turns the comparison into selection-on-test.
>
> Created 2026-09-11. See `docs/specs/parameter-free-routing.md`.

---

## Why this file exists

The project has no validation split and no honest held-out data, so **no router
may be fitted**. If several unfitted rules were tried and the best reported, the
test set would silently become a selection set. Freezing the candidate set in
advance removes that degree of freedom, exactly as `AGENTs.md` §6 requires for
any reported claim.

## Expert pool (fixed)

The four experts trained from `configs/*.yaml` on the canonical 10,847-sample
long-tailed training set, seeds {78, 88, 1034}:

| Config | Label | Loss |
|:--|:--|:--|
| `configs/ce.yaml` | `CE` | cross-entropy (ERM) |
| `configs/lal.yaml` | `LAL` | logit-adjusted, tau=1.0 |
| `configs/balanced_softmax.yaml` | `BalancedSoftmax` | balanced softmax |
| `configs/mixup.yaml` | `Mixup` | CE + mixup, alpha=1.0 |

**Recorded caveat:** with tau=1.0 the logit-adjusted loss and balanced softmax
differ only by a class-independent constant, so they are the *same objective*.
The pool therefore contains at most **3 distinct experts**, and κ(LAL, BS) is
expected to be ≈ 1.0. Any diversity claim from this pool must account for that.

## Candidate routing rules (frozen — exactly four)

| # | Rule | Combination | Parameters |
|:-:|:--|:--|:--:|
| 1 | `UniformRouter` | mean of expert logits, then argmax | 0 |
| 2 | `ProductRouter` | product (geometric mean) of expert probabilities | 0 |
| 3 | `ConfidenceRouter` | argmax of raw max-softmax confidence, then that expert's argmax | 0 |
| 4 | `TTARouter` | `ConfidenceRouter` over TTA-averaged logits | 0 |

No other rule may be added to the reported comparison. Rules from
`docs/routing-results-record.md` are **excluded by construction**: they cannot
be evaluated without held-out fitting data.

## Metrics and reporting

For every rule, evaluated on the balanced 10K CIFAR-100 test set:

- **Balanced Accuracy (BA)** — primary, mean per-class recall.
- **Tail accuracy** — mean recall over tail classes (< 20 training samples),
  using the immutable Head/Med/Tail definition in `AGENTs.md` §5.
- Head and Medium accuracy, for context only.
- Per-expert usage distribution, so a rule that collapses onto one expert is
  visible.

## Decision rule (fixed in advance)

A routing rule counts as a **success** only if, compared with `UniformRouter`
on the *same* expert pool and the *same* test set:

1. BA is **higher**, and
2. Tail accuracy is **higher**,

with the paired per-sample difference exceeding its own standard deviation, and
the direction consistent across all three seeds {78, 88, 1034}. A rule that
merely ties uniform is reported as **no gain** — never as a success
(`AGENTs.md` §6, and the `dace-fix-plan.md` §8 precedent).

## Expected outcome, declared before looking

The prior recorded evidence predicts **a tie with uniform**:
`phase0-results.md` reports 18 methods over 2 pools failing to beat uniform, with
a 44.2% all-wrong floor on the 3-expert pool. This is stated here so a tie is
reported as the pre-registered expectation rather than reframed afterwards.

## Integrity rules

- The test set is read only through `scripts/evaluate_experts.py`, which appends
  an entry to `docs/test-access-log.md` (command, timestamp, git hash) on every
  read, so accidental peeking is visible.
- No threshold, weight, temperature, or rule ordering may be tuned after a
  test-set number has been seen.
- A single test-set evaluation produces the numbers for all four rules at once,
  so the rules cannot be compared across separate peeks.
