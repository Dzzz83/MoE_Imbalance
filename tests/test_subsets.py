"""
Tests for the ensemble-size analysis.

Answers "why 2, why 3, why 4?" by evaluating a combination rule over **every
subset** of the expert pool, so accuracy can be read off as a function of how
many experts are included.

Methodological point this module pins down: choosing the *best* subset on the
test set is selection-on-test and its number is optimistically biased. The *mean*
over all subsets of a given size is unbiased. Both are reported, clearly
labelled, and the tests assert the labelling exists rather than letting an
optimistic number be quoted as a result.
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np

from scripts.router import UniformRouter, ProbabilityAverageRouter

import importlib

try:
    sub = importlib.import_module('scripts.subsets')
except ImportError:
    sub = None


def _fake_logits(n=200, e=4, c=5, seed=0):
    """Logits where each expert has a different, learnable-ish signal."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, c, n)
    logits = rng.normal(scale=1.0, size=(n, e, c)).astype(np.float32)
    # expert 0 is strong, expert 1 medium, expert 2 weak, expert 3 weak+scaled
    strengths = [3.0, 1.5, 0.6, 0.6]
    for i, s in enumerate(strengths):
        logits[np.arange(n), i, targets] += s
    logits[:, 3] *= 3.0            # deliberately different logit scale
    return logits, targets


COUNTS = np.array([50] * 5, dtype=np.int64)   # all classes "head", groups unused here


# ---------------------------------------------------------------------------
# Subset enumeration
# ---------------------------------------------------------------------------

def test_subsets_are_enumerated_by_size():
    a = sub.SubsetEnsembleAnalysis(*_fake_logits(), expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    assert len(a.subsets(1)) == 4, a.subsets(1)
    assert len(a.subsets(2)) == 6, a.subsets(2)
    assert len(a.subsets(3)) == 4, a.subsets(3)
    assert len(a.subsets(4)) == 1, a.subsets(4)
    assert len(a.subsets()) == 15, len(a.subsets())
    print("  ✅ subsets enumerated: 4 / 6 / 4 / 1 (15 total)")


def test_subsets_are_index_tuples_in_order():
    a = sub.SubsetEnsembleAnalysis(*_fake_logits(), expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    assert all(isinstance(s, tuple) for s in a.subsets())
    assert a.subsets(4) == [(0, 1, 2, 3)], a.subsets(4)
    assert (0, 2) in a.subsets(2), a.subsets(2)
    print("  ✅ subsets are ordered index tuples")


# ---------------------------------------------------------------------------
# Evaluation correctness
# ---------------------------------------------------------------------------

def test_full_subset_reproduces_the_full_pool_result():
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    full = a.evaluate_subset(UniformRouter, (0, 1, 2, 3))
    direct = UniformRouter(expert_names=list('ABCD')).predict_class(logits)
    direct_ba = sub.balanced_accuracy(targets, direct)
    assert abs(full['ba'] - direct_ba) < 1e-12, (full['ba'], direct_ba)
    print(f"  ✅ full subset reproduces the pool result ({full['ba']:.4f})")


def test_single_expert_subset_equals_that_expert():
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    for i in range(4):
        got = a.evaluate_subset(UniformRouter, (i,))['ba']
        own = sub.balanced_accuracy(targets, logits[:, i].argmax(axis=1))
        assert abs(got - own) < 1e-12, (i, got, own)
    print("  ✅ size-1 subsets equal the individual experts")


def test_evaluation_uses_only_the_subset_columns():
    """A weak expert's presence must not change a subset that excludes it."""
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    ab = a.evaluate_subset(UniformRouter, (0, 1))
    # blank out expert 2 and 3 entirely; the (0,1) result must be unchanged
    masked = logits.copy()
    masked[:, 2:] = 1e6
    b = sub.SubsetEnsembleAnalysis(masked, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    assert abs(ab['ba'] - b.evaluate_subset(UniformRouter, (0, 1))['ba']) < 1e-12
    print("  ✅ excluded experts cannot influence a subset's result")


def test_rule_can_be_swapped():
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    u = a.evaluate_subset(UniformRouter, (0, 1, 2, 3))['ba']
    p = a.evaluate_subset(ProbabilityAverageRouter, (0, 1, 2, 3))['ba']
    assert u != p, "the analysis did not actually switch rules"
    print(f"  ✅ rule swap works (uniform {u:.4f} vs probability {p:.4f})")


# ---------------------------------------------------------------------------
# Sizing table and the selection-bias warning
# ---------------------------------------------------------------------------

def test_size_table_reports_mean_and_best_separately():
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    table = a.size_table(UniformRouter)
    assert set(table) == {1, 2, 3, 4}, sorted(table)
    for size, row in table.items():
        for key in ('mean_ba', 'best_ba', 'worst_ba', 'best_subset', 'n_subsets'):
            assert key in row, f"size {size} missing '{key}': {sorted(row)}"
        assert row['best_ba'] >= row['mean_ba'] >= row['worst_ba'], row
        assert row['n_subsets'] == len(a.subsets(size)), row
    print("  ✅ size table: " + ", ".join(
        f"k={k}: {v['mean_ba']:.4f} (best {v['best_ba']:.4f})"
        for k, v in table.items()))


def test_best_subset_is_flagged_optimistic():
    """The analysis must label its best-of number as selection-on-test."""
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    table = a.size_table(UniformRouter)
    assert table[3]['optimistic'] is True, "best-of-N was not flagged"
    assert 'selection' in sub.SELECTION_WARNING.lower(), sub.SELECTION_WARNING
    print(f"  ✅ best-of-N flagged optimistic: {sub.SELECTION_WARNING[:58]}...")


def test_best_subset_really_is_the_best():
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    table = a.size_table(UniformRouter)
    for size, row in table.items():
        scores = {s: a.evaluate_subset(UniformRouter, s)['ba'] for s in a.subsets(size)}
        best = max(scores.values())
        assert abs(row['best_ba'] - best) < 1e-12, (size, row['best_ba'], best)
        assert abs(scores[row['best_subset']] - best) < 1e-12, (size, row)
    print("  ✅ best_subset is genuinely the argmax for every size")


def test_aggregate_across_seeds_is_supported():
    logits, targets = _fake_logits()
    a = sub.SubsetEnsembleAnalysis(logits, targets, expert_names=list('ABCD'),
                                   class_counts=COUNTS)
    multi = sub.aggregate_size_tables([a.size_table(UniformRouter) for _ in range(3)])
    assert multi[2]['mean_ba']['n'] == 3, multi[2]
    assert 'std' in multi[2]['mean_ba'], multi[2]
    print("  ✅ size tables aggregate across seeds")


TESTS = [
    ("Subsets by size", test_subsets_are_enumerated_by_size),
    ("Subsets are index tuples", test_subsets_are_index_tuples_in_order),
    ("Full subset == pool", test_full_subset_reproduces_the_full_pool_result),
    ("Size-1 == single expert", test_single_expert_subset_equals_that_expert),
    ("Only subset columns used", test_evaluation_uses_only_the_subset_columns),
    ("Rule can be swapped", test_rule_can_be_swapped),
    ("Size table mean vs best", test_size_table_reports_mean_and_best_separately),
    ("Best flagged optimistic", test_best_subset_is_flagged_optimistic),
    ("Best is really best", test_best_subset_really_is_the_best),
    ("Aggregate across seeds", test_aggregate_across_seeds_is_supported),
]


def main() -> int:
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 - surface the real cause
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 62}")
    print(f"  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
