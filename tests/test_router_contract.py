"""
Contract tests for the parameter-free router framework.

The project has no validation split. Experts train on the full long-tailed
training set, so there is no honest held-out data anywhere to fit a router on.
Rather than trusting discipline, the framework is built so that fitting on
held-out labels is **structurally impossible**:

  * the router interface has no `train()` method
  * no module under scripts/router/ may reference val_labels / val_logits
  * only parameter-free mechanisms are registered
  * the five fitted routers no longer exist

Run directly:
    python tests/test_router_contract.py
"""

import glob
import importlib
import inspect
import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np

from scripts import router as router_pkg

ROUTER_DIR = os.path.join(_proj_root, 'scripts', 'router')

#: Mechanisms removed because their fitting step needed a validation split.
DELETED_MODULES = ['correctness', 'pairwise', 'cluster', 'gate', 'selective']

#: Removed as mathematically redundant: the same classifier as logit averaging.
REDUNDANT_MODULES = ['product']

#: The full surviving registry. Parameter-free, or parameter-free by construction.
EXPECTED_REGISTRY = {'Uniform', 'Probability', 'Confidence', 'TTA'}

#: The measured results of the deleted mechanisms live here. `docs/` is a
#: tracked in records/, which IS committed, so these normally run; the skip is a
#: safety net for a local checkout that has deleted them.
RESULTS_RECORD = os.path.join(_proj_root, 'records', 'routing_mechanism.md')
PREREGISTRATION = os.path.join(_proj_root, 'records', 'routing-preregistration.md')


class _SkipTest(Exception):
    """Raised when a check cannot run in this checkout.

    A distinct type, not a return value: the runner counts a normal return as a
    pass, so returning here would report the pre-registration check as green
    even when the record is missing.
    """


def _skip_if_absent(path: str) -> bool:
    """False when the doc is present; otherwise skip the test explicitly."""
    if os.path.exists(path):
        return False
    raise _SkipTest(
        f"local-only doc not present: {os.path.relpath(path, _proj_root)}"
    )


def _synthetic_logits(n=16, k=3, c=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k, c)).astype(np.float32)


# ---------------------------------------------------------------------------
# Structural: no fitting anywhere
# ---------------------------------------------------------------------------

def test_no_router_mentions_held_out_labels():
    """No router module may reference validation labels or logits."""
    offenders = {}
    for path in sorted(glob.glob(os.path.join(ROUTER_DIR, '*.py'))):
        with open(path) as f:
            src = f.read()
        hits = [tok for tok in ('val_labels', 'val_logits', 'val_features')
                if tok in src]
        if hits:
            offenders[os.path.basename(path)] = hits
    assert not offenders, (
        f"router modules still reference held-out data: {offenders}"
    )
    print("  ✅ no router module references held-out labels/logits")


def test_base_router_has_no_train_method():
    """The interface must expose no fitting entry point."""
    from scripts.router.base import BaseRouter
    assert not hasattr(BaseRouter, 'train'), \
        "BaseRouter still exposes train() — fitting on held-out data is possible"
    methods = {n for n, _ in inspect.getmembers(BaseRouter, inspect.isfunction)}
    for banned in ('fit', 'calibrate', 'tune'):
        assert banned not in methods, f"BaseRouter exposes '{banned}()'"
    print(f"  ✅ BaseRouter has no fitting method (has: {sorted(methods)})")


def test_no_router_module_defines_a_fit_like_method():
    """No concrete router may grow its own fitting method."""
    banned_substrings = ('def train(', 'def fit(', 'def calibrate(',
                         'def tune', '.fit(')
    offenders = {}
    for path in sorted(glob.glob(os.path.join(ROUTER_DIR, '*.py'))):
        with open(path) as f:
            src = f.read()
        hits = [b for b in banned_substrings if b in src]
        if hits:
            offenders[os.path.basename(path)] = hits
    assert not offenders, f"fitting logic still present: {offenders}"
    print("  ✅ no router defines fitting logic")


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------

def test_fitted_routers_are_deleted():
    """The five val-dependent mechanisms must be gone from disk."""
    present = [m for m in DELETED_MODULES
               if os.path.exists(os.path.join(ROUTER_DIR, f'{m}.py'))]
    assert not present, f"fitted router modules still on disk: {present}"
    print(f"  ✅ deleted modules absent: {DELETED_MODULES}")


def test_registry_is_exactly_the_parameter_free_set():
    """Only parameter-free mechanisms may be registered."""
    from scripts.router import ROUTERS
    assert set(ROUTERS) == EXPECTED_REGISTRY, (
        f"registry {sorted(ROUTERS)} != expected {sorted(EXPECTED_REGISTRY)}"
    )
    print(f"  ✅ registry: {sorted(ROUTERS)}")


def test_removed_routers_are_not_importable():
    """Importing a deleted mechanism must fail loudly, not silently resolve."""
    for name in DELETED_MODULES:
        try:
            importlib.import_module(f'scripts.router.{name}')
        except ImportError:
            continue
        raise AssertionError(f"scripts.router.{name} is still importable")
    print("  ✅ deleted routers are not importable")


def test_confidence_router_has_no_calibration():
    """Confidence routing must be the raw, parameter-free variant."""
    from scripts.router import ConfidenceRouter
    params = inspect.signature(ConfidenceRouter.__init__).parameters
    assert 'calibrate' not in params, \
        "ConfidenceRouter still exposes a calibrate flag"
    r = ConfidenceRouter(expert_names=['A', 'B', 'C'])
    assert not hasattr(r, 'temperatures'), \
        "ConfidenceRouter still carries fitted temperatures"
    # A bare construction must fail rather than silently assume an expert pool.
    try:
        ConfidenceRouter()
    except TypeError:
        pass
    else:
        raise AssertionError("ConfidenceRouter() invented an expert pool")
    print("  ✅ ConfidenceRouter is parameter-free (no calibration)")


# ---------------------------------------------------------------------------
# Behaviour: every router predicts with no fitting step
# ---------------------------------------------------------------------------

def test_every_router_predicts_without_fitting():
    """Freshly constructed routers must predict immediately."""
    from scripts.router import ROUTERS

    logits = _synthetic_logits()
    for name, klass in ROUTERS.items():
        r = klass(expert_names=['A', 'B', 'C'])
        idx = r.predict(logits)
        assert idx.shape == (16,), f"{name}: predict shape {idx.shape}"
        assert idx.min() >= 0 and idx.max() < 3, f"{name}: expert index out of range"
        labels = r.predict_class(logits)
        assert labels.shape == (16,), f"{name}: predict_class shape {labels.shape}"
        w = r.predict_proba(logits)
        assert w.shape == (16, 3), f"{name}: predict_proba shape {w.shape}"
        print(f"  ✅ {name}: predicts without fitting")


def test_uniform_router_equals_logit_averaging():
    """Uniform routing must be exactly the logit-averaging baseline."""
    from scripts.router import UniformRouter
    logits = _synthetic_logits(seed=3)
    r = UniformRouter(expert_names=['A', 'B', 'C'])
    expected = logits.mean(axis=1).argmax(axis=1)
    assert np.array_equal(r.predict_class(logits), expected), \
        "uniform router is not logit averaging"
    w = r.predict_proba(logits)
    assert np.allclose(w, 1.0 / 3), "uniform router weights are not uniform"
    print("  ✅ UniformRouter == logit averaging")


def test_product_is_provably_the_logit_average():
    """Why ProductRouter was removed: it is the same classifier as UniformRouter.

        prod_e softmax(z_e)_c = exp(sum_e z_{e,c}) / prod_e sum_c' exp(z_{e,c'})

    The denominator does not depend on the class c, so the argmax over c equals
    the argmax of the mean logits — exactly what UniformRouter computes. This is
    the justification for deleting the rule, kept as a test rather than a note.
    """
    from scripts.utils.features import softmax

    logits = _synthetic_logits(n=64, seed=21)
    product = np.prod(softmax(logits), axis=1).argmax(axis=1)
    logit_average = logits.mean(axis=1).argmax(axis=1)
    agree = float((product == logit_average).mean())
    assert agree == 1.0, (
        f"product and logit averaging disagreed on {(1-agree)*100:.1f}% of samples — "
        f"the redundancy claim is wrong"
    )
    print(f"  OK product == logit average on 64/64 samples (identical classifier)")


def test_redundant_router_module_is_removed():
    """The redundant module must be gone from disk and unimportable."""
    import importlib

    for name in REDUNDANT_MODULES:
        path = os.path.join(ROUTER_DIR, f'{name}.py')
        assert not os.path.exists(path), f"{path} still on disk"
        try:
            importlib.import_module(f'scripts.router.{name}')
        except ImportError:
            continue
        raise AssertionError(f"scripts.router.{name} is still importable")
    print(f"  OK removed as redundant: {REDUNDANT_MODULES}")


def test_routers_are_deterministic():
    """Two identical calls must give identical predictions (no hidden state)."""
    from scripts.router import ROUTERS
    logits = _synthetic_logits(seed=5)
    for name, klass in ROUTERS.items():
        r = klass(expert_names=['A', 'B', 'C'])
        a = r.predict_class(logits)
        b = r.predict_class(logits)
        assert np.array_equal(a, b), f"{name}: non-deterministic predictions"
    print("  ✅ all routers deterministic across repeated calls")


# ---------------------------------------------------------------------------
# Documentation records
# ---------------------------------------------------------------------------

def test_results_record_exists_and_names_every_deleted_mechanism():
    """The results of every deleted mechanism must be recorded before deletion."""
    if _skip_if_absent(RESULTS_RECORD):
        return
    with open(RESULTS_RECORD) as f:
        text = f.read().lower()
    missing = [m for m in DELETED_MODULES if m not in text]
    assert not missing, f"results record does not mention: {missing}"
    for term in ('correctness', 'pairwise', 'cluster', 'gate', 'selective'):
        assert term in text, f"record missing '{term}'"
    print("  ✅ results record covers every deleted mechanism")


def test_preregistration_exists_before_any_test_evaluation():
    """The candidate set must be frozen before the test set is touched."""
    if _skip_if_absent(PREREGISTRATION):
        return
    with open(PREREGISTRATION) as f:
        text = f.read()
    for rule in ('UniformRouter', 'ProbabilityAverageRouter',
                 'ConfidenceRouter', 'TTARouter'):
        assert rule in text, f"preregistration does not name {rule}"
    assert 'Amendment' in text, "the post-hoc baseline addition is not recorded"
    assert 'Tail' in text, "preregistration must state the Tail-accuracy criterion"
    print("  ✅ pre-registration names the frozen candidate set")


# ---------------------------------------------------------------------------
# Probability averaging — the standard soft-vote ensemble
# ---------------------------------------------------------------------------

def test_probability_average_equals_mean_of_softmax():
    """The rule must be argmax of the mean softmax probability (a soft vote)."""
    from scripts.router import ProbabilityAverageRouter
    from scripts.utils.features import softmax

    logits = _synthetic_logits(seed=11)
    r = ProbabilityAverageRouter(expert_names=['A', 'B', 'C'])
    expected = softmax(logits).mean(axis=1).argmax(axis=1)
    assert np.array_equal(r.predict_class(logits), expected), \
        "probability averaging is not the mean of softmax probabilities"
    print("  ✅ ProbabilityAverage == argmax(mean softmax)")


def test_probability_average_resists_a_dominant_expert():
    """Why the two baselines differ: a logit average can be captured by one
    high-magnitude expert; a probability average cannot.

    Expert A is a confident outlier with huge logits; experts B and C agree with
    each other on a different class. The logit average follows the outlier
    because its magnitudes dominate the sum. The probability average follows the
    two experts who agree, because each contributes a distribution bounded by 1.
    """
    from scripts.router import ProbabilityAverageRouter, UniformRouter

    logits = np.zeros((1, 3, 3), dtype=np.float32)
    logits[0, 0, 0] = 100.0        # expert A: huge magnitude, says class 0
    logits[0, 1, 1] = 10.0         # expert B says class 1
    logits[0, 2, 1] = 10.0         # expert C says class 1

    logit_pred = UniformRouter(expert_names=['A', 'B', 'C']).predict_class(logits)[0]
    prob_pred = ProbabilityAverageRouter(expert_names=['A', 'B', 'C']).predict_class(logits)[0]

    assert logit_pred == 0, f"expected the outlier to capture the logit average, got {logit_pred}"
    assert prob_pred == 1, f"expected the majority to win the probability average, got {prob_pred}"
    print("  OK logit average captured by the outlier; probability average follows the majority")


def test_probability_average_is_not_a_duplicate_of_logit_average():
    """The two must be genuinely different rules, unlike Product == Uniform."""
    from scripts.router import ProbabilityAverageRouter, UniformRouter

    logits = _synthetic_logits(n=200, seed=13)
    logits[:, 0] *= 5.0            # mimic the real pool's 2.3x scale disparity
    a = UniformRouter(expert_names=['A', 'B', 'C']).predict_class(logits)
    b = ProbabilityAverageRouter(expert_names=['A', 'B', 'C']).predict_class(logits)
    differing = int((a != b).sum())
    assert differing > 0, "probability and logit averaging gave identical predictions"
    print(f"  ✅ the two baselines differ on {differing}/200 samples")


def test_probability_average_predict_proba_is_uniform():
    from scripts.router import ProbabilityAverageRouter
    r = ProbabilityAverageRouter(expert_names=['A', 'B', 'C'])
    w = r.predict_proba(_synthetic_logits())
    assert w.shape == (16, 3), w.shape
    assert np.allclose(w, 1 / 3), "equal-weight average should report uniform weights"
    print("  ✅ equal weights reported")


def test_probability_average_needs_no_tta():
    from scripts.router import ProbabilityAverageRouter
    assert ProbabilityAverageRouter.requires_tta is False
    print("  ✅ requires_tta False")


def _expert_0_is_wrong_logits(seed=0):
    """(logits, labels) where experts 1-2 are right and expert 0 is wrong."""
    rng = np.random.default_rng(seed)
    n, c = 300, 100
    labels = rng.integers(0, c, n)
    logits = (rng.normal(size=(n, 3, c)) * 0.1).astype(np.float32)
    logits[np.arange(n), 1, labels] += 10.0
    logits[np.arange(n), 2, labels] += 10.0
    logits[np.arange(n), 0, (labels + 1) % c] += 10.0
    return logits, labels


def test_every_registered_rule_evaluates_its_own_predictions():
    """evaluate() must score the rule's real decisions — for every registered rule.

    Uniform and Probability averaging make no per-sample expert choice: their
    ``predict()`` returns expert 0 as a sentinel, and ``compute_routing_metrics``
    read that sentinel literally, so both reported expert 0's accuracy under the
    rule's name (plus "100% usage of expert A").

    Iterating the whole registry — rather than the two rules known to be
    affected — is the point: a future combine-then-argmax rule that forgets
    ``selects_single_expert = False`` fails here instead of silently reporting
    expert 0's numbers.
    """
    from scripts.evaluation import balanced_accuracy
    from scripts.router import ROUTERS

    logits, labels = _expert_0_is_wrong_logits()
    for name, klass in ROUTERS.items():
        rule = klass(expert_names=['A', 'B', 'C'])
        expected = balanced_accuracy(labels, rule.predict_class(logits))
        report = rule.evaluate(logits, labels)
        assert abs(report['ba'] - expected) < 1e-12, (
            f"{name}.evaluate() reported BA {report['ba']:.4f}, but the rule's own "
            f"predictions score {expected:.4f} — a sentinel predict() was read as a "
            f"decision (combine-then-argmax rules must set "
            f"selects_single_expert = False)"
        )
    print(f"  ✅ all {len(ROUTERS)} registered rules evaluate their own predictions")


def test_combine_then_argmax_rules_report_equal_usage():
    """Combine-then-argmax rules must declare themselves and report equal usage.

    ``selects_single_expert`` only affects the *usage* report (accuracy is always
    taken from ``predict_class``), so a rule that forgets it would look right on
    BA while reporting "100% usage of expert A". The registry check below forces a
    deliberate decision whenever a rule is added or removed.
    """
    from scripts.router import ProbabilityAverageRouter, ROUTERS, UniformRouter

    combining = {name for name, klass in ROUTERS.items()
                 if not klass.selects_single_expert}
    assert combining == {'Uniform', 'Probability'}, (
        f"the set of combining rules is {sorted(combining)}; a new "
        f"combine-then-argmax rule must set selects_single_expert = False (and a "
        f"new selecting rule must keep it True) — update this test deliberately"
    )

    logits, labels = _expert_0_is_wrong_logits()
    for klass in (UniformRouter, ProbabilityAverageRouter):
        rule = klass(expert_names=['A', 'B', 'C'])
        usage = list(rule.evaluate(logits, labels)['expert_usage'].values())
        # predict_proba returns float32 weights, so the tolerance is 1e-5, not
        # 1e-9: the bug this pins produced 100.0/0.0/0.0, a 66-point error.
        assert all(abs(u - 100.0 / 3) < 1e-5 for u in usage), (
            f"{klass.__name__} reports usage {usage}; a combine-then-argmax rule "
            f"draws on every expert equally"
        )
    print("  ✅ combine-then-argmax rules declared and report equal usage")


def test_confidence_and_tta_select_the_most_confident_expert():
    """Exact decision values, not just 'the index is in range'."""
    from scripts.router import ConfidenceRouter, TTARouter

    n, c = 4, 5
    winners = [2, 0, 1, 0]
    logits = np.zeros((n, 3, c), dtype=np.float32)
    for i, expert in enumerate(winners):
        logits[i, expert, 0] = 5.0        # that expert is confidently class 0
        logits[i, expert, 1] = 1.0

    confidence = ConfidenceRouter(expert_names=['A', 'B', 'C'])
    tta = TTARouter(expert_names=['A', 'B', 'C'])
    assert confidence.predict(logits).tolist() == winners, confidence.predict(logits)
    assert tta.predict(logits).tolist() == winners, tta.predict(logits)
    assert confidence.predict_class(logits).tolist() == [0] * n

    # A single-expert rule reports hard usage, so collapse stays visible.
    usage = confidence.evaluate(logits, np.zeros(n, dtype=np.int64))['expert_usage']
    expected = {'A': 50.0, 'B': 25.0, 'C': 25.0}
    for name, value in expected.items():
        assert abs(usage[name] - value) < 1e-9, f"{name}: {usage} != {expected}"
    print(f"  ✅ confidence/TTA decisions exact; usage {usage}")


TESTS = [
    ("No held-out labels in routers", test_no_router_mentions_held_out_labels),
    ("BaseRouter has no train()", test_base_router_has_no_train_method),
    ("No fitting logic in routers", test_no_router_module_defines_a_fit_like_method),
    ("Fitted routers deleted", test_fitted_routers_are_deleted),
    ("Registry is parameter-free only", test_registry_is_exactly_the_parameter_free_set),
    ("Deleted routers not importable", test_removed_routers_are_not_importable),
    ("ConfidenceRouter uncalibrated", test_confidence_router_has_no_calibration),
    ("All routers predict without fitting", test_every_router_predicts_without_fitting),
    ("Uniform == logit averaging", test_uniform_router_equals_logit_averaging),
    ("Probability == mean softmax", test_probability_average_equals_mean_of_softmax),
    ("Probability resists a dominant expert", test_probability_average_resists_a_dominant_expert),
    ("Probability != logit average", test_probability_average_is_not_a_duplicate_of_logit_average),
    ("Probability weights uniform", test_probability_average_predict_proba_is_uniform),
    ("Probability needs no TTA", test_probability_average_needs_no_tta),
    ("Combine rules evaluate() correctly", test_every_registered_rule_evaluates_its_own_predictions),
    ("Combine rules report equal usage", test_combine_then_argmax_rules_report_equal_usage),
    ("Confidence/TTA decisions exact", test_confidence_and_tta_select_the_most_confident_expert),
    ("Product is the logit average", test_product_is_provably_the_logit_average),
    ("Redundant module removed", test_redundant_router_module_is_removed),
    ("Routers deterministic", test_routers_are_deterministic),
    ("Results record complete", test_results_record_exists_and_names_every_deleted_mechanism),
    ("Pre-registration exists", test_preregistration_exists_before_any_test_evaluation),
]


def main() -> int:
    passed = failed = skipped = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except _SkipTest as e:
            print(f"  ⊘ skipped {name} ({e})")
            skipped += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 - surface the real cause
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 62}")
    print(f"  {len(TESTS)} tests: {passed} passed, {skipped} skipped, {failed} failed")
    if skipped:
        print(f"  NOTE: {skipped} check(s) did not run — reported as skipped, "
              f"never as passed.")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
