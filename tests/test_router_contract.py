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

#: The full surviving registry. Parameter-free, or parameter-free by construction.
EXPECTED_REGISTRY = {'Uniform', 'Product', 'Confidence', 'TTA'}

#: The measured results of the deleted mechanisms live here.
RESULTS_RECORD = os.path.join(
    _proj_root, 'docs', 'routing-results-record.md')
PREREGISTRATION = os.path.join(
    _proj_root, 'docs', 'routing-preregistration.md')


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


def test_product_router_equals_geometric_mean():
    """Product routing must be the geometric mean of expert probabilities."""
    from scripts.router import ProductRouter
    from scripts.utils.features import softmax

    logits = _synthetic_logits(seed=4)
    r = ProductRouter(expert_names=['A', 'B', 'C'])
    probs = softmax(logits)
    expected = np.prod(probs, axis=1).argmax(axis=1)
    assert np.array_equal(r.predict_class(logits), expected), \
        "product router is not the geometric mean"
    print("  ✅ ProductRouter == geometric mean of probabilities")


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
    assert os.path.exists(RESULTS_RECORD), f"missing {RESULTS_RECORD}"
    with open(RESULTS_RECORD) as f:
        text = f.read().lower()
    missing = [m for m in DELETED_MODULES if m not in text]
    assert not missing, f"results record does not mention: {missing}"
    for term in ('correctness', 'pairwise', 'cluster', 'gate', 'selective'):
        assert term in text, f"record missing '{term}'"
    print("  ✅ results record covers every deleted mechanism")


def test_preregistration_exists_before_any_test_evaluation():
    """The candidate set must be frozen before the test set is touched."""
    assert os.path.exists(PREREGISTRATION), f"missing {PREREGISTRATION}"
    with open(PREREGISTRATION) as f:
        text = f.read()
    for rule in ('UniformRouter', 'ProductRouter', 'ConfidenceRouter', 'TTARouter'):
        assert rule in text, f"preregistration does not name {rule}"
    assert 'Tail' in text, "preregistration must state the Tail-accuracy criterion"
    print("  ✅ pre-registration names the frozen candidate set")


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
    ("Product == geometric mean", test_product_router_equals_geometric_mean),
    ("Routers deterministic", test_routers_are_deterministic),
    ("Results record complete", test_results_record_exists_and_names_every_deleted_mechanism),
    ("Pre-registration exists", test_preregistration_exists_before_any_test_evaluation),
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
