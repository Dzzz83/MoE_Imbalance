"""
Tests for real test-time augmentation (TTA).

The pre-registration defines the fourth routing rule as "`ConfidenceRouter` over
TTA-averaged logits". An earlier version of the code never implemented the
"TTA-averaged" part: `TTARouter` simply delegated to its base router on the same
single-view logits, so the TTA row was **bit-identical to the Confidence row**
and measured nothing about TTA. These tests pin the real behaviour.

Design choices pinned here:
  * views are generated with the training augmentation (random crop, pad 4, and
    horizontal flip), so TTA sees the distribution the model was trained on;
  * predictions are averaged in **probability** space, not logit space, because
    logits are scale-sensitive and one overconfident view would dominate;
  * the averaged distribution is returned as log-probabilities, so the existing
    routers (which apply `softmax` to their input) recover exactly the average.
"""

import os
import sys
import tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch

import importlib

from scripts import evaluation as ev

# Guarded so a not-yet-written module fails one test at a time, not the file.
try:
    _tta_mod = importlib.import_module('data.tta')
except ImportError:
    _tta_mod = None
AugmentationViews = getattr(_tta_mod, 'AugmentationViews', None)


def _batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=g)


# ---------------------------------------------------------------------------
# View generation
# ---------------------------------------------------------------------------

def test_views_preserve_shape():
    views = AugmentationViews(n_views=5, seed=0)(_batch())
    assert len(views) == 5, len(views)
    for v in views:
        assert v.shape == (4, 3, 32, 32), v.shape
    print(f"  ✅ {len(views)} views, shapes preserved")


def test_views_actually_augment():
    """Every view must differ from the original — otherwise TTA is a no-op."""
    x = _batch()
    views = AugmentationViews(n_views=4, seed=0)(x)
    identical = sum(1 for v in views if torch.allclose(v, x))
    assert identical == 0, f"{identical}/4 views are identical to the input"
    print("  ✅ all views differ from the original")


def test_views_are_seeded_and_reproducible():
    """Same seed -> identical views; different seed -> different views."""
    x = _batch()
    a = AugmentationViews(n_views=4, seed=7)(x)
    b = AugmentationViews(n_views=4, seed=7)(x)
    c = AugmentationViews(n_views=4, seed=8)(x)
    for va, vb in zip(a, b):
        assert torch.equal(va, vb), "same seed gave different views"
    assert any(not torch.equal(va, vc) for va, vc in zip(a, c)), \
        "different seeds gave identical views"
    print("  ✅ views reproducible under a seed, and seed-dependent")


def test_crop_keeps_values_in_range():
    """Cropping/padding must not create out-of-range pixel values."""
    x = _batch()
    views = AugmentationViews(n_views=3, seed=0)(x)
    for v in views:
        assert torch.isfinite(v).all(), "view contains non-finite values"
    print("  ✅ views finite")


# ---------------------------------------------------------------------------
# Averaging: probability space, not logit space
# ---------------------------------------------------------------------------

def test_averaging_happens_in_probability_space():
    """The average must be log(mean softmax), not softmax(mean logits)."""
    rng = np.random.default_rng(0)
    # one very overconfident view and one flat view: the two rules disagree
    v1 = np.zeros((2, 3, 5), dtype=np.float64)
    v1[:, 0, 0] = 50.0                      # extremely confident on class 0
    v2 = np.zeros((2, 3, 5), dtype=np.float64)   # uniform

    got = ev.tta_average_log_probs([v1, v2])

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    expected_prob = 0.5 * softmax(v1) + 0.5 * softmax(v2)
    expected = np.log(expected_prob)

    logit_average = np.log(softmax(0.5 * v1 + 0.5 * v2))

    assert np.allclose(got, expected, atol=1e-9), "not averaging probabilities"
    assert not np.allclose(got, logit_average, atol=1e-3), \
        "result matches logit-averaging — the wrong space"
    print("  ✅ averages softmax probabilities (not logits)")


def test_averaged_log_probs_recover_the_mean_distribution():
    """softmax(returned) must equal the mean of the per-view probabilities."""
    rng = np.random.default_rng(1)
    views = [rng.normal(size=(3, 2, 7)) for _ in range(5)]
    got = ev.tta_average_log_probs(views)

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    mean_prob = np.mean([softmax(v) for v in views], axis=0)
    assert np.allclose(softmax(got), mean_prob, atol=1e-9)
    assert np.allclose(softmax(got).sum(axis=-1), 1.0, atol=1e-9)
    print("  ✅ softmax(averaged) == mean of per-view probabilities")


def test_single_view_average_is_a_no_op():
    """With one view, the average must reproduce that view's distribution."""
    rng = np.random.default_rng(2)
    v = rng.normal(size=(2, 3, 6))

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    got = ev.tta_average_log_probs([v])
    assert np.allclose(softmax(got), softmax(v), atol=1e-9)
    print("  ✅ single-view TTA is a no-op")


# ---------------------------------------------------------------------------
# ExpertPool integration
# ---------------------------------------------------------------------------

def _fake_checkpoint(directory, label, seed=78):
    from models.resnet32 import ResNet32
    path = os.path.join(directory, f'{label}_seed{seed}_final.pt')
    torch.save({
        'epoch': 200, 'seed': seed, 'expert_name': label,
        'model_state_dict': ResNet32(num_classes=100).state_dict(),
        'optimiser_state_dict': {}, 'is_final': True, 'log': {},
    }, path)


def _pool(labels=('CE', 'LAL')):
    d = tempfile.mkdtemp(prefix='dsh_tta_')
    for label in labels:
        _fake_checkpoint(d, label)
    return ev.ExpertPool(list(labels), seeds=[78], checkpoint_dir=d, device='cpu').load()


def _loader(n=4):
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    return DataLoader(
        TensorDataset(torch.randn(n, 3, 32, 32, generator=g),
                      torch.randint(0, 100, (n,), generator=g)),
        batch_size=n,
    )


def test_pool_tta_changes_the_logits():
    """The bug: TTA logits used to be identical to plain logits."""
    pool = _pool()
    loader = _loader()
    plain, _ = pool.logits(loader)
    tta, _ = pool.logits(loader, n_augs=4, tta_seed=0)
    assert plain.shape == tta.shape, f"{plain.shape} vs {tta.shape}"
    assert not np.allclose(plain, tta, atol=1e-6), \
        "TTA logits are identical to plain logits — TTA is not being applied"
    print("  ✅ TTA logits differ from plain logits")


def test_pool_tta_is_reproducible():
    pool = _pool()
    loader = _loader()
    a, _ = pool.logits(loader, n_augs=3, tta_seed=5)
    b, _ = pool.logits(loader, n_augs=3, tta_seed=5)
    assert np.allclose(a, b, atol=1e-12), "TTA is not reproducible under a seed"
    print("  ✅ TTA reproducible under a seed")


def test_pool_single_view_matches_plain_inference():
    """n_augs=1 must be exactly the plain path, not an approximation."""
    pool = _pool()
    loader = _loader()
    plain, _ = pool.logits(loader)
    one, _ = pool.logits(loader, n_augs=1)
    assert np.allclose(plain, one, atol=1e-9), \
        "n_augs=1 should reproduce plain inference exactly"
    print("  ✅ n_augs=1 == plain inference")


def test_pool_tta_probabilities_are_a_valid_distribution():
    pool = _pool()
    tta, _ = pool.logits(_loader(), n_augs=3, tta_seed=0)
    probs = np.exp(tta)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6), \
        "TTA output is not a normalised distribution"
    print("  ✅ TTA output normalises to 1")


# ---------------------------------------------------------------------------
# The router must declare that it needs TTA logits
# ---------------------------------------------------------------------------

def test_tta_router_declares_the_requirement():
    """Only TTARouter may claim to need TTA logits."""
    from scripts.router import ROUTERS
    for name, klass in ROUTERS.items():
        expected = (name == 'TTA')
        assert klass.requires_tta is expected, (
            f"{name}.requires_tta is {klass.requires_tta}, expected {expected}"
        )
    print("  ✅ only TTARouter declares requires_tta")


def test_non_tta_routers_default_to_false():
    from scripts.router.base import BaseRouter
    assert BaseRouter.requires_tta is False
    print("  ✅ BaseRouter defaults requires_tta=False")


def test_evaluator_routes_each_rule_to_the_right_logits():
    """The evaluator must hand TTA rules the TTA pass, and others the plain pass."""
    from scripts import evaluate_experts as evr
    from scripts.router import ROUTERS

    plain = np.zeros((2, 3, 4))
    tta = np.ones((2, 3, 4))
    for name, klass in ROUTERS.items():
        got = evr.select_logits(klass, plain, tta)
        expected = tta if klass.requires_tta else plain
        assert np.array_equal(got, expected), f"{name} got the wrong logits"
    print("  ✅ each rule receives the correct logits")

    # A TTA rule with no TTA pass must fail loudly, not silently fall back.
    try:
        evr.select_logits(ROUTERS['TTA'], plain, None)
    except ev.EvaluationError as e:
        print(f"  ✅ TTA rule without a TTA pass is rejected: {str(e)[:60]}")
        return
    raise AssertionError("a TTA rule silently fell back to plain logits")


TESTS = [
    ("Views preserve shape", test_views_preserve_shape),
    ("Views actually augment", test_views_actually_augment),
    ("Views are seeded", test_views_are_seeded_and_reproducible),
    ("Views finite", test_crop_keeps_values_in_range),
    ("Averaging in probability space", test_averaging_happens_in_probability_space),
    ("Averaged recovers mean distribution", test_averaged_log_probs_recover_the_mean_distribution),
    ("Single view is a no-op", test_single_view_average_is_a_no_op),
    ("Pool TTA changes logits", test_pool_tta_changes_the_logits),
    ("Pool TTA reproducible", test_pool_tta_is_reproducible),
    ("Pool n_augs=1 == plain", test_pool_single_view_matches_plain_inference),
    ("Pool TTA normalised", test_pool_tta_probabilities_are_a_valid_distribution),
    ("TTARouter declares requires_tta", test_tta_router_declares_the_requirement),
    ("BaseRouter defaults False", test_non_tta_routers_default_to_false),
    ("Evaluator routes logits per rule", test_evaluator_routes_each_rule_to_the_right_logits),
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
