"""
Tests for losses/kl_divergence.py — KL / JS divergence and agreement labels.

Expected behavior:
  - KL(P || P) ≈ 0 for identical distributions
  - KL(P || Q) > 0 for different distributions
  - JS(P || Q) is symmetric and bounded [0, log(2)]
  - compute_agreement_label returns 0 for similar distributions, 1 for very different ones
"""

import sys
import os
import torch

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from losses.kl_divergence import (
    kl_divergence,
    symmetric_kl_divergence,
    js_divergence,
    compute_agreement_label,
)

# Fixtures below are random; seed them so a threshold assertion cannot
# flip between runs.
torch.manual_seed(0)


def test_kl_identical_distributions():
    """KL(P || P) should be ~0 for any distribution."""
    logits = torch.randn(4, 10)
    kl = kl_divergence(logits, logits, reduction="batch_mean")
    assert kl.item() < 1e-6, f"KL(P||P) should be ~0, got {kl.item()}"


def test_kl_different_distributions():
    """KL(P || Q) should be > 0 for different distributions."""
    logits_a = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)  # confident class 0
    logits_b = torch.tensor([[0.0, 2.0, 0.0, 0.0]], dtype=torch.float32)  # confident class 1
    kl = kl_divergence(logits_a, logits_b, reduction="batch_mean")
    assert kl.item() > 0.1, f"Different distributions should have KL > 0, got {kl.item()}"


def test_kl_asymmetry():
    """KL(P || Q) != KL(Q || P) in general (asymmetric).

    Use a sharp (low-entropy) P vs a diffuse (high-entropy) Q.
    The two KL directions should differ measurably.
    """
    # P is very confident in class 0; Q is uniform across 5 classes
    logits_a = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    logits_b = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    kl_ab = kl_divergence(logits_a, logits_b, reduction="none")
    kl_ba = kl_divergence(logits_b, logits_a, reduction="none")
    diff = (kl_ab - kl_ba).abs().item()
    assert diff > 1e-4, f"KL should be asymmetric, diff={diff}"


def test_kl_per_sample_return_shape():
    """'none' reduction returns (B,) tensor."""
    logits = torch.randn(8, 100)
    kl_ps = kl_divergence(logits, logits, reduction="none")
    assert kl_ps.shape == (8,), f"Expected (8,), got {kl_ps.shape}"


def test_symmetric_kl():
    """symmetric_kl should be ~2 * KL(P||Q) when KL(P||Q) ≈ KL(Q||P) is small."""
    logits = torch.randn(4, 10)
    sym_kl = symmetric_kl_divergence(logits, logits, reduction="batch_mean")
    assert sym_kl.item() < 1e-6, f"Symmetric KL(P,P) should be ~0, got {sym_kl.item()}"


def test_js_divergence():
    """JS(P || Q) should be symmetric and bounded [0, log(2)]."""
    logits_a = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32)
    logits_b = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float32)

    js_ab = js_divergence(logits_a, logits_b, reduction="batch_mean")
    js_ba = js_divergence(logits_b, logits_a, reduction="batch_mean")

    # Symmetry
    assert abs(js_ab.item() - js_ba.item()) < 1e-6, "JS should be symmetric"

    # Bounded by log(2) ≈ 0.693
    assert js_ab.item() < 0.7, f"JS should be <= log(2) ≈ 0.693, got {js_ab.item()}"

    # JS(P || P) = 0
    js_pp = js_divergence(logits_a, logits_a, reduction="batch_mean")
    assert js_pp.item() < 1e-6, f"JS(P,P) should be ~0, got {js_pp.item()}"


def test_agreement_label_agree():
    """Similar distributions → label = 0 (agree)."""
    logits = torch.randn(4, 10)
    labels = compute_agreement_label(logits, logits, threshold=0.5, metric="kl")
    assert (labels == 0).all(), "Identical distributions should be labeled 'agree'"


def test_agreement_label_disagree():
    """Very different distributions → label = 1 (disagree)."""
    logits_a = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32)
    logits_b = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float32)
    labels = compute_agreement_label(logits_a, logits_b, threshold=0.1, metric="kl")
    assert (labels == 1).all(), "Very different distributions should be labeled 'disagree'"


def test_agreement_label_all_metrics():
    """All metric options run without error."""
    logits_a = torch.randn(4, 10)
    logits_b = torch.randn(4, 10)
    for metric in ["kl", "sym_kl", "js"]:
        labels = compute_agreement_label(logits_a, logits_b, threshold=0.5, metric=metric)
        assert labels.shape == (4,)
        assert labels.dtype == torch.long


if __name__ == "__main__":
    # Run all tests and report
    tests = [
        ("KL identical distributions", test_kl_identical_distributions),
        ("KL different distributions", test_kl_different_distributions),
        ("KL asymmetry", test_kl_asymmetry),
        ("KL per-sample return shape", test_kl_per_sample_return_shape),
        ("Symmetric KL", test_symmetric_kl),
        ("JS divergence", test_js_divergence),
        ("Agreement label agree", test_agreement_label_agree),
        ("Agreement label disagree", test_agreement_label_disagree),
        ("Agreement label all metrics", test_agreement_label_all_metrics),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("  All tests passed!")
