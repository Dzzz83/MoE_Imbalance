"""
Synthetic dry-run for DACE Evaluation (prototype-based routing).

This file used to re-IMPLEMENT compute_prototypes / prototype_routing inside
the test, which meant the shipped code in scripts/evaluate_dace.py was never
exercised — and a device bug reached Kaggle undetected.  It now imports the
real functions, so what is tested is what actually runs.

Retained here: the metric helpers and the end-to-end pipeline sanity checks.
Real-function regression coverage lives in tests/test_evaluate_dace_real.py.

Kaggle-Gate compliance (AGENTs.md §11):
  - Runs entirely on CPU
  - No CIFAR-100 download needed
  - No pre-trained checkpoints required
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.resnet32 import ResNet32, ResNet32WithRouting

# Real shipped functions — no local copies
from scripts.evaluate_dace import (
    compute_prototypes,
    prototype_routing,
    balanced_accuracy,
    group_accuracies,
    get_class_groups,
)


def test_prototype_computation_shapes():
    """Prototype computation returns correctly shaped prototypes."""
    torch.manual_seed(0)
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()

    fake_images = torch.randn(16, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (16,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=8)

    prototypes = compute_prototypes(
        expert_a, expert_b, expert_c, val_loader, device='cpu',
    )

    for key in ['A', 'B', 'C']:
        assert 'agree' in prototypes[key]
        assert 'disagree' in prototypes[key]
        assert prototypes[key]['agree'].shape == (32,), \
            f"{key} agree shape: {prototypes[key]['agree'].shape}"
        assert prototypes[key]['disagree'].shape == (32,), \
            f"{key} disagree shape: {prototypes[key]['disagree'].shape}"

    print("  ✅ Prototypes computed correctly (A, B, C — 32-d each)")


def test_routing_output_shapes():
    """Prototype routing produces correct output shapes."""
    torch.manual_seed(0)
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()

    fake_images = torch.randn(8, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (8,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=8)

    prototypes = compute_prototypes(
        expert_a, expert_b, expert_c, val_loader, device='cpu',
    )

    with torch.no_grad():
        logits_a = expert_a(fake_images)
        logits_b, emb_b = expert_b(fake_images)
        logits_c, emb_c = expert_c(fake_images)

    final_logits, scores, avg_agreement, best_expert = prototype_routing(
        logits_a, logits_b, logits_c, emb_b, emb_c, prototypes,
    )

    assert final_logits.shape == (8, 100), f"Final logits: {final_logits.shape}"
    assert scores.shape == (8, 3), f"Scores: {scores.shape}"
    assert avg_agreement.shape == (8,), f"Agreement: {avg_agreement.shape}"
    assert best_expert.shape == (8,), f"Best expert: {best_expert.shape}"
    assert set(best_expert.tolist()).issubset({0, 1, 2})

    print("  ✅ Routing output shapes correct")


def test_uniform_fallback():
    """When embeddings are identical, uniform fallback is used."""
    torch.manual_seed(0)
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = expert_b  # same model -> identical embeddings

    fake_images = torch.randn(4, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (4,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=4)

    prototypes = compute_prototypes(
        expert_a, expert_b, expert_c, val_loader, device='cpu',
    )

    with torch.no_grad():
        logits_a = expert_a(fake_images)
        logits_b, emb_b = expert_b(fake_images)
        logits_c, emb_c = expert_c(fake_images)

    final_logits, scores, avg_agreement, best_expert = prototype_routing(
        logits_a, logits_b, logits_c, emb_b, emb_c, prototypes,
        threshold_agree=0.5,
    )

    uniform_logits = (logits_a + logits_b + logits_c) / 3.0
    assert torch.allclose(final_logits, uniform_logits, atol=1e-5), \
        "Should use uniform when embeddings are identical"
    print(f"  ✅ Uniform fallback works (cosine_sim={avg_agreement.mean():.4f})")


def test_routing_decisions_valid():
    """Routing decisions are valid (0, 1, or 2) for all samples."""
    torch.manual_seed(0)
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()

    fake_images = torch.randn(4, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (4,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=4)

    prototypes = compute_prototypes(
        expert_a, expert_b, expert_c, val_loader, device='cpu',
    )

    with torch.no_grad():
        logits_a = expert_a(fake_images)
        logits_b, emb_b = expert_b(fake_images)
        logits_c, emb_c = expert_c(fake_images)

    _, _, _, best_expert = prototype_routing(
        logits_a, logits_b, logits_c, emb_b, emb_c, prototypes,
    )

    assert all(d in [0, 1, 2] for d in best_expert.tolist())
    print(f"  ✅ Routing decisions valid: {best_expert.tolist()}")


def test_class_groups_from_real_helper():
    """get_class_groups splits classes by frequency on a synthetic fixture.

    The fixture below is 30/36/34 *by construction*, so it proves only that the
    thresholds are applied as written — it says nothing about the real split,
    which is 35/35/30 (asserted in tests/test_protocol_splits.py). Do not quote
    these numbers as the project's class-group sizes.
    """
    counts = np.array([410] * 30 + [50] * 36 + [5] * 34, dtype=np.int64)
    groups = get_class_groups(counts)

    assert len(groups['Head']) == 30, groups['Head']
    assert len(groups['Med']) == 36, groups['Med']
    assert len(groups['Tail']) == 34, groups['Tail']
    print("  ✅ synthetic fixture splits 30/36/34 by construction "
          "(the real split is 35/35/30)")


def test_metrics_computation():
    """Balanced accuracy and group accuracies compute without error."""
    rng = np.random.default_rng(0)
    all_targets = rng.integers(0, 100, size=1000)
    all_preds = rng.integers(0, 100, size=1000)

    ba = balanced_accuracy(all_targets, all_preds)
    assert 0.0 <= ba <= 1.0, f"BA should be in [0,1], got {ba}"

    groups = {
        'Head': np.arange(30),
        'Med': np.arange(30, 66),
        'Tail': np.arange(66, 100),
    }
    grp = group_accuracies(all_targets, all_preds, groups)
    for name in ['Head', 'Med', 'Tail']:
        assert 0.0 <= grp[name] <= 1.0, f"{name} acc: {grp[name]}"

    print(f"  ✅ Metrics: BA={ba:.2%}, Head={grp['Head']:.2%}, "
          f"Med={grp['Med']:.2%}, Tail={grp['Tail']:.2%}")


if __name__ == "__main__":
    print("=" * 50)
    print("  DACE Evaluation — Synthetic Dry-Run (real functions)")
    print("=" * 50)
    print()

    tests = [
        ("Prototype computation shapes", test_prototype_computation_shapes),
        ("Routing output shapes", test_routing_output_shapes),
        ("Uniform fallback", test_uniform_fallback),
        ("Routing decisions valid", test_routing_decisions_valid),
        ("Class groups from real helper", test_class_groups_from_real_helper),
        ("Metrics computation", test_metrics_computation),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
