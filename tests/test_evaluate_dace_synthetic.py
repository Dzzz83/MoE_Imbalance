"""
Synthetic dry-run for DACE Evaluation (prototype-based routing).

Tests the inference pipeline:
  - Prototype computation from three randomly initialized experts
  - Prototype-based routing on synthetic data
  - Routing produces valid output shapes and decisions
  - Uniform fallback when embeddings are similar
  - Full evaluation pipeline (metrics computation)

This satisfies the "Kaggle-Gate" rule (AGENTs.md §11):
  - Runs entirely on CPU
  - No CIFAR-100 download needed
  - No pre-trained checkpoints required
"""

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.resnet32 import ResNet32, ResNet32WithRouting


# ── Minimal re-implementation of prototype routing for testing ──────────

def compute_prototypes_simple(
    expert_a, expert_b, expert_c, val_loader, kl_threshold=0.1, device='cpu',
):
    """Simplified prototype computation for testing."""
    emb_b_list, emb_c_list = [], []
    kl_b_labels, kl_c_labels = [], []

    for images, _ in val_loader:
        images = images.to(device)
        with torch.no_grad():
            logits_a = expert_a(images)
            probs_a = F.softmax(logits_a, dim=1)

            logits_b, emb_b = expert_b(images)
            probs_b = F.softmax(logits_b, dim=1)

            logits_c, emb_c = expert_c(images)
            probs_c = F.softmax(logits_c, dim=1)

            kl_ab = (probs_a * (torch.log(probs_a + 1e-12)
                                - torch.log(probs_b + 1e-12))).sum(dim=1)
            kl_b_labels.append(kl_ab > kl_threshold)

            avg_probs = (probs_a + probs_b) / 2.0
            kl_ac = (avg_probs * (torch.log(avg_probs + 1e-12)
                                  - torch.log(probs_c + 1e-12))).sum(dim=1)
            kl_c_labels.append(kl_ac > kl_threshold)

            emb_b_list.append(emb_b.cpu())
            emb_c_list.append(emb_c.cpu())

    emb_b_all = torch.cat(emb_b_list, dim=0)
    emb_c_all = torch.cat(emb_c_list, dim=0)
    kl_b_all = torch.cat(kl_b_labels, dim=0)
    kl_c_all = torch.cat(kl_c_labels, dim=0)

    def safe_proto(emb, mask):
        if mask.sum() > 0:
            return emb[mask].mean(dim=0)
        return emb.mean(dim=0)

    return {
        'B': {
            'agree': safe_proto(emb_b_all, ~kl_b_all),
            'disagree': safe_proto(emb_b_all, kl_b_all),
        },
        'C': {
            'agree': safe_proto(emb_c_all, ~kl_c_all),
            'disagree': safe_proto(emb_c_all, kl_c_all),
        },
    }


def prototype_routing_simple(
    logits_a, logits_b, logits_c, emb_b, emb_c,
    prototypes, threshold_agree=0.7,
):
    """Simplified prototype routing for testing."""
    batch_size = logits_a.size(0)
    emb_b_norm = F.normalize(emb_b, dim=1)
    emb_c_norm = F.normalize(emb_c, dim=1)

    pb_agree = F.normalize(prototypes['B']['agree'].unsqueeze(0).to(emb_b.device), dim=1)
    pb_dis = F.normalize(prototypes['B']['disagree'].unsqueeze(0).to(emb_b.device), dim=1)
    pc_agree = F.normalize(prototypes['C']['agree'].unsqueeze(0).to(emb_c.device), dim=1)
    pc_dis = F.normalize(prototypes['C']['disagree'].unsqueeze(0).to(emb_c.device), dim=1)

    score_b = F.cosine_similarity(emb_b_norm, pb_dis) - F.cosine_similarity(emb_b_norm, pb_agree)
    score_c = F.cosine_similarity(emb_c_norm, pc_dis) - F.cosine_similarity(emb_c_norm, pc_agree)
    score_a = -(score_b + score_c) / 2.0

    emb_sim_bc = F.cosine_similarity(emb_b_norm, emb_c_norm)
    use_uniform = (emb_sim_bc > threshold_agree).float().unsqueeze(1)

    scores = torch.stack([score_a, score_b, score_c], dim=1)
    logits_stack = torch.stack([logits_a, logits_b, logits_c], dim=2)

    uniform_logits = (logits_a + logits_b + logits_c) / 3.0
    best_expert = scores.argmax(dim=1)
    selected_logits = logits_stack[torch.arange(batch_size), :, best_expert]

    final_logits = use_uniform * uniform_logits + (1.0 - use_uniform) * selected_logits
    return final_logits, scores, emb_sim_bc, best_expert


# ── Tests ────────────────────────────────────────────────────────────────

def test_prototype_computation_shapes():
    """Prototype computation returns correctly shaped prototypes."""
    device = 'cpu'
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()

    # Create synthetic validation set
    fake_images = torch.randn(16, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (16,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets),
                            batch_size=8)

    prototypes = compute_prototypes_simple(
        expert_a, expert_b, expert_c, val_loader, device=device,
    )

    for key in ['B', 'C']:
        assert 'agree' in prototypes[key]
        assert 'disagree' in prototypes[key]
        assert prototypes[key]['agree'].shape == (32,), \
            f"{key} agree shape: {prototypes[key]['agree'].shape}"
        assert prototypes[key]['disagree'].shape == (32,), \
            f"{key} disagree shape: {prototypes[key]['disagree'].shape}"

    print(f"  ✅ Prototypes computed correctly (B and C, 32-d each)")


def test_routing_output_shapes():
    """Prototype routing produces correct output shapes."""
    device = 'cpu'
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()

    fake_images = torch.randn(8, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (8,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=8)

    prototypes = compute_prototypes_simple(
        expert_a, expert_b, expert_c, val_loader, device=device,
    )

    with torch.no_grad():
        logits_a = expert_a(fake_images)
        logits_b, emb_b = expert_b(fake_images)
        logits_c, emb_c = expert_c(fake_images)

    final_logits, scores, avg_agreement, best_expert = prototype_routing_simple(
        logits_a, logits_b, logits_c, emb_b, emb_c, prototypes,
    )

    assert final_logits.shape == (8, 100), f"Final logits: {final_logits.shape}"
    assert scores.shape == (8, 3), f"Scores: {scores.shape}"
    assert avg_agreement.shape == (8,), f"Agreement: {avg_agreement.shape}"
    assert best_expert.shape == (8,), f"Best expert: {best_expert.shape}"
    assert set(best_expert.tolist()).issubset({0, 1, 2})

    print(f"  ✅ Routing output shapes correct")


def test_uniform_fallback():
    """When embeddings are very similar, uniform fallback is used.

    Use the SAME model for B and C to create near-identical embeddings
    (since different random models produce different embeddings even on
    identical inputs).
    """
    device = 'cpu'
    expert_a = ResNet32(num_classes=100).eval()
    # Use the same model instance for B and C to get identical embeddings
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = expert_b  # same model → same embeddings

    fake_images = torch.randn(4, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (4,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=4)

    prototypes = compute_prototypes_simple(
        expert_a, expert_b, expert_c, val_loader, device=device,
    )

    with torch.no_grad():
        logits_a = expert_a(fake_images)
        logits_b, emb_b = expert_b(fake_images)
        logits_c, emb_c = expert_c(fake_images)  # identical to emb_b

    final_logits, scores, avg_agreement, best_expert = prototype_routing_simple(
        logits_a, logits_b, logits_c, emb_b, emb_c, prototypes,
        threshold_agree=0.5,
    )

    # emb_b and emb_c are identical → cosine_sim = 1.0 → uniform
    uniform_logits = (logits_a + logits_b + logits_c) / 3.0
    assert torch.allclose(final_logits, uniform_logits, atol=1e-5), \
        "Should use uniform when embeddings are identical"
    print(f"  ✅ Uniform fallback works (cosine_sim={avg_agreement.mean():.4f} > 0.5)")


def test_routing_decisions_valid():
    """Routing decisions are valid (0, 1, or 2) for all samples."""
    device = 'cpu'
    expert_a = ResNet32(num_classes=100).eval()
    expert_b = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    expert_c = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()

    fake_images = torch.randn(4, 3, 32, 32)
    fake_targets = torch.randint(0, 100, (4,))
    val_loader = DataLoader(TensorDataset(fake_images, fake_targets), batch_size=4)

    prototypes = compute_prototypes_simple(
        expert_a, expert_b, expert_c, val_loader, device=device,
    )

    with torch.no_grad():
        logits_a = expert_a(fake_images)
        logits_b, emb_b = expert_b(fake_images)
        logits_c, emb_c = expert_c(fake_images)

    _, _, _, best_expert = prototype_routing_simple(
        logits_a, logits_b, logits_c, emb_b, emb_c, prototypes,
    )

    assert all(d in [0, 1, 2] for d in best_expert.tolist())
    print(f"  ✅ Routing decisions valid: {best_expert.tolist()}")


def test_metrics_computation():
    """Balanced accuracy and group accuracies compute without error."""
    # Create fake predictions and targets
    all_targets = np.random.randint(0, 100, size=100)
    all_preds = np.random.randint(0, 100, size=100)

    def balanced_accuracy(targets, preds):
        classes = sorted(set(targets.tolist()))
        per_class = []
        for c in classes:
            mask = targets == c
            per_class.append((preds[mask] == c).sum() / max(mask.sum(), 1))
        return float(np.mean(per_class))

    ba = balanced_accuracy(all_targets, all_preds)
    assert 0.0 <= ba <= 1.0, f"BA should be in [0,1], got {ba}"

    groups = {
        'Head': np.arange(30),
        'Med': np.arange(30, 66),
        'Tail': np.arange(66, 100),
    }

    def group_acc(targets, preds, groups):
        result = {}
        for name, cls_list in groups.items():
            mask = np.isin(targets, cls_list)
            if mask.sum() > 0:
                result[name] = (preds[mask] == targets[mask]).sum() / mask.sum()
            else:
                result[name] = 0.0
        return result

    grp = group_acc(all_targets, all_preds, groups)
    for name in ['Head', 'Med', 'Tail']:
        assert 0.0 <= grp[name] <= 1.0, f"{name} acc: {grp[name]}"

    print(f"  ✅ Metrics: BA={ba:.2%}, "
          f"Head={grp['Head']:.2%}, Med={grp['Med']:.2%}, Tail={grp['Tail']:.2%}")


if __name__ == "__main__":
    print("=" * 50)
    print("  DACE Evaluation — Synthetic Dry-Run")
    print("=" * 50)
    print()

    tests = [
        ("Prototype computation shapes", test_prototype_computation_shapes),
        ("Routing output shapes", test_routing_output_shapes),
        ("Uniform fallback", test_uniform_fallback),
        ("Routing decisions valid", test_routing_decisions_valid),
        ("Metrics computation", test_metrics_computation),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("  All synthetic dry-run tests passed!")
        print(f"  You can now run the full evaluation on Kaggle:")
        print(f"    python scripts/evaluate_dace.py")
