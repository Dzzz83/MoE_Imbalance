"""
Synthetic dry-run for DACE Expert C (Tail specialist, cascade).

Tests the components of the cascade training:
  - Two frozen experts (A and B) can produce logits
  - Ensemble average of A and B is computable
  - KL divergence between ensemble average and C is computable
  - Agreement labels from KL(avg(A,B) || C) are valid
  - Contrastive routing loss works with C's routing embeddings
  - Combined loss (CE + contrastive) is finite
  - Gradients flow through BOTH classifier and routing head
  - Backbone gradients confirm routing head signal flows through

This satisfies the "Kaggle-Gate" rule (AGENTs.md §11):
  - Runs entirely on CPU
  - No CIFAR-100 download needed
  - Tiny dummy batch (batch_size=4)
  - No pre-trained checkpoints required (uses randomly initialized models)
"""

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import torch.nn.functional as F
import numpy as np

from models.resnet32 import ResNet32, ResNet32WithRouting
from losses.ce_loss import CELoss
from losses.kl_divergence import kl_divergence, compute_agreement_label
from losses.contrastive_routing_loss import ContrastiveRoutingLoss


def test_ensemble_avg():
    """Ensemble average of two experts is computable."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)
        logits_b, _ = model_b(images)

    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)
    avg_probs = (probs_a + probs_b) / 2.0

    assert avg_probs.shape == (4, 100), f"Avg shape: {avg_probs.shape}"
    assert torch.allclose(avg_probs.sum(dim=1), torch.ones(4)), \
        "Avg probs should sum to 1"
    print(f"  ✅ Ensemble avg shape: {avg_probs.shape}, sums to 1")


def test_kl_ensemble_vs_c():
    """KL(avg(A,B) || C) is computable and finite."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    model_c = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)
        logits_b, _ = model_b(images)
    logits_c, _ = model_c(images)

    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)
    avg_logits = torch.log((probs_a + probs_b) / 2.0 + 1e-12)

    kl = kl_divergence(avg_logits, logits_c, reduction="batch_mean")
    assert torch.isfinite(kl), f"KL should be finite, got {kl}"
    print(f"  ✅ KL(avg(A,B)||C) = {kl.item():.4f} (finite)")


def test_agreement_labels_ensemble():
    """Agreement labels from KL(avg(A,B) || C) are valid."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    model_c = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)
        logits_b, _ = model_b(images)
    logits_c, _ = model_c(images)

    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)
    avg_logits = torch.log((probs_a + probs_b) / 2.0 + 1e-12)

    labels = compute_agreement_label(avg_logits, logits_c, threshold=0.5, metric='kl')
    assert labels.shape == (4,)
    assert set(labels.tolist()).issubset({0, 1})
    print(f"  ✅ Agreement labels: {labels.tolist()}")


def test_combined_loss():
    """Combined CE + contrastive loss is finite and backpropagates."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    model_c = ResNet32WithRouting(num_classes=100, routing_dim=32)
    loss_ce = CELoss()
    contrastive_loss_fn = ContrastiveRoutingLoss(temperature=0.5)

    images = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))

    # Forward A and B (frozen)
    with torch.no_grad():
        logits_a = model_a(images)
        logits_b, _ = model_b(images)
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        avg_probs = (probs_a + probs_b) / 2.0
        avg_logits = torch.log(avg_probs + 1e-12)

    # Forward C
    logits_c, routing_emb = model_c(images)

    # Classification loss
    loss_cls = loss_ce(logits_c, targets)

    # Agreement labels
    kl_labels = compute_agreement_label(avg_logits, logits_c, threshold=0.5, metric='kl')

    # Contrastive routing loss
    loss_contrastive, _ = contrastive_loss_fn(routing_emb, kl_labels)

    # Combined loss
    lambda_routing = 0.05
    loss = loss_cls + lambda_routing * loss_contrastive

    assert torch.isfinite(loss), f"Combined loss should be finite, got {loss}"
    print(f"  ✅ Combined loss = {loss.item():.4f} (finite)")

    loss.backward()

    # Check gradients
    cls_grad = model_c.classifier.weight.grad
    routing_grad = model_c.routing_head.net[0].weight.grad
    conv1_grad = model_c.backbone.conv1.weight.grad

    assert cls_grad is not None and cls_grad.abs().sum().item() > 0
    assert routing_grad is not None and routing_grad.abs().sum().item() > 0
    assert conv1_grad is not None and conv1_grad.abs().sum().item() > 0

    print(f"  ✅ Gradients: classifier={cls_grad.abs().sum().item():.4f}, "
          f"routing_head={routing_grad.abs().sum().item():.4f}, "
          f"backbone_conv1={conv1_grad.abs().sum().item():.6f}")


def test_routing_only_gradient():
    """Routing loss alone produces backbone gradients (no classifier supervision)."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    model_c = ResNet32WithRouting(num_classes=100, routing_dim=32)
    contrastive_loss_fn = ContrastiveRoutingLoss(temperature=0.5)

    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)
        logits_b, _ = model_b(images)
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        avg_logits = torch.log((probs_a + probs_b) / 2.0 + 1e-12)

    logits_c, routing_emb = model_c(images)
    kl_labels = compute_agreement_label(avg_logits, logits_c, threshold=0.5, metric='kl')

    # Only routing loss (no classification loss)
    loss, _ = contrastive_loss_fn(routing_emb, kl_labels)
    loss.backward()

    conv1_grad = model_c.backbone.conv1.weight.grad
    assert conv1_grad is not None and conv1_grad.abs().sum().item() > 0, \
        "Backbone should receive gradients from routing loss alone"
    print(f"  ✅ Backbone conv1 gradient (routing only): "
          f"{conv1_grad.abs().sum().item():.6f}")


if __name__ == "__main__":
    print("=" * 50)
    print("  DACE Expert C — Synthetic Dry-Run")
    print("=" * 50)
    print()

    tests = [
        ("Ensemble average", test_ensemble_avg),
        ("KL ensemble vs C", test_kl_ensemble_vs_c),
        ("Agreement labels ensemble", test_agreement_labels_ensemble),
        ("Combined loss + backward", test_combined_loss),
        ("Routing-only backbone gradient", test_routing_only_gradient),
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
        print(f"  You can now run the full training on Kaggle:")
        print(f"    python scripts/train_dace_c.py \\")
        print(f"      --dace-a-ckpt checkpoints/DACE_A_best.pt \\")
        print(f"      --dace-b-ckpt checkpoints/DACE_B_best.pt")
