"""
Synthetic dry-run for DACE Expert B (Med specialist, cascade).

Tests the components of the cascade training:
  - ResNet32WithRouting produces correct shapes (logits + routing embedding)
  - KL divergence between two models' outputs is computable
  - Agreement labels can be computed from two model outputs
  - Contrastive routing loss receives correct shapes
  - Combined loss (Mixup CE + contrastive) is finite
  - Gradients flow through BOTH classifier and routing head

This satisfies the "Kaggle-Gate" rule (AGENTs.md §11):
  - Runs entirely on CPU
  - No CIFAR-100 download needed
  - Tiny dummy batch (batch_size=4)
  - No pre-trained checkpoint required (uses randomly initialized models)
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


def mixup_data(images, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return lam * criterion(pred, targets_a) + (1.0 - lam) * criterion(pred, targets_b)


def test_resnet32_with_routing_shapes():
    """ResNet32WithRouting produces (logits, routing_emb) with correct shapes."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)
    logits, routing_emb = model(images)
    assert logits.shape == (4, 100), f"Logits: {logits.shape}"
    assert routing_emb.shape == (4, 32), f"Routing emb: {routing_emb.shape}"
    print(f"  ✅ Logits: {logits.shape}, Routing emb: {routing_emb.shape}")


def test_kl_between_two_models():
    """KL divergence between two random models' outputs is computable."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32(num_classes=100)
    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)
        logits_b = model_b(images)

    kl = kl_divergence(logits_a, logits_b, reduction="batch_mean")
    assert torch.isfinite(kl), "KL should be finite"
    assert kl.item() > 0, f"KL should be > 0 for different models, got {kl.item()}"
    print(f"  ✅ KL(A||B) = {kl.item():.4f} (finite, >0)")


def test_agreement_labels():
    """Agreement labels are computable and correctly shaped."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)
        logits_b, _ = model_b(images)

    labels = compute_agreement_label(logits_a, logits_b, threshold=0.5, metric='kl')
    assert labels.shape == (4,), f"Labels shape: {labels.shape}"
    assert labels.dtype == torch.long
    assert set(labels.tolist()).issubset({0, 1}), "Labels should be 0 or 1"
    print(f"  ✅ Agreement labels: {labels.tolist()}")


def test_contrastive_loss_on_routing_emb():
    """Contrastive routing loss receives routing embeddings and produces finite loss."""
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    contrastive_loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    images = torch.randn(4, 3, 32, 32)

    _, routing_emb = model_b(images)
    labels = torch.tensor([0, 0, 1, 1])

    loss, aux = contrastive_loss_fn(routing_emb, labels)
    assert torch.isfinite(loss), "Contrastive loss should be finite"
    assert loss.item() >= 0, f"Loss should be >= 0, got {loss.item()}"
    print(f"  ✅ Contrastive loss = {loss.item():.4f} (finite)")


def test_combined_loss():
    """Combined Mixup CE + contrastive loss is finite and backpropagates."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    loss_ce = CELoss()
    contrastive_loss_fn = ContrastiveRoutingLoss(temperature=0.5)

    images = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))

    # Forward Expert A (frozen)
    with torch.no_grad():
        logits_a = model_a(images)

    # Mixup + Forward Expert B
    mixed_images, targets_a, targets_b, lam = mixup_data(images, targets)
    logits_b, routing_emb = model_b(mixed_images)

    # Classification loss
    loss_cls = mixup_criterion(loss_ce, logits_b, targets_a, targets_b, lam)

    # KL labels on original images
    logits_b_original, routing_emb_original = model_b(images)
    kl_labels = compute_agreement_label(logits_a, logits_b_original, threshold=0.5, metric='kl')

    # Contrastive loss
    loss_contrastive, _ = contrastive_loss_fn(routing_emb_original, kl_labels)

    # Combined loss
    lambda_routing = 0.1
    loss = loss_cls + lambda_routing * loss_contrastive

    assert torch.isfinite(loss), f"Combined loss should be finite, got {loss}"
    print(f"  ✅ Combined loss = {loss.item():.4f} (finite)")

    # Backward pass
    loss.backward()

    # Check gradients in both classifier and routing head
    cls_grad = model_b.classifier.weight.grad
    routing_grad = model_b.routing_head.net[0].weight.grad

    assert cls_grad is not None and cls_grad.abs().sum().item() > 0, \
        "Classifier should have gradients"
    assert routing_grad is not None and routing_grad.abs().sum().item() > 0, \
        "Routing head should have gradients"
    print(f"  ✅ Gradients: classifier={cls_grad.abs().sum().item():.4f}, "
          f"routing_head={routing_grad.abs().sum().item():.4f}")


def test_backbone_gradient_from_routing():
    """Routing head gradient flows into backbone (first conv layer)."""
    model_a = ResNet32(num_classes=100)
    model_b = ResNet32WithRouting(num_classes=100, routing_dim=32)
    contrastive_loss_fn = ContrastiveRoutingLoss(temperature=0.5)

    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        logits_a = model_a(images)

    logits_b, routing_emb = model_b(images)
    kl_labels = compute_agreement_label(logits_a, logits_b, threshold=0.5, metric='kl')

    # Only routing loss (no classification loss)
    loss, _ = contrastive_loss_fn(routing_emb, kl_labels)
    loss.backward()

    conv1_grad = model_b.backbone.conv1.weight.grad
    assert conv1_grad is not None and conv1_grad.abs().sum().item() > 0, \
        "Backbone conv1 should have gradients from routing loss"
    print(f"  ✅ Backbone conv1 gradient from routing loss: "
          f"{conv1_grad.abs().sum().item():.6f}")


if __name__ == "__main__":
    print("=" * 50)
    print("  DACE Expert B — Synthetic Dry-Run")
    print("=" * 50)
    print()

    tests = [
        ("ResNet32WithRouting shapes", test_resnet32_with_routing_shapes),
        ("KL between two models", test_kl_between_two_models),
        ("Agreement labels", test_agreement_labels),
        ("Contrastive loss on routing emb", test_contrastive_loss_on_routing_emb),
        ("Combined loss + backward", test_combined_loss),
        ("Backbone gradient from routing", test_backbone_gradient_from_routing),
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
        print(f"    python scripts/train_dace_b.py --dace-a-ckpt checkpoints/DACE_A_best.pt")
