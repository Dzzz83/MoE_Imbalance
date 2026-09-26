"""
Tests for models/resnet32.py — RoutingHead and ResNet32WithRouting.

Expected behavior:
  - RoutingHead: 64 → 64 → ReLU → 32, shape (B, 32)
  - ResNet32WithRouting.forward() returns (logits, routing_emb)
  - forward_logits() returns logits only
  - forward_with_features() returns (logits, routing_emb, features)
  - Backward pass populates .grad for backbone, classifier, and routing head
  - Checkpoint loading compatible with ResNet32 (strict=False)
"""

import sys
import os
import torch

from pathlib import Path as _TestPath
import sys as _test_sys
_TEST_PACKAGE_DIR = str(_TestPath(__file__).resolve().parent.parent)
if _TEST_PACKAGE_DIR not in _test_sys.path:
    _test_sys.path.insert(0, _TEST_PACKAGE_DIR)
from repo_root import REPO_ROOT
if str(REPO_ROOT) not in _test_sys.path:
    _test_sys.path.insert(0, str(REPO_ROOT))
_proj_root = str(REPO_ROOT)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from models.resnet32 import RoutingHead, ResNet32WithRouting, ResNet32, PaCoResNet32

# Fixtures below are random; seed them so a threshold assertion cannot
# flip between runs.
torch.manual_seed(0)


def test_routing_head_shape():
    """RoutingHead produces (B, 32) embedding from (B, 64) features."""
    head = RoutingHead(feat_dim=64, routing_dim=32)
    features = torch.randn(4, 64)
    emb = head(features)
    assert emb.shape == (4, 32), f"Expected (4, 32), got {emb.shape}"


def test_routing_head_gradients():
    """Gradient flows through RoutingHead to features."""
    head = RoutingHead(feat_dim=64, routing_dim=32)
    features = torch.randn(4, 64, requires_grad=True)
    emb = head(features)
    loss = emb.pow(2).mean()
    loss.backward()
    assert features.grad is not None, "Gradient should flow to features"
    assert features.grad.abs().sum().item() > 0, "Gradient should be non-zero"


def test_resnet32_with_routing_shapes():
    """ResNet32WithRouting returns (logits, routing_emb) with correct shapes."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)
    logits, routing_emb = model(images)
    assert logits.shape == (4, 100), f"Logits shape expected (4, 100), got {logits.shape}"
    assert routing_emb.shape == (4, 32), f"Routing emb shape expected (4, 32), got {routing_emb.shape}"


def test_forward_logits_only():
    """forward_logits returns only logits (compatible with ResNet32 interface)."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)
    logits = model.forward_logits(images)
    assert logits.shape == (4, 100), f"Expected (4, 100), got {logits.shape}"


def test_forward_with_features():
    """forward_with_features returns (logits, routing_emb, features)."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)
    logits, routing_emb, features = model.forward_with_features(images)
    assert logits.shape == (4, 100)
    assert routing_emb.shape == (4, 32)
    assert features.shape == (4, 64), f"Features shape expected (4, 64), got {features.shape}"


def test_backward_pass():
    """Loss.backward() populates .grad for all trainable parameters."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 100, (4,))

    logits, routing_emb = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels) + routing_emb.pow(2).mean()
    loss.backward()

    # Check that all parameter groups have gradients
    has_grad_counts = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum().item() > 0:
            has_grad_counts += 1

    assert has_grad_counts > 0, f"No parameters received gradients"
    assert has_grad_counts >= total_params * 0.9, (
        f"Only {has_grad_counts}/{total_params} parameters have gradients"
    )


def test_classifier_gradient_flows():
    """Classifier gradient flows through backbone (shared features)."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 100, (4,))

    logits, _ = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # The first conv layer should receive gradient (confirms backbone is in the graph)
    assert model.backbone.conv1.weight.grad is not None
    assert model.backbone.conv1.weight.grad.abs().sum().item() > 0


def test_routing_head_gradient_flows_to_backbone():
    """Routing head gradient flows into backbone features."""
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    images = torch.randn(4, 3, 32, 32)

    _, routing_emb = model(images)
    loss = routing_emb.pow(2).mean()
    loss.backward()

    # The first conv layer should receive gradient (backbone is shared)
    assert model.backbone.conv1.weight.grad is not None
    assert model.backbone.conv1.weight.grad.abs().sum().item() > 0


def test_load_resnet32_checkpoint():
    """ResNet32WithRouting can load a ResNet32 checkpoint with strict=False.

    ResNet32 uses 'fc.*' keys; ResNet32WithRouting uses 'classifier.*'.
    The routing_head.* keys are new and not present in the checkpoint.
    """
    # Create a ResNet32 checkpoint
    resnet = ResNet32(num_classes=100)
    state_dict = resnet.state_dict()

    # Load into ResNet32WithRouting — should succeed with strict=False
    model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Expected missing keys: routing_head.* (new) AND classifier.* (named differently)
    # Since ResNet32 has 'fc.weight', 'fc.bias' but we have 'classifier.weight', 'classifier.bias'
    missing_keys = set(missing)
    assert 'routing_head.net.0.weight' in missing_keys, (
        f"Expected routing_head keys in missing, got {missing_keys}"
    )
    assert 'routing_head.net.2.weight' in missing_keys
    # 'classifier.*' are missing because ResNet32 uses 'fc.*'
    assert 'classifier.weight' in missing_keys

    # The 'fc.weight' and 'fc.bias' from ResNet32 should appear in unexpected
    unexpected_keys = {k for k in unexpected}
    assert 'fc.weight' in unexpected_keys, (
        f"Expected 'fc.weight' in unexpected, got {unexpected_keys}"
    )


def test_paco_queue_handles_batches_larger_than_queue():
    """Circular queue replacement must support any batch size."""
    model = PaCoResNet32(num_classes=3, dim=2, K=3, mlp=False)
    model.queue.zero_()
    model.queue_label.zero_()
    model.queue_ptr.fill_(1)
    keys = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    labels = torch.arange(7, dtype=torch.long)

    model._dequeue_and_enqueue(keys, labels)

    # Last K keys are written starting at (old_ptr + B - K) % K = 2.
    expected_queue = torch.tensor([[10., 11.], [12., 13.], [8., 9.]])
    expected_labels = torch.tensor([5, 6, 4])
    assert torch.equal(model.queue, expected_queue)
    assert torch.equal(model.queue_label, expected_labels)
    assert int(model.queue_ptr) == 2
    print("  ✅ PaCo queue handles batches larger than 2K")


def test_paco_rejects_nonpositive_queue_size():
    try:
        PaCoResNet32(num_classes=3, dim=2, K=0, mlp=False)
    except ValueError as exc:
        assert "K" in str(exc)
        return
    raise AssertionError("PaCoResNet32 must reject K=0")


if __name__ == "__main__":
    tests = [
        ("RoutingHead shape", test_routing_head_shape),
        ("RoutingHead gradients", test_routing_head_gradients),
        ("ResNet32WithRouting shapes", test_resnet32_with_routing_shapes),
        ("Forward logits only", test_forward_logits_only),
        ("Forward with features", test_forward_with_features),
        ("Backward pass", test_backward_pass),
        ("Classifier gradient flows", test_classifier_gradient_flows),
        ("Routing head gradient flows to backbone", test_routing_head_gradient_flows_to_backbone),
        ("Load ResNet32 checkpoint", test_load_resnet32_checkpoint),
        ("PaCo queue handles large batches", test_paco_queue_handles_batches_larger_than_queue),
        ("PaCo queue validates size", test_paco_rejects_nonpositive_queue_size),
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
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("  All tests passed!")
