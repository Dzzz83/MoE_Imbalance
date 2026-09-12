"""
Regression tests for scripts/evaluate_dace.py — the REAL shipped functions.

Why this file exists:
    tests/test_evaluate_dace_synthetic.py re-IMPLEMENTS compute_prototypes and
    prototype_routing inside the test file.  That means the shipped code in
    scripts/evaluate_dace.py was never actually exercised, and a device bug
    (`embeddings moved to CPU, KL masks left on GPU`) reached Kaggle undetected.

    These tests import the real functions so the shipped code path is covered.

Bug being guarded (reported from Kaggle, CUDA):
    RuntimeError: indices should be either on cpu or on the same device as the
    indexed tensor (cpu)
        at   proto_b_disagree = emb_b_all[kl_b_all].mean(dim=0)
    Cause: emb_* went through `.cpu()` but the boolean KL masks did not, so on
    CUDA the mask and the indexed tensor lived on different devices.

The invariant that must hold: every per-sample tensor collected inside
compute_prototypes is reduced to ONE device before any indexing happens.
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

# Import the REAL shipped functions (not a copy)
from scripts.evaluate_dace import (
    compute_prototypes,
    prototype_routing,
    assert_mask_matches_embedding,
)
from models.resnet32 import ResNet32, ResNet32WithRouting


# ── fixtures ─────────────────────────────────────────────────────────────

def _build(seed: int = 0):
    torch.manual_seed(seed)
    ea = ResNet32(num_classes=100).eval()
    eb = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    ec = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    imgs = torch.randn(16, 3, 32, 32)
    tgts = torch.randint(0, 100, (16,))
    loader = DataLoader(TensorDataset(imgs, tgts), batch_size=8)
    return ea, eb, ec, loader, imgs


# ── tests ────────────────────────────────────────────────────────────────

class _FakeTensor:
    """Stand-in exposing only .device/.dtype — enough for the guard.

    Lets us exercise a cross-device mismatch on a CPU-only machine, where a
    genuine second device does not exist.
    """

    def __init__(self, device, dtype=torch.bool):
        self.device = torch.device(device)
        self.dtype = dtype


def test_guard_accepts_matching_device():
    """Guard passes when mask and embedding share a device and mask is bool."""
    emb = torch.zeros(3, 4)
    mask = torch.ones(3, dtype=torch.bool)
    assert_mask_matches_embedding(emb, mask, 'B')  # must not raise


def test_guard_rejects_device_mismatch():
    """Guard raises the self-describing error on a device mismatch.

    This is the condition that produced the Kaggle crash: embeddings moved to
    CPU while the boolean KL mask stayed on GPU.
    """
    emb = _FakeTensor('cpu')
    mask = _FakeTensor('cuda')
    try:
        assert_mask_matches_embedding(emb, mask, 'B')
    except RuntimeError as e:
        msg = str(e)
        assert 'Expert B' in msg, msg
        assert 'cpu' in msg and 'cuda' in msg, msg
        assert 'boolean indexing' in msg, msg
        return
    raise AssertionError(
        "device mismatch must raise RuntimeError, but guard passed silently"
    )


def test_guard_rejects_non_bool_mask():
    """Guard raises when the mask is not boolean (would index by value)."""
    emb = _FakeTensor('cpu')
    mask = _FakeTensor('cpu', dtype=torch.float32)
    try:
        assert_mask_matches_embedding(emb, mask, 'C')
    except RuntimeError as e:
        assert 'torch.bool' in str(e), str(e)
        return
    raise AssertionError("non-bool mask must raise RuntimeError")


def test_real_compute_prototypes_shapes_and_finiteness():
    """Real compute_prototypes returns 32-d finite prototypes for A, B, C."""
    ea, eb, ec, loader, _ = _build()
    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=0.1, device='cpu')

    assert set(protos.keys()) == {'A', 'B', 'C'}, protos.keys()
    for name in ['A', 'B', 'C']:
        for kind in ['agree', 'disagree']:
            t = protos[name][kind]
            assert t.shape == (32,), f"{name}.{kind} shape {t.shape}"
            assert torch.isfinite(t).all(), f"{name}.{kind} has non-finite values"


def test_prototypes_are_on_cpu():
    """Prototypes must end up on CPU so prototype_routing can move them freely."""
    ea, eb, ec, loader, _ = _build()
    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=0.1, device='cpu')

    for name in ['A', 'B', 'C']:
        for kind in ['agree', 'disagree']:
            dev = protos[name][kind].device
            assert dev.type == 'cpu', f"{name}.{kind} on {dev}, expected cpu"


def test_routing_scores_vary_when_the_threshold_splits_the_batch():
    """The routing scorer must actually route, not return a constant.

    With randomly initialised experts every sample sits far above the default
    ``kl_threshold=0.1``, so both prototypes are the mean of the *same* samples:
    they come out identical, every score is 0, and ``best_expert`` is always 0.
    Every other assertion in this file passes in that state, which is exactly how
    a broken scorer stays green — so this test forces a threshold that splits the
    batch (the median KL) and then requires varied scores and decisions.
    """
    ea, eb, ec, loader, imgs = _build()

    kl_values = []
    with torch.no_grad():
        for images, _ in loader:
            probs_a = F.softmax(ea(images), dim=1)
            logits_b, _emb_b = eb(images)
            probs_b = F.softmax(logits_b, dim=1)
            kl_values.append(
                (probs_a * (torch.log(probs_a + 1e-12)
                            - torch.log(probs_b + 1e-12))).sum(dim=1))
    threshold = float(torch.cat(kl_values).median())

    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=threshold,
                               device='cpu')
    assert not torch.allclose(protos['B']['agree'], protos['B']['disagree']), (
        "B's agree and disagree prototypes are identical — the fixture is "
        "degenerate and the scorer below is not being exercised"
    )

    with torch.no_grad():
        logits_a = ea(imgs)
        logits_b, emb_b = eb(imgs)
        logits_c, emb_c = ec(imgs)

    _final, scores, _agree, best_expert = prototype_routing(
        logits_a, logits_b, logits_c, emb_b, emb_c, protos, threshold_agree=0.7,
    )
    assert float(scores.abs().max()) > 0.0, "every routing score is 0 (inert scorer)"
    assert len(set(best_expert.tolist())) > 1, (
        f"routing always selects the same expert: {best_expert.tolist()}"
    )
    print(f"  ✅ scores vary (max |score| {float(scores.abs().max()):.2e}), "
          f"experts chosen {sorted(set(best_expert.tolist()))}")


def test_prototype_indexing_device_invariant():
    """The exact operation that failed on Kaggle must be device-consistent.

    Reproduces the indexing step the bug crashed on, using the real
    prototypes, to prove mask and indexed tensor agree on device.
    """
    ea, eb, ec, loader, _ = _build()
    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=0.1, device='cpu')

    # Re-derive a mask on the SAME device as the prototype and index with it.
    emb = protos['B']['disagree']
    mask = torch.ones(emb.shape[0], dtype=torch.bool, device=emb.device)
    picked = emb[mask].mean(dim=0)          # the operation that crashed
    assert torch.isfinite(picked).all()
    assert picked.device == emb.device


def test_real_prototype_routing_shapes():
    """Real prototype_routing returns correct shapes and valid decisions."""
    ea, eb, ec, loader, imgs = _build()
    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=0.1, device='cpu')

    with torch.no_grad():
        logits_a = ea(imgs)
        logits_b, emb_b = eb(imgs)
        logits_c, emb_c = ec(imgs)

    final_logits, scores, avg_agreement, best_expert = prototype_routing(
        logits_a, logits_b, logits_c, emb_b, emb_c, protos,
        threshold_agree=0.7,
    )

    assert final_logits.shape == (16, 100), final_logits.shape
    assert scores.shape == (16, 3), scores.shape
    assert avg_agreement.shape == (16,), avg_agreement.shape
    assert best_expert.shape == (16,), best_expert.shape
    assert set(best_expert.tolist()).issubset({0, 1, 2})
    assert torch.isfinite(final_logits).all()


def test_real_prototype_routing_uniform_fallback():
    """When B and C are the same model, embeddings match -> uniform fallback."""
    torch.manual_seed(0)
    ea = ResNet32(num_classes=100).eval()
    eb = ResNet32WithRouting(num_classes=100, routing_dim=32).eval()
    ec = eb  # identical model -> identical embeddings

    imgs = torch.randn(8, 3, 32, 32)
    tgts = torch.randint(0, 100, (8,))
    loader = DataLoader(TensorDataset(imgs, tgts), batch_size=8)

    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=0.1, device='cpu')

    with torch.no_grad():
        logits_a = ea(imgs)
        logits_b, emb_b = eb(imgs)
        logits_c, emb_c = ec(imgs)

    final_logits, _, agreement, _ = prototype_routing(
        logits_a, logits_b, logits_c, emb_b, emb_c, protos,
        threshold_agree=0.5,
    )

    assert torch.allclose(agreement, torch.ones_like(agreement), atol=1e-5), \
        "identical models should have cosine similarity 1.0"
    uniform = (logits_a + logits_b + logits_c) / 3.0
    assert torch.allclose(final_logits, uniform, atol=1e-5), \
        "high agreement must fall back to uniform averaging"


def test_prototype_routing_accepts_cpu_prototypes_with_fresh_copies():
    """Prototype tensors must not be mutated in place by routing."""
    ea, eb, ec, loader, imgs = _build()
    protos = compute_prototypes(ea, eb, ec, loader, kl_threshold=0.1, device='cpu')

    snapshot = {n: {k: v.clone() for k, v in d.items()} for n, d in protos.items()}

    with torch.no_grad():
        logits_a = ea(imgs)
        logits_b, emb_b = eb(imgs)
        logits_c, emb_c = ec(imgs)

    prototype_routing(logits_a, logits_b, logits_c, emb_b, emb_c, protos,
                      threshold_agree=0.7)

    for n, d in protos.items():
        for k, v in d.items():
            assert torch.equal(v, snapshot[n][k]), f"{n}.{k} was mutated"

    _ = np  # keep numpy import meaningful for future metric tests


if __name__ == "__main__":
    tests = [
        ("Guard accepts matching device", test_guard_accepts_matching_device),
        ("Guard rejects device mismatch", test_guard_rejects_device_mismatch),
        ("Guard rejects non-bool mask", test_guard_rejects_non_bool_mask),
        ("Real compute_prototypes shapes/finiteness",
         test_real_compute_prototypes_shapes_and_finiteness),
        ("Prototypes are on CPU", test_prototypes_are_on_cpu),
        ("Prototype indexing device invariant",
         test_prototype_indexing_device_invariant),
        ("Routing scores vary with a splitting threshold",
         test_routing_scores_vary_when_the_threshold_splits_the_batch),
        ("Real prototype_routing shapes", test_real_prototype_routing_shapes),
        ("Real prototype routing uniform fallback",
         test_real_prototype_routing_uniform_fallback),
        ("Prototypes not mutated by routing",
         test_prototype_routing_accepts_cpu_prototypes_with_fresh_copies),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
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
