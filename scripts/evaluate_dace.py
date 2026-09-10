#!/usr/bin/env python3
"""
DACE Inference: Prototype-based disagreement routing.

After all three DACE experts are trained, this script:
  1. Loads the three experts (A: ResNet32, B/C: ResNet32WithRouting)
  2. Pre-computes divergence prototypes on the VALIDATION set
  3. Evaluates test-set performance with prototype-based routing
  4. Reports Balanced Accuracy, per-group accuracies, and comparisons

Inference procedure (DACE plan §5):
  - For each test sample, compute each expert's disagreement affinity:
      score_e = cos_sim(emb_e, proto_disagree_e) - cos_sim(emb_e, proto_agree_e)
  - If all experts' embeddings are similar (avg_pairwise_sim > threshold):
      use uniform averaging
  - Otherwise: select the expert with the highest disagreement affinity

Usage:
    python scripts/evaluate_dace.py [--data-root DATA] [--checkpoint-dir CKPT]
                                    [--batch-size 128]
                                    [--threshold-agree 0.7]
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32, ResNet32WithRouting
from data.cifar_lt import LongTailCIFAR100


# ── Class groups ─────────────────────────────────────────────────────────

def get_class_groups(class_counts: np.ndarray) -> dict[str, np.ndarray]:
    return {
        'Head': np.where(class_counts >= 100)[0],
        'Med':  np.where((class_counts >= 20) & (class_counts < 100))[0],
        'Tail': np.where(class_counts < 20)[0],
    }


def balanced_accuracy(all_targets: np.ndarray, all_preds: np.ndarray) -> float:
    classes = sorted(set(all_targets.tolist()))
    per_class = []
    for c in classes:
        mask = all_targets == c
        per_class.append((all_preds[mask] == c).sum() / max(mask.sum(), 1))
    return float(np.mean(per_class))


def group_accuracies(
    all_targets: np.ndarray, all_preds: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, float]:
    result = {}
    for name, cls_list in groups.items():
        mask = np.isin(all_targets, cls_list)
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = (all_preds[mask] == all_targets[mask]).sum() / mask.sum()
    return result


# ── Expert loading ───────────────────────────────────────────────────────

def load_expert(checkpoint_path: str, has_routing: bool,
                device: str) -> torch.nn.Module:
    """Load a DACE expert checkpoint."""
    if has_routing:
        model = ResNet32WithRouting(num_classes=100, routing_dim=32)
    else:
        model = ResNet32(num_classes=100)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt['model_state_dict'], strict=False
    )
    model = model.to(device)
    model.eval()
    return model


# ── Prototype computation ────────────────────────────────────────────────

def assert_mask_matches_embedding(emb: torch.Tensor, mask: torch.Tensor,
                                  name: str) -> None:
    """Fail loudly if a boolean mask cannot legally index its embedding.

    Boolean indexing (`emb[mask]`) requires the mask to be on the same device
    as the indexed tensor.  When they diverge, CUDA raises an opaque error:

        RuntimeError: indices should be either on cpu or on the same device
        as the indexed tensor (cpu)

    This helper converts that into a self-describing failure.  It only reads
    `.device` / `.dtype`, so it is unit-testable without an accelerator.

    Raises:
        RuntimeError: on device mismatch or a non-boolean mask.
    """
    if emb.device != mask.device:
        raise RuntimeError(
            f"compute_prototypes: Expert {name} embedding device "
            f"({emb.device}) != KL mask device ({mask.device}); "
            f"boolean indexing would fail. Move both to one device."
        )
    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"compute_prototypes: Expert {name} KL mask dtype is "
            f"{mask.dtype}, expected torch.bool"
        )


@torch.no_grad()
def compute_prototypes(
    expert_a: torch.nn.Module,
    expert_b: torch.nn.Module,
    expert_c: torch.nn.Module,
    val_loader: DataLoader,
    kl_threshold: float = 0.1,
    device: str = 'cpu',
) -> dict:
    """
    Pre-compute divergence prototypes on the validation set.

    For each expert, computes two 32-d prototypes:
      - proto_agree: mean embedding of samples where KL < threshold
      - proto_disagree: mean embedding of samples where KL > threshold

    Expert A has no routing head, so only B and C get prototypes.
    For A, we use B's KL(A || B) as a proxy (since A has no embedding).

    Returns:
        prototypes: dict with keys 'A', 'B', 'C', each containing
                    {'agree': tensor, 'disagree': tensor}
    """
    # Collect all routing embeddings and KL labels.
    #
    # DEVICE RULE: every per-sample tensor accumulated below is reduced to
    # `acc_device` in the same statement that appends it.  Embeddings and the
    # boolean KL masks are later combined by boolean indexing
    # (`emb_all[mask]`), which PyTorch only permits when both operands live on
    # the same device.  Moving only one of them raises, on CUDA:
    #     RuntimeError: indices should be either on cpu or on the same device
    #     as the indexed tensor (cpu)
    # Accumulating on CPU also keeps VRAM flat for the whole validation set.
    acc_device = torch.device('cpu')

    emb_b_list = []
    emb_c_list = []
    kl_b_labels = []   # KL(A || B) > threshold?
    kl_c_labels = []   # KL(avg(A,B) || C) > threshold?

    for images, _ in val_loader:
        images = images.to(device)

        # Expert A (no routing head)
        logits_a = expert_a(images)
        probs_a = F.softmax(logits_a, dim=1)

        # Expert B (with routing head)
        logits_b, emb_b = expert_b(images)
        probs_b = F.softmax(logits_b, dim=1)

        # Expert C (with routing head)
        logits_c, emb_c = expert_c(images)
        probs_c = F.softmax(logits_c, dim=1)

        # KL(A || B) for Expert B's agreement
        kl_ab = (probs_a * (torch.log(probs_a + 1e-12)
                            - torch.log(probs_b + 1e-12))).sum(dim=1)
        kl_b_labels.append((kl_ab > kl_threshold).detach().to(acc_device))

        # KL(avg(A,B) || C) for Expert C's agreement
        avg_probs = (probs_a + probs_b) / 2.0
        kl_ac = (avg_probs * (torch.log(avg_probs + 1e-12)
                              - torch.log(probs_c + 1e-12))).sum(dim=1)
        kl_c_labels.append((kl_ac > kl_threshold).detach().to(acc_device))

        emb_b_list.append(emb_b.detach().to(acc_device))
        emb_c_list.append(emb_c.detach().to(acc_device))

    # Concatenate
    emb_b_all = torch.cat(emb_b_list, dim=0)
    emb_c_all = torch.cat(emb_c_list, dim=0)
    kl_b_all = torch.cat(kl_b_labels, dim=0)
    kl_c_all = torch.cat(kl_c_labels, dim=0)

    # Invariant guard: turn a cryptic cross-device indexing error into an
    # explicit, self-describing failure if this rule is ever broken again.
    assert_mask_matches_embedding(emb_b_all, kl_b_all, 'B')
    assert_mask_matches_embedding(emb_c_all, kl_c_all, 'C')

    # Compute prototypes for Expert B
    has_disagree_b = kl_b_all.sum() > 0
    has_agree_b = (~kl_b_all).sum() > 0
    if has_disagree_b and has_agree_b:
        proto_b_disagree = emb_b_all[kl_b_all].mean(dim=0)
        proto_b_agree = emb_b_all[~kl_b_all].mean(dim=0)
    else:
        # Fallback: use all samples
        proto_b_disagree = emb_b_all.mean(dim=0)
        proto_b_agree = emb_b_all.mean(dim=0)

    # Compute prototypes for Expert C
    has_disagree_c = kl_c_all.sum() > 0
    has_agree_c = (~kl_c_all).sum() > 0
    if has_disagree_c and has_agree_c:
        proto_c_disagree = emb_c_all[kl_c_all].mean(dim=0)
        proto_c_agree = emb_c_all[~kl_c_all].mean(dim=0)
    else:
        proto_c_disagree = emb_c_all.mean(dim=0)
        proto_c_agree = emb_c_all.mean(dim=0)

    # Additive guard: a degenerate split collapses the routing scores to a
    # constant (score = sim(disagree) - sim(agree) = 0), which silently makes
    # argmax always return Expert A. Surface it instead of hiding it.
    n_agree_b = int((~kl_b_all).sum())
    n_disagree_b = int(kl_b_all.sum())
    n_agree_c = int((~kl_c_all).sum())
    n_disagree_c = int(kl_c_all.sum())

    if torch.equal(proto_b_agree, proto_b_disagree):
        print("  [Warning] Expert B agree/disagree prototypes are identical "
              f"(agree={n_agree_b}, disagree={n_disagree_b} of "
              f"{len(kl_b_all)}); B's routing score is constant. "
              "Consider raising --kl-threshold.")
    if torch.equal(proto_c_agree, proto_c_disagree):
        print("  [Warning] Expert C agree/disagree prototypes are identical "
              f"(agree={n_agree_c}, disagree={n_disagree_c} of "
              f"{len(kl_c_all)}); C's routing score is constant. "
              "Consider raising --kl-threshold.")

    # For Expert A (no routing head): create a dummy prototype
    # A is selected when both B and C show "agree" patterns with the ensemble
    proto_a_disagree = torch.zeros(32)
    proto_a_agree = torch.zeros(32)

    prototypes = {
        'A': {'agree': proto_a_agree, 'disagree': proto_a_disagree},
        'B': {'agree': proto_b_agree, 'disagree': proto_b_disagree},
        'C': {'agree': proto_c_agree, 'disagree': proto_c_disagree},
    }

    print(f"\n[Prototypes] Computed from {len(emb_b_all)} validation samples:")
    print(f"  Expert B: agree={int((~kl_b_all).sum())}, "
          f"disagree={int(kl_b_all.sum())} samples")
    print(f"  Expert C: agree={int((~kl_c_all).sum())}, "
          f"disagree={int(kl_c_all.sum())} samples")

    return prototypes


# ── Prototype-based routing ──────────────────────────────────────────────

@torch.no_grad()
def prototype_routing(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    logits_c: torch.Tensor,
    emb_b: torch.Tensor,
    emb_c: torch.Tensor,
    prototypes: dict,
    threshold_agree: float = 0.7,
) -> torch.Tensor:
    """
    Prototype-based disagreement routing for a batch.

    Args:
        logits_a, logits_b, logits_c: (B, C) tensors
        emb_b, emb_c: (B, 32) routing embeddings (Expert A has none)
        prototypes: dict from compute_prototypes()
        threshold_agree: if avg pairwise embedding similarity > this, use uniform

    Returns:
        final_logits: (B, C) tensor after routing decision
    """
    batch_size = logits_a.size(0)

    # Compute disagreement affinity scores for B and C
    emb_b_norm = F.normalize(emb_b, dim=1)
    emb_c_norm = F.normalize(emb_c, dim=1)

    proto_b_agree = F.normalize(prototypes['B']['agree'].unsqueeze(0).to(emb_b.device), dim=1)
    proto_b_disagree = F.normalize(prototypes['B']['disagree'].unsqueeze(0).to(emb_b.device), dim=1)
    proto_c_agree = F.normalize(prototypes['C']['agree'].unsqueeze(0).to(emb_c.device), dim=1)
    proto_c_disagree = F.normalize(prototypes['C']['disagree'].unsqueeze(0).to(emb_c.device), dim=1)

    score_b = F.cosine_similarity(emb_b_norm, proto_b_disagree) \
              - F.cosine_similarity(emb_b_norm, proto_b_agree)
    score_c = F.cosine_similarity(emb_c_norm, proto_c_disagree) \
              - F.cosine_similarity(emb_c_norm, proto_c_agree)

    # Expert A: use negative of avg(B,C) disagreement as proxy
    # (A is preferred when both B and C agree with ensemble)
    score_a = -(score_b + score_c) / 2.0

    # Compute average pairwise embedding similarity for agreement detection
    # We have emb_b and emb_c; emb_a is approximated as 0 (no routing head)
    # Use cosine similarity between B and C as the agreement signal
    emb_sim_bc = F.cosine_similarity(emb_b_norm, emb_c_norm)  # (B,)
    avg_agreement = emb_sim_bc  # for 2 experts, this is the pairwise similarity

    # Stack scores [A, B, C] for each sample
    scores = torch.stack([score_a, score_b, score_c], dim=1)  # (B, 3)
    logits_stack = torch.stack([logits_a, logits_b, logits_c], dim=2)  # (B, C, 3)

    # Decision: if agreement is high, use uniform; otherwise, pick best expert
    use_uniform = (avg_agreement > threshold_agree).float().unsqueeze(1)  # (B, 1)

    # Uniform averaging
    uniform_logits = (logits_a + logits_b + logits_c) / 3.0

    # Selected expert routing
    best_expert = scores.argmax(dim=1)  # (B,)
    selected_logits = logits_stack[torch.arange(batch_size), :, best_expert]  # (B, C)

    # Blend: uniform if agreement is high, selected expert otherwise
    final_logits = use_uniform * uniform_logits + (1.0 - use_uniform) * selected_logits

    return final_logits, scores, avg_agreement, best_expert


# ── Main evaluation ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DACE Evaluation: Prototype-based disagreement routing'
    )
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threshold-agree', type=float, default=0.7,
                        help='Embedding similarity threshold for uniform fallback')
    parser.add_argument('--kl-threshold', type=float, default=0.1,
                        help='KL threshold for prototype computation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    device = args.device

    # ── load experts ───────────────────────────────────────────────────
    print("Loading DACE experts...")
    expert_a = load_expert(os.path.join(ckpt_dir, 'DACE_A_best.pt'),
                           has_routing=False, device=device)
    expert_b = load_expert(os.path.join(ckpt_dir, 'DACE_B_best.pt'),
                           has_routing=True, device=device)
    expert_c = load_expert(os.path.join(ckpt_dir, 'DACE_C_best.pt'),
                           has_routing=True, device=device)
    print(f"  Expert A: {sum(p.numel() for p in expert_a.parameters()):,} params")
    print(f"  Expert B: {sum(p.numel() for p in expert_b.parameters()):,} params")
    print(f"  Expert C: {sum(p.numel() for p in expert_c.parameters()):,} params")

    # ── data ──────────────────────────────────────────────────────────
    val_idx = np.load(f'{args.data_root}/processed/lt_val_indices.npy')
    test_set = LongTailCIFAR100(
        root=args.data_root,
        train=False, download=False,
        use_test_set=True,
    )
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        already_subsampled=True,
    )

    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── class groups ──────────────────────────────────────────────────
    train_idx = np.load(f'{args.data_root}/processed/lt_train_indices.npy')
    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=train_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        already_subsampled=True,
    )
    class_counts = train_set.get_class_counts()
    groups = get_class_groups(class_counts)
    print(f"\nClass groups:")
    for name, indices in groups.items():
        print(f"  {name}: {len(indices)} classes ({indices.tolist()})")

    # ── compute prototypes ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: Computing divergence prototypes (validation set)")
    print(f"{'='*60}")
    prototypes = compute_prototypes(
        expert_a, expert_b, expert_c, val_loader,
        kl_threshold=args.kl_threshold, device=device,
    )

    # ── evaluate ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Test-set evaluation with prototype routing")
    print(f"  Agreement threshold: {args.threshold_agree}")
    print(f"{'='*60}")

    all_targets = []
    all_preds_uniform = []
    all_preds_routed = []
    all_preds_a = []
    all_preds_b = []
    all_preds_c = []
    all_scores = []
    routing_decisions = []

    for images, targets in test_loader:
        images = images.to(device)
        targets_np = targets.numpy()
        batch_size = images.size(0)

        # Forward all experts
        with torch.no_grad():
            logits_a = expert_a(images)
            logits_b, emb_b = expert_b(images)
            logits_c, emb_c = expert_c(images)

        # Prototype routing
        final_logits, scores, avg_agreement, best_expert = prototype_routing(
            logits_a, logits_b, logits_c, emb_b, emb_c,
            prototypes, threshold_agree=args.threshold_agree,
        )

        # Individual predictions
        pred_a = logits_a.argmax(dim=1).cpu().numpy()
        pred_b = logits_b.argmax(dim=1).cpu().numpy()
        pred_c = logits_c.argmax(dim=1).cpu().numpy()
        pred_uniform = ((logits_a + logits_b + logits_c) / 3.0).argmax(dim=1).cpu().numpy()
        pred_routed = final_logits.argmax(dim=1).cpu().numpy()

        all_targets.append(targets_np)
        all_preds_uniform.append(pred_uniform)
        all_preds_routed.append(pred_routed)
        all_preds_a.append(pred_a)
        all_preds_b.append(pred_b)
        all_preds_c.append(pred_c)
        all_scores.append(scores.cpu().numpy())
        routing_decisions.append(best_expert.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds_uniform = np.concatenate(all_preds_uniform)
    all_preds_routed = np.concatenate(all_preds_routed)
    all_preds_a = np.concatenate(all_preds_a)
    all_preds_b = np.concatenate(all_preds_b)
    all_preds_c = np.concatenate(all_preds_c)
    all_scores = np.concatenate(all_scores)
    routing_decisions = np.concatenate(routing_decisions)

    # ── metrics ───────────────────────────────────────────────────────
    ba_uniform = balanced_accuracy(all_targets, all_preds_uniform)
    ba_routed = balanced_accuracy(all_targets, all_preds_routed)
    ba_a = balanced_accuracy(all_targets, all_preds_a)
    ba_b = balanced_accuracy(all_targets, all_preds_b)
    ba_c = balanced_accuracy(all_targets, all_preds_c)

    grp_uniform = group_accuracies(all_targets, all_preds_uniform, groups)
    grp_routed = group_accuracies(all_targets, all_preds_routed, groups)

    # Routing statistics
    n_uniform = (routing_decisions == -1).sum()
    # We can't directly count uniform decisions from best_expert
    # Re-derive from the logic
    n_total = len(routing_decisions)
    n_selected_a = (routing_decisions == 0).sum()
    n_selected_b = (routing_decisions == 1).sum()
    n_selected_c = (routing_decisions == 2).sum()

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"\n  Per-Expert BA:")
    print(f"    Expert A (Head specialist): {ba_a:.2%}")
    print(f"    Expert B (Med specialist):  {ba_b:.2%}")
    print(f"    Expert C (Tail specialist): {ba_c:.2%}")

    print(f"\n  Ensemble BA:")
    print(f"    Uniform averaging:   {ba_uniform:.2%}")
    print(f"    DACE routed:         {ba_routed:.2%}")
    print(f"    Gain over uniform:   {ba_routed - ba_uniform:+.2%}")

    print(f"\n  Group Accuracies (DACE routed):")
    for name in ['Head', 'Med', 'Tail']:
        print(f"    {name}: {grp_routed[name]:.2%} "
              f"(uniform: {grp_uniform[name]:.2%})")

    print(f"\n  Routing Decisions (test set, n={n_total}):")
    print(f"    Selected Expert A: {n_selected_a:5d} ({100*n_selected_a/n_total:.1f}%)")
    print(f"    Selected Expert B: {n_selected_b:5d} ({100*n_selected_b/n_total:.1f}%)")
    print(f"    Selected Expert C: {n_selected_c:5d} ({100*n_selected_c/n_total:.1f}%)")

    print(f"\n{'='*60}")
    if ba_routed > ba_uniform:
        print(f"  ✅ DACE routing improves over uniform!")
    else:
        print(f"  ⚠️ DACE routing does not improve over uniform yet.")
        print(f"     Try tuning --threshold-agree, --kl-threshold")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
