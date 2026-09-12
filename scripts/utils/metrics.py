"""
Evaluation metrics for CIFAR-100-LT experiments.

Provides:
  - balanced_accuracy: Mean per-class recall (standard BA)
  - per_class_accuracy: Per-class accuracy dictionary
  - group_accuracies: Head / Medium / Tail accuracy
  - compute_routing_metrics: Full routing evaluation suite
  - ece: Expected Calibration Error
  - confidence_metrics: Calibration statistics
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# ── Core Metrics ─────────────────────────────────────────────────────────


def balanced_accuracy(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Mean per-class recall (Balanced Accuracy).

    Computes recall for each class independently, then averages.
    Handles missing classes gracefully (they contribute 0).
    """
    if num_classes is None:
        classes = sorted(set(all_targets.tolist()))
    else:
        classes = list(range(num_classes))

    per_class_recalls = []
    for c in classes:
        mask = all_targets == c
        if mask.sum() == 0:
            per_class_recalls.append(0.0)
        else:
            per_class_recalls.append(
                (all_preds[mask] == c).sum() / mask.sum()
            )
    return float(np.mean(per_class_recalls))


def per_class_accuracy(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int = 100,
) -> dict[int, float]:
    """Return dict mapping class index → accuracy for that class."""
    result = {}
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() == 0:
            result[c] = 0.0
        else:
            result[c] = float((all_preds[mask] == c).sum() / mask.sum())
    return result


def group_accuracies(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-group accuracy (Head / Med / Tail).

    Args:
        all_targets: Ground-truth labels, shape (N,).
        all_preds: Predicted labels, shape (N,).
        groups: Dict from ``get_class_groups()``.
    Returns:
        Dict like {'Head': 0.72, 'Med': 0.45, 'Tail': 0.18}.
    """
    result = {}
    for name, cls_list in groups.items():
        mask = np.isin(all_targets, cls_list)
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = float(
                (all_preds[mask] == all_targets[mask]).sum() / mask.sum()
            )
    return result


# ── Routing Metrics ──────────────────────────────────────────────────────


def compute_routing_metrics(
    predictions: np.ndarray,  # shape (N,) — which expert was chosen per sample
    labels: np.ndarray,       # shape (N,) — ground truth
    all_logits: np.ndarray,   # shape (N, num_experts, 100) — all expert logits
    expert_names: list[str],
    class_counts: np.ndarray | None = None,
    class_predictions: np.ndarray | None = None,
    routing_weights: np.ndarray | None = None,
) -> dict:
    """Compute comprehensive routing evaluation metrics.

    Args:
        predictions: Expert index (0..num_experts-1) chosen for each sample.
            Combine-then-argmax rules (uniform, probability averaging) make no
            such choice; they pass a sentinel and supply ``class_predictions``.
        labels: Ground-truth class labels.
        all_logits: All experts' logits [N, num_experts, num_classes].
        expert_names: Names of experts for readable output.
        class_counts: Per-class sample counts (for head/med/tail grouping).
                       If None, uses training set distribution.
        class_predictions: Optional (N,) predicted labels. Overrides the argmax of
            the chosen expert's logits; without it a sentinel ``predictions`` array
            would be read as "every sample chose expert 0".
        routing_weights: Optional (N, num_experts) contribution of each expert.
            Used for the usage report when no single expert is selected.
    Returns:
        Dict with keys: ba, head_acc, med_acc, tail_acc, accuracy,
                        oracle_ba, oracle_gap, all_wrong_pct,
                        per_expert_usage, per_expert_ba.
    """
    num_experts = len(expert_names)
    N = len(labels)

    # Predicted class from the chosen expert, or from the rule's own decision
    if class_predictions is None:
        chosen_logits = all_logits[np.arange(N), predictions]  # (N, 100)
        chosen_preds = chosen_logits.argmax(axis=1)
    else:
        chosen_preds = np.asarray(class_predictions)

    # Overall metrics
    ba = balanced_accuracy(labels, chosen_preds)
    accuracy = float((chosen_preds == labels).mean())

    # Group accuracies
    if class_counts is None:
        # Default CIFAR-100-LT training distribution
        class_counts = np.array([
            int(500 * (100 ** (-i / 99)))
            for i in range(100)
        ])
        class_counts = np.maximum(class_counts, 1)

    from scripts.utils.data import get_class_groups
    groups = get_class_groups(class_counts)
    group_acc = group_accuracies(labels, chosen_preds, groups)

    # Oracle: at least one expert correct?
    expert_preds = all_logits.argmax(axis=2)  # (N, num_experts)
    any_correct = (expert_preds == labels[:, None]).any(axis=1)
    # Proper oracle: for each sample, use the best expert
    oracle_preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        correct = expert_preds[i] == labels[i]
        if correct.any():
            # Pick the first correct expert
            oracle_preds[i] = expert_preds[i][np.where(correct)[0][0]]
        else:
            oracle_preds[i] = expert_preds[i][0]  # all wrong, pick first
    oracle_ba = balanced_accuracy(labels, oracle_preds)
    oracle_gap = oracle_ba - ba

    # All-wrong percentage
    all_wrong = (~any_correct).mean() * 100

    # Per-expert usage: hard selection when the rule picks an expert, otherwise
    # the weights the rule actually combined with.
    if routing_weights is not None:
        weights = np.asarray(routing_weights, dtype=np.float64)
        if weights.shape != (N, num_experts):
            raise ValueError(
                f"routing_weights must be ({N}, {num_experts}), got {weights.shape}"
            )
        expert_usage = weights.mean(axis=0) * 100.0
    else:
        expert_usage = np.array(
            [(predictions == e).mean() * 100 for e in range(num_experts)]
        )

    # Per-expert BA
    expert_ba = []
    for e in range(num_experts):
        e_preds = all_logits[:, e].argmax(axis=1)
        expert_ba.append(balanced_accuracy(labels, e_preds))

    return {
        "ba": ba,
        "accuracy": accuracy,
        "head_acc": group_acc.get("Head", 0.0),
        "med_acc": group_acc.get("Med", 0.0),
        "tail_acc": group_acc.get("Tail", 0.0),
        "oracle_ba": oracle_ba,
        "oracle_gap": oracle_gap,
        "all_wrong_pct": all_wrong,
        "expert_usage": {name: usage for name, usage in zip(expert_names, expert_usage)},
        "expert_ba": {name: ba for name, ba in zip(expert_names, expert_ba)},
    }


# ── Calibration Metrics ──────────────────────────────────────────────────


def ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    Bins are right-closed, ``(lo, hi]``, so a prediction with confidence exactly
    1.0 lands in the last bin instead of falling outside every bin. A max-softmax
    confidence is always > 0, so no sample is lost at the low end. This matches
    ``scripts/evaluation.expected_calibration_error``, the implementation the
    reported numbers come from.

    Args:
        confidences: Predicted confidence (max softmax) per sample.
        correct: Boolean, whether each prediction was correct.
        num_bins: Number of equal-width bins.
    Returns:
        ECE score (lower is better).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece_val = 0.0
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece_val += np.abs(bin_acc - bin_conf) * in_bin.sum()
    return ece_val / len(confidences)


def confidence_metrics(
    confidences: np.ndarray,
    correct: np.ndarray,
) -> dict:
    """Compute calibration statistics.

    Returns:
        Dict with 'ece', 'avg_conf_correct', 'avg_conf_wrong', 'overconfidence'.
    """
    return {
        "ece": ece(confidences, correct),
        "avg_conf_correct": float(confidences[correct].mean()) if correct.any() else 0.0,
        "avg_conf_wrong": float(confidences[~correct].mean()) if (~correct).any() else 0.0,
        "overconfidence": float(
            (confidences[~correct].mean() - (1 - correct.mean()))
            if (~correct).any() else 0.0
        ),
    }
