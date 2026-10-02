"""Pure metrics for expert complementarity and routed predictions.

All arrays are supplied by the caller. Logits use ``(N, 4, C)`` ordering,
predictions use ``(N, 4)``, and labels/counts use ``(N,)``/``(C,)``. These
functions do not load data or fit a router.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.base_trainer import compute_class_groups
from scripts.expert_diagnostics import ExpertDiagnostics

from .contracts import (
    DiagnosticsError,
    EXPERT_NAMES,
    aligned_labeled_arrays,
    finite_array,
    json_safe,
)


def _validate_class_counts(class_counts: Any, num_classes: int) -> np.ndarray:
    counts = finite_array(class_counts, name="class_counts", ndim=1)
    if counts.shape != (num_classes,):
        raise DiagnosticsError(
            f"class_counts must have shape ({num_classes},), got {counts.shape}"
        )
    if np.any(counts <= 0.0) or not np.equal(counts, np.floor(counts)).all():
        raise DiagnosticsError("class_counts must contain positive integer counts")
    return counts.astype(np.int64, copy=False)


def _validate_labels_and_predictions(
    labels: Any, predictions: Any, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(labels)
    predicted = np.asarray(predictions)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise DiagnosticsError("labels must be an integer vector")
    if predicted.ndim != 1 or not np.issubdtype(predicted.dtype, np.integer):
        raise DiagnosticsError("predictions must be an integer vector")
    if len(targets) == 0 or predicted.shape != targets.shape:
        raise DiagnosticsError("labels and predictions must be aligned and non-empty")
    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise DiagnosticsError("labels fall outside class_counts")
    if np.any(predicted < 0) or np.any(predicted >= num_classes):
        raise DiagnosticsError("predictions fall outside class_counts")
    return targets.astype(np.int64, copy=False), predicted.astype(np.int64, copy=False)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _spearman_rho(first: np.ndarray, second: np.ndarray) -> float | None:
    """Compute Spearman rank correlation; return null for constants or n<2."""
    if len(first) < 2 or len(second) != len(first):
        return None
    rank_first = _average_ranks(first)
    rank_second = _average_ranks(second)
    if np.std(rank_first) == 0.0 or np.std(rank_second) == 0.0:
        return None
    return float(np.corrcoef(rank_first, rank_second)[0, 1])


def classify_predictions(
    labels: Any,
    predictions: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Compute canonical sample and macro-recall metrics from aligned classes.

    ``labels`` and ``predictions`` are integer ``(N,)`` vectors; class counts
    are positive integer ``(C,)`` training counts used by the canonical
    Head/Medium/Tail grouping. Sample accuracy remains defined when labels are
    missing. BA is ``None`` unless every output class is represented, and each
    group recall is ``None`` if that group's class set is empty or any member
    class is absent. Missing classes are never silently scored as zero.
    """
    raw_counts = np.asarray(class_counts)
    if raw_counts.ndim != 1 or len(raw_counts) < 2:
        raise DiagnosticsError("class_counts must be a vector with at least two classes")
    counts = _validate_class_counts(raw_counts, len(raw_counts))
    targets, predicted = _validate_labels_and_predictions(
        labels, predictions, len(counts)
    )

    class_rows: list[dict[str, Any]] = []
    recalls = np.full(len(counts), np.nan, dtype=np.float64)
    sample_counts = np.bincount(targets, minlength=len(counts))
    groups = compute_class_groups(counts)
    complete_groups = all(len(class_ids) > 0 for class_ids in groups.values())
    canonical_metrics: dict[str, Any] | None = None
    if complete_groups and np.all(sample_counts > 0):
        # Reuse the project's canonical metric definitions when every
        # denominator is valid. The local fallback below preserves explicit
        # missing-class nulls, which the canonical evaluator intentionally
        # rejects for selection/evaluation work.
        from scripts.task3f_ridge import Task3FError, classification_metrics

        try:
            canonical_metrics = classification_metrics(targets, predicted, counts)
        except Task3FError as exc:
            raise DiagnosticsError(f"canonical classification metrics failed: {exc}") from exc
    for class_id in range(len(counts)):
        n_class = int(sample_counts[class_id])
        if canonical_metrics is not None:
            recall = float(canonical_metrics["per_class_recall"][str(class_id)])
        else:
            recall = (
                float(np.mean(predicted[targets == class_id] == class_id))
                if n_class
                else None
            )
        if recall is not None:
            recalls[class_id] = recall
        class_rows.append(
            {
                "class_id": class_id,
                "sample_count": n_class,
                "recall": recall,
            }
        )

    group_metrics: dict[str, dict[str, Any]] = {}
    flat_group_metrics: dict[str, float | None] = {}
    for group_name, class_ids in groups.items():
        missing = [int(class_id) for class_id in class_ids if sample_counts[class_id] == 0]
        group_recall = (
            float(np.mean(recalls[class_ids]))
            if len(class_ids) > 0 and not missing
            else None
        )
        group_metrics[group_name] = {
            "class_ids": [int(class_id) for class_id in class_ids],
            "num_classes": int(len(class_ids)),
            "sample_count": int(sample_counts[class_ids].sum()) if len(class_ids) else 0,
            "missing_class_ids": missing,
            "macro_recall": group_recall,
        }
        flat_group_metrics[f"{group_name}_accuracy"] = group_recall

    missing_classes = np.flatnonzero(sample_counts == 0).astype(int).tolist()
    balanced_accuracy = (
        canonical_metrics["balanced_accuracy"] if canonical_metrics is not None
        else float(np.mean(recalls)) if not missing_classes else None
    )
    scalar_metrics = {
        "ordinary_accuracy": (
            canonical_metrics["ordinary_accuracy"] if canonical_metrics is not None
            else float(np.mean(predicted == targets))
        ),
        "balanced_accuracy": balanced_accuracy,
        **{
            f"{group_name}_accuracy": (
                canonical_metrics[f"{group_name}_accuracy"]
                if canonical_metrics is not None else flat_group_metrics[f"{group_name}_accuracy"]
            )
            for group_name in groups
        },
    }
    return json_safe(
        {
            "n_samples": int(len(targets)),
            **scalar_metrics,
            "missing_class_ids": missing_classes,
            "per_class": class_rows,
            "groups": group_metrics,
        }
    )


def analyze_complementarity(
    logits: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Describe expert correctness overlap, class frequency, and H/M/T profiles.

    ``logits`` must have shape ``(N, 4, C)`` and labels/counts ``(N,)`` and
    ``(C,)``. Prediction ties use the first expert class index through NumPy's
    stable ``argmax`` rule. Class-level and group macro recalls are ``None``
    when a required class has no labeled rows; sample denominators only include
    observed examples, while macro denominators are class counts. Pairwise
    correctness fractions and agreement matrices use all ``N`` rows. Training
    frequencies divide by the sum of training counts, observed label
    frequencies divide by ``N``, and Spearman associations use only classes
    with observed labels (null for fewer than two classes or a constant input).
    """
    scores, targets = aligned_labeled_arrays(logits, labels)
    counts = _validate_class_counts(class_counts, scores.shape[2])
    groups = compute_class_groups(counts)
    class_to_group = {
        int(class_id): group_name
        for group_name, class_ids in groups.items()
        for class_id in class_ids
    }
    predictions = scores.argmax(axis=2).astype(np.int64)
    # Reuse the project's established overlap and pairwise definitions. Omit
    # class_counts here because that legacy API requires every label class to
    # be present; this diagnostic explicitly represents absent classes as null.
    base = ExpertDiagnostics(
        predictions=predictions,
        logits=scores,
        labels=targets,
        expert_names=EXPERT_NAMES,
    ).complementarity()

    n_samples, n_experts = predictions.shape
    correctness = predictions == targets[:, None]
    any_correct = correctness.any(axis=1)
    exclusive = correctness & (correctness.sum(axis=1) == 1)[:, None]
    sample_counts = np.bincount(targets, minlength=scores.shape[2])
    recall_matrix = np.full((scores.shape[2], n_experts), np.nan, dtype=np.float64)
    any_recall = np.full(scores.shape[2], np.nan, dtype=np.float64)
    all_wrong_recall = np.full(scores.shape[2], np.nan, dtype=np.float64)
    exclusive_recall = np.full((scores.shape[2], n_experts), np.nan, dtype=np.float64)

    class_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    count_total = int(counts.sum())
    pair_disagreement_by_class = np.full(scores.shape[2], np.nan, dtype=np.float64)
    sample_disagreement_by_class = np.full(scores.shape[2], np.nan, dtype=np.float64)
    pair_indices = [(left, right) for left in range(n_experts) for right in range(left + 1, n_experts)]
    for class_id in range(scores.shape[2]):
        mask = targets == class_id
        n_class = int(sample_counts[class_id])
        if n_class:
            recall_matrix[class_id] = correctness[mask].mean(axis=0)
            any_recall[class_id] = float(any_correct[mask].mean())
            all_wrong_recall[class_id] = float((~any_correct[mask]).mean())
            exclusive_recall[class_id] = exclusive[mask].mean(axis=0)
            class_expert_recall: dict[str, float] | None = {
                name: float(value)
                for name, value in zip(EXPERT_NAMES, recall_matrix[class_id])
            }
            ordered_recall = np.sort(recall_matrix[class_id])
            best_recall = float(ordered_recall[-1])
            strongest_experts: list[str] | None = [
                EXPERT_NAMES[index]
                for index, value in enumerate(recall_matrix[class_id])
                if np.isclose(value, best_recall, rtol=0.0, atol=1e-12)
            ]
            best_second_gap: float | None = float(ordered_recall[-1] - ordered_recall[-2])
            pair_disagreement_by_class[class_id] = float(
                np.mean(
                    [
                        np.mean(predictions[mask, left] != predictions[mask, right])
                        for left, right in pair_indices
                    ]
                )
            )
            sample_disagreement_by_class[class_id] = float(
                np.mean(np.any(predictions[mask] != predictions[mask, :1], axis=1))
            )
            any_fraction: float | None = float(any_recall[class_id])
            all_wrong_fraction: float | None = float(all_wrong_recall[class_id])
            exclusive_fraction: dict[str, float] | None = {
                name: float(value)
                for name, value in zip(EXPERT_NAMES, exclusive_recall[class_id])
            }
        else:
            class_expert_recall = None
            strongest_experts = None
            best_recall = None
            best_second_gap = None
            any_fraction = all_wrong_fraction = None
            exclusive_fraction = None
        class_rows.append(
            {
                "class_id": class_id,
                "sample_count": n_class,
                "expert_correct_fraction": class_expert_recall,
                "strongest_experts": strongest_experts,
                "strongest_expert_recall": best_recall,
                "best_second_recall_gap": best_second_gap,
                "pairwise_prediction_disagreement_fraction": (
                    float(pair_disagreement_by_class[class_id]) if n_class else None
                ),
                "sample_with_any_prediction_disagreement_fraction": (
                    float(sample_disagreement_by_class[class_id]) if n_class else None
                ),
                "any_correct_fraction": any_fraction,
                "all_wrong_fraction": all_wrong_fraction,
                "exclusive_correct_fraction": exclusive_fraction,
            }
        )
        frequency_rows.append(
            {
                "class_id": class_id,
                "group": class_to_group[class_id],
                "training_count": int(counts[class_id]),
                "training_fraction": float(counts[class_id] / count_total),
                "sample_count": n_class,
                "sample_fraction": float(n_class / n_samples),
                "observed_label_frequency": float(n_class / n_samples),
            }
        )

    per_expert: list[dict[str, Any]] = []
    individual_metrics = [
        classify_predictions(targets, predictions[:, expert_id], counts)
        for expert_id in range(n_experts)
    ]
    for expert_id, name in enumerate(EXPERT_NAMES):
        correct_count = int(correctness[:, expert_id].sum())
        complete = bool(np.isfinite(recall_matrix[:, expert_id]).all())
        per_expert.append(
            {
                "expert": name,
                "correct_count": correct_count,
                "correct_fraction": float(correct_count / n_samples),
                "sample_accuracy": float(correct_count / n_samples),
                "balanced_accuracy": (
                    float(np.mean(recall_matrix[:, expert_id])) if complete else None
                ),
                "class_coverage": float(np.mean(sample_counts > 0)),
                "canonical_metrics": individual_metrics[expert_id],
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group_name, class_ids in groups.items():
        ids = np.asarray(class_ids, dtype=np.int64)
        present = bool(len(ids)) and bool(np.all(sample_counts[ids] > 0))
        group_rows.append(
            {
                "group": group_name,
                "class_ids": ids.astype(int).tolist(),
                "num_classes": int(len(ids)),
                "sample_count": int(sample_counts[ids].sum()) if len(ids) else 0,
                "complete_class_coverage": present,
                "expert_macro_recall": (
                    {
                        name: float(value)
                        for name, value in zip(
                            EXPERT_NAMES, recall_matrix[ids].mean(axis=0)
                        )
                    }
                    if present
                    else None
                ),
                "any_correct_macro_recall": (
                    float(np.mean(any_recall[ids])) if present else None
                ),
                "all_wrong_macro_recall": (
                    float(np.mean(all_wrong_recall[ids])) if present else None
                ),
                "exclusive_correct_macro_recall": (
                    {
                        name: float(value)
                        for name, value in zip(
                            EXPERT_NAMES, exclusive_recall[ids].mean(axis=0)
                        )
                    }
                    if present
                    else None
                ),
            }
        )

    # Label-dependent hard-selection oracle. It selects the first correct
    # expert; when all experts miss, it falls back to expert 0.
    selected_expert = np.zeros(n_samples, dtype=np.int64)
    for sample_id in np.flatnonzero(any_correct):
        selected_expert[sample_id] = int(np.flatnonzero(correctness[sample_id])[0])
    oracle_predictions = predictions[np.arange(n_samples), selected_expert]
    oracle_metrics = classify_predictions(targets, oracle_predictions, counts)
    metric_comparisons: dict[str, Any] = {}
    metric_names = (
        "ordinary_accuracy",
        "balanced_accuracy",
        "head_accuracy",
        "medium_accuracy",
        "tail_accuracy",
    )
    for metric_name in metric_names:
        oracle_value = oracle_metrics[metric_name]
        candidate_values = [metrics[metric_name] for metrics in individual_metrics]
        defined = [
            (EXPERT_NAMES[index], value)
            for index, value in enumerate(candidate_values)
            if value is not None
        ]
        if oracle_value is None or not defined:
            metric_comparisons[metric_name] = {
                "oracle": oracle_value,
                "strongest_individual": None,
                "strongest_expert_names": [],
                "oracle_minus_strongest_individual": None,
            }
            continue
        strongest_value = max(float(value) for _, value in defined)
        metric_comparisons[metric_name] = {
            "oracle": float(oracle_value),
            "strongest_individual": strongest_value,
            "strongest_expert_names": [
                name
                for name, value in defined
                if np.isclose(value, strongest_value, rtol=0.0, atol=1e-12)
            ],
            "oracle_minus_strongest_individual": float(oracle_value - strongest_value),
        }

    agreement_counts = np.zeros((n_experts, n_experts), dtype=np.int64)
    for left in range(n_experts):
        for right in range(n_experts):
            agreement_counts[left, right] = int(
                np.sum(predictions[:, left] == predictions[:, right])
            )
    disagreement_counts = n_samples - agreement_counts

    frequency_associations: list[dict[str, Any]] = []
    association_outcomes: list[tuple[str, str | None, np.ndarray]] = [
        ("any_expert_correct_fraction", None, any_recall),
        ("all_experts_wrong_fraction", None, all_wrong_recall),
    ]
    for expert_id, expert_name in enumerate(EXPERT_NAMES):
        association_outcomes.append(
            ("expert_correct_fraction", expert_name, recall_matrix[:, expert_id])
        )
        association_outcomes.append(
            ("exclusive_correct_fraction", expert_name, exclusive_recall[:, expert_id])
        )
    observed_class_mask = sample_counts > 0
    for outcome, expert_name, outcome_values in association_outcomes:
        valid = observed_class_mask & np.isfinite(outcome_values)
        frequency_associations.append(
            {
                "frequency_variable": "training_count",
                "outcome": outcome,
                "expert": expert_name,
                "n_classes": int(valid.sum()),
                "spearman_rho": _spearman_rho(counts[valid], outcome_values[valid]),
                "interpretation": (
                    "descriptive class-level rank association; not a causal estimate"
                ),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for left, left_name in enumerate(EXPERT_NAMES):
        for right in range(left + 1, n_experts):
            right_name = EXPERT_NAMES[right]
            both_correct = correctness[:, left] & correctness[:, right]
            left_only = correctness[:, left] & ~correctness[:, right]
            right_only = ~correctness[:, left] & correctness[:, right]
            both_wrong = ~correctness[:, left] & ~correctness[:, right]
            prediction_agreement = predictions[:, left] == predictions[:, right]
            pair_key = f"{left_name}|{right_name}"
            pairwise_rows.append(
                {
                    "pair": pair_key,
                    "experts": [left_name, right_name],
                    "both_correct_count": int(both_correct.sum()),
                    "a_only_correct_count": int(left_only.sum()),
                    "b_only_correct_count": int(right_only.sum()),
                    "both_wrong_count": int(both_wrong.sum()),
                    "both_correct_fraction": float(both_correct.mean()),
                    "a_only_correct_fraction": float(left_only.mean()),
                    "b_only_correct_fraction": float(right_only.mean()),
                    "both_wrong_fraction": float(both_wrong.mean()),
                    "prediction_agreement_count": int(prediction_agreement.sum()),
                    "prediction_agreement_fraction": float(prediction_agreement.mean()),
                    "prediction_disagreement_fraction": float((~prediction_agreement).mean()),
                    "correctness_contingency": [
                        [int(both_correct.sum()), int(left_only.sum())],
                        [int(right_only.sum()), int(both_wrong.sum())],
                    ],
                    "base_diagnostic": base["pairwise"][pair_key],
                }
            )

    return json_safe(
        {
            "summary": {
                "n_samples": int(n_samples),
                "num_experts": int(n_experts),
                "num_classes": int(scores.shape[2]),
                "any_expert_correct_count": int(any_correct.sum()),
                "any_expert_correct_fraction": float(any_correct.mean()),
                "all_experts_wrong_count": int((~any_correct).sum()),
                "all_experts_wrong_fraction": float((~any_correct).mean()),
                "hard_selection_oracle_metrics": oracle_metrics,
                "hard_oracle_gaps_vs_strongest_individual": metric_comparisons,
                "agreement_pattern_counts": base["agreement_patterns"]["pattern_counts"],
                "agreement_pattern_fractions": base["agreement_patterns"]["pattern_fractions"],
            },
            "per_expert": per_expert,
            "per_class": class_rows,
            "group": group_rows,
            "pairwise": pairwise_rows,
            "expert_pairwise_agreement_counts": agreement_counts.tolist(),
            "expert_pairwise_agreement_fractions": (agreement_counts / n_samples).tolist(),
            "expert_pairwise_disagreement_counts": disagreement_counts.tolist(),
            "expert_pairwise_disagreement_fractions": (disagreement_counts / n_samples).tolist(),
            "frequency": frequency_rows,
            "frequency_associations": frequency_associations,
        }
    )
