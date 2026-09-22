"""Task 3F-D: combined inference-time Tail-signal diagnostics.

This module is a read-only diagnostic over the frozen Task 3C OOF logits.  It
constructs four prediction/confidence signals without labels, evaluates the
predefined signal conjunctions retrospectively, and conditions the saved
Task 3F-A Ridge weights on those masks.  It does not fit a router, alter a
weight, retrain an expert, or access an evaluation population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from scripts.task3e_fixed import (
    EXPERT_ORDER,
    RestrictedAnalysisDataset,
    load_restricted_analysis_dataset,
)
from scripts.task3f_ridge import (
    FEATURE_SET_CONFIDENCE,
    PERMITTED_ANALYSIS_INNER_FOLDS,
    combine_weighted_logits,
    extract_features,
)
from scripts.task3f_tail_signal_diagnostics import (
    HIGHLIGHTED_CONFIGURATION_ID,
    assign_class_groups,
    validate_signal_alignment,
)


TASK3FD_SCHEMA_VERSION = "task3f_combined_signal_diagnostics.v1"
TASK3FD_TASK_IDENTIFIER = "Task 3F-D"
TASK3FA_TASK_IDENTIFIER = "Task 3F-A"
TASK3FB_TASK_IDENTIFIER = "Task 3F-B"
TASK3FC_TASK_IDENTIFIER = "Task 3F-C"
DATASET_NAME = "CIFAR-100-LT"
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
OUTER_FOLD = 0
FOLD_GENERATION_SEED = 42
TRAINING_SEED = 78
GROUP_NAMES = ("head", "medium", "tail")
SIGNAL_NAMES = ("A", "B", "C", "D")
SIGNAL_COMBINATION_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A", ("A",)),
    ("B", ("B",)),
    ("C", ("C",)),
    ("D", ("D",)),
    ("AB", ("A", "B")),
    ("AC", ("A", "C")),
    ("AD", ("A", "D")),
    ("BC", ("B", "C")),
    ("BD", ("B", "D")),
    ("CD", ("C", "D")),
    ("ABC", ("A", "B", "C")),
    ("ABD", ("A", "B", "D")),
    ("ACD", ("A", "C", "D")),
    ("BCD", ("B", "C", "D")),
    ("ABCD", ("A", "B", "C", "D")),
)
SIGNAL_COMBINATION_IDS = tuple(
    name for name, _ in SIGNAL_COMBINATION_REQUIREMENTS
)
MIXUP_INDEX = EXPERT_ORDER.index("Mixup")
LAL_INDEX = EXPERT_ORDER.index("LAL")
BALANCEDSOFTMAX_INDEX = EXPERT_ORDER.index("BalancedSoftmax")
WEIGHT_SUM_ATOL = 2e-6


class Task3FDCombinedSignalDiagnosticError(OOFProtocolError):
    """Raised when Task 3F-D input or calculation contracts are violated."""


# A shorter compatibility name is useful when importing the module in focused
# tests and follows the naming used by the neighboring diagnostics.
Task3FDDiagnosticError = Task3FDCombinedSignalDiagnosticError


def _as_numeric(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FDCombinedSignalDiagnosticError(
            f"{name} must contain real numeric values"
        )
    if not np.isfinite(array).all():
        raise Task3FDCombinedSignalDiagnosticError(f"{name} contains non-finite values")
    return array


def _as_integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise Task3FDCombinedSignalDiagnosticError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FDCombinedSignalDiagnosticError(f"{name} must contain integers")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise Task3FDCombinedSignalDiagnosticError(
            f"{name} must contain finite integer values"
        )
    return array.astype(np.int64, copy=False)


def _validate_class_counts(class_counts: Any) -> np.ndarray:
    counts = _as_integer_vector(class_counts, name="class_counts")
    if len(counts) < 2 or np.any(counts < 1):
        raise Task3FDCombinedSignalDiagnosticError(
            "class_counts must contain at least two positive counts"
        )
    return counts


def _validate_predictions(
    expert_predictions: Any,
    *,
    class_counts: np.ndarray | None = None,
) -> np.ndarray:
    array = _as_numeric(expert_predictions, name="expert predictions")
    if array.ndim != 2 or array.shape[1] != len(EXPERT_ORDER):
        raise Task3FDCombinedSignalDiagnosticError(
            f"expert predictions must have shape (samples, {len(EXPERT_ORDER)})"
        )
    if not np.equal(array, np.floor(array)).all():
        raise Task3FDCombinedSignalDiagnosticError(
            "expert predictions must contain integer class IDs"
        )
    predictions = array.astype(np.int64, copy=False)
    if class_counts is not None and (
        np.any(predictions < 0) or np.any(predictions >= len(class_counts))
    ):
        raise Task3FDCombinedSignalDiagnosticError(
            "expert predictions fall outside class_counts"
        )
    return predictions


def _validate_confidences(confidences: Any, *, num_samples: int) -> np.ndarray:
    array = _as_numeric(confidences, name="expert confidences")
    if array.ndim != 2 or array.shape != (num_samples, len(EXPERT_ORDER)):
        raise Task3FDCombinedSignalDiagnosticError(
            f"expert confidences must have shape ({num_samples}, {len(EXPERT_ORDER)})"
        )
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise Task3FDCombinedSignalDiagnosticError(
            "expert confidences must lie in [0, 1]"
        )
    return array.astype(np.float64, copy=False)


def _validate_weights(weights: Any, *, num_samples: int) -> np.ndarray:
    array = _as_numeric(weights, name="Ridge weights")
    if array.ndim == 1:
        if array.shape != (len(EXPERT_ORDER),):
            raise Task3FDCombinedSignalDiagnosticError(
                "Ridge weights must have four expert columns"
            )
        array = np.repeat(array[None, :], num_samples, axis=0)
    if array.ndim != 2 or array.shape != (num_samples, len(EXPERT_ORDER)):
        raise Task3FDCombinedSignalDiagnosticError(
            f"Ridge weights must have shape ({num_samples}, {len(EXPERT_ORDER)})"
        )
    if np.any(array < 0.0) or not np.allclose(
        array.sum(axis=1), 1.0, rtol=0.0, atol=WEIGHT_SUM_ATOL
    ):
        raise Task3FDCombinedSignalDiagnosticError(
            "Ridge weights must be non-negative and sum to one"
        )
    return array.astype(np.float64, copy=False)


def _validate_signal_mapping(
    signals: Mapping[str, Any],
    *,
    required_names: Sequence[str],
) -> dict[str, np.ndarray]:
    if not isinstance(signals, Mapping):
        raise Task3FDCombinedSignalDiagnosticError("signals must be a mapping")
    missing = [name for name in required_names if name not in signals]
    if missing:
        raise Task3FDCombinedSignalDiagnosticError(
            "signals are missing: " + ", ".join(missing)
        )
    arrays: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for name in required_names:
        array = np.asarray(signals[name])
        if array.ndim != 1 or array.dtype != np.bool_:
            raise Task3FDCombinedSignalDiagnosticError(
                f"signal {name} must be a one-dimensional Boolean mask"
            )
        arrays[name] = array.astype(bool, copy=False)
        lengths.add(len(array))
    if len(lengths) != 1:
        raise Task3FDCombinedSignalDiagnosticError("signal masks are not sample-aligned")
    return arrays


def build_inference_time_signals(
    expert_predictions: Any,
    confidences: Any,
    class_counts: Any,
) -> dict[str, np.ndarray]:
    """Construct signals A--D using predictions/confidences only.

    ``class_counts`` is the canonical training-frequency vector and is used
    only to map predicted class IDs to the canonical predicted class group for
    signal C.  True labels are intentionally not an argument to this function.
    """
    counts = _validate_class_counts(class_counts)
    predictions = _validate_predictions(expert_predictions, class_counts=counts)
    confidence_array = _validate_confidences(
        confidences, num_samples=len(predictions)
    )
    predicted_groups = np.column_stack(
        [assign_class_groups(predictions[:, index], counts) for index in range(4)]
    )
    signals = {
        "A": predictions[:, LAL_INDEX] == predictions[:, BALANCEDSOFTMAX_INDEX],
        "B": (
            (predictions[:, MIXUP_INDEX] != predictions[:, LAL_INDEX])
            & (predictions[:, MIXUP_INDEX] != predictions[:, BALANCEDSOFTMAX_INDEX])
        ),
        "C": (
            (predicted_groups[:, LAL_INDEX] == "tail")
            & (predicted_groups[:, BALANCEDSOFTMAX_INDEX] == "tail")
        ),
        "D": (
            (confidence_array[:, LAL_INDEX] > confidence_array[:, MIXUP_INDEX])
            & (
                confidence_array[:, BALANCEDSOFTMAX_INDEX]
                > confidence_array[:, MIXUP_INDEX]
            )
        ),
    }
    return {name: np.asarray(mask, dtype=bool) for name, mask in signals.items()}


# Natural aliases make the label-free seam explicit to callers without
# creating separate implementations.
construct_inference_signals = build_inference_time_signals
build_signals = build_inference_time_signals


def build_signal_combinations(signals: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Build exactly the 15 frozen conjunction masks."""
    base = _validate_signal_mapping(signals, required_names=SIGNAL_NAMES)
    return {
        name: np.logical_and.reduce([base[signal] for signal in required])
        for name, required in SIGNAL_COMBINATION_REQUIREMENTS
    }


def build_all_signal_masks(
    expert_predictions: Any,
    confidences: Any,
    class_counts: Any,
) -> dict[str, np.ndarray]:
    """Return the four individual and 15 predefined signal masks."""
    individual = build_inference_time_signals(
        expert_predictions, confidences, class_counts
    )
    return {**individual, **build_signal_combinations(individual)}


def detect_equivalent_signal_masks(
    masks: Mapping[str, Any],
) -> list[list[str]]:
    """Group masks with identical membership, preserving mapping order."""
    if not isinstance(masks, Mapping):
        raise Task3FDCombinedSignalDiagnosticError("masks must be a mapping")
    arrays = _validate_signal_mapping(masks, required_names=tuple(masks.keys()))
    groups: list[list[str]] = []
    representatives: list[np.ndarray] = []
    for name, array in arrays.items():
        for group_index, representative in enumerate(representatives):
            if np.array_equal(array, representative):
                groups[group_index].append(str(name))
                break
        else:
            representatives.append(array)
            groups.append([str(name)])
    return groups


find_equivalent_signal_masks = detect_equivalent_signal_masks


def _fraction(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(count / denominator)


def _validate_labels(labels: Any, *, num_samples: int, class_counts: np.ndarray) -> np.ndarray:
    array = _as_integer_vector(labels, name="labels")
    if len(array) != num_samples:
        raise Task3FDCombinedSignalDiagnosticError(
            "labels and predictions are not sample-aligned"
        )
    if np.any(array < 0) or np.any(array >= len(class_counts)):
        raise Task3FDCombinedSignalDiagnosticError("labels fall outside class_counts")
    return array


def tail_detection_diagnostics(
    masks: Mapping[str, Any],
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Evaluate Tail selection after the inference-time masks are frozen."""
    arrays = _validate_signal_mapping(masks, required_names=SIGNAL_COMBINATION_IDS)
    counts = _validate_class_counts(class_counts)
    labels_array = _validate_labels(
        labels, num_samples=len(next(iter(arrays.values()))), class_counts=counts
    )
    true_groups = assign_class_groups(labels_array, counts)
    actual_group_counts = {
        name: int(np.sum(true_groups == name)) for name in GROUP_NAMES
    }
    actual_tail_count = actual_group_counts["tail"]
    reports: dict[str, Any] = {}
    for name, mask in arrays.items():
        selected_count = int(mask.sum())
        selected_groups = {
            group_name: int(np.sum(mask & (true_groups == group_name)))
            for group_name in GROUP_NAMES
        }
        tail_count = selected_groups["tail"]
        reports[name] = {
            "signal_id": name,
            "required_signals": list(
                dict(SIGNAL_COMBINATION_REQUIREMENTS)[name]
                if name in SIGNAL_COMBINATION_IDS
                else (name,)
            ),
            "selected_count": selected_count,
            "selected_fraction": _fraction(selected_count, len(labels_array)),
            "selected_true_group_counts": selected_groups,
            "false_positive_count": selected_count - tail_count,
            "tail_precision": _fraction(tail_count, selected_count),
            "tail_recall": _fraction(tail_count, actual_tail_count),
        }
    return {
        "population_count": int(len(labels_array)),
        "actual_true_group_counts": actual_group_counts,
        "tail_prevalence": _fraction(actual_tail_count, len(labels_array)),
        "actual_tail_count": actual_tail_count,
        "reports": reports,
    }


def _correctness_category_masks(
    expert_predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    lal_correct = expert_predictions[:, LAL_INDEX] == labels
    balanced_correct = expert_predictions[:, BALANCEDSOFTMAX_INDEX] == labels
    mixup_correct = expert_predictions[:, MIXUP_INDEX] == labels
    return {
        "LAL_correct_Mixup_wrong": lal_correct & ~mixup_correct,
        "BalancedSoftmax_correct_Mixup_wrong": balanced_correct & ~mixup_correct,
        "either_rebalanced_correct_Mixup_wrong": (
            (lal_correct | balanced_correct) & ~mixup_correct
        ),
        "Mixup_correct_both_rebalanced_wrong": (
            mixup_correct & ~lal_correct & ~balanced_correct
        ),
        "all_three_wrong": ~lal_correct & ~balanced_correct & ~mixup_correct,
    }


def _outcome_counts(
    selected_mask: np.ndarray,
    category_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    selected_count = int(selected_mask.sum())
    return {
        category: {
            "count": int(np.sum(selected_mask & category_mask)),
            "fraction": _fraction(
                int(np.sum(selected_mask & category_mask)), selected_count
            ),
            "denominator_count": selected_count,
        }
        for category, category_mask in category_masks.items()
    }


def correctness_opportunity_diagnostics(
    masks: Mapping[str, Any],
    expert_predictions: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Report retrospective expert-correctness outcomes for every mask."""
    counts = _validate_class_counts(class_counts)
    predictions = _validate_predictions(expert_predictions, class_counts=counts)
    labels_array = _validate_labels(
        labels, num_samples=len(predictions), class_counts=counts
    )
    arrays = _validate_signal_mapping(masks, required_names=SIGNAL_COMBINATION_IDS)
    if len(next(iter(arrays.values()))) != len(predictions):
        raise Task3FDCombinedSignalDiagnosticError(
            "signal masks and predictions are not sample-aligned"
        )
    true_groups = assign_class_groups(labels_array, counts)
    category_masks = _correctness_category_masks(predictions, labels_array)
    reports: dict[str, Any] = {}
    for name, selected_mask in arrays.items():
        reports[name] = {
            "selected_count": int(selected_mask.sum()),
            "overall": _outcome_counts(selected_mask, category_masks),
            "by_true_group": {
                group_name: {
                    "selected_count": int(
                        np.sum(selected_mask & (true_groups == group_name))
                    ),
                    "outcomes": _outcome_counts(
                        selected_mask & (true_groups == group_name), category_masks
                    ),
                }
                for group_name in GROUP_NAMES
            },
        }
    return {
        "category_definitions": {
            "LAL_correct_Mixup_wrong": "LAL correct, Mixup wrong",
            "BalancedSoftmax_correct_Mixup_wrong": "BalancedSoftmax correct, Mixup wrong",
            "either_rebalanced_correct_Mixup_wrong": (
                "LAL or BalancedSoftmax correct, Mixup wrong"
            ),
            "Mixup_correct_both_rebalanced_wrong": (
                "Mixup correct, both LAL and BalancedSoftmax wrong"
            ),
            "all_three_wrong": "LAL, BalancedSoftmax and Mixup all wrong",
        },
        "expert_order": list(EXPERT_ORDER),
        "reports": reports,
    }


def _weight_stats(mask: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    selected_count = int(mask.sum())
    selected_weights = weights[mask]
    if selected_count == 0:
        mean_weight = {expert: None for expert in EXPERT_ORDER}
        std_weight = {expert: None for expert in EXPERT_ORDER}
        highest_counts = {expert: 0 for expert in EXPERT_ORDER}
        highest_fractions = {expert: None for expert in EXPERT_ORDER}
        mixup_highest_count = 0
        mixup_highest_fraction = None
    else:
        highest = selected_weights == selected_weights.max(axis=1, keepdims=True)
        highest_counts = {
            expert: int(highest[:, index].sum())
            for index, expert in enumerate(EXPERT_ORDER)
        }
        highest_fractions = {
            expert: _fraction(highest_counts[expert], selected_count)
            for expert in EXPERT_ORDER
        }
        mean_weight = {
            expert: float(selected_weights[:, index].mean())
            for index, expert in enumerate(EXPERT_ORDER)
        }
        std_weight = {
            expert: float(selected_weights[:, index].std())
            for index, expert in enumerate(EXPERT_ORDER)
        }
        mixup_highest_count = highest_counts["Mixup"]
        mixup_highest_fraction = highest_fractions["Mixup"]
    return {
        "selected_count": selected_count,
        "mean_weight": mean_weight,
        "std_weight": std_weight,
        "highest_weight_counts": highest_counts,
        "highest_weight_fractions": highest_fractions,
        "mixup_highest_weight_count": mixup_highest_count,
        "mixup_highest_weight_fraction": mixup_highest_fraction,
        "highest_weight_ties_count": (
            None
            if selected_count == 0
            else int(
                np.sum(
                    (selected_weights == selected_weights.max(axis=1, keepdims=True)).sum(axis=1)
                    > 1
                )
            )
        ),
    }


def ridge_weight_diagnostics(
    masks: Mapping[str, Any],
    weights: Any,
) -> dict[str, Any]:
    """Condition the saved Ridge weights on every frozen signal mask."""
    arrays = _validate_signal_mapping(masks, required_names=SIGNAL_COMBINATION_IDS)
    num_samples = len(next(iter(arrays.values())))
    weight_array = _validate_weights(weights, num_samples=num_samples)
    overall = _weight_stats(np.ones(num_samples, dtype=bool), weight_array)
    reports: dict[str, Any] = {}
    for name, mask in arrays.items():
        stats = _weight_stats(mask, weight_array)
        stats["mean_weight_delta_vs_overall"] = {
            expert: (
                None
                if stats["mean_weight"][expert] is None
                else float(
                    stats["mean_weight"][expert] - overall["mean_weight"][expert]
                )
            )
            for expert in EXPERT_ORDER
        }
        reports[name] = stats
    return {
        "weight_definition": (
            "saved held-out weights from " + HIGHLIGHTED_CONFIGURATION_ID
        ),
        "mixup_highest_weight_tie_policy": "a tie counts as receiving the highest weight",
        "overall": overall,
        "reports": reports,
    }


def _opportunity_stat(
    correctness_report: Mapping[str, Any],
    *,
    group_name: str | None = None,
) -> Mapping[str, Any]:
    if group_name is None:
        return correctness_report["overall"]["either_rebalanced_correct_Mixup_wrong"]
    return correctness_report["by_true_group"][group_name]["outcomes"][
        "either_rebalanced_correct_Mixup_wrong"
    ]


def compare_combined_signals(
    masks: Mapping[str, Any],
    tail_report: Mapping[str, Any],
    correctness_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare conjunctions with each constituent without choosing a winner."""
    arrays = _validate_signal_mapping(masks, required_names=SIGNAL_COMBINATION_IDS)
    tail_reports = tail_report["reports"]
    correctness_reports = correctness_report["reports"]
    comparisons: dict[str, Any] = {}
    definitions = dict(SIGNAL_COMBINATION_REQUIREMENTS)
    for name in SIGNAL_COMBINATION_IDS:
        required = definitions[name]
        if len(required) == 1:
            continue
        combined_mask = arrays[name]
        constituent_reports: dict[str, Any] = {}
        for constituent in required:
            constituent_mask = arrays[constituent]
            combined_tail = tail_reports[name]
            constituent_tail = tail_reports[constituent]
            combined_opp = _opportunity_stat(correctness_reports[name])
            constituent_opp = _opportunity_stat(correctness_reports[constituent])
            constituent_reports[constituent] = {
                "selected_count": constituent_tail["selected_count"],
                "selected_true_group_counts": constituent_tail[
                    "selected_true_group_counts"
                ],
                "tail_count": constituent_tail["selected_true_group_counts"]["tail"],
                "tail_precision": constituent_tail["tail_precision"],
                "tail_recall": constituent_tail["tail_recall"],
                "opportunity_count": constituent_opp["count"],
                "opportunity_fraction": constituent_opp["fraction"],
                "opportunity_denominator_count": constituent_opp[
                    "denominator_count"
                ],
                "combined_is_subset": bool(np.all(~combined_mask | constituent_mask)),
                "overlap_count": int(np.sum(combined_mask & constituent_mask)),
                "excluded_from_constituent_count": int(
                    np.sum(constituent_mask & ~combined_mask)
                ),
                "tail_precision_delta": (
                    None
                    if combined_tail["tail_precision"] is None
                    or constituent_tail["tail_precision"] is None
                    else float(
                        combined_tail["tail_precision"]
                        - constituent_tail["tail_precision"]
                    )
                ),
                "tail_recall_delta": (
                    None
                    if combined_tail["tail_recall"] is None
                    or constituent_tail["tail_recall"] is None
                    else float(
                        combined_tail["tail_recall"]
                        - constituent_tail["tail_recall"]
                    )
                ),
                "opportunity_fraction_delta": (
                    None
                    if combined_opp["fraction"] is None
                    or constituent_opp["fraction"] is None
                    else float(combined_opp["fraction"] - constituent_opp["fraction"])
                ),
                "combined": {
                    "selected_count": combined_tail["selected_count"],
                    "tail_count": combined_tail["selected_true_group_counts"]["tail"],
                    "tail_precision": combined_tail["tail_precision"],
                    "tail_recall": combined_tail["tail_recall"],
                    "opportunity_count": combined_opp["count"],
                    "opportunity_fraction": combined_opp["fraction"],
                    "opportunity_denominator_count": combined_opp[
                        "denominator_count"
                    ],
                },
            }
        comparisons[name] = {
            "required_signals": list(required),
            "selected_count": int(combined_mask.sum()),
            "constituents": constituent_reports,
            "all_constituent_subset_checks_pass": all(
                value["combined_is_subset"] for value in constituent_reports.values()
            ),
        }
    return {
        "comparisons": comparisons,
        "interpretation_contract": (
            "Conjunctions are compared with their frozen constituents; no combination "
            "is selected as a routing rule."
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Task3FDCombinedSignalDiagnosticError(
            f"cannot hash input artifact: {path}"
        ) from exc
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Task3FDCombinedSignalDiagnosticError(
            f"cannot load {name}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise Task3FDCombinedSignalDiagnosticError(f"{name} must be a JSON object")
    return payload


def _load_npz(path: Path, *, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise Task3FDCombinedSignalDiagnosticError(
            f"cannot load {name}: {path}"
        ) from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task3FDCombinedSignalDiagnosticError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise Task3FDCombinedSignalDiagnosticError(
                f"refusing to overwrite incompatible output: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise Task3FDCombinedSignalDiagnosticError(
                f"refusing to overwrite incompatible output: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _find_unique_index(values: Any, target: str, *, name: str) -> int:
    array = np.asarray(values).astype(str)
    matches = np.flatnonzero(array == target)
    if len(matches) != 1:
        raise Task3FDCombinedSignalDiagnosticError(
            f"{name} must contain exactly one {target!r}, found {len(matches)}"
        )
    return int(matches[0])


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _validate_task3f_a_artifacts(
    *,
    dataset: RestrictedAnalysisDataset,
    config: Mapping[str, Any],
    ridge_results: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    fold_assignments: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact saved Ridge row alignment and return its highlighted arrays."""
    _require(
        config.get("task_identifier") == TASK3FA_TASK_IDENTIFIER,
        "Task 3F-A config has the wrong task identifier",
    )
    _require(
        config.get("expert_order") == list(EXPERT_ORDER),
        "Task 3F-A expert order is incompatible",
    )
    _require(
        config.get("analyzed_sample_count") == dataset.num_samples == ANALYSIS_SAMPLE_COUNT,
        "Task 3F-A population is not the permitted 6,507 rows",
    )
    _require(
        config.get("permitted_analysis_inner_folds")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "Task 3F-A permitted folds are incompatible",
    )
    _require(
        config.get("reserved_router_selection_inner_folds") == [0],
        "Task 3F-A reserved fold declaration is incompatible",
    )
    restrictions = config.get("data_restrictions", {})
    for key in (
        "inner_fold_zero_used",
        "reserved_outer_evaluation_used",
        "original_cifar_test_used",
        "expert_training_or_checkpoints_modified",
        "source_oof_logits_modified",
    ):
        _require(restrictions.get(key) is False, f"Task 3F-A restriction {key} failed")
    _require(
        fold_assignments.get("outer_fold") == OUTER_FOLD
        and fold_assignments.get("permitted_inner_fold_ids")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and fold_assignments.get("every_sample_held_out_exactly_once") is True,
        "Task 3F-A fold assignment provenance is incompatible",
    )
    _require(
        "model_fits" in ridge_results and "adaptive_results" in ridge_results,
        "Task 3F-A results are incomplete",
    )

    logits_predictions = dataset.logits.argmax(axis=2).astype(np.int64)
    confidences = extract_features(dataset.logits, FEATURE_SET_CONFIDENCE)
    for artifact in (predictions, scores):
        validate_signal_alignment(
            expected_sample_indices=dataset.sample_indices,
            expected_inner_fold_ids=dataset.inner_fold_ids,
            expected_labels=dataset.labels,
            artifact_sample_indices=artifact.get("sample_indices"),
            artifact_inner_fold_ids=artifact.get("inner_fold_ids"),
            artifact_labels=artifact.get("labels"),
            expert_predictions=logits_predictions,
            confidences=confidences,
        )

    adaptive_ids = np.asarray(predictions.get("adaptive_configuration_ids")).astype(str)
    adaptive_weights = np.asarray(predictions.get("adaptive_weights"))
    adaptive_predictions = np.asarray(predictions.get("adaptive_predictions"))
    _require(
        adaptive_ids.ndim == 1 and adaptive_weights.ndim == 3
        and adaptive_predictions.ndim == 2
        and adaptive_weights.shape[1:] == (dataset.num_samples, len(EXPERT_ORDER))
        and adaptive_predictions.shape == (len(adaptive_ids), dataset.num_samples)
        and len(adaptive_ids) == 600,
        "Task 3F-A adaptive prediction arrays have incompatible shapes",
    )
    _require(
        len(np.unique(adaptive_ids)) == len(adaptive_ids),
        "Task 3F-A adaptive configuration IDs are not unique",
    )
    highlighted_index = _find_unique_index(
        adaptive_ids,
        HIGHLIGHTED_CONFIGURATION_ID,
        name="adaptive configuration IDs",
    )
    highlighted_weights = _validate_weights(
        adaptive_weights[highlighted_index], num_samples=dataset.num_samples
    )
    highlighted_predictions = np.asarray(
        adaptive_predictions[highlighted_index], dtype=np.int64
    )
    if highlighted_predictions.shape != (dataset.num_samples,):
        raise Task3FDCombinedSignalDiagnosticError(
            "saved highlighted predictions are not sample-aligned"
        )
    if np.any(highlighted_predictions < 0) or np.any(
        highlighted_predictions >= NUM_CLASSES
    ):
        raise Task3FDCombinedSignalDiagnosticError(
            "saved highlighted predictions contain invalid class IDs"
        )
    recomputed_predictions = combine_weighted_logits(
        dataset.logits,
        highlighted_weights / highlighted_weights.sum(axis=1, keepdims=True),
    ).argmax(axis=1)
    _require(
        np.array_equal(recomputed_predictions, highlighted_predictions),
        "saved highlighted predictions do not match saved Ridge weights",
    )
    return highlighted_weights, highlighted_predictions


def _validate_task3f_b_and_c_artifacts(
    *,
    mixup_results: Mapping[str, Any],
    tail_results: Mapping[str, Any],
    task3f_a_hashes: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(
        mixup_results.get("task_identifier") == TASK3FB_TASK_IDENTIFIER,
        "Task 3F-B result identifier is incompatible",
    )
    _require(
        mixup_results.get("provenance", {}).get("analyzed_sample_count")
        == ANALYSIS_SAMPLE_COUNT,
        "Task 3F-B population is not the permitted 6,507 rows",
    )
    _require(
        tail_results.get("task_identifier") == TASK3FC_TASK_IDENTIFIER,
        "Task 3F-C result identifier is incompatible",
    )
    _require(
        tail_results.get("highlighted_configuration_id")
        == HIGHLIGHTED_CONFIGURATION_ID,
        "Task 3F-C highlighted configuration is incompatible",
    )
    tail_provenance = tail_results.get("provenance", {})
    _require(
        tail_provenance.get("analyzed_sample_count") == ANALYSIS_SAMPLE_COUNT
        and tail_provenance.get("outer_fold") == OUTER_FOLD
        and tail_provenance.get("permitted_inner_folds")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and tail_provenance.get("reserved_inner_folds") == [0],
        "Task 3F-C population or fold provenance is incompatible",
    )
    for key in (
        "original_cifar_test_used",
        "reserved_outer_evaluation_used",
        "experts_retrained",
        "router_refit",
        "oracle_weights_used",
    ):
        _require(
            tail_provenance.get(key) is False,
            f"Task 3F-C restriction {key} failed",
        )
    recorded_a = tail_provenance.get("task3f_a_artifacts", {})
    for name, details in task3f_a_hashes.items():
        _require(
            recorded_a.get(name, {}).get("sha256") == details["sha256"],
            f"Task 3F-C provenance does not match Task 3F-A artifact {name}",
        )
    recorded_source = tail_provenance.get("task3c_oof_artifacts", {})
    for name, details in source_hashes.items():
        _require(
            recorded_source.get(name, {}).get("sha256") == details["sha256"],
            f"Task 3F-C provenance does not match source artifact {name}",
        )


def _mask_hash(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([len(mask)], dtype=np.int64).tobytes())
    digest.update(np.packbits(mask.astype(np.uint8), bitorder="little").tobytes())
    return digest.hexdigest()


def _format_count_fraction(count: int, fraction: float | None) -> str:
    return "n/a" if fraction is None else f"{count} ({fraction:.2%})"


def _format_value(value: Any, *, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _render_summary(results: Mapping[str, Any]) -> str:
    population = results["population"]
    tail = results["investigation_a"]
    opportunities = results["investigation_b"]["reports"]
    ridge = results["investigation_c"]
    comparisons = results["investigation_d"]["comparisons"]
    reports = tail["reports"]
    overall_ridge = ridge["overall"]
    compound_ids = [name for name in SIGNAL_COMBINATION_IDS if len(name) > 1]

    precision_improved = 0
    opportunity_rate_improved = 0
    for name in compound_ids:
        comparison = comparisons[name]
        if all(
            constituent["tail_precision_delta"] is not None
            and constituent["tail_precision_delta"] > 0.0
            for constituent in comparison["constituents"].values()
        ):
            precision_improved += 1
        if all(
            constituent["opportunity_fraction_delta"] is not None
            and constituent["opportunity_fraction_delta"] > 0.0
            for constituent in comparison["constituents"].values()
        ):
            opportunity_rate_improved += 1
    lower_mixup = sum(
        ridge["reports"][name]["mean_weight"]["Mixup"] is not None
        and ridge["reports"][name]["mean_weight"]["Mixup"]
        < overall_ridge["mean_weight"]["Mixup"]
        for name in SIGNAL_COMBINATION_IDS
    )

    lines = [
        "# Task 3F-D — Combined Tail-signal diagnostics",
        "",
        f"This is a retrospective diagnostic on {population['sample_count']:,} "
        "held-out OOF images from outer fold 0 / inner folds 1–3. Signals A–D "
        "were constructed from expert predictions, raw maximum-softmax "
        "confidence, and canonical class frequencies only. True labels and true "
        "Head/Medium/Tail groups were used afterward for evaluation; no router was "
        "refit.",
        "",
        "## Population",
        "",
        f"Head: {population['true_group_counts']['head']:,}; Medium: "
        f"{population['true_group_counts']['medium']:,}; Tail: "
        f"{population['true_group_counts']['tail']:,}; Tail prevalence: "
        f"{population['true_group_counts']['tail']} / {population['sample_count']} "
        f"({population['tail_prevalence']:.2%}).",
        "",
        "## All predefined masks",
        "",
        "Percentages retain their count in the same cell or adjacent count columns.",
        "",
        "| ID | Required | Selected n | Head / Medium / Tail n | False positives n | Tail precision | Tail recall | Rebalanced opportunity n / fraction | Mixup mean weight | Mixup highest n / fraction |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    definitions = dict(SIGNAL_COMBINATION_REQUIREMENTS)
    for name in SIGNAL_COMBINATION_IDS:
        report = reports[name]
        opp = opportunities[name]["overall"][
            "either_rebalanced_correct_Mixup_wrong"
        ]
        ridge_report = ridge["reports"][name]
        groups = report["selected_true_group_counts"]
        lines.append(
            f"| {name} | {'+'.join(definitions[name])} | {report['selected_count']} | "
            f"{groups['head']} / {groups['medium']} / {groups['tail']} | "
            f"{report['false_positive_count']} | "
            f"{_format_count_fraction(groups['tail'], report['tail_precision'])} | "
            f"{_format_count_fraction(groups['tail'], report['tail_recall'])} | "
            f"{_format_count_fraction(opp['count'], opp['fraction'])} | "
            f"{_format_value(ridge_report['mean_weight']['Mixup'])} | "
            f"{_format_count_fraction(ridge_report['mixup_highest_weight_count'], ridge_report['mixup_highest_weight_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"1. **Tail identification.** {precision_improved} of the {len(compound_ids)} "
            "predefined conjunctions have higher Tail precision than every one of "
            "their non-empty constituent masks on this population. Conjunctions are "
            "subsets by construction, so any precision increase is accompanied by "
            "lost selected images and no increase in Tail recall. The table reports "
            "the counts needed to interpret each percentage; it does not identify a "
            "winning combination.",
            f"2. **Useful rebalanced predictions.** The retrospective opportunity is "
            "the union of cases where LAL or BalancedSoftmax is correct while Mixup "
            f"is wrong. {opportunity_rate_improved} of the {len(compound_ids)} "
            "conjunctions have a higher opportunity fraction than every constituent "
            "on this development population. Opportunity counts and fractions are "
            "reported overall and by true group in diagnostic_results.json; these "
            "labels were not used to form any mask.",
            f"3. **Frozen Ridge response.** The overall saved Ridge mean Mixup weight "
            f"is {_format_value(overall_ridge['mean_weight']['Mixup'])} "
            f"(n={overall_ridge['selected_count']}); {lower_mixup} of the 15 masks "
            "have a lower selected-subgroup Mixup mean. This is a conditioned "
            "association, not an appropriate-response test or a new routing rule. "
            "Raw confidence scales are not assumed calibrated across experts.",
            "4. **Cost of conjunctions.** Requiring multiple conditions progressively "
            "shrinks the selected population. For example, ABCD is compared with A, "
            "B, C and D in the JSON comparison table, including its excluded-image "
            "counts, Tail precision/recall deltas and opportunity-rate deltas. Very "
            "small selected groups can produce unstable percentages, especially for "
            "the 183 actual Tail images.",
            "5. **Feature-representation implication.** Non-equivalent masks indicate "
            "that the frozen signals contain distinct combinatorial information, but "
            "this retrospective result does not justify selecting a new feature "
            "representation or fitting a new router. Any such proposal requires a "
            "new frozen protocol and independent evaluation.",
            "",
            "## Ridge reference",
            "",
            "| Population | n | CE mean | LAL mean | BalancedSoftmax mean | Mixup mean | Mixup highest n / fraction |",
            "|:--|--:|--:|--:|--:|--:|--:|",
            f"| Overall | {overall_ridge['selected_count']} | "
            f"{_format_value(overall_ridge['mean_weight']['CE'])} | "
            f"{_format_value(overall_ridge['mean_weight']['LAL'])} | "
            f"{_format_value(overall_ridge['mean_weight']['BalancedSoftmax'])} | "
            f"{_format_value(overall_ridge['mean_weight']['Mixup'])} | "
            f"{_format_count_fraction(overall_ridge['mixup_highest_weight_count'], overall_ridge['mixup_highest_weight_fraction'])} |",
            "",
            "## Limitations",
            "",
            "This is exploratory development-data evidence from one seed and one "
            "outer fold. The population has only 183 actual Tail images. Raw "
            "cross-expert confidence comparisons may be affected by calibration "
            "differences. Correctness outcomes and true-group composition are "
            "retrospective labels, and no combination was selected as a routing "
            "rule. The reserved inner fold 0, reserved outer-evaluation population, "
            "original CIFAR-100 test set, and full-data expert predictions were not "
            "used.",
            "",
        ]
    )
    return "\n".join(lines)


def run_task3f_combined_signal_diagnostics(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    ridge_directory: str | Path = "artifacts/oof/task3f_ridge",
    mixup_diagnostics_directory: str | Path = "artifacts/oof/task3f_mixup_diagnostics",
    tail_signal_diagnostics_directory: str | Path = "artifacts/oof/task3f_tail_signal_diagnostics",
    output_directory: str | Path = "artifacts/oof/task3f_combined_signal_diagnostics",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run Task 3F-D against existing artifacts and write separate outputs."""
    project_root_path = Path(
        project_root or Path(__file__).resolve().parents[1]
    ).resolve()
    source_directory = Path(oof_directory).resolve()
    ridge_path = Path(ridge_directory).resolve()
    mixup_path = Path(mixup_diagnostics_directory).resolve()
    tail_path = Path(tail_signal_diagnostics_directory).resolve()
    output_path = Path(output_directory).resolve()

    task3f_paths = {
        name: ridge_path / filename
        for name, filename in {
            "experiment_config": "experiment_config.json",
            "ridge_results": "ridge_results.json",
            "held_out_predictions": "held_out_predictions.npz",
            "router_scores": "router_scores.npz",
            "fold_assignments": "fold_assignments.json",
        }.items()
    }
    task3fb_paths = {
        "diagnostic_results": mixup_path / "diagnostic_results.json",
        "summary": mixup_path / "summary.md",
    }
    task3fc_paths = {
        "diagnostic_results": tail_path / "diagnostic_results.json",
        "summary": tail_path / "summary.md",
    }
    for group_name, paths in (
        ("Task 3F-A", task3f_paths),
        ("Task 3F-B", task3fb_paths),
        ("Task 3F-C", task3fc_paths),
    ):
        for name, path in paths.items():
            if not path.exists():
                raise Task3FDCombinedSignalDiagnosticError(
                    f"missing {group_name} artifact {name}: {path}"
                )
    if output_path == source_directory or output_path == ridge_path:
        raise Task3FDCombinedSignalDiagnosticError(
            "output directory must be separate from existing input artifacts"
        )

    task3f_hashes_before = _artifact_hashes(task3f_paths)
    task3fb_hashes_before = _artifact_hashes(task3fb_paths)
    task3fc_hashes_before = _artifact_hashes(task3fc_paths)
    config = _load_json(task3f_paths["experiment_config"], name="Task 3F-A config")
    ridge_results = _load_json(task3f_paths["ridge_results"], name="Task 3F-A results")
    fold_assignments = _load_json(
        task3f_paths["fold_assignments"], name="Task 3F-A fold assignments"
    )
    predictions = _load_npz(
        task3f_paths["held_out_predictions"], name="Task 3F-A predictions"
    )
    scores = _load_npz(task3f_paths["router_scores"], name="Task 3F-A scores")
    mixup_results = _load_json(
        task3fb_paths["diagnostic_results"], name="Task 3F-B results"
    )
    tail_results = _load_json(
        task3fc_paths["diagnostic_results"], name="Task 3F-C results"
    )

    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=FOLD_GENERATION_SEED,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    _require(
        dataset.num_samples == ANALYSIS_SAMPLE_COUNT,
        "validated OOF population is not the permitted 6,507 rows",
    )
    _require(
        np.all(dataset.outer_fold_ids == OUTER_FOLD)
        and set(np.unique(dataset.inner_fold_ids))
        == set(PERMITTED_ANALYSIS_INNER_FOLDS),
        "validated OOF population contains a reserved or incompatible fold",
    )
    class_counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    _require(len(class_counts) == NUM_CLASSES, "canonical class counts are incompatible")
    source_hashes = {
        name: {"path": str(details["path"]), "sha256": str(details["sha256"])}
        for name, details in source_files.items()
    }
    _validate_task3f_b_and_c_artifacts(
        mixup_results=mixup_results,
        tail_results=tail_results,
        task3f_a_hashes=task3f_hashes_before,
        source_hashes=source_hashes,
    )
    highlighted_weights, highlighted_predictions = _validate_task3f_a_artifacts(
        dataset=dataset,
        config=config,
        ridge_results=ridge_results,
        predictions=predictions,
        scores=scores,
        fold_assignments=fold_assignments,
    )
    _require(
        np.array_equal(
            dataset.sample_indices,
            np.asarray(predictions["sample_indices"], dtype=np.int64),
        ),
        "saved Task 3F-A sample IDs changed during diagnostic setup",
    )

    # This is intentionally the only construction seam for candidate signals.
    # It receives no labels or true group membership.
    expert_predictions = dataset.logits.argmax(axis=2).astype(np.int64)
    confidences = extract_features(dataset.logits, FEATURE_SET_CONFIDENCE)
    individual_signals = build_inference_time_signals(
        expert_predictions, confidences, class_counts
    )
    signal_masks = {
        **individual_signals,
        **build_signal_combinations(individual_signals),
    }

    # Retrospective label-dependent calculations begin only after every mask is
    # constructed and frozen.
    tail_report = tail_detection_diagnostics(
        signal_masks, dataset.labels, class_counts
    )
    correctness_report = correctness_opportunity_diagnostics(
        signal_masks, expert_predictions, dataset.labels, class_counts
    )
    ridge_report = ridge_weight_diagnostics(signal_masks, highlighted_weights)
    comparison_report = compare_combined_signals(
        signal_masks, tail_report, correctness_report
    )
    equivalent_groups = detect_equivalent_signal_masks(
        {name: signal_masks[name] for name in SIGNAL_COMBINATION_IDS}
    )

    current_task3f_hashes = _artifact_hashes(task3f_paths)
    current_task3fb_hashes = _artifact_hashes(task3fb_paths)
    current_task3fc_hashes = _artifact_hashes(task3fc_paths)
    _require(
        current_task3f_hashes == task3f_hashes_before,
        "a Task 3F-A artifact changed during diagnostics",
    )
    _require(
        current_task3fb_hashes == task3fb_hashes_before,
        "a Task 3F-B artifact changed during diagnostics",
    )
    _require(
        current_task3fc_hashes == task3fc_hashes_before,
        "a Task 3F-C artifact changed during diagnostics",
    )
    current_source_hashes = _artifact_hashes(
        {name: Path(details["path"]) for name, details in source_hashes.items()}
    )
    _require(
        current_source_hashes == source_hashes,
        "a Task 3C source artifact changed during diagnostics",
    )

    true_groups = assign_class_groups(dataset.labels, class_counts)
    mask_records = {
        name: {
            "selected_count": int(mask.sum()),
            "sha256": _mask_hash(mask),
        }
        for name, mask in signal_masks.items()
    }
    results: dict[str, Any] = {
        "task_identifier": TASK3FD_TASK_IDENTIFIER,
        "schema_version": TASK3FD_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root_path),
        "highlighted_configuration_id": HIGHLIGHTED_CONFIGURATION_ID,
        "provenance": {
            "dataset": DATASET_NAME,
            "training_seed": TRAINING_SEED,
            "fold_generation_seed": FOLD_GENERATION_SEED,
            "outer_fold": OUTER_FOLD,
            "permitted_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
            "reserved_inner_folds": [0],
            "analyzed_sample_count": dataset.num_samples,
            "expert_order": list(EXPERT_ORDER),
            "inner_fold_zero_used": False,
            "reserved_outer_evaluation_used": False,
            "original_cifar_test_used": False,
            "original_full_data_predictions_used": False,
            "experts_retrained": False,
            "router_refit": False,
            "expert_weights_modified": False,
            "source_oof_logits_modified": False,
            "signal_construction_requires_true_labels": False,
            "labels_used_only_after_signal_construction": True,
            "task3c_oof_artifacts": source_hashes,
            "task3f_a_artifacts": task3f_hashes_before,
            "task3f_b_artifacts": task3fb_hashes_before,
            "task3f_c_artifacts": task3fc_hashes_before,
            "task3f_a_config_input_artifacts": config.get("input_artifacts", {}),
            "canonical_training_index_sha256": manager.canonical_training_index_sha256,
        },
        "population": {
            "sample_count": dataset.num_samples,
            "true_group_counts": {
                group_name: int(np.sum(true_groups == group_name))
                for group_name in GROUP_NAMES
            },
            "tail_prevalence": _fraction(
                int(np.sum(true_groups == "tail")), dataset.num_samples
            ),
        },
        "signal_contract": {
            "expert_order": list(EXPERT_ORDER),
            "confidence_definition": (
                "maximum softmax probability from each expert's original OOF logits"
            ),
            "raw_confidence_not_assumed_calibrated_across_experts": True,
            "canonical_group_source": "scripts.base_trainer.compute_class_groups",
            "signals": {
                "A": "pred_LAL == pred_BalancedSoftmax",
                "B": "pred_Mixup differs from both pred_LAL and pred_BalancedSoftmax",
                "C": "pred_LAL and pred_BalancedSoftmax both map to canonical Tail",
                "D": "confidence_LAL > confidence_Mixup and confidence_BalancedSoftmax > confidence_Mixup",
            },
            "true_label_inputs": [],
            "oracle_inputs": [],
        },
        "signal_combinations": [
            {"id": name, "required_signals": list(required)}
            for name, required in SIGNAL_COMBINATION_REQUIREMENTS
        ],
        "signal_masks": mask_records,
        "equivalent_signal_mask_groups": [
            group for group in equivalent_groups if len(group) > 1
        ],
        "distinct_signal_mask_count": len(equivalent_groups),
        "investigation_a": tail_report,
        "investigation_b": correctness_report,
        "investigation_c": ridge_report,
        "investigation_d": {
            **comparison_report,
            "all_signal_mask_groups": equivalent_groups,
        },
        "saved_highlighted_predictions_match_saved_weights": True,
        "saved_highlighted_prediction_count": int(len(highlighted_predictions)),
    }
    summary = _render_summary(results)
    result_path = output_path / "diagnostic_results.json"
    summary_path = output_path / "summary.md"
    _write_json_once(result_path, results)
    _write_text_once(summary_path, summary)
    return {"diagnostic_results": result_path, "summary": summary_path}


__all__ = [
    "ANALYSIS_SAMPLE_COUNT",
    "HIGHLIGHTED_CONFIGURATION_ID",
    "SIGNAL_COMBINATION_IDS",
    "SIGNAL_COMBINATION_REQUIREMENTS",
    "SIGNAL_NAMES",
    "Task3FDCombinedSignalDiagnosticError",
    "Task3FDDiagnosticError",
    "build_all_signal_masks",
    "build_inference_time_signals",
    "build_signal_combinations",
    "build_signals",
    "compare_combined_signals",
    "construct_inference_signals",
    "correctness_opportunity_diagnostics",
    "detect_equivalent_signal_masks",
    "find_equivalent_signal_masks",
    "ridge_weight_diagnostics",
    "run_task3f_combined_signal_diagnostics",
    "tail_detection_diagnostics",
]
