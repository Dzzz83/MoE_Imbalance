"""Task 3E-B: adaptive soft-mixture oracle feasibility analysis.

This module is deliberately an analysis-only layer over the validated Task
3C OOF artifact.  It solves one label-dependent maximum-margin linear
program per image.  The resulting weights are an oracle diagnostic: they are
not an inference-time router and are never exposed through the training or
production evaluation pipeline.

Only :class:`~scripts.task3e_fixed.RestrictedAnalysisDataset` is accepted by
the analyzer.  Consequently, the complete aligned OOF artifact cannot be
accidentally passed to a Task 3E-B metric calculation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linprog

from data.nested_oof import NestedOOFFoldManager
from scripts.analysis.artifacts import (
    git_commit,
    load_json_object,
    repository_relative,
    serialize_json,
    sha256_array,
    write_texts_once,
)
from scripts.base_trainer import compute_class_groups
from scripts.expert_diagnostics import ExpertDiagnostics
from scripts.task3e_fixed import (
    ANALYSIS_SAMPLE_COUNT,
    CANONICAL_POPULATION_SIZE,
    DATASET_NAME,
    EXPERT_ORDER,
    FOLD_GENERATION_SEED,
    FULL_ALIGNED_SAMPLE_COUNT,
    IMBALANCE_RATIO,
    NUM_CLASSES,
    OUTER_EVALUATION_SIZE,
    OUTER_FOLD,
    PERMITTED_ANALYSIS_INNER_FOLDS,
    RESERVED_ROUTER_SELECTION_INNER_FOLDS,
    RestrictedAnalysisDataset,
    Task3EError,
    TRAINING_SEED,
    _require_equal,
    _sha256_file,
    generate_fixed_weight_candidates,
    load_restricted_analysis_dataset,
    verify_uniform_baseline,
)


TASK3E_SOFT_TASK_IDENTIFIER = "Task 3E-B"
TASK3E_SOFT_SCHEMA_VERSION = "task3e_soft_feasibility.v1"
TASK3E_SOFT_RESULTS_SCHEMA_VERSION = "task3e_soft_results.v1"

MARGIN_TOLERANCE = 1e-6
SOLVER_FEASIBILITY_TOLERANCE = 1e-7
MARGIN_VERIFICATION_TOLERANCE = 1e-7
SOLVER_METHOD = "highs"
SOLVER_OPTIONS = {
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
    "ipm_optimality_tolerance": 1e-9,
    "presolve": True,
}
FIXED_REFERENCE_TOLERANCE = 1e-12

REQUIRED_FIXED_REFERENCE_IDS = (
    "fixed_020",  # uniform four-expert candidate
    "fixed_006",
    "fixed_007",
    "fixed_010",
    "fixed_011",
)
HISTORICAL_THREE_EXPERT_WEIGHTS = (0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class SoftOracleInputError(Task3EError):
    """Raised when one image cannot be represented by the frozen LP."""


class SoftOracleVerificationError(Task3EError):
    """Raised by callers that require a verified LP solution."""


def _as_metric_bundle(metrics: Mapping[str, Any]) -> dict[str, float]:
    required = ("accuracy", "ba", "head", "medium", "tail")
    if any(key not in metrics or metrics[key] is None for key in required):
        raise Task3EError("the canonical metric implementation did not produce all metrics")
    return {
        "ordinary_accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["ba"]),
        "head_accuracy": float(metrics["head"]),
        "medium_accuracy": float(metrics["medium"]),
        "tail_accuracy": float(metrics["tail"]),
    }


def _hash_indices(indices: Sequence[int]) -> str:
    values = np.asarray(sorted(int(value) for value in indices), dtype="<i8")
    return sha256_array(values)


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    return load_json_object(path, name=name, error_type=Task3EError)


def _repository_relative(path: Path, project_root: Path) -> str:
    return repository_relative(path, project_root)


def _git_commit(project_root: Path) -> str:
    return git_commit(project_root, error_type=Task3EError)


@dataclass(frozen=True)
class SoftOracleResult:
    """One independently solved and numerically audited image LP."""

    status: str
    optimal_margin: float | None
    reconstructed_margin: float | None
    oracle_weights: tuple[float, ...] | None
    solver_status: int | None
    solver_message: str
    verification_passed: bool
    verification_errors: tuple[str, ...]
    constraint_count: int
    reported_margin: float | None = None
    objective_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "soft_oracle_status": self.status,
            "optimal_margin": self.optimal_margin,
            "reconstructed_margin": self.reconstructed_margin,
            "reported_margin": self.reported_margin,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "verification_passed": self.verification_passed,
            "verification_errors": list(self.verification_errors),
            "constraint_count": self.constraint_count,
            "objective_value": self.objective_value,
            "oracle_weights": (
                list(self.oracle_weights) if self.oracle_weights is not None else None
            ),
            "oracle_weights_are_label_dependent": True,
        }


class SoftMixtureOracle:
    """Solve and verify the frozen maximum-margin convex-logit LP."""

    def __init__(
        self,
        *,
        num_experts: int = len(EXPERT_ORDER),
        num_classes: int = NUM_CLASSES,
        solver: Callable[..., Any] = linprog,
        method: str = SOLVER_METHOD,
        solver_options: Mapping[str, Any] | None = None,
        margin_tolerance: float = MARGIN_TOLERANCE,
        feasibility_tolerance: float = SOLVER_FEASIBILITY_TOLERANCE,
        verification_tolerance: float = MARGIN_VERIFICATION_TOLERANCE,
    ) -> None:
        if num_experts != len(EXPERT_ORDER):
            raise SoftOracleInputError("Task 3E-B requires exactly four experts")
        if num_classes < 2:
            raise SoftOracleInputError("the LP requires at least two classes")
        if margin_tolerance <= 0 or feasibility_tolerance <= 0 or verification_tolerance <= 0:
            raise SoftOracleInputError("oracle tolerances must be positive")
        if method != SOLVER_METHOD:
            raise SoftOracleInputError("Task 3E-B requires the HiGHS linprog method")
        self.num_experts = int(num_experts)
        self.num_classes = int(num_classes)
        self.solver = solver
        self.method = method
        self.solver_options = dict(solver_options or SOLVER_OPTIONS)
        self.margin_tolerance = float(margin_tolerance)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.verification_tolerance = float(verification_tolerance)

    def _validate_logits_and_label(
        self, logits: np.ndarray, true_label: int
    ) -> tuple[np.ndarray, int]:
        array = np.asarray(logits)
        expected_shape = (self.num_experts, self.num_classes)
        if array.shape != expected_shape:
            raise SoftOracleInputError(
                f"one-image logits must have shape {expected_shape}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise SoftOracleInputError("one-image logits must be real numeric values")
        if not np.isfinite(array).all():
            raise SoftOracleInputError("one-image logits contain NaN or infinity")
        label_array = np.asarray(true_label)
        if label_array.ndim != 0 or not np.issubdtype(label_array.dtype, np.number):
            raise SoftOracleInputError("true_label must be one integer class index")
        if np.iscomplexobj(label_array) or not np.isfinite(label_array):
            raise SoftOracleInputError("true_label must be finite")
        if float(label_array) != float(np.floor(float(label_array))):
            raise SoftOracleInputError("true_label must be an integer class index")
        label = int(label_array)
        if not 0 <= label < self.num_classes:
            raise SoftOracleInputError(
                f"true_label must lie in [0, {self.num_classes}), got {label}"
            )
        # This is a numerical calculation copy, not a rescaling or a change to
        # the source aligned logits.
        return array.astype(np.float64, copy=False), label

    def build_problem(self, logits: np.ndarray, true_label: int) -> dict[str, Any]:
        """Construct the exact LP matrices for one image."""
        logits_array, label = self._validate_logits_and_label(logits, true_label)
        competitors = np.flatnonzero(np.arange(self.num_classes) != label).astype(np.int64)
        differences = (
            logits_array[:, label][:, None] - logits_array[:, competitors]
        ).T
        objective = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        a_ub = np.hstack(
            (-differences, np.ones((len(competitors), 1), dtype=np.float64))
        )
        b_ub = np.zeros(len(competitors), dtype=np.float64)
        a_eq = np.asarray([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
        b_eq = np.asarray([1.0], dtype=np.float64)
        bounds = [(0.0, 1.0)] * self.num_experts + [(None, None)]
        return {
            "objective": objective,
            "A_ub": a_ub,
            "b_ub": b_ub,
            "A_eq": a_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "difference_matrix": differences,
            "competitor_classes": competitors,
            "logits": logits_array,
            "true_label": label,
        }

    def _failure(
        self,
        *,
        solver_status: int | None,
        solver_message: str,
        errors: Sequence[str],
        constraint_count: int,
        reported_margin: float | None = None,
        objective_value: float | None = None,
        reconstructed_margin: float | None = None,
        weights: Sequence[float] | None = None,
    ) -> SoftOracleResult:
        return SoftOracleResult(
            status="solver_failure",
            optimal_margin=None,
            reconstructed_margin=(
                None
                if reconstructed_margin is None
                else float(reconstructed_margin)
            ),
            oracle_weights=(
                None if weights is None else tuple(float(value) for value in weights)
            ),
            solver_status=solver_status,
            solver_message=str(solver_message),
            verification_passed=False,
            verification_errors=tuple(str(error) for error in errors),
            constraint_count=int(constraint_count),
            reported_margin=reported_margin,
            objective_value=objective_value,
        )

    def solve(self, logits: np.ndarray, true_label: int) -> SoftOracleResult:
        """Solve one image LP, then independently verify the returned point."""
        problem = self.build_problem(logits, true_label)
        constraint_count = len(problem["competitor_classes"])
        try:
            result = self.solver(
                problem["objective"],
                A_ub=problem["A_ub"],
                b_ub=problem["b_ub"],
                A_eq=problem["A_eq"],
                b_eq=problem["b_eq"],
                bounds=problem["bounds"],
                method=self.method,
                options=self.solver_options,
            )
        except Exception as exc:  # pragma: no cover - exact SciPy exception varies
            return self._failure(
                solver_status=None,
                solver_message=f"solver raised {type(exc).__name__}: {exc}",
                errors=("solver_call_exception",),
                constraint_count=constraint_count,
            )

        raw_status = getattr(result, "status", None)
        solver_status: int | None
        try:
            solver_status = None if raw_status is None else int(raw_status)
        except (TypeError, ValueError, OverflowError):
            solver_status = None
        solver_message = str(getattr(result, "message", ""))
        if not bool(getattr(result, "success", False)):
            return self._failure(
                solver_status=solver_status,
                solver_message=solver_message,
                errors=("solver_unsuccessful",),
                constraint_count=constraint_count,
                objective_value=self._finite_float(getattr(result, "fun", None)),
            )

        errors: list[str] = []
        raw_x = np.asarray(getattr(result, "x", np.asarray([])))
        if raw_x.shape != (self.num_experts + 1,):
            return self._failure(
                solver_status=solver_status,
                solver_message=solver_message,
                errors=(f"solution_shape_{raw_x.shape}",),
                constraint_count=constraint_count,
                objective_value=self._finite_float(getattr(result, "fun", None)),
            )
        if not np.issubdtype(raw_x.dtype, np.number) or np.iscomplexobj(raw_x):
            return self._failure(
                solver_status=solver_status,
                solver_message=solver_message,
                errors=("solution_not_real_numeric",),
                constraint_count=constraint_count,
            )
        if not np.isfinite(raw_x).all():
            return self._failure(
                solver_status=solver_status,
                solver_message=solver_message,
                errors=("solution_contains_nonfinite_values",),
                constraint_count=constraint_count,
            )

        x = raw_x.astype(np.float64, copy=False)
        weights = x[: self.num_experts]
        reported_margin = float(x[-1])
        objective_value = self._finite_float(getattr(result, "fun", None))
        reconstructed_margin: float | None = None

        if np.any(weights < -self.feasibility_tolerance):
            errors.append("negative_weight_constraint_violation")
        if np.any(weights > 1.0 + self.feasibility_tolerance):
            errors.append("upper_weight_constraint_violation")
        if abs(float(weights.sum()) - 1.0) > self.feasibility_tolerance:
            errors.append("weight_sum_constraint_violation")
        if objective_value is None:
            errors.append("solver_objective_missing_or_nonfinite")
        elif abs(-objective_value - reported_margin) > self.verification_tolerance:
            errors.append("reported_margin_disagrees_with_objective")

        mixed_logits = weights @ problem["logits"]
        competitors = problem["competitor_classes"]
        reconstructed_margin = float(
            np.min(mixed_logits[problem["true_label"]] - mixed_logits[competitors])
        )
        if abs(reconstructed_margin - reported_margin) > self.verification_tolerance:
            errors.append("reported_margin_disagrees_with_reconstruction")

        primal_residuals = problem["difference_matrix"] @ weights - reported_margin
        if np.any(primal_residuals < -self.feasibility_tolerance):
            errors.append("primal_margin_constraint_violation")

        if errors:
            return self._failure(
                solver_status=solver_status,
                solver_message=solver_message,
                errors=errors,
                constraint_count=constraint_count,
                reported_margin=reported_margin,
                objective_value=objective_value,
                reconstructed_margin=reconstructed_margin,
                weights=weights,
            )

        if reconstructed_margin > self.margin_tolerance:
            status = "correctable"
        elif reconstructed_margin < -self.margin_tolerance:
            status = "not_strictly_correctable"
        else:
            status = "numerically_ambiguous"
        return SoftOracleResult(
            status=status,
            optimal_margin=reported_margin,
            reconstructed_margin=reconstructed_margin,
            oracle_weights=tuple(float(value) for value in weights),
            solver_status=solver_status,
            solver_message=solver_message,
            verification_passed=True,
            verification_errors=(),
            constraint_count=constraint_count,
            reported_margin=reported_margin,
            objective_value=objective_value,
        )

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if np.isfinite(result) else None

    solve_one = solve


def _coverage_metrics(
    labels: np.ndarray,
    event_mask: np.ndarray,
    class_groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute ordinary and macro class coverage for a boolean event."""
    labels = np.asarray(labels, dtype=np.int64)
    event_mask = np.asarray(event_mask, dtype=bool)
    if labels.ndim != 1 or event_mask.shape != labels.shape:
        raise Task3EError("coverage labels and event mask are not aligned")
    classes = np.unique(labels)
    per_class: dict[str, float] = {}
    denominators: dict[str, int] = {}
    for class_index in classes.tolist():
        mask = labels == class_index
        denominators[str(int(class_index))] = int(mask.sum())
        per_class[str(int(class_index))] = float(event_mask[mask].mean())

    def group_coverage(group_classes: np.ndarray) -> float | None:
        values = [
            per_class[str(int(class_index))]
            for class_index in group_classes.tolist()
            if str(int(class_index)) in per_class
        ]
        return float(np.mean(values)) if values else None

    return {
        "sample_count": int(len(labels)),
        "event_count": int(event_mask.sum()),
        "ordinary_coverage": float(event_mask.mean()),
        "balanced_accuracy": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "head_coverage": group_coverage(class_groups["head"]),
        "medium_coverage": group_coverage(class_groups["medium"]),
        "tail_coverage": group_coverage(class_groups["tail"]),
        "per_class_coverage": per_class,
        "per_class_sample_count": denominators,
        "present_class_count": len(per_class),
        "coverage_interpretation": (
            "macro class coverage of a feasibility/correctness event; it is not "
            "a predicted-class accuracy unless the event is an actual prediction"
        ),
    }


def _group_event_counts(
    labels: np.ndarray,
    event_mask: np.ndarray,
    class_groups: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    event_mask = np.asarray(event_mask, dtype=bool)
    output: dict[str, dict[str, Any]] = {}
    overall_count = int(event_mask.sum())
    output["all"] = {
        "count": overall_count,
        "fraction": float(overall_count / len(labels)),
        "denominator": int(len(labels)),
    }
    for name, classes in class_groups.items():
        group_mask = np.isin(labels, classes)
        count = int((event_mask & group_mask).sum())
        denominator = int(group_mask.sum())
        class_values = []
        for class_index in classes.tolist():
            class_mask = labels == class_index
            if class_mask.any():
                class_values.append(float(event_mask[class_mask].mean()))
        output[name] = {
            "count": count,
            "fraction": float(count / denominator) if denominator else 0.0,
            "denominator": denominator,
            "macro_class_fraction": (
                float(np.mean(class_values)) if class_values else None
            ),
            "classes": [int(value) for value in classes.tolist()],
        }
    return output


def _class_status_distribution(
    labels: np.ndarray, statuses: Sequence[str], class_count: int
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    status_array = np.asarray(statuses, dtype=object)
    output: dict[str, Any] = {}
    for status in (
        "correctable",
        "not_strictly_correctable",
        "numerically_ambiguous",
        "solver_failure",
    ):
        counts = [
            int(np.sum((labels == class_index) & (status_array == status)))
            for class_index in range(class_count)
        ]
        denominators = [int(np.sum(labels == class_index)) for class_index in range(class_count)]
        output[status] = {
            "count": int(sum(counts)),
            "fraction": float(sum(counts) / len(labels)),
            "counts_by_class": counts,
            "fractions_by_class": [
                float(count / denominator) if denominator else None
                for count, denominator in zip(counts, denominators)
            ],
        }
    return output


def _tail_class_report(
    labels: np.ndarray,
    uniform_correct: np.ndarray,
    any_expert_correct: np.ndarray,
    soft_correctable: np.ndarray,
    statuses: Sequence[str],
    tail_classes: np.ndarray,
) -> list[dict[str, Any]]:
    status_array = np.asarray(statuses, dtype=object)
    rows: list[dict[str, Any]] = []
    for class_index in tail_classes.tolist():
        mask = labels == class_index
        sample_count = int(mask.sum())
        rows.append(
            {
                "class_index": int(class_index),
                "sample_count": sample_count,
                "uniform_correct": int((uniform_correct & mask).sum()),
                "any_expert_correct": int((any_expert_correct & mask).sum()),
                "soft_correctable": int((soft_correctable & mask).sum()),
                "newly_soft_correctable": int(
                    ((~any_expert_correct) & soft_correctable & mask).sum()
                ),
                "uncorrectable": int(
                    (mask & (status_array == "not_strictly_correctable")).sum()
                ),
                "numerically_ambiguous": int(
                    (mask & (status_array == "numerically_ambiguous")).sum()
                ),
                "solver_failure": int(
                    (mask & (status_array == "solver_failure")).sum()
                ),
                "unresolved": int(
                    (mask
                     & np.isin(status_array, ["numerically_ambiguous", "solver_failure"])).sum()
                ),
                "uniform_recall": (
                    float((uniform_correct & mask).sum() / sample_count)
                    if sample_count
                    else None
                ),
                "hard_selection_recall": (
                    float((any_expert_correct & mask).sum() / sample_count)
                    if sample_count
                    else None
                ),
                "soft_feasibility_recall": (
                    float((soft_correctable & mask).sum() / sample_count)
                    if sample_count
                    else None
                ),
            }
        )
    return rows


def _fixed_artifact_input_records(
    *,
    fixed_directory: Path,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    records = {}
    for name, filename in (
        ("task3e_fixed_config", "experiment_config.json"),
        ("task3e_fixed_results", "fixed_weight_results.json"),
        ("task3e_fixed_summary", "summary.md"),
    ):
        path = fixed_directory / filename
        if not path.is_file():
            raise Task3EError(f"missing Task 3E-A artifact: {path}")
        records[name] = {
            "path": _repository_relative(path, project_root),
            "sha256": _sha256_file(path),
        }
    return records


def _validate_task3c_reference(
    reference_path: Path,
    *,
    expected_diagnostics_hash: str | None = None,
) -> dict[str, Any]:
    if not reference_path.is_file():
        raise Task3EError(f"missing Task 3C diagnostics artifact: {reference_path}")
    actual_hash = _sha256_file(reference_path)
    if expected_diagnostics_hash is not None and actual_hash != expected_diagnostics_hash:
        raise Task3EError("Task 3C diagnostics hash does not match the Task 3E-A provenance")
    payload = _json_object(reference_path, name="Task 3C diagnostics")
    _require_equal(payload, "schema_version", "task3c_diagnostics.v1", name="diagnostics")
    _require_equal(payload, "dataset", DATASET_NAME, name="diagnostics")
    _require_equal(payload, "imbalance_ratio", IMBALANCE_RATIO, name="diagnostics")
    _require_equal(payload, "training_seed", TRAINING_SEED, name="diagnostics")
    _require_equal(payload, "fold_generation_seed", FOLD_GENERATION_SEED, name="diagnostics")
    _require_equal(payload, "outer_fold_id", OUTER_FOLD, name="diagnostics")
    _require_equal(payload, "expert_order", list(EXPERT_ORDER), name="diagnostics")
    primary = payload.get("primary_router_fit_partition")
    if not isinstance(primary, Mapping):
        raise Task3EError("Task 3C diagnostics lack the primary router-fit partition")
    _require_equal(primary, "sample_count", ANALYSIS_SAMPLE_COUNT, name="diagnostics.primary_router_fit_partition")
    _require_equal(primary, "inner_fold_ids", list(PERMITTED_ANALYSIS_INNER_FOLDS), name="diagnostics.primary_router_fit_partition")
    uniform = primary.get("uniform_logit_ensemble")
    if not isinstance(uniform, Mapping) or uniform.get("combination") != "uniform logit average":
        raise Task3EError("Task 3C diagnostics lack the stored uniform logit baseline")
    uniform_metrics = uniform.get("metrics")
    hard = primary.get("hard_routing_oracle_headroom")
    if not isinstance(uniform_metrics, Mapping) or not isinstance(hard, Mapping):
        raise Task3EError("Task 3C primary reference metrics are malformed")
    if hard.get("n_samples") != ANALYSIS_SAMPLE_COUNT:
        raise Task3EError("Task 3C hard-selection reference has the wrong sample count")
    return {
        "path": reference_path,
        "sha256": actual_hash,
        "payload": payload,
        "uniform_metrics": dict(uniform_metrics),
        "hard_metrics": dict(hard.get("hard_selection_oracle_metrics", {})),
        "hard_headroom": dict(hard),
    }


def validate_fixed_reference_artifacts(
    fixed_results_directory: str | Path,
    *,
    project_root: str | Path,
    current_source_files: Mapping[str, Mapping[str, Any]],
    reference_diagnostics: str | Path,
    manager: NestedOOFFoldManager,
) -> dict[str, Any]:
    """Validate E-A configuration/results before using reference rows."""
    fixed_directory = Path(fixed_results_directory).resolve()
    project_root_path = Path(project_root).resolve()
    config_path = fixed_directory / "experiment_config.json"
    results_path = fixed_directory / "fixed_weight_results.json"
    summary_path = fixed_directory / "summary.md"
    config = _json_object(config_path, name="Task 3E-A experiment configuration")
    results = _json_object(results_path, name="Task 3E-A fixed-weight results")
    _require_equal(config, "task_identifier", "Task 3E-A", name="Task 3E-A config")
    _require_equal(config, "schema_version", "task3e_fixed_feasibility.v1", name="Task 3E-A config")
    _require_equal(config, "dataset", DATASET_NAME, name="Task 3E-A config")
    _require_equal(config, "imbalance_ratio", IMBALANCE_RATIO, name="Task 3E-A config")
    _require_equal(config, "training_seed", TRAINING_SEED, name="Task 3E-A config")
    _require_equal(config, "fold_generation_seed", FOLD_GENERATION_SEED, name="Task 3E-A config")
    _require_equal(config, "outer_fold", OUTER_FOLD, name="Task 3E-A config")
    _require_equal(config, "analyzed_sample_count", ANALYSIS_SAMPLE_COUNT, name="Task 3E-A config")
    _require_equal(config, "expert_order", list(EXPERT_ORDER), name="Task 3E-A config")
    _require_equal(config, "canonical_training_index_sha256", manager.canonical_training_index_sha256, name="Task 3E-A config")
    if config.get("permitted_analysis_inner_folds") != list(PERMITTED_ANALYSIS_INNER_FOLDS):
        raise Task3EError("Task 3E-A config has incompatible analysis folds")

    input_artifacts = config.get("input_artifacts")
    if not isinstance(input_artifacts, Mapping):
        raise Task3EError("Task 3E-A config lacks input artifact provenance")
    for name, details in current_source_files.items():
        entry = input_artifacts.get(name)
        if not isinstance(entry, Mapping) or entry.get("sha256") != details.get("sha256"):
            raise Task3EError(f"Task 3E-A input hash mismatch for {name}")
        if _sha256_file(Path(details["path"])) != str(entry["sha256"]):
            raise Task3EError(f"current input hash is invalid for {name}")
    diagnostics_reference = Path(reference_diagnostics).resolve()
    diagnostic_entry = input_artifacts.get("task3c_diagnostics")
    if not isinstance(diagnostic_entry, Mapping):
        raise Task3EError("Task 3E-A config lacks Task 3C diagnostics provenance")
    diagnostics_reference_data = _validate_task3c_reference(
        diagnostics_reference,
        expected_diagnostics_hash=str(diagnostic_entry.get("sha256")),
    )

    expected_fixed_inputs = _fixed_artifact_input_records(
        fixed_directory=fixed_directory,
        project_root=project_root_path,
    )
    _require_equal(results, "schema_version", "task3e_fixed_results.v1", name="Task 3E-A results")
    _require_equal(results, "candidate_count", 35, name="Task 3E-A results")
    partition = results.get("analysis_partition")
    if not isinstance(partition, Mapping):
        raise Task3EError("Task 3E-A results lack analysis partition provenance")
    _require_equal(partition, "outer_fold", OUTER_FOLD, name="Task 3E-A results.analysis_partition")
    _require_equal(partition, "inner_fold_ids", list(PERMITTED_ANALYSIS_INNER_FOLDS), name="Task 3E-A results.analysis_partition")
    _require_equal(partition, "sample_count", ANALYSIS_SAMPLE_COUNT, name="Task 3E-A results.analysis_partition")
    if partition.get("outer_evaluation_excluded") is not True or partition.get("router_selection_inner_fold_excluded") is not True:
        raise Task3EError("Task 3E-A results do not exclude reserved populations")

    expected_candidates = {candidate.candidate_id: candidate for candidate in generate_fixed_weight_candidates()}
    rows = results.get("candidates")
    if not isinstance(rows, list) or len(rows) != 35:
        raise Task3EError("Task 3E-A candidate rows are incomplete")
    actual_ids = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise Task3EError("Task 3E-A candidate row is malformed")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise Task3EError("Task 3E-A candidate identifier is malformed")
        actual_ids.append(candidate_id)
        expected = expected_candidates.get(candidate_id)
        if expected is None:
            raise Task3EError(f"unknown Task 3E-A candidate: {candidate_id!r}")
        try:
            actual_units = tuple(int(value) for value in row.get("weight_units", ()))
            actual_weights = np.asarray(row.get("weight_vector"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Task3EError(f"Task 3E-A weights are malformed for {candidate_id}") from exc
        if actual_units != expected.units:
            raise Task3EError(f"Task 3E-A units disagree for {candidate_id}")
        if actual_weights.shape != (len(EXPERT_ORDER),) or not np.isfinite(actual_weights).all():
            raise Task3EError(f"Task 3E-A weights are non-finite for {candidate_id}")
        if not np.array_equal(actual_weights, np.asarray(expected.weights, dtype=np.float64)):
            raise Task3EError(f"Task 3E-A weights disagree for {candidate_id}")
        for metric_name in (
            "ordinary_accuracy",
            "balanced_accuracy",
            "head_accuracy",
            "medium_accuracy",
            "tail_accuracy",
        ):
            try:
                metric_value = float(row[metric_name])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise Task3EError(
                    f"Task 3E-A metric {metric_name} is malformed for {candidate_id}"
                ) from exc
            if not np.isfinite(metric_value):
                raise Task3EError(
                    f"Task 3E-A metric {metric_name} is non-finite for {candidate_id}"
                )
    if actual_ids != [candidate.candidate_id for candidate in generate_fixed_weight_candidates()]:
        raise Task3EError("Task 3E-A candidate ordering is not the frozen ordering")
    if not isinstance(results.get("uniform_baseline"), Mapping):
        raise Task3EError("Task 3E-A results lack the uniform baseline")
    if results.get("uniform_baseline", {}).get("candidate_id") != "fixed_020":
        raise Task3EError("Task 3E-A uniform baseline is not fixed_020")
    if results.get("uniform_baseline_verification", {}).get("matches") is not True:
        raise Task3EError("Task 3E-A uniform baseline verification did not pass")
    return {
        "directory": fixed_directory,
        "config": config,
        "results": results,
        "input_artifacts": expected_fixed_inputs,
        "task3c_reference": diagnostics_reference_data,
    }


class SoftFeasibilityAnalyzer:
    """Run the soft-mixture oracle over one restricted analysis view."""

    def __init__(
        self,
        dataset: RestrictedAnalysisDataset,
        class_counts: np.ndarray,
        *,
        oracle: SoftMixtureOracle | None = None,
    ) -> None:
        if not isinstance(dataset, RestrictedAnalysisDataset):
            raise Task3EError(
                "SoftFeasibilityAnalyzer accepts only RestrictedAnalysisDataset; "
                "the complete aligned OOF dataset is not an analysis input"
            )
        counts = np.asarray(class_counts)
        if counts.shape != (dataset.logits.shape[2],):
            raise Task3EError("class_counts must match the restricted logits class dimension")
        if not np.issubdtype(counts.dtype, np.number) or np.iscomplexobj(counts):
            raise Task3EError("class_counts must be real numeric values")
        if np.any(counts < 1) or not np.equal(counts, np.floor(counts)).all():
            raise Task3EError("class_counts must contain positive integer canonical counts")
        labels = np.asarray(dataset.labels, dtype=np.int64)
        if np.any(labels < 0) or np.any(labels >= dataset.logits.shape[2]):
            raise Task3EError("restricted labels are outside the logits class range")
        if set(np.unique(dataset.inner_fold_ids).tolist()) - set(PERMITTED_ANALYSIS_INNER_FOLDS):
            raise Task3EError("soft analysis received a reserved inner-fold-0 sample")
        if np.any(dataset.outer_fold_ids != OUTER_FOLD):
            raise Task3EError("soft analysis received a reserved outer-evaluation sample")
        self.dataset = dataset
        self.class_counts = counts.astype(np.int64, copy=False)
        self.class_groups = compute_class_groups(self.class_counts)
        self.oracle = oracle or SoftMixtureOracle(
            num_experts=len(EXPERT_ORDER), num_classes=dataset.logits.shape[2]
        )
        if self.oracle.num_classes != dataset.logits.shape[2]:
            raise Task3EError("oracle class dimension does not match the restricted dataset")
        self.diagnostics = ExpertDiagnostics(
            logits=dataset.logits,
            labels=labels,
            class_counts=self.class_counts,
            expert_names=dataset.expert_names,
        )

    def _fixed_comparisons(
        self,
        fixed_reference: Mapping[str, Any],
        *,
        soft_correctable: np.ndarray,
        statuses: Sequence[str],
        uniform_metrics: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        results = fixed_reference["results"]
        rows_by_id = {row["candidate_id"]: row for row in results["candidates"]}
        labels = np.asarray(self.dataset.labels, dtype=np.int64)
        unresolved = np.isin(statuses, ["numerically_ambiguous", "solver_failure"])
        comparisons = []
        for candidate_id in REQUIRED_FIXED_REFERENCE_IDS:
            row = rows_by_id[candidate_id]
            weights = np.asarray(row["weight_vector"], dtype=np.float64)
            weight_matrix = np.repeat(weights[None, :], self.dataset.num_samples, axis=0)
            report = self.diagnostics.evaluate_soft_mixture(weight_matrix, "logits")
            metrics = _as_metric_bundle(report["metrics"])
            stored_metrics = {
                key: float(row[key])
                for key in (
                    "ordinary_accuracy",
                    "balanced_accuracy",
                    "head_accuracy",
                    "medium_accuracy",
                    "tail_accuracy",
                )
            }
            differences = {
                key: metrics[key] - stored_metrics[key] for key in stored_metrics
            }
            if any(abs(value) > FIXED_REFERENCE_TOLERANCE for value in differences.values()):
                raise Task3EError(
                    f"restricted data does not reproduce Task 3E-A row {candidate_id}: "
                    f"{differences}"
                )
            fixed_correct = report["predictions"] == labels
            comparisons.append(
                {
                    "candidate_id": candidate_id,
                    "weight_vector": [float(value) for value in weights.tolist()],
                    "stored_metrics": stored_metrics,
                    "recomputed_metrics": metrics,
                    "metric_differences": differences,
                    "metrics_match": True,
                    "fixed_correct_count": int(fixed_correct.sum()),
                    "fixed_correct_fraction": float(fixed_correct.mean()),
                    "oracle_strict_correctable_count": int(soft_correctable.sum()),
                    "oracle_strict_correctable_fraction": float(soft_correctable.mean()),
                    "fixed_correct_and_soft_feasible_count": int(
                        (fixed_correct & soft_correctable).sum()
                    ),
                    "fixed_incorrect_but_soft_feasible_count": int(
                        ((~fixed_correct) & soft_correctable).sum()
                    ),
                    "fixed_correct_but_not_strictly_feasible_count": int(
                        (fixed_correct & ~soft_correctable).sum()
                    ),
                    "ambiguous_or_unresolved_count": int(unresolved.sum()),
                    "ambiguous_or_unresolved_fraction": float(unresolved.mean()),
                    "delta_from_uniform": {
                        key: metrics[key] - float(uniform_metrics[key])
                        for key in metrics
                    },
                }
            )
        return comparisons

    def analyze(
        self,
        *,
        fixed_reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        logits = np.asarray(self.dataset.logits).astype(np.float64, copy=False)
        labels = np.asarray(self.dataset.labels, dtype=np.int64)
        expert_predictions = self.dataset.logits.argmax(axis=2).astype(np.int64)
        expert_correct = expert_predictions == labels[:, None]
        any_expert_correct = expert_correct.any(axis=1)
        uniform_weights = np.full(
            (self.dataset.num_samples, len(EXPERT_ORDER)),
            0.25,
            dtype=np.float64,
        )
        uniform_report = self.diagnostics.evaluate_soft_mixture(uniform_weights, "logits")
        uniform_predictions = uniform_report["predictions"]
        uniform_correct = uniform_predictions == labels
        uniform_margins = np.empty(self.dataset.num_samples, dtype=np.float64)
        for row_index, label in enumerate(labels.tolist()):
            competitors = np.arange(logits.shape[2]) != label
            uniform_margins[row_index] = float(
                uniform_report["combined_logits"][row_index, label]
                - np.max(uniform_report["combined_logits"][row_index, competitors])
            )

        statuses: list[str] = []
        oracle_results: list[SoftOracleResult] = []
        records: list[dict[str, Any]] = []
        for row_index in range(self.dataset.num_samples):
            result = self.oracle.solve(logits[row_index], int(labels[row_index]))
            oracle_results.append(result)
            statuses.append(result.status)
            all_wrong_soft = (not bool(any_expert_correct[row_index])) and result.status == "correctable"
            uniform_wrong_soft = (not bool(uniform_correct[row_index])) and result.status == "correctable"
            record = {
                "sample_index": int(self.dataset.sample_indices[row_index]),
                "true_label": int(labels[row_index]),
                "inner_fold_id": int(self.dataset.inner_fold_ids[row_index]),
                "uniform_correct": bool(uniform_correct[row_index]),
                "uniform_true_class_margin": float(uniform_margins[row_index]),
                "number_of_correct_experts": int(expert_correct[row_index].sum()),
                "hard_oracle_correct": bool(any_expert_correct[row_index]),
                "all_experts_wrong_but_soft_correctable": bool(all_wrong_soft),
                "uniform_wrong_but_soft_correctable": bool(uniform_wrong_soft),
            }
            record.update(result.to_dict())
            records.append(record)

        status_array = np.asarray(statuses, dtype=object)
        soft_correctable = status_array == "correctable"
        unresolved = np.isin(status_array, ["numerically_ambiguous", "solver_failure"])
        status_counts = {
            status: int((status_array == status).sum())
            for status in (
                "correctable",
                "not_strictly_correctable",
                "numerically_ambiguous",
                "solver_failure",
            )
        }
        hard_report = self.diagnostics.hard_routing_headroom()
        hard_coverage = _coverage_metrics(labels, any_expert_correct, self.class_groups)
        hard_metrics = _as_metric_bundle(hard_report["hard_selection_oracle_metrics"])
        hard_metric_differences = {
            "ordinary_accuracy": hard_coverage["ordinary_coverage"] - hard_metrics["ordinary_accuracy"],
            "balanced_accuracy": hard_coverage["balanced_accuracy"] - hard_metrics["balanced_accuracy"],
            "head_accuracy": hard_coverage["head_coverage"] - hard_metrics["head_accuracy"],
            "medium_accuracy": hard_coverage["medium_coverage"] - hard_metrics["medium_accuracy"],
            "tail_accuracy": hard_coverage["tail_coverage"] - hard_metrics["tail_accuracy"],
        }
        if any(abs(value) > FIXED_REFERENCE_TOLERANCE for value in hard_metric_differences.values()):
            raise Task3EError(
                "hard-selection correctness indicators disagree with the canonical hard oracle"
            )

        individual_positive_margins = np.empty_like(expert_predictions, dtype=np.float64)
        for row_index, label in enumerate(labels.tolist()):
            competitors = np.arange(logits.shape[2]) != label
            individual_positive_margins[row_index] = (
                logits[row_index][:, label, None]
                - logits[row_index][:, competitors]
            ).min(axis=1)
        individual_positive = (individual_positive_margins > self.oracle.margin_tolerance).any(axis=1)
        uniform_positive_violation = np.flatnonzero(
            uniform_correct
            & (uniform_margins > self.oracle.margin_tolerance)
            & ~soft_correctable
        )
        individual_positive_violation = np.flatnonzero(individual_positive & ~soft_correctable)
        numerical_invariants = {
            "uniform_positive_margin_count": int(
                (uniform_correct & (uniform_margins > self.oracle.margin_tolerance)).sum()
            ),
            "uniform_positive_margin_violations": [int(value) for value in uniform_positive_violation.tolist()],
            "strictly_positive_individual_expert_margin_count": int(individual_positive.sum()),
            "strictly_positive_individual_expert_margin_violations": [
                int(value) for value in individual_positive_violation.tolist()
            ],
            "max_abs_reported_reconstructed_margin_difference": float(
                max(
                    (
                        abs(float(result.reported_margin) - float(result.reconstructed_margin))
                        for result in oracle_results
                        if result.verification_passed
                        and result.reported_margin is not None
                        and result.reconstructed_margin is not None
                    ),
                    default=0.0,
                )
            ),
            "verification_failure_count": int(status_counts["solver_failure"]),
        }
        if uniform_positive_violation.size or individual_positive_violation.size:
            raise SoftOracleVerificationError(
                "a strictly positive feasible individual/uniform prediction was not "
                "accepted by the soft oracle; inspect numerical verification"
            )

        soft_coverage = _coverage_metrics(labels, soft_correctable, self.class_groups)
        all_wrong_soft = (~any_expert_correct) & soft_correctable
        uniform_wrong_soft = (~uniform_correct) & soft_correctable
        not_strictly = status_array == "not_strictly_correctable"
        class_status = _class_status_distribution(labels, statuses, logits.shape[2])
        tail_classes = self.class_groups["tail"]
        tail_report = _tail_class_report(
            labels,
            uniform_correct,
            any_expert_correct,
            soft_correctable,
            statuses,
            tail_classes,
        )

        aggregate = {
            "status_counts": status_counts,
            "status_fractions": {
                key: float(value / self.dataset.num_samples)
                for key, value in status_counts.items()
            },
            "soft_oracle": {
                "strict_correctable": soft_coverage,
                "strict_correctable_count": int(soft_correctable.sum()),
                "strict_correctable_fraction": float(soft_correctable.mean()),
                "ordinary_correctable_fraction": float(soft_correctable.mean()),
                "balanced_correctable_coverage": soft_coverage["balanced_accuracy"],
                "head_correctable_coverage": soft_coverage["head_coverage"],
                "medium_correctable_coverage": soft_coverage["medium_coverage"],
                "tail_correctable_coverage": soft_coverage["tail_coverage"],
                "lower_bound_count": int(soft_correctable.sum()),
                "lower_bound_fraction": float(soft_correctable.mean()),
                "potential_upper_bound_count": int((soft_correctable | unresolved).sum()),
                "potential_upper_bound_fraction": float(
                    (soft_correctable | unresolved).mean()
                ),
                "unresolved_count": int(unresolved.sum()),
                "unresolved_fraction": float(unresolved.mean()),
                "negative_optimum_count": int(not_strictly.sum()),
            },
            "uniform": {
                "metrics": _as_metric_bundle(uniform_report["metrics"]),
                "correct_count": int(uniform_correct.sum()),
                "correct_fraction": float(uniform_correct.mean()),
            },
            "hard_selection_oracle": {
                "metrics": hard_metrics,
                "correct_count": int(any_expert_correct.sum()),
                "correct_fraction": float(any_expert_correct.mean()),
                "coverage_metrics": hard_coverage,
                "stored_headroom": hard_report,
                "canonical_metric_differences": hard_metric_differences,
            },
            "uniform_wrong_but_soft_correctable": {
                "counts_by_group": _group_event_counts(labels, uniform_wrong_soft, self.class_groups),
                "coverage_metrics": _coverage_metrics(labels, uniform_wrong_soft, self.class_groups),
            },
            "all_experts_wrong_but_soft_correctable": {
                "counts_by_group": _group_event_counts(labels, all_wrong_soft, self.class_groups),
                "coverage_metrics": _coverage_metrics(labels, all_wrong_soft, self.class_groups),
                "count": int(all_wrong_soft.sum()),
                "fraction": float(all_wrong_soft.mean()),
            },
            "class_status_distribution": class_status,
            "numerical_verification": numerical_invariants,
        }

        fixed_comparisons = []
        baseline_comparison: dict[str, Any] = {
            "task3c_reference": {
                "uniform_metrics": fixed_reference["task3c_reference"]["uniform_metrics"]
                if fixed_reference is not None
                else None,
                "hard_headroom": fixed_reference["task3c_reference"]["hard_headroom"]
                if fixed_reference is not None
                else None,
            },
            "uniform_reproduction": None,
            "hard_selection_reproduction": None,
        }
        if fixed_reference is not None:
            task3c = fixed_reference["task3c_reference"]
            uniform_verification = verify_uniform_baseline(
                aggregate["uniform"]["metrics"], task3c["uniform_metrics"]
            )
            stored_hard = _as_metric_bundle(task3c["hard_headroom"]["hard_selection_oracle_metrics"])
            hard_differences = {
                key: hard_metrics[key] - stored_hard[key] for key in hard_metrics
            }
            if any(abs(value) > FIXED_REFERENCE_TOLERANCE for value in hard_differences.values()):
                raise Task3EError(f"hard-selection reference mismatch: {hard_differences}")
            baseline_comparison["uniform_reproduction"] = uniform_verification
            baseline_comparison["hard_selection_reproduction"] = {
                "matches": True,
                "tolerance": FIXED_REFERENCE_TOLERANCE,
                "stored_metrics": stored_hard,
                "restricted_metrics": hard_metrics,
                "absolute_differences": {key: abs(value) for key, value in hard_differences.items()},
            }
            fixed_comparisons = self._fixed_comparisons(
                fixed_reference,
                soft_correctable=soft_correctable,
                statuses=status_array,
                uniform_metrics=aggregate["uniform"]["metrics"],
            )
            baseline_comparison["fixed_weight_references"] = fixed_comparisons

        return {
            "analysis_partition": {
                "outer_fold": OUTER_FOLD,
                "inner_fold_ids": [int(value) for value in PERMITTED_ANALYSIS_INNER_FOLDS],
                "sample_count": self.dataset.num_samples,
                "sample_index_sha256": _hash_indices(self.dataset.sample_indices.tolist()),
                "outer_evaluation_excluded": True,
                "router_selection_inner_fold_excluded": True,
            },
            "aggregate": aggregate,
            "baseline_comparison": baseline_comparison,
            "fixed_weight_comparisons": fixed_comparisons,
            "historical_three_expert_reference": {
                "reference_id": "task3c_three_expert_uniform_without_ce",
                "weight_vector": list(HISTORICAL_THREE_EXPERT_WEIGHTS),
                "balanced_accuracy": 0.3719,
                "tail_accuracy": 0.1385,
                "approximately": True,
                "source": "historical Task 3C reference specified by the Task 3E-B protocol",
                "not_a_task3e_a_grid_candidate": True,
                "not_recomputed_or_used_for_optimization": True,
            },
            "tail_class_analysis": tail_report,
            "per_image": records,
            "oracle_definition": {
                "status_threshold": self.oracle.margin_tolerance,
                "statuses": [
                    "correctable",
                    "not_strictly_correctable",
                    "numerically_ambiguous",
                    "solver_failure",
                ],
                "weights_are_label_dependent": True,
                "not_an_inference_time_router": True,
            },
        }


def build_experiment_config(
    *,
    project_root: Path,
    source_files: Mapping[str, Mapping[str, Any]],
    fixed_reference: Mapping[str, Any],
    manager: NestedOOFFoldManager,
    dataset: RestrictedAnalysisDataset,
    execution_command: str | None = None,
) -> dict[str, Any]:
    groups = compute_class_groups(np.asarray(manager.canonical_class_counts, dtype=np.int64))
    router_fit_indices = manager.outer_fold(OUTER_FOLD).router_development.fit_indices
    input_artifacts = {
        name: {
            "path": _repository_relative(Path(details["path"]), project_root),
            "sha256": str(details["sha256"]),
        }
        for name, details in source_files.items()
    }
    input_artifacts["task3c_diagnostics"] = {
        "path": _repository_relative(fixed_reference["task3c_reference"]["path"], project_root),
        "sha256": fixed_reference["task3c_reference"]["sha256"],
    }
    input_artifacts.update(fixed_reference["input_artifacts"])
    return {
        "task_identifier": TASK3E_SOFT_TASK_IDENTIFIER,
        "schema_version": TASK3E_SOFT_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root),
        "execution_command": execution_command,
        "input_artifacts": input_artifacts,
        "dataset": DATASET_NAME,
        "imbalance_ratio": IMBALANCE_RATIO,
        "canonical_training_population_size": CANONICAL_POPULATION_SIZE,
        "full_aligned_sample_count": FULL_ALIGNED_SAMPLE_COUNT,
        "training_seed": TRAINING_SEED,
        "fold_generation_seed": FOLD_GENERATION_SEED,
        "outer_fold": OUTER_FOLD,
        "outer_evaluation_population_size": OUTER_EVALUATION_SIZE,
        "permitted_analysis_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "reserved_router_selection_inner_folds": list(RESERVED_ROUTER_SELECTION_INNER_FOLDS),
        "analyzed_sample_count": dataset.num_samples,
        "analysis_sample_index_sha256": _hash_indices(dataset.sample_indices.tolist()),
        "router_fit_sample_index_sha256": _hash_indices(router_fit_indices),
        "canonical_training_index_sha256": manager.canonical_training_index_sha256,
        "expert_order": list(EXPERT_ORDER),
        "number_of_classes": NUM_CLASSES,
        "original_logit_dtype": str(dataset.logits.dtype),
        "calculation_dtype": "float64",
        "lp_formulation": {
            "variables": ["w_CE", "w_LAL", "w_BalancedSoftmax", "w_Mixup", "t"],
            "difference_matrix": "D[c,e] = z[e,true_label] - z[e,c] for c != true_label",
            "objective": [0.0, 0.0, 0.0, 0.0, -1.0],
            "inequality": "-D @ w + t <= 0",
            "equality": [[1.0, 1.0, 1.0, 1.0, 0.0]],
            "equality_rhs": [1.0],
            "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [None, None]],
            "constraints_per_image": NUM_CLASSES - 1,
            "logits_are_unmodified": True,
            "weights_are_optimized_independently_per_image": True,
        },
        "solver": {
            "library": "scipy.optimize.linprog",
            "scipy_version": scipy.__version__,
            "method": SOLVER_METHOD,
            "options": dict(SOLVER_OPTIONS),
        },
        "numerical_tolerances": {
            "margin_tolerance": MARGIN_TOLERANCE,
            "solver_feasibility_tolerance": SOLVER_FEASIBILITY_TOLERANCE,
            "margin_verification_tolerance": MARGIN_VERIFICATION_TOLERANCE,
            "fixed_reference_tolerance": FIXED_REFERENCE_TOLERANCE,
        },
        "correctability_criteria": {
            "correctable": "verified reconstructed optimal margin > 1e-6",
            "not_strictly_correctable": "verified reconstructed optimal margin < -1e-6",
            "numerically_ambiguous": "verified reconstructed optimal margin in [-1e-6, 1e-6]",
            "solver_failure": "solver failure or failed numerical verification",
            "lower_bound": "strictly correctable count divided by analyzed sample count",
            "potential_upper_bound": "strictly correctable plus ambiguous or failed count divided by analyzed sample count",
        },
        "class_group_definitions": {
            "source": "scripts.base_trainer.compute_class_groups",
            "head": [int(value) for value in groups["head"].tolist()],
            "medium": [int(value) for value in groups["medium"].tolist()],
            "tail": [int(value) for value in groups["tail"].tolist()],
        },
        "metric_definitions": {
            "ordinary_accuracy": "canonical prediction accuracy for references; ordinary feasibility fraction for oracle events",
            "balanced_accuracy": "mean per-class recall for predictions or mean per-class feasibility coverage for oracle events",
            "head_accuracy": "macro recall/coverage over canonical Head classes",
            "medium_accuracy": "macro recall/coverage over canonical Medium classes",
            "tail_accuracy": "macro recall/coverage over canonical Tail classes",
        },
        "fixed_reference_ids": list(REQUIRED_FIXED_REFERENCE_IDS),
        "historical_three_expert_reference": {
            "weights": list(HISTORICAL_THREE_EXPERT_WEIGHTS),
            "balanced_accuracy_approx": 0.3719,
            "tail_accuracy_approx": 0.1385,
            "not_a_grid_candidate": True,
        },
        "scientific_limitations": [
            "The oracle uses each image's true label and is a label-dependent feasibility diagnostic, not a deployable router.",
            "All new calculations use only outer-fold-0 inner folds 1–3; earlier Task 3C diagnostics inspected inner fold 0 descriptively.",
            "The reserved outer-evaluation population and original CIFAR-100 test set are not accessed.",
            "Strict correctability is an existence result for convex logit mixtures and does not predict Ridge, Sinkhorn, or any learned router performance.",
            "The historical three-expert reference is approximate context and is not optimized or added to the E-A grid.",
        ],
        "source_restriction": {
            "accepted_analysis_type": "RestrictedAnalysisDataset",
            "complete_aligned_dataset_rejected": True,
            "inner_fold_0_rejected": True,
            "outer_evaluation_rejected": True,
            "test_loader_constructed": False,
            "expert_training_started": False,
        },
    }


def build_results_payload(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_identifier": TASK3E_SOFT_TASK_IDENTIFIER,
        "schema_version": TASK3E_SOFT_RESULTS_SCHEMA_VERSION,
        "analysis_partition": dict(analysis["analysis_partition"]),
        "oracle_definition": dict(analysis["oracle_definition"]),
        "aggregate": dict(analysis["aggregate"]),
        "baseline_comparison": dict(analysis["baseline_comparison"]),
        "fixed_weight_comparisons": list(analysis["fixed_weight_comparisons"]),
        "historical_three_expert_reference": dict(analysis["historical_three_expert_reference"]),
        "tail_class_analysis": list(analysis["tail_class_analysis"]),
        "per_image": list(analysis["per_image"]),
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.4f}%"


def _pp(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):+.4f} pp"


def _metric_line(metrics: Mapping[str, Any]) -> str:
    return (
        f"ordinary={_pct(metrics.get('ordinary_accuracy'))}, "
        f"BA={_pct(metrics.get('balanced_accuracy'))}, "
        f"Head={_pct(metrics.get('head_accuracy'))}, "
        f"Medium={_pct(metrics.get('medium_accuracy'))}, "
        f"Tail={_pct(metrics.get('tail_accuracy'))}"
    )


def render_summary(
    experiment_config: Mapping[str, Any], results: Mapping[str, Any]
) -> str:
    aggregate = results["aggregate"]
    soft = aggregate["soft_oracle"]
    hard = aggregate["hard_selection_oracle"]
    uniform = aggregate["uniform"]["metrics"]
    baseline = results["baseline_comparison"]
    lines = [
        "# Task 3E-B — Adaptive Soft-Mixture Oracle Feasibility Study",
        "",
        "This is a label-dependent, exploratory feasibility diagnostic. It solves an independent maximum-margin LP for each permitted OOF image; it is not an inference-time router and is not an independently validated generalization result.",
        "",
        "## Provenance and restrictions",
        "",
        f"- Source commit: `{experiment_config['source_git_commit']}`",
        f"- Dataset: {experiment_config['dataset']}, IR={int(experiment_config['imbalance_ratio'])}; canonical population={experiment_config['canonical_training_population_size']}",
        f"- Analyzed population: {experiment_config['analyzed_sample_count']} samples, outer fold {experiment_config['outer_fold']}, inner folds {experiment_config['permitted_analysis_inner_folds']}",
        f"- Expert order: `{', '.join(experiment_config['expert_order'])}`; logits calculated in `{experiment_config['calculation_dtype']}` from original `{experiment_config['original_logit_dtype']}` values",
            "- Inner fold 0, the reserved outer-evaluation population, and the CIFAR-100 test set were excluded from new calculations.",
        "",
        "## Numerical verification",
        "",
        f"- SciPy `{experiment_config['solver']['scipy_version']}`, `linprog(method='highs')`; margin tolerance `{experiment_config['numerical_tolerances']['margin_tolerance']}`.",
        f"- Status counts: `{aggregate['status_counts']}`.",
        f"- Strict correctability lower bound: **{_pct(soft['lower_bound_fraction'])}** ({soft['strict_correctable_count']}/{experiment_config['analyzed_sample_count']}). Potential upper bound if all unresolved cases were feasible: **{_pct(soft['potential_upper_bound_fraction'])}**.",
        f"- Uniform-positive-margin invariant violations: `{aggregate['numerical_verification']['uniform_positive_margin_violations']}`; positive-individual-expert violations: `{aggregate['numerical_verification']['strictly_positive_individual_expert_margin_violations']}`.",
        "",
        "## Reference results",
        "",
        f"- Uniform restricted-data metrics: {_metric_line(uniform)}",
        f"- Hard-selection oracle restricted-data metrics: {_metric_line(hard['metrics'])}",
        f"- Uniform reproduction verified: **{baseline['uniform_reproduction']['matches'] if baseline['uniform_reproduction'] else 'not run'}**; hard-selection reproduction verified: **{baseline['hard_selection_reproduction']['matches'] if baseline['hard_selection_reproduction'] else 'not run'}**.",
        "",
        "| Reference | Weights | BA | Tail | Fixed correct & soft-feasible | Fixed incorrect & soft-feasible |",
        "|---|---|---:|---:|---:|---:|",
        f"| Uniform | (0.25, 0.25, 0.25, 0.25) | {_pct(uniform['balanced_accuracy'])} | {_pct(uniform['tail_accuracy'])} | n/a | n/a |",
    ]
    for comparison in results["fixed_weight_comparisons"]:
        weights = "(" + ", ".join(f"{value:.2f}" for value in comparison["weight_vector"]) + ")"
        metrics = comparison["recomputed_metrics"]
        lines.append(
            f"| {comparison['candidate_id']} | {weights} | {_pct(metrics['balanced_accuracy'])} | {_pct(metrics['tail_accuracy'])} | {comparison['fixed_correct_and_soft_feasible_count']} | {comparison['fixed_incorrect_but_soft_feasible_count']} |"
        )
    historical = results["historical_three_expert_reference"]
    lines.append(
        f"| Historical three-expert reference (not E-A grid) | (0, 1/3, 1/3, 1/3) | {_pct(historical['balanced_accuracy'])} approx. | {_pct(historical['tail_accuracy'])} approx. | n/a | n/a |"
    )
    lines.extend(
        [
            "",
            "## Soft-oracle feasibility and headroom",
            "",
            f"- Soft-oracle strict correctability: **{_pct(soft['strict_correctable_fraction'])}**; macro BA feasibility **{_pct(soft['strict_correctable']['balanced_accuracy'])}**, Head **{_pct(soft['strict_correctable']['head_coverage'])}**, Medium **{_pct(soft['strict_correctable']['medium_coverage'])}**, Tail **{_pct(soft['strict_correctable']['tail_coverage'])}**.",
            f"- All experts wrong but soft-correctable: **{aggregate['all_experts_wrong_but_soft_correctable']['count']}** ({_pct(aggregate['all_experts_wrong_but_soft_correctable']['fraction'])}); its macro class coverage is **{_pct(aggregate['all_experts_wrong_but_soft_correctable']['coverage_metrics']['balanced_accuracy'])}** and Tail event coverage is **{_pct(aggregate['all_experts_wrong_but_soft_correctable']['coverage_metrics']['tail_coverage'])}**.",
            f"- Uniform wrong but soft-correctable: {aggregate['uniform_wrong_but_soft_correctable']['counts_by_group']['all']['count']} ({_pct(aggregate['uniform_wrong_but_soft_correctable']['counts_by_group']['all']['fraction'])}).",
            "- These are existence measurements using true labels. They do not estimate the accuracy of Ridge, Sinkhorn, or another learned router.",
            "",
            "## Tail-class feasibility",
            "",
            "| Class | N | Uniform | Any expert | Soft | Newly soft | Negative optimum | Ambiguous | Solver failure |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["tail_class_analysis"]:
        lines.append(
            f"| {row['class_index']} | {row['sample_count']} | {row['uniform_correct']} | {row['any_expert_correct']} | {row['soft_correctable']} | {row['newly_soft_correctable']} | {row['uncorrectable']} | {row['numerically_ambiguous']} | {row['solver_failure']} |"
        )
    lines.extend(
        [
            "",
            "Tail counts are development-population counts; classes with very few permitted images are not reliable population-level estimates. The soft Tail macro coverage should therefore be interpreted alongside the per-class denominators above.",
            "",
            "## Limitations and implications",
            "",
            "- The LP sees the true class and returns label-dependent oracle weights. This establishes convex-mixture feasibility only; it does not establish that an inference-time model can predict those weights.",
            "- The strict result counts only verified margins above the frozen `1e-6` threshold. Ambiguous and solver-failure cases remain unresolved rather than being treated as correctable or uncorrectable; the potential upper bound is intentionally non-definitive.",
            "- No post-hoc tolerance sensitivity analysis was used to replace the frozen result; the reported statuses use the predeclared `1e-6` margin band.",
            "- Earlier Task 3C diagnostics did inspect inner fold 0 descriptively, so that partition cannot be described as untouched for all prior research decisions even though E-B excludes it completely.",
            "- The result cannot establish generalization to other folds, seeds, the reserved outer evaluation population, or the balanced CIFAR-100 test set.",
            "- Task 3E-B does not implement or authorize Ridge, Sinkhorn, adaptive routing, new experts, or a final expert-weight selection. Any next experiment must use a separate, pre-registered protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs_once(files: Mapping[Path, str]) -> None:
    write_texts_once(files, error_type=Task3EError)


def run_task3e_soft(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    fixed_results_directory: str | Path = "artifacts/oof/task3e_fixed_feasibility",
    reference_diagnostics: str | Path | None = None,
    output_directory: str | Path = "artifacts/oof/task3e_soft_feasibility",
    project_root: str | Path | None = None,
    execution_command: str | None = None,
) -> dict[str, Path]:
    """Execute E-B and write the three required artifacts after validation."""
    project_root_path = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    source_directory = Path(oof_directory).resolve()
    reference_path = Path(reference_diagnostics or source_directory / "diagnostics.json").resolve()
    output_path = Path(output_directory).resolve()
    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=FOLD_GENERATION_SEED,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    groups = compute_class_groups(np.asarray(manager.canonical_class_counts, dtype=np.int64))
    expected_tail = np.arange(70, 100, dtype=np.int64)
    if not np.array_equal(groups["tail"], expected_tail):
        raise Task3EError("canonical Task 3E-B Tail classes are not 70–99")
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    fixed_reference = validate_fixed_reference_artifacts(
        fixed_results_directory,
        project_root=project_root_path,
        current_source_files=source_files,
        reference_diagnostics=reference_path,
        manager=manager,
    )
    analyzer = SoftFeasibilityAnalyzer(
        dataset,
        np.asarray(manager.canonical_class_counts, dtype=np.int64),
    )
    analysis = analyzer.analyze(fixed_reference=fixed_reference)
    source_files_with_references = dict(source_files)
    source_files_with_references["task3c_diagnostics"] = fixed_reference["task3c_reference"]
    experiment_config = build_experiment_config(
        project_root=project_root_path,
        source_files=source_files_with_references,
        fixed_reference=fixed_reference,
        manager=manager,
        dataset=dataset,
        execution_command=execution_command,
    )
    results = build_results_payload(analysis)
    summary = render_summary(experiment_config, results)
    files = {
        output_path / "experiment_config.json": serialize_json(experiment_config),
        output_path / "soft_oracle_results.json": serialize_json(results),
        output_path / "summary.md": summary,
    }
    _write_outputs_once(files)
    return {
        "experiment_config": output_path / "experiment_config.json",
        "soft_oracle_results": output_path / "soft_oracle_results.json",
        "summary": output_path / "summary.md",
    }


__all__ = [
    "FIXED_REFERENCE_TOLERANCE",
    "MARGIN_TOLERANCE",
    "SoftFeasibilityAnalyzer",
    "SoftMixtureOracle",
    "SoftOracleInputError",
    "SoftOracleResult",
    "TASK3E_SOFT_RESULTS_SCHEMA_VERSION",
    "TASK3E_SOFT_SCHEMA_VERSION",
    "TASK3E_SOFT_TASK_IDENTIFIER",
    "build_experiment_config",
    "build_results_payload",
    "render_summary",
    "run_task3e_soft",
    "validate_fixed_reference_artifacts",
]
