"""Task 3E-A: fixed-weight ensemble feasibility analysis.

This module is an analysis-only seam over the validated Task 3C aligned OOF
artifact.  It does not load images, checkpoints, or the CIFAR-100 test set.
The only label-dependent calculations are made after the aligned artifact has
been reduced to the exact outer-0 router-fit partition (inner folds 1--3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.nested_oof import FoldManifest, NestedOOFFoldManager, OOFProtocolError
from scripts.analysis.artifacts import (
    git_commit,
    load_json_object,
    repository_relative,
    serialize_json,
    sha256_file,
    validate_sha256,
    write_texts_once,
)
from scripts.base_trainer import compute_class_groups
from scripts.expert_diagnostics import ExpertDiagnostics
from scripts.task3c_oof import (
    ALIGNED_OOF_SCHEMA_VERSION,
    AlignedOOFDataset,
)


TASK3E_SCHEMA_VERSION = "task3e_fixed_feasibility.v1"
TASK3E_RESULTS_SCHEMA_VERSION = "task3e_fixed_results.v1"
TASK3E_TASK_IDENTIFIER = "Task 3E-A"

DATASET_NAME = "CIFAR-100-LT"
IMBALANCE_RATIO = 100.0
CANONICAL_POPULATION_SIZE = 10_847
FULL_ALIGNED_SAMPLE_COUNT = 8_677
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
TRAINING_SEED = 78
FOLD_GENERATION_SEED = 42
OUTER_FOLD = 0
PERMITTED_ANALYSIS_INNER_FOLDS = (1, 2, 3)
RESERVED_ROUTER_SELECTION_INNER_FOLDS = (0,)
OUTER_EVALUATION_SIZE = 2_170
EXPERT_ORDER = ("CE", "LAL", "BalancedSoftmax", "Mixup")
EXPERT_KEYS = ("ce", "logit_adjusted", "balanced_softmax", "mixup")
EXPECTED_ALIGNED_INNER_FOLDS = (0, 1, 2, 3)
UNIFORM_BASELINE_TOLERANCE = 1e-12


class Task3EError(OOFProtocolError):
    """Raised when Task 3E-A validation or analysis is unsafe."""


def _sha256_file(path: Path) -> str:
    return sha256_file(path, error_type=Task3EError, description="input artifact")


def _valid_sha256(value: Any, *, name: str) -> str:
    return validate_sha256(value, name=name, error_type=Task3EError)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    return load_json_object(path, name=name, error_type=Task3EError)


def _require_equal(payload: Mapping[str, Any], key: str, expected: Any, *, name: str) -> None:
    if payload.get(key) != expected:
        raise Task3EError(
            f"{name}.{key} is incompatible: expected {expected!r}, "
            f"got {payload.get(key)!r}"
        )


def _integer_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    """Return non-negative integer compositions in lexicographic order."""
    if total < 0 or parts < 1:
        raise Task3EError("composition total must be non-negative and parts positive")
    if parts == 1:
        return ((total,),)
    compositions = []
    for first in range(total + 1):
        for rest in _integer_compositions(total - first, parts - 1):
            compositions.append((first, *rest))
    return tuple(compositions)


@dataclass(frozen=True)
class FixedWeightCandidate:
    """One point on the frozen four-unit fixed-weight grid."""

    candidate_id: str
    units: tuple[int, ...]
    weights: tuple[float, ...]


def generate_fixed_weight_candidates() -> tuple[FixedWeightCandidate, ...]:
    """Generate exactly the 35 four-expert grid candidates.

    Integer units are generated first and divided by four only when the public
    weight vector is formed.  This makes the candidate set independent of
    floating-point sum/filter behaviour.
    """
    compositions = _integer_compositions(4, len(EXPERT_ORDER))
    candidates = tuple(
        FixedWeightCandidate(
            candidate_id=f"fixed_{position:03d}",
            units=units,
            weights=tuple(unit / 4.0 for unit in units),
        )
        for position, units in enumerate(compositions)
    )
    if len(candidates) != 35 or len({candidate.units for candidate in candidates}) != 35:
        raise Task3EError("the frozen four-unit grid must contain exactly 35 candidates")
    return candidates


def _candidate_for_units(units: tuple[int, ...]) -> FixedWeightCandidate:
    for candidate in generate_fixed_weight_candidates():
        if candidate.units == units:
            return candidate
    raise Task3EError(f"weight units are not on the frozen grid: {units!r}")


def _validate_frozen_manager(manager: NestedOOFFoldManager) -> None:
    if (
        len(manager.canonical_indices) != CANONICAL_POPULATION_SIZE
        or manager.num_classes != NUM_CLASSES
        or manager.fold_generation_seed != FOLD_GENERATION_SEED
        or manager.outer_fold_count != 5
        or manager.inner_fold_count != 4
        or manager.expert_order != EXPERT_ORDER
    ):
        raise Task3EError("fold manager does not match the frozen Task 3E-A configuration")
    outer = manager.outer_fold(OUTER_FOLD)
    router = outer.router_development
    if len(outer.evaluation_indices) != OUTER_EVALUATION_SIZE:
        raise Task3EError("outer fold 0 does not reserve exactly 2,170 samples")
    if tuple(router.fit_inner_fold_ids) != PERMITTED_ANALYSIS_INNER_FOLDS:
        raise Task3EError("fold manager router-fit folds are not inner folds 1–3")
    if tuple(router.selection_inner_fold_ids) != RESERVED_ROUTER_SELECTION_INNER_FOLDS:
        raise Task3EError("fold manager router-selection fold is not inner fold 0")
    if len(router.fit_indices) != ANALYSIS_SAMPLE_COUNT:
        raise Task3EError("fold manager router-fit partition is not 6,507 samples")


def _validate_source_provenance(
    source_directory: Path,
    aligned: AlignedOOFDataset,
    manager: NestedOOFFoldManager,
) -> dict[str, dict[str, Any]]:
    """Validate Task 3C sidecars and return the hashed input files."""
    metadata = aligned.metadata
    expected_metadata = {
        "schema_version": ALIGNED_OOF_SCHEMA_VERSION,
        "expert_names": list(EXPERT_ORDER),
        "fold_generation_seed": FOLD_GENERATION_SEED,
        "inner_fold_ids": list(EXPECTED_ALIGNED_INNER_FOLDS),
        "num_classes": NUM_CLASSES,
        "num_samples": FULL_ALIGNED_SAMPLE_COUNT,
        "outer_evaluation_excluded": True,
        "outer_evaluation_size": OUTER_EVALUATION_SIZE,
        "outer_fold_id": OUTER_FOLD,
        "router_fit_inner_fold_ids": list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "router_selection_inner_fold_ids": list(RESERVED_ROUTER_SELECTION_INNER_FOLDS),
        "primary_diagnostic_partition": "router_fit_inner_folds_1_2_3",
    }
    for key, expected in expected_metadata.items():
        _require_equal(metadata, key, expected, name="aligned_oof_metadata")
    if metadata.get("canonical_training_index_sha256") != manager.canonical_training_index_sha256:
        raise Task3EError("aligned metadata canonical training-index hash disagrees with the fold manager")

    batch_path = source_directory / "batch_manifest.json"
    fold_path = source_directory / "fold_manifest.json"
    batch = _load_json(batch_path, name="Task 3C batch manifest")
    _require_equal(batch, "schema_version", "task3c_oof_batch.v1", name="batch_manifest")
    _require_equal(batch, "experiment_id", "task3c_oof", name="batch_manifest")
    _require_equal(batch, "dataset", DATASET_NAME, name="batch_manifest")
    _require_equal(batch, "imbalance_ratio", IMBALANCE_RATIO, name="batch_manifest")
    _require_equal(batch, "canonical_population_size", CANONICAL_POPULATION_SIZE, name="batch_manifest")
    _require_equal(batch, "training_seed", TRAINING_SEED, name="batch_manifest")
    _require_equal(batch, "fold_generation_seed", FOLD_GENERATION_SEED, name="batch_manifest")
    _require_equal(batch, "outer_fold_id", OUTER_FOLD, name="batch_manifest")
    _require_equal(batch, "outer_evaluation_size", OUTER_EVALUATION_SIZE, name="batch_manifest")
    _require_equal(batch, "inner_fold_ids", list(EXPECTED_ALIGNED_INNER_FOLDS), name="batch_manifest")
    _require_equal(batch, "expert_order", list(EXPERT_ORDER), name="batch_manifest")
    router_development = batch.get("router_development")
    if not isinstance(router_development, Mapping):
        raise Task3EError("batch_manifest.router_development must be an object")
    _require_equal(router_development, "fit_inner_fold_ids", list(PERMITTED_ANALYSIS_INNER_FOLDS), name="batch_manifest.router_development")
    _require_equal(router_development, "selection_inner_fold_ids", list(RESERVED_ROUTER_SELECTION_INNER_FOLDS), name="batch_manifest.router_development")
    _require_equal(router_development, "outer_evaluation_excluded", True, name="batch_manifest.router_development")
    _require_equal(router_development, "router_fitting_implemented", False, name="batch_manifest.router_development")

    expected_job_ids = {
        f"{expert_key}/inner_{inner_id}"
        for expert_key in EXPERT_KEYS
        for inner_id in EXPECTED_ALIGNED_INNER_FOLDS
    }
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or any(not isinstance(job, Mapping) for job in jobs):
        raise Task3EError("batch manifest jobs must be objects")
    if len(jobs) != len(expected_job_ids) or {job.get("job_id") for job in jobs} != expected_job_ids:
        raise Task3EError("batch manifest does not describe the complete frozen 16-job expert matrix")

    try:
        fold_manifest = FoldManifest.from_json(fold_path.read_bytes())
    except (OSError, OOFProtocolError) as exc:
        raise Task3EError(f"invalid Task 3C fold manifest: {fold_path}") from exc
    if fold_manifest.to_dict() != manager.manifest().to_dict():
        raise Task3EError("Task 3C fold manifest does not match the regenerated fold manager")
    if batch.get("fold_manifest_sha256") != _sha256_file(fold_path):
        raise Task3EError("batch manifest fold-manifest hash does not match the fold manifest")

    source_checkpoints = metadata.get("source_checkpoints")
    if not isinstance(source_checkpoints, Mapping) or set(source_checkpoints) != expected_job_ids:
        raise Task3EError("aligned metadata does not identify every frozen source job")
    source_job_ids = metadata.get("source_job_ids")
    if (
        not isinstance(source_job_ids, list)
        or len(source_job_ids) != len(expected_job_ids)
        or set(source_job_ids) != expected_job_ids
    ):
        raise Task3EError("aligned metadata source_job_ids do not match the frozen source jobs")
    for job_id, checkpoint in source_checkpoints.items():
        if not isinstance(checkpoint, Mapping):
            raise Task3EError(f"source checkpoint provenance is malformed for {job_id}")
        _valid_sha256(checkpoint.get("sha256"), name=f"source checkpoint {job_id}")

    input_files = {
        "aligned_oof_arrays": source_directory / "aligned_oof.npz",
        "aligned_oof_metadata": source_directory / "aligned_oof_metadata.json",
        "task3c_batch_manifest": batch_path,
        "task3c_fold_manifest": fold_path,
    }
    return {
        name: {"path": path, "sha256": _sha256_file(path)}
        for name, path in input_files.items()
    }


@dataclass(frozen=True)
class RestrictedAnalysisDataset:
    """The only dataset view accepted by the fixed-weight analyzer."""

    sample_indices: np.ndarray
    labels: np.ndarray
    inner_fold_ids: np.ndarray
    outer_fold_ids: np.ndarray
    logits: np.ndarray
    expert_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices)
        labels = np.asarray(self.labels)
        inner_ids = np.asarray(self.inner_fold_ids)
        outer_ids = np.asarray(self.outer_fold_ids)
        logits = np.asarray(self.logits)
        if any(array.ndim != 1 for array in (indices, labels, inner_ids, outer_ids)):
            raise Task3EError("restricted analysis index and label arrays must be one-dimensional")
        if len(indices) == 0:
            raise Task3EError("restricted analysis dataset must not be empty")
        if not all(np.issubdtype(array.dtype, np.integer) for array in (indices, labels, inner_ids, outer_ids)):
            raise Task3EError("restricted analysis indices and labels must be integer arrays")
        if len({len(indices), len(labels), len(inner_ids), len(outer_ids)}) != 1:
            raise Task3EError("restricted analysis arrays are not aligned")
        if logits.ndim != 3 or logits.shape[:2] != (len(indices), len(EXPERT_ORDER)):
            raise Task3EError("restricted analysis logits must have shape (samples, 4, classes)")
        if not np.issubdtype(logits.dtype, np.number) or np.iscomplexobj(logits) or not np.isfinite(logits).all():
            raise Task3EError("restricted analysis logits must be finite real numeric values")
        if len(np.unique(indices)) != len(indices):
            raise Task3EError("restricted analysis sample IDs contain duplicates")
        if tuple(self.expert_names) != EXPERT_ORDER:
            raise Task3EError("restricted analysis expert ordering is not CE, LAL, BalancedSoftmax, Mixup")
        if np.any(outer_ids != OUTER_FOLD):
            raise Task3EError("restricted analysis contains a nonzero outer fold")
        allowed = set(PERMITTED_ANALYSIS_INNER_FOLDS)
        if not set(int(value) for value in np.unique(inner_ids)) <= allowed:
            raise Task3EError("restricted analysis contains the reserved inner-fold-0 partition")

        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "inner_fold_ids", inner_ids)
        object.__setattr__(self, "outer_fold_ids", outer_ids)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "expert_names", tuple(self.expert_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_samples(self) -> int:
        return int(len(self.sample_indices))

    @classmethod
    def from_aligned(
        cls,
        aligned: AlignedOOFDataset,
        manager: NestedOOFFoldManager,
        *,
        enforce_frozen_population: bool = True,
    ) -> "RestrictedAnalysisDataset":
        """Validate the complete artifact, then return only inner folds 1--3."""
        try:
            aligned.validate(manager)
        except (OOFProtocolError, ValueError) as exc:
            raise Task3EError("aligned OOF artifact fails fold or label validation") from exc
        selected = aligned.select_inner_folds(PERMITTED_ANALYSIS_INNER_FOLDS)
        expected = set(manager.outer_fold(OUTER_FOLD).router_development.fit_indices)
        actual = set(int(value) for value in selected.sample_indices.tolist())
        if actual != expected:
            raise Task3EError("selected samples are not exactly the frozen router-fit partition")
        outer_evaluation = set(manager.outer_fold(OUTER_FOLD).evaluation_indices)
        if actual & outer_evaluation:
            raise Task3EError("restricted analysis overlaps the reserved outer-evaluation population")
        if set(int(value) for value in np.unique(selected.inner_fold_ids)) != set(PERMITTED_ANALYSIS_INNER_FOLDS):
            raise Task3EError("restricted analysis does not contain exactly inner folds 1–3")
        expected_inner_by_index = {
            int(index): inner_id
            for inner_id in EXPECTED_ALIGNED_INNER_FOLDS
            for index in manager.inner_fold(OUTER_FOLD, inner_id).prediction_indices
        }
        if any(expected_inner_by_index[int(index)] != int(inner_id) for index, inner_id in zip(selected.sample_indices, selected.inner_fold_ids)):
            raise Task3EError("restricted analysis sample-to-inner-fold membership is incorrect")
        labels_by_index = dict(zip(manager.canonical_indices, manager.training_labels))
        if any(labels_by_index[int(index)] != int(label) for index, label in zip(selected.sample_indices, selected.labels)):
            raise Task3EError("restricted analysis labels disagree with the canonical training labels")
        if enforce_frozen_population and selected.num_samples != ANALYSIS_SAMPLE_COUNT:
            raise Task3EError("restricted analysis must contain exactly 6,507 samples")
        return cls(
            sample_indices=selected.sample_indices.copy(),
            labels=selected.labels.copy(),
            inner_fold_ids=selected.inner_fold_ids.copy(),
            outer_fold_ids=selected.outer_fold_ids.copy(),
            logits=selected.logits.copy(),
            expert_names=selected.expert_names,
            metadata={
                **selected.metadata,
                "restricted_analysis_inner_fold_ids": list(PERMITTED_ANALYSIS_INNER_FOLDS),
                "restricted_analysis_sample_count": selected.num_samples,
            },
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        sample_indices: np.ndarray,
        labels: np.ndarray,
        inner_fold_ids: np.ndarray,
        logits: np.ndarray,
        expert_names: Sequence[str] = EXPERT_ORDER,
        outer_fold_ids: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RestrictedAnalysisDataset":
        """Build a small synthetic restricted view for unit tests."""
        indices = np.asarray(sample_indices)
        if outer_fold_ids is None:
            outer_fold_ids = np.zeros(len(indices), dtype=np.int64)
        return cls(
            sample_indices=indices,
            labels=np.asarray(labels),
            inner_fold_ids=np.asarray(inner_fold_ids),
            outer_fold_ids=np.asarray(outer_fold_ids),
            logits=np.asarray(logits),
            expert_names=tuple(expert_names),
            metadata=dict(metadata or {"synthetic_fixture": True}),
        )


def load_restricted_analysis_dataset(
    source_directory: str | Path,
    manager: NestedOOFFoldManager,
) -> tuple[RestrictedAnalysisDataset, dict[str, dict[str, Any]]]:
    """Load, validate, and immediately restrict the existing Task 3C artifact."""
    _validate_frozen_manager(manager)
    source_directory = Path(source_directory)
    try:
        aligned = AlignedOOFDataset.load(source_directory, manager=manager)
    except (OOFProtocolError, OSError, ValueError) as exc:
        raise Task3EError(f"cannot load the validated aligned OOF artifact from {source_directory}") from exc
    restricted = RestrictedAnalysisDataset.from_aligned(aligned, manager)
    source_files = _validate_source_provenance(source_directory, aligned, manager)
    return restricted, source_files


def _metric_bundle(metrics: Mapping[str, Any]) -> dict[str, float]:
    required = ("accuracy", "ba", "head", "medium", "tail")
    if any(key not in metrics or metrics[key] is None for key in required):
        raise Task3EError("the canonical metric implementation did not produce all required metrics")
    return {
        "ordinary_accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["ba"]),
        "head_accuracy": float(metrics["head"]),
        "medium_accuracy": float(metrics["medium"]),
        "tail_accuracy": float(metrics["tail"]),
    }


class FixedWeightAnalyzer:
    """Evaluate the predefined global fixed-weight candidates."""

    def __init__(self, dataset: RestrictedAnalysisDataset, class_counts: np.ndarray) -> None:
        if not isinstance(dataset, RestrictedAnalysisDataset):
            raise Task3EError(
                "FixedWeightAnalyzer accepts only RestrictedAnalysisDataset; "
                "the complete aligned OOF dataset is not an analysis input"
            )
        self.dataset = dataset
        self.class_counts = np.asarray(class_counts)
        self.diagnostics = ExpertDiagnostics(
            logits=dataset.logits,
            labels=dataset.labels,
            class_counts=self.class_counts,
            expert_names=dataset.expert_names,
        )

    def weight_matrix(self, candidate: FixedWeightCandidate) -> np.ndarray:
        """Return one identical row per sample; no labels are consulted."""
        if len(candidate.weights) != len(EXPERT_ORDER):
            raise Task3EError("candidate has the wrong number of expert weights")
        return np.repeat(
            np.asarray(candidate.weights, dtype=np.float64)[None, :],
            self.dataset.num_samples,
            axis=0,
        )

    def evaluate_candidate(self, candidate: FixedWeightCandidate) -> dict[str, Any]:
        report = self.diagnostics.evaluate_soft_mixture(
            self.weight_matrix(candidate), "logits"
        )
        return {
            "candidate_id": candidate.candidate_id,
            "weight_units": list(candidate.units),
            "weight_vector": list(candidate.weights),
            **_metric_bundle(report["metrics"]),
        }

    def analyze(self) -> dict[str, Any]:
        candidates = generate_fixed_weight_candidates()
        rows = [self.evaluate_candidate(candidate) for candidate in candidates]
        uniform = _candidate_for_units((1, 1, 1, 1))
        uniform_row = next(row for row in rows if row["candidate_id"] == uniform.candidate_id)
        metric_names = (
            "balanced_accuracy",
            "head_accuracy",
            "medium_accuracy",
            "tail_accuracy",
        )
        for row in rows:
            row["delta_ba"] = row["balanced_accuracy"] - uniform_row["balanced_accuracy"]
            row["delta_head"] = row["head_accuracy"] - uniform_row["head_accuracy"]
            row["delta_medium"] = row["medium_accuracy"] - uniform_row["medium_accuracy"]
            row["delta_tail"] = row["tail_accuracy"] - uniform_row["tail_accuracy"]
            row["joint_improvement"] = row["delta_ba"] > 0.0 and row["delta_tail"] > 0.0
            if not all(np.isfinite(row[name]) for name in metric_names):
                raise Task3EError(f"candidate {row['candidate_id']} produced a non-finite metric")

        frontier_ids = pareto_frontier(rows)
        frontier_set = set(frontier_ids)
        for row in rows:
            row["pareto_frontier"] = row["candidate_id"] in frontier_set
        categories = _tradeoff_categories(rows)
        return {
            "candidate_count": len(rows),
            "candidates": rows,
            "uniform_baseline": uniform_row,
            "pareto_frontier_candidate_ids": list(frontier_ids),
            "joint_improvement_candidate_ids": [
                row["candidate_id"] for row in rows if row["joint_improvement"]
            ],
            "tradeoff_categories": categories,
        }


def pareto_frontier(results: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return all non-dominated results in the supplied deterministic order."""
    frontier: list[str] = []
    for index, candidate in enumerate(results):
        ba = float(candidate["balanced_accuracy"])
        tail = float(candidate["tail_accuracy"])
        dominated = False
        for other_index, other in enumerate(results):
            if index == other_index:
                continue
            other_ba = float(other["balanced_accuracy"])
            other_tail = float(other["tail_accuracy"])
            if (
                other_ba >= ba
                and other_tail >= tail
                and (other_ba > ba or other_tail > tail)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["candidate_id"]))
    return tuple(frontier)


def _tradeoff_categories(results: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    categories = {
        "improves_ba_and_tail": [],
        "improves_ba_reduces_tail": [],
        "improves_tail_reduces_ba": [],
        "reduces_ba_and_tail": [],
        "no_strict_change_or_mixed": [],
    }
    for row in results:
        delta_ba = float(row["delta_ba"])
        delta_tail = float(row["delta_tail"])
        candidate_id = str(row["candidate_id"])
        if delta_ba > 0.0 and delta_tail > 0.0:
            categories["improves_ba_and_tail"].append(candidate_id)
        elif delta_ba > 0.0 and delta_tail < 0.0:
            categories["improves_ba_reduces_tail"].append(candidate_id)
        elif delta_ba < 0.0 and delta_tail > 0.0:
            categories["improves_tail_reduces_ba"].append(candidate_id)
        elif delta_ba < 0.0 and delta_tail < 0.0:
            categories["reduces_ba_and_tail"].append(candidate_id)
        else:
            categories["no_strict_change_or_mixed"].append(candidate_id)
    return categories


def _metric_aliases(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "ordinary_accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["ba"]),
        "head_accuracy": float(metrics["head"]),
        "medium_accuracy": float(metrics["medium"]),
        "tail_accuracy": float(metrics["tail"]),
    }


def verify_uniform_baseline(
    actual: Mapping[str, Any],
    stored_task3c_metrics: Mapping[str, Any],
    *,
    tolerance: float = UNIFORM_BASELINE_TOLERANCE,
) -> dict[str, Any]:
    """Require Task 3E-A's uniform row to reproduce stored Task 3C values."""
    stored = _metric_aliases(stored_task3c_metrics)
    differences = {
        key: float(actual[key]) - float(stored[key])
        for key in stored
    }
    failures = {
        key: value for key, value in differences.items() if abs(value) > tolerance
    }
    if failures:
        raise Task3EError(
            "uniform baseline does not reproduce the stored Task 3C diagnostic "
            f"within tolerance {tolerance}: {failures}"
        )
    return {
        "matches": True,
        "tolerance": tolerance,
        "stored_task3c_metrics": stored,
        "task3e_metrics": {key: float(actual[key]) for key in stored},
        "absolute_differences": {key: abs(value) for key, value in differences.items()},
    }


def _load_stored_uniform_metrics(path: Path) -> dict[str, Any]:
    payload = _load_json(path, name="Task 3C diagnostics")
    _require_equal(payload, "schema_version", "task3c_diagnostics.v1", name="diagnostics")
    _require_equal(payload, "dataset", DATASET_NAME, name="diagnostics")
    _require_equal(payload, "imbalance_ratio", IMBALANCE_RATIO, name="diagnostics")
    _require_equal(payload, "training_seed", TRAINING_SEED, name="diagnostics")
    _require_equal(payload, "fold_generation_seed", FOLD_GENERATION_SEED, name="diagnostics")
    _require_equal(payload, "outer_fold_id", OUTER_FOLD, name="diagnostics")
    _require_equal(payload, "expert_order", list(EXPERT_ORDER), name="diagnostics")
    partition = payload.get("primary_router_fit_partition")
    if not isinstance(partition, Mapping):
        raise Task3EError("Task 3C diagnostics lack the primary router-fit partition")
    _require_equal(partition, "sample_count", ANALYSIS_SAMPLE_COUNT, name="diagnostics.primary_router_fit_partition")
    _require_equal(partition, "inner_fold_ids", list(PERMITTED_ANALYSIS_INNER_FOLDS), name="diagnostics.primary_router_fit_partition")
    uniform = partition.get("uniform_logit_ensemble")
    if not isinstance(uniform, Mapping) or uniform.get("combination") != "uniform logit average":
        raise Task3EError("Task 3C diagnostics lack the stored uniform logit baseline")
    metrics = uniform.get("metrics")
    if not isinstance(metrics, Mapping):
        raise Task3EError("stored Task 3C uniform baseline metrics are malformed")
    return dict(metrics)


def _repository_relative(path: Path, project_root: Path) -> str:
    return repository_relative(path, project_root)


def _git_commit(project_root: Path) -> str:
    return git_commit(project_root, error_type=Task3EError)


def build_experiment_config(
    *,
    project_root: Path,
    source_files: Mapping[str, Mapping[str, Any]],
    reference_diagnostics: Path,
    manager: NestedOOFFoldManager,
    dataset: RestrictedAnalysisDataset,
) -> dict[str, Any]:
    groups = compute_class_groups(np.asarray(manager.canonical_class_counts, dtype=np.int64))
    return {
        "task_identifier": TASK3E_TASK_IDENTIFIER,
        "schema_version": TASK3E_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root),
        "input_artifacts": {
            name: {
                "path": _repository_relative(Path(details["path"]), project_root),
                "sha256": str(details["sha256"]),
            }
            for name, details in source_files.items()
        }
        | {
            "task3c_diagnostics": {
                "path": _repository_relative(reference_diagnostics, project_root),
                "sha256": _sha256_file(reference_diagnostics),
            }
        },
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
        "expert_order": list(EXPERT_ORDER),
        "number_of_classes": NUM_CLASSES,
        "canonical_training_index_sha256": manager.canonical_training_index_sha256,
        "candidate_grid": {
            "generation": "all non-negative integer compositions of four units across four experts",
            "unit_total": 4,
            "denominator": 4,
            "allowed_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "ordering": "lexicographic integer-unit tuple in the frozen expert order",
            "number_of_candidates": 35,
        },
        "metric_definitions": {
            "ordinary_accuracy": "fraction of samples with predicted class equal to the label",
            "balanced_accuracy": "mean recall over all 100 classes",
            "head_accuracy": "macro recall over canonical classes with training count >= 100",
            "medium_accuracy": "macro recall over canonical classes with 20 <= training count < 100",
            "tail_accuracy": "macro recall over canonical classes with training count < 20",
            "class_group_source": "scripts.base_trainer.compute_class_groups",
        },
        "class_group_definitions": {
            "source": "scripts.base_trainer.compute_class_groups",
            "head": [int(value) for value in groups["head"].tolist()],
            "medium": [int(value) for value in groups["medium"].tolist()],
            "tail": [int(value) for value in groups["tail"].tolist()],
        },
        "uniform_baseline_tolerance": UNIFORM_BASELINE_TOLERANCE,
        "scientific_limitations": [
            "All fixed-weight results are exploratory, label-dependent development results on inner folds 1–3.",
            "Inner fold 0 is excluded from every new Task 3E-A metric calculation; earlier Task 3C diagnostics did use it descriptively.",
            "The reserved outer-evaluation population and the CIFAR-100 test set are not accessed.",
            "The fixed grid does not establish generalization, adaptive routing, or a final expert-weight choice.",
        ],
    }


def build_results_payload(
    analysis: Mapping[str, Any],
    baseline_verification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_identifier": TASK3E_TASK_IDENTIFIER,
        "schema_version": TASK3E_RESULTS_SCHEMA_VERSION,
        "analysis_partition": {
            "outer_fold": OUTER_FOLD,
            "inner_fold_ids": list(PERMITTED_ANALYSIS_INNER_FOLDS),
            "sample_count": ANALYSIS_SAMPLE_COUNT,
            "outer_evaluation_excluded": True,
            "router_selection_inner_fold_excluded": True,
        },
        "candidate_count": int(analysis["candidate_count"]),
        "uniform_baseline": analysis["uniform_baseline"],
        "uniform_baseline_verification": dict(baseline_verification),
        "candidates": list(analysis["candidates"]),
        "joint_improvement_candidate_ids": list(analysis["joint_improvement_candidate_ids"]),
        "pareto_frontier_candidate_ids": list(analysis["pareto_frontier_candidate_ids"]),
        "tradeoff_categories": dict(analysis["tradeoff_categories"]),
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.4f}%"


def _pp(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):+.4f} pp"


def _format_weights(weights: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):.2f}" for value in weights) + ")"


def render_summary(
    experiment_config: Mapping[str, Any],
    results: Mapping[str, Any],
) -> str:
    rows = results["candidates"]
    baseline_check = results["uniform_baseline_verification"]
    lines = [
        "# Task 3E-A — Fixed-Weight Ensemble Feasibility Study",
        "",
        "Exploratory analysis of the predefined 35 global fixed-weight logit ensembles.",
        "Every result below uses only the outer-fold-0 router-fit partition (inner folds 1–3); it is not an independently validated generalization result.",
        "",
        "## Provenance and protocol",
        "",
        f"- Source commit: `{experiment_config['source_git_commit']}`",
        f"- Dataset: {experiment_config['dataset']}, IR={int(experiment_config['imbalance_ratio'])}; canonical population={experiment_config['canonical_training_population_size']}",
        f"- Analyzed population: {experiment_config['analyzed_sample_count']} samples; inner folds {experiment_config['permitted_analysis_inner_folds']}; outer fold {experiment_config['outer_fold']}",
        f"- Expert order: `{', '.join(experiment_config['expert_order'])}`",
        "- Logits: original aligned OOF logits, without normalization or calibration",
        "",
        "## Uniform baseline reproduction",
        "",
        f"Stored Task 3C primary-partition values were compared with tolerance `{baseline_check['tolerance']}`. Match: **{baseline_check['matches']}**.",
        "",
        "| Metric | Stored Task 3C | Task 3E-A | Absolute difference |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("ordinary_accuracy", "Ordinary accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("head_accuracy", "Head accuracy"),
        ("medium_accuracy", "Medium accuracy"),
        ("tail_accuracy", "Tail accuracy"),
    ):
        lines.append(
            f"| {label} | {_pct(baseline_check['stored_task3c_metrics'][key])} | "
            f"{_pct(baseline_check['task3e_metrics'][key])} | "
            f"{baseline_check['absolute_differences'][key]:.3e} |"
        )

    lines.extend(
        [
            "",
            "## All 35 fixed-weight candidates",
            "",
            "Deltas are percentage-point changes from the uniform row. Candidate IDs follow lexicographic integer-unit ordering and are stable across runs.",
            "",
            "| ID | Weights (CE, LAL, BS, Mixup) | Accuracy | BA | Head | Medium | Tail | ΔBA | ΔHead | ΔMedium | ΔTail | Both BA+Tail? | Pareto? |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['candidate_id']} | {_format_weights(row['weight_vector'])} | {_pct(row['ordinary_accuracy'])} | "
            f"{_pct(row['balanced_accuracy'])} | {_pct(row['head_accuracy'])} | {_pct(row['medium_accuracy'])} | "
            f"{_pct(row['tail_accuracy'])} | {_pp(row['delta_ba'])} | {_pp(row['delta_head'])} | "
            f"{_pp(row['delta_medium'])} | {_pp(row['delta_tail'])} | "
            f"{'yes' if row['joint_improvement'] else 'no'} | {'yes' if row['pareto_frontier'] else 'no'} |"
        )

    categories = results["tradeoff_categories"]
    lines.extend(["", "## Joint improvement and BA–Tail trade-offs", ""])
    lines.append(
        f"- Candidates improving both BA and Tail: **{len(results['joint_improvement_candidate_ids'])}** "
        f"({', '.join(results['joint_improvement_candidate_ids']) or 'none'})."
    )
    for key, label in (
        ("improves_ba_reduces_tail", "improve BA while reducing Tail"),
        ("improves_tail_reduces_ba", "improve Tail while reducing BA"),
        ("reduces_ba_and_tail", "reduce both BA and Tail"),
        ("no_strict_change_or_mixed", "have no strict change or a zero-delta boundary"),
    ):
        lines.append(f"- Candidates that {label}: {', '.join(categories[key]) or 'none'}.")
    joint_rows = [row for row in rows if row["joint_improvement"]]
    if joint_rows:
        lines.append(
            "- Head/Medium deltas for the joint-improvement candidates: "
            + "; ".join(
                f"{row['candidate_id']} ({_pp(row['delta_head'])} Head, {_pp(row['delta_medium'])} Medium)"
                for row in joint_rows
            )
            + "."
        )

    lines.extend(["", "## Observed Pareto frontier", ""])
    lines.append(
        "This is the complete non-dominated frontier among the 35 predefined grid points only; it is not the frontier of all convex weight vectors. Exact BA–Tail ties are retained deterministically."
    )
    lines.extend(
        [
            "",
            "| ID | Weights | BA | Tail | Head | Medium | ΔBA | ΔTail |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    frontier_ids = set(results["pareto_frontier_candidate_ids"])
    for row in rows:
        if row["candidate_id"] in frontier_ids:
            lines.append(
                f"| {row['candidate_id']} | {_format_weights(row['weight_vector'])} | {_pct(row['balanced_accuracy'])} | "
                f"{_pct(row['tail_accuracy'])} | {_pct(row['head_accuracy'])} | {_pct(row['medium_accuracy'])} | "
                f"{_pp(row['delta_ba'])} | {_pp(row['delta_tail'])} |"
            )

    lines.extend(
        [
            "",
            "## Limitations and next-step implication",
            "",
            "- Inner fold 0 was excluded completely from new Task 3E-A calculations. Earlier Task 3C exploratory diagnostics did inspect it, so the router-selection partition cannot be described as untouched for all prior research decisions.",
            "- The reserved outer-evaluation population and the original CIFAR-100 test set were not accessed. No checkpoint, method, or final weight choice should be selected from this report alone.",
            "- The one-third three-expert vector `(0, 1/3, 1/3, 1/3)` is not an allowed grid point and was not silently added as a 36th candidate.",
            "- The next soft-mixture feasibility experiment must remain a separate exploratory analysis; these fixed global weights do not establish the value of adaptive per-sample weighting.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_texts_once(files: Mapping[Path, str]) -> None:
    write_texts_once(files, error_type=Task3EError)


def run_task3e_fixed(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    reference_diagnostics: str | Path | None = None,
    output_directory: str | Path = "artifacts/oof/task3e_fixed_feasibility",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Execute Task 3E-A and write its three required artifacts."""
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
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    analyzer = FixedWeightAnalyzer(dataset, np.asarray(manager.canonical_class_counts, dtype=np.int64))
    analysis = analyzer.analyze()
    stored_metrics = _load_stored_uniform_metrics(reference_path)
    baseline_verification = verify_uniform_baseline(
        analysis["uniform_baseline"],
        stored_metrics,
    )
    experiment_config = build_experiment_config(
        project_root=project_root_path,
        source_files=source_files,
        reference_diagnostics=reference_path,
        manager=manager,
        dataset=dataset,
    )
    results = build_results_payload(analysis, baseline_verification)
    summary = render_summary(experiment_config, results)
    output_files = {
        output_path / "experiment_config.json": serialize_json(experiment_config),
        output_path / "fixed_weight_results.json": serialize_json(results),
        output_path / "summary.md": summary,
    }
    _write_texts_once(output_files)
    return {
        "experiment_config": output_path / "experiment_config.json",
        "fixed_weight_results": output_path / "fixed_weight_results.json",
        "summary": output_path / "summary.md",
    }


__all__ = [
    "ANALYSIS_SAMPLE_COUNT",
    "EXPERT_ORDER",
    "FixedWeightAnalyzer",
    "FixedWeightCandidate",
    "PERMITTED_ANALYSIS_INNER_FOLDS",
    "RestrictedAnalysisDataset",
    "TASK3E_RESULTS_SCHEMA_VERSION",
    "TASK3E_SCHEMA_VERSION",
    "Task3EError",
    "build_experiment_config",
    "build_results_payload",
    "generate_fixed_weight_candidates",
    "load_restricted_analysis_dataset",
    "pareto_frontier",
    "run_task3e_fixed",
    "verify_uniform_baseline",
]
