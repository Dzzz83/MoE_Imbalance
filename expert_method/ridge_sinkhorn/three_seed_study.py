"""Array-only nested OOF study engine for the three-seed Ridge/Sinkhorn study.

The module consumes validated prediction arrays and never opens datasets,
checkpoints, or outer labels during fitting and prediction. Artifact access and
CLI concerns live in the companion runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from scripts.analysis import ArtifactReader, ImmutableArtifactWriter
from scripts.ridge_sinkhorn_model import fit_residual_ridge
from scripts.ridge_sinkhorn_oracle import (
    FIXED_007_ANCHOR,
    compute_oracle_targets,
    residuals_to_weights,
    smooth_positive_kernel,
)
from scripts.ridge_sinkhorn_ot import apply_log_bias, fit_frozen_dual_prices
from scripts.task3f_ridge import (
    EXPERT_ORDER,
    FIXED_REFERENCE_UNITS,
    _weighted_probability_predictions,
    classification_metrics,
    combine_weighted_logits,
    compute_contribution_targets,
    compute_sample_weights,
    extract_features,
    fit_ridge_router,
    scores_to_weights,
)


STUDY_ID = "ridge_sinkhorn_3seed_v1"
SCHEMA_VERSION = "ridge_sinkhorn_3seed_study.v1"
ANCHOR = tuple(float(value) for value in FIXED_007_ANCHOR)
METHOD_IDS = (
    "contribution_ridge",
    "contribution_ridge_sinkhorn",
    "residual_ridge",
    "residual_ridge_sinkhorn",
    "selective_residual_ridge_sinkhorn",
)
CONTROL_IDS = (
    "uniform_logit", "uniform_probability", "fixed_006", "fixed_007", "fixed_010",
    "fixed_011", "residual_anchor", "prior_only_control",
)
STUDY_PREDICTION_IDS = METHOD_IDS + CONTROL_IDS
METRIC_NAMES = (
    "balanced_accuracy", "head_accuracy", "medium_accuracy", "tail_accuracy", "ordinary_accuracy"
)


class StudyError(ValueError):
    """Raised when study arrays, locks, or artifacts violate their contract."""


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StudyError("study value is not finite JSON data") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode("ascii") + repr(array.shape).encode("ascii") + array.tobytes()).hexdigest()


def _environment_versions() -> dict[str, Any]:
    """Return the runtime and numerical-library versions used for study work."""
    packages = ("numpy", "scipy", "scikit-learn", "torch")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(), "packages": versions}


def _membership_hash(sample_ids: np.ndarray) -> str:
    canonical = np.asarray(sorted(int(value) for value in sample_ids), dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise StudyError(f"{name} must be real numeric data")
    if not np.isfinite(array).all():
        raise StudyError(f"{name} contains non-finite values")
    if ndim is not None and array.ndim != ndim:
        raise StudyError(f"{name} must have {ndim} dimensions")
    return array.astype(np.float64, copy=False)


def _validate_simplex(weights: Any, *, rows: int | None = None) -> np.ndarray:
    array = _finite_array(weights, name="weights", ndim=2)
    if array.shape[1] != 4 or (rows is not None and len(array) != rows):
        raise StudyError("weights must have shape (N, 4)")
    if len(array) == 0 or np.any(array < 0.0) or not np.allclose(
        array.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise StudyError("weights must be nonnegative and row-stochastic")
    return array


@dataclass(frozen=True)
class StudyConfig:
    """Frozen study protocol, grids, priors, metrics, and bootstrap settings."""

    seeds: tuple[int, ...] = (78, 88, 1034)
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    inner_folds: tuple[int, ...] = (0, 1, 2, 3)
    expert_order: tuple[str, ...] = EXPERT_ORDER
    contribution_alphas: tuple[float, ...] = (0.1, 10.0, 1000.0)
    gammas: tuple[float, ...] = (0.0, 1.0)
    contribution_temperatures: tuple[float, ...] = (1.0, 2.0)
    contribution_shrinkages: tuple[float, ...] = (0.5, 0.75, 1.0)
    residual_penalties: tuple[float, ...] = (0.1, 1.0, 10.0)
    residual_alphas: tuple[float, ...] = (0.1, 10.0, 1000.0)
    residual_scales: tuple[float, ...] = (0.5, 0.75, 1.0)
    sinkhorn_rhos: tuple[float, ...] = (0.1, 1.0, 10.0)
    selective_taus: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    priors: tuple[tuple[str, tuple[float, ...]], ...] = field(default_factory=tuple)
    metric_names: tuple[str, ...] = METRIC_NAMES
    bootstrap_seed: int = 20260924
    bootstrap_replicates: int = 10_000
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.seeds != (78, 88, 1034) or self.outer_folds != (0, 1, 2, 3, 4):
            raise StudyError("the frozen seed or outer-fold inventory was changed")
        if self.inner_folds != (0, 1, 2, 3) or tuple(self.expert_order) != EXPERT_ORDER:
            raise StudyError("the frozen inner folds or expert ordering was changed")
        if self.schema_version != SCHEMA_VERSION:
            raise StudyError("unsupported study configuration schema")
        if tuple(self.metric_names) != METRIC_NAMES:
            raise StudyError("the frozen metric inventory was changed")
        if self.bootstrap_seed != 20260924 or self.bootstrap_replicates != 10_000:
            raise StudyError("the frozen hierarchical bootstrap rule was changed")
        expected = _frozen_priors()
        if not self.priors:
            object.__setattr__(self, "priors", expected)
        elif self.priors != expected:
            raise StudyError("the frozen Sinkhorn prior grid was changed")
        for field_name in (
            "contribution_alphas", "gammas", "contribution_temperatures",
            "contribution_shrinkages", "residual_penalties", "residual_alphas",
            "residual_scales", "sinkhorn_rhos", "selective_taus",
        ):
            values = tuple(getattr(self, field_name))
            if not values or any(not np.isfinite(float(value)) for value in values):
                raise StudyError(f"invalid frozen grid: {field_name}")
        expected_grids = {
            "contribution_alphas": (0.1, 10.0, 1000.0), "gammas": (0.0, 1.0),
            "contribution_temperatures": (1.0, 2.0), "contribution_shrinkages": (0.5, 0.75, 1.0),
            "residual_penalties": (0.1, 1.0, 10.0), "residual_alphas": (0.1, 10.0, 1000.0),
            "residual_scales": (0.5, 0.75, 1.0), "sinkhorn_rhos": (0.1, 1.0, 10.0),
            "selective_taus": (0.1, 0.25, 0.5, 1.0),
        }
        for name, exact in expected_grids.items():
            if tuple(getattr(self, name)) != exact:
                raise StudyError(f"the frozen {name} grid was changed")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable protocol snapshot."""
        return {
            "study_id": STUDY_ID,
            "schema_version": self.schema_version,
            "seeds": list(self.seeds), "outer_folds": list(self.outer_folds),
            "inner_folds": list(self.inner_folds), "expert_order": list(self.expert_order),
            "grids": {
                "contribution_alpha": list(self.contribution_alphas), "gamma": list(self.gammas),
                "contribution_temperature": list(self.contribution_temperatures),
                "contribution_shrinkage": list(self.contribution_shrinkages),
                "residual_penalty": list(self.residual_penalties), "residual_alpha": list(self.residual_alphas),
                "residual_scale": list(self.residual_scales), "sinkhorn_rho": list(self.sinkhorn_rhos),
                "selective_tau": list(self.selective_taus),
            },
            "priors": {name: list(values) for name, values in self.priors},
            "metrics": list(self.metric_names),
            "bootstrap": {"kind": "stratified_class_then_sample_paired", "seed": self.bootstrap_seed,
                          "replicates": self.bootstrap_replicates},
            "outer_population_note": (
                "retrospective nested-OOF evidence; the population and some inner-fold history influenced earlier development"
            ),
        }

    @property
    def sha256(self) -> str:
        """Hash the canonical resolved protocol configuration."""
        return _sha256_text(_canonical_json(self.to_dict()))


def _frozen_priors() -> tuple[tuple[str, tuple[float, ...]], ...]:
    uniform = np.full(4, 0.25, dtype=np.float64)
    priors: list[tuple[str, tuple[float, ...]]] = [("uniform", tuple(uniform.tolist()))]
    for name, units in FIXED_REFERENCE_UNITS.items():
        fixed = np.asarray(units, dtype=np.float64) / 4.0
        smoothed = 0.95 * fixed + 0.05 * uniform
        priors.append((name, tuple(float(value) for value in smoothed)))
    return tuple(priors)


@dataclass(frozen=True)
class ExpertRunSpec:
    """Identity of one expert/seed/outer/inner training job."""

    study_id: str
    stage: str
    expert_name: str
    training_seed: int
    outer_fold_id: int
    inner_fold_id: int | None
    job_id: str

    def __post_init__(self) -> None:
        if self.study_id != STUDY_ID or self.stage not in {"inner", "outer"}:
            raise StudyError("expert run spec targets the wrong study or stage")
        if self.expert_name not in EXPERT_ORDER or self.training_seed not in (78, 88, 1034):
            raise StudyError("expert run spec has an unknown expert or seed")
        if self.outer_fold_id not in range(5):
            raise StudyError("expert run spec has an invalid outer fold")
        expected_inner = self.inner_fold_id in range(4) if self.stage == "inner" else self.inner_fold_id is None
        if not expected_inner:
            raise StudyError("expert run spec has an invalid inner-fold marker")
        if not self.job_id:
            raise StudyError("expert run spec requires a stable job ID")


@dataclass(frozen=True)
class DatasetProvenance:
    """Validated source and membership record attached to labeled router data."""

    role: str
    training_seed: int
    outer_fold_id: int
    included_inner_folds: tuple[int, ...]
    sample_membership_sha256: str
    source_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.role not in {"inner-training", "inner-validation", "inner-oof"}:
            raise StudyError("dataset provenance has an invalid role")
        if self.training_seed not in (78, 88, 1034) or self.outer_fold_id not in range(5):
            raise StudyError("dataset provenance has an invalid seed or outer fold")
        if any(fold not in range(4) for fold in self.included_inner_folds):
            raise StudyError("dataset provenance has an invalid inner fold")
        if len(self.sample_membership_sha256) != 64:
            raise StudyError("dataset membership hash is malformed")
        for _name, digest in self.source_hashes:
            if len(digest) != 64:
                raise StudyError("dataset source hash is malformed")


def _aligned_labeled_arrays(
    logits: Any, labels: Any, sample_ids: Any, inner_fold_ids: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logit_array = _finite_array(logits, name="logits", ndim=3)
    ids = np.asarray(sample_ids)
    targets = np.asarray(labels)
    folds = np.asarray(inner_fold_ids)
    if len(logit_array) == 0 or logit_array.shape[1] != 4 or logit_array.shape[2] < 2:
        raise StudyError("logits must have shape (N, 4, C) with N>0 and C>=2")
    if ids.ndim != 1 or targets.shape != ids.shape or folds.shape != ids.shape or len(ids) != len(logit_array):
        raise StudyError("logits, labels, sample IDs, and inner folds are misaligned")
    if not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != len(ids) or np.any(ids < 0):
        raise StudyError("sample IDs must be unique nonnegative integers")
    if not np.issubdtype(targets.dtype, np.integer) or np.any(targets < 0) or np.any(targets >= logit_array.shape[2]):
        raise StudyError("labels must be integer class IDs within the logit range")
    if not np.issubdtype(folds.dtype, np.integer) or np.any(~np.isin(folds, (0, 1, 2, 3))):
        raise StudyError("inner fold IDs must be integer values from 0 through 3")
    return (
        _readonly(logit_array, dtype=np.float64), _readonly(targets, dtype=np.int64),
        _readonly(ids, dtype=np.int64), _readonly(folds, dtype=np.int64),
    )


@dataclass(frozen=True)
class InnerOOFDataset:
    """Four-fold inner OOF logits assembled only from validated run artifacts."""

    logits: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    inner_fold_ids: np.ndarray
    training_seed: int
    outer_fold_id: int
    source_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        logits, labels, ids, folds = _aligned_labeled_arrays(
            self.logits, self.labels, self.sample_ids, self.inner_fold_ids
        )
        if tuple(sorted(np.unique(folds).tolist())) != (0, 1, 2, 3):
            raise StudyError("inner OOF dataset must contain all four held-out folds")
        if self.training_seed not in (78, 88, 1034) or self.outer_fold_id not in range(5):
            raise StudyError("inner OOF dataset has an invalid seed or outer fold")
        if not self.source_hashes:
            raise StudyError("inner OOF dataset has no validated prediction source hashes")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "inner_fold_ids", folds)
        object.__setattr__(self, "source_hashes", tuple(sorted(self.source_hashes)))

    @property
    def provenance(self) -> DatasetProvenance:
        """Return source provenance for the complete inner OOF population."""
        return DatasetProvenance(
            "inner-oof", self.training_seed, self.outer_fold_id, (0, 1, 2, 3),
            _membership_hash(self.sample_ids), self.source_hashes,
        )


@dataclass(frozen=True)
class RouterTrainingDataset:
    """Aligned labeled rows with an explicit inner-training role."""

    logits: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    inner_fold_ids: np.ndarray
    provenance: DatasetProvenance

    def __post_init__(self) -> None:
        logits, labels, ids, folds = _aligned_labeled_arrays(
            self.logits, self.labels, self.sample_ids, self.inner_fold_ids
        )
        if self.provenance.role != "inner-training" or tuple(sorted(np.unique(folds).tolist())) != self.provenance.included_inner_folds:
            raise StudyError("router fit accepts only its explicitly validated inner-training role")
        if _membership_hash(ids) != self.provenance.sample_membership_sha256:
            raise StudyError("router training membership disagrees with its provenance")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "inner_fold_ids", folds)


@dataclass(frozen=True)
class RouterValidationDataset:
    """Labeled inner validation rows for metric calculation only."""

    logits: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    inner_fold_id: int
    provenance: DatasetProvenance

    def __post_init__(self) -> None:
        logit_array = _finite_array(self.logits, name="validation logits", ndim=3)
        ids = np.asarray(self.sample_ids)
        labels = np.asarray(self.labels)
        if logit_array.shape[1] != 4 or logit_array.shape[2] < 2 or len(logit_array) == 0:
            raise StudyError("validation logits must have shape (N, 4, C)")
        if ids.shape != (len(logit_array),) or labels.shape != ids.shape:
            raise StudyError("validation labels, IDs, and logits are misaligned")
        if not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != len(ids):
            raise StudyError("validation sample IDs must be unique integers")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= logit_array.shape[2]):
            raise StudyError("validation labels are outside the logit class range")
        if self.inner_fold_id not in range(4) or self.provenance.role != "inner-validation":
            raise StudyError("validation rows require an inner-validation provenance role")
        if self.provenance.included_inner_folds != (self.inner_fold_id,):
            raise StudyError("validation provenance names a different inner fold")
        if _membership_hash(ids) != self.provenance.sample_membership_sha256:
            raise StudyError("validation membership disagrees with its provenance")
        object.__setattr__(self, "logits", _readonly(logit_array, dtype=np.float64))
        object.__setattr__(self, "labels", _readonly(labels, dtype=np.int64))
        object.__setattr__(self, "sample_ids", _readonly(ids, dtype=np.int64))


@dataclass(frozen=True)
class InferenceBatch:
    """Label-free inference data; fold IDs and class groups are intentionally absent."""

    logits: np.ndarray
    sample_ids: np.ndarray

    def __post_init__(self) -> None:
        logits = _finite_array(self.logits, name="inference logits", ndim=3)
        ids = np.asarray(self.sample_ids)
        if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1] != 4 or logits.shape[2] < 2:
            raise StudyError("inference logits must have shape (N, 4, C)")
        if ids.shape != (len(logits),) or not np.issubdtype(ids.dtype, np.integer):
            raise StudyError("inference sample IDs must be an aligned integer vector")
        if len(np.unique(ids)) != len(ids):
            raise StudyError("inference sample IDs contain duplicates")
        object.__setattr__(self, "logits", _readonly(logits, dtype=np.float64))
        object.__setattr__(self, "sample_ids", _readonly(ids, dtype=np.int64))


@dataclass(frozen=True)
class MethodSpec:
    """One primary method's router/allocator composition and parameter names."""

    method_id: str
    router_template: str
    allocator_template: str
    hyperparameter_names: tuple[str, ...]


METHOD_REGISTRY: tuple[MethodSpec, ...] = (
    MethodSpec("contribution_ridge", "contribution_ridge", "identity", ("alpha", "gamma", "temperature", "shrinkage")),
    MethodSpec("contribution_ridge_sinkhorn", "contribution_ridge", "frozen_price_sinkhorn", ("alpha", "gamma", "temperature", "shrinkage", "prior", "rho")),
    MethodSpec("residual_ridge", "residual_ridge", "identity", ("penalty", "alpha", "gamma", "scale")),
    MethodSpec("residual_ridge_sinkhorn", "residual_ridge", "frozen_price_sinkhorn", ("penalty", "alpha", "gamma", "scale", "prior", "rho")),
    MethodSpec("selective_residual_ridge_sinkhorn", "residual_ridge", "selective_blend", ("penalty", "alpha", "gamma", "scale", "prior", "rho", "tau")),
)
if tuple(spec.method_id for spec in METHOD_REGISTRY) != METHOD_IDS:
    raise AssertionError("primary method registry order changed")


@dataclass(frozen=True)
class MethodSelection:
    """Selected settings and pooled cross-fit evidence for one method."""

    method_id: str
    configuration: tuple[tuple[str, Any], ...]
    metrics: tuple[tuple[str, float], ...]
    maximin_delta: float
    candidate_id: str
    sinkhorn_diagnostics: tuple[Mapping[str, Any], ...] = ()

    def config(self) -> dict[str, Any]:
        return dict(self.configuration)

    def metric_dict(self) -> dict[str, float]:
        return dict(self.metrics)


@dataclass(frozen=True)
class InnerFoldMembership:
    """Validation membership and its three-fold router fit complement."""

    inner_fold_id: int
    validation_sample_ids: tuple[int, ...]
    validation_membership_sha256: str
    router_training_membership_sha256: str
    router_training_sample_count: int


@dataclass(frozen=True)
class FoldLock:
    """Immutable fitted fold decision, with complete source/membership hashes."""

    training_seed: int
    outer_fold_id: int
    selected: tuple[MethodSelection, ...]
    control_selections: tuple[MethodSelection, ...]
    inner_fold_memberships: tuple[InnerFoldMembership, ...]
    fitted_states: tuple[tuple[str, Mapping[str, Any]], ...]
    fitted_state_sha256s: tuple[tuple[str, str], ...]
    training_membership_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    plan_sha256: str
    config_sha256: str
    source_commit: str
    lock_sha256: str
    schema_version: str = "ridge_sinkhorn_3seed_fold_lock.v1"

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "outer_fold_id": self.outer_fold_id,
            "selected": [
                {"method_id": item.method_id, "configuration": dict(item.configuration),
                 "metrics": dict(item.metrics), "maximin_delta": item.maximin_delta,
                 "candidate_id": item.candidate_id,
                 "sinkhorn_diagnostics": list(item.sinkhorn_diagnostics)} for item in self.selected
            ],
            "control_selections": [
                {"method_id": item.method_id, "configuration": dict(item.configuration),
                 "metrics": dict(item.metrics), "maximin_delta": item.maximin_delta,
                 "candidate_id": item.candidate_id}
                for item in self.control_selections
            ],
            "inner_fold_memberships": [
                {"inner_fold_id": item.inner_fold_id,
                 "validation_sample_ids": list(item.validation_sample_ids),
                 "validation_membership_sha256": item.validation_membership_sha256,
                 "router_training_membership_sha256": item.router_training_membership_sha256,
                 "router_training_sample_count": item.router_training_sample_count}
                for item in self.inner_fold_memberships
            ],
            "fitted_states": {name: dict(state) for name, state in self.fitted_states},
            "fitted_state_sha256s": dict(self.fitted_state_sha256s),
            "training_membership_sha256": self.training_membership_sha256,
            "source_hashes": dict(self.source_hashes),
            "plan_sha256": self.plan_sha256,
            "config_sha256": self.config_sha256,
            "source_commit": self.source_commit,
        }
        if include_digest:
            data["lock_sha256"] = self.lock_sha256
        return data

    def validate(self, config: StudyConfig) -> None:
        """Check schema, completeness, nested hashes, and config linkage."""
        if self.schema_version != "ridge_sinkhorn_3seed_fold_lock.v1":
            raise StudyError("unsupported fold-lock schema")
        if self.training_seed not in config.seeds or self.outer_fold_id not in config.outer_folds:
            raise StudyError("fold lock is outside the frozen inventory")
        if tuple(item.method_id for item in self.selected) != METHOD_IDS:
            raise StudyError("fold lock is missing or reorders a primary method")
        if tuple(item.method_id for item in self.control_selections) != ("prior_only_control",):
            raise StudyError("fold lock is missing the selected prior-only control")
        if tuple(item.inner_fold_id for item in self.inner_fold_memberships) != (0, 1, 2, 3):
            raise StudyError("fold lock is missing inner cross-fit membership records")
        validation_ids: list[int] = []
        per_fold_ids: dict[int, tuple[int, ...]] = {}
        for item in self.inner_fold_memberships:
            ids = tuple(int(value) for value in item.validation_sample_ids)
            if not ids or len(ids) != len(set(ids)) or any(value < 0 for value in ids):
                raise StudyError("fold lock has malformed validation sample IDs")
            if item.validation_membership_sha256 != _membership_hash(np.asarray(ids, dtype=np.int64)):
                raise StudyError("fold-lock validation membership hash differs from sample IDs")
            per_fold_ids[item.inner_fold_id] = ids
            validation_ids.extend(ids)
        if len(validation_ids) != len(set(validation_ids)):
            raise StudyError("fold-lock inner validation memberships overlap")
        if _membership_hash(np.asarray(validation_ids, dtype=np.int64)) != self.training_membership_sha256:
            raise StudyError("inner validation union differs from full refit training membership")
        for fold_id, item in enumerate(self.inner_fold_memberships):
            complement = [sample_id for other_fold, ids in per_fold_ids.items() if other_fold != fold_id for sample_id in ids]
            if item.router_training_sample_count != len(complement):
                raise StudyError("fold-lock router-training sample count differs from its complement")
            if item.router_training_membership_sha256 != _membership_hash(np.asarray(complement, dtype=np.int64)):
                raise StudyError("fold-lock router-training membership hash differs from its complement")
        if self.config_sha256 != config.sha256 or len(self.plan_sha256) != 64 or len(self.source_commit) not in {40, 64}:
            raise StudyError("fold lock protocol/config hash is incompatible")
        try:
            int(self.source_commit, 16)
        except ValueError as exc:
            raise StudyError("fold lock source commit is not a hexadecimal Git object ID") from exc
        if _sha256_text(_canonical_json(self.payload(include_digest=False))) != self.lock_sha256:
            raise StudyError("fold-lock content hash does not match")
        states = dict(self.fitted_states)
        required = {"contribution", "residual", "contribution_prices", "residual_prices", "prior_only"}
        if set(states) != required:
            raise StudyError("fold lock has an incomplete fitted parameter set")
        state_hashes = dict(self.fitted_state_sha256s)
        if set(state_hashes) != required:
            raise StudyError("fold lock is missing fitted parameter hashes")
        for value in states.values():
            if not isinstance(value, Mapping):
                raise StudyError("fold lock fitted state is malformed")
        for name, state in states.items():
            if state_hashes[name] != _sha256_text(_canonical_json(state)):
                raise StudyError(f"fold lock fitted parameter hash differs for {name}")


@runtime_checkable
class FittedRouter(Protocol):
    """A fitted router whose prediction seam accepts inference-only batches."""

    def predict_weights(self, batch: InferenceBatch) -> np.ndarray:
        """Return finite row-stochastic expert weights for inference rows."""


@runtime_checkable
class RouterStrategy(Protocol):
    """A supervised strategy constrained to the inner-training dataset type."""

    def fit(self, training: RouterTrainingDataset, parameters: Mapping[str, Any]) -> FittedRouter:
        """Fit targets and parameters using only explicitly marked training rows."""


@runtime_checkable
class FittedAllocator(Protocol):
    """A fitted allocation map over unlabeled weight matrices."""

    def transform(self, weights: np.ndarray) -> np.ndarray:
        """Transform a weight matrix without labels or fold metadata."""


@runtime_checkable
class AllocationStrategy(Protocol):
    """A weight-only allocation strategy."""

    def fit(self, weights: np.ndarray, auxiliary_weights: np.ndarray | None = None) -> FittedAllocator:
        """Fit from router weights only."""


@dataclass(frozen=True)
class FittedContributionRouter:
    """Task 3F contribution Ridge score model plus soft-weight controls."""

    fit: Any
    temperature: float
    shrinkage: float
    num_classes: int

    def predict_weights(self, batch: InferenceBatch) -> np.ndarray:
        if batch.logits.shape[2] != self.num_classes:
            raise StudyError("inference class dimension differs from contribution fit")
        return _validate_simplex(scores_to_weights(
            self.fit.predict_scores(batch.logits), self.temperature, self.shrinkage
        ), rows=len(batch.logits))

    def state_dict(self) -> dict[str, Any]:
        """Export deterministic model parameters without pickle serialization."""
        return {
            "kind": "contribution_ridge",
            "feature_set": self.fit.feature_set,
            "alpha": float(self.fit.alpha), "gamma": float(self.fit.gamma),
            "temperature": float(self.temperature), "shrinkage": float(self.shrinkage),
            "num_classes": int(self.num_classes),
            "scaler_mean": np.asarray(self.fit.scaler.mean_, dtype=np.float64).tolist(),
            "scaler_scale": np.asarray(self.fit.scaler.scale_, dtype=np.float64).tolist(),
            "coef": np.asarray(self.fit.model.coef_, dtype=np.float64).tolist(),
            "intercept": np.asarray(self.fit.model.intercept_, dtype=np.float64).tolist(),
        }


@dataclass(frozen=True)
class FittedResidualRouter:
    """Task 3F residual Ridge model projected around fixed_007."""

    fit: Any
    penalty: float
    scale: float
    num_classes: int
    oracle_diagnostics: Mapping[str, Any]

    def predict_weights(self, batch: InferenceBatch) -> np.ndarray:
        if batch.logits.shape[2] != self.num_classes:
            raise StudyError("inference class dimension differs from residual fit")
        return _validate_simplex(residuals_to_weights(
            self.fit.predict_residual(batch.logits), ANCHOR, self.scale
        ), rows=len(batch.logits))

    def state_dict(self) -> dict[str, Any]:
        """Export deterministic model parameters without pickle serialization."""
        diagnostics = self.oracle_diagnostics
        compact_diagnostics = {
            key: diagnostics[key] for key in (
                "solver", "scipy_version", "penalty", "margin", "ftol", "max_iterations",
                "row_count", "all_succeeded", "exact_anchor_rows", "max_primal_residual",
                "max_objective_residual",
            ) if key in diagnostics
        }
        return {
            "kind": "residual_ridge", "feature_set": self.fit.feature_set,
            "alpha": float(self.fit.alpha), "gamma": float(self.fit.gamma),
            "penalty": float(self.penalty), "scale": float(self.scale),
            "num_classes": int(self.num_classes), "anchor": list(ANCHOR),
            "scaler_mean": np.asarray(self.fit.scaler.mean_, dtype=np.float64).tolist(),
            "scaler_scale": np.asarray(self.fit.scaler.scale_, dtype=np.float64).tolist(),
            "coef": np.asarray(self.fit.model.coef_, dtype=np.float64).tolist(),
            "intercept": np.asarray(self.fit.model.intercept_, dtype=np.float64).tolist(),
            "oracle_diagnostics": compact_diagnostics,
        }


class ContributionRidgeStrategy:
    """Fit Task 3F-A contribution targets then map predicted scores to weights."""

    def fit(self, training: RouterTrainingDataset, parameters: Mapping[str, Any]) -> FittedContributionRouter:
        required = {"alpha", "gamma", "temperature", "shrinkage"}
        if set(parameters) != required:
            raise StudyError("contribution router parameters do not match the frozen template")
        fitted = fit_ridge_router(
            training.logits, training.labels, "confidence_only",
            float(parameters["alpha"]), float(parameters["gamma"]),
        )
        return FittedContributionRouter(
            fitted, float(parameters["temperature"]), float(parameters["shrinkage"]),
            int(training.logits.shape[2]),
        )


class ResidualRidgeStrategy:
    """Fit Task 3F hinge-oracle residual targets around the fixed_007 anchor."""

    def fit(self, training: RouterTrainingDataset, parameters: Mapping[str, Any]) -> FittedResidualRouter:
        required = {"penalty", "alpha", "gamma", "scale"}
        if set(parameters) != required:
            raise StudyError("residual router parameters do not match the frozen template")
        penalty, alpha, gamma, scale = (float(parameters[key]) for key in ("penalty", "alpha", "gamma", "scale"))
        targets = compute_oracle_targets(training.logits, training.labels, ANCHOR, penalty)
        sample_weights = compute_sample_weights(training.labels, gamma)
        fitted = fit_residual_ridge(
            training.logits, targets.residuals, sample_weights,
            feature_set="confidence_only", alpha=alpha, gamma=gamma,
        )
        return FittedResidualRouter(
            fitted, penalty, scale, int(training.logits.shape[2]), targets.diagnostics,
        )


class IdentityAllocationStrategy:
    """Keep the router's row-stochastic weights unchanged."""

    def fit(self, weights: np.ndarray, auxiliary_weights: np.ndarray | None = None) -> "FittedIdentityAllocator":
        _validate_simplex(weights)
        if auxiliary_weights is not None:
            raise StudyError("identity allocation does not accept auxiliary weights")
        return FittedIdentityAllocator()


@dataclass(frozen=True)
class FittedIdentityAllocator:
    """Identity allocation over one four-column weight matrix."""

    def transform(self, weights: np.ndarray) -> np.ndarray:
        return _readonly(_validate_simplex(weights))


class FrozenPriceSinkhornStrategy:
    """Fit global relaxed-Sinkhorn prices on training weights, then freeze them."""

    def __init__(self, rho: float, q: Sequence[float]) -> None:
        self.rho = float(rho)
        self.q = tuple(float(value) for value in q)

    def fit(self, weights: np.ndarray, auxiliary_weights: np.ndarray | None = None) -> "FittedFrozenPriceAllocator":
        if auxiliary_weights is not None:
            raise StudyError("frozen-price Sinkhorn accepts one weight matrix")
        smoothed = smooth_positive_kernel(_validate_simplex(weights))
        prices = fit_frozen_dual_prices(smoothed, self.rho, self.q)
        if not prices.diagnostics["converged"]:
            raise StudyError("frozen-price Sinkhorn did not converge")
        return FittedFrozenPriceAllocator(self.rho, self.q, prices.log_prices, dict(prices.diagnostics), len(smoothed))


@dataclass(frozen=True)
class FittedFrozenPriceAllocator:
    """Row-independent frozen dual prices and their convergence diagnostics."""

    rho: float
    q: tuple[float, ...]
    log_prices: np.ndarray
    diagnostics: Mapping[str, Any]
    fit_sample_count: int

    def __post_init__(self) -> None:
        prices = _finite_array(self.log_prices, name="frozen dual prices", ndim=1)
        if prices.shape != (4,) or self.fit_sample_count < 1:
            raise StudyError("frozen price allocator parameters are malformed")
        object.__setattr__(self, "log_prices", _readonly(prices))

    def transform(self, weights: np.ndarray) -> np.ndarray:
        smoothed = smooth_positive_kernel(_validate_simplex(weights))
        return _validate_simplex(apply_log_bias(smoothed, self.log_prices), rows=len(smoothed))

    def state_dict(self) -> dict[str, Any]:
        """Return serializable prices, fit count, and solver diagnostics."""
        return {
            "rho": self.rho, "q": list(self.q), "log_prices": self.log_prices.tolist(),
            "diagnostics": dict(self.diagnostics), "fit_sample_count": self.fit_sample_count,
        }


class SelectiveBlendAllocationStrategy:
    """Blend residual Ridge and frozen Sinkhorn weights by anchor distance."""

    def __init__(self, tau: float) -> None:
        self.tau = float(tau)

    def fit(self, weights: np.ndarray, auxiliary_weights: np.ndarray | None = None) -> "FittedSelectiveBlendAllocator":
        if auxiliary_weights is None:
            raise StudyError("selective blending requires Ridge and Sinkhorn weight matrices")
        first = _validate_simplex(weights)
        second = _validate_simplex(auxiliary_weights, rows=len(first))
        return FittedSelectiveBlendAllocator(self.tau, np.asarray(ANCHOR), len(first))


@dataclass(frozen=True)
class FittedSelectiveBlendAllocator:
    """Selective blend using only two Nx4 weight blocks at inference."""

    tau: float
    anchor: np.ndarray
    fit_sample_count: int

    def __post_init__(self) -> None:
        anchor = _finite_array(self.anchor, name="selective anchor", ndim=1)
        if anchor.shape != (4,) or not np.isclose(anchor.sum(), 1.0) or self.tau <= 0.0:
            raise StudyError("selective blend parameters are malformed")
        object.__setattr__(self, "anchor", _readonly(anchor))

    def transform(self, weights: np.ndarray) -> np.ndarray:
        packed = _finite_array(weights, name="selective blend input", ndim=2)
        if packed.shape[1] != 8 or len(packed) == 0:
            raise StudyError("selective blend input must concatenate two Nx4 matrices")
        ridge = _validate_simplex(packed[:, :4])
        sinkhorn = _validate_simplex(packed[:, 4:])
        distance = np.abs(ridge - self.anchor[None, :]).sum(axis=1)
        gate = np.minimum(1.0, distance / self.tau)[:, None]
        blended = (1.0 - gate) * ridge + gate * sinkhorn
        return _validate_simplex(blended, rows=len(ridge))


@dataclass(frozen=True)
class SerializedContributionRouter:
    """Inference-only contribution router reconstructed from a locked JSON state."""

    feature_set: str
    temperature: float
    shrinkage: float
    num_classes: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    def predict_weights(self, batch: InferenceBatch) -> np.ndarray:
        if batch.logits.shape[2] != self.num_classes:
            raise StudyError("inference class dimension differs from locked contribution fit")
        features = extract_features(batch.logits, self.feature_set)
        standardized = (features - self.scaler_mean) / self.scaler_scale
        scores = standardized @ self.coef.T + self.intercept
        return _validate_simplex(scores_to_weights(scores, self.temperature, self.shrinkage), rows=len(features))


@dataclass(frozen=True)
class SerializedResidualRouter:
    """Inference-only residual router reconstructed from a locked JSON state."""

    feature_set: str
    scale: float
    num_classes: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    def predict_weights(self, batch: InferenceBatch) -> np.ndarray:
        if batch.logits.shape[2] != self.num_classes:
            raise StudyError("inference class dimension differs from locked residual fit")
        features = extract_features(batch.logits, self.feature_set)
        standardized = (features - self.scaler_mean) / self.scaler_scale
        residuals = standardized @ self.coef.T + self.intercept
        return _validate_simplex(residuals_to_weights(residuals, ANCHOR, self.scale), rows=len(features))


def router_from_state(state: Mapping[str, Any]) -> FittedRouter:
    """Reconstruct a fitted router from an immutable fold-lock parameter state."""
    kind = state.get("kind")
    if kind not in {"contribution_ridge", "residual_ridge"}:
        raise StudyError("locked router state has an unsupported kind")
    feature_set = str(state.get("feature_set", ""))
    if feature_set != "confidence_only":
        raise StudyError("locked router state uses an unsupported feature set")
    mean = _finite_array(state.get("scaler_mean"), name="locked scaler mean", ndim=1)
    scale = _finite_array(state.get("scaler_scale"), name="locked scaler scale", ndim=1)
    coef = _finite_array(state.get("coef"), name="locked Ridge coefficients", ndim=2)
    intercept = _finite_array(state.get("intercept"), name="locked Ridge intercept", ndim=1)
    if mean.shape != (4,) or scale.shape != (4,) or np.any(scale <= 0.0):
        raise StudyError("locked feature scaler is malformed")
    if coef.shape != (4, 4) or intercept.shape != (4,):
        raise StudyError("locked Ridge parameters have unexpected dimensions")
    num_classes = int(state.get("num_classes", 0))
    if num_classes < 2:
        raise StudyError("locked router class dimension is malformed")
    arrays = tuple(_readonly(item) for item in (mean, scale, coef, intercept))
    if kind == "contribution_ridge":
        return SerializedContributionRouter(
            feature_set, float(state["temperature"]), float(state["shrinkage"]),
            num_classes, *arrays,
        )
    return SerializedResidualRouter(
        feature_set, float(state["scale"]), num_classes, *arrays,
    )


@dataclass(frozen=True)
class FoldEvaluation:
    """One locked fold's predictions, weights, metrics, and source hashes."""

    training_seed: int
    outer_fold_id: int
    sample_ids: np.ndarray
    labels: np.ndarray
    predictions: tuple[tuple[str, np.ndarray], ...]
    weights: tuple[tuple[str, np.ndarray], ...]
    metrics: tuple[tuple[str, Mapping[str, Any]], ...]
    selected_configurations: tuple[tuple[str, Mapping[str, Any]], ...]
    sinkhorn_diagnostics: tuple[tuple[str, Mapping[str, Any]], ...]
    source_hashes: tuple[tuple[str, str], ...]
    lock_sha256: str

    def __post_init__(self) -> None:
        ids = np.asarray(self.sample_ids)
        labels = np.asarray(self.labels)
        if (
            ids.ndim != 1 or labels.shape != ids.shape
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(labels.dtype, np.integer)
            or len(np.unique(ids)) != len(ids)
        ):
            raise StudyError("fold-evaluation labels and sample IDs are malformed")
        prediction_map = dict(self.predictions)
        weight_map = dict(self.weights)
        if set(prediction_map) != set(weight_map) or set(prediction_map) != set(STUDY_PREDICTION_IDS):
            raise StudyError("fold-evaluation predictions/weights do not match the frozen method/control set")
        for name, prediction in prediction_map.items():
            values = np.asarray(prediction)
            if values.shape != ids.shape or not np.issubdtype(values.dtype, np.integer):
                raise StudyError(f"fold-evaluation predictions are malformed for {name}")
        for name, weights in weight_map.items():
            _validate_simplex(weights, rows=len(ids))
        object.__setattr__(self, "sample_ids", _readonly(ids, dtype=np.int64))
        object.__setattr__(self, "labels", _readonly(labels, dtype=np.int64))
        object.__setattr__(self, "predictions", tuple(
            (name, _readonly(prediction_map[name], dtype=np.int64)) for name in sorted(prediction_map)
        ))
        object.__setattr__(self, "weights", tuple(
            (name, _readonly(weight_map[name], dtype=np.float64)) for name in sorted(weight_map)
        ))
        object.__setattr__(self, "selected_configurations", tuple(sorted(self.selected_configurations)))
        object.__setattr__(self, "sinkhorn_diagnostics", tuple(sorted(self.sinkhorn_diagnostics)))


@dataclass(frozen=True)
class StudyResult:
    """Fold, seed, aggregate, and paired-comparison results from saved predictions."""

    fold_results: tuple[Mapping[str, Any], ...]
    seed_results: Mapping[str, Any]
    aggregate_results: Mapping[str, Any]
    paired_comparisons: Mapping[str, Any]
    bootstrap_intervals: Mapping[str, Any]
    success_by_seed: Mapping[str, Any]
    success_by_method: Mapping[str, bool]
    schema_version: str = "ridge_sinkhorn_3seed_result.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return machine-readable immutable study results."""
        return {
            "schema_version": self.schema_version,
            "study_id": STUDY_ID,
            "interpretation": "retrospective nested-OOF evidence, not independent confirmation",
            "fold_results": list(self.fold_results),
            "seed_results": dict(self.seed_results),
            "aggregate_results": dict(self.aggregate_results),
            "paired_comparisons": dict(self.paired_comparisons),
            "bootstrap_intervals": dict(self.bootstrap_intervals),
            "success_by_seed": dict(self.success_by_seed),
            "success_by_method": dict(self.success_by_method),
        }


class StudyEvaluator:
    """Predict with locked inference-only states, then score outer labels once."""

    def __init__(self, config: StudyConfig | None = None) -> None:
        self.config = config or StudyConfig()

    def predict_locked_methods(
        self, lock: FoldLock, batch: InferenceBatch,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return primary and control predictions/weights without access to labels."""
        lock.validate(self.config)
        selected = {item.method_id: item.config() for item in lock.selected}
        states = dict(lock.fitted_states)
        contribution_state = states["contribution"]
        residual_state = states["residual"]
        for state, config, keys in (
            (contribution_state, selected["contribution_ridge"], ("alpha", "gamma", "temperature", "shrinkage")),
            (residual_state, selected["residual_ridge"], ("penalty", "alpha", "gamma", "scale")),
        ):
            if any(float(state[key]) != float(config[key]) for key in keys):
                raise StudyError("locked router parameters disagree with the selected configuration")
        contribution_router = router_from_state(contribution_state)
        residual_router = router_from_state(residual_state)
        contribution_weights = contribution_router.predict_weights(batch)
        residual_weights = residual_router.predict_weights(batch)

        prior_by_name = dict(self.config.priors)
        contribution_config = selected["contribution_ridge_sinkhorn"]
        residual_config = selected["residual_ridge_sinkhorn"]
        selective_config = selected["selective_residual_ridge_sinkhorn"]
        if any(selective_config.get(key) != residual_config.get(key) for key in ("prior", "rho")):
            raise StudyError("locked selective method changed its residual Sinkhorn settings")
        contribution_prices_state = states["contribution_prices"]
        residual_prices_state = states["residual_prices"]
        for price_state, method_config in (
            (contribution_prices_state, contribution_config), (residual_prices_state, residual_config),
        ):
            if float(price_state["rho"]) != float(method_config["rho"]):
                raise StudyError("locked Sinkhorn rho differs from its selected configuration")
            expected_prior = prior_by_name[str(method_config["prior"])]
            if not np.allclose(price_state["q"], expected_prior, rtol=0.0, atol=0.0):
                raise StudyError("locked Sinkhorn prior differs from its selected configuration")
        contribution_sinkhorn = FittedFrozenPriceAllocator(
            float(contribution_prices_state["rho"]), tuple(contribution_prices_state["q"]),
            np.asarray(contribution_prices_state["log_prices"], dtype=np.float64),
            dict(contribution_prices_state["diagnostics"]), int(contribution_prices_state["fit_sample_count"]),
        ).transform(contribution_weights)
        residual_sinkhorn = FittedFrozenPriceAllocator(
            float(residual_prices_state["rho"]), tuple(residual_prices_state["q"]),
            np.asarray(residual_prices_state["log_prices"], dtype=np.float64),
            dict(residual_prices_state["diagnostics"]), int(residual_prices_state["fit_sample_count"]),
        ).transform(residual_weights)

        tau = float(selected["selective_residual_ridge_sinkhorn"]["tau"])
        selective_allocator = SelectiveBlendAllocationStrategy(tau).fit(residual_weights, residual_sinkhorn)
        selective_weights = selective_allocator.transform(np.concatenate((residual_weights, residual_sinkhorn), axis=1))
        weights: dict[str, np.ndarray] = {
            "contribution_ridge": contribution_weights,
            "contribution_ridge_sinkhorn": contribution_sinkhorn,
            "residual_ridge": residual_weights,
            "residual_ridge_sinkhorn": residual_sinkhorn,
            "selective_residual_ridge_sinkhorn": selective_weights,
        }
        uniform = np.full(4, 0.25, dtype=np.float64)
        weights["uniform_logit"] = np.repeat(uniform[None, :], len(batch.logits), axis=0)
        weights["uniform_probability"] = weights["uniform_logit"]
        weights["residual_anchor"] = np.repeat(np.asarray(ANCHOR)[None, :], len(batch.logits), axis=0)
        for name, units in FIXED_REFERENCE_UNITS.items():
            weights[name] = np.repeat((np.asarray(units, dtype=np.float64) / 4.0)[None, :], len(batch.logits), axis=0)
        prior_state = states["prior_only"]
        prior_selection = lock.control_selections[0].config()
        if str(prior_state["prior_name"]) != str(prior_selection["prior"]):
            raise StudyError("locked prior-only weights disagree with their selected prior")
        expected_prior = prior_by_name[str(prior_state["prior_name"])]
        if not np.allclose(prior_state["weights"], expected_prior, rtol=0.0, atol=0.0):
            raise StudyError("locked prior-only vector differs from the frozen prior grid")
        weights["prior_only_control"] = np.repeat(np.asarray(expected_prior)[None, :], len(batch.logits), axis=0)
        predictions = {
            name: _prediction_from_weights(batch.logits, value) for name, value in weights.items()
        }
        predictions["uniform_probability"] = _weighted_probability_predictions(batch.logits, weights["uniform_probability"])
        for name, value in weights.items():
            _validate_simplex(value, rows=len(batch.logits))
            value.setflags(write=False)
        return predictions, weights

    def evaluate_fold(
        self,
        lock: FoldLock,
        batch: InferenceBatch,
        labels: np.ndarray,
        canonical_class_counts: np.ndarray,
        *,
        source_hashes: Sequence[tuple[str, str]],
        expected_outer_sample_ids: np.ndarray,
    ) -> FoldEvaluation:
        """Predict from labels-free input first, then calculate the outer metrics."""
        predictions, weights = self.predict_locked_methods(lock, batch)
        expected_ids = np.asarray(expected_outer_sample_ids, dtype=np.int64)
        if not np.array_equal(batch.sample_ids, expected_ids):
            raise StudyError("outer inference samples differ from this fold's canonical evaluation membership")
        targets = np.asarray(labels)
        if targets.shape != (len(batch.logits),) or not np.issubdtype(targets.dtype, np.integer):
            raise StudyError("outer evaluation labels do not align with locked predictions")
        if np.any(targets < 0) or np.any(targets >= batch.logits.shape[2]):
            raise StudyError("outer evaluation labels are outside the class range")
        counts = np.asarray(canonical_class_counts, dtype=np.int64)
        if counts.shape != (batch.logits.shape[2],):
            raise StudyError("canonical class counts do not match outer logits")
        metrics = tuple(
            (name, classification_metrics(targets, predictions[name], counts))
            for name in sorted(predictions)
        )
        selections = {item.method_id: item for item in lock.selected}
        selected_configurations = tuple(
            (method, selections[method].config()) for method in METHOD_IDS
        )
        sinkhorn_diagnostics = tuple(
            (method, {"fold_diagnostics": [dict(value) for value in selections[method].sinkhorn_diagnostics]})
            for method in ("contribution_ridge_sinkhorn", "residual_ridge_sinkhorn", "selective_residual_ridge_sinkhorn")
        )
        return FoldEvaluation(
            lock.training_seed, lock.outer_fold_id, batch.sample_ids, targets,
            tuple(predictions.items()), tuple(weights.items()), metrics,
            selected_configurations, sinkhorn_diagnostics,
            tuple(sorted(source_hashes)), lock.lock_sha256,
        )

    def aggregate(
        self,
        evaluations: Sequence[FoldEvaluation],
        canonical_sample_ids: np.ndarray,
        canonical_class_counts: np.ndarray,
        expected_outer_sample_ids: Mapping[int, np.ndarray],
    ) -> StudyResult:
        """Recompute per-seed results and paired intervals from saved predictions."""
        expected_ids = np.asarray(canonical_sample_ids, dtype=np.int64)
        counts = np.asarray(canonical_class_counts, dtype=np.int64)
        if set(expected_outer_sample_ids) != set(self.config.outer_folds):
            raise StudyError("expected outer memberships do not cover exactly five folds")
        if len(evaluations) != len(self.config.seeds) * len(self.config.outer_folds):
            raise StudyError("cannot report an incomplete 15-fold result matrix")
        by_seed: dict[int, list[FoldEvaluation]] = {seed: [] for seed in self.config.seeds}
        seen_fold_keys: set[tuple[int, int]] = set()
        fold_payload: list[dict[str, Any]] = []
        for result in evaluations:
            key = (result.training_seed, result.outer_fold_id)
            if key in seen_fold_keys or result.training_seed not in by_seed or result.outer_fold_id not in self.config.outer_folds:
                raise StudyError("duplicate or out-of-inventory fold prediction")
            seen_fold_keys.add(key)
            expected_fold_ids = np.asarray(expected_outer_sample_ids[result.outer_fold_id], dtype=np.int64)
            if not np.array_equal(result.sample_ids, expected_fold_ids):
                raise StudyError("saved predictions do not match their exact outer-fold membership")
            by_seed[result.training_seed].append(result)
            recomputed_fold_metrics = {
                name: classification_metrics(result.labels, prediction, counts)
                for name, prediction in result.predictions
            }
            if set(dict(result.metrics)) != set(STUDY_PREDICTION_IDS) or _canonical_json(recomputed_fold_metrics) != _canonical_json(dict(result.metrics)):
                raise StudyError("saved fold metrics do not match metrics recomputed from predictions and labels")
            fold_mass = {
                name: {expert: float(value) for expert, value in zip(EXPERT_ORDER, weights.mean(axis=0))}
                for name, weights in result.weights
            }
            fold_payload.append({
                "training_seed": result.training_seed,
                "outer_fold_id": result.outer_fold_id,
                "sample_count": len(result.sample_ids),
                "lock_sha256": result.lock_sha256,
                "source_hashes": dict(result.source_hashes),
                "metrics": recomputed_fold_metrics,
                "selected_configurations": {name: dict(value) for name, value in result.selected_configurations},
                "sinkhorn_diagnostics": {name: dict(value) for name, value in result.sinkhorn_diagnostics},
                "mean_expert_mass": fold_mass,
            })
        if seen_fold_keys != {(seed, fold) for seed in self.config.seeds for fold in self.config.outer_folds}:
            raise StudyError("fold prediction matrix is missing a seed/outer pair")

        methods = tuple(dict(evaluations[0].predictions))
        if set(methods) != set(STUDY_PREDICTION_IDS):
            raise StudyError("evaluation matrix has an unsupported method/control inventory")
        seed_metrics: dict[str, Any] = {}
        seed_expert_mass: dict[str, Any] = {}
        seed_predictions: dict[int, dict[str, np.ndarray]] = {}
        reference_labels: np.ndarray | None = None
        for seed in self.config.seeds:
            folds = sorted(by_seed[seed], key=lambda value: value.outer_fold_id)
            ids = np.concatenate([item.sample_ids for item in folds])
            labels = np.concatenate([item.labels for item in folds])
            if len(ids) != len(expected_ids) or not np.array_equal(np.sort(ids), np.sort(expected_ids)):
                raise StudyError(f"seed {seed} does not have exactly one canonical prediction per sample")
            order = np.argsort(ids, kind="stable")
            if len(np.unique(ids)) != len(ids):
                raise StudyError(f"seed {seed} contains duplicate held-out sample predictions")
            labels = labels[order]
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise StudyError("outer label artifacts disagree across training seeds")
            predictions: dict[str, np.ndarray] = {}
            for name in methods:
                combined = np.concatenate([dict(item.predictions)[name] for item in folds])[order]
                predictions[name] = combined
            seed_predictions[seed] = predictions
            seed_metrics[ str(seed) ] = {
                name: classification_metrics(labels, predictions[name], counts)
                for name in methods
            }
            seed_expert_mass[str(seed)] = {
                name: {
                    expert: float(np.concatenate([dict(item.weights)[name] for item in folds], axis=0).mean(axis=0)[axis])
                    for axis, expert in enumerate(EXPERT_ORDER)
                }
                for name in methods
            }
        if reference_labels is None:
            raise StudyError("no seed predictions were aggregated")

        metric_names = self.config.metric_names
        aggregate_metrics = {
            method: {
                metric: {
                    "mean": float(np.mean([seed_metrics[str(seed)][method][metric] for seed in self.config.seeds])),
                    "std": float(np.std([seed_metrics[str(seed)][method][metric] for seed in self.config.seeds], ddof=0)),
                }
                for metric in metric_names
            }
            for method in methods
        }
        for method in methods:
            aggregate_metrics[method]["mean_expert_mass"] = {
                expert: {
                    "mean": float(np.mean([seed_expert_mass[str(seed)][method][expert] for seed in self.config.seeds])),
                    "std": float(np.std([seed_expert_mass[str(seed)][method][expert] for seed in self.config.seeds], ddof=0)),
                }
                for expert in EXPERT_ORDER
            }
        paired: dict[str, Any] = {}
        adaptive_not_dominated: dict[str, dict[str, bool]] = {}
        fixed_names = ("fixed_006", "fixed_007", "fixed_010", "fixed_011")
        for seed in self.config.seeds:
            baseline = seed_metrics[str(seed)]["uniform_logit"]
            seed_deltas: dict[str, Any] = {}
            adaptive_not_dominated[str(seed)] = {}
            for method in METHOD_IDS:
                references = ["uniform_logit", "uniform_probability", *fixed_names]
                if method == "contribution_ridge_sinkhorn":
                    references.append("contribution_ridge")
                elif method in {"residual_ridge_sinkhorn", "selective_residual_ridge_sinkhorn"}:
                    references.append("residual_ridge")
                seed_deltas[method] = {
                    reference: {
                        metric: float(seed_metrics[str(seed)][method][metric] - seed_metrics[str(seed)][reference][metric])
                        for metric in metric_names
                    }
                    for reference in tuple(dict.fromkeys(references))
                    if reference != method
                }
                adaptive_not_dominated[str(seed)][method] = all(
                    not (
                        seed_metrics[str(seed)][reference]["balanced_accuracy"] >= seed_metrics[str(seed)][method]["balanced_accuracy"]
                        and seed_metrics[str(seed)][reference]["tail_accuracy"] >= seed_metrics[str(seed)][method]["tail_accuracy"]
                    ) for reference in fixed_names
                )
            paired[str(seed)] = seed_deltas
        aggregate_deltas: dict[str, Any] = {}
        for method in METHOD_IDS:
            references = tuple(paired[str(self.config.seeds[0])][method])
            aggregate_deltas[method] = {
                reference: {
                    metric: {
                        "mean": float(np.mean([
                            paired[str(seed)][method][reference][metric] for seed in self.config.seeds
                        ])),
                        "std": float(np.std([
                            paired[str(seed)][method][reference][metric] for seed in self.config.seeds
                        ], ddof=0)),
                    }
                    for metric in metric_names
                }
                for reference in references
            }
        success_by_seed = {
            str(seed): {
                **{
                    method: (
                        seed_metrics[str(seed)][method]["balanced_accuracy"] > seed_metrics[str(seed)]["uniform_logit"]["balanced_accuracy"]
                        and seed_metrics[str(seed)][method]["tail_accuracy"] > seed_metrics[str(seed)]["uniform_logit"]["tail_accuracy"]
                    ) for method in METHOD_IDS
                },
                "all_primary_methods": all(
                    seed_metrics[str(seed)][method]["balanced_accuracy"] > seed_metrics[str(seed)]["uniform_logit"]["balanced_accuracy"]
                    and seed_metrics[str(seed)][method]["tail_accuracy"] > seed_metrics[str(seed)]["uniform_logit"]["tail_accuracy"]
                    for method in METHOD_IDS
                ),
            } for seed in self.config.seeds
        }
        success_by_method = {
            method: all(bool(success_by_seed[str(seed)][method]) for seed in self.config.seeds)
            for method in METHOD_IDS
        }
        intervals = self._hierarchical_bootstrap(seed_predictions, reference_labels, counts)
        return StudyResult(
            tuple(sorted(fold_payload, key=lambda value: (value["training_seed"], value["outer_fold_id"]))),
            {seed: {"metrics": seed_metrics[seed], "mean_expert_mass": seed_expert_mass[seed]}
             for seed in seed_metrics}, aggregate_metrics,
            {"by_seed": paired, "mean_std_across_seeds": aggregate_deltas,
             "fixed_mixture_non_domination": adaptive_not_dominated},
            intervals, success_by_seed, success_by_method,
        )

    def _hierarchical_bootstrap(
        self,
        seed_predictions: Mapping[int, Mapping[str, np.ndarray]],
        labels: np.ndarray,
        class_counts: np.ndarray,
    ) -> dict[str, Any]:
        """Compute paired class-then-sample intervals conditional on saved models."""
        from scripts.base_trainer import compute_class_groups

        names = tuple(sorted(next(iter(seed_predictions.values()))))
        methods = tuple(name for name in names if name != "uniform_logit")
        pairs: set[tuple[str, str]] = set()
        for method in METHOD_IDS:
            pairs.add((method, "uniform_logit"))
            pairs.add((method, "uniform_probability"))
            pairs.update((method, name) for name in ("fixed_006", "fixed_007", "fixed_010", "fixed_011"))
            matched = "contribution_ridge" if method == "contribution_ridge_sinkhorn" else (
                "residual_ridge" if method in {"residual_ridge_sinkhorn", "selective_residual_ridge_sinkhorn"} else None
            )
            if matched is not None:
                pairs.add((method, matched))
        pair_list = tuple(sorted((left, right) for left, right in pairs if left != right))
        correct = np.stack([
            np.stack([(seed_predictions[seed][name] == labels).astype(np.float64) for seed in sorted(seed_predictions)])
            for name in names
        ])  # method, seed, sample
        groups = compute_class_groups(class_counts)
        class_rows = [np.flatnonzero(labels == class_id) for class_id in range(len(class_counts))]
        if any(rows.size == 0 for rows in class_rows):
            raise StudyError("saved outer labels do not cover every canonical class")
        rng = np.random.default_rng(self.config.bootstrap_seed)
        samples = {pair: np.empty((self.config.bootstrap_replicates, 3), dtype=np.float64) for pair in pair_list}
        name_axis = {name: index for index, name in enumerate(names)}
        total_classes = len(class_counts)
        for replicate in range(self.config.bootstrap_replicates):
            group_values: dict[str, list[np.ndarray]] = {name: [] for name in groups}
            class_sample_acc: list[np.ndarray] = []
            class_sample_counts: list[int] = []
            for group_name, classes in groups.items():
                selected_classes = rng.choice(classes, size=len(classes), replace=True)
                for class_id in selected_classes:
                    rows = class_rows[int(class_id)]
                    sampled_rows = rng.choice(rows, size=len(rows), replace=True)
                    group_values[group_name].append(correct[:, :, sampled_rows].mean(axis=(1, 2)))
                    class_sample_acc.append(correct[:, :, sampled_rows].mean(axis=(1, 2)))
                    class_sample_counts.append(len(sampled_rows))
            group_means = {
                name: np.mean(np.stack(values, axis=0), axis=0)
                for name, values in group_values.items()
            }
            metric_values = np.stack((
                sum(group_means[name] * len(classes) for name, classes in groups.items()) / total_classes,
                group_means["tail"],
                np.average(np.stack(class_sample_acc, axis=0), axis=0, weights=np.asarray(class_sample_counts)),
            ), axis=1)
            for pair in pair_list:
                samples[pair][replicate] = metric_values[name_axis[pair[0]]] - metric_values[name_axis[pair[1]]]
        metric_names = tuple(
            name for name in self.config.metric_names
            if name in {"balanced_accuracy", "tail_accuracy", "ordinary_accuracy"}
        )
        return {
            f"{left}_minus_{right}": {
                metric: np.quantile(samples[(left, right)][:, axis], [0.025, 0.975]).tolist()
                for axis, metric in enumerate(metric_names)
            }
            for left, right in pair_list
        }


@dataclass(frozen=True)
class _CandidateEvaluation:
    """Internal pooled inner cross-fit evidence for one frozen candidate."""

    candidate_id: str
    parameters: Mapping[str, Any]
    metrics: Mapping[str, Any]
    weights: np.ndarray
    predictions: np.ndarray
    sinkhorn_diagnostics: tuple[Mapping[str, Any], ...] = ()


def _prediction_from_weights(logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return combine_weighted_logits(logits, weights).argmax(axis=1).astype(np.int64)


def _metric_payload(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray, class_counts: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    checked = _validate_simplex(weights, rows=len(logits))
    prediction = _prediction_from_weights(logits, checked)
    return classification_metrics(labels, prediction, class_counts), prediction


def _metric_tuple(metrics: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    return tuple((name, float(metrics[name])) for name in METRIC_NAMES)


def _candidate_id(parameters: Mapping[str, Any]) -> str:
    return _canonical_json({key: parameters[key] for key in sorted(parameters)})


def _tie_order(
    family: str, parameters: Mapping[str, Any], *, phase: str,
) -> tuple[Any, ...]:
    """Frozen regularization, routing, OT, and stable-ID tie-break fields."""
    if family == "contribution":
        common: tuple[Any, ...] = (-float(parameters["alpha"]),)
        less_routing = (float(parameters["shrinkage"]), -float(parameters["temperature"]))
        if phase == "sinkhorn":
            return (*common, *less_routing, float(parameters["rho"]), _candidate_id(parameters))
        return (*common, *less_routing, _candidate_id(parameters))
    if family == "residual":
        common = (-float(parameters["alpha"]), -float(parameters["penalty"]))
        less_routing = (float(parameters["scale"]),)
        if phase == "sinkhorn":
            return (*common, *less_routing, float(parameters["rho"]), _candidate_id(parameters))
        if phase == "selective":
            return (*common, *less_routing, float(parameters["rho"]), -float(parameters["tau"]), _candidate_id(parameters))
        return (*common, *less_routing, _candidate_id(parameters))
    return (_candidate_id(parameters),)


def _candidate_sort_key(
    candidate: _CandidateEvaluation,
    baseline: Mapping[str, Any],
    family: str,
    *,
    phase: str,
) -> tuple[Any, ...]:
    ba = float(candidate.metrics["balanced_accuracy"])
    tail = float(candidate.metrics["tail_accuracy"])
    maximin = min(
        ba - float(baseline["balanced_accuracy"]),
        tail - float(baseline["tail_accuracy"]),
    )
    return (-maximin, -ba, -tail, *_tie_order(family, candidate.parameters, phase=phase))


def _training_view(dataset: InnerOOFDataset, mask: np.ndarray) -> RouterTrainingDataset:
    ids, folds = dataset.sample_ids[mask], dataset.inner_fold_ids[mask]
    unique_folds = tuple(sorted(int(value) for value in np.unique(folds)))
    provenance = DatasetProvenance(
        "inner-training", dataset.training_seed, dataset.outer_fold_id, unique_folds,
        _membership_hash(ids), dataset.source_hashes,
    )
    return RouterTrainingDataset(dataset.logits[mask], dataset.labels[mask], ids, folds, provenance)


def _validation_view(dataset: InnerOOFDataset, fold_id: int) -> RouterValidationDataset:
    mask = dataset.inner_fold_ids == fold_id
    ids = dataset.sample_ids[mask]
    provenance = DatasetProvenance(
        "inner-validation", dataset.training_seed, dataset.outer_fold_id, (fold_id,),
        _membership_hash(ids), dataset.source_hashes,
    )
    return RouterValidationDataset(dataset.logits[mask], dataset.labels[mask], ids, fold_id, provenance)


def _router_training_from_inner(dataset: InnerOOFDataset) -> RouterTrainingDataset:
    provenance = DatasetProvenance(
        "inner-training", dataset.training_seed, dataset.outer_fold_id, (0, 1, 2, 3),
        _membership_hash(dataset.sample_ids), dataset.source_hashes,
    )
    return RouterTrainingDataset(
        dataset.logits, dataset.labels, dataset.sample_ids, dataset.inner_fold_ids, provenance,
    )


class InnerCVSelector:
    """Four-fold cross-fit candidate selection with the frozen maximin rule."""

    def __init__(self, config: StudyConfig | None = None) -> None:
        self.config = config or StudyConfig()
        self.contribution_strategy: RouterStrategy = ContributionRidgeStrategy()
        self.residual_strategy: RouterStrategy = ResidualRidgeStrategy()

    def select(
        self, dataset: InnerOOFDataset, canonical_class_counts: np.ndarray,
    ) -> tuple[tuple[MethodSelection, ...], tuple[MethodSelection, ...], dict[str, Any]]:
        """Select all method configurations from pooled four-fold predictions."""
        counts = np.asarray(canonical_class_counts, dtype=np.int64)
        if counts.shape != (dataset.logits.shape[2],) or np.any(counts <= 0):
            raise StudyError("canonical class counts do not match inner logits")
        baseline_weights = np.full((len(dataset.logits), 4), 0.25, dtype=np.float64)
        baseline_metrics, _baseline_prediction = _metric_payload(
            dataset.logits, dataset.labels, baseline_weights, counts,
        )

        contribution_candidates: list[_CandidateEvaluation] = []
        for alpha in self.config.contribution_alphas:
            for gamma in self.config.gammas:
                pooled_scores = self._crossfit_contribution_scores(dataset, alpha, gamma)
                for temperature in self.config.contribution_temperatures:
                    for shrinkage in self.config.contribution_shrinkages:
                        parameters = {
                            "alpha": alpha, "gamma": gamma,
                            "temperature": temperature, "shrinkage": shrinkage,
                        }
                        weights = scores_to_weights(pooled_scores, temperature, shrinkage)
                        metrics, predictions = _metric_payload(dataset.logits, dataset.labels, weights, counts)
                        contribution_candidates.append(_CandidateEvaluation(
                            _candidate_id(parameters), parameters, metrics, weights, predictions,
                        ))
        contribution_base = min(
            contribution_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, baseline_metrics, "contribution", phase="base"),
        )

        residual_candidates: list[_CandidateEvaluation] = []
        residual_target_cache: dict[tuple[int, float], Any] = {}
        for penalty in self.config.residual_penalties:
            for gamma in self.config.gammas:
                for alpha in self.config.residual_alphas:
                    pooled_residuals = self._crossfit_residuals(
                        dataset, penalty, alpha, gamma, residual_target_cache,
                    )
                    for scale in self.config.residual_scales:
                        parameters = {"penalty": penalty, "alpha": alpha, "gamma": gamma, "scale": scale}
                        weights = residuals_to_weights(pooled_residuals, ANCHOR, scale)
                        metrics, predictions = _metric_payload(dataset.logits, dataset.labels, weights, counts)
                        residual_candidates.append(_CandidateEvaluation(
                            _candidate_id(parameters), parameters, metrics, weights, predictions,
                        ))
        residual_base = min(
            residual_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, baseline_metrics, "residual", phase="base"),
        )

        contribution_selected = self._selection(
            "contribution_ridge", contribution_base, baseline_metrics,
        )
        contribution_sinkhorn_candidates = self._crossfit_sinkhorn_grid(
            dataset, "contribution", contribution_base.parameters, counts,
        )
        contribution_sinkhorn = min(
            contribution_sinkhorn_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, baseline_metrics, "contribution", phase="sinkhorn"),
        )
        contribution_sinkhorn_selected = self._selection(
            "contribution_ridge_sinkhorn", contribution_sinkhorn, baseline_metrics,
        )

        residual_selected = self._selection("residual_ridge", residual_base, baseline_metrics)
        residual_sinkhorn_candidates = self._crossfit_sinkhorn_grid(
            dataset, "residual", residual_base.parameters, counts,
        )
        residual_sinkhorn = min(
            residual_sinkhorn_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, baseline_metrics, "residual", phase="sinkhorn"),
        )
        residual_sinkhorn_selected = self._selection(
            "residual_ridge_sinkhorn", residual_sinkhorn, baseline_metrics,
        )

        selective_candidates: list[_CandidateEvaluation] = []
        for tau in self.config.selective_taus:
            configuration = {**dict(residual_sinkhorn.parameters), "tau": tau}
            ridge_weights = residual_base.weights
            sinkhorn_weights = residual_sinkhorn.weights
            packed = np.concatenate((ridge_weights, sinkhorn_weights), axis=1)
            allocator = SelectiveBlendAllocationStrategy(tau).fit(ridge_weights, sinkhorn_weights)
            blended = allocator.transform(packed)
            metrics, predictions = _metric_payload(dataset.logits, dataset.labels, blended, counts)
            selective_candidates.append(_CandidateEvaluation(
                _candidate_id(configuration), configuration, metrics, blended, predictions,
                residual_sinkhorn.sinkhorn_diagnostics,
            ))
        selective = min(
            selective_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, baseline_metrics, "residual", phase="selective"),
        )
        selective_selected = self._selection(
            "selective_residual_ridge_sinkhorn", selective, baseline_metrics,
        )

        prior_candidates: list[_CandidateEvaluation] = []
        for prior_name, prior in self.config.priors:
            parameters = {"prior": prior_name}
            weights = np.repeat(np.asarray(prior, dtype=np.float64)[None, :], len(dataset.logits), axis=0)
            metrics, predictions = _metric_payload(dataset.logits, dataset.labels, weights, counts)
            prior_candidates.append(_CandidateEvaluation(
                _candidate_id(parameters), parameters, metrics, weights, predictions,
            ))
        prior_only = min(
            prior_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, baseline_metrics, "control", phase="base"),
        )
        prior_selection = self._selection("prior_only_control", prior_only, baseline_metrics)

        selected = (
            contribution_selected, contribution_sinkhorn_selected,
            residual_selected, residual_sinkhorn_selected, selective_selected,
        )
        diagnostic = {
            "uniform_logit": baseline_metrics,
            "uniform_probability": classification_metrics(
                dataset.labels,
                _weighted_probability_predictions(dataset.logits, baseline_weights), counts,
            ),
            "fixed_references": self._fixed_inner_controls(dataset, counts),
            "selected_base_weights": {
                "contribution": contribution_base.weights,
                "residual": residual_base.weights,
            },
            "selected_sinkhorn_weights": {
                "contribution": contribution_sinkhorn.weights,
                "residual": residual_sinkhorn.weights,
            },
            "selected_prior_only_weights": prior_only.weights,
            "prior_only_selection": prior_selection,
            "residual_target_cache_entries": len(residual_target_cache),
        }
        return selected, (prior_selection,), diagnostic

    def _selection(
        self, method_id: str, candidate: _CandidateEvaluation, baseline: Mapping[str, Any],
    ) -> MethodSelection:
        metrics = candidate.metrics
        maximin = min(
            float(metrics["balanced_accuracy"]) - float(baseline["balanced_accuracy"]),
            float(metrics["tail_accuracy"]) - float(baseline["tail_accuracy"]),
        )
        return MethodSelection(
            method_id, tuple(sorted(candidate.parameters.items())), _metric_tuple(metrics),
            float(maximin), candidate.candidate_id, candidate.sinkhorn_diagnostics,
        )

    def _crossfit_contribution_scores(
        self, dataset: InnerOOFDataset, alpha: float, gamma: float,
    ) -> np.ndarray:
        """Fit one contribution model per fold, independent of weight-map settings."""
        scores = np.empty((len(dataset.logits), 4), dtype=np.float64)
        for fold_id in self.config.inner_folds:
            validation_mask = dataset.inner_fold_ids == fold_id
            training = _training_view(dataset, ~validation_mask)
            validation = _validation_view(dataset, fold_id)
            fitted = fit_ridge_router(
                training.logits, training.labels, "confidence_only", alpha, gamma,
            )
            scores[validation_mask] = fitted.predict_scores(validation.logits)
        if not np.isfinite(scores).all():
            raise StudyError("cross-fit contribution scores contain non-finite values")
        return _readonly(scores)

    def _crossfit_residuals(
        self,
        dataset: InnerOOFDataset,
        penalty: float,
        alpha: float,
        gamma: float,
        target_cache: dict[tuple[int, float], Any],
    ) -> np.ndarray:
        """Fit one residual model per fold/settings, reusing each oracle target."""
        residuals = np.empty((len(dataset.logits), 4), dtype=np.float64)
        for fold_id in self.config.inner_folds:
            validation_mask = dataset.inner_fold_ids == fold_id
            training = _training_view(dataset, ~validation_mask)
            validation = _validation_view(dataset, fold_id)
            targets = target_cache.get((fold_id, float(penalty)))
            if targets is None:
                targets = compute_oracle_targets(training.logits, training.labels, ANCHOR, penalty)
                target_cache[(fold_id, float(penalty))] = targets
            fitted = fit_residual_ridge(
                training.logits, targets.residuals, compute_sample_weights(training.labels, gamma),
                feature_set="confidence_only", alpha=alpha, gamma=gamma,
            )
            residuals[validation_mask] = fitted.predict_residual(validation.logits)
        if not np.isfinite(residuals).all():
            raise StudyError("cross-fit residual predictions contain non-finite values")
        return _readonly(residuals)

    def _crossfit_router_inputs(
        self, dataset: InnerOOFDataset, family: str, parameters: Mapping[str, Any],
        *, target_cache: dict[tuple[int, float], Any] | None = None,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Fit each router once per inner split and return fit/heldout kernels."""
        per_fold: list[tuple[np.ndarray, np.ndarray]] = []
        pooled = np.empty((len(dataset.logits), 4), dtype=np.float64)
        for fold_id in self.config.inner_folds:
            validation_mask = dataset.inner_fold_ids == fold_id
            training = _training_view(dataset, ~validation_mask)
            validation = _validation_view(dataset, fold_id)
            batch = InferenceBatch(validation.logits, validation.sample_ids)
            if family == "contribution":
                fitted = self.contribution_strategy.fit(training, parameters)
            elif family == "residual":
                if target_cache is not None:
                    penalty = float(parameters["penalty"])
                    target = target_cache.get((fold_id, penalty))
                    if target is None:
                        target = compute_oracle_targets(training.logits, training.labels, ANCHOR, penalty)
                        target_cache[(fold_id, penalty)] = target
                    gamma = float(parameters["gamma"])
                    fitted_model = fit_residual_ridge(
                        training.logits, target.residuals, compute_sample_weights(training.labels, gamma),
                        feature_set="confidence_only", alpha=float(parameters["alpha"]), gamma=gamma,
                    )
                    fitted = FittedResidualRouter(
                        fitted_model, penalty, float(parameters["scale"]), dataset.logits.shape[2], target.diagnostics,
                    )
                else:
                    fitted = self.residual_strategy.fit(training, parameters)
            else:
                raise StudyError(f"unknown router family {family!r}")
            train_batch = InferenceBatch(training.logits, training.sample_ids)
            train_weights = fitted.predict_weights(train_batch)
            validation_weights = fitted.predict_weights(batch)
            pooled[validation_mask] = validation_weights
            per_fold.append((train_weights, validation_weights))
        return per_fold, _readonly(_validate_simplex(pooled, rows=len(dataset.logits)))

    def _crossfit_sinkhorn_grid(
        self, dataset: InnerOOFDataset, family: str, base_parameters: Mapping[str, Any],
        class_counts: np.ndarray,
    ) -> list[_CandidateEvaluation]:
        target_cache: dict[tuple[int, float], Any] = {}
        fold_inputs, _base_pooled = self._crossfit_router_inputs(
            dataset, family, base_parameters,
            target_cache=target_cache if family == "residual" else None,
        )
        candidates: list[_CandidateEvaluation] = []
        for prior_name, prior in self.config.priors:
            for rho in self.config.sinkhorn_rhos:
                parameters = {**dict(base_parameters), "prior": prior_name, "rho": rho}
                pooled = np.empty((len(dataset.logits), 4), dtype=np.float64)
                diagnostics: list[Mapping[str, Any]] = []
                for fold_index, fold_id in enumerate(self.config.inner_folds):
                    train_weights, validation_weights = fold_inputs[fold_index]
                    allocator = FrozenPriceSinkhornStrategy(rho, prior).fit(train_weights)
                    validation_mask = dataset.inner_fold_ids == fold_id
                    pooled[validation_mask] = allocator.transform(validation_weights)
                    diagnostics.append(dict(allocator.diagnostics))
                metrics, predictions = _metric_payload(dataset.logits, dataset.labels, pooled, class_counts)
                candidates.append(_CandidateEvaluation(
                    _candidate_id(parameters), parameters, metrics, pooled, predictions, tuple(diagnostics),
                ))
        return candidates

    @staticmethod
    def _fixed_inner_controls(dataset: InnerOOFDataset, counts: np.ndarray) -> dict[str, Mapping[str, Any]]:
        controls: dict[str, Mapping[str, Any]] = {}
        uniform = np.full(4, 0.25, dtype=np.float64)
        for name, weights in (("uniform_logit", uniform),):
            _metrics, predictions = _metric_payload(
                dataset.logits, dataset.labels,
                np.repeat(weights[None, :], len(dataset.logits), axis=0), counts,
            )
            controls[name] = classification_metrics(dataset.labels, predictions, counts)
        controls["residual_anchor"] = classification_metrics(
            dataset.labels,
            _prediction_from_weights(
                dataset.logits,
                np.repeat(np.asarray(ANCHOR)[None, :], len(dataset.logits), axis=0),
            ), counts,
        )
        for name, units in FIXED_REFERENCE_UNITS.items():
            fixed = np.asarray(units, dtype=np.float64) / 4.0
            controls[name] = classification_metrics(
                dataset.labels,
                _prediction_from_weights(dataset.logits, np.repeat(fixed[None, :], len(dataset.logits), axis=0)),
                counts,
            )
        return controls


class FoldStudyRunner:
    """Select, refit, and freeze five method configurations for one outer fold."""

    def __init__(self, config: StudyConfig | None = None) -> None:
        self.config = config or StudyConfig()
        self.selector = InnerCVSelector(self.config)

    def lock(
        self,
        dataset: InnerOOFDataset,
        canonical_class_counts: np.ndarray,
        *,
        plan_sha256: str,
        source_commit: str,
    ) -> FoldLock:
        """Perform pooled inner selection and refit selected routers on all four folds."""
        if len(plan_sha256) != 64 or len(source_commit) not in {40, 64}:
            raise StudyError("plan hash or source commit has an invalid length")
        for digest in (plan_sha256, source_commit):
            try:
                int(digest, 16)
            except ValueError as exc:
                raise StudyError("plan hash/source commit is not hexadecimal") from exc
        selections, controls, selection_diagnostics = self.selector.select(dataset, canonical_class_counts)
        selected = {item.method_id: item.config() for item in selections}
        contribution_config = {
            name: selected["contribution_ridge"][name]
            for name in ("alpha", "gamma", "temperature", "shrinkage")
        }
        residual_config = {
            name: selected["residual_ridge"][name]
            for name in ("penalty", "alpha", "gamma", "scale")
        }
        if any(selected["contribution_ridge_sinkhorn"].get(key) != value for key, value in contribution_config.items()):
            raise StudyError("Contribution Ridge + Sinkhorn did not reuse the selected Ridge kernel")
        if any(selected["residual_ridge_sinkhorn"].get(key) != value for key, value in residual_config.items()):
            raise StudyError("Residual Ridge + Sinkhorn did not reuse the selected Ridge kernel")
        if any(selected["selective_residual_ridge_sinkhorn"].get(key) != value for key, value in residual_config.items()):
            raise StudyError("selective allocation did not reuse the selected residual Ridge kernel")
        for key in ("prior", "rho"):
            if selected["selective_residual_ridge_sinkhorn"].get(key) != selected["residual_ridge_sinkhorn"].get(key):
                raise StudyError("selective allocation did not reuse the selected Sinkhorn configuration")

        training = _router_training_from_inner(dataset)
        contribution = self.selector.contribution_strategy.fit(training, contribution_config)
        residual = self.selector.residual_strategy.fit(training, residual_config)
        contribution_train_weights = contribution.predict_weights(InferenceBatch(training.logits, training.sample_ids))
        residual_train_weights = residual.predict_weights(InferenceBatch(training.logits, training.sample_ids))
        prior_by_name = dict(self.config.priors)
        contribution_sink_config = selected["contribution_ridge_sinkhorn"]
        residual_sink_config = selected["residual_ridge_sinkhorn"]
        contribution_prices = FrozenPriceSinkhornStrategy(
            float(contribution_sink_config["rho"]), prior_by_name[str(contribution_sink_config["prior"])],
        ).fit(contribution_train_weights)
        residual_prices = FrozenPriceSinkhornStrategy(
            float(residual_sink_config["rho"]), prior_by_name[str(residual_sink_config["prior"])],
        ).fit(residual_train_weights)
        prior_name = str(controls[0].config()["prior"])
        states: dict[str, Mapping[str, Any]] = {
            "contribution": contribution.state_dict(),
            "residual": residual.state_dict(),
            "contribution_prices": contribution_prices.state_dict(),
            "residual_prices": residual_prices.state_dict(),
            "prior_only": {
                "prior_name": prior_name,
                "weights": list(prior_by_name[prior_name]),
                "selected_metrics": dict(controls[0].metrics),
            },
        }
        state_tuple = tuple((name, states[name]) for name in sorted(states))
        state_hashes = tuple(
            (name, _sha256_text(_canonical_json(states[name]))) for name in sorted(states)
        )
        fold_memberships = []
        for fold_id in self.config.inner_folds:
            validation_ids = tuple(sorted(
                int(value) for value in dataset.sample_ids[dataset.inner_fold_ids == fold_id]
            ))
            training_ids = tuple(
                int(value) for value in dataset.sample_ids[dataset.inner_fold_ids != fold_id]
            )
            fold_memberships.append(InnerFoldMembership(
                fold_id, validation_ids, _membership_hash(np.asarray(validation_ids, dtype=np.int64)),
                _membership_hash(np.asarray(training_ids, dtype=np.int64)), len(training_ids),
            ))
        lock = FoldLock(
            training_seed=dataset.training_seed,
            outer_fold_id=dataset.outer_fold_id,
            selected=selections,
            control_selections=controls,
            inner_fold_memberships=tuple(fold_memberships),
            fitted_states=state_tuple,
            fitted_state_sha256s=state_hashes,
            training_membership_sha256=_membership_hash(training.sample_ids),
            source_hashes=dataset.source_hashes,
            plan_sha256=plan_sha256.lower(),
            config_sha256=self.config.sha256,
            source_commit=source_commit.lower(),
            lock_sha256="",
        )
        digest = _sha256_text(_canonical_json(lock.payload(include_digest=False)))
        locked = FoldLock(
            training_seed=lock.training_seed, outer_fold_id=lock.outer_fold_id,
            selected=lock.selected, control_selections=lock.control_selections,
            inner_fold_memberships=lock.inner_fold_memberships,
            fitted_states=lock.fitted_states, fitted_state_sha256s=lock.fitted_state_sha256s,
            training_membership_sha256=lock.training_membership_sha256,
            source_hashes=lock.source_hashes, plan_sha256=lock.plan_sha256,
            config_sha256=lock.config_sha256, source_commit=lock.source_commit,
            lock_sha256=digest,
        )
        locked.validate(self.config)
        # Keep compact cross-fit control diagnostics linked to the lock via selected records.
        if not selection_diagnostics["fixed_references"]:
            raise StudyError("inner control calculations were not produced")
        return locked


def _method_selection_from_dict(payload: Mapping[str, Any]) -> MethodSelection:
    return MethodSelection(
        method_id=str(payload["method_id"]),
        configuration=tuple(sorted(dict(payload["configuration"]).items())),
        metrics=tuple(sorted((str(key), float(value)) for key, value in dict(payload["metrics"]).items())),
        maximin_delta=float(payload["maximin_delta"]),
        candidate_id=str(payload["candidate_id"]),
        sinkhorn_diagnostics=tuple(payload.get("sinkhorn_diagnostics", ())),
    )


def fold_lock_from_dict(payload: Mapping[str, Any]) -> FoldLock:
    """Load a lock payload without trusting its schema or embedded digest."""
    try:
        states = payload["fitted_states"]
        if not isinstance(states, Mapping):
            raise StudyError("fold-lock fitted states must be a mapping")
        return FoldLock(
            training_seed=int(payload["training_seed"]),
            outer_fold_id=int(payload["outer_fold_id"]),
            selected=tuple(_method_selection_from_dict(item) for item in payload["selected"]),
            control_selections=tuple(
                _method_selection_from_dict(item) for item in payload["control_selections"]
            ),
            inner_fold_memberships=tuple(
                InnerFoldMembership(
                    int(item["inner_fold_id"]),
                    tuple(int(value) for value in item["validation_sample_ids"]),
                    str(item["validation_membership_sha256"]),
                    str(item["router_training_membership_sha256"]),
                    int(item["router_training_sample_count"]),
                ) for item in payload["inner_fold_memberships"]
            ),
            fitted_states=tuple((str(key), dict(value)) for key, value in sorted(states.items())),
            fitted_state_sha256s=tuple(sorted(
                (str(key), str(value)) for key, value in dict(payload["fitted_state_sha256s"]).items()
            )),
            training_membership_sha256=str(payload["training_membership_sha256"]),
            source_hashes=tuple(sorted(
                (str(key), str(value)) for key, value in dict(payload["source_hashes"]).items()
            )),
            plan_sha256=str(payload["plan_sha256"]),
            config_sha256=str(payload["config_sha256"]),
            source_commit=str(payload["source_commit"]),
            lock_sha256=str(payload["lock_sha256"]),
            schema_version=str(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, StudyError):
            raise
        raise StudyError("fold-lock JSON is malformed") from exc


class StudyArtifactRepository:
    """Immutable study-level artifact adapter beside native OOF run artifacts."""

    def __init__(self, artifact_root: str | Path, config: StudyConfig | None = None) -> None:
        self.config = config or StudyConfig()
        self.study_root = Path(artifact_root).resolve() / STUDY_ID
        self.base_dir = self.study_root / "study_analysis"
        self.writer = ImmutableArtifactWriter(error_type=StudyError)
        self.reader = ArtifactReader(error_type=StudyError)

    @property
    def config_path(self) -> Path:
        return self.base_dir / "study_config.json"

    @property
    def study_lock_path(self) -> Path:
        return self.base_dir / "study_lock.json"

    def lock_path(self, training_seed: int, outer_fold_id: int) -> Path:
        return self.base_dir / "locks" / f"seed_{training_seed}_outer_{outer_fold_id}.json"

    def evaluation_path(self, training_seed: int, outer_fold_id: int) -> Path:
        return self.base_dir / "evaluations" / f"seed_{training_seed}_outer_{outer_fold_id}.json"

    def evaluation_arrays_path(self, training_seed: int, outer_fold_id: int) -> Path:
        return self.base_dir / "evaluations" / f"seed_{training_seed}_outer_{outer_fold_id}.npz"

    @property
    def result_path(self) -> Path:
        return self.base_dir / "results.json"

    @property
    def report_path(self) -> Path:
        return self.base_dir / "report.md"

    def write_config(self) -> Path:
        """Write the canonical frozen configuration idempotently."""
        self.writer.write_json_once(self.config_path, self.config.to_dict())
        return self.config_path

    def write_study_lock(
        self,
        *,
        plan_sha256: str,
        source_commit: str,
        job_manifest_sha256: str,
        job_inventory: Sequence[Mapping[str, Any]],
    ) -> Path:
        """Freeze source commit, plan/config hashes, and the full job inventory."""
        if len(plan_sha256) != 64 or len(job_manifest_sha256) != 64 or len(source_commit) not in {40, 64}:
            raise StudyError("study lock has malformed plan, manifest, or source-commit identity")
        try:
            int(plan_sha256, 16)
            int(job_manifest_sha256, 16)
            int(source_commit, 16)
        except ValueError as exc:
            raise StudyError("study lock plan, manifest, and source-commit hashes must be hexadecimal") from exc
        inventory = [dict(item) for item in job_inventory]
        if len(inventory) != 300 or any(not isinstance(item.get("job_id"), str) for item in inventory):
            raise StudyError("study lock requires the complete 300-job inventory")
        inventory.sort(key=lambda item: item["job_id"])
        job_ids = [item["job_id"] for item in inventory]
        if len(set(job_ids)) != 300:
            raise StudyError("study lock requires 300 unique stable expert job IDs")
        if sum(item.get("stage") == "inner" for item in inventory) != 240 or sum(
            item.get("stage") == "outer" for item in inventory
        ) != 60:
            raise StudyError("study lock inventory must contain exactly 240 inner and 60 outer jobs")
        for item in inventory:
            job_id = item["job_id"]
            expected_shard_key = f"{int.from_bytes(hashlib.sha256(job_id.encode('utf-8')).digest()[:8], 'big'):016x}"
            if (
                item.get("shard_key") != expected_shard_key
                or item.get("shard_rule") != "first_8_sha256_bytes_big_endian_mod_shard_count"
            ):
                raise StudyError(f"study lock inventory has an invalid shard assignment for {job_id}")
        payload = {
            "schema_version": "ridge_sinkhorn_3seed_study_lock.v1",
            "study_id": STUDY_ID,
            "plan_sha256": plan_sha256,
            "config_sha256": self.config.sha256,
            "source_commit": source_commit,
            "job_manifest_sha256": job_manifest_sha256,
            "job_ids": job_ids,
            "job_inventory": inventory,
            "job_inventory_sha256": _sha256_text(_canonical_json(inventory)),
            "shard_rule": "unsigned_big_endian(first_8_bytes(SHA-256(job_id))) % shard_count",
            "environment": _environment_versions(),
            "test_accessed": False,
            "outer_read_after_lock": True,
        }
        payload["study_lock_sha256"] = _sha256_text(_canonical_json(payload))
        self.write_config()
        self.writer.write_json_once(self.study_lock_path, payload)
        return self.study_lock_path

    def write_fold_lock(self, lock: FoldLock) -> Path:
        """Persist one validated immutable fold lock."""
        lock.validate(self.config)
        self.write_config()
        path = self.lock_path(lock.training_seed, lock.outer_fold_id)
        self.writer.write_json_once(path, lock.payload())
        return path

    def read_fold_lock(self, training_seed: int, outer_fold_id: int) -> FoldLock:
        """Load and fully validate one immutable fold lock."""
        payload = self.reader.read_json(self.lock_path(training_seed, outer_fold_id), name="fold lock")
        lock = fold_lock_from_dict(payload)
        lock.validate(self.config)
        if (lock.training_seed, lock.outer_fold_id) != (training_seed, outer_fold_id):
            raise StudyError("fold-lock path identity disagrees with its payload")
        return lock

    def validate_complete_lock_matrix(
        self, *, plan_sha256: str, source_commit: str,
    ) -> tuple[FoldLock, ...]:
        """Validate all 15 locks without opening any outer run or label artifact."""
        if len(plan_sha256) != 64 or len(source_commit) not in {40, 64}:
            raise StudyError("expected a plan SHA-256 and exact Git source commit")
        config_payload = self.reader.read_json(self.config_path, name="study configuration")
        if _canonical_json(config_payload) != _canonical_json(self.config.to_dict()):
            raise StudyError("saved study configuration differs from the frozen StudyConfig")
        study_lock = self.reader.read_json(self.study_lock_path, name="study lock")
        recorded_digest = study_lock.pop("study_lock_sha256", None)
        if recorded_digest != _sha256_text(_canonical_json(study_lock)):
            raise StudyError("study-lock content hash does not match")
        if (
            study_lock.get("schema_version") != "ridge_sinkhorn_3seed_study_lock.v1"
            or study_lock.get("study_id") != STUDY_ID
            or study_lock.get("plan_sha256") != plan_sha256
            or study_lock.get("config_sha256") != self.config.sha256
            or study_lock.get("source_commit") != source_commit
            or study_lock.get("test_accessed") is not False
        ):
            raise StudyError("study lock disagrees with the requested protocol identity")
        locked_inventory = study_lock.get("job_inventory")
        if not isinstance(locked_inventory, list) or len(locked_inventory) != 300:
            raise StudyError("study lock does not carry the complete 300-job inventory")
        if any(not isinstance(item, Mapping) or not isinstance(item.get("job_id"), str) for item in locked_inventory):
            raise StudyError("study-lock inventory contains malformed job entries")
        locked_inventory = sorted(locked_inventory, key=lambda item: item["job_id"])
        locked_job_ids = [item["job_id"] for item in locked_inventory]
        if len(set(locked_job_ids)) != 300 or study_lock.get("job_ids") != locked_job_ids:
            raise StudyError("study lock job IDs do not match its full unique inventory")
        if study_lock.get("job_inventory_sha256") != _sha256_text(_canonical_json(locked_inventory)):
            raise StudyError("study-lock job inventory hash does not match")
        if study_lock.get("shard_rule") != "unsigned_big_endian(first_8_bytes(SHA-256(job_id))) % shard_count":
            raise StudyError("study lock has an unsupported shard assignment rule")
        if not isinstance(study_lock.get("environment"), Mapping):
            raise StudyError("study lock is missing environment and package versions")
        matrix_manifest_path = self.study_root / "job_manifest.json"
        matrix_manifest = self.reader.read_json(matrix_manifest_path, name="matrix job manifest")
        if study_lock.get("job_manifest_sha256") != self.reader.sha256_file(
            matrix_manifest_path, description="matrix job manifest",
        ):
            raise StudyError("study lock points to a different matrix job manifest")
        manifest_inventory = matrix_manifest.get("inventory")
        if not isinstance(manifest_inventory, list) or len(manifest_inventory) != 300:
            raise StudyError("matrix job manifest does not contain the complete 300-job inventory")
        if any(not isinstance(item, Mapping) or not isinstance(item.get("job_id"), str) for item in manifest_inventory):
            raise StudyError("matrix job manifest contains malformed inventory entries")
        manifest_inventory = sorted(manifest_inventory, key=lambda item: item["job_id"])
        manifest_job_ids = [item["job_id"] for item in manifest_inventory]
        if len(set(manifest_job_ids)) != 300:
            raise StudyError("matrix job manifest contains duplicate job IDs")
        if sum(item.get("stage") == "inner" for item in manifest_inventory) != 240 or sum(
            item.get("stage") == "outer" for item in manifest_inventory
        ) != 60:
            raise StudyError("matrix job manifest must contain exactly 240 inner and 60 outer jobs")
        if (
            matrix_manifest.get("study_id") != STUDY_ID
            or matrix_manifest.get("plan_sha256") != plan_sha256
            or matrix_manifest.get("protocol_config_sha256") != self.config.sha256
            or matrix_manifest.get("source_commit") != source_commit
            or manifest_job_ids != locked_job_ids
            or manifest_inventory != locked_inventory
            or study_lock.get("shard_rule") != matrix_manifest.get("shard_rule")
        ):
            raise StudyError("study lock inventory/protocol identity differs from current matrix manifest")
        locks: list[FoldLock] = []
        for seed in self.config.seeds:
            for outer_fold in self.config.outer_folds:
                lock = self.read_fold_lock(seed, outer_fold)
                if lock.plan_sha256 != plan_sha256 or lock.source_commit != source_commit:
                    raise StudyError("fold lock points to a different plan or source commit")
                if len(lock.source_hashes) != 48 or len({name for name, _digest in lock.source_hashes}) != 48:
                    raise StudyError("fold lock must identify checkpoint, prediction, and config hashes for all 16 inner jobs")
                for name, digest in lock.source_hashes:
                    if not name or len(digest) != 64:
                        raise StudyError("fold lock has malformed inner source hashes")
                    try:
                        int(digest, 16)
                    except ValueError as exc:
                        raise StudyError("fold lock source hash is not hexadecimal") from exc
                try:
                    int(lock.training_membership_sha256, 16)
                except ValueError as exc:
                    raise StudyError("fold lock training-membership hash is invalid") from exc
                locks.append(lock)
        if len(locks) != 15 or len({(item.training_seed, item.outer_fold_id) for item in locks}) != 15:
            raise StudyError("the complete 15-fold lock matrix is missing or duplicated")
        return tuple(locks)

    def write_fold_evaluation(
        self, evaluation: FoldEvaluation, *, expected_outer_sample_ids: np.ndarray,
    ) -> tuple[Path, Path]:
        """Persist one fold's metrics and predictions as immutable JSON/NPZ."""
        lock = self.read_fold_lock(evaluation.training_seed, evaluation.outer_fold_id)
        if evaluation.lock_sha256 != lock.lock_sha256:
            raise StudyError("fold evaluation does not reference its immutable fold lock")
        expected_configurations = {item.method_id: item.config() for item in lock.selected}
        if {name: dict(value) for name, value in evaluation.selected_configurations} != expected_configurations:
            raise StudyError("fold evaluation selected settings differ from its immutable lock")
        expected_diagnostics = {
            method: {
                "fold_diagnostics": [
                    dict(value)
                    for item in lock.selected if item.method_id == method
                    for value in item.sinkhorn_diagnostics
                ]
            }
            for method in (
                "contribution_ridge_sinkhorn", "residual_ridge_sinkhorn",
                "selective_residual_ridge_sinkhorn",
            )
        }
        if {name: dict(value) for name, value in evaluation.sinkhorn_diagnostics} != expected_diagnostics:
            raise StudyError("fold evaluation Sinkhorn diagnostics differ from its immutable lock")
        expected_ids = np.asarray(expected_outer_sample_ids, dtype=np.int64)
        if not np.array_equal(evaluation.sample_ids, expected_ids):
            raise StudyError("fold evaluation does not match its exact outer membership")
        metadata = {
            "schema_version": "ridge_sinkhorn_3seed_fold_evaluation.v1",
            "training_seed": evaluation.training_seed,
            "outer_fold_id": evaluation.outer_fold_id,
            "lock_sha256": evaluation.lock_sha256,
            "sample_count": len(evaluation.sample_ids),
            "source_hashes": dict(evaluation.source_hashes),
            "metrics": {name: dict(value) for name, value in evaluation.metrics},
            "selected_configurations": {
                name: dict(value) for name, value in evaluation.selected_configurations
            },
            "sinkhorn_diagnostics": {
                name: dict(value) for name, value in evaluation.sinkhorn_diagnostics
            },
        }
        arrays = {"sample_ids": evaluation.sample_ids, "labels": evaluation.labels}
        arrays.update({f"prediction__{name}": value for name, value in evaluation.predictions})
        arrays.update({f"weights__{name}": value for name, value in evaluation.weights})
        json_path = self.evaluation_path(evaluation.training_seed, evaluation.outer_fold_id)
        npz_path = self.evaluation_arrays_path(evaluation.training_seed, evaluation.outer_fold_id)
        self.writer.write_npz_once(npz_path, arrays)
        metadata["prediction_arrays"] = {
            "path": npz_path.name,
            "size_bytes": npz_path.stat().st_size,
            "sha256": self.reader.sha256_file(npz_path, description="fold prediction arrays"),
        }
        self.writer.write_json_once(json_path, metadata)
        return json_path, npz_path

    def read_fold_evaluation(
        self, training_seed: int, outer_fold_id: int, *, expected_outer_sample_ids: np.ndarray,
    ) -> FoldEvaluation:
        """Read saved predictions only after checking their lock and array contract."""
        metadata = self.reader.read_json(
            self.evaluation_path(training_seed, outer_fold_id), name="fold evaluation",
        )
        lock = self.read_fold_lock(training_seed, outer_fold_id)
        array_path = self.evaluation_arrays_path(training_seed, outer_fold_id)
        array_record = metadata.get("prediction_arrays")
        if not isinstance(array_record, Mapping) or array_record.get("path") != array_path.name:
            raise StudyError("fold evaluation has a malformed prediction-array reference")
        if not array_path.is_file() or array_record.get("size_bytes") != array_path.stat().st_size:
            raise StudyError("fold prediction-array byte size differs from its JSON manifest")
        if array_record.get("sha256") != self.reader.sha256_file(array_path, description="fold prediction arrays"):
            raise StudyError("fold prediction-array SHA-256 differs from its JSON manifest")
        if metadata.get("training_seed") != training_seed or metadata.get("outer_fold_id") != outer_fold_id:
            raise StudyError("fold evaluation path identity disagrees with its JSON metadata")
        arrays = self.reader.read_npz(array_path)
        expected_arrays = {"sample_ids", "labels"}
        expected_arrays.update(f"prediction__{name}" for name in STUDY_PREDICTION_IDS)
        expected_arrays.update(f"weights__{name}" for name in STUDY_PREDICTION_IDS)
        if set(arrays) != expected_arrays:
            raise StudyError("saved fold prediction arrays do not match the frozen array schema")
        prediction_arrays = {
            key.removeprefix("prediction__"): value
            for key, value in arrays.items() if key.startswith("prediction__")
        }
        weight_arrays = {
            key.removeprefix("weights__"): value
            for key, value in arrays.items() if key.startswith("weights__")
        }
        if set(prediction_arrays) != set(weight_arrays):
            raise StudyError("saved fold predictions and weights have different method sets")
        if metadata.get("schema_version") != "ridge_sinkhorn_3seed_fold_evaluation.v1":
            raise StudyError("unsupported fold evaluation schema")
        if metadata.get("lock_sha256") != lock.lock_sha256:
            raise StudyError("fold evaluation is linked to a different immutable lock")
        if metadata.get("sample_count") != len(arrays["sample_ids"]):
            raise StudyError("fold evaluation sample count disagrees with its NPZ")
        expected_ids = set(STUDY_PREDICTION_IDS)
        if set(prediction_arrays) != expected_ids or set(metadata.get("metrics", {})) != expected_ids:
            raise StudyError("saved fold evaluation has a different frozen method/control set")
        if set(metadata.get("selected_configurations", {})) != set(METHOD_IDS):
            raise StudyError("saved fold evaluation lacks selected settings for all primary methods")
        if set(metadata.get("sinkhorn_diagnostics", {})) != {
            "contribution_ridge_sinkhorn", "residual_ridge_sinkhorn", "selective_residual_ridge_sinkhorn"
        }:
            raise StudyError("saved fold evaluation lacks required Sinkhorn diagnostics")
        expected_configurations = {item.method_id: item.config() for item in lock.selected}
        actual_configurations = {
            str(name): dict(value) for name, value in metadata["selected_configurations"].items()
        }
        if _canonical_json(actual_configurations) != _canonical_json(expected_configurations):
            raise StudyError("saved selected configurations differ from their immutable fold lock")
        expected_diagnostics = {
            method: {
                "fold_diagnostics": [
                    dict(value)
                    for item in lock.selected if item.method_id == method
                    for value in item.sinkhorn_diagnostics
                ]
            }
            for method in (
                "contribution_ridge_sinkhorn", "residual_ridge_sinkhorn",
                "selective_residual_ridge_sinkhorn",
            )
        }
        actual_diagnostics = {
            str(name): dict(value) for name, value in metadata["sinkhorn_diagnostics"].items()
        }
        if _canonical_json(actual_diagnostics) != _canonical_json(expected_diagnostics):
            raise StudyError("saved Sinkhorn diagnostics differ from their immutable fold lock")
        source_hashes = metadata.get("source_hashes")
        if not isinstance(source_hashes, Mapping) or any(
            not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64
            for name, digest in source_hashes.items()
        ):
            raise StudyError("saved fold evaluation has malformed source hashes")
        try:
            for digest in source_hashes.values():
                int(digest, 16)
        except ValueError as exc:
            raise StudyError("saved fold evaluation source hashes must be hexadecimal") from exc
        result = FoldEvaluation(
            training_seed, outer_fold_id, arrays["sample_ids"], arrays["labels"],
            tuple(prediction_arrays.items()), tuple(weight_arrays.items()),
            tuple((str(name), value) for name, value in metadata["metrics"].items()),
            tuple((str(name), value) for name, value in metadata["selected_configurations"].items()),
            tuple((str(name), value) for name, value in metadata["sinkhorn_diagnostics"].items()),
            tuple(sorted((str(key), str(value)) for key, value in source_hashes.items())),
            lock.lock_sha256,
        )
        expected_ids = np.asarray(expected_outer_sample_ids, dtype=np.int64)
        if not np.array_equal(result.sample_ids, expected_ids):
            raise StudyError("saved fold evaluation does not match its exact outer membership")
        return result

    def validate_complete_evaluation_matrix(
        self, *, expected_outer_sample_ids: Mapping[int, np.ndarray],
    ) -> tuple[FoldEvaluation, ...]:
        """Require exact saved fold predictions and matching immutable locks."""
        evaluations = tuple(
            self.read_fold_evaluation(
                seed, fold, expected_outer_sample_ids=expected_outer_sample_ids[fold],
            )
            for seed in self.config.seeds for fold in self.config.outer_folds
        )
        keys = {(item.training_seed, item.outer_fold_id) for item in evaluations}
        if len(evaluations) != 15 or len(keys) != 15:
            raise StudyError("complete outer evaluation matrix is missing or duplicated")
        return evaluations

    def write_result(self, result: StudyResult, markdown: str) -> tuple[Path, Path]:
        """Write machine-readable results and their derived Markdown report once."""
        if result.schema_version != "ridge_sinkhorn_3seed_result.v1":
            raise StudyError("unsupported aggregate result schema")
        self.writer.write_json_once(self.result_path, result.to_dict())
        self.writer.write_text_once(self.report_path, markdown)
        return self.result_path, self.report_path


class MarkdownReportBuilder:
    """Render a concise report solely from the immutable ``StudyResult``."""

    def __init__(self, config: StudyConfig | None = None) -> None:
        self.config = config or StudyConfig()

    def build(self, result: StudyResult) -> str:
        """Build the study report without recalculating outcomes."""
        lines = [
            "# Three-seed Ridge–Sinkhorn nested OOF study",
            "",
            "Retrospective nested-OOF evidence: this population and some inner-fold history influenced earlier development. These results are not independent confirmation.",
            "",
            "The original balanced CIFAR-100 test set was out of scope. Each seed pools five held-out outer folds; confidence intervals condition on the trained models and do not capture retraining variability or dependence from overlapping inner expert training populations.",
            "",
            "| Method | BA mean ± SD | Head mean ± SD | Medium mean ± SD | Tail mean ± SD | Sample accuracy mean ± SD |",
            "|:--|--:|--:|--:|--:|--:|",
        ]
        for method, values in result.aggregate_results.items():
            def display(metric: str) -> str:
                item = values[metric]
                return f"{item['mean']:.4f} ± {item['std']:.4f}"
            lines.append(
                f"| {method} | {display('balanced_accuracy')} | {display('head_accuracy')} | "
                f"{display('medium_accuracy')} | {display('tail_accuracy')} | {display('ordinary_accuracy')} |"
            )
        lines.extend(("", "Mean expert mass across seeds (rows are methods; columns follow the frozen expert order CE, LAL, BalancedSoftmax, Mixup):", "", "| Method | CE | LAL | BalancedSoftmax | Mixup |", "|:--|--:|--:|--:|--:|"))
        for method, values in result.aggregate_results.items():
            masses = values["mean_expert_mass"]
            lines.append("| " + method + " | " + " | ".join(
                f"{masses[expert]['mean']:.4f} ± {masses[expert]['std']:.4f}" for expert in EXPERT_ORDER
            ) + " |")
        lines.extend(("", "Per-method success requires BA and Tail accuracy to exceed uniform-logit results for every seed. Fixed mixtures are assessed separately for Pareto domination.", ""))
        for seed, success in sorted(result.success_by_seed.items()):
            lines.append(f"- Seed {seed}: " + ", ".join(
                f"{method}={bool(success[method])}" for method in METHOD_IDS
            ))
        lines.append("- All-seed success: " + ", ".join(
            f"{method}={bool(result.success_by_method[method])}" for method in METHOD_IDS
        ))
        non_dominated = result.paired_comparisons["fixed_mixture_non_domination"]
        for seed, by_method in sorted(non_dominated.items()):
            lines.append(f"- Seed {seed} fixed-mixture non-domination: " + ", ".join(
                f"{method}={bool(by_method[method])}" for method in METHOD_IDS
            ))
        lines.extend(("", "Paired deltas against uniform, fixed mixtures, and matched Ridge-only methods (mean ± SD across seeds; intervals are paired hierarchical bootstrap 95% intervals):", "", "| Method | Reference | ΔBA mean ± SD [95% CI] | ΔTail mean ± SD [95% CI] | ΔSample accuracy mean ± SD |", "|:--|:--|--:|--:|--:|"))
        paired_mean = result.paired_comparisons["mean_std_across_seeds"]
        for method, references in paired_mean.items():
            for reference, metrics in references.items():
                interval_key = f"{method}_minus_{reference}"
                interval = result.bootstrap_intervals.get(interval_key, {})
                ba = metrics["balanced_accuracy"]
                tail = metrics["tail_accuracy"]
                sample = metrics["ordinary_accuracy"]
                ba_ci = interval.get("balanced_accuracy", [float("nan"), float("nan")])
                tail_ci = interval.get("tail_accuracy", [float("nan"), float("nan")])
                lines.append(
                    f"| {method} | {reference} | {ba['mean']:.4f} ± {ba['std']:.4f} [{ba_ci[0]:.4f}, {ba_ci[1]:.4f}] | "
                    f"{tail['mean']:.4f} ± {tail['std']:.4f} [{tail_ci[0]:.4f}, {tail_ci[1]:.4f}] | "
                    f"{sample['mean']:.4f} ± {sample['std']:.4f} |"
                )
        lines.extend(("", "Selected settings and frozen-price Sinkhorn convergence diagnostics by fold:", ""))
        for fold in result.fold_results:
            lines.append(f"- Seed {fold['training_seed']}, outer fold {fold['outer_fold_id']}: selections `{_canonical_json(fold['selected_configurations'])}`")
            lines.append(f"  Sinkhorn: `{_canonical_json(fold['sinkhorn_diagnostics'])}`")
            lines.append(f"  Mean expert mass: `{_canonical_json(fold['mean_expert_mass'])}`")
        lines.extend(("", "Paired hierarchical bootstrap intervals use 10,000 stratified class-then-sample resamples with seed 20260924. They describe paired prediction uncertainty conditional on the fitted experts and routers.", ""))
        return "\n".join(lines)
