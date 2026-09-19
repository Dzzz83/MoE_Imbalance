"""Deterministic nested out-of-fold data protocol.

This module owns fold membership and provenance contracts only.  It does not
load images, train experts, fit routers, or read the CIFAR-100 test split.
The canonical population and labels are supplied by the caller, normally via
``data.protocol_splits``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = "nested_oof_manifest.v1"
FOLD_ALGORITHM = "numpy.default_rng.per_class_balanced_remainder.v1"
DEFAULT_EXPERT_ORDER = ("CE", "LAL", "BalancedSoftmax", "Mixup")


class OOFProtocolError(ValueError):
    """Base error for invalid nested-OOF configuration or provenance."""


class FoldConfigurationError(OOFProtocolError):
    """Raised when the requested folds cannot preserve class coverage."""


class FoldMembershipError(OOFProtocolError):
    """Raised when a constructed or serialized fold violates an invariant."""


class OOFArtifactValidationError(OOFProtocolError):
    """Raised when an OOF prediction record or artifact is not trustworthy."""


OOF_PREDICTION_SCHEMA_VERSION = "nested_oof_prediction.v1"


def _as_int_tuple(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    array = np.asarray(list(values))
    if array.ndim != 1:
        raise OOFProtocolError(f"{name} must be one-dimensional")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise OOFProtocolError(f"{name} must contain integer values")
    return tuple(int(value) for value in array.tolist())


def _hash_indices(indices: Iterable[int]) -> str:
    """Hash an index membership set in canonical sorted int64 form."""
    array = np.asarray(sorted(int(value) for value in indices), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OOFProtocolError(f"cannot hash canonical index artifact: {path}") from exc
    return digest.hexdigest()


def _class_counts(
    indices: Iterable[int], labels_by_index: Mapping[int, int], num_classes: int
) -> tuple[int, ...]:
    counts = np.zeros(num_classes, dtype=np.int64)
    for index in indices:
        try:
            label = labels_by_index[int(index)]
        except KeyError as exc:
            raise FoldMembershipError(
                f"sample index {index} is not in the canonical population"
            ) from exc
        if label < 0 or label >= num_classes:
            raise FoldMembershipError(
                f"sample index {index} has label {label}, outside [0, {num_classes})"
            )
        counts[label] += 1
    return tuple(int(value) for value in counts.tolist())


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise OOFProtocolError(f"value is not JSON serializable: {value!r}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise OOFArtifactValidationError(f"{name} must be a 64-character SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise OOFArtifactValidationError(f"{name} is not valid hexadecimal SHA-256") from exc
    return value.lower()


def _derived_seed(seed: int, outer_fold_id: int, stage: int) -> int:
    """Derive independent deterministic seeds without relying on call order."""
    sequence = np.random.SeedSequence([int(seed), int(outer_fold_id), int(stage)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _stratified_partition(
    indices: Sequence[int],
    labels_by_index: Mapping[int, int],
    *,
    num_classes: int,
    n_folds: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign every class to balanced, deterministic fold buckets."""
    if n_folds < 2:
        raise FoldConfigurationError(f"n_folds must be at least 2, got {n_folds}")

    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    rng = np.random.default_rng(seed)
    sorted_indices = np.asarray(sorted(int(value) for value in indices), dtype=np.int64)

    for label in range(num_classes):
        class_indices = np.asarray(
            [index for index in sorted_indices if labels_by_index[int(index)] == label],
            dtype=np.int64,
        )
        shuffled = class_indices[rng.permutation(len(class_indices))]
        # Give every fold the class quotient first, then place that class's
        # remainder in the currently smallest buckets.  This preserves
        # per-class stratification without accumulating all remainders in fold 0.
        fold_order = sorted(range(n_folds), key=lambda fold: (len(buckets[fold]), fold))
        for position, index in enumerate(shuffled.tolist()):
            buckets[fold_order[position % n_folds]].append(int(index))

    return tuple(tuple(sorted(bucket)) for bucket in buckets)


@dataclass(frozen=True)
class InnerFoldDefinition:
    """One inner held-out prediction partition and its expert-training set."""

    outer_fold_id: int
    inner_fold_id: int
    expert_training_indices: tuple[int, ...]
    prediction_indices: tuple[int, ...]
    expert_training_class_counts: tuple[int, ...]
    prediction_class_counts: tuple[int, ...]

    @property
    def expert_training_membership_hash(self) -> str:
        return _hash_indices(self.expert_training_indices)


@dataclass(frozen=True)
class RouterDevelopmentPartition:
    """Disjoint OOF rows for router fitting and hyperparameter selection."""

    fit_inner_fold_ids: tuple[int, ...]
    selection_inner_fold_ids: tuple[int, ...]
    fit_indices: tuple[int, ...]
    selection_indices: tuple[int, ...]
    fit_class_counts: tuple[int, ...]
    selection_class_counts: tuple[int, ...]


@dataclass(frozen=True)
class OuterFoldDefinition:
    """An outer evaluation fold and all development partitions beneath it."""

    outer_fold_id: int
    expert_training_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    expert_training_class_counts: tuple[int, ...]
    evaluation_class_counts: tuple[int, ...]
    inner_folds: tuple[InnerFoldDefinition, ...]
    router_development: RouterDevelopmentPartition

    @property
    def expert_training_membership_hash(self) -> str:
        return _hash_indices(self.expert_training_indices)


@dataclass(frozen=True)
class FoldManifest:
    """Versioned, JSON-serializable description of a nested fold map."""

    schema_version: str
    canonical_training_index_sha256: str
    canonical_training_indices: tuple[int, ...]
    canonical_training_labels: tuple[int, ...]
    fold_generation_seed: int
    fold_algorithm: str
    experiment_config: Mapping[str, Any]
    outer_folds: tuple[OuterFoldDefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of the complete manifest."""
        return {
            "schema_version": self.schema_version,
            "canonical_training_index_sha256": self.canonical_training_index_sha256,
            "canonical_training_indices": list(self.canonical_training_indices),
            "canonical_training_labels": list(self.canonical_training_labels),
            "fold_generation_seed": self.fold_generation_seed,
            "fold_algorithm": self.fold_algorithm,
            "experiment_config": json.loads(_canonical_json(dict(self.experiment_config))),
            "outer_folds": [
                {
                    "outer_fold_id": outer.outer_fold_id,
                    "training_indices": list(outer.expert_training_indices),
                    "held_out_indices": list(outer.evaluation_indices),
                    "training_class_counts": list(outer.expert_training_class_counts),
                    "held_out_class_counts": list(outer.evaluation_class_counts),
                    "router_development": {
                        "fit_inner_fold_ids": list(
                            outer.router_development.fit_inner_fold_ids
                        ),
                        "selection_inner_fold_ids": list(
                            outer.router_development.selection_inner_fold_ids
                        ),
                        "fit_indices": list(outer.router_development.fit_indices),
                        "selection_indices": list(
                            outer.router_development.selection_indices
                        ),
                        "fit_class_counts": list(
                            outer.router_development.fit_class_counts
                        ),
                        "selection_class_counts": list(
                            outer.router_development.selection_class_counts
                        ),
                    },
                    "inner_folds": [
                        {
                            "inner_fold_id": inner.inner_fold_id,
                            "training_indices": list(inner.expert_training_indices),
                            "held_out_indices": list(inner.prediction_indices),
                            "training_class_counts": list(
                                inner.expert_training_class_counts
                            ),
                            "held_out_class_counts": list(inner.prediction_class_counts),
                            "training_membership_sha256": (
                                inner.expert_training_membership_hash
                            ),
                        }
                        for inner in outer.inner_folds
                    ],
                }
                for outer in self.outer_folds
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoldManifest":
        """Restore a manifest and reject missing or malformed provenance."""
        if not isinstance(payload, Mapping):
            raise OOFProtocolError("fold manifest must be a mapping")
        required = {
            "schema_version",
            "canonical_training_index_sha256",
            "canonical_training_indices",
            "canonical_training_labels",
            "fold_generation_seed",
            "fold_algorithm",
            "experiment_config",
            "outer_folds",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise OOFProtocolError(
                "fold manifest is missing required fields: " + ", ".join(missing)
            )
        if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise OOFProtocolError(
                f"unsupported fold manifest schema: {payload['schema_version']!r}"
            )
        if not isinstance(payload["experiment_config"], Mapping):
            raise OOFProtocolError("experiment_config must be a mapping")
        if not isinstance(payload["outer_folds"], list):
            raise OOFProtocolError("outer_folds must be a list")

        outer_folds: list[OuterFoldDefinition] = []
        for outer_payload in payload["outer_folds"]:
            if not isinstance(outer_payload, Mapping):
                raise OOFProtocolError("each outer fold must be a mapping")
            outer_required = {
                "outer_fold_id",
                "training_indices",
                "held_out_indices",
                "training_class_counts",
                "held_out_class_counts",
                "router_development",
                "inner_folds",
            }
            missing_outer = sorted(outer_required - set(outer_payload))
            if missing_outer:
                raise OOFProtocolError(
                    "outer fold is missing required fields: "
                    + ", ".join(missing_outer)
                )
            router_payload = outer_payload["router_development"]
            if not isinstance(router_payload, Mapping):
                raise OOFProtocolError("router_development must be a mapping")
            router_required = {
                "fit_inner_fold_ids",
                "selection_inner_fold_ids",
                "fit_indices",
                "selection_indices",
                "fit_class_counts",
                "selection_class_counts",
            }
            missing_router = sorted(router_required - set(router_payload))
            if missing_router:
                raise OOFProtocolError(
                    "router_development is missing required fields: "
                    + ", ".join(missing_router)
                )

            inner_definitions: list[InnerFoldDefinition] = []
            if not isinstance(outer_payload["inner_folds"], list):
                raise OOFProtocolError("inner_folds must be a list")
            for inner_payload in outer_payload["inner_folds"]:
                if not isinstance(inner_payload, Mapping):
                    raise OOFProtocolError("each inner fold must be a mapping")
                inner_required = {
                    "inner_fold_id",
                    "training_indices",
                    "held_out_indices",
                    "training_class_counts",
                    "held_out_class_counts",
                    "training_membership_sha256",
                }
                missing_inner = sorted(inner_required - set(inner_payload))
                if missing_inner:
                    raise OOFProtocolError(
                        "inner fold is missing required fields: "
                        + ", ".join(missing_inner)
                    )
                inner_definition = InnerFoldDefinition(
                    outer_fold_id=int(outer_payload["outer_fold_id"]),
                    inner_fold_id=int(inner_payload["inner_fold_id"]),
                    expert_training_indices=_as_int_tuple(
                        inner_payload["training_indices"], name="inner training_indices"
                    ),
                    prediction_indices=_as_int_tuple(
                        inner_payload["held_out_indices"], name="inner held_out_indices"
                    ),
                    expert_training_class_counts=_as_int_tuple(
                        inner_payload["training_class_counts"],
                        name="inner training_class_counts",
                    ),
                    prediction_class_counts=_as_int_tuple(
                        inner_payload["held_out_class_counts"],
                        name="inner held_out_class_counts",
                    ),
                )
                if str(
                    inner_payload["training_membership_sha256"]
                ) != inner_definition.expert_training_membership_hash:
                    raise FoldMembershipError(
                        "inner training membership hash does not match its indices"
                    )
                inner_definitions.append(inner_definition)

            outer_folds.append(
                OuterFoldDefinition(
                    outer_fold_id=int(outer_payload["outer_fold_id"]),
                    expert_training_indices=_as_int_tuple(
                        outer_payload["training_indices"], name="outer training_indices"
                    ),
                    evaluation_indices=_as_int_tuple(
                        outer_payload["held_out_indices"], name="outer held_out_indices"
                    ),
                    expert_training_class_counts=_as_int_tuple(
                        outer_payload["training_class_counts"],
                        name="outer training_class_counts",
                    ),
                    evaluation_class_counts=_as_int_tuple(
                        outer_payload["held_out_class_counts"],
                        name="outer held_out_class_counts",
                    ),
                    inner_folds=tuple(inner_definitions),
                    router_development=RouterDevelopmentPartition(
                        fit_inner_fold_ids=_as_int_tuple(
                            router_payload["fit_inner_fold_ids"],
                            name="fit_inner_fold_ids",
                        ),
                        selection_inner_fold_ids=_as_int_tuple(
                            router_payload["selection_inner_fold_ids"],
                            name="selection_inner_fold_ids",
                        ),
                        fit_indices=_as_int_tuple(
                            router_payload["fit_indices"], name="router fit_indices"
                        ),
                        selection_indices=_as_int_tuple(
                            router_payload["selection_indices"],
                            name="router selection_indices",
                        ),
                        fit_class_counts=_as_int_tuple(
                            router_payload["fit_class_counts"],
                            name="router fit_class_counts",
                        ),
                        selection_class_counts=_as_int_tuple(
                            router_payload["selection_class_counts"],
                            name="router selection_class_counts",
                        ),
                    ),
                )
            )

        manifest = cls(
            schema_version=str(payload["schema_version"]),
            canonical_training_index_sha256=str(
                payload["canonical_training_index_sha256"]
            ),
            canonical_training_indices=_as_int_tuple(
                payload["canonical_training_indices"],
                name="canonical_training_indices",
            ),
            canonical_training_labels=_as_int_tuple(
                payload["canonical_training_labels"],
                name="canonical_training_labels",
            ),
            fold_generation_seed=int(payload["fold_generation_seed"]),
            fold_algorithm=str(payload["fold_algorithm"]),
            experiment_config=dict(payload["experiment_config"]),
            outer_folds=tuple(outer_folds),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_json(cls, serialized: str | bytes) -> "FoldManifest":
        try:
            payload = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OOFProtocolError("fold manifest is not valid JSON") from exc
        return cls.from_dict(payload)

    def validate(self) -> None:
        """Validate a restored manifest without constructing a manager."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise FoldMembershipError(
                f"unsupported fold manifest schema: {self.schema_version!r}"
            )
        if self.fold_algorithm != FOLD_ALGORITHM:
            raise FoldMembershipError(
                f"unsupported fold-generation algorithm: {self.fold_algorithm!r}"
            )
        _validate_sha256(
            self.canonical_training_index_sha256,
            name="canonical_training_index_sha256",
        )
        if len(self.canonical_training_indices) != len(self.canonical_training_labels):
            raise FoldMembershipError(
                "canonical training indices and labels have different lengths"
            )
        if len(set(self.canonical_training_indices)) != len(self.canonical_training_indices):
            raise FoldMembershipError("manifest canonical indices contain duplicates")
        config = dict(self.experiment_config)
        required_config = {
            "dataset",
            "imbalance_ratio",
            "canonical_population_size",
            "num_classes",
            "outer_folds",
            "inner_folds",
            "router_selection_inner_fold_ids",
            "expert_order",
            "head_medium_tail_source",
            "canonical_class_counts",
        }
        missing_config = sorted(required_config - set(config))
        if missing_config:
            raise FoldMembershipError(
                "manifest experiment_config is missing: "
                + ", ".join(missing_config)
            )
        if config["dataset"] != "CIFAR-100-LT" or float(config["imbalance_ratio"]) != 100.0:
            raise FoldMembershipError(
                "manifest experiment_config does not identify canonical CIFAR-100-LT IR=100"
            )
        if int(config["canonical_population_size"]) != len(self.canonical_training_indices):
            raise FoldMembershipError(
                "manifest canonical population size is inconsistent"
            )
        if config["head_medium_tail_source"] != "scripts.base_trainer.compute_class_groups":
            raise FoldMembershipError(
                "manifest does not use the canonical Head/Medium/Tail definition"
            )
        expert_order = tuple(config["expert_order"])
        if not expert_order or len(set(expert_order)) != len(expert_order):
            raise FoldMembershipError("manifest expert_order must be unique and non-empty")
        try:
            num_classes = int(config["num_classes"])
            outer_count = int(config["outer_folds"])
            inner_count = int(config["inner_folds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FoldMembershipError(
                "manifest experiment_config lacks valid fold dimensions"
            ) from exc
        configured_selection_ids = tuple(
            int(value) for value in config["router_selection_inner_fold_ids"]
        )
        if not configured_selection_ids or not set(configured_selection_ids) <= set(
            range(inner_count)
        ):
            raise FoldMembershipError(
                "manifest router-selection inner fold IDs are out of range"
            )
        if len(self.outer_folds) != outer_count:
            raise FoldMembershipError(
                f"manifest has {len(self.outer_folds)} outer folds, expected {outer_count}"
            )
        if tuple(sorted(self.canonical_training_indices)) != self.canonical_training_indices:
            raise FoldMembershipError("manifest canonical indices must be sorted")
        labels_by_index = dict(
            zip(self.canonical_training_indices, self.canonical_training_labels)
        )
        if any(label < 0 or label >= num_classes for label in self.canonical_training_labels):
            raise FoldMembershipError("manifest contains an out-of-range training label")
        actual_counts = _class_counts(
            self.canonical_training_indices, labels_by_index, num_classes
        )
        recorded_counts = tuple(config["canonical_class_counts"])
        if tuple(int(value) for value in recorded_counts) != actual_counts:
            raise FoldMembershipError("manifest canonical class counts are inconsistent")

        canonical = set(self.canonical_training_indices)
        outer_ids = [outer.outer_fold_id for outer in self.outer_folds]
        if outer_ids != list(range(outer_count)):
            raise FoldMembershipError(f"outer fold IDs must be 0..{outer_count - 1}")
        all_outer_eval = [
            index for outer in self.outer_folds for index in outer.evaluation_indices
        ]
        if set(all_outer_eval) != canonical or len(all_outer_eval) != len(canonical):
            raise FoldMembershipError(
                "outer held-out partitions do not reconstruct the canonical population"
            )

        for outer in self.outer_folds:
            evaluation = set(outer.evaluation_indices)
            training = set(outer.expert_training_indices)
            if len(evaluation) != len(outer.evaluation_indices) or len(training) != len(
                outer.expert_training_indices
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} contains duplicate sample IDs"
                )
            if training & evaluation or training | evaluation != canonical:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} is not a canonical partition"
                )
            if outer.expert_training_class_counts != _class_counts(
                training, labels_by_index, num_classes
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} training class counts are inconsistent"
                )
            if outer.evaluation_class_counts != _class_counts(
                evaluation, labels_by_index, num_classes
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} held-out class counts are inconsistent"
                )
            if [inner.inner_fold_id for inner in outer.inner_folds] != list(range(inner_count)):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} inner IDs must be 0..{inner_count - 1}"
                )

            inner_predictions = [
                index for inner in outer.inner_folds for index in inner.prediction_indices
            ]
            if set(inner_predictions) != training or len(inner_predictions) != len(training):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} inner predictions do not cover training once"
                )
            for inner in outer.inner_folds:
                inner_training = set(inner.expert_training_indices)
                prediction = set(inner.prediction_indices)
                if inner_training & prediction or inner_training | prediction != training:
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "is not a partition of outer training"
                    )
                if inner.expert_training_class_counts != _class_counts(
                    inner_training, labels_by_index, num_classes
                ):
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "training class counts are inconsistent"
                    )
                if inner.prediction_class_counts != _class_counts(
                    prediction, labels_by_index, num_classes
                ):
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "held-out class counts are inconsistent"
                    )
                if inner.expert_training_membership_hash != _hash_indices(inner_training):
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "training membership hash is inconsistent"
                    )
                if any(count == 0 for count in inner.expert_training_class_counts):
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} drops a class"
                    )

            router = outer.router_development
            fit = set(router.fit_indices)
            selection = set(router.selection_indices)
            if len(fit) != len(router.fit_indices) or len(selection) != len(
                router.selection_indices
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router partitions contain duplicates"
                )
            if fit & selection or fit | selection != training:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router partitions are inconsistent"
                )
            if set(router.fit_inner_fold_ids) & set(router.selection_inner_fold_ids):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router fold IDs overlap"
                )
            if set(router.fit_inner_fold_ids) | set(router.selection_inner_fold_ids) != set(
                range(inner_count)
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router fold IDs do not cover inner folds"
                )
            if tuple(router.selection_inner_fold_ids) != configured_selection_ids:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router-selection IDs disagree "
                    "with experiment_config"
                )
            if router.fit_class_counts != _class_counts(fit, labels_by_index, num_classes):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router-fit class counts are inconsistent"
                )
            if router.selection_class_counts != _class_counts(
                selection, labels_by_index, num_classes
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router-selection class counts "
                    "are inconsistent"
                )
            if any(count == 0 for count in router.fit_class_counts) or any(
                count == 0 for count in router.selection_class_counts
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router development drops a class"
                )


class NestedOOFFoldManager:
    """Construct and validate deterministic five-by-four nested folds.

    The manager accepts aligned canonical sample indices and training labels so
    synthetic fixtures can exercise the exact same membership contracts as the
    real CIFAR-100-LT population.  It never accesses image data or test data.
    """

    def __init__(
        self,
        canonical_indices: Sequence[int] | np.ndarray,
        training_labels: Sequence[int] | np.ndarray,
        *,
        seed: int = 42,
        outer_folds: int = 5,
        inner_folds: int = 4,
        num_classes: int | None = None,
        expert_order: Sequence[str] = DEFAULT_EXPERT_ORDER,
        canonical_training_index_sha256: str | None = None,
    ) -> None:
        indices = np.asarray(canonical_indices)
        labels = np.asarray(training_labels)
        if indices.ndim != 1 or labels.ndim != 1 or len(indices) != len(labels):
            raise OOFProtocolError(
                "canonical_indices and training_labels must be aligned one-dimensional arrays"
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise OOFProtocolError("canonical_indices must contain integers")
        if not np.issubdtype(labels.dtype, np.integer):
            raise OOFProtocolError("training_labels must contain integers")
        if len(indices) == 0:
            raise OOFProtocolError("canonical population must not be empty")

        order = np.argsort(indices, kind="stable")
        indices = indices[order].astype(np.int64, copy=False)
        labels = labels[order].astype(np.int64, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise OOFProtocolError("canonical_indices contain duplicate sample IDs")
        if np.any(indices < 0) or np.any(indices >= 50000):
            raise OOFProtocolError(
                "canonical_indices must address only the CIFAR-100 training range [0, 50000)"
            )

        if num_classes is None:
            num_classes = int(labels.max()) + 1
        if isinstance(num_classes, bool) or not isinstance(num_classes, (int, np.integer)):
            raise OOFProtocolError("num_classes must be an integer")
        num_classes = int(num_classes)
        if num_classes < 1:
            raise OOFProtocolError(f"num_classes must be positive, got {num_classes}")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise OOFProtocolError(
                f"training_labels must be in [0, {num_classes})"
            )

        expert_order = tuple(str(name) for name in expert_order)
        if not expert_order or len(set(expert_order)) != len(expert_order):
            raise OOFProtocolError("expert_order must contain unique expert identities")
        if any(not name for name in expert_order):
            raise OOFProtocolError("expert_order entries must be non-empty")

        self.canonical_indices = tuple(int(value) for value in indices.tolist())
        self.training_labels = tuple(int(value) for value in labels.tolist())
        self.fold_generation_seed = int(seed)
        self.outer_fold_count = int(outer_folds)
        self.inner_fold_count = int(inner_folds)
        self.num_classes = num_classes
        self.expert_order = expert_order
        self.canonical_training_index_sha256 = (
            canonical_training_index_sha256 or _hash_indices(self.canonical_indices)
        )
        _validate_sha256(
            self.canonical_training_index_sha256,
            name="canonical_training_index_sha256",
        )

        if self.outer_fold_count < 2 or self.inner_fold_count < 2:
            raise FoldConfigurationError(
                "outer_folds and inner_folds must both be at least 2"
            )

        self._labels_by_index = dict(zip(self.canonical_indices, self.training_labels))
        self.canonical_class_counts = _class_counts(
            self.canonical_indices, self._labels_by_index, self.num_classes
        )
        self._validate_class_coverage_configuration()
        self._outer_folds = self._build_folds()
        self.validate_membership()

    @classmethod
    def from_canonical_training_data(
        cls,
        data_root: str | Path = "./data",
        *,
        seed: int = 42,
        outer_folds: int = 5,
        inner_folds: int = 4,
        expert_order: Sequence[str] = DEFAULT_EXPERT_ORDER,
    ) -> "NestedOOFFoldManager":
        """Build a manager from the project's canonical train artifact only."""
        from data.protocol_splits import (
            LT_TRAIN_FILENAME,
            load_lt_train_indices,
            load_lt_train_labels,
        )

        data_root = Path(data_root)
        indices = load_lt_train_indices(str(data_root))
        labels = load_lt_train_labels(str(data_root))
        if len(indices) != 10847 or len(labels) != 10847:
            raise FoldConfigurationError(
                "canonical CIFAR-100-LT training population must contain exactly "
                f"10,847 samples, got indices={len(indices)}, labels={len(labels)}"
            )
        artifact = data_root / "processed" / LT_TRAIN_FILENAME
        return cls(
            indices,
            labels,
            seed=seed,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            num_classes=100,
            expert_order=expert_order,
            canonical_training_index_sha256=_hash_file(artifact),
        )

    def _validate_class_coverage_configuration(self) -> None:
        missing = [
            label
            for label, count in enumerate(self.canonical_class_counts)
            if count == 0
        ]
        if missing:
            raise FoldConfigurationError(
                "canonical population is missing required classes: "
                + ", ".join(str(label) for label in missing)
            )

        impossible = []
        for label, count in enumerate(self.canonical_class_counts):
            minimum_outer_train = count - int(np.ceil(count / self.outer_fold_count))
            if count < self.outer_fold_count or minimum_outer_train < self.inner_fold_count:
                impossible.append(
                    f"class {label}: count={count}, requires at least "
                    f"{self.outer_fold_count} for outer folds and "
                    f"{self.inner_fold_count} remaining for inner training"
                )
        if impossible:
            raise FoldConfigurationError(
                "requested nested folds cannot preserve every class in every "
                "expert-training partition; " + "; ".join(impossible)
            )

    def _build_folds(self) -> tuple[OuterFoldDefinition, ...]:
        outer_parts = _stratified_partition(
            self.canonical_indices,
            self._labels_by_index,
            num_classes=self.num_classes,
            n_folds=self.outer_fold_count,
            seed=self.fold_generation_seed,
        )
        folds: list[OuterFoldDefinition] = []
        all_indices = set(self.canonical_indices)

        for outer_id, evaluation in enumerate(outer_parts):
            evaluation_set = set(evaluation)
            training = tuple(sorted(all_indices - evaluation_set))
            training_counts = _class_counts(
                training, self._labels_by_index, self.num_classes
            )
            evaluation_counts = _class_counts(
                evaluation, self._labels_by_index, self.num_classes
            )

            inner_parts = _stratified_partition(
                training,
                self._labels_by_index,
                num_classes=self.num_classes,
                n_folds=self.inner_fold_count,
                seed=_derived_seed(self.fold_generation_seed, outer_id, 1),
            )
            inner_definitions: list[InnerFoldDefinition] = []
            training_set = set(training)
            for inner_id, prediction in enumerate(inner_parts):
                prediction_set = set(prediction)
                inner_training = tuple(sorted(training_set - prediction_set))
                inner_definitions.append(
                    InnerFoldDefinition(
                        outer_fold_id=outer_id,
                        inner_fold_id=inner_id,
                        expert_training_indices=inner_training,
                        prediction_indices=prediction,
                        expert_training_class_counts=_class_counts(
                            inner_training, self._labels_by_index, self.num_classes
                        ),
                        prediction_class_counts=_class_counts(
                            prediction, self._labels_by_index, self.num_classes
                        ),
                    )
                )

            selection_ids = (0,)
            fit_ids = tuple(
                inner.inner_fold_id
                for inner in inner_definitions
                if inner.inner_fold_id not in selection_ids
            )
            selection_indices = tuple(
                sorted(
                    index
                    for inner in inner_definitions
                    if inner.inner_fold_id in selection_ids
                    for index in inner.prediction_indices
                )
            )
            fit_indices = tuple(
                sorted(
                    index
                    for inner in inner_definitions
                    if inner.inner_fold_id in fit_ids
                    for index in inner.prediction_indices
                )
            )
            router_development = RouterDevelopmentPartition(
                fit_inner_fold_ids=fit_ids,
                selection_inner_fold_ids=selection_ids,
                fit_indices=fit_indices,
                selection_indices=selection_indices,
                fit_class_counts=_class_counts(
                    fit_indices, self._labels_by_index, self.num_classes
                ),
                selection_class_counts=_class_counts(
                    selection_indices, self._labels_by_index, self.num_classes
                ),
            )
            folds.append(
                OuterFoldDefinition(
                    outer_fold_id=outer_id,
                    expert_training_indices=training,
                    evaluation_indices=evaluation,
                    expert_training_class_counts=training_counts,
                    evaluation_class_counts=evaluation_counts,
                    inner_folds=tuple(inner_definitions),
                    router_development=router_development,
                )
            )
        return tuple(folds)

    @property
    def outer_folds(self) -> tuple[OuterFoldDefinition, ...]:
        return self._outer_folds

    def outer_fold(self, outer_fold_id: int) -> OuterFoldDefinition:
        if isinstance(outer_fold_id, bool) or not isinstance(
            outer_fold_id, (int, np.integer)
        ):
            raise FoldMembershipError(f"unknown outer fold ID: {outer_fold_id!r}")
        outer_fold_id = int(outer_fold_id)
        if outer_fold_id < 0 or outer_fold_id >= len(self._outer_folds):
            raise FoldMembershipError(f"unknown outer fold ID: {outer_fold_id!r}")
        try:
            return self._outer_folds[outer_fold_id]
        except (IndexError, TypeError) as exc:
            raise FoldMembershipError(f"unknown outer fold ID: {outer_fold_id!r}") from exc

    def inner_fold(self, outer_fold_id: int, inner_fold_id: int) -> InnerFoldDefinition:
        outer = self.outer_fold(outer_fold_id)
        if isinstance(inner_fold_id, bool) or not isinstance(
            inner_fold_id, (int, np.integer)
        ):
            raise FoldMembershipError(
                f"unknown inner fold ID {inner_fold_id!r} for outer fold {outer_fold_id}"
            )
        inner_fold_id = int(inner_fold_id)
        if inner_fold_id < 0 or inner_fold_id >= len(outer.inner_folds):
            raise FoldMembershipError(
                f"unknown inner fold ID {inner_fold_id!r} for outer fold {outer_fold_id}"
            )
        try:
            return outer.inner_folds[inner_fold_id]
        except (IndexError, TypeError) as exc:
            raise FoldMembershipError(
                f"unknown inner fold ID {inner_fold_id!r} for outer fold {outer_fold_id}"
            ) from exc

    def validate_membership(self) -> None:
        """Assert all outer, inner, and router-development invariants."""
        canonical = set(self.canonical_indices)
        outer_evaluation = [
            index for fold in self._outer_folds for index in fold.evaluation_indices
        ]
        if len(outer_evaluation) != len(canonical) or len(set(outer_evaluation)) != len(canonical):
            raise FoldMembershipError(
                "outer evaluation folds contain duplicate or missing canonical samples"
            )
        if set(outer_evaluation) != canonical:
            raise FoldMembershipError(
                "outer evaluation folds do not reconstruct the canonical population"
            )

        for outer in self._outer_folds:
            evaluation = set(outer.evaluation_indices)
            training = set(outer.expert_training_indices)
            if len(evaluation) != len(outer.evaluation_indices) or len(training) != len(
                outer.expert_training_indices
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} contains duplicate sample IDs"
                )
            if training & evaluation:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} training/evaluation overlap"
                )
            if training | evaluation != canonical:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} does not partition the canonical population"
                )

            inner_prediction = [
                index for inner in outer.inner_folds for index in inner.prediction_indices
            ]
            if len(inner_prediction) != len(training) or len(set(inner_prediction)) != len(
                training
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} inner predictions have duplicate "
                    "or missing samples"
                )
            if set(inner_prediction) != training:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} inner predictions escape outer training"
                )

            for inner in outer.inner_folds:
                inner_training = set(inner.expert_training_indices)
                prediction = set(inner.prediction_indices)
                if inner_training & prediction:
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "expert training/prediction overlap"
                    )
                if inner_training | prediction != training:
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "does not partition outer training"
                    )
                if any(count == 0 for count in inner.expert_training_class_counts):
                    raise FoldMembershipError(
                        f"outer {outer.outer_fold_id}, inner {inner.inner_fold_id} "
                        "drops a class from expert training"
                    )

            router = outer.router_development
            fit = set(router.fit_indices)
            selection = set(router.selection_indices)
            if len(fit) != len(router.fit_indices) or len(selection) != len(
                router.selection_indices
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router partitions contain duplicates"
                )
            if fit & selection or fit | selection != training:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router fit/selection is not a partition"
                )
            if fit & evaluation or selection & evaluation:
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} outer evaluation leaked into "
                    "router development"
                )
            if any(count == 0 for count in router.fit_class_counts) or any(
                count == 0 for count in router.selection_class_counts
            ):
                raise FoldMembershipError(
                    f"outer fold {outer.outer_fold_id} router development drops a class"
                )

    def validate_oof_record(self, record: "OOFPredictionRecord") -> None:
        """Validate one OOF row against its exact expert-training membership."""
        if not isinstance(record, OOFPredictionRecord):
            raise OOFArtifactValidationError(
                f"OOF record must be OOFPredictionRecord, got {type(record).__name__}"
            )
        record.validate_provenance()
        if record.expert_id not in self.expert_order:
            raise OOFArtifactValidationError(
                f"expert {record.expert_id!r} is absent from the fixed expert ordering"
            )

        try:
            outer = self.outer_fold(record.outer_fold_id)
        except FoldMembershipError as exc:
            raise OOFArtifactValidationError(str(exc)) from exc

        if record.sample_index not in self._labels_by_index:
            raise OOFArtifactValidationError(
                f"sample ID {record.sample_index} is outside the canonical population"
            )
        expected_label = self._labels_by_index[record.sample_index]
        if record.training_label != expected_label:
            raise OOFArtifactValidationError(
                f"sample ID {record.sample_index} has training label "
                f"{record.training_label}, expected {expected_label}"
            )

        if record.inner_fold_id is None:
            expected_population = set(outer.evaluation_indices)
            expected_training = outer.expert_training_indices
            population_name = "outer evaluation population"
        else:
            try:
                inner = self.inner_fold(record.outer_fold_id, record.inner_fold_id)
            except FoldMembershipError as exc:
                raise OOFArtifactValidationError(str(exc)) from exc
            expected_population = set(inner.prediction_indices)
            expected_training = inner.expert_training_indices
            population_name = "inner prediction population"

        if record.sample_index in set(expected_training):
            raise OOFArtifactValidationError(
                f"sample ID {record.sample_index} belongs to the expert training population; "
                "an OOF prediction may not evaluate an in-training sample"
            )
        if record.sample_index not in expected_population:
            raise OOFArtifactValidationError(
                f"sample ID {record.sample_index} is not in the declared "
                f"{population_name} for outer fold {record.outer_fold_id}"
            )

        expected_hash = _hash_indices(expected_training)
        if record.expert_training_membership_hash != expected_hash:
            raise OOFArtifactValidationError(
                "expert-training membership hash does not match the declared fold"
            )
        if len(record.logits) != self.num_classes:
            raise OOFArtifactValidationError(
                f"logits contain {len(record.logits)} classes, expected {self.num_classes}"
            )

    def distribution_report(self) -> dict[str, Any]:
        """Return class counts for every manifest training/held-out role."""
        outer_reports = []
        for outer in self.outer_folds:
            inner_reports = []
            for inner in outer.inner_folds:
                inner_reports.append(
                    {
                        "inner_fold_id": inner.inner_fold_id,
                        "training_size": len(inner.expert_training_indices),
                        "held_out_size": len(inner.prediction_indices),
                        "training_class_counts": list(inner.expert_training_class_counts),
                        "held_out_class_counts": list(inner.prediction_class_counts),
                        "training_membership_sha256": inner.expert_training_membership_hash,
                    }
                )
            router = outer.router_development
            outer_reports.append(
                {
                    "outer_fold_id": outer.outer_fold_id,
                    "training_size": len(outer.expert_training_indices),
                    "held_out_size": len(outer.evaluation_indices),
                    "training_class_counts": list(outer.expert_training_class_counts),
                    "held_out_class_counts": list(outer.evaluation_class_counts),
                    "training_membership_sha256": outer.expert_training_membership_hash,
                    "inner": inner_reports,
                    "router_development": {
                        "fit_size": len(router.fit_indices),
                        "selection_size": len(router.selection_indices),
                        "fit_class_counts": list(router.fit_class_counts),
                        "selection_class_counts": list(router.selection_class_counts),
                    },
                }
            )

        from scripts.base_trainer import compute_class_groups

        groups = compute_class_groups(np.asarray(self.canonical_class_counts))
        return {
            "canonical": {
                "population_size": len(self.canonical_indices),
                "class_counts": list(self.canonical_class_counts),
            },
            "head_medium_tail": {
                "source": "scripts.base_trainer.compute_class_groups",
                "classes": {
                    name: [int(value) for value in values.tolist()]
                    for name, values in groups.items()
                },
            },
            "outer": outer_reports,
        }

    def manifest(self) -> FoldManifest:
        """Return the complete versioned manifest for this fold map."""
        config = {
            "dataset": "CIFAR-100-LT",
            "imbalance_ratio": 100.0,
            "canonical_population_size": len(self.canonical_indices),
            "num_classes": self.num_classes,
            "outer_folds": self.outer_fold_count,
            "inner_folds": self.inner_fold_count,
            "router_selection_inner_fold_ids": [0],
            "expert_order": list(self.expert_order),
            "head_medium_tail_source": "scripts.base_trainer.compute_class_groups",
            "canonical_class_counts": list(self.canonical_class_counts),
        }
        manifest = FoldManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            canonical_training_index_sha256=self.canonical_training_index_sha256,
            canonical_training_indices=self.canonical_indices,
            canonical_training_labels=self.training_labels,
            fold_generation_seed=self.fold_generation_seed,
            fold_algorithm=FOLD_ALGORITHM,
            experiment_config=config,
            outer_folds=self._outer_folds,
        )
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class OOFPredictionRecord:
    """One expert prediction with enough provenance to audit its membership."""

    schema_version: str
    sample_index: int
    training_label: int
    outer_fold_id: int
    inner_fold_id: int | None
    expert_id: str
    training_seed: int
    expert_training_membership_hash: str
    checkpoint_path: str
    checkpoint_sha256: str
    resolved_config_json: str
    resolved_config_sha256: str
    logits: tuple[float, ...]
    features: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != OOF_PREDICTION_SCHEMA_VERSION:
            raise OOFArtifactValidationError(
                f"unsupported OOF prediction schema: {self.schema_version!r}"
            )
        for name in ("sample_index", "training_label", "outer_fold_id", "training_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise OOFArtifactValidationError(f"{name} must be an integer")
        if self.inner_fold_id is not None and (
            isinstance(self.inner_fold_id, bool)
            or not isinstance(self.inner_fold_id, (int, np.integer))
        ):
            raise OOFArtifactValidationError("inner_fold_id must be an integer or null")
        if not self.expert_id:
            raise OOFArtifactValidationError("expert_id must be non-empty")
        if not self.checkpoint_path:
            raise OOFArtifactValidationError("checkpoint_path must be non-empty")
        membership_hash = _validate_sha256(
            self.expert_training_membership_hash,
            name="expert_training_membership_hash",
        )
        checkpoint_hash = _validate_sha256(
            self.checkpoint_sha256, name="checkpoint_sha256"
        )
        config_hash = _validate_sha256(
            self.resolved_config_sha256,
            name="resolved_config_sha256",
        )
        try:
            config = json.loads(self.resolved_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OOFArtifactValidationError(
                "resolved_config_json is not valid JSON"
            ) from exc
        if not isinstance(config, Mapping):
            raise OOFArtifactValidationError("resolved_config_json must encode a mapping")

        logits = np.asarray(self.logits, dtype=np.float64)
        if logits.ndim != 1 or logits.size == 0 or not np.isfinite(logits).all():
            raise OOFArtifactValidationError(
                "logits must be a non-empty one-dimensional finite vector"
            )
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "training_label", int(self.training_label))
        object.__setattr__(self, "outer_fold_id", int(self.outer_fold_id))
        object.__setattr__(
            self,
            "inner_fold_id",
            None if self.inner_fold_id is None else int(self.inner_fold_id),
        )
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(self, "expert_training_membership_hash", membership_hash)
        object.__setattr__(self, "checkpoint_sha256", checkpoint_hash)
        object.__setattr__(self, "resolved_config_sha256", config_hash)
        object.__setattr__(self, "logits", tuple(float(value) for value in logits.tolist()))
        if self.features is not None:
            features = np.asarray(self.features, dtype=np.float64)
            if features.ndim != 1 or not np.isfinite(features).all():
                raise OOFArtifactValidationError(
                    "features must be a one-dimensional finite vector when supplied"
                )
            object.__setattr__(
                self, "features", tuple(float(value) for value in features.tolist())
            )

    @classmethod
    def create(
        cls,
        *,
        sample_index: int,
        training_label: int,
        outer_fold_id: int,
        inner_fold_id: int | None,
        expert_id: str,
        training_seed: int,
        expert_training_membership_hash: str,
        checkpoint_path: str,
        checkpoint_sha256: str,
        resolved_config: Mapping[str, Any],
        logits: Sequence[float] | np.ndarray,
        features: Sequence[float] | np.ndarray | None = None,
    ) -> "OOFPredictionRecord":
        resolved_config_json = _canonical_json(dict(resolved_config))
        return cls(
            schema_version=OOF_PREDICTION_SCHEMA_VERSION,
            sample_index=sample_index,
            training_label=training_label,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
            expert_id=expert_id,
            training_seed=training_seed,
            expert_training_membership_hash=expert_training_membership_hash,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            resolved_config_json=resolved_config_json,
            resolved_config_sha256=_sha256_text(resolved_config_json),
            logits=tuple(float(value) for value in np.asarray(logits).tolist()),
            features=(
                None
                if features is None
                else tuple(float(value) for value in np.asarray(features).tolist())
            ),
        )

    def validate_provenance(self) -> None:
        """Validate hashes tying the record to its checkpoint and config."""
        _validate_sha256(
            self.expert_training_membership_hash,
            name="expert_training_membership_hash",
        )
        _validate_sha256(self.checkpoint_sha256, name="checkpoint_sha256")
        expected_config_hash = _sha256_text(self.resolved_config_json)
        if self.resolved_config_sha256 != expected_config_hash:
            raise OOFArtifactValidationError(
                "resolved configuration hash does not match resolved_config_json"
            )

    def to_dict(self) -> dict[str, Any]:
        config = json.loads(self.resolved_config_json)
        return {
            "schema_version": self.schema_version,
            "sample_index": self.sample_index,
            "training_label": self.training_label,
            "outer_fold_id": self.outer_fold_id,
            "inner_fold_id": self.inner_fold_id,
            "expert_id": self.expert_id,
            "training_seed": self.training_seed,
            "expert_training_membership_sha256": self.expert_training_membership_hash,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "resolved_config": config,
            "resolved_config_sha256": self.resolved_config_sha256,
            "logits": list(self.logits),
            "features": None if self.features is None else list(self.features),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OOFPredictionRecord":
        if not isinstance(payload, Mapping):
            raise OOFArtifactValidationError("OOF prediction record must be a mapping")
        required = {
            "schema_version",
            "sample_index",
            "training_label",
            "outer_fold_id",
            "inner_fold_id",
            "expert_id",
            "training_seed",
            "expert_training_membership_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "resolved_config",
            "resolved_config_sha256",
            "logits",
            "features",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise OOFArtifactValidationError(
                "OOF prediction record is missing: " + ", ".join(missing)
            )
        try:
            resolved_config_json = _canonical_json(dict(payload["resolved_config"]))
        except (TypeError, ValueError) as exc:
            raise OOFArtifactValidationError("resolved_config must be a mapping") from exc
        return cls(
            schema_version=str(payload["schema_version"]),
            sample_index=payload["sample_index"],
            training_label=payload["training_label"],
            outer_fold_id=payload["outer_fold_id"],
            inner_fold_id=payload["inner_fold_id"],
            expert_id=str(payload["expert_id"]),
            training_seed=payload["training_seed"],
            expert_training_membership_hash=str(
                payload["expert_training_membership_sha256"]
            ),
            checkpoint_path=str(payload["checkpoint_path"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            resolved_config_json=resolved_config_json,
            resolved_config_sha256=str(payload["resolved_config_sha256"]),
            logits=tuple(payload["logits"]),
            features=(
                None if payload["features"] is None else tuple(payload["features"])
            ),
        )


@dataclass(frozen=True)
class OOFPredictionArtifact:
    """Collection of aligned OOF records with a fixed expert axis."""

    expert_order: tuple[str, ...]
    num_classes: int
    records: tuple[OOFPredictionRecord, ...]
    schema_version: str = OOF_PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OOF_PREDICTION_SCHEMA_VERSION:
            raise OOFArtifactValidationError(
                f"unsupported OOF artifact schema: {self.schema_version!r}"
            )
        order = tuple(str(name) for name in self.expert_order)
        if not order or len(set(order)) != len(order):
            raise OOFArtifactValidationError(
                "OOF artifact expert ordering must be non-empty and unique"
            )
        if isinstance(self.num_classes, bool) or not isinstance(
            self.num_classes, (int, np.integer)
        ) or int(self.num_classes) < 1:
            raise OOFArtifactValidationError("OOF artifact num_classes must be positive")
        object.__setattr__(self, "expert_order", order)
        object.__setattr__(self, "num_classes", int(self.num_classes))
        object.__setattr__(self, "records", tuple(self.records))

    def validate(
        self,
        manager: NestedOOFFoldManager,
        *,
        require_complete: bool = False,
    ) -> None:
        """Validate record membership, provenance, alignment, and completeness."""
        if self.schema_version != OOF_PREDICTION_SCHEMA_VERSION:
            raise OOFArtifactValidationError(
                f"unsupported OOF artifact schema: {self.schema_version!r}"
            )
        if self.expert_order != manager.expert_order:
            raise OOFArtifactValidationError(
                "expert ordering is inconsistent with the fold manifest"
            )
        if self.num_classes != manager.num_classes:
            raise OOFArtifactValidationError(
                f"artifact has {self.num_classes} classes, expected {manager.num_classes}"
            )

        seen: set[tuple[int, int, int | None, str]] = set()
        provenance: dict[tuple[int, int | None, str], tuple[Any, ...]] = {}
        for record in self.records:
            if not isinstance(record, OOFPredictionRecord):
                raise OOFArtifactValidationError("artifact contains a non-record value")
            try:
                record.validate_provenance()
                manager.validate_oof_record(record)
            except OOFProtocolError as exc:
                if isinstance(exc, OOFArtifactValidationError):
                    raise
                raise OOFArtifactValidationError(str(exc)) from exc
            key = (
                record.sample_index,
                record.outer_fold_id,
                record.inner_fold_id,
                record.expert_id,
            )
            if key in seen:
                raise OOFArtifactValidationError(
                    f"duplicate OOF record for sample/fold/expert key {key}"
                )
            seen.add(key)
            if len(record.logits) != self.num_classes:
                raise OOFArtifactValidationError(
                    f"record logits have {len(record.logits)} classes, expected {self.num_classes}"
                )
            group_key = (record.outer_fold_id, record.inner_fold_id, record.expert_id)
            signature = (
                record.training_seed,
                record.expert_training_membership_hash,
                record.checkpoint_path,
                record.checkpoint_sha256,
                record.resolved_config_json,
                record.resolved_config_sha256,
            )
            previous = provenance.get(group_key)
            if previous is not None and previous != signature:
                raise OOFArtifactValidationError(
                    "inconsistent expert checkpoint/config provenance within "
                    f"outer={record.outer_fold_id}, inner={record.inner_fold_id}, "
                    f"expert={record.expert_id}"
                )
            provenance[group_key] = signature

        if require_complete:
            expected: set[tuple[int, int, int, str]] = set()
            for outer in manager.outer_folds:
                for inner in outer.inner_folds:
                    for sample_index in inner.prediction_indices:
                        for expert_id in manager.expert_order:
                            expected.add(
                                (
                                    sample_index,
                                    outer.outer_fold_id,
                                    inner.inner_fold_id,
                                    expert_id,
                                )
                            )
            actual = {
                (sample, outer, inner, expert)
                for sample, outer, inner, expert in seen
                if inner is not None
            }
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise OOFArtifactValidationError(
                    "OOF artifact is incomplete or contains unexpected rows; "
                    f"missing={missing[:3]}, extra={extra[:3]}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expert_order": list(self.expert_order),
            "num_classes": self.num_classes,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OOFPredictionArtifact":
        if not isinstance(payload, Mapping):
            raise OOFArtifactValidationError("OOF artifact must be a mapping")
        required = {"schema_version", "expert_order", "num_classes", "records"}
        missing = sorted(required - set(payload))
        if missing:
            raise OOFArtifactValidationError(
                "OOF artifact is missing: " + ", ".join(missing)
            )
        if not isinstance(payload["records"], list):
            raise OOFArtifactValidationError("OOF artifact records must be a list")
        return cls(
            schema_version=str(payload["schema_version"]),
            expert_order=tuple(payload["expert_order"]),
            num_classes=payload["num_classes"],
            records=tuple(OOFPredictionRecord.from_dict(item) for item in payload["records"]),
        )

    @classmethod
    def from_json(cls, serialized: str | bytes) -> "OOFPredictionArtifact":
        try:
            payload = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OOFArtifactValidationError("OOF artifact is not valid JSON") from exc
        return cls.from_dict(payload)
