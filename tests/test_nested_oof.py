"""Synthetic regression tests for the nested OOF protocol.

These tests use only hand-built training-like populations.  They never load
the CIFAR-100 test split, checkpoints, or model predictions.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.nested_oof import (
    FoldConfigurationError,
    FoldManifest,
    FoldMembershipError,
    NestedOOFFoldManager,
    OOFArtifactValidationError,
    OOFPredictionArtifact,
    OOFPredictionRecord,
)


def _synthetic_population() -> tuple[np.ndarray, np.ndarray]:
    """Return a small population with a five-sample rare class."""
    labels = np.repeat(np.arange(3, dtype=np.int64), [10, 7, 5])
    indices = np.arange(100, 100 + len(labels), dtype=np.int64)
    return indices, labels


def test_nested_folds_are_deterministic_and_reconstruct_the_population():
    indices, labels = _synthetic_population()

    first = NestedOOFFoldManager(
        indices,
        labels,
        seed=17,
        num_classes=3,
        expert_order=("A", "B"),
    )
    second = NestedOOFFoldManager(
        indices,
        labels,
        seed=17,
        num_classes=3,
        expert_order=("A", "B"),
    )

    assert first.manifest().to_dict() == second.manifest().to_dict()

    outer_eval = np.concatenate([
        np.asarray(f.evaluation_indices, dtype=np.int64)
        for f in first.outer_folds
    ])
    assert np.array_equal(np.sort(outer_eval), indices)
    assert len(np.unique(outer_eval)) == len(indices)


def test_different_fold_seeds_change_the_assignment():
    indices, labels = _synthetic_population()
    first = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    second = NestedOOFFoldManager(
        indices, labels, seed=18, num_classes=3, expert_order=("A", "B")
    )

    first_outer = [fold.evaluation_indices for fold in first.outer_folds]
    second_outer = [fold.evaluation_indices for fold in second.outer_folds]
    assert first_outer != second_outer


def test_inner_oof_and_router_development_partitions_cover_outer_training_once():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )

    for outer in manager.outer_folds:
        outer_training = set(outer.expert_training_indices)
        inner_predictions = [
            sample
            for inner in outer.inner_folds
            for sample in inner.prediction_indices
        ]
        assert set(inner_predictions) == outer_training
        assert len(inner_predictions) == len(set(inner_predictions))

        for inner in outer.inner_folds:
            assert not set(inner.expert_training_indices) & set(inner.prediction_indices)
            assert (
                set(inner.expert_training_indices) | set(inner.prediction_indices)
                == outer_training
            )

        router = outer.router_development
        assert set(router.fit_indices) | set(router.selection_indices) == outer_training
        assert not set(router.fit_indices) & set(router.selection_indices)
        assert not set(outer.evaluation_indices) & (
            set(router.fit_indices) | set(router.selection_indices)
        )


def test_five_sample_class_has_one_outer_holdout_and_survives_inner_training():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )

    for outer in manager.outer_folds:
        assert outer.evaluation_class_counts[2] == 1
        assert outer.expert_training_class_counts[2] == 4
        for inner in outer.inner_folds:
            assert inner.prediction_class_counts[2] == 1
            assert inner.expert_training_class_counts[2] == 3

        router = outer.router_development
        assert router.fit_class_counts[2] == 3
        assert router.selection_class_counts[2] == 1


def test_impossible_rare_class_configuration_fails_clearly():
    indices = np.arange(12, dtype=np.int64)
    labels = np.repeat(np.arange(3, dtype=np.int64), [5, 5, 2])

    try:
        NestedOOFFoldManager(indices, labels, num_classes=3)
    except FoldConfigurationError as exc:
        assert "class 2" in str(exc)
        assert "cannot preserve" in str(exc)
    else:
        raise AssertionError("a two-sample class was accepted for five outer folds")


def test_manifest_round_trips_and_records_membership_provenance():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices,
        labels,
        seed=17,
        num_classes=3,
        expert_order=("A", "B"),
        canonical_training_index_sha256="f" * 64,
    )
    manifest = manager.manifest()
    payload = manifest.to_dict()

    assert payload["schema_version"] == "nested_oof_manifest.v1"
    assert payload["canonical_training_index_sha256"] == "f" * 64
    assert payload["fold_generation_seed"] == 17
    assert payload["fold_algorithm"]
    assert len(payload["canonical_training_indices"]) == len(indices)
    assert len(payload["outer_folds"]) == 5
    assert payload["outer_folds"][0]["inner_folds"][0]["training_membership_sha256"]

    restored = FoldManifest.from_json(manifest.to_json())
    restored.validate()
    assert restored.to_dict() == payload


def _label_map(indices: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    return {int(index): int(label) for index, label in zip(indices, labels)}


def _valid_oof_record(
    manager: NestedOOFFoldManager,
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    sample_index: int | None = None,
    outer_fold_id: int = 0,
    inner_fold_id: int = 1,
    expert_id: str = "A",
) -> OOFPredictionRecord:
    inner = manager.inner_fold(outer_fold_id, inner_fold_id)
    if sample_index is None:
        sample_index = inner.prediction_indices[0]
    return OOFPredictionRecord.create(
        sample_index=sample_index,
        training_label=_label_map(indices, labels)[sample_index],
        outer_fold_id=outer_fold_id,
        inner_fold_id=inner_fold_id,
        expert_id=expert_id,
        training_seed=78,
        expert_training_membership_hash=inner.expert_training_membership_hash,
        checkpoint_path="synthetic/checkpoint.pt",
        checkpoint_sha256="a" * 64,
        resolved_config={"expert": expert_id, "seed": 78, "synthetic": True},
        logits=np.zeros(3, dtype=np.float32),
        features=np.ones(2, dtype=np.float32),
    )


def test_oof_record_accepts_only_the_inner_prediction_population():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    valid = _valid_oof_record(manager, indices, labels)
    manager.validate_oof_record(valid)

    inner = manager.inner_fold(0, 1)
    leaking_sample = inner.expert_training_indices[0]
    leaking = _valid_oof_record(
        manager, indices, labels, sample_index=leaking_sample
    )
    try:
        manager.validate_oof_record(leaking)
    except OOFArtifactValidationError as exc:
        assert "training population" in str(exc)
    else:
        raise AssertionError("an in-training OOF sample was accepted")


def test_outer_expert_records_are_distinct_from_inner_oof_records():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    outer = manager.outer_fold(0)
    label_map = _label_map(indices, labels)
    outer_record = OOFPredictionRecord.create(
        sample_index=outer.evaluation_indices[0],
        training_label=label_map[outer.evaluation_indices[0]],
        outer_fold_id=0,
        inner_fold_id=None,
        expert_id="A",
        training_seed=78,
        expert_training_membership_hash=outer.expert_training_membership_hash,
        checkpoint_path="synthetic/outer-checkpoint.pt",
        checkpoint_sha256="d" * 64,
        resolved_config={"expert": "A", "seed": 78, "outer": True},
        logits=np.zeros(3),
    )
    manager.validate_oof_record(outer_record)

    wrong_outer_record = OOFPredictionRecord.create(
        sample_index=outer.expert_training_indices[0],
        training_label=label_map[outer.expert_training_indices[0]],
        outer_fold_id=0,
        inner_fold_id=None,
        expert_id="A",
        training_seed=78,
        expert_training_membership_hash=outer.expert_training_membership_hash,
        checkpoint_path="synthetic/outer-checkpoint.pt",
        checkpoint_sha256="d" * 64,
        resolved_config={"expert": "A", "seed": 78, "outer": True},
        logits=np.zeros(3),
    )
    try:
        manager.validate_oof_record(wrong_outer_record)
    except OOFArtifactValidationError as exc:
        assert "training population" in str(exc)
    else:
        raise AssertionError("outer expert record accepted an outer-training sample")


def test_oof_record_rejects_inconsistent_sample_fold_and_membership_metadata():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    valid = _valid_oof_record(manager, indices, labels)

    cases = [
        (
            valid.__class__.create(
                sample_index=999,
                training_label=0,
                outer_fold_id=0,
                inner_fold_id=1,
                expert_id="A",
                training_seed=78,
                expert_training_membership_hash=valid.expert_training_membership_hash,
                checkpoint_path=valid.checkpoint_path,
                checkpoint_sha256=valid.checkpoint_sha256,
                resolved_config={"expert": "A", "seed": 78, "synthetic": True},
                logits=np.zeros(3),
            ),
            "canonical population",
        ),
        (
            OOFPredictionRecord.create(
                sample_index=valid.sample_index,
                training_label=valid.training_label,
                outer_fold_id=99,
                inner_fold_id=1,
                expert_id="A",
                training_seed=78,
                expert_training_membership_hash=valid.expert_training_membership_hash,
                checkpoint_path=valid.checkpoint_path,
                checkpoint_sha256=valid.checkpoint_sha256,
                resolved_config={"expert": "A", "seed": 78, "synthetic": True},
                logits=np.zeros(3),
            ),
            "outer fold",
        ),
        (
            OOFPredictionRecord.create(
                sample_index=valid.sample_index,
                training_label=valid.training_label,
                outer_fold_id=0,
                inner_fold_id=1,
                expert_id="A",
                training_seed=78,
                expert_training_membership_hash="b" * 64,
                checkpoint_path=valid.checkpoint_path,
                checkpoint_sha256=valid.checkpoint_sha256,
                resolved_config={"expert": "A", "seed": 78, "synthetic": True},
                logits=np.zeros(3),
            ),
            "membership hash",
        ),
    ]
    for record, message in cases:
        try:
            manager.validate_oof_record(record)
        except OOFArtifactValidationError as exc:
            assert message in str(exc), (message, exc)
        else:
            raise AssertionError(f"inconsistent OOF metadata was accepted: {message}")


def test_prediction_artifact_enforces_fixed_expert_order_and_provenance_consistency():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    record = _valid_oof_record(manager, indices, labels)
    artifact = OOFPredictionArtifact(
        expert_order=("A", "B"), num_classes=3, records=(record,)
    )
    artifact.validate(manager)

    try:
        OOFPredictionArtifact(
            expert_order=("B", "A"), num_classes=3, records=(record,)
        ).validate(manager)
    except OOFArtifactValidationError as exc:
        assert "expert ordering" in str(exc)
    else:
        raise AssertionError("inconsistent expert ordering was accepted")

    inconsistent = OOFPredictionRecord(
        schema_version=record.schema_version,
        sample_index=record.sample_index,
        training_label=record.training_label,
        outer_fold_id=record.outer_fold_id,
        inner_fold_id=record.inner_fold_id,
        expert_id=record.expert_id,
        training_seed=record.training_seed,
        expert_training_membership_hash=record.expert_training_membership_hash,
        checkpoint_path=record.checkpoint_path,
        checkpoint_sha256=record.checkpoint_sha256,
        resolved_config_json=record.resolved_config_json,
        resolved_config_sha256="c" * 64,
        logits=record.logits,
        features=record.features,
    )
    try:
        OOFPredictionArtifact(
            expert_order=("A", "B"), num_classes=3, records=(inconsistent,)
        ).validate(manager)
    except OOFArtifactValidationError as exc:
        assert "resolved configuration" in str(exc)
    else:
        raise AssertionError("inconsistent resolved configuration provenance was accepted")


def test_real_canonical_training_population_builds_without_test_access():
    manager = NestedOOFFoldManager.from_canonical_training_data(
        Path(_PROJECT_ROOT) / "data",
        seed=17,
    )

    assert len(manager.canonical_indices) == 10847
    assert len(manager.training_labels) == 10847
    artifact = Path(_PROJECT_ROOT) / "data" / "processed" / "lt_ir100_train_indices.npy"
    assert manager.canonical_training_index_sha256 == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert min(manager.canonical_class_counts) == 5
    assert [
        label for label, count in enumerate(manager.canonical_class_counts) if count == 5
    ] == [96, 97, 98, 99]
    assert manager.expert_order == ("CE", "LAL", "BalancedSoftmax", "Mixup")

    from scripts.base_trainer import compute_class_groups

    groups = compute_class_groups(np.asarray(manager.canonical_class_counts))
    assert tuple(len(groups[name]) for name in ("head", "medium", "tail")) == (
        35,
        35,
        30,
    )
    assert 69 in groups["medium"]
    for outer in manager.outer_folds:
        assert len(outer.evaluation_indices) in (2169, 2170)
        assert outer.evaluation_class_counts[96:] == (1, 1, 1, 1)
        for inner in outer.inner_folds:
            assert all(count > 0 for count in inner.expert_training_class_counts)


def test_manifest_rejects_tampered_membership_hash_and_reports_distributions():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    payload = manager.manifest().to_dict()
    payload["outer_folds"][0]["inner_folds"][0]["training_membership_sha256"] = "0" * 64
    try:
        FoldManifest.from_dict(payload)
    except Exception as exc:  # noqa: BLE001 - assert the public rejection contract
        assert "membership hash" in str(exc)
    else:
        raise AssertionError("tampered inner training membership hash was accepted")

    report = manager.distribution_report()
    assert report["canonical"]["class_counts"] == [10, 7, 5]
    assert len(report["outer"]) == 5
    assert report["outer"][0]["inner"][0]["held_out_class_counts"][2] == 1
    assert report["head_medium_tail"]["source"] == "scripts.base_trainer.compute_class_groups"


def test_manager_rejects_tampered_derived_class_counts():
    """Membership validation must not trust serialized class-count tuples."""
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    outer = manager.outer_fold(0)
    tampered = replace(
        outer,
        expert_training_class_counts=(999,) + outer.expert_training_class_counts[1:],
    )
    manager._outer_folds = (tampered,) + manager.outer_folds[1:]

    try:
        manager.validate_membership()
    except FoldMembershipError as exc:
        assert "training class counts" in str(exc)
    else:
        raise AssertionError("tampered outer class counts were accepted")


def test_manifest_rejects_router_indices_inconsistent_with_declared_inner_folds():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    payload = manager.manifest().to_dict()
    router = payload["outer_folds"][0]["router_development"]
    router["fit_indices"], router["selection_indices"] = (
        router["selection_indices"],
        router["fit_indices"],
    )
    router["fit_class_counts"], router["selection_class_counts"] = (
        router["selection_class_counts"],
        router["fit_class_counts"],
    )

    try:
        FoldManifest.from_dict(payload)
    except FoldMembershipError as exc:
        assert "indices do not match" in str(exc)
    else:
        raise AssertionError(
            "router indices inconsistent with declared inner folds were accepted"
        )


def test_complete_oof_artifact_requires_each_inner_sample_and_expert():
    indices, labels = _synthetic_population()
    manager = NestedOOFFoldManager(
        indices, labels, seed=17, num_classes=3, expert_order=("A", "B")
    )
    records = []
    for outer in manager.outer_folds:
        for inner in outer.inner_folds:
            for sample_index in inner.prediction_indices:
                for expert_id in manager.expert_order:
                    records.append(
                        _valid_oof_record(
                            manager,
                            indices,
                            labels,
                            sample_index=sample_index,
                            outer_fold_id=outer.outer_fold_id,
                            inner_fold_id=inner.inner_fold_id,
                            expert_id=expert_id,
                        )
                    )
    artifact = OOFPredictionArtifact(
        expert_order=manager.expert_order,
        num_classes=manager.num_classes,
        records=tuple(records),
    )
    artifact.validate(manager, require_complete=True)
    restored = OOFPredictionArtifact.from_json(artifact.to_json())
    restored.validate(manager, require_complete=True)

    incomplete = OOFPredictionArtifact(
        expert_order=manager.expert_order,
        num_classes=manager.num_classes,
        records=tuple(records[:-1]),
    )
    try:
        incomplete.validate(manager, require_complete=True)
    except OOFArtifactValidationError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete OOF artifact was accepted")

    outer = manager.outer_fold(0)
    labels_by_index = _label_map(indices, labels)
    outer_record = OOFPredictionRecord.create(
        sample_index=outer.evaluation_indices[0],
        training_label=labels_by_index[outer.evaluation_indices[0]],
        outer_fold_id=0,
        inner_fold_id=None,
        expert_id="A",
        training_seed=78,
        expert_training_membership_hash=outer.expert_training_membership_hash,
        checkpoint_path="synthetic/outer-checkpoint.pt",
        checkpoint_sha256="d" * 64,
        resolved_config={"expert": "A", "seed": 78, "outer": True},
        logits=np.zeros(3),
    )
    with_outer = OOFPredictionArtifact(
        expert_order=manager.expert_order,
        num_classes=manager.num_classes,
        records=tuple(records) + (outer_record,),
    )
    try:
        with_outer.validate(manager, require_complete=True)
    except OOFArtifactValidationError as exc:
        assert "unexpected rows" in str(exc)
    else:
        raise AssertionError(
            "complete inner OOF artifact accepted an extra outer-evaluation record"
        )


TESTS = [
    (
        "deterministic folds reconstruct the population",
        test_nested_folds_are_deterministic_and_reconstruct_the_population,
    ),
    ("different seeds change assignments", test_different_fold_seeds_change_the_assignment),
    (
        "inner and router partitions are complete",
        test_inner_oof_and_router_development_partitions_cover_outer_training_once,
    ),
    (
        "five-sample class is retained",
        test_five_sample_class_has_one_outer_holdout_and_survives_inner_training,
    ),
    (
        "impossible class coverage fails",
        test_impossible_rare_class_configuration_fails_clearly,
    ),
    (
        "manifest round trip",
        test_manifest_round_trips_and_records_membership_provenance,
    ),
    (
        "OOF prediction membership",
        test_oof_record_accepts_only_the_inner_prediction_population,
    ),
    (
        "outer expert versus inner OOF",
        test_outer_expert_records_are_distinct_from_inner_oof_records,
    ),
    (
        "OOF sample/fold/provenance validation",
        test_oof_record_rejects_inconsistent_sample_fold_and_membership_metadata,
    ),
    (
        "fixed expert order and provenance",
        test_prediction_artifact_enforces_fixed_expert_order_and_provenance_consistency,
    ),
    (
        "real canonical training manifest",
        test_real_canonical_training_population_builds_without_test_access,
    ),
    (
        "manifest tamper rejection and reporting",
        test_manifest_rejects_tampered_membership_hash_and_reports_distributions,
    ),
    ("derived class-count tamper rejection", test_manager_rejects_tampered_derived_class_counts),
    (
        "router membership tamper rejection",
        test_manifest_rejects_router_indices_inconsistent_with_declared_inner_folds,
    ),
    (
        "complete OOF artifact",
        test_complete_oof_artifact_requires_each_inner_sample_and_expert,
    ),
]


def main() -> int:
    passed = failed = 0
    for name, test in TESTS:
        try:
            test()
            print(f"  PASS {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - standalone test harness
            print(f"  FAIL {name}: {exc}")
            failed += 1
    print(f"nested OOF protocol: {passed} passed, {failed} failed")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
