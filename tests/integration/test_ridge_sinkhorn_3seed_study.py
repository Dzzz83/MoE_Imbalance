"""Focused synthetic checks for the three-seed study engine and artifacts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pathlib import Path as _TestPath
import sys as _test_sys
_TEST_PACKAGE_DIR = str(_TestPath(__file__).resolve().parent.parent)
if _TEST_PACKAGE_DIR not in _test_sys.path:
    _test_sys.path.insert(0, _TEST_PACKAGE_DIR)
from repo_root import REPO_ROOT
if str(REPO_ROOT) not in _test_sys.path:
    _test_sys.path.insert(0, str(REPO_ROOT))
_PROJECT_ROOT = str(REPO_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.nested_oof import NestedOOFFoldManager
from expert_method.ridge_sinkhorn.three_seed_study import (
    METHOD_IDS,
    STUDY_PREDICTION_IDS,
    FoldEvaluation,
    FoldStudyRunner,
    InferenceBatch,
    InnerOOFDataset,
    MarkdownReportBuilder,
    StudyArtifactRepository,
    StudyConfig,
    StudyError,
    StudyEvaluator,
    SelectiveBlendAllocationStrategy,
    _CandidateEvaluation,
    _candidate_sort_key,
    _canonical_json,
    _membership_hash,
    _sha256_text,
    classification_metrics,
)
from expert_method.ridge_sinkhorn.matrix import (
    STUDY_ID as MATRIX_STUDY_ID,
    OOFMatrixPlanner,
    build_job_inventory,
    canonical_json_bytes,
)


def _synthetic_manager() -> NestedOOFFoldManager:
    """Create a tiny canonical population with non-empty Head/Medium/Tail groups."""
    class_counts = (100, 20, 5, 5, 5, 5)
    labels = np.concatenate([
        np.full(count, class_id, dtype=np.int64)
        for class_id, count in enumerate(class_counts)
    ])
    return NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=6,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )


def _inner_dataset(seed: int = 78, outer: int = 0) -> InnerOOFDataset:
    rng = np.random.default_rng(seed + outer)
    sample_ids = np.arange(80, dtype=np.int64) + outer * 1000
    fold_ids = np.repeat(np.arange(4, dtype=np.int64), 20)
    labels = (np.arange(80, dtype=np.int64) * 7 + outer) % 6
    logits = rng.normal(0.0, 0.5, size=(80, 4, 6))
    logits[np.arange(80), :, labels] += 0.5
    sources = tuple((f"job-{index}/prediction", f"{index:064x}") for index in range(16))
    return InnerOOFDataset(
        logits, labels, sample_ids, fold_ids, seed, outer, sources,
    )


def _lock(runner: FoldStudyRunner | None = None):
    runner = runner or FoldStudyRunner()
    return runner.lock(
        _inner_dataset(), np.asarray((100, 20, 5, 5, 5, 5), dtype=np.int64),
        plan_sha256="a" * 64, source_commit="b" * 40,
    )


def _canonical_labels(manager: NestedOOFFoldManager) -> dict[int, int]:
    return dict(zip(manager.canonical_indices, manager.training_labels))


def _complete_synthetic_evaluations(manager: NestedOOFFoldManager) -> tuple[FoldEvaluation, ...]:
    labels_by_id = _canonical_labels(manager)
    counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    evaluations: list[FoldEvaluation] = []
    rng = np.random.default_rng(2026)
    for seed in (78, 88, 1034):
        for outer in range(5):
            ids = np.asarray(manager.outer_fold(outer).evaluation_indices, dtype=np.int64)
            labels = np.asarray([labels_by_id[int(value)] for value in ids], dtype=np.int64)
            predictions: dict[str, np.ndarray] = {}
            weights: dict[str, np.ndarray] = {}
            for method in STUDY_PREDICTION_IDS:
                predicted = labels.copy()
                if method not in {"uniform_logit", "fixed_007"}:
                    noise = rng.random(len(labels)) < 0.2
                    predicted[noise] = (predicted[noise] + 1) % 6
                predictions[method] = predicted
                weights[method] = np.full((len(ids), 4), 0.25, dtype=np.float64)
            metrics = tuple(
                (name, classification_metrics(labels, predictions[name], counts))
                for name in STUDY_PREDICTION_IDS
            )
            evaluations.append(FoldEvaluation(
                training_seed=seed, outer_fold_id=outer, sample_ids=ids, labels=labels,
                predictions=tuple(predictions.items()), weights=tuple(weights.items()),
                metrics=metrics,
                selected_configurations=tuple((method, {"synthetic": True}) for method in METHOD_IDS),
                sinkhorn_diagnostics=tuple(
                    (method, {"fold_diagnostics": [{"converged": True}]})
                    for method in (
                        "contribution_ridge_sinkhorn", "residual_ridge_sinkhorn",
                        "selective_residual_ridge_sinkhorn",
                    )
                ),
                source_hashes=((f"seed-{seed}-outer-{outer}", "c" * 64),),
                lock_sha256=f"{seed:03d}{outer:01d}".ljust(64, "d"),
            ))
    return tuple(evaluations)


class _SyntheticArtifactView:
    """Small validated-view stand-in that records access ordering for CLI tests."""

    def __init__(self, manager: NestedOOFFoldManager, artifact_root: Path) -> None:
        self.manager = manager
        self.planner = SimpleNamespace(
            inventory=build_job_inventory(manager),
            validate_complete_lock_matrix=lambda frozen=None: self.events.append(("view_lock", frozen)),
        )
        self.artifact_root = artifact_root
        self.events: list[tuple] = []
        self.labels_by_id = _canonical_labels(manager)
        self.jobs = {job.job_id: job for job in self.planner.inventory}
        self.references = {job_id: self._reference(job) for job_id, job in self.jobs.items()}

    @staticmethod
    def _digest(job_id: str, component: str) -> str:
        import hashlib

        return hashlib.sha256(f"{job_id}:{component}".encode()).hexdigest()

    def _reference(self, job):
        return SimpleNamespace(
            job=job,
            checkpoint_sha256=self._digest(job.job_id, "checkpoint"),
            prediction_sha256=self._digest(job.job_id, "prediction"),
            resolved_config_sha256=self._digest(job.job_id, "config"),
        )

    def resolve(self, job_id: str):
        self.events.append(("resolve", job_id))
        return self.references[job_id]

    def _prediction(self, job, sample_ids: np.ndarray):
        rng = np.random.default_rng(
            job.training_seed + 100 * job.outer_fold_id + 10 * (job.inner_fold_id or 0)
            + (0 if job.expert_key == "ce" else 1 if job.expert_key == "logit_adjusted" else 2 if job.expert_key == "balanced_softmax" else 3)
        )
        labels = np.asarray([self.labels_by_id[int(value)] for value in sample_ids], dtype=np.int64)
        logits = rng.normal(size=(len(sample_ids), 6))
        logits[np.arange(len(sample_ids)), labels] += 0.3
        return logits, labels

    def load_logits(self, job_id: str):
        job = self.jobs[job_id]
        ids = np.asarray(self.manager.inner_fold(job.outer_fold_id, job.inner_fold_id).prediction_indices, dtype=np.int64)
        logits, labels = self._prediction(job, ids)
        reference = self.references[job_id]
        self.events.append(("inner_logits", job_id))
        return SimpleNamespace(
            job_id=job_id, stage="inner", expert_name=job.expert_name,
            training_seed=job.training_seed, outer_fold_id=job.outer_fold_id,
            inner_fold_id=job.inner_fold_id, sample_ids=ids, labels=labels, logits=logits,
            checkpoint_sha256=reference.checkpoint_sha256,
            prediction_sha256=reference.prediction_sha256,
            resolved_config_sha256=reference.resolved_config_sha256,
        )

    def load_outer_logits(self, job_id: str):
        job = self.jobs[job_id]
        ids = np.asarray(self.manager.outer_fold(job.outer_fold_id).evaluation_indices, dtype=np.int64)
        logits, _labels = self._prediction(job, ids)
        reference = self.references[job_id]
        self.events.append(("outer_logits", job_id))
        return SimpleNamespace(
            job_id=job_id, stage="outer", expert_name=job.expert_name,
            training_seed=job.training_seed, outer_fold_id=job.outer_fold_id,
            sample_ids=ids, logits=logits,
            checkpoint_sha256=reference.checkpoint_sha256,
            prediction_sha256=reference.prediction_sha256,
            resolved_config_sha256=reference.resolved_config_sha256,
        )

    def load_outer_labels(self, job_id: str):
        job = self.jobs[job_id]
        ids = np.asarray(self.manager.outer_fold(job.outer_fold_id).evaluation_indices, dtype=np.int64)
        labels = np.asarray([self.labels_by_id[int(value)] for value in ids], dtype=np.int64)
        self.events.append(("outer_labels", job_id))
        return ids, labels


def test_frozen_configuration_and_smoothed_priors() -> None:
    config = StudyConfig()
    assert config.bootstrap_replicates == 10_000
    assert config.bootstrap_seed == 20260924
    priors = dict(config.priors)
    assert tuple(priors) == ("uniform", "fixed_006", "fixed_007", "fixed_010", "fixed_011")
    np.testing.assert_allclose(priors["fixed_007"], 0.95 * np.asarray((0.0, 0.25, 0.5, 0.25)) + 0.05 / 4)
    with pytest.raises(StudyError, match="frozen"):
        StudyConfig(bootstrap_replicates=10)


def test_candidate_selection_uses_maximin_then_frozen_tie_breaks() -> None:
    baseline = {"balanced_accuracy": 0.5, "tail_accuracy": 0.3}

    def candidate(candidate_id: str, ba: float, tail: float, **parameters) -> _CandidateEvaluation:
        return _CandidateEvaluation(candidate_id, parameters, {
            "balanced_accuracy": ba, "tail_accuracy": tail,
        }, np.ones((1, 4)) / 4, np.zeros(1, dtype=np.int64))

    high_ba_low_tail = candidate("ba", 0.9, 0.2, alpha=1000.0, gamma=0.0, temperature=1.0, shrinkage=1.0)
    balanced = candidate("balanced", 0.6, 0.4, alpha=0.1, gamma=0.0, temperature=1.0, shrinkage=1.0)
    assert _candidate_sort_key(balanced, baseline, "contribution", phase="base") < _candidate_sort_key(
        high_ba_low_tail, baseline, "contribution", phase="base",
    )

    more_regularized = candidate("alpha1000", 0.6, 0.4, alpha=1000.0, gamma=1.0, temperature=1.0, shrinkage=0.75)
    less_regularized = candidate("alpha10", 0.6, 0.4, alpha=10.0, gamma=0.0, temperature=1.0, shrinkage=0.5)
    assert _candidate_sort_key(more_regularized, baseline, "contribution", phase="base") < _candidate_sort_key(
        less_regularized, baseline, "contribution", phase="base",
    )
    weaker_route = candidate("shrink05", 0.6, 0.4, alpha=10.0, gamma=0.0, temperature=2.0, shrinkage=0.5)
    stronger_route = candidate("shrink1", 0.6, 0.4, alpha=10.0, gamma=0.0, temperature=1.0, shrinkage=1.0)
    assert _candidate_sort_key(weaker_route, baseline, "contribution", phase="base") < _candidate_sort_key(
        stronger_route, baseline, "contribution", phase="base",
    )


def test_selective_blend_tau_endpoints() -> None:
    anchor = np.asarray((0.0, 0.25, 0.5, 0.25))
    ridge = np.stack((anchor, np.full(4, 0.25)))
    sinkhorn = np.asarray(((0.25, 0.25, 0.25, 0.25), (0.0, 0.0, 0.0, 1.0)))
    packed = np.concatenate((ridge, sinkhorn), axis=1)

    tau_small = SelectiveBlendAllocationStrategy(0.1).fit(ridge, sinkhorn).transform(packed)
    tau_large = SelectiveBlendAllocationStrategy(1.0).fit(ridge, sinkhorn).transform(packed)
    np.testing.assert_allclose(tau_small[0], anchor)
    np.testing.assert_allclose(tau_large[0], anchor)
    np.testing.assert_allclose(tau_small[1], sinkhorn[1])
    np.testing.assert_allclose(tau_large[1], 0.5 * ridge[1] + 0.5 * sinkhorn[1])


def test_inference_batch_has_no_label_channel() -> None:
    with pytest.raises(TypeError):
        InferenceBatch(np.ones((2, 4, 6)), np.asarray((0, 1)), labels=np.asarray((0, 1)))


def test_analysis_and_matrix_clis_use_the_same_reuse_root_name(tmp_path: Path) -> None:
    from expert_method.ridge_sinkhorn.matrix import parse_reuse_roots
    from scripts.run_ridge_sinkhorn_3seed import _parse_reuse_root

    path = tmp_path / "rs3-reuse"
    expected = parse_reuse_roots((str(path),))
    actual: dict[str, Path] = {}
    _parse_reuse_root(str(path), actual)
    assert actual == expected


def test_locked_prediction_is_batch_and_outer_label_invariant() -> None:
    lock = _lock()
    evaluator = StudyEvaluator()
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(12, 4, 6))
    ids = np.arange(12, dtype=np.int64)
    full_predictions, full_weights = evaluator.predict_locked_methods(lock, InferenceBatch(logits, ids))
    split_predictions: dict[str, list[np.ndarray]] = {name: [] for name in STUDY_PREDICTION_IDS}
    split_weights: dict[str, list[np.ndarray]] = {name: [] for name in STUDY_PREDICTION_IDS}
    for rows in (slice(0, 3), slice(3, 8), slice(8, 12)):
        part_predictions, part_weights = evaluator.predict_locked_methods(
            lock, InferenceBatch(logits[rows], ids[rows]),
        )
        for name in STUDY_PREDICTION_IDS:
            split_predictions[name].append(part_predictions[name])
            split_weights[name].append(part_weights[name])
    for name in STUDY_PREDICTION_IDS:
        np.testing.assert_array_equal(full_predictions[name], np.concatenate(split_predictions[name]))
        np.testing.assert_allclose(full_weights[name], np.concatenate(split_weights[name]), rtol=0, atol=1e-12)

    changed_labels = (np.arange(12, dtype=np.int64) + 1) % 6
    changed = evaluator.evaluate_fold(
        lock, InferenceBatch(logits, ids), changed_labels,
        np.asarray((100, 20, 5, 5, 5, 5)), source_hashes=(("outer", "e" * 64),),
        expected_outer_sample_ids=ids,
    )
    for name in STUDY_PREDICTION_IDS:
        np.testing.assert_array_equal(full_predictions[name], dict(changed.predictions)[name])
    assert lock.lock_sha256 == _lock().lock_sha256
    assert all(item.sinkhorn_diagnostics for item in lock.selected if "sinkhorn" in item.method_id)


def test_canonical_fold_membership_validator_rejects_a_repartition(tmp_path: Path) -> None:
    from scripts.run_ridge_sinkhorn_3seed import _validate_fold_lock_membership

    manager = _synthetic_manager()
    ids = np.asarray(manager.outer_fold(0).expert_training_indices, dtype=np.int64)
    fold_ids = np.empty(len(ids), dtype=np.int64)
    positions = {int(value): index for index, value in enumerate(ids)}
    for fold in range(4):
        fold_ids[[positions[int(value)] for value in manager.inner_fold(0, fold).prediction_indices]] = fold
    labels_by_id = _canonical_labels(manager)
    dataset = InnerOOFDataset(
        np.random.default_rng(3).normal(size=(len(ids), 4, 6)),
        np.asarray([labels_by_id[int(value)] for value in ids]), ids, fold_ids, 78, 0,
        tuple((f"source-{i}", f"{i:064x}") for i in range(16)),
    )
    runner = FoldStudyRunner()
    valid_lock = runner.lock(
        dataset, np.asarray(manager.canonical_class_counts),
        plan_sha256="a" * 64, source_commit="b" * 40,
    )
    _validate_fold_lock_membership(valid_lock, manager)
    first = valid_lock.inner_fold_memberships[0]
    mutated_ids = (first.validation_sample_ids[0] + 1,) + first.validation_sample_ids[1:]
    mutated = replace(
        first,
        validation_sample_ids=mutated_ids,
        validation_membership_sha256=_membership_hash(np.asarray(mutated_ids, dtype=np.int64)),
    )
    bad_memberships = (mutated, *valid_lock.inner_fold_memberships[1:])
    bad = replace(valid_lock, inner_fold_memberships=bad_memberships, lock_sha256="")
    from expert_method.ridge_sinkhorn.three_seed_study import _sha256_text

    bad = replace(bad, lock_sha256=_sha256_text(_canonical_json(bad.payload(include_digest=False))))
    with pytest.raises(StudyError, match="canonical inner fold"):
        _validate_fold_lock_membership(bad, manager)


def test_matrix_manifest_binds_every_canonical_job_record(tmp_path: Path) -> None:
    from scripts.run_ridge_sinkhorn_3seed import _read_frozen_matrix_identity

    manager = _synthetic_manager()
    planner = OOFMatrixPlanner(manager=manager, artifact_root=tmp_path)
    manifest = planner.frozen_manifest()
    study_dir = tmp_path / MATRIX_STUDY_ID
    study_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = study_dir / "job_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    loaded, _digest = _read_frozen_matrix_identity(
        tmp_path, StudyConfig(), expected_manifest=manifest,
    )
    assert loaded["inventory"] == manifest["inventory"]

    corrupt = dict(manifest)
    corrupt_inventory = [dict(item) for item in manifest["inventory"]]
    corrupt_inventory[0]["training_seed"] = -1
    corrupt["inventory"] = corrupt_inventory
    manifest_path.write_bytes(canonical_json_bytes(corrupt))
    with pytest.raises(StudyError, match="inventory"):
        _read_frozen_matrix_identity(tmp_path, StudyConfig(), expected_manifest=manifest)


def test_study_lock_persists_full_sharded_inventory_and_rejects_mismatch(tmp_path: Path) -> None:
    manager = _synthetic_manager()
    planner = OOFMatrixPlanner(manager=manager, artifact_root=tmp_path)
    manifest = planner.frozen_manifest()
    manifest_bytes = canonical_json_bytes(manifest)
    job_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    repository = StudyArtifactRepository(tmp_path / "good")
    inventory = manifest["inventory"]
    repository.write_study_lock(
        plan_sha256=manifest["plan_sha256"], source_commit=manifest["source_commit"],
        job_manifest_sha256=job_manifest_sha256, job_inventory=inventory,
    )
    repository.write_study_lock(
        plan_sha256=manifest["plan_sha256"], source_commit=manifest["source_commit"],
        job_manifest_sha256=job_manifest_sha256, job_inventory=inventory,
    )
    payload = json.loads(repository.study_lock_path.read_text())
    assert len(payload["job_inventory"]) == 300
    assert payload["job_inventory_sha256"]
    assert payload["job_inventory"][0]["shard_key"]

    bad_root = tmp_path / "bad"
    bad_repository = StudyArtifactRepository(bad_root)
    bad_inventory = [dict(item) for item in inventory]
    bad_inventory[0]["run_relative_path"] = "incorrect/source/path"
    bad_repository.write_study_lock(
        plan_sha256=manifest["plan_sha256"], source_commit=manifest["source_commit"],
        job_manifest_sha256=job_manifest_sha256, job_inventory=bad_inventory,
    )
    manifest_path = bad_root / MATRIX_STUDY_ID / "job_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    with pytest.raises(StudyError, match="inventory/protocol"):
        bad_repository.validate_complete_lock_matrix(
            plan_sha256=manifest["plan_sha256"], source_commit=manifest["source_commit"],
        )


def test_outer_stage_prerequisite_gate_validates_all_300_after_locks(tmp_path: Path) -> None:
    from scripts.run_ridge_sinkhorn_3seed import (
        _reference_source_hashes,
        _validate_outer_stage_prerequisites,
    )

    manager = _synthetic_manager()
    view = _SyntheticArtifactView(manager, tmp_path)
    locks = []
    for seed in (78, 88, 1034):
        for outer in range(5):
            memberships = tuple(SimpleNamespace(
                inner_fold_id=inner,
                validation_sample_ids=manager.inner_fold(outer, inner).prediction_indices,
            ) for inner in range(4))
            source_hashes = tuple(sorted(
                source_hash
                for job in view.planner.inventory
                if job.stage == "inner" and job.training_seed == seed and job.outer_fold_id == outer
                for source_hash in _reference_source_hashes(view.references[job.job_id])
            ))
            locks.append(SimpleNamespace(
                training_seed=seed, outer_fold_id=outer,
                inner_fold_memberships=memberships, source_hashes=source_hashes,
            ))

    class LockRepository:
        def validate_complete_lock_matrix(self, **_kwargs):
            view.events.append(("repository_locks",))
            return tuple(locks)

    manifest = {"plan_sha256": "a" * 64, "source_commit": "b" * 40}
    result_locks, references = _validate_outer_stage_prerequisites(
        view=view, manager=manager, repository=LockRepository(), manifest=manifest,
    )
    assert len(result_locks) == 15
    assert len(references) == 300
    assert len([event for event in view.events if event[0] == "resolve"]) == 300
    assert view.events[0][0] == "repository_locks"
    assert view.events[1] == ("view_lock", manifest)
    assert not any(event[0] in {"outer_logits", "outer_labels"} for event in view.events)


def test_evaluation_npz_is_hash_bound_and_resume_is_idempotent(tmp_path: Path) -> None:
    lock = _lock()
    repository = StudyArtifactRepository(tmp_path)
    lock_path = repository.write_fold_lock(lock)
    repository.write_fold_lock(lock)
    loaded_lock = repository.read_fold_lock(78, 0)
    assert loaded_lock.lock_sha256 == lock.lock_sha256
    assert json.loads(lock_path.read_text()) == lock.payload()
    evaluator = StudyEvaluator()
    ids = np.arange(16, dtype=np.int64)
    logits = np.random.default_rng(5).normal(size=(16, 4, 6))
    labels = np.arange(16, dtype=np.int64) % 6
    evaluation = evaluator.evaluate_fold(
        lock, InferenceBatch(logits, ids), labels,
        np.asarray((100, 20, 5, 5, 5, 5)), source_hashes=(("outer-source", "f" * 64),),
        expected_outer_sample_ids=ids,
    )
    json_path, arrays_path = repository.write_fold_evaluation(
        evaluation, expected_outer_sample_ids=ids,
    )
    repository.write_fold_evaluation(evaluation, expected_outer_sample_ids=ids)
    loaded = repository.read_fold_evaluation(78, 0, expected_outer_sample_ids=ids)
    assert loaded.lock_sha256 == lock.lock_sha256
    metadata = json.loads(json_path.read_text())
    assert metadata["prediction_arrays"]["sha256"]
    assert set(metadata["selected_configurations"]) == set(METHOD_IDS)
    assert "fold_diagnostics" in metadata["sinkhorn_diagnostics"]["residual_ridge_sinkhorn"]
    assert arrays_path.is_file()
    contents = arrays_path.read_bytes()
    arrays_path.write_bytes(contents[:-1] + bytes((contents[-1] ^ 0x01,)))
    with pytest.raises(StudyError, match="SHA-256"):
        repository.read_fold_evaluation(78, 0, expected_outer_sample_ids=ids)


def test_aggregate_recomputes_metrics_and_report_includes_mass_and_selections(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _synthetic_manager()
    evaluations = list(_complete_synthetic_evaluations(manager))
    evaluator = StudyEvaluator()
    assert evaluator.config.bootstrap_replicates == 10_000
    monkeypatch.setattr(
        evaluator, "_hierarchical_bootstrap",
        lambda _predictions, _labels, _counts: {"test-seam": {"balanced_accuracy": [0.0, 1.0]}},
    )
    expected_outer = {
        fold: np.asarray(manager.outer_fold(fold).evaluation_indices, dtype=np.int64)
        for fold in range(5)
    }
    result = evaluator.aggregate(
        evaluations, np.asarray(manager.canonical_indices, dtype=np.int64),
        np.asarray(manager.canonical_class_counts, dtype=np.int64), expected_outer,
    )
    report = MarkdownReportBuilder().build(result)
    assert "Mean expert mass across seeds" in report
    assert "Selected settings and frozen-price Sinkhorn convergence diagnostics" in report
    assert "CE | LAL | BalancedSoftmax | Mixup" in report
    assert "test-seam" in result.bootstrap_intervals

    broken = evaluations[0]
    wrong_metrics = dict(broken.metrics)
    wrong_metrics["uniform_logit"] = {**wrong_metrics["uniform_logit"], "ordinary_accuracy": 0.123}
    evaluations[0] = replace(broken, metrics=tuple(wrong_metrics.items()))
    with pytest.raises(StudyError, match="recomputed"):
        evaluator.aggregate(
            evaluations, np.asarray(manager.canonical_indices, dtype=np.int64),
            np.asarray(manager.canonical_class_counts, dtype=np.int64), expected_outer,
        )
