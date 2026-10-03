"""Real artifact-reader safety checks for retrospective inner diagnostics.

The fixture keeps the matrix shape required by the public reader (300 jobs,
240 inner predictions, 16 historical sources, and 15 locks) while replacing
the population-size guard with a small 100-class nested fold manifest. The
checkpoint model-state compatibility check is also bypassed because these
synthetic runs intentionally contain metadata-only checkpoint states.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data.nested_oof import (
    NestedOOFFoldManager,
    OOFPredictionArtifact,
    OOFPredictionRecord,
)
from expert_method import cli as study_cli
from expert_method.config import load_study
from expert_method.diagnostics import DiagnosticsError, InnerArtifactReader, StudyDiagnosticsRunner
from expert_method.diagnostics import artifacts as diagnostics_artifacts
from expert_method.diagnostics.artifacts import STUDY_ID
from expert_method.ridge_sinkhorn.matrix import (
    COMPATIBILITY_SCHEMA_VERSION,
    HISTORICAL_INNER_EXPERIMENT,
    HISTORICAL_PILOT_EXPERIMENT,
    OOFMatrixPlanner,
    OOFArtifactStore,
    canonical_json_bytes,
)
from expert_method.ridge_sinkhorn.three_seed_study import (
    ANCHOR,
    InferenceBatch,
    METHOD_IDS,
    ContributionRidgeStrategy,
    FoldLock,
    FrozenPriceSinkhornStrategy,
    InnerCVSelector,
    InnerFoldMembership,
    MethodSelection,
    ResidualRidgeStrategy,
    StudyArtifactRepository,
    StudyEvaluator,
    _canonical_json,
    _membership_hash,
    _sha256_text,
)
from scripts.config import TrainingConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/studies/ridge_sinkhorn_3seed_v1.yaml"
TRAINING_COMMIT = "b" * 40
SNAPSHOT_COMMIT = "d" * 40
ROOT_NAME = "rs3"


def _synthetic_manager() -> NestedOOFFoldManager:
    """Build a tiny, fully class-covered five-by-four nested-fold manifest."""
    labels = np.repeat(np.arange(100, dtype=np.int64), 5)
    return NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=100,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )


def _training_configs(definition: Any) -> dict[str, TrainingConfig]:
    repository_root = definition.source_path.parents[2]
    return {
        expert: TrainingConfig.from_file(repository_root / relative_path)
        for expert, relative_path in definition.expert_config_paths.items()
    }


@contextmanager
def _synthetic_reader_contracts() -> Iterator[None]:
    """Relax only population/model-architecture checks for tiny synthetic data."""
    def validate_small_manager(manager: NestedOOFFoldManager, protocol: Mapping[str, Any]) -> None:
        assert manager.num_classes == int(protocol["num_classes"]) == 100
        assert manager.fold_generation_seed == int(protocol["fold_generation_seed"]) == 42
        assert manager.outer_fold_count == int(protocol["outer_folds"]) == 5
        assert manager.inner_fold_count == int(protocol["inner_folds"]) == 4
        assert tuple(manager.expert_order) == tuple(protocol["expert_order"])
        assert manager.canonical_class_counts == (5,) * 100

    def skip_model_architecture_check(*_args: Any, **_kwargs: Any) -> None:
        return None

    with (
        patch.object(diagnostics_artifacts, "_validate_manager_protocol", validate_small_manager),
        patch.object(OOFArtifactStore, "_validate_model_state_compatibility", staticmethod(skip_model_architecture_check)),
    ):
        yield


@contextmanager
def _synthetic_fixture_contracts() -> Iterator[None]:
    """Also stamp a distinct training commit while constructing frozen files."""
    original_manifest = OOFMatrixPlanner.frozen_manifest

    def stable_training_identity(planner: OOFMatrixPlanner) -> dict[str, Any]:
        manifest = original_manifest(planner)
        manifest["source_commit"] = TRAINING_COMMIT
        manifest["source_tree_dirty"] = False
        return manifest

    with _synthetic_reader_contracts(), patch.object(
        OOFMatrixPlanner, "frozen_manifest", stable_training_identity,
    ):
        yield


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pretty_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_synthetic_run(
    *,
    planner: OOFMatrixPlanner,
    job: Any,
    output_root: Path,
    experiment_id: str,
    store_cache: dict[tuple[Path, str], OOFArtifactStore],
) -> tuple[Path, Path, Path]:
    """Write one full, ordinary OOF artifact through the production store."""
    context = job.run_spec(experiment_id=experiment_id, epochs=200).resolve(planner.manager)
    root = output_root.resolve()
    cache_key = (root, experiment_id)
    store = store_cache.get(cache_key)
    if store is None:
        store = OOFArtifactStore(
            root=root,
            manager=planner.manager,
            experiment_id=experiment_id,
            canonical_checkpoint_dir=root.parent / "canonical-checkpoints",
        )
        store_cache[cache_key] = store

    resolved_config = planner.resolved_config(job)
    # A source run keeps its own checkpoint directory; that machine-specific
    # value is deliberately normalized by the planner's recipe validator.
    resolved_config = json.loads(_canonical_json(resolved_config))
    resolved_config["checkpoint"]["dir"] = str(store.run_dir(context) / "checkpoints")
    store.prepare_run(context, resolved_config)
    checkpoint_path = store.checkpoint_path(context)
    torch.save({
        "expert_name": context.expert_name,
        "seed": context.training_seed,
        "epoch": 200,
        "is_final": True,
        "model_state_dict": {},
    }, checkpoint_path)
    checkpoint_sha256 = store.record_checkpoint(context, checkpoint_path)

    labels_by_id = dict(zip(planner.manager.canonical_indices, planner.manager.training_labels))
    records = tuple(
        OOFPredictionRecord.create(
            sample_index=int(sample_id),
            training_label=int(labels_by_id[int(sample_id)]),
            outer_fold_id=context.outer_fold_id,
            inner_fold_id=context.inner_fold_id,
            expert_id=context.expert_name,
            training_seed=context.training_seed,
            expert_training_membership_hash=context.training_membership_hash,
            checkpoint_path=str(checkpoint_path.resolve()),
            checkpoint_sha256=checkpoint_sha256,
            resolved_config=resolved_config,
            logits=np.zeros(100, dtype=np.float64),
        )
        for sample_id in context.prediction_indices
    )
    artifact = OOFPredictionArtifact(
        expert_order=planner.manager.expert_order,
        num_classes=100,
        records=records,
    )
    store.write_predictions(context, artifact)
    return (
        store.checkpoint_path(context),
        store.prediction_path(context),
        store.metadata_path(context),
    )


def _locked_states() -> tuple[dict[str, Any], dict[str, str]]:
    contribution = {
        "kind": "contribution_ridge", "feature_set": "confidence_only",
        "alpha": 0.1, "gamma": 0.0, "temperature": 1.0, "shrinkage": 1.0,
        "num_classes": 100, "scaler_mean": [0.0] * 4,
        "scaler_scale": [1.0] * 4, "coef": [[0.0] * 4 for _ in range(4)],
        "intercept": [0.0] * 4,
    }
    residual = {
        "kind": "residual_ridge", "feature_set": "confidence_only",
        "alpha": 0.1, "gamma": 0.0, "penalty": 1.0, "scale": 1.0,
        "num_classes": 100, "anchor": list(ANCHOR),
        "scaler_mean": [0.0] * 4, "scaler_scale": [1.0] * 4,
        "coef": [[0.0] * 4 for _ in range(4)], "intercept": [0.0] * 4,
        "oracle_diagnostics": {},
    }
    prices = {
        "rho": 1.0, "q": [0.25] * 4, "log_prices": [0.0] * 4,
        "diagnostics": {"converged": True}, "fit_sample_count": 1,
    }
    states: dict[str, Any] = {
        "contribution": contribution,
        "residual": residual,
        "contribution_prices": prices,
        "residual_prices": prices,
        "prior_only": {
            "prior_name": "uniform", "weights": [0.25] * 4,
            "selected_metrics": {"balanced_accuracy": 0.0, "tail_accuracy": 0.0},
        },
    }
    hashes = {name: _sha256_text(_canonical_json(value)) for name, value in states.items()}
    return states, hashes


def _selections() -> tuple[tuple[MethodSelection, ...], tuple[MethodSelection, ...]]:
    contribution = {"alpha": 0.1, "gamma": 0.0, "temperature": 1.0, "shrinkage": 1.0}
    residual = {"penalty": 1.0, "alpha": 0.1, "gamma": 0.0, "scale": 1.0}
    configurations = (
        contribution,
        {**contribution, "prior": "uniform", "rho": 1.0},
        residual,
        {**residual, "prior": "uniform", "rho": 1.0},
        {**residual, "prior": "uniform", "rho": 1.0, "tau": 0.25},
    )
    recorded_metrics = tuple(
        (name, 0.0) for name in (
            "balanced_accuracy", "head_accuracy", "medium_accuracy",
            "tail_accuracy", "ordinary_accuracy",
        )
    )
    selected = tuple(
        MethodSelection(
            method_id=method_id,
            configuration=tuple(sorted(configuration.items())),
            metrics=recorded_metrics,
            maximin_delta=0.0,
            candidate_id=f"synthetic-{method_id}",
        )
        for method_id, configuration in zip(METHOD_IDS, configurations)
    )
    controls = (MethodSelection(
        method_id="prior_only_control",
        configuration=(("prior", "uniform"),),
        metrics=recorded_metrics,
        maximin_delta=0.0,
        candidate_id="synthetic-prior-only",
    ),)
    return selected, controls


@pytest.fixture(scope="module")
def real_inner_artifact_tree(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Create real persisted runs and locks for the actual read-only adapter."""
    root = tmp_path_factory.mktemp("inner-diagnostics-real-reader")
    artifact_root = root / "result-snapshot"
    reuse_root = root / "reuse" / ROOT_NAME
    artifact_root.mkdir(parents=True)
    reuse_root.mkdir(parents=True)
    (artifact_root / diagnostics_artifacts.SNAPSHOT_MARKER_NAME).write_text(
        json.dumps({
            "schema_version": "diagnostics_snapshot.v1",
            "snapshot_commit": SNAPSHOT_COMMIT,
            "file_hashes": {},
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    definition = load_study(CONFIG_PATH)
    config = definition.to_study_config()
    manager = _synthetic_manager()
    training_configs = _training_configs(definition)
    with _synthetic_fixture_contracts():
        planner = OOFMatrixPlanner(
            manager=manager,
            study_id=STUDY_ID,
            artifact_root=artifact_root,
            reuse_roots={ROOT_NAME: reuse_root},
            training_configs=training_configs,
            study_config=config,
            training_epochs=200,
            read_only=False,
        )
        store_cache: dict[tuple[Path, str], OOFArtifactStore] = {}
        paths_by_job: dict[str, tuple[Path, Path, Path]] = {}
        for job in planner.inventory:
            if job.stage != "inner":
                continue
            if job.training_seed == 78 and job.outer_fold_id == 0:
                experiment_id = (
                    HISTORICAL_PILOT_EXPERIMENT
                    if job.expert_key == "ce" and job.inner_fold_id == 0
                    else HISTORICAL_INNER_EXPERIMENT
                )
                source_root = reuse_root
            else:
                experiment_id = STUDY_ID
                source_root = artifact_root
            paths_by_job[job.job_id] = _write_synthetic_run(
                planner=planner,
                job=job,
                output_root=source_root,
                experiment_id=experiment_id,
                store_cache=store_cache,
            )

        reuse_rows = [
            dict(row) for row in planner.audit_compatibility(
                tuple(job for job in planner.inventory if job.stage == "inner")
            )
        ]
        assert len(reuse_rows) == 16
        inventory = study_cli._inventory_payload(planner.inventory)
        base_manifest = planner.frozen_manifest()
        freeze_payload: dict[str, Any] = {
            "schema_version": study_cli._FREEZE_SCHEMA,
            "study_id": STUDY_ID,
            "source_commit": TRAINING_COMMIT,
            "source_tree_dirty": False,
            "plan_sha256": base_manifest["plan_sha256"],
            "study_config_sha256": definition.scientific_sha256,
            "study_config_yaml_sha256": hashlib.sha256(definition.source_path.read_bytes()).hexdigest(),
            "resolved_study_config": json.loads(definition.canonical_scientific_json),
            "expert_config_sha256": study_cli._expert_config_hashes(definition),
            "protocol_config_sha256": config.sha256,
            "fold_manifest_sha256": base_manifest["fold_manifest_sha256"],
            "inventory_count": len(inventory),
            "inventory": inventory,
            "job_inventory_sha256": hashlib.sha256(_canonical_json(inventory).encode("utf-8")).hexdigest(),
            "shard_rule": base_manifest["shard_rule"],
            "reuse_decisions": {
                "inner": reuse_rows,
                "outer": "deferred_until_all_inner_locks_are_valid",
            },
            "runtime_root_names": [ROOT_NAME],
        }
        freeze_payload["freeze_sha256"] = hashlib.sha256(
            _canonical_json(freeze_payload).encode("utf-8")
        ).hexdigest()
        study_root = artifact_root / STUDY_ID
        (study_root / "manifests").mkdir(parents=True, exist_ok=True)
        (study_root / "manifests" / "study-freeze.json").write_text(
            _canonical_json(freeze_payload) + "\n", encoding="utf-8",
        )
        planner.freeze_sha256 = freeze_payload["freeze_sha256"]
        frozen_manifest = planner.frozen_manifest()
        assert frozen_manifest["inventory"] == inventory
        (study_root / "job_manifest.json").write_bytes(canonical_json_bytes(frozen_manifest))
        (study_root / "fold_manifest.json").write_text(
            manager.manifest().to_json(), encoding="utf-8",
        )
        (study_root / "reuse_compatibility_inner.json").write_bytes(canonical_json_bytes({
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "study_freeze_sha256": freeze_payload["freeze_sha256"],
            "stage": "inner",
            "rows": reuse_rows,
        }))

        repository = StudyArtifactRepository(artifact_root, config)
        repository.write_study_lock(
            plan_sha256=frozen_manifest["plan_sha256"],
            source_commit=TRAINING_COMMIT,
            job_manifest_sha256=_sha256(study_root / "job_manifest.json"),
            job_inventory=inventory,
        )
        states, state_hashes = _locked_states()
        selected, controls = _selections()
        for seed in config.seeds:
            for outer in config.outer_folds:
                memberships = []
                all_validation_ids: list[int] = []
                for inner in config.inner_folds:
                    validation_ids = tuple(
                        int(value) for value in manager.inner_fold(outer, inner).prediction_indices
                    )
                    train_ids = tuple(
                        value for other in config.inner_folds if other != inner
                        for value in manager.inner_fold(outer, other).prediction_indices
                    )
                    memberships.append(InnerFoldMembership(
                        inner_fold_id=inner,
                        validation_sample_ids=validation_ids,
                        validation_membership_sha256=_membership_hash(np.asarray(validation_ids, dtype=np.int64)),
                        router_training_membership_sha256=_membership_hash(np.asarray(train_ids, dtype=np.int64)),
                        router_training_sample_count=len(train_ids),
                    ))
                    all_validation_ids.extend(validation_ids)

                source_hashes = []
                for job in planner.inventory:
                    if job.stage != "inner" or (job.training_seed, job.outer_fold_id) != (seed, outer):
                        continue
                    checkpoint_path, prediction_path, metadata_path = paths_by_job[job.job_id]
                    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    source_hashes.extend((
                        (f"{job.job_id}/checkpoint", _sha256(checkpoint_path)),
                        (f"{job.job_id}/prediction", _sha256(prediction_path)),
                        (f"{job.job_id}/resolved_config", run_metadata["resolved_config_sha256"]),
                    ))
                lock = FoldLock(
                    training_seed=seed,
                    outer_fold_id=outer,
                    selected=selected,
                    control_selections=controls,
                    inner_fold_memberships=tuple(memberships),
                    fitted_states=tuple(sorted(states.items())),
                    fitted_state_sha256s=tuple(sorted(state_hashes.items())),
                    training_membership_sha256=_membership_hash(np.asarray(all_validation_ids, dtype=np.int64)),
                    source_hashes=tuple(sorted(source_hashes)),
                    plan_sha256=frozen_manifest["plan_sha256"],
                    config_sha256=config.sha256,
                    source_commit=TRAINING_COMMIT,
                    lock_sha256="",
                )
                lock_sha = _sha256_text(_canonical_json(lock.payload(include_digest=False)))
                lock = FoldLock(
                    training_seed=lock.training_seed,
                    outer_fold_id=lock.outer_fold_id,
                    selected=lock.selected,
                    control_selections=lock.control_selections,
                    inner_fold_memberships=lock.inner_fold_memberships,
                    fitted_states=lock.fitted_states,
                    fitted_state_sha256s=lock.fitted_state_sha256s,
                    training_membership_sha256=lock.training_membership_sha256,
                    source_hashes=lock.source_hashes,
                    plan_sha256=lock.plan_sha256,
                    config_sha256=lock.config_sha256,
                    source_commit=lock.source_commit,
                    lock_sha256=lock_sha,
                )
                repository.write_fold_lock(lock)

    return {
        "root": root,
        "artifact_root": artifact_root,
        "reuse_root": reuse_root,
        "reader": lambda: InnerArtifactReader(
            config_path=CONFIG_PATH,
            artifact_root=artifact_root,
            reuse_roots={ROOT_NAME: reuse_root},
        ),
        "paths_by_job": paths_by_job,
        "planner": planner,
        "manager": manager,
        "config": config,
    }


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tamper_prediction(
    paths: tuple[Path, Path, Path],
    mutate: Any,
) -> tuple[bytes, bytes]:
    _checkpoint_path, prediction_path, metadata_path = paths
    original_prediction = prediction_path.read_bytes()
    original_metadata = metadata_path.read_bytes()
    payload = json.loads(original_prediction)
    mutate(payload)
    modified = _pretty_json(payload)
    prediction_path.write_bytes(modified)
    metadata = json.loads(original_metadata)
    metadata["prediction"]["sha256"] = hashlib.sha256(modified).hexdigest()
    metadata_path.write_bytes(_pretty_json(metadata))
    return original_prediction, original_metadata


def _restore_prediction(paths: tuple[Path, Path, Path], originals: tuple[bytes, bytes]) -> None:
    _checkpoint_path, prediction_path, metadata_path = paths
    prediction_path.write_bytes(originals[0])
    metadata_path.write_bytes(originals[1])


def test_real_reader_validate_only_preserves_artifact_bytes_and_creates_no_output(
    real_inner_artifact_tree: dict[str, Any],
) -> None:
    artifact_root = real_inner_artifact_tree["artifact_root"]
    output_root = real_inner_artifact_tree["root"] / "diagnostics-must-not-exist"
    before = _file_hashes(artifact_root)
    reader = real_inner_artifact_tree["reader"]()
    with _synthetic_reader_contracts():
        result = StudyDiagnosticsRunner(
            reader, output_root=output_root, make_figures=False,
        ).validate_only()
    assert result["valid"] is True
    assert result["validation_counts"] == {
        "locks": 15, "inner_jobs": 240,
        "native_inner_jobs": 224, "historical_inner_jobs": 16,
    }
    assert result["training_source_commit"] == TRAINING_COMMIT
    assert result["result_snapshot_commit"] == SNAPSHOT_COMMIT
    assert result["training_source_commit"] != result["result_snapshot_commit"]
    assert not output_root.exists()
    assert _file_hashes(artifact_root) == before


def test_real_reader_rejects_prediction_hash_change_without_mutating_or_creating_output(
    real_inner_artifact_tree: dict[str, Any],
) -> None:
    job = next(
        job for job in real_inner_artifact_tree["planner"].inventory
        if job.stage == "inner" and job.training_seed != 78
    )
    paths = real_inner_artifact_tree["paths_by_job"][job.job_id]
    original_prediction = paths[1].read_bytes()
    output_root = real_inner_artifact_tree["root"] / "hash-failure-output"
    try:
        paths[1].write_bytes(original_prediction + b" ")
        with pytest.raises(DiagnosticsError, match="invalid inner artifact|prediction hash|hash does not match"):
            with _synthetic_reader_contracts():
                StudyDiagnosticsRunner(
                    real_inner_artifact_tree["reader"](),
                    output_root=output_root,
                    make_figures=False,
                ).validate_only()
    finally:
        paths[1].write_bytes(original_prediction)
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("label", "training label|expected"),
        ("nonfinite", "finite|non-finite|prediction artifact is invalid"),
    ],
)
def test_real_reader_rejects_invalid_saved_labels_and_nonfinite_logits(
    real_inner_artifact_tree: dict[str, Any],
    mutation: str,
    message: str,
) -> None:
    job = next(
        job for job in real_inner_artifact_tree["planner"].inventory
        if job.stage == "inner" and job.training_seed != 78
    )
    paths = real_inner_artifact_tree["paths_by_job"][job.job_id]
    record = json.loads(paths[1].read_bytes())["records"][0]

    def mutate(payload: dict[str, Any]) -> None:
        row = payload["records"][0]
        if mutation == "label":
            row["training_label"] = (int(row["training_label"]) + 1) % 100
        else:
            row["logits"][0] = float("nan")

    originals = _tamper_prediction(paths, mutate)
    try:
        with pytest.raises(DiagnosticsError, match=message):
            with _synthetic_reader_contracts():
                StudyDiagnosticsRunner(
                    real_inner_artifact_tree["reader"](),
                    output_root=real_inner_artifact_tree["root"] / f"{mutation}-failure-output",
                    make_figures=False,
                ).validate_only()
    finally:
        _restore_prediction(paths, originals)
    assert record["sample_index"] == json.loads(paths[1].read_bytes())["records"][0]["sample_index"]


def test_real_reader_runner_aligns_saved_inner_batches_without_outer_reads_or_refitting(
    real_inner_artifact_tree: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real assembler, evaluator, and read-only reader after locks exist."""
    from expert_method.analysis import assemble_inner_dataset
    from expert_method.ridge_sinkhorn import three_seed_study as study_module
    import scripts.ridge_sinkhorn_data as sinkhorn_data
    import scripts.ridge_sinkhorn_outer as outer_module
    import scripts.task3e_fixed as task3e

    reader = real_inner_artifact_tree["reader"]()
    with _synthetic_reader_contracts():
        validated = reader.validate()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("inner diagnostics attempted fitting, selection, or an outer/data read")

    monkeypatch.setattr(study_module.FoldStudyRunner, "lock", forbidden)
    monkeypatch.setattr(study_module.InnerCVSelector, "select", forbidden)
    monkeypatch.setattr(study_module.ContributionRidgeStrategy, "fit", forbidden)
    monkeypatch.setattr(study_module.ResidualRidgeStrategy, "fit", forbidden)
    monkeypatch.setattr(study_module.FrozenPriceSinkhornStrategy, "fit", forbidden)
    monkeypatch.setattr(sinkhorn_data, "load_ridge_sinkhorn_development_data", forbidden)
    monkeypatch.setattr(outer_module, "_load_outer_predictions", forbidden)
    monkeypatch.setattr(task3e, "load_restricted_analysis_dataset", forbidden)

    loaded_job_ids: list[str] = []
    original_load_logits = reader.load_logits

    def checked_load_logits(job_id: str):
        prediction = original_load_logits(job_id)
        assert prediction.stage == "inner"
        loaded_job_ids.append(job_id)
        return prediction

    monkeypatch.setattr(reader, "load_logits", checked_load_logits)
    with _synthetic_reader_contracts():
        dataset = assemble_inner_dataset(
            reader, reader.job_lookup, validated.manager, 78, 0,
        )
    expected_ids = np.concatenate([
        np.asarray(validated.manager.inner_fold(0, inner).prediction_indices, dtype=np.int64)
        for inner in range(4)
    ])
    np.testing.assert_array_equal(dataset.sample_ids, expected_ids)
    assert len(loaded_job_ids) == 16
    jobs_by_id = {job.job_id: job for job in validated.planner.inventory}
    assert [jobs_by_id[job_id].inner_fold_id for job_id in loaded_job_ids] == [
        inner for inner in range(4) for _expert in range(4)
    ]

    evaluator = StudyEvaluator(reader.config)
    locked_batch = InferenceBatch(dataset.logits, dataset.sample_ids)
    full_predictions, full_weights = evaluator.predict_locked_methods(
        validated.locks[(78, 0)], locked_batch,
    )
    boundaries = (0, 37, 142, 303, len(dataset.sample_ids))
    chunk_predictions: dict[str, list[np.ndarray]] = {
        method: [] for method in full_predictions
    }
    chunk_weights: dict[str, list[np.ndarray]] = {method: [] for method in full_weights}
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        predictions, weights = evaluator.predict_locked_methods(
            validated.locks[(78, 0)],
            InferenceBatch(dataset.logits[start:stop], dataset.sample_ids[start:stop]),
        )
        for method in full_predictions:
            chunk_predictions[method].append(predictions[method])
        for method in full_weights:
            chunk_weights[method].append(weights[method])
    for method in full_predictions:
        np.testing.assert_array_equal(
            full_predictions[method], np.concatenate(chunk_predictions[method]),
        )
    for method in full_weights:
        np.testing.assert_allclose(
            full_weights[method], np.concatenate(chunk_weights[method]),
            rtol=0.0, atol=1e-12,
        )

    runner = StudyDiagnosticsRunner(
        reader,
        output_root=tmp_path / "guarded-real-reader-diagnostics",
        make_figures=False,
    )
    with _synthetic_reader_contracts():
        result = runner.run(limit_pairs=1)
    assert result["pairs_analyzed"] == 1
    output_root = Path(result["output_root"])
    assert (output_root / "analysis.json").is_file()
    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["training_source_commit"] == TRAINING_COMMIT
    assert provenance["analysis_source_commit"] != provenance["training_source_commit"]
