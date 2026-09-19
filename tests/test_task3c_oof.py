"""Synthetic regression tests for the Task 3C batch seam.

The fixtures are training-like only. They never load CIFAR images, the
balanced test set, or the repository's canonical expert checkpoints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.nested_oof import (  # noqa: E402
    NestedOOFFoldManager,
    OOFPredictionArtifact,
    OOFPredictionRecord,
)
from scripts.oof_pipeline import (  # noqa: E402
    OOFArtifactStore,
    OOFCompletedRun,
    OOFTrainingOrchestrator,
)
from scripts.task3c_oof import (  # noqa: E402
    OOFAlignmentBuilder,
    Task3CBatchPlan,
    Task3CBatchRunner,
    Task3CDiagnosticReporter,
    Task3CError,
)


def _resnet32_checkpoint_state(
    *, expert_name: str, seed: int = 78, epoch: int = 200
) -> dict:
    """Return a structurally valid frozen-protocol checkpoint for synthetic tests."""
    from models.resnet32 import ResNet32

    model = ResNet32(num_classes=100)
    return {
        "epoch": epoch,
        "seed": seed,
        "expert_name": expert_name,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": {},
        "is_final": epoch == 200,
        "log": {},
    }


def _manager() -> NestedOOFFoldManager:
    labels = np.repeat(np.arange(3, dtype=np.int64), [10, 7, 5])
    indices = np.arange(100, 100 + len(labels), dtype=np.int64)
    return NestedOOFFoldManager(
        indices,
        labels,
        seed=42,
        num_classes=3,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )


def _plan(root: Path) -> Task3CBatchPlan:
    return Task3CBatchPlan(
        data_root=root / "data",
        config_root=Path(_PROJECT_ROOT) / "configs",
        artifact_root=root / "artifacts" / "oof",
        pilot_root=root / "artifacts" / "oof" / "task3b_pilot_ce_s78_o0_i0",
        device="cpu",
        manager=_manager(),
        enforce_canonical_population=False,
    )


def _resolved_config(plan: Task3CBatchPlan, job, checkpoint_dir: Path) -> dict:
    config = plan.config_for(job).replace(
        seed=78,
        device="cpu",
        epochs=200,
        checkpoint_dir=str(checkpoint_dir),
    )
    payload = config.to_dict()
    payload["resolved_device"] = config.resolved_device
    return payload


def _write_synthetic_pilot(
    plan: Task3CBatchPlan,
    *,
    mutate_config=None,
) -> tuple[Path, dict[str, str]]:
    pilot_root = plan.pilot_root
    store = OOFArtifactStore(
        root=pilot_root.parent,
        manager=plan.manager,
        experiment_id=pilot_root.name,
        canonical_checkpoint_dir=plan.artifact_root / "canonical-checkpoints",
    )
    job = next(job for job in plan.jobs if job.is_pilot)
    context = job.spec.__class__(
        experiment_id=pilot_root.name,
        expert=job.expert_key,
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(plan.manager)
    payload = _resolved_config(plan, job, store.run_dir(context) / "checkpoints")
    if mutate_config is not None:
        mutate_config(payload)
    store.prepare_run(context, payload)
    checkpoint_path = store.checkpoint_path(context)
    torch.save(_resnet32_checkpoint_state(expert_name="CE"), checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    labels_by_index = dict(zip(plan.manager.canonical_indices, plan.manager.training_labels))
    records = tuple(
        OOFPredictionRecord.create(
            sample_index=sample_index,
            training_label=labels_by_index[sample_index],
            outer_fold_id=0,
            inner_fold_id=0,
            expert_id="CE",
            training_seed=78,
            expert_training_membership_hash=context.training_membership_hash,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            resolved_config=payload,
            logits=np.array([1.0, 0.0, 0.0]),
        )
        for sample_index in context.prediction_indices
    )
    artifact = OOFPredictionArtifact(
        expert_order=plan.manager.expert_order,
        num_classes=plan.manager.num_classes,
        records=records,
    )
    store.record_checkpoint(context, checkpoint_path, checkpoint_sha)
    store.write_predictions(context, artifact)
    run_dir = plan.pilot_root / "expert_CE" / "seed_78" / "outer_0" / "inner_0"
    return run_dir, {
        "metadata": (run_dir / "run_metadata.json").read_text(),
        "config": (run_dir / "resolved_config.json").read_text(),
        "predictions": (run_dir / "predictions.json").read_text(),
        "manifest": (plan.pilot_root / "fold_manifest.json").read_text(),
    }


def _write_synthetic_run(
    plan: Task3CBatchPlan,
    job,
    *,
    checkpoint_state: dict | None = None,
) -> tuple[Path, dict[str, str]]:
    """Write one complete synthetic non-pilot run through the artifact store."""
    store = OOFArtifactStore(
        root=plan.artifact_root,
        manager=plan.manager,
        experiment_id=plan.experiment_id,
        canonical_checkpoint_dir=plan.artifact_root / "canonical-checkpoints",
    )
    context = plan.context_for(job)
    payload = _resolved_config(plan, job, store.run_dir(context) / "checkpoints")
    store.prepare_run(context, payload)
    checkpoint_path = store.checkpoint_path(context)
    torch.save(
        checkpoint_state
        or _resnet32_checkpoint_state(expert_name=job.expert_name),
        checkpoint_path,
    )
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    labels_by_index = dict(zip(plan.manager.canonical_indices, plan.manager.training_labels))
    records = tuple(
        OOFPredictionRecord.create(
            sample_index=sample_index,
            training_label=labels_by_index[sample_index],
            outer_fold_id=0,
            inner_fold_id=job.inner_fold_id,
            expert_id=job.expert_name,
            training_seed=78,
            expert_training_membership_hash=context.training_membership_hash,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            resolved_config=payload,
            logits=np.eye(plan.manager.num_classes, dtype=np.float32)[
                int(sample_index) % plan.manager.num_classes
            ],
        )
        for sample_index in context.prediction_indices
    )
    artifact = OOFPredictionArtifact(
        expert_order=plan.manager.expert_order,
        num_classes=plan.manager.num_classes,
        records=records,
    )
    store.record_checkpoint(context, checkpoint_path, checkpoint_sha)
    store.write_predictions(context, artifact)
    run_dir = store.run_dir(context)
    return run_dir, {
        "metadata": (run_dir / "run_metadata.json").read_text(),
        "config": (run_dir / "resolved_config.json").read_text(),
        "predictions": (run_dir / "predictions.json").read_text(),
        "manifest": (
            plan.artifact_root / plan.experiment_id / "fold_manifest.json"
        ).read_text(),
    }


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _pilot_files(run_dir: Path, pilot_root: Path) -> dict[str, bytes]:
    return {
        "checkpoint": (run_dir / "checkpoints" / "CE_seed78_final.pt").read_bytes(),
        "metadata": (run_dir / "run_metadata.json").read_bytes(),
        "config": (run_dir / "resolved_config.json").read_bytes(),
        "predictions": (run_dir / "predictions.json").read_bytes(),
        "manifest": (pilot_root / "fold_manifest.json").read_bytes(),
    }


def test_dry_run_finds_one_read_only_pilot_and_fifteen_missing_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        _run_dir, before = _write_synthetic_pilot(plan)
        runner = Task3CBatchRunner(plan, trainer_builder=lambda **_kwargs: None)

        statuses = runner.inspect()
        summary = runner.summarize(statuses)
        assert summary == {
            "required": 16,
            "validated_existing": 1,
            "missing": 15,
            "partial": 0,
            "invalid": 0,
        }
        assert statuses[0].job_id == "ce/inner_0"
        assert statuses[0].state == "validated_existing"
        assert "read-only" in statuses[0].detail

        pilot_root = plan.pilot_root
        after = {
            "metadata": (
                pilot_root / "expert_CE" / "seed_78" / "outer_0" / "inner_0" / "run_metadata.json"
            ).read_text(),
            "config": (
                pilot_root / "expert_CE" / "seed_78" / "outer_0" / "inner_0" / "resolved_config.json"
            ).read_text(),
            "predictions": (
                pilot_root / "expert_CE" / "seed_78" / "outer_0" / "inner_0" / "predictions.json"
            ).read_text(),
            "manifest": (pilot_root / "fold_manifest.json").read_text(),
        }
        assert before == after


def _assert_batch_refuses_invalid_pilot(plan: Task3CBatchPlan, run_dir: Path) -> None:
    calls = {"builders": 0}

    def trainer_builder(**_kwargs):
        calls["builders"] += 1
        raise AssertionError("invalid pilot must stop before trainer construction")

    runner = Task3CBatchRunner(plan, trainer_builder=trainer_builder)
    statuses = runner.inspect()
    pilot_status = next(status for status in statuses if status.job_id == "ce/inner_0")
    assert pilot_status.state == "invalid"
    try:
        runner.run_missing(execute_full=True)
    except Task3CError as exc:
        assert "pilot" in str(exc).lower() or "invalid" in str(exc).lower()
    else:
        raise AssertionError("invalid pilot was allowed to enter batch execution")
    assert calls["builders"] == 0


def _rewrite_prediction_checkpoint_hash(run_dir: Path, checkpoint_sha: str) -> None:
    artifact = OOFPredictionArtifact.from_json(
        (run_dir / "predictions.json").read_text()
    )
    records = tuple(
        OOFPredictionRecord.create(
            sample_index=record.sample_index,
            training_label=record.training_label,
            outer_fold_id=record.outer_fold_id,
            inner_fold_id=record.inner_fold_id,
            expert_id=record.expert_id,
            training_seed=record.training_seed,
            expert_training_membership_hash=record.expert_training_membership_hash,
            checkpoint_path=record.checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            resolved_config=json.loads(record.resolved_config_json),
            logits=record.logits,
            features=record.features,
        )
        for record in artifact.records
    )
    rewritten = OOFPredictionArtifact(
        expert_order=artifact.expert_order,
        num_classes=artifact.num_classes,
        records=records,
    )
    prediction_path = run_dir / "predictions.json"
    prediction_path.write_text(rewritten.to_json())
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint"]["sha256"] = checkpoint_sha
    metadata["prediction"]["sha256"] = hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def test_corrupted_pilot_checkpoint_is_invalid_and_never_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(plan)
        checkpoint_path = run_dir / "checkpoints" / "CE_seed78_final.pt"
        checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"corrupt")
        before = _pilot_files(run_dir, plan.pilot_root)
        _assert_batch_refuses_invalid_pilot(plan, run_dir)
        assert _pilot_files(run_dir, plan.pilot_root) == before


def test_corrupted_pilot_prediction_is_invalid_and_never_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(plan)
        prediction_path = run_dir / "predictions.json"
        prediction_path.write_bytes(b"not-json")
        before = _pilot_files(run_dir, plan.pilot_root)
        _assert_batch_refuses_invalid_pilot(plan, run_dir)
        assert _pilot_files(run_dir, plan.pilot_root) == before


def test_pilot_rejects_incompatible_loss_configuration():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(
            plan,
            mutate_config=lambda payload: payload["loss"].update({"tau": 0.5}),
        )
        _assert_batch_refuses_invalid_pilot(plan, run_dir)


def test_pilot_identity_fold_provenance_and_final_epoch_are_checked():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)

        run_dir, _before = _write_synthetic_pilot(plan)
        _rewrite_json(
            run_dir / "run_metadata.json",
            lambda payload: payload.update({"expert_name": "LAL"}),
        )
        _assert_batch_refuses_invalid_pilot(plan, run_dir)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(plan)
        _rewrite_json(
            run_dir / "run_metadata.json",
            lambda payload: payload.update({"training_seed": 79}),
        )
        _assert_batch_refuses_invalid_pilot(plan, run_dir)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(plan)
        _rewrite_json(
            run_dir / "run_metadata.json",
            lambda payload: payload.update({"training_membership_sha256": "f" * 64}),
        )
        _assert_batch_refuses_invalid_pilot(plan, run_dir)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(plan)
        checkpoint_path = run_dir / "checkpoints" / "CE_seed78_final.pt"
        torch.save(
            _resnet32_checkpoint_state(expert_name="CE", epoch=199),
            checkpoint_path,
        )
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        _rewrite_prediction_checkpoint_hash(run_dir, checkpoint_sha)
        _rewrite_json(
            run_dir / "run_metadata.json",
            lambda payload: payload["checkpoint"].update(
                {"sha256": checkpoint_sha, "epoch": 199}
            ),
        )
        _assert_batch_refuses_invalid_pilot(plan, run_dir)


def test_pilot_rejects_mismatched_fold_manifest_and_missing_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        run_dir, _before = _write_synthetic_pilot(plan)
        _rewrite_json(
            plan.pilot_root / "fold_manifest.json",
            lambda payload: payload.update({"fold_generation_seed": 43}),
        )
        _assert_batch_refuses_invalid_pilot(plan, run_dir)

    for missing_name in ("directory", "checkpoint", "prediction"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = _plan(root)
            run_dir, _before = _write_synthetic_pilot(plan)
            if missing_name == "directory":
                shutil.rmtree(plan.pilot_root)
                runner = Task3CBatchRunner(plan, trainer_builder=lambda **_kwargs: None)
                status = next(
                    value for value in runner.inspect() if value.job_id == "ce/inner_0"
                )
                assert status.state == "missing"
            else:
                target = (
                    run_dir / "checkpoints" / "CE_seed78_final.pt"
                    if missing_name == "checkpoint"
                    else run_dir / "predictions.json"
                )
                target.unlink()
                runner = Task3CBatchRunner(plan, trainer_builder=lambda **_kwargs: None)
                status = next(
                    value for value in runner.inspect() if value.job_id == "ce/inner_0"
                )
                assert status.state == "invalid"


def test_relocated_pilot_validates_without_rewriting_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source_plan = _plan(root)
        source_run, before = _write_synthetic_pilot(source_plan)
        source_snapshot = _pilot_files(source_run, source_plan.pilot_root)
        relocated_root = root / "relocated" / source_plan.pilot_root.name
        relocated_root.parent.mkdir(parents=True)
        shutil.copytree(source_plan.pilot_root, relocated_root)
        relocated_plan = Task3CBatchPlan(
            data_root=root / "other-data",
            config_root=Path(_PROJECT_ROOT) / "configs",
            artifact_root=root / "other-artifacts" / "oof",
            pilot_root=relocated_root,
            device="cpu",
            manager=_manager(),
            enforce_canonical_population=False,
        )
        runner = Task3CBatchRunner(relocated_plan, trainer_builder=lambda **_kwargs: None)
        status = next(value for value in runner.inspect() if value.job_id == "ce/inner_0")
        assert status.state == "validated_existing"
        assert "read-only" in status.detail
        assert _pilot_files(
            relocated_root / "expert_CE" / "seed_78" / "outer_0" / "inner_0",
            relocated_root,
        ) == source_snapshot


class _RecoveryModel(torch.nn.Module):
    def forward(self, images):
        return torch.zeros(
            (images.shape[0], 3), dtype=images.dtype, device=images.device
        )


class _RecoveryDataModule:
    def __init__(self, manager):
        self.manager = manager
        self.labels_by_index = dict(
            zip(manager.canonical_indices, manager.training_labels)
        )

    def validate_context(self, _context):
        return None

    def class_counts(self, context):
        return np.asarray(context.training_class_counts, dtype=np.int64)

    def training_loader(self, _context):
        raise AssertionError("prediction recovery must not train")

    def prediction_loader(self, context):
        indices = tuple(context.prediction_indices)
        labels = tuple(self.labels_by_index[index] for index in indices)
        return [
            (
                torch.zeros((len(indices), 3, 4, 4), dtype=torch.float32),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(indices, dtype=torch.int64),
            )
        ]


class _RecoveryTrainer:
    expert_name = "CE"
    seed = 78
    device = "cpu"

    def __init__(self, calls, checkpoint_path):
        self.calls = calls
        self.checkpoint_path = checkpoint_path
        self.model = _RecoveryModel()

    def load_checkpoint(self, path):
        assert Path(path) == self.checkpoint_path
        self.calls["loads"] += 1

    def train(self, _loader):
        self.calls["trains"] += 1
        raise AssertionError("valid-checkpoint prediction recovery must not train")


def _recovery_runner(plan: Task3CBatchPlan, checkpoint_path: Path):
    calls = {"loads": 0, "trains": 0}

    def trainer_builder(*_args, **_kwargs):
        return _RecoveryTrainer(calls, checkpoint_path)

    runner = Task3CBatchRunner(plan, trainer_builder=trainer_builder)
    runner.pipeline.data_module_factory = lambda **_kwargs: _RecoveryDataModule(
        plan.manager
    )
    return runner, calls


def test_valid_final_checkpoint_recovers_missing_predictions_without_training():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        _write_synthetic_pilot(plan)
        job = next(job for job in plan.jobs if job.job_id == "ce/inner_1")
        run_dir, _before = _write_synthetic_run(plan, job)
        prediction_path = run_dir / "predictions.json"
        prediction_path.unlink()
        checkpoint_path = run_dir / "checkpoints" / "CE_seed78_final.pt"
        runner, calls = _recovery_runner(plan, checkpoint_path)

        statuses = runner.inspect()
        status = next(value for value in statuses if value.job_id == job.job_id)
        assert status.state == "partial"
        assert "prediction" in status.detail.lower()

        final_statuses = runner.run_missing(max_jobs=1, execute_full=False)
        final = next(value for value in final_statuses if value.job_id == job.job_id)
        assert final.state == "validated_existing"
        assert calls == {"loads": 2, "trains": 0}
        assert prediction_path.exists()


def test_invalid_existing_nonpilot_checkpoint_is_invalid_and_not_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        _write_synthetic_pilot(plan)
        job = next(job for job in plan.jobs if job.job_id == "ce/inner_1")
        run_dir, _before = _write_synthetic_run(plan, job)
        checkpoint_path = run_dir / "checkpoints" / "CE_seed78_final.pt"
        checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"corrupt")
        before = checkpoint_path.read_bytes()
        calls = {"builders": 0}

        def trainer_builder(**_kwargs):
            calls["builders"] += 1
            raise AssertionError("invalid artifacts must stop before training")

        runner = Task3CBatchRunner(plan, trainer_builder=trainer_builder)
        status = next(value for value in runner.inspect() if value.job_id == job.job_id)
        assert status.state == "invalid"
        try:
            runner.run_missing(max_jobs=1, execute_full=True)
        except Task3CError as exc:
            assert "invalid" in str(exc).lower()
        else:
            raise AssertionError("invalid non-pilot artifact was accepted")
        assert calls["builders"] == 0
        assert checkpoint_path.read_bytes() == before


def test_completed_nonpilot_run_is_skipped_and_max_jobs_counts_pending_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        _write_synthetic_pilot(plan)
        completed_job = next(job for job in plan.jobs if job.job_id == "ce/inner_1")
        _write_synthetic_run(plan, completed_job)
        runner = Task3CBatchRunner(plan, trainer_builder=lambda **_kwargs: None)
        statuses = runner.inspect()
        summary = runner.summarize(statuses)
        assert summary["validated_existing"] == 2
        assert summary["missing"] == 14
        assert len({job.job_id for job in plan.jobs}) == 16

        calls = []
        runner.pipeline.run = lambda spec, config: calls.append(spec)  # type: ignore[method-assign]
        runner.store.validate_completed_run = lambda _context: SimpleNamespace(  # type: ignore[method-assign]
            resolved_config={}
        )
        runner._validate_frozen_resolved_config = lambda *_args: None  # type: ignore[method-assign]
        runner.inspect = lambda: statuses  # type: ignore[method-assign]
        runner.plan.write_batch_manifest()
        try:
            runner.run_missing(max_jobs=1, execute_full=True)
        except Exception as exc:  # pragma: no cover - assertion context below
            raise AssertionError(f"max_jobs test should not execute fake failure: {exc}")
        assert len(calls) == 1
        assert calls[0].inner_fold_id == 2


def test_dry_run_and_validation_never_build_or_train_experts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        _write_synthetic_pilot(plan)
        calls = {"builders": 0}

        def trainer_builder(**_kwargs):
            calls["builders"] += 1
            raise AssertionError("inspection must not construct an expert trainer")

        runner = Task3CBatchRunner(plan, trainer_builder=trainer_builder)
        runner.inspect()
        try:
            runner.run_missing(max_jobs=1, execute_full=False)
        except Task3CError as exc:
            assert "execute-full" in str(exc)
        else:
            raise AssertionError("unauthorized batch execution was accepted")
        assert calls["builders"] == 0


def test_missing_final_checkpoint_never_substitutes_a_canonical_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        job = next(job for job in plan.jobs if job.job_id == "ce/inner_1")
        store = OOFArtifactStore(
            root=root / "artifacts" / "oof",
            manager=plan.manager,
            experiment_id=plan.experiment_id,
            canonical_checkpoint_dir=root / "canonical-checkpoints",
        )
        context = plan.context_for(job)
        checkpoint_dir = store.run_dir(context) / "checkpoints"
        config = plan.config_for(job).replace(
            seed=78,
            device="cpu",
            epochs=200,
            checkpoint_dir=str(checkpoint_dir),
        )
        payload = config.to_dict()
        payload["resolved_device"] = config.resolved_device
        store.prepare_run(context, payload)

        canonical_path = store.canonical_checkpoint_dir / "CE_seed78_final.pt"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_resnet32_checkpoint_state(expert_name="CE"), canonical_path)
        canonical_before = canonical_path.read_bytes()
        expected_path = store.checkpoint_path(context)
        calls = {"train": 0, "load": 0}

        class _Trainer:
            expert_name = "CE"
            seed = 78
            device = "cpu"

            def train(self, _loader):
                calls["train"] += 1
                torch.save(_resnet32_checkpoint_state(expert_name="CE"), expected_path)
                return []

            def save_history(self, path):
                Path(path).write_text("[]")

            def load_checkpoint(self, _path):
                calls["load"] += 1
                raise AssertionError("canonical checkpoint was implicitly reused")

        class _DataModule:
            def validate_context(self, _context):
                return None

            def class_counts(self, _context):
                return np.asarray(context.training_class_counts, dtype=np.int64)

            def training_loader(self, _context):
                return []

        result = OOFTrainingOrchestrator(
            trainer_builder=lambda *_args, **_kwargs: _Trainer()
        ).train(
            context=context,
            config=config,
            data_module=_DataModule(),
            store=store,
            resolved_config=payload,
        )
        assert calls == {"train": 1, "load": 0}
        assert result.checkpoint_path == expected_path
        assert canonical_path.read_bytes() == canonical_before


def _synthetic_runs(plan: Task3CBatchPlan) -> dict[str, OOFCompletedRun]:
    labels_by_index = dict(zip(plan.manager.canonical_indices, plan.manager.training_labels))
    runs = {}
    for expert_index, job in enumerate(plan.jobs):
        context = plan.context_for(job)
        checkpoint_sha = f"{expert_index + 1:064x}"
        records = []
        for sample_index in context.prediction_indices:
            prediction = (int(sample_index) + expert_index) % plan.manager.num_classes
            logits = np.full(plan.manager.num_classes, -1.0)
            logits[prediction] = 2.0
            records.append(
                OOFPredictionRecord.create(
                    sample_index=sample_index,
                    training_label=labels_by_index[sample_index],
                    outer_fold_id=0,
                    inner_fold_id=job.inner_fold_id,
                    expert_id=job.expert_name,
                    training_seed=78,
                    expert_training_membership_hash=context.training_membership_hash,
                    checkpoint_path=f"{job.job_id}/checkpoint.pt",
                    checkpoint_sha256=checkpoint_sha,
                    resolved_config={"expert": job.expert_key, "seed": 78},
                    logits=logits,
                )
            )
        artifact = OOFPredictionArtifact(
            expert_order=plan.manager.expert_order,
            num_classes=plan.manager.num_classes,
            records=tuple(records),
        )
        runs[job.job_id] = OOFCompletedRun(
            context=context,
            run_dir=Path(job.job_id),
            resolved_config={"expert": job.expert_key, "seed": 78},
            metadata={"checkpoint": {"sha256": checkpoint_sha}},
            checkpoint_path=Path(job.job_id) / "checkpoint.pt",
            prediction_path=Path(job.job_id) / "predictions.json",
            artifact=artifact,
        )
    return runs


def test_alignment_persistence_and_primary_diagnostic_partition():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        dataset = OOFAlignmentBuilder(plan).build(_synthetic_runs(plan))
        rebuilt = OOFAlignmentBuilder(plan).build(_synthetic_runs(plan))
        dataset.validate(plan.manager)
        assert dataset.num_samples == len(plan.manager.outer_fold(0).expert_training_indices)
        assert set(dataset.inner_fold_ids.tolist()) == {0, 1, 2, 3}
        assert set(dataset.sample_indices.tolist()).isdisjoint(
            plan.manager.outer_fold(0).evaluation_indices
        )
        labels_by_index = dict(
            zip(plan.manager.canonical_indices, plan.manager.training_labels)
        )
        assert np.array_equal(dataset.sample_indices, np.sort(dataset.sample_indices))
        assert np.array_equal(dataset.sample_indices, rebuilt.sample_indices)
        assert np.array_equal(
            dataset.labels,
            np.asarray(
                [labels_by_index[index] for index in dataset.sample_indices],
                dtype=np.int64,
            ),
        )
        assert dataset.logits.shape == (
            dataset.num_samples,
            len(plan.manager.expert_order),
            plan.manager.num_classes,
        )
        assert dataset.expert_names == plan.manager.expert_order
        router = plan.manager.outer_fold(0).router_development
        assert set(router.fit_indices).isdisjoint(router.selection_indices)

        output = root / "aligned"
        arrays_path, metadata_path = dataset.save(output, manager=plan.manager)
        restored = dataset.load(output, manager=plan.manager)
        assert arrays_path.exists() and metadata_path.exists()
        assert np.array_equal(restored.sample_indices, dataset.sample_indices)
        assert np.array_equal(restored.logits, dataset.logits)

        report = Task3CDiagnosticReporter(plan).build(dataset)
        primary = report["primary_router_fit_partition"]
        assert primary["inner_fold_ids"] == [1, 2, 3]
        assert primary["no_router_fitted"] is True
        assert "inner-fold-0 labels" in report["full_development_descriptive_partition"]["disclosure"]
        assert report["research_boundary"]["router_selection_inner_fold_ids"] == [0]
        assert report["research_boundary"]["outer_evaluation_excluded"] is True


def _replace_run_record(runs, job_id, position, **changes):
    updated = dict(runs)
    run = runs[job_id]
    original = run.artifact.records[position]
    values = {
        "sample_index": original.sample_index,
        "training_label": original.training_label,
        "outer_fold_id": original.outer_fold_id,
        "inner_fold_id": original.inner_fold_id,
        "expert_id": original.expert_id,
        "training_seed": original.training_seed,
        "expert_training_membership_hash": original.expert_training_membership_hash,
        "checkpoint_path": original.checkpoint_path,
        "checkpoint_sha256": original.checkpoint_sha256,
        "resolved_config": json.loads(original.resolved_config_json),
        "logits": original.logits,
        "features": original.features,
    }
    values.update(changes)
    records = list(run.artifact.records)
    records[position] = OOFPredictionRecord.create(**values)
    artifact = OOFPredictionArtifact(
        expert_order=run.artifact.expert_order,
        num_classes=run.artifact.num_classes,
        records=tuple(records),
    )
    updated[job_id] = dataclasses.replace(run, artifact=artifact)
    return updated


def _assert_alignment_rejects(plan, runs, expected_text):
    try:
        OOFAlignmentBuilder(plan).build(runs)
    except Exception as exc:  # noqa: BLE001 - standalone regression assertion
        assert expected_text.lower() in str(exc).lower(), str(exc)
    else:
        raise AssertionError("invalid OOF run collection was accepted")


def test_alignment_rejects_all_listed_provenance_and_shape_corruptions():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan(Path(tmp))
        valid = _synthetic_runs(plan)
        job = next(job for job in plan.jobs if job.job_id == "ce/inner_0")
        record = valid[job.job_id].artifact.records[0]

        duplicate = dict(valid)
        duplicate[job.job_id] = dataclasses.replace(
            valid[job.job_id],
            artifact=dataclasses.replace(
                valid[job.job_id].artifact,
                records=(record, record) + valid[job.job_id].artifact.records[2:],
            ),
        )
        _assert_alignment_rejects(plan, duplicate, "duplicate")

        labels_by_index = dict(
            zip(plan.manager.canonical_indices, plan.manager.training_labels)
        )
        wrong_label = _replace_run_record(
            valid,
            job.job_id,
            0,
            training_label=(labels_by_index[record.sample_index] + 1) % plan.manager.num_classes,
        )
        _assert_alignment_rejects(plan, wrong_label, "label")

        wrong_inner = _replace_run_record(
            valid,
            job.job_id,
            0,
            inner_fold_id=1,
        )
        _assert_alignment_rejects(plan, wrong_inner, "sample")

        wrong_seed = valid
        for position in range(len(valid[job.job_id].artifact.records)):
            wrong_seed = _replace_run_record(
                wrong_seed, job.job_id, position, training_seed=999
            )
        _assert_alignment_rejects(plan, wrong_seed, "seed")

        wrong_membership = _replace_run_record(
            valid, job.job_id, 0, expert_training_membership_hash="a" * 64
        )
        _assert_alignment_rejects(plan, wrong_membership, "membership")

        unexpected_sample = _replace_run_record(
            valid,
            job.job_id,
            0,
            sample_index=int(plan.manager.outer_fold(0).evaluation_indices[0]),
            training_label=int(
                labels_by_index[plan.manager.outer_fold(0).evaluation_indices[0]]
            ),
        )
        _assert_alignment_rejects(plan, unexpected_sample, "sample")

        outer_sample = _replace_run_record(
            valid,
            job.job_id,
            0,
            sample_index=int(plan.manager.outer_fold(0).evaluation_indices[1]),
            training_label=int(
                labels_by_index[plan.manager.outer_fold(0).evaluation_indices[1]]
            ),
        )
        _assert_alignment_rejects(plan, outer_sample, "sample")

        wrong_dimension = dict(valid)
        wrong_dimension[job.job_id] = dataclasses.replace(
            valid[job.job_id],
            artifact=dataclasses.replace(
                valid[job.job_id].artifact,
                num_classes=4,
            ),
        )
        _assert_alignment_rejects(plan, wrong_dimension, "classes")

        incomplete = dict(valid)
        del incomplete[job.job_id]
        _assert_alignment_rejects(plan, incomplete, "missing")

        try:
            OOFPredictionRecord.create(
                sample_index=record.sample_index,
                training_label=record.training_label,
                outer_fold_id=record.outer_fold_id,
                inner_fold_id=record.inner_fold_id,
                expert_id=record.expert_id,
                training_seed=record.training_seed,
                expert_training_membership_hash=record.expert_training_membership_hash,
                checkpoint_path=record.checkpoint_path,
                checkpoint_sha256=record.checkpoint_sha256,
                resolved_config=json.loads(record.resolved_config_json),
                logits=np.full(plan.manager.num_classes, np.nan),
            )
        except Exception as exc:  # noqa: BLE001 - constructor contract
            assert "finite" in str(exc)
        else:
            raise AssertionError("non-finite OOF logits were accepted")


def test_incomplete_job_collection_refuses_diagnostic_generation():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan(Path(tmp))
        runs = _synthetic_runs(plan)
        del runs["mixup/inner_3"]
        try:
            OOFAlignmentBuilder(plan).build(runs)
        except Task3CError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("diagnostic alignment accepted incomplete jobs")


def test_interrupted_run_adopts_only_a_valid_unrecorded_final_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = _plan(root)
        job = next(job for job in plan.jobs if job.job_id == "ce/inner_1")
        context = plan.context_for(job)
        store = OOFArtifactStore(
            root=root / "task3c-artifacts",
            manager=plan.manager,
            experiment_id="task3c_oof",
            canonical_checkpoint_dir=root / "canonical-checkpoints",
        )
        checkpoint_dir = store.run_dir(context) / "checkpoints"
        config = plan.config_for(job).replace(
            seed=78,
            device="cpu",
            epochs=200,
            checkpoint_dir=str(checkpoint_dir),
        )
        payload = config.to_dict()
        payload["resolved_device"] = config.resolved_device
        store.prepare_run(context, payload)
        checkpoint_path = store.checkpoint_path(context)
        torch.save(_resnet32_checkpoint_state(expert_name="CE"), checkpoint_path)

        class _DataModule:
            def validate_context(self, _context):
                return None

            def class_counts(self, _context):
                return np.asarray(context.training_class_counts, dtype=np.int64)

            def training_loader(self, _context):
                raise AssertionError("a valid interrupted final checkpoint should be reused")

        class _Trainer:
            expert_name = "CE"
            seed = 78
            device = "cpu"

            def load_checkpoint(self, path):
                assert Path(path) == checkpoint_path

        orchestrator = OOFTrainingOrchestrator(
            trainer_builder=lambda *_args, **_kwargs: _Trainer()
        )
        result = orchestrator.train(
            context=context,
            config=config,
            data_module=_DataModule(),
            store=store,
            resolved_config=payload,
        )
        assert result.checkpoint_path == checkpoint_path
        metadata = json.loads(store.metadata_path(context).read_text())
        assert metadata["status"] == "training_complete"
        assert metadata["checkpoint"]["epoch"] == 200


TESTS = [
    (
        "read-only pilot validation and 16-job dry-run accounting",
        test_dry_run_finds_one_read_only_pilot_and_fifteen_missing_jobs,
    ),
    (
        "corrupted pilot checkpoint is refused without overwrite",
        test_corrupted_pilot_checkpoint_is_invalid_and_never_overwritten,
    ),
    (
        "corrupted pilot predictions are refused without overwrite",
        test_corrupted_pilot_prediction_is_invalid_and_never_overwritten,
    ),
    (
        "pilot loss configuration is frozen",
        test_pilot_rejects_incompatible_loss_configuration,
    ),
    (
        "pilot identity, fold provenance, and final epoch are frozen",
        test_pilot_identity_fold_provenance_and_final_epoch_are_checked,
    ),
    (
        "pilot manifest and required files are present",
        test_pilot_rejects_mismatched_fold_manifest_and_missing_artifacts,
    ),
    (
        "relocated pilot remains read-only and valid",
        test_relocated_pilot_validates_without_rewriting_provenance,
    ),
    (
        "valid checkpoint recovers predictions without training",
        test_valid_final_checkpoint_recovers_missing_predictions_without_training,
    ),
    (
        "invalid non-pilot checkpoint is not overwritten",
        test_invalid_existing_nonpilot_checkpoint_is_invalid_and_not_overwritten,
    ),
    (
        "completed jobs are skipped and max-jobs counts pending work",
        test_completed_nonpilot_run_is_skipped_and_max_jobs_counts_pending_only,
    ),
    (
        "inspection and unauthorized execution do not train",
        test_dry_run_and_validation_never_build_or_train_experts,
    ),
    (
        "missing final checkpoints do not use canonical checkpoints",
        test_missing_final_checkpoint_never_substitutes_a_canonical_checkpoint,
    ),
    (
        "aligned persistence and diagnostic partition disclosure",
        test_alignment_persistence_and_primary_diagnostic_partition,
    ),
    (
        "alignment rejects provenance and shape corruption",
        test_alignment_rejects_all_listed_provenance_and_shape_corruptions,
    ),
    (
        "incomplete jobs refuse diagnostic generation",
        test_incomplete_job_collection_refuses_diagnostic_generation,
    ),
    (
        "interrupted final-checkpoint recovery",
        test_interrupted_run_adopts_only_a_valid_unrecorded_final_checkpoint,
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
    print(f"Task 3C OOF: {passed} passed, {failed} failed")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
