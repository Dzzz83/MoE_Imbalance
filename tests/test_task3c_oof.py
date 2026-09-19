"""Synthetic regression tests for the Task 3C batch seam.

The fixtures are training-like only. They never load CIFAR images, the
balanced test set, or the repository's canonical expert checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

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
)


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


def _write_synthetic_pilot(plan: Task3CBatchPlan) -> tuple[Path, dict[str, str]]:
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
    store.prepare_run(context, payload)
    checkpoint_path = store.checkpoint_path(context)
    torch.save(
        {
            "epoch": 200,
            "seed": 78,
            "expert_name": "CE",
            "model_state_dict": {},
            "optimiser_state_dict": {},
            "is_final": True,
            "log": {},
        },
        checkpoint_path,
    )
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
        dataset.validate(plan.manager)
        assert dataset.num_samples == len(plan.manager.outer_fold(0).expert_training_indices)
        assert set(dataset.inner_fold_ids.tolist()) == {0, 1, 2, 3}
        assert set(dataset.sample_indices.tolist()).isdisjoint(
            plan.manager.outer_fold(0).evaluation_indices
        )

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
        torch.save(
            {
                "epoch": 200,
                "seed": 78,
                "expert_name": "CE",
                "model_state_dict": {},
                "optimiser_state_dict": {},
                "is_final": True,
                "log": {},
            },
            checkpoint_path,
        )

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
        "aligned persistence and diagnostic partition disclosure",
        test_alignment_persistence_and_primary_diagnostic_partition,
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
