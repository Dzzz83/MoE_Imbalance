"""Integrity checks for isolated evaluation provenance and publication."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expert_method.cli import build_parser
from expert_method.config import load_runtime_profile, load_study
from expert_method.provenance import (
    EvaluationRunStore,
    StudyTrainingIdentity,
    canonical_json_bytes,
)
from expert_method.ridge_sinkhorn.matrix import MatrixError, OOFMatrixPlanner
from expert_method.ridge_sinkhorn.three_seed_study import StudyArtifactRepository, StudyConfig
from expert_method.workflow import WorkflowError, _EvaluationRunContext


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _training_identity() -> StudyTrainingIdentity:
    return StudyTrainingIdentity(
        study_id="example-study",
        source_commit="a" * 40,
        freeze_sha256=_digest("freeze"),
        plan_sha256=_digest("plan"),
        study_config_sha256=_digest("config"),
        study_config_yaml_sha256=_digest("yaml"),
        protocol_config_sha256=_digest("protocol"),
        fold_manifest_sha256=_digest("folds"),
        job_inventory_sha256=_digest("jobs"),
    )


def _start_record(evaluation_id: str = "test-run") -> dict[str, object]:
    return {
        "evaluation_id": evaluation_id,
        "study_id": "example-study",
        "training_identity": _training_identity().to_dict(),
        "evaluator_identity": {
            "commit": "b" * 40,
            "dirty": False,
            "repository_root": "/checkout",
        },
        "scientific_settings": {"folds": 5, "bootstrap_replicates": 10_000},
        "configuration_inputs": {"files": {"study": {"sha256": _digest("study")}}},
        "runtime_resolution": {"profile_id": "test"},
        "command": ["python", "-m", "expert_method", "study", "evaluate"],
        "python_executable": "/usr/bin/python",
        "working_directory": "/checkout",
        "started_at": "2026-10-03T00:00:00+00:00",
        "runtime_versions": {"python": "3.12"},
    }


def _completion_record(evaluation_id: str = "test-run") -> dict[str, object]:
    record = _start_record(evaluation_id)
    record.update({
        "finished_at": None,
        "duration_seconds": None,
        "input_artifacts": {f"job-{index}": {"sha256": _digest(str(index))} for index in range(300)},
        "input_manifests": {"folds": _digest("folds")},
        "input_evaluation_outputs": {},
        "input_evaluation_records": {},
        "fold_memberships": {f"fold-{index}": {"lock_sha256": _digest(str(index))} for index in range(15)},
        "training_population": {"size": 10_847, "indices_sha256": _digest("indices")},
    })
    return record


def _publish_evaluation(store: EvaluationRunStore, *, start_sha256: str) -> None:
    staged = store.stage_directory("evaluate")
    (staged / "evaluations").mkdir()
    (staged / "evaluations" / "fold.json").write_text("{}\n", encoding="utf-8")
    output = store.publish(
        "evaluate", staged, expected_start_sha256=start_sha256
    )
    store.complete(
        "evaluate",
        _completion_record(store.evaluation_id),
        output,
        started_monotonic=time.monotonic() - 1.0,
        expected_start_sha256=start_sha256,
    )


def _freeze_record(identity: StudyTrainingIdentity) -> dict[str, object]:
    return {
        "study_id": identity.study_id,
        "source_commit": identity.source_commit,
        "source_tree_dirty": False,
        "freeze_sha256": identity.freeze_sha256,
        "plan_sha256": identity.plan_sha256,
        "study_config_sha256": identity.study_config_sha256,
        "study_config_yaml_sha256": identity.study_config_yaml_sha256,
        "protocol_config_sha256": identity.protocol_config_sha256,
        "fold_manifest_sha256": identity.fold_manifest_sha256,
        "job_inventory_sha256": identity.job_inventory_sha256,
    }


def test_evaluation_completion_hashes_isolated_outputs(tmp_path: Path) -> None:
    study_root = tmp_path / "runs" / "example-study"
    baseline = study_root / "study_analysis"
    baseline.mkdir(parents=True)
    sentinel = baseline / "results.json"
    sentinel.write_text("baseline\n", encoding="utf-8")
    store = EvaluationRunStore(study_root, "test-run")
    start_path = store.begin("evaluate", _start_record())
    start_sha256 = hashlib.sha256(start_path.read_bytes()).hexdigest()

    _publish_evaluation(store, start_sha256=start_sha256)

    completed = store.read_complete("evaluate")
    assert len(completed["provenance"]["input_artifacts"]) == 300
    assert len(completed["provenance"]["fold_memberships"]) == 15
    assert sentinel.read_text(encoding="utf-8") == "baseline\n"
    assert (store.run_dir / "outputs" / "evaluations" / "fold.json").is_file()
    assert not (baseline / "evaluations").exists()

    (store.run_dir / "outputs" / "evaluations" / "fold.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outputs differ"):
        store.read_complete("evaluate")


def test_tampered_start_record_blocks_publication(tmp_path: Path) -> None:
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "test-run")
    start_path = store.begin("evaluate", _start_record())
    pinned_sha256 = hashlib.sha256(start_path.read_bytes()).hexdigest()
    staged = store.stage_directory("evaluate")
    (staged / "fold.json").write_text("validated output\n", encoding="utf-8")

    changed = json.loads(start_path.read_text(encoding="utf-8"))
    changed["command"] = ["altered-after-start"]
    changed.pop("record_sha256")
    changed["record_sha256"] = hashlib.sha256(
        canonical_json_bytes(changed)
    ).hexdigest()
    start_path.write_bytes(canonical_json_bytes(changed))

    with pytest.raises(ValueError, match="start record changed"):
        store.validate_completion_record(
            "evaluate",
            _completion_record(),
            expected_start_sha256=pinned_sha256,
        )
    with pytest.raises(ValueError, match="start record changed"):
        store.publish(
            "evaluate", staged, expected_start_sha256=pinned_sha256
        )
    assert not (store.run_dir / "outputs").exists()
    assert staged.is_dir()


def test_source_gate_failure_never_publishes_staged_outputs(tmp_path: Path) -> None:
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "test-run")
    start_path = store.begin("evaluate", _start_record())
    staged = store.stage_directory("evaluate")
    (staged / "fold.json").write_text("validated output\n", encoding="utf-8")
    context = object.__new__(_EvaluationRunContext)
    context.stage = "evaluate"
    context.store = store
    context.staged_directory = staged
    context.started_monotonic = time.monotonic() - 1.0
    context.start_record_sha256 = hashlib.sha256(start_path.read_bytes()).hexdigest()
    context.completion_record = lambda **_kwargs: _completion_record()

    def reject_changed_sources(_session: object) -> None:
        raise WorkflowError("study input bytes changed before publication")

    context.verify_integrity = reject_changed_sources
    with pytest.raises(WorkflowError, match="input bytes changed"):
        context.publish_and_complete(
            manager=None, config=None, locks=(), references={}, session=None
        )
    assert not (store.run_dir / "outputs").exists()
    assert staged.is_dir()


def test_missing_completion_provenance_is_rejected_before_publish(tmp_path: Path) -> None:
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "test-run")
    start_path = store.begin("evaluate", _start_record())
    start_sha256 = hashlib.sha256(start_path.read_bytes()).hexdigest()
    staged = store.stage_directory("evaluate")
    (staged / "fold.json").write_text("validated output\n", encoding="utf-8")
    incomplete = _completion_record()
    incomplete.pop("input_artifacts")

    with pytest.raises(ValueError, match="required study provenance"):
        store.validate_completion_record(
            "evaluate", incomplete, expected_start_sha256=start_sha256
        )
    assert not (store.run_dir / "outputs").exists()


def test_report_repository_reads_paired_outputs_and_writes_separately(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    evaluate_outputs = tmp_path / "evaluation-runs" / "run-1" / "outputs"
    report_stage = tmp_path / "evaluation-runs" / "run-1" / ".report-staging"
    repository = StudyArtifactRepository(
        artifact_root,
        config=StudyConfig(),
        output_dir=report_stage,
        evaluation_input_dir=evaluate_outputs,
    )

    assert repository.config_path == artifact_root / repository.study_root.name / "study_analysis" / "study_config.json"
    assert repository.lock_path(78, 0).parent == repository.study_analysis_dir / "locks"
    assert repository.evaluation_path(78, 0) == evaluate_outputs / "evaluations" / "seed_78_outer_0.json"
    assert repository.evaluation_arrays_path(78, 0) == evaluate_outputs / "evaluations" / "seed_78_outer_0.npz"
    assert repository.result_path == report_stage / "results.json"
    assert repository.report_path == report_stage / "report.md"


def test_failure_cleanup_does_not_follow_retargeted_output_root(tmp_path: Path) -> None:
    study_root = tmp_path / "runs" / "example-study"
    store = EvaluationRunStore(study_root, "test-run")
    store.begin("evaluate", _start_record())
    staged = store.stage_directory("evaluate")
    moved_runs = study_root / "saved-evaluation-runs"
    store.runs_root.rename(moved_runs)

    outside = tmp_path / "redirected-output"
    redirected_run = outside / store.evaluation_id
    redirected_stage = redirected_run / staged.name
    redirected_stage.mkdir(parents=True)
    victim = redirected_stage / "keep.txt"
    victim.write_text("do not remove", encoding="utf-8")
    store.runs_root.symlink_to(outside, target_is_directory=True)

    context = object.__new__(_EvaluationRunContext)
    context.store = store
    context.stage = "evaluate"
    context.owns_run = True
    context.staged_directory = staged
    context.fail(RuntimeError("simulated evaluation failure"))

    assert victim.read_text(encoding="utf-8") == "do not remove"
    assert not (redirected_run / "evaluate.failed.json").exists()
    assert (moved_runs / store.evaluation_id / staged.name).is_dir()


def test_evaluation_run_rejects_symlinked_study_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "real-studies"
    target.mkdir()
    alias = tmp_path / "study-alias"
    alias.symlink_to(target, target_is_directory=True)
    store = EvaluationRunStore(alias / "example-study", "test-run")

    with pytest.raises(ValueError, match="resolves through a symlink"):
        store.begin("evaluate", _start_record())
    assert list(target.iterdir()) == []


def test_duplicate_command_cannot_fail_an_owned_run(tmp_path: Path) -> None:
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "test-run")
    store.begin("evaluate", _start_record())
    duplicate = object.__new__(_EvaluationRunContext)
    duplicate.store = store
    duplicate.stage = "evaluate"
    duplicate.owns_run = False
    duplicate.staged_directory = None

    duplicate.fail(ValueError("evaluation ID already exists"))

    assert (store.run_dir / "evaluate.start.json").is_file()
    assert not (store.run_dir / "evaluate.failed.json").exists()


def test_actual_integrity_gate_rejects_tampered_start_and_changed_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import expert_method.cli as cli

    study = load_study(REPO_ROOT / "configs/studies/ridge_sinkhorn_3seed_v1.yaml")
    profile = load_runtime_profile(REPO_ROOT / "configs/profiles/local-smoke.yaml")
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "test-run")
    start_path = store.begin("evaluate", _start_record())
    context = object.__new__(_EvaluationRunContext)
    context.store = store
    context.stage = "evaluate"
    context.start_record_sha256 = hashlib.sha256(start_path.read_bytes()).hexdigest()
    context.evaluator_identity = _start_record()["evaluator_identity"]
    context.training_identity = _training_identity().to_dict()
    context.configuration_inputs = _start_record()["configuration_inputs"]
    context.study = study
    context.profile = profile
    session = SimpleNamespace(verify_sources_unchanged=lambda: None)

    start_record = json.loads(start_path.read_text(encoding="utf-8"))
    start_record["command"] = ["tampered"]
    start_record.pop("record_sha256")
    start_record["record_sha256"] = hashlib.sha256(
        canonical_json_bytes(start_record)
    ).hexdigest()
    start_path.write_bytes(canonical_json_bytes(start_record))
    with pytest.raises(ValueError, match="start record changed"):
        context.verify_integrity(session)

    evaluator_store = EvaluationRunStore(
        tmp_path / "evaluator-check" / "example-study", "evaluator-run"
    )
    evaluator_start = evaluator_store.begin(
        "evaluate", _start_record("evaluator-run")
    )
    context.store = evaluator_store
    context.start_record_sha256 = hashlib.sha256(evaluator_start.read_bytes()).hexdigest()
    monkeypatch.setattr(
        cli,
        "_source_identity",
        lambda _root: {"commit": "c" * 40, "dirty": False},
    )
    with pytest.raises(WorkflowError, match="evaluator source identity changed"):
        context.verify_integrity(session)


def test_report_rejects_evaluation_identity_and_rehashed_sidecar_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import expert_method.cli as cli
    import expert_method.workflow as workflow

    study = load_study(REPO_ROOT / "configs/studies/ridge_sinkhorn_3seed_v1.yaml")
    identity = _training_identity()
    freeze = _freeze_record(identity)
    start_record = _start_record()
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "test-run")
    evaluate_start = store.begin("evaluate", start_record)
    _publish_evaluation(
        store, start_sha256=hashlib.sha256(evaluate_start.read_bytes()).hexdigest()
    )
    previous = store.read_complete("evaluate")
    pinned_evaluate_sidecars = store.sidecar_hashes("evaluate")

    context = object.__new__(_EvaluationRunContext)
    context.study = study
    context.profile = SimpleNamespace()
    context.stage = "report"
    context.store = store
    context.training_identity = identity.to_dict()
    context.evaluator_identity = start_record["evaluator_identity"]
    context.scientific_settings = start_record["scientific_settings"]
    context.configuration_inputs = start_record["configuration_inputs"]
    context._runtime_resolution = lambda: start_record["runtime_resolution"]
    context.previous_evaluation = previous
    context.previous_evaluation_sidecar_hashes = pinned_evaluate_sidecars
    report_start = store.begin("report", _start_record())
    context.start_record_sha256 = hashlib.sha256(report_start.read_bytes()).hexdigest()

    mismatched_identity = dict(previous["provenance"])
    mismatched_identity["evaluator_identity"] = {"commit": "c" * 40, "dirty": False}
    with pytest.raises(WorkflowError, match="different evaluator identity"):
        context._require_matching_evaluation(mismatched_identity)

    complete_path = store.run_dir / "evaluate.complete.json"
    rewritten = json.loads(complete_path.read_text(encoding="utf-8"))
    first_job = sorted(rewritten["provenance"]["input_artifacts"])[0]
    rewritten["provenance"]["input_artifacts"][first_job]["checkpoint_sha256"] = _digest(
        "rewritten but rehashed"
    )
    rewritten.pop("record_sha256")
    rewritten["record_sha256"] = hashlib.sha256(canonical_json_bytes(rewritten)).hexdigest()
    complete_path.write_bytes(canonical_json_bytes(rewritten))
    assert store.read_complete("evaluate")["output_files"] == previous["output_files"]

    monkeypatch.setattr(
        cli,
        "_source_identity",
        lambda _root: {"commit": context.evaluator_identity["commit"], "dirty": False},
    )
    monkeypatch.setattr(cli, "_freeze_path", lambda *_args: Path("unused-freeze"))
    monkeypatch.setattr(cli, "_read_freeze", lambda _path: freeze)
    monkeypatch.setattr(cli, "_validate_frozen_training_inputs", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_analysis_configuration_inputs",
        lambda *_args: context.configuration_inputs,
    )
    session = SimpleNamespace(verify_sources_unchanged=lambda: None)
    with pytest.raises(WorkflowError, match="provenance records changed"):
        context.verify_integrity(session)

    context.previous_evaluation_sidecar_hashes = store.sidecar_hashes("evaluate")
    with pytest.raises(WorkflowError, match="provenance changed during report"):
        context.verify_integrity(session)


def test_incomplete_evaluation_cannot_supply_report(tmp_path: Path) -> None:
    store = EvaluationRunStore(tmp_path / "runs" / "example-study", "incomplete")
    store.begin("evaluate", _start_record("incomplete"))
    with pytest.raises(ValueError, match="completion record is missing"):
        store.read_complete("evaluate")


def test_evaluation_run_rejects_symlinked_output_root(tmp_path: Path) -> None:
    study_root = tmp_path / "runs" / "example-study"
    outside = tmp_path / "outside"
    outside.mkdir()
    study_root.mkdir(parents=True)
    (study_root / "evaluation_runs").symlink_to(outside, target_is_directory=True)
    store = EvaluationRunStore(study_root, "test-run")
    with pytest.raises(ValueError, match="cannot be a symlink"):
        store.begin("evaluate", _start_record())
    assert list(outside.iterdir()) == []


def test_evaluation_id_is_available_only_for_evaluate_and_report() -> None:
    parser = build_parser()
    for stage in ("evaluate", "report"):
        parsed = parser.parse_args(["study", stage, "--evaluation-id", "run-1"])
        assert parsed.evaluation_id == "run-1"
    parsed = parser.parse_args(["study", "lock"])
    assert not hasattr(parsed, "evaluation_id")


def test_frozen_training_identity_requires_clean_record_and_read_only_planner() -> None:
    identity = _training_identity()
    with pytest.raises(MatrixError, match="read-only analysis plan"):
        OOFMatrixPlanner(
            manager=None,
            study_id=identity.study_id,
            freeze_sha256=identity.freeze_sha256,
            training_identity=identity,
            read_only=False,
        )

    freeze = {
        "study_id": identity.study_id,
        "source_commit": identity.source_commit,
        "source_tree_dirty": True,
        "freeze_sha256": identity.freeze_sha256,
        "plan_sha256": identity.plan_sha256,
        "study_config_sha256": identity.study_config_sha256,
        "study_config_yaml_sha256": identity.study_config_yaml_sha256,
        "protocol_config_sha256": identity.protocol_config_sha256,
        "fold_manifest_sha256": identity.fold_manifest_sha256,
        "job_inventory_sha256": identity.job_inventory_sha256,
    }
    with pytest.raises(ValueError, match="recorded as dirty"):
        StudyTrainingIdentity.from_freeze(freeze)


def test_separate_evaluator_rejects_other_checkout_and_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import expert_method.cli as cli

    study = load_study(REPO_ROOT / "configs/studies/ridge_sinkhorn_3seed_v1.yaml")
    profile = load_runtime_profile(REPO_ROOT / "configs/profiles/local-smoke.yaml")
    profile = replace(profile, run_root=str(tmp_path / "runs"))

    other_checkout_study = replace(
        study,
        source_path=tmp_path / "other" / "configs" / "studies" / "study.yaml",
    )
    with pytest.raises(WorkflowError, match="same checkout"):
        _EvaluationRunContext(
            other_checkout_study, profile, "wrong-checkout", stage="evaluate"
        ).begin()
    assert not (tmp_path / "runs" / study.study_id / "evaluation_runs").exists()

    monkeypatch.setattr(
        cli,
        "_source_identity",
        lambda _root: {"commit": "b" * 40, "dirty": True},
    )
    with pytest.raises(WorkflowError, match="clean committed evaluator"):
        _EvaluationRunContext(study, profile, "dirty-evaluator", stage="evaluate").begin()
    assert not (tmp_path / "runs" / study.study_id / "evaluation_runs").exists()
