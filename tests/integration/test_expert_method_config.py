"""Public YAML contracts for the config-driven experiment CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

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

from data.nested_oof import NestedOOFFoldManager
from expert_method.config import ConfigError, load_runtime_profile, load_study
from expert_method.cli import main as cli_main
from expert_method.oof.pipeline import OOFArtifactError
from expert_method.ridge_sinkhorn.matrix import OOFMatrixPlanner


ROOT = REPO_ROOT
STUDY_PATH = ROOT / "configs" / "studies" / "ridge_sinkhorn_3seed_v1.yaml"


def _copy_config_tree(tmp_path: Path) -> tuple[Path, Path]:
    config_root = tmp_path / "configs"
    (config_root / "studies").mkdir(parents=True)
    (config_root / "profiles").mkdir()
    (config_root / "experts").mkdir()
    for name in ("ce.yaml", "lal.yaml", "balanced_softmax.yaml", "mixup.yaml"):
        shutil.copy2(ROOT / "configs" / "experts" / name, config_root / "experts" / name)
    study = config_root / "studies" / STUDY_PATH.name
    shutil.copy2(STUDY_PATH, study)
    profile = config_root / "profiles" / "local-smoke.yaml"
    shutil.copy2(ROOT / "configs" / "profiles" / "local-smoke.yaml", profile)
    return study, profile


def test_canonical_study_and_profile_resolve_strictly() -> None:
    study = load_study(STUDY_PATH)
    profile = load_runtime_profile(ROOT / "configs" / "profiles" / "local-smoke.yaml")

    assert study.study_id == "ridge_sinkhorn_3seed_v1"
    assert study.protocol["training_seeds"] == (78, 88, 1034)
    assert study.protocol["priors"] == {
        "uniform": (0.25, 0.25, 0.25, 0.25),
        "fixed_006": (0.0125, 0.25, 0.25, 0.4875),
        "fixed_007": (0.0125, 0.25, 0.4875, 0.25),
        "fixed_010": (0.0125, 0.4875, 0.25, 0.25),
        "fixed_011": (0.0125, 0.4875, 0.4875, 0.0125),
    }
    assert tuple(study.resolved_expert_configs) == (
        "ce", "logit_adjusted", "balanced_softmax", "mixup"
    )
    assert study.scientific_sha256 == hashlib.sha256(study.canonical_scientific_json.encode()).hexdigest()
    assert profile.profile_id == "local-smoke"
    assert profile.device == "cpu"
    assert profile.max_jobs == 1


def test_loaded_study_constructs_the_executable_protocol_from_yaml() -> None:
    from expert_method.ridge_sinkhorn.three_seed_study import StudyConfig

    study = load_study(STUDY_PATH)
    config = study.to_study_config()

    assert isinstance(config, StudyConfig)
    assert config.seeds == (78, 88, 1034)
    assert config.outer_folds == (0, 1, 2, 3, 4)
    assert config.inner_folds == (0, 1, 2, 3)
    assert config.expert_order == ("CE", "LAL", "BalancedSoftmax", "Mixup")
    assert dict(config.priors) == study.protocol["priors"]
    assert config.sha256 == StudyConfig().sha256


def test_unknown_study_or_profile_keys_fail_before_use(tmp_path: Path) -> None:
    study_path, profile_path = _copy_config_tree(tmp_path)
    study_path.write_text(study_path.read_text() + "\nunexpected: true\n")
    with pytest.raises(ConfigError, match="unknown"):
        load_study(study_path)

    profile_path.write_text(profile_path.read_text() + "\nmax_jobz: 2\n")
    with pytest.raises(ConfigError, match="unknown"):
        load_runtime_profile(profile_path)

    profile_path.write_text(profile_path.read_text().replace("max_jobs: 1", "max_jobs: 1\nmax_jobs: 2"))
    with pytest.raises(ConfigError, match="duplicate"):
        load_runtime_profile(profile_path)


def test_scientific_hash_tracks_expert_recipe_but_not_profile_paths(tmp_path: Path) -> None:
    study_path, profile_path = _copy_config_tree(tmp_path)
    original = load_study(study_path)

    profile_text = profile_path.read_text().replace("max_jobs: 1", "max_jobs: 4")
    profile_path.write_text(profile_text)
    assert load_study(study_path).scientific_sha256 == original.scientific_sha256

    expert = tmp_path / "configs" / "experts" / "ce.yaml"
    recipe = expert.read_text()
    runtime_only = recipe.replace("device: auto", "device: cuda").replace(
        "root: ./data", "root: /mounted/cifar"
    ).replace("dir: ./checkpoints", "dir: /temporary/checkpoints")
    expert.write_text(runtime_only)
    assert load_study(study_path).scientific_sha256 == original.scientific_sha256

    expert.write_text(runtime_only.replace("lr: 0.1", "lr: 0.2", 1))
    changed = load_study(study_path)
    assert changed.scientific_sha256 != original.scientific_sha256


def test_runtime_profile_rejects_invalid_shard_and_absolute_reuse_name(tmp_path: Path) -> None:
    profile_path = ROOT / "configs" / "profiles" / "local-smoke.yaml"
    text = profile_path.read_text().replace("shard_count: 1", "shard_count: 0")
    bad_profile = tmp_path / "bad.yaml"
    bad_profile.write_text(text)
    with pytest.raises(ConfigError, match="shard_count"):
        load_runtime_profile(bad_profile)

    bad_profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "reuse_roots: {}", "reuse_roots:\n  /bad-name: /some/path"
        )
    )
    with pytest.raises(ConfigError, match="reuse root name"):
        load_runtime_profile(bad_profile)


def test_read_only_matrix_planner_does_not_create_artifact_root(tmp_path: Path) -> None:
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    artifact_root = tmp_path / "preview-must-not-create"
    planner = OOFMatrixPlanner(manager=manager, artifact_root=artifact_root, reuse_roots={}, read_only=True)
    assert not artifact_root.exists()
    with pytest.raises(OOFArtifactError, match="read-only"):
        planner.native_store.write_manifest()
    assert not artifact_root.exists()


def test_matrix_identity_uses_loaded_executable_config_and_freeze_hash(tmp_path: Path) -> None:
    study = load_study(STUDY_PATH)
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    freeze_hash = "a" * 64
    planner = OOFMatrixPlanner(
        manager=manager,
        artifact_root=tmp_path / "matrix-identity",
        reuse_roots={},
        study_config=study.to_study_config(),
        freeze_sha256=freeze_hash,
        read_only=True,
    )

    identity = planner.frozen_manifest()
    assert identity["protocol_config_sha256"] == study.to_study_config().sha256
    assert identity["study_freeze_sha256"] == freeze_hash
    assert not (tmp_path / "matrix-identity").exists()


def test_validate_cli_emits_config_identity_without_creating_run_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "validation-must-not-create"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "validate"
    ]) == 0
    import json
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["study_id"] == "ridge_sinkhorn_3seed_v1"
    assert len(payload["scientific_sha256"]) == 64
    assert not run_root.exists()


def test_plan_cli_uses_read_only_preview_and_reports_inner_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "plan-must-not-create"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda _study, _profile: manager)

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "plan", "--stage", "inner"
    ]) == 0
    import json
    payload = json.loads(capsys.readouterr().out)
    assert payload["inventory_count"] == 240
    assert payload["counts"]["missing"] == 240
    assert payload["stage"] == "inner"
    assert not run_root.exists()


def test_outer_plan_cli_stops_at_lock_gate_before_planning_outer_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_root = tmp_path / "outer-plan-must-not-touch-jobs"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda _study, _profile: manager)

    from expert_method.ridge_sinkhorn.matrix import OOFMatrixPlanner

    events: list[str] = []
    original_gate = OOFMatrixPlanner.validate_complete_lock_matrix
    original_plan = OOFMatrixPlanner.plan
    original_native = OOFMatrixPlanner._validate_native
    original_reuse = OOFMatrixPlanner._validate_historical

    def gate_spy(self, **kwargs):
        events.append("lock_gate")
        return original_gate(self, **kwargs)

    def plan_spy(self, *, stage, **kwargs):
        if stage == "outer":
            events.append("outer_plan")
        return original_plan(self, stage=stage, **kwargs)

    def native_spy(self, job):
        if job.stage == "outer":
            events.append("outer_native_artifact")
        return original_native(self, job)

    def reuse_spy(self, job, source_experiment):
        if job.stage == "outer":
            events.append("outer_reuse_artifact")
        return original_reuse(self, job, source_experiment)

    monkeypatch.setattr(OOFMatrixPlanner, "validate_complete_lock_matrix", gate_spy)
    monkeypatch.setattr(OOFMatrixPlanner, "plan", plan_spy)
    monkeypatch.setattr(OOFMatrixPlanner, "_validate_native", native_spy)
    monkeypatch.setattr(OOFMatrixPlanner, "_validate_historical", reuse_spy)

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile),
        "study", "plan", "--stage", "outer",
    ]) == 2
    capsys.readouterr()
    assert events == ["lock_gate"]
    assert not run_root.exists()


def test_legacy_inner_plan_freezes_identity_and_reuse_audit_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from scripts.run_ridge_sinkhorn_matrix import main as legacy_matrix_main

    run_root = tmp_path / "legacy-plan-artifacts"
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda *_args: manager)
    source = cli_main.__globals__["_source_identity"]
    actual_commit = source(ROOT)["commit"]
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": actual_commit, "dirty": False},
    )
    args = [
        "--plan", "--stage", "inner", "--study-id", "ridge_sinkhorn_3seed_v1",
        "--data-root", "synthetic-training-only", "--artifact-root", str(run_root),
        "--device", "cpu",
    ]

    assert legacy_matrix_main(args) == 0
    first_output = capsys.readouterr().out
    freeze_path = run_root / "ridge_sinkhorn_3seed_v1" / "manifests/study-freeze.json"
    audit_path = run_root / "ridge_sinkhorn_3seed_v1" / "reuse_compatibility_inner.json"
    assert freeze_path.is_file()
    assert audit_path.is_file()
    first_freeze = freeze_path.read_bytes()
    first_audit = audit_path.read_bytes()
    assert json.loads(first_output)["frozen"] is True

    assert legacy_matrix_main(args) == 0
    capsys.readouterr()
    assert freeze_path.read_bytes() == first_freeze
    assert audit_path.read_bytes() == first_audit


def test_legacy_outer_plan_freezes_reuse_audit_only_after_lock_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from expert_method.ridge_sinkhorn.matrix import OOFMatrixPlanner
    from scripts.run_ridge_sinkhorn_matrix import main as legacy_matrix_main

    run_root = tmp_path / "legacy-outer-plan-artifacts"
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda *_args: manager)
    source = cli_main.__globals__["_source_identity"]
    actual_commit = source(ROOT)["commit"]
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": actual_commit, "dirty": False},
    )

    events: list[str] = []
    original_freeze = OOFMatrixPlanner.freeze
    original_plan = OOFMatrixPlanner.plan

    def lock_gate(self, *, frozen=None):
        events.append("lock_gate")
        self._locks_validated = True
        return tuple(range(15))

    def freeze_spy(self, *, stage="inner", freeze_sha256=None):
        result = original_freeze(self, stage=stage, freeze_sha256=freeze_sha256)
        if stage == "outer":
            events.append("outer_audit")
        return result

    def plan_spy(self, *, stage, **kwargs):
        if stage == "outer":
            audit_path = run_root / "ridge_sinkhorn_3seed_v1" / "reuse_compatibility_outer.json"
            assert audit_path.is_file()
            assert json.loads(audit_path.read_text())["stage"] == "outer"
            events.append("outer_plan")
        return original_plan(self, stage=stage, **kwargs)

    monkeypatch.setattr(OOFMatrixPlanner, "validate_complete_lock_matrix", lock_gate)
    monkeypatch.setattr(OOFMatrixPlanner, "freeze", freeze_spy)
    monkeypatch.setattr(OOFMatrixPlanner, "plan", plan_spy)

    assert legacy_matrix_main([
        "--plan", "--stage", "outer", "--study-id", "ridge_sinkhorn_3seed_v1",
        "--data-root", "synthetic-training-only", "--artifact-root", str(run_root),
        "--device", "cpu",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "outer"
    assert events.index("lock_gate") < events.index("outer_audit") < events.index("outer_plan")


def test_freeze_requires_clean_committed_source_before_data_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "freeze-must-not-create"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": "a" * 40, "dirty": True},
    )
    monkeypatch.setattr(
        "expert_method.cli._load_manager",
        lambda *_args: pytest.fail("dirty source must be rejected before loading canonical data"),
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "freeze"
    ]) == 2
    assert "clean committed source" in capsys.readouterr().err
    assert not run_root.exists()


def test_status_without_freeze_reports_next_action_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "status-must-not-create"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "status"
    ]) == 0
    import json
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "unfrozen"
    assert payload["next_action"] == "study freeze"
    assert not run_root.exists()


def test_status_reports_running_attempts_before_freeze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from expert_method.attempts import AttemptWorkspace

    run_root = tmp_path / "running-attempts"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )
    attempt = AttemptWorkspace.create(
        run_root=run_root,
        study_id="ridge_sinkhorn_3seed_v1",
        job_id="synthetic-job",
        stage="inner",
        expert_key="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
        device="cpu",
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "status"
    ]) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["attempts"]["running"] == 1
    assert payload["attempts"]["failed"] == 0
    assert payload["attempts"]["locations"][0]["path"] == str(attempt.path)
    assert payload["counts"]["attempts"]["running"] == 1


def test_doctor_reports_preflight_without_mutation_or_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "doctor-run-root"
    bundle_root = tmp_path / "doctor-bundles"
    profile = tmp_path / "doctor-profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text()
        .replace("data_root: data", f"data_root: {tmp_path}")
        .replace("run_root: runs/local-smoke", f"run_root: {run_root}")
        .replace("bundle_output_dir: runs/local-smoke/bundles", f"bundle_output_dir: {bundle_root}")
    )
    monkeypatch.setattr(
        "expert_method.cli._load_manager",
        lambda *_args: pytest.fail("doctor must not load fold or test data"),
    )

    result = cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "doctor"
    ])
    payload = __import__("json").loads(capsys.readouterr().out)
    assert result in {0, 1}
    assert payload["test_accessed"] is False
    assert payload["mutated"] is False
    assert payload["git"]["head"]
    assert "dirty" in payload["git"]
    assert "cuda_available" in payload["device"]
    assert "free_bytes" in payload["disk"]["run_root"]
    assert not run_root.exists()
    assert not bundle_root.exists()


@pytest.mark.parametrize("command", ("run", "session"))
def test_training_commands_require_explicit_full_run_confirmation_before_loading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {tmp_path / 'no-training'}"
        )
    )
    monkeypatch.setattr(
        "expert_method.cli._load_manager",
        lambda *_args: pytest.fail("confirmation must be checked before loading canonical data"),
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile),
        "study", command, "--stage", "inner",
    ]) == 2
    assert "--execute-full" in capsys.readouterr().err
    assert not (tmp_path / "no-training").exists()


def test_freeze_records_commit_plan_config_and_inventory_immutably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "freeze-output"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {run_root}"
        )
    )
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda _study, _profile: manager)
    source = cli_main.__globals__["_source_identity"]
    actual_commit = source(STUDY_PATH.parents[2])["commit"]
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": actual_commit, "dirty": False},
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "freeze"
    ]) == 0
    import json
    output = json.loads(capsys.readouterr().out)
    freeze_path = Path(output["path"])
    freeze = json.loads(freeze_path.read_text())
    assert freeze["source_commit"] == actual_commit
    assert freeze["plan_sha256"] == hashlib.sha256((ROOT / "docs" / "PLAN.md").read_bytes()).hexdigest()
    assert freeze["study_config_sha256"] == load_study(STUDY_PATH).scientific_sha256
    assert freeze["inventory_count"] == 300
    assert len(freeze["inventory"]) == 300
    assert freeze["job_inventory_sha256"] == hashlib.sha256(
        json.dumps(freeze["inventory"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    matrix_root = run_root / "ridge_sinkhorn_3seed_v1"
    assert (matrix_root / "fold_manifest.json").is_file()
    job_manifest = json.loads((matrix_root / "job_manifest.json").read_text())
    assert job_manifest["study_freeze_sha256"] == freeze["freeze_sha256"]
    assert job_manifest["protocol_config_sha256"] == load_study(STUDY_PATH).to_study_config().sha256
    inner_audit = json.loads((matrix_root / "reuse_compatibility_inner.json").read_text())
    assert inner_audit["study_freeze_sha256"] == freeze["freeze_sha256"]
    assert len(freeze["reuse_decisions"]["inner"]) == 16
    assert freeze["reuse_decisions"]["outer"] == "deferred_until_all_inner_locks_are_valid"
    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "freeze"
    ]) == 0
    capsys.readouterr()
    assert freeze_path.read_text() == (json.dumps(freeze, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "status"
    ]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "inner_incomplete"
    assert status["counts"]["inner"]["missing"] == 240
    assert status["counts"]["outer_access_gated"] is True


def test_freeze_identity_excludes_runtime_profile_paths_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda _study, _profile: manager)
    source = cli_main.__globals__["_source_identity"]
    actual_commit = source(ROOT)["commit"]
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": actual_commit, "dirty": False},
    )

    profiles = []
    for suffix, device, shard_index, shard_count, max_jobs in (
        ("a", "cpu", 0, 1, 1),
        ("b", "cuda", 2, 4, 8),
    ):
        runtime_root = tmp_path / f"SENTINEL_{suffix}_RUNTIME_ROOT"
        reuse_root = tmp_path / f"reuse-{suffix}"
        reuse_root.mkdir()
        bundle_root = tmp_path / f"SENTINEL_{suffix}_BUNDLE_ROOT"
        bundle_input = tmp_path / f"SENTINEL_{suffix}_BUNDLE_INPUT.tar"
        profile = tmp_path / f"profile-{suffix}.yaml"
        profile.write_text(
            "\n".join((
                "schema_version: expert_method.runtime_profile.v1",
                f"profile_id: runtime-{suffix}",
                f"data_root: {tmp_path / f'SENTINEL_{suffix}_DATA_ROOT'}",
                f"run_root: {runtime_root}",
                "reuse_roots:",
                f"  rs3: {reuse_root}",
                f"device: {device}",
                f"shard_index: {shard_index}",
                f"shard_count: {shard_count}",
                f"max_jobs: {max_jobs}",
                "bundle_inputs:",
                f"  - {bundle_input}",
                f"bundle_output_dir: {bundle_root}",
                "",
            )),
            encoding="utf-8",
        )
        profiles.append((profile, runtime_root, bundle_root, bundle_input))

    freezes = []
    for profile, _runtime_root, _bundle_root, _bundle_input in profiles:
        assert cli_main([
            "--config", str(STUDY_PATH), "--profile", str(profile), "study", "freeze"
        ]) == 0
        freezes.append(json.loads(capsys.readouterr().out))

    freeze_paths = [Path(result["path"]) for result in freezes]
    first_bytes, second_bytes = (path.read_bytes() for path in freeze_paths)
    assert first_bytes == second_bytes
    first = json.loads(first_bytes)
    assert first["runtime_root_names"] == ["rs3"]
    assert "runtime_profile" not in first
    for profile, runtime_root, bundle_root, bundle_input in profiles:
        for sentinel in (str(runtime_root), str(bundle_root), str(bundle_input), "SENTINEL_"):
            assert sentinel.encode() not in first_bytes
    assert freezes[0]["freeze_sha256"] == freezes[1]["freeze_sha256"]


def test_session_exports_valid_jobs_when_a_later_training_attempt_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The training boundary may fail mid-batch without losing earlier valid output."""
    import json
    from types import SimpleNamespace

    import torch

    from data.nested_oof import OOFPredictionArtifact, OOFPredictionRecord
    from models.resnet32 import ResNet32
    from expert_method.oof.pipeline import OOFRunSpec

    run_root = tmp_path / "SENTINEL_session-artifacts"
    bundle_root = tmp_path / "SENTINEL_session-bundles"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (ROOT / "configs" / "profiles" / "local-smoke.yaml").read_text()
        .replace("data_root: data", f"data_root: {tmp_path / 'SENTINEL_data'}")
        .replace("run_root: runs/local-smoke", f"run_root: {run_root}")
        .replace("max_jobs: 1", "max_jobs: 2")
        .replace("bundle_output_dir: runs/local-smoke/bundles", f"bundle_output_dir: {bundle_root}")
    )
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    manager = NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda *_args: manager)
    source = cli_main.__globals__["_source_identity"]
    actual_commit = source(ROOT)["commit"]
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": actual_commit, "dirty": False},
    )

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "freeze"
    ]) == 0
    capsys.readouterr()

    calls = 0

    def training_boundary(pipeline, spec: OOFRunSpec, base_config, **run_options):
        nonlocal calls
        calls += 1
        phase_callback = run_options.get("phase_callback")
        if phase_callback is not None:
            phase_callback("training")
        if calls == 2:
            context = spec.resolve(pipeline.manager)
            config = base_config.replace(
                seed=context.training_seed,
                device=spec.device,
                epochs=spec.epochs,
                checkpoint_dir=str(pipeline.store.run_dir(context) / "checkpoints"),
            )
            provenance_dir = Path(run_options.get("provenance_run_dir", pipeline.store.run_dir(context)))
            provenance_config = base_config.replace(
                seed=context.training_seed,
                device=spec.device,
                epochs=spec.epochs,
                checkpoint_dir=str(provenance_dir / "checkpoints"),
            )
            resolved_config = provenance_config.to_dict()
            resolved_config["resolved_device"] = config.resolved_device
            pipeline.store.prepare_run(context, resolved_config)
            pipeline.store.checkpoint_path(context).write_bytes(b"partial checkpoint for failure fixture")
            metrics_sink = run_options.get("metrics_sink")
            if metrics_sink is not None:
                metrics_sink({"epoch": 1, "train_loss": 1.25})
            print("synthetic training was interrupted")
            raise RuntimeError("synthetic interruption after one completed job")
        context = spec.resolve(pipeline.manager)
        config = base_config.replace(
            seed=context.training_seed,
            device=spec.device,
            epochs=spec.epochs,
            checkpoint_dir=str(pipeline.store.run_dir(context) / "checkpoints"),
        )
        provenance_dir = Path(run_options.get("provenance_run_dir", pipeline.store.run_dir(context)))
        provenance_config = base_config.replace(
            seed=context.training_seed,
            device=spec.device,
            epochs=spec.epochs,
            checkpoint_dir=str(provenance_dir / "checkpoints"),
        )
        resolved_config = provenance_config.to_dict()
        resolved_config["resolved_device"] = config.resolved_device
        pipeline.store.prepare_run(context, resolved_config)
        if calls == 1:
            canonical_run_dir = Path(provenance_dir)
            stale_lock = canonical_run_dir.parent / f".{canonical_run_dir.name}.promotion.lock"
            stale_lock.parent.mkdir(parents=True, exist_ok=True)
            stale_lock.write_text("left behind by a crashed process")
        metrics_sink = run_options.get("metrics_sink")
        if metrics_sink is not None:
            metrics_sink({"epoch": 1, "train_loss": 1.0})
            metrics_sink({"epoch": spec.epochs, "train_loss": 0.5})
        checkpoint_path = pipeline.store.checkpoint_path(context)
        model = ResNet32(num_classes=100)
        torch.save({
            "epoch": spec.epochs,
            "seed": context.training_seed,
            "expert_name": context.expert_name,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": {},
            "is_final": True,
            "log": {},
        }, checkpoint_path)
        checkpoint_sha256 = pipeline.store.record_checkpoint(context, checkpoint_path)
        labels_by_id = dict(zip(pipeline.manager.canonical_indices, pipeline.manager.training_labels))
        records = tuple(
            OOFPredictionRecord.create(
                sample_index=sample_id,
                training_label=int(labels_by_id[sample_id]),
                outer_fold_id=context.outer_fold_id,
                inner_fold_id=context.inner_fold_id,
                expert_id=context.expert_name,
                training_seed=context.training_seed,
                expert_training_membership_hash=context.training_membership_hash,
                checkpoint_path=str(provenance_dir / "checkpoints" / checkpoint_path.name),
                checkpoint_sha256=checkpoint_sha256,
                resolved_config=resolved_config,
                logits=np.zeros(pipeline.manager.num_classes, dtype=np.float32),
            )
            for sample_id in context.prediction_indices
        )
        artifact = OOFPredictionArtifact(
            expert_order=pipeline.manager.expert_order,
            num_classes=pipeline.manager.num_classes,
            records=records,
        )
        prediction_path = pipeline.store.write_predictions(context, artifact)
        return SimpleNamespace(
            checkpoint_sha256=checkpoint_sha256,
            prediction_path=prediction_path,
        )

    monkeypatch.setattr("expert_method.oof.pipeline.OOFPipeline.run", training_boundary)
    monkeypatch.setattr("expert_method.ridge_sinkhorn.matrix.cuda_preflight", lambda _device: None)
    result = cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "session",
        "--stage", "inner", "--execute-full",
    ])
    output = capsys.readouterr().out
    payload = json.loads(output[output.rfind("\n{") + 1:])
    assert result == 1
    assert calls == 2
    assert payload["success"] is False
    assert payload["completed_count"] == 1
    assert payload["failures"][0]["error"] == "synthetic interruption after one completed job"
    completed = payload["completed"][0]
    completed_attempt = Path(completed["attempt_path"])
    completed_record = json.loads((completed_attempt / "attempt.json").read_text())
    assert completed_record["status"] == "succeeded"
    assert completed["canonical_path"] == completed_record["locations"]["canonical_job"]
    assert Path(completed["canonical_path"]).is_dir()
    canonical_run = Path(completed["canonical_path"])
    canonical_metadata = __import__("json").loads((canonical_run / "run_metadata.json").read_text())
    prediction_artifact = __import__("json").loads((canonical_run / "predictions.json").read_text())
    expected_checkpoint = canonical_run / canonical_metadata["checkpoint"]["path"]
    assert prediction_artifact["records"][0]["checkpoint_path"] == str(expected_checkpoint)
    assert [json.loads(line)["epoch"] for line in (completed_attempt / "metrics.jsonl").read_text().splitlines()] == [1, 200]
    failure = payload["failures"][0]
    assert failure["attempt_id"]
    failed_attempt = Path(failure["attempt_path"])
    attempt_record = json.loads((failed_attempt / "attempt.json").read_text())
    assert attempt_record["status"] == "failed"
    assert attempt_record["job_id"] == failure["job_id"]
    assert attempt_record["failure"]["exception_type"] == "RuntimeError"
    assert "synthetic interruption" in attempt_record["failure"]["traceback"]
    assert (failed_attempt / "resolved_config.json").is_file()
    assert "synthetic training was interrupted" in (failed_attempt / "execution.log").read_text()
    assert any(failed_attempt.rglob("*.pt")), "failed attempt did not retain its partial checkpoint"
    assert [json.loads(line)["epoch"] for line in (failed_attempt / "metrics.jsonl").read_text().splitlines()] == [1]
    assert payload["bundle_path"] is not None
    bundle_path = Path(payload["bundle_path"])
    assert bundle_path.is_file()
    from expert_method.ridge_sinkhorn.matrix import _read_bundle

    bundle_manifest, bundle_payloads = _read_bundle(bundle_path)
    freeze_member = "ridge_sinkhorn_3seed_v1/manifests/study-freeze.json"
    bundled_freeze = bundle_payloads[freeze_member]
    assert b"runtime_profile" not in bundled_freeze
    assert b"SENTINEL_" not in bundled_freeze
    assert str(run_root).encode() not in bundled_freeze
    assert str(bundle_root).encode() not in bundled_freeze
    selected = bundle_manifest["shard"]["selected_job_ids"]
    assert len(selected) == 1
    assert selected[0] != failure["job_id"]
    assert not any("/attempts/" in record["path"] for record in bundle_manifest["files"])
    assert not (run_root / "ridge_sinkhorn_3seed_v1" / "study_analysis" / "study_lock.json").exists()
    assert not (run_root / "ridge_sinkhorn_3seed_v1" / "study_analysis" / "results.json").exists()

    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "status"
    ]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["attempts"]["failed"] == 1
    assert status["attempts"]["succeeded"] == 1
    assert status["counts"]["inner"]["invalid"] == 0

    # A retry gets a fresh attempt ID and leaves the failed attempt untouched.
    assert cli_main([
        "--config", str(STUDY_PATH), "--profile", str(profile), "study", "session",
        "--stage", "inner", "--execute-full",
    ]) == 0
    retry_output = capsys.readouterr().out
    retry_payload = json.loads(retry_output[retry_output.rfind("\n{") + 1:])
    retried = next(row for row in retry_payload["completed"] if row["job_id"] == failure["job_id"])
    assert retried["attempt_id"] != failure["attempt_id"]
    assert json.loads((failed_attempt / "attempt.json").read_text())["status"] == "failed"
    assert json.loads((Path(retried["attempt_path"]) / "attempt.json").read_text())["status"] == "succeeded"


def test_concurrent_batches_promote_a_job_at_most_once(tmp_path: Path) -> None:
    """Spawned promotion writers serialize while reusing a stale lock inode."""
    import json
    import multiprocessing

    from tests.integration.promotion_worker import promote_in_spawned_process

    process_context = multiprocessing.get_context("spawn")
    expected_config = {"protocol": "frozen-fixture-v1"}
    canonical_run = tmp_path / "canonical" / "job-1"
    canonical_run.parent.mkdir(parents=True)
    lock_path = canonical_run.parent / f".{canonical_run.name}.promotion.lock"
    lock_path.write_text("left over after the locking process crashed")
    attempts = [tmp_path / "attempt-1", tmp_path / "attempt-2"]
    for index, path in enumerate(attempts, start=1):
        path.mkdir()
        (path / "validated.json").write_text(
            json.dumps(expected_config), encoding="utf-8"
        )
        (path / "writer.txt").write_text(str(index), encoding="utf-8")
    start_barrier = process_context.Barrier(2)
    result_queue = process_context.Queue()
    processes = [
        process_context.Process(
            target=promote_in_spawned_process,
            args=(str(source), str(canonical_run), expected_config, start_barrier, result_queue),
        )
        for index, source in enumerate(attempts, start=1)
    ]
    started = []
    messages = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        messages = [result_queue.get(timeout=20) for _ in processes]
    finally:
        unjoined = []
        for process in started:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                unjoined.append(process.pid)
        result_queue.close()
        result_queue.join_thread()
        assert not unjoined, f"promotion workers could not be stopped cleanly: {unjoined}"

    assert all(process.exitcode == 0 for process in started)
    assert all(kind == "ok" for kind, _value in messages), messages
    assert sorted(value for _kind, value in messages) == [False, True]
    assert json.loads((canonical_run / "validated.json").read_text()) == expected_config
    assert (canonical_run / "writer.txt").read_text() in {"1", "2"}
    assert sum(path.exists() for path in attempts) == 1
    assert lock_path.is_file(), "promotion must not unlink the persistent lock inode"
