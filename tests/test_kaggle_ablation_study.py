import json
import os
from pathlib import Path
import subprocess

import pytest

from pipeline import run_ablation
from scripts import kaggle_ablation_study as kaggle


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


@pytest.fixture
def results_repository(tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir()
    remote = tmp_path / "remote.git"
    git("init", "--bare", str(remote), cwd=tmp_path)
    git("init", "--initial-branch", kaggle.RESULTS_BRANCH, cwd=root)
    git("config", "user.name", "Ablation tests", cwd=root)
    git("config", "user.email", "ablation@example.invalid", cwd=root)
    git("remote", "add", "origin", str(remote), cwd=root)
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", root)
    return root


def test_publication_pushes_reports_only_and_retries_existing_commit(tmp_path, results_repository, monkeypatch):
    study = tmp_path / "study"
    kaggle.write_json(study / "study_summary.json", {"balanced_accuracy": 0.4})
    kaggle.write_json(study / "split/split.json", {"fitting_positions": [1, 2]})
    (study / "split/fitting.npz").write_bytes(b"training images")
    (study / "final.pt").write_bytes(b"checkpoint")
    (study / "features.pt").write_bytes(b"features")
    # Even an unrelated pre-staged file must not enter the publication commit.
    (results_repository / "private.pt").write_bytes(b"unrelated staged artifact")
    git("add", "private.pt", cwd=results_repository)
    real_run = kaggle.run
    pushes = []

    def fail_first_push(command, **kwargs):
        if command[:2] == ["git", "push"]:
            pushes.append(command)
            if len(pushes) == 1:
                raise subprocess.CalledProcessError(1, command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(kaggle, "run", fail_first_push)
    with pytest.raises(subprocess.CalledProcessError):
        kaggle.publish_reports(study, "a" * 40, "save pilot", os.environ.copy())
    assert git("rev-list", "--count", "HEAD", cwd=results_repository) == "1"
    kaggle.publish_reports(study, "a" * 40, "retry pilot", os.environ.copy())
    assert len(pushes) == 2
    assert git("rev-list", "--count", "HEAD", cwd=results_repository) == "1"
    tracked = git("ls-tree", "-r", "--name-only", "HEAD", cwd=results_repository).splitlines()
    assert tracked == ["README.md", "manifest.json", "reports/split/split.json", "reports/study_summary.json"]
    manifest = json.loads((results_repository / "manifest.json").read_text())
    assert manifest["report_hashes"]["study_summary.json"] == kaggle.sha256(study / "study_summary.json")
    (study / "study_summary.json").unlink()
    kaggle.publish_reports(study, "a" * 40, "remove obsolete summary", os.environ.copy())
    assert not (results_repository / "reports/study_summary.json").exists()
    assert (results_repository / "private.pt").read_bytes() == b"unrelated staged artifact"
    with pytest.raises(RuntimeError, match="mix source"):
        kaggle.publish_reports(study, "b" * 40, "changed source", {})


@pytest.mark.parametrize("path", ["../outside.json", "/absolute.json", "checkpoints/final.pt"])
def test_report_paths_cannot_escape_or_publish_binary_artifacts(path):
    with pytest.raises(ValueError, match="report path"):
        kaggle.report_path(path)


def test_interrupted_runs_are_archived_without_touching_completed_or_failed_pilots(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path)
    study = tmp_path / "study"
    kaggle.write_json(study / "baselines/seed0/config.json", {"seed": 0})
    kaggle.write_json(study / "pilot/lr_0.02/pilot_failure.json", {"reason": "Nonfinite loss"})
    kaggle.write_json(study / "study/seed0/ot/complete.json", {"completed": True})
    monkeypatch.setattr(kaggle, "RESTART_INTERRUPTED_RUNS", False)
    with pytest.raises(RuntimeError, match="Interrupted runs"):
        kaggle.archive_interrupted_runs(study)
    assert (study / "baselines/seed0/config.json").exists()
    monkeypatch.setattr(kaggle, "RESTART_INTERRUPTED_RUNS", True)
    kaggle.archive_interrupted_runs(study)
    assert not (study / "baselines/seed0").exists()
    saved = list((tmp_path / "interruptions").glob("*/baselines/seed0/config.json"))
    assert len(saved) == 1
    assert (study / "pilot/lr_0.02/pilot_failure.json").exists()
    assert (study / "study/seed0/ot/complete.json").exists()


def test_restore_preserves_artifacts_and_does_not_overwrite_local_workspace(tmp_path, monkeypatch):
    source = tmp_path / "attached"
    kaggle.write_json(source / "study/study_config.json", {"seed": 0})
    (source / "data").mkdir()
    (source / "data/train.npz").write_bytes(b"original pool")
    working = tmp_path / "working"
    monkeypatch.setattr(kaggle, "RESTORE_ROOT", source)
    monkeypatch.setattr(kaggle, "WORK_ROOT", working)
    monkeypatch.setattr(kaggle, "IS_STRENGTH_STUDY", False)
    kaggle.restore_workspace()
    assert (working / "data/train.npz").read_bytes() == b"original pool"
    (working / "data/train.npz").write_bytes(b"local edits")
    kaggle.restore_workspace()
    assert (working / "data/train.npz").read_bytes() == b"local edits"

    (source / "data/train.npz").unlink()
    strength_working = tmp_path / "strength-working"
    (strength_working / "data").mkdir(parents=True)
    (strength_working / "data/train.npz").write_bytes(b"data from an earlier attempt")
    monkeypatch.setattr(kaggle, "WORK_ROOT", strength_working)
    with pytest.raises(ValueError, match="data/train.npz"):
        kaggle.restore_workspace()
    monkeypatch.setattr(kaggle, "IS_STRENGTH_STUDY", True)
    kaggle.restore_workspace()
    assert (strength_working / "study/study_config.json").is_file()
    assert (strength_working / "data/train.npz").read_bytes() == b"data from an earlier attempt"


def test_repository_checkout_reuses_published_source_pin(tmp_path, monkeypatch):
    code = tmp_path / "code"
    (code / "pipeline").mkdir(parents=True)
    (code / "pipeline/run_ablation.py").write_text("# existing implementation\n")
    results = tmp_path / "results"
    kaggle.write_json(results / "manifest.json", {"source_commit": "a" * 40})
    monkeypatch.setattr(kaggle, "CODE_ROOT", code)
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", results)
    monkeypatch.setattr(kaggle, "SOURCE_COMMIT", None)
    commands = []

    def fake_git(command, **kwargs):
        commands.append(command)
        if command[1:4] == ["remote", "get-url", "origin"]:
            return kaggle.REPOSITORY
        if command[1:3] == ["branch", "--show-current"]:
            return kaggle.RESULTS_BRANCH
        if command[1:3] == ["status", "--porcelain"]:
            return ""
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "a" * 40
        return ""

    monkeypatch.setattr(kaggle, "run", fake_git)
    assert kaggle.prepare_repositories({}) == "a" * 40
    assert ["git", "fetch", "origin", "a" * 40] in commands
    assert not any(command[1] == "ls-remote" for command in commands)
    monkeypatch.setattr(kaggle, "SOURCE_COMMIT", "b" * 40)
    with pytest.raises(RuntimeError, match="Source commit changed"):
        kaggle.prepare_repositories({})


def test_development_stages_publish_progress_without_freezing_or_running_test(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "working")
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "STAGES", ("prepare", "pilot", "study"))
    monkeypatch.setattr("pipeline.runtime.resolve_device", lambda device: "cpu")
    root = kaggle.WORK_ROOT / "study"
    monkeypatch.setattr(run_ablation, "prepare_study", lambda config, path: (config, root, {}))
    calls, publications = [], []

    def pilot(*args, on_progress):
        calls.append("pilot")
        on_progress("pilot run complete")

    def study(*args, on_progress):
        calls.append("study")
        on_progress("study run complete")

    def forbidden(*args, **kwargs):
        pytest.fail("Development defaults must not freeze the recipe or evaluate test data")

    monkeypatch.setattr(run_ablation, "run_pilot", pilot)
    monkeypatch.setattr(run_ablation, "run_study", study)
    monkeypatch.setattr(run_ablation, "freeze_recipe", forbidden)
    monkeypatch.setattr(run_ablation, "run_final", forbidden)
    monkeypatch.setattr(kaggle, "publish_reports", lambda root, commit, message, auth: publications.append(message))
    kaggle.execute_stages({}, "a" * 40, tmp_path / "raw", {})
    assert calls == ["pilot", "study"]
    assert publications == ["initialize ablation study", "complete prepare", "pilot run complete", "complete pilot",
                            "study run complete", "complete study"]
    kaggle.write_json(kaggle.RESULTS_ROOT / "manifest.json", {"source_commit": "a" * 40})
    with pytest.raises(RuntimeError, match="restore the saved Kaggle workspace"):
        kaggle.execute_stages({}, "a" * 40, tmp_path / "raw", {})
    kaggle.write_json(root / "study_config.json", {})
    kaggle.write_json(kaggle.RESULTS_ROOT / "reports/study_config.json", {"different": True})
    kaggle.write_json(kaggle.RESULTS_ROOT / "reports/split/split.json", {})
    with pytest.raises(RuntimeError, match="does not match the published study"):
        kaggle.execute_stages({}, "a" * 40, tmp_path / "raw", {})


def test_strength_stage_publishes_separately_and_rejects_an_older_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "working")
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr("pipeline.runtime.resolve_device", lambda device: "cpu")
    original = kaggle.WORK_ROOT / "study"
    strength = original / "strength"
    publications = []
    with monkeypatch.context() as old_checkout:
        old_checkout.delattr(run_ablation, "run_strength_study")
        with pytest.raises(RuntimeError, match="lacks the stronger OT stage"):
            kaggle.execute_strength("a" * 40, {})
    with pytest.raises(RuntimeError, match="Restore the completed"):
        kaggle.execute_strength("a" * 40, {})
    kaggle.write_json(original / "study_summary.json", {})

    def run_strength(root, device, *, on_progress):
        assert root == original and device == "cpu"
        kaggle.write_json(strength / "strength_config.json", {"weight_exponent": 3})
        on_progress("save stronger OT run")

    monkeypatch.setattr(run_ablation, "run_strength_study", run_strength)
    monkeypatch.setattr(kaggle, "publish_reports",
                        lambda root, commit, message, auth: publications.append((root, message)))
    kaggle.execute_strength("a" * 40, {})
    assert publications == [(strength, "save stronger OT run"), (strength, "complete stronger OT study")]
    kaggle.write_json(kaggle.RESULTS_ROOT / "manifest.json", {"report_hashes": {
        "strength_config.json": kaggle.sha256(strength / "strength_config.json"),
        "study/seed0/strong_ot/complete.json": "published",
    }})
    with pytest.raises(RuntimeError, match="Restore the latest stronger OT workspace"):
        kaggle.execute_strength("a" * 40, {})
    assert len(publications) == 2


def test_initial_checkout_resolves_a_real_branch_and_keeps_source_clean(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "pipeline").mkdir(parents=True)
    (source / "pipeline/run_ablation.py").write_text("# published source\n")
    git("init", "--initial-branch", "ablation_study", cwd=source)
    git("config", "user.name", "Ablation tests", cwd=source)
    git("config", "user.email", "ablation@example.invalid", cwd=source)
    git("add", "pipeline/run_ablation.py", cwd=source)
    git("commit", "-m", "test: create source", cwd=source)
    commit = git("rev-parse", "HEAD", cwd=source)
    monkeypatch.setattr(kaggle, "REPOSITORY", str(source))
    monkeypatch.setattr(kaggle, "CODE_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "SOURCE_COMMIT", None)
    monkeypatch.setattr(kaggle, "SOURCE_REF", "ablation_study")
    assert kaggle.prepare_repositories(os.environ.copy()) == commit
    assert git("rev-parse", "HEAD", cwd=kaggle.CODE_ROOT) == commit
    assert git("status", "--porcelain", cwd=kaggle.CODE_ROOT) == ""
