import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from pipeline import run_ablation
from pipeline.ablation_data import prepare_split
from pipeline.data import TrainingData, save_training_data
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


def test_strength_restores_complete_attached_study_and_preserves_prepared_data(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    attached = inputs / "notebooks" / "saved-output" / "ot-ablation" / "study"
    incomplete = inputs / "incomplete-output"
    working = tmp_path / "working"
    (working / "data").mkdir(parents=True)
    (working / "data/train.npz").write_bytes(b"data from an earlier attempt")
    for folder in (attached, incomplete):
        kaggle.write_json(folder / "study_config.json", {"seeds": [0]})
        kaggle.write_json(folder / "selected_schedule.json", {"learning_rate": 0.002})
        for name in ("study_summary.json", "study_results.json", "implementation.json",
                     "pilot_results.json", "split/split.json"):
            kaggle.write_json(folder / name, {})
        for name in ("split/fitting.npz", "split/development.npz",
                     "baselines/seed0/final.pt", "caches/seed0/features_weights.pt"):
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"saved artifact")
    for control in ("study/seed0/ot", "pilot/lr_0.002"):
        kaggle.write_json(attached / control / "complete.json", {})
        kaggle.write_json(attached / control / "metrics.json", {})
    monkeypatch.setattr(kaggle, "INPUT_ROOT", inputs)
    monkeypatch.setattr(kaggle, "WORK_ROOT", working)
    monkeypatch.setattr(kaggle, "RESTORE_ROOT", None)
    monkeypatch.setattr(kaggle, "IS_STRENGTH_STUDY", True)
    kaggle.restore_workspace()
    assert (working / "study/baselines/seed0/final.pt").read_bytes() == b"saved artifact"
    assert (working / "data/train.npz").read_bytes() == b"data from an earlier attempt"


def test_strength_missing_artifacts_stops_before_git_setup(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    reports = inputs / "github-reports"
    kaggle.write_json(reports / "study_config.json", {"seeds": [0]})
    monkeypatch.setattr(kaggle, "INPUT_ROOT", inputs)
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "working")
    monkeypatch.setattr(kaggle, "RESTORE_ROOT", None)
    monkeypatch.setattr(kaggle, "IS_STRENGTH_STUDY", True)
    monkeypatch.setattr(kaggle, "IS_WEIGHT_AUDIT", False)
    assert kaggle.restore_workspace() is False
    monkeypatch.setattr(kaggle, "REBUILD_MISSING_STUDY", False)
    with pytest.raises(RuntimeError, match="GitHub reports alone cannot restore") as failure:
        kaggle.main()
    assert "caches/seed0/features_weights.pt" in str(failure.value)
    assert not kaggle.WORK_ROOT.exists()


@pytest.mark.parametrize("audit_only", [False, True])
def test_rebuild_uses_published_split_and_schedule_and_pushes_results(tmp_path, monkeypatch, audit_only):
    work = tmp_path / "working"
    original = work / "study"
    output = work / "weight-audit" if audit_only else original / "strength"
    prepared = work / "data/train.npz"
    labels = np.repeat(np.arange(100, dtype=np.int64), 5)
    images = np.random.default_rng(3).integers(0, 256, (len(labels), 32, 32, 3), dtype=np.uint8)
    save_training_data(prepared, TrainingData(images, labels, np.bincount(labels),
                                              {"imbalance_factor": 0.01, "data_seed": 42}))
    config = run_ablation.resolve_study_config({
        "train_data": "/original/train.npz", "seeds": [0], "shuffle_seeds": [0],
        "num_workers": 0, "learning_rates": [0.002],
        "baseline": {"epochs": 1}, "classifier": {"epochs": 1},
    })
    split = prepare_split(prepared, tmp_path / "published-split")
    published = {
        "study_config.json": config,
        "split/split.json": split,
        "pilot_results.json": [{"learning_rate": 0.002, "status": "completed"}],
        "selected_schedule.json": {"learning_rate": 0.002, "pilot_results_sha256": "original"},
    }
    reports_commit = "b" * 40
    commands, publications = [], []

    def read_results(command, **kwargs):
        commands.append(command)
        if command[1] == "rev-parse":
            return reports_commit
        if command[1] == "show":
            return json.dumps(published[command[2].split(":reports/")[1]])
        assert command == ["git", "fetch", "--no-tags", "origin", "refs/heads/ablation-study-results"]

    monkeypatch.setattr(kaggle, "WORK_ROOT", work)
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "run", read_results)
    monkeypatch.setattr(kaggle, "prepare_data", lambda: (prepared, prepared.parent))
    monkeypatch.setattr("pipeline.runtime.resolve_device", lambda device: torch.device("cpu"))
    archive_calls = []

    def upload(study, caches, *_):
        archive_calls.append(study)
        assert caches[0] == kaggle.sha256(study / "caches/seed0/features_weights.pt")
        return {"asset_id": 11, "sha256": "archive", "download_url": "https://example.invalid/ot.zip"}

    def publish(root, commit, message, auth):
        assert (root / "weight_audit.json").is_file() and (root / "weight_artifacts.json").is_file()
        publications.append((root, message))

    monkeypatch.setattr(kaggle, "upload_weight_artifacts", upload)
    monkeypatch.setattr(kaggle, "publish_reports", publish)
    if audit_only:
        kaggle.execute_rebuilt_weight_audit("a" * 40, {})
        assert publications[-1] == (output, "complete rebuilt OT weight audit")
        assert not (original / "study").exists()
        assert (output / "baselines/seed0/complete.json").is_file()
        assert (output / "split/split.json").is_file()
    else:
        kaggle.execute_strength("a" * 40, {}, rebuild=True)
        assert publications[-1] == (output, "complete stronger OT study")
        summary = json.loads((output / "study_summary.json").read_text())
        assert set(summary) == {"class_balanced", "ot", "strong_ot", "shuffled_strong_ot",
                                "strong_ot_vs_shuffled_strong_ot"}
    assert archive_calls == [original]
    assert {root for root, _ in publications} == {output}
    assert all(command[1] in ("fetch", "rev-parse", "show") for command in commands)
    schedule = json.loads((original / "selected_schedule.json").read_text())
    assert schedule["learning_rate"] == 0.002
    assert schedule["pilot_results_sha256"] == kaggle.sha256(original / "pilot_results.json")
    assert json.loads((output / "rebuild.json").read_text())["development_results_commit"] == reports_commit
    assert json.loads((original / "study_config.json").read_text()) == config | {"train_data": str(prepared)}


def test_repository_checkout_reuses_published_source_pin(tmp_path, monkeypatch):
    code = tmp_path / "code"
    (code / "pipeline").mkdir(parents=True)
    (code / "pipeline/run_ablation.py").write_text("# existing implementation\n")
    results = tmp_path / "results"
    kaggle.write_json(results / "manifest.json", {"source_commit": "a" * 40})
    monkeypatch.setattr(kaggle, "CODE_ROOT", code)
    monkeypatch.setattr(kaggle, "IS_WEIGHT_AUDIT", False)
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
        kaggle.write_json(strength / "strength_config.json", {"weight_exponent": 3, "feature_caches": {"0": "abc"}})
        on_progress("save stronger OT run")

    monkeypatch.setattr(run_ablation, "run_strength_study", run_strength)
    monkeypatch.setattr(kaggle, "read_weight_audit_inputs", lambda *args: {})
    monkeypatch.setattr(kaggle, "save_weight_audit", lambda *args: None)
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


def test_weight_audit_uses_matching_artifacts_and_handles_weight_ties(tmp_path, monkeypatch):
    work = tmp_path / "working"
    inputs = tmp_path / "input"
    study = inputs / "saved-notebook" / "study"
    baseline = study / "baselines/seed0/final.pt"
    baseline.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"fc.weight": torch.tensor([[1.], [-1.]]),
                                    "fc.bias": torch.zeros(2)}}, baseline)
    cache = study / "caches/seed0/features_weights.pt"
    cache.parent.mkdir(parents=True)
    torch.save({
        "inputs": {"seed": 0, "baseline_sha256": kaggle.sha256(baseline)},
        "labels": torch.tensor([0] * 4 + [1] * 4),
        "features": torch.tensor([[-2.], [-1.], [1.], [2.], [2.], [-2.], [1.], [-1.]]),
        "ot_weights": torch.tensor([1., 1., 1., 4., 4., 4., 1., 1.]),
        "prototype_diagnostics": [{"class_index": i, "used_ot": True} for i in range(2)],
    }, cache)
    local_study = work / "study"
    local_cache = local_study / "caches/seed0/features_weights.pt"
    local_baseline = local_study / "baselines/seed0/final.pt"
    local_cache.parent.mkdir(parents=True)
    local_baseline.parent.mkdir(parents=True)
    local_cache.write_bytes(cache.read_bytes())
    local_baseline.write_bytes(b"unrelated baseline")
    expected = {"0": kaggle.sha256(cache)}
    publications = []
    manifest = {"report_hashes": {}}
    release = None
    uploaded_zip = None
    interrupted_uploads = []
    asset_downloads = []

    def read_results(command, **kwargs):
        if command[1] == "fetch":
            assert command[-1] == "refs/heads/ablation-strength-results"
        elif command[1] == "rev-parse":
            return "a" * 40
        elif command[1] == "show":
            if command[-1].endswith("strength_config.json"):
                return json.dumps({"feature_caches": expected})
            if command[-1].endswith(":manifest.json"):
                return json.dumps(manifest)
            return (work / "weight-audit/weight_artifacts.json").read_text()
        else:
            pytest.fail(f"Unexpected command: {command}")

    def release_request(url, auth, *, data=None, content_type=None, binary=False, method=None):
        nonlocal release, uploaded_zip
        if url.startswith(kaggle.RELEASES_API + "/tags/"):
            if release is None:
                raise kaggle.HTTPError(url, 404, "Not found", None, None)
            return io.BytesIO(json.dumps(release).encode())
        if url == kaggle.RELEASES_API:
            assert content_type == "application/json"
            assert json.loads(data)["target_commitish"] == "b" * 40
            release = {"id": 7, "assets": []}
            return io.BytesIO(json.dumps(release).encode())
        if method == "DELETE":
            assert url == kaggle.RELEASES_API + "/assets/9"
            assert release["assets"] == [{"id": 9, "name": "ot-weight-inputs.zip", "state": "starter", "size": 0}]
            release["assets"].clear()
            return io.BytesIO()
        if url.startswith(kaggle.RELEASE_UPLOADS + "/7/assets?"):
            assert content_type == "application/zip" and uploaded_zip is None
            if not interrupted_uploads:
                interrupted_uploads.append(url)
                release["assets"].append({"id": 9, "name": "ot-weight-inputs.zip", "state": "starter", "size": 0})
                raise kaggle.HTTPError(url, 502, "Simulated upload failure", None, None)
            uploaded_zip = data
            asset = {
                "id": 11, "name": "ot-weight-inputs.zip", "state": "uploaded", "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "browser_download_url": "https://example.invalid/ot.zip",
            }
            release["assets"].append(asset)
            return io.BytesIO(json.dumps(asset).encode())
        assert binary and url == kaggle.RELEASES_API + "/assets/11"
        asset_downloads.append(url)
        return io.BytesIO(uploaded_zip)

    monkeypatch.setattr(kaggle, "WORK_ROOT", work)
    monkeypatch.setattr(kaggle, "INPUT_ROOT", inputs)
    monkeypatch.setattr(kaggle, "RESTORE_ROOT", None)
    monkeypatch.setattr(kaggle, "run", read_results)
    monkeypatch.setattr(kaggle, "github_release_request", release_request)
    monkeypatch.setattr(kaggle, "publish_reports",
                        lambda root, *_: publications.append(root))
    with pytest.raises(kaggle.HTTPError, match="Simulated upload failure"):
        kaggle.execute_weight_audit("b" * 40, {})
    assert publications == []
    kaggle.execute_weight_audit("b" * 40, {})
    report = json.loads((work / "weight-audit/weight_audit.json").read_text())
    assert report["strength_results_commit"] == "a" * 40
    assert report["seeds"][0]["confidence_percentile_shift"] == pytest.approx(0.1875)
    assert report["seeds"][0]["classes"][0]["baseline_accuracy_shift"] == pytest.approx(0.5)
    assert report["seeds"][0]["classes"][1]["confidence_percentile_shift"] == pytest.approx(0)
    assert publications == [work / "weight-audit"]
    assert json.loads((work / "weight-audit/weight_artifacts.json").read_text())["asset_id"] == 11
    assert kaggle.upload_weight_artifacts(study, {0: expected["0"]}, "b" * 40, {})["asset_id"] == 11
    good_digest = expected["0"]
    expected["0"] = "invalid-cache-hash"
    with pytest.raises(RuntimeError, match="completed stronger OT run"):
        kaggle.execute_weight_audit("b" * 40, {})
    assert publications == [work / "weight-audit"]
    expected["0"] = good_digest
    manifest["report_hashes"]["weight_artifacts.json"] = "published"
    cache.unlink()
    local_cache.unlink()
    kaggle.execute_weight_audit("b" * 40, {})
    assert len(asset_downloads) == 1
    assert json.loads((work / "weight-audit/weight_audit.json").read_text()) == report


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
    monkeypatch.setattr(kaggle, "IS_WEIGHT_AUDIT", False)
    monkeypatch.setattr(kaggle, "CODE_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "SOURCE_COMMIT", None)
    monkeypatch.setattr(kaggle, "SOURCE_REF", "ablation_study")
    assert kaggle.prepare_repositories(os.environ.copy()) == commit
    assert git("rev-parse", "HEAD", cwd=kaggle.CODE_ROOT) == commit
    assert git("status", "--porcelain", cwd=kaggle.CODE_ROOT) == ""
