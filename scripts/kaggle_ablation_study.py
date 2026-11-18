"""Paste into a Kaggle cell to run OT ablations and publish JSON reports."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


REPOSITORY = "https://github.com/Dzzz83/MoE_Imbalance.git"
SOURCE_REF = "ablation_study"
# Optional exact published commit. Otherwise the first run resolves SOURCE_REF,
# and later sessions reuse the commit recorded on the results branch.
SOURCE_COMMIT = None
# Run development first. For the follow-up use ("strength",) on saved artifacts.
STAGES = ("prepare", "pilot", "study")
IS_STRENGTH_STUDY = STAGES == ("strength",)
RESULTS_BRANCH = "ablation-strength-results" if IS_STRENGTH_STUDY else "ablation-study-results"
GIT_USER_NAME = "Dzzz83"
GIT_USER_EMAIL = "phattuan43@gmail.com"

CODE_ROOT = Path("/kaggle/working") / ("ot-strength-code" if IS_STRENGTH_STUDY else "ot-ablation-code")
RESULTS_ROOT = Path("/kaggle/working") / ("ot-strength-results" if IS_STRENGTH_STUDY else "ot-ablation-results")
WORK_ROOT = Path("/kaggle/working/ot-ablation")
RAW_CIFAR_ROOT = Path("/kaggle/input/datasets/dzzz833/cifar100")
# Attach a previous saved WORK_ROOT as a Kaggle dataset, then point here at its
# root containing study/. The original stages also need data/train.npz.
RESTORE_ROOT = None
ALLOW_DOWNLOAD = False
RESTART_INTERRUPTED_RUNS = True
# Optional overrides to configs/ablation_study.json, e.g. {"num_workers": 2}.
# Keep these fixed across sessions for the same study.
CONFIG_OVERRIDES = {"num_workers": 2}


def run(command, cwd=None, env=None, capture=False):
    print("+", " ".join(map(str, command)), flush=True)
    result = subprocess.run(
        list(map(str, command)), cwd=cwd, env=env, text=True, check=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_repositories(auth_env):
    if not RESULTS_ROOT.exists():
        remote = run(["git", "ls-remote", "--heads", REPOSITORY, f"refs/heads/{RESULTS_BRANCH}"],
                     env=auth_env, capture=True)
        if remote:
            run(["git", "clone", "--single-branch", "--branch", RESULTS_BRANCH,
                 REPOSITORY, RESULTS_ROOT], env=auth_env)
        else:
            RESULTS_ROOT.mkdir(parents=True)
            run(["git", "init", "--initial-branch", RESULTS_BRANCH], cwd=RESULTS_ROOT)
            run(["git", "remote", "add", "origin", REPOSITORY], cwd=RESULTS_ROOT)
    if run(["git", "remote", "get-url", "origin"], cwd=RESULTS_ROOT, capture=True) != REPOSITORY:
        raise RuntimeError("Unexpected results repository")
    if run(["git", "branch", "--show-current"], cwd=RESULTS_ROOT, capture=True) != RESULTS_BRANCH:
        raise RuntimeError("Unexpected results branch")
    run(["git", "config", "user.name", GIT_USER_NAME], cwd=RESULTS_ROOT)
    run(["git", "config", "user.email", GIT_USER_EMAIL], cwd=RESULTS_ROOT)

    manifest_path = RESULTS_ROOT / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    commit = SOURCE_COMMIT or (previous["source_commit"] if previous else None)
    if commit is None:
        remote = run(["git", "ls-remote", "--heads", REPOSITORY, f"refs/heads/{SOURCE_REF}"],
                     env=auth_env, capture=True)
        if not remote:
            raise RuntimeError(f"Publish the implementation branch {SOURCE_REF!r} before running Kaggle")
        commit = remote.split()[0]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("SOURCE_COMMIT must be a full published Git commit hash")
    if previous and previous["source_commit"] != commit:
        raise RuntimeError("Source commit changed; use a new results branch and study folder")
    if not CODE_ROOT.exists():
        run(["git", "clone", REPOSITORY, CODE_ROOT], env=auth_env)
    if run(["git", "remote", "get-url", "origin"], cwd=CODE_ROOT, capture=True) != REPOSITORY:
        raise RuntimeError("Unexpected source repository")
    if run(["git", "status", "--porcelain"], cwd=CODE_ROOT, capture=True):
        raise RuntimeError("Source checkout has local edits")
    run(["git", "fetch", "origin", commit], cwd=CODE_ROOT, env=auth_env)
    run(["git", "checkout", "--detach", commit], cwd=CODE_ROOT)
    if run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, capture=True) != commit:
        raise RuntimeError("Source commit mismatch")
    if not (CODE_ROOT / "pipeline/run_ablation.py").is_file():
        raise RuntimeError("The source commit does not contain the ablation implementation")
    return commit


def restore_workspace():
    if RESTORE_ROOT is None:
        return
    source = Path(RESTORE_ROOT)
    if not (source / "study/study_config.json").is_file():
        raise ValueError("RESTORE_ROOT must contain the saved study/ directory")
    if not IS_STRENGTH_STUDY and not (source / "data/train.npz").is_file():
        raise ValueError("RESTORE_ROOT must also contain data/train.npz for the original stages")
    if WORK_ROOT.exists():
        # Re-running the cell should not overwrite its current local artifacts.
        print(f"Using existing local workspace {WORK_ROOT}; restore skipped", flush=True)
        return
    shutil.copytree(source, WORK_ROOT, symlinks=True)


def prepare_data():
    from pipeline.data import prepare_data as prepare_retained_data

    data_root = WORK_ROOT / "data"
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    attached = RAW_CIFAR_ROOT
    if (attached / "cifar-100-python").is_dir():
        attached = attached / "cifar-100-python"
    if attached.is_dir() and (attached / "train").is_file():
        link = raw_root / "cifar-100-python"
        if link.is_symlink() and not link.exists():
            link.unlink()  # A previous session's dataset mount may have moved.
        if not link.exists():
            link.symlink_to(attached, target_is_directory=True)
    prepared = data_root / "train.npz"
    if not prepared.exists():
        prepare_retained_data(raw_root, prepared, imbalance_factor=0.01, seed=42, download=ALLOW_DOWNLOAD)
    return prepared, raw_root


def archive_interrupted_runs(study_root):
    patterns = ("baselines/*", "pilot/*", "study/*/*", "final/baselines/*",
                "final/runs/*/*", "final/evaluation/*/*")
    interrupted = [path for pattern in patterns for path in study_root.glob(pattern)
                   if path.is_dir() and not (path / "complete.json").exists()
                   and not (path / "pilot_failure.json").exists()]
    if interrupted and not RESTART_INTERRUPTED_RUNS:
        raise RuntimeError("Interrupted runs found; enable RESTART_INTERRUPTED_RUNS to archive and restart them")
    for path in interrupted:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = WORK_ROOT / "interruptions" / stamp / path.relative_to(study_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)
        print(f"Archived {path}; its next attempt starts from the beginning", flush=True)


def report_path(relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("Invalid published report path")
    return RESULTS_ROOT / "reports" / str(relative)


def publish_reports(study_root, source_commit, message, auth_env):
    """Publish JSON only. Training data, feature caches, and models stay local."""
    manifest_path = RESULTS_ROOT / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["source_commit"] != source_commit:
        raise RuntimeError("Cannot mix source commits on the results branch")
    files = {}
    for source in sorted(study_root.rglob("*.json")):
        relative = source.relative_to(study_root).as_posix()
        if source.stat().st_size > 95 * 1024 * 1024:
            raise RuntimeError(f"Report exceeds the Git file-size limit: {source}")
        # Validate JSON before replacing a previously published report.
        json.loads(source.read_text())
        destination = report_path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative] = sha256(source)
    previous_files = previous["report_hashes"] if previous else {}
    for relative in previous_files:
        if relative not in files:
            report_path(relative).unlink(missing_ok=True)
    write_json(manifest_path, {
        "source_commit": source_commit, "source_ref": SOURCE_REF, "runner_version": 1,
        "report_hashes": files,
        "artifact_policy": "JSON reports only; preserve Kaggle study/ and data/ to reuse training artifacts",
    })
    is_strength_study = "strength_config.json" in files
    title = "Stronger OT ablation results" if is_strength_study else "OT reweighting ablation results"
    final_location = "" if is_strength_study else "Final test reports, when explicitly run: `reports/final/`.\n"
    (RESULTS_ROOT / "README.md").write_text(
        f"# {title}\n\n"
        f"Source commit: `{source_commit}`.\n\n"
        "Development reports: `reports/study_results.json` and `reports/study_summary.json`.\n"
        f"{final_location}\n"
        "The manifest lists current report files and their hashes. Scores are fractions.\n"
        "Datasets, cached features, and checkpoints are kept in Kaggle outputs, not Git.\n"
    )
    published_paths = ["manifest.json", "README.md"] + [
        report_path(relative).relative_to(RESULTS_ROOT).as_posix()
        for relative in sorted(set(files) | set(previous_files))
    ]
    run(["git", "add", "--", *published_paths], cwd=RESULTS_ROOT)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *published_paths], cwd=RESULTS_ROOT)
    if changed.returncode not in (0, 1):
        raise RuntimeError("Could not inspect staged reports")
    if changed.returncode == 1:
        run(["git", "commit", "--only", "-m", f"results: {message}", "--", *published_paths], cwd=RESULTS_ROOT)
    # Retry the push even if the preceding attempt already created its commit.
    run(["git", "push", "origin", f"HEAD:refs/heads/{RESULTS_BRANCH}"], cwd=RESULTS_ROOT, env=auth_env)


def execute_stages(config, source_commit, raw_root, auth_env):
    from pipeline import run_ablation
    from pipeline.runtime import resolve_device

    if (RESULTS_ROOT / "manifest.json").exists() and not (WORK_ROOT / "study/study_config.json").exists():
        raise RuntimeError("Results already exist: restore the saved Kaggle workspace or choose a new results branch")
    config, root, split = run_ablation.prepare_study(config, WORK_ROOT / "study")
    if (RESULTS_ROOT / "manifest.json").exists():
        for relative, expected in (("study_config.json", config), ("split/split.json", split)):
            saved = report_path(relative)
            if not saved.is_file() or json.loads(saved.read_text()) != expected:
                raise RuntimeError("The restored workspace does not match the published study settings and split")
    archive_interrupted_runs(root)
    device = resolve_device("cuda")

    def publish(message):
        publish_reports(root, source_commit, message, auth_env)

    publish("initialize ablation study")
    for stage in STAGES:
        if stage == "pilot":
            run_ablation.run_pilot(config, root, split, device, on_progress=publish)
        elif stage == "study":
            run_ablation.run_study(config, root, split, device, on_progress=publish)
        elif stage == "freeze":
            run_ablation.freeze_recipe(config, root)
        elif stage == "final":
            run_ablation.run_final(config, root, split, device, raw_root,
                                   download=ALLOW_DOWNLOAD, on_progress=publish)
        publish(f"complete {stage}")


def execute_strength(source_commit, auth_env):
    from pipeline.run_ablation import run_strength_study
    from pipeline.runtime import resolve_device

    original = WORK_ROOT / "study"
    if not (original / "study_summary.json").is_file():
        raise RuntimeError("Restore the completed Kaggle development study using RESTORE_ROOT")
    output = original / "strength"
    manifest = RESULTS_ROOT / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())["report_hashes"]
        required = ["strength_config.json", *(path for path in previous if path.endswith("/complete.json"))]
        for path in required:
            relative = report_path(path).relative_to(RESULTS_ROOT / "reports")
            local = output / relative
            if not local.is_file() or sha256(local) != previous.get(path):
                raise RuntimeError("Restore the latest stronger OT workspace to continue publishing")
    archive_interrupted_runs(output)

    def publish(message):
        publish_reports(output, source_commit, message, auth_env)

    run_strength_study(original, resolve_device("cuda"), on_progress=publish)
    publish("complete stronger OT study")


def main():
    allowed = ("prepare", "pilot", "study", "freeze", "final")
    if not IS_STRENGTH_STUDY:
        if not STAGES or len(set(STAGES)) != len(STAGES) or any(stage not in allowed for stage in STAGES):
            raise ValueError('Use ("strength",) alone, or distinct prepare/pilot/study/freeze/final stages')
        if list(STAGES) != sorted(STAGES, key=allowed.index):
            raise ValueError("STAGES must follow prepare, pilot, study, freeze, final order")
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Enable the Kaggle secret GITHUB_TOKEN with repository write access")
    os.environ.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    os.environ.pop("GITHUB_TOKEN", None)
    os.environ.pop("GIT_ASKPASS", None)
    with tempfile.TemporaryDirectory(prefix="ablation-git-") as helper_dir:
        askpass = Path(helper_dir) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in\n'
            '  *Username*) printf "%s\\n" "$GIT_AUTH_USER" ;;\n'
            '  *Password*) printf "%s\\n" "$GITHUB_TOKEN" ;;\nesac\n'
        )
        askpass.chmod(0o700)
        auth_env = os.environ | {
            "GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTH_USER": GIT_USER_NAME, "GITHUB_TOKEN": token,
        }
        source_commit = prepare_repositories(auth_env)
        sys.path.insert(0, str(CODE_ROOT))
        import pipeline

        if Path(pipeline.__file__).resolve().parent != CODE_ROOT / "pipeline":
            raise RuntimeError("Restart the notebook kernel: another pipeline checkout is already imported")
        import torch
        import torchvision

        print(f"torch={torch.__version__}, torchvision={torchvision.__version__}", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU accelerator in Kaggle")
        restore_workspace()
        if IS_STRENGTH_STUDY:
            execute_strength(source_commit, auth_env)
        else:
            prepared, raw_root = prepare_data()
            config = json.loads((CODE_ROOT / "configs/ablation_study.json").read_text()) | CONFIG_OVERRIDES
            config["train_data"] = str(prepared)
            execute_stages(config, source_commit, raw_root, auth_env)
        print(f"Reports pushed to {RESULTS_BRANCH}. Preserve {WORK_ROOT} in Kaggle outputs.", flush=True)


if __name__ == "__main__":
    main()
