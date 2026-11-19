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
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZipInfo


REPOSITORY_SLUG = "Dzzz83/MoE_Imbalance"
REPOSITORY = f"https://github.com/{REPOSITORY_SLUG}.git"
RELEASES_API = f"https://api.github.com/repos/{REPOSITORY_SLUG}/releases"
RELEASE_UPLOADS = f"https://uploads.github.com/repos/{REPOSITORY_SLUG}/releases"
SOURCE_REF = "ablation_study"
# Optional exact published commit. Otherwise the first run resolves SOURCE_REF,
# and later sessions reuse the commit recorded on the results branch.
SOURCE_COMMIT = None
# Run Stage B from the published audit features and completed Stage A references.
# Use ("audit",) to audit an existing stronger OT run without training.
STAGES = ("simple_controls",)
IS_SIMPLE_CONTROLS = STAGES == ("simple_controls",)
IS_ASSIGNMENT_STUDY = STAGES == ("assignment",)
IS_WEIGHT_AUDIT = STAGES == ("audit",)
IS_AUDIT_REBUILD = STAGES == ("rebuild_audit",)
IS_STRENGTH_STUDY = STAGES == ("strength",)
DEVELOPMENT_RESULTS_BRANCH = "ablation-study-results"
STRENGTH_RESULTS_BRANCH = "ablation-strength-results"
if IS_SIMPLE_CONTROLS:
    RESULTS_BRANCH, RESULTS_FOLDER = "ablation-simple-controls-results", "ot-simple-controls-results"
elif IS_ASSIGNMENT_STUDY:
    RESULTS_BRANCH, RESULTS_FOLDER = "ablation-assignment-results", "ot-assignment-results"
elif IS_WEIGHT_AUDIT:
    RESULTS_BRANCH, RESULTS_FOLDER = "ablation-weight-audit-results", "ot-weight-audit-results"
elif IS_AUDIT_REBUILD:
    RESULTS_BRANCH, RESULTS_FOLDER = "ablation-rebuilt-audit-results", "ot-rebuilt-audit-results"
elif IS_STRENGTH_STUDY:
    RESULTS_BRANCH, RESULTS_FOLDER = STRENGTH_RESULTS_BRANCH, "ot-strength-results"
else:
    RESULTS_BRANCH, RESULTS_FOLDER = DEVELOPMENT_RESULTS_BRANCH, "ot-ablation-results"
GIT_USER_NAME = "Dzzz83"
GIT_USER_EMAIL = "phattuan43@gmail.com"

CODE_ROOT = Path("/kaggle/working") / (
    "ot-simple-controls-code" if IS_SIMPLE_CONTROLS else
    "ot-assignment-code" if IS_ASSIGNMENT_STUDY else
    "ot-strength-code" if IS_STRENGTH_STUDY else "ot-ablation-code"
)
RESULTS_ROOT = Path("/kaggle/working") / RESULTS_FOLDER
WORK_ROOT = Path("/kaggle/working") / (
    "ot-simple-controls" if IS_SIMPLE_CONTROLS else
    "ot-weight-assignment" if IS_ASSIGNMENT_STUDY else
    "ot-ablation-audit-rebuild" if IS_AUDIT_REBUILD else "ot-ablation"
)
INPUT_ROOT = Path("/kaggle/input")
RAW_CIFAR_ROOT = INPUT_ROOT / "datasets/dzzz833/cifar100"
# Stronger OT finds a single complete study among the attached Kaggle inputs.
# Set this to its saved workspace or study folder to choose one explicitly.
# The original stages expect a workspace with study/ and data/train.npz.
RESTORE_ROOT = None
# If there is no saved workspace, rebuild the fitting baselines and controls.
REBUILD_MISSING_STUDY = True
ALLOW_DOWNLOAD = False
RESTART_INTERRUPTED_RUNS = True
# Optional overrides to configs/ablation_study.json, e.g. {"num_workers": 2}.
# Keep these fixed across sessions for the same study.
CONFIG_OVERRIDES = {"num_workers": 2}
# Assignment settings use their own config; keep research settings fixed.
ASSIGNMENT_CONFIG_OVERRIDES = {}
ASSIGNMENT_PREFLIGHT_ONLY = False
# Stage B settings are frozen; preflight validates references/geometry without training.
SIMPLE_CONTROLS_PREFLIGHT_ONLY = False


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
    if IS_WEIGHT_AUDIT:
        return commit  # The audit applies the saved linear classifier directly.
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


def missing_strength_artifacts(study):
    required = [
        "study_config.json", "study_summary.json", "study_results.json",
        "implementation.json", "selected_schedule.json", "pilot_results.json",
        "split/split.json", "split/fitting.npz", "split/development.npz",
    ]
    config_path = study / "study_config.json"
    seeds = []
    if config_path.is_file():
        try:
            seeds = json.loads(config_path.read_text())["seeds"]
            if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
                raise ValueError("Invalid seed list")
        except (OSError, ValueError, KeyError, TypeError):
            return ["valid study_config.json with training seeds"]
    schedule_path = study / "selected_schedule.json"
    rate = None
    if schedule_path.is_file():
        try:
            rate = json.loads(schedule_path.read_text())["learning_rate"]
            if type(rate) not in (int, float):
                raise ValueError("Invalid learning rate")
        except (OSError, ValueError, KeyError, TypeError):
            return ["valid selected_schedule.json with the pilot learning rate"]
    for seed in seeds:
        required.extend((f"baselines/seed{seed}/final.pt", f"caches/seed{seed}/features_weights.pt"))
        controls = [f"study/seed{seed}/ot"]
        if seed != 0:
            controls.append(f"study/seed{seed}/class_balanced")
        elif rate is not None:
            controls.append(f"pilot/lr_{rate}")
        for control in controls:
            required.extend((f"{control}/complete.json", f"{control}/metrics.json"))
    return [name for name in required if not (study / name).is_file()]


def find_strength_study(local_study):
    if local_study.exists():
        candidates = [local_study]
    elif RESTORE_ROOT is not None:
        source = Path(RESTORE_ROOT)
        candidates = [source if (source / "study_config.json").is_file() else source / "study"]
    else:
        candidates = sorted({path.parent.resolve() for path in INPUT_ROOT.rglob("study_config.json")})

    complete_studies = []
    incomplete_studies = []
    for study in candidates:
        missing = missing_strength_artifacts(study)
        if missing:
            incomplete_studies.append(f"{study}: {', '.join(missing)}")
        else:
            complete_studies.append(study)
    if len(complete_studies) == 1:
        return complete_studies[0]
    if complete_studies:
        raise RuntimeError("Set RESTORE_ROOT to one saved study: " + ", ".join(map(str, complete_studies)))
    details = "\nMissing files by location:\n" + "\n".join(incomplete_studies) if incomplete_studies else ""
    can_rebuild = not local_study.exists() or (local_study.parent / "rebuild.json").is_file()
    if REBUILD_MISSING_STUDY and RESTORE_ROOT is None and can_rebuild:
        print(f"No saved development workspace available. Rebuilding from CIFAR-100.{details}", flush=True)
        return None
    if local_study.exists():
        details += "\nThe incomplete local study was preserved. Use an unused WORK_ROOT to restore a complete copy."
    raise RuntimeError(
        "No complete development study found. Attach the saved output from the successful development run "
        f"under {INPUT_ROOT}. It must include the split, baseline checkpoints, and feature caches. "
        f"GitHub reports alone cannot restore these files.{details}"
    )


def restore_workspace():
    local_study = WORK_ROOT / "study"
    if IS_STRENGTH_STUDY:
        source_study = find_strength_study(local_study)
        if source_study is None:
            return False
        if source_study == local_study:
            return True
        print(f"Restoring stronger OT inputs from {source_study}", flush=True)
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
    else:
        if RESTORE_ROOT is None:
            return
        source = Path(RESTORE_ROOT)
        source_study = source / "study"
        if not (source_study / "study_config.json").is_file():
            raise ValueError("RESTORE_ROOT must contain the saved study/ directory")
        if not (source / "data/train.npz").is_file():
            raise ValueError("RESTORE_ROOT must also contain data/train.npz for the original stages")
        if not WORK_ROOT.exists():
            shutil.copytree(source, WORK_ROOT, symlinks=True)
            return
        if local_study.exists():
            print(f"Using existing local study {local_study}; restore skipped", flush=True)
            return

    # A failed startup can leave prepared data behind without a study directory.
    with tempfile.TemporaryDirectory(dir=WORK_ROOT) as staging_dir:
        staged_study = Path(staging_dir) / "study"
        shutil.copytree(source_study, staged_study, symlinks=True)
        prepared = WORK_ROOT / "data/train.npz"
        if not IS_STRENGTH_STUDY and not prepared.exists():
            prepared.parent.mkdir(exist_ok=True)
            staged_data = Path(staging_dir) / "train.npz"
            shutil.copy2(source_study.parent / "data/train.npz", staged_data)
            staged_data.rename(prepared)
        staged_study.rename(local_study)
    return True


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
        if not ALLOW_DOWNLOAD and not (raw_root / "cifar-100-python/train").is_file():
            raise RuntimeError(f"Attach CIFAR-100 at {RAW_CIFAR_ROOT} or set ALLOW_DOWNLOAD = True.")
        prepare_retained_data(raw_root, prepared, imbalance_factor=0.01, seed=42, download=ALLOW_DOWNLOAD)
    return prepared, raw_root


def archive_interrupted_runs(study_root):
    patterns = ("baselines/*", "pilot/*", "study/*/*", "runs/*/*", "final/baselines/*",
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
    """Publish JSON reports and references to any separately uploaded artifacts."""
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
        "artifact_policy": "JSON in Git; reusable weight audit inputs in GitHub Releases when available",
    })
    is_strength_study = "strength_config.json" in files
    title = "Stronger OT ablation results" if is_strength_study else "OT reweighting ablation results"
    is_assignment = (study_root / "study_assets.json").is_file()
    if is_assignment:
        title = "OT assignment ablation results"
        config = json.loads((study_root / "study_config.json").read_text())
        if config.get("config", {}).get("stage") == "simple_controls":
            title = "OT simple controls ablation results"
    report_locations = ""
    if "study_summary.json" in files or is_strength_study:
        report_locations += "Development reports: `reports/study_results.json` and `reports/study_summary.json`.\n"
    if not is_assignment and not is_strength_study and "weight_audit.json" not in files:
        report_locations += "Final test reports, when explicitly run: `reports/final/`.\n"
    if "weight_audit.json" in files:
        if not is_strength_study:
            title = "OT example weight audit"
        report_locations += "Audit report: `reports/weight_audit.json`.\n"
    if "weight_artifacts.json" in files:
        artifacts = json.loads((study_root / "weight_artifacts.json").read_text())
        report_locations += f"Saved baselines and feature caches: [download ZIP]({artifacts['download_url']}).\n"
    if is_assignment:
        report_locations += "Run status: `reports/study_status.json`. Reusable continuation assets: `reports/study_assets.json`.\n"
    (RESULTS_ROOT / "README.md").write_text(
        f"# {title}\n\n"
        f"Source commit: `{source_commit}`.\n\n"
        f"{report_locations}\n"
        "The manifest lists current report files and their hashes. Scores are fractions.\n"
        "Raw training images are kept out of the published artifacts.\n"
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
        raise RuntimeError(
            "Results already exist for the original study: restore the saved Kaggle workspace with RESTORE_ROOT. "
            'For the stronger OT follow-up, set STAGES = ("strength",) and restore the completed study.'
        )
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


def prepare_rebuilt_study(source_commit, auth_env):
    """Prepare the original fitting split and schedule from published reports."""
    from pipeline import run_ablation
    from pipeline.runtime import atomic_write_json

    marker_path = WORK_ROOT / "rebuild.json"
    saved = json.loads(marker_path.read_text()) if marker_path.exists() else None
    reports_ref = saved["development_results_commit"] if saved else f"refs/heads/{DEVELOPMENT_RESULTS_BRANCH}"
    run(["git", "fetch", "--no-tags", "origin", reports_ref], cwd=CODE_ROOT, env=auth_env)
    reports_commit = run(["git", "rev-parse", "FETCH_HEAD"], cwd=CODE_ROOT, capture=True)
    recovery = {"development_results_commit": reports_commit, "training_source_commit": source_commit}
    if saved is not None and recovery != saved:
        raise RuntimeError("Rebuild inputs changed. Resume with the original source commit.")

    published = {}
    for name in ("study_config.json", "split/split.json", "pilot_results.json", "selected_schedule.json"):
        published[name] = json.loads(run(["git", "show", f"{reports_commit}:reports/{name}"],
                                         cwd=CODE_ROOT, capture=True))
    prepared, _ = prepare_data()
    atomic_write_json(marker_path, recovery)
    config = published["study_config.json"] | {"train_data": str(prepared)}
    config, root, split = run_ablation.prepare_study(config, WORK_ROOT / "study")
    previous_split = published["split/split.json"]
    for name in ("split_seed", "original_class_counts", "fitting_positions", "development_positions"):
        if split[name] != previous_split[name]:
            raise RuntimeError("Rebuilt data does not match the published development split.")

    pilot_path = root / "pilot_results.json"
    atomic_write_json(pilot_path, published["pilot_results.json"])
    schedule = published["selected_schedule.json"] | {"pilot_results_sha256": sha256(pilot_path)}
    atomic_write_json(root / "selected_schedule.json", schedule)
    atomic_write_json(root / "rebuild.json", recovery)
    return config, root, split


def rebuild_development_study(source_commit, auth_env):
    """Recreate the fitting models and the original classifier comparisons."""
    from pipeline import run_ablation
    from pipeline.runtime import resolve_device

    if (RESULTS_ROOT / "manifest.json").exists():
        raise RuntimeError("Stronger OT results already exist. Restore their saved workspace to resume.")
    config, root, split = prepare_rebuilt_study(source_commit, auth_env)
    archive_interrupted_runs(root)
    print("Rebuilding the development baselines and controls.", flush=True)
    run_ablation.run_study(config, root, split, resolve_device("cuda"),
                           on_progress=lambda message: print(message, flush=True))


def execute_strength(source_commit, auth_env, *, rebuild=False):
    from pipeline import run_ablation
    from pipeline.runtime import resolve_device

    if not hasattr(run_ablation, "run_strength_study"):
        raise RuntimeError(f"Source commit {source_commit} lacks the stronger OT stage. Use the updated source branch.")
    if rebuild:
        rebuild_development_study(source_commit, auth_env)
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
    recovery = original / "rebuild.json"
    if recovery.is_file():
        output.mkdir(exist_ok=True)
        shutil.copyfile(recovery, output / recovery.name)

    handoff_saved = False

    def publish(message):
        nonlocal handoff_saved
        if not handoff_saved:
            config = json.loads((output / "strength_config.json").read_text())
            expected = {int(seed): digest for seed, digest in config["feature_caches"].items()}
            bundles = read_weight_audit_inputs(original, expected)
            if bundles is None:
                raise RuntimeError("The exact stronger OT inputs are missing or have changed.")
            save_weight_audit(original, bundles, expected, output, source_commit, auth_env)
            handoff_saved = True
        publish_reports(output, source_commit, message, auth_env)

    run_ablation.run_strength_study(original, resolve_device("cuda"), on_progress=publish)
    publish("complete stronger OT study")


def read_weight_audit_inputs(study, expected_caches):
    import torch

    bundles = {}
    for seed, digest in expected_caches.items():
        cache = study / f"caches/seed{seed}/features_weights.pt"
        baseline = study / f"baselines/seed{seed}/final.pt"
        if not baseline.is_file() or not cache.is_file() or sha256(cache) != digest:
            return None
        bundle = torch.load(cache, map_location="cpu", weights_only=True)
        if bundle["inputs"]["seed"] != seed or sha256(baseline) != bundle["inputs"]["baseline_sha256"]:
            return None
        bundles[seed] = bundle
    return bundles


def find_weight_audit_inputs(expected_caches):
    local = WORK_ROOT / "study"
    if RESTORE_ROOT is not None:
        root = Path(RESTORE_ROOT)
        candidates = [root if (root / "caches").is_dir() else root / "study"]
    else:
        mounted = {p.parents[2] for p in INPUT_ROOT.glob("**/caches/seed*/features_weights.pt")}
        candidates = [local, *sorted(mounted)]
    for study in candidates:
        bundles = read_weight_audit_inputs(study, expected_caches)
        if bundles is not None:
            return study, bundles
    raise RuntimeError(
        "The exact baseline checkpoints and feature caches from the completed stronger OT run "
        "aren't available on GitHub or in this session. If the run's files are still on Kaggle, "
        "run this audit once in that workspace to publish them."
    )


def audit_seed_weights(study, seed, bundle):
    import torch
    from torch.nn.functional import linear

    baseline = study / f"baselines/seed{seed}/final.pt"
    state = torch.load(baseline, map_location="cpu", weights_only=True)["model_state_dict"]
    labels = bundle["labels"]
    with torch.inference_mode():
        logits = linear(bundle["features"], state["fc.weight"], state["fc.bias"])
        if not torch.isfinite(logits).all():
            raise ValueError(f"Nonfinite baseline scores for seed {seed}")
        correct = logits.argmax(dim=1) == labels
        true_scores = logits.gather(1, labels[:, None]).squeeze(1)
        logits.scatter_(1, labels[:, None], -torch.inf)
        margins = true_scores - logits.amax(dim=1)

    classes = []
    for diagnostic in bundle["prototype_diagnostics"]:
        if not diagnostic["used_ot"]:
            continue
        class_index = diagnostic["class_index"]
        members = labels == class_index
        weights = bundle["ot_weights"][members]
        class_margins = margins[members].double()
        top_count = (len(weights) + 3) // 4
        cutoff = weights.topk(top_count).values[-1]
        above, tied = weights > cutoff, weights == cutoff
        # Give tied examples an equal share of the remaining top-quarter places.
        in_top = above.double() + tied.double() * (top_count - above.sum()) / tied.sum()
        _, ranks, counts = torch.unique(class_margins, return_inverse=True, return_counts=True)
        percentiles = (counts.cumsum(0) - counts.double() / 2)[ranks] / len(weights)
        accuracy = correct[members].double()
        classes.append({
            "class_index": class_index, "fitting_count": len(weights),
            "confidence_percentile_shift": (in_top @ percentiles / top_count).item() - 0.5,
            "correct_class_margin_shift": (in_top @ class_margins / top_count - class_margins.mean()).item(),
            "baseline_accuracy_shift": (in_top @ accuracy / top_count - accuracy.mean()).item(),
        })
    shifts = ("confidence_percentile_shift", "correct_class_margin_shift", "baseline_accuracy_shift")
    return {
        "seed": seed, "ot_active_class_count": len(classes),
        **{name: sum(c[name] for c in classes) / len(classes) if classes else None for name in shifts},
        "classes": classes,
    }


def github_release_request(url, auth_env, *, data=None, content_type=None, binary=False, method=None):
    request = Request(url, data=data, method=method, headers={
        "Accept": "application/octet-stream" if binary else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    # An asset download can redirect to signed storage. Keep the token on GitHub.
    request.add_unredirected_header("Authorization", f"Bearer {auth_env['GITHUB_TOKEN']}")
    if content_type:
        request.add_header("Content-Type", content_type)
    return urlopen(request, timeout=120)


def weight_artifact_paths(seeds):
    return [path for seed in sorted(seeds) for path in (
        f"baselines/seed{seed}/final.pt", f"caches/seed{seed}/features_weights.pt",
    )]


def upload_weight_artifacts(study, expected_caches, source_commit, auth_env):
    return upload_artifacts(study, weight_artifact_paths(expected_caches), source_commit, auth_env)


def upload_artifacts(study, paths, source_commit, auth_env, prefix="ot-weight-inputs"):
    """Upload an explicitly enumerated archive; never include data by directory scan."""
    with tempfile.TemporaryDirectory(prefix="ot-artifacts-") as temporary:
        bundle = Path(temporary) / f"{prefix}.zip"
        with ZipFile(bundle, "w") as archive:
            for relative in paths:
                archive.writestr(ZipInfo(relative), (study / relative).read_bytes())
        digest = sha256(bundle)
        tag = f"{prefix}-{digest}"
        try:
            with github_release_request(f"{RELEASES_API}/tags/{tag}", auth_env) as response:
                release = json.load(response)
        except HTTPError as error:
            if error.code != 404:
                raise
            error.close()
            request = json.dumps({
                "tag_name": tag, "target_commitish": source_commit,
                "name": ("OT audit inputs " if prefix == "ot-weight-inputs" else "OT assignment files ") + digest[:12],
                "prerelease": True, "make_latest": "false",
            }).encode()
            with github_release_request(RELEASES_API, auth_env, data=request,
                                        content_type="application/json") as response:
                release = json.load(response)
        asset = next((entry for entry in release["assets"] if entry["name"] == bundle.name), None)
        if asset is not None and asset["state"] == "starter" and asset["size"] == 0:
            # GitHub can leave an empty placeholder after a failed upload.
            with github_release_request(f"{RELEASES_API}/assets/{int(asset['id'])}", auth_env, method="DELETE"):
                pass
            asset = None
        if asset is None:
            upload_url = f"{RELEASE_UPLOADS}/{int(release['id'])}/assets?name={bundle.name}"
            with github_release_request(upload_url, auth_env, data=bundle.read_bytes(),
                                        content_type="application/zip") as response:
                asset = json.load(response)
        if (asset["state"] != "uploaded" or asset["size"] != bundle.stat().st_size
                or asset.get("digest") not in (None, f"sha256:{digest}")):
            raise RuntimeError("GitHub did not save the expected OT artifact archive")
        return {"asset_id": asset["id"], "sha256": digest, "download_url": asset["browser_download_url"]}


def download_weight_artifacts(artifacts, expected_caches, destination, auth_env):
    bundle = destination / "ot-weight-inputs.zip"
    url = f"{RELEASES_API}/assets/{int(artifacts['asset_id'])}"
    with github_release_request(url, auth_env, binary=True) as response, bundle.open("wb") as target:
        shutil.copyfileobj(response, target)
    if sha256(bundle) != artifacts["sha256"]:
        raise ValueError("The published OT artifact archive did not match its checksum")
    paths = weight_artifact_paths(expected_caches)
    with ZipFile(bundle) as archive:
        if sorted(archive.namelist()) != sorted(paths):
            raise ValueError("The published OT artifact archive contains unexpected files")
        for relative in paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(relative) as source, target.open("wb") as file:
                shutil.copyfileobj(source, file)
    return destination


def save_weight_audit(study, bundles, expected, output, source_commit, auth_env,
                      *, strength_commit=None, artifacts=None):
    seeds = [audit_seed_weights(study, seed, bundles[seed]) for seed in sorted(expected)]
    if artifacts is None:
        artifacts = upload_weight_artifacts(study, expected, source_commit, auth_env)
    write_json(output / "weight_artifacts.json", artifacts)
    write_json(output / "weight_audit.json", {
        "strength_results_commit": strength_commit, "feature_caches": expected,
        "comparison": "Top quarter by original OT weight versus the full class, among OT-active fitting classes.",
        "confidence": "Correct-class score minus the highest score for another class.",
        "seeds": seeds,
    })
    for result in seeds:
        shift = result["confidence_percentile_shift"]
        if shift is not None:
            print(f"Seed {result['seed']}: top-weight confidence rank changed by {100 * shift:+.2f} percentile points.",
                  flush=True)


def execute_weight_audit(source_commit, auth_env):
    run(["git", "fetch", "--no-tags", "origin", f"refs/heads/{STRENGTH_RESULTS_BRANCH}"],
        cwd=RESULTS_ROOT, env=auth_env)
    strength_commit = run(["git", "rev-parse", "FETCH_HEAD"], cwd=RESULTS_ROOT, capture=True)
    strength = json.loads(run(["git", "show", f"{strength_commit}:reports/strength_config.json"],
                              cwd=RESULTS_ROOT, capture=True))
    expected = {int(seed): digest for seed, digest in strength["feature_caches"].items()}
    if not expected:
        raise RuntimeError("The stronger OT results do not list any feature caches")
    manifest = json.loads(run(["git", "show", f"{strength_commit}:manifest.json"],
                              cwd=RESULTS_ROOT, capture=True))
    artifacts = None
    if "weight_artifacts.json" in manifest["report_hashes"]:
        artifacts = json.loads(run(["git", "show", f"{strength_commit}:reports/weight_artifacts.json"],
                                   cwd=RESULTS_ROOT, capture=True))
    else:
        previous_report = report_path("weight_audit.json")
        previous_artifacts = report_path("weight_artifacts.json")
        if previous_report.is_file() and previous_artifacts.is_file():
            previous = json.loads(previous_report.read_text())
            if previous["feature_caches"] == strength["feature_caches"]:
                artifacts = json.loads(previous_artifacts.read_text())
    output = WORK_ROOT / "weight-audit"
    with tempfile.TemporaryDirectory(prefix="ot-audit-inputs-") as temporary:
        if artifacts is not None:
            study = download_weight_artifacts(artifacts, expected, Path(temporary), auth_env)
            bundles = read_weight_audit_inputs(study, expected)
            if bundles is None:
                raise ValueError("The published baselines and feature caches do not match the stronger OT run")
        else:
            study, bundles = find_weight_audit_inputs(expected)
        save_weight_audit(study, bundles, expected, output, source_commit, auth_env,
                          strength_commit=strength_commit, artifacts=artifacts)
    publish_reports(output, source_commit, "complete OT example weight audit", auth_env)


def execute_rebuilt_weight_audit(source_commit, auth_env):
    from pipeline import run_ablation
    from pipeline.runtime import resolve_device

    has_reports = all(report_path(name).is_file() for name in ("weight_audit.json", "weight_artifacts.json"))
    if has_reports and not run(["git", "status", "--porcelain", "--", "manifest.json", "reports"],
                               cwd=RESULTS_ROOT, capture=True):
        run(["git", "push", "origin", f"HEAD:refs/heads/{RESULTS_BRANCH}"], cwd=RESULTS_ROOT, env=auth_env)
        print("The rebuilt audit and its inputs are already on GitHub.", flush=True)
        return

    config, root, _ = prepare_rebuilt_study(source_commit, auth_env)
    archive_interrupted_runs(root)
    device = resolve_device("cuda")
    fitting, development = root / "split/fitting.npz", root / "split/development.npz"
    expected, bundles = {}, {}
    for seed in config["seeds"]:
        checkpoint = run_ablation._baseline(config, fitting, seed, root / f"baselines/seed{seed}", device)
        bundle = run_ablation._seed_inputs(
            config, root / f"caches/seed{seed}", fitting, development, checkpoint, seed, device,
        )
        expected[seed] = bundle["cache_sha256"]
        bundles[seed] = bundle
        print(f"Baseline and feature cache ready for seed {seed}.", flush=True)

    output = WORK_ROOT / "weight-audit"
    reports = [
        "study_config.json", "implementation.json", "split/split.json", "pilot_results.json",
        "selected_schedule.json", "rebuild.json",
    ]
    for seed in config["seeds"]:
        reports.extend(f"baselines/seed{seed}/{name}" for name in ("config.json", "history.json", "complete.json"))
    for relative in reports:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    save_weight_audit(root, bundles, expected, output, source_commit, auth_env)
    publish_reports(output, source_commit, "complete rebuilt OT weight audit", auth_env)


ASSIGNMENT_RUN_FILES = {
    "config.json", "weights.pt", "weight_diagnostics.json", "starting_metrics.json",
    "history.json", "metrics.json", "final.pt", "complete.json",
    "prototype_weight_diagnostics.json", "assignment_finetuning_diagnostics.json",
}


def is_assignment_asset_path(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) == 3 and parts[0] == "diagnostics" and parts[1] in {"seed0", "seed1", "seed2"}:
        return parts[2] in {"scores.pt", "assignment_audit.json"}
    conditions = {"class_balanced", "ot", "strong_ot", "easy_strong", "boundary_strong", "hard_strong"}
    conditions.update(f"shuffled_strong_ot_shuffle{s}" for s in (0, 1, 2))
    return (len(parts) == 4 and parts[0] == "runs" and parts[1] in {"seed0", "seed1", "seed2"}
            and parts[2] in conditions and parts[3] in ASSIGNMENT_RUN_FILES)


def assignment_asset_paths(root, seed=None):
    return cached_asset_paths(root, seed, is_assignment_asset_path)


def is_simple_controls_asset_path(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) == 3 and parts[0] == "diagnostics" and parts[1] in {"seed0", "seed1", "seed2"}:
        return parts[2] in {"geometry.pt", "controls.json"}
    conditions = {"cluster_balanced", "prototype_distance", "easy_rank",
                  "cluster_rank_strong", "distance_rank_strong"}
    return (len(parts) == 4 and parts[0] == "runs" and parts[1] in {"seed0", "seed1", "seed2"}
            and parts[2] in conditions and parts[3] in ASSIGNMENT_RUN_FILES)


def cached_asset_paths(root, seed, accept_path):
    paths = []
    for current in ([seed] if seed is not None else (0, 1, 2)):
        folders = [root / f"diagnostics/seed{current}"]
        folders.extend(p for p in sorted((root / f"runs/seed{current}").glob("*"))
                       if (p / "complete.json").is_file())
        for folder in folders:
            for path in sorted(folder.glob("*")):
                relative = path.relative_to(root).as_posix()
                if path.is_file() and accept_path(relative):
                    paths.append(relative)
    return paths


def restore_assignment_study(auth_env):
    """Restore published JSON and its checksum-linked completed continuation files."""
    local = WORK_ROOT / "study"
    return restore_cached_study(RESULTS_ROOT, local, auth_env, is_assignment_asset_path)


def restore_cached_study(published_root, local, auth_env, accept_path):
    """Restore a report tree and only its allowlisted, receipted binary artifacts."""
    if local.exists():
        return local
    published = published_root / "reports/study_config.json"
    if not published.is_file():
        return local
    manifest = json.loads((published_root / "manifest.json").read_text())
    for relative, digest in manifest["report_hashes"].items():
        parts = PurePosixPath(relative).parts
        if (not parts or ".." in parts or PurePosixPath(relative).is_absolute()
                or sha256(published_root / "reports" / relative) != digest):
            raise ValueError(f"Published cached-study report changed: {relative}")
    asset_report = published_root / "reports/study_assets.json"
    if not asset_report.is_file():
        raise ValueError("Published assignment study has no reusable continuation assets")
    assets = json.loads(asset_report.read_text())
    selected = [assets["complete"]] if "complete" in assets else list(assets["seeds"].values())
    local.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cached-study-restore-", dir=local.parent) as temporary:
        staged = Path(temporary) / "study"
        shutil.copytree(published_root / "reports", staged)
        for index, asset in enumerate(selected):
            archive_path = Path(temporary) / f"asset{index}.zip"
            url = f"{RELEASES_API}/assets/{int(asset['asset_id'])}"
            with github_release_request(url, auth_env, binary=True) as response, archive_path.open("wb") as target:
                shutil.copyfileobj(response, target)
            if sha256(archive_path) != asset["sha256"]:
                raise ValueError("Assignment continuation archive checksum mismatch")
            with ZipFile(archive_path) as archive:
                names = archive.namelist()
                if (len(names) != len(set(names)) or sorted(names) != sorted(asset["paths"])
                        or any(not accept_path(name) for name in names)):
                    raise ValueError("Unexpected assignment continuation archive entries")
                for name in names:
                    contents = archive.read(name)
                    target = staged / name
                    if target.suffix == ".json" and (not target.is_file() or target.read_bytes() != contents):
                        raise ValueError(f"Continuation archive disagrees with published JSON: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
        # Each completed published run must actually have its binary artifacts.
        for receipt in staged.glob("runs/*/*/complete.json"):
            saved = json.loads(receipt.read_text())
            for name, digest in saved["files"].items():
                if name not in ASSIGNMENT_RUN_FILES or not (receipt.parent / name).is_file():
                    raise ValueError("Incomplete published continuation archive")
                if sha256(receipt.parent / name) != digest:
                    raise ValueError("Changed published continuation artifact")
        staged.rename(local)
    return local


def prepare_assignment_inputs(config, auth_env):
    from pipeline.ablation_inputs import AUDIT_REPORT_HASHES, load_audit_inputs

    destination = WORK_ROOT / "inputs"
    if destination.exists():
        load_audit_inputs(destination, config)
        return destination
    if RESTORE_ROOT is not None:
        source = Path(RESTORE_ROOT)
        if (source / "inputs").is_dir():
            source = source / "inputs"
        load_audit_inputs(source, config)
        return source
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    commit = config["input_results_commit"]
    run(["git", "fetch", "origin", commit], cwd=RESULTS_ROOT, env=auth_env)
    with tempfile.TemporaryDirectory(prefix="assignment-inputs-", dir=WORK_ROOT) as temporary:
        staged = Path(temporary) / "inputs"
        staged.mkdir()
        for relative, digest in AUDIT_REPORT_HASHES.items():
            response = subprocess.run(["git", "show", f"{commit}:reports/{relative}"],
                                      cwd=RESULTS_ROOT, env=auth_env, check=True, stdout=subprocess.PIPE)
            target = staged / "audit" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.stdout)
            if sha256(target) != digest:
                raise ValueError(f"Pinned input report checksum mismatch: {relative}")
        audit = json.loads((staged / "audit/weight_audit.json").read_text())
        artifacts = json.loads((staged / "audit/weight_artifacts.json").read_text())
        download_weight_artifacts(artifacts, {int(s): h for s, h in audit["feature_caches"].items()}, staged, auth_env)
        (staged / "ot-weight-inputs.zip").unlink()
        verified = load_audit_inputs(staged, config)
        write_json(staged / "inputs.json", verified["provenance"])
        staged.rename(destination)
    return destination


def execute_assignment(source_commit, auth_env):
    from pipeline.run_weight_study import resolve_config, run_study
    from pipeline.runtime import resolve_device

    config = resolve_config(json.loads((CODE_ROOT / "configs/weight_assignment.json").read_text())
                            | ASSIGNMENT_CONFIG_OVERRIDES)
    inputs = prepare_assignment_inputs(config, auth_env)
    if ASSIGNMENT_PREFLIGHT_ONLY:
        print("Assignment preflight passed: three verified baselines; 27 planned continuations.", flush=True)
        return
    root = restore_assignment_study(auth_env)
    archive_interrupted_runs(root)

    def publish(study, seed, message):
        publish_cached_study(study, seed, message, source_commit, auth_env,
                             "ot-assignment", is_assignment_asset_path)

    run_study(config, inputs, root, resolve_device("cuda"), on_progress=publish)


def publish_cached_study(study, seed, message, source_commit, auth_env, asset_prefix, accept_path):
    """Publish one completed-run snapshot; retrying reuses its content-addressed asset."""
    path = study / "study_assets.json"
    assets = json.loads(path.read_text()) if path.exists() else {"seeds": {}}
    paths = cached_asset_paths(study, seed, accept_path)
    suffix = f"seed{seed}" if seed is not None else "complete"
    asset = upload_artifacts(study, paths, source_commit, auth_env, f"{asset_prefix}-{suffix}") | {"paths": paths}
    if seed is None:
        assets["complete"] = asset
    else:
        assets["seeds"][str(seed)] = asset
    write_json(path, assets)
    publish_reports(study, source_commit, message, auth_env)


def prepare_simple_controls_parent(auth_env):
    """Fetch exactly the pinned parent reports and their complete release archive."""
    from pipeline.ablation_references import PARENT_COMMIT, validate_parent_manifest

    destination = WORK_ROOT / "parent"
    if destination.exists():
        validate_parent_manifest(destination)
        return destination
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    run(["git", "fetch", "origin", PARENT_COMMIT], cwd=RESULTS_ROOT, env=auth_env)
    with tempfile.TemporaryDirectory(prefix="simple-controls-parent-", dir=WORK_ROOT) as temporary:
        published = Path(temporary) / "published"
        published.mkdir()

        def read_pinned(relative):
            response = subprocess.run(["git", "show", f"{PARENT_COMMIT}:{relative}"],
                                      cwd=RESULTS_ROOT, env=auth_env, check=True, stdout=subprocess.PIPE)
            target = published / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.stdout)

        read_pinned("manifest.json")
        manifest = json.loads((published / "manifest.json").read_text())
        for relative in manifest["report_hashes"]:
            parts = PurePosixPath(relative).parts
            if not parts or ".." in parts or PurePosixPath(relative).is_absolute():
                raise ValueError("Invalid pinned parent report path")
            read_pinned(f"reports/{relative}")
        validate_parent_manifest(published)
        restored = Path(temporary) / "parent"
        restored.mkdir()
        restore_cached_study(published, restored / "reports", auth_env, is_assignment_asset_path)
        shutil.copyfile(published / "manifest.json", restored / "manifest.json")
        validate_parent_manifest(restored)
        restored.rename(destination)
    return destination


def execute_simple_controls(source_commit, auth_env):
    import torch
    from pipeline.run_simple_controls import prepare_study, resolve_config
    from pipeline.run_weight_study import run_study
    from pipeline.runtime import resolve_device

    config = resolve_config(json.loads((CODE_ROOT / "configs/weight_simple_controls.json").read_text()))
    # Match the recorded parent backend settings before checking compatibility.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    inputs = prepare_assignment_inputs(config, auth_env)
    parent = prepare_simple_controls_parent(auth_env)
    if SIMPLE_CONTROLS_PREFLIGHT_ONLY:
        prepare_study(config, inputs, parent)
        print("Stage B preflight passed: parent references, three geometry replays and 15 weight vectors; "
              "training environment is checked when training starts.", flush=True)
        return
    root = restore_cached_study(RESULTS_ROOT, WORK_ROOT / "study", auth_env, is_simple_controls_asset_path)
    archive_interrupted_runs(root)

    def publish(study, seed, message):
        publish_cached_study(study, seed, message, source_commit, auth_env,
                             "ot-simple-controls", is_simple_controls_asset_path)

    run_study(config, inputs, root, resolve_device("cuda"), on_progress=publish, parent_root=parent)


def main():
    allowed = ("prepare", "pilot", "study", "freeze", "final")
    is_assignment = STAGES == ("assignment",)
    is_simple_controls = STAGES == ("simple_controls",)
    if not (is_simple_controls or is_assignment or IS_WEIGHT_AUDIT or IS_AUDIT_REBUILD or IS_STRENGTH_STUDY):
        if not STAGES or len(set(STAGES)) != len(STAGES) or any(stage not in allowed for stage in STAGES):
            raise ValueError('Use ("simple_controls",), ("assignment",), ("rebuild_audit",), ("audit",), ("strength",), or original stages')
        if list(STAGES) != sorted(STAGES, key=allowed.index):
            raise ValueError("STAGES must follow prepare, pilot, study, freeze, final order")
    print(f"Kaggle stages: {', '.join(STAGES)}. Results branch: {RESULTS_BRANCH}.", flush=True)
    rebuild = IS_STRENGTH_STUDY and not restore_workspace()
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
        if IS_WEIGHT_AUDIT:
            execute_weight_audit(source_commit, auth_env)
            print(f"Report pushed to {RESULTS_BRANCH}.", flush=True)
            return
        sys.path.insert(0, str(CODE_ROOT))
        import pipeline

        if Path(pipeline.__file__).resolve().parent != CODE_ROOT / "pipeline":
            raise RuntimeError("Restart the notebook kernel: another pipeline checkout is already imported")
        import torch
        import torchvision

        print(f"torch={torch.__version__}, torchvision={torchvision.__version__}", flush=True)
        preflight_only = ((is_assignment and ASSIGNMENT_PREFLIGHT_ONLY)
                          or (is_simple_controls and SIMPLE_CONTROLS_PREFLIGHT_ONLY))
        if not torch.cuda.is_available() and not preflight_only:
            raise RuntimeError("Enable a GPU accelerator in Kaggle")
        if is_simple_controls:
            execute_simple_controls(source_commit, auth_env)
        elif is_assignment:
            execute_assignment(source_commit, auth_env)
        elif IS_AUDIT_REBUILD:
            execute_rebuilt_weight_audit(source_commit, auth_env)
        elif IS_STRENGTH_STUDY:
            execute_strength(source_commit, auth_env, rebuild=rebuild)
        else:
            restore_workspace()
            prepared, raw_root = prepare_data()
            config = json.loads((CODE_ROOT / "configs/ablation_study.json").read_text()) | CONFIG_OVERRIDES
            config["train_data"] = str(prepared)
            execute_stages(config, source_commit, raw_root, auth_env)
        if preflight_only:
            print("Preflight complete; no training or report publication was performed.", flush=True)
        else:
            print(f"Reports pushed to {RESULTS_BRANCH}. Preserve {WORK_ROOT} in Kaggle outputs.", flush=True)


if __name__ == "__main__":
    main()
