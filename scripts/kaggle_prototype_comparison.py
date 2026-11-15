"""Paste into Kaggle to compare OT reweighting and feature synthesis."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY = "https://github.com/Dzzz83/MoE_Imbalance.git"
SOURCE_COMMIT = "01840f81d159ecdfd09659fcb5b4984e0fc141fc"
RESULTS_BRANCH = "ot-methods-results"
GIT_USER_NAME = "Dzzz83"
GIT_USER_EMAIL = "phattuan43@gmail.com"

CODE_ROOT = Path("/kaggle/working/ot-methods-code")
STATE_ROOT = Path("/kaggle/working/ot-methods-results")
DATA_ROOT = Path("/kaggle/working/ot-methods-data")
RAW_CIFAR_ROOT = Path("/kaggle/input/datasets/dzzz833/cifar100")
TRAINING_SEEDS = (0, 1, 2)
# Choose "reweighting", "synthesis", or "both". Both paths share each baseline.
EXPERIMENT_PATH = "both"
# Optional shortcut to run only SAVA within the reweighting path.
SAVA_ONLY = False
# Optional seed -> attached baseline checkpoint path. Each must match that seed
# and the pinned baseline recipe and retained pool.
BASELINE_CHECKPOINTS = {}
MODES = ("uniform", "class_balanced", "cluster_balanced", "sinkhorn", "sava")


def run(command, cwd=None, env=None, capture=False):
    print("+", " ".join(map(str, command)), flush=True)
    result = subprocess.run(
        list(map(str, command)), cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE if capture else None, check=True,
    )
    return result.stdout.strip() if capture else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def push_state(message, auth_env):
    for path in STATE_ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(STATE_ROOT).parts:
            if path.stat().st_size > 95 * 1024 * 1024:
                raise RuntimeError(f"Result exceeds the Git storage limit: {path}")
    run(["git", "add", "-A"], cwd=STATE_ROOT)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=STATE_ROOT)
    if changed.returncode not in (0, 1):
        raise RuntimeError("Could not inspect staged experiment results")
    if changed.returncode == 1:
        run(["git", "commit", "-m", message], cwd=STATE_ROOT)
    # Always push, including when retrying a previous failed push.
    run(["git", "push", "origin", f"HEAD:refs/heads/{RESULTS_BRANCH}"], cwd=STATE_ROOT, env=auth_env)


def prepare_repositories(auth_env):
    if not CODE_ROOT.exists():
        run(["git", "clone", REPOSITORY, CODE_ROOT], env=auth_env)
    if run(["git", "remote", "get-url", "origin"], cwd=CODE_ROOT, capture=True) != REPOSITORY:
        raise RuntimeError("Unexpected source repository")
    if run(["git", "status", "--porcelain"], cwd=CODE_ROOT, capture=True):
        raise RuntimeError("Source checkout has local edits")
    run(["git", "fetch", "origin", SOURCE_COMMIT], cwd=CODE_ROOT, env=auth_env)
    run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=CODE_ROOT)
    if run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, capture=True) != SOURCE_COMMIT:
        raise RuntimeError("Source commit mismatch")

    if not STATE_ROOT.exists():
        remote = run(
            ["git", "ls-remote", "--heads", REPOSITORY, f"refs/heads/{RESULTS_BRANCH}"],
            env=auth_env, capture=True,
        )
        if remote:
            run(["git", "clone", "--single-branch", "--branch", RESULTS_BRANCH,
                 REPOSITORY, STATE_ROOT], env=auth_env)
        else:
            STATE_ROOT.mkdir(parents=True)
            run(["git", "init", "--initial-branch", RESULTS_BRANCH], cwd=STATE_ROOT)
            run(["git", "remote", "add", "origin", REPOSITORY], cwd=STATE_ROOT)
    if run(["git", "remote", "get-url", "origin"], cwd=STATE_ROOT, capture=True) != REPOSITORY:
        raise RuntimeError("Unexpected results repository")
    if run(["git", "branch", "--show-current"], cwd=STATE_ROOT, capture=True) != RESULTS_BRANCH:
        raise RuntimeError("Unexpected results branch")
    run(["git", "config", "user.name", GIT_USER_NAME], cwd=STATE_ROOT)
    run(["git", "config", "user.email", GIT_USER_EMAIL], cwd=STATE_ROOT)


def prepare_data(training_env):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    raw_root = DATA_ROOT / "raw"
    raw_root.mkdir(exist_ok=True)
    # Accept either the CIFAR directory itself or its parent in the attached dataset.
    attached = RAW_CIFAR_ROOT
    if (attached / "cifar-100-python").is_dir():
        attached = attached / "cifar-100-python"
    raw_link = raw_root / "cifar-100-python"
    if attached.is_dir() and (attached / "train").is_file():
        if not raw_link.exists():
            raw_link.symlink_to(attached, target_is_directory=True)
    prepared = DATA_ROOT / "train.npz"
    if not prepared.exists():
        run([sys.executable, "-m", "pipeline.data", "--data-root", raw_root,
             "--output", prepared, "--imbalance-factor", "0.01", "--seed", "42", "--download"],
            cwd=CODE_ROOT, env=training_env)
    return prepared, raw_root


def train_run(name, config, auth_env, training_env):
    output_dir = STATE_ROOT / "runs" / name
    saved_config = save_stage_config(name, config)
    if not (output_dir / "final.pt").exists():
        # Keep unfinished attempts outside the result repository. A fresh attempt
        # starts from the source checkpoint because the pipeline has no epoch resume.
        attempt = Path(tempfile.mkdtemp(prefix=f"{name}-", dir="/kaggle/working")) / "run"
        run([sys.executable, "-m", "pipeline.train", "--config", saved_config,
             "--output-dir", attempt, "--device", "cuda"], cwd=CODE_ROOT, env=training_env)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        attempt.rename(output_dir)
        attempt.parent.rmdir()
    push_state(f"save {name} final checkpoint", auth_env)
    return output_dir


def prepare_baseline(seed, prepared, auth_env, training_env):
    baseline_checkpoint = BASELINE_CHECKPOINTS.get(seed)
    baseline_config = json.loads((CODE_ROOT / "configs/baseline.json").read_text())
    baseline_config.update(train_data=str(prepared), num_workers=2, seed=seed)
    if baseline_checkpoint is not None:
        # Match the complete resolved baseline recipe and pool metadata before reuse.
        sys.path.insert(0, str(CODE_ROOT))
        import torch
        from pipeline import train
        from pipeline.data import load_training_data
        from pipeline.methods.finetuning import _load_pretrained_baseline
        from pipeline.model import create_model

        checkpoint = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["config"] != train.resolve_config(baseline_config):
            # Training paths and worker counts may differ across machines.
            original = checkpoint["config"] | {"train_data": str(prepared), "num_workers": 2}
            if original != train.resolve_config(baseline_config):
                raise RuntimeError("Attached baseline recipe does not match the experiment")
        _load_pretrained_baseline(create_model(), load_training_data(prepared), baseline_checkpoint)
        destination = STATE_ROOT / "runs" / f"baseline_seed{seed}"
        if not (destination / "final.pt").exists():
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir="/kaggle/working") as temporary_dir:
                pending_checkpoint = Path(temporary_dir) / "final.pt"
                shutil.copyfile(baseline_checkpoint, pending_checkpoint)
                pending_checkpoint.replace(destination / "final.pt")
        push_state("save attached baseline", auth_env)
    else:
        destination = train_run(f"baseline_seed{seed}", baseline_config, auth_env, training_env)
    return destination


def run_comparison(seed, prepared, raw_root, auth_env, training_env):
    modes = ("sava",) if SAVA_ONLY else MODES
    suffix = f"_sava_seed{seed}" if SAVA_ONLY else ("" if seed == 0 else f"_seed{seed}")
    if SAVA_ONLY:
        if not (CODE_ROOT / "configs/sava_finetuning.json").is_file():
            raise RuntimeError("Pin SOURCE_COMMIT to a published SAVA implementation commit")
        if not (STATE_ROOT / "runs" / f"baseline_seed{seed}" / "final.pt").is_file():
            raise RuntimeError(f"SAVA requires the completed baseline for seed {seed}")
    baseline_checkpoint = BASELINE_CHECKPOINTS.get(seed)
    manifest = {
        "source_commit": SOURCE_COMMIT, "seed": seed, "data_seed": 42,
        "imbalance_factor": 0.01, "modes": list(modes),
        "baseline_input": str(baseline_checkpoint) if baseline_checkpoint else None,
        "runner_version": 1,
    }
    manifest_path = STATE_ROOT / f"experiment{suffix}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Experiment settings changed; choose a new RESULTS_BRANCH and STATE_ROOT")
    write_json(manifest_path, manifest)
    push_state("initialize prototype comparison", auth_env)
    destination = prepare_baseline(seed, prepared, auth_env, training_env)
    runs = {"baseline": destination}
    for mode in modes:
        config = json.loads((CODE_ROOT / "configs" / f"{mode}_finetuning.json").read_text())
        config.update(train_data=str(prepared), num_workers=2, seed=seed)
        config["method_options"]["pretrained_checkpoint"] = str(destination / "final.pt")
        runs[mode] = train_run(f"{mode}_seed{seed}", config, auth_env, training_env)
    summary = {}
    for mode, output_dir in runs.items():
        summary[mode] = evaluate_run(mode, output_dir, raw_root, auth_env, training_env)
    write_json(STATE_ROOT / f"comparison{suffix}.json", summary)
    push_state(f"save seed {seed} prototype comparison summary", auth_env)
    print(json.dumps(summary, indent=2))
    return summary


def evaluate_run(name, output_dir, raw_root, auth_env, training_env):
    metrics = output_dir / "metrics.json"
    if not metrics.exists():
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as temporary_dir:
            pending_metrics = Path(temporary_dir) / "metrics.json"
            run([sys.executable, "-m", "pipeline.evaluate",
                 "--checkpoint", output_dir / "final.pt",
                 "--data-root", raw_root, "--output", pending_metrics, "--device", "cuda",
                 "--num-workers", "2", "--download"], cwd=CODE_ROOT, env=training_env)
            json.loads(pending_metrics.read_text())
            pending_metrics.replace(metrics)
    push_state(f"evaluate {name}", auth_env)
    return json.loads(metrics.read_text())


def save_stage_config(name, config):
    path = STATE_ROOT / "configs" / f"{name}.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"Saved configuration differs for {name}")
    write_json(path, config)
    return path


def run_synthesis(seed, prepared, raw_root, auth_env, training_env):
    save_stage_config(f"experiment_synthesis_seed{seed}", {
        "source_commit": SOURCE_COMMIT, "seed": seed, "data_seed": 42,
        "imbalance_factor": 0.01,
        "baseline_input": str(BASELINE_CHECKPOINTS[seed]) if seed in BASELINE_CHECKPOINTS else None,
    })
    baseline = prepare_baseline(seed, prepared, auth_env, training_env)
    checkpoint = baseline / "final.pt"
    cache = baseline / "train_features.pt"
    if not cache.exists():
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as temporary_dir:
            pending_cache = Path(temporary_dir) / "train_features.pt"
            run([sys.executable, "-m", "pipeline.extract_features",
                 "--checkpoint", checkpoint, "--train-data", prepared,
                 "--output", pending_cache, "--device", "cuda", "--num-workers", "2"],
                cwd=CODE_ROOT, env=training_env)
            pending_cache.replace(cache)
        push_state(f"save seed {seed} training features", auth_env)
    generation_dir = STATE_ROOT / f"synthesis_seed{seed}"
    generation = json.loads((CODE_ROOT / "configs/synthesis_final.json").read_text())
    generation.update(feature_cache=str(cache), output_dir=str(generation_dir), seed=seed)
    generation_config = save_stage_config(f"synthesis_seed{seed}", generation)
    # Generation resumes per class and verifies the cache, seed, and recipe.
    run([sys.executable, "-m", "pipeline.methods.synthesis", "--config", generation_config,
         "--device", "cuda", "--resume"], cwd=CODE_ROOT, env=training_env)
    push_state(f"save seed {seed} synthetic features", auth_env)
    output_dir = STATE_ROOT / "runs" / f"synthetic_seed{seed}"
    classifier = json.loads((CODE_ROOT / "configs/classifier_synthesis.json").read_text())
    classifier.update(pretrained_checkpoint=str(checkpoint), feature_cache=str(cache),
                      synthetic_features=str(generation_dir / "synthetic_features.pt"),
                      output_dir=str(output_dir), seed=seed, num_workers=0,
                      target_count=generation["method_options"]["target_count"])
    classifier_config = save_stage_config(f"synthetic_seed{seed}", classifier)
    # The synthesis classifier resumes at epoch boundaries, including RNG state.
    run([sys.executable, "-m", "pipeline.train_classifier", "--config", classifier_config,
         "--device", "cuda", "--resume"], cwd=CODE_ROOT, env=training_env)
    push_state(f"save seed {seed} synthetic classifier", auth_env)
    summary = {
        name: evaluate_run(name, path, raw_root, auth_env, training_env)
        for name, path in (("baseline", baseline), ("synthetic", output_dir))
    }
    write_json(STATE_ROOT / f"comparison_synthesis_seed{seed}.json", summary)
    push_state(f"save seed {seed} synthesis comparison", auth_env)
    return summary


def main():
    if EXPERIMENT_PATH not in {"reweighting", "synthesis", "both"}:
        raise ValueError("EXPERIMENT_PATH must be reweighting, synthesis, or both")
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Enable the Kaggle secret GITHUB_TOKEN")
    training_env = os.environ.copy()
    training_env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    training_env.pop("GITHUB_TOKEN", None)
    training_env.pop("GIT_ASKPASS", None)
    # The helper contains no token. Only Git subprocesses receive the secret.
    with tempfile.TemporaryDirectory(prefix="prototype-git-") as helper_dir:
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
        prepare_repositories(auth_env)
        # Keep Kaggle's installed torch/torchvision pair. Fail early if incompatible.
        run([sys.executable, "-c",
             "import torch, torchvision; print(torch.__version__, torchvision.__version__); "
             "assert torch.cuda.is_available(), 'Enable a GPU accelerator in Kaggle'"],
            cwd=CODE_ROOT, env=training_env)
        prepared, raw_root = prepare_data(training_env)
        for seed in TRAINING_SEEDS:
            summary = {}
            if EXPERIMENT_PATH in {"reweighting", "both"}:
                summary.update(run_comparison(seed, prepared, raw_root, auth_env, training_env))
            if EXPERIMENT_PATH in {"synthesis", "both"}:
                summary.update(run_synthesis(seed, prepared, raw_root, auth_env, training_env))
            write_json(STATE_ROOT / f"comparison_ot_seed{seed}.json", summary)
            push_state(f"save seed {seed} OT methods comparison", auth_env)


if __name__ == "__main__":
    main()
