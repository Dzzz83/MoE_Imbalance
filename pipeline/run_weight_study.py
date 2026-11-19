"""Matched assignment ablation on published, frozen fitting/development features."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from pipeline.ablation_inputs import ARCHIVE_SHA256, AUDIT_COMMIT, load_audit_inputs
from pipeline.ablation_weights import (
    assignment_audit, baseline_scores, reassign_weights, transform_weights, weight_diagnostics,
)
from pipeline.methods import reweighting
from pipeline.model import create_model
from pipeline.run_ablation import _completed_run, _finish_run, evaluate_features, summarize_results
from pipeline.runtime import (
    atomic_torch_save, atomic_write_json, file_sha256, resolve_device,
    save_checkpoint_with_metadata, seed_everything,
)

PRIMARY_REFERENCES = ("strong_ot", "class_balanced", "shuffled_strong_ot")


def resolve_config(config):
    if isinstance(config, dict) and config.get("stage") == "simple_controls":
        from pipeline.run_simple_controls import resolve_config as resolve_controls

        return resolve_controls(config)
    defaults = {
        "stage": "assignment", "input_results_commit": AUDIT_COMMIT, "archive_sha256": ARCHIVE_SHA256,
        "seeds": [0, 1, 2], "shuffle_seeds": [0, 1, 2], "tie_seed": 0,
        "batch_size": 128, "weight_exponent": 3, "classifier": {"learning_rate": 0.002},
        "practical_gain": 0.005,
    }
    if not isinstance(config, dict) or config.keys() - defaults.keys():
        raise ValueError("Unknown assignment-study configuration settings")
    resolved = defaults | config
    if (resolved["stage"] != "assignment" or type(resolved["weight_exponent"]) is not int
            or resolved["weight_exponent"] != 3):
        raise ValueError("This runner implements the exponent-3 assignment stage only")
    if resolved["input_results_commit"] != AUDIT_COMMIT or resolved["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Use the pinned rebuilt audit commit and archive checksum")
    for name in ("seeds", "shuffle_seeds"):
        if resolved[name] != [0, 1, 2] or any(type(s) is not int for s in resolved[name]):
            raise ValueError(f"{name} must be [0, 1, 2] for the matched study")
    for name in ("tie_seed", "batch_size"):
        value = resolved[name]
        if type(value) is not int or not (0 if name == "tie_seed" else 1) <= value < 2**32:
            raise ValueError(f"Invalid {name}")
    gain = resolved["practical_gain"]
    if type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0:
        raise ValueError("practical_gain must be finite and positive")
    if not isinstance(resolved["classifier"], dict):
        raise ValueError("classifier must be an object")
    resolved["classifier"] = reweighting.prepare_classifier_options(defaults["classifier"] | resolved["classifier"])
    return resolved


def conditions(config):
    return [(name, None) for name in ("class_balanced", "ot", "strong_ot")] + [
        ("shuffled_strong_ot", seed) for seed in config["shuffle_seeds"]
    ] + [(name, None) for name in ("easy_strong", "boundary_strong", "hard_strong")]


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def environment(device):
    return {
        "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cpu_threads": torch.get_num_threads(),
    }


def summarize_assignment(records, config):
    expected = {(seed, condition, shuffle) for seed in config["seeds"] for condition, shuffle in conditions(config)}
    identities = [(r["seed"], r["condition"], r["shuffle_seed"]) for r in records]
    context = {(seed, "baseline", None) for seed in config["seeds"]}
    actual = [(r["seed"], r["condition"], r["shuffle_seed"]) for r in records if r["condition"] != "baseline"]
    if len(identities) != len(set(identities)) or not set(identities) <= expected | context:
        raise ValueError("Unexpected or duplicate assignment results")
    if set(identities) != expected | context:
        return {"status": "incomplete", "completed_runs": len(actual), "expected_runs": len(expected),
                "decision": {"status": "incomplete"}}
    summary = summarize_results(records)
    contrasts = {}
    for reference in PRIMARY_REFERENCES:
        result = summarize_results(records, reference)["boundary_strong"]["balanced_accuracy"]
        contrasts[f"boundary_strong_minus_{reference}"] = result
    secondary = {}
    for reference in PRIMARY_REFERENCES:
        paired = summarize_results(records, reference)
        for condition in ("ot", "easy_strong", "hard_strong"):
            secondary[f"{condition}_minus_{reference}"] = paired[condition]["balanced_accuracy"]
    complete = all(r["paired_difference_mean"] is not None for r in contrasts.values())
    target_met = complete and all(
        r["paired_difference_mean"] >= config["practical_gain"]
        and sum(p["paired_difference"] > 0 for p in r["per_seed"]) >= 2 for r in contrasts.values()
    )
    return {
        "status": "complete", "completed_runs": len(actual), "expected_runs": len(expected),
        "conditions": summary, "primary_contrasts": contrasts, "secondary_contrasts": secondary,
        "decision": {
            "status": ("target_met" if target_met else "target_not_met") if complete else "incomplete",
            "practical_gain": config["practical_gain"],
            "interpretation": "Research budget criterion only; not significance, equivalence, or proof of mechanism.",
            "secondary_conditions": ["ot", "easy_strong", "hard_strong"],
        },
    }


def _continuation(config, inputs, seed, condition, shuffle, weights, scores, root, device):
    bundle, checkpoint = inputs["bundles"][seed], inputs["checkpoints"][seed]
    split = inputs["split"]
    identity = {
        "seed": seed, "condition": condition, "shuffle_seed": shuffle, "config": config,
        "files": inputs["provenance"]["seed_files"][str(seed)],
        "weights_sha256": _tensor_hash(weights), "margins_sha256": _tensor_hash(scores["margins"]),
        "original_class_counts": split["original_class_counts"],
    }
    if _completed_run(root, identity):
        return json.loads((root / "metrics.json").read_text())
    root.mkdir(parents=True)
    seed_everything(seed)
    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    training_config = {
        "method": "assignment_finetuning", "seed": seed, "batch_size": config["batch_size"],
        "num_workers": 0, "input_results_commit": config["input_results_commit"],
        "baseline_sha256": identity["files"]["baseline"],
        "method_options": {"condition": condition, "shuffle_seed": shuffle,
                           "classifier": config["classifier"]},
    }
    atomic_write_json(root / "config.json", training_config)
    atomic_torch_save({"weights": weights, "labels": bundle["labels"], "inputs": identity}, root / "weights.pt")
    diagnostics = weight_diagnostics(weights, bundle["labels"], split["original_class_counts"],
                                     bundle["prototype_diagnostics"])
    diagnostics["assignment"] = assignment_audit(
        weights, bundle["labels"], scores["margins"], scores["correct"], bundle["prototype_diagnostics"],
    )
    atomic_write_json(root / "weight_diagnostics.json", diagnostics)
    active = [d["class_index"] for d in bundle["prototype_diagnostics"] if d["used_ot"]]

    def evaluate(current):
        return evaluate_features(current, bundle["development_features"], bundle["development_labels"],
                                 split["original_class_counts"], device, config["batch_size"], active)

    atomic_write_json(root / "starting_metrics.json", evaluate(model))
    history = []

    def record_epoch(current, record):
        if not math.isfinite(record["loss"]) or any(not torch.isfinite(p).all() for p in current.fc.parameters()):
            raise RuntimeError("Nonfinite classifier state")
        record["development"] = evaluate(current)
        history.append(record.copy())
        atomic_write_json(root / "history.json", history)

    reweighting.train_classifier_from_features(
        model, bundle["features"], bundle["labels"], training_config, root, device,
        epoch_offset=checkpoint["config"]["method_options"]["epochs"], use_ot=False,
        weight_builder=lambda *args: (weights.to(device), bundle["prototype_diagnostics"]),
        epoch_callback=record_epoch,
    )
    # A frozen backbone includes every buffer, not only trainable parameters.
    for name, value in model.state_dict().items():
        if not name.startswith("fc.") and not torch.equal(value.cpu(), checkpoint["model_state_dict"][name]):
            raise RuntimeError(f"Frozen backbone changed: {name}")
    save_checkpoint_with_metadata(root / "final.pt", model, training_config,
                                  checkpoint["class_counts"], checkpoint["data_settings"])
    metrics = history[-1]["development"]
    atomic_write_json(root / "metrics.json", metrics)
    _finish_run(root, identity)
    return metrics


def run_study(config, input_root, output_root, device, on_progress=None, *, parent_root=None):
    if config.get("stage") == "simple_controls":
        from pipeline.run_simple_controls import run_study as run_controls

        if parent_root is None:
            raise ValueError("Stage B requires the pinned Stage A --parent directory")
        return run_controls(config, input_root, parent_root, output_root, device, on_progress)
    config = resolve_config(config)
    inputs = load_audit_inputs(input_root, config)
    root, device = Path(output_root), torch.device(device)
    source_root = Path(__file__).parent
    implementation = {name: file_sha256(source_root / name) for name in (
        "run_weight_study.py", "ablation_inputs.py", "ablation_weights.py", "run_ablation.py",
        "runtime.py", "model.py", "evaluate.py", "extract_features.py", "methods/reweighting.py",
    )}
    manifest = {"config": config, "inputs": inputs["provenance"], "implementation": implementation,
                "environment": environment(device),
                "primary_contrasts": [f"boundary_strong_minus_{r}" for r in PRIMARY_REFERENCES],
                "expected_runs": [{"seed": s, "condition": c, "shuffle_seed": t}
                                  for s in config["seeds"] for c, t in conditions(config)]}
    manifest_path = root / "study_config.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Study inputs, implementation, or environment changed; choose a new output folder")
    elif root.exists():
        raise FileExistsError("Incomplete study initialization; preserve it and choose a new output folder")
    else:
        root.mkdir(parents=True)
        atomic_write_json(manifest_path, manifest)
    atomic_write_json(root / "inputs.json", inputs["provenance"])
    atomic_write_json(root / "implementation.json", implementation)
    records = []

    def publish_state(status, error=None):
        atomic_write_json(root / "study_results.json", records)
        summary = summarize_assignment(records, config)
        if status == "failed":
            summary["status"], summary["decision"] = "failed", {"status": "incomplete"}
        atomic_write_json(root / "study_summary.json", summary)
        atomic_write_json(root / "study_status.json", {"status": status, "error": error,
                                                       "completed_runs": summary["completed_runs"]})

    try:
        publish_state("running")
        for seed in config["seeds"]:
            bundle, checkpoint = inputs["bundles"][seed], inputs["checkpoints"][seed]
            margins, correct = baseline_scores(bundle["features"], bundle["labels"], checkpoint["model_state_dict"])
            tie_order = torch.randperm(len(margins), generator=torch.Generator().manual_seed(config["tie_seed"]))
            scores = {"margins": margins, "correct": correct, "tie_order": tie_order}
            score_dir = root / f"diagnostics/seed{seed}"
            score_dir.mkdir(parents=True, exist_ok=True)
            score_path = score_dir / "scores.pt"
            if score_path.exists():
                saved = torch.load(score_path, map_location="cpu", weights_only=True)
                if saved.keys() != scores.keys() or any(not torch.equal(saved[k], v) for k, v in scores.items()):
                    raise ValueError("Saved assignment scores or tie order changed")
            else:
                atomic_torch_save(scores, score_path)
            atomic_write_json(score_dir / "assignment_audit.json", assignment_audit(
                bundle["ot_weights"], bundle["labels"], margins, correct, bundle["prototype_diagnostics"],
            ))
            baseline_model = create_model().to(device)
            baseline_model.load_state_dict(checkpoint["model_state_dict"])
            active = [d["class_index"] for d in bundle["prototype_diagnostics"] if d["used_ot"]]
            baseline_metrics = evaluate_features(baseline_model, bundle["development_features"],
                bundle["development_labels"], inputs["split"]["original_class_counts"], device,
                config["batch_size"], active)
            del baseline_model
            records.append({"seed": seed, "condition": "baseline", "shuffle_seed": None, "metrics": baseline_metrics})
            strong = transform_weights(bundle["ot_weights"], bundle["labels"], "strong_ot")
            for condition, shuffle in conditions(config):
                if condition in ("easy_strong", "boundary_strong", "hard_strong"):
                    weights = reassign_weights(strong, bundle["labels"], margins, condition, tie_order)
                elif condition == "strong_ot":
                    weights = strong.clone()
                elif condition == "shuffled_strong_ot":
                    weights = transform_weights(strong, bundle["labels"], "shuffled_ot", shuffle)
                else:
                    weights = transform_weights(bundle["ot_weights"], bundle["labels"], condition, shuffle)
                if not torch.isfinite(weights).all() or (weights <= 0).any():
                    raise ValueError("Invalid or underflowed assignment weights")
                name = condition if shuffle is None else f"{condition}_shuffle{shuffle}"
                run_root = root / f"runs/seed{seed}" / name
                metrics = _continuation(config, inputs, seed, condition, shuffle, weights, scores, run_root, device)
                records.append({"seed": seed, "condition": condition, "shuffle_seed": shuffle, "metrics": metrics,
                                "output_dir": run_root.relative_to(root).as_posix()})
                publish_state("running")
                if on_progress is not None:
                    on_progress(root, seed, f"save assignment seed {seed} {name}")
        publish_state("complete")
        if on_progress is not None:
            on_progress(root, None, "complete assignment study")
    except Exception as error:
        publish_state("failed", f"{type(error).__name__}: {error}")
        raise
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--config", default="configs/weight_assignment.json")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--parent", help="Stage A directory containing manifest.json and reports/")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = resolve_config(json.loads(Path(args.config).read_text()))
    if args.command == "validate":
        if config["stage"] == "simple_controls":
            from pipeline.run_simple_controls import prepare_study

            if args.parent is None:
                parser.error("Stage B requires --parent")
            inputs, _, _ = prepare_study(config, args.inputs, args.parent)
        else:
            inputs = load_audit_inputs(args.inputs, config)
        print(json.dumps(inputs["provenance"], indent=2))
    else:
        output = args.output_dir or f"outputs/weight_{config['stage']}"
        run_study(config, args.inputs, output, resolve_device(args.device), parent_root=args.parent)


if __name__ == "__main__":
    main()
