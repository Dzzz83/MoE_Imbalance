"""Bounded Stage C image-OT and feature-SMOTE studies on pinned audit inputs."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch
import torch.nn.functional as functional
import torchvision

from pipeline.ablation_inputs import ARCHIVE_SHA256, AUDIT_COMMIT, load_audit_inputs
from pipeline.controlled_bias_head import HeadConvergenceError, fit_head
from pipeline.feature_smote import bank_diagnostics, build_banks
from pipeline.followup_images import prepare_images
from pipeline.followup_reporting import evaluate_logits, readable_report, summarize
from pipeline.followup_weights import (
    FEATURE_CONDITIONS, IMAGE_CONDITIONS, build_feature_weights, build_image_weights, describe_weights,
)
from pipeline.image_reweighting import image_logits, train_images
from pipeline.model import create_model
from pipeline.run_ablation import _completed_run, _finish_run
from pipeline.run_weight_study import environment
from pipeline.runtime import (
    atomic_torch_save, atomic_write_json, content_identity, file_sha256, resolve_device,
    save_checkpoint_with_metadata, seed_everything,
)
from pipeline.simple_control_weights import replay_geometry

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
RUN_FILES = {"config.json", "weights.pt", "weight_diagnostics.json", "history.json", "optimization.json",
             "metrics.json", "predictions.pt", "final.pt"}
FEATURE_RUN_FILES = RUN_FILES | {"head.pt", "cast_predictions.pt", "cast_metrics.json"}
STAGES = ("synthetic_reweighting", "ot_contribution")
EXPECTED_ACTIVE_CLASSES = (75, 70, 73)
EXPECTED_FITTING_ROWS = 8716
EXPECTED_AUGMENTED_ROWS = 11442


def resolve_config(config):
    if not isinstance(config, dict) or config.get("stage") not in STAGES:
        raise ValueError("Choose synthetic_reweighting or ot_contribution")
    defaults = json.loads((CONFIG_ROOT / f"{config['stage']}.json").read_text()) | {
        "input_results_commit": AUDIT_COMMIT, "archive_sha256": ARCHIVE_SHA256,
    }
    resolved = defaults | config
    section = "head" if config["stage"] == "synthetic_reweighting" else "image"
    if isinstance(config.get(section), dict):
        resolved[section] = defaults[section] | config[section]
    if json.dumps(resolved, sort_keys=True) != json.dumps(defaults, sort_keys=True):
        raise ValueError("Stage C settings are frozen to PLAN.md; use a new protocol for changes")
    return resolved


def prepare_study(config, input_root):
    config = resolve_config(config)
    torch.set_num_threads(config["cpu_threads"])
    inputs = load_audit_inputs(input_root, config)
    prepared = {}
    for seed in config["seeds"]:
        bundle = inputs["bundles"][seed]
        geometry = replay_geometry(bundle, seed)
        active = sum(d["used_ot"] for d in bundle["prototype_diagnostics"])
        if active != EXPECTED_ACTIVE_CLASSES[seed]:
            raise ValueError(f"Pinned real-only active-class count changed for seed {seed}: {active}")
        if config["stage"] == "synthetic_reweighting":
            banks = build_banks(bundle["features"], bundle["labels"])
            if len(banks["smote"]["labels"]) != EXPECTED_AUGMENTED_ROWS or len(bundle["labels"]) != EXPECTED_FITTING_ROWS:
                raise ValueError("Pinned SMOTE population size changed")
            weights, solver = build_feature_weights(bundle, geometry, banks)
            diagnostics = {name: bank_diagnostics(bank, bundle["features"]) for name, bank in banks.items()}
            prepared[seed] = {"geometry": geometry, "banks": banks, "weights": weights,
                              "solver": solver, "bank_diagnostics": diagnostics}
        else:
            weights, diagnostic = build_image_weights(bundle, geometry, inputs["checkpoints"][seed]["model_state_dict"])
            prepared[seed] = {"geometry": geometry, "weights": weights, **diagnostic}
    first = inputs["bundles"][0]
    for bundle in inputs["bundles"].values():
        for name in ("labels", "development_labels"):
            if not torch.equal(first[name], bundle[name]):
                raise ValueError("Audit representations have different row labels")
    return inputs, prepared


def _implementation():
    source = Path(__file__).parent
    names = ("run_ot_followups.py", "followup_weights.py", "followup_reporting.py", "feature_smote.py",
             "image_reweighting.py", "followup_images.py", "controlled_bias_head.py", "controlled_bias_weights.py",
             "ablation_inputs.py", "ablation_weights.py", "simple_control_weights.py",
             "ablation_data.py", "run_ablation.py", "run_weight_study.py", "runtime.py", "data.py",
             "model.py", "evaluate.py", "methods/prototype_weights.py", "methods/baseline.py")
    return {name: file_sha256(source / name) for name in names}


def initialize_study(config, inputs, prepared, output_root, device):
    root = Path(output_root)
    manifest = {"config": config, "inputs": inputs["provenance"], "implementation": _implementation(),
                "environment": environment(device) | {"torchvision": str(torchvision.__version__)},
                "preparation": content_identity(prepared)}
    if root.exists():
        if not (root / "study_config.json").is_file():
            raise ValueError("Incomplete study initialization; preserve it and choose a new output folder")
        if json.loads((root / "study_config.json").read_text()) != manifest:
            raise ValueError("Follow-up inputs, preparation, source or environment changed; choose a new output folder")
        receipt = json.loads((root / "preparation.json").read_text())
        expected = {f"diagnostics/seed{s}/{name}" for s in config["seeds"]
                    for name in ("preparation.pt", "preparation_report.json")}
        if set(receipt) != expected:
            raise ValueError("Incomplete saved preparation receipt")
        for relative, digest in receipt.items():
            if file_sha256(root / relative) != digest:
                raise ValueError("Saved preparation changed")
            if relative.endswith(".pt"):
                saved = torch.load(root / relative, weights_only=True, map_location="cpu")
                seed = int(Path(relative).parent.name.removeprefix("seed"))
                if content_identity(saved) != content_identity(prepared[seed]):
                    raise ValueError("Saved bank, geometry or weights disagree with deterministic replay")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="followup-init-", dir=root.parent) as temporary:
            staged = Path(temporary) / "study"
            staged.mkdir()
            atomic_write_json(staged / "study_config.json", manifest)
            receipt = {}
            for seed in config["seeds"]:
                path = staged / f"diagnostics/seed{seed}/preparation.pt"
                path.parent.mkdir(parents=True)
                atomic_torch_save(prepared[seed], path)
                receipt[path.relative_to(staged).as_posix()] = file_sha256(path)
                report = {"solver": prepared[seed]["solver"],
                          "active_class_count": sum(d["used_ot"] for d in inputs["bundles"][seed]["prototype_diagnostics"]),
                          "bank_diagnostics": prepared[seed].get("bank_diagnostics"),
                          "weights": {name: content_identity(value) for name, value in prepared[seed]["weights"].items()}}
                atomic_write_json(path.parent / "preparation_report.json", report)
                report_path = path.parent / "preparation_report.json"
                receipt[report_path.relative_to(staged).as_posix()] = file_sha256(report_path)
            atomic_write_json(staged / "preparation.json", receipt)
            staged.rename(root)
    return root, file_sha256(root / "study_config.json")


def _archive_interrupted(root, run):
    if run.exists() and not (run / "complete.json").exists():
        if not (run / "failure.json").exists():
            atomic_write_json(run / "failure.json", {
                "reason": "Interrupted before completion; restart from the original checkpoint or zero head.",
            })
        archive = root / "attempts" / run.relative_to(root)
        archive.mkdir(parents=True, exist_ok=True)
        attempt = archive / f"attempt{len(list(archive.iterdir())) + 1}"
        shutil.move(str(run), attempt)


def _condition_data(bundle, values, condition):
    if condition.startswith("feature_real_") or condition.startswith("image_"):
        return bundle["features"], bundle["labels"], torch.zeros(len(bundle["labels"]), dtype=torch.bool)
    kind = "duplicate" if condition.startswith("feature_duplicate_") else "smote"
    bank = values["banks"][kind]
    return bank["features"], bank["labels"], bank["is_added"]


def fit_condition(config, inputs, values, seed, condition, root, preparation_hash, device, images=None):
    bundle, checkpoint = inputs["bundles"][seed], inputs["checkpoints"][seed]
    features, labels, origins = _condition_data(bundle, values, condition)
    weights = values["weights"][condition]
    identity = {"preparation_sha256": preparation_hash, "seed": seed, "condition": condition,
                "weights": content_identity(weights), "features": content_identity(features),
                "labels": content_identity(labels)}
    run = root / f"runs/seed{seed}/{condition}"
    _archive_interrupted(root, run)
    expected_files = FEATURE_RUN_FILES if config["stage"] == "synthetic_reweighting" else RUN_FILES
    if _completed_run(run, identity):
        receipt = json.loads((run / "complete.json").read_text())
        if set(receipt["files"]) != expected_files:
            raise ValueError("Incomplete follow-up run receipt")
        return json.loads((run / "metrics.json").read_text())
    run.mkdir(parents=True)
    atomic_write_json(run / "config.json", identity | {"config": config})
    atomic_torch_save({"weights": weights, "labels": labels, "is_added": origins}, run / "weights.pt")
    atomic_write_json(run / "weight_diagnostics.json", describe_weights(
        weights, labels, origins, inputs["split"]["original_class_counts"], bundle["prototype_diagnostics"]))
    seed_everything(seed)
    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    started = time.monotonic()
    try:
        if config["stage"] == "synthetic_reweighting":
            head, optimization, history = fit_head(features, labels, weights, 100, config["head"])
            optimization["elapsed_seconds"] = time.monotonic() - started
            atomic_torch_save(head, run / "head.pt")
            logits = functional.linear(bundle["development_features"].double(), head["weight"], head["bias"])
            metrics, predictions = evaluate_logits(logits, bundle["development_labels"], inputs["split"]["original_class_counts"])
            with torch.no_grad():
                model.fc.weight.copy_(head["weight"].to(model.fc.weight))
                model.fc.bias.copy_(head["bias"].to(model.fc.bias))
                cast_logits = model.fc(bundle["development_features"].to(device)).cpu()
            cast_metrics, cast_predictions = evaluate_logits(cast_logits, bundle["development_labels"], inputs["split"]["original_class_counts"])
            cast_metrics["prediction_disagreements_with_float64"] = int((cast_predictions["predictions"] != predictions["predictions"]).sum())
            atomic_write_json(run / "cast_metrics.json", cast_metrics)
            atomic_torch_save(cast_predictions, run / "cast_predictions.pt")
            for name, tensor in model.state_dict().items():
                if not name.startswith("fc.") and not torch.equal(tensor.cpu(), checkpoint["model_state_dict"][name]):
                    raise RuntimeError("Frozen feature backbone changed")
        else:
            history, optimization = train_images(model, images["fitting"], weights, seed, config["image"], device,
                                                  lambda h: atomic_write_json(run / "history.json", h))
            logits, development_labels = image_logits(model, images["development"], device, config["image"]["batch_size"])
            if not torch.equal(development_labels, bundle["development_labels"]):
                raise ValueError("Development image labels changed")
            metrics, predictions = evaluate_logits(logits, development_labels, inputs["split"]["original_class_counts"])
        atomic_write_json(run / "history.json", history)
        atomic_write_json(run / "optimization.json", optimization)
        atomic_write_json(run / "metrics.json", metrics)
        atomic_torch_save(predictions, run / "predictions.pt")
        save_checkpoint_with_metadata(run / "final.pt", model,
                                      {"method": config["stage"], "seed": seed, "condition": condition, "study": config,
                                       "fitting_class_counts": checkpoint["class_counts"].tolist()},
                                      torch.tensor(inputs["split"]["original_class_counts"], dtype=torch.int64),
                                      checkpoint["data_settings"])
        if {p.name for p in run.iterdir()} != expected_files:
            raise ValueError("Missing or unexpected follow-up outputs")
        _finish_run(run, identity)
    except Exception as error:
        if isinstance(error, HeadConvergenceError):
            atomic_write_json(run / "optimization.json", error.diagnostics)
            atomic_write_json(run / "history.json", error.history)
        atomic_write_json(run / "failure.json", {"error_type": type(error).__name__, "message": str(error),
                                                "elapsed_seconds": time.monotonic() - started})
        raise
    return metrics


def run_study(config, input_root, output_root, device="cpu", on_progress=None, *, preflight_only=False,
              retained_path=None, split_root=None):
    config = resolve_config(config)
    if config["stage"] == "synthetic_reweighting" and str(device) != "cpu":
        raise ValueError("The feature study requires CPU float64 fitting")
    device = resolve_device(device)
    inputs, prepared = prepare_study(config, input_root)
    images = None
    if config["stage"] == "ot_contribution":
        if retained_path is None or split_root is None:
            raise ValueError("Image continuations require retained training images and --split-dir")
        images = prepare_images(inputs, retained_path, split_root)
        inputs["provenance"] = inputs["provenance"] | {"images": images["provenance"]}
    conditions = IMAGE_CONDITIONS if config["stage"] == "ot_contribution" else FEATURE_CONDITIONS
    if preflight_only:
        return {"status": "preflight_passed", "expected_runs": 3 * len(conditions), "scientific_fits": 0,
                "preparation": content_identity(prepared)}
    root, digest = initialize_study(config, inputs, prepared, output_root, device)
    records = []

    def publish(status, error=None, seed=None):
        summary = summarize(records, config)
        if status == "failed":
            summary["status"] = "failed"
            summary["decision"] = {"status": "incomplete"}
        atomic_write_json(root / "study_results.json", records)
        atomic_write_json(root / "study_summary.json", summary)
        atomic_write_json(root / "readable_report.json", readable_report(summary))
        atomic_write_json(root / "study_status.json", {"status": status, "error": error,
                          "completed_runs": len(records), "expected_runs": 3 * len(conditions)})
        if on_progress is not None:
            on_progress(root, seed, f"{config['stage']}: {status}, {len(records)}/{3 * len(conditions)} fits")

    try:
        for seed in config["seeds"]:
            for condition in conditions:
                print(f"{config['stage']}: seed {seed}, {condition}", flush=True)
                metrics = fit_condition(config, inputs, prepared[seed], seed, condition, root, digest, device, images)
                records.append({"seed": seed, "condition": condition, "metrics": metrics})
                publish("running", seed=seed)
        publish("complete")
    except Exception as error:
        # Record the scientific failure even when external publication is unavailable.
        try:
            publish("failed", f"{type(error).__name__}: {error}")
        except Exception:
            pass
        raise
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run_study(config, args.inputs, args.output_dir, args.device, preflight_only=args.preflight_only,
                       retained_path=args.train_data, split_root=args.split_dir)
    if args.preflight_only:
        print(json.dumps({k: v for k, v in result.items() if k != "preparation"}), flush=True)


if __name__ == "__main__":
    main()
