"""Bounded, gated fresh-head diagnostic on pinned frozen features."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from pipeline.ablation_inputs import ARCHIVE_SHA256, AUDIT_COMMIT, load_audit_inputs
from pipeline import controlled_bias_data as data
from pipeline.controlled_bias_head import HeadConvergenceError, evaluate_head, fit_head
from pipeline.controlled_bias_reporting import GATE_ONE, GATE_TWO, gate_one, readable_report, summarize
from pipeline.controlled_bias_weights import (
    build_geometry_weights, class_balanced, geometry_diagnostics, known_correction, weight_diagnostics,
)
from pipeline.run_ablation import _completed_run, _finish_run
from pipeline.run_weight_study import environment
from pipeline.runtime import atomic_torch_save, atomic_write_json, content_identity, file_sha256

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/controlled_bias.json"
RUN_FILES = {"config.json", "head.pt", "history.json", "optimization.json", "metrics.json", "predictions.pt"}


def resolve_config(config):
    defaults = json.loads(CONFIG_PATH.read_text()) | {
        "input_results_commit": AUDIT_COMMIT, "archive_sha256": ARCHIVE_SHA256,
        "controller_results_commit": data.CONTROLLER_COMMIT,
        "controller_archive_sha256": data.CONTROLLER_ARCHIVE_SHA256,
    }
    if not isinstance(config, dict) or config.keys() - defaults.keys():
        raise ValueError("Unknown controlled-bias configuration settings")
    resolved = defaults | config
    for name in ("head", "gate"):
        if isinstance(config.get(name), dict):
            resolved[name] = defaults[name] | config[name]
    if json.dumps(resolved, sort_keys=True) != json.dumps(defaults, sort_keys=True):
        raise ValueError("Controlled-bias settings are frozen to PLAN.md; record a new protocol before changing them")
    return resolved


def prepare_study(config, input_root, controller_root):
    """Prepare all fitting arrays and all 27 fixed weights before fitting any head."""
    config = resolve_config(config)
    torch.set_num_threads(config["cpu_threads"])
    inputs = load_audit_inputs(input_root, config)
    data.validate_cache_alignment(inputs["bundles"])
    controller, controller_provenance = data.load_controller(controller_root)
    sampling, sampling_report = data.prepare_sampling(
        inputs["bundles"][0]["labels"], inputs["split"]["original_class_counts"], controller, config)
    for name, count in data.EXPECTED_COUNTS.items():
        if name != "development" and len(sampling[name]) != count:
            raise ValueError(f"Pinned sampling count changed: {name}")
    labels = inputs["bundles"][0]["labels"]
    label_map = sampling["label_map"]
    class_count = len(sampling["eligible_original_classes"])
    development_labels = label_map[inputs["bundles"][0]["development_labels"]]
    selected_development = development_labels[development_labels >= 0]
    if (len(selected_development) != data.EXPECTED_COUNTS["development"]
            or (torch.bincount(selected_development, minlength=class_count) == 0).any()):
        raise ValueError("Pinned eligible development counts changed")
    unbiased_labels = label_map[labels[sampling["unbiased"]]]
    retained_labels = label_map[labels[sampling["retained"]]]
    reference_labels = label_map[labels[sampling["reference"]]]
    base_weights = {
        "unbiased_cb": class_balanced(unbiased_labels, class_count),
        "biased_cb": class_balanced(retained_labels, class_count),
        "known_correction": known_correction(retained_labels, torch.bincount(unbiased_labels, minlength=class_count),
                                              sampling["inclusion_probabilities"]),
    }
    prepared = {}
    for seed in config["seeds"]:
        features = inputs["bundles"][seed]["features"]
        weights, geometry, method_report = build_geometry_weights(
            features[sampling["retained"]], retained_labels, features[sampling["reference"]], reference_labels,
            class_count, seed, config)
        weights = base_weights | weights
        diagnostics = {}
        for condition, values in weights.items():
            positions = sampling["unbiased" if condition == "unbiased_cb" else "retained"]
            diagnostics[condition] = weight_diagnostics(
                values, label_map[labels[positions]], class_count, sampling["controller_groups"][positions],
                unbiased_labels, sampling["controller_groups"][sampling["unbiased"]])
        prepared[seed] = {"weights": weights, "geometry": geometry, "method_report": method_report,
                          "weight_diagnostics": diagnostics,
                          "geometry_diagnostics": geometry_diagnostics(
                              geometry, retained_labels, sampling["controller_groups"][sampling["retained"]],
                              reference_labels, sampling["controller_groups"][sampling["reference"]],
                              unbiased_labels, sampling["controller_groups"][sampling["unbiased"]])}
    return {"inputs": inputs, "controller": controller, "controller_provenance": controller_provenance,
            "sampling": sampling, "sampling_report": sampling_report, "prepared": prepared}


def verify_receipt(root, receipt, expected_files):
    if set(receipt["files"]) != set(expected_files):
        raise ValueError("Incomplete diagnostic receipt")
    for name, digest in receipt["files"].items():
        path = root / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Diagnostic artifact missing or changed: {name}")


def initialize_study(config, prepared, output_root):
    root = Path(output_root)
    source = Path(__file__).parent
    names = ("run_controlled_bias.py", "controlled_bias_data.py", "controlled_bias_weights.py",
             "controlled_bias_head.py", "controlled_bias_reporting.py", "ablation_inputs.py",
             "methods/prototype_weights.py", "runtime.py", "run_ablation.py", "run_weight_study.py")
    implementation = {name: file_sha256(source / name) for name in names}
    implementation["../configs/controlled_bias.json"] = file_sha256(CONFIG_PATH)
    implementation["../scripts/kaggle_ablation_study.py"] = file_sha256(source.parent / "scripts/kaggle_ablation_study.py")
    artifacts = {"preparation/sampling.pt": prepared["sampling"],
                 "preparation/sampling.json": prepared["sampling_report"]}
    for seed, current in prepared["prepared"].items():
        artifacts[f"diagnostics/seed{seed}/geometry.pt"] = current["geometry"]
        artifacts[f"diagnostics/seed{seed}/weights.pt"] = current["weights"]
        artifacts[f"diagnostics/seed{seed}/diagnostics.json"] = {
            name: current[name] for name in ("method_report", "weight_diagnostics", "geometry_diagnostics")}
    manifest = {"config": config, "inputs": prepared["inputs"]["provenance"],
                "controller": prepared["controller_provenance"], "implementation": implementation,
                "environment": environment(torch.device("cpu")), "preparation": content_identity(artifacts),
                "maximum_fits": 27, "feature_dtype": str(prepared["inputs"]["bundles"][0]["features"].dtype),
                "head_dtype": "torch.float64", "initialization": "zero", "device": "cpu"}
    path = root / "study_config.json"
    receipt_path = root / "preparation/complete.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Diagnostic inputs, preparation, implementation or environment changed; choose a new output folder")
        if not receipt_path.is_file():
            raise ValueError("Incomplete preparation; preserve it and choose a new output folder")
        receipt = json.loads(receipt_path.read_text())
        verify_receipt(root, receipt, artifacts.keys() | {"study_config.json"})
    elif root.exists():
        raise FileExistsError("Incomplete diagnostic initialization; choose a new output folder")
    else:
        root.mkdir(parents=True)
        atomic_write_json(path, manifest)
        for name, value in artifacts.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".pt":
                atomic_torch_save(value, target)
            else:
                atomic_write_json(target, value)
        atomic_write_json(receipt_path, {"files": {name: file_sha256(root / name) for name in artifacts.keys() | {"study_config.json"}}})
    return root, file_sha256(receipt_path)


def archive_interrupted_run(root, run):
    """Keep failed/interrupted attempts; a retry always starts a fresh zero head."""
    if not run.exists() or (run / "complete.json").exists():
        return
    relative = run.relative_to(root / "runs")
    attempts = root / "attempts" / relative
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (attempts / f"attempt{attempt}").exists():
        attempt += 1
    shutil.move(str(run), attempts / f"attempt{attempt}")


def fit_condition(config, prepared, context, root, preparation_sha256, seed, condition):
    sampling, inputs = prepared["sampling"], prepared["inputs"]
    bundle = inputs["bundles"][seed]
    positions = sampling["unbiased" if condition == "unbiased_cb" else "retained"]
    labels = sampling["label_map"][bundle["labels"][positions]]
    weights = prepared["prepared"][seed]["weights"][condition]
    identity = {"preparation_sha256": preparation_sha256, "seed": seed, "condition": condition,
                "positions": content_identity(positions), "weights": content_identity(weights)}
    run = root / f"runs/seed{seed}/{condition}"
    if (run / "complete.json").is_file():
        receipt = json.loads((run / "complete.json").read_text())
        verify_receipt(run, receipt, RUN_FILES)
        _completed_run(run, identity)
    else:
        archive_interrupted_run(root, run)
        run.mkdir(parents=True)
        atomic_write_json(run / "config.json", identity)
        try:
            head, optimization, history = fit_head(
                bundle["features"][positions], labels, weights, len(sampling["eligible_original_classes"]), config["head"])
            metrics, predictions = evaluate_head(
                head, bundle["development_features"][context["positions"]], context["labels"],
                context["controller_groups"], sampling["biased_groups"], sampling["eligible_original_classes"])
            predictions["development_positions"] = context["positions"]
            atomic_torch_save({"artifact_type": "controlled_bias_linear_head", "state_dict": head,
                               "eligible_original_classes": sampling["eligible_original_classes"],
                               "objective": config["head"], "inputs": identity}, run / "head.pt")
            atomic_write_json(run / "history.json", history)
            atomic_write_json(run / "optimization.json", optimization)
            atomic_write_json(run / "metrics.json", metrics)
            atomic_torch_save(predictions, run / "predictions.pt")
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error), "identity": identity}
            if isinstance(error, HeadConvergenceError):
                failure["optimization"] = error.diagnostics
                atomic_write_json(run / "history.json", error.history)
            atomic_write_json(run / "failure.json", failure)
            raise
        _finish_run(run, identity)
    return {"seed": seed, "condition": condition, "output_dir": run.relative_to(root).as_posix(),
            "metrics": json.loads((run / "metrics.json").read_text()),
            "optimization": json.loads((run / "optimization.json").read_text()),
            "completion_sha256": file_sha256(run / "complete.json")}


def freeze_gate(root, records, config, preparation_sha256):
    gate = {"preparation_sha256": preparation_sha256, "decision": gate_one(records, config),
            "completed_runs": {row["output_dir"]: row["completion_sha256"] for row in records if row["condition"] in GATE_ONE}}
    path = root / "gate_one.json"
    if path.exists():
        if json.loads(path.read_text()) != gate:
            raise ValueError("Persisted Gate-1 receipt changed")
    else:
        atomic_write_json(path, gate)
    for name, digest in gate["completed_runs"].items():
        if file_sha256(root / name / "complete.json") != digest:
            raise ValueError("Gate-1 run receipt changed")
        verify_receipt(root / name, json.loads((root / name / "complete.json").read_text()), RUN_FILES)
    return gate["decision"]


def run_study(config, input_root, controller_root, output_root, on_progress=None, *, preflight_only=False):
    config = resolve_config(config)
    prepared = prepare_study(config, input_root, controller_root)
    root, preparation_sha256 = initialize_study(config, prepared, output_root)
    if preflight_only:
        return {"status": "preflight_complete", "maximum_fits": 27,
                "sampling_counts": prepared["sampling_report"]["counts"], "output_root": str(root)}
    # This reporter-only construction happens after all fitted state is frozen.
    context = data.development_context(prepared["inputs"]["bundles"][0], prepared["controller"], prepared["sampling"])
    records = []

    def publish_state(status=None, error=None):
        summary = summarize(records, config, technical_failure=error is not None)
        atomic_write_json(root / "study_results.json", records)
        atomic_write_json(root / "study_summary.json", summary)
        atomic_write_json(root / "readable_report.json", readable_report(summary, records))
        atomic_write_json(root / "study_status.json", {
            "status": status or summary["status"], "error": error, "completed_runs": len(records),
            "expected_runs": summary["expected_runs"], "maximum_runs": 27})
        return summary

    try:
        for conditions in (GATE_ONE, GATE_TWO):
            for seed in config["seeds"]:
                for condition in conditions:
                    records.append(fit_condition(config, prepared, context, root, preparation_sha256, seed, condition))
                    publish_state("running")
                    if on_progress:
                        on_progress(root, seed, f"save controlled bias seed {seed} {condition}")
            if conditions == GATE_ONE:
                gate = freeze_gate(root, records, config, preparation_sha256)
                publish_state()
                if on_progress:
                    on_progress(root, None, "save controlled bias gate one")
                if not gate["passed"]:
                    return publish_state()
                # Re-read, recompute and verify all nine receipts immediately before Gate 2.
                if not freeze_gate(root, records, config, preparation_sha256)["passed"]:
                    raise ValueError("Gate 2 requires a verified passing Gate 1")
        summary = publish_state()
        if on_progress:
            on_progress(root, None, "complete controlled bias study")
        return summary
    except Exception as error:
        publish_state("failed", f"{type(error).__name__}: {error}")
        if on_progress:
            on_progress(root, None, "save failed controlled bias study")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(json.loads(args.config.read_text()), args.inputs, args.controller, args.output,
                       preflight_only=args.command == "preflight")
    print(json.dumps(result if args.command == "preflight" else {k: result[k] for k in ("status", "completed_runs", "gate_one", "decision")}, indent=2))


if __name__ == "__main__":
    main()
