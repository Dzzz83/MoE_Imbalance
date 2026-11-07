"""Stage 5: run the planned controls and report shared CIFAR-100 metrics."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from pipeline.evaluate import evaluate_checkpoint
from pipeline.methods import synthesis
from pipeline.train_classifier import (
    file_sha256, prepare_training_features, resolve_config, run_training,
    validate_real_cache, write_json,
)


METRICS = ("accuracy", "balanced_accuracy", "head_recall", "medium_recall", "tail_recall")


def generation_variants(base):
    """Only the two specified components differ from the full generator."""
    variants = {}
    for name, uniform in (("uniform_ot", True), ("loss_shifted_ot", False)):
        config = json.loads(json.dumps(base))
        config["method_options"] = synthesis.resolve_options(config.get("method_options", {}))
        config["method_options"].update({"uniform_targets": uniform, "guidance_weight": 0.0})
        variants[name] = config
    return variants


def inspect_generation(synthetic_path):
    path = Path(synthetic_path).parent / "generation_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary["all_classes_processed"] or len(summary["classes"]) != 100:
        raise ValueError(f"Incomplete generation summary: {path}")
    generated = [record for record in summary["classes"] if record["synthetic_count"] > 0]
    for record in generated:
        for key in ("objective", "row_l1_error", "column_l1_error", "working_spread"):
            if not math.isfinite(record[key]):
                raise ValueError(f"Nonfinite generation diagnostic: {path}")
    return {
        "summary_path": str(path),
        "synthetic_total": summary["synthetic_total"],
        "iteration_limit_classes": [
            record["class_id"] for record in generated
            if record["stopping_reason"] == "iteration_limit"
        ],
        "max_row_l1_error": max((r["row_l1_error"] for r in generated), default=0.0),
        "max_column_l1_error": max((r["column_l1_error"] for r in generated), default=0.0),
    }


def write_tables(records, class_counts, output):
    """Write percentages, percentage-point differences, and all class recalls."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not records or records[0]["name"] != "baseline":
        raise ValueError("First comparison record must be the baseline")
    support = records[0]["metrics"]["test_support"]
    if len(support) != 100 or len(class_counts) != 100:
        raise ValueError("Expected 100 classes")
    names = [record["name"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate comparison names")
    for record in records:
        metrics = record["metrics"]
        if metrics["test_support"] != support or len(metrics["per_class_recall"]) != 100:
            raise ValueError("All methods must use the same test population and 100 classes")
        for key in METRICS:
            value = metrics[key]
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid metric {key}")
    baseline = records[0]["metrics"]
    columns = ["method"] + [key + "_percent" for key in METRICS] + [
        "accuracy_delta_pp", "tail_recall_delta_pp", "classifier_examples_per_epoch",
        "classifier_total_examples_processed", "classifier_training_seconds",
        "generation_seconds_this_invocation", "generation_iteration_limit_classes", "checkpoint",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metrics, training = record["metrics"], record.get("training_summary", {})
            row = {"method": record["name"], "checkpoint": record["checkpoint"]}
            row.update({key + "_percent": None if metrics[key] is None else 100 * metrics[key]
                        for key in METRICS})
            for key in ("accuracy", "tail_recall"):
                row[key + "_delta_pp"] = (
                    None if metrics[key] is None or baseline[key] is None
                    else 100 * (metrics[key] - baseline[key])
                )
            row.update({
                "classifier_examples_per_epoch": training.get("examples_per_epoch"),
                "classifier_total_examples_processed": training.get("total_examples_processed"),
                "classifier_training_seconds": training.get("training_seconds"),
                "generation_seconds_this_invocation": record.get("generation_seconds_this_invocation"),
                "generation_iteration_limit_classes": len(
                    record.get("generation_diagnostics", {}).get("iteration_limit_classes", [])
                ) if "generation_diagnostics" in record else None,
            })
            writer.writerow(row)
    with (output / "per_class_recall.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["class_id", "real_train_count", "group"] + [name + "_recall_percent" for name in names])
        for class_id, count in enumerate(class_counts):
            group = "Head" if count >= 100 else "Medium" if count >= 20 else "Tail"
            recalls = [record["metrics"]["per_class_recall"][class_id] for record in records]
            writer.writerow([class_id, count, group] + [
                None if recall is None else 100 * recall for recall in recalls
            ])
    write_json(output / "comparison.json", records)
    print("\nMethod                  Accuracy     Balanced     Head       Medium     Tail")
    for record in records:
        values = ["n/a" if record["metrics"][key] is None
                  else f"{100 * record['metrics'][key]:.2f}%" for key in METRICS]
        print(f"{record['name']:<23} " + " ".join(f"{value:>10}" for value in values))


def run_comparison(config, device="cpu", resume=False):
    allowed = {"classifier_config", "generation_config", "output_dir", "evaluation"}
    if not isinstance(config, dict) or config.keys() - allowed:
        raise ValueError("Invalid comparison configuration keys")
    classifier = resolve_config(json.loads(Path(config["classifier_config"]).read_text(encoding="utf-8")))
    if classifier["mode"] != "synthetic":
        raise ValueError("classifier_config must describe the full synthetic method")
    generation = json.loads(Path(config["generation_config"]).read_text(encoding="utf-8"))
    options = synthesis.resolve_options(generation.get("method_options", {}))
    if options["guidance_weight"] <= 0 or options["uniform_targets"]:
        raise ValueError("Full synthesis must enable guidance and loss-shifted target masses")
    if options["target_count"] != classifier["target_count"]:
        raise ValueError("Generation and classifier target counts differ")
    if generation.get("seed", 0) != classifier["seed"]:
        raise ValueError("Generation and classifier seeds must match")
    real_hash = file_sha256(classifier["feature_cache"])
    if file_sha256(generation["feature_cache"]) != real_hash:
        raise ValueError("Generation and classifier feature caches differ")
    pretrained = torch.load(classifier["pretrained_checkpoint"], map_location="cpu", weights_only=True)
    if pretrained["config"].get("seed", 0) != classifier["seed"]:
        raise ValueError("For a new experiment seed, pretrain and extract matching features first")
    cache = torch.load(classifier["feature_cache"], map_location="cpu", weights_only=True)
    validate_real_cache(cache, pretrained, file_sha256(classifier["pretrained_checkpoint"]))
    full = torch.load(classifier["synthetic_features"], map_location="cpu", weights_only=True)
    if (full["generation_config"]["method_options"] != options
            or full["generation_config"]["seed"] != classifier["seed"]):
        raise ValueError("Full synthetic dataset does not match the specified generation recipe")
    prepare_training_features(cache, classifier, real_hash, full)
    full_diagnostics = inspect_generation(classifier["synthetic_features"])
    if max(full_diagnostics["max_row_l1_error"], full_diagnostics["max_column_l1_error"]) > options["marginal_tolerance"]:
        raise ValueError("Full synthesis did not satisfy the configured marginal tolerance")
    evaluation = {"data_root": "data/raw", "download": True, "batch_size": 256, "num_workers": 4}
    specified_evaluation = config.get("evaluation", {})
    if not isinstance(specified_evaluation, dict) or specified_evaluation.keys() - evaluation.keys():
        raise ValueError("Invalid evaluation settings")
    evaluation.update(specified_evaluation)
    if (type(evaluation["batch_size"]) is not int or evaluation["batch_size"] < 1
            or type(evaluation["num_workers"]) is not int or evaluation["num_workers"] < 0
            or type(evaluation["download"]) is not bool
            or not isinstance(evaluation["data_root"], str)):
        raise ValueError("Invalid evaluation setting values")
    output = Path(config["output_dir"])
    manifest = {
        "classifier": classifier, "generation_options": options, "evaluation": evaluation,
        "pretrained_sha256": file_sha256(classifier["pretrained_checkpoint"]),
        "real_cache_sha256": real_hash,
        "full_synthetic_sha256": file_sha256(classifier["synthetic_features"]),
        "comparison_code_sha256": file_sha256(__file__),
        "synthesis_code_sha256": file_sha256(synthesis.__file__),
    }
    manifest_path = output / "experiment_config.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Comparison folder exists: {output}; use --resume")
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires identical experiment settings and inputs")
    else:
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)
    experiments = []
    for name, mode in (("real_only", "real"), ("random_duplicate", "duplicate")):
        current = classifier | {
            "mode": mode, "synthetic_features": None,
            "output_dir": str(output / "classifiers" / name),
        }
        path = run_training(current, device, resume)
        experiments.append({"name": name, "checkpoint": str(path)})
    for name, variant in generation_variants(generation).items():
        generated_dir = output / "generation" / name
        generated_path = generated_dir / "synthetic_features.pt"
        print(f"\nGenerating control: {name}", flush=True)
        was_complete = generated_path.exists()
        started = time.perf_counter()
        synthesis.run_generation(variant, device=device, output_dir=generated_dir, resume=resume)
        generation_seconds = None if was_complete else time.perf_counter() - started
        diagnostics = inspect_generation(generated_path)
        if max(diagnostics["max_row_l1_error"], diagnostics["max_column_l1_error"]) > options["marginal_tolerance"]:
            raise ValueError(f"Control {name} did not satisfy marginal tolerances")
        current = classifier | {
            "synthetic_features": str(generated_path),
            "output_dir": str(output / "classifiers" / name),
        }
        path = run_training(current, device, resume)
        experiments.append({
            "name": name, "checkpoint": str(path), "generation_diagnostics": diagnostics,
            "generation_seconds_this_invocation": generation_seconds,
        })
    full_path = run_training(classifier, device, resume)
    experiments.append({
        "name": "full_ot", "checkpoint": str(full_path),
        "generation_diagnostics": full_diagnostics,
        "generation_seconds_this_invocation": None,
    })
    experiments.insert(0, {"name": "baseline", "checkpoint": classifier["pretrained_checkpoint"]})
    # All training is complete before the first test-set evaluation in this runner.
    for experiment in experiments:
        checkpoint = torch.load(experiment["checkpoint"], map_location="cpu", weights_only=True)
        if (not torch.equal(checkpoint["class_counts"], pretrained["class_counts"])
                or checkpoint["data_settings"] != pretrained["data_settings"]):
            raise ValueError("Comparison checkpoints do not share retained-data settings")
        experiment["checkpoint_sha256"] = file_sha256(experiment["checkpoint"])
        experiment["training_summary"] = checkpoint.get("training_summary", {})
        experiment["metrics"] = evaluate_checkpoint(
            checkpoint_path=experiment["checkpoint"],
            data_root=evaluation["data_root"],
            output_path=output / f"{experiment['name']}_metrics.json",
            device=device, batch_size=evaluation["batch_size"],
            num_workers=evaluation["num_workers"], download=evaluation["download"],
        )
        if experiment["metrics"]["test_support"] != [100] * 100:
            raise ValueError("Expected the original balanced 10,000-image CIFAR-100 test set")
    write_tables(experiments, pretrained["class_counts"].tolist(), output)
    print(f"\nSaved comparison tables to {output}", flush=True)
    for experiment in experiments:
        limited = experiment.get("generation_diagnostics", {}).get("iteration_limit_classes", [])
        if limited:
            print(f"Convergence review needed for {experiment['name']}, classes: {limited}", flush=True)
    return experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        run_comparison(config, args.device, args.resume)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
