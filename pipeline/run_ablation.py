"""Run the development split, pilot, and matched prototype OT ablations."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from pipeline.ablation_data import prepare_split
from pipeline.ablation_weights import STRONG_OT_EXPONENT, transform_weights, weight_diagnostics
from pipeline.data import load_training_data
from pipeline.evaluate import compute_metrics, evaluate_checkpoint
from pipeline.extract_features import extract_features
from pipeline.methods import finetuning, prototype_weights, reweighting
from pipeline.model import create_model
from pipeline.runtime import (
    atomic_torch_save, atomic_write_json, file_sha256, load_checkpoint,
    resolve_device, save_checkpoint, seed_everything, write_json,
)
from pipeline.train import resolve_config, run_training


def resolve_study_config(config):
    defaults = {
        "train_data": "data/prepared/train.npz", "split_seed": 43,
        "seeds": [0, 1, 2], "shuffle_seeds": [0, 1, 2],
        "learning_rates": [0.00002, 0.0002, 0.002, 0.02],
        "batch_size": 128, "num_workers": 4, "baseline": {}, "classifier": {},
        "final_baselines": {},
    }
    if not isinstance(config, dict) or config.keys() - defaults.keys():
        raise ValueError("Unknown ablation configuration settings")
    resolved = defaults | config
    for name in ("seeds", "shuffle_seeds"):
        values = resolved[name]
        if (
            not isinstance(values, list) or not values
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"{name} must be a nonempty list of unique integer seeds")
    if resolved["seeds"][0] != 0:
        raise ValueError("The first training seed must be 0 for the schedule pilot")
    rates = resolved["learning_rates"]
    if (
        not isinstance(rates, list) or not rates
        or any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates)
        or len(rates) != len(set(rates))
    ):
        raise ValueError("learning_rates must be distinct finite positive numbers")
    # Use existing validators for shared training settings and recipes.
    shared = resolve_config({
        "method": "baseline", "train_data": resolved["train_data"],
        "batch_size": resolved["batch_size"], "num_workers": resolved["num_workers"],
        "method_options": resolved["baseline"],
    })
    resolved["baseline"] = shared["method_options"]
    if not isinstance(resolved["classifier"], dict):
        raise ValueError("classifier must be an object")
    if "learning_rate" in resolved["classifier"]:
        raise ValueError("Use learning_rates for the pilot; classifier learning_rate is selected by it")
    resolved["classifier"] = reweighting.prepare_classifier_options(resolved["classifier"])
    resolved["classifier"].pop("learning_rate")
    if type(resolved["split_seed"]) is not int or not 0 <= resolved["split_seed"] < 2**32:
        raise ValueError("split_seed must be in [0, 2**32)")
    sources = resolved["final_baselines"]
    if not isinstance(sources, dict) or any(
        seed not in {str(s) for s in resolved["seeds"]} or not isinstance(path, str) or not path
        for seed, path in sources.items()
    ):
        raise ValueError("final_baselines must map configured seed strings to checkpoint paths")
    resolved["train_data"] = str(Path(resolved["train_data"]).resolve())
    resolved["final_baselines"] = {seed: str(Path(path).resolve()) for seed, path in sources.items()}
    return resolved


def _read(path):
    return json.loads(Path(path).read_text())


def prepare_study(config, output_dir):
    config = resolve_study_config(config)
    output_dir = Path(output_dir).resolve()
    config_path = output_dir / "study_config.json"
    if config_path.exists():
        if _read(config_path) != config:
            raise ValueError("Study settings changed; choose a new output folder")
    elif output_dir.exists():
        raise FileExistsError(f"Choose a new study output folder: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
        write_json(config_path, config)
    pipeline_dir = Path(__file__).parent
    implementation = {name: file_sha256(pipeline_dir / name) for name in (
        "run_ablation.py", "ablation_data.py", "ablation_weights.py", "data.py", "model.py",
        "runtime.py", "train.py", "evaluate.py", "extract_features.py", "methods/baseline.py",
        "methods/finetuning.py", "methods/reweighting.py", "methods/prototype_weights.py",
    )}
    provenance = output_dir / "implementation.json"
    if provenance.exists() and _read(provenance) != implementation:
        raise ValueError("Implementation changed; choose a new study output folder")
    if not provenance.exists():
        write_json(provenance, implementation)
    split = prepare_split(config["train_data"], output_dir / "split", config["split_seed"])
    return config, output_dir, split


def _completed_run(output_dir, identity):
    """Reuse only completed, unchanged runs; never resume an interrupted optimizer."""
    if not output_dir.exists():
        return False
    receipt_path = output_dir / "complete.json"
    if not receipt_path.exists():
        raise FileExistsError(f"Incomplete run exists: {output_dir}; preserve it and choose a new study folder")
    receipt = _read(receipt_path)
    if receipt["inputs"] != identity:
        raise ValueError(f"Run inputs changed: {output_dir}")
    for name, digest in receipt["files"].items():
        if file_sha256(output_dir / name) != digest:
            raise ValueError(f"Completed run file changed: {output_dir / name}")
    return True


def _finish_run(output_dir, identity):
    atomic_write_json(output_dir / "complete.json", {
        "inputs": identity,
        "files": {p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    })


def _baseline(config, data_path, seed, output_dir, device, external=None):
    training_config = resolve_config({
        "method": "baseline", "train_data": str(data_path), "seed": seed,
        "batch_size": config["batch_size"], "num_workers": config["num_workers"],
        "method_options": config["baseline"],
    })
    if external is not None:
        data = load_training_data(data_path)
        model = create_model()
        finetuning._load_pretrained_baseline(model, data, external)
        _, checkpoint = load_checkpoint(external, "cpu")
        if checkpoint["config"] != training_config:
            # Paths can differ when the same retained file is moved between machines.
            source = checkpoint["config"] | {"train_data": str(data_path)}
            if source != training_config:
                raise ValueError("External baseline recipe/seed does not match the study")
        return Path(external)
    identity = {"config": training_config, "data_sha256": file_sha256(data_path)}
    if not _completed_run(output_dir, identity):
        run_training(training_config, output_dir, str(device))
        _finish_run(output_dir, identity)
    return output_dir / "final.pt"


def _seed_inputs(config, root, data_path, development_path, baseline_path, seed, device):
    """Cache raw features and the original OT vector once per baseline seed."""
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "baseline_sha256": file_sha256(baseline_path),
        "fitting_sha256": file_sha256(data_path),
        "development_sha256": file_sha256(development_path) if development_path else None,
        "seed": seed,
        "prototype_options": prototype_weights.prepare_options({"weighting": "sinkhorn"}),
    }
    path = root / "features_weights.pt"
    if path.exists():
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["inputs"] != identity:
            raise ValueError(f"Feature/weight inputs changed: {path}")
        return bundle | {"cache_sha256": file_sha256(path)}
    model, _ = load_checkpoint(baseline_path, device)
    fitting = load_training_data(data_path)
    features, labels = extract_features(
        model, fitting, str(device), config["batch_size"], config["num_workers"], seed,
    )
    ot_weights, diagnostics = prototype_weights.build_weights(
        features, labels, {"seed": seed, "method_options": identity["prototype_options"]}, device,
    )
    bundle = {
        "inputs": identity, "features": features, "labels": labels,
        "ot_weights": ot_weights.cpu(), "prototype_diagnostics": diagnostics,
    }
    if development_path is not None:
        development = load_training_data(development_path)
        dev_features, dev_labels = extract_features(
            model, development, str(device), config["batch_size"], config["num_workers"], seed,
        )
        bundle.update(development_features=dev_features, development_labels=dev_labels)
    atomic_torch_save(bundle, path)
    return bundle | {"cache_sha256": file_sha256(path)}


@torch.inference_mode()
def evaluate_features(model, features, labels, original_counts, device, batch_size, active_classes):
    """Use one common loss and original-count reporting groups for every condition."""
    predictions = []
    total_loss = 0.0
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for start in range(0, len(labels), batch_size):
            logits = model.fc(features[start:start + batch_size].to(device))
            targets = labels[start:start + batch_size].to(device)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite development logits")
            total_loss += functional.cross_entropy(logits, targets, reduction="sum").item()
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    finally:
        for module, training in states:
            module.training = training
    metrics = compute_metrics(np.asarray(predictions), labels.numpy(), np.asarray(original_counts))
    metrics["development_support"] = metrics.pop("test_support")
    metrics["cross_entropy"] = total_loss / len(labels)
    recalls = metrics["per_class_recall"]
    for name, members in (
        ("ot_active", active_classes),
        ("fallback", [c for c in range(100) if c not in active_classes]),
    ):
        values = [recalls[c] for c in members]
        metrics[f"{name}_class_count"] = len(members)
        metrics[f"{name}_recall"] = float(np.mean(values)) if values and None not in values else None
    return metrics


def _continuation(config, data_path, baseline_path, bundle, original_counts, seed,
                  condition, rate, output_dir, device, shuffle_seed=None):
    weights = transform_weights(bundle["ot_weights"], bundle["labels"], condition, shuffle_seed)
    classifier = config["classifier"] | {"learning_rate": rate}
    training_config = {
        "method": "ablation_finetuning", "train_data": str(data_path), "seed": seed,
        "batch_size": config["batch_size"], "num_workers": config["num_workers"],
        "method_options": {"pretrained_checkpoint": str(baseline_path), "classifier": classifier,
                           "condition": condition, "shuffle_seed": shuffle_seed},
    }
    if condition in ("strong_ot", "shuffled_strong_ot"):
        training_config["method_options"]["weight_exponent"] = STRONG_OT_EXPONENT
    identity = {"config": training_config, "cache_inputs": bundle["inputs"],
                "cache_sha256": bundle["cache_sha256"],
                "original_counts": original_counts}
    if _completed_run(output_dir, identity):
        return _read(output_dir / "metrics.json") if "development_features" in bundle else None
    data = load_training_data(data_path)
    seed_everything(seed)
    model = create_model().to(device)
    offset = finetuning._load_pretrained_baseline(model, data, baseline_path)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "config.json", training_config)
    torch.save({"weights": weights, "labels": bundle["labels"],
                "cache_inputs": bundle["inputs"]}, output_dir / "weights.pt")
    diagnostics = weight_diagnostics(weights, bundle["labels"], original_counts, bundle["prototype_diagnostics"])
    # Coverage refers to the original OT solve, even for shuffled and mixed controls.
    write_json(output_dir / "weight_diagnostics.json", diagnostics)
    active = [r["class_index"] for r in bundle["prototype_diagnostics"] if r["used_ot"]]

    def development_metrics(current_model):
        return evaluate_features(
            current_model, bundle["development_features"], bundle["development_labels"],
            original_counts, device, config["batch_size"], active,
        )

    def record_epoch(current_model, record):
        if not math.isfinite(record["loss"]) or any(
            not torch.isfinite(parameter).all() for parameter in current_model.fc.parameters()
        ):
            raise RuntimeError("Nonfinite classifier training state")
        if has_development:
            record["development"] = development_metrics(current_model)

    has_development = "development_features" in bundle
    if has_development:
        write_json(output_dir / "starting_metrics.json", development_metrics(model))
    history = reweighting.train_classifier(
        model, data, training_config, output_dir, device, epoch_offset=offset, use_ot=False,
        weight_builder=lambda features, labels, cfg, dev: (weights.to(dev), bundle["prototype_diagnostics"]),
        cached_features=(bundle["features"], bundle["labels"]),
        epoch_callback=record_epoch,
    )
    write_json(output_dir / "history.json", history)
    save_checkpoint(output_dir / "final.pt", model, training_config, data)
    if has_development:
        metrics = history[-1]["development"]
        write_json(output_dir / "metrics.json", metrics)
    else:
        metrics = None
    _finish_run(output_dir, identity)
    return metrics


def run_pilot(config, root, split, device, on_progress=None):
    if (root / "final_recipe.json").exists():
        raise ValueError("The recipe is frozen; use the existing pilot results")
    fitting = root / "split/fitting.npz"
    development = root / "split/development.npz"
    source = _baseline(config, fitting, 0, root / "baselines/seed0", device)
    if on_progress is not None:
        on_progress("save fitting baseline seed 0")
    bundle = _seed_inputs(config, root / "caches/seed0", fitting, development, source, 0, device)
    results = []
    for rate in config["learning_rates"]:
        output = root / "pilot" / f"lr_{rate}"
        failure_path = output / "pilot_failure.json"
        if failure_path.exists():
            failure = _read(failure_path)
            if failure["cache_sha256"] != bundle["cache_sha256"] or failure["learning_rate"] != rate:
                raise ValueError("Failed pilot inputs changed")
            results.append({"learning_rate": rate, "status": "failed", "reason": failure["reason"]})
            write_json(root / "pilot_results.json", results)
            if on_progress is not None:
                on_progress(f"record failed pilot rate {rate}")
            continue
        try:
            metrics = _continuation(
                config, fitting, source, bundle, split["original_class_counts"], 0,
                "class_balanced", rate, output, device,
            )
        except RuntimeError as error:
            # Only numerical failures can be excluded from selection. Infrastructure
            # failures (including OOM) must stop rather than silently alter the grid.
            if "Nonfinite" not in str(error):
                raise
            write_json(failure_path, {"learning_rate": rate, "cache_sha256": bundle["cache_sha256"],
                                      "reason": str(error)})
            results.append({"learning_rate": rate, "status": "failed", "reason": str(error)})
            write_json(root / "pilot_results.json", results)
            if on_progress is not None:
                on_progress(f"record failed pilot rate {rate}")
            continue
        results.append({"learning_rate": rate, "status": "completed", "metrics": metrics,
                        "output_dir": str(output)})
        write_json(root / "pilot_results.json", results)
        if on_progress is not None:
            on_progress(f"save pilot rate {rate}")
    successful = [r for r in results if r["status"] == "completed"]
    write_json(root / "pilot_results.json", results)
    if not successful:
        raise RuntimeError("All pilot rates failed; no schedule selected")
    best = max(successful, key=lambda r: (r["metrics"]["balanced_accuracy"], -r["learning_rate"]))
    atomic_write_json(root / "selected_schedule.json", {
        "learning_rate": best["learning_rate"], "pilot_results_sha256": file_sha256(root / "pilot_results.json"),
        "selection": "highest final-epoch development balanced accuracy; smaller rate breaks ties",
    })
    return best


def _conditions(config):
    return [(name, None) for name in ("class_balanced", "ot", "half_ot", "uniform")] + [
        ("shuffled_ot", seed) for seed in config["shuffle_seeds"]
    ]


def _condition_name(condition, shuffle_seed):
    return condition if shuffle_seed is None else f"{condition}_{shuffle_seed}"


def summarize_results(records, reference_condition="class_balanced"):
    """Average shuffles within training seed before calculating paired effects."""
    metric_names = ("balanced_accuracy", "accuracy", "head_recall", "medium_recall", "tail_recall")
    by_condition = {}
    for row in records:
        by_condition.setdefault(row["condition"], {}).setdefault(row["seed"], []).append(row["metrics"])
    references = by_condition[reference_condition]
    summary = {}
    for condition, seeds in by_condition.items():
        summary[condition] = {}
        for metric in metric_names:
            means, differences, per_seed = [], [], []
            for seed, metrics in sorted(seeds.items()):
                values = [m[metric] for m in metrics]
                reference_values = [m[metric] for m in references[seed]]
                if None in reference_values or None in values:
                    per_seed.append({"seed": seed, "mean": None, "paired_difference": None})
                    continue
                mean = float(np.mean(values))
                means.append(mean)
                difference = mean - float(np.mean(reference_values))
                differences.append(difference)
                per_seed.append({
                    "seed": seed, "mean": mean, "paired_difference": difference,
                    "shuffle_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "individual_values": values,
                })
            complete = len(means) == len(seeds)
            summary[condition][metric] = {
                "mean": float(np.mean(means)) if complete else None,
                "sample_sd": float(np.std(means, ddof=1)) if complete and len(means) > 1 else None,
                "paired_difference_mean": float(np.mean(differences)) if complete else None,
                "paired_difference_sd": float(np.std(differences, ddof=1)) if complete and len(means) > 1 else None,
                "training_seed_count": len(seeds), "per_seed": per_seed,
            }
    return summary


def _selected_schedule(root):
    selected = _read(root / "selected_schedule.json")
    if selected["pilot_results_sha256"] != file_sha256(root / "pilot_results.json"):
        raise ValueError("Pilot results changed after schedule selection")
    return selected


def run_study(config, root, split, device, on_progress=None):
    if (root / "final_recipe.json").exists():
        raise ValueError("The recipe is frozen; use the saved study results")
    rate = _selected_schedule(root)["learning_rate"]
    fitting, development = root / "split/fitting.npz", root / "split/development.npz"
    records = []
    for seed in config["seeds"]:
        source = _baseline(config, fitting, seed, root / f"baselines/seed{seed}", device)
        if on_progress is not None:
            on_progress(f"save fitting baseline seed {seed}")
        bundle = _seed_inputs(config, root / f"caches/seed{seed}", fitting, development, source, seed, device)
        baseline_model, _ = load_checkpoint(source, device)
        active = [r["class_index"] for r in bundle["prototype_diagnostics"] if r["used_ot"]]
        baseline_metrics = evaluate_features(
            baseline_model, bundle["development_features"], bundle["development_labels"],
            split["original_class_counts"], device, config["batch_size"], active,
        )
        records.append({"seed": seed, "condition": "baseline", "shuffle_seed": None,
                        "metrics": baseline_metrics, "checkpoint": str(source)})
        del baseline_model
        for condition, shuffle_seed in _conditions(config):
            name = _condition_name(condition, shuffle_seed)
            # Reuse the selected pilot run rather than retraining seed 0.
            output = (root / "pilot" / f"lr_{rate}" if seed == 0 and condition == "class_balanced"
                      else root / "study" / f"seed{seed}" / name)
            metrics = _continuation(
                config, fitting, source, bundle, split["original_class_counts"], seed,
                condition, rate, output, device, shuffle_seed,
            )
            records.append({"seed": seed, "condition": condition, "shuffle_seed": shuffle_seed,
                            "metrics": metrics, "output_dir": str(output)})
            write_json(root / "study_results.json", records)
            if on_progress is not None:
                on_progress(f"save development seed {seed} {name}")
    write_json(root / "study_results.json", records)
    write_json(root / "study_summary.json", summarize_results(records))
    return records


def run_strength_study(root, device, on_progress=None):
    """Train stronger OT and matched shuffles from an existing development study."""
    root = Path(root).resolve()
    config = _read(root / "study_config.json")
    split = _read(root / "split/split.json")
    rate = _selected_schedule(root)["learning_rate"]
    classifier = config["classifier"] | {"learning_rate": rate}
    original_records = _read(root / "study_results.json")
    expected = {(seed, condition, shuffle) for seed in config["seeds"]
                for condition, shuffle in [("baseline", None)] + _conditions(config)}
    if (
        len(original_records) != len(expected)
        or {(r["seed"], r["condition"], r["shuffle_seed"]) for r in original_records} != expected
        or not (root / "study_summary.json").is_file()
    ):
        raise ValueError("Complete the original development study before running stronger OT")
    implementation = _read(root / "implementation.json")
    pipeline_dir = Path(__file__).parent
    for name, digest in implementation.items():
        # This follow-up extends the driver and its weight transforms.
        if name not in ("run_ablation.py", "ablation_weights.py") and file_sha256(pipeline_dir / name) != digest:
            raise ValueError(f"The original study used a different training implementation: {name}")

    fitting, development = root / "split/fitting.npz", root / "split/development.npz"
    for path in (fitting, development):
        if file_sha256(path) != split["file_hashes"][path.stem]:
            raise ValueError(f"The saved development split changed: {path}")
    original_by_seed = {}
    for seed in config["seeds"]:
        baseline = root / f"baselines/seed{seed}/final.pt"
        cache_dir = root / f"caches/seed{seed}"
        if not baseline.is_file() or not (cache_dir / "features_weights.pt").is_file():
            raise FileNotFoundError(f"Restore the original baseline and feature cache for seed {seed}")
        bundle = _seed_inputs(config, cache_dir, fitting, development, baseline, seed, device)
        controls = [row for row in original_records
                    if row["seed"] == seed and row["condition"] in ("class_balanced", "ot")]
        for row in controls:
            control = (root / "pilot" / f"lr_{rate}" if seed == 0 and row["condition"] == "class_balanced"
                       else root / f"study/seed{seed}" / row["condition"])
            receipt = _read(control / "complete.json")
            control_config = receipt["inputs"]["config"]
            control_options = control_config["method_options"]
            if (
                receipt["inputs"]["cache_sha256"] != bundle["cache_sha256"]
                or receipt["inputs"]["original_counts"] != split["original_class_counts"]
                or control_config["seed"] != seed
                or any(control_config[name] != config[name] for name in ("batch_size", "num_workers"))
                or control_options["condition"] != row["condition"]
                or control_options["classifier"] != classifier
                or receipt["files"]["metrics.json"] != file_sha256(control / "metrics.json")
                or _read(control / "metrics.json") != row["metrics"]
            ):
                raise ValueError(f"Original run no longer matches the saved inputs: {control}")
        original_by_seed[seed] = baseline, bundle, controls

    output = root / "strength"
    provenance = {
        "weight_exponent": STRONG_OT_EXPONENT, "learning_rate": rate,
        "original_reports": {name: file_sha256(root / name) for name in (
            "study_config.json", "split/split.json", "selected_schedule.json", "study_results.json",
        )},
        "feature_caches": {str(seed): bundle["cache_sha256"]
                           for seed, (_, bundle, _) in original_by_seed.items()},
        "implementation": {name: file_sha256(pipeline_dir / name)
                           for name in ("run_ablation.py", "ablation_weights.py")},
    }
    config_path = output / "strength_config.json"
    if config_path.exists() and _read(config_path) != provenance:
        raise ValueError("Stronger OT inputs changed; preserve the previous strength folder before continuing")
    if not config_path.exists():
        output.mkdir(exist_ok=True)
        atomic_write_json(config_path, provenance)
    if on_progress is not None:
        on_progress("initialize stronger OT study")

    conditions = [("strong_ot", None)] + [("shuffled_strong_ot", seed) for seed in config["shuffle_seeds"]]
    records = []
    for seed, (baseline, bundle, controls) in original_by_seed.items():
        records.extend(controls)
        for condition, shuffle_seed in conditions:
            name = _condition_name(condition, shuffle_seed)
            run_dir = output / f"study/seed{seed}" / name
            metrics = _continuation(config, fitting, baseline, bundle, split["original_class_counts"],
                                    seed, condition, rate, run_dir, device, shuffle_seed)
            records.append({"seed": seed, "condition": condition, "shuffle_seed": shuffle_seed,
                            "metrics": metrics, "output_dir": str(run_dir)})
            if on_progress is not None:
                on_progress(f"save stronger OT seed {seed} {name}")
    summary = summarize_results(records)
    summary["strong_ot_vs_shuffled_strong_ot"] = summarize_results(records, "shuffled_strong_ot")["strong_ot"]
    atomic_write_json(output / "study_results.json", records)
    atomic_write_json(output / "study_summary.json", summary)
    return records


def freeze_recipe(config, root):
    schedule = _selected_schedule(root)
    records = _read(root / "study_results.json")
    expected = {(seed, condition, shuffle) for seed in config["seeds"]
                for condition, shuffle in [("baseline", None)] + _conditions(config)}
    if {(r["seed"], r["condition"], r["shuffle_seed"]) for r in records} != expected:
        raise ValueError("The full development comparison must finish before freezing")
    recipe = {
        "config": config, "learning_rate": schedule["learning_rate"],
        "study_results_sha256": file_sha256(root / "study_results.json"),
        "study_summary_sha256": file_sha256(root / "study_summary.json"),
        "selected_schedule_sha256": file_sha256(root / "selected_schedule.json"),
        "external_baseline_hashes": {seed: file_sha256(path) for seed, path in config["final_baselines"].items()},
        "final_conditions": [list(item) for item in _conditions(config)],
    }
    path = root / "final_recipe.json"
    if path.exists() and _read(path) != recipe:
        raise ValueError("An incompatible final recipe already exists")
    atomic_write_json(path, recipe)
    return recipe


def run_final(config, root, split, device, data_root, download=False, on_progress=None):
    recipe = _read(root / "final_recipe.json")
    if (
        recipe["config"] != config
        or any(
            recipe[field] != file_sha256(root / name) for field, name in (
                ("study_results_sha256", "study_results.json"),
                ("study_summary_sha256", "study_summary.json"),
                ("selected_schedule_sha256", "selected_schedule.json"),
            )
        )
        or recipe["learning_rate"] != _selected_schedule(root)["learning_rate"]
        or recipe["final_conditions"] != [list(item) for item in _conditions(config)]
        or recipe["external_baseline_hashes"] != {
            seed: file_sha256(path) for seed, path in config["final_baselines"].items()
        }
    ):
        raise ValueError("Frozen recipe inputs changed")
    records = []
    data_path = Path(config["train_data"])
    for seed in config["seeds"]:
        source = _baseline(
            config, data_path, seed, root / f"final/baselines/seed{seed}", device,
            config["final_baselines"].get(str(seed)),
        )
        if on_progress is not None:
            on_progress(f"save full-pool baseline seed {seed}")
        bundle = _seed_inputs(config, root / f"final/caches/seed{seed}", data_path, None, source, seed, device)
        outputs = [("baseline", None, source, root / f"final/evaluation/seed{seed}/baseline")]
        for condition, shuffle_seed in recipe["final_conditions"]:
            output = root / f"final/runs/seed{seed}" / _condition_name(condition, shuffle_seed)
            _continuation(
                config, data_path, source, bundle, split["original_class_counts"], seed,
                condition, recipe["learning_rate"], output, device, shuffle_seed,
            )
            if on_progress is not None:
                on_progress(f"save full-pool seed {seed} {_condition_name(condition, shuffle_seed)}")
            outputs.append((condition, shuffle_seed, output / "final.pt",
                            root / f"final/evaluation/seed{seed}" / _condition_name(condition, shuffle_seed)))
        for condition, shuffle_seed, checkpoint, evaluation_dir in outputs:
            identity = {"checkpoint_sha256": file_sha256(checkpoint),
                        "recipe_sha256": file_sha256(root / "final_recipe.json"),
                        "data_root": str(Path(data_root).resolve())}
            if not _completed_run(evaluation_dir, identity):
                evaluation_dir.mkdir(parents=True)
                evaluate_checkpoint(
                    checkpoint, data_root, evaluation_dir / "metrics.json", str(device),
                    config["batch_size"], config["num_workers"], download,
                )
                _finish_run(evaluation_dir, identity)
            records.append({"seed": seed, "condition": condition, "shuffle_seed": shuffle_seed,
                            "metrics": _read(evaluation_dir / "metrics.json")})
            write_json(root / "final/results.json", records)
            if on_progress is not None:
                on_progress(f"save test seed {seed} {_condition_name(condition, shuffle_seed)}")
    write_json(root / "final/results.json", records)
    write_json(root / "final/summary.json", summarize_results(records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "study", "strength", "freeze", "final"))
    parser.add_argument("--config", type=Path, default=Path("configs/ablation_study.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation_study"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--download", action="store_true", help="Allow CIFAR test download for final evaluation only")
    args = parser.parse_args()
    try:
        if args.stage == "strength":
            root = args.output_dir.resolve()
            run_strength_study(root, resolve_device(args.device))
        else:
            config, root, split = prepare_study(_read(args.config), args.output_dir)
            device = resolve_device(args.device)
            if args.stage == "pilot":
                run_pilot(config, root, split, device)
            elif args.stage == "study":
                run_study(config, root, split, device)
            elif args.stage == "freeze":
                freeze_recipe(config, root)
            elif args.stage == "final":
                run_final(config, root, split, device, args.data_root, args.download)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Completed ablation stage '{args.stage}' in {root}")


if __name__ == "__main__":
    main()
