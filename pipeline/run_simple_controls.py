"""Fifteen exploratory Stage B continuations, with immutable Stage A references."""

import json
from pathlib import Path
import statistics

import torch

from pipeline.ablation_inputs import load_audit_inputs
from pipeline import ablation_references as references
from pipeline import run_weight_study as assignment
from pipeline.runtime import atomic_torch_save, atomic_write_json, file_sha256
from pipeline.simple_control_weights import CONDITIONS, NATIVE_CONDITIONS, build_controls, replay_geometry

METRICS = ("balanced_accuracy", "accuracy", "head_recall", "medium_recall", "tail_recall",
           "ot_active_recall", "fallback_recall", "cross_entropy")


def resolve_config(config):
    defaults = references.ASSIGNMENT_CONFIG | {
        "stage": "simple_controls", "parent_results_commit": references.PARENT_COMMIT,
        "parent_archive_sha256": references.PARENT_ARCHIVE_SHA256,
    }
    if not isinstance(config, dict) or config.keys() - defaults.keys():
        raise ValueError("Unknown simple-controls configuration settings")
    resolved = defaults | config
    if isinstance(config.get("classifier"), dict):
        resolved["classifier"] = defaults["classifier"] | config["classifier"]
    if json.dumps(resolved, sort_keys=True) != json.dumps(defaults, sort_keys=True):
        raise ValueError("Stage B settings are frozen to PLAN.md and the pinned Stage A recipe")
    return resolved


def prepare_study(config, input_root, parent_root, device=None):
    config = resolve_config(config)
    inputs = load_audit_inputs(input_root, config)
    parent = references.load_parent(parent_root, inputs)
    if device is not None:
        references.validate_environment(parent["environment"], device)
    prepared = {}
    for seed in config["seeds"]:
        bundle = inputs["bundles"][seed]
        geometry = replay_geometry(bundle, seed)
        weights, diagnostics = build_controls(bundle, geometry, parent["scores"][seed], parent["strong"][seed])
        prepared[seed] = {"geometry": geometry, "weights": weights, "diagnostics": diagnostics}
    return inputs, parent, prepared


def _paired_metrics(rows, condition, reference):
    result = {}
    for metric in METRICS:
        per_seed = []
        for seed in (0, 1, 2):
            values = [r["metrics"][metric] for r in rows if r["seed"] == seed and r["condition"] == condition]
            baseline = [r["metrics"][metric] for r in rows if r["seed"] == seed and r["condition"] == reference]
            mean = statistics.mean(values) if values and None not in values else None
            ref_mean = statistics.mean(baseline) if baseline and None not in baseline else None
            per_seed.append({"seed": seed, "mean": mean, "reference_mean": ref_mean,
                             "individual_values": values, "reference_individual_values": baseline,
                             "paired_difference": mean - ref_mean if mean is not None and ref_mean is not None else None})
        differences = [r["paired_difference"] for r in per_seed]
        means = [r["mean"] for r in per_seed]
        result[metric] = {
            "per_seed": per_seed, "training_seed_count": 3,
            "mean": statistics.mean(means) if None not in means else None,
            "sample_sd": statistics.stdev(means) if None not in means else None,
            "paired_difference_mean": statistics.mean(differences) if None not in differences else None,
            "paired_difference_sd": statistics.stdev(differences) if None not in differences else None,
        }
    return result


def _class_contributions(rows, condition, reference, active_by_seed):
    per_seed = []
    for seed in (0, 1, 2):
        current = [r["metrics"] for r in rows if r["seed"] == seed and r["condition"] == condition]
        previous = [r["metrics"] for r in rows if r["seed"] == seed and r["condition"] == reference]
        classes = []
        for c, support in enumerate(current[0]["development_support"]):
            recall = statistics.mean(r["per_class_recall"][c] for r in current)
            reference_recall = statistics.mean(r["per_class_recall"][c] for r in previous)
            classes.append({
                "class_index": c, "development_support": support, "original_ot_active": c in active_by_seed[seed],
                "correct_count_mean": statistics.mean(round(r["per_class_recall"][c] * support) for r in current),
                "reference_correct_count_mean": statistics.mean(round(r["per_class_recall"][c] * support) for r in previous),
                "recall_difference": recall - reference_recall,
                "balanced_accuracy_contribution": (recall - reference_recall) / 100,
            })
        totals = {name: sum(r["balanced_accuracy_contribution"] for r in classes if predicate(r))
                  for name, predicate in (
                      ("ot_active_contribution", lambda r: r["original_ot_active"]),
                      ("fallback_contribution", lambda r: not r["original_ot_active"]),
                      ("one_image_class_contribution", lambda r: r["development_support"] == 1),
                  )}
        per_seed.append({"seed": seed, **totals, "classes": classes,
                         "balanced_accuracy_difference": sum(r["balanced_accuracy_contribution"] for r in classes)})
    return per_seed


def summarize_controls(records, config, active_by_seed):
    expected_parent = {(s, c, t) for s in config["seeds"]
                       for c, t in [("baseline", None)] + assignment.conditions(config)}
    expected_new = {(s, c, None) for s in config["seeds"] for c in CONDITIONS}
    identities = [(r["seed"], r["condition"], r["shuffle_seed"]) for r in records]
    if len(identities) != len(set(identities)) or not set(identities) <= expected_parent | expected_new:
        raise ValueError("Unexpected or duplicate Stage B result")
    completed = len(set(identities) & expected_new)
    if set(identities) != expected_parent | expected_new:
        return {"status": "incomplete", "completed_runs": completed, "expected_runs": 15,
                "decision": {"status": "incomplete"}}
    primary = [("ot", c) for c in NATIVE_CONDITIONS]
    usefulness = [(c, "class_balanced") for c in CONDITIONS] + [(c, "ot") for c in NATIVE_CONDITIONS]
    secondary = [
        ("strong_ot", "cluster_rank_strong"), ("strong_ot", "distance_rank_strong"),
        ("strong_ot", "shuffled_strong_ot"), ("cluster_rank_strong", "shuffled_strong_ot"),
        ("distance_rank_strong", "shuffled_strong_ot"), ("easy_strong", "easy_rank"),
    ]
    groups = {}
    for name, pairs in (("primary_contrasts", primary), ("usefulness_contrasts", usefulness),
                        ("secondary_contrasts", secondary)):
        groups[name] = {f"{c}_minus_{r}": _paired_metrics(records, c, r) for c, r in pairs}
    contributions = {f"{c}_minus_{r}": _class_contributions(records, c, r, active_by_seed)
                     for c, r in dict.fromkeys(primary + usefulness + secondary)}

    def meets_target(contrast):
        metric = contrast["balanced_accuracy"]
        return (metric["paired_difference_mean"] >= config["practical_gain"]
                and sum(r["paired_difference"] > 0 for r in metric["per_seed"]) >= 2)

    candidates = [c for c in NATIVE_CONDITIONS if all(
        meets_target(groups["usefulness_contrasts"][f"{c}_minus_{r}"]) for r in ("class_balanced", "ot"))]
    ot_advantage = {c: meets_target(groups["primary_contrasts"][f"ot_minus_{c}"]) for c in NATIVE_CONDITIONS}
    parent_ot = _paired_metrics(records, "ot", "class_balanced")
    return {
        "status": "complete", "completed_runs": 15, "expected_runs": 15, "reused_parent_records": 30,
        "conditions": {c: _paired_metrics(records, c, "class_balanced")
                       for c in dict.fromkeys(r["condition"] for r in records)},
        **groups, "class_contributions": contributions,
        "decision": {
            "status": "confirmation_candidate" if candidates else "review_deprioritize_or_unresolved",
            "exploratory_native_candidates": candidates, "ot_practical_advantage_by_native": ot_advantage,
            "parent_ot_versus_class_balanced": parent_ot,
            "parent_ot_usefulness_target_met": meets_target(parent_ot),
            "practical_gain": config["practical_gain"], "minimum_positive_seeds": 2,
            "interpretation": "Research-budget rules, not significance or equivalence tests. Native comparisons "
                              "compare complete recipes. Matched rank/shuffle spectra address assignment. "
                              "Review one-image class contributions before selecting a separate confirmation.",
            "next_stage": "Requires review; no automatic sweep, new split, backbone training or official test.",
        },
    }


def run_study(config, input_root, parent_root, output_root, device, on_progress=None):
    config = resolve_config(config)
    device = torch.device(device)
    inputs, parent, prepared = prepare_study(config, input_root, parent_root, device)
    root = Path(output_root)
    source = Path(__file__).parent
    names = (
        "run_simple_controls.py", "simple_control_weights.py", "ablation_references.py",
        "run_weight_study.py", "ablation_inputs.py", "ablation_weights.py", "run_ablation.py",
        "runtime.py", "model.py", "evaluate.py", "extract_features.py",
        "methods/reweighting.py", "methods/prototype_weights.py",
    )
    implementation = {name: file_sha256(source / name) for name in names}
    manifest = {"config": config, "inputs": inputs["provenance"], "parent": parent["provenance"],
                "implementation": implementation, "environment": assignment.environment(device),
                "expected_runs": [{"seed": s, "condition": c, "shuffle_seed": None}
                                  for s in config["seeds"] for c in CONDITIONS]}
    path = root / "study_config.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Stage B inputs, references, implementation or environment changed; choose a new output folder")
    elif root.exists():
        raise FileExistsError("Incomplete Stage B initialization; preserve it and choose a new output folder")
    else:
        root.mkdir(parents=True)
        atomic_write_json(path, manifest)
    atomic_write_json(root / "inputs.json", inputs["provenance"])
    atomic_write_json(root / "parent.json", parent["provenance"])
    atomic_write_json(root / "implementation.json", implementation)
    records = []
    for row in parent["records"]:
        record = dict(row, origin="stage_a", parent_results_commit=references.PARENT_COMMIT)
        if "output_dir" in record:
            record["parent_output_dir"] = record.pop("output_dir")
        records.append(record)
    active = {s: [d["class_index"] for d in inputs["bundles"][s]["prototype_diagnostics"] if d["used_ot"]]
              for s in config["seeds"]}

    def publish_state(status, error=None):
        atomic_write_json(root / "study_results.json", records)
        summary = summarize_controls(records, config, active)
        if status == "failed":
            summary["status"], summary["decision"] = "failed", {"status": "incomplete"}
        atomic_write_json(root / "study_summary.json", summary)
        atomic_write_json(root / "study_status.json", {
            "status": status, "error": error, "completed_runs": summary["completed_runs"], "expected_runs": 15,
        })

    try:
        publish_state("running")
        for seed in config["seeds"]:
            values = prepared[seed]
            folder = root / f"diagnostics/seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            geometry_path = folder / "geometry.pt"
            if geometry_path.exists():
                saved = torch.load(geometry_path, map_location="cpu", weights_only=True)
                if len(saved) != len(values["geometry"]):
                    raise ValueError("Saved Stage B geometry changed")
                for current, previous in zip(values["geometry"], saved):
                    if current.keys() != previous.keys():
                        raise ValueError("Saved Stage B geometry changed")
                    for name, value in current.items():
                        if isinstance(value, torch.Tensor):
                            is_equal = isinstance(previous[name], torch.Tensor) and torch.equal(value, previous[name])
                        else:
                            is_equal = value == previous[name]
                        if not is_equal:
                            raise ValueError("Saved Stage B geometry changed")
            else:
                atomic_torch_save(values["geometry"], geometry_path)
            diagnostic = values["diagnostics"] | {
                "geometry_sha256": file_sha256(geometry_path),
                "scores_sha256": {k: assignment._tensor_hash(v) for k, v in parent["scores"][seed].items()},
                "weights_sha256": {k: assignment._tensor_hash(v) for k, v in values["weights"].items()},
                "replay": {"active_mask_and_cluster_sizes": "exact", "ot_rtol": 1e-5, "ot_atol": 1e-8},
            }
            atomic_write_json(folder / "controls.json", diagnostic)
            for condition in CONDITIONS:
                run = root / f"runs/seed{seed}/{condition}"
                metrics = assignment._continuation(
                    config, inputs, seed, condition, None, values["weights"][condition],
                    parent["scores"][seed], run, device,
                )
                records.append({"seed": seed, "condition": condition, "shuffle_seed": None, "metrics": metrics,
                                "output_dir": run.relative_to(root).as_posix(), "origin": "stage_b"})
                publish_state("running")
                if on_progress is not None:
                    on_progress(root, seed, f"save simple controls seed {seed} {condition}")
        publish_state("complete")
        if on_progress is not None:
            on_progress(root, None, "complete simple controls study")
    except Exception as error:
        publish_state("failed", f"{type(error).__name__}: {error}")
        raise
    return records
