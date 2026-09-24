"""One locked outer-fold-0 evaluation, after four full OOF expert jobs exist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data.nested_oof import NestedOOFFoldManager
from scripts.analysis import ArtifactReader, ImmutableArtifactWriter
from scripts.oof_pipeline import OOFArtifactStore, OOFRunSpec
from scripts.ridge_sinkhorn_study import ANCHOR, evaluate
from scripts.task3e_fixed import EXPERT_KEYS, EXPERT_ORDER
from scripts.task3f_ridge import (
    FIXED_REFERENCE_UNITS, HISTORICAL_NO_CE_WEIGHTS,
    _weighted_probability_predictions, classification_metrics, extract_features,
)


EXPERIMENT_ID = "ridge_sinkhorn_outer_s78_o0"
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_REPLICATES = 10_000


def _load_outer_predictions(
    manager: NestedOOFFoldManager, artifact_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    store = OOFArtifactStore(root=artifact_root, manager=manager, experiment_id=EXPERIMENT_ID)
    expected = tuple(manager.outer_fold(0).evaluation_indices)
    labels_by_index = dict(zip(manager.canonical_indices, manager.training_labels))
    logits = np.empty((len(expected), len(EXPERT_ORDER), manager.num_classes), dtype=np.float64)
    provenance = {}
    reader = ArtifactReader()
    for expert_axis, (expert_key, expert_name) in enumerate(zip(EXPERT_KEYS, EXPERT_ORDER)):
        context = OOFRunSpec(
            experiment_id=EXPERIMENT_ID, expert=expert_key,
            training_seed=78, outer_fold_id=0,
        ).resolve(manager)
        completed = store.validate_completed_run(context)
        if context.role != "outer_evaluation" or context.inner_fold_id is not None:
            raise ValueError("outer loader received a non-outer expert artifact")
        records = {record.sample_index: record for record in completed.artifact.records}
        if set(records) != set(expected):
            raise ValueError(f"{expert_name} outer predictions do not cover the reserved population")
        for row, sample_id in enumerate(expected):
            record = records[sample_id]
            if record.expert_id != expert_name or record.training_label != labels_by_index[sample_id]:
                raise ValueError(f"{expert_name} outer record order, expert, or label is invalid")
            logits[row, expert_axis] = record.logits
        provenance[expert_name] = {
            "prediction_path": str(completed.prediction_path),
            "prediction_sha256": reader.sha256_file(completed.prediction_path),
            "checkpoint_sha256": completed.metadata["checkpoint"]["sha256"],
        }
    labels = np.asarray([labels_by_index[sample_id] for sample_id in expected], dtype=np.int64)
    return np.asarray(expected, dtype=np.int64), labels, logits, provenance


def _locked_weights(logits: np.ndarray, model: dict, arrays: dict[str, np.ndarray]) -> np.ndarray:
    from scripts.ridge_sinkhorn_oracle import residuals_to_weights, smooth_positive_kernel
    from scripts.ridge_sinkhorn_ot import apply_log_bias

    if model["schema_version"] != "ridge_sinkhorn_locked_outer.v1":
        raise ValueError("outer model schema is incompatible")
    if model["feature_set"] != "confidence_only" or tuple(model["expert_order"]) != EXPERT_ORDER:
        raise ValueError("locked feature set or expert order changed")
    features = extract_features(logits, model["feature_set"])
    standardized = (features - arrays["scaler_mean"]) / arrays["scaler_scale"]
    residual = standardized @ arrays["ridge_coef"].T + arrays["ridge_intercept"]
    weights = residuals_to_weights(residual, arrays["anchor"], float(model["scale"]))
    if bool(model["ot"]):
        weights = apply_log_bias(smooth_positive_kernel(weights), arrays["log_prices"])
    return weights


def _reference_predictions(logits: np.ndarray) -> dict[str, np.ndarray]:
    references = {
        "uniform_logit": np.mean(logits, axis=1).argmax(axis=1),
        "uniform_probability": _weighted_probability_predictions(logits, np.full(4, 0.25)),
        "uniform_without_ce": np.einsum("e,nec->nc", np.asarray(HISTORICAL_NO_CE_WEIGHTS), logits).argmax(axis=1),
    }
    for identifier, units in FIXED_REFERENCE_UNITS.items():
        references[identifier] = np.einsum("e,nec->nc", np.asarray(units) / 4, logits).argmax(axis=1)
    return references


def hierarchical_paired_intervals(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    class_counts: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, dict[str, list[float]]]:
    """Paired class-then-sample bootstrap, stratified by canonical group."""
    from scripts.base_trainer import compute_class_groups

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    groups = compute_class_groups(class_counts)
    indices_by_class = [np.flatnonzero(labels == class_id) for class_id in range(len(class_counts))]
    if any(len(indices) == 0 for indices in indices_by_class):
        raise ValueError("outer evaluation is missing a class")
    methods = tuple(predictions)
    correct = {key: (predictions[key] == labels).astype(np.float64) for key in methods}
    diffs = {key: np.empty((replicates, 3), dtype=np.float64) for key in methods if key != "uniform_logit"}
    if "candidate" in predictions:
        diffs.update({
            f"candidate_minus_{key}": np.empty((replicates, 3), dtype=np.float64)
            for key in methods if key not in ("uniform_logit", "candidate")
        })
    for replicate in range(replicates):
        per_group: dict[str, dict[str, list[float]]] = {
            key: {group: [] for group in ("head", "medium", "tail")} for key in methods
        }
        accuracy_sum = {key: 0.0 for key in methods}
        sampled_count = 0
        for group, classes in groups.items():
            sampled_classes = rng.choice(classes, size=len(classes), replace=True)
            for class_id in sampled_classes:
                class_rows = indices_by_class[int(class_id)]
                sampled_rows = rng.choice(class_rows, size=len(class_rows), replace=True)
                sampled_count += len(sampled_rows)
                for key in methods:
                    score = float(correct[key][sampled_rows].mean())
                    per_group[key][group].append(score)
                    accuracy_sum[key] += score * len(sampled_rows)
        metric = {
            key: np.array([
                sum(np.mean(per_group[key][group]) * len(groups[group]) for group in groups) / len(class_counts),
                np.mean(per_group[key]["tail"]),
                accuracy_sum[key] / sampled_count,
            ]) for key in methods
        }
        for key in methods:
            if key != "uniform_logit":
                diffs[key][replicate] = metric[key] - metric["uniform_logit"]
            if key not in ("uniform_logit", "candidate") and "candidate" in predictions:
                diffs[f"candidate_minus_{key}"][replicate] = metric["candidate"] - metric[key]
    return {
        key: {
            metric_name: np.quantile(values[:, axis], [0.025, 0.975]).tolist()
            for axis, metric_name in enumerate(("balanced_accuracy", "tail_accuracy", "ordinary_accuracy"))
        } for key, values in diffs.items()
    }


def run_outer(
    *,
    data_root: str | Path = "data",
    artifact_root: str | Path = "artifacts/oof",
    development_directory: str | Path = "artifacts/oof/ridge_sinkhorn_v3",
) -> dict:
    root = Path(development_directory)
    reader, writer = ArtifactReader(), ImmutableArtifactWriter()
    decision_path = root / "selection_v1" / "decision.json"
    model_path = root / "locked_outer_v1" / "model.json"
    decision = reader.read_json(decision_path)
    model = reader.read_json(model_path)
    config_path = root / "allocation_v1" / "config.json"
    if model["development_config_sha256"] != reader.sha256_file(config_path):
        raise ValueError("development configuration changed after outer lock")
    for name, digest in reader.read_json(config_path)["source_code_sha256"].items():
        if reader.sha256_file(Path(__file__).resolve().parents[1] / name) != digest:
            raise ValueError(f"locked development source code changed: {name}")
    if decision["selected_candidate_id"] != model["candidate_id"]:
        raise ValueError("locked model and immutable selection decision disagree")
    if model["selection_decision_sha256"] != reader.sha256_file(decision_path):
        raise ValueError("selection decision changed after outer lock")
    arrays_path = root / "locked_outer_v1" / "model_arrays.npz"
    if model["model_arrays_sha256"] != reader.sha256_file(arrays_path):
        raise ValueError("locked model arrays changed after selection")
    for source in model["source_files"].values():
        if reader.sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("development source artifact changed after selection")
    arrays = reader.read_npz(arrays_path)
    manager = NestedOOFFoldManager.from_canonical_training_data(data_root, seed=42)
    if model["canonical_training_index_sha256"] != manager.canonical_training_index_sha256:
        raise ValueError("locked model targets a different canonical population")
    if set(arrays["fit_sample_indices"].tolist()) != set(manager.outer_fold(0).expert_training_indices):
        raise ValueError("locked model was not refitted on exactly the four inner folds")
    # The only outer read follows verification of the immutable decision/model.
    sample_ids, labels, logits, provenance = _load_outer_predictions(manager, artifact_root)
    routed = _locked_weights(logits, model, arrays)
    metrics, prediction = evaluate(logits, labels, routed, np.asarray(manager.canonical_class_counts))
    predictions = _reference_predictions(logits)
    predictions["candidate"] = prediction
    metric_rows = {
        key: classification_metrics(labels, values, np.asarray(manager.canonical_class_counts))
        for key, values in predictions.items()
    }
    fixed = ("fixed_006", "fixed_007", "fixed_010", "fixed_011", "uniform_probability", "uniform_without_ce")
    baseline = metric_rows["uniform_logit"]
    passes = (
        metrics["balanced_accuracy"] > baseline["balanced_accuracy"]
        and metrics["tail_accuracy"] > baseline["tail_accuracy"]
        and all(
            not (
                metric_rows[key]["balanced_accuracy"] >= metrics["balanced_accuracy"]
                and metric_rows[key]["tail_accuracy"] >= metrics["tail_accuracy"]
            ) for key in fixed
        )
    )
    intervals = hierarchical_paired_intervals(
        labels, predictions, np.asarray(manager.canonical_class_counts)
    )
    output = root / "outer_evaluation_v1"
    report = {
        "schema_version": "ridge_sinkhorn_outer_evaluation.v1",
        "candidate_id": model["candidate_id"],
        "outer_fold": 0, "training_seed": 78,
        "sample_count": len(sample_ids),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "type": "stratified class/sample hierarchical paired"},
        "metrics": metric_rows,
        "paired_differences": {
            key: {
                metric: value[metric] - baseline[metric]
                for metric in ("ordinary_accuracy", "balanced_accuracy", "head_accuracy", "medium_accuracy", "tail_accuracy")
            } for key, value in metric_rows.items() if key != "uniform_logit"
        },
        "candidate_paired_differences_vs_references": {
            key: {
                metric: metrics[metric] - metric_rows[key][metric]
                for metric in ("ordinary_accuracy", "balanced_accuracy", "head_accuracy", "medium_accuracy", "tail_accuracy")
            } for key in metric_rows if key != "candidate"
        },
        "paired_intervals_vs_uniform": {
            key: value for key, value in intervals.items() if not key.startswith("candidate_minus_")
        },
        "candidate_paired_intervals_vs_references": {
            key.removeprefix("candidate_minus_"): value
            for key, value in intervals.items() if key.startswith("candidate_minus_")
        },
        "expansion_gate_passed": passes,
        "outer_source_artifacts": provenance,
        "locked_model_sha256": reader.sha256_file(model_path),
        "original_test_set_accessed": False,
    }
    writer.write_json_once(output / "results.json", report)
    writer.write_npz_once(output / "predictions.npz", {
        "sample_indices": sample_ids,
        "labels": labels,
        "candidate_weights": routed,
        **{f"prediction_{key}": value.astype(np.int16) for key, value in predictions.items()},
    })
    return report
