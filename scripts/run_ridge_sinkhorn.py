#!/usr/bin/env python3
"""Run the staged, CPU-only Ridge/Sinkhorn development selection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from data.nested_oof import NestedOOFFoldManager  # noqa: E402
from scripts.analysis import ArtifactReader, ImmutableArtifactWriter  # noqa: E402
from scripts.ridge_sinkhorn_data import load_ridge_sinkhorn_development_data  # noqa: E402
from scripts.ridge_sinkhorn_model import fit_residual_ridge  # noqa: E402
from scripts.ridge_sinkhorn_study import (  # noqa: E402
    ANCHOR, ALPHAS, GAMMAS, PENALTIES, SCALES, RHOS,
    StudyArrays, allocation_stage, fixed_references, residual_stage, selected_diagnostics,
)
from scripts.task3f_ridge import compute_sample_weights  # noqa: E402


SCHEMA = "ridge_sinkhorn_development.v3"
SOURCE_FILES = (
    "scripts/ridge_sinkhorn_data.py",
    "scripts/ridge_sinkhorn_model.py",
    "scripts/ridge_sinkhorn_oracle.py",
    "scripts/ridge_sinkhorn_ot.py",
    "scripts/ridge_sinkhorn_selection.py",
    "scripts/ridge_sinkhorn_study.py",
    "scripts/ridge_sinkhorn_outer.py",
    "scripts/run_ridge_sinkhorn.py",
    "scripts/run_ridge_sinkhorn_outer.py",
    "scripts/task3f_ridge.py",
    "scripts/task3e_fixed.py",
    "scripts/task3c_oof.py",
    "scripts/oof_pipeline.py",
    "scripts/base_trainer.py",
    "data/nested_oof.py",
)


def _write_stage(
    directory: Path,
    stage: dict,
    *,
    config: dict,
    writer: ImmutableArtifactWriter,
) -> None:
    rows = stage["rows"]
    identifiers = [row["candidate_id"] for row in rows]
    arrays = {
        "sample_indices": np.asarray(config["selection_sample_indices"], dtype=np.int64),
        "predictions": np.stack([stage["predictions"][key] for key in identifiers]).astype(np.int16),
        "weights": np.stack([stage["weights"][key] for key in identifiers]),
    }
    payload = {key: value for key, value in stage.items() if key not in {"predictions", "weights", "highlighted_fit_kernel", "_oracle_targets"}}
    payload["candidate_ids_in_array_order"] = identifiers
    payload["schema_version"] = SCHEMA
    writer.write_json_once(directory / "config.json", config)
    writer.write_json_once(directory / "results.json", payload)
    writer.write_npz_once(directory / "selection_arrays.npz", arrays)


def _selected_row(stage: dict, identifier: str) -> dict:
    return next(row for row in stage["rows"] if row["candidate_id"] == identifier)


def _refit_locked_model(data: StudyArrays, selected: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Refit the one selected family on all inner OOF rows after method lock."""
    from scripts.ridge_sinkhorn_oracle import compute_oracle_targets, residuals_to_weights, smooth_positive_kernel
    from scripts.ridge_sinkhorn_ot import DEFAULT_Q, fit_frozen_dual_prices

    logits = np.concatenate((data.fit_logits, data.selection_logits))
    labels = np.concatenate((data.fit_labels, data.selection_labels))
    indices = np.concatenate((data.fit_indices, data.selection_indices))
    penalty, gamma, alpha, scale = (
        float(selected[key]) for key in ("penalty", "gamma", "alpha", "scale")
    )
    target = compute_oracle_targets(logits, labels, ANCHOR, penalty).residuals
    weights = compute_sample_weights(labels, gamma)
    fit = fit_residual_ridge(
        logits, target, weights, feature_set="confidence_only", alpha=alpha, gamma=gamma
    )
    arrays = {
        "fit_sample_indices": indices.astype(np.int64),
        "scaler_mean": np.asarray(fit.scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(fit.scaler.scale_, dtype=np.float64),
        "ridge_coef": np.asarray(fit.model.coef_, dtype=np.float64),
        "ridge_intercept": np.asarray(fit.model.intercept_, dtype=np.float64),
        "anchor": ANCHOR.copy(),
    }
    fitted_price_diagnostics = None
    if selected["ot"]:
        fit_routed = residuals_to_weights(fit.predict_residual(logits), ANCHOR, scale)
        fit_kernel = smooth_positive_kernel(fit_routed)
        fitted = fit_frozen_dual_prices(fit_kernel, float(selected["rho"]), DEFAULT_Q)
        arrays["log_prices"] = np.asarray(fitted.log_prices, dtype=np.float64)
        fitted_price_diagnostics = fitted.diagnostics
    metadata = {
        "candidate_id": selected["candidate_id"],
        "family": selected["kind"],
        "feature_set": "confidence_only",
        "penalty": penalty, "gamma": gamma, "alpha": alpha, "scale": scale,
        "rho": selected.get("rho"),
        "prior": list(DEFAULT_Q) if selected["ot"] else None,
        "ot": bool(selected["ot"]),
        "fitted_price_diagnostics": fitted_price_diagnostics,
        "refit_role": "all_four_inner_oof_folds_after_selection_lock",
        "refit_sample_count": len(indices),
        "inference_inputs": "four expert original logits only",
        "inference_output": "weighted original-logit soft mixture",
    }
    return metadata, arrays


def run_development(
    *,
    data_root: str | Path = "data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    output_directory: str | Path = "artifacts/oof/ridge_sinkhorn_v3",
) -> dict:
    manager = NestedOOFFoldManager.from_canonical_training_data(data_root, seed=42)
    source = load_ridge_sinkhorn_development_data(oof_directory, manager)
    data = StudyArrays(
        fit_logits=source.fit.logits,
        fit_labels=source.fit.labels,
        fit_indices=source.fit.sample_indices,
        selection_logits=source.selection.logits,
        selection_labels=source.selection.labels,
        selection_indices=source.selection.sample_indices,
        canonical_class_counts=np.asarray(manager.canonical_class_counts),
    )
    output = Path(output_directory)
    if output.resolve() == Path(oof_directory).resolve():
        raise ValueError("output directory must differ from the immutable input directory")
    reader, writer = ArtifactReader(), ImmutableArtifactWriter()
    code_hashes = {
        name: reader.sha256_file(Path(_PROJECT_ROOT) / name)
        for name in SOURCE_FILES
    }
    references = fixed_references(data)
    config = {
        "schema_version": SCHEMA,
        "source_git_commit": reader.git_commit(_PROJECT_ROOT, required=True),
        "source_code_sha256": code_hashes,
        "source_files": {key: {"path": str(value["path"]), "sha256": value["sha256"]} for key, value in source.source_files.items()},
        "canonical_training_index_sha256": manager.canonical_training_index_sha256,
        "expert_order": list(manager.expert_order),
        "outer_fold": 0, "fit_inner_folds": [1, 2, 3], "selection_inner_folds": [0],
        "fit_sample_count": len(data.fit_labels), "selection_sample_count": len(data.selection_labels),
        "selection_sample_indices": data.selection_indices.tolist(),
        "original_test_set_accessed": False,
        "selection_fold_prior_exposure": "Task 3C previously inspected inner fold 0 descriptively",
        "grid": {"alpha": ALPHAS, "gamma": GAMMAS, "penalty": PENALTIES, "scale": SCALES, "rho": RHOS},
        "residual_anchor": ANCHOR.tolist(),
        "logit_semantics": "weighted original logits; all four experts",
    }
    allocation = allocation_stage(data)
    _write_stage(output / "allocation_v1", allocation, config=config, writer=writer)
    residual = residual_stage(data, allocation["selected_rho"], references)
    _write_stage(output / "residual_v1", residual, config=config, writer=writer)
    writer.write_json_once(output / "selection_v1" / "references.json", references)
    writer.write_json_once(output / "selection_v1" / "decision.json", residual["selection"])
    selected_id = residual["selection"]["selected_candidate_id"]
    if selected_id is None:
        return {"selected_candidate_id": None, "outer_training_needed": False, "output_directory": str(output)}
    selected = _selected_row(residual, selected_id)
    diagnostics = selected_diagnostics(
        data, selected, residual["weights"][selected_id], residual["_oracle_targets"],
    )
    _write_stage(output / "selected_diagnostics_v1", diagnostics, config=config, writer=writer)
    # The immutable selection decision above is already installed before the
    # labels of selection rows are admitted to this final all-inner refit.
    metadata, arrays = _refit_locked_model(data, selected)
    model_arrays_path = output / "locked_outer_v1" / "model_arrays.npz"
    writer.write_npz_once(model_arrays_path, arrays)
    writer.write_json_once(output / "locked_outer_v1" / "model.json", {
        **metadata,
        "schema_version": "ridge_sinkhorn_locked_outer.v1",
        "selection_decision_sha256": reader.sha256_file(output / "selection_v1" / "decision.json"),
        "development_config_sha256": reader.sha256_file(output / "allocation_v1" / "config.json"),
        "model_arrays_sha256": reader.sha256_file(model_arrays_path),
        "source_files": {key: {"path": str(value["path"]), "sha256": value["sha256"]} for key, value in source.source_files.items()},
        "canonical_training_index_sha256": manager.canonical_training_index_sha256,
        "expert_order": list(manager.expert_order),
    })
    return {"selected_candidate_id": selected_id, "outer_training_needed": True, "output_directory": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--oof-directory", default="artifacts/oof/task3c_oof")
    parser.add_argument("--output-directory", default="artifacts/oof/ridge_sinkhorn_v3")
    args = parser.parse_args(argv)
    try:
        print(run_development(data_root=args.data_root, oof_directory=args.oof_directory, output_directory=args.output_directory))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ridge/Sinkhorn development error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
