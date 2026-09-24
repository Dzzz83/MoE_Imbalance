"""Staged array-only development study for Ridge scores and OT allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.ridge_sinkhorn_model import fit_residual_ridge
from scripts.ridge_sinkhorn_selection import select_candidate
from scripts.task3f_ridge import (
    FIXED_REFERENCE_UNITS,
    HISTORICAL_NO_CE_WEIGHTS,
    classification_metrics,
    combine_weighted_logits,
    compute_sample_weights,
    fit_global_score_control,
    fit_ridge_router,
    scores_to_weights,
    _weighted_probability_predictions,
)


from scripts.ridge_sinkhorn_oracle import FIXED_007_ANCHOR


ANCHOR = np.asarray(FIXED_007_ANCHOR, dtype=np.float64)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMAS = (0.0, 0.5, 1.0)
PENALTIES = (0.1, 1.0, 10.0)
SCALES = (0.25, 0.5, 0.75, 1.0)
RHOS = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class StudyArrays:
    fit_logits: np.ndarray
    fit_labels: np.ndarray
    fit_indices: np.ndarray
    selection_logits: np.ndarray
    selection_labels: np.ndarray
    selection_indices: np.ndarray
    canonical_class_counts: np.ndarray

    def __post_init__(self) -> None:
        if self.fit_logits.ndim != 3 or self.selection_logits.ndim != 3:
            raise ValueError("logits must have (rows, experts, classes) shape")
        if self.fit_logits.shape[1:] != self.selection_logits.shape[1:] or self.fit_logits.shape[1] != 4:
            raise ValueError("expert axes must match the frozen four-expert order")
        for name in ("fit", "selection"):
            logits = getattr(self, name + "_logits")
            labels = getattr(self, name + "_labels")
            indices = getattr(self, name + "_indices")
            if labels.shape != (len(logits),) or indices.shape != labels.shape:
                raise ValueError(f"{name} rows, labels and IDs are misaligned")
            if not np.isfinite(logits).all() or not np.isfinite(labels).all():
                raise ValueError(f"{name} rows contain non-finite values")
        if set(self.fit_indices.tolist()) & set(self.selection_indices.tolist()):
            raise ValueError("router-fit and selection sample IDs overlap")
        if self.canonical_class_counts.shape != (self.fit_logits.shape[2],):
            raise ValueError("canonical class counts have the wrong shape")


def evaluate(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray, counts: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    prediction = combine_weighted_logits(logits, weights).argmax(axis=1)
    return classification_metrics(labels, prediction, counts), prediction


def weight_semantics_diagnostics(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray, counts: np.ndarray,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Report probability, top-2, and hard routing without selection eligibility."""
    rows, predictions, routed_weights = [], {}, {}
    top = np.argsort(-weights, axis=1, kind="stable")
    for name, candidate_weights in (
        ("top2", np.where(
            np.arange(4)[None, :] == top[:, :1],
            weights, np.where(np.arange(4)[None, :] == top[:, 1:2], weights, 0.0)
        )),
        ("hard", np.eye(4)[top[:, 0]]),
    ):
        candidate_weights = candidate_weights / candidate_weights.sum(axis=1, keepdims=True)
        identifier = f"{prefix}_{name}"
        metrics, prediction = evaluate(logits, labels, candidate_weights, counts)
        rows.append({"candidate_id": identifier, "kind": "weight_semantics_diagnostic", "metrics": metrics, "selection_eligible": False})
        predictions[identifier], routed_weights[identifier] = prediction, candidate_weights
    identifier = f"{prefix}_probability"
    pred = _weighted_probability_predictions(logits, weights)
    rows.append({
        "candidate_id": identifier, "kind": "weight_semantics_diagnostic",
        "metrics": classification_metrics(labels, pred, counts), "selection_eligible": False,
    })
    predictions[identifier], routed_weights[identifier] = pred, weights
    return rows, predictions, routed_weights


def fixed_references(data: StudyArrays) -> dict[str, dict[str, Any]]:
    logits, labels, counts = data.selection_logits, data.selection_labels, data.canonical_class_counts
    uniform = np.full(4, 0.25)
    references = {}
    references["uniform_logit"], _ = evaluate(logits, labels, uniform, counts)
    prob_prediction = _weighted_probability_predictions(logits, uniform)
    references["uniform_probability"] = classification_metrics(labels, prob_prediction, counts)
    references["uniform_without_ce"], _ = evaluate(logits, labels, np.asarray(HISTORICAL_NO_CE_WEIGHTS), counts)
    for identifier, units in FIXED_REFERENCE_UNITS.items():
        references[identifier], _ = evaluate(logits, labels, np.asarray(units, dtype=float) / 4, counts)
    return references


def allocation_stage(data: StudyArrays) -> dict[str, Any]:
    """Study the same highlighted Ridge kernel under four allocation rules."""
    from scripts.ridge_sinkhorn_ot import (
        DEFAULT_Q, balanced_sinkhorn, fit_frozen_dual_prices,
        relaxed_sinkhorn, apply_log_bias,
    )

    fit = fit_ridge_router(data.fit_logits, data.fit_labels, "confidence_only", 1000.0, 1.0)
    fit_kernel = scores_to_weights(fit.predict_scores(data.fit_logits), 2.0, 0.75)
    selection_kernel = scores_to_weights(fit.predict_scores(data.selection_logits), 2.0, 0.75)
    no_ot, no_ot_prediction = evaluate(data.selection_logits, data.selection_labels, selection_kernel, data.canonical_class_counts)
    rows = [{"candidate_id": "highlighted_no_ot", "kind": "no_ot", "metrics": no_ot}]
    predictions = {"highlighted_no_ot": no_ot_prediction}
    weights = {"highlighted_no_ot": selection_kernel}
    diagnostic_rows, diagnostic_predictions, diagnostic_weights = weight_semantics_diagnostics(
        data.selection_logits, data.selection_labels, selection_kernel,
        data.canonical_class_counts, "highlighted",
    )
    rows.extend(diagnostic_rows)
    predictions.update(diagnostic_predictions)
    weights.update(diagnostic_weights)
    for rho in RHOS:
        token = str(rho).replace(".", "p")
        fitted = fit_frozen_dual_prices(fit_kernel, rho, DEFAULT_Q)
        frozen_weights = fitted.apply(selection_kernel)
        metrics, pred = evaluate(data.selection_logits, data.selection_labels, frozen_weights, data.canonical_class_counts)
        identifier = f"highlighted_frozen_rho_{token}"
        rows.append({"candidate_id": identifier, "kind": "frozen_price", "rho": rho, "metrics": metrics, "diagnostics": fitted.diagnostics})
        predictions[identifier], weights[identifier] = pred, frozen_weights
        live = relaxed_sinkhorn(selection_kernel, rho, DEFAULT_Q)
        live_id = f"highlighted_batch_rho_{token}"
        live_metrics, live_pred = evaluate(data.selection_logits, data.selection_labels, live.weights, data.canonical_class_counts)
        rows.append({"candidate_id": live_id, "kind": "transductive_diagnostic", "rho": rho, "metrics": live_metrics, "diagnostics": live.diagnostics})
        predictions[live_id], weights[live_id] = live_pred, live.weights
    balanced = balanced_sinkhorn(selection_kernel)
    balanced_metrics, balanced_pred = evaluate(data.selection_logits, data.selection_labels, balanced.weights, data.canonical_class_counts)
    rows.append({"candidate_id": "highlighted_balanced_batch", "kind": "transductive_diagnostic", "metrics": balanced_metrics, "diagnostics": balanced.diagnostics})
    predictions["highlighted_balanced_batch"], weights["highlighted_balanced_batch"] = balanced_pred, balanced.weights
    prior_weights = apply_log_bias(selection_kernel, np.log(np.asarray(DEFAULT_Q) / 0.25))
    prior_metrics, prior_pred = evaluate(data.selection_logits, data.selection_labels, prior_weights, data.canonical_class_counts)
    rows.append({"candidate_id": "highlighted_prior_bias", "kind": "global_bias_control", "metrics": prior_metrics})
    predictions["highlighted_prior_bias"], weights["highlighted_prior_bias"] = prior_pred, prior_weights
    # Matched alternative score sources are interpretation controls only.
    for feature_set in ("full_13",):
        alt = fit_ridge_router(data.fit_logits, data.fit_labels, feature_set, 1000.0, 1.0)
        alt_weights = scores_to_weights(alt.predict_scores(data.selection_logits), 2.0, 0.75)
        alt_metrics, alt_pred = evaluate(data.selection_logits, data.selection_labels, alt_weights, data.canonical_class_counts)
        rows.append({"candidate_id": "full13_no_ot", "kind": "score_diagnostic", "metrics": alt_metrics})
        predictions["full13_no_ot"], weights["full13_no_ot"] = alt_pred, alt_weights
    global_fit = fit_global_score_control(data.fit_logits, data.fit_labels, 1.0)
    global_weights = scores_to_weights(global_fit.predict_scores(len(data.selection_logits)), 2.0, 0.75)
    global_metrics, global_pred = evaluate(data.selection_logits, data.selection_labels, global_weights, data.canonical_class_counts)
    rows.append({"candidate_id": "global_score_no_ot", "kind": "score_diagnostic", "metrics": global_metrics})
    predictions["global_score_no_ot"], weights["global_score_no_ot"] = global_pred, global_weights
    eligible = [
        row for row in rows
        if row["kind"] == "frozen_price"
        and row["metrics"]["balanced_accuracy"] > no_ot["balanced_accuracy"]
        and row["metrics"]["tail_accuracy"] > no_ot["tail_accuracy"]
    ]
    eligible.sort(key=lambda row: (-row["metrics"]["balanced_accuracy"], -row["metrics"]["tail_accuracy"], row["candidate_id"]))
    return {
        "rows": rows, "predictions": predictions, "weights": weights,
        "highlighted_fit_kernel": fit_kernel,
        "selected_rho": eligible[0]["rho"] if eligible else None,
        "frozen_price_gate_passed": bool(eligible),
        "prior": np.asarray(DEFAULT_Q).tolist(),
    }


def residual_stage(data: StudyArrays, selected_rho: float | None, references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the frozen residual grid once on selection inner fold 0."""
    from scripts.ridge_sinkhorn_oracle import compute_oracle_targets, residuals_to_weights, smooth_positive_kernel
    from scripts.ridge_sinkhorn_ot import DEFAULT_Q, fit_frozen_dual_prices

    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    targets = {}
    for penalty in PENALTIES:
        targets[penalty] = compute_oracle_targets(data.fit_logits, data.fit_labels, ANCHOR, penalty)
    for penalty in PENALTIES:
        target = targets[penalty].residuals
        for gamma in GAMMAS:
            sample_weights = compute_sample_weights(data.fit_labels, gamma)
            for alpha in ALPHAS:
                fit = fit_residual_ridge(
                    data.fit_logits, target, sample_weights,
                    feature_set="confidence_only", alpha=alpha, gamma=gamma,
                )
                fit_residual = fit.predict_residual(data.fit_logits)
                selection_residual = fit.predict_residual(data.selection_logits)
                for scale in SCALES:
                    identifier = f"residual_p{penalty:g}_g{gamma:g}_a{alpha:g}_s{scale:g}"
                    fit_weights = residuals_to_weights(fit_residual, ANCHOR, scale)
                    selection_weights = residuals_to_weights(selection_residual, ANCHOR, scale)
                    metrics, pred = evaluate(data.selection_logits, data.selection_labels, selection_weights, data.canonical_class_counts)
                    rows.append({"candidate_id": identifier, "kind": "residual_no_ot", "ot": False, "metrics": metrics, "penalty": penalty, "gamma": gamma, "alpha": alpha, "scale": scale, "selection_eligible": True})
                    predictions[identifier], weights[identifier] = pred, selection_weights
                    if selected_rho is not None:
                        fit_kernel = smooth_positive_kernel(fit_weights)
                        selection_kernel = smooth_positive_kernel(selection_weights)
                        prices = fit_frozen_dual_prices(fit_kernel, selected_rho, DEFAULT_Q)
                        routed = prices.apply(selection_kernel)
                        ot_metrics, ot_pred = evaluate(data.selection_logits, data.selection_labels, routed, data.canonical_class_counts)
                        ot_id = identifier + f"_frozen_rho{selected_rho:g}"
                        rows.append({"candidate_id": ot_id, "kind": "residual_frozen_price", "ot": True, "metrics": ot_metrics, "penalty": penalty, "gamma": gamma, "alpha": alpha, "scale": scale, "rho": selected_rho, "selection_eligible": True, "diagnostics": prices.diagnostics})
                        predictions[ot_id], weights[ot_id] = ot_pred, routed
    decision = select_candidate(rows, references)
    return {
        "rows": rows, "predictions": predictions, "weights": weights,
        "oracle_diagnostics": {str(k): value.diagnostics for k, value in targets.items()},
        "_oracle_targets": {k: value.residuals for k, value in targets.items()},
        "selection": decision,
    }


def selected_diagnostics(
    data: StudyArrays, selected: dict[str, Any], selected_weights: np.ndarray,
    cached_targets: dict[float, np.ndarray],
) -> dict[str, Any]:
    """Secondary checks at the one locked configuration; none enter selection."""
    from scripts.ridge_sinkhorn_oracle import compute_oracle_targets, residuals_to_weights, smooth_positive_kernel
    from scripts.ridge_sinkhorn_ot import DEFAULT_Q, fit_frozen_dual_prices, relaxed_sinkhorn

    rows, predictions, weights = weight_semantics_diagnostics(
        data.selection_logits, data.selection_labels, selected_weights,
        data.canonical_class_counts, "selected",
    )
    penalty, gamma, alpha, scale = (float(selected[key]) for key in ("penalty", "gamma", "alpha", "scale"))
    sample_weights = compute_sample_weights(data.fit_labels, gamma)
    target = cached_targets[penalty]
    primary_fit = fit_residual_ridge(
        data.fit_logits, target, sample_weights,
        feature_set="confidence_only", alpha=alpha, gamma=gamma,
    )
    primary_no_ot = residuals_to_weights(
        primary_fit.predict_residual(data.selection_logits), ANCHOR, scale,
    )
    for feature_set, margin, suffix in (
        ("full_13", 0.0, "full13"),
        ("confidence_only", 1.0, "margin1"),
    ):
        if margin == 1.0:
            target = compute_oracle_targets(data.fit_logits, data.fit_labels, ANCHOR, penalty, margin=1.0).residuals
        fit = fit_residual_ridge(
            data.fit_logits, target, sample_weights,
            feature_set=feature_set, alpha=alpha, gamma=gamma,
        )
        routed = residuals_to_weights(fit.predict_residual(data.selection_logits), ANCHOR, scale)
        metrics, prediction = evaluate(data.selection_logits, data.selection_labels, routed, data.canonical_class_counts)
        identifier = f"selected_{suffix}"
        rows.append({"candidate_id": identifier, "kind": "target_or_feature_sensitivity", "metrics": metrics, "selection_eligible": False})
        predictions[identifier], weights[identifier] = prediction, routed
    smoothed = smooth_positive_kernel(primary_no_ot)
    smooth_metrics, smooth_prediction = evaluate(
        data.selection_logits, data.selection_labels, smoothed, data.canonical_class_counts
    )
    rows.append({"candidate_id": "selected_smoothing", "kind": "numerical_control", "metrics": smooth_metrics, "selection_eligible": False})
    predictions["selected_smoothing"], weights["selected_smoothing"] = smooth_prediction, smoothed
    rho = float(selected["rho"]) if selected["ot"] else 1.0
    live = relaxed_sinkhorn(smoothed, rho, DEFAULT_Q)
    live_metrics, live_prediction = evaluate(
        data.selection_logits, data.selection_labels, live.weights, data.canonical_class_counts
    )
    rows.append({"candidate_id": "selected_batch_ot", "kind": "transductive_diagnostic", "rho": rho, "metrics": live_metrics, "diagnostics": live.diagnostics, "selection_eligible": False})
    predictions["selected_batch_ot"], weights["selected_batch_ot"] = live_prediction, live.weights
    if not selected["ot"]:
        # If Stage B failed, this is the sole prespecified strength-1 frozen
        # residual+OT check. It cannot become a selection candidate.
        fit_kernel = smooth_positive_kernel(
            residuals_to_weights(primary_fit.predict_residual(data.fit_logits), ANCHOR, scale)
        )
        price = fit_frozen_dual_prices(fit_kernel, 1.0, DEFAULT_Q)
        routed = price.apply(smoothed)
        metrics, prediction = evaluate(
            data.selection_logits, data.selection_labels, routed, data.canonical_class_counts
        )
        rows.append({"candidate_id": "selected_frozen_ot_strength1", "kind": "ineligible_diagnostic", "rho": 1.0, "metrics": metrics, "diagnostics": price.diagnostics, "selection_eligible": False})
        predictions["selected_frozen_ot_strength1"], weights["selected_frozen_ot_strength1"] = prediction, routed
    return {"rows": rows, "predictions": predictions, "weights": weights}
