"""Orchestrate validated inner diagnostics without fitting or selecting models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from expert_method.analysis import assemble_inner_dataset
from expert_method import cli as study_cli
from expert_method.ridge_sinkhorn.three_seed_study import (
    ANCHOR,
    METHOD_IDS,
    InferenceBatch,
    StudyEvaluator,
)
from scripts.base_trainer import compute_class_groups
from scripts.ridge_sinkhorn_oracle import smooth_positive_kernel

from .aggregation import (
    add_profile_aggregates,
    add_profile_rows,
    add_stability_rows,
    aggregate_pair_tables,
    append_metric_rows,
    add_sample_entropy_aggregates,
    descriptive_metrics,
    diagnostic_highlights,
    METRIC_NAMES,
)
from .artifacts import DiagnosticsInputs, InnerArtifactReader, STUDY_ID
from .complementarity import analyze_complementarity, classify_predictions
from .contracts import DiagnosticsError, EXPERT_NAMES, json_safe
from .reporting import TABLE_FILES, assert_no_output_symlinks, write_outputs
from .router import analyze_router_weights
from .sinkhorn import sinkhorn_stage_diagnostics
from .prior import (
    analyze_prior_relation,
    correctness_transition,
    mean_weight_counterfactual,
    scoped_correctness_transitions,
)
from .stability import selective_gate


_ADAPTIVE_METHODS = tuple(METHOD_IDS)
_SINKHORN_METHODS = (
    "contribution_ridge_sinkhorn", "residual_ridge_sinkhorn",
)


def _seed_outer(seed: int, outer: int) -> dict[str, int]:
    return {"training_seed": int(seed), "outer_fold_id": int(outer)}


class StudyDiagnosticsRunner:
    """Coordinate read-only analysis of 15 validated inner seed/fold pairs."""

    def __init__(
        self,
        reader: InnerArtifactReader,
        *,
        output_root: str | Path,
        command_arguments: Sequence[str] = (),
        make_figures: bool = True,
    ) -> None:
        self.reader = reader
        self.output_root = Path(output_root).expanduser()
        self.command_arguments = tuple(str(argument) for argument in command_arguments)
        self.make_figures = bool(make_figures)

    def _validate_output_location(self) -> Path:
        """Reject outputs that overlap an input root in either direction."""
        assert_no_output_symlinks(self.output_root, make_figures=self.make_figures)
        output = self.output_root.resolve()
        input_roots = [self.reader.artifact_root, *self.reader.reuse_roots.values()]
        for input_path in input_roots:
            source = input_path.resolve()
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                raise DiagnosticsError(
                    f"output root overlaps input root: output={output}, input={source}"
                )
        return output

    @staticmethod
    def _analysis_identity() -> dict[str, Any]:
        root = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD^{commit}"],
                cwd=root, check=True, capture_output=True, text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root, check=True, capture_output=True, text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DiagnosticsError("cannot identify analysis source Git commit and dirty state") from exc
        if len(commit) not in {40, 64}:
            raise DiagnosticsError("analysis source Git commit is malformed")

        tracked_code = (
            "expert_method/diagnostics",
            "expert_method/analysis.py",
            "expert_method/cli.py",
            "expert_method/config.py",
            "expert_method/oof/pipeline.py",
            "expert_method/ridge_sinkhorn/matrix.py",
            "expert_method/ridge_sinkhorn/three_seed_study.py",
            "scripts/analysis/artifacts.py",
            "scripts/base_trainer.py",
            "scripts/config.py",
            "scripts/expert_diagnostics.py",
            "scripts/ridge_sinkhorn_model.py",
            "scripts/ridge_sinkhorn_oracle.py",
            "scripts/ridge_sinkhorn_ot.py",
            "scripts/task3f_ridge.py",
            "data/nested_oof.py",
        )
        source_paths = [
            root / relative for relative in tracked_code
            if (root / relative).is_file()
        ]
        diagnostics_dir = root / "expert_method" / "diagnostics"
        source_paths.extend(sorted(
            path for path in diagnostics_dir.glob("*.py") if path.is_file()
        ))
        source_hashes = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(source_paths))
        }
        digest = hashlib.sha256()
        for relative, value in sorted(source_hashes.items()):
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(value))
        versions: dict[str, str | None] = {}
        for package in ("numpy", "scipy", "scikit-learn", "torch", "matplotlib", "PyYAML"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {
            "analysis_source_commit": commit,
            "analysis_source_dirty": bool(status.strip()),
            "diagnostics_source_sha256": digest.hexdigest(),
            "diagnostics_source_file_sha256": source_hashes,
            "package_versions": {
                "python": platform.python_version(),
                **versions,
            },
        }

    def validate_only(self) -> dict[str, Any]:
        """Validate identities and serialized inner payloads without metrics or outputs."""
        inputs = self.reader.validate()
        return {
            "valid": True,
            "study_id": STUDY_ID,
            "stage": "inner",
            "validation_counts": dict(inputs.validation_counts),
            "training_source_commit": str(inputs.manifest["source_commit"]),
            "result_snapshot_commit": inputs.snapshot_commit,
            "manifest_sha256": inputs.manifest_sha256,
            "study_config_sha256": self.reader.config.sha256,
            "config_yaml_sha256": inputs.config_file_sha256,
        }

    def run(self, *, limit_pairs: int | None = None) -> dict[str, Any]:
        """Analyze complete data or a deterministic prefix for an explicit smoke run."""
        output = self._validate_output_location()
        inputs = self.reader.validate()
        if limit_pairs is not None and (isinstance(limit_pairs, bool) or limit_pairs < 1):
            raise DiagnosticsError("limit_pairs must be a positive integer")
        expected_pairs = [
            (seed, outer)
            for seed in self.reader.config.seeds
            for outer in self.reader.config.outer_folds
        ]
        selected_pairs = expected_pairs if limit_pairs is None else expected_pairs[:limit_pairs]
        if len(selected_pairs) != len(expected_pairs) and limit_pairs is None:
            raise DiagnosticsError("complete analysis did not select all 15 pairs")

        tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_FILES}
        pair_router_profiles: dict[tuple[int, int], dict[str, dict[int, np.ndarray | None]]] = {}
        pair_router_overall: dict[tuple[int, int], dict[str, np.ndarray | None]] = {}
        pair_router_groups: dict[tuple[int, int], dict[str, dict[str, Mapping[str, Any]]]] = {}
        job_lookup = self.reader.job_lookup
        evaluator = StudyEvaluator(self.reader.config)
        class_counts = np.asarray(inputs.manager.canonical_class_counts, dtype=np.int64)

        for training_seed, outer_fold_id in selected_pairs:
            pair = (training_seed, outer_fold_id)
            dataset = assemble_inner_dataset(
                self.reader,
                job_lookup,
                inputs.manager,
                training_seed,
                outer_fold_id,
            )
            lock = inputs.locks[pair]
            batch = InferenceBatch(dataset.logits, dataset.sample_ids)
            # This is the serialized inference path. No strategy .fit or selector
            # is invoked; the only router parameters come from the saved lock.
            predictions, weights = evaluator.predict_locked_methods(lock, batch)
            complement = analyze_complementarity(dataset.logits, dataset.labels, class_counts)
            pair_key = _seed_outer(training_seed, outer_fold_id)
            self._record_expert_diagnostics(
                tables, pair_key, complement, dataset.logits, dataset.labels,
            )
            router_report = analyze_router_weights(
                dataset.logits,
                dataset.labels,
                weights,
                class_counts,
                predictions_by_method=predictions,
            )
            for association in router_report["class_frequency_correlations"]:
                tables["frequency_associations"].append({
                    **dict(pair_key), "scope": "outer_fold",
                    "source": "saved_router_behavior", **dict(association),
                })
            class_profiles, overall_profiles = add_profile_rows(
                tables,
                pair_key=pair_key,
                router_report=router_report,
            )
            pair_router_profiles[pair] = class_profiles
            pair_router_overall[pair] = overall_profiles
            pair_router_groups[pair] = {
                method: {
                    str(profile["scope_id"]): profile
                    for profile in method_report["group_profiles"]
                }
                for method, method_report in router_report["methods"].items()
            }
            for method, method_report in router_report["methods"].items():
                metrics = classify_predictions(
                    dataset.labels, predictions[method], class_counts,
                )
                append_metric_rows(
                    tables["router_fit_set_metrics"],
                    evidence="saved_router_fit_set_prediction",
                    training_seed=training_seed,
                    outer_fold_id=outer_fold_id,
                    method=method,
                    metrics=metrics,
                )
            self._record_selection_rows(tables, pair_key, lock)
            self._record_prior_and_sinkhorn(
                tables,
                pair_key,
                dataset.logits,
                dataset.labels,
                predictions,
                weights,
                lock,
                class_counts,
            )

        add_profile_aggregates(
            tables, pair_router_overall, pair_router_profiles, pair_router_groups, selected_pairs,
        )
        add_sample_entropy_aggregates(tables, selected_pairs)
        add_stability_rows(tables["stability"], pair_router_profiles, selected_pairs)
        aggregate_pair_tables(tables, selected_pairs)
        descriptive = descriptive_metrics(tables)
        highlights = diagnostic_highlights(tables, expected_pairs)
        complete = len(selected_pairs) == 15
        summary = {
            "study_id": STUDY_ID,
            "stage": "inner",
            "complete_matrix": complete,
            "pairs_analyzed": len(selected_pairs),
            "expected_pair_count": 15,
            "validation_counts": dict(inputs.validation_counts),
            "evidence_labels": {
                "expert_oof": "saved inner OOF predictions",
                "selection_cv": "metrics recorded in immutable method locks",
                "router_fit_set": "saved final router state applied to rows used to fit it",
            },
            "descriptive_metrics": descriptive,
            "diagnostic_highlights": highlights,
            "scientific_limits": [
                "Inner fold development populations overlap across outer folds and seeds.",
                "Stable saved class profiles do not establish a flat hyperparameter optimum.",
                "Saved-state fit-set gains do not establish generalization.",
                "The hard-selection oracle is label-dependent and not an inference method.",
                "No outer prediction, outer label, or original CIFAR-100 test input was loaded.",
            ],
        }
        provenance = self._provenance(inputs, complete=complete, selected_pairs=selected_pairs)
        written = write_outputs(
            output,
            tables=tables,
            summary=summary,
            provenance=provenance,
            make_figures=self.make_figures,
        )
        return {
            "complete_matrix": complete,
            "pairs_analyzed": len(selected_pairs),
            "output_root": str(output),
            "validation_counts": dict(inputs.validation_counts),
            "output_hashes": written,
        }

    @staticmethod
    def _record_expert_diagnostics(
        tables: dict[str, list[dict[str, Any]]],
        pair_key: Mapping[str, int],
        report: Mapping[str, Any],
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        for item in report["per_expert"]:
            metrics = item["canonical_metrics"]
            append_metric_rows(
                tables["expert_oof_metrics"],
                evidence="expert_inner_oof",
                training_seed=pair_key["training_seed"],
                outer_fold_id=pair_key["outer_fold_id"],
                method=str(item["expert"]),
                metrics=metrics,
            )
        summary = report["summary"]
        for metric, values in summary["hard_oracle_gaps_vs_strongest_individual"].items():
            tables["oracle_opportunity"].append(
                {
                    **dict(pair_key),
                    "scope": "outer_fold",
                    "metric": metric,
                    "oracle": values["oracle"],
                    "strongest_individual": values["strongest_individual"],
                    "strongest_expert_names": values["strongest_expert_names"],
                    "oracle_minus_strongest_individual": values["oracle_minus_strongest_individual"],
                    "oracle_accuracy": summary["hard_selection_oracle_metrics"].get(metric),
                    "oracle_policy": "first correct expert in canonical order; falls back to CE when all are wrong",
                }
            )
        for count, frequency in summary["agreement_pattern_counts"].items():
            tables["complementarity"].append(
                {
                    **dict(pair_key),
                    "scope": "outer_fold",
                    "kind": "agreement_pattern",
                    "pattern": count,
                    "count": frequency,
                    "fraction": summary["agreement_pattern_fractions"].get(count),
                }
            )
        correct_by_sample = (np.asarray(logits).argmax(axis=2) == np.asarray(labels)[:, None]).sum(axis=1)
        n_samples = int(len(labels))
        summary_row = {
            **dict(pair_key),
            "scope": "outer_fold",
            "kind": "correctness_summary",
            "any_expert_correct_fraction": summary["any_expert_correct_fraction"],
            "all_experts_wrong_fraction": summary["all_experts_wrong_fraction"],
            "n_samples": n_samples,
        }
        tables["complementarity"].append(summary_row)
        for number_correct in range(5):
            count = int(np.sum(correct_by_sample == number_correct))
            tables["complementarity"].append(
                {
                    **dict(pair_key),
                    "scope": "outer_fold",
                    "kind": "correct_count_distribution",
                    "num_correct_experts": number_correct,
                    "count": count,
                    "fraction": float(count / n_samples),
                }
            )
        expert_correct = np.asarray(logits).argmax(axis=2) == np.asarray(labels)[:, None]
        bitmasks = np.sum(
            expert_correct.astype(np.uint8)
            * (1 << np.arange(len(EXPERT_NAMES), dtype=np.uint8))[None, :],
            axis=1,
        )
        for pattern in range(1 << len(EXPERT_NAMES)):
            count = int(np.count_nonzero(bitmasks == pattern))
            tables["complementarity"].append({
                **dict(pair_key), "scope": "outer_fold",
                "kind": "correctness_pattern", "correct_expert_bitmask": pattern,
                "correct_experts": [
                    name for index, name in enumerate(EXPERT_NAMES)
                    if pattern & (1 << index)
                ],
                "count": count, "fraction": float(count / n_samples),
            })
        for matrix_name in (
            "expert_pairwise_agreement_counts",
            "expert_pairwise_agreement_fractions",
            "expert_pairwise_disagreement_counts",
            "expert_pairwise_disagreement_fractions",
        ):
            matrix = report[matrix_name]
            kind = matrix_name.removeprefix("expert_pairwise_")
            for left, left_name in enumerate(EXPERT_NAMES):
                for right, right_name in enumerate(EXPERT_NAMES):
                    tables["complementarity"].append({
                        **dict(pair_key), "scope": "outer_fold", "kind": "expert_matrix",
                        "matrix": kind, "expert_a": left_name, "expert_b": right_name,
                        "value": matrix[left][right],
                    })
        for row in report["pairwise"]:
            tables["complementarity"].append(
                {**dict(pair_key), "scope": "outer_fold", "kind": "pairwise_correctness", **dict(row)}
            )
        frequency = {int(row["class_id"]): row for row in report["frequency"]}
        for class_row in report["per_class"]:
            recalls = class_row.get("expert_correct_fraction") or {}
            exclusive = class_row.get("exclusive_correct_fraction") or {}
            frequency_row = frequency[int(class_row["class_id"])]
            for expert in EXPERT_NAMES:
                tables["per_class_specialization"].append(
                    {
                        **dict(pair_key),
                        "scope": "outer_fold",
                        "class_id": class_row["class_id"],
                        "training_count": frequency_row["training_count"],
                        "sample_count": class_row["sample_count"],
                        "group": frequency_row["group"],
                        "expert": expert,
                        "class_recall": recalls.get(expert),
                        "exclusive_correct_fraction": exclusive.get(expert),
                        "any_correct_fraction": class_row.get("any_correct_fraction"),
                        "all_wrong_fraction": class_row.get("all_wrong_fraction"),
                        "strongest_experts": class_row.get("strongest_experts"),
                        "strongest_expert_recall": class_row.get("strongest_expert_recall"),
                        "best_second_recall_gap": class_row.get("best_second_recall_gap"),
                        "pairwise_prediction_disagreement_fraction": class_row.get("pairwise_prediction_disagreement_fraction"),
                    }
                )
        for row in report["frequency_associations"]:
            tables["frequency_associations"].append({**dict(pair_key), "scope": "outer_fold", **dict(row)})
        for item in report["group"]:
            tables["complementarity"].append(
                {
                    **dict(pair_key),
                    "scope": "outer_fold",
                    "kind": "canonical_group",
                    "group": item["group"],
                    "class_ids": item["class_ids"],
                    "sample_count": item["sample_count"],
                    "complete_class_coverage": item["complete_class_coverage"],
                    "expert_macro_recall": item["expert_macro_recall"],
                    "any_correct_macro_recall": item["any_correct_macro_recall"],
                    "all_wrong_macro_recall": item["all_wrong_macro_recall"],
                    "exclusive_correct_macro_recall": item["exclusive_correct_macro_recall"],
                }
            )

    @staticmethod
    def _record_selection_rows(
        tables: dict[str, list[dict[str, Any]]], pair_key: Mapping[str, int], lock: Any,
    ) -> None:
        """Write the selected lock's recorded cross-fit metrics without recalculation."""
        for selection in (*lock.selected, *lock.control_selections):
            metrics = dict(selection.metrics)
            for metric in METRIC_NAMES:
                tables["selection_cv_metrics"].append(
                    {
                        **dict(pair_key),
                        "scope": "outer_fold",
                        "evidence": "recorded_selection_used_inner_cv",
                        "method": selection.method_id,
                        "metric": metric,
                        "value": metrics[metric],
                        "configuration": dict(selection.configuration),
                        "candidate_id": selection.candidate_id,
                        "maximin_delta": selection.maximin_delta,
                        "sinkhorn_solver_diagnostics": selection.sinkhorn_diagnostics,
                    }
                )

    @staticmethod
    def _record_prior_and_sinkhorn(
        tables: dict[str, list[dict[str, Any]]],
        pair_key: Mapping[str, int],
        logits: np.ndarray,
        labels: np.ndarray,
        predictions: Mapping[str, np.ndarray],
        weights: Mapping[str, np.ndarray],
        lock: Any,
        class_counts: np.ndarray,
    ) -> None:
        """Record prior departures, prediction transitions, and saved Sinkhorn states."""
        selected = {selection.method_id: selection for selection in lock.selected}
        states = dict(lock.fitted_states)
        prior_selection = next(
            item for item in lock.control_selections if item.method_id == "prior_only_control"
        )
        prior_name = str(dict(prior_selection.configuration)["prior"])
        prior_control_weights = np.asarray(weights["prior_only_control"], dtype=np.float64)
        prior_weights = prior_control_weights.mean(axis=0)
        if not np.allclose(prior_control_weights, prior_weights[None, :], rtol=0.0, atol=1e-10):
            raise DiagnosticsError("saved prior-only control must use one fixed allocation")
        n_samples = len(labels)

        for method in _ADAPTIVE_METHODS:
            adaptive_weights = np.asarray(weights[method], dtype=np.float64)
            relation = analyze_prior_relation(adaptive_weights, prior_weights, labels, class_counts)
            overall = relation["scoped_departures"][0]
            tables["prior_diagnostics"].append({
                **dict(pair_key), "scope": "outer_fold",
                "diagnostic": "adaptive_mean_vs_selected_prior_only",
                "method": method, "selected_prior_name": prior_name,
                **relation["mean_profile_comparison"],
                "mean_weights": relation["mean_profile"],
                "selected_prior_weights": relation["reference_profile"],
                "mean_samplewise_l1_departure": overall["mean_l1_departure"],
                "samplewise_l1_departure_quantiles": relation["overall_samplewise_l1_departure"],
            })
            for departure in relation["scoped_departures"]:
                profile_scope = departure["scope"]
                profile_scope_id = departure["scope_id"]
                tables["prior_diagnostics"].append({
                    **dict(pair_key), "scope": "outer_fold",
                    "diagnostic": "adaptive_allocation_departure",
                    "method": method, "reference_name": prior_name,
                    "profile_scope": profile_scope,
                    "profile_scope_id": profile_scope_id,
                    **{key: value for key, value in departure.items()
                       if key not in {"scope", "scope_id"}},
                })
            control_transition = correctness_transition(
                labels,
                predictions["prior_only_control"],
                predictions[method],
                class_counts,
                baseline_name="prior_only_control",
                candidate_name=method,
            )
            tables["prior_diagnostics"].append({
                **dict(pair_key), "scope": "outer_fold",
                "diagnostic": "adaptive_vs_selected_prior_only_predictions",
                "method": method, "selected_prior_name": prior_name,
                **control_transition,
            })
            for transition in scoped_correctness_transitions(
                labels,
                predictions["prior_only_control"],
                predictions[method],
                class_counts,
                baseline_name="prior_only_control",
                candidate_name=method,
            ):
                profile_scope = transition["scope"]
                profile_scope_id = transition["scope_id"]
                tables["prior_diagnostics"].append({
                    **dict(pair_key), "scope": "outer_fold",
                    "diagnostic": "adaptive_vs_selected_prior_only_scoped_predictions",
                    "method": method, "selected_prior_name": prior_name,
                    "profile_scope": profile_scope,
                    "profile_scope_id": profile_scope_id,
                    **{key: value for key, value in transition.items()
                       if key not in {"scope", "scope_id"}},
                })
            counterfactual = mean_weight_counterfactual(
                logits, labels, adaptive_weights, predictions[method], class_counts,
            )
            tables["prior_diagnostics"].append({
                **dict(pair_key), "scope": "outer_fold",
                "diagnostic": "mean_weight_counterfactual",
                "method": method, "selected_prior_name": prior_name,
                **counterfactual,
            })
            for transition in counterfactual["scoped_transitions"]:
                profile_scope = transition["scope"]
                profile_scope_id = transition["scope_id"]
                tables["prior_diagnostics"].append({
                    **dict(pair_key), "scope": "outer_fold",
                    "diagnostic": "mean_weight_counterfactual_scoped_transition",
                    "method": method, "selected_prior_name": prior_name,
                    "profile_scope": profile_scope,
                    "profile_scope_id": profile_scope_id,
                    **{key: value for key, value in transition.items()
                       if key not in {"scope", "scope_id"}},
                })

        for method in _SINKHORN_METHODS:
            base_method = "contribution_ridge" if method.startswith("contribution") else "residual_ridge"
            state_key = "contribution_prices" if base_method == "contribution_ridge" else "residual_prices"
            price_state = states[state_key]
            solver_diagnostics = dict(price_state["diagnostics"])
            raw = np.asarray(weights[base_method], dtype=np.float64)
            smoothed = smooth_positive_kernel(raw)
            adjusted = np.asarray(weights[method], dtype=np.float64)
            target_prior = np.asarray(price_state["q"], dtype=np.float64)
            prices = np.asarray(price_state["log_prices"], dtype=np.float64)
            report = sinkhorn_stage_diagnostics(
                logits, labels, raw, smoothed, adjusted, prices, target_prior,
                class_counts, method,
            )
            common = {
                "target_prior": report["sinkhorn"]["target_prior"],
                "adjusted_marginal": report["sinkhorn"]["adjusted_marginal"],
                "marginal_delta": report["sinkhorn"]["marginal_delta"],
                "max_abs_marginal_residual": report["sinkhorn"]["max_abs_marginal_residual"],
                "marginal_kl_nats": report["sinkhorn"]["marginal_kl_nats"],
                "mean_row_kl_adjusted_vs_smoothed_nats": report["sinkhorn"]["mean_row_kl_adjusted_vs_smoothed_nats"],
                "prices": report["sinkhorn"]["prices"],
                "centered_prices": report["sinkhorn"]["centered_prices"],
                "centered_price_range": report["sinkhorn"]["centered_price_range"],
                "solver_diagnostics": solver_diagnostics,
                "saved_selection_diagnostics": selected[method].sinkhorn_diagnostics,
                "decomposition": report["decomposition"],
            }
            for stage in report["stages"]:
                for expert_index, expert in enumerate(EXPERT_NAMES):
                    tables["sinkhorn_adjustments"].append({
                        **dict(pair_key), "scope": "outer_fold", "method": method,
                        "stage": stage["stage"], "expert": expert,
                        "mean_weight": stage["mean_weights"][expert_index],
                        "std_weight": stage["std_weights"][expert_index],
                        "allocation_entropy_bits": stage["allocation_entropy_bits"],
                        "mean_sample_entropy_bits": stage["mean_sample_entropy_bits"],
                        "stage_fit_set_metrics": stage["fit_set_metrics"],
                        **common,
                    })
            for scope_row in report["scoped_rows"]:
                tables["sinkhorn_adjustments"].append({
                    **dict(pair_key), "scope": "outer_fold", "method": method,
                    "stage": "scoped_summary",
                    "profile_scope": scope_row["scope"],
                    "scope_id": scope_row["scope_id"],
                    "class_ids": scope_row["class_ids"],
                    "sample_count": scope_row["sample_count"],
                    "missing_class_ids": scope_row["missing_class_ids"],
                    "stage_summaries": scope_row["stage_summaries"],
                    "mean_raw_to_smoothed_l1": scope_row["mean_raw_to_smoothed_l1"],
                    "mean_smoothed_to_adjusted_l1": scope_row["mean_smoothed_to_adjusted_l1"],
                    "mean_raw_to_adjusted_l1": scope_row["mean_raw_to_adjusted_l1"],
                    "mean_kl_adjusted_vs_smoothed_nats": scope_row["mean_kl_adjusted_vs_smoothed_nats"],
                    "transitions": scope_row["transitions"],
                    **common,
                })

            relation = analyze_prior_relation(adjusted, target_prior, labels, class_counts)
            overall = relation["scoped_departures"][0]
            tables["prior_diagnostics"].append({
                **dict(pair_key), "scope": "outer_fold",
                "diagnostic": "adaptive_mean_vs_sinkhorn_target_prior",
                "method": method,
                "selected_prior_name": dict(selected[method].configuration)["prior"],
                "target_prior": relation["reference_profile"],
                "mean_weights": relation["mean_profile"],
                **relation["mean_profile_comparison"],
                "mean_samplewise_l1_departure": overall["mean_l1_departure"],
                "samplewise_l1_departure_quantiles": relation["overall_samplewise_l1_departure"],
                "max_abs_marginal_residual": report["sinkhorn"]["max_abs_marginal_residual"],
                "marginal_kl_nats": report["sinkhorn"]["marginal_kl_nats"],
                "solver_diagnostics": solver_diagnostics,
                "solver_converged": solver_diagnostics["converged"],
            })
            for departure in relation["scoped_departures"]:
                profile_scope = departure["scope"]
                profile_scope_id = departure["scope_id"]
                tables["prior_diagnostics"].append({
                    **dict(pair_key), "scope": "outer_fold",
                    "diagnostic": "sinkhorn_target_prior_departure",
                    "method": method, "reference_name": "sinkhorn_target_prior",
                    "profile_scope": profile_scope,
                    "profile_scope_id": profile_scope_id,
                    **{key: value for key, value in departure.items()
                       if key not in {"scope", "scope_id"}},
                })

        selective_method = "selective_residual_ridge_sinkhorn"
        selective_selection = selected[selective_method]
        tau = float(dict(selective_selection.configuration)["tau"])
        ridge = np.asarray(weights["residual_ridge"], dtype=np.float64)
        residual_sinkhorn = np.asarray(weights["residual_ridge_sinkhorn"], dtype=np.float64)
        blended, gates = selective_gate(ridge, residual_sinkhorn, tau, np.asarray(ANCHOR))
        saved_blended = np.asarray(weights[selective_method], dtype=np.float64)
        if not np.allclose(blended, saved_blended, rtol=0.0, atol=1e-10):
            raise DiagnosticsError("selective gate diagnostics differ from saved locked inference")
        gate_scopes: list[tuple[str, int | str, np.ndarray]] = [
            ("overall", "overall", np.ones(n_samples, dtype=bool))
        ]
        gate_scopes.extend(("class", class_id, np.asarray(labels) == class_id)
                           for class_id in range(len(class_counts)))
        gate_scopes.extend(
            (group, group, np.isin(labels, class_ids))
            for group, class_ids in compute_class_groups(class_counts).items()
        )
        solver_diagnostics = dict(states["residual_prices"]["diagnostics"])
        for scope, scope_id, mask in gate_scopes:
            selected_gates = gates[mask]
            values = saved_blended[mask]
            tables["sinkhorn_adjustments"].append({
                **dict(pair_key), "scope": "outer_fold", "method": selective_method,
                "stage": "selective_gate", "profile_scope": scope, "scope_id": scope_id,
                "group": scope_id if scope in {"overall", "head", "medium", "tail"} else None,
                "tau": tau, "gate_mean": float(selected_gates.mean()) if len(selected_gates) else None,
                "gate_p05": float(np.quantile(selected_gates, .05)) if len(selected_gates) else None,
                "gate_median": float(np.quantile(selected_gates, .50)) if len(selected_gates) else None,
                "gate_p95": float(np.quantile(selected_gates, .95)) if len(selected_gates) else None,
                "gate_min": float(selected_gates.min()) if len(selected_gates) else None,
                "gate_max": float(selected_gates.max()) if len(selected_gates) else None,
                "mean_blended_weights": values.mean(axis=0).tolist() if len(values) else None,
                "sample_count": int(mask.sum()),
                "solver_diagnostics": solver_diagnostics,
                "saved_selection_diagnostics": selective_selection.sinkhorn_diagnostics,
                "interpretation": "saved selective gate based on L1 distance from the fixed_007 anchor",
            })
        gate_transition = correctness_transition(
            labels, predictions["residual_ridge_sinkhorn"], predictions[selective_method],
            class_counts, baseline_name="residual_ridge_sinkhorn", candidate_name=selective_method,
        )
        tables["sinkhorn_adjustments"].append({
            **dict(pair_key), "scope": "outer_fold", "method": selective_method,
            "stage": "selective_blend_transition", "tau": tau,
            "mean_gate": float(gates.mean()), "n_samples": n_samples,
            "solver_diagnostics": solver_diagnostics,
            **gate_transition,
        })
        blended_mean = saved_blended.mean(axis=0)
        blended_entropy = -float(np.sum(blended_mean[blended_mean > 0] * np.log2(blended_mean[blended_mean > 0])))
        for expert_index, expert in enumerate(EXPERT_NAMES):
            tables["sinkhorn_adjustments"].append({
                **dict(pair_key), "scope": "outer_fold", "method": selective_method,
                "stage": "selective_blend", "expert": expert,
                "mean_weight": float(blended_mean[expert_index]),
                "allocation_entropy_bits": blended_entropy,
                "mean_sample_entropy_bits": float(np.mean(
                    -np.sum(np.where(saved_blended > 0, saved_blended * np.log2(np.maximum(saved_blended, 1e-300)), 0), axis=1)
                )),
                "mean_gate": float(gates.mean()), "tau": tau,
                "mean_l1_from_residual_sinkhorn": float(np.mean(np.abs(saved_blended - residual_sinkhorn).sum(axis=1))),
                "solver_diagnostics": solver_diagnostics,
                "saved_selection_diagnostics": selective_selection.sinkhorn_diagnostics,
            })

    def _provenance(
        self,
        inputs: DiagnosticsInputs,
        *,
        complete: bool,
        selected_pairs: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        source_identity = self._analysis_identity()
        freeze_path = inputs.repository.study_root / "manifests" / "study-freeze.json"
        audit_path = inputs.planner.native_store.base_dir / "reuse_compatibility_inner.json"
        fold_path = inputs.planner.native_store.manifest_path
        lock_hashes: dict[str, dict[str, str]] = {}
        for pair, lock in sorted(inputs.locks.items()):
            path = inputs.repository.lock_path(*pair)
            lock_hashes[f"seed_{pair[0]}_outer_{pair[1]}"] = {
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "lock_sha256": lock.lock_sha256,
            }
        source_hashes: dict[str, dict[str, Any]] = {}
        for job_id, reference in sorted(inputs.references.items()):
            source_hashes[job_id] = {
                "source_experiment_id": reference.source_experiment_id,
                "source_root_name": reference.source_root_name,
                "checkpoint_sha256": reference.checkpoint_sha256,
                "prediction_sha256": reference.prediction_sha256,
                "resolved_config_sha256": reference.resolved_config_sha256,
                "run_manifest_sha256": reference.manifest_sha256,
            }
        return json_safe(
            {
                "schema_version": "expert_method.inner_diagnostics_provenance.v1",
                "study_id": STUDY_ID,
                "stage": "inner",
                "complete_matrix": complete,
                "pairs_analyzed": [list(pair) for pair in selected_pairs],
                "validation_counts": dict(inputs.validation_counts),
                "training_source_commit": inputs.manifest["source_commit"],
                "training_source_tree_dirty": inputs.manifest["source_tree_dirty"],
                "result_snapshot_commit": inputs.snapshot_commit,
                "result_snapshot_marker_sha256": inputs.snapshot_marker_sha256,
                **source_identity,
                "configuration": {
                    "study_config_sha256": self.reader.config.sha256,
                    "study_yaml_sha256": inputs.config_file_sha256,
                    "expert_config_sha256": study_cli._expert_config_hashes(self.reader.definition),
                    "plan_sha256": inputs.manifest["plan_sha256"],
                },
                "artifact_identity": {
                    "freeze_sha256": json.loads(freeze_path.read_text())["freeze_sha256"],
                    "freeze_file_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                    "matrix_manifest_sha256": inputs.manifest_sha256,
                    "fold_manifest_file_sha256": hashlib.sha256(fold_path.read_bytes()).hexdigest(),
                    "reuse_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                    "locks": lock_hashes,
                    "inner_source_hashes": source_hashes,
                },
                "package_versions": source_identity["package_versions"],
                "command_arguments": list(self.command_arguments),
                "output_root": str(self.output_root.resolve()),
                "figure_output": self.make_figures,
                "outer_predictions_loaded": False,
                "outer_labels_loaded": False,
                "original_test_loaded": False,
                "routers_refit": False,
                "selection_repeated": False,
            }
        )
