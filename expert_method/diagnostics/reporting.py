"""Deterministic CSV, JSON, Markdown, and figure output for diagnostics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scripts.analysis import ImmutableArtifactWriter

from .contracts import DiagnosticsError, json_safe


TABLE_FILES = {
    "expert_oof_metrics": "expert_oof_metrics.csv",
    "complementarity": "complementarity.csv",
    "per_class_specialization": "per_class_specialization.csv",
    "frequency_associations": "frequency_associations.csv",
    "oracle_opportunity": "oracle_opportunity.csv",
    "selection_cv_metrics": "selection_cv_metrics.csv",
    "router_fit_set_metrics": "router_fit_set_metrics.csv",
    "router_weight_profiles": "router_weight_profiles.csv",
    "router_class_profiles": "router_class_profiles.csv",
    "router_group_allocations": "router_group_allocations.csv",
    "stability": "router_stability.csv",
    "prior_diagnostics": "prior_diagnostics.csv",
    "sinkhorn_adjustments": "sinkhorn_adjustments.csv",
    "seed_aggregates": "seed_aggregates.csv",
}

_FIGURE_FILES = (
    "complementarity.png", "complementarity.svg",
    "specialization.png", "specialization.svg",
    "allocation_entropy.png", "allocation_entropy.svg",
    "stability.png", "stability.svg",
    "sinkhorn_adjustments.png", "sinkhorn_adjustments.svg",
)


def _assert_no_symlink_components(output_root: Path, relative_path: Path) -> None:
    """Reject symlink targets beneath the output root before writing files."""
    root = Path(os.path.abspath(output_root))
    target = Path(os.path.abspath(root / relative_path))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise DiagnosticsError(f"output path escapes its root: {target}") from exc
    current = root
    if current.is_symlink():
        raise DiagnosticsError(f"output path contains a symlink: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DiagnosticsError(f"output path contains a symlink: {current}")


def assert_no_output_symlinks(output_root: str | Path, *, make_figures: bool) -> None:
    """Preflight all possible output targets, including existing dangling links."""
    root = Path(os.path.abspath(Path(output_root).expanduser()))
    if root.is_symlink():
        raise DiagnosticsError(f"output root must not be a symlink: {root}")
    relative_paths = [
        *(Path(filename) for filename in TABLE_FILES.values()),
        Path("analysis.json"), Path("provenance.json"), Path("report.md"),
    ]
    if make_figures:
        relative_paths.extend(Path("figures") / filename for filename in _FIGURE_FILES)
    for relative_path in relative_paths:
        _assert_no_symlink_components(root, relative_path)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def render_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render stable CSV columns in first-seen order; nested cells use JSON."""
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            name = str(key)
            if name not in seen:
                columns.append(name)
                seen.add(name)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    if columns:
        writer.writeheader()
        for row in rows:
            writer.writerow({str(key): _cell(value) for key, value in row.items()})
    return buffer.getvalue()


def _write_bytes_once(
    path: Path, payload: bytes, *, output_root: Path | None = None,
) -> None:
    """Install one immutable binary artifact with the same no-clobber rule."""
    if output_root is not None:
        try:
            relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(output_root)))
        except ValueError as exc:
            raise DiagnosticsError(f"output path escapes its root: {path}") from exc
        _assert_no_symlink_components(output_root, relative)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise DiagnosticsError(f"refusing to overwrite an incompatible output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise DiagnosticsError(f"refusing to overwrite an incompatible output: {path}")
    except OSError as exc:
        raise DiagnosticsError(f"cannot write immutable binary output: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _markdown(summary: Mapping[str, Any]) -> str:
    validation = summary.get("validation_counts", {})
    metrics = summary.get("descriptive_metrics", [])
    highlights = summary.get("diagnostic_highlights", {})
    lines = [
        "# Inner Ridge/Sinkhorn diagnostics",
        "",
        "This report describes validated inner OOF expert predictions and saved router states.",
        "It does not read outer predictions, outer labels, or the original CIFAR-100 test set.",
        "",
        "## Validated inputs",
        "",
        f"- Fold locks: {validation.get('locks', 'unknown')} (expected 15).",
        f"- Inner expert jobs: {validation.get('inner_jobs', 'unknown')} (expected 240).",
        f"- Native inner jobs: {validation.get('native_inner_jobs', 'unknown')}.",
        f"- Historical inner jobs: {validation.get('historical_inner_jobs', 'unknown')}.",
        f"- Diagnostic pairs analyzed: {summary.get('pairs_analyzed', 0)} / {summary.get('expected_pair_count', 15)}; complete matrix: {str(bool(summary.get('complete_matrix', False))).lower()}.",
        "",
        "## Evidence labels",
        "",
        "- Expert OOF rows are diagnostics on the saved inner out-of-fold predictions.",
        "- Selection CV rows reproduce metrics recorded in the immutable locks.",
        "- Router fit-set rows apply saved final router states to the rows used to fit them; they are descriptive fit-set results.",
        "- Fold populations overlap across outer folds and seeds. Summaries average folds within each seed, then report the mean and sample standard deviation across the three seeds.",
        "",
        "## Descriptive metrics",
        "",
        "Sample accuracy and macro recall are separate. Undefined class or group statistics are blank in CSV and null in JSON.",
        "",
        "| Evidence | Method | Metric | Mean across seeds | Sample SD across seeds |",
        "|:--|:--|:--|--:|--:|",
    ]
    for row in metrics:
        lines.append(
            "| {evidence} | {method} | {metric} | {mean} | {sd} |".format(
                evidence=row.get("evidence", ""),
                method=row.get("method", ""),
                metric=row.get("metric", ""),
                mean=_number(row.get("mean_across_seeds")),
                sd=_number(row.get("sample_sd_across_seeds")),
            )
        )
    lines.extend(
        [
            "",
            "## Complementarity and specialization",
            "",
            f"- Fraction with at least one correct expert: {_number(highlights.get('any_expert_correct_fraction'))}.",
            f"- Fraction with all four experts wrong: {_number(highlights.get('all_experts_wrong_fraction'))}.",
            f"- Label-dependent hard-selection oracle BA: {_number(highlights.get('hard_oracle_balanced_accuracy'))}; its gap over the strongest individual expert is {_number(highlights.get('hard_oracle_ba_gap'))}.",
            f"- Mean class-profile disagreement is {_number(highlights.get('mean_class_disagreement'))}; specialization ties are retained in the class table.",
            "",
            "## Saved router profiles",
            "",
            f"- Mean normalized allocation entropy: {_number(highlights.get('mean_normalized_router_entropy'))}.",
            f"- Mean class-profile L1 distance across folds within seeds: {_number(highlights.get('fold_profile_mean_l1'))}.",
            "These are descriptive profiles of locked states applied to their router fit rows.",
            "",
            "## Prior and Sinkhorn diagnostics",
            "",
            f"- Mean L1 distance from the selected prior-only control: {_number(highlights.get('mean_l1_from_prior_only'))}.",
            f"- Mean maximum absolute Sinkhorn marginal residual: {_number(highlights.get('mean_maximum_abs_sinkhorn_marginal_residual'))}.",
            f"- Mean selective gate: {_number(highlights.get('mean_selective_gate'))}.",
            "Marginal residual describes fit-population balance against a saved target. It does not establish solver failure or population generalization.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Stable class allocation profiles describe repeatability across these saved folds; they do not establish a flat hyperparameter optimum.",
            "Prediction changes from saved states describe this development population and do not establish generalization.",
            "The hard-selection oracle uses labels and is a diagnostic, not an inference-time method or a soft-mixture upper bound.",
            "Sinkhorn marginal mismatch is reported as a descriptive residual and does not itself indicate solver non-convergence.",
            "",
            "## Files",
            "",
            "CSV tables preserve fold-level rows and descriptive seed aggregation. Oracle, prior comparisons, target priors, solver diagnostics, and selective gates are recorded in dedicated tables. Provenance is recorded in `provenance.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def _sinkhorn_plot_series(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, str], list[float]]:
    """Return per-expert mean marginal weights for the three plotted methods."""
    expert_names = ("CE", "LAL", "BalancedSoftmax", "Mixup")
    rows = [
        row for row in tables.get("sinkhorn_adjustments", ())
        if row.get("scope") == "mean_across_seeds"
    ]
    requested = {
        "contribution_ridge_sinkhorn": ("raw", "smoothed", "adjusted"),
        "residual_ridge_sinkhorn": ("raw", "smoothed", "adjusted"),
        "selective_residual_ridge_sinkhorn": ("selective_blend",),
    }
    result: dict[tuple[str, str], list[float]] = {}
    for method, stages in requested.items():
        for stage in stages:
            result[(method, stage)] = [
                next((float(row["mean_weight_across_seeds"]) for row in rows
                      if row.get("method") == method and row.get("stage") == stage
                      and row.get("expert") == expert
                         and row.get("mean_weight_across_seeds") is not None), float("nan"))
                for expert in expert_names
            ]
    return result


def _figure_payloads(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, bytes]:
    """Create stable Agg figures for the five requested diagnostic views."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["svg.hashsalt"] = "expert_method.diagnostics.v1"
        matplotlib.rcParams["timezone"] = "UTC"
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise DiagnosticsError(
            "figures require Matplotlib; install requirements-analysis.txt or pass --no-figures"
        ) from exc

    figures: dict[str, bytes] = {}

    def save(name: str, figure: Any) -> None:
        for extension in ("png", "svg"):
            buffer = io.BytesIO()
            metadata = (
                {"Software": "expert_method.diagnostics"}
                if extension == "png"
                else {"Date": None, "Creator": "expert_method.diagnostics"}
            )
            figure.savefig(
                buffer,
                format=extension,
                dpi=150,
                bbox_inches="tight",
                metadata=metadata,
            )
            figures[f"{name}.{extension}"] = buffer.getvalue()
        plt.close(figure)

    distribution = [
        row for row in tables.get("complementarity", ())
        if row.get("scope") == "mean_across_seeds" and row.get("kind") == "correct_count_distribution"
        and row.get("mean_fraction_across_seeds") is not None
    ]
    expert_names = ["CE", "LAL", "BalancedSoftmax", "Mixup"]
    fractions_by_count = {
        int(row["num_correct_experts"]): float(row["mean_fraction_across_seeds"])
        for row in distribution
        if row.get("mean_fraction_across_seeds") is not None
        and np.isfinite(float(row["mean_fraction_across_seeds"]))
    }
    if set(fractions_by_count) == {0, 1, 2, 3, 4}:
        figure, axis = plt.subplots(figsize=(7, 4))
        correct_counts = [0, 1, 2, 3, 4]
        bottoms = np.zeros(1, dtype=np.float64)
        colors = ["#AA3377", "#EE7733", "#CCBB44", "#228833", "#4477AA"]
        for number_correct, color in zip(correct_counts, colors):
            fraction = fractions_by_count[number_correct]
            axis.bar(["Inner OOF rows"], [fraction], bottom=bottoms, label=f"{number_correct} correct", color=color)
            bottoms += fraction
        axis.set_ylabel("Fraction of samples")
        axis.set_title("Expert correctness overlap and oracle opportunity")
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        save("complementarity", figure)

    specialization_rows = list(tables.get("per_class_specialization", ()))
    selected = [
        row for row in specialization_rows
        if row.get("scope") == "mean_across_seeds" and row.get("class_recall") is not None
    ]
    if selected:
        classes = sorted({int(row["class_id"]) for row in selected})
        matrix = np.full((len(expert_names), len(classes)), np.nan)
        class_index = {class_id: index for index, class_id in enumerate(classes)}
        expert_index = {name: index for index, name in enumerate(expert_names)}
        for row in selected:
            matrix[expert_index[str(row["expert"])], class_index[int(row["class_id"])]] = float(row["class_recall"])
        figure, axis = plt.subplots(figsize=(11, 3.8))
        image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_yticks(range(len(expert_names)), expert_names)
        axis.set_xlabel("Canonical class ID")
        axis.set_title("Per-class expert recall across seeds")
        figure.colorbar(image, ax=axis, label="Recall")
        save("specialization", figure)

    profile_rows = list(tables.get("router_weight_profiles", ()))
    profile = [
        row for row in profile_rows
        if row.get("scope") == "mean_across_seeds" and row.get("expert")
        and row.get("mean_weight_across_seeds") is not None
        and np.isfinite(float(row["mean_weight_across_seeds"]))
    ]
    if profile:
        method_names = list(dict.fromkeys(str(row.get("method")) for row in profile))
        expert_values = ["CE", "LAL", "BalancedSoftmax", "Mixup"]
        weights_by_method_expert = {
            (str(row["method"]), str(row["expert"])): float(row["mean_weight_across_seeds"])
            for row in profile
        }
        if not all((method, expert) in weights_by_method_expert
                   for method in method_names for expert in expert_values):
            profile = []
    if profile:
        x = np.arange(len(method_names), dtype=np.float64)
        figure, axis = plt.subplots(figsize=(11, 4.5))
        bottoms = np.zeros(len(method_names), dtype=np.float64)
        colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
        for expert_id, expert in enumerate(expert_values):
            heights = [weights_by_method_expert[(method, expert)] for method in method_names]
            axis.bar(x, heights, bottom=bottoms, label=expert, color=colors[expert_id])
            bottoms += np.asarray(heights)
        axis.set_xticks(x, method_names, rotation=24, ha="right")
        axis.set_ylabel("Mean expert weight")
        axis.set_title("Saved router allocation and normalized entropy")
        entropy_rows = [row for row in profile_rows if row.get("scope") == "mean_across_seeds" and row.get("normalized_allocation_entropy") is not None and row.get("expert") == "CE"]
        if entropy_rows:
            entropy_by_method = [
                next((float(row["normalized_allocation_entropy"]) for row in entropy_rows if row.get("method") == method), np.nan)
                for method in method_names
            ]
            axis.plot(x, entropy_by_method, color="#AA3377", marker="o", label="Allocation entropy")
            sample_entropy = [
                next((float(row["mean_normalized_sample_entropy"]) for row in entropy_rows
                      if row.get("method") == method and row.get("mean_normalized_sample_entropy") is not None), np.nan)
                for method in method_names
            ]
            if any(np.isfinite(sample_entropy)):
                axis.plot(x, sample_entropy, color="#117733", marker="s", label="Mean sample entropy")
            axis.set_ylim(0.0, 1.05)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        save("allocation_entropy", figure)

    stability_rows = [
        row for row in tables.get("stability", ())
        if row.get("comparison_scope") == "mean_across_seeds"
        and row.get("mean_l1_distance") is not None
        and np.isfinite(float(row["mean_l1_distance"]))
    ]
    if stability_rows:
        method_names = list(dict.fromkeys(str(row.get("method")) for row in stability_rows))
        values_by_method = {str(row["method"]): float(row["mean_l1_distance"]) for row in stability_rows}
        values = [values_by_method[method] for method in method_names]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.bar(method_names, values, color="#66CCEE")
        axis.set_ylabel("Mean class-profile L1 distance")
        axis.set_title("Saved router profile stability across folds")
        axis.tick_params(axis="x", rotation=22)
        save("stability", figure)

    sinkhorn_profiles = _sinkhorn_plot_series(tables)
    if any(np.isfinite(values).any() for values in sinkhorn_profiles.values()):
        stages = ["raw", "smoothed", "adjusted"]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for axis, method in zip(axes[:2], ("contribution_ridge_sinkhorn", "residual_ridge_sinkhorn")):
            x = np.arange(len(expert_names))
            width = 0.25
            for index, stage in enumerate(stages):
                means = sinkhorn_profiles[(method, stage)]
                axis.bar(x + (index - 1) * width, means, width, label=stage)
            axis.set_xticks(x, expert_names, rotation=15)
            axis.set_title(method.replace("_", " "))
            axis.set_ylabel("Mean marginal weight")
            axis.legend()
        selective_axis = axes[2]
        x = np.arange(len(expert_names))
        selective_means = sinkhorn_profiles[("selective_residual_ridge_sinkhorn", "selective_blend")]
        selective_axis.bar(x, selective_means, color="#228833", label="selective blend")
        selective_axis.set_xticks(x, expert_names, rotation=15)
        selective_axis.set_title("Selective residual blend")
        selective_axis.set_ylabel("Mean marginal weight")
        selective_axis.legend()
        figure.suptitle("Sinkhorn smoothing and frozen-price adjustment")
        save("sinkhorn_adjustments", figure)

    return figures


def write_outputs(
    output_root: str | Path,
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    make_figures: bool,
) -> dict[str, str]:
    """Persist all outputs idempotently; conflicting bytes are never replaced."""
    assert_no_output_symlinks(output_root, make_figures=make_figures)
    root = Path(output_root).expanduser().resolve()
    writer = ImmutableArtifactWriter(error_type=DiagnosticsError)
    written: dict[str, str] = {}
    root.mkdir(parents=True, exist_ok=True)
    for name, filename in TABLE_FILES.items():
        rows = list(tables.get(name, ()))
        path = root / filename
        writer.write_text_once(path, render_csv(rows))
        written[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    safe_summary = json_safe(summary)
    safe_provenance = json_safe(provenance)
    for filename, payload in (("analysis.json", safe_summary), ("provenance.json", safe_provenance)):
        path = root / filename
        writer.write_json_once(path, payload)
        written[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    report_path = root / "report.md"
    writer.write_text_once(report_path, _markdown(safe_summary))
    written["report.md"] = hashlib.sha256(report_path.read_bytes()).hexdigest()

    if make_figures:
        figures = _figure_payloads(tables)
        for filename, payload in figures.items():
            path = root / "figures" / filename
            _write_bytes_once(path, payload, output_root=root)
            written[f"figures/{filename}"] = hashlib.sha256(payload).hexdigest()
    return written
