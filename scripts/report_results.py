"""Build descriptive paper tables and figures from the archived OT evaluations."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
METHODS = {
    "reweighting": (
        ("baseline", "Baseline"),
        ("uniform", "Uniform continuation"),
        ("class_balanced", "Class-balanced"),
        ("cluster_balanced", "Cluster-balanced"),
        ("sinkhorn", "Prototype Sinkhorn"),
        ("sava", "SAVA + class balance"),
    ),
    "synthesis": (
        ("baseline", "Baseline"),
        ("real_only", "Real-only continuation"),
        ("random_duplicate", "Random duplication"),
        ("uniform_ot", "Uniform OT"),
        ("loss_shifted_ot", "Loss-shifted OT"),
        ("full_ot", "Full guided OT"),
    ),
}
METRICS = ("accuracy", "head_recall", "medium_recall", "tail_recall")
COLUMNS = (
    ("accuracy_percent", "Accuracy (%)"),
    ("head_recall_percent", "Head (%)"),
    ("medium_recall_percent", "Medium (%)"),
    ("tail_recall_percent", "Tail (%)"),
    ("accuracy_delta_pp", "Delta accuracy (pp)"),
)
SEED_STYLES = (("o", "#0072B2"), ("s", "#B34700"), ("^", "#007F5F"))


def validate_metrics(metrics, path):
    """Require complete balanced-test results; never turn missing values into zero."""
    for key in (*METRICS, "balanced_accuracy"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{path}: expected a finite fraction for {key}")
    if metrics.get("test_support") != [100] * 100:
        raise ValueError(f"{path}: expected 100 test examples in each of 100 classes")
    recalls = metrics.get("per_class_recall")
    if not isinstance(recalls, list) or len(recalls) != 100:
        raise ValueError(f"{path}: expected 100 per-class recalls")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in recalls):
        raise ValueError(f"{path}: per-class recalls must be complete finite fractions")
    if not math.isclose(metrics["accuracy"], statistics.mean(recalls), abs_tol=1e-12):
        raise ValueError(f"{path}: accuracy disagrees with balanced test class recalls")
    if not math.isclose(metrics["accuracy"], metrics["balanced_accuracy"], abs_tol=1e-12):
        raise ValueError(f"{path}: balanced accuracy disagrees with accuracy")


def load_records(results_root):
    records, inputs = [], []
    for family, methods in METHODS.items():
        for seed in SEEDS:
            seed_records = []
            for method, label in methods:
                relative = (
                    Path(f"reweighting/runs/{method}_seed{seed}/metrics.json")
                    if family == "reweighting"
                    else Path(f"synthesis/comparison_5000_seed{seed}/{method}_metrics.json")
                )
                path = results_root / relative
                metrics = json.loads(path.read_text())
                validate_metrics(metrics, path)
                inputs.append(path)
                seed_records.append({
                    "family": family, "method": method, "label": label, "seed": seed,
                    "source_file": relative.as_posix(),
                    **{key + "_percent": 100 * metrics[key] for key in METRICS},
                })
            baseline = seed_records[0]
            for record in seed_records:
                for metric in ("accuracy", "tail_recall"):
                    record[metric + "_delta_pp"] = (
                        record[metric + "_percent"] - baseline[metric + "_percent"]
                    )
                records.append(record)
    return records, inputs


def summarize_method(records):
    if sorted(record["seed"] for record in records) != list(SEEDS):
        raise ValueError("Each method must have exactly one result for seeds 0, 1, and 2")
    columns = [key for key in records[0] if key.endswith(("_percent", "_pp"))]
    return {
        key: {"mean": statistics.mean(record[key] for record in records),
              "sd": statistics.stdev(record[key] for record in records)}
        for key in columns
    }


def summarize(records):
    return {
        family: {
            method: summarize_method([
                row for row in records if row["family"] == family and row["method"] == method
            ])
            for method, _ in methods
        }
        for family, methods in METHODS.items()
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(family, summary, output):
    methods = METHODS[family]
    best = {key: max(summary[method][key]["mean"] for method, _ in methods) for key, _ in COLUMNS}
    markdown = [
        f"# {'Reweighting' if family == 'reweighting' else 'Synthesis'} results", "",
        "Mean ± sample SD across training seeds 0, 1, 2 (n = 3). Higher is better.", "",
        "| Method | " + " | ".join(label + " ↑" for _, label in COLUMNS) + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    recipe = "warm-up/multistep" if family == "reweighting" else "cosine"
    caption = (
        f"{family.capitalize()} results on CIFAR-100-LT (imbalance factor 0.01), "
        f"using the {recipe} baseline. Values are mean $\\pm$ sample SD over "
        "training seeds 0, 1, 2. Head/Medium/Tail are macro class recalls. "
        "Accuracy changes are computed within seed against this family's baseline "
        "and measured in percentage points (pp). Bold marks the highest mean "
        "within each column, not statistical significance."
    )
    latex = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\caption{" + caption + "}",
        r"\label{tab:ot-" + family + "}", r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & Accuracy (\%) & Head (\%) & Medium (\%) & Tail (\%) & $\Delta$ accuracy (pp) \\",
        r"\midrule",
    ]
    csv_rows = []
    for method, label in methods:
        md_cells, tex_cells = [], []
        csv_row = {"method": method, "label": label, "n_seeds": len(SEEDS)}
        for key, _ in COLUMNS:
            mean, sd = summary[method][key]["mean"], summary[method][key]["sd"]
            csv_row[key + "_mean"] = mean
            csv_row[key + "_sd"] = sd
            if method == "baseline" and key == "accuracy_delta_pp":
                md_cells.append("—")
                tex_cells.append("---")
                continue
            signed = "+" if key.endswith("_pp") else ""
            md_value = f"{mean:{signed}.2f} ± {sd:.2f}"
            tex_value = f"{mean:{signed}.2f} \\pm {sd:.2f}"
            if math.isclose(mean, best[key], abs_tol=1e-12):
                md_value = f"**{md_value}**"
                tex_value = r"\boldsymbol{" + tex_value + "}"
            md_cells.append(md_value)
            tex_cells.append("$" + tex_value + "$")
        markdown.append("| " + label + " | " + " | ".join(md_cells) + " |")
        latex.append(label + " & " + " & ".join(tex_cells) + r" \\")
        csv_rows.append(csv_row)
    markdown.extend([
        "", "Head: retained class count ≥100; Medium: 20–99; Tail: <20.", "",
        "Delta accuracy is calculated per seed before computing its mean and SD. "
        "Bold marks the highest unrounded mean in each column (including ties); "
        "it does not indicate statistical significance.", "",
        f"Baseline recipe: {recipe}. Compare methods within this table; "
        "the other experiment family used a different baseline and classifier recipe.", "",
    ])
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / f"{family}.md").write_text("\n".join(markdown), encoding="utf-8")
    (output / f"{family}.tex").write_text("\n".join(latex), encoding="utf-8")
    write_csv(output / f"{family}.csv", csv_rows)


def plot_changes(records, summaries, metric, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    key = metric + "_delta_pp"
    title = "Overall accuracy improvement" if metric == "accuracy" else "Tail recall improvement"
    limits = [0.0]
    for family, methods in METHODS.items():
        for method, _ in methods[1:]:
            values = summaries[family][method][key]
            limits.extend((values["mean"] - values["sd"], values["mean"] + values["sd"]))
            limits.extend(row[key] for row in records if row["family"] == family and row["method"] == method)
    lower, upper = min(limits), max(limits)
    padding = 0.08 * max(upper - lower, 0.1)
    style = {
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "text.usetex": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, layout="constrained")
        for panel, (family, methods), ax in zip("AB", METHODS.items(), axes, strict=True):
            for index, (method, _) in enumerate(methods[1:]):
                rows = sorted(
                    (row for row in records if row["family"] == family and row["method"] == method),
                    key=lambda row: row["seed"],
                )
                for offset, row, (marker, color) in zip((-0.24, -0.12, 0), rows, SEED_STYLES, strict=True):
                    ax.plot(row[key], index + offset, marker=marker, color=color,
                            markersize=4, linestyle="none", markeredgewidth=0.7, zorder=4)
                values = summaries[family][method][key]
                ax.errorbar(values["mean"], index + 0.24, xerr=values["sd"],
                            fmt="D", color="#222222", markersize=3.8,
                            elinewidth=1.1, capsize=2.5, zorder=3)
            recipe = "warm-up/multistep baseline" if family == "reweighting" else "cosine baseline"
            ax.set_title(f"{panel}  {family.capitalize()}\n{recipe}", loc="left", pad=10)
            ax.set_yticks(range(len(methods) - 1), [label for _, label in methods[1:]])
            ax.set_ylim(len(methods) - 1 - 0.5, -0.55)
            ax.set_xlim(lower - padding, upper + padding)
            ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=1)
            ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel("Change from own baseline (pp)", labelpad=7)
        handles = [
            Line2D([], [], marker=marker, color=color, linestyle="none", markersize=4,
                   label=f"Seed {seed}")
            for seed, (marker, color) in zip(SEEDS, SEED_STYLES, strict=True)
        ]
        handles.append(Line2D([], [], marker="D", color="#222222", markersize=4,
                              linewidth=1, label="Mean ± SD"))
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(title, fontsize=11)
        for extension in ("pdf", "svg", "png"):
            metadata = {"Date": None} if extension == "svg" else None
            figure_path = output / f"{metric}_improvement.{extension}"
            fig.savefig(figure_path, dpi=300,
                        facecolor="white", transparent=False, metadata=metadata)
            if extension == "svg":
                # Preserve path-command separators while removing backend trailing spaces.
                figure_path.write_text("\n".join(line.rstrip() for line in figure_path.read_text().splitlines()) + "\n")
        plt.close(fig)


def build_report(results_root, output, overwrite=False):
    results_root, output = results_root.resolve(), output.resolve()
    if output == results_root or any(output.is_relative_to(results_root / family) for family in METHODS):
        raise ValueError("Output must be separate from the imported source results")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Report directory exists: {output}; use --overwrite to regenerate")
    records, inputs = load_records(results_root)
    summaries = summarize(records)
    import matplotlib
    matplotlib.use("Agg")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_metrics.csv", records)
    for family in METHODS:
        write_tables(family, summaries[family], output)
    for metric in ("accuracy", "tail_recall"):
        plot_changes(records, summaries, metric, output)
    inputs.append(results_root / "manifest.json")
    metadata = {
        "training_seeds": list(SEEDS), "n_seeds": 3,
        "aggregation": "Arithmetic mean and sample SD (ddof=1) across training seeds",
        "deltas": "100 * (method fraction - same-family, same-seed baseline fraction); SD of these paired differences",
        "excluded": ["synthesis/comparison_seed0: additional historical seed-0 experiment"],
        "missing_data": "Require complete finite metrics for every included method and seed; no imputation or exclusions within the included experiment sets",
        "inference": "Descriptive only; no hypothesis tests or significance claims",
        "figure": {"width_inches": 7.2, "height_inches": 3.8, "png_dpi": 300,
                   "axis_limits": "Shared across the two panels for each outcome; include zero, all seeds, and mean ± SD",
                   "seed_vertical_offsets": [-0.24, -0.12, 0], "mean_vertical_offset": 0.24},
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [{"path": path.relative_to(results_root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs],
        "summaries": summaries,
    }
    (output / "analysis.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results/paper")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate report files in an existing output directory")
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="ot-report-mpl-") as cache:
            os.environ.setdefault("MPLCONFIGDIR", cache)
            output = build_report(args.results_root, args.output_dir, args.overwrite)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved descriptive tables and figures to {output}")


if __name__ == "__main__":
    main()
