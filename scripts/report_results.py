"""Build descriptive paper tables and figures from the archived OT evaluations."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import textwrap


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
METRICS = ("accuracy", "balanced_accuracy", "head_recall", "medium_recall", "tail_recall")
COLUMNS = (
    ("accuracy_percent", "Accuracy (%)"),
    ("head_recall_percent", "Head (%)"),
    ("medium_recall_percent", "Medium (%)"),
    ("tail_recall_percent", "Tail (%)"),
    ("accuracy_delta_pp", "Delta accuracy (pp)"),
)
CHARTS = {
    "balanced_accuracy": ("balanced_accuracy_percent", "Balanced accuracy (%)"),
    "tail_accuracy": ("tail_recall_percent", "Tail accuracy (macro recall, %)"),
}
LEGACY_OUTPUTS = tuple(
    f"{metric}_improvement.{extension}"
    for metric in ("accuracy", "tail_recall")
    for extension in ("pdf", "svg", "png")
) + ("analysis.json", "seed_metrics.csv", "reweighting_accuracy.png", "synthesis_accuracy.png")


def validate_metrics(metrics, path):
    """Require complete balanced-test results; never turn missing values into zero."""
    for key in METRICS:
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
    records = []
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
    return records


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


def plot_metric_bars(family, summary, chart, output, upper_limit):
    import matplotlib.pyplot as plt

    methods = METHODS[family]
    key, axis_label = CHARTS[chart]
    means = [summary[method][key]["mean"] for method, _ in methods]
    labels = [textwrap.fill(label, width=17, break_long_words=False) for _, label in methods]
    color = "#0072B2" if family == "reweighting" else "#007F5F"
    recipe = "warm-up/multistep baseline" if family == "reweighting" else "cosine baseline"
    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "text.usetex": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        bars = ax.bar(
            range(len(methods)), means, width=0.65,
            color=["#777777"] + [color] * (len(methods) - 1),
            zorder=3,
        )
        for bar, mean in zip(bars, means, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.6,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
        title = f"{family.capitalize()} methods"
        if chart == "tail_accuracy":
            title += " — tail accuracy"
        elif chart == "balanced_accuracy":
            title += " — balanced accuracy"
        ax.set_title(title, loc="left", fontsize=13, pad=27)
        ax.text(0, 1.02, f"Mean across seeds 0, 1, 2 · {recipe}",
                transform=ax.transAxes, fontsize=8, color="#444444")
        ax.set_ylabel(axis_label)
        ax.set_xticks(range(len(methods)), labels)
        ax.tick_params(axis="x", length=0, pad=8, labelsize=8)
        ax.set_ylim(0, upper_limit)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        fig.savefig(output / f"{family}_{chart}.png", dpi=300,
                    facecolor="white", transparent=False)
        plt.close(fig)


def build_report(results_root, output, overwrite=False):
    results_root, output = results_root.resolve(), output.resolve()
    if output == results_root or any(output.is_relative_to(results_root / family) for family in METHODS):
        raise ValueError("Output must be separate from the imported source results")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Report directory exists: {output}; use --overwrite to regenerate")
    summaries = summarize(load_records(results_root))
    import matplotlib
    matplotlib.use("Agg")
    output.mkdir(parents=True, exist_ok=True)
    for family in METHODS:
        write_tables(family, summaries[family], output)
    for chart, (key, _) in CHARTS.items():
        # Each outcome uses the same zero-based scale for both method families.
        largest_mean = max(
            values[key]["mean"]
            for family_summary in summaries.values() for values in family_summary.values()
        )
        upper_limit = 5 * math.ceil((largest_mean + 2) / 5)
        for family in METHODS:
            plot_metric_bars(family, summaries[family], chart, output, upper_limit)
    if overwrite:
        # Remove only exports owned by the previous reporting script.
        for name in LEGACY_OUTPUTS:
            (output / name).unlink(missing_ok=True)
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
