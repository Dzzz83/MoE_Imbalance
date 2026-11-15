import copy
from pathlib import Path

import pytest

from scripts import report_results as report


def valid_metrics():
    return {
        "accuracy": 0.5, "balanced_accuracy": 0.5,
        "head_recall": 0.5, "medium_recall": 0.5, "tail_recall": 0.5,
        "per_class_recall": [0.5] * 100, "test_support": [100] * 100,
    }


def test_paired_difference_sd_is_not_the_sd_of_absolute_scores():
    rows = [
        {"seed": seed, "accuracy_percent": score, "accuracy_delta_pp": 2.0}
        for seed, score in enumerate((12.0, 22.0, 32.0))
    ]
    summary = report.summarize_method(rows)
    assert summary["accuracy_percent"] == {"mean": 22.0, "sd": 10.0}
    assert summary["accuracy_delta_pp"] == {"mean": 2.0, "sd": 0.0}
    with pytest.raises(ValueError, match="exactly one result"):
        report.summarize_method([rows[0], rows[1], rows[1]])


@pytest.mark.parametrize("change", [
    {"accuracy": None}, {"tail_recall": float("nan")},
    {"head_recall": float("inf")}, {"medium_recall": -0.1},
    {"test_support": [100] * 99}, {"per_class_recall": [None] * 100},
    {"balanced_accuracy": 0.6}, {"accuracy": 0.6},
])
def test_missing_or_inconsistent_metrics_are_rejected(change):
    with pytest.raises(ValueError):
        report.validate_metrics(valid_metrics() | change, "example.json")


def test_archived_results_have_three_seeds_and_no_legacy_duplicate():
    root = Path(__file__).resolve().parents[1] / "results"
    records = report.load_records(root)
    assert len(records) == 36
    assert all("/comparison_seed0/" not in row["source_file"] for row in records)
    summaries = report.summarize(records)
    assert summaries["reweighting"]["class_balanced"]["accuracy_percent"]["mean"] == pytest.approx(38.9933333333)
    assert summaries["reweighting"]["sava"]["accuracy_delta_pp"]["mean"] == pytest.approx(0.2933333333)
    assert summaries["synthesis"]["full_ot"]["accuracy_delta_pp"]["mean"] == pytest.approx(1.88)
    for family, methods in report.METHODS.items():
        for method, _ in methods:
            rows = [row for row in records if row["family"] == family and row["method"] == method]
            assert sorted(row["seed"] for row in rows) == [0, 1, 2]
            for row in rows:
                baseline = next(r for r in records if r["family"] == family
                                and r["method"] == "baseline" and r["seed"] == row["seed"])
                assert row["tail_recall_delta_pp"] == pytest.approx(
                    row["tail_recall_percent"] - baseline["tail_recall_percent"]
                )


def test_table_marks_ties_and_labels_baseline_changes(tmp_path):
    metric_summary = {
        key: {"mean": 50.0, "sd": 1.0}
        for key, _ in report.COLUMNS
    }
    summary = {method: copy.deepcopy(metric_summary) for method, _ in report.METHODS["reweighting"]}
    report.write_tables("reweighting", summary, tmp_path)
    markdown = (tmp_path / "reweighting.md").read_text()
    assert markdown.count("**50.00 ± 1.00**") == 24
    assert "| Baseline |" in markdown and "| — |" in markdown
    assert "not indicate statistical significance" in markdown
    assert r"\boldsymbol{50.00 \pm 1.00}" in (tmp_path / "reweighting.tex").read_text()


def test_output_cannot_replace_raw_results_or_implicitly_overwrite(tmp_path):
    root = tmp_path / "results"
    for output in (root, root / "reweighting/report", root / "synthesis/report"):
        with pytest.raises(ValueError, match="separate from the imported"):
            report.build_report(root, output, overwrite=True)
        assert not output.exists()
    existing = tmp_path / "paper"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="--overwrite"):
        report.build_report(root, existing)
    assert sentinel.read_text() == "keep"


def test_report_exports_only_tables_and_pngs_and_preserves_unrelated_files(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    root = Path(__file__).resolve().parents[1] / "results"
    for name in report.LEGACY_OUTPUTS:
        (tmp_path / name).write_text("previous export")
    (tmp_path / "keep.txt").write_text("unrelated")
    chart_limits = []

    def save_chart(family, summary, chart, output, upper_limit):
        chart_limits.append((chart, upper_limit))
        (output / f"{family}_{chart}.png").write_bytes(b"chart")

    monkeypatch.setattr(report, "plot_metric_bars", save_chart)
    report.build_report(root, tmp_path, overwrite=True)
    expected = {f"{family}.{extension}" for family in report.METHODS for extension in ("md", "tex", "csv")}
    expected |= {f"{family}_{chart}.png" for family in report.METHODS for chart in ("accuracy", "tail_accuracy")} | {"keep.txt"}
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert (tmp_path / "keep.txt").read_text() == "unrelated"
    assert chart_limits == [("accuracy", 45), ("accuracy", 45), ("tail_accuracy", 20), ("tail_accuracy", 20)]
