import csv
import json

import pytest

from pipeline.methods.synthesis import resolve_options
from pipeline.run_comparison import generation_variants, write_tables


def test_controls_change_only_target_policy_and_guidance():
    base = {"seed": 0, "feature_cache": "cache.pt", "output_dir": "unused",
            "method_options": {"entropy_coefficient": .0005, "max_iterations": 1000}}
    original = json.loads(json.dumps(base))
    variants = generation_variants(base)
    assert base == original
    expected = resolve_options(base["method_options"])
    for name, config in variants.items():
        options = config["method_options"]
        assert options["guidance_weight"] == 0
        assert options["uniform_targets"] == (name == "uniform_ot")
        for key in expected.keys() - {"guidance_weight", "uniform_targets"}:
            assert options[key] == expected[key]
        assert config["seed"] == base["seed"]
        assert config["feature_cache"] == base["feature_cache"]


def metric_record(name, accuracy):
    return {
        "name": name, "checkpoint": f"{name}/final.pt",
        "metrics": {
            "accuracy": accuracy, "balanced_accuracy": accuracy,
            "head_recall": accuracy, "medium_recall": accuracy, "tail_recall": accuracy,
            "per_class_recall": [accuracy] * 100, "test_support": [100] * 100,
        },
    }


def test_percentage_units_differences_and_real_count_groups(tmp_path):
    records = [metric_record("baseline", .5), metric_record("full_ot", .55)]
    counts = [500, 50] + [5] * 98
    write_tables(records, counts, tmp_path)
    with (tmp_path / "comparison.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert float(rows[1]["accuracy_percent"]) == pytest.approx(55)
    assert float(rows[1]["accuracy_delta_pp"]) == pytest.approx(5)
    assert float(rows[1]["tail_recall_delta_pp"]) == pytest.approx(5)
    with (tmp_path / "per_class_recall.csv").open(newline="") as source:
        classes = list(csv.DictReader(source))
    assert [row["group"] for row in classes[:3]] == ["Head", "Medium", "Tail"]
    assert len(classes) == 100
    assert float(classes[99]["full_ot_recall_percent"]) == pytest.approx(55)


def test_different_test_populations_are_rejected(tmp_path):
    records = [metric_record("baseline", .5), metric_record("full_ot", .55)]
    records[1]["metrics"]["test_support"][0] = 99
    with pytest.raises(ValueError, match="same test population"):
        write_tables(records, [5] * 100, tmp_path)
