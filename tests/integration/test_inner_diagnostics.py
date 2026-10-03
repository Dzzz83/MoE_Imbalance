"""Integration checks for validated inner diagnostics, CLI guards, and outputs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from expert_method.config import load_study
from expert_method.diagnostics import DiagnosticsError, InnerArtifactReader, StudyDiagnosticsRunner
from expert_method.diagnostics.artifacts import (
    COMPATIBILITY_SCHEMA_VERSION,
    STUDY_ID,
    _require_same_reuse_rows,
    _validate_manager_protocol,
    _validate_stage_audit,
)
from expert_method.diagnostics.cli import main
from expert_method.diagnostics.reporting import (
    TABLE_FILES,
    _figure_payloads,
    _markdown,
    _sinkhorn_plot_series,
    write_outputs,
)
from expert_method.diagnostics.aggregation import (
    add_profile_rows,
    add_sample_entropy_aggregates,
    diagnostic_highlights,
)
from expert_method.diagnostics.complementarity import classify_predictions
from expert_method.diagnostics.router import analyze_router_weights
from expert_method.ridge_sinkhorn.matrix import canonical_json_bytes
from expert_method.ridge_sinkhorn.three_seed_study import (
    FoldStudyRunner,
    InferenceBatch,
    InnerOOFDataset,
    StudyEvaluator,
    StudyConfig,
)
from data.nested_oof import NestedOOFFoldManager
from scripts.task3f_ridge import classification_metrics


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/studies/ridge_sinkhorn_3seed_v1.yaml"


def _synthetic_dataset() -> InnerOOFDataset:
    rng = np.random.default_rng(721)
    sample_ids = np.arange(80, dtype=np.int64)
    labels = (np.arange(80, dtype=np.int64) * 7) % 6
    logits = rng.normal(size=(80, 4, 6))
    logits[np.arange(80), :, labels] += 0.7
    return InnerOOFDataset(
        logits=logits,
        labels=labels,
        sample_ids=sample_ids,
        inner_fold_ids=np.repeat(np.arange(4, dtype=np.int64), 20),
        training_seed=78,
        outer_fold_id=0,
        source_hashes=tuple((f"job-{index}", f"{index + 1:064x}") for index in range(16)),
    )


def _synthetic_manager() -> NestedOOFFoldManager:
    counts = (100, 20, 5, 5, 5, 5)
    labels = np.concatenate([
        np.full(count, class_id, dtype=np.int64) for class_id, count in enumerate(counts)
    ])
    return NestedOOFFoldManager(
        np.arange(len(labels), dtype=np.int64), labels, seed=42,
        outer_folds=5, inner_folds=4, num_classes=6,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )


def test_real_reader_rejects_corrupt_freeze_before_artifact_resolution(tmp_path: Path) -> None:
    artifact_root = tmp_path / "snapshot"
    study_root = artifact_root / STUDY_ID
    (study_root / "manifests").mkdir(parents=True)
    (study_root / "job_manifest.json").write_text("{}\n", encoding="utf-8")
    (study_root / "fold_manifest.json").write_text("{}\n", encoding="utf-8")
    (study_root / "manifests" / "study-freeze.json").write_text("{broken", encoding="utf-8")
    reader = InnerArtifactReader(config_path=CONFIG_PATH, artifact_root=artifact_root)

    with pytest.raises(DiagnosticsError, match="study-freeze.json"):
        reader.validate()
    assert set(artifact_root.rglob("*")) == {
        study_root, study_root / "manifests", study_root / "job_manifest.json",
        study_root / "fold_manifest.json", study_root / "manifests" / "study-freeze.json",
    }


def test_reader_uses_strict_executable_study_schema_for_full_config(tmp_path: Path) -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    altered = source.replace("training_samples: 10847", "training_samples: 10846")
    config = tmp_path / "changed.yaml"
    config.write_text(altered, encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="study configuration"):
        InnerArtifactReader(config_path=config, artifact_root=tmp_path / "missing")


def test_fold_manager_must_match_executable_protocol_population() -> None:
    protocol = dict(load_study(CONFIG_PATH).protocol)
    manager = _synthetic_manager()
    with pytest.raises(DiagnosticsError, match="canonical population"):
        _validate_manager_protocol(manager, protocol)


def test_diagnostics_reuses_canonical_metrics_and_keeps_sample_macro_separate() -> None:
    counts = np.asarray((100, 20, 5, 5, 5, 5), dtype=np.int64)
    labels = np.asarray((0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 5), dtype=np.int64)
    predictions = labels.copy()
    predictions[[0, 6, 7, 9, 11]] = (1, 0, 0, 0, 0)
    expected = classification_metrics(labels, predictions, counts)
    actual = classify_predictions(labels, predictions, counts)
    for key in (
        "ordinary_accuracy", "balanced_accuracy", "head_accuracy",
        "medium_accuracy", "tail_accuracy",
    ):
        assert actual[key] == pytest.approx(expected[key])
    assert actual["ordinary_accuracy"] != actual["balanced_accuracy"]
    incomplete = classify_predictions(labels[:-1], predictions[:-1], counts)
    assert incomplete["ordinary_accuracy"] is not None
    assert incomplete["balanced_accuracy"] is None
    assert incomplete["tail_accuracy"] is None


def test_native_inner_audit_identity_is_validated(tmp_path: Path) -> None:
    audit_path = tmp_path / "reuse_compatibility_inner.json"
    invalid = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "study_freeze_sha256": "b" * 64,
        "stage": "outer",
    }
    audit_path.write_bytes(canonical_json_bytes(invalid))
    planner = SimpleNamespace(native_store=SimpleNamespace(base_dir=tmp_path))
    with pytest.raises(DiagnosticsError, match="identity"):
        _validate_stage_audit(planner, {"freeze_sha256": "a" * 64})


def test_reuse_decision_error_names_audit_job_and_expected_source_identity(tmp_path: Path) -> None:
    audit_path = tmp_path / "reuse_compatibility_inner.json"
    frozen = [{
        "target_job_id": "ce_s78_o0_i0",
        "source_experiment_id": "task3b_pilot_ce_s78_o0_i0",
        "source_root_name": "rs3",
        "status": "compatible",
    }]
    resolved = [{
        "target_job_id": "ce_s78_o0_i0",
        "source_experiment_id": "task3c_oof",
        "source_root_name": "local-oof",
        "status": "compatible",
    }]

    with pytest.raises(DiagnosticsError) as caught:
        _require_same_reuse_rows(audit_path, frozen, resolved)
    message = str(caught.value)
    assert str(audit_path) in message
    assert "ce_s78_o0_i0" in message
    assert "expected source_experiment_id='task3b_pilot_ce_s78_o0_i0'" in message
    assert "source_root_name='rs3'" in message
    assert "resolved source_experiment_id='task3c_oof'" in message
    assert "source_root_name='local-oof'" in message


def test_validate_only_cli_does_not_create_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_root = tmp_path / "missing_snapshot"
    output_root = tmp_path / "must_not_exist"
    status = main([
        "--config", str(CONFIG_PATH), "--artifact-root", str(artifact_root),
        "--output-root", str(output_root), "--validate-only",
    ])
    assert status == 2
    assert not output_root.exists()
    assert "job_manifest.json" in capsys.readouterr().err


def test_output_location_rejects_figures_symlink_into_input_before_writing(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    sentinel = input_root / "preserve.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    before = {path.relative_to(input_root): path.read_bytes() for path in input_root.rglob("*") if path.is_file()}
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "figures").symlink_to(input_root, target_is_directory=True)
    reader = SimpleNamespace(artifact_root=input_root, reuse_roots={})
    runner = StudyDiagnosticsRunner(reader, output_root=output_root, make_figures=True)

    with pytest.raises(DiagnosticsError, match="symlink"):
        runner._validate_output_location()
    after = {path.relative_to(input_root): path.read_bytes() for path in input_root.rglob("*") if path.is_file()}
    assert after == before


def test_writer_rejects_figure_symlink_before_any_output_file_is_written(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "preserve.txt").write_text("unchanged\n", encoding="utf-8")
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "figures").symlink_to(input_root, target_is_directory=True)
    tables = {name: [] for name in TABLE_FILES}

    with pytest.raises(DiagnosticsError, match="symlink"):
        write_outputs(
            output_root, tables=tables, summary={}, provenance={}, make_figures=True,
        )

    assert sorted(path.name for path in output_root.iterdir()) == ["figures"]
    assert (input_root / "preserve.txt").read_text(encoding="utf-8") == "unchanged\n"


def test_real_locked_inference_orchestrator_keeps_evidence_tables_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real runner/evaluator/schema path on one small locked pair."""
    dataset = _synthetic_dataset()
    config = StudyConfig()
    counts = np.asarray((100, 20, 5, 5, 5, 5), dtype=np.int64)
    lock = FoldStudyRunner(config).lock(
        dataset, counts, plan_sha256="a" * 64, source_commit="b" * 40,
    )
    analysis_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert lock.source_commit != analysis_commit
    manager = SimpleNamespace(canonical_class_counts=tuple(counts.tolist()))
    inputs = SimpleNamespace(
        manager=manager,
        locks={(78, 0): lock},
        validation_counts={"locks": 15, "inner_jobs": 240, "native_inner_jobs": 224, "historical_inner_jobs": 16},
        manifest={"source_commit": "b" * 40},
        manifest_sha256="c" * 64,
        snapshot_commit="d" * 40,
        config_file_sha256="e" * 64,
    )

    class ReaderStub:
        def __init__(self) -> None:
            self.artifact_root = tmp_path / "inputs"
            self.reuse_roots = {}
            self.config = config
            self.job_lookup = {}
        def validate(self):
            return inputs

    reader = ReaderStub()
    runner = StudyDiagnosticsRunner(
        reader, output_root=tmp_path / "diagnostics", make_figures=False,
    )
    monkeypatch.setattr(
        "expert_method.diagnostics.runner.assemble_inner_dataset",
        lambda *_args: dataset,
    )
    monkeypatch.setattr(
        runner, "_provenance",
        lambda *_args, **_kwargs: {"schema_version": "synthetic-test"},
    )
    result = runner.run(limit_pairs=1)

    assert result["pairs_analyzed"] == 1
    root = Path(result["output_root"])
    assert set(TABLE_FILES.values()).issubset({path.name for path in root.glob("*.csv")})
    summary = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    assert summary["complete_matrix"] is False
    assert summary["diagnostic_highlights"]["any_expert_correct_fraction"] is None
    assert "1 / 15; complete matrix: false" in (root / "report.md").read_text(encoding="utf-8")
    assert summary["evidence_labels"]["expert_oof"] == "saved inner OOF predictions"
    assert summary["evidence_labels"]["selection_cv"].startswith("metrics recorded")
    selection_rows = (root / "selection_cv_metrics.csv").read_text(encoding="utf-8")
    fit_rows = (root / "router_fit_set_metrics.csv").read_text(encoding="utf-8")
    assert "recorded_selection_used_inner_cv" in selection_rows
    assert "saved_router_fit_set_prediction" in fit_rows
    prior_rows = (root / "prior_diagnostics.csv").read_text(encoding="utf-8")
    assert "adaptive_vs_selected_prior_only_scoped_predictions" in prior_rows
    assert "mean_weight_counterfactual_scoped_transition" in prior_rows
    sinkhorn_rows = (root / "sinkhorn_adjustments.csv").read_text(encoding="utf-8")
    assert "scoped_summary" in sinkhorn_rows
    assert "selective_gate" in sinkhorn_rows
    assert "selective_blend" in sinkhorn_rows

    first_hashes = result["output_hashes"]
    rerun = runner.run(limit_pairs=1)
    assert rerun["output_hashes"] == first_hashes
    (root / "report.md").write_text("modified\n", encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="refusing to overwrite"):
        runner.run(limit_pairs=1)


def test_saved_locked_predictions_follow_reordered_sample_ids() -> None:
    dataset = _synthetic_dataset()
    counts = np.asarray((100, 20, 5, 5, 5, 5), dtype=np.int64)
    lock = FoldStudyRunner(StudyConfig()).lock(
        dataset, counts, plan_sha256="a" * 64, source_commit="c" * 40,
    )
    evaluator = StudyEvaluator(StudyConfig())
    original = InferenceBatch(dataset.logits, dataset.sample_ids)
    predictions, weights = evaluator.predict_locked_methods(lock, original)
    order = np.arange(len(dataset.sample_ids))[::-1]
    reordered_predictions, reordered_weights = evaluator.predict_locked_methods(
        lock, InferenceBatch(dataset.logits[order], dataset.sample_ids[order]),
    )
    inverse = np.argsort(order)
    for method in predictions:
        np.testing.assert_array_equal(predictions[method], reordered_predictions[method][inverse])
        np.testing.assert_allclose(weights[method], reordered_weights[method][inverse], rtol=0, atol=1e-12)


def test_all_five_figures_render_deterministically() -> None:
    pytest.importorskip("matplotlib")
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    for count in range(5):
        tables["complementarity"].append({
            "scope": "mean_across_seeds", "kind": "correct_count_distribution",
            "num_correct_experts": count, "mean_fraction_across_seeds": 0.2,
        })
    for expert in ("CE", "LAL", "BalancedSoftmax", "Mixup"):
        tables["per_class_specialization"].append({
            "scope": "mean_across_seeds", "class_id": 0, "expert": expert,
            "class_recall": 0.5,
        })
    for method in ("contribution_ridge_sinkhorn", "residual_ridge_sinkhorn",
                   "selective_residual_ridge_sinkhorn"):
        for expert in ("CE", "LAL", "BalancedSoftmax", "Mixup"):
            tables["router_weight_profiles"].append({
                "scope": "mean_across_seeds", "method": method, "expert": expert,
                "mean_weight_across_seeds": 0.25, "normalized_allocation_entropy": 1.0,
                "mean_normalized_sample_entropy": 0.8,
            })
    for method in ("contribution_ridge_sinkhorn", "residual_ridge_sinkhorn"):
        for stage in ("raw", "smoothed", "adjusted"):
            for expert in ("CE", "LAL", "BalancedSoftmax", "Mixup"):
                tables["sinkhorn_adjustments"].append({
                    "scope": "mean_across_seeds", "method": method, "stage": stage,
                    "expert": expert, "mean_weight_across_seeds": 0.25,
                })
    for expert in ("CE", "LAL", "BalancedSoftmax", "Mixup"):
        tables["sinkhorn_adjustments"].append({
            "scope": "mean_across_seeds", "method": "selective_residual_ridge_sinkhorn",
            "stage": "selective_blend", "expert": expert, "mean_weight_across_seeds": 0.25,
        })
    tables["stability"].append({
        "comparison_scope": "mean_across_seeds", "method": "residual_ridge",
        "mean_l1_distance": 0.2,
    })
    first = _figure_payloads(tables)
    second = _figure_payloads(tables)
    assert set(first) == {
        "complementarity.png", "complementarity.svg",
        "specialization.png", "specialization.svg",
        "allocation_entropy.png", "allocation_entropy.svg",
        "stability.png", "stability.svg",
        "sinkhorn_adjustments.png", "sinkhorn_adjustments.svg",
    }
    assert first == second


def test_selective_sinkhorn_plot_series_uses_selective_blend_rows() -> None:
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    for expert, value in zip(("CE", "LAL", "BalancedSoftmax", "Mixup"), (0.1, 0.2, 0.3, 0.4)):
        tables["sinkhorn_adjustments"].append({
            "scope": "mean_across_seeds", "method": "selective_residual_ridge_sinkhorn",
            "stage": "selective_blend", "expert": expert,
            "mean_weight_across_seeds": value,
        })
    plotted = _sinkhorn_plot_series(tables)
    assert plotted[("selective_residual_ridge_sinkhorn", "selective_blend")] == [0.1, 0.2, 0.3, 0.4]


def test_smoke_plot_payloads_skip_undefined_aggregates() -> None:
    pytest.importorskip("matplotlib")
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    for count in range(5):
        tables["complementarity"].append({
            "scope": "mean_across_seeds", "kind": "correct_count_distribution",
            "num_correct_experts": count, "mean_fraction_across_seeds": None,
        })
    for method in ("contribution_ridge_sinkhorn", "residual_ridge_sinkhorn"):
        for expert in ("CE", "LAL", "BalancedSoftmax", "Mixup"):
            tables["router_weight_profiles"].append({
                "scope": "mean_across_seeds", "method": method, "expert": expert,
                "mean_weight_across_seeds": None, "normalized_allocation_entropy": None,
            })
        for stage in ("raw", "smoothed", "adjusted"):
            for expert in ("CE", "LAL", "BalancedSoftmax", "Mixup"):
                tables["sinkhorn_adjustments"].append({
                    "scope": "mean_across_seeds", "method": method, "stage": stage,
                    "expert": expert, "mean_weight_across_seeds": None,
                })
    tables["stability"].append({
        "comparison_scope": "mean_across_seeds", "method": "residual_ridge",
        "mean_l1_distance": None,
    })
    assert _figure_payloads(tables) == {}


def test_markdown_reports_smoke_matrix_coverage() -> None:
    report = _markdown({
        "pairs_analyzed": 1, "expected_pair_count": 15, "complete_matrix": False,
        "validation_counts": {}, "descriptive_metrics": [], "diagnostic_highlights": {},
    })
    assert "Diagnostic pairs analyzed: 1 / 15; complete matrix: false." in report


def test_highlights_use_executable_seed_ids_and_nested_full_matrix_means() -> None:
    expected_pairs = [(seed, fold) for seed in (78, 88, 1034) for fold in range(5)]
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    for seed, fold in expected_pairs:
        identity = {"training_seed": seed, "outer_fold_id": fold, "scope": "outer_fold"}
        tables["expert_oof_metrics"].append({**identity, "method": "CE", "metric": "BA", "value": 0.5})
        tables["complementarity"].append({
            **identity, "kind": "correctness_summary", "any_expert_correct_fraction": 0.7,
            "all_experts_wrong_fraction": 0.2,
        })
        tables["oracle_opportunity"].append({
            **identity, "metric": "balanced_accuracy", "oracle": 0.8,
            "oracle_minus_strongest_individual": 0.1,
        })
        tables["per_class_specialization"].append({
            **identity, "expert": "CE", "class_id": 0,
            "pairwise_prediction_disagreement_fraction": 0.3,
        })
        tables["router_weight_profiles"].append({
            **identity, "expert": "CE", "method": "residual_ridge",
            "normalized_allocation_entropy": 0.6,
        })
        tables["prior_diagnostics"].extend([
            {**identity, "diagnostic": "adaptive_mean_vs_selected_prior_only", "l1_distance": 0.1},
            {**identity, "diagnostic": "adaptive_mean_vs_sinkhorn_target_prior",
             "max_abs_marginal_residual": 0.02},
        ])
        tables["sinkhorn_adjustments"].append({
            **identity, "stage": "selective_gate", "group": "overall", "gate_mean": 0.4,
        })
    tables["stability"].append({
        "comparison_scope": "mean_across_seeds", "method": "residual_ridge",
        "mean_l1_distance": 0.25,
    })
    result = diagnostic_highlights(tables, expected_pairs)
    assert result["complete_matrix"] is True
    assert result["any_expert_correct_fraction"] == pytest.approx(0.7)
    assert result["hard_oracle_balanced_accuracy"] == pytest.approx(0.8)


def test_sample_entropy_copy_respects_outer_fold_and_group_profiles_export_ties() -> None:
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    for fold, value in ((0, 0.2), (1, 0.8)):
        for expert in ("CE", "LAL"):
            tables["router_class_profiles"].append({
                "scope": "outer_fold", "training_seed": 78, "outer_fold_id": fold,
                "method": "method", "class_id": 3, "expert": expert,
                "mean_normalized_sample_entropy": value if expert == "CE" else None,
            })
    add_sample_entropy_aggregates(tables, [(78, 0), (78, 1)])
    lal_rows = [row for row in tables["router_class_profiles"] if row["expert"] == "LAL"]
    assert [row["mean_normalized_sample_entropy"] for row in lal_rows] == [0.2, 0.8]

    labels = np.arange(100, dtype=np.int64)
    weights = np.full((100, 4), 0.25, dtype=np.float64)
    logits = np.zeros((100, 4, 100), dtype=np.float64)
    report = analyze_router_weights(
        logits, labels, {"uniform": weights},
        np.asarray([100] * 30 + [20] * 30 + [5] * 40, dtype=np.int64),
    )
    add_profile_rows(tables, pair_key={"training_seed": 78, "outer_fold_id": 0}, router_report=report)
    head = [row for row in tables["router_group_allocations"] if row.get("group") == "head"]
    assert head
    assert all(row["preferred_expert_tie_count"] == 4 for row in head)
    assert all(row["class_macro_preferred_expert_tie_count"] == 4 for row in head)


def test_sample_entropy_index_preserves_nested_aggregates_and_row_order() -> None:
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    seeds = (78, 88, 1034)
    expected_seed_means = {78: 0.2, 88: 0.3, 1034: 0.4}
    for seed_index, seed in enumerate(seeds):
        for fold in range(5):
            value = 0.1 + seed_index * 0.1 + fold * 0.05
            for expert in ("CE", "LAL"):
                tables["router_class_profiles"].append({
                    "scope": "outer_fold", "training_seed": seed, "outer_fold_id": fold,
                    "method": "method", "class_id": 3, "expert": expert,
                    "mean_normalized_sample_entropy": value if expert == "CE" else None,
                })
        for scope, row_seed in (("seed_mean", seed), ("mean_across_seeds", None)):
            for expert in ("CE", "LAL"):
                tables["router_class_profiles"].append({
                    "scope": scope, "training_seed": row_seed, "outer_fold_id": None,
                    "method": "method", "class_id": 3, "expert": expert,
                    "mean_normalized_sample_entropy": None,
                })
    before = [tuple(row.items()) for row in tables["router_class_profiles"]]

    add_sample_entropy_aggregates(
        tables,
        [(seed, fold) for seed in seeds for fold in range(5)],
    )

    rows = tables["router_class_profiles"]
    assert [tuple(row.items())[:6] for row in rows] == [entry[:6] for entry in before]
    for seed, expected in expected_seed_means.items():
        for expert in ("CE", "LAL"):
            row = next(row for row in rows if row["scope"] == "seed_mean"
                       and row["training_seed"] == seed and row["expert"] == expert)
            assert row["mean_normalized_sample_entropy"] == pytest.approx(expected)
    for expert in ("CE", "LAL"):
        row = next(row for row in rows if row["scope"] == "mean_across_seeds" and row["expert"] == expert)
        assert row["mean_normalized_sample_entropy"] == pytest.approx(0.3)
        assert row["sample_sd_normalized_sample_entropy_across_seeds"] == pytest.approx(0.1)
