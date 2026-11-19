import copy
import hashlib
import io
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from pipeline import ablation_inputs, ablation_references as references, run_simple_controls as controls
from pipeline import run_weight_study as assignment
from pipeline.ablation_weights import transform_weights
from pipeline.methods import prototype_weights, reweighting
from pipeline.runtime import file_sha256, write_json
from pipeline.simple_control_weights import CONDITIONS, build_controls, replay_geometry
from scripts import kaggle_ablation_study as kaggle
from test_weight_study import audit_inputs  # Reuse the published-input fixture and its validation.


@pytest.fixture
def stage_b_inputs(audit_inputs, monkeypatch):
    audit_path = audit_inputs / "audit/weight_audit.json"
    audit = json.loads(audit_path.read_text())
    for seed in (0, 1, 2):
        cache = audit_inputs / f"caches/seed{seed}/features_weights.pt"
        bundle = torch.load(cache, weights_only=True)
        generator = torch.Generator().manual_seed(seed)
        bundle["features"][:12] = torch.randn(12, 64, generator=generator) * 0.01
        bundle["features"][:5, 0] += 1
        bundle["features"][5:12, 1] += 1
        bundle["ot_weights"], bundle["prototype_diagnostics"] = prototype_weights.build_weights(
            bundle["features"], bundle["labels"],
            {"seed": seed, "method_options": bundle["inputs"]["prototype_options"]}, torch.device("cpu"))
        assert bundle["prototype_diagnostics"][0]["used_ot"]
        torch.save(bundle, cache)
        audit["feature_caches"][str(seed)] = file_sha256(cache)
    write_json(audit_path, audit)
    monkeypatch.setitem(ablation_inputs.AUDIT_REPORT_HASHES, "weight_audit.json", file_sha256(audit_path))
    # Artificial integration uses one epoch; production settings have no override.
    monkeypatch.setattr(references, "ASSIGNMENT_CONFIG", assignment.resolve_config({"classifier": {"epochs": 1}}))
    return audit_inputs


@pytest.fixture
def stage_b_parent(stage_b_inputs, tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    reports = parent / "reports"
    assignment.run_study(references.ASSIGNMENT_CONFIG, stage_b_inputs, reports, "cpu")
    config = json.loads((reports / "study_config.json").read_text())
    config["implementation"].update(references.PARENT_EXTENDED_MODULES)
    write_json(reports / "study_config.json", config)
    write_json(reports / "study_assets.json", {
        "complete": {"sha256": references.PARENT_ARCHIVE_SHA256, "asset_id": 1,
                     "paths": kaggle.assignment_asset_paths(reports)}, "seeds": {}})
    manifest = {
        "source_commit": references.PARENT_SOURCE_COMMIT,
        "report_hashes": {p.relative_to(reports).as_posix(): file_sha256(p) for p in reports.rglob("*.json")},
    }
    write_json(parent / "manifest.json", manifest)
    monkeypatch.setattr(references, "PARENT_MANIFEST_SHA256", file_sha256(parent / "manifest.json"))
    return parent


def test_geometry_and_five_rules_preserve_masses_ties_and_fallbacks(stage_b_inputs):
    inputs = ablation_inputs.load_audit_inputs(stage_b_inputs, controls.resolve_config({}))
    bundle = inputs["bundles"][0]
    before = torch.random.get_rng_state().clone()
    geometry = replay_geometry(bundle, 0)
    assert torch.equal(before, torch.random.get_rng_state())
    strong = transform_weights(bundle["ot_weights"], bundle["labels"], "strong_ot")
    scores = {"margins": torch.tensor([-2., -1., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.] + [0.] * 396),
              "tie_order": torch.arange(408).flip(0)}
    weights, diagnostics = build_controls(bundle, geometry, scores, strong)
    sizes = torch.bincount(geometry[0]["assignments"])
    assert sorted(sizes.tolist()) == [5, 7]
    expected = 1 / (200 * sizes[geometry[0]["assignments"]].double())
    assert torch.allclose(weights["cluster_balanced"][:12].double(), expected)
    assert weights["easy_rank"][1] == weights["easy_rank"][2]
    assert weights["easy_rank"][11] / weights["easy_rank"][0] == pytest.approx(4)
    distance = (1 - torch.nn.functional.normalize(bundle["features"][:12], dim=1)
                @ geometry[0]["centers"].T).clamp_min(0).min(1).values
    assert torch.allclose(weights["prototype_distance"][:12], torch.softmax(-distance.double() / .1, 0).float() / 100)
    for name, values in weights.items():
        assert torch.equal(values[12:], strong[12:])
        assert torch.allclose(values[:12].sum(), torch.tensor(.01))
        if name.endswith("rank_strong"):
            assert torch.equal(values[:12].sort().values, strong[:12].sort().values)
    assert diagnostics["priority_counts"]["cluster_rank_strong"]["active_classes_with_ties"] == 1
    uniform_scores = scores | {"margins": torch.zeros(408)}
    tied, _ = build_controls(bundle, geometry, uniform_scores, strong)
    assert torch.equal(tied["easy_rank"][:12], tied["easy_rank"][0].expand(12))
    damaged = copy.deepcopy(bundle)
    damaged["prototype_diagnostics"][0]["cluster_sizes"] = [6, 6]
    with pytest.raises(ValueError, match="replay mismatch"):
        replay_geometry(damaged, 0)
    damaged = copy.deepcopy(bundle)
    damaged["ot_weights"][0] *= 2
    with pytest.raises(ValueError, match="original OT weights"):
        replay_geometry(damaged, 0)
    with pytest.raises(ValueError, match="Invalid saved margins"):
        build_controls(bundle, geometry, scores | {"margins": scores["margins"] * float("nan")}, strong)
    with pytest.raises(ValueError, match="permutation"):
        build_controls(bundle, geometry, scores | {"tie_order": torch.zeros(408, dtype=torch.int64)}, strong)


def test_equal_clusters_and_tied_distances_have_constant_priority(stage_b_inputs):
    inputs = ablation_inputs.load_audit_inputs(stage_b_inputs, controls.resolve_config({}))
    bundle = copy.deepcopy(inputs["bundles"][0])
    bundle["features"][:12] = 0
    bundle["features"][:6, 0] = 1
    bundle["features"][6:12, 1] = 1
    bundle["ot_weights"], bundle["prototype_diagnostics"] = prototype_weights.build_weights(
        bundle["features"], bundle["labels"],
        {"seed": 0, "method_options": bundle["inputs"]["prototype_options"]}, torch.device("cpu"))
    strong = transform_weights(bundle["ot_weights"], bundle["labels"], "strong_ot")
    weights, report = build_controls(bundle, replay_geometry(bundle, 0),
        {"margins": torch.zeros(408), "tie_order": torch.arange(408)}, strong)
    for name in ("cluster_rank_strong", "distance_rank_strong"):
        assert report["priority_counts"][name]["active_classes_with_constant_priority"] == 1
    for name in CONDITIONS:
        assert torch.equal(weights[name][:12], weights[name][0].expand(12))


def test_parent_validation_rejects_missing_changed_and_incompatible_references(stage_b_inputs, stage_b_parent, monkeypatch):
    inputs = ablation_inputs.load_audit_inputs(stage_b_inputs, controls.resolve_config({}))
    parent = references.load_parent(stage_b_parent, inputs)
    assert len(parent["records"]) == 30
    with pytest.raises(ValueError, match="environment mismatch"):
        references.validate_environment(parent["environment"] | {"device_name": "different GPU"}, "cpu")
    provenance = json.loads((stage_b_parent / "reports/study_config.json").read_text())
    provenance["implementation"]["model.py"] = "0" * 64
    with pytest.raises(ValueError, match="Shared training implementation"):
        references.validate_shared_implementation(provenance)
    saved = references.PARENT_FUNCTION_HASHES
    monkeypatch.setattr(references, "PARENT_FUNCTION_HASHES",
                        saved | {"run_weight_study.py": {"_continuation": "0" * 64}})
    with pytest.raises(ValueError, match="Shared training functions"):
        references.validate_shared_implementation(json.loads((stage_b_parent / "reports/study_config.json").read_text()))
    monkeypatch.setattr(references, "PARENT_FUNCTION_HASHES", saved)
    binary = stage_b_parent / "reports/runs/seed0/strong_ot/weights.pt"
    original = binary.read_bytes()
    binary.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Completed run file changed"):
        references.load_parent(stage_b_parent, inputs)
    binary.write_bytes(original)
    receipt = stage_b_parent / "reports/runs/seed2/hard_strong/complete.json"
    receipt.unlink()
    with pytest.raises(ValueError, match="report missing or changed"):
        references.load_parent(stage_b_parent, inputs)


@pytest.mark.parametrize("override", [
    {"classifier": {"learning_rate": .02}}, {"seeds": [0]}, {"batch_size": 64},
    {"parent_results_commit": "a" * 40}, {"parent_archive_sha256": "b" * 64},
    {"tie_seed": True}, {"distance_temperature": .2},
])
def test_stage_b_rejects_settings_outside_frozen_plan(override):
    with pytest.raises(ValueError):
        assignment.resolve_config({"stage": "simple_controls"} | override)


def test_paired_reporting_averages_shuffles_and_highlights_one_image_changes():
    records = []
    config = controls.resolve_config({})
    for seed in (0, 1, 2):
        for condition, shuffle in [("baseline", None)] + assignment.conditions(config) + [(c, None) for c in CONDITIONS]:
            recalls = [.4] * 99 + [0.]
            if condition == "cluster_balanced":
                recalls[0] += .6
            if condition == "prototype_distance":
                recalls[99] = 1.
            if shuffle is not None:
                recalls[0] += (shuffle - 1) * .1
            value = sum(recalls) / 100
            metrics = {name: value for name in controls.METRICS}
            metrics.update(per_class_recall=recalls, development_support=[100] * 99 + [1])
            records.append({"seed": seed, "condition": condition, "shuffle_seed": shuffle, "metrics": metrics})
    result = controls.summarize_controls(records, config, {s: [0] for s in (0, 1, 2)})
    assert result["primary_contrasts"]["ot_minus_cluster_balanced"]["balanced_accuracy"]["paired_difference_mean"] == pytest.approx(-.006)
    assert result["decision"]["exploratory_native_candidates"] == ["cluster_balanced", "prototype_distance"]
    shuffle = result["secondary_contrasts"]["strong_ot_minus_shuffled_strong_ot"]["balanced_accuracy"]
    assert shuffle["paired_difference_mean"] == pytest.approx(0)
    assert shuffle["training_seed_count"] == 3
    assert len(shuffle["per_seed"][0]["reference_individual_values"]) == 3
    classes = result["class_contributions"]["prototype_distance_minus_class_balanced"]
    assert classes[0]["one_image_class_contribution"] == pytest.approx(.01)
    assert classes[0]["fallback_contribution"] == pytest.approx(.01)
    assert classes[0]["classes"][99]["correct_count_mean"] == 1
    assert classes[0]["classes"][99]["reference_correct_count_mean"] == 0


def test_all_fifteen_continuations_reports_and_completed_reuse(stage_b_inputs, stage_b_parent, tmp_path, monkeypatch):
    root = tmp_path / "stage_b"
    config = controls.resolve_config({})
    records = assignment.run_study(config, stage_b_inputs, root, "cpu", parent_root=stage_b_parent)
    assert len(records) == 45 and sum(r["origin"] == "stage_b" for r in records) == 15
    summary = json.loads((root / "study_summary.json").read_text())
    assert summary["completed_runs"] == 15 and summary["status"] == "complete"
    assert len(summary["primary_contrasts"]) == 3
    assert len(summary["usefulness_contrasts"]) == 8
    assert len(summary["secondary_contrasts"]) == 6
    for contrast in summary["class_contributions"].values():
        for row in contrast:
            assert row["balanced_accuracy_difference"] == pytest.approx(row["ot_active_contribution"] + row["fallback_contribution"])
    baseline = torch.load(stage_b_inputs / "baselines/seed0/final.pt", weights_only=True)
    for condition in CONDITIONS:
        checkpoint = torch.load(root / f"runs/seed0/{condition}/final.pt", weights_only=True)
        for name, value in checkpoint["model_state_dict"].items():
            if not name.startswith("fc."):
                assert torch.equal(value, baseline["model_state_dict"][name])
    final = root / "runs/seed0/cluster_balanced/final.pt"
    digest = file_sha256(final)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    monkeypatch.setattr(reweighting, "train_classifier_from_features",
                        lambda *a, **kw: pytest.fail("Completed Stage B continuations must not retrain"))
    assert assignment.run_study(config, stage_b_inputs, relocated, "cpu", parent_root=stage_b_parent) == records
    assert file_sha256(final) == digest
    assert controls.summarize_controls(records[:-1], config, {0: [0], 1: [0], 2: [0]})["decision"]["status"] == "incomplete"
    with pytest.raises(ValueError, match="duplicate"):
        controls.summarize_controls(records + [records[-1]], config, {})
    geometry_path = relocated / "diagnostics/seed0/geometry.pt"
    geometry = torch.load(geometry_path, weights_only=True)
    geometry[0]["centers"][0, 0] += .1
    torch.save(geometry, geometry_path)
    with pytest.raises(ValueError, match="Saved Stage B geometry changed"):
        assignment.run_study(config, stage_b_inputs, relocated, "cpu", parent_root=stage_b_parent)


def test_stage_b_interruption_archive_and_publication_retry(stage_b_inputs, stage_b_parent, tmp_path, monkeypatch):
    config = controls.resolve_config({})
    root = tmp_path / "study"
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "workspace")
    with pytest.raises(RuntimeError, match="publication unavailable"):
        controls.run_study(config, stage_b_inputs, stage_b_parent, root, "cpu",
            on_progress=lambda *a: (_ for _ in ()).throw(RuntimeError("publication unavailable")))
    completed = root / "runs/seed0/cluster_balanced/final.pt"
    digest = file_sha256(completed)
    partial = root / "runs/seed0/prototype_distance"
    partial.mkdir()
    (partial / "partial.txt").write_text("interrupted")
    kaggle.archive_interrupted_runs(root)
    assert not partial.exists() and list((kaggle.WORK_ROOT / "interruptions").rglob("partial.txt"))
    controls.run_study(config, stage_b_inputs, stage_b_parent, root, "cpu")
    assert file_sha256(completed) == digest
    paths = kaggle.cached_asset_paths(root, None, kaggle.is_simple_controls_asset_path)
    assert "diagnostics/seed0/geometry.pt" in paths
    assert not kaggle.is_simple_controls_asset_path("runs/seed0/ot/final.pt")
    assert not kaggle.is_simple_controls_asset_path("../geometry.pt")
    archive = io.BytesIO()
    with ZipFile(archive, "w") as target:
        for path in paths:
            target.write(root / path, path)
    contents = archive.getvalue()
    write_json(root / "study_assets.json", {"complete": {
        "asset_id": 1, "sha256": hashlib.sha256(contents).hexdigest(), "paths": paths}, "seeds": {}})
    published = tmp_path / "published"
    for path in root.rglob("*.json"):
        destination = published / "reports" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(published / "manifest.json", {"report_hashes": {
        path.relative_to(published / "reports").as_posix(): file_sha256(path)
        for path in (published / "reports").rglob("*.json")}})
    monkeypatch.setattr(kaggle, "github_release_request", lambda *a, **kw: io.BytesIO(contents))
    restored = kaggle.restore_cached_study(published, tmp_path / "restored", {}, kaggle.is_simple_controls_asset_path)
    assert (restored / "diagnostics/seed0/geometry.pt").read_bytes() == (root / "diagnostics/seed0/geometry.pt").read_bytes()
    assert file_sha256(restored / "runs/seed0/cluster_balanced/final.pt") == digest


def test_kaggle_stage_b_preflight_and_dispatch_do_not_train_stage_a(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "CODE_ROOT", Path(__file__).resolve().parents[1])
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(kaggle, "prepare_assignment_inputs", lambda *a: tmp_path / "inputs")
    monkeypatch.setattr(kaggle, "prepare_simple_controls_parent", lambda *a: tmp_path / "parent")
    forbidden = lambda *a, **kw: pytest.fail("Stage B must not prepare images, train baselines or evaluate official test")
    monkeypatch.setattr(kaggle, "prepare_data", forbidden)
    monkeypatch.setattr("pipeline.run_ablation._baseline", forbidden)
    monkeypatch.setattr("pipeline.evaluate.evaluate_checkpoint", forbidden)
    monkeypatch.setattr("pipeline.runtime.resolve_device", lambda *a: torch.device("cpu"))
    calls = []
    monkeypatch.setattr(controls, "prepare_study", lambda *a: calls.append("preflight"))
    monkeypatch.setattr(assignment, "run_study", forbidden)
    monkeypatch.setattr(kaggle, "SIMPLE_CONTROLS_PREFLIGHT_ONLY", True)
    threads = torch.get_num_threads()
    try:
        kaggle.execute_simple_controls("a" * 40, {})
        assert calls == ["preflight"]
        monkeypatch.setattr(kaggle, "SIMPLE_CONTROLS_PREFLIGHT_ONLY", False)
        monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
        uploads = []

        def execute(config, inputs, root, device, on_progress, parent_root):
            assert config["stage"] == "simple_controls" and config["classifier"]["epochs"] == 40
            assert inputs == tmp_path / "inputs" and parent_root == tmp_path / "parent"
            run = root / "runs/seed0/cluster_balanced"
            write_json(run / "complete.json", {})
            (run / "final.pt").write_bytes(b"checkpoint")
            on_progress(root, 0, "first")
            on_progress(root, None, "complete")

        monkeypatch.setattr(assignment, "run_study", execute)
        monkeypatch.setattr(kaggle, "upload_artifacts", lambda root, paths, commit, env, prefix:
                            uploads.append((paths, prefix)) or {"asset_id": 1, "sha256": "a" * 64})
        monkeypatch.setattr(kaggle, "publish_reports", lambda *a: calls.append(a[2]))
        kaggle.execute_simple_controls("a" * 40, {})
        assert calls == ["preflight", "first", "complete"]
        assert [u[1] for u in uploads] == ["ot-simple-controls-seed0", "ot-simple-controls-complete"]
    finally:
        torch.set_num_threads(threads)
