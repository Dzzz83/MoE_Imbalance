import copy
import io
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from pipeline import ablation_inputs, run_weight_study as study
from pipeline.ablation_weights import assignment_audit, baseline_scores, reassign_weights, transform_weights
from pipeline.methods import baseline, prototype_weights, reweighting
from pipeline.model import create_model
from pipeline.runtime import file_sha256, load_checkpoint, save_checkpoint_with_metadata, write_json
from scripts import kaggle_ablation_study as kaggle


@pytest.fixture
def audit_inputs(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    labels = torch.cat((torch.zeros(12, dtype=torch.int64), torch.arange(1, 100).repeat(4)))
    development_labels = torch.cat((torch.zeros(3, dtype=torch.int64), torch.arange(1, 100)))
    counts = torch.bincount(labels, minlength=100)
    dev_counts = torch.bincount(development_labels, minlength=100)
    options = baseline.prepare_method_options({})
    settings = {"imbalance_factor": 0.01, "data_seed": 42}
    split = {
        "split_seed": 43, "original_class_counts": (counts + dev_counts).tolist(),
        "fitting_class_counts": counts.tolist(), "development_class_counts": dev_counts.tolist(),
        "fitting_positions": list(range(len(labels))),
        "development_positions": list(range(len(labels), len(labels) + len(development_labels))),
        "file_hashes": {"fitting": "a" * 64, "development": "b" * 64},
    }
    audit = {"feature_caches": {}}
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = create_model()
        checkpoint = root / f"baselines/seed{seed}/final.pt"
        checkpoint.parent.mkdir(parents=True)
        save_checkpoint_with_metadata(checkpoint, model, {
            "method": "baseline", "seed": seed, "method_options": options,
        }, counts, settings)
        weights = (1 / (100 * counts[labels].double())).float()
        weights[:12] = torch.arange(1, 13).float() / 78 / 100
        diagnostics = [{"class_index": c, "used_ot": c == 0,
                        "fallback_reason": None if c == 0 else "too_few_examples",
                        "cluster_sizes": [6, 6] if c == 0 else None} for c in range(100)]
        bundle = {
            "inputs": {"seed": seed, "baseline_sha256": file_sha256(checkpoint),
                       "fitting_sha256": "a" * 64, "development_sha256": "b" * 64,
                       "prototype_options": prototype_weights.prepare_options({"weighting": "sinkhorn"})},
            "features": torch.randn(len(labels), 64), "labels": labels,
            "development_features": torch.randn(len(development_labels), 64),
            "development_labels": development_labels, "ot_weights": weights,
            "prototype_diagnostics": diagnostics,
        }
        cache = root / f"caches/seed{seed}/features_weights.pt"
        cache.parent.mkdir(parents=True)
        torch.save(bundle, cache)
        audit["feature_caches"][str(seed)] = file_sha256(cache)
    reports = {
        "weight_audit.json": audit,
        "weight_artifacts.json": {"sha256": ablation_inputs.ARCHIVE_SHA256, "asset_id": 1},
        "study_config.json": {"baseline": options, "split_seed": 43, "seeds": [0, 1, 2]},
        "split/split.json": split, "implementation.json": {"synthetic": "fixture"},
    }
    hashes = {}
    for name, value in reports.items():
        write_json(root / "audit" / name, value)
        hashes[name] = file_sha256(root / "audit" / name)
    monkeypatch.setattr(ablation_inputs, "AUDIT_REPORT_HASHES", hashes)
    return root


def test_ranked_assignments_handle_mistakes_ties_and_preserve_rng():
    labels = torch.arange(100).repeat_interleave(4)
    weights = torch.tensor([1., 2., 3., 4.]).repeat(100) / 1000
    margins = torch.tensor([-5., -0.1, 0.1, 5.]).repeat(100)
    tie_order = torch.arange(len(labels)).flip(0)
    before = torch.random.get_rng_state().clone()
    easy = reassign_weights(weights, labels, margins, "easy_strong", tie_order)
    boundary = reassign_weights(weights, labels, margins, "boundary_strong", tie_order)
    hard = reassign_weights(weights, labels, margins, "hard_strong", tie_order)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert easy[:4].tolist() == weights[:4].tolist()
    assert hard[0] == weights[3] and hard[3] == weights[0]
    assert boundary[1] == weights[3] and boundary[2] == weights[2]
    for candidate in (easy, boundary, hard):
        assert torch.equal(candidate.reshape(100, 4).sort().values, weights.reshape(100, 4).sort().values)
    uniform = torch.full_like(weights, 0.0025)
    assert torch.equal(reassign_weights(uniform, labels, margins, "boundary_strong", tie_order), uniform)
    with pytest.raises(ValueError, match="permutation"):
        reassign_weights(weights, labels, margins, "boundary_strong", torch.zeros_like(labels))
    with pytest.raises(ValueError, match="finite"):
        reassign_weights(weights, labels, margins * float("nan"), "easy_strong", tie_order)


def test_assignment_diagnostics_share_ties_and_handle_constant_weights():
    labels = torch.arange(100).repeat_interleave(4)
    margins = torch.tensor([-2., -1., 1., 2.]).repeat(100)
    weights = torch.full((400,), 0.0025)
    diagnostics = [{"used_ot": c == 0} for c in range(100)]
    report = assignment_audit(weights, labels, margins, margins > 0, diagnostics)
    row = report["classes"][0]
    assert row["margin_weight_spearman"] is None
    assert row["top_quarter_confidence_percentile_shift"] == pytest.approx(0)
    assert row["mistake_mass_fraction"] == pytest.approx(0.5)
    assert row["uncertain_quarter_mass_fraction"] == pytest.approx(0.25)
    assert report["summary"]["ot_active"]["class_count"] == 1
    assert report["summary"]["all"]["nonconstant_class_count"] == 0


def test_pinned_inputs_reject_changed_reports_caches_and_checkpoints(audit_inputs):
    config = study.resolve_config({})
    verified = ablation_inputs.load_audit_inputs(audit_inputs, config)
    assert set(verified["bundles"]) == {0, 1, 2}
    checkpoint = audit_inputs / "baselines/seed2/final.pt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="baseline checkpoint"):
        ablation_inputs.load_audit_inputs(audit_inputs, config)
    (audit_inputs / "caches/seed1/features_weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="feature cache"):
        ablation_inputs.load_audit_inputs(audit_inputs, config)
    (audit_inputs / "audit/split/split.json").write_text("{}")
    with pytest.raises(ValueError, match="report"):
        ablation_inputs.load_audit_inputs(audit_inputs, config)


def test_shared_feature_loop_matches_existing_entry_and_keeps_partial_batch(tmp_path, training_data, monkeypatch):
    torch.manual_seed(7)
    first = create_model()
    second = copy.deepcopy(first)
    features = torch.randn(len(training_data.labels), 64)
    labels = torch.tensor(training_data.labels)
    config = {"method": "uniform_finetuning", "seed": 3, "batch_size": 4,
              "method_options": {"classifier": reweighting.prepare_classifier_options({"epochs": 1})}}
    seen = []
    original_loss = reweighting._weighted_cross_entropy

    def checked_loss(logits, targets, weights, count):
        seen.append((len(targets), count))
        return original_loss(logits, targets, weights, count)

    monkeypatch.setattr(reweighting, "_weighted_cross_entropy", checked_loss)
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
    before = copy.deepcopy(first.state_dict())
    reweighting.train_classifier(first, training_data, config, tmp_path / "first", torch.device("cpu"),
                                epoch_offset=0, use_ot=False, cached_features=(features, labels))
    reweighting.train_classifier_from_features(second, features, labels, config, tmp_path / "second",
                                               torch.device("cpu"), epoch_offset=0, use_ot=False)
    assert seen == [(4, 7), (3, 7), (4, 7), (3, 7)]
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
        if not name.startswith("fc."):
            assert torch.equal(value, before[name])
    assert not torch.equal(first.fc.weight, before["fc.weight"])


@pytest.fixture
def completed_study(audit_inputs, tmp_path):
    config = study.resolve_config({"classifier": {"epochs": 1, "learning_rate": 0.002}})
    root = tmp_path / "study"
    records = study.run_study(config, audit_inputs, root, "cpu")
    return config, root, records


def test_all_conditions_checkpoint_and_relocated_completed_reuse(completed_study, audit_inputs, tmp_path, monkeypatch):
    config, root, records = completed_study
    assert len(records) == 30
    summary = json.loads((root / "study_summary.json").read_text())
    assert summary["status"] == "complete" and summary["completed_runs"] == 27
    assert len(summary["primary_contrasts"]) == 3
    baseline = torch.load(audit_inputs / "baselines/seed0/final.pt", weights_only=True)
    model, checkpoint = load_checkpoint(root / "runs/seed0/boundary_strong/final.pt", "cpu")
    model.eval()
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)
    assert checkpoint["class_counts"].tolist() == [12] + [4] * 99
    reference = torch.load(root / "runs/seed0/strong_ot/weights.pt", weights_only=True)
    for condition in ("easy_strong", "boundary_strong", "hard_strong", "shuffled_strong_ot_shuffle0"):
        candidate = torch.load(root / f"runs/seed0/{condition}/weights.pt", weights_only=True)
        for c in range(100):
            selected = reference["labels"] == c
            assert torch.equal(candidate["weights"][selected].sort().values,
                               reference["weights"][selected].sort().values)
    for name, value in checkpoint["model_state_dict"].items():
        if not name.startswith("fc."):
            assert torch.equal(value, baseline["model_state_dict"][name])
    assert study.summarize_assignment(records[:-1], config)["decision"]["status"] == "incomplete"
    with pytest.raises(ValueError, match="duplicate"):
        study.summarize_assignment(records + [records[-1]], config)
    relocated = tmp_path / "moved"
    shutil.copytree(root, relocated)

    def forbidden(*args, **kwargs):
        pytest.fail("A completed run must not retrain")

    monkeypatch.setattr(reweighting, "train_classifier_from_features", forbidden)
    assert study.run_study(config, audit_inputs, relocated, "cpu") == records
    (relocated / "runs/seed2/hard_strong/final.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed run file changed"):
        study.run_study(config, audit_inputs, relocated, "cpu")
    assert json.loads((relocated / "study_status.json").read_text())["status"] == "failed"


def test_assignment_archives_restore_completed_runs_and_reject_corruption(completed_study, tmp_path, monkeypatch):
    _, root, _ = completed_study
    paths = kaggle.assignment_asset_paths(root)
    assert all(kaggle.is_assignment_asset_path(p) for p in paths)
    assert not kaggle.is_assignment_asset_path("../secret.json")
    assert not kaggle.is_assignment_asset_path("runs/seed0/class_balanced/secret.pt")
    archive = io.BytesIO()
    with ZipFile(archive, "w") as target:
        for relative in paths:
            target.write(root / relative, relative)
    import hashlib

    contents = archive.getvalue()
    asset = {"asset_id": 1, "sha256": hashlib.sha256(contents).hexdigest(), "paths": paths}
    write_json(root / "study_assets.json", {"seeds": {}, "complete": asset})
    results = tmp_path / "results"
    for source in root.rglob("*.json"):
        relative = source.relative_to(root)
        destination = results / "reports" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(results / "manifest.json", {"report_hashes": {
        p.relative_to(results / "reports").as_posix(): file_sha256(p)
        for p in (results / "reports").rglob("*.json")}})
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "restored")
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", results)
    monkeypatch.setattr(kaggle, "github_release_request", lambda *a, **kw: io.BytesIO(contents))
    restored = kaggle.restore_assignment_study({})
    assert (restored / "runs/seed2/hard_strong/final.pt").read_bytes() == (root / "runs/seed2/hard_strong/final.pt").read_bytes()
    shutil.rmtree(restored)
    monkeypatch.setattr(kaggle, "github_release_request", lambda *a, **kw: io.BytesIO(b"broken"))
    with pytest.raises(ValueError, match="checksum"):
        kaggle.restore_assignment_study({})


def test_interruption_and_publication_retry_reuse_finished_run(audit_inputs, tmp_path, monkeypatch):
    config = study.resolve_config({"classifier": {"epochs": 1}})
    root = tmp_path / "study"

    def interrupted(*args):
        raise RuntimeError("publication unavailable")

    with pytest.raises(RuntimeError, match="publication unavailable"):
        study.run_study(config, audit_inputs, root, "cpu", on_progress=interrupted)
    final = root / "runs/seed0/class_balanced/final.pt"
    digest = file_sha256(final)
    partial = root / "runs/seed0/ot"
    partial.mkdir()
    (partial / "partial.txt").write_text("keep me")
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "workspace")
    kaggle.archive_interrupted_runs(root)
    assert not partial.exists() and final.exists()
    assert list((tmp_path / "workspace/interruptions").rglob("partial.txt"))
    study.run_study(config, audit_inputs, root, "cpu")
    assert file_sha256(final) == digest


@pytest.mark.parametrize("override", [{"seeds": [0]}, {"shuffle_seeds": [0, 1]},
    {"tie_seed": True}, {"practical_gain": float("nan")}, {"stage": "final"}, {"unknown": 1}])
def test_assignment_config_rejects_invalid_settings(override):
    with pytest.raises(ValueError):
        study.resolve_config(override)


def test_classifier_override_preserves_study_learning_rate():
    config = study.resolve_config({"classifier": {"epochs": 1}})
    assert config["classifier"]["learning_rate"] == 0.002


def test_primary_decision_uses_paired_seed_means_and_shuffle_means():
    config = study.resolve_config({})
    records = []
    for seed in config["seeds"]:
        for condition, shuffle in [("baseline", None)] + study.conditions(config):
            value = 0.406 if condition == "boundary_strong" else 0.4
            if shuffle is not None:
                value += (shuffle - 1) * 0.001
            records.append({"seed": seed, "condition": condition, "shuffle_seed": shuffle,
                "metrics": {name: value for name in ("balanced_accuracy", "accuracy", "head_recall",
                                                     "medium_recall", "tail_recall")}})
    result = study.summarize_assignment(records, config)
    assert result["decision"]["status"] == "target_met"
    contrast = result["primary_contrasts"]["boundary_strong_minus_shuffled_strong_ot"]
    assert contrast["paired_difference_mean"] == pytest.approx(0.006)
    assert contrast["training_seed_count"] == 3
    for row in records:
        if row["seed"] == 2 and row["condition"] == "boundary_strong":
            row["metrics"]["balanced_accuracy"] = 0.1
    assert study.summarize_assignment(records, config)["decision"]["status"] == "target_not_met"


def test_assignment_kaggle_preflight_dispatch_and_incremental_assets(audit_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "CODE_ROOT", Path(__file__).resolve().parents[1])
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(kaggle, "prepare_assignment_inputs", lambda *a: audit_inputs)
    monkeypatch.setattr(kaggle, "ASSIGNMENT_CONFIG_OVERRIDES", {})
    monkeypatch.setattr("pipeline.runtime.resolve_device", lambda *a: torch.device("cpu"))

    def forbidden(*args, **kwargs):
        pytest.fail("Assignment must not prepare images, train baselines, or evaluate official test data")

    monkeypatch.setattr(kaggle, "prepare_data", forbidden)
    monkeypatch.setattr("pipeline.run_ablation._baseline", forbidden)
    monkeypatch.setattr("pipeline.evaluate.evaluate_checkpoint", forbidden)
    monkeypatch.setattr(study, "run_study", forbidden)
    monkeypatch.setattr(kaggle, "ASSIGNMENT_PREFLIGHT_ONLY", True)
    kaggle.execute_assignment("a" * 40, {})
    monkeypatch.setattr(kaggle, "ASSIGNMENT_PREFLIGHT_ONLY", False)
    root = kaggle.WORK_ROOT / "study"
    monkeypatch.setattr(kaggle, "restore_assignment_study", lambda *a: root)
    publications, uploads = [], []

    def execute(config, inputs, output, device, on_progress):
        assert config["classifier"]["epochs"] == 40
        assert config["classifier"]["learning_rate"] == 0.002
        assert inputs == audit_inputs and output == root
        run = root / "runs/seed0/class_balanced"
        run.mkdir(parents=True)
        write_json(run / "complete.json", {})
        (run / "final.pt").write_bytes(b"checkpoint")
        on_progress(root, 0, "save run")
        on_progress(root, None, "complete")

    def upload(root, paths, commit, env, prefix):
        uploads.append((paths, prefix))
        return {"asset_id": len(uploads), "sha256": "a" * 64}

    monkeypatch.setattr(study, "run_study", execute)
    monkeypatch.setattr(kaggle, "upload_artifacts", upload)
    monkeypatch.setattr(kaggle, "publish_reports", lambda root, commit, message, env: publications.append(message))
    kaggle.execute_assignment("a" * 40, {})
    assert publications == ["save run", "complete"]
    assert uploads[0][1] == "ot-assignment-seed0" and uploads[1][1] == "ot-assignment-complete"
    assert all(kaggle.is_assignment_asset_path(p) for paths, _ in uploads for p in paths)
    assert "complete" in json.loads((root / "study_assets.json").read_text())
