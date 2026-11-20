import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from pipeline import followup_reporting as reporting, run_ot_followups as study
from pipeline.ablation_data import prepare_split
from pipeline.controlled_bias_head import HeadConvergenceError
from pipeline.data import TrainingData, save_training_data
from pipeline.feature_smote import bank_diagnostics, build_banks, validate_bank
from pipeline import followup_images
from pipeline.followup_weights import (
    FEATURE_CONDITIONS, IMAGE_CONDITIONS, build_feature_weights, build_image_weights,
    normalize_weights, shuffle_weights, validate_weights, weights_on_geometry,
)
from pipeline.image_reweighting import IndexedAugmentations, image_logits, train_images
from pipeline.runtime import file_sha256, load_checkpoint, write_json
from scripts import kaggle_ablation_study as kaggle
from test_simple_controls import stage_b_inputs
from test_weight_study import audit_inputs


@pytest.fixture
def followup_inputs(stage_b_inputs, monkeypatch):
    monkeypatch.setattr(study, "EXPECTED_ACTIVE_CLASSES", (1, 1, 1))
    monkeypatch.setattr(study, "EXPECTED_FITTING_ROWS", 408)
    monkeypatch.setattr(study, "EXPECTED_AUGMENTED_ROWS", 1632)
    return stage_b_inputs


def test_smote_count_policy_ties_parent_identity_raw_coordinates_and_rng():
    labels = torch.cat((torch.zeros(100, dtype=torch.int64), torch.arange(1, 100).repeat_interleave(2)))
    features = torch.arange(len(labels) * 3, dtype=torch.float32).reshape(-1, 3) + 1
    before = torch.random.get_rng_state().clone()
    banks = build_banks(features, labels)
    assert torch.equal(before, torch.random.get_rng_state())
    smote, duplicate = banks["smote"], banks["duplicate"]
    assert len(smote["labels"]) == 100 + 99 * 8
    assert torch.equal(smote["features"][:len(labels)], features)
    assert not (labels[smote["anchor_positions"]] == 0).any()
    assert torch.equal(smote["anchor_positions"], duplicate["anchor_positions"])
    assert torch.equal(smote["features"][len(labels):].amin(dim=1) > 1,
                       torch.ones(len(smote["lambda"]), dtype=torch.bool))
    a, b = features[smote["anchor_positions"]], features[smote["neighbor_positions"]]
    assert (smote["features"][len(labels):] >= torch.minimum(a, b) - .001).all()
    assert (smote["features"][len(labels):] <= torch.maximum(a, b) + .001).all()
    assert torch.equal(duplicate["features"][len(labels):], a)
    # Two-row classes force the other real row, never self or a synthetic parent.
    assert (smote["anchor_positions"] != smote["neighbor_positions"]).all()
    changed = build_banks(features * 3, labels)["smote"]
    assert torch.equal(changed["anchor_positions"], smote["anchor_positions"])
    assert torch.equal(changed["lambda"], smote["lambda"])
    tied_features = torch.ones(400, 3)
    tied_labels = torch.arange(100).repeat_interleave(4)
    tied = build_banks(tied_features, tied_labels)["smote"]
    # Independently replay rank draws against sorted cache positions under exact ties.
    ranks = torch.randint(3, (12,), generator=torch.Generator().manual_seed(701))
    for i in range(12):
        anchor = int(tied["anchor_positions"][i])
        available = [p for p in range(4) if p != anchor]
        assert tied["neighbor_positions"][i] == available[int(ranks[i])]
    bad = copy.deepcopy(smote)
    bad["features"][-1, 0] += 1
    with pytest.raises(ValueError, match="interpolation"):
        validate_bank(bad, features, labels, "smote")
    with pytest.raises(ValueError, match="at least two"):
        build_banks(torch.ones(100, 3), torch.arange(100))


def test_duplicate_parent_diversity_excludes_unused_neighbors():
    labels = torch.arange(100).repeat_interleave(4)
    features = torch.arange(800, dtype=torch.float32).reshape(400, 2) + 1
    bank = build_banks(features, labels)["duplicate"]
    # Twelve copies of one real anchor have one contributing parent, even
    # though a distinct neighbor was saved for the matched SMOTE proposal.
    bank["anchor_positions"][:12] = 0
    bank["neighbor_positions"][:12] = 1
    bank["features"][len(labels):len(labels) + 12] = features[0]
    validate_bank(bank, features, labels, "duplicate")
    report = bank_diagnostics(bank, features)
    assert report["classes"][0]["unique_anchor_count"] == 1
    assert report["classes"][0]["unique_parent_count"] == 1
    assert report["classes"][0]["exact_real_duplicate_count"] == 12


def test_fixed_real_geometry_fallback_mass_shuffle_and_solver_failure(followup_inputs, monkeypatch):
    config = study.resolve_config({"stage": "synthetic_reweighting"})
    inputs, prepared = study.prepare_study(config, followup_inputs)
    bundle, values = inputs["bundles"][0], prepared[0]
    weights, _ = build_feature_weights(bundle, values["geometry"], values["banks"])
    assert set(weights) == set(FEATURE_CONDITIONS)
    for condition, weight in weights.items():
        _, labels, origins = study._condition_data(bundle, values, condition)
        validate_weights(weight, labels)
        # Class 1 has sixteen augmented rows, but stays a real-only fallback.
        assert torch.equal(weight[labels == 1], weight[labels == 1][0].expand(int((labels == 1).sum())))
        if "shuffle" in condition:
            reference = weights["feature_smote_ot"]
            for c in range(100):
                for origin in (False, True):
                    mask = (labels == c) & (origins == origin)
                    assert torch.equal(weight[mask].sort().values, reference[mask].sort().values)
    image, _ = build_image_weights(bundle, values["geometry"], inputs["checkpoints"][0]["model_state_dict"])
    assert set(image) == set(IMAGE_CONDITIONS)
    active_values = image["image_easy_rank"][bundle["labels"] == 0]
    assert float(active_values.max() / active_values.min()) == pytest.approx(4)
    real = image["image_ot"]
    assert torch.equal(shuffle_weights(real, bundle["labels"], 1).sort().values, real.sort().values)
    bad = real.clone()
    bad[0] *= 2
    with pytest.raises(ValueError, match="mass"):
        validate_weights(bad, bundle["labels"])
    def fail(*args, **kwargs):
        raise RuntimeError("forced transport failure")
    monkeypatch.setattr("pipeline.followup_weights._relaxed_sinkhorn", fail)
    with pytest.raises(RuntimeError, match="transport failure"):
        weights_on_geometry(bundle["features"], bundle["labels"], values["geometry"],
                            bundle["prototype_diagnostics"], bundle["inputs"]["prototype_options"])


def test_preflight_checks_all_three_seeds_and_never_fits(followup_inputs, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("preflight fitted a scientific head")
    monkeypatch.setattr(study, "fit_head", fail)
    result = study.run_study({"stage": "synthetic_reweighting"}, followup_inputs, tmp_path / "unused",
                             preflight_only=True)
    assert result["expected_runs"] == 30 and result["scientific_fits"] == 0
    assert set(result["preparation"]) == {"0", "1", "2"}
    assert not (tmp_path / "unused").exists()


def test_feature_matrix_reuse_cast_standalone_checkpoint_and_tampering(followup_inputs, tmp_path, monkeypatch):
    root = tmp_path / "study"
    events = []
    records = study.run_study({"stage": "synthetic_reweighting"}, followup_inputs, root,
                              on_progress=lambda *args: events.append(args[1]))
    assert len(records) == 30 and len(events) == 31
    summary = json.loads((root / "study_summary.json").read_text())
    assert summary["status"] == "complete"
    assert set(summary["contrasts"]) == set(reporting.contrast_definitions("synthetic_reweighting"))
    run = root / "runs/seed0/feature_smote_ot"
    optimization = json.loads((run / "optimization.json").read_text())
    assert optimization["is_converged"] and optimization["gradient_infinity_norm"] <= 1e-6
    head = torch.load(run / "head.pt", weights_only=True)
    model, checkpoint = load_checkpoint(run / "final.pt", "cpu")
    assert model.fc.weight.dtype == torch.float32 and head["weight"].dtype == torch.float64
    assert torch.equal(model.fc.weight, head["weight"].float())
    split = json.loads((followup_inputs / "audit/split/split.json").read_text())
    assert checkpoint["class_counts"].tolist() == split["original_class_counts"]
    assert checkpoint["config"]["fitting_class_counts"] == split["fitting_class_counts"]
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)
    metrics = json.loads((run / "metrics.json").read_text())
    cast = json.loads((run / "cast_metrics.json").read_text())
    assert len(metrics["per_class_recall"]) == 100
    assert "prediction_disagreements_with_float64" in cast
    def fail(*args, **kwargs):
        pytest.fail("a compatible completed head was refitted")
    monkeypatch.setattr(study, "fit_head", fail)
    assert study.run_study({"stage": "synthetic_reweighting"}, followup_inputs, root) == records
    (run / "head.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Completed run file changed"):
        study.run_study({"stage": "synthetic_reweighting"}, followup_inputs, root)


def test_preparation_tamper_and_interrupted_recovery(followup_inputs, tmp_path, monkeypatch):
    config = study.resolve_config({"stage": "synthetic_reweighting"})
    inputs, prepared = study.prepare_study(config, followup_inputs)
    root, digest = study.initialize_study(config, inputs, prepared, tmp_path / "study", torch.device("cpu"))
    run = root / "runs/seed0/feature_real_cb"
    run.mkdir(parents=True)
    (run / "failure.txt").write_text("interrupted")
    study.fit_condition(config, inputs, prepared[0], 0, "feature_real_cb", root, digest, torch.device("cpu"))
    assert (root / "attempts/runs/seed0/feature_real_cb/attempt1/failure.txt").read_text() == "interrupted"
    damaged = root / "diagnostics/seed1/preparation.pt"
    saved = torch.load(damaged, weights_only=True)
    saved["banks"]["smote"]["features"][-1, 0] += 1
    torch.save(saved, damaged)
    with pytest.raises(ValueError, match="Saved preparation changed"):
        study.initialize_study(config, inputs, prepared, root, torch.device("cpu"))
    with pytest.raises(ValueError, match="frozen"):
        study.resolve_config({"stage": "synthetic_reweighting", "head": {"regularization": .01}})
    with pytest.raises(ValueError, match="frozen"):
        study.resolve_config({"stage": "ot_contribution", "image": {"epochs": 1}})


class TinyImageModel(torch.nn.Module):
    def __init__(self, feature_count=2):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2),
                                             torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                             torch.nn.Linear(2, feature_count))
        self.fc = torch.nn.Linear(feature_count, 100)

    def forward(self, images):
        return self.fc(self.backbone(images))


def image_data(count=201):
    labels = np.arange(count, dtype=np.int64) % 100
    return TrainingData(np.random.default_rng(42).integers(0, 256, (count, 32, 32, 3), dtype=np.uint8),
                        labels, np.bincount(labels, minlength=100), {"imbalance_factor": .01, "data_seed": 42})


def test_image_updates_partial_batch_bn_and_matched_augmentation_rng(tmp_path):
    torch.set_num_threads(2)
    data = image_data()
    weights = normalize_weights(torch.ones(len(data.labels)), torch.from_numpy(data.labels))
    torch.manual_seed(7)
    first = TinyImageModel()
    second = copy.deepcopy(first)
    before = copy.deepcopy(first.state_dict())
    dataset = IndexedAugmentations(data, 0)
    dataset.next_epoch()
    rng = torch.random.get_rng_state().clone()
    assert torch.equal(dataset[3][0], dataset[3][0])
    assert torch.equal(torch.random.get_rng_state(), rng)
    options = {"epochs": 2, "learning_rate": .002, "batch_size": 128}
    for model in (first, second):
        history, result = train_images(model, data, weights, 0, options, torch.device("cpu"))
        assert result["optimizer_steps"] == 4 and result["bn_buffers_unchanged"]
        assert result["backbone_parameter_movement_l2"] > 0
        assert result["maximum_backbone_gradient"] > 0
    assert len(history) == 2
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            assert torch.equal(value, before[name])
    logits, labels = image_logits(first, data, "cpu", 128)
    assert logits.shape == (201, 100) and torch.equal(labels, torch.from_numpy(data.labels))
    bad = weights.clone()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        train_images(first, data, bad, 0, options, "cpu")


def test_paired_reporting_shuffle_mean_interaction_and_inclusive_gate():
    metrics = {m: .5 for m in reporting.METRICS} | {"per_class_recall": [.5] * 100,
                                                    "development_support": [1] * 100}
    values = {"feature_real_cb": .5, "feature_real_ot": .5, "feature_duplicate_cb": .5,
              "feature_duplicate_ot": .5, "feature_smote_cb": .5, "feature_smote_ot": .505,
              "feature_smote_prototype_distance": .5,
              "feature_smote_ot_shuffle0": .49, "feature_smote_ot_shuffle1": .5,
              "feature_smote_ot_shuffle2": .51}
    records = [{"seed": s, "condition": c, "metrics": metrics | {"balanced_accuracy": v,
                "per_class_recall": [v] * 100}} for s in (0, 1, 2) for c, v in values.items()]
    config = study.resolve_config({"stage": "synthetic_reweighting"})
    result = reporting.summarize(records, config)
    assert result["decision"]["candidates"]["synthesis_plus_ot"]["target_met"]
    assert not result["decision"]["candidates"]["synthesis_without_ot"]["target_met"]
    assert result["contrasts"]["interaction"]["metrics"]["balanced_accuracy"]["mean"] == pytest.approx(.005)
    assert result["contrasts"]["ot_minus_shuffle_mean"]["positive_seeds"] == 3
    assert reporting.readable_report(result)["contrasts"]["ot_after_synthesis"]["mean_gain_pp"] == pytest.approx(.5)
    assert reporting.summarize(records[:-1], config)["decision"]["status"] == "incomplete"
    with pytest.raises(ValueError, match="Duplicate"):
        reporting.summarize(records + records[:1], config)
    image_rows = [{"seed": s, "condition": c, "metrics": metrics | {
        "balanced_accuracy": .52 if s == 0 and c == "image_ot" else .5}}
        for s in (0, 1, 2) for c in IMAGE_CONDITIONS]
    image_summary = reporting.summarize(image_rows, study.resolve_config({"stage": "ot_contribution"}))
    shuffle_contrast = image_summary["contrasts"]["ot_minus_shuffle_mean"]
    assert shuffle_contrast["metrics"]["balanced_accuracy"]["mean"] > .005
    assert shuffle_contrast["positive_seeds"] == 1 and not shuffle_contrast["target_met"]


def test_image_reconstruction_rejects_changed_pixels_and_row_labels(tmp_path, monkeypatch):
    data = image_data(500)
    retained = tmp_path / "retained.npz"
    save_training_data(retained, data)
    split_root = tmp_path / "split"
    split = prepare_split(retained, split_root, 43)
    inputs = {"split": split, "bundles": {0: {
        "labels": torch.from_numpy(data.labels[split["fitting_positions"]]),
        "development_labels": torch.from_numpy(data.labels[split["development_positions"]])}}}
    from pipeline.data import load_training_data
    hashes = {"retained": followup_images.array_hashes(data)} | {
        name: followup_images.array_hashes(load_training_data(split_root / f"{name}.npz"))
        for name in ("fitting", "development")}
    monkeypatch.setattr(followup_images, "IMAGE_ARRAY_HASHES", hashes)
    result = study.prepare_images(inputs, retained, split_root)
    assert np.array_equal(result["fitting"].images, data.images[split["fitting_positions"]])
    # Different ZIP serialization is acceptable only with identical array bytes.
    alternate = tmp_path / "uncompressed.npz"
    np.savez(alternate, images=data.images, labels=data.labels, class_counts=data.class_counts,
             **data.data_settings)
    assert file_sha256(alternate) != file_sha256(retained)
    replay = study.prepare_images(inputs, alternate, tmp_path / "alternate_split")
    assert np.array_equal(replay["fitting"].images, result["fitting"].images)
    data.images[0, 0, 0, 0] ^= 1
    tampered = tmp_path / "tampered.npz"
    save_training_data(tampered, data)
    with pytest.raises(ValueError, match="Retained image arrays"):
        study.prepare_images(inputs, tampered, tmp_path / "tampered_split")
    inputs["bundles"][0]["labels"][0] = 99
    with pytest.raises(ValueError, match="row order"):
        study.prepare_images(inputs, retained, split_root)
    with (split_root / "fitting.npz").open("ab") as target:
        target.write(b"changed pixels or file bytes")
    with pytest.raises(ValueError, match="data has changed"):
        study.prepare_images(inputs, retained, split_root)


def test_image_matrix_restores_each_checkpoint_and_requires_real_images(followup_inputs, tmp_path, monkeypatch):
    config = study.resolve_config({"stage": "ot_contribution"})
    inputs, prepared = study.prepare_study(config, followup_inputs)
    # Use a tiny image model for matrix orchestration, keeping production weight preparation.
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        inputs["checkpoints"][seed]["model_state_dict"] = TinyImageModel(64).state_dict()
    monkeypatch.setattr(study, "create_model", lambda: TinyImageModel(64))
    monkeypatch.setattr(study, "prepare_study", lambda *args: (inputs, prepared))
    fitting, development = image_data(408), image_data(102)
    fitting.labels[:] = inputs["bundles"][0]["labels"].numpy()
    development.labels[:] = inputs["bundles"][0]["development_labels"].numpy()
    monkeypatch.setattr(study, "prepare_images", lambda *args: {
        "fitting": fitting, "development": development, "provenance": {"artificial": True}})
    # Only the artificial test's schedule is shortened, not the public configuration.
    real_train = study.train_images
    monkeypatch.setattr(study, "train_images", lambda model, data, weights, seed, options, device, callback:
                        real_train(model, data, weights, seed, options | {"epochs": 1}, device, callback))
    with pytest.raises(ValueError, match="retained training images"):
        study.run_study(config, followup_inputs, tmp_path / "unused")
    records = study.run_study(config, followup_inputs, tmp_path / "image_study", "cpu",
                              retained_path=tmp_path / "retained.npz", split_root=tmp_path / "split")
    assert len(records) == 18
    summary = json.loads((tmp_path / "image_study/study_summary.json").read_text())
    assert summary["status"] == "complete" and summary["expected_runs"] == 18
    assert len(summary["decision"]["candidates"]["image_ot"]["required_contrasts"]) == 3


def test_solver_failure_is_incomplete_and_preserves_diagnostics(followup_inputs, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise HeadConvergenceError("forced convergence failure", {"is_converged": False}, [{"evaluation": 1}])
    monkeypatch.setattr(study, "fit_head", fail)
    root = tmp_path / "failed"
    with pytest.raises(HeadConvergenceError):
        study.run_study({"stage": "synthetic_reweighting"}, followup_inputs, root)
    status = json.loads((root / "study_status.json").read_text())
    assert status["status"] == "failed" and status["completed_runs"] == 0
    assert json.loads((root / "study_summary.json").read_text())["decision"]["status"] == "incomplete"
    run = root / "runs/seed0/feature_real_cb"
    assert not (run / "complete.json").exists()
    assert json.loads((run / "optimization.json").read_text())["is_converged"] is False


@pytest.mark.parametrize("stage", ["synthetic_reweighting", "ot_contribution"])
def test_kaggle_dispatch_uses_separate_config_device_preflight_and_publication(stage, tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "STAGES", (stage,))
    monkeypatch.setattr(kaggle, "CODE_ROOT", Path(__file__).resolve().parents[1])
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "FOLLOWUP_PREFLIGHT_ONLY", False)
    monkeypatch.setattr(kaggle, "prepare_assignment_inputs", lambda *args: tmp_path / "inputs")
    monkeypatch.setattr(kaggle, "restore_cached_study", lambda *args, **kwargs: tmp_path / "study")
    monkeypatch.setattr(kaggle, "prepare_data", lambda: (tmp_path / "train.npz", tmp_path / "raw"))
    calls, publications = [], []
    monkeypatch.setattr(kaggle, "publish_cached_study", lambda *args: publications.append(args))
    def run(config, inputs, output, device, **kwargs):
        calls.append((config["stage"], device, kwargs))
        if kwargs["on_progress"]:
            kwargs["on_progress"](output, 0, "completed")
        return {"status": "preflight_passed", "preparation": {}}
    monkeypatch.setattr(study, "run_study", run)
    kaggle.execute_followup("a" * 40, {})
    assert calls[0][0:2] == (stage, "cuda" if stage == "ot_contribution" else "cpu")
    assert publications[0][-2] == "ot-" + stage.replace("_", "-")
    monkeypatch.setattr(kaggle, "FOLLOWUP_PREFLIGHT_ONLY", True)
    kaggle.execute_followup("a" * 40, {})
    assert calls[-1][1] == "cpu" and calls[-1][2]["on_progress"] is None
    assert len(publications) == 1


def test_followup_assets_recover_banks_completed_heads_and_failed_attempts(tmp_path, monkeypatch):
    study_root = tmp_path / "study"
    for s in (0, 1, 2):
        folder = study_root / f"diagnostics/seed{s}"
        folder.mkdir(parents=True)
        (folder / "preparation.pt").write_bytes(b"bank")
        write_json(folder / "preparation_report.json", {"seed": s})
    run = study_root / "runs/seed0/feature_real_cb"
    run.mkdir(parents=True)
    (run / "head.pt").write_bytes(b"head")
    write_json(run / "complete.json", {"inputs": {}, "files": {"head.pt": file_sha256(run / "head.pt")}})
    attempt = study_root / "attempts/runs/seed0/feature_real_cb/attempt1"
    attempt.mkdir(parents=True)
    (attempt / "weights.pt").write_bytes(b"failed weights")
    paths = kaggle.followup_asset_paths(study_root, 0)
    assert "diagnostics/seed2/preparation.pt" in paths
    assert "attempts/runs/seed0/feature_real_cb/attempt1/weights.pt" in paths
    assert "runs/seed0/feature_real_cb/head.pt" in paths
    assert not kaggle.is_followup_asset_path("../runs/seed0/feature_real_cb/head.pt")
    assert not kaggle.is_followup_asset_path("runs/seed0/feature_real_cb/private.pt")
    published = tmp_path / "published"
    for p in study_root.rglob("*.json"):
        target = published / "reports" / p.relative_to(study_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write_json(published / "reports/study_config.json", {"config": {"stage": "synthetic_reweighting"}})
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for p in paths:
            archive.writestr(p, (study_root / p).read_bytes())
    import hashlib
    archive_bytes = buffer.getvalue()
    write_json(published / "reports/study_assets.json", {"seeds": {"0": {
        "asset_id": 1, "paths": paths, "sha256": hashlib.sha256(archive_bytes).hexdigest()}}})
    write_json(published / "manifest.json", {"report_hashes": {
        p.relative_to(published / "reports").as_posix(): file_sha256(p) for p in (published / "reports").rglob("*.json")}})
    monkeypatch.setattr(kaggle, "github_release_request", lambda *args, **kwargs: io.BytesIO(archive_bytes))
    restored = kaggle.restore_cached_study(published, tmp_path / "restored", {}, kaggle.is_followup_asset_path,
                                         run_files=study.FEATURE_RUN_FILES)
    assert (restored / "diagnostics/seed2/preparation.pt").read_bytes() == b"bank"
    assert (restored / "attempts/runs/seed0/feature_real_cb/attempt1/weights.pt").read_bytes() == b"failed weights"
