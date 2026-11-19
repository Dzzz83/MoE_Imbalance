import copy
from fractions import Fraction
import hashlib
import io
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from pipeline import controlled_bias_data as data
from pipeline import controlled_bias_reporting as reporting
from pipeline import run_controlled_bias as study
from pipeline.controlled_bias_head import HeadConvergenceError, evaluate_head, fit_head
from pipeline.controlled_bias_weights import build_geometry_weights, class_balanced, known_correction, validate_weights
from pipeline.methods.prototype_weights import _relaxed_sinkhorn
from pipeline.runtime import file_sha256, write_json
from scripts import kaggle_ablation_study as kaggle


@pytest.fixture
def diagnostic_inputs(tmp_path, monkeypatch):
    """Two artificial eligible classes, retaining the real selection/solver code."""
    monkeypatch.setattr(data, "EXPECTED_CLASSES", [0, 1])
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {"eligible": 160, "reference": 32, "unbiased": 128, "development": 40})
    labels = torch.tensor([0, 1] * 80)
    geometry = []
    for c in range(2):
        assignments = torch.tensor([0] * 40 + [1] * 40)
        centers = torch.zeros(2, 64)
        centers[0, c * 2] = 1
        centers[1, c * 2 + 1] = 1
        geometry.append({"class_index": c, "assignments": assignments, "centers": centers})
    bundles = {}
    for seed in (0, 1, 2):
        generator = torch.Generator().manual_seed(seed)
        # Rotate each representation's axes so cross-seed center use is detectable.
        rotation = torch.roll(torch.eye(64), shifts=seed * 5, dims=1)
        features = torch.randn(160, 64, generator=generator) * .02
        development = torch.randn(40, 64, generator=generator) * .02
        for c in range(2):
            features[labels == c] += geometry[c]["centers"][geometry[c]["assignments"]] @ rotation
            development[torch.tensor([0, 1] * 20) == c] += geometry[c]["centers"][torch.tensor([0] * 10 + [1] * 10)] @ rotation
        bundles[seed] = {"features": features, "labels": labels.clone(), "development_features": development,
                         "development_labels": torch.tensor([0, 1] * 20),
                         "inputs": {"fitting_sha256": "f", "development_sha256": "e"}}
    inputs = {"bundles": bundles, "split": {"original_class_counts": [100, 100]},
              "provenance": {"seed_files": {str(s): {"cache": f"cache{s}"} for s in (0, 1, 2)}}}
    controller = tmp_path / "controller/diagnostics/seed0"
    controller.mkdir(parents=True)
    torch.save(geometry, controller / "geometry.pt")
    write_json(controller / "controls.json", {"geometry_sha256": file_sha256(controller / "geometry.pt")})
    monkeypatch.setattr(data, "CONTROLLER_REPORT_SHA256", file_sha256(controller / "controls.json"))
    monkeypatch.setattr(data, "CONTROLLER_GEOMETRY_SHA256", file_sha256(controller / "geometry.pt"))
    monkeypatch.setattr(study, "load_audit_inputs", lambda *_: inputs)
    return inputs, geometry, controller.parents[1]


def test_sampling_exact_sizes_probabilities_disjointness_and_cache_alignment(diagnostic_inputs):
    inputs, geometry, _ = diagnostic_inputs
    before = torch.random.get_rng_state().clone()
    config = study.resolve_config({})
    selected, report = data.prepare_sampling(inputs["bundles"][0]["labels"], [100, 100], geometry, config)
    assert report["counts"] == {"eligible": 160, "reference": 32, "unbiased": 128, "retained": 80, "removed": 48}
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(selected["inclusion_probabilities"].unique(), torch.tensor([.25, 1.], dtype=torch.float64))
    for name in ("reference", "unbiased", "retained", "removed"):
        assert torch.equal(selected[name], selected[name].sort().values)
    all_positions = torch.cat([selected[n] for n in ("reference", "retained", "removed")])
    assert torch.equal(all_positions.sort().values, torch.arange(160))
    repeat, _ = data.prepare_sampling(inputs["bundles"][0]["labels"], [100, 100], geometry, config)
    assert study.content_identity(repeat) == study.content_identity(selected)
    data.validate_cache_alignment(inputs["bundles"])
    changed = copy.deepcopy(inputs["bundles"])
    changed[1]["labels"] = changed[1]["labels"].flip(0)
    with pytest.raises(ValueError, match="label order"):
        data.validate_cache_alignment(changed)
    changed = copy.deepcopy(inputs["bundles"])
    changed[2]["inputs"]["fitting_sha256"] = "different"
    with pytest.raises(ValueError, match="split identity"):
        data.validate_cache_alignment(changed)
    geometry[0]["assignments"][:20] = 1
    with pytest.raises(ValueError, match="eligibility changed"):
        data.prepare_sampling(inputs["bundles"][0]["labels"], [100, 100], geometry, config)


def test_sampling_uses_actual_ceil_probability_and_known_loss_is_unbiased(diagnostic_inputs):
    inputs, geometry, _ = diagnostic_inputs
    # 39/41 groups reserve 7/8 references, so one U group has 33 members.
    geometry = copy.deepcopy(geometry)
    for current in geometry:
        current["assignments"] = torch.tensor([0] * 39 + [1] * 41)
    sampling, report = data.prepare_sampling(inputs["bundles"][0]["labels"], [100, 100], geometry, study.resolve_config({}))
    for row in report["classes"]:
        biased = row["groups"][row["biased_group"]]
        assert biased["retained"] == (biased["unbiased"] + 3) // 4
        assert biased["inclusion_probability"] == biased["retained"] / biased["unbiased"]
    # Exhaust all fixed-size samples from a tiny population; the fixed-logit
    # loss identity is exact rational arithmetic, not a generalization test.
    all_losses = [Fraction(2), Fraction(4), Fraction(5), Fraction(7), Fraction(9)]
    estimates = []
    for kept in itertools.combinations(range(1, 5), 2):
        estimates.append(all_losses[0] / 5 + sum(all_losses[i] / (5 * Fraction(2, 4)) for i in kept))
    assert sum(estimates) / len(estimates) == sum(all_losses) / len(all_losses)
    retained_labels = inputs["bundles"][0]["labels"][sampling["retained"]]
    unbiased_labels = inputs["bundles"][0]["labels"][sampling["unbiased"]]
    weights = known_correction(retained_labels, torch.bincount(unbiased_labels), sampling["inclusion_probabilities"])
    validate_weights(weights, retained_labels, 2)
    for c in range(2):
        for g in (0, 1):
            rows = (retained_labels == c) & (sampling["controller_groups"][sampling["retained"]] == g)
            desired = ((unbiased_labels == c) & (sampling["controller_groups"][sampling["unbiased"]] == g)).sum() / (unbiased_labels == c).sum()
            assert weights[rows].sum().item() == pytest.approx(desired.item() / 2)


def test_geometry_targets_smoothing_shared_fallbacks_and_exact_shuffles():
    config = study.resolve_config({})
    features = torch.tensor([[1., 0.]] * 20 + [[0., 1.]] * 10 + [[1., 0.]] * 8)
    labels = torch.tensor([0] * 30 + [1] * 8)
    # No references assigned to one valid prototype; both q entries stay positive.
    references = torch.tensor([[1., 0.]] * 12 + [[1., 0.]] * 4)
    reference_labels = torch.tensor([0] * 12 + [1] * 4)
    weights, geometry, diagnostics = build_geometry_weights(features, labels, references, reference_labels, 2, 0, config)
    assert diagnostics[0]["is_active"]
    assert sorted(diagnostics[0]["reference_counts"]) == [0, 12]
    assert sorted(diagnostics[0]["target"]) == pytest.approx([.5 / 13, 12.5 / 13])
    assert not diagnostics[1]["is_active"]
    assert geometry[1]["centers"] is None
    assert diagnostics[1]["fallback_reason"] == "too_few_examples"
    assignments = geometry[0]["assignments"]
    for g in (0, 1):
        assert weights["reference_cluster_ratio"][:30][assignments == g].sum().item() == pytest.approx(geometry[0]["target"][g].item() / 2)
    for values in weights.values():
        validate_weights(values, labels, 2)
        assert torch.equal(values[30:], torch.full((8,), 1 / 16, dtype=torch.float64))
    assert not torch.equal(weights["equal_target_ot"], weights["reference_target_ot"])
    for seed in (0, 1, 2):
        for c in (0, 1):
            assert torch.equal(weights[f"shuffled_reference_ot_shuffle{seed}"][labels == c].sort().values,
                               weights["reference_target_ot"][labels == c].sort().values)
    with pytest.raises(RuntimeError, match="did not converge"):
        build_geometry_weights(features, labels, references, reference_labels, 2, 0, config | {"sinkhorn_iterations": 1})


def test_nonuniform_sinkhorn_quotas_and_legacy_default():
    cost = torch.tensor([[0., 1.], [1., 0.], [1., 1.]])
    options = {"epsilon": .1, "row_kl_strength": .2, "sinkhorn_iterations": 200, "sinkhorn_tolerance": .001}
    uniform, diagnostic = _relaxed_sinkhorn(cost, **options)
    explicit, _ = _relaxed_sinkhorn(cost, prototype_masses=[.5, .5], **options)
    assert torch.equal(uniform, explicit)
    plan, _ = _relaxed_sinkhorn(cost, prototype_masses=[.8, .2], **options)
    torch.testing.assert_close(plan.sum(0), torch.tensor([.8, .2], dtype=torch.float64), rtol=1e-10, atol=1e-12)
    assert diagnostic["sinkhorn_residual"] <= .001
    for masses in ([1., 0.], [.4, .4], [.2, .3, .5], [float("nan"), .5]):
        with pytest.raises(ValueError, match="Prototype masses"):
            _relaxed_sinkhorn(cost, prototype_masses=masses, **options)


def test_preparation_information_boundaries_and_reporter_coordinate_frames(diagnostic_inputs, monkeypatch):
    inputs, geometry, controller_root = diagnostic_inputs
    prepared = study.prepare_study({}, "unused", controller_root)
    sampling = prepared["sampling"]
    before = study.content_identity(prepared["prepared"])
    # No removed-row or controller data is in the ordinary builder interface.
    original_builder = study.build_geometry_weights
    seen = []
    def check_builder(features, labels, references, reference_labels, class_count, seed, config):
        bundle = inputs["bundles"][seed]
        assert torch.equal(features, bundle["features"][sampling["retained"]])
        assert torch.equal(references, bundle["features"][sampling["reference"]])
        seen.append(seed)
        return original_builder(features, labels, references, reference_labels, class_count, seed, config)
    monkeypatch.setattr(study, "build_geometry_weights", check_builder)
    # Development changes cannot alter preparation or weights.
    for bundle in inputs["bundles"].values():
        bundle["development_features"] += 100
    changed = study.prepare_study({}, "unused", controller_root)
    assert before == study.content_identity(changed["prepared"])
    assert seen == [0, 1, 2]
    # Group assignments are constructed once in the seed-0 coordinate system.
    context = data.development_context(inputs["bundles"][0], geometry, sampling)
    for c in (0, 1):
        selected = context["labels"] == c
        raw = inputs["bundles"][0]["development_features"][context["positions"][selected]]
        expected = (torch.nn.functional.normalize(raw, dim=1) @ geometry[c]["centers"].T).argmax(1)
        assert torch.equal(context["controller_groups"][selected], expected)
    # Each method geometry carries its own feature seed, including rotated data.
    for seed in (0, 1, 2):
        assert all(g["feature_seed"] == seed for g in prepared["prepared"][seed]["geometry"])


def test_fresh_head_convergence_determinism_and_failure():
    features = torch.tensor([[1., 0.], [1.2, .1], [0., 1.], [.1, 1.2]])
    labels = torch.tensor([0, 0, 1, 1])
    weights = class_balanced(labels, 2)
    options = study.resolve_config({})["head"]
    before = torch.random.get_rng_state().clone()
    head, diagnostics, history = fit_head(features, labels, weights, 2, options)
    assert diagnostics["is_converged"] and diagnostics["gradient_infinity_norm"] <= 1e-6
    assert head["weight"].shape == (2, 2) and head["weight"].dtype == torch.float64
    assert torch.equal(before, torch.random.get_rng_state())
    repeat, repeated, _ = fit_head(features, labels, weights, 2, options)
    assert torch.equal(head["weight"], repeat["weight"]) and diagnostics == repeated
    assert history[-1]["objective"] < history[0]["objective"]
    with pytest.raises(HeadConvergenceError, match="gradient convergence"):
        fit_head(features, labels, weights, 2, options | {"max_iterations": 1})
    with pytest.raises(HeadConvergenceError, match="evaluation budget"):
        fit_head(features, labels, weights, 2, options | {"max_evaluations": 1})
    with pytest.raises(ValueError, match="features"):
        fit_head(features * float("nan"), labels, weights, 2, options)
    metrics, predictions = evaluate_head(head, features, labels, torch.zeros(4, dtype=torch.int64),
                                         torch.tensor([1, 0]), torch.tensor([4, 8]))
    assert metrics["diagnostic_balanced_accuracy"] == 1
    assert metrics["classes"][0]["original_class"] == 4
    assert metrics["controller_groups"][1]["recall"] is None
    assert torch.equal(predictions["labels"], labels)


def test_nonfinite_final_head_failure_remains_json_serializable(monkeypatch):
    original_step = torch.optim.LBFGS.step
    def corrupt_final_state(optimizer, closure):
        result = original_step(optimizer, closure)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("inf"))
        return result
    monkeypatch.setattr(torch.optim.LBFGS, "step", corrupt_final_state)
    labels = torch.tensor([0, 1])
    with pytest.raises(HeadConvergenceError) as failure:
        fit_head(torch.eye(2), labels, class_balanced(labels, 2), 2, study.resolve_config({})["head"])
    json.dumps(failure.value.diagnostics, allow_nan=False)
    assert failure.value.diagnostics["is_converged"] is False
    assert failure.value.diagnostics["objective"] is None


def gate_records(damage=.02, recovery=.01):
    records = []
    for seed in (0, 1, 2):
        for condition, score in (("unbiased_cb", .4 + damage), ("biased_cb", .4), ("known_correction", .4 + recovery)):
            records.append({"seed": seed, "condition": condition, "metrics": {m: score for m in reporting.METRICS},
                            "optimization": {"is_converged": True}})
    return records


def test_gates_missing_failed_borderline_and_passing():
    config = study.resolve_config({})
    assert reporting.gate_one([], config)["status"] == "incomplete"
    assert reporting.gate_one([], config, technical_failure=True)["reasons"] == ["technical_failure"]
    passed = reporting.gate_one(gate_records(), config)
    assert passed["passed"] and passed["recovery_fraction"] == pytest.approx(.5)
    # Exact binary fractions make the inclusive thresholds unambiguous.
    boundary = config | {"gate": {"minimum_damage": .125, "minimum_recovery": .0625,
                                  "minimum_recovery_fraction": .5, "minimum_positive_seeds": 2}}
    rows = gate_records()
    for row in rows:
        score = {"biased_cb": .5, "unbiased_cb": .625, "known_correction": .5625}[row["condition"]]
        row["metrics"] = {m: score for m in reporting.METRICS}
    assert reporting.gate_one(rows, boundary)["passed"]
    assert reporting.gate_one(gate_records(.01, .005), config)["passed"]
    assert reporting.gate_one(gate_records(.005, .001), config)["reasons"] == ["insufficient_damage", "insufficient_recovery"]
    assert reporting.gate_one(gate_records(.1, .01), config)["reasons"] == ["insufficient_recovery"]
    rows = gate_records()
    rows[0]["optimization"]["is_converged"] = False
    with pytest.raises(ValueError, match="optimization"):
        reporting.gate_one(rows, config)
    with pytest.raises(ValueError, match="duplicate"):
        reporting.gate_one(gate_records() + gate_records()[:1], config)


def test_preflight_receipts_gate_failure_reuse_and_tampering(diagnostic_inputs, tmp_path, monkeypatch):
    _, _, controller = diagnostic_inputs
    root = tmp_path / "study"
    result = study.run_study({}, "unused", controller, root, preflight_only=True)
    assert result["status"] == "preflight_complete"
    assert not (root / "runs").exists()
    assert (root / "preparation/complete.json").is_file()
    result = study.run_study({}, "unused", controller, root)
    assert result["status"] == "stopped_gate_one"
    assert result["completed_runs"] == 9
    assert len(list((root / "runs").glob("*/*/head.pt"))) == 9
    original = study.fit_head
    monkeypatch.setattr(study, "fit_head", lambda *_: pytest.fail("Completed compatible heads must be reused"))
    assert study.run_study({}, "unused", controller, root) == result
    monkeypatch.setattr(study, "fit_head", original)
    path = root / "runs/seed0/biased_cb/metrics.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        study.run_study({}, "unused", controller, root)
    # Preparation hashes protect arrays as well as the run receipts.
    (root / "diagnostics/seed2/weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        study.run_study({}, "unused", controller, root, preflight_only=True)


def test_full_gate_two_routing_and_interrupted_restart(diagnostic_inputs, tmp_path, monkeypatch):
    _, _, controller = diagnostic_inputs
    # This tests routing, not whether artificial data meet the scientific gate.
    real_gate = study.gate_one
    def passing_gate(records, config, **kwargs):
        gate = real_gate(records, config, **kwargs)
        if gate["completed_fits"] == 9 and not kwargs.get("technical_failure"):
            gate = gate | {"passed": True, "status": "passed", "reasons": []}
        return gate
    monkeypatch.setattr(study, "gate_one", passing_gate)
    monkeypatch.setattr(reporting, "gate_one", passing_gate)
    root = tmp_path / "study"
    study.run_study({}, "unused", controller, root, preflight_only=True)
    interrupted = root / "runs/seed0/unbiased_cb"
    interrupted.mkdir(parents=True)
    (interrupted / "config.json").write_text("{}")
    (interrupted / "failure.json").write_text('{"message": "simulated interruption"}')
    result = study.run_study({}, "unused", controller, root)
    assert result["status"] == "complete" and result["completed_runs"] == 27
    assert (root / "attempts/seed0/unbiased_cb/attempt1/failure.json").exists()
    assert len(list((root / "runs").glob("*/*/complete.json"))) == 27
    assert len(result["contrasts"]) == 10
    assert result["contrasts"]["reference_target_ot_minus_shuffled_reference_ot"]["diagnostic_balanced_accuracy"]["per_seed"][0]["reference_shuffle_sample_sd"] is not None
    head = torch.load(root / "runs/seed0/unbiased_cb/head.pt", weights_only=True)
    assert head["artifact_type"] == "controlled_bias_linear_head"
    assert head["state_dict"]["weight"].shape == (2, 64)
    receipt = root / "gate_one.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="Gate-1 receipt changed"):
        study.run_study({}, "unused", controller, root)
    original_fit = study.fit_head
    attempted = []
    def fail_first_gate_two_fit(*args):
        attempted.append(True)
        if len(attempted) == 10:
            raise HeadConvergenceError("Gate-2 gradient failure", {"is_converged": False}, [])
        return original_fit(*args)
    monkeypatch.setattr(study, "fit_head", fail_first_gate_two_fit)
    failed_root = tmp_path / "failed-gate-two"
    with pytest.raises(HeadConvergenceError, match="Gate-2 gradient"):
        study.run_study({}, "unused", controller, failed_root)
    failed = json.loads((failed_root / "study_summary.json").read_text())
    assert failed["status"] == "failed" and failed["expected_runs"] == 27
    assert failed["gate_one"]["passed"] and failed["completed_runs"] == 9


def test_failed_fit_is_recorded_and_blocks_gate_two(diagnostic_inputs, tmp_path, monkeypatch):
    _, _, controller = diagnostic_inputs
    def fail(*_):
        raise HeadConvergenceError("simulated gradient failure", {"is_converged": False}, [{"evaluation": 1}])
    monkeypatch.setattr(study, "fit_head", fail)
    with pytest.raises(HeadConvergenceError):
        study.run_study({}, "unused", controller, tmp_path / "study")
    root = tmp_path / "study"
    assert json.loads((root / "study_summary.json").read_text())["gate_one"]["reasons"] == ["technical_failure"]
    assert (root / "runs/seed0/unbiased_cb/failure.json").is_file()
    assert not (root / "gate_one.json").exists()


def test_kaggle_controller_download_and_diagnostic_archive_restore(diagnostic_inputs, tmp_path, monkeypatch):
    _, _, controller = diagnostic_inputs
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name in ("controls.json", "geometry.pt"):
            archive.write(controller / f"diagnostics/seed0/{name}", f"diagnostics/seed0/{name}")
    monkeypatch.setattr(data, "CONTROLLER_ARCHIVE_SHA256", hashlib.sha256(buffer.getvalue()).hexdigest())
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(kaggle, "github_release_request", lambda *_args, **_kwargs: io.BytesIO(buffer.getvalue()))
    restored_controller = kaggle.prepare_controlled_bias_controller({})
    assert file_sha256(restored_controller / "diagnostics/seed0/geometry.pt") == data.CONTROLLER_GEOMETRY_SHA256
    assert kaggle.prepare_controlled_bias_controller({}) == restored_controller
    root = tmp_path / "study"
    study.run_study({}, "unused", restored_controller, root)
    paths = kaggle.controlled_bias_asset_paths(root, seed=0)
    assert "preparation/sampling.pt" in paths and "diagnostics/seed2/weights.pt" in paths
    assert "runs/seed0/biased_cb/head.pt" in paths
    assert not kaggle.is_controlled_bias_asset_path("../private.pt")
    assert not kaggle.is_controlled_bias_asset_path("runs/seed0/unknown/head.pt")
    published = tmp_path / "published"
    for source in root.rglob("*.json"):
        target = published / "reports" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    # Include every completed run in the complete archive.
    paths = kaggle.controlled_bias_asset_paths(root)
    asset = io.BytesIO()
    with ZipFile(asset, "w") as archive:
        for name in paths:
            archive.write(root / name, name)
    write_json(published / "reports/study_assets.json", {"seeds": {}, "complete": {
        "asset_id": 3, "paths": paths, "sha256": hashlib.sha256(asset.getvalue()).hexdigest()}})
    write_json(published / "manifest.json", {"report_hashes": {
        p.relative_to(published / "reports").as_posix(): file_sha256(p) for p in (published / "reports").rglob("*.json")}})
    monkeypatch.setattr(kaggle, "github_release_request", lambda *_args, **_kwargs: io.BytesIO(asset.getvalue()))
    restored = kaggle.restore_cached_study(published, tmp_path / "restored", {}, kaggle.is_controlled_bias_asset_path, run_files=study.RUN_FILES)
    result = study.run_study({}, "unused", restored_controller, restored)
    assert result["completed_runs"] == 9


def test_new_incremental_publication_invalidates_older_gate_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    write_json(root / "study_assets.json", {"seeds": {"1": {"asset_id": 1}}, "complete": {"asset_id": 2}})
    monkeypatch.setattr(kaggle, "controlled_bias_asset_paths", lambda *_: ["preparation/sampling.pt"])
    monkeypatch.setattr(kaggle, "upload_artifacts", lambda *_: {"asset_id": 3})
    published = []
    monkeypatch.setattr(kaggle, "publish_reports", lambda *_: published.append(True))
    kaggle.publish_cached_study(root, 0, "new conditional fit", "a" * 40, {}, "ot-controlled-bias", kaggle.is_controlled_bias_asset_path)
    assets = json.loads((root / "study_assets.json").read_text())
    assert "complete" not in assets
    assert assets["seeds"]["0"]["asset_id"] == 3 and assets["seeds"]["1"]["asset_id"] == 1
    assert published == [True]


def test_kaggle_preflight_uses_cpu_and_its_own_inputs(diagnostic_inputs, tmp_path, monkeypatch):
    _, _, controller = diagnostic_inputs
    monkeypatch.setattr(kaggle, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(kaggle, "CODE_ROOT", Path(study.__file__).resolve().parents[1])
    monkeypatch.setattr(kaggle, "RESULTS_ROOT", tmp_path / "unpublished")
    monkeypatch.setattr(kaggle, "CONTROLLED_BIAS_PREFLIGHT_ONLY", True)
    monkeypatch.setattr(kaggle, "prepare_assignment_inputs", lambda *_: "unused")
    monkeypatch.setattr(kaggle, "prepare_controlled_bias_controller", lambda *_: controller)
    monkeypatch.setattr(kaggle, "publish_reports", lambda *_: pytest.fail("Preflight must not publish results"))
    kaggle.execute_controlled_bias("a" * 40, {})
    manifest = json.loads((kaggle.WORK_ROOT / "study/study_config.json").read_text())
    assert manifest["environment"]["device"] == "cpu"
    assert not (kaggle.WORK_ROOT / "study/runs").exists()


def test_frozen_config_rejects_unplanned_scientific_changes():
    for change in ({"retention_divisor": 8}, {"head": {"regularization": .01}}, {"gate": {"minimum_damage": 0}}, {"seeds": [0]}, {"unknown": 1}):
        with pytest.raises(ValueError):
            study.resolve_config(change)
