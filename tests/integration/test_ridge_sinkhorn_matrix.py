"""Focused synthetic checks for the resumable expert matrix and bundle boundary."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from pathlib import Path as _TestPath
import sys as _test_sys
_TEST_PACKAGE_DIR = str(_TestPath(__file__).resolve().parent.parent)
if _TEST_PACKAGE_DIR not in _test_sys.path:
    _test_sys.path.insert(0, _TEST_PACKAGE_DIR)
from repo_root import REPO_ROOT
if str(REPO_ROOT) not in _test_sys.path:
    _test_sys.path.insert(0, str(REPO_ROOT))
_PROJECT_ROOT = str(REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.nested_oof import NestedOOFFoldManager
from expert_method.ridge_sinkhorn.three_seed_study import StudyConfig
from expert_method.ridge_sinkhorn.matrix import (
    BUNDLE_SCHEMA_VERSION,
    COMPATIBILITY_SCHEMA_VERSION,
    DEFAULT_EXPERT_KEYS,
    DEFAULT_SEEDS,
    EXPERT_NAMES,
    MATRIX_SCHEMA_VERSION,
    STUDY_ID,
    MatrixError,
    OOFMatrixPlanner,
    StudyArtifactView,
    _read_bundle,
    _tar_info,
    build_job_inventory,
    canonical_json_bytes,
    cuda_preflight,
    export_bundle,
    import_bundle,
    membership_sha256,
    merge_bundles,
    shard_for_job,
    shard_key,
)


def synthetic_manager() -> NestedOOFFoldManager:
    labels = np.repeat(np.arange(5, dtype=np.int64), 20)
    indices = np.arange(len(labels), dtype=np.int64)
    return NestedOOFFoldManager(
        indices,
        labels,
        seed=42,
        outer_folds=5,
        inner_folds=4,
        num_classes=5,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )


def _inventory_payload(manager: NestedOOFFoldManager) -> list[dict]:
    jobs = build_job_inventory(manager)
    return [
        {
            **job.__dict__,
            "run_relative_path": job.run_relative_path,
            "shard_key": f"{shard_key(job.job_id):016x}",
            "shard_rule": "first_8_sha256_bytes_big_endian_mod_shard_count",
        }
        for job in jobs
    ]


def _valid_bundle_fixture(tmp_path: Path) -> tuple[Path, dict, dict[str, bytes]]:
    manager = synthetic_manager()
    inventory = _inventory_payload(manager)
    job = next(item for item in inventory if item["stage"] == "inner")
    job_id = job["job_id"]
    expert_key = job["expert_key"]
    expert_name = EXPERT_NAMES[expert_key]
    seed = job["training_seed"]
    outer = job["outer_fold_id"]
    inner = job["inner_fold_id"]
    run_relative = job["run_relative_path"]
    run_files = {
        f"checkpoints/{expert_name}_seed{seed}_final.pt": b"synthetic-final-checkpoint",
        "predictions.json": b'{"schema_version":"nested_oof_prediction.v1","records":[]}',
        "resolved_config.json": b'{"schedule":{"epochs":200}}',
        "run_metadata.json": b'{"schema_version":"nested_oof_run.v1","status":"complete"}',
    }
    payloads: dict[str, bytes] = {
        f"{STUDY_ID}/fold_manifest.json": b'{"schema_version":"nested_oof_manifest.v1"}',
    }
    files: list[dict] = []
    for relative, data in run_files.items():
        archive_path = f"{STUDY_ID}/{run_relative}/{relative}"
        payloads[archive_path] = data
        schema = (
            "torch_checkpoint.final.v1" if relative.startswith("checkpoints/") else
            "nested_oof_prediction.v1" if relative == "predictions.json" else
            "resolved_training_config.v1" if relative == "resolved_config.json" else
            "nested_oof_run.v1"
        )
        files.append(
            {
                "path": archive_path,
                "run_relative_path": relative,
                "schema": schema,
                "byte_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    files.sort(key=lambda entry: entry["path"])
    run_identity_sha = hashlib.sha256(
        json.dumps(
            {"job_id": job_id, "run_relative_path": run_relative, "files": files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    job_payload = {
        "job_id": job_id,
        "stage": "inner",
        "expert": expert_key,
        "training_seed": seed,
        "outer_fold_id": outer,
        "inner_fold_id": inner,
        "run_relative_path": run_relative,
        "training_membership_sha256": job["training_membership_sha256"],
        "prediction_membership_sha256": job["prediction_membership_sha256"],
        "shard_key": f"{shard_key(job_id):016x}",
        "shard_index": 0,
        "shard_count": 1,
        "checkpoint_sha256": next(item["sha256"] for item in files if item["run_relative_path"].startswith("checkpoints/")),
        "prediction_sha256": next(item["sha256"] for item in files if item["run_relative_path"] == "predictions.json"),
        "resolved_config_sha256": next(item["sha256"] for item in files if item["run_relative_path"] == "resolved_config.json"),
        "run_identity_sha256": run_identity_sha,
        "files": list(files),
    }

    plan_hash = "a" * 64
    config_hash = "b" * 64
    source_commit = "c" * 40
    lock = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "plan_sha256": plan_hash,
        "study_config_sha256": config_hash,
        "protocol_config_sha256": StudyConfig().sha256,
        "source_commit": source_commit,
        "source_tree_dirty": False,
        "fold_manifest_sha256": "d" * 64,
        "inventory_count": len(inventory),
        "inventory": inventory,
        "shard_rule": "unsigned_big_endian(first_8_bytes(SHA-256(job_id))) % shard_count",
    }
    compatibility = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "stage": "inner",
        "rows": [],
    }
    for path, data, schema in (
        (f"{STUDY_ID}/job_manifest.json", canonical_json_bytes(lock), "study_lock.v1"),
        (
            f"{STUDY_ID}/reuse_compatibility_inner.json",
            canonical_json_bytes(compatibility),
            COMPATIBILITY_SCHEMA_VERSION,
        ),
    ):
        payloads[path] = data
        files.append(
            {
                "path": path,
                "run_relative_path": Path(path).name,
                "schema": schema,
                "byte_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    fold_path = f"{STUDY_ID}/fold_manifest.json"
    files.append(
        {
            "path": fold_path,
            "run_relative_path": "fold_manifest.json",
            "schema": "nested_oof_manifest.v1",
            "byte_size": len(payloads[fold_path]),
            "sha256": hashlib.sha256(payloads[fold_path]).hexdigest(),
        }
    )
    files.sort(key=lambda entry: entry["path"])
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "study_config_sha256": config_hash,
        "protocol_config_sha256": StudyConfig().sha256,
        "plan_sha256": plan_hash,
        "source_commit": source_commit,
        "shard": {"stage": "inner", "index": 0, "count": 1, "selected_job_ids": [job_id]},
        "jobs": [job_payload],
        "historical_references": [],
        "files": files,
    }
    bundle = tmp_path / "valid.tar"
    _write_tar(bundle, manifest, payloads)
    return bundle, manifest, payloads


def _write_tar(path: Path, manifest: dict, payloads: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w", format=tarfile.GNU_FORMAT) as archive:
        encoded = canonical_json_bytes(manifest)
        archive.addfile(_tar_info("manifest.json", len(encoded)), io.BytesIO(encoded))
        for name in sorted(payloads):
            data = payloads[name]
            archive.addfile(_tar_info(name, len(data)), io.BytesIO(data))


def test_job_inventory_and_sharding_are_stable() -> None:
    inventory = build_job_inventory(synthetic_manager())
    assert len(inventory) == 300
    assert sum(job.stage == "inner" for job in inventory) == 240
    assert sum(job.stage == "outer" for job in inventory) == 60
    assert len({job.job_id for job in inventory}) == 300
    assert tuple(job.job_id for job in inventory) == tuple(sorted(job.job_id for job in inventory))
    for job in inventory[:20]:
        expected = int.from_bytes(hashlib.sha256(job.job_id.encode()).digest()[:8], "big")
        assert shard_key(job.job_id) == expected
        assert shard_for_job(job.job_id, 7) == expected % 7


def test_planner_reports_stage_scoped_inventory_and_freezes_inner_only(tmp_path: Path) -> None:
    planner = OOFMatrixPlanner(
        manager=synthetic_manager(),
        artifact_root=tmp_path / "oof",
        reuse_roots={},
    )
    path = planner.freeze(stage="inner")
    assert path.is_file()
    compatibility = json.loads((path.parent / "reuse_compatibility_inner.json").read_text())
    assert compatibility["stage"] == "inner"
    assert len(compatibility["rows"]) == 16
    assert not (path.parent / "reuse_compatibility_outer.json").exists()
    plan = planner.plan(stage="inner", shard_index=0, shard_count=11, max_jobs=3)
    assert plan.inventory_count == len(planner.jobs(stage="inner", shard_index=0, shard_count=11))
    assert plan.missing_count == plan.inventory_count
    assert plan.selected_count == min(3, plan.inventory_count)
    assert plan.estimated_gpu_hours >= 0
    assert plan.available_output_bytes >= 0


def test_valid_bundle_import_is_idempotent_and_preserves_bytes(tmp_path: Path) -> None:
    bundle, manifest, payloads = _valid_bundle_fixture(tmp_path)
    parsed, actual = _read_bundle(bundle)
    assert parsed["study_id"] == STUDY_ID
    assert actual == payloads
    destination = tmp_path / "artifacts"
    imported = import_bundle(bundle, artifact_root=destination)
    assert imported == (manifest["jobs"][0]["job_id"],)
    imported_path = destination / f"{STUDY_ID}/fold_manifest.json"
    assert imported_path.read_bytes() == payloads[f"{STUDY_ID}/fold_manifest.json"]
    assert import_bundle(bundle, artifact_root=destination) == imported
    imported_path.write_bytes(b"different immutable bytes")
    with pytest.raises(MatrixError, match="refusing to overwrite different existing bytes"):
        import_bundle(bundle, artifact_root=destination)
    assert imported_path.read_bytes() == b"different immutable bytes"


def test_public_restore_bootstraps_a_fresh_root_from_cumulative_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An inner bundle restores the immutable freeze and all native manifests."""
    from expert_method.cli import main as config_cli_main
    from expert_method.config import load_runtime_profile, load_study
    from expert_method.workflow import restore_bundles

    project_root = REPO_ROOT
    study_path = project_root / "configs/studies/ridge_sinkhorn_3seed_v1.yaml"
    source_root = tmp_path / "source-artifacts"
    source_profile_path = tmp_path / "source-profile.yaml"
    source_profile_path.write_text(
        (project_root / "configs/profiles/local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {source_root}"
        )
    )
    monkeypatch.setattr("expert_method.cli._load_manager", lambda *_args: synthetic_manager())
    current_source = config_cli_main.__globals__["_source_identity"](project_root)
    monkeypatch.setattr(
        "expert_method.cli._source_identity",
        lambda _root: {"commit": current_source["commit"], "dirty": False},
    )

    assert config_cli_main([
        "--config", str(study_path), "--profile", str(source_profile_path), "study", "freeze"
    ]) == 0
    capsys.readouterr()
    study = load_study(study_path)
    freeze_path = source_root / STUDY_ID / "manifests/study-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    matrix_root = source_root / STUDY_ID

    bundle, manifest, payloads = _valid_bundle_fixture(tmp_path)
    frozen_matrix = json.loads((matrix_root / "job_manifest.json").read_text())
    manifest.update({
        "study_freeze_sha256": freeze["freeze_sha256"],
        "study_config_sha256": frozen_matrix["study_config_sha256"],
        "protocol_config_sha256": frozen_matrix["protocol_config_sha256"],
        "plan_sha256": frozen_matrix["plan_sha256"],
        "source_commit": frozen_matrix["source_commit"],
    })
    payloads[f"{STUDY_ID}/fold_manifest.json"] = (matrix_root / "fold_manifest.json").read_bytes()
    payloads[f"{STUDY_ID}/job_manifest.json"] = (matrix_root / "job_manifest.json").read_bytes()
    payloads[f"{STUDY_ID}/reuse_compatibility_inner.json"] = (
        matrix_root / "reuse_compatibility_inner.json"
    ).read_bytes()
    freeze_member = f"{STUDY_ID}/manifests/study-freeze.json"
    payloads[freeze_member] = freeze_path.read_bytes()
    records_by_path = {record["path"]: record for record in manifest["files"]}
    for name, data in payloads.items():
        record = records_by_path.get(name)
        if record is None:
            record = {
                "path": name,
                "run_relative_path": name.removeprefix(f"{STUDY_ID}/"),
                "schema": "expert_method.study_freeze.v1",
            }
            records_by_path[name] = record
        record["byte_size"] = len(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
    manifest["files"] = sorted(records_by_path.values(), key=lambda item: item["path"])
    _write_tar(bundle, manifest, payloads)

    fresh_root = tmp_path / "fresh-kaggle-run-root"
    target_profile_path = tmp_path / "target-profile.yaml"
    target_profile_path.write_text(
        (project_root / "configs/profiles/local-smoke.yaml").read_text().replace(
            "run_root: runs/local-smoke", f"run_root: {fresh_root}"
        )
    )
    target_profile = load_runtime_profile(target_profile_path)
    restored = restore_bundles(study, target_profile, paths=(str(bundle),))

    assert restored["restored_job_count"] == 1
    assert restored["freeze_sha256"] == freeze["freeze_sha256"]
    assert (fresh_root / STUDY_ID / "manifests/study-freeze.json").read_bytes() == freeze_path.read_bytes()
    assert (fresh_root / STUDY_ID / "job_manifest.json").read_bytes() == (
        matrix_root / "job_manifest.json"
    ).read_bytes()
    again = restore_bundles(study, target_profile, paths=(str(bundle),))
    assert again["job_ids"] == restored["job_ids"]


def test_merge_accepts_compatible_bundles_with_different_shard_counts(tmp_path: Path) -> None:
    bundle, manifest, payloads = _valid_bundle_fixture(tmp_path)
    other_manifest = copy.deepcopy(manifest)
    other_count = 7
    job_id = other_manifest["jobs"][0]["job_id"]
    other_index = shard_for_job(job_id, other_count)
    other_manifest["shard"]["index"] = other_index
    other_manifest["shard"]["count"] = other_count
    other_manifest["jobs"][0]["shard_index"] = other_index
    other_manifest["jobs"][0]["shard_count"] = other_count
    other_bundle = tmp_path / "same-job-other-sharding.tar"
    _write_tar(other_bundle, other_manifest, payloads)

    destination = tmp_path / "merged-artifacts"
    assert merge_bundles([bundle, other_bundle], artifact_root=destination) == (job_id,)
    for relative, contents in payloads.items():
        assert (destination / relative).read_bytes() == contents


def test_bundle_rejects_parent_traversal_member(tmp_path: Path) -> None:
    bundle = tmp_path / "traversal.tar"
    with tarfile.open(bundle, mode="w", format=tarfile.GNU_FORMAT) as archive:
        first = canonical_json_bytes({"schema_version": BUNDLE_SCHEMA_VERSION})
        archive.addfile(_tar_info("manifest.json", len(first)), io.BytesIO(first))
        archive.addfile(_tar_info("../outside.txt", 1), io.BytesIO(b"x"))
    with pytest.raises(MatrixError, match="unsafe archive member"):
        _read_bundle(bundle)


def test_bundle_rejects_unowned_destination_and_job_identity(tmp_path: Path) -> None:
    bundle, manifest, payloads = _valid_bundle_fixture(tmp_path)
    outside_manifest = copy.deepcopy(manifest)
    outside_payloads = dict(payloads)
    outside_data = b"not a study artifact"
    outside_path = f"{STUDY_ID}/unowned/extra.bin"
    outside_payloads[outside_path] = outside_data
    outside_manifest["files"].append(
        {
            "path": outside_path,
            "run_relative_path": "extra.bin",
            "schema": "unowned.v1",
            "byte_size": len(outside_data),
            "sha256": hashlib.sha256(outside_data).hexdigest(),
        }
    )
    outside_bundle = tmp_path / "unowned.tar"
    _write_tar(outside_bundle, outside_manifest, outside_payloads)
    with pytest.raises(MatrixError, match="payload ownership mismatch"):
        _read_bundle(outside_bundle)

    wrong_id_manifest = copy.deepcopy(manifest)
    wrong_id_manifest["jobs"][0]["job_id"] = (
        f"{STUDY_ID}/inner/mixup/seed_78/outer_0/inner_0"
    )
    wrong_id_manifest["shard"]["selected_job_ids"] = [wrong_id_manifest["jobs"][0]["job_id"]]
    wrong_id_bundle = tmp_path / "wrong-job-id.tar"
    _write_tar(wrong_id_bundle, wrong_id_manifest, payloads)
    with pytest.raises(MatrixError, match="job ID fields do not match"):
        _read_bundle(wrong_id_bundle)


@pytest.mark.parametrize("operation", ("plan", "resolve", "export"))
def test_outer_artifact_access_is_gated_by_complete_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    planner = OOFMatrixPlanner(manager=synthetic_manager(), artifact_root=tmp_path / "oof")
    opened_outer_artifacts: list[str] = []

    def fail_lock_gate(**_kwargs):
        raise MatrixError("complete lock matrix required")

    monkeypatch.setattr(planner, "validate_complete_lock_matrix", fail_lock_gate)
    monkeypatch.setattr(
        planner,
        "_validate_native",
        lambda job: opened_outer_artifacts.append(job.job_id),
    )
    monkeypatch.setattr(
        planner,
        "_validate_historical",
        lambda job, _source: (opened_outer_artifacts.append(job.job_id), {}),
    )
    outer_job = next(job for job in planner.inventory if job.stage == "outer")
    if operation == "plan":
        with pytest.raises(MatrixError, match="complete lock matrix required"):
            planner.plan(stage="outer")
    elif operation == "resolve":
        with pytest.raises(MatrixError, match="complete lock matrix required"):
            planner.resolve_reference(outer_job)
    else:
        view = StudyArtifactView(
            manager=planner.manager,
            artifact_root=planner.artifact_root,
            training_configs=planner.training_configs,
        )
        # Make the view share the spy-gated planner instance.
        view.planner = planner
        with pytest.raises(MatrixError, match="complete lock matrix required"):
            export_bundle(
                view=view,
                destination=tmp_path / "outer.tar",
                stage="outer",
            )
        assert not (tmp_path / "outer.tar").exists()
    assert opened_outer_artifacts == []


@pytest.mark.parametrize("invalid_field", ("membership", "source_hashes"))
def test_outer_lock_validation_rechecks_inner_memberships_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_field: str
) -> None:
    from expert_method.ridge_sinkhorn.three_seed_study import StudyArtifactRepository

    manager = synthetic_manager()
    planner = OOFMatrixPlanner(manager=manager, artifact_root=tmp_path / "oof")
    source_digests = {
        "checkpoint": "1" * 64,
        "prediction": "2" * 64,
        "resolved_config": "3" * 64,
    }
    locks = []
    for seed in DEFAULT_SEEDS:
        for outer in range(5):
            fold_ids = [
                list(manager.inner_fold(outer, inner).prediction_indices)
                for inner in range(4)
            ]
            if invalid_field == "membership" and seed == DEFAULT_SEEDS[0] and outer == 0:
                # Keep a disjoint partition of the canonical outer-training set,
                # but move IDs across folds so it is internally plausible and
                # differs from the fold manager's canonical assignment.
                fold_ids[0][0], fold_ids[1][0] = fold_ids[1][0], fold_ids[0][0]
                fold_ids[0].sort()
                fold_ids[1].sort()
            memberships = []
            for inner, ids in enumerate(fold_ids):
                complement = sorted(
                    sample_id
                    for other, values in enumerate(fold_ids)
                    if other != inner
                    for sample_id in values
                )
                memberships.append(
                    SimpleNamespace(
                        inner_fold_id=inner,
                        validation_sample_ids=tuple(ids),
                        validation_membership_sha256=membership_sha256(ids),
                        router_training_membership_sha256=membership_sha256(complement),
                        router_training_sample_count=len(complement),
                    )
                )

            entries = [
                (f"{job.job_id}/{name}", digest)
                for job in planner.inventory
                if job.stage == "inner"
                and job.training_seed == seed
                and job.outer_fold_id == outer
                for name, digest in source_digests.items()
            ]
            if invalid_field == "source_hashes" and seed == DEFAULT_SEEDS[0] and outer == 0:
                entries[0] = (entries[0][0], "f" * 64)
            locks.append(
                SimpleNamespace(
                    training_seed=seed,
                    outer_fold_id=outer,
                    inner_fold_memberships=tuple(memberships),
                    source_hashes=tuple(sorted(entries)),
                )
            )

    monkeypatch.setattr(
        StudyArtifactRepository,
        "validate_complete_lock_matrix",
        lambda _repository, **_kwargs: tuple(locks),
    )
    resolved_jobs: list[str] = []

    def resolve_inner(job):
        assert job.stage == "inner"
        resolved_jobs.append(job.job_id)
        return (
            SimpleNamespace(
                checkpoint_sha256=source_digests["checkpoint"],
                prediction_sha256=source_digests["prediction"],
                resolved_config_sha256=source_digests["resolved_config"],
            ),
            "validated",
            "synthetic validated inner source",
        )

    monkeypatch.setattr(planner, "resolve_reference", resolve_inner)
    frozen = {"plan_sha256": "a" * 64, "source_commit": "b" * 40}
    with pytest.raises(MatrixError, match=(
        "canonical inner fold" if invalid_field == "membership" else "inner source hashes differ"
    )):
        planner.validate_complete_lock_matrix(frozen=frozen)

    assert planner._locks_validated is False
    assert all(job_id.startswith(f"{STUDY_ID}/inner/") for job_id in resolved_jobs)
    if invalid_field == "membership":
        assert resolved_jobs == []
    else:
        assert len(resolved_jobs) == 240


def test_cuda_preflight_fails_without_silent_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MatrixError, match="CUDA is unavailable"):
        cuda_preflight("cuda")
    cuda_preflight("cpu")


def test_outer_bundle_checks_locks_before_reading_any_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "study_config_sha256": "a" * 64,
        "protocol_config_sha256": StudyConfig().sha256,
        "plan_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "shard": {"stage": "outer", "index": 0, "count": 1, "selected_job_ids": []},
    }
    payloads = {f"{STUDY_ID}/expert_CE/seed_78/outer_0/outer_eval/predictions.json": b"outer labels"}
    bundle = tmp_path / "outer.tar"
    _write_tar(bundle, manifest, payloads)
    original_extractfile = tarfile.TarFile.extractfile
    extracted: list[str] = []

    def spy_extractfile(archive, member):
        name = member.name if isinstance(member, tarfile.TarInfo) else str(member)
        extracted.append(name)
        if name != "manifest.json":
            raise AssertionError("outer payload was read before its lock gate")
        return original_extractfile(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy_extractfile)
    with pytest.raises(MatrixError, match="frozen job_manifest.json"):
        _read_bundle(bundle, artifact_root=tmp_path / "empty-artifacts")
    assert extracted == ["manifest.json"]


@pytest.mark.parametrize(
    "gate_error",
    (
        "fold-lock validation membership differs from the canonical inner fold",
        "fold-lock inner source hashes differ from validated references",
    ),
)
def test_outer_bundle_uses_strong_planner_gate_before_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_error: str,
) -> None:
    import expert_method.ridge_sinkhorn.matrix as matrix_module

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "study_config_sha256": "a" * 64,
        "protocol_config_sha256": StudyConfig().sha256,
        "plan_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "shard": {"stage": "outer", "index": 0, "count": 1, "selected_job_ids": []},
    }
    artifact_root = tmp_path / "artifacts"
    study_root = artifact_root / STUDY_ID
    study_root.mkdir(parents=True)
    (study_root / "job_manifest.json").write_text(json.dumps({
        "study_id": STUDY_ID,
        "study_config_sha256": manifest["study_config_sha256"],
        "protocol_config_sha256": manifest["protocol_config_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "source_commit": manifest["source_commit"],
    }))

    calls: list[tuple[str, object]] = []
    manager = object()
    monkeypatch.setattr(
        matrix_module, "manager_from_fold_manifest",
        lambda path: calls.append(("manager", path)) or manager,
    )

    class RejectingPlanner:
        def __init__(self, **kwargs):
            calls.append(("planner", kwargs))

        def validate_complete_lock_matrix(self, *, frozen):
            calls.append(("gate", frozen))
            raise MatrixError(gate_error)

    monkeypatch.setattr(matrix_module, "OOFMatrixPlanner", RejectingPlanner)
    payloads = {
        f"{STUDY_ID}/expert_CE/seed_78/outer_0/outer_eval/predictions.json": b"outer labels",
    }
    bundle = tmp_path / "outer-strong-gate.tar"
    _write_tar(bundle, manifest, payloads)
    original_extractfile = tarfile.TarFile.extractfile
    extracted: list[str] = []

    def spy_extractfile(archive, member):
        name = member.name if isinstance(member, tarfile.TarInfo) else str(member)
        extracted.append(name)
        if name != "manifest.json":
            raise AssertionError("outer payload was read before the strong planner gate")
        return original_extractfile(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy_extractfile)
    with pytest.raises(MatrixError, match=gate_error):
        _read_bundle(
            bundle,
            artifact_root=artifact_root,
            reuse_roots={"historical": tmp_path / "reuse"},
        )
    assert [name for name, _value in calls] == ["manager", "planner", "gate"]
    assert extracted == ["manifest.json"]


def test_parser_accepts_documented_command_forms_verbatim() -> None:
    from scripts.run_ridge_sinkhorn_matrix import build_parser

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--plan", "--stage", "inner", "--study-id", STUDY_ID,
            "--data-root", "/kaggle/input/cifar100", "--artifact-root", "/kaggle/working/rs3/artifacts",
            "--reuse-root", "/kaggle/input/rs3-reuse", "--device", "cuda",
            "--shard-index", "0", "--shard-count", "4",
        ]
    )
    assert parsed.plan and parsed.stage == "inner" and parsed.shard_count == 4
    parsed = parser.parse_args(
        [
            "--study-id", STUDY_ID, "--import-bundle", "/kaggle/input/inner-s0.tar",
            "--artifact-root", "/kaggle/working/rs3/artifacts",
        ]
    )
    assert parsed.import_bundle.endswith("inner-s0.tar")
    parsed = parser.parse_args(
        [
            "--study-id", STUDY_ID, "--merge-bundles", "/kaggle/input/inner-s0.tar",
            "/kaggle/input/inner-s1.tar", "--artifact-root", "/kaggle/working/rs3/artifacts",
        ]
    )
    assert len(parsed.merge_bundles) == 2
