"""Synthetic regression tests for the fold-aware OOF execution pipeline.

The tests deliberately use a tiny in-memory training-like population.  They do
not load CIFAR-100 images, the CIFAR-100 test split, or canonical checkpoints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.nested_oof import (  # noqa: E402
    NestedOOFFoldManager,
    OOFPredictionArtifact,
    OOFArtifactValidationError,
)
from data.oof_datamodule import (  # noqa: E402
    FoldAwareDataModule,
    OOFDataError,
)
from scripts.oof_pipeline import (  # noqa: E402
    OOFArtifactError,
    OOFArtifactStore,
    OOFPipeline,
    OOFPredictionCollector,
    OOFRunSpec,
)
from scripts.config import TrainingConfig  # noqa: E402


def _population() -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(3, dtype=np.int64), [10, 7, 5])
    indices = np.arange(100, 100 + len(labels), dtype=np.int64)
    return indices, labels


def _manager() -> NestedOOFFoldManager:
    indices, labels = _population()
    return NestedOOFFoldManager(
        indices,
        labels,
        seed=17,
        num_classes=3,
        expert_order=("CE", "LAL"),
    )


class _SyntheticDataset(Dataset):
    def __init__(self, sample_indices, label_map, train):
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.sample_targets = np.asarray(
            [label_map[int(index)] for index in self.sample_indices], dtype=np.int64
        )
        self.train = bool(train)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, position):
        sample_index = int(self.sample_indices[position])
        image = torch.full((3, 4, 4), sample_index / 1000.0, dtype=torch.float32)
        return image, int(self.sample_targets[position])


def _dataset_factory(label_map):
    def factory(*, base_train_indices, train, **_kwargs):
        assert _kwargs.get("use_test_set") is False
        return _SyntheticDataset(base_train_indices, label_map, train)

    return factory


def _data_module(manager: NestedOOFFoldManager) -> FoldAwareDataModule:
    label_map = dict(zip(manager.canonical_indices, manager.training_labels))
    return FoldAwareDataModule(
        manager=manager,
        root="synthetic",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        dataset_factory=_dataset_factory(label_map),
        seed=78,
    )


def _config_payload(expert: str = "ce") -> dict:
    return {
        "expert": expert,
        "seed": 78,
        "device": "cpu",
        "model": {"arch": "synthetic", "num_classes": 3},
    }


def _checkpoint(path: Path, *, expert_name: str = "CE", seed: int = 78) -> str:
    model = nn.Linear(48, 3)
    state = {
        "epoch": 1,
        "seed": seed,
        "expert_name": expert_name,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": {},
        "is_final": True,
        "log": {},
    }
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def test_run_spec_distinguishes_inner_and_outer_memberships():
    manager = _manager()
    inner = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(manager)
    outer = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=None,
    ).resolve(manager)

    assert inner.role == "inner_oof"
    assert outer.role == "outer_evaluation"
    assert set(inner.training_indices).isdisjoint(inner.prediction_indices)
    assert set(outer.training_indices).isdisjoint(outer.prediction_indices)
    assert set(inner.training_indices) != set(outer.training_indices)
    assert inner.expert_name == "CE"
    assert inner.training_class_counts == manager.inner_fold(0, 0).expert_training_class_counts


def test_fold_aware_loaders_use_declared_memberships_and_counts():
    manager = _manager()
    data = _data_module(manager)
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=1,
    ).resolve(manager)

    assert np.array_equal(
        data.class_counts(context), np.asarray(context.training_class_counts)
    )
    training_loader = data.training_loader(context)
    prediction_loader = data.prediction_loader(context)
    assert tuple(training_loader.dataset.sample_indices.tolist()) == context.training_indices
    assert prediction_loader.dataset.sample_indices.tolist() == list(context.prediction_indices)
    assert training_loader.dataset.train is True
    assert prediction_loader.dataset.base_dataset.train is False
    assert set(training_loader.dataset.sample_indices).isdisjoint(
        prediction_loader.dataset.sample_indices
    )

    observed_prediction_indices = []
    for _images, _labels, sample_indices in prediction_loader:
        observed_prediction_indices.extend(int(value) for value in sample_indices.tolist())
    assert observed_prediction_indices == list(context.prediction_indices)


def test_outer_prediction_loader_excludes_all_development_data():
    manager = _manager()
    data = _data_module(manager)
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=2,
    ).resolve(manager)
    outer = manager.outer_fold(2)
    development = set(outer.expert_training_indices)
    observed = set(data.prediction_loader(context).dataset.sample_indices.tolist())
    assert observed == set(outer.evaluation_indices)
    assert observed.isdisjoint(development)
    assert observed.isdisjoint(outer.router_development.fit_indices)
    assert observed.isdisjoint(outer.router_development.selection_indices)


def test_membership_tampering_is_rejected_before_dataset_construction():
    manager = _manager()
    data = _data_module(manager)
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(manager)
    tampered = dataclasses.replace(
        context,
        training_indices=context.training_indices[:-1]
        + (manager.canonical_indices[-1],),
    )
    try:
        data.training_loader(tampered)
    except OOFDataError as exc:
        assert "membership" in str(exc)
    else:
        raise AssertionError("tampered training membership reached dataset construction")


def test_artifact_store_refuses_canonical_checkpoint_directory():
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        canonical = Path(tmp) / "checkpoints"
        try:
            OOFArtifactStore(
                root=canonical,
                manager=manager,
                experiment_id="synthetic",
                canonical_checkpoint_dir=canonical,
            )
        except OOFArtifactError as exc:
            assert "canonical checkpoint" in str(exc)
        else:
            raise AssertionError("OOF artifacts were allowed in canonical checkpoints")


def test_artifact_store_rejects_incompatible_run_reuse():
    manager = _manager()
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(manager)
    with tempfile.TemporaryDirectory() as tmp:
        store = OOFArtifactStore(
            root=Path(tmp) / "artifacts" / "oof",
            manager=manager,
            experiment_id="synthetic",
            canonical_checkpoint_dir=Path(tmp) / "canonical-checkpoints",
        )
        store.prepare_run(context, _config_payload())
        mismatched = dict(_config_payload())
        mismatched["seed"] = 999
        try:
            store.prepare_run(context, mismatched)
        except OOFArtifactError as exc:
            assert "overwrite" in str(exc) or "incompatible" in str(exc)
        else:
            raise AssertionError("incompatible run reuse was accepted")


def test_prediction_collection_is_deterministic_and_records_provenance():
    manager = _manager()
    data = _data_module(manager)
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(manager)

    class _Model(nn.Module):
        def forward(self, images):
            flat = images.reshape(images.shape[0], -1)
            return torch.stack((flat[:, 0], flat[:, 1] + 1, flat[:, 2] + 2), dim=1)

    class _Trainer:
        def __init__(self):
            self.model = _Model()
            self.device = "cpu"
            self.expert_name = "CE"
            self.seed = 78

        def load_checkpoint(self, _path):
            return None

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "checkpoint.pt"
        checkpoint_sha = _checkpoint(checkpoint_path)
        config = _config_payload()
        collector = OOFPredictionCollector(manager=manager, data_module=data)
        first = collector.collect(
            context=context,
            trainer=_Trainer(),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            resolved_config=config,
        )
        second = collector.collect(
            context=context,
            trainer=_Trainer(),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            resolved_config=config,
        )

        assert first.to_json() == second.to_json()
        first.validate(manager, require_complete=False)
        assert [record.sample_index for record in first.records] == list(
            context.prediction_indices
        )
        assert all(record.expert_id == "CE" for record in first.records)
        assert all(
            record.expert_training_membership_hash == context.training_membership_hash
            for record in first.records
        )


def test_prediction_collection_rejects_wrong_checkpoint_identity():
    manager = _manager()
    data = _data_module(manager)
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(manager)

    class _Trainer:
        def __init__(self):
            self.model = nn.Identity()
            self.device = "cpu"
            self.expert_name = "CE"
            self.seed = 78

        def load_checkpoint(self, _path):
            raise AssertionError("wrong checkpoint must be rejected before loading")

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "wrong.pt"
        checkpoint_sha = _checkpoint(checkpoint_path, expert_name="LAL")
        try:
            OOFPredictionCollector(manager=manager, data_module=data).collect(
                context=context,
                trainer=_Trainer(),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                resolved_config=_config_payload(),
            )
        except OOFArtifactError as exc:
            assert "incompatible" in str(exc)
        else:
            raise AssertionError("checkpoint from a different expert was accepted")


def test_store_rejects_incomplete_duplicate_or_mismatched_prediction_records():
    manager = _manager()
    data = _data_module(manager)
    context = OOFRunSpec(
        experiment_id="synthetic",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
    ).resolve(manager)

    with tempfile.TemporaryDirectory() as tmp:
        store = OOFArtifactStore(
            root=Path(tmp) / "artifacts" / "oof",
            manager=manager,
            experiment_id="synthetic",
            canonical_checkpoint_dir=Path(tmp) / "canonical-checkpoints",
        )
        store.prepare_run(context, _config_payload())
        checkpoint_path = store.checkpoint_path(context)
        checkpoint_sha = _checkpoint(checkpoint_path)
        class _Model(nn.Module):
            def forward(self, images):
                flat = images.reshape(images.shape[0], -1)
                return flat[:, :3]

        artifact = OOFPredictionCollector(manager=manager, data_module=data).collect(
            context=context,
            trainer=type(
                "Trainer",
                (),
                {
                    "model": _Model(),
                    "device": "cpu",
                    "expert_name": "CE",
                    "seed": 78,
                    "load_checkpoint": lambda self, _path: None,
                },
            )(),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            resolved_config=_config_payload(),
        )
        store.record_checkpoint(context, checkpoint_path, checkpoint_sha)
        store.write_predictions(context, artifact)
        restored = store.load_predictions(context)
        assert len(restored.records) == len(context.prediction_indices)

        duplicate = OOFPredictionArtifact(
            expert_order=manager.expert_order,
            num_classes=manager.num_classes,
            records=artifact.records[:-1] + (artifact.records[-2], artifact.records[-1]),
        )
        try:
            store.write_predictions(context, duplicate)
        except OOFArtifactValidationError as exc:
            assert "duplicate" in str(exc) or "sample" in str(exc)
        else:
            raise AssertionError("duplicate prediction records were accepted")

        misordered = OOFPredictionArtifact(
            expert_order=manager.expert_order,
            num_classes=manager.num_classes,
            records=(artifact.records[1], artifact.records[0]) + artifact.records[2:],
        )
        try:
            store.write_predictions(context, misordered)
        except OOFArtifactValidationError as exc:
            assert "misordered" in str(exc)
        else:
            raise AssertionError("misordered prediction records were accepted")


def test_pipeline_smoke_trains_only_the_declared_population():
    manager = _manager()
    context_spec = OOFRunSpec(
        experiment_id="synthetic_smoke",
        expert="ce",
        training_seed=78,
        outer_fold_id=0,
        inner_fold_id=0,
        device="cpu",
        epochs=1,
        max_batches=1,
    )
    observed = {"train_calls": 0}

    class _Model(nn.Module):
        def forward(self, images):
            flat = images.reshape(images.shape[0], -1)
            return flat[:, :3]

    class _Trainer:
        def __init__(self, config):
            self.model = _Model()
            self.device = "cpu"
            self.expert_name = "CE"
            self.seed = config.seed
            self.config = config

        def train(self, loader):
            observed["train_calls"] += 1
            observed["training_indices"] = tuple(
                loader.dataset.dataset.sample_indices.tolist()
                if hasattr(loader.dataset, "dataset")
                else loader.dataset.sample_indices.tolist()
            )
            next(iter(loader))
            path = (
                Path(self.config.checkpoint.dir)
                / f"CE_seed{self.seed}_final.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": self.config.schedule.epochs,
                    "seed": self.seed,
                    "expert_name": self.expert_name,
                    "model_state_dict": self.model.state_dict(),
                    "optimiser_state_dict": {},
                    "is_final": True,
                    "log": {},
                },
                path,
            )
            return [{"epoch": 1, "train_loss": 0.0}]

        def save_history(self, path):
            Path(path).write_text(json.dumps([{"epoch": 1}]))

        def load_checkpoint(self, _path):
            return None

    def trainer_builder(config, *, class_counts, device):
        observed["class_counts"] = tuple(class_counts.tolist())
        observed["device"] = device
        return _Trainer(config)

    with tempfile.TemporaryDirectory() as tmp:
        store = OOFArtifactStore(
            root=Path(tmp) / "artifacts" / "oof",
            manager=manager,
            experiment_id="synthetic_smoke",
            canonical_checkpoint_dir=Path(tmp) / "canonical-checkpoints",
        )
        pipeline = OOFPipeline(
            manager=manager,
            store=store,
            data_module_factory=lambda **_kwargs: _data_module(manager),
            trainer_builder=trainer_builder,
        )
        config = TrainingConfig.from_file("configs/ce.yaml").replace(
            device="cpu", epochs=1
        )
        result = pipeline.run(context_spec, config)
        recovered = pipeline.run(context_spec, config)

        expected = manager.inner_fold(0, 0)
        assert observed["train_calls"] == 1
        assert observed["training_indices"] == expected.expert_training_indices
        assert observed["class_counts"] == expected.expert_training_class_counts
        assert result.prediction_path.exists()
        assert result.checkpoint_path.is_relative_to(Path(tmp).resolve())
        assert result.context.prediction_indices == expected.prediction_indices
        assert recovered.prediction_artifact.to_json() == result.prediction_artifact.to_json()


TESTS = [
    ("inner versus outer run memberships", test_run_spec_distinguishes_inner_and_outer_memberships),
    ("fold-aware loaders and class counts", test_fold_aware_loaders_use_declared_memberships_and_counts),
    ("outer prediction excludes development", test_outer_prediction_loader_excludes_all_development_data),
    ("membership tampering is rejected early", test_membership_tampering_is_rejected_before_dataset_construction),
    ("canonical checkpoint guard", test_artifact_store_refuses_canonical_checkpoint_directory),
    ("incompatible run reuse is rejected", test_artifact_store_rejects_incompatible_run_reuse),
    ("deterministic prediction provenance", test_prediction_collection_is_deterministic_and_records_provenance),
    ("checkpoint identity validation", test_prediction_collection_rejects_wrong_checkpoint_identity),
    ("prediction completeness and duplicates", test_store_rejects_incomplete_duplicate_or_mismatched_prediction_records),
    ("one-fold training and prediction smoke", test_pipeline_smoke_trains_only_the_declared_population),
]


def main() -> int:
    passed = failed = 0
    for name, test in TESTS:
        try:
            test()
            print(f"  PASS {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - standalone test harness
            print(f"  FAIL {name}: {exc}")
            failed += 1
    print(f"OOF pipeline: {passed} passed, {failed} failed")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
