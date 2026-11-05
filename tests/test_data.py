from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline import data


def test_retention_profile_and_fixed_population():
    labels = np.repeat(np.arange(100), 500)
    selected = data.select_long_tail(labels)
    counts = np.bincount(labels[selected], minlength=100)
    assert (len(selected), counts[0], counts[-1]) == (10847, 500, 5)
    assert np.all(np.diff(selected) > 0)
    np.testing.assert_array_equal(selected, data.select_long_tail(labels, seed=42))
    assert not np.array_equal(selected, data.select_long_tail(labels, seed=43))
    np.testing.assert_array_equal(data.select_long_tail(labels, 1.0), np.arange(50_000))


@pytest.mark.parametrize("factor", [0, -0.1, 1.1, np.nan, np.inf, "0.01", True])
def test_invalid_imbalance_factor(factor):
    with pytest.raises(ValueError, match="imbalance_factor"):
        data.select_long_tail(np.repeat(np.arange(100), 500), factor)


def test_small_positive_factor_keeps_every_class():
    labels = np.repeat(np.arange(100), 500)
    selected = data.select_long_tail(labels, 1e-320)
    counts = np.bincount(labels[selected], minlength=100)
    assert counts[0] == 500
    assert np.all(counts >= 1)


def test_preparation_selects_only_the_training_split(tmp_path, monkeypatch):
    labels = np.repeat(np.arange(100), 500)
    # A broadcast view identifies each source image without allocating 150 MB.
    images = np.broadcast_to(
        (np.arange(50_000) % 256).astype(np.uint8)[:, None, None, None],
        (50_000, 32, 32, 3),
    )
    calls = []

    def source(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=images, targets=labels.tolist())

    monkeypatch.setattr(data.datasets, "CIFAR100", source)
    path = tmp_path / "train.npz"
    prepared = data.prepare_data(tmp_path / "raw", path)
    assert calls == [{"root": str(tmp_path / "raw"), "train": True, "download": False}]
    selected = data.select_long_tail(labels)
    np.testing.assert_array_equal(prepared.images[:, 0, 0, 0], selected % 256)
    loaded = data.load_training_data(path)
    np.testing.assert_array_equal(loaded.labels, labels[selected])
    np.testing.assert_array_equal(loaded.images, prepared.images)
    assert loaded.data_settings == {"imbalance_factor": 0.01, "data_seed": 42}
    with pytest.raises(FileExistsError):
        data.prepare_data(tmp_path / "raw", path)
    assert len(calls) == 1


def test_saved_data_round_trip_and_views(tmp_path, training_data):
    path = tmp_path / "train.npz"
    data.save_training_data(path, training_data)
    loaded = data.load_training_data(path)
    np.testing.assert_array_equal(loaded.images, training_data.images)
    np.testing.assert_array_equal(loaded.class_counts, training_data.class_counts)
    view = data.make_dataset(loaded, data.evaluation_transform())
    image, label = view[0]
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert label == 0
    torch.testing.assert_close(image, view[0][0], rtol=0, atol=0)
    training_image, _ = data.make_dataset(loaded, data.training_transform())[0]
    assert training_image.shape == image.shape
    np.testing.assert_array_equal(loaded.images, training_data.images)
    with pytest.raises(FileExistsError):
        data.save_training_data(path, training_data)


def test_reject_mismatched_counts_and_discarded_source_shape(tmp_path, training_data):
    with pytest.raises(ValueError, match="class_counts do not match"):
        data.save_training_data(
            tmp_path / "bad.npz", replace(training_data, class_counts=np.ones(100, dtype=np.int64))
        )
    assert not (tmp_path / "bad.npz").exists()
    with pytest.raises(ValueError, match="50,000"):
        data.select_long_tail(training_data.labels)


def test_scalar_labels_are_reported_as_invalid_data(tmp_path, training_data):
    with pytest.raises(ValueError, match="labels must be"):
        data.save_training_data(
            tmp_path / "bad.npz", replace(training_data, labels=np.array(0, dtype=np.int64))
        )


def test_loader_rejects_inconsistent_file(tmp_path, training_data):
    path = tmp_path / "bad.npz"
    np.savez(
        path, images=training_data.images, labels=training_data.labels,
        class_counts=np.ones(100, dtype=np.int64), imbalance_factor=0.01, data_seed=42,
    )
    with pytest.raises(ValueError, match="class_counts do not match"):
        data.load_training_data(path)
