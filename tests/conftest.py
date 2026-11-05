import numpy as np
import pytest
import torch

from pipeline.data import TrainingData


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    # Tiny images and batches are slower with a large CPU thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def training_data():
    labels = np.array([0, 0, 1, 1, 2, 3, 99], dtype=np.int64)
    images = np.random.default_rng(0).integers(0, 256, (7, 32, 32, 3), dtype=np.uint8)
    return TrainingData(
        images=images,
        labels=labels,
        class_counts=np.bincount(labels, minlength=100),
        data_settings={"imbalance_factor": 0.01, "data_seed": 42},
    )
