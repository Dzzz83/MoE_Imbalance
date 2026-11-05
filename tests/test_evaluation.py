import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.data import evaluation_transform, make_dataset
from pipeline.evaluate import compute_metrics, predict
from pipeline.model import create_model


def test_metrics_use_macro_recall_and_training_group_boundaries():
    labels = np.array([0, 0, 0, 1, *range(2, 100)])
    predictions = np.array([0, 1, 1, 1, *range(2, 100)])
    counts = np.full(100, 19)
    counts[:2] = 100
    counts[2:4] = [20, 99]
    metrics = compute_metrics(predictions, labels, counts)
    assert metrics["accuracy"] == pytest.approx(100 / 102)
    assert metrics["balanced_accuracy"] == pytest.approx((99 + 1 / 3) / 100)
    assert metrics["head_recall"] == pytest.approx((1 / 3 + 1) / 2)
    assert metrics["medium_recall"] == 1
    assert metrics["tail_recall"] == 1
    assert metrics["test_support"] == [3, *([1] * 99)]


def test_missing_classes_and_empty_groups_are_explicit():
    metrics = compute_metrics([0, 0], [0, 0], np.full(100, 100))
    assert metrics["accuracy"] == 1
    assert metrics["per_class_recall"] == [1, *([None] * 99)]
    for name in ("balanced_accuracy", "head_recall", "medium_recall", "tail_recall"):
        assert metrics[name] is None


@pytest.mark.parametrize(
    "predictions,labels", [([], []), ([0], [0, 1]), ([-1], [0]), ([100], [0]), ([0.5], [0])]
)
def test_invalid_metric_inputs(predictions, labels):
    with pytest.raises(ValueError):
        compute_metrics(predictions, labels, np.ones(100, dtype=np.int64))


def test_inference_keeps_partial_batches_and_model_state(training_data):
    model = create_model()
    loader = DataLoader(
        make_dataset(training_data, evaluation_transform()), batch_size=3, shuffle=False
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    predictions, labels = predict(model, loader, torch.device("cpu"))
    assert predictions.shape == (7,)
    np.testing.assert_array_equal(labels, training_data.labels)
    repeated_predictions, _ = predict(model, loader, torch.device("cpu"))
    np.testing.assert_array_equal(predictions, repeated_predictions)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
