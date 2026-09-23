"""Stage 0 characterization tests for numerical and compatibility contracts.

All fixtures in this module are synthetic.  The committed numerical references
are read-only development summaries; no dataset, checkpoint, test population,
or OOF prediction array is loaded by these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.base_trainer import (
    balanced_accuracy as trainer_balanced_accuracy,
    compute_class_groups,
    group_accuracies as trainer_group_accuracies,
)
from scripts.evaluation import (
    balanced_accuracy as evaluation_balanced_accuracy,
    evaluate_predictions,
    group_accuracies as evaluation_group_accuracies,
)
from scripts.expert_diagnostics import DiagnosticInputError, ExpertDiagnostics
from scripts.router import (
    ConfidenceRouter,
    ProbabilityAverageRouter,
    UniformRouter,
)
from scripts.subsets import balanced_accuracy as subset_balanced_accuracy
from scripts.task3f_ridge import (
    Task3FError,
    classification_metrics,
    combine_weighted_logits,
    compute_contribution_targets,
    extract_features,
    scores_to_weights,
)
from scripts.utils.features import softmax
from scripts.utils.metrics import (
    balanced_accuracy as routing_balanced_accuracy,
    group_accuracies as routing_group_accuracies,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_raw_numpy_conversion_error(call):
    with pytest.raises(ValueError) as caught:
        call()
    assert type(caught.value) is ValueError
    assert "setting an array element with a sequence" in str(caught.value)


def _unequal_group_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels/predictions with unequal counts inside every group."""
    # Canonical groups are Head={0,1}, Medium={2,3}, Tail={4,5}.
    counts = np.array([100, 100, 20, 20, 5, 5], dtype=np.int64)
    labels = np.array(
        [0] + [1] * 9 + [2] + [3] * 3 + [4] + [5] * 3,
        dtype=np.int64,
    )
    # One class in each group is always correct and the other is always wrong.
    predictions = np.array(
        [0] + [0] * 9 + [2] + [2] * 3 + [4] + [4] * 3,
        dtype=np.int64,
    )
    return labels, predictions, counts


def test_active_softmax_is_stable_shape_preserving_and_normalized():
    """The active feature softmax uses max-shifting over the final axis."""
    logits = np.array(
        [
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[1000.0, 0.0, -1000.0], [-1000.0, -1001.0, -1002.0]],
            [[7.0, 7.0, 7.0], [0.0, 1000.0, 0.0]],
        ],
        dtype=np.float64,
    )

    probabilities = softmax(logits)

    assert probabilities.shape == logits.shape
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.allclose(
        probabilities[0, 0], np.exp(logits[0, 0]) / np.exp(logits[0, 0]).sum()
    )
    assert np.allclose(probabilities[2, 0], np.full(3, 1.0 / 3.0))
    assert probabilities[1, 0, 0] > 1.0 - 1e-12
    assert probabilities[2, 1, 1] > 1.0 - 1e-12


def test_softmax_nonfinite_behavior_is_distinct_from_ridge_input_validation():
    """The standalone helper propagates non-finite values; Ridge rejects them."""
    with np.errstate(all="ignore"):
        propagated = softmax(np.array([[np.nan, 0.0], [np.inf, 0.0]]))
    assert np.isnan(propagated).all()

    invalid_logits = np.zeros((1, 4, 3), dtype=np.float64)
    invalid_logits[0, 0, 0] = np.nan
    with pytest.raises(Task3FError, match="non-finite"):
        extract_features(invalid_logits, "confidence_only")


def test_parameter_free_routers_preserve_order_sentinels_and_first_ties():
    names = ["CE", "LAL", "BalancedSoftmax", "Mixup"]
    logits = np.zeros((2, 4, 3), dtype=np.float64)

    uniform = UniformRouter(names)
    probability = ProbabilityAverageRouter(names)
    confidence = ConfidenceRouter(names)

    # Combine-then-argmax routers retain the historical expert-zero sentinel.
    assert np.array_equal(uniform.predict(logits), np.zeros(2, dtype=np.int64))
    assert np.array_equal(probability.predict(logits), np.zeros(2, dtype=np.int64))
    assert uniform.selects_single_expert is False
    assert probability.selects_single_expert is False

    # NumPy argmax is the historical first-expert tie policy for all rules.
    assert np.array_equal(uniform.predict_class(logits), [0, 0])
    assert np.array_equal(probability.predict_class(logits), [0, 0])
    assert np.array_equal(confidence.predict(logits), [0, 0])


def test_weighted_logit_and_probability_combinations_keep_adaptive_rows_distinct():
    """Weighted logits and weighted probabilities are different operations."""
    logits = np.zeros((2, 4, 3), dtype=np.float64)
    # A high-scale outlier wins logit averaging; three bounded probabilities
    # agree on a different class.
    logits[0, 0, 0] = 100.0
    logits[0, 1:, 1] = 10.0
    logits[1, 0, 2] = 4.0
    logits[1, 1:, 1] = 3.0
    weights = np.array([[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]])

    combined_logits = combine_weighted_logits(logits, weights)
    combined_probabilities = np.einsum("ne,nec->nc", weights, softmax(logits))

    assert combined_logits.shape == (2, 3)
    assert combined_probabilities.shape == (2, 3)
    assert np.allclose(combined_logits, np.einsum("ne,nec->nc", weights, logits))
    assert np.allclose(combined_probabilities.sum(axis=1), 1.0)
    assert combined_logits.argmax(axis=1).tolist() == [0, 2]
    assert combined_probabilities.argmax(axis=1).tolist() == [1, 2]

    # The Ridge score transform is row-wise, not a single global composition.
    scores = np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    adaptive_weights = scores_to_weights(scores, temperature=1.0, shrinkage=0.5)
    assert adaptive_weights.shape == (2, 4)
    assert np.allclose(adaptive_weights.sum(axis=1), 1.0)
    assert not np.allclose(adaptive_weights[0], adaptive_weights[1])


def test_legacy_callers_preserve_raw_numpy_errors_for_ragged_inputs():
    ragged = [[0, 1], [2]]
    diagnostic_logits = np.zeros((2, 2, 3), dtype=np.float64)
    ridge_logits = np.zeros((2, 4, 3), dtype=np.float64)
    diagnostic = ExpertDiagnostics(logits=diagnostic_logits)

    calls = (
        lambda: ExpertDiagnostics(predictions=ragged),
        lambda: ExpertDiagnostics(
            predictions=np.zeros((2, 1), dtype=np.int64), labels=ragged
        ),
        lambda: diagnostic.evaluate_soft_mixture(ragged),
        lambda: extract_features(ragged, "confidence_only"),
        lambda: compute_contribution_targets(ridge_logits, ragged),
        lambda: scores_to_weights(ragged, temperature=1.0, shrinkage=1.0),
        lambda: combine_weighted_logits(ragged, np.full(4, 0.25)),
        lambda: combine_weighted_logits(ridge_logits, ragged),
    )
    for call in calls:
        _assert_raw_numpy_conversion_error(call)


def test_legacy_validation_keeps_task_specific_errors_after_conversion_fix():
    diagnostic = ExpertDiagnostics(logits=np.zeros((2, 2, 3), dtype=np.float64))
    logits = np.zeros((2, 4, 3), dtype=np.float64)

    with pytest.raises(DiagnosticInputError, match="finite"):
        ExpertDiagnostics(predictions=np.array([[0.0, np.nan]]))
    with pytest.raises(DiagnosticInputError, match="integer"):
        ExpertDiagnostics(predictions=np.array([[0.5, 1.0]]))
    with pytest.raises(DiagnosticInputError, match="shape"):
        diagnostic.evaluate_soft_mixture(np.ones((2, 3), dtype=np.float64))
    with pytest.raises(DiagnosticInputError, match="non-negative"):
        diagnostic.evaluate_soft_mixture(
            np.array([[0.5, 0.5], [1.1, -0.1]], dtype=np.float64)
        )
    with pytest.raises(DiagnosticInputError, match="sum to one"):
        diagnostic.evaluate_soft_mixture(
            np.array([[0.5, 0.5], [0.4, 0.4]], dtype=np.float64)
        )

    invalid_logits = logits.copy()
    invalid_logits[0, 0, 0] = np.nan
    with pytest.raises(Task3FError, match="finite"):
        extract_features(invalid_logits, "confidence_only")
    with pytest.raises(Task3FError, match="integer"):
        compute_contribution_targets(logits, np.array([0.5, 1.0]))
    with pytest.raises(Task3FError, match=r"\[0, 3\)"):
        compute_contribution_targets(logits, np.array([0, 3]))
    with pytest.raises(Task3FError, match="shape"):
        combine_weighted_logits(logits, np.ones((2, 3), dtype=np.float64))
    with pytest.raises(Task3FError, match="non-negative"):
        combine_weighted_logits(
            logits, np.array([[0.5, 0.5, 0.1, -0.1], [0.25] * 4])
        )
    with pytest.raises(Task3FError, match="sum to one"):
        combine_weighted_logits(
            logits, np.array([[0.4, 0.4, 0.1, 0.0], [0.25] * 4])
        )


def test_valid_legacy_callers_keep_exact_numeric_outputs():
    logits = np.array(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -2.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 2.0]],
            [[-2.0, 1.0, 0.0], [2.0, -1.0, 0.5], [0.0, 3.0, 1.0], [1.0, 0.0, -1.0]],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1], dtype=np.int64)
    weights = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])

    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    expected_features = probabilities.max(axis=2)
    expected_combined_logits = np.einsum("ne,nec->nc", weights, logits)
    expected_combined_probabilities = np.einsum(
        "ne,nec->nc", weights, probabilities
    )
    ensemble_logits = logits.mean(axis=1)
    ensemble_shifted = ensemble_logits - ensemble_logits.max(axis=1, keepdims=True)
    ensemble_probabilities = np.exp(ensemble_shifted)
    ensemble_probabilities /= ensemble_probabilities.sum(axis=1, keepdims=True)
    deltas = logits - ensemble_logits[:, None, :]
    expected_targets = (
        np.take_along_axis(deltas, labels[:, None, None], axis=2).squeeze(axis=2)
        - np.einsum("nc,nec->ne", ensemble_probabilities, deltas)
    )

    assert np.array_equal(extract_features(logits, "confidence_only"), expected_features)
    assert np.array_equal(compute_contribution_targets(logits, labels), expected_targets)
    assert np.array_equal(combine_weighted_logits(logits, weights), expected_combined_logits)

    diagnostic = ExpertDiagnostics(logits=logits)
    probability_report = diagnostic.evaluate_soft_mixture(weights, combination="probability")
    assert np.array_equal(probability_report["probabilities"], expected_combined_probabilities)
    assert np.array_equal(
        probability_report["predictions"], expected_combined_probabilities.argmax(axis=1)
    )


def test_metric_callers_preserve_sample_weighted_and_macro_group_definitions():
    labels, predictions, class_counts = _unequal_group_fixture()
    groups = compute_class_groups(class_counts)
    expected_sample = {"head": 0.1, "medium": 0.25, "tail": 0.25}
    expected_macro = {"head": 0.5, "medium": 0.5, "tail": 0.5}

    trainer_ba, _per_class = trainer_balanced_accuracy(labels, predictions)
    assert trainer_ba == pytest.approx(0.5)
    assert evaluation_balanced_accuracy(labels, predictions) == pytest.approx(0.5)
    assert routing_balanced_accuracy(labels, predictions) == pytest.approx(0.5)
    assert subset_balanced_accuracy(labels, predictions) == pytest.approx(0.5)

    trainer_groups = trainer_group_accuracies(labels, predictions, groups)
    evaluation_groups = evaluation_group_accuracies(labels, predictions, groups)
    routing_groups = routing_group_accuracies(
        labels,
        predictions,
        {"Head": groups["head"], "Med": groups["medium"], "Tail": groups["tail"]},
    )
    assert trainer_groups == pytest.approx(expected_sample)
    assert evaluation_groups == pytest.approx(expected_sample)
    assert routing_groups == pytest.approx(
        {"Head": 0.1, "Med": 0.25, "Tail": 0.25}
    )

    probabilities = np.eye(len(class_counts), dtype=np.float64)[predictions]
    evaluated = evaluate_predictions(labels, predictions, probabilities, class_counts)
    assert evaluated["head"] == pytest.approx(0.1)
    assert evaluated["medium"] == pytest.approx(0.25)
    assert evaluated["tail"] == pytest.approx(0.25)

    diagnostic = ExpertDiagnostics(
        predictions=predictions[:, None],
        labels=labels,
        class_counts=class_counts,
    )
    macro = diagnostic.hard_routing_headroom()["hard_selection_oracle_metrics"]
    assert {name: macro[name] for name in ("head", "medium", "tail")} == pytest.approx(
        expected_macro
    )

    ridge_metrics = classification_metrics(labels, predictions, class_counts)
    assert {
        "head": ridge_metrics["head_accuracy"],
        "medium": ridge_metrics["medium_accuracy"],
        "tail": ridge_metrics["tail_accuracy"],
    } == pytest.approx(expected_macro)
    assert evaluated["head"] != macro["head"]


@pytest.mark.parametrize("predictions, expected", [("correct", 1.0), ("wrong", 0.0)])
def test_metric_callers_define_all_correct_and_all_wrong_cases(predictions, expected):
    labels, _original_predictions, class_counts = _unequal_group_fixture()
    if predictions == "correct":
        selected = labels.copy()
    else:
        selected = (labels + 1) % len(class_counts)
    probabilities = np.eye(len(class_counts), dtype=np.float64)[selected]

    assert evaluation_balanced_accuracy(labels, selected) == pytest.approx(expected)
    assert routing_balanced_accuracy(labels, selected) == pytest.approx(expected)
    assert evaluate_predictions(labels, selected, probabilities, class_counts)["ba"] == pytest.approx(
        expected
    )
    assert classification_metrics(labels, selected, class_counts)["balanced_accuracy"] == pytest.approx(
        expected
    )


def test_absent_class_policies_remain_explicit_and_unified_by_no_refactor():
    labels = np.array([0, 0, 1], dtype=np.int64)
    predictions = np.array([0, 1, 1], dtype=np.int64)
    class_counts = np.array([100, 20, 5], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64)[predictions]

    # Present-class implementations omit class 2; the explicit class-count
    # argument in scripts.utils.metrics includes it with zero recall.
    assert evaluation_balanced_accuracy(labels, predictions) == pytest.approx(0.75)
    assert subset_balanced_accuracy(labels, predictions) == pytest.approx(0.75)
    assert routing_balanced_accuracy(labels, predictions) == pytest.approx(0.75)
    assert routing_balanced_accuracy(labels, predictions, num_classes=3) == pytest.approx(0.5)
    trainer_ba, _ = trainer_balanced_accuracy(labels, predictions)
    assert trainer_ba == pytest.approx(0.75)

    groups = compute_class_groups(class_counts)
    assert evaluation_group_accuracies(labels, predictions, groups)["tail"] == 0.0
    assert trainer_group_accuracies(labels, predictions, groups)["tail"] == 0.0

    with pytest.raises(DiagnosticInputError, match="missing classes"):
        ExpertDiagnostics(
            predictions=predictions[:, None],
            labels=labels,
            class_counts=class_counts,
        )
    with pytest.raises(Task3FError, match="missing classes"):
        classification_metrics(labels, predictions, class_counts)


def test_committed_development_references_use_full_precision_machine_artifacts():
    """Protect the frozen Stage 3 references without rerunning an analysis."""
    fixed_path = _PROJECT_ROOT / "artifacts/oof/task3e_fixed_feasibility/fixed_weight_results.json"
    ridge_path = _PROJECT_ROOT / "artifacts/oof/task3f_ridge/ridge_results.json"
    fixed = json.loads(fixed_path.read_text())
    ridge = json.loads(ridge_path.read_text())

    assert fixed["analysis_partition"] == {
        "inner_fold_ids": [1, 2, 3],
        "outer_evaluation_excluded": True,
        "outer_fold": 0,
        "router_selection_inner_fold_excluded": True,
        "sample_count": 6507,
    }
    fixed_rows = {row["candidate_id"]: row for row in fixed["candidates"]}
    assert fixed_rows["fixed_020"]["balanced_accuracy"] == pytest.approx(
        0.35932756391263865, abs=1e-15
    )
    assert fixed_rows["fixed_020"]["tail_accuracy"] == pytest.approx(
        0.07009379509379508, abs=1e-15
    )
    assert fixed_rows["fixed_006"]["balanced_accuracy"] == pytest.approx(
        0.3721341147209944, abs=1e-15
    )
    assert fixed_rows["fixed_006"]["tail_accuracy"] == pytest.approx(
        0.09911495911495911, abs=1e-15
    )
    assert fixed_rows["fixed_007"]["balanced_accuracy"] == pytest.approx(
        0.36440310027522765, abs=1e-15
    )
    assert fixed_rows["fixed_007"]["tail_accuracy"] == pytest.approx(
        0.14728956228956228, abs=1e-15
    )

    ridge_row = next(
        row
        for row in ridge["adaptive_results"]
        if row["configuration_id"]
        == "adaptive_confidence_only_alpha1000_gamma1_temperature2_lambda0p75"
    )
    assert ridge_row["metrics"]["balanced_accuracy"] == pytest.approx(
        0.3655023874404003, abs=1e-15
    )
    assert ridge_row["metrics"]["tail_accuracy"] == pytest.approx(
        0.07379749879749879, abs=1e-15
    )


def test_legacy_test_loader_authorizes_but_does_not_consume_its_grant(
    monkeypatch, tmp_path: Path
):
    """Characterize the legacy capability gap without constructing CIFAR data."""
    import scripts.utils.data as data_utils

    events: list[tuple[str, str]] = []

    class SyntheticGrant:
        def __init__(self):
            self.consumed = False

        def consume(self):
            self.consumed = True

    grant = SyntheticGrant()

    class SyntheticAccessLog:
        def __init__(self, path):
            events.append(("init", str(path)))

        def authorize(self, command, note=""):
            events.append((command, note))
            return grant

    class SyntheticDataset:
        def __init__(self, *args, **kwargs):
            events.append(("dataset", kwargs.get("train"), kwargs.get("use_test_set")))

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return np.zeros((3, 2, 2), dtype=np.float32), 0

        def get_class_counts(self):
            return np.array([1], dtype=np.int64)

    monkeypatch.setattr(data_utils, "TestAccessLog", SyntheticAccessLog)
    monkeypatch.setattr(data_utils, "LongTailCIFAR100", SyntheticDataset)
    data_utils.create_cifar_loader(
        "test",
        data_root=tmp_path,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    assert events[1][1] == "legacy create_cifar_loader test-set read"
    assert events[2] == ("dataset", False, True)
    assert grant.consumed is False
