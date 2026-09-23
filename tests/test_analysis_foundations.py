"""Synthetic tests for Stage 1 analysis foundations.

No project dataset, checkpoint, OOF population, or historical result artifact
is loaded here.  Expected numerical values are calculated independently of the
shared implementation where practical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.artifacts import (
    ArtifactError,
    canonical_json,
    git_commit,
    load_json_object,
    load_npz_arrays,
    repository_relative,
    serialize_json,
    sha256_array,
    sha256_file,
    verify_file_hash,
    write_json_once,
    write_npz_once,
    write_text_once,
)
from scripts.analysis.combination import (
    combine_weighted_logits,
    combine_weighted_probabilities,
    stable_softmax,
)
from scripts.analysis.validation import (
    validate_expert_weights,
    validate_integer_vector,
    validate_numeric_array,
)


def _logit_fixture() -> np.ndarray:
    return np.asarray(
        [
            [
                [1000.0, 0.0, -1000.0],
                [0.0, 10.0, -10.0],
                [-2.0, 3.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [-1000.0, -1001.0, -1002.0],
                [4.0, 3.0, 2.0],
                [0.0, 0.0, 0.0],
                [2.0, 5.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )


def _independent_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def test_numeric_and_integer_validators_enforce_explicit_shapes_and_finiteness():
    accepted = validate_numeric_array(
        [[1, 2], [3, 4]], shape=(2, 2), cast_dtype=np.float64
    )
    assert accepted.dtype == np.float64
    assert accepted.shape == (2, 2)

    with pytest.raises(ValueError, match="shape"):
        validate_numeric_array(np.zeros((2, 3)), shape=(2, 2))
    with pytest.raises(ValueError, match="non-finite"):
        validate_numeric_array(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="real numeric"):
        validate_numeric_array(np.array([1.0 + 2.0j]))

    integer_values = validate_integer_vector([1.0, 2.0, 3.0], shape=(3,))
    assert np.array_equal(integer_values, np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="integer"):
        validate_integer_vector([1.0, 2.5, 3.0])
    with pytest.raises(ValueError, match="shape"):
        validate_integer_vector(np.zeros((2, 2), dtype=np.int64), shape=(3,))
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        validate_integer_vector([0, 3], upper_bound=3)


def test_shared_validators_can_preserve_raw_numpy_conversion_errors():
    ragged = [[0, 1], [2]]

    with pytest.raises(ValueError) as numeric_error:
        validate_numeric_array(ragged, wrap_conversion_errors=False)
    assert type(numeric_error.value) is ValueError
    assert "setting an array element with a sequence" in str(numeric_error.value)

    with pytest.raises(ValueError) as integer_error:
        validate_integer_vector(ragged, wrap_conversion_errors=False)
    assert type(integer_error.value) is ValueError
    assert "setting an array element with a sequence" in str(integer_error.value)

    # The generic shared interface continues to provide its task-independent
    # wrapped error by default.
    with pytest.raises(ValueError, match="numeric array"):
        validate_numeric_array(ragged)
    with pytest.raises(ValueError, match="integer values"):
        validate_integer_vector(ragged)


def test_expert_weight_validation_preserves_order_and_sum_tolerance():
    global_weights = validate_expert_weights(
        [0.1, 0.2, 0.3, 0.4],
        num_experts=4,
        num_samples=2,
        allow_vector=True,
    )
    assert global_weights.shape == (2, 4)
    assert np.array_equal(global_weights[0], [0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(global_weights[1], global_weights[0])

    within_tolerance = np.asarray(
        [[0.25, 0.25, 0.25, 0.25000005]], dtype=np.float64
    )
    validate_expert_weights(
        within_tolerance,
        num_experts=4,
        num_samples=1,
        sum_atol=1e-7,
    )
    with pytest.raises(ValueError, match="sum to one"):
        validate_expert_weights(
            [[0.25, 0.25, 0.25, 0.250001]],
            num_experts=4,
            num_samples=1,
            sum_atol=1e-7,
        )
    with pytest.raises(ValueError, match="non-negative"):
        validate_expert_weights(
            [[0.5, 0.5, 0.1, -0.1]], num_experts=4, num_samples=1
        )
    with pytest.raises(ValueError, match="shape"):
        validate_expert_weights(
            [[0.5, 0.5, 0.0]], num_experts=4, num_samples=1
        )


def test_weighted_logit_and_probability_combination_match_independent_formulas():
    logits = _logit_fixture()
    weights = np.asarray(
        [[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]],
        dtype=np.float64,
    )
    expected_logits = np.einsum("ne,nec->nc", weights, logits)
    expected_probabilities = np.einsum(
        "ne,nec->nc", weights, _independent_softmax(logits)
    )

    combined_logits = combine_weighted_logits(logits, weights, num_experts=4)
    combined_probabilities = combine_weighted_probabilities(
        logits, weights, num_experts=4
    )

    assert np.array_equal(combined_logits, expected_logits)
    assert np.array_equal(combined_probabilities, expected_probabilities)
    assert np.array_equal(
        combined_logits.argmax(axis=1), expected_logits.argmax(axis=1)
    )
    assert np.array_equal(
        combined_probabilities.argmax(axis=1), expected_probabilities.argmax(axis=1)
    )
    assert np.allclose(combined_probabilities.sum(axis=1), 1.0)

    reordered_logits = logits[:, [3, 1, 0, 2], :]
    reordered_weights = weights[:, [3, 1, 0, 2]]
    assert np.allclose(
        combine_weighted_logits(reordered_logits, reordered_weights, num_experts=4),
        combined_logits,
    )


def test_stable_softmax_handles_extreme_values_but_combination_rejects_nonfinite_inputs():
    extreme = np.asarray([[1000.0, 0.0, -1000.0], [7.0, 7.0, 7.0]])
    probabilities = stable_softmax(extreme)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > 1.0 - 1e-12
    assert np.allclose(probabilities[1], 1.0 / 3.0)

    with np.errstate(all="ignore"):
        propagated = stable_softmax(np.asarray([[np.nan, 0.0]]))
    assert np.isnan(propagated).all()
    with pytest.raises(ValueError, match="non-finite"):
        combine_weighted_logits(
            np.asarray([[[np.nan, 0.0], [0.0, 1.0]]]),
            [0.5, 0.5],
            num_experts=2,
        )
    with pytest.raises(ValueError, match="sum to one"):
        combine_weighted_probabilities(
            np.zeros((1, 2, 2)), [[0.4, 0.4]], num_experts=2
        )


def test_artifact_json_round_trip_and_canonical_serialization(tmp_path: Path):
    payload = {
        "schema_version": "synthetic.v1",
        "array": np.asarray([1, 2, 3], dtype=np.int64),
        "scalar": np.float64(2.5),
        "nested": {"b": True, "a": ("x", "y")},
    }
    path = tmp_path / "artifact.json"
    write_json_once(path, payload)
    loaded = load_json_object(path)
    assert loaded == {
        "array": [1, 2, 3],
        "nested": {"a": ["x", "y"], "b": True},
        "scalar": 2.5,
        "schema_version": "synthetic.v1",
    }
    assert path.read_text() == serialize_json(payload)
    write_json_once(path, payload)
    with pytest.raises(ArtifactError, match="overwrite"):
        write_json_once(path, {"schema_version": "synthetic.v2"})

    assert canonical_json({"b": 2, "a": [1, 3]}) == '{"a":[1,3],"b":2}'
    assert json.loads(canonical_json({"value": 1})) == {"value": 1}


def test_artifact_npz_round_trip_write_once_and_array_hashes(tmp_path: Path):
    arrays = {
        "ids": np.asarray([1, 2, 3], dtype=np.int64),
        "scores": np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64),
    }
    path = tmp_path / "arrays.npz"
    write_npz_once(path, arrays)
    restored = load_npz_arrays(path)
    assert set(restored) == set(arrays)
    assert restored["ids"].dtype == arrays["ids"].dtype
    assert np.array_equal(restored["ids"], arrays["ids"])
    assert np.array_equal(restored["scores"], arrays["scores"], equal_nan=True)
    write_npz_once(path, arrays)
    with pytest.raises(ArtifactError, match="array differs"):
        write_npz_once(path, {**arrays, "ids": np.asarray([1, 2, 4], dtype=np.int64)})
    assert sha256_array(arrays["ids"]) == hashlib.sha256(
        np.ascontiguousarray(arrays["ids"]).tobytes()
    ).hexdigest()


def test_file_hash_verification_and_provenance_helpers(tmp_path: Path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"immutable synthetic source")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected
    assert verify_file_hash(path, expected.upper(), name="source") == expected
    with pytest.raises(ArtifactError, match="does not match"):
        verify_file_hash(path, "0" * 64, name="source")

    assert repository_relative(path, tmp_path) == "source.bin"
    assert len(git_commit(Path(__file__).resolve().parents[1])) == 40


def test_write_text_once_preserves_incompatible_existing_content(tmp_path: Path):
    path = tmp_path / "text" / "result.txt"
    write_text_once(path, "original\n")
    write_text_once(path, "original\n")
    with pytest.raises(ArtifactError, match="overwrite"):
        write_text_once(path, "replacement\n")
    assert path.read_text() == "original\n"
