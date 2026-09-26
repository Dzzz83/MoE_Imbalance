"""Focused regression tests for numerical and protocol hardening fixes."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.utils.features import compute_energy


def test_compute_energy_is_finite_for_large_logits():
    logits = np.array([[1000.0, 0.0], [-1000.0, -1001.0]])
    got = compute_energy(logits)
    expected = np.asarray([-1000.0, 999.6867383124818])
    assert np.all(np.isfinite(got))
    assert np.allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_compute_energy_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        compute_energy(np.zeros((1, 2)), temperature=temperature)


def test_compute_energy_rejects_nonfinite_logits():
    with pytest.raises(ValueError, match="finite"):
        compute_energy(np.array([[np.inf, 0.0]]))
