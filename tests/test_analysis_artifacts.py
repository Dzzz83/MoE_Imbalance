"""Regression tests for dependency-light immutable analysis artifacts."""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis import ArtifactReader, ImmutableArtifactWriter


def _run_concurrent(calls):
    errors = []

    def run(call):
        try:
            call()
        except Exception as exc:  # pragma: no cover - asserted by callers
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(call,)) for call in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    return errors


def test_json_writer_is_idempotent_and_rejects_conflicts(tmp_path):
    path = tmp_path / "nested" / "result.json"
    writer = ImmutableArtifactWriter()

    writer.write_json_once(path, {"value": np.int64(3)})
    first = path.read_bytes()
    writer.write_json_once(path, {"value": np.int64(3)})
    assert path.read_bytes() == first

    with pytest.raises(ValueError, match="incompatible"):
        writer.write_json_once(path, {"value": 4})
    assert path.read_bytes() == first


def test_text_writer_is_idempotent_and_npz_writer_compares_arrays(tmp_path):
    writer = ImmutableArtifactWriter()
    text_path = tmp_path / "summary.md"
    writer.write_text_once(text_path, "stable\n")
    writer.write_text_once(text_path, "stable\n")
    with pytest.raises(ValueError, match="incompatible"):
        writer.write_text_once(text_path, "changed\n")

    npz_path = tmp_path / "arrays.npz"
    arrays = {"values": np.array([1.0, np.nan]), "ids": np.array([1, 2])}
    writer.write_npz_once(npz_path, arrays)
    writer.write_npz_once(npz_path, arrays)
    loaded = ArtifactReader().read_npz(npz_path)
    np.testing.assert_equal(loaded["values"], arrays["values"])
    np.testing.assert_array_equal(loaded["ids"], arrays["ids"])
    with pytest.raises(ValueError, match="differ"):
        writer.write_npz_once(
            npz_path,
            {"values": np.array([1.0, 4.0]), "ids": np.array([1, 2])},
        )


@pytest.mark.parametrize("kind", ["json", "npz"])
@pytest.mark.parametrize("conflicting", [False, True])
def test_concurrent_writers_are_no_clobber(monkeypatch, tmp_path, kind, conflicting):
    """A link race accepts equal payloads and rejects incompatible winners."""
    writer = ImmutableArtifactWriter()
    path = tmp_path / ("result.json" if kind == "json" else "arrays.npz")
    rendezvous = threading.Barrier(2)
    real_link = os.link

    def coordinated_link(source, destination, *args, **kwargs):
        # Both temporary files are complete before either writer attempts the
        # install.  Exactly one real link can therefore win the race.
        rendezvous.wait(timeout=5)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", coordinated_link)

    if kind == "json":
        payloads = (
            {"value": 1},
            {"value": 1 if not conflicting else 2},
        )
        calls = [lambda payload=payload: writer.write_json_once(path, payload) for payload in payloads]
    else:
        arrays = {"values": np.array([1, 2])}
        competing_arrays = {"values": np.array([1, 2 if not conflicting else 3])}
        calls = [
            lambda payload=arrays: writer.write_npz_once(path, payload),
            lambda payload=competing_arrays: writer.write_npz_once(path, payload),
        ]

    errors = _run_concurrent(calls)
    if conflicting:
        assert len(errors) == 1
        assert "incompatible" in str(errors[0]) or "differ" in str(errors[0])
    else:
        assert errors == []
    assert path.exists()
    assert set(tmp_path.iterdir()) == {path}
