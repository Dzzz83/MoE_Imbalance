"""
Tests for the CIFAR-100-LT data protocol (single training artifact, no validation).

Protocol (current):
    CIFAR-100 train (50K, balanced)
      └── apply imbalance factor 0.01 (IR=100)  -> 10,847 samples
           └── ALL of them are the training set      -> lt_ir100_train_indices.npy
    CIFAR-100 test (10K, balanced) is the only evaluation set.

There is deliberately NO validation split. These tests are the enforcement
mechanism: if any of them fails, a training run could silently see data it must
not, or the reported protocol could drift away from the standard benchmark.
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np

# Imported as a module (not as names) so that a missing function reports as one
# failing test instead of aborting the whole file at import time.
from data import protocol_splits as ps

DATA_ROOT = os.path.join(_proj_root, 'data')
PROCESSED = os.path.join(DATA_ROOT, 'processed')

# Standard CIFAR-100-LT IR=100 profile: n_i = 500 * 100 ** (-i / 99)
EXPECTED_TOTAL = 10847
EXPECTED_HEAD = 500
EXPECTED_TAIL = 5
N_CLASSES = 100


def _targets() -> np.ndarray:
    from torchvision import datasets
    return np.asarray(
        datasets.CIFAR100(root=DATA_ROOT, train=True, download=False).targets
    )


def _expected_profile() -> np.ndarray:
    i = np.arange(N_CLASSES, dtype=np.float64)
    raw = 500.0 * (100.0 ** (-i / (N_CLASSES - 1)))
    return np.maximum(raw.astype(np.int64), 1)


def test_canonical_artifact_exists():
    """The single canonical training artifact must exist."""
    path = os.path.join(PROCESSED, ps.LT_TRAIN_FILENAME)
    assert os.path.exists(path), f"missing canonical artifact: {path}"
    print(f"  ✅ {ps.LT_TRAIN_FILENAME} exists")


def test_no_validation_artifact():
    """No validation artifact the protocol could read may exist.

    The protocol owns three legacy filenames (see ``protocol_splits``); all three
    must be absent. Any *other* artifact whose name mentions a validation split is
    reported, not failed: data files are never deleted by a test, and the retired
    scripts that produced them are already documented as dead.
    """
    for stale in (ps.LEGACY_VAL_FILENAME, ps.LEGACY_TRAIN_FILENAME,
                  ps.LEGACY_ALL_FILENAME):
        path = os.path.join(PROCESSED, stale)
        assert not os.path.exists(path), (
            f"validation artifact still present: {path} — the protocol has no val split"
        )

    # No live module may read a validation artifact either. The training path is
    # covered by tests/test_training_protocol.py; this covers the data layer.
    # `protocol_splits` itself is exempt: it names the removed artifacts on
    # purpose, to detect them.
    banned = ('lt_val_indices', 'lt_all_indices', 'lt_train_indices.npy',
              'balanced_val_indices', 'val_targets')
    for rel in ('data/lt_datamodule.py', 'data/cifar_lt.py',
                'scripts/utils/data.py', 'scripts/evaluate_experts.py',
                'scripts/analyze_subsets.py'):
        with open(os.path.join(_proj_root, rel)) as f:
            hits = [token for token in banned if token in f.read()]
        assert not hits, f"{rel} still references removed artifacts: {hits}"

    leftovers = sorted(name for name in os.listdir(PROCESSED)
                       if 'val' in name and name != ps.LT_TRAIN_FILENAME)
    if leftovers:
        print(f"  ⚠️  retired-split leftovers on disk (unread by the live path): "
              f"{leftovers}")
    print("  ✅ no validation artifact the protocol can read")


def test_class_group_sizes_match_the_artifact():
    """Head/Med/Tail sizes must describe the committed split, not a fixture.

    README and docs state these counts. They were once copied from a synthetic
    test fixture (30/36/34) instead of the real split, which is 35/35/30 under
    the frozen thresholds (Head >= 100, Medium 20 <= n < 100, Tail < 20).
    """
    from scripts.base_trainer import compute_class_groups

    counts = np.bincount(_targets()[ps.load_lt_train_indices(DATA_ROOT)],
                         minlength=N_CLASSES)
    groups = compute_class_groups(counts)
    sizes = (len(groups['head']), len(groups['medium']), len(groups['tail']))
    assert sizes == (35, 35, 30), (
        f"the committed split gives Head/Med/Tail = {sizes}, but the documented "
        f"counts are 35/35/30 — one of the two is wrong"
    )
    boundary = np.where(counts == 20)[0]
    assert len(boundary) == 1, f"expected one class with exactly 20 samples, got {boundary}"
    assert boundary[0] in groups['medium'], (
        f"class {boundary[0]} has exactly 20 samples and must be Medium "
        f"(Medium is 20 <= n < 100)"
    )
    print(f"  ✅ class groups: head={sizes[0]}, medium={sizes[1]}, tail={sizes[2]} "
          f"(boundary class {int(boundary[0])} has 20 samples)")


def test_legacy_loader_reads_the_canonical_artifact_and_refuses_val():
    """``scripts/utils/data.create_cifar_loader`` must obey the protocol.

    It used to read ``processed/lt_train_indices.npy`` (a removed artifact) and
    to expose a ``'val'`` split, bypassing ``protocol_splits`` entirely — the one
    place the no-leakage rule is meant to be structural.
    """
    from data.protocol_splits import load_lt_train_indices
    from scripts.utils.data import create_cifar_loader

    loader, counts = create_cifar_loader(
        'train', data_root=DATA_ROOT, batch_size=64, num_workers=0, pin_memory=False)
    expected = load_lt_train_indices(DATA_ROOT)
    assert len(loader.dataset) == EXPECTED_TOTAL, len(loader.dataset)
    assert np.array_equal(loader.dataset.sample_indices, expected), \
        "the training loader is not serving the canonical artifact"
    assert int(counts.sum()) == EXPECTED_TOTAL, counts.sum()

    try:
        create_cifar_loader('val', data_root=DATA_ROOT, num_workers=0, pin_memory=False)
    except ps.ProtocolError as exc:
        print(f"  ✅ legacy loader uses the canonical artifact; 'val' rejected: {exc}")
        return
    raise AssertionError("create_cifar_loader('val') must raise: no val split exists")


def test_artifact_profile_matches_standard_protocol():
    """Counts, total and IR must match n_i = 500 * 100**(-i/99) exactly."""
    idx = ps.load_lt_train_indices(DATA_ROOT)
    targets = _targets()
    counts = np.bincount(targets[idx], minlength=N_CLASSES)

    assert len(idx) == EXPECTED_TOTAL, f"total {len(idx)} != {EXPECTED_TOTAL}"
    assert np.array_equal(counts, _expected_profile()), (
        f"per-class profile does not match the formula\n"
        f"  got      head={counts[0]} tail={counts[-1]} "
        f"IR={counts.max() / max(counts.min(), 1):.1f}\n"
        f"  expected head={EXPECTED_HEAD} tail={EXPECTED_TAIL} IR=100.0"
    )
    assert counts.min() >= 1, f"{(counts == 0).sum()} classes have zero samples"
    assert len(np.unique(targets[idx])) == N_CLASSES, "not all 100 classes present"
    print(f"  ✅ profile: n={len(idx)}, head={counts[0]}, tail={counts[-1]}, "
          f"IR={counts.max() // counts.min()}")


def test_artifact_hygiene():
    """Indices must be int64, sorted, unique, and inside the CIFAR-100 train range."""
    idx = ps.load_lt_train_indices(DATA_ROOT)
    assert idx.dtype == np.int64, f"dtype {idx.dtype} != int64"
    assert np.all(np.diff(idx) > 0), "indices not strictly increasing (unsorted or dupes)"
    assert len(np.unique(idx)) == len(idx), "duplicate indices"
    assert idx.min() >= 0 and idx.max() < 50000, "index outside CIFAR-100 train range"
    print(f"  ✅ hygiene: dtype={idx.dtype}, sorted+unique, range=[{idx.min()}, {idx.max()}]")


def test_artifact_is_deterministic_and_regenerable():
    """Regenerating from the committed script must reproduce the artifact exactly."""
    from utils.create_lt_split import build_lt_indices

    targets = _targets()
    a = build_lt_indices(targets, imbalance_ratio=100.0, seed=42)
    b = build_lt_indices(targets, imbalance_ratio=100.0, seed=42)
    assert np.array_equal(a, b), "generator is not deterministic under a fixed seed"
    assert np.array_equal(a, ps.load_lt_train_indices(DATA_ROOT)), (
        "artifact on disk differs from a fresh generation — it was not produced by "
        "utils/create_lt_split.py with seed 42"
    )
    print(f"  ✅ regenerates identically ({len(a)} indices, seed 42)")


def test_imbalance_factor_mapping():
    """The recorded imbalance factor 0.01 must correspond to IR=100."""
    from utils.create_lt_split import imbalance_factor_to_ir, IR_DEFAULT

    assert IR_DEFAULT == 100.0, f"IR_DEFAULT {IR_DEFAULT} != 100.0"
    assert abs(imbalance_factor_to_ir(0.01) - 100.0) < 1e-9
    print("  ✅ imbalance factor 0.01 -> IR 100")


def test_protocol_splits_have_no_val_key():
    """load_protocol_splits must not expose a validation split."""
    s = ps.load_protocol_splits(DATA_ROOT, seed=42)
    assert 'val' not in s, "load_protocol_splits still returns a 'val' split"
    print(f"  ✅ no 'val' key; keys = {sorted(s)}")


def test_all_train_is_the_canonical_artifact():
    """all_train must be exactly the canonical artifact, not a subset of it."""
    s = ps.load_protocol_splits(DATA_ROOT, seed=42)
    assert np.array_equal(s['all_train'], ps.load_lt_train_indices(DATA_ROOT)), \
        "all_train != canonical artifact"
    print("  ✅ all_train == canonical artifact")


def test_train_core_and_routing_dev_disjoint():
    """The honest-label carve must not overlap train_core."""
    s = ps.load_protocol_splits(DATA_ROOT, seed=42)
    overlap = np.intersect1d(s['train_core'], s['routing_dev'])
    assert overlap.size == 0, f"LEAKAGE: {overlap.size} shared indices"
    print("  ✅ train_core ∩ routing_dev = ∅")


def test_union_reconstructs_all_train():
    """train_core + routing_dev must cover the training set exactly once."""
    s = ps.load_protocol_splits(DATA_ROOT, seed=42)
    union = np.sort(np.concatenate([s['train_core'], s['routing_dev']]))
    assert np.array_equal(union, np.sort(s['all_train'])), \
        "union of splits does not reconstruct all_train"
    print("  ✅ train_core ∪ routing_dev == all_train (exact)")


def test_split_is_reproducible():
    """Same seed -> identical carve."""
    a = ps.load_protocol_splits(DATA_ROOT, seed=42)
    b = ps.load_protocol_splits(DATA_ROOT, seed=42)
    assert np.array_equal(a['train_core'], b['train_core'])
    assert np.array_equal(a['routing_dev'], b['routing_dev'])
    print("  ✅ same seed reproduces the carve exactly")


def test_different_seed_changes_split():
    """A different seed must actually repartition (else the seed is ignored)."""
    a = ps.load_protocol_splits(DATA_ROOT, seed=42)
    b = ps.load_protocol_splits(DATA_ROOT, seed=123)
    assert not np.array_equal(a['routing_dev'], b['routing_dev']), \
        "routing_dev identical across seeds — seed is not being used"
    assert len(a['routing_dev']) == len(b['routing_dev']), \
        "holdout size should be seed-independent"
    print("  ✅ different seed repartitions with stable size")


def test_tail_classes_present_in_routing_dev():
    """Tail classes must not be starved out of the honest-label carve."""
    s = ps.load_protocol_splits(DATA_ROOT, seed=42)
    targets = _targets()
    dev_targets = targets[s['routing_dev']]
    uniq = np.unique(dev_targets)
    assert len(uniq) == N_CLASSES, (
        f"routing_dev covers {len(uniq)}/100 classes — some classes are absent, "
        f"which would bias the routing labels"
    )
    min_c = min((dev_targets == c).sum() for c in uniq)
    print(f"  ✅ routing_dev covers all 100 classes (min {min_c} samples/class)")


def test_stratified_split_preserves_class_profile():
    """Holdout fraction should be applied per class, not globally."""
    idx = np.arange(1000)
    targets = np.array([0] * 900 + [1] * 100)
    core, dev = ps.stratified_split(idx, targets, 0.2, seed=0)
    n_dev_0 = (targets[dev] == 0).sum()
    n_dev_1 = (targets[dev] == 1).sum()
    assert n_dev_0 == 180, f"class 0 holdout {n_dev_0}, expected 180"
    assert n_dev_1 == 20, f"class 1 holdout {n_dev_1}, expected 20"
    print(f"  ✅ per-class stratification: {n_dev_0} + {n_dev_1} held out")


def test_verify_detects_injected_leakage():
    """verify_protocol_splits must actually catch overlapping splits."""
    bad = {
        'train_core': np.array([1, 2, 3, 4]),
        'routing_dev': np.array([4, 5, 6]),   # 4 overlaps
        'all_train': np.array([1, 2, 3, 4, 5, 6]),
    }
    try:
        ps.verify_protocol_splits(bad)
    except ps.ProtocolError as e:
        assert 'LEAKAGE' in str(e), str(e)
        print(f"  ✅ injected overlap detected: {e}")
        return
    raise AssertionError("overlapping splits were NOT detected")


def test_verify_rejects_any_val_split():
    """A reintroduced validation split must be rejected structurally."""
    bad = {
        'train_core': np.array([1, 2, 3]),
        'routing_dev': np.array([4, 5]),
        'all_train': np.array([1, 2, 3, 4, 5]),
        'val': np.array([9, 10]),
    }
    try:
        ps.verify_protocol_splits(bad)
    except ps.ProtocolError as e:
        assert 'validation' in str(e).lower(), str(e)
        print(f"  ✅ validation split rejected: {e}")
        return
    raise AssertionError("a 'val' split was accepted")


def test_missing_artifact_raises():
    """A missing artifact must raise, never silently substitute."""
    try:
        ps.load_lt_train_indices('/nonexistent_root')
    except ps.ProtocolError as e:
        assert 'Missing' in str(e), str(e)
        print(f"  ✅ missing artifact raises: {e}")
        return
    raise AssertionError("missing artifact did not raise ProtocolError")


def test_canonical_artifact_is_not_gitignored():
    """The artifact must survive a fresh clone — Kaggle has no local data dir.

    `data/processed/` was ignored wholesale, so a Kaggle clone received zero
    index files and training died with ProtocolError before the first epoch.
    """
    import subprocess

    path = os.path.join(PROCESSED, ps.LT_TRAIN_FILENAME)
    result = subprocess.run(
        ['git', 'check-ignore', '-q', path],
        cwd=_proj_root, capture_output=True,
    )
    assert result.returncode != 0, (
        f"{ps.LT_TRAIN_FILENAME} is gitignored, so a fresh clone (Kaggle) will "
        f"not have it and training cannot start"
    )
    print(f"  ✅ {ps.LT_TRAIN_FILENAME} is not gitignored")


def test_canonical_artifact_is_tracked_by_git():
    """The artifact must actually be committed, not merely un-ignored."""
    import subprocess

    rel = f'data/processed/{ps.LT_TRAIN_FILENAME}'
    result = subprocess.run(
        ['git', 'ls-files', '--error-unmatch', rel],
        cwd=_proj_root, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"{rel} is not tracked by git — Kaggle would clone a repo without the "
        f"canonical split. Run: git add {rel}"
    )
    print(f"  ✅ {rel} is tracked by git")


def test_missing_artifact_error_names_the_remedy():
    """The failure must be actionable, not just a bare path."""
    try:
        ps.load_lt_train_indices('/nonexistent_root')
    except ps.ProtocolError as e:
        msg = str(e)
        assert 'create_lt_split' in msg or 'git' in msg.lower(), (
            f"error does not say how to fix it: {msg}"
        )
        print("  ✅ missing-artifact error names the remedy")
        return
    raise AssertionError("missing artifact did not raise ProtocolError")


def test_generator_reproduces_the_committed_artifact():
    """Regeneration from raw data must reproduce the tracked artifact.

    This is the fallback on a machine (like Kaggle) whose clone lacked the
    artifact: `utils/create_lt_split.py` needs only `data/cifar-100-python`.
    """
    from utils.create_lt_split import build_lt_indices

    idx = build_lt_indices(_targets(), imbalance_ratio=100.0, seed=42)
    assert len(idx) == EXPECTED_TOTAL, len(idx)
    assert np.array_equal(idx, ps.load_lt_train_indices(DATA_ROOT)), \
        "regeneration does not reproduce the committed artifact"
    print(f"  ✅ regenerates from raw data ({len(idx)} indices)")


def test_index_range_guard():
    """An out-of-range index array (e.g. full CIFAR train) must be rejected."""
    ps.assert_no_test_leakage(np.array([0, 100, 49999]), 'train_core')
    try:
        ps.assert_no_test_leakage(np.array([0, 50000]), 'train_core')
    except ps.ProtocolError as e:
        print(f"  ✅ out-of-range indices rejected: {e}")
        return
    raise AssertionError("out-of-range indices were accepted")


TESTS = [
    ("Canonical artifact exists", test_canonical_artifact_exists),
    ("No validation artifact", test_no_validation_artifact),
    ("Legacy loader obeys the protocol", test_legacy_loader_reads_the_canonical_artifact_and_refuses_val),
    ("Artifact profile matches standard protocol", test_artifact_profile_matches_standard_protocol),
    ("Class group sizes match the artifact", test_class_group_sizes_match_the_artifact),
    ("Artifact hygiene", test_artifact_hygiene),
    ("Artifact deterministic and regenerable", test_artifact_is_deterministic_and_regenerable),
    ("Imbalance factor -> IR mapping", test_imbalance_factor_mapping),
    ("No 'val' key in protocol splits", test_protocol_splits_have_no_val_key),
    ("all_train == canonical artifact", test_all_train_is_the_canonical_artifact),
    ("train_core / routing_dev disjoint", test_train_core_and_routing_dev_disjoint),
    ("Union reconstructs all_train", test_union_reconstructs_all_train),
    ("Split is reproducible", test_split_is_reproducible),
    ("Different seed changes split", test_different_seed_changes_split),
    ("Tail classes present in routing_dev", test_tail_classes_present_in_routing_dev),
    ("Per-class stratification", test_stratified_split_preserves_class_profile),
    ("Injected overlap detected", test_verify_detects_injected_leakage),
    ("Any val split rejected", test_verify_rejects_any_val_split),
    ("Missing artifact raises", test_missing_artifact_raises),
    ("Artifact not gitignored", test_canonical_artifact_is_not_gitignored),
    ("Artifact tracked by git", test_canonical_artifact_is_tracked_by_git),
    ("Missing-artifact error actionable", test_missing_artifact_error_names_the_remedy),
    ("Generator reproduces artifact", test_generator_reproduces_the_committed_artifact),
    ("Index range guard", test_index_range_guard),
]


def main() -> int:
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 - surface the real cause
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 52}")
    print(f"  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
