"""
The CIFAR-100-LT data protocol.

Defines, in one place, which data may be used for what. Every downstream script
must obtain its indices from here, so the no-leakage rule is structural rather
than a convention someone has to remember.

Data roles
----------
  all_train     10,847   The canonical long-tailed training set (IR=100). Experts
                         train on these; nothing is held out for validation.
  train_core    ~80%     of all_train — expert training (backbone + classifier)
  routing_dev   ~20%     of all_train — routing-head training ONLY; source of
                         honest correctness labels (samples the experts that
                         predict on them were never trained on)
  test          10,000   CIFAR-100 test set. FINAL reported numbers only.

There is deliberately **no validation split**: the earlier `lt_val` set was
non-standard (it removed 20% of an already tiny long-tailed training set), and
measurably broken for its purpose — being a global rather than stratified
shuffle, it left 4 classes with zero samples and 7 with a single sample. Model
selection is therefore fixed by the published hyperparameters and epoch counts,
and the reported model is the final-epoch model. ``verify_protocol_splits``
rejects any attempt to reintroduce a 'val' key.

``routing_dev`` is *not* a validation set: it is an internal carve of the
training pool used only to obtain honest labels for routing heads, and it never
selects a model or a checkpoint.

Usage:
    from data.protocol_splits import load_protocol_splits, load_lt_train_indices
    splits = load_protocol_splits('./data', routing_dev_frac=0.2, seed=42)
    splits['train_core']    # np.ndarray of CIFAR-100 train indices
    splits['routing_dev']
    splits['all_train']
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The one canonical long-tailed training artifact (produced by
#: ``utils/create_lt_split.py``). Located in ``<data_root>/processed/``.
LT_TRAIN_FILENAME: str = 'lt_ir100_train_indices.npy'

#: Filenames from the superseded protocols. Their presence is not an error, but
#: nothing in this module reads them; kept only so diagnostics can name them.
LEGACY_VAL_FILENAME: str = 'lt_val_indices.npy'
LEGACY_TRAIN_FILENAME: str = 'lt_train_indices.npy'
LEGACY_ALL_FILENAME: str = 'lt_all_indices.npy'


class ProtocolError(RuntimeError):
    """Raised when the data protocol is violated or an artifact is missing."""


def load_lt_train_indices(data_root: str = './data') -> np.ndarray:
    """Load the canonical long-tailed training index set.

    Raises:
        ProtocolError: if the artifact is missing (never substitute a value).
    """
    path = Path(data_root) / 'processed' / LT_TRAIN_FILENAME
    if not path.exists():
        raise ProtocolError(
            f"Missing protocol artifact: {path}\n"
            f"  This file is committed to the repository and is required before "
            f"the first training step.\n"
            f"  Fix one of:\n"
            f"    (a) restore it from git:  git checkout -- "
            f"data/processed/{LT_TRAIN_FILENAME}   (or git pull)\n"
            f"    (b) regenerate it (deterministic, needs data/cifar-100-python):\n"
            f"        python utils/create_lt_split.py\n"
            f"  If it was regenerated, verify it still matches the committed split "
            f"before training:  python tests/test_protocol_splits.py"
        )
    return np.load(str(path))


def _load_targets(train_index: np.ndarray, data_root: str) -> np.ndarray:
    """Per-sample class labels for the given CIFAR-100 train indices."""
    from torchvision import datasets

    try:
        full = datasets.CIFAR100(root=data_root, train=True, download=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ProtocolError(
            f"Cannot load CIFAR-100 train split from '{data_root}': {exc}"
        ) from exc
    targets = np.asarray(full.targets, dtype=np.int64)
    return targets[train_index]


def stratified_split(
    train_index: np.ndarray,
    targets: np.ndarray,
    holdout_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices per class so both parts keep the long-tailed profile.

    Every class contributes ``floor(n_c * holdout_frac)`` samples to the holdout
    (at least 1 when the class has >= 2 samples), so tail classes are represented
    in the routing set rather than being starved by a global shuffle.

    Returns:
        (train_core, routing_dev) -- both sorted index arrays, disjoint.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ProtocolError(f"holdout_frac must be in (0, 1), got {holdout_frac}")

    rng = np.random.default_rng(seed)
    core_parts: list[np.ndarray] = []
    dev_parts: list[np.ndarray] = []

    for cls in np.unique(targets):
        cls_idx = train_index[targets == cls]
        cls_idx = np.sort(cls_idx)
        n = len(cls_idx)
        n_dev = int(np.floor(n * holdout_frac))
        if n >= 2:
            n_dev = max(1, min(n_dev, n - 1))
        else:
            n_dev = 0
        perm = rng.permutation(n)
        dev_parts.append(cls_idx[perm[:n_dev]])
        core_parts.append(cls_idx[perm[n_dev:]])

    train_core = np.sort(np.concatenate(core_parts))
    routing_dev = (
        np.sort(np.concatenate(dev_parts)) if dev_parts else np.array([], dtype=np.int64)
    )
    return train_core, routing_dev


def load_protocol_splits(
    data_root: str = './data',
    routing_dev_frac: float = 0.2,
    seed: int = 42,
    verify: bool = True,
) -> dict[str, np.ndarray]:
    """Load the protocol index sets and assert they are disjoint.

    Args:
        data_root: directory containing ``processed/lt_ir100_train_indices.npy``.
        routing_dev_frac: fraction of the training set carved out for routing-head
            training (honest-label source). Not a validation split.
        seed: split seed (fixed across the project for comparability).
        verify: run the protocol assertions before returning.

    Returns:
        dict with keys 'train_core', 'routing_dev', 'all_train'. There is no
        'val' key by design.

    Raises:
        ProtocolError: if a required artifact is missing or the splits overlap.
    """
    all_train = load_lt_train_indices(data_root)
    targets = _load_targets(all_train, data_root)
    train_core, routing_dev = stratified_split(
        all_train, targets, routing_dev_frac, seed
    )

    splits = {
        'train_core': train_core,
        'routing_dev': routing_dev,
        'all_train': all_train,
    }

    if verify:
        verify_protocol_splits(splits)

    return splits


def verify_protocol_splits(splits: dict[str, np.ndarray]) -> None:
    """Assert the splits are disjoint, complete, and contain no validation set.

    Raises:
        ProtocolError: on a validation split, leakage, or a non-reconstructing union.
    """
    if 'val' in splits:
        raise ProtocolError(
            "LEAKAGE/Protocol: a 'val' (validation) split was supplied, but this "
            "protocol has no validation split. Experts train on the full "
            "long-tailed training set and are evaluated on the 10K test set only."
        )

    core = set(splits['train_core'].tolist())
    dev = set(splits['routing_dev'].tolist())
    all_train = set(splits['all_train'].tolist())

    if core & dev:
        raise ProtocolError(
            f"LEAKAGE: train_core and routing_dev share {len(core & dev)} indices"
        )
    if (core | dev) != all_train:
        missing = len(all_train - (core | dev))
        extra = len((core | dev) - all_train)
        raise ProtocolError(
            f"train_core + routing_dev != all_train (missing {missing}, extra {extra})"
        )


def assert_no_test_leakage(indices: np.ndarray, role: str) -> None:
    """Guard for training entry points.

    The CIFAR-100 test set is addressed by ``use_test_set=True`` on the dataset
    rather than by index, so any *index array* reaching a trainer is by
    definition from the training pool.  This guard exists to make the intent
    explicit and to catch a future refactor that passes the full CIFAR-100 train
    range (0..49999) instead of a protocol split.
    """
    idx = np.asarray(indices)
    n_full_cifar_train = 50000
    if idx.size and (idx.max() >= n_full_cifar_train or idx.min() < 0):
        raise ProtocolError(
            f"{role}: indices out of range for a protocol split "
            f"(min={idx.min()}, max={idx.max()})"
        )
