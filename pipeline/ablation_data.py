"""Fixed fitting/development split for the retained training population."""

import json
from pathlib import Path

import numpy as np

from pipeline.data import NUM_CLASSES, TrainingData, load_training_data, save_training_data
from pipeline.runtime import atomic_write_json, file_sha256


def split_positions(labels, seed=43):
    labels = np.asarray(labels)
    if (
        labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= NUM_CLASSES))
    ):
        raise ValueError("Split labels must be an integer vector in [0, 99]")
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if (counts < 5).any():
        raise ValueError("The ablation split requires at least five retained examples per class")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("split_seed must be an integer in [0, 2**32)")
    generator = np.random.default_rng(seed)
    development = np.sort(np.concatenate([
        generator.choice(np.flatnonzero(labels == c), max(1, int(count // 5)), replace=False)
        for c, count in enumerate(counts)
    ]))
    fitting = np.setdiff1d(np.arange(len(labels)), development)
    return fitting, development


def _subset(data, positions):
    labels = data.labels[positions]
    return TrainingData(
        data.images[positions], labels, np.bincount(labels, minlength=NUM_CLASSES),
        data.data_settings.copy(),
    )


def prepare_split(source_path, output_dir, seed=43):
    """Create once; reuse only after checking source, split positions, and files."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "split.json"
    data = load_training_data(source_path)
    fitting, development = split_positions(data.labels, seed)
    manifest = {
        "source_sha256": file_sha256(source_path), "split_seed": seed,
        "original_class_counts": data.class_counts.tolist(),
        "fitting_positions": fitting.tolist(), "development_positions": development.tolist(),
        "fitting_class_counts": np.bincount(data.labels[fitting], minlength=NUM_CLASSES).tolist(),
        "development_class_counts": np.bincount(data.labels[development], minlength=NUM_CLASSES).tolist(),
    }
    paths = {name: output_dir / f"{name}.npz" for name in ("fitting", "development")}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(saved.get(key) != value for key, value in manifest.items()):
            raise ValueError("Saved split does not match the retained pool or split settings")
        for name, path in paths.items():
            if file_sha256(path) != saved["file_hashes"][name]:
                raise ValueError(f"Saved {name} data has changed")
        return saved
    if output_dir.exists():
        raise FileExistsError(f"Incomplete split folder exists; choose a new output: {output_dir}")
    output_dir.mkdir(parents=True)
    save_training_data(paths["fitting"], _subset(data, fitting))
    save_training_data(paths["development"], _subset(data, development))
    manifest["file_hashes"] = {name: file_sha256(path) for name, path in paths.items()}
    atomic_write_json(manifest_path, manifest)
    return manifest
