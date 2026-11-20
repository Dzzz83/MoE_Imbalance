"""Exact split-43 image identities independent of NPZ container serialization."""

from pathlib import Path

import torch

from pipeline.ablation_data import prepare_split
from pipeline.data import load_training_data
from pipeline.runtime import content_identity

# Derived from the checksum-verified official CIFAR-100 training file using
# the audit's unchanged preparation code, seed 42 and split 43. Array hashes
# avoid depending on a particular Python/NumPy ZIP serialization version.
IMAGE_ARRAY_HASHES = {
    "retained": {
        "images": "19191649b3fec4d9b698d5de534327fcfcced4a54f63efc131406c2dbf7a56c1",
        "labels": "e9c42eeb5394889c6aea7a43e653db965774ebd1ae7d99c0a15d6c9056c1e3ef",
    },
    "fitting": {
        "images": "d4e6ce920d9d3b45e5b3d1138bb953eefe66253236e26ad1ae2a7c322da03b77",
        "labels": "fc74cd909353483b7303e06c9a50bf7e57468815c6d3bb68351390c277c056e2",
    },
    "development": {
        "images": "d8f1c6c7705b0d270d4d8fd9a68a0be3bb2310a759535e1267e9534ef7c63404",
        "labels": "a1d1bfeea987722460368f6388f2af267c9d59473bede87ec42f1e60521ea232",
    },
}


def array_hashes(data):
    return {name: content_identity(torch.from_numpy(getattr(data, name)))["sha256"]
            for name in ("images", "labels")}


def prepare_images(inputs, retained_path, split_root):
    """Check exact pixels/labels and audit positions, never only counts or features."""
    retained = load_training_data(retained_path)
    if (array_hashes(retained) != IMAGE_ARRAY_HASHES["retained"]
            or retained.data_settings != {"imbalance_factor": .01, "data_seed": 42}):
        raise ValueError("Retained image arrays disagree with the pinned official-data reconstruction")
    split = prepare_split(retained_path, split_root, 43)
    container_fields = {"source_sha256", "file_hashes"}
    if ({k: v for k, v in split.items() if k not in container_fields}
            != {k: v for k, v in inputs["split"].items() if k not in container_fields}):
        raise ValueError("Reconstructed image positions/counts disagree with the pinned audit split")
    data = {name: load_training_data(Path(split_root) / f"{name}.npz") for name in ("fitting", "development")}
    for name, current in data.items():
        if array_hashes(current) != IMAGE_ARRAY_HASHES[name]:
            raise ValueError(f"Pinned {name} image arrays changed")
    for bundle in inputs["bundles"].values():
        for name, key in (("fitting", "labels"), ("development", "development_labels")):
            if not torch.equal(torch.from_numpy(data[name].labels), bundle[key]):
                raise ValueError("Image and fitting-cache row order disagree")
    data["provenance"] = {"canonical_arrays": IMAGE_ARRAY_HASHES,
                          "audit_container_hashes": {k: inputs["split"][k] for k in container_fields},
                          "validation": "Exact array hashes, audit split metadata and cache label order passed."}
    return data
