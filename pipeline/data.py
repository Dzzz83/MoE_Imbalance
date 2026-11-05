"""Prepare the retained training pool and expose ordinary image datasets."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

NUM_CLASSES = 100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass
class TrainingData:
    images: np.ndarray
    labels: np.ndarray
    class_counts: np.ndarray
    data_settings: dict


def _validate_preparation_settings(imbalance_factor, seed):
    if (
        type(imbalance_factor) not in (int, float)
        or not np.isfinite(imbalance_factor)
        or not 0 < imbalance_factor <= 1
    ):
        raise ValueError("imbalance_factor must be in (0, 1]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("data seed must be an integer in [0, 2**32)")


def select_long_tail(labels, imbalance_factor=0.01, seed=42):
    """Select from the original balanced training labels, in source order."""
    _validate_preparation_settings(imbalance_factor, seed)
    labels = np.asarray(labels)
    if (
        labels.shape != (50_000,)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= NUM_CLASSES))
        or not np.all(np.bincount(labels, minlength=NUM_CLASSES) == 500)
    ):
        raise ValueError("Expected the original 50,000 labels, 500 per class")

    counts = 500 * imbalance_factor ** (np.arange(NUM_CLASSES) / (NUM_CLASSES - 1))
    counts = np.maximum(counts.astype(np.int64), 1)
    generator = np.random.default_rng(seed)
    selected = [
        generator.choice(np.flatnonzero(labels == label), count, replace=False)
        for label, count in enumerate(counts)
    ]
    return np.sort(np.concatenate(selected))


def _validate_training_data(data):
    if data.labels.dtype != np.int64 or data.labels.ndim != 1 or not len(data.labels):
        raise ValueError("labels must be a nonempty int64 array with shape (N,)")
    if data.images.dtype != np.uint8 or data.images.shape != (len(data.labels), 32, 32, 3):
        raise ValueError("images must be uint8 with shape (N, 32, 32, 3)")
    if np.any((data.labels < 0) | (data.labels >= NUM_CLASSES)):
        raise ValueError("labels must be in [0, 99]")
    if data.class_counts.dtype != np.int64 or data.class_counts.shape != (NUM_CLASSES,):
        raise ValueError("class_counts must be int64 with shape (100,)")
    if not np.array_equal(data.class_counts, np.bincount(data.labels, minlength=NUM_CLASSES)):
        raise ValueError("class_counts do not match labels")
    _validate_preparation_settings(
        data.data_settings["imbalance_factor"], data.data_settings["data_seed"]
    )


def save_training_data(path, data):
    """Save only retained images, without replacing an existing population."""
    _validate_training_data(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        np.savez_compressed(
            destination,
            images=data.images,
            labels=data.labels,
            class_counts=data.class_counts,
            **data.data_settings,
        )


def load_training_data(path) -> TrainingData:
    with np.load(path, allow_pickle=False) as prepared:
        required = {"images", "labels", "class_counts", "imbalance_factor", "data_seed"}
        missing = required - set(prepared.files)
        if missing:
            raise ValueError(f"Prepared data is missing fields: {', '.join(sorted(missing))}")
        data = TrainingData(
            images=prepared["images"],
            labels=prepared["labels"],
            class_counts=prepared["class_counts"],
            data_settings={
                "imbalance_factor": prepared["imbalance_factor"].item(),
                "data_seed": prepared["data_seed"].item(),
            },
        )
    _validate_training_data(data)
    return data


def prepare_data(data_root, output, imbalance_factor=0.01, seed=42, download=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Prepared data already exists: {output}")
    _validate_preparation_settings(imbalance_factor, seed)
    source = datasets.CIFAR100(root=str(data_root), train=True, download=download)
    labels = np.asarray(source.targets, dtype=np.int64)
    selected = select_long_tail(labels, imbalance_factor, seed)
    retained_labels = labels[selected]
    data = TrainingData(
        images=source.data[selected],
        labels=retained_labels,
        class_counts=np.bincount(retained_labels, minlength=NUM_CLASSES),
        data_settings={"imbalance_factor": imbalance_factor, "data_seed": seed},
    )
    save_training_data(output, data)
    return data


def training_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def evaluation_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


class ImageDataset(Dataset):
    def __init__(self, data, transform):
        self.images = data.images
        self.labels = data.labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = to_pil_image(self.images[index])
        return self.transform(image), int(self.labels[index])


def make_dataset(data, transform) -> Dataset:
    """Create a view; transforms receive a PIL image, as in torchvision CIFAR."""
    return ImageDataset(data, transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/train.npz"))
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        data = prepare_data(
            args.data_root, args.output, args.imbalance_factor, args.seed, args.download
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved {len(data.labels):,} retained training images to {args.output}")


if __name__ == "__main__":
    main()
