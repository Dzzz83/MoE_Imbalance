"""Restore a shared checkpoint and evaluate on the original CIFAR-100 test set."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from pipeline.data import NUM_CLASSES, evaluation_transform
from pipeline.runtime import load_checkpoint, resolve_device, write_json


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for images, batch_labels in loader:
        logits = model(images.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced nonfinite evaluation logits")
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
    if not predictions:
        raise ValueError("Evaluation dataset is empty")
    return np.concatenate(predictions), np.concatenate(labels)


def compute_metrics(predictions, labels, class_counts) -> dict:
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    class_counts = np.asarray(class_counts)
    if predictions.ndim != 1 or labels.shape != predictions.shape or not len(labels):
        raise ValueError("predictions and labels must be matching nonempty vectors")
    for name, values in (("predictions", predictions), ("labels", labels)):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= NUM_CLASSES)):
            raise ValueError(f"{name} must contain integer class labels in [0, 99]")
    if (
        class_counts.shape != (NUM_CLASSES,)
        or not np.issubdtype(class_counts.dtype, np.integer)
        or np.any(class_counts < 0)
    ):
        raise ValueError("class_counts must contain 100 nonnegative integers")

    support = np.bincount(labels, minlength=NUM_CLASSES)
    correct = np.bincount(labels[predictions == labels], minlength=NUM_CLASSES)
    recall = np.divide(correct, support, out=np.zeros(NUM_CLASSES), where=support > 0)

    def mean_recall(group):
        if not np.any(group) or np.any(support[group] == 0):
            return None
        return float(recall[group].mean())

    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": mean_recall(np.ones(NUM_CLASSES, dtype=bool)),
        "per_class_recall": [float(value) if count else None for value, count in zip(recall, support)],
        "test_support": support.tolist(),
        "head_recall": mean_recall(class_counts >= 100),
        "medium_recall": mean_recall((class_counts >= 20) & (class_counts < 100)),
        "tail_recall": mean_recall(class_counts < 20),
    }


def evaluate_checkpoint(
    checkpoint_path, data_root, output_path, device="cpu", batch_size=256,
    num_workers=4, download=False,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if Path(checkpoint_path).resolve() == Path(output_path).resolve():
        raise ValueError("Metric output must not replace the checkpoint")
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    test_data = datasets.CIFAR100(
        root=str(data_root), train=False, transform=evaluation_transform(), download=download
    )
    loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    predictions, labels = predict(model, loader, device)
    metrics = compute_metrics(predictions, labels, checkpoint["class_counts"].cpu().numpy())
    write_json(output_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        metrics = evaluate_checkpoint(
            args.checkpoint, args.data_root, args.output, args.device,
            args.batch_size, args.num_workers, args.download,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    for name in ("accuracy", "balanced_accuracy", "head_recall", "medium_recall", "tail_recall"):
        score = metrics[name]
        print(f"{name}: {'n/a' if score is None else f'{score * 100:.2f}%'}")
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
