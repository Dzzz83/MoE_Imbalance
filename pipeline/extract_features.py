import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.data import evaluation_transform, load_training_data, make_dataset
from pipeline.runtime import load_checkpoint, resolve_device


def extract_features(model, train_data, device, batch_size=128, num_workers=0):
    """Freeze the backbone and return CPU features and labels in source order."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    device = resolve_device(device)
    model.to(device)
    model.backbone.requires_grad_(False)
    model.eval()
    loader = DataLoader(
        make_dataset(train_data, evaluation_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            batch = model.backbone(images.to(device))
            if batch.shape != (len(targets), 64) or not torch.isfinite(batch).all():
                raise ValueError("Expected finite 64-dimensional backbone features")
            features.append(batch.cpu())
            labels.append(targets)
    features, labels = torch.cat(features), torch.cat(labels)
    if not np.array_equal(labels.numpy(), train_data.labels):
        raise ValueError("Extracted labels do not match retained training order")
    return features, labels


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_extraction(checkpoint_path, train_data_path, output, device="cpu",
                   batch_size=128, num_workers=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Feature cache already exists: {output}")
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    train_data = load_training_data(train_data_path)
    if checkpoint["data_settings"] != train_data.data_settings:
        raise ValueError("Training data settings do not match the checkpoint")
    counts = torch.from_numpy(train_data.class_counts.copy())
    if not torch.equal(checkpoint["class_counts"].cpu(), counts):
        raise ValueError("Training class counts do not match the checkpoint")
    features, labels = extract_features(
        model, train_data, device, batch_size, num_workers
    )
    # Store original coordinates. Stage 3 uses X = features / feature_scale
    # and multiplies synthetic X by the same scale before classifier training.
    scale = features.double().square().sum(dim=1).mean().sqrt().clamp_min(1e-12)
    cache = {
        "features": features,
        "labels": labels,
        "class_counts": counts,
        "data_settings": train_data.data_settings,
        "feature_scale": scale.item(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "train_data_sha256": file_sha256(train_data_path),
        "preprocessing": "pipeline.data.evaluation_transform",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        torch.save(cache, destination)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=Path("data/prepared/train.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    try:
        cache = run_extraction(
            args.checkpoint, args.train_data, args.output, args.device,
            args.batch_size, args.num_workers,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved features {tuple(cache['features'].shape)} to {args.output}")


if __name__ == "__main__":
    main()
