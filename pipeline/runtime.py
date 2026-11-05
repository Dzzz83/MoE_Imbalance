"""Small shared helpers for randomness, JSON, devices, and model checkpoints."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from pipeline.model import create_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """PyTorch supplies each worker's seed; also seed NumPy and Python there."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(name):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available; use --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}")
    return device


def write_json(path, value):
    # Serialize first, so invalid method output cannot truncate an existing file.
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def save_checkpoint(path, model, config, train_data):
    torch.save({
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "config": config,
        "class_counts": torch.from_numpy(train_data.class_counts.copy()),
        "data_settings": train_data.data_settings,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
