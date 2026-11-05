"""Ordinary cross-entropy training: the example for future methods."""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from pipeline.data import make_dataset, training_transform
from pipeline.runtime import seed_worker

DEFAULT_OPTIONS = {
    "epochs": 200,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "scheduler": "cosine",
}


def resolve_options(options):
    unknown = options.keys() - DEFAULT_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown baseline options: {', '.join(sorted(unknown))}")
    resolved = DEFAULT_OPTIONS | options
    if type(resolved["epochs"]) is not int or resolved["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    for name in ("learning_rate", "momentum", "weight_decay"):
        value = resolved[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if resolved["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if resolved["momentum"] >= 1:
        raise ValueError("momentum must be less than 1")
    if resolved["scheduler"] != "cosine":
        raise ValueError("The baseline supports scheduler='cosine'")
    return resolved


def train(model, train_data, config, output_dir, device) -> list[dict]:
    options = config["method_options"]
    loader = DataLoader(
        make_dataset(train_data, training_transform()),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=options["learning_rate"],
        momentum=options["momentum"],
        weight_decay=options["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options["epochs"])
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, options["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            sample_count += len(labels)
        mean_loss = loss_sum / sample_count
        history.append({"epoch": epoch, "loss": mean_loss, "learning_rate": learning_rate})
        scheduler.step()
        print(f"Epoch {epoch}/{options['epochs']} loss={mean_loss:.4f} lr={learning_rate:.6g}", flush=True)
    return history
