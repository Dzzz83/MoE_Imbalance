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
    "weight_decay": 0.0002,
    "scheduler": "multistep",
    "warmup_epochs": 5,
    "decay_epochs": [160, 180],
    "decay_factor": 0.01,
}


def prepare_method_options(options):
    unknown = options.keys() - DEFAULT_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown baseline options: {', '.join(sorted(unknown))}")
    method_options = DEFAULT_OPTIONS | options
    if type(method_options["epochs"]) is not int or method_options["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    for name in ("learning_rate", "momentum", "weight_decay"):
        value = method_options[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if method_options["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if method_options["momentum"] >= 1:
        raise ValueError("momentum must be less than 1")
    if method_options["scheduler"] not in ("multistep", "cosine"):
        raise ValueError("The baseline supports scheduler='multistep' or 'cosine'")
    if method_options["scheduler"] == "cosine":
        # Historical synthesis pretraining had no warm-up or milestone decays.
        schedule_keys = {"warmup_epochs", "decay_epochs", "decay_factor"}
        if options.keys() & schedule_keys:
            raise ValueError("Cosine scheduling does not use warmup_epochs, decay_epochs, or decay_factor")
        return {name: value for name, value in method_options.items() if name not in schedule_keys}
    warmup_epochs = method_options["warmup_epochs"]
    if type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    decay_epochs = method_options["decay_epochs"]
    if (
        not isinstance(decay_epochs, list)
        or any(type(epoch) is not int or epoch < 1 for epoch in decay_epochs)
        or decay_epochs != sorted(set(decay_epochs))
    ):
        raise ValueError("decay_epochs must be a strictly increasing list of positive integers")
    if any(epoch < warmup_epochs for epoch in decay_epochs):
        raise ValueError("decay_epochs must not precede the end of warm-up")
    decay_factor = method_options["decay_factor"]
    if (
        type(decay_factor) not in (int, float)
        or not math.isfinite(decay_factor)
        or not 0 < decay_factor < 1
    ):
        raise ValueError("decay_factor must be a finite number between 0 and 1, exclusive")
    method_options["decay_epochs"] = list(decay_epochs)
    return method_options


def _learning_rate_for_epoch(epoch, options):
    """Follow LDAM's schedule using one-based epochs and drops after milestones."""
    if epoch <= options["warmup_epochs"]:
        return options["learning_rate"] * epoch / options["warmup_epochs"]
    completed_milestones = sum(epoch > milestone for milestone in options["decay_epochs"])
    return options["learning_rate"] * options["decay_factor"] ** completed_milestones


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
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options["epochs"])
        if options["scheduler"] == "cosine" else None
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, options["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        learning_rate = (
            optimizer.param_groups[0]["lr"] if scheduler is not None
            else _learning_rate_for_epoch(epoch, options)
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
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
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch}/{options['epochs']} loss={mean_loss:.4f} lr={learning_rate:.6g}", flush=True)
    return history
