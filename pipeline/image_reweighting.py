"""Matched image continuations with fixed weights and frozen BN statistics."""

import time

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from pipeline.data import evaluation_transform, make_dataset, training_transform
from pipeline.followup_weights import validate_weights


class IndexedAugmentations(Dataset):
    """Local per-row RNG seeds make augmentation independent of condition weights."""

    def __init__(self, data, seed):
        self.dataset = make_dataset(data, training_transform())
        self.generator = torch.Generator().manual_seed(seed + 20000)
        self.seeds = None

    def next_epoch(self):
        self.seeds = torch.randint(2**63 - 1, (len(self),), generator=self.generator)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # torchvision's crop/flip use torch's CPU RNG. Restore it after each row.
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(torch.Generator().manual_seed(int(self.seeds[index])).get_state())
            image, label = self.dataset[index]
        return image, label, index


def freeze_bn_statistics(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_images(model, data, weights, seed, options, device, on_epoch=None):
    labels = torch.from_numpy(data.labels)
    validate_weights(weights, labels)
    dataset = IndexedAugmentations(data, seed)
    loader = DataLoader(dataset, batch_size=options["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + 10000), drop_last=False)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    bn_initial = {name: b.detach().cpu().clone() for name, b in model.named_buffers()
                  if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=options["learning_rate"], momentum=.9,
                                nesterov=True, weight_decay=.0005)
    weights = weights.to(device)
    history, steps = [], 0
    maximum_backbone_gradient = torch.zeros((), device=device)
    started = time.monotonic()
    for epoch in range(options["epochs"]):
        dataset.next_epoch()
        model.train()
        freeze_bn_statistics(model)
        weighted_loss = torch.zeros((), dtype=torch.float64, device=device)
        ordinary_loss = torch.zeros((), dtype=torch.float64, device=device)
        for images, targets, positions in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = functional.cross_entropy(model(images), targets, reduction="none")
            loss = (len(dataset) * weights[positions.to(device)] * losses).mean()
            if not torch.isfinite(loss) or not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite image training loss")
            loss.backward()
            gradient_maxima = {name: parameter.grad.detach().abs().max()
                              for name, parameter in model.named_parameters() if parameter.grad is not None}
            if not torch.isfinite(torch.stack(list(gradient_maxima.values()))).all():
                raise RuntimeError("Nonfinite image training gradient")
            backbone_gradient = torch.stack([value for name, value in gradient_maxima.items()
                                              if not name.startswith("fc.")]).max()
            maximum_backbone_gradient = torch.maximum(maximum_backbone_gradient, backbone_gradient)
            optimizer.step()
            ordinary_loss += losses.detach().double().sum()
            weighted_loss += loss.detach() * len(targets)
            steps += 1
        if not torch.isfinite(torch.stack([p.detach().abs().max() for p in model.parameters()])).all():
            raise RuntimeError("Nonfinite image training parameters")
        record = {"epoch": epoch + 1, "loss": float(weighted_loss) / len(dataset),
                  "unweighted_loss": float(ordinary_loss) / len(dataset), "optimizer_steps": steps,
                  "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        if on_epoch is not None:
            on_epoch(history)
    buffers = dict(model.named_buffers())
    if any(not torch.equal(buffers[name].cpu(), before) for name, before in bn_initial.items()):
        raise RuntimeError("Frozen batch-normalization buffers changed")
    movement = {name: float((p.detach().cpu() - initial[name]).double().square().sum())
                for name, p in model.named_parameters()}
    backbone_movement = sum(v for k, v in movement.items() if not k.startswith("fc.")) ** .5
    expected_steps = options["epochs"] * ((len(dataset) + options["batch_size"] - 1) // options["batch_size"])
    maximum_backbone_gradient = float(maximum_backbone_gradient)
    if steps != expected_steps or backbone_movement <= 0 or maximum_backbone_gradient <= 0:
        raise RuntimeError("Image continuation did not perform the planned backbone updates")
    return history, {"optimizer_steps": steps, "expected_steps": expected_steps,
                     "backbone_parameter_movement_l2": backbone_movement,
                     "head_parameter_movement_l2": sum(v for k, v in movement.items() if k.startswith("fc.")) ** .5,
                     "maximum_backbone_gradient": maximum_backbone_gradient,
                     "bn_buffers_unchanged": True, "elapsed_seconds": time.monotonic() - started}


@torch.no_grad()
def image_logits(model, data, device, batch_size=128):
    model.eval()
    loader = DataLoader(make_dataset(data, evaluation_transform()), batch_size=batch_size, shuffle=False,
                        num_workers=0, generator=torch.Generator().manual_seed(0))
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)
