"""Frozen-feature OT reweighting over the complete retained training pool."""

import math

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from pipeline.data import NUM_CLASSES, evaluation_transform, make_dataset
from pipeline.methods import baseline
from pipeline.runtime import seed_worker, write_json

DEFAULT_CLASSIFIER_OPTIONS = {
    "epochs": 40,
    "learning_rate": 0.00002,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "nesterov": True,
    "scheduler": "constant",
}

DEFAULT_OPTIONS = {
    "pretraining": {},
    "classifier": {},
    "score_learning_rate": 0.001,
    "score_updates": 1,
    "epsilon": 0.1,
    "sinkhorn_iterations": 200,
    "marginal_tolerance": 0.001,
    "feature_cost_scale": 0.5,
    "label_mismatch_cost": 0.01,
}


def prepare_method_options(options):
    unknown = options.keys() - DEFAULT_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown reweighting options: {', '.join(sorted(unknown))}")
    method_options = DEFAULT_OPTIONS | options
    for name in ("pretraining", "classifier"):
        if not isinstance(method_options[name], dict):
            raise ValueError(f"{name} must be a JSON object")
    method_options["pretraining"] = baseline.prepare_method_options(method_options["pretraining"])
    classifier_options = method_options["classifier"]
    unknown = classifier_options.keys() - DEFAULT_CLASSIFIER_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown classifier options: {', '.join(sorted(unknown))}")
    classifier_options = DEFAULT_CLASSIFIER_OPTIONS | classifier_options
    for name, value in (
        ("classifier epochs", classifier_options["epochs"]),
        ("score_updates", method_options["score_updates"]),
        ("sinkhorn_iterations", method_options["sinkhorn_iterations"]),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("score_learning_rate", "epsilon", "marginal_tolerance"):
        value = method_options[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    for name in ("feature_cost_scale", "label_mismatch_cost"):
        value = method_options[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if method_options["feature_cost_scale"] == 0 and method_options["label_mismatch_cost"] == 0:
        raise ValueError("At least one OT cost coefficient must be positive")
    for name in ("learning_rate", "momentum", "weight_decay"):
        value = classifier_options[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"classifier {name} must be a finite nonnegative number")
    if classifier_options["learning_rate"] == 0:
        raise ValueError("classifier learning_rate must be positive")
    if classifier_options["momentum"] >= 1:
        raise ValueError("classifier momentum must be less than 1")
    if type(classifier_options["nesterov"]) is not bool:
        raise ValueError("classifier nesterov must be a boolean")
    if classifier_options["nesterov"] and classifier_options["momentum"] == 0:
        raise ValueError("classifier Nesterov requires positive momentum")
    if classifier_options["scheduler"] != "constant":
        raise ValueError("The classifier supports scheduler='constant'")
    method_options["classifier"] = classifier_options
    return method_options


def _pretrain(model, train_data, config, output_dir, device) -> list[dict]:
    """Train the full model with the baseline recipe before learning weights."""
    pretraining_options = baseline.prepare_method_options(
        config["method_options"].get("pretraining", {})
    )
    pretraining_config = config | {"method_options": pretraining_options}
    history = baseline.train(model, train_data, pretraining_config, output_dir, device)
    return [record | {"stage": "pretraining"} for record in history]


def _freeze_backbone(model):
    """Fix backbone parameters and batch-norm state for Stage 2.

    Later classifier training must use model.fc.train(), since model.train()
    would put the backbone's batch-normalization layers back in training mode.
    """
    model.backbone.requires_grad_(False)
    for parameter in model.backbone.parameters():
        parameter.grad = None
    model.backbone.eval()


@torch.no_grad()
def _cache_training_features(model, train_data, config, device):
    """Freeze the backbone and return raw CPU features and labels in retained order.

    Keep feature magnitudes for classifier training; normalization for OT is a
    separate operation. Use no_grad rather than inference_mode so cached tensors
    can later participate in classifier gradient computation.
    """
    _freeze_backbone(model)
    loader = DataLoader(
        make_dataset(train_data, evaluation_transform()),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    feature_batches = []
    label_batches = []
    for images, labels in loader:
        features = model.backbone(images.to(device))
        if not torch.isfinite(features).all():
            raise RuntimeError("Nonfinite backbone features during training feature extraction")
        feature_batches.append(features.cpu())
        label_batches.append(labels)
    return torch.cat(feature_batches), torch.cat(label_batches)


@torch.no_grad()
def _build_ot_references(raw_features, labels):
    """Return normalized examples, normalized class means, and equal class masses.

    Reference row c represents class c. Average raw features before normalizing;
    leave the raw classifier cache unchanged. Zero vectors remain zero under
    normalization, so their dot product with any normalized vector is zero.
    """
    class_counts = torch.bincount(labels, minlength=NUM_CLASSES)
    missing_classes = torch.nonzero(class_counts == 0).flatten().tolist()
    if missing_classes:
        raise ValueError(f"Cannot build OT references for missing classes: {missing_classes}")
    class_means = torch.stack([
        raw_features[labels == class_index].mean(dim=0)
        for class_index in range(NUM_CLASSES)
    ])
    if not torch.isfinite(class_means).all():
        raise RuntimeError("Nonfinite class means during OT reference construction")
    normalized_features = functional.normalize(raw_features, dim=1)
    normalized_class_means = functional.normalize(class_means, dim=1)
    reference_masses = raw_features.new_full((NUM_CLASSES,), 1 / NUM_CLASSES)
    return normalized_features, normalized_class_means, reference_masses


@torch.no_grad()
def _build_ot_cost(
    normalized_features, normalized_class_means, labels, *,
    feature_cost_scale=0.5, label_mismatch_cost=0.01,
):
    """Build detached example-by-class costs using the authors' CIFAR-100 scales.

    Inputs come from _build_ot_references; column c represents class c. The
    released half mean-squared one-hot label distance is 0.01 for a mismatch.
    """
    for name, value in (
        ("feature_cost_scale", feature_cost_scale),
        ("label_mismatch_cost", label_mismatch_cost),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if feature_cost_scale == 0 and label_mismatch_cost == 0:
        raise ValueError("At least one OT cost coefficient must be positive")
    # Clamp numerical roundoff so identical unit vectors cannot have negative cost.
    cosine_similarity = (normalized_features @ normalized_class_means.T).clamp(-1, 1)
    reference_labels = torch.arange(NUM_CLASSES, device=labels.device)
    label_mismatches = (labels[:, None] != reference_labels[None, :]).to(cosine_similarity.dtype)
    cost = feature_cost_scale * (1 - cosine_similarity) + label_mismatch_cost * label_mismatches
    if not torch.isfinite(cost).all():
        raise RuntimeError("Nonfinite OT costs")
    return cost


def _initialize_example_scores(example_count, device, dtype=torch.float32):
    """Create one zero-initialized learnable score per retained-array position."""
    if type(example_count) is not int or example_count < 1:
        raise ValueError("example_count must be a positive integer")
    return torch.nn.Parameter(torch.zeros(example_count, device=device, dtype=dtype))


def _make_score_optimizer(example_scores, learning_rate=0.001):
    """Create Adam once for Stage 2 and reuse it across global score updates."""
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("score learning_rate must be a finite positive number")
    return torch.optim.Adam(
        [example_scores], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
    )


def _global_example_weights(example_scores):
    """Normalize the complete score vector before indexing any shuffled batches.

    Preserve gradients for OT learning. Classifier training will detach these
    weights. Reject softmax underflow rather than silently changing the masses
    by clamping, since the transport solve requires strictly positive masses.
    """
    if example_scores.ndim != 1 or not example_scores.numel():
        raise ValueError("example_scores must be a nonempty one-dimensional tensor")
    if not torch.isfinite(example_scores).all():
        raise RuntimeError("Nonfinite global example scores")
    weights = torch.softmax(example_scores, dim=0)
    if not torch.isfinite(weights).all():
        raise RuntimeError("Nonfinite global example weights")
    if (weights <= 0).any():
        raise RuntimeError("Global example weights underflowed to zero")
    return weights


def _sinkhorn_transport(cost, example_weights, reference_masses, *, epsilon=0.1, max_iterations=200):
    """Solve entropy-regularized OT, preserving gradients into example weights.

    Rows correspond to retained examples and columns to class references. Costs
    stay detached. Use a fixed iteration budget for reproducible differentiated
    solves; report maximum absolute marginal errors without claiming convergence.
    The outer weight objective is (transport * cost).sum(), without entropy.
    """
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if cost.ndim != 2 or not cost.shape[0] or not cost.shape[1]:
        raise ValueError("cost must be a nonempty two-dimensional tensor")
    if not cost.is_floating_point() or not torch.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("cost must contain finite nonnegative floating-point values")
    for name, masses, size in (
        ("example_weights", example_weights, cost.shape[0]),
        ("reference_masses", reference_masses, cost.shape[1]),
    ):
        if masses.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if masses.dtype != cost.dtype or masses.device != cost.device:
            raise ValueError(f"{name} must have the same dtype and device as cost")
        if not torch.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError(f"{name} must contain finite positive masses")
        if not torch.isclose(masses.sum(), masses.new_tensor(1.0)):
            raise ValueError(f"{name} must sum to one")

    log_kernel = -cost.detach() / epsilon
    if not torch.isfinite(log_kernel).all():
        raise RuntimeError("Nonfinite Sinkhorn log kernel; check cost scale and epsilon")
    log_example_weights = example_weights.log()
    log_reference_masses = reference_masses.log()
    log_column_scale = torch.zeros_like(reference_masses)
    for _ in range(max_iterations):
        log_row_scale = log_example_weights - torch.logsumexp(
            log_kernel + log_column_scale[None, :], dim=1
        )
        log_column_scale = log_reference_masses - torch.logsumexp(
            log_kernel + log_row_scale[:, None], dim=0
        )
    transport = (log_kernel + log_row_scale[:, None] + log_column_scale[None, :]).exp()
    if not torch.isfinite(transport).all():
        raise RuntimeError("Nonfinite Sinkhorn transport plan")
    with torch.no_grad():
        diagnostics = {
            "sinkhorn_iterations": max_iterations,
            "row_marginal_error": (transport.sum(dim=1) - example_weights).abs().max().item(),
            "column_marginal_error": (transport.sum(dim=0) - reference_masses).abs().max().item(),
            "row_relative_error": ((transport.sum(dim=1) - example_weights).abs() / example_weights).max().item(),
            "column_relative_error": ((transport.sum(dim=0) - reference_masses).abs() / reference_masses).max().item(),
        }
    return transport, diagnostics


def _update_example_scores(
    example_scores, optimizer, cost, reference_masses, *,
    score_updates=1, epsilon=0.1, max_iterations=200, marginal_tolerance=0.001,
):
    """Update global scores, then return fixed weights for one classifier epoch.

    The caller supplies a persistent optimizer over example_scores and calls
    this before each classifier epoch. Diagnostics describe the last solve
    before its optimizer step; returned weights reflect all completed steps.
    """
    if type(score_updates) is not int or score_updates < 1:
        raise ValueError("score_updates must be a positive integer")
    if (
        type(marginal_tolerance) not in (int, float)
        or not math.isfinite(marginal_tolerance)
        or marginal_tolerance <= 0
    ):
        raise ValueError("marginal_tolerance must be a finite positive number")
    fixed_cost = cost.detach()
    fixed_reference_masses = reference_masses.detach()
    for _ in range(score_updates):
        optimizer.zero_grad(set_to_none=True)
        weights = _global_example_weights(example_scores)
        transport, diagnostics = _sinkhorn_transport(
            fixed_cost, weights, fixed_reference_masses,
            epsilon=epsilon, max_iterations=max_iterations,
        )
        if max(diagnostics["row_relative_error"], diagnostics["column_relative_error"]) > marginal_tolerance:
            raise RuntimeError(
                f"Sinkhorn marginal relative errors exceed tolerance {marginal_tolerance}: "
                f"rows={diagnostics['row_relative_error']:.6g}, "
                f"columns={diagnostics['column_relative_error']:.6g}. "
                "Increase sinkhorn_iterations before retrying."
            )
        ot_cost = (transport * fixed_cost).sum()
        if not torch.isfinite(ot_cost):
            raise RuntimeError("Nonfinite OT cost during example-score update")
        ot_cost.backward()
        if example_scores.grad is None or not torch.isfinite(example_scores.grad).all():
            raise RuntimeError("Missing or nonfinite example-score gradients")
        optimizer.step()
    with torch.no_grad():
        classifier_weights = _global_example_weights(example_scores)
    return classifier_weights, diagnostics | {
        "score_updates": score_updates,
        "ot_cost_before_update": ot_cost.item(),
    }


def _weighted_cross_entropy(logits, labels, batch_weights, example_count):
    losses = functional.cross_entropy(logits, labels, reduction="none")
    return (example_count * batch_weights.detach() * losses).mean()


def _effective_sample_size(weights):
    # Account for rounding in total probability mass without modifying training
    # weights. For masses summing exactly to one this is 1 / sum(weights**2).
    masses = weights.to(torch.float64)
    return (masses.sum().square() / masses.square().sum()).item()


def _train_classifier_epoch(model, loader, optimizer, weights, device):
    """Train on cached features using global weights indexed by retained position."""
    model.backbone.eval()
    model.fc.train()
    loss_sum = 0.0
    sample_count = 0
    for features, labels, positions in loader:
        features, labels = features.to(device), labels.to(device)
        batch_weights = weights[positions.to(weights.device)]
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_cross_entropy(model.fc(features), labels, batch_weights, len(weights))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite weighted classifier loss")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        sample_count += len(labels)
    return loss_sum / sample_count


def train(model, train_data, config, output_dir, device) -> list[dict]:
    options = config["method_options"]
    # Fail before pretraining if a balanced reference cannot be constructed.
    if (train_data.class_counts == 0).any():
        raise ValueError("Reweighting requires retained training examples in every class")
    history = _pretrain(model, train_data, config, output_dir, device)
    print("Caching deterministic training features for frozen-backbone Stage 2", flush=True)
    raw_features, labels = _cache_training_features(model, train_data, config, device)
    normalized_features, references, reference_masses = _build_ot_references(raw_features, labels)
    cost = _build_ot_cost(
        normalized_features.to(device), references.to(device), labels.to(device),
        feature_cost_scale=options["feature_cost_scale"],
        label_mismatch_cost=options["label_mismatch_cost"],
    )
    reference_masses = reference_masses.to(device)
    example_scores = _initialize_example_scores(len(labels), device, dtype=cost.dtype)
    score_optimizer = _make_score_optimizer(example_scores, options["score_learning_rate"])
    classifier_options = options["classifier"]
    classifier_optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=classifier_options["learning_rate"],
        momentum=classifier_options["momentum"], weight_decay=classifier_options["weight_decay"],
        nesterov=classifier_options["nesterov"],
    )
    # Cached tensors are small; worker processes offer no image preprocessing benefit.
    loader = DataLoader(
        TensorDataset(raw_features, labels, torch.arange(len(labels))),
        batch_size=config["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    epoch_diagnostics = []
    for stage_epoch in range(1, classifier_options["epochs"] + 1):
        print(f"Reweighting epoch {stage_epoch}/{classifier_options['epochs']}: updating global scores", flush=True)
        weights, transport_diagnostics = _update_example_scores(
            example_scores, score_optimizer, cost, reference_masses,
            score_updates=options["score_updates"], epsilon=options["epsilon"],
            max_iterations=options["sinkhorn_iterations"], marginal_tolerance=options["marginal_tolerance"],
        )
        loss = _train_classifier_epoch(model, loader, classifier_optimizer, weights, device)
        epoch = options["pretraining"]["epochs"] + stage_epoch
        record = {
            "epoch": epoch, "stage": "reweighting", "stage_epoch": stage_epoch,
            "loss": loss, "learning_rate": classifier_optimizer.param_groups[0]["lr"],
            "score_learning_rate": score_optimizer.param_groups[0]["lr"],
            **transport_diagnostics,
            "maximum_weight": weights.max().item(),
            "effective_sample_size": _effective_sample_size(weights),
        }
        history.append(record)
        class_weight_masses = torch.zeros(NUM_CLASSES, device=device, dtype=weights.dtype)
        class_weight_masses.scatter_add_(0, labels.to(device), weights)
        epoch_diagnostics.append({"epoch": epoch, "class_weight_masses": class_weight_masses.cpu().tolist()})
        write_json(output_dir / "reweighting_diagnostics.json", epoch_diagnostics)
        print(
            f"Reweighting epoch {stage_epoch}/{classifier_options['epochs']} loss={loss:.4f} "
            f"ot_cost={record['ot_cost_before_update']:.6g} "
            f"row_error={record['row_relative_error']:.3g} col_error={record['column_relative_error']:.3g}",
            flush=True,
        )
    return history
