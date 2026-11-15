"""SAVA-inspired hierarchical dual valuation on frozen training features.

This uses a feature/label cost, not the original SAVA OTDD pixel cost.
"""

import math

import torch
import torch.nn.functional as functional

from pipeline.data import NUM_CLASSES


def prepare_options(options):
    if not isinstance(options, dict):
        raise ValueError("sava options must be a JSON object")
    defaults = dict(reference_per_class=5, reference_seed=42, scoring_batch_size=256,
                    epsilon=0.1, label_cost=1.0, temperature=1.0,
                    sinkhorn_iterations=1000, sinkhorn_tolerance=0.001)
    unknown = options.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown SAVA options: {', '.join(sorted(unknown))}")
    resolved = defaults | options
    for name, default in defaults.items():
        value = resolved[name]
        minimum = 0 if name == "reference_seed" else 1
        if type(default) is int:
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if resolved["scoring_batch_size"] < 2:
        raise ValueError("scoring_batch_size must be >= 2")
    if resolved["reference_seed"] >= 2**32:
        raise ValueError("reference_seed must be less than 2**32")
    return resolved


def _transport(cost, source_mass, target_mass, options):
    """Balanced entropy-regularized OT with explicit marginal verification."""
    if not torch.isfinite(cost).all():
        raise ValueError("Nonfinite SAVA transport cost")
    epsilon = options["epsilon"]
    kernel = -cost / epsilon
    log_target_scale = torch.zeros_like(target_mass)
    for iteration in range(1, options["sinkhorn_iterations"] + 1):
        log_source_scale = source_mass.log() - torch.logsumexp(kernel + log_target_scale, dim=1)
        log_target_scale = target_mass.log() - torch.logsumexp(
            kernel + log_source_scale[:, None], dim=0,
        )
        plan = (kernel + log_source_scale[:, None] + log_target_scale).exp()
        error = max(((plan.sum(1) / source_mass) - 1).abs().max().item(),
                    ((plan.sum(0) / target_mass) - 1).abs().max().item())
        if math.isfinite(error) and error <= options["sinkhorn_tolerance"]:
            potential = epsilon * log_source_scale
            # Remove the arbitrary dual offset before combining batch-pair scores.
            potential = potential - (potential * source_mass).sum()
            return plan, potential, iteration, error
    raise RuntimeError("SAVA transport did not converge")


def _scores_to_weights(scores, labels, temperature):
    weights = torch.empty_like(scores)
    for class_index in range(NUM_CLASSES):
        rows = labels == class_index
        values = scores[rows]
        centered = values - values.mean()
        scale = centered.square().mean().sqrt()
        # Constant scores recover equal within-class weights.
        standardized = centered / scale.clamp_min(1e-12)
        weights[rows] = torch.softmax(-standardized / temperature, dim=0) / NUM_CLASSES
    return weights


def _scoring_batches(positions, batch_size):
    # At size 2, an odd population needs one batch of 3 rather than a singleton.
    batch_count = min(math.ceil(len(positions) / batch_size), len(positions) // 2)
    return torch.tensor_split(positions, batch_count)


@torch.no_grad()
def build_weights(raw_features, labels, config, device):
    options = config["method_options"]["sava"]
    if not torch.isfinite(raw_features).all():
        raise ValueError("Nonfinite SAVA features")
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    reference_count = options["reference_per_class"]
    if (counts < reference_count).any():
        raise ValueError("Every class must contain reference_per_class retained images")
    reference_generator = torch.Generator().manual_seed(options["reference_seed"])
    references = torch.cat([
        torch.where(labels == c)[0][torch.randperm(int(counts[c]), generator=reference_generator)[:reference_count]]
        for c in range(NUM_CLASSES)
    ])
    generator = torch.Generator().manual_seed(config["seed"])
    references = references[torch.randperm(len(references), generator=generator)]
    order = torch.randperm(len(labels), generator=generator)
    # Balanced chunks avoid singleton final batches, whose calibrated dual is undefined.
    size = options["scoring_batch_size"]
    source_batches = _scoring_batches(order, size)
    reference_batches = _scoring_batches(references, size)
    features = functional.normalize(raw_features.to(device=device, dtype=torch.float64), dim=1)
    device_labels = labels.to(device)
    costs = features.new_empty((len(source_batches), len(reference_batches)))
    potentials = {}
    max_error, max_iterations = 0.0, 0
    for i, source in enumerate(source_batches):
        for j, target in enumerate(reference_batches):
            source_rows, target_rows = source.to(device), target.to(device)
            cost = (1 - features[source_rows] @ features[target_rows].T).clamp_min(0)
            cost += options["label_cost"] * (device_labels[source_rows, None] != device_labels[target_rows])
            a = cost.new_full((len(source),), 1 / len(source))
            b = cost.new_full((len(target),), 1 / len(target))
            plan, potential, iterations, error = _transport(cost, a, b, options)
            costs[i, j] = (plan * cost).sum()
            potentials[i, j] = potential * len(source) / (len(source) - 1)
            max_error, max_iterations = max(max_error, error), max(max_iterations, iterations)
    a = features.new_tensor([len(batch) / len(labels) for batch in source_batches])
    b = features.new_tensor([len(batch) / len(references) for batch in reference_batches])
    # Scaling keeps outer entropy relative to observed batch costs.
    outer_cost = costs / costs.max().clamp_min(1e-12)
    batch_plan, _, iterations, error = _transport(outer_cost, a, b, options)
    scores = features.new_empty(len(labels))
    for i, source in enumerate(source_batches):
        scores[source.to(device)] = sum(
            batch_plan[i, j] / a[i] * potentials[i, j] for j in range(len(reference_batches))
        )
    if not torch.isfinite(scores).all():
        raise RuntimeError("Nonfinite SAVA scores")
    weights = _scores_to_weights(scores, device_labels, options["temperature"])
    if not torch.isfinite(weights).all():
        raise RuntimeError("Nonfinite SAVA weights; check temperature")
    membership = torch.zeros(len(labels), dtype=torch.bool, device=device)
    membership[references.to(device)] = True
    diagnostics = dict(
        reference_positions=references.tolist(), scores=scores.cpu().tolist(),
        weights=weights.cpu().tolist(), maximum_marginal_relative_error=max(max_error, error),
        maximum_sinkhorn_iterations=max(max_iterations, iterations),
        reference_mean_weight=weights[membership].mean().item(),
        other_mean_weight=weights[~membership].mean().item() if (~membership).any() else None,
        classes=[dict(
            class_index=c,
            score_standard_deviation=scores[device_labels == c].std(unbiased=False).item(),
            minimum_weight=weights[device_labels == c].min().item(),
            maximum_weight=weights[device_labels == c].max().item(),
        ) for c in range(NUM_CLASSES)],
    )
    return weights.to(raw_features.dtype), diagnostics
