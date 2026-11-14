"""Fixed sample weights for class and prototype balancing."""

import math

import torch
import torch.nn.functional as functional

from pipeline.data import NUM_CLASSES


def prepare_options(options):
    weighting = options.get("weighting")
    if weighting not in ("class_balanced", "cluster_balanced", "sinkhorn"):
        raise ValueError("weighting must be class_balanced, cluster_balanced, or sinkhorn")
    defaults = {}
    if weighting != "class_balanced":
        defaults["minimum_cluster_size"] = 5
    if weighting == "sinkhorn":
        defaults.update(
            epsilon=0.1, row_kl_strength=0.2, sinkhorn_iterations=200, sinkhorn_tolerance=0.001,
        )
    unknown = options.keys() - defaults.keys() - {"weighting"}
    if unknown:
        raise ValueError(f"Unknown prototype weighting options: {', '.join(sorted(unknown))}")
    resolved = defaults | options
    for name in defaults:
        value = resolved[name]
        if type(defaults[name]) is int:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return resolved


def _cluster_class(features, minimum_size, generator):
    if len(features) < 2 * minimum_size:
        return None
    first_index = torch.randint(len(features), (), generator=generator)
    first_center = features[first_index]
    distances = (1 - features @ first_center).clamp_min(0)
    distances[first_index] = 0
    if distances.max() <= 1e-6:
        return None
    second_center = features[torch.multinomial(distances, 1, generator=generator)[0]]
    centers = torch.stack((first_center, second_center))
    previous_assignments = None
    for _ in range(50):
        assignments = (features @ centers.T).argmax(dim=1)
        sizes = torch.bincount(assignments, minlength=2)
        if (sizes == 0).any():
            return None
        if previous_assignments is not None and torch.equal(assignments, previous_assignments):
            if (sizes < minimum_size).any():
                return None
            return assignments, sizes, centers
        centers = torch.stack([features[assignments == cluster].mean(dim=0) for cluster in range(2)])
        if (centers.norm(dim=1) <= 1e-6).any():
            return None
        centers = functional.normalize(centers, dim=1)
        previous_assignments = assignments
    return None


def _relaxed_sinkhorn(cost, *, epsilon, row_kl_strength, sinkhorn_iterations, sinkhorn_tolerance):
    """Fix equal prototype masses and penalize departure from uniform sample masses."""
    log_kernel = -cost.to(torch.float64) / epsilon
    if not torch.isfinite(log_kernel).all():
        raise RuntimeError("Nonfinite prototype Sinkhorn kernel")
    log_sample_mass = -math.log(cost.shape[0])
    log_prototype_mass = -math.log(cost.shape[1])
    row_exponent = 1 / (1 + epsilon / row_kl_strength)
    log_column_scale = log_kernel.new_zeros(cost.shape[1])
    for iteration in range(1, sinkhorn_iterations + 1):
        log_row_scale = row_exponent * (
            log_sample_mass - torch.logsumexp(log_kernel + log_column_scale, dim=1)
        )
        next_column_scale = log_prototype_mass - torch.logsumexp(
            log_kernel + log_row_scale[:, None], dim=0
        )
        # Column sums alone cannot detect convergence because each update enforces them.
        residual = (next_column_scale - log_column_scale).abs().max().item()
        log_column_scale = next_column_scale
        if residual <= sinkhorn_tolerance:
            plan = (log_kernel + log_row_scale[:, None] + log_column_scale).exp()
            column_error = (plan.sum(dim=0) * cost.shape[1] - 1).abs().max().item()
            if not math.isfinite(column_error) or column_error > sinkhorn_tolerance:
                raise RuntimeError("Prototype Sinkhorn did not meet its column quotas")
            return plan, {
                "sinkhorn_iterations": iteration,
                "sinkhorn_residual": residual,
                "column_relative_error": column_error,
            }
    raise RuntimeError("Prototype Sinkhorn did not converge; check the iteration budget")


@torch.no_grad()
def build_weights(raw_features, labels, config, device):
    options = config["method_options"]
    weighting = options["weighting"]
    generator = torch.Generator().manual_seed(config["seed"])
    weights = torch.empty(len(labels), dtype=raw_features.dtype)
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    if (counts == 0).any():
        raise ValueError("Prototype fine-tuning requires retained training examples in every class")
    class_diagnostics = []
    for class_index, count in enumerate(counts.tolist()):
        class_rows = labels == class_index
        class_weights = weights.new_full((count,), 1 / count)
        diagnostics = {"class_index": class_index, "cluster_sizes": None}
        if weighting != "class_balanced":
            features = functional.normalize(raw_features[class_rows], dim=1)
            clusters = _cluster_class(features, options["minimum_cluster_size"], generator)
            if clusters is not None:
                assignments, sizes, centers = clusters
                diagnostics["cluster_sizes"] = sizes.tolist()
                if weighting == "cluster_balanced":
                    class_weights = 0.5 / sizes[assignments].to(weights.dtype)
                else:
                    cost = (1 - features @ centers.T).clamp_min(0)
                    plan, transport_diagnostics = _relaxed_sinkhorn(
                        cost, epsilon=options["epsilon"], row_kl_strength=options["row_kl_strength"],
                        sinkhorn_iterations=options["sinkhorn_iterations"],
                        sinkhorn_tolerance=options["sinkhorn_tolerance"],
                    )
                    class_weights = plan.sum(dim=1).to(weights.dtype)
                    diagnostics.update(transport_diagnostics)
        weights[class_rows] = class_weights / NUM_CLASSES
        diagnostics["minimum_weight_multiplier"] = (count * class_weights.min()).item()
        diagnostics["maximum_weight_multiplier"] = (count * class_weights.max()).item()
        class_diagnostics.append(diagnostics)
    return weights.to(device), class_diagnostics
