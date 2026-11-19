"""K-class weighting rules; ordinary builders receive only retained/reference rows."""

import torch
import torch.nn.functional as functional

from pipeline.methods.prototype_weights import _cluster_class, _relaxed_sinkhorn

GEOMETRY_CONDITIONS = ("equal_target_ot", "reference_target_ot", "reference_cluster_ratio")


def validate_weights(weights, labels, class_count):
    if (weights.dtype != torch.float64 or weights.shape != labels.shape
            or not torch.isfinite(weights).all() or (weights <= 0).any()
            or labels.dtype != torch.int64 or (labels < 0).any() or (labels >= class_count).any()):
        raise ValueError("Invalid diagnostic weights or labels")
    masses = torch.zeros(class_count, dtype=torch.float64).scatter_add_(0, labels, weights)
    if not torch.allclose(masses, torch.full_like(masses, 1 / class_count), rtol=1e-10, atol=1e-12):
        raise ValueError("Diagnostic class masses must equal 1/K")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.), rtol=1e-10, atol=1e-12):
        raise ValueError("Diagnostic total mass must equal one")


def class_balanced(labels, class_count):
    counts = torch.bincount(labels, minlength=class_count)
    if len(counts) != class_count or (counts == 0).any():
        raise ValueError("Every eligible class needs fitting rows")
    weights = 1 / (class_count * counts[labels].double())
    validate_weights(weights, labels, class_count)
    return weights


def known_correction(labels, unbiased_counts, probabilities):
    if (probabilities.shape != labels.shape or not torch.isfinite(probabilities).all()
            or (probabilities <= 0).any() or (probabilities > 1).any()):
        raise ValueError("Invalid inclusion probabilities")
    weights = 1 / (len(unbiased_counts) * unbiased_counts[labels].double() * probabilities.double())
    validate_weights(weights, labels, len(unbiased_counts))
    return weights


@torch.no_grad()
def build_geometry_weights(features, labels, reference_features, reference_labels, class_count, seed, config):
    """One T-only geometry per seed; reference rows estimate only prototype quotas."""
    base = class_balanced(labels, class_count)
    weights = {name: base.clone() for name in GEOMETRY_CONDITIONS}
    geometry, reports = [], []
    generator = torch.Generator().manual_seed(seed)
    for c in range(class_count):
        rows = labels == c
        normalized = functional.normalize(features[rows], dim=1)
        diagnostic = {"class_index": c, "fallback_reason": None, "is_active": False}
        clusters = _cluster_class(normalized, config["minimum_cluster_size"], generator, diagnostic)
        saved = {"class_index": c, "feature_seed": seed, "assignments": None, "centers": None}
        if clusters is not None:
            assignments, sizes, centers = clusters
            normalized_reference = functional.normalize(reference_features[reference_labels == c], dim=1)
            if len(normalized_reference) == 0:
                raise ValueError("Every eligible class needs reference rows")
            reference_assignments = (normalized_reference @ centers.T).argmax(dim=1)
            reference_counts = torch.bincount(reference_assignments, minlength=2)
            smoothing = config["reference_pseudocount"]
            target = (reference_counts.double() + smoothing) / (len(normalized_reference) + 2 * smoothing)
            cost = (1 - normalized @ centers.T).clamp_min(0)
            transports = {}
            for name, masses in (("equal_target_ot", None), ("reference_target_ot", target)):
                plan, transport = _relaxed_sinkhorn(
                    cost, epsilon=config["epsilon"], row_kl_strength=config["row_kl_strength"],
                    sinkhorn_iterations=config["sinkhorn_iterations"],
                    sinkhorn_tolerance=config["sinkhorn_tolerance"], prototype_masses=masses,
                )
                values = plan.sum(dim=1) / class_count
                weights[name][rows] = values
                hard_mass = torch.zeros(2, dtype=torch.float64).scatter_add_(0, assignments, values) * class_count
                transports[name] = transport | {"soft_column_masses": plan.sum(0).tolist(),
                                               "hard_group_fractions": hard_mass.tolist(),
                                               "hard_group_target_absolute_error": (hard_mass - (target if masses is not None else .5)).abs().tolist()}
            weights["reference_cluster_ratio"][rows] = target[assignments] / (class_count * sizes[assignments].double())
            simple_mass = torch.zeros(2, dtype=torch.float64).scatter_add_(
                0, assignments, weights["reference_cluster_ratio"][rows])
            if not torch.allclose(simple_mass, target / class_count, rtol=1e-10, atol=1e-12):
                raise RuntimeError("Simple correction failed reference cluster masses")
            saved.update(assignments=assignments.clone(), centers=centers.clone(), target=target,
                         reference_assignments=reference_assignments)
            diagnostic.update(is_active=True, cluster_sizes=sizes.tolist(),
                              reference_counts=reference_counts.tolist(),
                              raw_reference_proportions=(reference_counts.double() / len(normalized_reference)).tolist(),
                              target=target.tolist(), transport=transports)
        geometry.append(saved)
        reports.append(diagnostic)
    for name, values in weights.items():
        validate_weights(values, labels, class_count)
        for c, diagnostic in enumerate(reports):
            if not diagnostic["is_active"] and not torch.equal(values[labels == c], base[labels == c]):
                raise RuntimeError(f"Fallback weights changed: {name}, class {c}")
    for shuffle_seed in config["shuffle_seeds"]:
        generator = torch.Generator().manual_seed(shuffle_seed)
        original = weights["reference_target_ot"]
        shuffled = original.clone()
        for c in range(class_count):
            rows = torch.where(labels == c)[0]
            shuffled[rows] = original[rows[torch.randperm(len(rows), generator=generator)]]
            if not torch.equal(shuffled[rows].sort().values, original[rows].sort().values):
                raise RuntimeError("Shuffle changed the weight multiset")
        validate_weights(shuffled, labels, class_count)
        weights[f"shuffled_reference_ot_shuffle{shuffle_seed}"] = shuffled
    return weights, geometry, reports


def weight_diagnostics(weights, labels, class_count, controller_groups, unbiased_labels, unbiased_groups):
    """Privileged reporting is separate from all ordinary weight construction."""
    validate_weights(weights, labels, class_count)
    classes = []
    for c in range(class_count):
        selected = labels == c
        values = weights[selected]
        desired = torch.bincount(unbiased_groups[unbiased_labels == c], minlength=2).double()
        desired /= desired.sum()
        mass = torch.zeros(2, dtype=torch.float64).scatter_add_(0, controller_groups[selected], values) * class_count
        multipliers = values * class_count * len(values)
        classes.append({"class_index": c, "count": len(values), "class_mass": float(values.sum()),
                        "controller_group_masses": (mass / class_count).tolist(),
                        "controller_group_fractions": mass.tolist(), "desired_unbiased_fractions": desired.tolist(),
                        "controller_mass_absolute_error": float((mass - desired).abs().mean()),
                        "minimum_multiplier": float(multipliers.min()), "maximum_multiplier": float(multipliers.max()),
                        "effective_sample_size": float(values.sum().square() / values.square().sum()),
                        "effective_sample_size_fraction": float(values.sum().square() / values.square().sum() / len(values))})
    return {"classes": classes, "macro_controller_mass_absolute_error": sum(r["controller_mass_absolute_error"] for r in classes) / class_count}


def geometry_diagnostics(geometry, retained_labels, retained_groups, reference_labels, reference_groups,
                         unbiased_labels, unbiased_groups):
    reports = []
    for c, current in enumerate(geometry):
        desired = torch.bincount(unbiased_groups[unbiased_labels == c], minlength=2).double()
        desired /= desired.sum()
        reference_fraction = torch.bincount(reference_groups[reference_labels == c], minlength=2).double()
        reference_fraction /= reference_fraction.sum()
        report = {"class_index": c, "is_active": current["assignments"] is not None,
                  "reference_controller_fractions": reference_fraction.tolist(),
                  "reference_versus_unbiased_absolute_error": float((reference_fraction - desired).abs().mean())}
        if current["assignments"] is None:
            reports.append(report)
            continue
        contingency = torch.bincount(retained_groups[retained_labels == c] * 2 + current["assignments"], minlength=4).reshape(2, 2)
        should_swap = bool(contingency.flip(1).trace() > contingency.trace())
        aligned_target = current["target"].flip(0) if should_swap else current["target"]
        reports.append(report | {"controller_by_method_contingency": contingency.tolist(),
                        "reporting_method_label_permutation": [1, 0] if should_swap else [0, 1],
                        "aligned_method_target_versus_unbiased_error": float((aligned_target - desired).abs().mean())})
    return reports
