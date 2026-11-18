"""Fixed controls and diagnostics for prototype OT example weights."""

import torch

from pipeline.data import NUM_CLASSES

STRONG_OT_EXPONENT = 3


def class_balanced_weights(labels):
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    if (counts == 0).any():
        raise ValueError("Ablation weights require fitting examples in all 100 classes")
    return (1 / (NUM_CLASSES * counts[labels].double())).to(torch.float32)


def transform_weights(ot_weights, labels, condition, shuffle_seed=None):
    """Preserve class mass; shuffle only with a dedicated CPU generator."""
    if ot_weights.shape != labels.shape or not torch.isfinite(ot_weights).all() or (ot_weights <= 0).any():
        raise ValueError("OT weights must be a finite positive vector aligned with labels")
    balanced = class_balanced_weights(labels).to(ot_weights)
    masses = torch.zeros(NUM_CLASSES, dtype=ot_weights.dtype, device=ot_weights.device)
    masses.scatter_add_(0, labels.to(ot_weights.device), ot_weights)
    if not torch.allclose(masses, torch.full_like(masses, 1 / NUM_CLASSES), rtol=1e-5, atol=1e-8):
        raise ValueError("Original OT weights must give every class mass 1/100")
    if condition == "class_balanced":
        return balanced
    if condition == "ot":
        return ot_weights.clone()
    if condition == "half_ot":
        return 0.5 * balanced + 0.5 * ot_weights
    if condition == "uniform":
        return torch.full_like(ot_weights, 1 / len(labels))
    if condition not in ("strong_ot", "shuffled_ot", "shuffled_strong_ot"):
        raise ValueError(f"Unknown ablation condition: {condition}")
    is_strong = condition in ("strong_ot", "shuffled_strong_ot")
    is_shuffled = condition in ("shuffled_ot", "shuffled_strong_ot")
    if is_shuffled and (type(shuffle_seed) is not int or not 0 <= shuffle_seed < 2**32):
        raise ValueError("Shuffled OT requires a shuffle seed in [0, 2**32)")
    generator = torch.Generator().manual_seed(shuffle_seed) if is_shuffled else None
    weights = ot_weights.clone()
    for c in range(NUM_CLASSES):
        positions = torch.where(labels == c)[0]
        values = ot_weights[positions]
        if is_strong:
            # Cube relative weights, then restore the original 1/100 class total.
            amplified = (values.double() / values.max().double()).pow(STRONG_OT_EXPONENT)
            values = (amplified / amplified.sum() / NUM_CLASSES).to(values.dtype)
        if generator is not None:
            order = torch.randperm(len(positions), generator=generator).to(positions.device)
            values = values[order]
        weights[positions] = values
    return weights


def weight_diagnostics(weights, labels, original_counts, prototype_diagnostics):
    records = []
    for c in range(NUM_CLASSES):
        values = weights[labels == c].double()
        multipliers = values * NUM_CLASSES * len(values)
        ess = (values.sum().square() / values.square().sum()).item()
        records.append({
            "class_index": c, "fitting_count": len(values), "class_mass": values.sum().item(),
            "minimum_multiplier": multipliers.min().item(),
            "maximum_multiplier": multipliers.max().item(),
            "multiplier_sd": multipliers.std(unbiased=False).item(),
            "effective_sample_size": ess, "effective_sample_fraction": ess / len(values),
            "used_ot": prototype_diagnostics[c]["used_ot"],
            "fallback_reason": prototype_diagnostics[c]["fallback_reason"],
        })
    coverage = {}
    for name, selected in (
        ("all", [True] * NUM_CLASSES),
        ("head", [n >= 100 for n in original_counts]),
        ("medium", [20 <= n < 100 for n in original_counts]),
        ("tail", [n < 20 for n in original_counts]),
    ):
        classes = [r for r, keep in zip(records, selected) if keep]
        active = sum(r["used_ot"] for r in classes)
        coverage[name] = {"class_count": len(classes), "ot_active": active, "fallback": len(classes) - active}
    return {"classes": records, "prototype_ot_coverage": coverage}
