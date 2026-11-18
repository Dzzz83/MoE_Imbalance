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


@torch.inference_mode()
def baseline_scores(features, labels, state):
    """Score fitting examples once, without consuming a training generator."""
    logits = torch.nn.functional.linear(features, state["fc.weight"], state["fc.bias"])
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite baseline logits")
    correct = logits.argmax(dim=1) == labels
    true_scores = logits.gather(1, labels[:, None]).squeeze(1)
    logits.scatter_(1, labels[:, None], -torch.inf)
    margins = true_scores - logits.amax(dim=1)
    if not torch.isfinite(margins).all():
        raise ValueError("Nonfinite fitting margins")
    return margins, correct


def reassign_weights(strong_weights, labels, margins, condition, tie_order):
    """Reassign one fixed within-class multiset; larger priority gets larger mass."""
    priorities = {"easy_strong": margins, "boundary_strong": -margins.abs(), "hard_strong": -margins}
    if condition not in priorities:
        raise ValueError(f"Unknown ranked condition: {condition}")
    if (margins.shape != labels.shape or not torch.isfinite(margins).all()
            or tie_order.dtype != torch.int64 or tie_order.shape != labels.shape
            or not torch.equal(tie_order.sort().values, torch.arange(len(labels)))):
        raise ValueError("Expected finite aligned margins and a complete tie permutation")
    weights = transform_weights(strong_weights, labels, "ot")
    priority = priorities[condition]
    for c in range(NUM_CLASSES):
        positions = tie_order[labels[tie_order] == c]
        ordered = positions[torch.argsort(priority[positions], stable=True)]
        weights[ordered] = strong_weights[positions].sort().values
        if not torch.equal(weights[positions].sort().values, strong_weights[positions].sort().values):
            raise RuntimeError("Reassignment changed the class weight multiset")
    return weights


def _midranks(values):
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    return (counts.cumsum(0) - counts.double() / 2)[inverse]


def _quarter_membership(values, *, largest):
    count = (len(values) + 3) // 4
    cutoff = values.topk(count, largest=largest).values[-1]
    selected = values > cutoff if largest else values < cutoff
    ties = values == cutoff
    membership = selected.double() + ties.double() * (count - selected.sum()) / ties.sum()
    return membership, count, cutoff.item()


def assignment_audit(weights, labels, margins, correct, prototype_diagnostics):
    """Macro class diagnostics; ties share quarter membership and rank positions."""
    classes = []
    for c, diagnostic in enumerate(prototype_diagnostics):
        selected = labels == c
        values, scores = weights[selected].double(), margins[selected].double()
        ranks, score_ranks = _midranks(values), _midranks(scores)
        centered_weights, centered_scores = ranks - ranks.mean(), score_ranks - score_ranks.mean()
        denominator = centered_weights.norm() * centered_scores.norm()
        top, count, _ = _quarter_membership(values, largest=True)
        uncertain, _, cutoff = _quarter_membership(scores.abs(), largest=False)
        nonconstant = bool((values != values[0]).any())
        classes.append({
            "class_index": c, "fitting_count": len(values), "used_ot": diagnostic["used_ot"],
            "nonconstant_weights": nonconstant,
            "margin_weight_spearman": (centered_weights @ centered_scores / denominator).item()
            if denominator > 0 else None,
            "undefined_correlation_reason": None if denominator > 0 else "constant_weights_or_margins",
            "top_quarter_confidence_percentile_shift": (top @ score_ranks / len(values) / count).item() - 0.5,
            "mistake_mass_fraction": (values @ (~correct[selected]).double() / values.sum()).item(),
            "uncertain_quarter_mass_fraction": (values @ uncertain / values.sum()).item(),
            "uncertain_quarter_margin_cutoff": cutoff, "quarter_support": count,
        })
    summaries = {}
    metrics = ("margin_weight_spearman", "top_quarter_confidence_percentile_shift",
               "mistake_mass_fraction", "uncertain_quarter_mass_fraction")
    for name, members in (("all", classes), ("ot_active", [r for r in classes if r["used_ot"]])):
        summary = {"class_count": len(members), "nonconstant_class_count": sum(r["nonconstant_weights"] for r in members)}
        for metric in metrics:
            values = [r[metric] for r in members if r[metric] is not None]
            summary[metric] = sum(values) / len(values) if values else None
            summary[f"{metric}_class_count"] = len(values)
        summaries[name] = summary
    return {"classes": classes, "summary": summaries}
