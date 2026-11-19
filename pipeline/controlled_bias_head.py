"""Fresh, regularized CPU linear heads and reporter-only development scoring."""

import math

import torch
import torch.nn.functional as functional

from pipeline.controlled_bias_weights import validate_weights


class HeadConvergenceError(RuntimeError):
    def __init__(self, message, diagnostics, history):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.history = history


def fit_head(features, labels, weights, class_count, options):
    """No development/reference inputs; every solve starts at exactly zero."""
    validate_weights(weights, labels, class_count)
    features = features.detach().to(device="cpu", dtype=torch.float64)
    if features.ndim != 2 or len(features) != len(labels) or not torch.isfinite(features).all():
        raise ValueError("Invalid head fitting features")
    # Creating parameters directly avoids consuming the global random generator.
    matrix = torch.nn.Parameter(torch.zeros(class_count, features.shape[1], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(class_count, dtype=torch.float64))
    parameters = [matrix, bias]
    optimizer = torch.optim.LBFGS(
        parameters, lr=options["learning_rate"], history_size=options["history_size"],
        max_iter=options["max_iterations"], max_eval=options["max_evaluations"],
        tolerance_grad=options["gradient_tolerance"], tolerance_change=options["change_tolerance"],
        line_search_fn="strong_wolfe",
    )
    history = []

    def objective():
        ce = (weights * functional.cross_entropy(functional.linear(features, matrix, bias), labels, reduction="none")).sum()
        regularizer = options["regularization"] / 2 * (matrix.square().sum() + bias.square().sum())
        return ce, regularizer

    def closure():
        # L-BFGS checks max_eval after a line search; enforce the actual hard
        # objective-evaluation budget here as well.
        if len(history) >= options["max_evaluations"]:
            raise HeadConvergenceError("Head objective evaluation budget exhausted", {"evaluations": len(history)}, history)
        optimizer.zero_grad()
        ce, regularizer = objective()
        loss = ce + regularizer
        if not torch.isfinite(loss):
            raise HeadConvergenceError("Nonfinite head objective", {"evaluations": len(history)}, history)
        loss.backward()
        gradient_norm = max(float(p.grad.abs().max()) for p in parameters)
        if not all(torch.isfinite(p.grad).all() for p in parameters):
            raise HeadConvergenceError("Nonfinite head gradient", {"evaluations": len(history)}, history)
        history.append({"evaluation": len(history) + 1, "objective": float(loss.detach()),
                        "weighted_cross_entropy": float(ce.detach()), "regularizer": float(regularizer.detach()),
                        "gradient_infinity_norm": gradient_norm})
        return loss

    optimizer.step(closure)
    optimizer.zero_grad()
    ce, regularizer = objective()
    (ce + regularizer).backward()
    gradient_norm = max(float(p.grad.abs().max()) for p in parameters)
    state = optimizer.state[matrix]
    diagnostics = {"objective": float((ce + regularizer).detach()), "weighted_cross_entropy": float(ce.detach()),
                   "regularizer": float(regularizer.detach()), "iterations": state["n_iter"],
                   "evaluations": len(history), "verification_evaluations": 1,
                   "gradient_infinity_norm": gradient_norm,
                   "parameter_norm": float(torch.cat([p.detach().flatten() for p in parameters]).norm()),
                   "is_converged": bool(gradient_norm <= options["accepted_gradient_norm"]
                                        and all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() for p in parameters)
                                        and torch.isfinite(ce + regularizer))}
    if not diagnostics["is_converged"]:
        # Failure diagnostics must still be valid JSON when a numerical value
        # is nonfinite; the failure itself must not be hidden by serialization.
        diagnostics = {name: None if isinstance(value, float) and not math.isfinite(value) else value
                       for name, value in diagnostics.items()}
        raise HeadConvergenceError("Head failed the final full-gradient convergence check", diagnostics, history)
    return {"weight": matrix.detach().clone(), "bias": bias.detach().clone()}, diagnostics, history


@torch.no_grad()
def evaluate_head(head, features, labels, controller_groups, biased_groups, original_classes):
    logits = functional.linear(features.double(), head["weight"], head["bias"])
    losses = functional.cross_entropy(logits, labels, reduction="none")
    if not torch.isfinite(logits).all() or not torch.isfinite(losses).all():
        raise RuntimeError("Nonfinite development scores")
    predictions = logits.argmax(dim=1)
    correct = predictions == labels
    class_count = len(original_classes)
    classes, groups = [], []
    for c, original_class in enumerate(original_classes.tolist()):
        selected = labels == c
        support = int(selected.sum())
        count = int(correct[selected].sum())
        classes.append({"class_index": c, "original_class": original_class, "support": support,
                        "correct": count, "recall": count / support if support else None,
                        "cross_entropy": float(losses[selected].mean()) if support else None})
        for g in (0, 1):
            mask = selected & (controller_groups == g)
            group_support = int(mask.sum())
            group_correct = int(correct[mask].sum())
            groups.append({"class_index": c, "controller_group": g,
                           "role": "downsampled" if g == int(biased_groups[c]) else "retained_all",
                           "support": group_support, "correct": group_correct,
                           "recall": group_correct / group_support if group_support else None,
                           "balanced_accuracy_contribution": group_correct / support / class_count if support else None})
    recalls = [r["recall"] for r in classes]
    metrics = {"diagnostic_balanced_accuracy": sum(recalls) / class_count if None not in recalls else None,
               "accuracy": float(correct.double().mean()), "cross_entropy": float(losses.mean()),
               "class_macro_cross_entropy": sum(r["cross_entropy"] for r in classes) / class_count if None not in recalls else None,
               "classes": classes, "controller_groups": groups}
    return metrics, {"predictions": predictions, "labels": labels.clone(), "cross_entropy": losses,
                     "controller_groups": controller_groups.clone()}
