"""100-class development endpoints and prespecified paired Stage C decisions."""

import math
import statistics

import numpy as np
import torch
import torch.nn.functional as functional

from pipeline.evaluate import compute_metrics
from pipeline.followup_weights import FEATURE_CONDITIONS, IMAGE_CONDITIONS

METRICS = ("balanced_accuracy", "accuracy", "head_recall", "medium_recall", "tail_recall",
           "cross_entropy", "class_macro_cross_entropy")


def evaluate_logits(logits, labels, original_counts):
    if logits.shape != (len(labels), 100) or not torch.isfinite(logits).all():
        raise ValueError("Invalid 100-class development logits")
    losses = functional.cross_entropy(logits, labels, reduction="none")
    if not torch.isfinite(losses).all():
        raise ValueError("Nonfinite development cross-entropy")
    predictions = logits.argmax(dim=1)
    metrics = compute_metrics(predictions.numpy(), labels.numpy(), np.asarray(original_counts))
    metrics["development_support"] = metrics.pop("test_support")
    class_losses = [float(losses[labels == c].mean()) if (labels == c).any() else None for c in range(100)]
    metrics.update(cross_entropy=float(losses.mean()), per_class_cross_entropy=class_losses,
                   class_macro_cross_entropy=statistics.mean(class_losses) if None not in class_losses else None)
    return metrics, {"predictions": predictions, "labels": labels.clone(), "cross_entropy": losses}


def contrast_definitions(stage):
    if stage == "ot_contribution":
        shuffle = {f"image_ot_shuffle{s}": -1 / 3 for s in (0, 1, 2)}
        return {
            "ot_minus_cb": {"image_ot": 1, "image_cb": -1},
            "ot_minus_easy_rank": {"image_ot": 1, "image_easy_rank": -1},
            "ot_minus_shuffle_mean": {"image_ot": 1, **shuffle},
            "easy_rank_minus_cb": {"image_easy_rank": 1, "image_cb": -1},
            "shuffle_mean_minus_cb": {k: -v for k, v in shuffle.items()} | {"image_cb": -1},
        }
    a, b, c, d = "feature_real_cb", "feature_real_ot", "feature_smote_cb", "feature_smote_ot"
    return {
        "synthesis_with_cb": {c: 1, a: -1}, "ot_after_synthesis": {d: 1, c: -1},
        "synthesis_with_ot": {d: 1, b: -1}, "ot_without_synthesis": {b: 1, a: -1},
        "interaction": {d: 1, c: -1, b: -1, a: 1},
        "ot_minus_prototype_distance": {d: 1, "feature_smote_prototype_distance": -1},
        "ot_minus_shuffle_mean": {d: 1, **{f"feature_smote_ot_shuffle{s}": -1 / 3 for s in (0, 1, 2)}},
        "synthesis_minus_duplicate_cb": {c: 1, "feature_duplicate_cb": -1},
        "synthesis_minus_duplicate_ot": {d: 1, "feature_duplicate_ot": -1},
        "combination_minus_real_cb": {d: 1, a: -1},
    }


def _mean_sd(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)} if None not in values else {
        "mean": None, "sample_sd": None}


def summarize(records, config):
    stage = config["stage"]
    conditions = IMAGE_CONDITIONS if stage == "ot_contribution" else FEATURE_CONDITIONS
    expected = {(s, c) for s in config["seeds"] for c in conditions}
    keys = [(r["seed"], r["condition"]) for r in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Duplicate or unexpected follow-up result")
    result = {"status": "incomplete", "completed_runs": len(keys), "expected_runs": len(expected),
              "decision": {"status": "incomplete"}}
    if set(keys) != expected:
        return result
    rows = {(r["seed"], r["condition"]): r["metrics"] for r in records}
    summaries = {c: {m: _mean_sd([rows[s, c][m] for s in config["seeds"]]) for m in METRICS}
                 for c in conditions}
    contrasts = {}
    for name, terms in contrast_definitions(stage).items():
        per_seed, metric_differences = [], {m: [] for m in METRICS}
        for seed in config["seeds"]:
            differences = {m: sum(v * rows[seed, c][m] for c, v in terms.items())
                           if all(rows[seed, c][m] is not None for c in terms) else None for m in METRICS}
            for m, value in differences.items():
                metric_differences[m].append(value)
            support = rows[seed, next(iter(terms))]["development_support"]
            contributions = [{"class_index": c, "development_support": support[c],
                              "balanced_accuracy_contribution": sum(
                                  v * rows[seed, condition]["per_class_recall"][c] / 100
                                  for condition, v in terms.items())} for c in range(100)]
            per_seed.append({"seed": seed, "differences": differences, "classes": contributions,
                             "one_image_class_contribution": sum(r["balanced_accuracy_contribution"]
                                                                  for r in contributions if r["development_support"] == 1)})
        primary = metric_differences["balanced_accuracy"]
        mean = statistics.mean(primary)
        # A shuffled mean can leave ~1e-17 residuals even for identical scores.
        positive_seeds = sum(v > 1e-12 for v in primary)
        target_met = ((mean >= config["practical_gain"] or math.isclose(mean, config["practical_gain"], rel_tol=0, abs_tol=1e-12))
                      and positive_seeds >= 2)
        contrasts[name] = {"terms": terms, "per_seed": per_seed,
                           "metrics": {m: _mean_sd(v) for m, v in metric_differences.items()},
                           "positive_seeds": positive_seeds, "target_met": target_met}
    if stage == "ot_contribution":
        requirements = {"image_ot": ["ot_minus_cb", "ot_minus_easy_rank", "ot_minus_shuffle_mean"]}
    else:
        ot = ["ot_after_synthesis", "ot_minus_prototype_distance", "ot_minus_shuffle_mean"]
        requirements = {"synthesis_without_ot": ["synthesis_with_cb", "synthesis_minus_duplicate_cb"],
                        "ot_after_synthesis": ot,
                        "synthesis_plus_ot": ot + ["combination_minus_real_cb", "synthesis_with_ot", "synthesis_minus_duplicate_ot"]}
    decisions = {name: {"required_contrasts": names, "target_met": all(contrasts[n]["target_met"] for n in names)}
                 for name, names in requirements.items()}
    result.update(status="complete", conditions=summaries, contrasts=contrasts,
                  decision={"status": "review_before_confirmation", "candidates": decisions,
                            "practical_gain": config["practical_gain"], "minimum_positive_seeds": 2,
                            "interpretation": "Budget screening only; three matched seeds on one split do not establish significance or equivalence.",
                            "next_stage": "Review both studies before conditional split-44 confirmation. No automatic confirmation or DeepSMOTE."})
    return result


def readable_report(summary):
    decisions = summary["decision"].get("candidates", {})
    conclusions = {}
    for name, values in decisions.items():
        if values["target_met"]:
            conclusions[name] = "Cleared every required screening contrast. Review both studies before a separate confirmation."
        else:
            conclusions[name] = "Did not clear every required screening contrast. This study does not justify advancing this claim."
    return {"status": summary["status"], "completed_runs": summary["completed_runs"],
            "expected_runs": summary["expected_runs"], "score_unit": "percentage points for differences",
            "conclusions": conclusions,
            "contrasts": {name: {"mean_gain_pp": 100 * values["metrics"]["balanced_accuracy"]["mean"],
                                  "sample_sd_pp": 100 * values["metrics"]["balanced_accuracy"]["sample_sd"],
                                  "per_seed_gain_pp": [100 * r["differences"]["balanced_accuracy"] for r in values["per_seed"]],
                                  "screening_target_met": values["target_met"]}
                          for name, values in summary.get("contrasts", {}).items()},
            "decision": summary["decision"],
            "limits": "Development results, not official-test results. Synthetic rows and shuffled controls are not independent replications."}
