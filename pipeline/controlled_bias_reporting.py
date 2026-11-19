"""Prespecified gates and paired descriptive reports for three feature seeds."""

import math
import statistics

GATE_ONE = ("unbiased_cb", "biased_cb", "known_correction")
GATE_TWO = ("equal_target_ot", "reference_target_ot", "reference_cluster_ratio",
            "shuffled_reference_ot_shuffle0", "shuffled_reference_ot_shuffle1", "shuffled_reference_ot_shuffle2")
METRICS = ("diagnostic_balanced_accuracy", "accuracy", "cross_entropy", "class_macro_cross_entropy")
CONDITION_LABELS = {
    "unbiased_cb": "Before removal, class balancing",
    "biased_cb": "After removal, class balancing",
    "known_correction": "Known sampling correction",
    "equal_target_ot": "OT with equal prototype targets",
    "reference_target_ot": "OT with reference targets",
    "reference_cluster_ratio": "Simple reference correction",
    **{f"shuffled_reference_ot_shuffle{s}": f"Reference OT shuffled within class (shuffle {s})" for s in (0, 1, 2)},
}
OUTCOME_EXPLANATIONS = {
    "insufficient_damage": "The introduced removal did not cause the required consistent 1-point accuracy loss.",
    "insufficient_recovery": "Known correction did not meet the required recovery; stop before testing OT.",
    "technical_failure": "A technical failure prevents a valid decision; inspect the saved error and convergence records.",
    "incomplete": "The required fits are not all complete yet.",
    "ot_capability_merits_separate_natural_data_followup": "Reference OT cleared the usefulness, simple-rule and shuffle targets. Design a separate natural-data follow-up before expanding.",
    "target_information_helps_transport_value_unestablished": "Reference targets help versus equal targets, but an additional benefit from transport is not established.",
    "prefer_investigating_simple_correction": "The simple correction is useful; OT has not shown the required extra benefit over it.",
    "known_correction_helps_geometry_not_established": "Known correction helps, but the geometry methods did not clear the usefulness target. Inspect geometry and target diagnostics before proposing another experiment.",
    "unresolved": "The comparisons do not support a clear next direction. Stop this bounded experiment and discuss the findings.",
}


def _at_least(value, threshold):
    # Inclusive prespecified boundaries should not flip on subtraction roundoff.
    return value >= threshold or math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)


def index_records(records):
    indexed = {}
    for row in records:
        key = row["seed"], row["condition"]
        if key in indexed or row["seed"] not in (0, 1, 2) or row["condition"] not in GATE_ONE + GATE_TWO:
            raise ValueError("Unexpected or duplicate controlled-bias result")
        if (not row["optimization"]["is_converged"] or any(
                type(row["metrics"].get(m)) not in (int, float) or not math.isfinite(row["metrics"][m]) for m in METRICS)):
            raise ValueError("A result has invalid optimization or development metrics")
        indexed[key] = row
    return indexed


def paired_contrast(records, condition, reference, gain):
    indexed = index_records(records)
    result = {}
    for metric in METRICS:
        paired = []
        for seed in (0, 1, 2):
            def rows(name):
                names = [f"shuffled_reference_ot_shuffle{s}" for s in (0, 1, 2)] if name == "shuffled_reference_ot" else [name]
                return [indexed[seed, n]["metrics"][metric] for n in names]
            values, reference_values = rows(condition), rows(reference)
            current, previous = statistics.mean(values), statistics.mean(reference_values)
            paired.append({"seed": seed, "value": current, "reference": previous,
                           "individual_values": values, "reference_individual_values": reference_values,
                           "shuffle_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                           "reference_shuffle_sample_sd": statistics.stdev(reference_values) if len(reference_values) > 1 else None,
                           "difference": current - previous})
        differences = [p["difference"] for p in paired]
        result[metric] = {"per_seed": paired, "mean_difference": statistics.mean(differences),
                          "sample_sd_difference": statistics.stdev(differences)}
    primary = result["diagnostic_balanced_accuracy"]
    result["is_promising"] = (_at_least(primary["mean_difference"], gain)
                              and sum(p["difference"] > 0 for p in primary["per_seed"]) >= 2)
    return result


def gate_one(records, config, *, technical_failure=False):
    indexed = index_records(records)
    expected = {(s, c) for s in (0, 1, 2) for c in GATE_ONE}
    if not expected <= indexed.keys():
        return {"status": "technical_failure" if technical_failure else "incomplete", "passed": False,
                "reasons": ["technical_failure"] if technical_failure else [], "completed_fits": len(expected & indexed.keys()),
                "expected_fits": 9, "thresholds": config["gate"]}
    damage = paired_contrast(records, "unbiased_cb", "biased_cb", config["practical_gain"])["diagnostic_balanced_accuracy"]
    recovery = paired_contrast(records, "known_correction", "biased_cb", config["practical_gain"])["diagnostic_balanced_accuracy"]
    mean_damage, mean_recovery = damage["mean_difference"], recovery["mean_difference"]
    fraction = mean_recovery / mean_damage if mean_damage > 0 else None
    thresholds = config["gate"]
    reasons = []
    if (not _at_least(mean_damage, thresholds["minimum_damage"]) or sum(p["difference"] > 0 for p in damage["per_seed"]) < thresholds["minimum_positive_seeds"]):
        reasons.append("insufficient_damage")
    if (not _at_least(mean_recovery, thresholds["minimum_recovery"]) or fraction is None
            or not _at_least(fraction, thresholds["minimum_recovery_fraction"])
            or sum(p["difference"] > 0 for p in recovery["per_seed"]) < thresholds["minimum_positive_seeds"]):
        reasons.append("insufficient_recovery")
    if technical_failure:
        reasons.append("technical_failure")
    return {"status": "passed" if not reasons else "failed", "passed": not reasons, "reasons": reasons,
            "completed_fits": 9, "expected_fits": 9, "damage": damage, "recovery": recovery,
            "recovery_fraction": fraction, "thresholds": thresholds}


def contributions(records, condition, reference):
    indexed = index_records(records)
    output = []
    for seed in (0, 1, 2):
        def metrics(name):
            names = [f"shuffled_reference_ot_shuffle{s}" for s in (0, 1, 2)] if name == "shuffled_reference_ot" else [name]
            return [indexed[seed, n]["metrics"] for n in names]
        current, previous = metrics(condition), metrics(reference)
        class_count = len(current[0]["classes"])
        classes = []
        for c, row in enumerate(current[0]["classes"]):
            difference = statistics.mean(m["classes"][c]["recall"] for m in current) - statistics.mean(m["classes"][c]["recall"] for m in previous)
            classes.append({"original_class": row["original_class"], "support": row["support"],
                            "recall_difference": difference, "balanced_accuracy_contribution": difference / class_count})
        group_changes = []
        for g, row in enumerate(current[0]["controller_groups"]):
            change = statistics.mean(m["controller_groups"][g]["balanced_accuracy_contribution"] for m in current) - statistics.mean(m["controller_groups"][g]["balanced_accuracy_contribution"] for m in previous)
            group_changes.append({"class_index": row["class_index"], "controller_group": row["controller_group"],
                                  "role": row["role"], "support": row["support"], "balanced_accuracy_contribution": change})
        output.append({"seed": seed, "classes": classes, "controller_groups": group_changes,
                       "largest_class_contributors": sorted(classes, key=lambda r: abs(r["balanced_accuracy_contribution"]), reverse=True)[:5]})
    return output


def summarize(records, config, *, technical_failure=False):
    indexed = index_records(records)
    gate_one_runs = {(s, c) for s in (0, 1, 2) for c in GATE_ONE}
    # A later Gate-2 failure does not invalidate the nine verified Gate-1 fits.
    gate = gate_one(records, config, technical_failure=technical_failure and not gate_one_runs <= indexed.keys())
    base = {"completed_runs": len(records), "maximum_runs": 27, "gate_one": gate,
            "interpretation": "Exploratory budget criteria, not significance/equivalence tests. Three representations share images and masks. Fixed-backbone classifier diagnostic, not a universal OT verdict."}
    if gate["completed_fits"] == 9:
        base["conditions"] = {}
        for condition in GATE_ONE + GATE_TWO:
            per_seed = [{"seed": s, **indexed[s, condition]["metrics"]} for s in (0, 1, 2) if (s, condition) in indexed]
            if len(per_seed) == 3:
                base["conditions"][condition] = {
                    metric: {"per_seed": [{"seed": r["seed"], "value": r[metric]} for r in per_seed],
                             "mean": statistics.mean(r[metric] for r in per_seed),
                             "sample_sd": statistics.stdev(r[metric] for r in per_seed)} for metric in METRICS}
        gate_pairs = [("unbiased_cb", "biased_cb"), ("known_correction", "biased_cb")]
        base["gate_one_contrasts"] = {
            f"{c}_minus_{r}": paired_contrast(records, c, r, config["practical_gain"]) for c, r in gate_pairs}
        base["gate_one_contributions"] = {
            f"{c}_minus_{r}": contributions(records, c, r) for c, r in gate_pairs}
    if not gate["passed"]:
        base.update(status="failed" if technical_failure else "stopped_gate_one" if gate["completed_fits"] == 9 else "incomplete",
                    expected_runs=9, decision={"outcomes": gate["reasons"], "next_stage": "Stop; no automatic redesign or sweep."})
        return base
    expected = {(s, c) for s in (0, 1, 2) for c in GATE_ONE + GATE_TWO}
    if technical_failure or indexed.keys() != expected:
        base.update(status="failed" if technical_failure else "incomplete", expected_runs=27,
                    decision={"outcomes": ["technical_failure"] if technical_failure else ["incomplete"]})
        return base
    pairs = [("reference_target_ot", "reference_cluster_ratio"), ("reference_target_ot", "shuffled_reference_ot"),
             ("reference_target_ot", "equal_target_ot"), ("shuffled_reference_ot", "biased_cb")]
    pairs += [(c, reference) for c in GATE_TWO[:3] for reference in ("biased_cb", "known_correction")]
    contrasts = {f"{c}_minus_{r}": paired_contrast(records, c, r, config["practical_gain"]) for c, r in pairs}
    def promising(c, r):
        return contrasts[f"{c}_minus_{r}"]["is_promising"]
    outcomes = []
    if all(promising("reference_target_ot", r) for r in ("biased_cb", "reference_cluster_ratio", "shuffled_reference_ot")):
        outcomes.append("ot_capability_merits_separate_natural_data_followup")
    if promising("reference_target_ot", "equal_target_ot") and not all(promising("reference_target_ot", r) for r in ("reference_cluster_ratio", "shuffled_reference_ot")):
        outcomes.append("target_information_helps_transport_value_unestablished")
    if promising("reference_cluster_ratio", "biased_cb") and not promising("reference_target_ot", "reference_cluster_ratio"):
        outcomes.append("prefer_investigating_simple_correction")
    if not any(promising(c, "biased_cb") for c in GATE_TWO[:3]):
        outcomes.append("known_correction_helps_geometry_not_established")
    if not outcomes:
        outcomes.append("unresolved")
    base.update(status="complete", expected_runs=27, contrasts=contrasts,
                contributions={f"{c}_minus_{r}": contributions(records, c, r) for c, r in pairs},
                decision={"outcomes": outcomes, "next_stage": "Review results before any new intervention, natural-data or backbone experiment.", "practical_gain": config["practical_gain"]})
    return base


def readable_report(summary, records):
    lines = ["Controlled-bias diagnostic", f"Status: {summary['status']}; completed fits: {len(records)} (maximum 27).",
             "Scores concern 34 eligible classes with fresh heads, not historical 100-class continuations."]
    gate = summary["gate_one"]
    lines.append(f"Gate 1: {gate['status']}; reasons: {', '.join(gate['reasons']) or 'none'}.")
    if "damage" in gate:
        lines.append(f"Mean removal damage: {gate['damage']['mean_difference'] * 100:.3f} percentage points; known-correction recovery: {gate['recovery']['mean_difference'] * 100:.3f} points.")
    indexed = index_records(records)
    for condition in GATE_ONE + GATE_TWO:
        values = [indexed[s, condition]["metrics"]["diagnostic_balanced_accuracy"] for s in (0, 1, 2) if (s, condition) in indexed]
        if values:
            lines.append(f"{CONDITION_LABELS[condition]}: " + ", ".join(f"{v * 100:.3f}%" for v in values) + f"; mean {statistics.mean(values) * 100:.3f}%.")
    for name, contrast in summary.get("contrasts", {}).items():
        primary = contrast["diagnostic_balanced_accuracy"]
        lines.append(f"{name}: mean {primary['mean_difference'] * 100:+.3f} points; paired SD {primary['sample_sd_difference'] * 100:.3f}; promising={contrast['is_promising']}.")
    lines += [OUTCOME_EXPLANATIONS[outcome] for outcome in summary["decision"]["outcomes"]]
    lines += [summary["interpretation"]]
    return {"lines": lines}
