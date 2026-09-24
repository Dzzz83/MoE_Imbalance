"""Frozen development-selection rule for the Ridge/Sinkhorn study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Select one new BA/Tail Pareto point on the selection fold.

    References must contain the selection-fold uniform-logit row. A tie with a
    fixed reference is dominated, so the result cannot claim a new point.
    """
    required = {
        "uniform_logit", "uniform_probability", "uniform_without_ce",
        "fixed_006", "fixed_007", "fixed_010", "fixed_011",
    }
    if set(references) != required:
        raise ValueError("selection references differ from the frozen set")
    baseline = references["uniform_logit"]
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        identifier = str(candidate["candidate_id"])
        if identifier in seen:
            raise ValueError(f"duplicate candidate ID: {identifier}")
        seen.add(identifier)
        metrics = candidate["metrics"]
        ba = float(metrics["balanced_accuracy"])
        tail = float(metrics["tail_accuracy"])
        joint = ba > float(baseline["balanced_accuracy"]) and tail > float(baseline["tail_accuracy"])
        dominators = sorted(
            name for name, ref in references.items()
            if float(ref["balanced_accuracy"]) >= ba and float(ref["tail_accuracy"]) >= tail
        )
        records.append({
            "candidate_id": identifier,
            "metrics": dict(metrics),
            "joint_uniform_improvement": joint,
            "dominating_references": dominators,
            "eligible": joint and not dominators and bool(candidate.get("selection_eligible", True)),
            "ot": bool(candidate.get("ot", False)),
        })
    eligible = [row for row in records if row["eligible"]]
    eligible.sort(key=lambda row: (
        -float(row["metrics"]["balanced_accuracy"]),
        -float(row["metrics"]["tail_accuracy"]),
        row["ot"],
        row["candidate_id"],
    ))
    return {
        "selected_candidate_id": eligible[0]["candidate_id"] if eligible else None,
        "eligible_candidate_ids": [row["candidate_id"] for row in eligible],
        "candidate_decisions": records,
        "selection_rule": "joint BA/Tail improvement over selection uniform; no frozen reference weakly dominates; highest BA, Tail, no OT, stable ID",
    }
