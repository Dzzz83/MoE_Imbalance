"""Fold-within-seed descriptive aggregation for inner diagnostics tables."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .complementarity import EXPERT_NAMES
from .stability import compare_profiles


METRIC_NAMES = (
    "ordinary_accuracy", "balanced_accuracy", "head_accuracy",
    "medium_accuracy", "tail_accuracy",
)


def append_metric_rows(
    rows: list[dict[str, Any]], *, evidence: str, training_seed: int,
    outer_fold_id: int, method: str, metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the same named sample and macro metrics into long-form rows."""
    for metric in METRIC_NAMES:
        rows.append({
            "scope": "outer_fold", "evidence": evidence,
            "training_seed": training_seed, "outer_fold_id": outer_fold_id,
            "method": method, "metric": metric, "value": metrics.get(metric),
            **dict(extra or {}),
        })


def aggregate_metric_rows(
    rows: list[dict[str, Any]], *, key_columns: tuple[str, ...], expected_folds: int,
) -> list[dict[str, Any]]:
    """Append fold means per seed and mean/sample-SD across three seed means.

    A seed mean exists only when every expected fold value is defined. This
    prevents missing-class rows from silently changing the estimand.
    """
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scope") == "outer_fold":
            by_key[tuple(row.get(key) for key in key_columns)].append(row)
    additions: list[dict[str, Any]] = []
    for group_rows in by_key.values():
        seed_rows: list[dict[str, Any]] = []
        for seed in sorted({int(row["training_seed"]) for row in group_rows}):
            selected = sorted(
                (row for row in group_rows if int(row["training_seed"]) == seed),
                key=lambda row: int(row["outer_fold_id"]),
            )
            values = [row.get("value") for row in selected]
            value = (
                float(np.mean(values))
                if len(selected) == expected_folds and all(item is not None for item in values)
                else None
            )
            summary = dict(selected[0])
            summary.update(scope="seed_mean", training_seed=seed, outer_fold_id=None,
                           value=value, fold_count=len(selected))
            _clear_fold_specific_metadata(summary, selected)
            additions.append(summary)
            seed_rows.append(summary)
        values = [row["value"] for row in seed_rows]
        complete = len(values) == 3 and all(value is not None for value in values)
        summary = dict(group_rows[0])
        mean = float(np.mean(values)) if complete else None
        summary.update(
            scope="mean_across_seeds", training_seed=None, outer_fold_id=None,
            value=mean, mean_across_seeds=mean,
            sample_sd_across_seeds=float(np.std(values, ddof=1)) if complete else None,
            seed_count=len(values) if complete else sum(value is not None for value in values),
        )
        _clear_fold_specific_metadata(summary, group_rows)
        additions.append(summary)
    rows.extend(additions)
    return additions


def _clear_fold_specific_metadata(row: dict[str, Any], source_rows: Sequence[Mapping[str, Any]]) -> None:
    """Prevent one fold's selection choice from masquerading as an aggregate."""
    candidate_ids = sorted({
        str(source["candidate_id"]) for source in source_rows
        if source.get("candidate_id") is not None
    })
    for name in ("configuration", "candidate_id", "maximin_delta", "sinkhorn_solver_diagnostics"):
        if name in row:
            row[name] = None
    if candidate_ids:
        row["candidate_ids_across_folds"] = candidate_ids


def _entropy_bits(profile: np.ndarray) -> float:
    positive = profile > 0.0
    return float(-np.sum(profile[positive] * np.log2(profile[positive])))


def _preferred(profile: np.ndarray | None) -> tuple[str | None, list[str] | None]:
    if profile is None:
        return None, None
    largest = float(np.max(profile))
    names = [name for name, value in zip(EXPERT_NAMES, profile)
             if np.isclose(value, largest, rtol=0.0, atol=1e-12)]
    return names[0], names


def _tie_count(names: list[str] | None) -> int | None:
    """Keep the tie count missing when the corresponding profile is undefined."""
    return None if names is None else len(names)


def add_profile_rows(
    tables: dict[str, list[dict[str, Any]]], *, pair_key: Mapping[str, int],
    router_report: Mapping[str, Any],
) -> tuple[dict[str, dict[int, np.ndarray | None]], dict[str, np.ndarray | None]]:
    """Flatten one saved router's allocation profiles and return stability arrays."""
    classes: dict[str, dict[int, np.ndarray | None]] = {}
    overall_vectors: dict[str, np.ndarray | None] = {}
    for method, method_report in router_report["methods"].items():
        overall = method_report["overall_profile"]
        vector = overall["mean_weights"]
        mean = None if vector is None else np.asarray(vector, dtype=np.float64)
        overall_vectors[method] = mean
        preferred, ties = _preferred(mean)
        for index, expert in enumerate(EXPERT_NAMES):
            tables["router_weight_profiles"].append({
                **dict(pair_key), "scope": "outer_fold", "method": method,
                "expert": expert,
                "mean_weight": None if mean is None else float(mean[index]),
                "std_weight": None if overall["std_weights"] is None else float(overall["std_weights"][index]),
                "weight_quantiles": overall["weight_quantiles"],
                "allocation_entropy_bits": overall["allocation_entropy_bits"],
                "normalized_allocation_entropy": overall["normalized_allocation_entropy"],
                "mean_sample_entropy_bits": overall["mean_sample_entropy_bits"],
                "mean_normalized_sample_entropy": overall["mean_normalized_sample_entropy"],
                "row_preferred_fraction": (
                    None if overall["preferred_fraction"] is None
                    else overall["preferred_fraction"][index]
                ),
                "row_preferred_tie_count_histogram": overall["per_row_preferred_tie_count_histogram"],
                "preferred_expert": preferred, "preferred_expert_tie_names": ties,
                "preferred_expert_tie_count": _tie_count(ties),
                "tie_policy": "canonical expert order: CE, LAL, BalancedSoftmax, Mixup",
            })
        class_profiles: dict[int, np.ndarray | None] = {}
        for profile in method_report["class_profiles"]:
            class_id = int(profile["class_id"])
            raw = profile["mean_weights"]
            values = None if raw is None else np.asarray(raw, dtype=np.float64)
            class_profiles[class_id] = values
            class_preferred, class_ties = _preferred(values)
            for index, expert in enumerate(EXPERT_NAMES):
                tables["router_class_profiles"].append({
                    **dict(pair_key), "scope": "outer_fold", "method": method,
                    "class_id": class_id, "training_count": profile["training_count"],
                    "sample_count": profile["sample_count"], "expert": expert,
                    "mean_weight": None if values is None else float(values[index]),
                    "allocation_entropy_bits": profile["allocation_entropy_bits"],
                    "normalized_allocation_entropy": profile["normalized_allocation_entropy"],
                    "mean_sample_entropy_bits": profile["mean_sample_entropy_bits"],
                    "mean_normalized_sample_entropy": profile["mean_normalized_sample_entropy"],
                    "row_preferred_fraction": (
                        None if profile["preferred_fraction"] is None
                        else profile["preferred_fraction"][index]
                    ),
                    "row_preferred_tie_count_histogram": profile["per_row_preferred_tie_count_histogram"],
                    "preferred_expert": class_preferred,
                    "preferred_expert_tie_names": class_ties,
                    "preferred_expert_tie_count": _tie_count(class_ties),
                })
        classes[method] = class_profiles
        for profile in method_report["group_profiles"]:
            sample_profile = (
                None if profile["mean_weights"] is None
                else np.asarray(profile["mean_weights"], dtype=np.float64)
            )
            macro_profile = (
                None if profile["class_macro_mean_weights"] is None
                else np.asarray(profile["class_macro_mean_weights"], dtype=np.float64)
            )
            sample_preferred, sample_ties = _preferred(sample_profile)
            macro_preferred, macro_ties = _preferred(macro_profile)
            for index, expert in enumerate(EXPERT_NAMES):
                sample_mean = profile["mean_weights"]
                macro_mean = profile["class_macro_mean_weights"]
                tables["router_group_allocations"].append({
                    **dict(pair_key), "scope": "outer_fold", "method": method,
                    "group": profile["scope_id"], "expert": expert,
                    "mean_sample_weight": None if sample_mean is None else float(sample_mean[index]),
                    "class_macro_mean_weight": None if macro_mean is None else float(macro_mean[index]),
                    "allocation_entropy_bits": profile["allocation_entropy_bits"],
                    "normalized_allocation_entropy": profile["normalized_allocation_entropy"],
                    "mean_sample_entropy_bits": profile["mean_sample_entropy_bits"],
                    "mean_normalized_sample_entropy": profile["mean_normalized_sample_entropy"],
                    "sample_count": profile["sample_count"],
                    "missing_class_ids": profile["missing_class_ids"],
                    "row_preferred_fraction": (
                        None if profile["preferred_fraction"] is None
                        else profile["preferred_fraction"][index]
                    ),
                    "row_preferred_tie_count_histogram": profile["per_row_preferred_tie_count_histogram"],
                    "preferred_expert": sample_preferred,
                    "preferred_expert_tie_names": sample_ties,
                    "preferred_expert_tie_count": _tie_count(sample_ties),
                    "class_macro_preferred_expert": macro_preferred,
                    "class_macro_preferred_expert_tie_names": macro_ties,
                    "class_macro_preferred_expert_tie_count": _tie_count(macro_ties),
                })
    return classes, overall_vectors


def compare_class_profiles(
    first: Mapping[int, np.ndarray | None], second: Mapping[int, np.ndarray | None],
) -> dict[str, Any]:
    """Compare profiles only when both carry the same complete class inventory."""
    keys = sorted(set(first) | set(second))
    missing = [key for key in keys if first.get(key) is None or second.get(key) is None]
    if missing:
        return {
            "n_classes": len(keys), "missing_class_ids": missing,
            "l1_distance": None, "l2_distance": None,
            "cosine_similarity": None, "correlation": None,
            "preferred_agreement": None, "entropy_difference_bits": None,
        }
    return {
        "n_classes": len(keys), "missing_class_ids": [],
        **compare_profiles(np.stack([first[key] for key in keys]),
                           np.stack([second[key] for key in keys])),
    }


def add_stability_rows(
    table: list[dict[str, Any]],
    profiles: Mapping[tuple[int, int], Mapping[str, Mapping[int, np.ndarray | None]]],
    selected_pairs: Sequence[tuple[int, int]],
) -> None:
    """Add separate fold-within-seed and seed-within-fold comparisons."""
    methods = sorted({name for pair in profiles.values() for name in pair})
    by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for seed in sorted({seed for seed, _ in selected_pairs}):
        folds = sorted(outer for current, outer in selected_pairs if current == seed)
        for method in methods:
            for left_pos, left in enumerate(folds):
                for right in folds[left_pos + 1:]:
                    row = {
                        "comparison_scope": "fold_within_seed", "method": method,
                        "training_seed": seed, "outer_fold_a": left,
                        "outer_fold_b": right,
                        **compare_class_profiles(profiles[(seed, left)].get(method, {}),
                                                 profiles[(seed, right)].get(method, {})),
                    }
                    table.append(row)
                    by_method_seed[(method, seed)].append(row)
    for outer in sorted({outer for _, outer in selected_pairs}):
        seeds = sorted(seed for seed, fold in selected_pairs if fold == outer)
        for method in methods:
            for left_pos, seed_a in enumerate(seeds):
                for seed_b in seeds[left_pos + 1:]:
                    table.append({
                        "comparison_scope": "seed_within_fold", "method": method,
                        "outer_fold_id": outer, "training_seed_a": seed_a,
                        "training_seed_b": seed_b,
                        **compare_class_profiles(profiles[(seed_a, outer)].get(method, {}),
                                                 profiles[(seed_b, outer)].get(method, {})),
                    })
    for method in methods:
        seed_values: list[float | None] = []
        for seed in sorted({seed for seed, _ in selected_pairs}):
            rows = by_method_seed.get((method, seed), [])
            values = [row["l1_distance"] for row in rows if row.get("l1_distance") is not None]
            complete = len(rows) == 10 and len(values) == 10
            mean = float(np.mean(values)) if complete else None
            seed_values.append(mean)
            table.append({
                "comparison_scope": "seed_mean", "method": method,
                "training_seed": seed, "mean_l1_distance": mean,
                "fold_pair_count": len(rows),
            })
        valid = [value for value in seed_values if value is not None]
        complete = len(seed_values) == 3 and len(valid) == 3
        table.append({
            "comparison_scope": "mean_across_seeds", "method": method,
            "mean_l1_distance": float(np.mean(valid)) if complete else None,
            "sample_sd_across_seeds": float(np.std(valid, ddof=1)) if complete else None,
            "seed_count": len(valid),
        })


def add_profile_aggregates(
    tables: dict[str, list[dict[str, Any]]],
    overall: Mapping[tuple[int, int], Mapping[str, np.ndarray | None]],
    classes: Mapping[tuple[int, int], Mapping[str, Mapping[int, np.ndarray | None]]],
    groups: Mapping[tuple[int, int], Mapping[str, Mapping[str, Mapping[str, Any]]]],
    selected_pairs: Sequence[tuple[int, int]],
) -> None:
    """Average each fold's profiles within seed, then summarize seed profiles."""
    seeds = sorted({seed for seed, _ in selected_pairs})
    methods = sorted({method for pair in overall.values() for method in pair})

    def emit(table: list[dict[str, Any]], *, method: str, scope: str,
             profile: np.ndarray | None, sd: np.ndarray | None,
             expert_column: str = "expert", training_seed: int | None = None,
             extra: Mapping[str, Any] | None = None) -> None:
        preferred, ties = _preferred(profile)
        entropy = None if profile is None else _entropy_bits(profile)
        for index, expert in enumerate(EXPERT_NAMES):
            table.append({
                "scope": scope, "training_seed": training_seed, "method": method,
                expert_column: expert, **dict(extra or {}),
                "mean_weight": None if profile is None else float(profile[index]),
                "mean_weight_across_seeds": None if profile is None else float(profile[index]),
                "sample_sd_across_seeds": None if sd is None else float(sd[index]),
                "mean_weights": None if profile is None else profile.tolist(),
                "allocation_entropy_bits": entropy,
                "normalized_allocation_entropy": None if entropy is None else entropy / 2.0,
                "preferred_expert": preferred, "preferred_expert_tie_names": ties,
                "preferred_expert_tie_count": _tie_count(ties),
            })

    overall_seed: dict[tuple[int, str], np.ndarray | None] = {}
    # This is the frozen study's five-fold protocol. A limited smoke run must
    # not turn one selected fold into a complete one-fold estimate.
    expected_folds = 5
    for seed in seeds:
        folds = [outer for current, outer in selected_pairs if current == seed]
        for method in methods:
            vectors = [overall[(seed, fold)].get(method) for fold in folds]
            profile = (np.mean(np.stack(vectors), axis=0)
                       if len(folds) == expected_folds and vectors and all(v is not None for v in vectors)
                       else None)
            overall_seed[(seed, method)] = profile
            emit(tables["router_weight_profiles"], method=method, scope="seed_mean",
                 profile=profile, sd=None, training_seed=seed)
    for method in methods:
        seed_profiles = [overall_seed.get((seed, method)) for seed in seeds]
        profile = (np.mean(np.stack(seed_profiles), axis=0)
                   if len(seeds) == 3 and all(v is not None for v in seed_profiles) else None)
        sd = np.std(np.stack(seed_profiles), axis=0, ddof=1) if profile is not None else None
        emit(tables["router_weight_profiles"], method=method, scope="mean_across_seeds",
             profile=profile, sd=sd)

    class_ids = sorted({class_id for pair in classes.values() for mapping in pair.values() for class_id in mapping})
    for method in methods:
        for class_id in class_ids:
            seed_profiles: list[np.ndarray | None] = []
            for seed in seeds:
                folds = [outer for current, outer in selected_pairs if current == seed]
                vectors = [classes[(seed, fold)].get(method, {}).get(class_id) for fold in folds]
                profile = (np.mean(np.stack(vectors), axis=0)
                           if len(folds) == expected_folds and vectors and all(v is not None for v in vectors)
                           else None)
                seed_profiles.append(profile)
                entropy = None if profile is None else _entropy_bits(profile)
                preferred, ties = _preferred(profile)
                for index, expert in enumerate(EXPERT_NAMES):
                    tables["router_class_profiles"].append({
                        "scope": "seed_mean", "training_seed": seed, "method": method,
                        "class_id": class_id, "expert": expert,
                        "mean_weight": None if profile is None else float(profile[index]),
                        "allocation_entropy_bits": entropy,
                        "normalized_allocation_entropy": None if entropy is None else entropy / 2.0,
                        "preferred_expert": preferred,
                        "preferred_expert_tie_names": ties,
                        "preferred_expert_tie_count": _tie_count(ties),
                    })
            mean = (np.mean(np.stack(seed_profiles), axis=0)
                    if len(seed_profiles) == 3 and all(v is not None for v in seed_profiles) else None)
            sd = np.std(np.stack(seed_profiles), axis=0, ddof=1) if mean is not None else None
            entropy = None if mean is None else _entropy_bits(mean)
            preferred, ties = _preferred(mean)
            for index, expert in enumerate(EXPERT_NAMES):
                tables["router_class_profiles"].append({
                    "scope": "mean_across_seeds", "training_seed": None,
                    "method": method, "class_id": class_id,
                    "expert": expert, "mean_weight": None if mean is None else float(mean[index]),
                    "sample_sd_across_seeds": None if sd is None else float(sd[index]),
                    "allocation_entropy_bits": entropy,
                    "normalized_allocation_entropy": None if entropy is None else entropy / 2.0,
                    "preferred_expert": preferred,
                    "preferred_expert_tie_names": ties,
                    "preferred_expert_tie_count": _tie_count(ties),
                })

    group_names = sorted({
        group for pair in groups.values() for method_groups in pair.values()
        for group in method_groups
    })
    for method in methods:
        for group in group_names:
            fold_profiles: dict[int, list[np.ndarray | None]] = defaultdict(list)
            fold_macro_profiles: dict[int, list[np.ndarray | None]] = defaultdict(list)
            for seed, outer in selected_pairs:
                record = groups[(seed, outer)][method][group]
                fold_profiles[seed].append(record["mean_weights"])
                fold_macro_profiles[seed].append(record["class_macro_mean_weights"])
            seed_profiles: list[np.ndarray | None] = []
            seed_macro_profiles: list[np.ndarray | None] = []
            for seed in seeds:
                vectors = [None if value is None else np.asarray(value, dtype=np.float64)
                           for value in fold_profiles.get(seed, [])]
                profile = (np.mean(np.stack(vectors), axis=0)
                           if len(vectors) == expected_folds and vectors
                           and all(value is not None for value in vectors) else None)
                macro_vectors = [None if value is None else np.asarray(value, dtype=np.float64)
                                 for value in fold_macro_profiles.get(seed, [])]
                macro_profile = (np.mean(np.stack(macro_vectors), axis=0)
                                 if len(macro_vectors) == expected_folds and macro_vectors
                                 and all(value is not None for value in macro_vectors) else None)
                seed_profiles.append(profile)
                seed_macro_profiles.append(macro_profile)
                entropy = None if profile is None else _entropy_bits(profile)
                macro_entropy = None if macro_profile is None else _entropy_bits(macro_profile)
                preferred, ties = _preferred(profile)
                macro_preferred, macro_ties = _preferred(macro_profile)
                for index, expert in enumerate(EXPERT_NAMES):
                    tables["router_group_allocations"].append({
                        "scope": "seed_mean", "training_seed": seed, "method": method,
                        "group": group, "expert": expert,
                        "mean_sample_weight": None if profile is None else float(profile[index]),
                        "normalized_allocation_entropy": None if entropy is None else entropy / 2.0,
                        "class_macro_mean_weight": None if macro_profile is None else float(macro_profile[index]),
                        "class_macro_normalized_allocation_entropy": (
                            None if macro_entropy is None else macro_entropy / 2.0
                        ),
                        "preferred_expert": preferred,
                        "preferred_expert_tie_names": ties,
                        "preferred_expert_tie_count": _tie_count(ties),
                        "class_macro_preferred_expert": macro_preferred,
                        "class_macro_preferred_expert_tie_names": macro_ties,
                        "class_macro_preferred_expert_tie_count": _tie_count(macro_ties),
                    })
            mean = (np.mean(np.stack(seed_profiles), axis=0)
                    if len(seed_profiles) == 3 and all(value is not None for value in seed_profiles)
                    else None)
            sd = np.std(np.stack(seed_profiles), axis=0, ddof=1) if mean is not None else None
            mean_macro = (np.mean(np.stack(seed_macro_profiles), axis=0)
                          if len(seed_macro_profiles) == 3
                          and all(value is not None for value in seed_macro_profiles) else None)
            sd_macro = np.std(np.stack(seed_macro_profiles), axis=0, ddof=1) if mean_macro is not None else None
            entropy = None if mean is None else _entropy_bits(mean)
            macro_entropy = None if mean_macro is None else _entropy_bits(mean_macro)
            preferred, ties = _preferred(mean)
            macro_preferred, macro_ties = _preferred(mean_macro)
            for index, expert in enumerate(EXPERT_NAMES):
                tables["router_group_allocations"].append({
                    "scope": "mean_across_seeds", "training_seed": None,
                    "method": method, "group": group, "expert": expert,
                    "mean_sample_weight": None if mean is None else float(mean[index]),
                    "sample_sd_across_seeds": None if sd is None else float(sd[index]),
                    "normalized_allocation_entropy": None if entropy is None else entropy / 2.0,
                    "class_macro_mean_weight": None if mean_macro is None else float(mean_macro[index]),
                    "class_macro_sample_sd_across_seeds": None if sd_macro is None else float(sd_macro[index]),
                    "class_macro_normalized_allocation_entropy": (
                        None if macro_entropy is None else macro_entropy / 2.0
                    ),
                    "preferred_expert": preferred,
                    "preferred_expert_tie_names": ties,
                    "preferred_expert_tie_count": _tie_count(ties),
                    "class_macro_preferred_expert": macro_preferred,
                    "class_macro_preferred_expert_tie_names": macro_ties,
                    "class_macro_preferred_expert_tie_count": _tie_count(macro_ties),
                })


def add_sample_entropy_aggregates(
    tables: dict[str, list[dict[str, Any]]], selected_pairs: Sequence[tuple[int, int]],
) -> None:
    """Aggregate within-row normalized entropy separately from allocation entropy."""
    for table_name, profile_keys, value_field in (
        ("router_weight_profiles", ("method",), "mean_normalized_sample_entropy"),
        ("router_class_profiles", ("method", "class_id"), "mean_normalized_sample_entropy"),
        ("router_group_allocations", ("method", "group"), "mean_normalized_sample_entropy"),
    ):
        rows = tables[table_name]
        rows_by_profile: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        base_by_profile: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            profile_key = tuple(row.get(key) for key in profile_keys)
            rows_by_profile[profile_key].append(row)
            if row.get("scope") == "outer_fold" and row.get("expert") == EXPERT_NAMES[0]:
                base_by_profile[profile_key].append(row)

        seed_ids = sorted({seed for seed, _fold in selected_pairs})
        folds_by_seed: dict[int, list[int]] = defaultdict(list)
        for seed, fold in selected_pairs:
            folds_by_seed[int(seed)].append(int(fold))

        for profile_key, base in base_by_profile.items():
            values_by_seed: dict[int, list[float]] = defaultdict(list)
            for row in base:
                if row.get(value_field) is not None:
                    values_by_seed[int(row["training_seed"])].append(float(row[value_field]))
            seed_values: dict[int, float | None] = {}
            for seed in seed_ids:
                folds = folds_by_seed[seed]
                fold_values = values_by_seed.get(seed, [])
                seed_values[seed] = (
                    float(np.mean(fold_values)) if len(fold_values) == len(folds) == 5 else None
                )
            across = [value for value in seed_values.values() if value is not None]
            aggregate_mean = float(np.mean(across)) if len(across) == 3 else None
            aggregate_sd = float(np.std(across, ddof=1)) if len(across) == 3 else None
            for row in rows_by_profile[profile_key]:
                if row.get("expert") != EXPERT_NAMES[0]:
                    continue
                if row.get("scope") == "seed_mean":
                    value = seed_values.get(int(row["training_seed"]))
                    row["mean_normalized_sample_entropy"] = value
                    row["mean_sample_entropy_bits"] = None if value is None else value * 2.0
                elif row.get("scope") == "mean_across_seeds":
                    row["mean_normalized_sample_entropy"] = aggregate_mean
                    row["sample_sd_normalized_sample_entropy_across_seeds"] = aggregate_sd
                    row["mean_sample_entropy_bits"] = None if aggregate_mean is None else aggregate_mean * 2.0
                    row["sample_sd_mean_sample_entropy_bits_across_seeds"] = (
                        None if aggregate_sd is None else aggregate_sd * 2.0
                    )

        # Carry the shared profile entropy to each expert with one indexed
        # lookup. Include fold identity so a row cannot inherit another fold's
        # value when the saved sample entropies differ.
        entropy_fields = (
            "mean_normalized_sample_entropy", "mean_sample_entropy_bits",
            "sample_sd_normalized_sample_entropy_across_seeds",
            "sample_sd_mean_sample_entropy_bits_across_seeds",
        )
        ce_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
        for profile_key in base_by_profile:
            for row in rows_by_profile[profile_key]:
                if row.get("expert") == EXPERT_NAMES[0]:
                    identity = (
                        profile_key, row.get("scope"), row.get("training_seed"),
                        row.get("outer_fold_id"),
                    )
                    ce_by_identity.setdefault(identity, row)
        for row in rows:
            if row.get("expert") == EXPERT_NAMES[0]:
                continue
            profile_key = tuple(row.get(key) for key in profile_keys)
            if profile_key not in base_by_profile:
                continue
            identity = (
                profile_key, row.get("scope"), row.get("training_seed"),
                row.get("outer_fold_id"),
            )
            matching = ce_by_identity.get(identity)
            if matching is not None:
                for field in entropy_fields:
                    row[field] = matching.get(field)


def aggregate_pair_tables(
    tables: dict[str, list[dict[str, Any]]], selected_pairs: Sequence[tuple[int, int]],
) -> None:
    """Aggregate separate evidence tables and per-class specialization."""
    # Smoke limits do not change the study's five-fold aggregation contract.
    expected_folds = 5
    for name in ("expert_oof_metrics", "router_fit_set_metrics", "selection_cv_metrics"):
        additions = aggregate_metric_rows(
            tables[name], key_columns=("evidence", "method", "metric"),
            expected_folds=expected_folds,
        )
        for row in additions:
            tables["seed_aggregates"].append({key: row.get(key) for key in (
                "scope", "evidence", "training_seed", "method", "metric", "value",
                "mean_across_seeds", "sample_sd_across_seeds", "seed_count", "fold_count",
            )})

    rows = tables["per_class_specialization"]
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scope") == "outer_fold":
            grouped[(int(row["training_seed"]), str(row["expert"]), int(row["class_id"]))].append(row)
    seeds = sorted({key[0] for key in grouped})
    classes = sorted({key[2] for key in grouped})
    for seed in seeds:
        for expert in EXPERT_NAMES:
            for class_id in classes:
                fold_rows = grouped.get((seed, expert, class_id), [])
                values = [row.get("class_recall") for row in fold_rows]
                complete = len(fold_rows) == expected_folds and all(v is not None for v in values)
                if fold_rows:
                    summary = dict(fold_rows[0])
                    summary.update(scope="seed_mean", training_seed=seed, outer_fold_id=None,
                                   class_recall=float(np.mean(values)) if complete else None)
                    for field in (
                        "sample_count", "any_correct_fraction", "all_wrong_fraction",
                        "exclusive_correct_fraction", "strongest_experts",
                        "strongest_expert_recall", "best_second_recall_gap",
                        "pairwise_prediction_disagreement_fraction",
                        "sample_with_any_prediction_disagreement_fraction",
                    ):
                        if field in summary:
                            fold_values = [row.get(field) for row in fold_rows]
                            if field in {
                                "any_correct_fraction", "all_wrong_fraction",
                                "exclusive_correct_fraction",
                                "pairwise_prediction_disagreement_fraction",
                                "sample_with_any_prediction_disagreement_fraction",
                            } and all(value is not None for value in fold_values):
                                if isinstance(fold_values[0], Mapping):
                                    summary[field] = {
                                        key: float(np.mean([value[key] for value in fold_values]))
                                        for key in fold_values[0]
                                    }
                                else:
                                    summary[field] = float(np.mean(fold_values))
                            else:
                                summary[field] = None
                    rows.append(summary)
    for expert in EXPERT_NAMES:
        for class_id in classes:
            selected = [row for row in rows if row.get("scope") == "seed_mean"
                        and row.get("expert") == expert and row.get("class_id") == class_id]
            values = [row["class_recall"] for row in selected if row.get("class_recall") is not None]
            complete = len(selected) == 3 and len(values) == 3
            rows.append({
                "scope": "mean_across_seeds", "expert": expert, "class_id": class_id,
                "class_recall": float(np.mean(values)) if complete else None,
                "sample_sd_across_seeds": float(np.std(values, ddof=1)) if complete else None,
            })
    # Recompute winner/tie/gap from the aggregate class recalls, rather than
    # inheriting the first fold's specialization annotations.
    for scope in ("seed_mean", "mean_across_seeds"):
        keys = sorted({
            (row.get("training_seed"), int(row["class_id"]))
            for row in rows if row.get("scope") == scope
        }, key=lambda item: (-1 if item[0] is None else int(item[0]), item[1]))
        for seed, class_id in keys:
            candidates = [row for row in rows if row.get("scope") == scope
                          and row.get("training_seed") == seed
                          and int(row["class_id"]) == class_id]
            recalls = [row.get("class_recall") for row in candidates]
            defined = len(candidates) == len(EXPERT_NAMES) and all(value is not None for value in recalls)
            if defined:
                ranked = sorted(float(value) for value in recalls)
                best = ranked[-1]
                winners = [str(row["expert"]) for row in candidates
                           if np.isclose(float(row["class_recall"]), best, rtol=0.0, atol=1e-12)]
                gap = best - ranked[-2]
            else:
                winners, best, gap = None, None, None
            for row in candidates:
                row["strongest_experts"] = winners
                row["strongest_expert_recall"] = best
                row["best_second_recall_gap"] = gap

    for table_name, keys, field in (
        ("sinkhorn_adjustments", ("method", "stage", "expert"), "mean_weight"),
        ("prior_diagnostics", ("method", "diagnostic"), "l1_distance"),
    ):
        candidates = [row for row in tables[table_name] if row.get("scope") == "outer_fold" and row.get(field) is not None]
        temporary = [dict(row, value=row[field]) for row in candidates]
        additions = aggregate_metric_rows(temporary, key_columns=keys, expected_folds=expected_folds)
        for row in additions:
            result = dict(row)
            result[field] = result.pop("value")
            result["mean_weight_across_seeds"] = result.get("mean_across_seeds")
            tables[table_name].append(result)

    # Preserve rates as fold means, then summarize seed means. Counts remain
    # available in their original fold rows and are not pooled across the
    # overlapping development populations.
    _aggregate_scalar_field(
        tables["complementarity"], key_columns=("kind", "num_correct_experts"),
        field="fraction", expected_folds=expected_folds,
        row_filter=lambda row: row.get("kind") == "correct_count_distribution",
        aliases={"mean_fraction_across_seeds": "mean_across_seeds"},
    )
    _aggregate_scalar_field(
        tables["complementarity"],
        key_columns=("kind", "correct_expert_bitmask"), field="fraction",
        expected_folds=expected_folds,
        row_filter=lambda row: row.get("kind") == "correctness_pattern",
        aliases={"mean_fraction_across_seeds": "mean_across_seeds"},
    )
    _aggregate_scalar_fields(
        tables["complementarity"], key_columns=("kind",),
        fields=("any_expert_correct_fraction", "all_experts_wrong_fraction"),
        expected_folds=expected_folds,
        row_filter=lambda row: row.get("kind") == "correctness_summary",
    )
    _aggregate_scalar_fields(
        tables["complementarity"], key_columns=("kind", "pair"), fields=(
        "both_correct_fraction", "a_only_correct_fraction", "b_only_correct_fraction",
        "both_wrong_fraction", "prediction_agreement_fraction",
        "prediction_disagreement_fraction",),
        expected_folds=expected_folds,
        row_filter=lambda row: row.get("kind") == "pairwise_correctness",
    )
    _aggregate_scalar_field(
        tables["complementarity"], key_columns=("kind", "matrix", "expert_a", "expert_b"),
        field="value", expected_folds=expected_folds,
        row_filter=lambda row: row.get("kind") == "expert_matrix"
        and row.get("matrix", "").endswith("fractions"),
    )
    _aggregate_scalar_fields(
        tables["oracle_opportunity"], key_columns=("metric",),
        fields=("oracle", "strongest_individual", "oracle_minus_strongest_individual"),
        expected_folds=expected_folds,
    )
    _aggregate_scalar_field(
        tables["frequency_associations"],
        key_columns=("source", "method", "expert", "frequency_variable", "outcome"),
        field="spearman_rho", expected_folds=expected_folds,
    )
    # Flatten canonical H/M/T expert macro recalls to make the group
    # denominator visible for descriptive seed summaries.
    group_rows = [row for row in tables["complementarity"]
                  if row.get("kind") == "canonical_group" and row.get("scope") == "outer_fold"]
    _aggregate_scalar_fields(
        tables["complementarity"], key_columns=("kind", "group"),
        fields=("any_correct_macro_recall", "all_wrong_macro_recall"),
        expected_folds=expected_folds,
        row_filter=lambda row: row.get("kind") == "canonical_group",
        output_kind="canonical_group_aggregate",
    )
    for map_field in ("expert_macro_recall", "exclusive_correct_macro_recall"):
        expanded = []
        for row in group_rows:
            for expert in EXPERT_NAMES:
                values = row[map_field]
                expanded.append({
                    **{key: row[key] for key in (
                        "scope", "training_seed", "outer_fold_id", "kind", "group",
                    )},
                    "expert": expert,
                    map_field: None if values is None else values[expert],
                })
        _aggregate_scalar_fields(
            tables["complementarity"], key_columns=("kind", "group", "expert"),
            fields=(map_field,), expected_folds=expected_folds,
            selected_records=expanded,
            output_kind="canonical_group_aggregate",
        )


def _aggregate_scalar_field(
    rows: list[dict[str, Any]], *, key_columns: tuple[str, ...], field: str,
    expected_folds: int, row_filter: Any = None,
    aliases: Mapping[str, str] | None = None,
) -> None:
    """Append clean summaries for one scalar without copying unrelated fold data."""
    _aggregate_scalar_fields(
        rows, key_columns=key_columns, fields=(field,), expected_folds=expected_folds,
        row_filter=row_filter, aliases=aliases,
    )


def _aggregate_scalar_fields(
    rows: list[dict[str, Any]], *, key_columns: tuple[str, ...], fields: tuple[str, ...],
    expected_folds: int, row_filter: Any = None,
    aliases: Mapping[str, str] | None = None,
    selected_records: Sequence[Mapping[str, Any]] | None = None,
    output_kind: str | None = None,
) -> None:
    """Append one minimal summary row per key, carrying all named metrics."""
    selected = [row for row in list(rows)
                if row.get("scope") == "outer_fold" and (row_filter is None or row_filter(row))] \
        if selected_records is None else list(selected_records)
    for summary in _summarize_scalar_records(
        selected, key_columns=key_columns, fields=fields, expected_folds=expected_folds,
    ):
        if output_kind is not None:
            summary["kind"] = output_kind
        for alias, source in (aliases or {}).items():
            summary[alias] = summary.get(source)
        rows.append(summary)


def _summarize_scalar_records(
    records: Sequence[Mapping[str, Any]], *, key_columns: tuple[str, ...],
    fields: tuple[str, ...], expected_folds: int,
) -> list[dict[str, Any]]:
    """Build minimal nested means, retaining only identity and stated metrics."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key) for key in key_columns)].append(record)
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        key_payload = dict(zip(key_columns, key))
        seed_values: dict[str, list[float | None]] = {field: [] for field in fields}
        for seed in sorted({int(row["training_seed"]) for row in group}):
            fold_rows = [row for row in group if int(row["training_seed"]) == seed]
            payload: dict[str, Any] = {
                **key_payload, "scope": "seed_mean", "training_seed": seed,
                "fold_count": len(fold_rows),
            }
            for field in fields:
                values = [row.get(field) for row in fold_rows]
                complete = len(fold_rows) == expected_folds and all(value is not None for value in values)
                mean = float(np.mean(values)) if complete else None
                payload[field] = mean
                seed_values[field].append(mean)
            output.append(payload)
        payload = {
            **key_payload, "scope": "mean_across_seeds", "training_seed": None,
            "fold_count": len(group),
        }
        for field in fields:
            values = seed_values[field]
            valid = [value for value in values if value is not None]
            complete = len(values) == 3 and len(valid) == 3
            mean = float(np.mean(valid)) if complete else None
            sd = float(np.std(valid, ddof=1)) if complete else None
            payload[field] = mean
            payload[f"{field}_mean_across_seeds"] = mean
            payload[f"{field}_sample_sd_across_seeds"] = sd
            if len(fields) == 1:
                payload["value"] = mean
                payload["mean_across_seeds"] = mean
                payload["sample_sd_across_seeds"] = sd
            payload["seed_count"] = len(valid)
        output.append(payload)
    return output


def descriptive_metrics(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for table_name in ("expert_oof_metrics", "selection_cv_metrics", "router_fit_set_metrics"):
        for row in tables[table_name]:
            if row.get("scope") == "mean_across_seeds":
                result.append({key: row.get(key) for key in (
                    "evidence", "method", "metric", "mean_across_seeds", "sample_sd_across_seeds",
                )})
    return result


def diagnostic_highlights(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Return complete-matrix means with folds averaged before the seeds.

    Highlights are intentionally null for a limited smoke run. This avoids
    presenting a one-pair execution as if it summarized the full study.
    """
    lock_rows = [row for row in tables["expert_oof_metrics"] if row.get("scope") == "outer_fold"]
    observed_pairs = {
        (int(row["training_seed"]), int(row["outer_fold_id"])) for row in lock_rows
    }
    pairs = set(expected_pairs)
    seeds = sorted({seed for seed, _fold in pairs})
    folds_by_seed = {
        seed: sorted(fold for current_seed, fold in pairs if current_seed == seed)
        for seed in seeds
    }
    complete_matrix = len(pairs) == 15 and observed_pairs == pairs

    def nested_mean(
        rows: Sequence[Mapping[str, Any]], field: str,
    ) -> float | None:
        by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            value = row[field]
            if value is None:
                return None
            by_pair[(int(row["training_seed"]), int(row["outer_fold_id"]))].append(float(value))
        if set(by_pair) != pairs or not pairs:
            return None
        fold_means = {pair: float(np.mean(values)) for pair, values in by_pair.items() if values}
        if set(fold_means) != pairs:
            return None
        seed_means = []
        for seed in seeds:
            expected_folds = folds_by_seed[seed]
            values = [fold_means[(seed, fold)] for fold in expected_folds
                      if (seed, fold) in fold_means]
            if len(values) != len(expected_folds):
                return None
            seed_means.append(float(np.mean(values)))
        return float(np.mean(seed_means))

    if not complete_matrix:
        return {
            "complete_matrix": False,
            "any_expert_correct_fraction": None,
            "all_experts_wrong_fraction": None,
            "hard_oracle_balanced_accuracy": None,
            "hard_oracle_ba_gap": None,
            "mean_class_disagreement": None,
            "mean_normalized_router_entropy": None,
            "fold_profile_mean_l1": None,
            "fold_profile_mean_l1_by_method": {},
            "mean_l1_from_prior_only": None,
            "mean_maximum_abs_sinkhorn_marginal_residual": None,
            "mean_selective_gate": None,
        }

    complement = tables["complementarity"]
    correctness = [row for row in complement if row["kind"] == "correctness_summary"
                   and row["scope"] == "outer_fold"]
    oracle = [row for row in tables["oracle_opportunity"]
              if row["metric"] == "balanced_accuracy" and row["scope"] == "outer_fold"]
    class_disagreement = [
        row for row in tables["per_class_specialization"]
        if row["scope"] == "outer_fold" and row["expert"] == EXPERT_NAMES[0]
    ]
    entropy_rows = [
        row for row in tables["router_weight_profiles"]
        if row["scope"] == "outer_fold" and row["expert"] == EXPERT_NAMES[0]
    ]
    prior_rows = [
        row for row in tables["prior_diagnostics"]
        if row["diagnostic"] == "adaptive_mean_vs_selected_prior_only"
        and row["scope"] == "outer_fold"
    ]
    sinkhorn_rows = [
        row for row in tables["prior_diagnostics"]
        if row["diagnostic"] == "adaptive_mean_vs_sinkhorn_target_prior"
        and row["scope"] == "outer_fold"
    ]
    gate_rows = [
        row for row in tables["sinkhorn_adjustments"]
        if row["stage"] == "selective_gate" and row["group"] == "overall"
        and row["scope"] == "outer_fold"
    ]
    stability_by_method = {
        str(row["method"]): row["mean_l1_distance"]
        for row in tables["stability"] if row["comparison_scope"] == "mean_across_seeds"
    }
    defined_stability = [float(value) for value in stability_by_method.values() if value is not None]
    return {
        "complete_matrix": True,
        "any_expert_correct_fraction": nested_mean(correctness, "any_expert_correct_fraction"),
        "all_experts_wrong_fraction": nested_mean(correctness, "all_experts_wrong_fraction"),
        "hard_oracle_balanced_accuracy": nested_mean(oracle, "oracle"),
        "hard_oracle_ba_gap": nested_mean(oracle, "oracle_minus_strongest_individual"),
        "mean_class_disagreement": nested_mean(class_disagreement, "pairwise_prediction_disagreement_fraction"),
        "mean_normalized_router_entropy": nested_mean(entropy_rows, "normalized_allocation_entropy"),
        "fold_profile_mean_l1": (
            float(np.mean(defined_stability))
            if stability_by_method and len(defined_stability) == len(stability_by_method) else None
        ),
        "fold_profile_mean_l1_by_method": stability_by_method,
        "mean_l1_from_prior_only": nested_mean(prior_rows, "l1_distance"),
        "mean_maximum_abs_sinkhorn_marginal_residual": nested_mean(
            sinkhorn_rows, "max_abs_marginal_residual",
        ),
        "mean_selective_gate": nested_mean(gate_rows, "gate_mean"),
    }
