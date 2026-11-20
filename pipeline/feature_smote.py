"""Deterministic, real-parent interpolation and matched anchor-copy controls."""

import torch
import torch.nn.functional as functional


def build_banks(features, labels):
    if (features.device.type != "cpu" or features.ndim != 2 or len(features) != len(labels)
            or not features.is_floating_point() or not torch.isfinite(features).all()
            or labels.dtype != torch.int64 or labels.ndim != 1 or (labels < 0).any()):
        raise ValueError("Expected finite CPU fitting features and integer labels")
    counts = torch.bincount(labels)
    if len(counts) != 100 or (counts < 2).any():
        raise ValueError("SMOTE requires at least two real rows in each of 100 classes")
    anchor_rng = torch.Generator().manual_seed(700)
    neighbor_rng = torch.Generator().manual_seed(701)
    lambda_rng = torch.Generator().manual_seed(702)
    anchors, neighbors, coefficients, added_labels, distances = [], [], [], [], []
    for c, count in enumerate(counts.tolist()):
        added_count = (min(100, 4 * count) if count < 100 else count) - count
        if not added_count:
            continue
        positions = torch.where(labels == c)[0]
        normalized = functional.normalize(features[positions], dim=1)
        cost = (1 - normalized @ normalized.T).clamp_min(0)
        cost.fill_diagonal_(torch.inf)
        # Stable sort resolves equal distances by the ascending cache position.
        nearest = cost.argsort(dim=1, stable=True)[:, :min(5, count - 1)]
        anchor = torch.randint(count, (added_count,), generator=anchor_rng)
        ranks = torch.randint(nearest.shape[1], (added_count,), generator=neighbor_rng)
        neighbor = nearest[anchor, ranks]
        anchors.append(positions[anchor])
        neighbors.append(positions[neighbor])
        coefficients.append(torch.rand(added_count, generator=lambda_rng, dtype=torch.float64))
        added_labels.append(torch.full((added_count,), c, dtype=torch.int64))
        distances.append(cost[anchor, neighbor])
    anchor = torch.cat(anchors) if anchors else torch.empty(0, dtype=torch.int64)
    neighbor = torch.cat(neighbors) if neighbors else torch.empty(0, dtype=torch.int64)
    coefficient = torch.cat(coefficients) if coefficients else torch.empty(0, dtype=torch.float64)
    new_labels = torch.cat(added_labels) if added_labels else torch.empty(0, dtype=torch.int64)
    # Interpolate raw classifier coordinates, then retain the cache's feature dtype.
    added = ((1 - coefficient[:, None]) * features[anchor].double()
             + coefficient[:, None] * features[neighbor].double()).to(features.dtype)
    common = {"labels": torch.cat((labels, new_labels)),
              "is_added": torch.cat((torch.zeros(len(labels), dtype=torch.bool),
                                     torch.ones(len(anchor), dtype=torch.bool))),
              "anchor_positions": anchor, "neighbor_positions": neighbor,
              "neighbor_cosine_distance": torch.cat(distances) if distances else features.new_empty(0)}
    banks = {
        "smote": common | {"features": torch.cat((features, added)), "lambda": coefficient},
        "duplicate": common | {"features": torch.cat((features, features[anchor])),
                                "lambda": torch.zeros_like(coefficient)},
    }
    for name, bank in banks.items():
        validate_bank(bank, features, labels, name)
    return banks


def validate_bank(bank, features, labels, kind):
    """Check provenance and raw-coordinate identities before a head sees a bank."""
    count = len(labels)
    anchor, neighbor, coefficient = (bank[k] for k in ("anchor_positions", "neighbor_positions", "lambda"))
    if (anchor.dtype != torch.int64 or neighbor.dtype != torch.int64
            or anchor.shape != neighbor.shape or coefficient.shape != anchor.shape
            or (anchor < 0).any() or (anchor >= count).any()
            or (neighbor < 0).any() or (neighbor >= count).any()
            or (anchor == neighbor).any() or not torch.equal(labels[anchor], labels[neighbor])
            or not torch.isfinite(coefficient).all() or (coefficient < 0).any() or (coefficient >= 1).any()):
        raise ValueError("Invalid same-class real-parent provenance")
    expected_labels = torch.cat((labels, labels[anchor]))
    expected_origin = torch.arange(len(expected_labels)) >= count
    if (not torch.equal(bank["labels"], expected_labels)
            or not torch.equal(bank["is_added"], expected_origin)
            or not torch.equal(bank["features"][:count], features)):
        raise ValueError("Bank changed real rows, labels or origins")
    real_counts = torch.bincount(labels, minlength=100)
    expected_counts = torch.where(real_counts >= 100, real_counts, (4 * real_counts).clamp_max(100))
    if not torch.equal(torch.bincount(bank["labels"], minlength=100), expected_counts):
        raise ValueError("Bank does not follow the fixed count policy")
    if kind == "duplicate" and torch.count_nonzero(coefficient):
        raise ValueError("Duplicate coefficients must be zero")
    expected = ((1 - coefficient[:, None]) * features[anchor].double()
                + coefficient[:, None] * features[neighbor].double()).to(features.dtype)
    if not torch.equal(bank["features"][count:], expected):
        raise ValueError("Bank does not match its raw-coordinate interpolation")


def bank_diagnostics(bank, real_features):
    records = []
    count = len(real_features)
    added_features = bank["features"][count:]
    for c in range(100):
        real = bank["labels"][:count] == c
        added = bank["labels"][count:] == c
        values = added_features[added]
        # Exact duplicates are descriptive; they are never rejected or redrawn.
        real_values = real_features[real]
        duplicate_real = sum(bool((real_values == row).all(dim=1).any()) for row in values)
        # Zero-lambda copies use only the anchor; the saved candidate neighbor
        # did not contribute feature coordinates to these rows.
        contributing_neighbors = added & (bank["lambda"] > 0)
        parents = torch.cat((bank["anchor_positions"][added],
                             bank["neighbor_positions"][contributing_neighbors]))
        records.append({"class_index": c, "real_count": int(real.sum()), "added_count": len(values),
                        "unique_anchor_count": len(bank["anchor_positions"][added].unique()),
                        "unique_parent_count": len(parents.unique()),
                        "exact_real_duplicate_count": duplicate_real,
                        "repeated_added_count": len(values) - len(values.unique(dim=0)),
                        "mean_neighbor_cosine_distance": float(bank["neighbor_cosine_distance"][added].mean())
                        if added.any() else None})
    return {"real_count": count, "added_count": len(added_features), "total_count": len(bank["labels"]),
            "classes": records}
