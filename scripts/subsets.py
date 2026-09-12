"""
Ensemble-size analysis: how many experts, and which ones?

Answers "why 2, why 3, why 4?" empirically by evaluating a combination rule over
**every subset** of the expert pool, so accuracy can be read as a function of
ensemble size.

A methodological trap this module exists to avoid
-------------------------------------------------
Choosing the best subset *on the test set* is selection-on-test: with 15 subsets
you will always find one that looks good by luck. So two numbers are reported for
every size and they are never conflated:

    mean_ba  over all subsets of that size  - unbiased estimate
    best_ba  over all subsets of that size  - optimistically biased

``best_ba`` is labelled ``optimistic=True`` in the table and the module carries
an explicit warning string, so an optimistic number cannot be quoted as a result
by accident.

The rules themselves are parameter-free, so none of this needs held-out data.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from scripts.evaluation import EvaluationError, aggregate_across_seeds

#: Shown alongside any best-of-N number.
SELECTION_WARNING = (
    "best_ba is the maximum over subsets chosen on the test set, so it is "
    "selection-on-test and optimistically biased; quote mean_ba as the result"
)


def balanced_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    """Mean per-class recall (local copy so this module stays standalone)."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    recalls = []
    for c in np.unique(targets):
        mask = targets == c
        recalls.append((preds[mask] == c).sum() / max(mask.sum(), 1))
    return float(np.mean(recalls)) if recalls else 0.0


class SubsetEnsembleAnalysis:
    """Evaluates a routing rule over every subset of a fixed expert pool.

    Args:
        logits: (N, num_experts, num_classes) expert logits for one split.
        targets: (N,) ground-truth labels.
        expert_names: label per expert column.
        class_counts: per-class training counts, used for head/med/tail groups.
    """

    def __init__(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        expert_names: list[str],
        class_counts: np.ndarray | None = None,
    ) -> None:
        logits = np.asarray(logits)
        targets = np.asarray(targets)
        if logits.ndim != 3:
            raise EvaluationError(
                f"logits must be (N, experts, classes), got {logits.shape}"
            )
        if len(targets) != logits.shape[0]:
            raise EvaluationError(f"{len(targets)} labels for {logits.shape[0]} samples")
        if len(expert_names) != logits.shape[1]:
            raise EvaluationError(
                f"{len(expert_names)} names for {logits.shape[1]} experts"
            )
        self.logits = logits
        self.targets = targets
        self.expert_names = list(expert_names)
        self.class_counts = class_counts
        self.num_experts = logits.shape[1]

    # ── subsets ───────────────────────────────────────────────────────

    def subsets(self, size: int | None = None) -> list[tuple[int, ...]]:
        """Every subset of the given size, as ordered index tuples."""
        sizes = [size] if size is not None else range(1, self.num_experts + 1)
        out: list[tuple[int, ...]] = []
        for k in sizes:
            if not 1 <= k <= self.num_experts:
                raise EvaluationError(
                    f"subset size {k} out of range 1..{self.num_experts}"
                )
            out.extend(combinations(range(self.num_experts), k))
        return out

    def names(self, subset: tuple[int, ...]) -> list[str]:
        return [self.expert_names[i] for i in subset]

    # ── evaluation ────────────────────────────────────────────────────

    def evaluate_subset(self, rule_class, subset: tuple[int, ...]) -> dict:
        """Apply a router class to one subset of the expert columns."""
        if not subset:
            raise EvaluationError("empty subset")
        names = self.names(subset)
        rule = rule_class(expert_names=names)
        sliced = self.logits[:, list(subset), :]
        preds = rule.predict_class(sliced)
        ba = balanced_accuracy(self.targets, preds)

        out = {'ba': ba, 'subset': tuple(subset), 'names': tuple(names)}
        if self.class_counts is not None:
            from scripts.evaluation import evaluate_predictions
            probs = np.einsum('ne,nec->nc', rule.predict_proba(sliced),
                              _softmax(sliced))
            metrics = evaluate_predictions(self.targets, preds, probs,
                                           self.class_counts)
            out['tail'] = metrics['tail']
            out['head'] = metrics['head']
        return out

    def size_table(self, rule_class) -> dict[int, dict]:
        """Per-size summary: unbiased mean, optimistic best, and the winner."""
        table: dict[int, dict] = {}
        for size in range(1, self.num_experts + 1):
            results = [self.evaluate_subset(rule_class, s) for s in self.subsets(size)]
            bas = np.array([r['ba'] for r in results])
            best_idx = int(bas.argmax())
            row = {
                'n_subsets': len(results),
                'mean_ba': float(bas.mean()),
                'std_ba': float(bas.std(ddof=1)) if len(bas) > 1 else 0.0,
                'best_ba': float(bas.max()),
                'worst_ba': float(bas.min()),
                'best_subset': results[best_idx]['subset'],
                'best_names': results[best_idx]['names'],
                # only meaningful (and only flagged) where a choice was made
                'optimistic': len(results) > 1,
            }
            if 'tail' in results[0]:
                tails = np.array([r['tail'] for r in results])
                row['mean_tail'] = float(tails.mean())
                row['best_tail'] = float(tails.max())
            table[size] = row
        return table


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def aggregate_size_tables(tables: list[dict[int, dict]]) -> dict[int, dict]:
    """Mean/std of the per-size numbers across seeds."""
    if not tables:
        raise EvaluationError("no size tables to aggregate")
    sizes = sorted(tables[0])
    out: dict[int, dict] = {}
    for size in sizes:
        rows = [t[size] for t in tables]
        agg = aggregate_across_seeds(
            [{k: r[k] for k in ('mean_ba', 'std_ba', 'best_ba', 'worst_ba')}
             for r in rows]
        )
        out[size] = {
            **agg,
            'n_subsets': rows[0]['n_subsets'],
            'optimistic': rows[0]['optimistic'],
        }
    return out
