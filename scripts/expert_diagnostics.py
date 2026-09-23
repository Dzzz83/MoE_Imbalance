"""Array-only diagnostics for a supplied pool of classification experts.

This module deliberately has no data, checkpoint, or filesystem dependency.
Callers supply aligned expert logits or predictions and, for label-dependent
reports, the corresponding labels.  The stacked-logit convention is
``(samples, experts, classes)`` and the prediction convention is
``(samples, experts)``.

The diagnostics in this module are descriptive or oracle measurements.  They
do not fit a router and do not turn label-dependent quantities into inference
rules.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from scripts.base_trainer import compute_class_groups
from scripts.analysis.combination import (
    combine_weighted_logits,
    combine_weighted_probabilities,
    stable_softmax,
)
from scripts.analysis.validation import validate_expert_weights, validate_integer_vector
from scripts.evaluation import balanced_accuracy


class DiagnosticInputError(ValueError):
    """Raised when supplied diagnostic arrays are malformed or misaligned."""


def _as_integer_array(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    """Validate an array of integer-valued labels or predictions."""
    return validate_integer_vector(
        value,
        name=name,
        ndim=ndim,
        error_type=DiagnosticInputError,
        wrap_conversion_errors=False,
    )


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the final axis."""
    return stable_softmax(logits)


def _fraction(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


class ExpertDiagnostics:
    """Compute diagnostics from already supplied expert outputs.

    Args:
        predictions: Optional expert class predictions, shape ``(N, E)``.
        logits: Optional expert logits, shape ``(N, E, C)``.  If predictions
            are omitted, they are derived with ``argmax``.  When both are
            supplied they must agree exactly, which catches sample/expert
            ordering mistakes early.
        labels: Optional ground-truth class labels, shape ``(N,)``.  Required
            by correctness, complementarity, headroom, and metric reports.
        class_counts: Optional per-class training counts, shape ``(C,)``.
            When present, the canonical project grouping implementation is
            used for Head/Medium/Tail reports.
        expert_names: Optional names in the exact column order of predictions
            and logits.  Names must be unique.

    No method in this class loads data or checkpoints, fits parameters, or
    evaluates a dataset.  It only operates on the arrays passed by the caller.
    """

    def __init__(
        self,
        predictions: np.ndarray | None = None,
        logits: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        class_counts: np.ndarray | None = None,
        expert_names: Sequence[str] | None = None,
    ) -> None:
        if predictions is None and logits is None:
            raise DiagnosticInputError("supply predictions or logits")

        logits_array: np.ndarray | None = None
        if logits is not None:
            logits_array = np.asarray(logits)
            if logits_array.ndim != 3:
                raise DiagnosticInputError(
                    "logits must have shape (samples, experts, classes), "
                    f"got {logits_array.shape}"
                )
            if not np.issubdtype(logits_array.dtype, np.number) or np.iscomplexobj(logits_array):
                raise DiagnosticInputError("logits must contain real numeric values")
            if not np.isfinite(logits_array).all():
                raise DiagnosticInputError("logits contain non-finite values")
            if logits_array.shape[0] == 0:
                raise DiagnosticInputError("logits must contain at least one sample")
            if logits_array.shape[1] == 0:
                raise DiagnosticInputError("logits must contain at least one expert")
            if logits_array.shape[2] == 0:
                raise DiagnosticInputError("logits must contain at least one class")

        inferred_predictions: np.ndarray | None = None
        if predictions is not None:
            inferred_predictions = _as_integer_array(
                predictions, name="predictions", ndim=2
            )
            if inferred_predictions.shape[0] == 0:
                raise DiagnosticInputError(
                    "predictions must contain at least one sample"
                )
            if inferred_predictions.shape[1] == 0:
                raise DiagnosticInputError(
                    "predictions must contain at least one expert"
                )

        if logits_array is not None and inferred_predictions is not None:
            if inferred_predictions.shape != logits_array.shape[:2]:
                raise DiagnosticInputError(
                    "predictions and logits are misaligned: predictions shape "
                    f"{inferred_predictions.shape}, logits leading shape "
                    f"{logits_array.shape[:2]}"
                )
            derived = logits_array.argmax(axis=2)
            if not np.array_equal(inferred_predictions, derived):
                raise DiagnosticInputError(
                    "predictions disagree with logits.argmax(axis=2); verify "
                    "expert ordering and sample alignment"
                )

        if logits_array is not None:
            n_samples, n_experts, n_classes = logits_array.shape
        else:
            assert inferred_predictions is not None  # checked above
            n_samples, n_experts = inferred_predictions.shape
            if class_counts is not None:
                supplied_counts = np.asarray(class_counts)
                if supplied_counts.ndim != 1 or len(supplied_counts) == 0:
                    raise DiagnosticInputError(
                        "class_counts must be a non-empty one-dimensional array"
                    )
                n_classes = len(supplied_counts)
            else:
                n_classes = int(inferred_predictions.max()) + 1

        if labels is not None:
            labels_array = _as_integer_array(labels, name="labels", ndim=1)
            if len(labels_array) != n_samples:
                raise DiagnosticInputError(
                    f"labels and expert outputs are misaligned: {len(labels_array)} "
                    f"labels for {n_samples} samples"
                )
            if labels_array.size and labels_array.max() >= n_classes:
                raise DiagnosticInputError(
                    f"labels contain class {int(labels_array.max())}, but outputs "
                    f"have only {n_classes} classes"
                )
            if labels_array.size and labels_array.min() < 0:
                raise DiagnosticInputError("labels must be non-negative class indices")
        else:
            labels_array = None

        if inferred_predictions is None:
            assert logits_array is not None
            inferred_predictions = logits_array.argmax(axis=2).astype(np.int64)

        if inferred_predictions.min() < 0 or inferred_predictions.max() >= n_classes:
            raise DiagnosticInputError(
                f"predictions contain a class outside [0, {n_classes})"
            )

        counts_array: np.ndarray | None = None
        if class_counts is not None:
            counts_array = np.asarray(class_counts)
            if counts_array.ndim != 1 or len(counts_array) != n_classes:
                raise DiagnosticInputError(
                    "class_counts must have shape (classes,), matching logits; "
                    f"got {counts_array.shape} for {n_classes} classes"
                )
            if not np.issubdtype(counts_array.dtype, np.number) or np.iscomplexobj(counts_array):
                raise DiagnosticInputError("class_counts must contain real numeric counts")
            if not np.isfinite(counts_array).all():
                raise DiagnosticInputError("class_counts contain non-finite values")
            if (counts_array < 1).any() or not np.equal(counts_array, np.floor(counts_array)).all():
                missing = np.flatnonzero(counts_array < 1).tolist()
                raise DiagnosticInputError(
                    "class_counts must contain a positive integer count for every "
                    f"class; missing/empty classes: {missing}"
                )
            counts_array = counts_array.astype(np.int64, copy=False)
            if labels_array is not None:
                present = np.bincount(labels_array, minlength=n_classes) > 0
                missing = np.flatnonzero(~present).tolist()
                if missing:
                    raise DiagnosticInputError(
                        "labels are missing classes required by class_counts: "
                        f"{missing}"
                    )

        if expert_names is None:
            names = [f"E{i}" for i in range(n_experts)]
        else:
            names = list(expert_names)
            if len(names) != n_experts:
                raise DiagnosticInputError(
                    f"expert_names has {len(names)} entries for {n_experts} experts"
                )
            if any(not isinstance(name, str) or not name for name in names):
                raise DiagnosticInputError("expert_names must be non-empty strings")
            if len(set(names)) != len(names):
                raise DiagnosticInputError("expert_names must be unique and ordered")

        self.logits = logits_array
        self.predictions = inferred_predictions
        self.labels = labels_array
        self.class_counts = counts_array
        self.expert_names = names
        self.n_samples = n_samples
        self.num_experts = n_experts
        self.num_classes = n_classes

    def _require_labels(self, report: str) -> np.ndarray:
        if self.labels is None:
            raise DiagnosticInputError(f"{report} requires labels")
        return self.labels

    def _require_logits(self, report: str) -> np.ndarray:
        if self.logits is None:
            raise DiagnosticInputError(f"{report} requires logits")
        return self.logits

    def _correct_matrix(self) -> np.ndarray:
        labels = self._require_labels("correctness diagnostics")
        return self.predictions == labels[:, None]

    def _pattern_key(self, row: np.ndarray) -> str:
        _, counts = np.unique(row, return_counts=True)
        counts = sorted((int(count) for count in counts), reverse=True)
        if len(counts) == 1:
            return f"{self.num_experts}-0"
        return "-".join(str(count) for count in counts)

    def agreement_patterns(self) -> dict:
        """Report arbitrary-expert prediction partitions and lone dissenters.

        A unique dissenter exists only for the ``(E-1)-1`` partition.  In
        particular, a four-expert ``2-1-1`` sample has two dissenters and is
        never included in the unique-dissenter denominator.
        """
        labels = self._require_labels("agreement-pattern diagnostics")
        correct = self._correct_matrix()
        pattern_counts: dict[str, int] = {}
        unique_samples: list[int] = []
        agreeing_correct = 0
        dissenting_correct = 0
        dissenter_indices: list[int] = []

        for sample, row in enumerate(self.predictions):
            key = self._pattern_key(row)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

            values, counts = np.unique(row, return_counts=True)
            order = np.argsort(-counts, kind="stable")
            counts_desc = counts[order].tolist()
            if self.num_experts >= 3 and counts_desc == [self.num_experts - 1, 1]:
                majority_class = values[order[0]]
                dissent_class = values[order[1]]
                dissenter = int(np.flatnonzero(row == dissent_class)[0])
                unique_samples.append(sample)
                dissenter_indices.append(dissenter)
                agreeing_correct += int(majority_class == labels[sample])
                dissenting_correct += int(correct[sample, dissenter])

        unique_count = len(unique_samples)
        return {
            "n_samples": self.n_samples,
            "num_experts": self.num_experts,
            "pattern_counts": pattern_counts,
            "pattern_fractions": {
                key: _fraction(count, self.n_samples)
                for key, count in pattern_counts.items()
            },
            "unique_dissenter": {
                "pattern": f"{self.num_experts - 1}-1",
                "count": unique_count,
                "fraction_of_all": _fraction(unique_count, self.n_samples),
                "denominator": unique_count,
                "sample_indices": unique_samples,
                "dissenter_indices": dissenter_indices,
                "agreeing_group_correct_count": agreeing_correct,
                "agreeing_group_correct_fraction": _fraction(
                    agreeing_correct, unique_count
                ),
                "dissenting_expert_correct_count": dissenting_correct,
                "dissenting_expert_correct_fraction": _fraction(
                    dissenting_correct, unique_count
                ),
            },
            "non_unique_dissenter_count": self.n_samples - unique_count,
        }

    def correctness_diagnostics(self) -> dict:
        """Separate correctness counts from confidence-based events."""
        self._require_labels("correctness diagnostics")
        logits = self._require_logits("confidence diagnostics")
        correct = self._correct_matrix()
        num_correct = correct.sum(axis=1).astype(np.int64)

        def event(count: int) -> dict:
            return {
                "count": int(count),
                "fraction": _fraction(int(count), self.n_samples),
                "denominator": self.n_samples,
            }

        counts = {
            str(k): int((num_correct == k).sum())
            for k in range(self.num_experts + 1)
        }
        confidence = _stable_softmax(logits).max(axis=2)
        global_expert = confidence.argmax(axis=1)  # stable first-column tie break
        global_correct = correct[np.arange(self.n_samples), global_expert]

        any_correct = num_correct > 0
        order = np.argsort(-confidence, axis=1, kind="stable")
        rank_matrix = np.empty_like(order)
        rank_matrix[np.arange(self.n_samples)[:, None], order] = np.arange(
            self.num_experts
        )[None, :]
        best_correct_rank = np.full(self.n_samples, -1, dtype=np.int64)
        unique_highest_correct = np.zeros(self.n_samples, dtype=bool)
        for sample in np.flatnonzero(any_correct):
            correct_ranks = rank_matrix[sample, correct[sample]]
            best_correct_rank[sample] = int(correct_ranks.min()) + 1
            correct_confidence = confidence[sample, correct[sample]]
            unique_highest_correct[sample] = (
                int((correct_confidence == correct_confidence.max()).sum()) == 1
            )

        rank_counts = {
            str(rank): int((best_correct_rank[any_correct] == rank).sum())
            for rank in range(1, self.num_experts + 1)
        }
        rank_denominator = int(any_correct.sum())
        return {
            "n_samples": self.n_samples,
            "num_correct_experts": num_correct.tolist(),
            "correct_count_distribution": counts,
            "exactly_one_correct": event((num_correct == 1).sum()),
            "multiple_correct": event((num_correct >= 2).sum()),
            "no_correct": event((num_correct == 0).sum()),
            "all_correct": event((num_correct == self.num_experts).sum()),
            "at_least_one_correct": event(any_correct.sum()),
            "globally_most_confident": {
                "correct_count": int(global_correct.sum()),
                "correct_fraction": _fraction(
                    int(global_correct.sum()), self.n_samples
                ),
                "denominator": self.n_samples,
                "selected_expert_indices": global_expert.tolist(),
                "tie_policy": "first expert in supplied ordering",
            },
            "unique_highest_confidence_among_correct": {
                "count": int(unique_highest_correct[any_correct].sum()),
                "fraction": _fraction(
                    int(unique_highest_correct[any_correct].sum()),
                    rank_denominator,
                ),
                "denominator": rank_denominator,
                "conditioning_event": "samples with at least one correct expert",
            },
            "confidence_ranking_among_correct": {
                "denominator": rank_denominator,
                "conditioning_event": "samples with at least one correct expert",
                "best_correct_rank_counts": rank_counts,
                "best_correct_rank_fractions": {
                    rank: _fraction(count, rank_denominator)
                    for rank, count in rank_counts.items()
                },
                "mean_best_correct_rank": (
                    float(best_correct_rank[any_correct].mean())
                    if rank_denominator
                    else None
                ),
                "rank_definition": (
                    "one plus the highest-confidence correct expert's rank "
                    "among all experts; confidence ties use supplied ordering"
                ),
            },
        }

    def _class_recalls(self, predictions: np.ndarray) -> np.ndarray:
        """Return recall for every output class, with NaN for absent labels."""
        labels = self._require_labels("metric diagnostics")
        recalls = np.full(self.num_classes, np.nan, dtype=np.float64)
        for class_index in np.unique(labels):
            mask = labels == class_index
            recalls[class_index] = float(
                np.mean(predictions[mask] == class_index)
            )
        return recalls

    def _prediction_metrics(self, predictions: np.ndarray) -> dict:
        """Compute sample accuracy, BA, and optional macro group recalls."""
        labels = self._require_labels("metric diagnostics")
        predictions = np.asarray(predictions, dtype=np.int64)
        recalls = self._class_recalls(predictions)
        metrics = {
            "accuracy": float(np.mean(predictions == labels)),
            "ba": balanced_accuracy(labels, predictions),
        }
        if self.class_counts is not None:
            groups = compute_class_groups(self.class_counts)
            for group_name, classes in groups.items():
                if len(classes) == 0:
                    metrics[group_name] = None
                    continue
                group_recalls = recalls[classes]
                if np.isnan(group_recalls).any():
                    missing = classes[np.isnan(group_recalls)].tolist()
                    raise DiagnosticInputError(
                        f"labels are missing classes in {group_name} group: {missing}"
                    )
                metrics[group_name] = float(np.mean(group_recalls))
        return metrics

    def _group_complementarity(self, correct: np.ndarray) -> dict:
        """Aggregate per-class correctness as macro H/M/T complementarity."""
        if self.class_counts is None:
            return {}
        labels = self._require_labels("group complementarity diagnostics")
        groups = compute_class_groups(self.class_counts)
        any_correct = correct.any(axis=1)
        all_wrong = ~any_correct
        exclusive = correct & (correct.sum(axis=1) == 1)[:, None]
        output: dict[str, dict] = {}
        for group_name, classes in groups.items():
            if len(classes) == 0:
                output[group_name] = {
                    "classes": [],
                    "num_classes": 0,
                    "sample_count": 0,
                    "expert_macro_recall": None,
                    "any_correct_macro_recall": None,
                    "all_wrong_macro_recall": None,
                    "exclusive_correct_macro_recall": None,
                }
                continue
            per_class_masks = [labels == class_index for class_index in classes]
            if any(not mask.any() for mask in per_class_masks):
                missing = [
                    int(class_index)
                    for class_index, mask in zip(classes, per_class_masks)
                    if not mask.any()
                ]
                raise DiagnosticInputError(
                    f"labels are missing classes in {group_name} group: {missing}"
                )
            expert_macro = []
            exclusive_macro = []
            any_macro = []
            all_wrong_macro = []
            for mask in per_class_masks:
                expert_macro.append(correct[mask].mean(axis=0))
                exclusive_macro.append(exclusive[mask].mean(axis=0))
                any_macro.append(float(any_correct[mask].mean()))
                all_wrong_macro.append(float(all_wrong[mask].mean()))
            output[group_name] = {
                "classes": [int(class_index) for class_index in classes],
                "num_classes": len(classes),
                "sample_count": int(sum(mask.sum() for mask in per_class_masks)),
                "expert_macro_recall": {
                    name: float(value)
                    for name, value in zip(self.expert_names, np.mean(expert_macro, axis=0))
                },
                "any_correct_macro_recall": float(np.mean(any_macro)),
                "all_wrong_macro_recall": float(np.mean(all_wrong_macro)),
                "exclusive_correct_macro_recall": {
                    name: float(value)
                    for name, value in zip(
                        self.expert_names, np.mean(exclusive_macro, axis=0)
                    )
                },
            }
        return output

    def complementarity(self) -> dict:
        """Measure correctness overlap without treating disagreement as gain."""
        labels = self._require_labels("complementarity diagnostics")
        correct = self._correct_matrix()
        any_correct = correct.any(axis=1)
        exclusive = correct & (correct.sum(axis=1) == 1)[:, None]

        per_expert = {}
        for index, name in enumerate(self.expert_names):
            metrics = self._prediction_metrics(self.predictions[:, index])
            per_expert[name] = {
                "correct_count": int(correct[:, index].sum()),
                "correct_fraction": _fraction(int(correct[:, index].sum()), self.n_samples),
                **metrics,
            }

        pairwise = {}
        for left in range(self.num_experts):
            for right in range(left + 1, self.num_experts):
                key = f"{self.expert_names[left]}|{self.expert_names[right]}"
                joint_correct = correct[:, left] & correct[:, right]
                joint_error = ~correct[:, left] & ~correct[:, right]
                pairwise[key] = {
                    "experts": [self.expert_names[left], self.expert_names[right]],
                    "joint_correct_count": int(joint_correct.sum()),
                    "joint_correct_fraction": _fraction(
                        int(joint_correct.sum()), self.n_samples
                    ),
                    "joint_error_count": int(joint_error.sum()),
                    "joint_error_fraction": _fraction(
                        int(joint_error.sum()), self.n_samples
                    ),
                    "prediction_disagreement_fraction": float(
                        np.mean(self.predictions[:, left] != self.predictions[:, right])
                    ),
                }

        exclusive_report = {}
        for index, name in enumerate(self.expert_names):
            count = int(exclusive[:, index].sum())
            exclusive_report[name] = {
                "count": count,
                "fraction_of_all_samples": _fraction(count, self.n_samples),
                "fraction_given_any_expert_correct": _fraction(
                    count, int(any_correct.sum())
                ),
                "denominator_all_samples": self.n_samples,
                "denominator_any_expert_correct": int(any_correct.sum()),
            }

        per_class = {}
        for class_index in range(self.num_classes):
            mask = labels == class_index
            if not mask.any():
                continue
            per_class[str(class_index)] = {
                "sample_count": int(mask.sum()),
                "expert_correct_fraction": {
                    name: float(value)
                    for name, value in zip(self.expert_names, correct[mask].mean(axis=0))
                },
                "any_correct_fraction": float(any_correct[mask].mean()),
                "all_wrong_fraction": float((~any_correct[mask]).mean()),
                "exclusive_correct_fraction": {
                    name: float(value)
                    for name, value in zip(self.expert_names, exclusive[mask].mean(axis=0))
                },
            }

        return {
            "n_samples": self.n_samples,
            "num_experts": self.num_experts,
            "per_expert": per_expert,
            "pairwise": pairwise,
            "exclusive_correct": exclusive_report,
            "num_correct_experts": correct.sum(axis=1).astype(np.int64).tolist(),
            "per_class": per_class,
            "head_medium_tail": self._group_complementarity(correct),
            "agreement_patterns": self.agreement_patterns(),
        }

    def hard_routing_headroom(self) -> dict:
        """Report label-dependent headroom for a hard expert selector.

        The oracle selects the first correct expert in supplied order and
        falls back to expert zero when all experts are wrong.  This is an
        oracle diagnostic only.  In particular, the all-wrong fraction is not
        an upper bound on arbitrary soft logit/probability combinations.
        """
        self._require_labels("hard-routing headroom diagnostics")
        correct = self._correct_matrix()
        any_correct = correct.any(axis=1)
        selected = np.zeros(self.n_samples, dtype=np.int64)
        for sample in np.flatnonzero(any_correct):
            selected[sample] = int(np.flatnonzero(correct[sample])[0])
        oracle_predictions = self.predictions[
            np.arange(self.n_samples), selected
        ]
        oracle_metrics = self._prediction_metrics(oracle_predictions)
        return {
            "n_samples": self.n_samples,
            "all_experts_wrong_fraction": _fraction(
                int((~any_correct).sum()), self.n_samples
            ),
            "at_least_one_expert_correct_fraction": _fraction(
                int(any_correct.sum()), self.n_samples
            ),
            "hard_selection_oracle_accuracy": oracle_metrics["accuracy"],
            "hard_selection_oracle_balanced_accuracy": oracle_metrics["ba"],
            "hard_selection_oracle_metrics": oracle_metrics,
            "interpretation": (
                "label-dependent hard-selection oracle; not an inference-time "
                "router and not a soft-mixture upper bound"
            ),
        }

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        return validate_expert_weights(
            weights,
            num_experts=self.num_experts,
            num_samples=self.n_samples,
            sum_atol=1e-7,
            name="routing weights",
            error_type=DiagnosticInputError,
            wrap_conversion_errors=False,
        )

    def evaluate_soft_mixture(
        self,
        weights: np.ndarray,
        combination: str = "logit",
    ) -> dict:
        """Evaluate supplied row-stochastic soft expert weights.

        ``combination='logit'`` computes ``softmax(sum_e w_e z_e)``.
        ``combination='probability'`` computes ``sum_e w_e softmax(z_e)``.
        The weights are supplied by the caller; this method performs no fitting
        or optimization and does not use labels except for optional reporting.
        """
        logits = self._require_logits("soft-mixture diagnostics")
        weights = self._validate_weights(weights)
        mode = str(combination).lower()
        aliases = {
            "logit": "logit",
            "logits": "logit",
            "probability": "probability",
            "probabilities": "probability",
        }
        if mode not in aliases:
            raise DiagnosticInputError(
                "combination must be 'logit' or 'probability', "
                f"got {combination!r}"
            )
        mode = aliases[mode]

        if mode == "logit":
            combined_logits = combine_weighted_logits(
                logits,
                weights,
                num_experts=self.num_experts,
                weight_sum_atol=1e-7,
                error_type=DiagnosticInputError,
            )
            probabilities = _stable_softmax(combined_logits)
        else:
            combined_logits = None
            probabilities = combine_weighted_probabilities(
                logits,
                weights,
                num_experts=self.num_experts,
                weight_sum_atol=1e-7,
                error_type=DiagnosticInputError,
            )
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        return {
            "combination": mode,
            "weights": weights,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": (
                self._prediction_metrics(predictions)
                if self.labels is not None
                else None
            ),
            "combined_logits": combined_logits,
            "label_dependent": self.labels is not None,
        }

    def ensemble_contribution(self) -> dict:
        """Compare uniform logit averaging with every leave-one-out ensemble.

        All comparisons use the same samples, so each reported delta is a
        paired change.  ``delta_*`` is ``without_expert - complete``; positive
        means that removing the named expert improved that metric on the
        supplied development data.
        """
        logits = self._require_logits("ensemble-contribution diagnostics")
        self._require_labels("ensemble-contribution diagnostics")
        if self.class_counts is None:
            raise DiagnosticInputError(
                "ensemble contribution requires class_counts to report Tail "
                "macro recall"
            )
        if self.num_experts < 2:
            raise DiagnosticInputError(
                "ensemble contribution requires at least two experts"
            )

        complete_predictions = logits.mean(axis=1).argmax(axis=1).astype(np.int64)
        complete_metrics = self._prediction_metrics(complete_predictions)
        without = {}
        for removed, name in enumerate(self.expert_names):
            remaining = np.delete(logits, removed, axis=1)
            predictions = remaining.mean(axis=1).argmax(axis=1).astype(np.int64)
            metrics = self._prediction_metrics(predictions)
            complete_correct = complete_predictions == self.labels
            without_correct = predictions == self.labels
            without[name] = {
                "removed_expert_index": removed,
                "metrics": metrics,
                "delta_ba": metrics["ba"] - complete_metrics["ba"],
                "delta_tail": metrics["tail"] - complete_metrics["tail"],
                "paired_sample_accuracy": {
                    "improved": int((without_correct & ~complete_correct).sum()),
                    "degraded": int((complete_correct & ~without_correct).sum()),
                    "unchanged": int((complete_correct == without_correct).sum()),
                    "denominator": self.n_samples,
                },
            }
        return {
            "combination": "uniform logit average",
            "complete": complete_metrics,
            "without_expert": without,
            "delta_definition": "without_expert - complete",
            "label_dependent": True,
        }


__all__ = ["DiagnosticInputError", "ExpertDiagnostics"]
