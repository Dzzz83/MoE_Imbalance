"""
Evaluation library: metrics, expert-pool loading, routing headroom, run health.

Three questions this module answers, all without touching the test set until the
candidate rules are frozen:

1. **Are the trained experts correct?** → :class:`ExpertPool` plus the metric
   functions (BA, Head/Med/Tail, ECE).
2. **How much room does routing have at all?** → :class:`HeadroomAnalyzer`
   (all-wrong floor, oracle ceiling, pairwise Cohen's kappa).
3. **Did a training run actually behave?** → :class:`RunHealthChecker`, which
   reads the per-epoch history JSON a run writes.

The stacked-logits convention throughout is ``(N, num_experts, num_classes)``,
matching the router interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from scripts.base_trainer import compute_class_groups, group_accuracies


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot be carried out as requested."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def tta_average_log_probs(
    per_view_logits: list[np.ndarray], eps: float = 1e-12
) -> np.ndarray:
    """Average predictions over augmented views, in probability space.

    Computes ``log(mean_v softmax(logits_v))`` — the mean of the per-view
    probability distributions, returned as log-probabilities.

    Averaging **probabilities** rather than raw logits is the standard TTA
    ensemble: logits sit on an arbitrary scale, so a single overconfident view
    would dominate a logit average. Returning log-probabilities keeps the router
    interface unchanged — routers apply ``softmax`` to their input, which
    recovers exactly the averaged distribution.

    With a single view this is the identity (up to the eps clamp).
    """
    if not per_view_logits:
        raise EvaluationError("TTA needs at least one view")
    stack = np.stack([_softmax(v) for v in per_view_logits], axis=0)
    mean_prob = stack.mean(axis=0)
    return np.log(np.maximum(mean_prob, eps))


def aggregate_across_seeds(per_seed_metrics: list[dict]) -> dict:
    """Summarise per-seed metric dicts as mean / std / n per key.

    AGENTs.md section 6 requires every reported number to be averaged over at
    least 3 seeds, so the spread has to travel with the mean.
    """
    if not per_seed_metrics:
        raise EvaluationError("no per-seed metrics to aggregate")
    keys: set[str] = set()
    for m in per_seed_metrics:
        keys |= set(m)
    out: dict[str, dict] = {}
    for key in sorted(keys):
        values = [m[key] for m in per_seed_metrics
                  if isinstance(m.get(key), (int, float)) and not isinstance(m.get(key), bool)]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            'mean': float(arr.mean()),
            'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            'n': int(len(arr)),
        }
    return out


def balanced_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    """Mean per-class recall over the classes present in `targets`."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    classes = np.unique(targets)
    recalls = []
    for c in classes:
        mask = targets == c
        recalls.append((preds[mask] == c).sum() / max(mask.sum(), 1))
    return float(np.mean(recalls)) if recalls else 0.0


def expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> float:
    """|accuracy - confidence| averaged over equal-width confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(targets)
    if total == 0:
        return 0.0

    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_predictions(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_counts: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Full metric bundle for one predictor on one split."""
    groups = compute_class_groups(np.asarray(class_counts))
    out = {
        'ba': balanced_accuracy(targets, preds),
        'ece': expected_calibration_error(probs, targets, n_bins=n_bins),
        'n': int(len(targets)),
        'avg_confidence': float(np.asarray(probs).max(axis=1).mean()),
    }
    out.update(group_accuracies(np.asarray(targets), np.asarray(preds), groups))
    return out


# ---------------------------------------------------------------------------
# Expert pool
# ---------------------------------------------------------------------------

class ExpertPool:
    """Loads expert checkpoints written by the config-driven training runs.

    Checkpoint naming follows ``scripts/base_trainer.py``:
    ``{expert_name}_seed{seed}_final.pt``.
    """

    def __init__(
        self,
        expert_names: list[str],
        seeds: list[int],
        checkpoint_dir: str = './checkpoints',
        device: str = 'cpu',
    ) -> None:
        self.expert_names = list(expert_names)
        self.seeds = list(seeds)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._models: dict[str, torch.nn.Module] = {}

    def available(self) -> dict[tuple[str, int], Path]:
        """Map each (expert label, seed) to its final checkpoint.

        Keyed by both, because a 3-seed pool has three checkpoints per expert.
        An earlier version keyed by label alone and the seed loop overwrote
        itself, so a 3-seed evaluation silently reported a single seed.
        """
        found: dict[tuple[str, int], Path] = {}
        for name in self.expert_names:
            for seed in self.seeds:
                path = self.checkpoint_dir / f'{name}_seed{seed}_final.pt'
                if path.exists():
                    found[(name, seed)] = path
        return found

    def seeds_present(self) -> list[int]:
        """Every seed that has at least one checkpoint on disk."""
        return sorted({seed for (_, seed) in self.available()})

    def seeds_for(self, expert: str) -> list[int]:
        """Seeds available for one expert."""
        return sorted(seed for (name, seed) in self.available() if name == expert)

    def missing(self) -> list[str]:
        """Expert labels with no checkpoint for any requested seed."""
        present = {name for (name, _) in self.available()}
        return [n for n in self.expert_names if n not in present]

    def _check_device(self) -> None:
        """Fail clearly when a CUDA device is requested but unusable.

        Observed for real: the NVIDIA driver wedged around a reboot,
        ``torch.cuda.is_available()`` went False, and ``torch.load`` raised a
        traceback about deserialisation that named neither the cause nor the
        remedy. Checked at call time, not cached, so a driver that recovers is
        picked up without restarting the process.
        """
        if str(self.device).startswith('cuda') and not torch.cuda.is_available():
            raise EvaluationError(
                f"device '{self.device}' was requested, but no working CUDA device "
                f"is available (torch.cuda.is_available() is False).\n"
                f"  Fix one of:\n"
                f"    (a) run on CPU:     add --device cpu\n"
                f"    (b) check the GPU:  nvidia-smi   — if it cannot reach the "
                f"driver, a reboot (or `sudo modprobe nvidia`) usually clears it"
            )

    def load(self, seed: int | None = None) -> 'ExpertPool':
        """Load the experts for one seed. Needs at least two to be a pool.

        Args:
            seed: which seed to load. Required when more than one seed is on
                disk — guessing would silently drop or mix runs.
        """
        from models.resnet32 import ResNet32

        self._check_device()
        available = self.available()
        if not available:
            raise EvaluationError(
                f"no expert checkpoints found in {self.checkpoint_dir} for "
                f"{self.expert_names} x seeds {self.seeds}"
            )

        if seed is None:
            seeds = self.seeds_present()
            if len(seeds) > 1:
                raise EvaluationError(
                    f"multiple seeds present ({seeds}); pass seed=<one of them> "
                    f"so this evaluation reports a single seed's numbers"
                )
            seed = seeds[0]

        paths = {name: p for (name, s), p in available.items() if s == seed}
        if len(paths) < 2:
            raise EvaluationError(
                f"routing needs at least 2 experts for seed {seed}; found "
                f"{sorted(paths)} (missing: {self.missing()})"
            )
        for name, path in paths.items():
            state = torch.load(path, map_location=self.device, weights_only=False)
            model = ResNet32(num_classes=100)
            model.load_state_dict(state['model_state_dict'])
            model.to(self.device).eval()
            self._models[name] = model
        self.expert_names = list(self._models)
        self.loaded_seed = seed
        return self

    @torch.no_grad()
    def logits_from_batch(self, images: torch.Tensor) -> np.ndarray:
        """Stacked logits for one batch, shape (N, num_experts, num_classes)."""
        if not self._models:
            raise EvaluationError("call load() before requesting logits")
        images = images.to(self.device)
        per_expert = [m(images).cpu().numpy() for m in self._models.values()]
        return np.stack(per_expert, axis=1)

    @torch.no_grad()
    def logits(
        self,
        loader,
        n_augs: int = 1,
        tta_seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stacked logits and labels for a whole loader.

        Args:
            loader: yields (images, targets).
            n_augs: number of augmented views per sample. ``1`` is plain
                single-view inference; ``> 1`` enables test-time augmentation,
                averaging the per-view probability distributions.
            tta_seed: fixes the view sampling so TTA is reproducible.

        Returns:
            (logits, labels). With ``n_augs > 1`` the logits are
            log-probabilities of the view-averaged distribution.
        """
        from data.tta import AugmentationViews

        if n_augs < 1:
            raise EvaluationError(f"n_augs must be >= 1, got {n_augs}")
        if n_augs == 1 and tta_seed != 0:
            raise EvaluationError("tta_seed is meaningless when n_augs == 1")

        views = None if n_augs == 1 else AugmentationViews(n_views=n_augs, seed=tta_seed)
        chunks, labels = [], []
        for batch in loader:
            images, targets = batch[0], batch[1]
            if views is None:
                chunks.append(self.logits_from_batch(images))
            else:
                per_view = [self.logits_from_batch(v) for v in views(images)]
                chunks.append(tta_average_log_probs(per_view))
            labels.append(np.asarray(targets))
        if not chunks:
            raise EvaluationError("loader produced no batches")
        return np.concatenate(chunks, axis=0), np.concatenate(labels, axis=0)

    @property
    def loaded(self) -> list[str]:
        return list(self._models)


# ---------------------------------------------------------------------------
# Routing headroom
# ---------------------------------------------------------------------------

class HeadroomAnalyzer:
    """How much routing *could* gain, given a pool of experts.

    All quantities are label-dependent but parameter-free: nothing is fitted, so
    no held-out data is needed to compute them.
    """

    def __init__(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        expert_names: list[str] | None = None,
    ) -> None:
        logits = np.asarray(logits)
        if logits.ndim != 3:
            raise EvaluationError(
                f"logits must be (N, num_experts, num_classes), got {logits.shape}"
            )
        self.logits = logits
        self.targets = np.asarray(targets)
        n, e, _ = logits.shape
        if len(self.targets) != n:
            raise EvaluationError(
                f"{len(self.targets)} labels for {n} samples"
            )
        self.expert_names = list(expert_names) if expert_names else [f'E{i}' for i in range(e)]
        if len(self.expert_names) != e:
            raise EvaluationError(
                f"{len(self.expert_names)} names for {e} experts"
            )

    def correct_matrix(self) -> np.ndarray:
        """Boolean (N, num_experts): did expert e get sample n right?"""
        preds = self.logits.argmax(axis=2)                 # (N, E)
        return preds == self.targets[:, None]

    def correctness_counts(self) -> np.ndarray:
        """How many samples had exactly k experts correct, for k = 0..E."""
        return np.bincount(self.correct_matrix().sum(axis=1),
                           minlength=len(self.expert_names) + 1)

    def all_wrong_fraction(self) -> float:
        """Fraction of samples no expert gets right — the routing floor."""
        return float((self.correct_matrix().sum(axis=1) == 0).mean())

    def oracle_accuracy(self) -> float:
        """BA achievable with a perfect per-sample expert choice (the ceiling)."""
        counts = self.correct_matrix().sum(axis=1)
        return float((counts > 0).mean())

    def pairwise_kappa(self) -> dict[tuple[str, str], float]:
        """Cohen's kappa of predicted labels for every expert pair."""
        preds = self.logits.argmax(axis=2)                 # (N, E)
        out: dict[tuple[str, str], float] = {}
        for i in range(len(self.expert_names)):
            for j in range(i + 1, len(self.expert_names)):
                out[(self.expert_names[i], self.expert_names[j])] = _cohen_kappa(
                    preds[:, i], preds[:, j]
                )
        return out

    def summary(self) -> dict:
        """Everything needed for a headroom report."""
        counts = self.correctness_counts()
        return {
            'n': int(len(self.targets)),
            'num_experts': len(self.expert_names),
            'all_wrong_fraction': self.all_wrong_fraction(),
            'oracle_accuracy': self.oracle_accuracy(),
            'correctness_counts': counts.tolist(),
            'pairwise_kappa': {f'{a}|{b}': v for (a, b), v in self.pairwise_kappa().items()},
        }


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two label vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b):
        conf[index[x], index[y]] += 1
    total = conf.sum()
    if total == 0:
        return 0.0
    po = np.trace(conf) / total
    pe = float((conf.sum(axis=0) * conf.sum(axis=1)).sum()) / (total ** 2)
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return float((po - pe) / (1.0 - pe))


# ---------------------------------------------------------------------------
# Run health
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    """Outcome of checking one training run's history."""

    expert: str
    seed: int
    epochs: int
    ok: bool
    problems: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = 'OK' if self.ok else f'{len(self.problems)} problem(s)'
        return f"{self.expert} seed={self.seed}: {self.epochs} epochs — {status}"


class RunHealthChecker:
    """Validates a training run against the published protocol.

    Checks the things that silently ruin a run: a schedule that never decayed,
    a diverged loss, a truncated run, and a missing final checkpoint.
    """

    def __init__(
        self,
        expert: str,
        seed: int,
        checkpoint_dir: str | None = None,
        expected_epochs: int = 200,
        decay_epochs: tuple[int, int] = (160, 180),
        decay_factors: tuple[float, float] = (0.01, 0.0001),
    ) -> None:
        self.expert = expert
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.expected_epochs = expected_epochs
        self.decay_epochs = tuple(decay_epochs)
        self.decay_factors = tuple(decay_factors)

    def check(self, history: list[dict]) -> HealthReport:
        """Validate a per-epoch history list."""
        problems: list[str] = []
        if not history:
            return HealthReport(self.expert, self.seed, 0, False, ['empty history'])

        epochs = len(history)
        by_epoch = {int(row.get('epoch', i + 1)): row for i, row in enumerate(history)}

        # 1. full budget
        if epochs < self.expected_epochs:
            problems.append(
                f"run stopped after {epochs} epochs, expected {self.expected_epochs}"
            )

        # 2. schedule actually decayed at the milestones.
        # Both factors are relative to the *base* lr, exactly as in the
        # reference `adjust_learning_rate`: lr=base for e<=160, base*0.01 for
        # 161..180, base*0.0001 after 180. So the reference for each milestone
        # is the plateau lr, never the previous milestone's lr.
        first_decay, second_decay = self.decay_epochs
        plateau = by_epoch.get(first_decay)
        after_first = by_epoch.get(first_decay + 1)
        after_second = by_epoch.get(second_decay + 1)
        base_lr = plateau['lr'] if plateau else None
        if base_lr:
            if after_first:
                ratio = after_first['lr'] / base_lr
                if abs(ratio / self.decay_factors[0] - 1.0) > 1e-6:
                    problems.append(
                        f"lr at epoch {first_decay + 1} is {ratio:.4g} x base, "
                        f"expected {self.decay_factors[0]}"
                    )
            if after_second:
                ratio = after_second['lr'] / base_lr
                if abs(ratio / self.decay_factors[1] - 1.0) > 1e-6:
                    problems.append(
                        f"lr at epoch {second_decay + 1} is {ratio:.4g} x base, "
                        f"expected {self.decay_factors[1]}"
                    )

        # 3. finite losses
        losses = [row.get('train_loss') for row in history]
        n_bad = sum(1 for v in losses if v is None or not np.isfinite(v))
        if n_bad:
            problems.append(f"{n_bad} epoch(s) with a missing or non-finite loss")

        # 4. loss descended overall
        finite = [v for v in losses if v is not None and np.isfinite(v)]
        if len(finite) >= 20:
            head = float(np.mean(finite[:10]))
            tail = float(np.mean(finite[-10:]))
            if tail >= head:
                problems.append(
                    f"loss did not descend (first 10 epochs mean {head:.4f} -> "
                    f"last 10 mean {tail:.4f})"
                )

        # 5. final checkpoint present
        if self.checkpoint_dir is not None:
            expected = self.checkpoint_dir / f'{self.expert}_seed{self.seed}_final.pt'
            if not expected.exists():
                problems.append(f"missing final checkpoint: {expected}")

        return HealthReport(self.expert, self.seed, epochs, not problems, problems)


def load_history(path: str | Path) -> list[dict]:
    """Read a `*_history.json` written by the trainer."""
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f"missing history file: {path}")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise EvaluationError(f"{path}: expected a list of epoch records")
    return data
