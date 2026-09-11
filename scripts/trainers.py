"""
The four experts, each consuming a typed TrainingConfig.

Structure
---------
``ConfigDrivenTrainer`` translates a :class:`~scripts.config.TrainingConfig` into
the granular knobs ``BaseTrainer`` needs, so each concrete expert only has to say
how to build its model and its loss. ``TrainerRegistry`` maps a config's
``expert`` key to its class, which is what makes ``scripts/train.py`` fully
config-driven: adding an expert means adding a class and a YAML file, not editing
a dispatcher.

Note on LA vs Balanced Softmax: with tau=1.0 the logit-adjusted loss adds
``log(n_y) - log(N)`` and Balanced Softmax adds ``log(n_y)``. The two differ only
by a class-independent constant, so they are the *same objective* and give no
ensemble diversity. Both are kept for comparability with the recorded results.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.mixup import Mixup
from losses.balanced_softmax_loss import BalancedSoftmaxLoss
from losses.ce_loss import CELoss
from losses.lal_loss import LALLoss
from models.resnet32 import ResNet32
from scripts.base_trainer import BaseTrainer
from scripts.config import ConfigError, TrainingConfig

EXPECTED_NUM_CLASSES = 100


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TrainerRegistry:
    """Maps an expert key (from the config) to its trainer class."""

    _registry: dict[str, type['ConfigDrivenTrainer']] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(klass: type['ConfigDrivenTrainer']) -> type['ConfigDrivenTrainer']:
            if key in cls._registry:
                raise ValueError(f"expert '{key}' is already registered by "
                                 f"{cls._registry[key].__name__}")
            cls._registry[key] = klass
            klass.expert_key = key
            return klass
        return decorator

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def get(cls, key: str) -> type['ConfigDrivenTrainer']:
        if key not in cls._registry:
            raise ConfigError(
                f"no trainer registered for expert '{key}'; available: {cls.keys()}"
            )
        return cls._registry[key]

    @classmethod
    def build(
        cls,
        config: TrainingConfig,
        class_counts: np.ndarray | None = None,
        device: str | None = None,
    ) -> 'ConfigDrivenTrainer':
        return cls.get(config.expert)(config, class_counts=class_counts, device=device)


def build_trainer(
    config: TrainingConfig,
    class_counts: np.ndarray | None = None,
    device: str | None = None,
) -> 'ConfigDrivenTrainer':
    """Build the trainer described by `config`."""
    return TrainerRegistry.build(config, class_counts=class_counts, device=device)


# ---------------------------------------------------------------------------
# Configured base
# ---------------------------------------------------------------------------

class ConfigDrivenTrainer(BaseTrainer):
    """Translates a TrainingConfig into BaseTrainer settings."""

    expert_key: str = ''
    expert_name: str = ''

    def __init__(
        self,
        config: TrainingConfig,
        class_counts: np.ndarray | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        counts = None if class_counts is None else np.asarray(class_counts)

        model = self.build_model(config)
        loss_fn = self.build_loss(config, counts)

        super().__init__(
            model=model,
            expert_name=self.expert_name,
            loss_fn=loss_fn,
            class_counts=counts,
            device=device or config.resolved_device,
            lr=config.optimiser.lr,
            weight_decay=config.optimiser.weight_decay,
            momentum=config.optimiser.momentum,
            batch_size=config.data.batch_size,
            epochs=config.schedule.epochs,
            warmup_epochs=config.schedule.warmup_epochs,
            decay_epochs=config.schedule.decay_epochs,
            decay_factors=config.schedule.decay_factors,
            checkpoint_dir=config.checkpoint.dir,
            seed=config.seed,
        )

    # ── to be overridden ──────────────────────────────────────────────

    def build_model(self, config: TrainingConfig) -> nn.Module:
        if config.model.arch != 'resnet32':
            raise ConfigError(
                f"unsupported arch '{config.model.arch}' (only 'resnet32')"
            )
        if config.model.num_classes != EXPECTED_NUM_CLASSES:
            raise ConfigError(
                f"num_classes must be {EXPECTED_NUM_CLASSES} for CIFAR-100, "
                f"got {config.model.num_classes}"
            )
        return ResNet32(num_classes=config.model.num_classes)

    def build_loss(
        self, config: TrainingConfig, class_counts: np.ndarray | None
    ) -> nn.Module | None:
        return None

    @staticmethod
    def _require_counts(
        class_counts: np.ndarray | None, expert: str
    ) -> np.ndarray:
        if class_counts is None:
            raise ConfigError(
                f"expert '{expert}' needs class counts to build its loss; "
                f"none were supplied"
            )
        counts = np.asarray(class_counts, dtype=np.float64)
        if counts.shape != (EXPECTED_NUM_CLASSES,):
            raise ConfigError(
                f"class_counts must have {EXPECTED_NUM_CLASSES} entries, "
                f"got shape {counts.shape}"
            )
        return counts

    def train(self, train_loader: DataLoader) -> list[dict]:  # noqa: D102 - inherited
        return super().train(train_loader)


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------

@TrainerRegistry.register('ce')
class CETrainer(ConfigDrivenTrainer):
    """Standard cross-entropy — the ERM baseline."""

    expert_name = 'CE'

    def build_loss(self, config, class_counts):
        return CELoss()

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        if weights is not None:
            unreduced = F.cross_entropy(logits, targets, reduction='none')
            loss = (unreduced * weights).mean()
        else:
            loss = self.loss_fn(logits, targets)
        return loss, logits, {}


@TrainerRegistry.register('logit_adjusted')
class LALTrainer(ConfigDrivenTrainer):
    """Logit-adjusted loss — Menon et al., ICLR 2021 (tau from the config)."""

    expert_name = 'LAL'

    def build_loss(self, config, class_counts):
        counts = self._require_counts(class_counts, config.expert)
        if config.loss.tau is None:
            raise ConfigError("expert 'logit_adjusted' requires loss.tau")
        priors = torch.tensor(counts / counts.sum(), dtype=torch.float32)
        return LALLoss(class_priors=priors, tau=config.loss.tau)

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        if weights is not None:
            adjusted = logits + self.loss_fn.tau * self.loss_fn.log_prior.unsqueeze(0)
            unreduced = F.cross_entropy(adjusted, targets, reduction='none')
            loss = (unreduced * weights).mean()
        else:
            loss = self.loss_fn(logits, targets)
        return loss, logits, {}


@TrainerRegistry.register('balanced_softmax')
class BalancedSoftmaxTrainer(ConfigDrivenTrainer):
    """Balanced Softmax — Ren et al., NeurIPS 2020."""

    expert_name = 'BalancedSoftmax'

    def build_loss(self, config, class_counts):
        counts = self._require_counts(class_counts, config.expert)
        return BalancedSoftmaxLoss(class_counts=torch.tensor(counts, dtype=torch.float32))

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        if weights is not None:
            adjusted = logits + self.loss_fn.log_counts.unsqueeze(0)
            unreduced = F.cross_entropy(adjusted, targets, reduction='none')
            loss = (unreduced * weights).mean()
        else:
            loss = self.loss_fn(logits, targets)
        return loss, logits, {}


@TrainerRegistry.register('mixup')
class MixupTrainer(ConfigDrivenTrainer):
    """Mixup + cross-entropy — Zhang et al., ICLR 2018 (alpha from the config)."""

    expert_name = 'Mixup'
    #: Logits come from mixed inputs, so training accuracy would be meaningless.
    reports_train_accuracy = False

    def build_loss(self, config, class_counts):
        if config.loss.alpha is None:
            raise ConfigError("expert 'mixup' requires loss.alpha")
        self.mixup = Mixup(alpha=config.loss.alpha)
        return CELoss()

    @property
    def mixup_alpha(self) -> float:
        """The Beta mixing coefficient, sourced from the config."""
        return self.mixup.alpha

    def _compute_loss(self, images, targets, weights=None):
        mixed, targets_a, targets_b, lam = self.mixup(images, targets)
        logits = self.model(mixed)
        loss = Mixup.criterion(self.loss_fn, logits, targets_a, targets_b, lam)
        return loss, logits, {'lam': lam}
