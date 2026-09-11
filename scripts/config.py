"""
Typed configuration for expert training.

Every expert is fully described by one YAML file under ``configs/``; the value
objects here are the schema for those files. Loading is strict on purpose:

    * unknown keys raise, so a typo like ``epochz`` cannot be silently ignored
    * missing sections raise
    * an unknown expert raises
    * loss hyperparameters that do not belong to an expert raise (e.g. ``tau`` on CE)
    * a non-canonical imbalance ratio raises (the project fixes it at 0.01 => IR 100)

The dataclasses are frozen: overrides return a new config rather than mutating one,
so the config a run started from can always be printed and recorded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Expert keys that have a registered trainer (see ``scripts/trainers.py``).
#: Kept here rather than imported from the registry to avoid a circular import;
#: a test asserts the two sets agree.
SUPPORTED_EXPERTS: frozenset[str] = frozenset(
    {'ce', 'logit_adjusted', 'balanced_softmax', 'mixup'}
)

#: The project fixes the imbalance factor at 0.01 (IR=100); other values would
#: break comparability with the published benchmark (AGENTs.md section 10).
CANONICAL_IMBALANCE_RATIO: float = 100.0


class ConfigError(ValueError):
    """Raised when a training config is malformed, mis-typed or inconsistent."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    arch: str = 'resnet32'
    num_classes: int = 100


@dataclass(frozen=True)
class LossConfig:
    """Loss hyperparameters. Which ones are mandatory depends on the expert."""

    tau: float | None = None      # logit-adjusted loss only
    alpha: float | None = None    # mixup only


@dataclass(frozen=True)
class OptimiserConfig:
    name: str = 'sgd'
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 2e-4
    nesterov: bool = False


@dataclass(frozen=True)
class ScheduleConfig:
    epochs: int = 200
    warmup_epochs: int = 5
    decay_epochs: tuple[int, int] = (160, 180)
    decay_factors: tuple[float, float] = (0.01, 0.0001)


@dataclass(frozen=True)
class DataConfig:
    root: str = './data'
    imbalance_ratio: float = CANONICAL_IMBALANCE_RATIO
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str = './checkpoints'


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Everything one training run needs, loaded from a single YAML file."""

    expert: str
    seed: int = 0
    device: str = 'auto'
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # ── loading ───────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> 'TrainingConfig':
        """Load and validate a config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = '<dict>') -> 'TrainingConfig':
        """Validate a raw mapping and build a config."""
        sections = {
            'model': ModelConfig,
            'loss': LossConfig,
            'optimiser': OptimiserConfig,
            'schedule': ScheduleConfig,
            'data': DataConfig,
            'checkpoint': CheckpointConfig,
        }
        allowed = {'expert', 'seed', 'device'} | set(sections)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(
                f"{source}: unknown key(s) {unknown}; allowed top-level keys are "
                f"{sorted(allowed)}"
            )
        if 'expert' not in raw:
            raise ConfigError(f"{source}: missing required key 'expert'")

        built: dict[str, Any] = {}
        for name, klass in sections.items():
            if name not in raw:
                raise ConfigError(f"{source}: missing required section '{name}'")
            built[name] = _build_section(klass, raw[name], f"{source}:{name}")

        cfg = cls(
            expert=raw['expert'],
            seed=raw.get('seed', cls.seed),
            device=raw.get('device', cls.device),
            **built,
        )
        cfg.validate(source)
        return cfg

    # ── validation ────────────────────────────────────────────────────

    def validate(self, source: str = '<config>') -> None:
        """Cross-field checks that individual sections cannot make alone."""
        if self.expert not in SUPPORTED_EXPERTS:
            raise ConfigError(
                f"{source}: unknown expert '{self.expert}'; supported: "
                f"{sorted(SUPPORTED_EXPERTS)}"
            )
        if abs(self.data.imbalance_ratio - CANONICAL_IMBALANCE_RATIO) > 1e-9:
            raise ConfigError(
                f"{source}: imbalance_ratio must be {CANONICAL_IMBALANCE_RATIO} "
                f"(imb_factor 0.01), got {self.data.imbalance_ratio}"
            )

        # Loss hyperparameters must match the expert: required where meaningful,
        # rejected where they would be silently ignored.
        if self.expert == 'logit_adjusted':
            if self.loss.tau is None:
                raise ConfigError(
                    f"{source}: expert 'logit_adjusted' requires loss.tau "
                    f"(published value 1.0)"
                )
        elif self.loss.tau is not None:
            raise ConfigError(
                f"{source}: loss.tau is set but expert '{self.expert}' does not use it"
            )

        if self.expert == 'mixup':
            if self.loss.alpha is None:
                raise ConfigError(
                    f"{source}: expert 'mixup' requires loss.alpha "
                    f"(published value 1.0)"
                )
        elif self.loss.alpha is not None:
            raise ConfigError(
                f"{source}: loss.alpha is set but expert '{self.expert}' does not use it"
            )

        if self.schedule.decay_epochs[0] >= self.schedule.decay_epochs[1]:
            raise ConfigError(
                f"{source}: decay_epochs must be increasing, got {self.schedule.decay_epochs}"
            )
        if self.schedule.warmup_epochs >= self.schedule.decay_epochs[0]:
            raise ConfigError(
                f"{source}: warmup_epochs must precede the first decay"
            )
        if self.optimiser.name != 'sgd':
            raise ConfigError(
                f"{source}: only 'sgd' is supported, got '{self.optimiser.name}'"
            )
        if self.optimiser.nesterov:
            raise ConfigError(
                f"{source}: nesterov must be False — the reference CIFAR-LT recipe "
                f"uses plain momentum SGD"
            )

    # ── derived / overrides ───────────────────────────────────────────

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to cuda when available, else cpu."""
        if self.device != 'auto':
            return self.device
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def replace(
        self,
        *,
        seed: int | None = None,
        device: str | None = None,
        epochs: int | None = None,
        checkpoint_dir: str | None = None,
    ) -> 'TrainingConfig':
        """Return a new config with the given CLI/override values applied."""
        schedule = self.schedule
        if epochs is not None:
            schedule = dataclasses.replace(schedule, epochs=epochs)
        checkpoint = self.checkpoint
        if checkpoint_dir is not None:
            checkpoint = dataclasses.replace(checkpoint, dir=checkpoint_dir)
        return dataclasses.replace(
            self,
            seed=self.seed if seed is None else seed,
            device=self.device if device is None else device,
            schedule=schedule,
            checkpoint=checkpoint,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, for recording the effective config of a run."""
        out = dataclasses.asdict(self)
        out['device'] = self.device
        return out


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------

def _build_section(klass: type, data: Any, path: str) -> Any:
    """Build a flat frozen dataclass section, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")

    names = {f.name for f in dataclasses.fields(klass)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}; allowed: {sorted(names)}"
        )
    kwargs = dict(data)
    for key in ('decay_epochs', 'decay_factors'):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(kwargs[key])
    try:
        return klass(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
