"""Strict schemas and scientific hashing for study and runtime YAML files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


STUDY_SCHEMA_VERSION = "expert_method.study.v1"
PROFILE_SCHEMA_VERSION = "expert_method.runtime_profile.v1"
_EXPERT_KEYS = ("ce", "logit_adjusted", "balanced_softmax", "mixup")


class ConfigError(ValueError):
    """Raised when a study or runtime profile violates its declared schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ConfigError(f"duplicate YAML mapping key: {key!r}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class StudyDefinition:
    """Validated scientific protocol and its resolved expert recipes."""

    study_id: str
    protocol: Mapping[str, Any]
    resolved_expert_configs: Mapping[str, Mapping[str, Any]]
    expert_config_paths: Mapping[str, str]
    canonical_scientific_json: str
    scientific_sha256: str
    source_path: Path

    def to_study_config(self) -> Any:
        """Construct the executable Ridge/Sinkhorn config from this YAML.

        The YAML validator owns the human-facing schema; the legacy study
        implementation remains the execution model. Mapping every executable
        field here keeps that model driven by the loaded scientific file.
        """
        from expert_method.ridge_sinkhorn.three_seed_study import SCHEMA_VERSION, StudyConfig

        grids = self.protocol["grids"]
        bootstrap = self.protocol["bootstrap"]
        return StudyConfig(
            seeds=tuple(self.protocol["training_seeds"]),
            outer_folds=tuple(range(self.protocol["outer_folds"])),
            inner_folds=tuple(range(self.protocol["inner_folds"])),
            expert_order=tuple(self.protocol["expert_order"]),
            contribution_alphas=tuple(grids["contribution_alpha"]),
            gammas=tuple(grids["gamma"]),
            contribution_temperatures=tuple(grids["contribution_temperature"]),
            contribution_shrinkages=tuple(grids["contribution_shrinkage"]),
            residual_penalties=tuple(grids["residual_penalty"]),
            residual_alphas=tuple(grids["residual_alpha"]),
            residual_scales=tuple(grids["residual_scale"]),
            sinkhorn_rhos=tuple(grids["sinkhorn_rho"]),
            selective_taus=tuple(grids["selective_tau"]),
            priors=tuple(
                (name, tuple(values)) for name, values in self.protocol["priors"].items()
            ),
            metric_names=tuple(self.protocol["metrics"]),
            bootstrap_seed=bootstrap["seed"],
            bootstrap_replicates=bootstrap["replicates"],
            # Schema version is structural metadata, not a scientific choice;
            # pin it explicitly rather than inheriting the dataclass default.
            schema_version=SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class RuntimeProfile:
    """Machine-specific paths and execution limits, separate from science."""

    profile_id: str
    data_root: str
    run_root: str
    reuse_roots: Mapping[str, str]
    device: str
    shard_index: int
    shard_count: int
    max_jobs: int
    bundle_inputs: tuple[str, ...]
    bundle_output_dir: str
    source_path: Path


def load_study(path: str | Path) -> StudyDefinition:
    """Load a strict study file and hash its resolved scientific contents.

    Paths and devices from expert recipes are normalized away before hashing;
    model, loss, optimizer, schedule, and data-ordering parameters remain part
    of the scientific identity.
    """
    study_path = Path(path).expanduser().resolve()
    raw = _read_yaml(study_path)
    _keys(raw, {"schema_version", "study_id", "protocol", "expert_configs"}, "study")
    if raw["schema_version"] != STUDY_SCHEMA_VERSION:
        raise ConfigError(f"unsupported study schema_version: {raw['schema_version']!r}")
    study_id = _string(raw["study_id"], "study.study_id")
    if study_id != "ridge_sinkhorn_3seed_v1":
        raise ConfigError("study.study_id is frozen at 'ridge_sinkhorn_3seed_v1'")

    protocol = _validate_protocol(raw["protocol"])
    config_refs = raw["expert_configs"]
    if not isinstance(config_refs, dict):
        raise ConfigError("study.expert_configs must be a mapping")
    _keys(config_refs, set(_EXPERT_KEYS), "study.expert_configs")

    repository_root = _repository_root(study_path)
    resolved_recipes: dict[str, Mapping[str, Any]] = {}
    resolved_paths: dict[str, str] = {}
    for expert in _EXPERT_KEYS:
        reference = _string(config_refs[expert], f"study.expert_configs.{expert}")
        ref_path = Path(reference)
        if ref_path.is_absolute():
            raise ConfigError(f"expert config reference must be relative: {expert}")
        candidate = (repository_root / ref_path).resolve()
        if not candidate.is_relative_to(repository_root) or candidate.is_symlink() or not candidate.is_file():
            raise ConfigError(f"expert config for {expert} must resolve to a regular file inside the repository")
        try:
            from scripts.config import ConfigError as TrainingConfigError
            from scripts.config import TrainingConfig

            training_config = TrainingConfig.from_file(candidate)
        except (TrainingConfigError, OSError) as exc:
            raise ConfigError(f"cannot resolve expert config {expert}: {exc}") from exc
        if training_config.expert != expert:
            raise ConfigError(
                f"expert config {expert} declares expert {training_config.expert!r}"
            )
        if training_config.schedule.epochs != protocol["epochs"]:
            raise ConfigError(f"expert config {expert} does not use the frozen epoch count")
        resolved_recipes[expert] = _freeze_mapping(_scientific_recipe(training_config.to_dict()))
        resolved_paths[expert] = candidate.relative_to(repository_root).as_posix()

    scientific_payload = {
        "study_id": study_id,
        "protocol": _thaw(protocol),
        "resolved_expert_configs": {
            expert: _thaw(resolved_recipes[expert]) for expert in _EXPERT_KEYS
        },
    }
    canonical = _canonical_json(scientific_payload)
    return StudyDefinition(
        study_id=study_id,
        protocol=protocol,
        resolved_expert_configs=MappingProxyType(resolved_recipes),
        expert_config_paths=MappingProxyType(resolved_paths),
        canonical_scientific_json=canonical,
        scientific_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source_path=study_path,
    )


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    """Load and validate execution paths, device, and batch limits."""
    profile_path = Path(path).expanduser().resolve()
    raw = _read_yaml(profile_path)
    _keys(
        raw,
        {
            "schema_version", "profile_id", "data_root", "run_root", "reuse_roots",
            "device", "shard_index", "shard_count", "max_jobs", "bundle_inputs",
            "bundle_output_dir",
        },
        "runtime profile",
    )
    if raw["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ConfigError(f"unsupported runtime profile schema_version: {raw['schema_version']!r}")
    profile_id = _string(raw["profile_id"], "profile.profile_id")
    device = _string(raw["device"], "profile.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("profile.device must be one of: auto, cpu, cuda")
    shard_index = _integer(raw["shard_index"], "profile.shard_index", minimum=0)
    shard_count = _integer(raw["shard_count"], "profile.shard_count", minimum=1)
    if shard_index >= shard_count:
        raise ConfigError("profile.shard_index must be less than shard_count")
    max_jobs = _integer(raw["max_jobs"], "profile.max_jobs", minimum=1)

    reuse = raw["reuse_roots"]
    if not isinstance(reuse, dict):
        raise ConfigError("profile.reuse_roots must be a mapping of logical names to paths")
    reuse_roots: dict[str, str] = {}
    for name, value in reuse.items():
        if not isinstance(name, str) or not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ConfigError(f"invalid reuse root name: {name!r}")
        reuse_roots[name] = _string(value, f"profile.reuse_roots.{name}")

    bundle_inputs = raw["bundle_inputs"]
    if not isinstance(bundle_inputs, list) or any(not isinstance(value, str) or not value for value in bundle_inputs):
        raise ConfigError("profile.bundle_inputs must be a list of non-empty path strings")
    return RuntimeProfile(
        profile_id=profile_id,
        data_root=_string(raw["data_root"], "profile.data_root"),
        run_root=_string(raw["run_root"], "profile.run_root"),
        reuse_roots=MappingProxyType(reuse_roots),
        device=device,
        shard_index=shard_index,
        shard_count=shard_count,
        max_jobs=max_jobs,
        bundle_inputs=tuple(bundle_inputs),
        bundle_output_dir=_string(raw["bundle_output_dir"], "profile.bundle_output_dir"),
        source_path=profile_path,
    )


def _validate_protocol(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("study.protocol must be a mapping")
    _keys(
        value,
        {
            "dataset", "imbalance_ratio", "training_samples", "num_classes", "training_seeds",
            "fold_generation_seed", "outer_folds", "inner_folds", "fold_algorithm",
            "expert_order", "epochs", "checkpoint_selection", "grids",
            "priors", "metrics", "bootstrap", "methods", "controls", "selection",
            "outer_population_note",
        },
        "study.protocol",
    )
    seeds = _int_list(value["training_seeds"], "protocol.training_seeds")
    experts = _string_list(value["expert_order"], "protocol.expert_order")
    if seeds != (78, 88, 1034) or experts != ("CE", "LAL", "BalancedSoftmax", "Mixup"):
        raise ConfigError("study seeds and expert order are frozen by the protocol")
    fold_generation_seed = _integer(value["fold_generation_seed"], "protocol.fold_generation_seed", minimum=0)
    dataset = _string(value["dataset"], "protocol.dataset")
    imbalance_ratio = value["imbalance_ratio"]
    if isinstance(imbalance_ratio, bool) or not isinstance(imbalance_ratio, (int, float)) or float(imbalance_ratio) != 100.0:
        raise ConfigError("protocol.imbalance_ratio is frozen at 100")
    training_samples = _integer(value["training_samples"], "protocol.training_samples", minimum=1)
    num_classes = _integer(value["num_classes"], "protocol.num_classes", minimum=1)
    outer_folds = _integer(value["outer_folds"], "protocol.outer_folds", minimum=2)
    inner_folds = _integer(value["inner_folds"], "protocol.inner_folds", minimum=2)
    epochs = _integer(value["epochs"], "protocol.epochs", minimum=1)
    if (dataset, float(imbalance_ratio), training_samples, num_classes) != (
        "cifar100_lt", 100.0, 10847, 100
    ):
        raise ConfigError("dataset, imbalance ratio, and canonical training population are frozen")
    if (fold_generation_seed, outer_folds, inner_folds, epochs) != (42, 5, 4, 200):
        raise ConfigError("fold and training settings are frozen by the protocol")
    fold_algorithm = _string(value["fold_algorithm"], "protocol.fold_algorithm")
    checkpoint_selection = _string(value["checkpoint_selection"], "protocol.checkpoint_selection")
    if fold_algorithm != "numpy.default_rng.per_class_balanced_remainder.v1":
        raise ConfigError("unsupported frozen fold algorithm")
    if checkpoint_selection != "final_epoch":
        raise ConfigError("checkpoint_selection must be final_epoch")

    grids = value["grids"]
    if not isinstance(grids, dict):
        raise ConfigError("protocol.grids must be a mapping")
    expected_grids = {
        "contribution_alpha": (0.1, 10.0, 1000.0),
        "gamma": (0.0, 1.0),
        "contribution_temperature": (1.0, 2.0),
        "contribution_shrinkage": (0.5, 0.75, 1.0),
        "residual_penalty": (0.1, 1.0, 10.0),
        "residual_alpha": (0.1, 10.0, 1000.0),
        "residual_scale": (0.5, 0.75, 1.0),
        "sinkhorn_rho": (0.1, 1.0, 10.0),
        "selective_tau": (0.1, 0.25, 0.5, 1.0),
    }
    _keys(grids, set(expected_grids), "protocol.grids")
    normalized_grids = {name: _number_list(grids[name], f"protocol.grids.{name}") for name in expected_grids}
    if normalized_grids != expected_grids:
        raise ConfigError("study search grids are frozen by the protocol")

    priors = value["priors"]
    expected_priors = {
        "uniform": (0.25, 0.25, 0.25, 0.25),
        "fixed_006": (0.0125, 0.25, 0.25, 0.4875),
        "fixed_007": (0.0125, 0.25, 0.4875, 0.25),
        "fixed_010": (0.0125, 0.4875, 0.25, 0.25),
        "fixed_011": (0.0125, 0.4875, 0.4875, 0.0125),
    }
    if not isinstance(priors, dict):
        raise ConfigError("protocol.priors must be a mapping")
    _keys(priors, set(expected_priors), "protocol.priors")
    normalized_priors = {name: _number_list(priors[name], f"protocol.priors.{name}") for name in expected_priors}
    if normalized_priors != expected_priors:
        raise ConfigError("study priors must match the frozen smoothed prior grid")

    metrics = _string_list(value["metrics"], "protocol.metrics")
    if metrics != ("balanced_accuracy", "head_accuracy", "medium_accuracy", "tail_accuracy", "ordinary_accuracy"):
        raise ConfigError("study metric inventory is frozen by the protocol")
    bootstrap = value["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise ConfigError("protocol.bootstrap must be a mapping")
    _keys(bootstrap, {"kind", "seed", "replicates"}, "protocol.bootstrap")
    bootstrap_kind = _string(bootstrap["kind"], "protocol.bootstrap.kind")
    bootstrap_seed = _integer(bootstrap["seed"], "protocol.bootstrap.seed", minimum=0)
    bootstrap_replicates = _integer(bootstrap["replicates"], "protocol.bootstrap.replicates", minimum=1)
    if (bootstrap_kind, bootstrap_seed, bootstrap_replicates) != (
        "stratified_class_then_sample_paired", 20260924, 10_000
    ):
        raise ConfigError("study bootstrap settings are frozen by the protocol")

    methods = _string_list(value["methods"], "protocol.methods")
    expected_methods = (
        "contribution_ridge", "contribution_ridge_sinkhorn", "residual_ridge",
        "residual_ridge_sinkhorn", "selective_residual_ridge_sinkhorn",
    )
    controls = _string_list(value["controls"], "protocol.controls")
    expected_controls = (
        "uniform_logit", "uniform_probability", "fixed_006", "fixed_007", "fixed_010",
        "fixed_011", "residual_anchor", "prior_only_control",
    )
    if methods != expected_methods or controls != expected_controls:
        raise ConfigError("study method and control inventories are frozen by the protocol")

    selection = value["selection"]
    if not isinstance(selection, dict):
        raise ConfigError("protocol.selection must be a mapping")
    _keys(selection, {"objective", "tie_breaks"}, "protocol.selection")
    if _string(selection["objective"], "protocol.selection.objective") != "maximin_delta_balanced_accuracy_tail":
        raise ConfigError("unsupported study selection objective")
    tie_breaks = _string_list(selection["tie_breaks"], "protocol.selection.tie_breaks")
    if tie_breaks != ("balanced_accuracy", "tail_accuracy", "stronger_regularization", "less_routing_or_ot", "stable_configuration_id"):
        raise ConfigError("study selection tie breaks are frozen by the protocol")

    note = _string(value["outer_population_note"], "protocol.outer_population_note")
    if not note:
        raise ConfigError("protocol.outer_population_note must not be empty")
    return _freeze_mapping({
        "dataset": dataset,
        "imbalance_ratio": float(imbalance_ratio),
        "training_samples": training_samples,
        "num_classes": num_classes,
        "training_seeds": seeds,
        "fold_generation_seed": fold_generation_seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "fold_algorithm": fold_algorithm,
        "expert_order": experts,
        "epochs": epochs,
        "checkpoint_selection": checkpoint_selection,
        "grids": normalized_grids,
        "priors": normalized_priors,
        "metrics": metrics,
        "bootstrap": {"kind": bootstrap_kind, "seed": bootstrap_seed, "replicates": bootstrap_replicates},
        "methods": methods,
        "controls": controls,
        "selection": {"objective": selection["objective"], "tie_breaks": tie_breaks},
        "outer_population_note": note,
    })


def _scientific_recipe(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove placement and device values, retaining training semantics."""
    normalized = json.loads(_canonical_json(dict(payload)))
    normalized.pop("seed", None)
    normalized.pop("device", None)
    normalized.pop("resolved_device", None)
    data = normalized.get("data")
    if isinstance(data, dict):
        data["root"] = "<data-root>"
    checkpoint = normalized.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint["dir"] = "<run-checkpoint-dir>"
    return normalized


def _repository_root(study_path: Path) -> Path:
    try:
        root = study_path.parents[2]
    except IndexError as exc:
        raise ConfigError("study config must live under configs/studies/") from exc
    if study_path.parent.name != "studies":
        raise ConfigError("study config must live under configs/studies/")
    return root.resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfigError(f"configuration file is missing or not a regular file: {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read YAML file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ConfigError(f"{name} has unknown key(s): {unknown}")
    if missing:
        raise ConfigError(f"{name} is missing required key(s): {missing}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _int_list(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list of integers")
    return tuple(_integer(item, name, minimum=0) for item in value)


def _number_list(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list of numbers")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{name} must contain only numbers")
        output.append(float(item))
    return tuple(output)


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConfigError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError("configuration contains values that cannot be canonically serialized") from exc


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({
        key: _freeze_mapping(item) if isinstance(item, dict) else tuple(item) if isinstance(item, list) else item
        for key, item in value.items()
    })


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
