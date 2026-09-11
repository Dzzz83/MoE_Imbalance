"""
Tests for the config-driven training path.

Every expert is described by its own YAML config file holding all of its
parameters. These tests enforce that the configs exist, that they encode the
published CIFAR-LT recipe, that the schema rejects typos and missing sections,
and that the registry turns a config into the right trainer.
"""

import os
import sys
import tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch

import importlib

# Imported as modules (not names) so a not-yet-written module fails one test at a
# time instead of aborting the whole file at import time.
def _safe(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


cfg_mod = _safe('scripts.config')
trainers_mod = _safe('scripts.trainers')
train_mod = _safe('scripts.train')

CONFIG_DIR = os.path.join(_proj_root, 'configs')

#: expert key -> (config filename, expected trainer class name, expected checkpoint label)
EXPECTED = {
    'ce': ('ce.yaml', 'CETrainer', 'CE'),
    'logit_adjusted': ('lal.yaml', 'LALTrainer', 'LAL'),
    'balanced_softmax': ('balanced_softmax.yaml', 'BalancedSoftmaxTrainer', 'BalancedSoftmax'),
    'mixup': ('mixup.yaml', 'MixupTrainer', 'Mixup'),
}


def _counts():
    return np.array([500] + [5] * 99, dtype=np.int64)


# ---------------------------------------------------------------------------
# Config files exist and parse
# ---------------------------------------------------------------------------

def test_all_expert_configs_exist():
    """Each model must have its own config file."""
    missing = [f for f, _, _ in EXPECTED.values()
               if not os.path.exists(os.path.join(CONFIG_DIR, f))]
    assert not missing, f"missing config files: {missing}"
    print(f"  ✅ all {len(EXPECTED)} expert configs present")


def test_configs_parse_into_training_config():
    """Every config file must load into a TrainingConfig."""
    for key, (fname, _, _) in EXPECTED.items():
        cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, fname))
        assert isinstance(cfg, cfg_mod.TrainingConfig)
        assert cfg.expert == key, f"{fname}: expert '{cfg.expert}' != '{key}'"
        print(f"  ✅ {fname} -> expert '{cfg.expert}'")


def test_registry_covers_exactly_the_config_experts():
    """The registry keys and the config set must agree."""
    keys = set(trainers_mod.TrainerRegistry.keys())
    assert keys == set(EXPECTED), f"registry {sorted(keys)} != {sorted(EXPECTED)}"
    print(f"  ✅ registry keys: {sorted(keys)}")


# ---------------------------------------------------------------------------
# The configs encode the published recipe
# ---------------------------------------------------------------------------

def test_configs_encode_published_schedule():
    """200 epochs, warmup 5, x0.01 at 160, x1e-4 at 180 - in every config."""
    for fname, _, _ in EXPECTED.values():
        cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, fname))
        s = cfg.schedule
        assert s.epochs == 200, f"{fname}: epochs {s.epochs}"
        assert s.warmup_epochs == 5, f"{fname}: warmup {s.warmup_epochs}"
        assert tuple(s.decay_epochs) == (160, 180), f"{fname}: decay_epochs {s.decay_epochs}"
        assert tuple(s.decay_factors) == (0.01, 0.0001), f"{fname}: factors {s.decay_factors}"
        print(f"  ✅ {fname}: 200 ep, warmup 5, decay 160/180")


def test_configs_encode_published_optimiser():
    """SGD momentum 0.9, lr 0.1, wd 2e-4, no nesterov - in every config."""
    for fname, _, _ in EXPECTED.values():
        cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, fname))
        o = cfg.optimiser
        assert o.name == 'sgd', f"{fname}: optimiser {o.name}"
        assert o.lr == 0.1, f"{fname}: lr {o.lr}"
        assert o.momentum == 0.9, f"{fname}: momentum {o.momentum}"
        assert abs(o.weight_decay - 2e-4) < 1e-12, f"{fname}: wd {o.weight_decay}"
        assert o.nesterov is False, f"{fname}: nesterov must be False"
        print(f"  ✅ {fname}: SGD lr 0.1 mom 0.9 wd 2e-4 nesterov False")


def _all_keys(obj) -> set:
    """Every key appearing anywhere in a nested mapping/list."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


def test_configs_declare_no_validation():
    """No config may declare a validation split (checked on keys, not comments)."""
    import yaml

    banned = {'val', 'validation', 'val_split', 'val_indices', 'val_loader',
              'validation_split'}
    for fname, _, _ in EXPECTED.values():
        with open(os.path.join(CONFIG_DIR, fname)) as f:
            raw = yaml.safe_load(f)
        found = _all_keys(raw) & banned
        assert not found, f"{fname} declares validation key(s): {sorted(found)}"
    print("  ✅ no config declares a validation split")


def test_checkpoint_policy_in_configs():
    """save_from_epoch 160 and save_every 20 in every config."""
    for fname, _, _ in EXPECTED.values():
        cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, fname))
        assert cfg.checkpoint.save_from_epoch == 160, fname
        assert cfg.checkpoint.save_every == 20, fname
    print("  ✅ checkpoint policy: every 20 from 160 + final")


def test_loss_hyperparameters_live_in_their_own_config():
    """LA carries tau=1.0; Mixup carries alpha=1.0; CE carries neither."""
    la = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'lal.yaml'))
    mu = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'mixup.yaml'))
    ce = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    assert la.loss.tau == 1.0, f"LA tau {la.loss.tau}"
    assert mu.loss.alpha == 1.0, f"Mixup alpha {mu.loss.alpha}"
    assert ce.loss.tau is None and ce.loss.alpha is None, "CE should set no loss hyperparams"
    print("  ✅ tau=1.0 (LA), alpha=1.0 (Mixup), none (CE)")


def test_each_expert_has_a_distinct_config():
    """The four configs must not be the same file content."""
    bodies = []
    for fname, _, _ in EXPECTED.values():
        with open(os.path.join(CONFIG_DIR, fname)) as f:
            bodies.append(f.read())
    assert len(set(bodies)) == len(bodies), "duplicate config files"
    print("  ✅ four distinct config files")


# ---------------------------------------------------------------------------
# Schema strictness
# ---------------------------------------------------------------------------

def _write_yaml(text):
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    return path


VALID = """
expert: ce
seed: 0
model: {arch: resnet32, num_classes: 100}
loss: {}
optimiser: {name: sgd, lr: 0.1, momentum: 0.9, weight_decay: 2.0e-4, nesterov: false}
schedule: {epochs: 200, warmup_epochs: 5, decay_epochs: [160, 180], decay_factors: [0.01, 0.0001]}
data: {root: ./data, imbalance_ratio: 100.0, batch_size: 128, num_workers: 2, pin_memory: true}
checkpoint: {dir: ./checkpoints, save_from_epoch: 160, save_every: 20}
"""


def test_valid_config_round_trips():
    """A well-formed config parses."""
    cfg = cfg_mod.TrainingConfig.from_file(_write_yaml(VALID))
    assert cfg.expert == 'ce' and cfg.schedule.epochs == 200
    print("  ✅ valid config parses")


def test_unknown_key_is_rejected():
    """A typo in the config must fail loudly, not be silently ignored."""
    bad = VALID.replace('epochs: 200', 'epochz: 200')
    try:
        cfg_mod.TrainingConfig.from_file(_write_yaml(bad))
    except cfg_mod.ConfigError as e:
        assert 'epochz' in str(e), str(e)
        print(f"  ✅ unknown key rejected: {str(e)[:80]}")
        return
    raise AssertionError("an unknown key was accepted")


def test_missing_section_is_rejected():
    """A missing required section must fail loudly."""
    bad = VALID.replace('checkpoint: {dir: ./checkpoints, save_from_epoch: 160, save_every: 20}\n', '')
    try:
        cfg_mod.TrainingConfig.from_file(_write_yaml(bad))
    except cfg_mod.ConfigError as e:
        assert 'checkpoint' in str(e), str(e)
        print(f"  ✅ missing section rejected: {str(e)[:80]}")
        return
    raise AssertionError("a missing section was accepted")


def test_unknown_expert_is_rejected():
    """An expert with no registered trainer must fail loudly."""
    bad = VALID.replace('expert: ce', 'expert: nonexistent')
    try:
        cfg_mod.TrainingConfig.from_file(_write_yaml(bad))
    except cfg_mod.ConfigError as e:
        assert 'nonexistent' in str(e), str(e)
        print(f"  ✅ unknown expert rejected: {str(e)[:80]}")
        return
    raise AssertionError("an unknown expert was accepted")


def test_non_canonical_imbalance_ratio_is_rejected():
    """The project fixes the imbalance factor at 0.01 (IR=100)."""
    bad = VALID.replace('imbalance_ratio: 100.0', 'imbalance_ratio: 50.0')
    try:
        cfg_mod.TrainingConfig.from_file(_write_yaml(bad))
    except cfg_mod.ConfigError as e:
        print(f"  ✅ non-canonical IR rejected: {str(e)[:80]}")
        return
    raise AssertionError("a non-canonical imbalance ratio was accepted")


def test_missing_loss_hyperparameter_is_rejected():
    """LA without tau must fail loudly."""
    la = open(os.path.join(CONFIG_DIR, 'lal.yaml')).read().replace('tau: 1.0', '')
    path = _write_yaml(la)
    try:
        t = trainers_mod.build_trainer(
            cfg_mod.TrainingConfig.from_file(path), _counts(), device='cpu'
        )
        t = None
    except cfg_mod.ConfigError as e:
        print(f"  ✅ LA without tau rejected: {str(e)[:80]}")
        return
    raise AssertionError("LA without tau was accepted")


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

def test_registry_builds_expected_trainer_classes():
    """Each expert key must build its own trainer class."""
    for key, (fname, class_name, label) in EXPECTED.items():
        cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, fname))
        t = trainers_mod.build_trainer(cfg, _counts(), device='cpu')
        assert type(t).__name__ == class_name, f"{key}: got {type(t).__name__}"
        assert t.expert_name == label, f"{key}: label '{t.expert_name}' != '{label}'"
        print(f"  ✅ {key} -> {class_name} (label {label})")


def test_trainer_takes_settings_from_config():
    """The built trainer must carry the config's optimiser and schedule."""
    cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    t = trainers_mod.build_trainer(cfg, _counts(), device='cpu')
    pg = t.optimiser.param_groups[0]
    assert t.epochs == cfg.schedule.epochs == 200
    assert t.warmup_epochs == cfg.schedule.warmup_epochs
    assert tuple(t.decay_epochs) == tuple(cfg.schedule.decay_epochs)
    assert pg['lr'] == cfg.optimiser.lr
    assert pg['momentum'] == cfg.optimiser.momentum
    assert abs(pg['weight_decay'] - cfg.optimiser.weight_decay) < 1e-15
    assert pg['nesterov'] == cfg.optimiser.nesterov
    assert t.save_from_epoch == cfg.checkpoint.save_from_epoch
    assert t.save_every == cfg.checkpoint.save_every
    print("  ✅ trainer settings come from the config")


def test_mixup_trainer_gets_alpha_from_config():
    """Mixup's alpha must be sourced from its config, not a hardcoded default."""
    cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'mixup.yaml'))
    t = trainers_mod.build_trainer(cfg, _counts(), device='cpu')
    assert t.mixup_alpha == cfg.loss.alpha == 1.0
    print("  ✅ mixup_alpha comes from config")


def test_lal_trainer_gets_tau_from_config():
    """LA's tau must be sourced from its config."""
    cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'lal.yaml'))
    t = trainers_mod.build_trainer(cfg, _counts(), device='cpu')
    assert t.loss_fn.tau == cfg.loss.tau == 1.0
    print("  ✅ LA tau comes from config")


def test_overrides_do_not_mutate_original_config():
    """CLI overrides must produce a new config object."""
    cfg = cfg_mod.TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    new = cfg.replace(seed=99, device='cpu', epochs=3)
    assert new.seed == 99 and new.schedule.epochs == 3
    assert cfg.seed == 0 and cfg.schedule.epochs == 200, "original config was mutated"
    print("  ✅ replace() returns a new config")


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end_and_writes_final_checkpoint():
    """`train.py --config ...` must train and write a final checkpoint."""
    out = tempfile.mkdtemp(prefix='dsh_cli_')
    rc = train_mod.main([
        '--config', os.path.join(CONFIG_DIR, 'ce.yaml'),
        '--device', 'cpu',
        '--epochs', '1',
        '--seed', '3',
        '--max-batches', '2',
        '--checkpoint-dir', out,
    ])
    assert rc == 0, f"CLI returned {rc}"
    files = sorted(os.listdir(out))
    assert any(f.endswith('_final.pt') for f in files), f"no final checkpoint: {files}"
    assert any('CE' in f and '3' in f for f in files), f"label/seed missing: {files}"
    print(f"  ✅ CLI wrote {files}")


TESTS = [
    ("All expert configs exist", test_all_expert_configs_exist),
    ("Configs parse into TrainingConfig", test_configs_parse_into_training_config),
    ("Registry covers config experts", test_registry_covers_exactly_the_config_experts),
    ("Configs encode published schedule", test_configs_encode_published_schedule),
    ("Configs encode published optimiser", test_configs_encode_published_optimiser),
    ("Configs declare no validation", test_configs_declare_no_validation),
    ("Checkpoint policy in configs", test_checkpoint_policy_in_configs),
    ("Loss hyperparameters per config", test_loss_hyperparameters_live_in_their_own_config),
    ("Configs are distinct", test_each_expert_has_a_distinct_config),
    ("Valid config round-trips", test_valid_config_round_trips),
    ("Unknown key rejected", test_unknown_key_is_rejected),
    ("Missing section rejected", test_missing_section_is_rejected),
    ("Unknown expert rejected", test_unknown_expert_is_rejected),
    ("Non-canonical IR rejected", test_non_canonical_imbalance_ratio_is_rejected),
    ("Missing loss param rejected", test_missing_loss_hyperparameter_is_rejected),
    ("Registry builds trainer classes", test_registry_builds_expected_trainer_classes),
    ("Trainer settings from config", test_trainer_takes_settings_from_config),
    ("Mixup alpha from config", test_mixup_trainer_gets_alpha_from_config),
    ("LA tau from config", test_lal_trainer_gets_tau_from_config),
    ("Overrides are immutable", test_overrides_do_not_mutate_original_config),
    ("CLI end-to-end + final checkpoint", test_cli_runs_end_to_end_and_writes_final_checkpoint),
]


def main() -> int:
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 - surface the real cause
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
