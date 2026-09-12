"""
Tests for the published CIFAR-LT training protocol.

Protocol enforced here:
    - 200 epochs, SGD (momentum 0.9, weight_decay 2e-4, no nesterov), batch 128
    - linear warmup over the first 5 epochs from 0 to base_lr
    - base_lr for epochs 6..160, x0.01 for 161..180, x0.0001 for 181..200
    - NO validation split, NO early stopping, NO val-based selection
    - checkpoints every 20 epochs from 160, plus the final epoch (the reported model)
    - NaN/Inf in logits or loss raises immediately

Reference: Cao et al. (LDAM-DRW) `cifar_train.py` `adjust_learning_rate`.
"""

import glob
import inspect
import os
import sys
import tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Imported as modules (not names) so a missing symbol fails one test, not the file.
from scripts import base_trainer as bt

EXPERT_SCRIPTS = [
    'scripts/train.py',
    'scripts/trainers.py',
    'scripts/config.py',
    'data/lt_datamodule.py',
    'data/mixup.py',
]

#: The one place allowed (and required) to read the canonical index artifact.
CANONICAL_LOADER_OWNER = 'data/lt_datamodule.py'


def _tiny_loader(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=g)
    targets = torch.randint(0, 100, (n,), generator=g)
    return DataLoader(TensorDataset(images, targets), batch_size=4)


class _TinyTrainer(bt.BaseTrainer):
    """Minimal concrete trainer used to exercise the loop on CPU."""

    def __init__(self, **kwargs):
        super().__init__(model=torch.nn.Linear(3 * 32 * 32, 100), expert_name='Tiny', **kwargs)

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images.flatten(1))
        return F.cross_entropy(logits, targets), logits, {}


class _NaNLossTrainer(_TinyTrainer):
    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images.flatten(1))
        return torch.full((), float('nan'), requires_grad=True), logits, {}


class _NaNLogitsTrainer(_TinyTrainer):
    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images.flatten(1))
        logits = logits.clone()
        logits[0, 0] = float('inf')
        return F.cross_entropy(logits, targets), logits, {}


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def test_lr_schedule_matches_reference():
    """lr must match LDAM-DRW's adjust_learning_rate exactly."""
    expected = {
        1: 0.02,     # warmup: lr * 1/5
        5: 0.1,      # warmup end
        6: 0.1,      # plateau
        160: 0.1,    # last epoch at full lr
        161: 0.001,  # x0.01
        180: 0.001,  # last epoch at x0.01
        181: 1e-5,   # x0.0001
        200: 1e-5,   # final epoch
    }
    for epoch, want in expected.items():
        got = bt.step_lr(epoch, base_lr=0.1)
        assert abs(got - want) < 1e-12, f"epoch {epoch}: lr {got} != {want}"
    print("  ✅ lr schedule: warmup/plateau/160/x0.01/180/x1e-4 all match")


def test_lr_schedule_scales_with_base_lr():
    """The schedule is multiplicative in base_lr."""
    assert abs(bt.step_lr(100, base_lr=0.05) - 0.05) < 1e-12
    assert abs(bt.step_lr(170, base_lr=0.05) - 0.0005) < 1e-12
    print("  ✅ schedule scales with base_lr")


def test_defaults_are_the_published_recipe():
    """Constructor defaults must encode the published recipe."""
    p = inspect.signature(bt.BaseTrainer.__init__).parameters
    assert p['lr'].default == 0.1, p['lr'].default
    assert p['weight_decay'].default == 2e-4, p['weight_decay'].default
    assert p['momentum'].default == 0.9, p['momentum'].default
    assert p['batch_size'].default == 128, p['batch_size'].default
    assert p['epochs'].default == 200, p['epochs'].default
    assert p['warmup_epochs'].default == 5, p['warmup_epochs'].default
    assert tuple(p['decay_epochs'].default) == (160, 180), p['decay_epochs'].default
    print("  ✅ defaults: 200 ep, lr 0.1, wd 2e-4, mom 0.9, batch 128, warmup 5, 160/180")


def test_optimiser_matches_reference():
    """SGD without nesterov, reference momentum and weight decay."""
    t = _TinyTrainer(device='cpu')
    pg = t.optimiser.param_groups[0]
    assert pg['momentum'] == 0.9, pg['momentum']
    assert abs(pg['weight_decay'] - 2e-4) < 1e-12, pg['weight_decay']
    assert pg['nesterov'] is False, "nesterov must be False (LDAM-DRW uses plain SGD)"
    assert isinstance(t.optimiser, torch.optim.SGD)
    print("  ✅ optimiser: SGD mom=0.9 wd=2e-4 nesterov=False")


def test_no_cosine_scheduler():
    """The cosine scheduler from the previous implementation must be gone."""
    t = _TinyTrainer(device='cpu')
    assert not hasattr(t, 'scheduler'), "cosine scheduler still present"
    print("  ✅ no cosine scheduler attribute")


# ---------------------------------------------------------------------------
# No validation split
# ---------------------------------------------------------------------------

def test_train_accepts_only_train_loader():
    """The training entry point must not take a validation loader."""
    params = list(inspect.signature(bt.BaseTrainer.train).parameters)
    assert params == ['self', 'train_loader'], (
        f"train() takes {params} — a validation loader or class_counts is still there"
    )
    print("  ✅ train(train_loader) only")


def test_no_validation_machinery_on_trainer():
    """No val-based selection state may exist on the trainer."""
    t = _TinyTrainer(device='cpu')
    for attr in ('best_metric_val', 'val_loader'):
        assert not hasattr(t, attr), f"validation machinery still present: {attr}"
    print("  ✅ no best_metric_val / val_loader")


def test_history_has_no_validation_metrics():
    """Per-epoch logs must not contain val_* keys."""
    t = _TinyTrainer(device='cpu', epochs=2, checkpoint_dir=_tmp_ckpt('history'))
    hist = t.train(_tiny_loader())
    assert hist, "empty history"
    keys = set().union(*[set(h) for h in hist])
    bad = {k for k in keys if k.startswith('val_') or k == 'val_ba' or k == 'val_loss'}
    assert not bad, f"validation metrics still logged: {sorted(bad)}"
    print(f"  ✅ history keys: {sorted(keys)}")


def test_training_scripts_reference_only_canonical_artifact():
    """Training scripts must reference the canonical artifact and nothing removed."""
    # Removed artifact *filenames*. Note that the legitimate loader
    # `load_lt_train_indices()` and the canonical `lt_ir100_train_indices.npy`
    # must NOT trip this check, hence full filenames rather than bare substrings.
    banned = ('lt_val_indices', 'lt_all_indices', 'balanced_val_indices',
              'val_targets', 'lt_train_indices.npy', 'processed/lt_train_indices')
    offenders = {}
    for path in EXPERT_SCRIPTS + ['scripts/base_trainer.py']:
        with open(os.path.join(_proj_root, path)) as f:
            src = f.read()
        hits = [b for b in banned if b in src]
        if hits:
            offenders[path] = hits
    assert not offenders, f"still referencing removed artifacts: {offenders}"

    # The canonical index artifact must be read through the data module only.
    with open(os.path.join(_proj_root, CANONICAL_LOADER_OWNER)) as f:
        src = f.read()
    assert 'load_lt_train_indices' in src, (
        f"{CANONICAL_LOADER_OWNER} does not use load_lt_train_indices()"
    )
    print("  ✅ training path reads only the canonical artifact, via the data module")


# ---------------------------------------------------------------------------
# Checkpoint policy
# ---------------------------------------------------------------------------

def _tmp_ckpt(name):
    """A scratch checkpoint dir outside the repo, so tests never pollute it."""
    return tempfile.mkdtemp(prefix=f'dsh_ckpt_{name}_')


def test_only_the_final_checkpoint_is_written():
    """Exactly one checkpoint per run: the final epoch (the reported model).

    Milestone checkpoints (epoch 160/180) were removed: they were inspection-only,
    cost ~7.7 MB per expert, and pushing them to the repo made every future clone
    slower, since git history is permanent.
    """
    d = _tmp_ckpt('policy')
    # 180 epochs would previously have produced epoch160 + epoch180 + final;
    # only the final may exist now.
    t = _TinyTrainer(device='cpu', epochs=180, checkpoint_dir=d, seed=0)
    t.train(_tiny_loader())
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, '*.pt')))
    assert len(names) == 1, f"expected exactly 1 checkpoint, got {names}"
    assert 'final' in names[0], f"the only checkpoint must be the final one: {names}"
    print(f"  ✅ exactly one checkpoint written: {names}")


def test_no_milestone_checkpoint_knobs_remain():
    """The save_from_epoch / save_every machinery must be gone."""
    p = inspect.signature(bt.BaseTrainer.__init__).parameters
    for banned in ('save_from_epoch', 'save_every'):
        assert banned not in p, f"BaseTrainer still accepts '{banned}'"
    t = _TinyTrainer(device='cpu', epochs=1, checkpoint_dir=_tmp_ckpt('noknobs'))
    for banned in ('save_from_epoch', 'save_every', '_should_save'):
        assert not hasattr(t, banned), f"trainer still carries '{banned}'"
    print("  ✅ no milestone-checkpoint knobs remain")


def test_checkpoint_names_include_expert_and_seed():
    """Three seeds must not clobber each other."""
    d = _tmp_ckpt('seednames')
    t = _TinyTrainer(device='cpu', epochs=1, checkpoint_dir=d, seed=123)
    t.train(_tiny_loader())
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, '*.pt')))
    assert all('Tiny' in n for n in names), names
    assert all('123' in n for n in names), names
    print(f"  ✅ checkpoint names carry expert + seed: {names}")


def test_final_checkpoint_is_self_describing():
    """The saved state must record expert, seed and epoch."""
    d = _tmp_ckpt('selfdescribing')
    t = _TinyTrainer(device='cpu', epochs=2, checkpoint_dir=d, seed=7)
    t.train(_tiny_loader())
    finals = glob.glob(os.path.join(d, '*final*.pt'))
    assert finals, "no final checkpoint written"
    state = torch.load(finals[0], map_location='cpu')
    for key in ('model_state_dict', 'expert_name', 'epoch', 'seed'):
        assert key in state, f"checkpoint missing '{key}': {sorted(state)}"
    assert state['epoch'] == 2, state['epoch']
    assert state['seed'] == 7, state['seed']
    print(f"  ✅ final checkpoint self-describing: {sorted(state)}")


# ---------------------------------------------------------------------------
# NaN / Inf guards
# ---------------------------------------------------------------------------

def test_nan_loss_raises():
    """A non-finite loss must stop training immediately."""
    t = _NaNLossTrainer(device='cpu', epochs=1,
                        checkpoint_dir=_tmp_ckpt('nanloss'))
    try:
        t.train(_tiny_loader())
    except FloatingPointError as e:
        print(f"  ✅ NaN loss raised: {e}")
        return
    raise AssertionError("a NaN loss did not raise")


def test_nonfinite_logits_raise():
    """Non-finite logits must stop training immediately."""
    t = _NaNLogitsTrainer(device='cpu', epochs=1,
                          checkpoint_dir=_tmp_ckpt('nanlogits'))
    try:
        t.train(_tiny_loader())
    except FloatingPointError as e:
        print(f"  ✅ non-finite logits raised: {e}")
        return
    raise AssertionError("non-finite logits did not raise")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_seed_makes_initialisation_reproducible():
    """set_seed must make torch and numpy reproducible."""
    bt.set_seed(0)
    a = torch.randn(4)
    n_a = np.random.rand(2)
    bt.set_seed(0)
    b = torch.randn(4)
    n_b = np.random.rand(2)
    assert torch.equal(a, b), "torch RNG not reproducible"
    assert np.array_equal(n_a, n_b), "numpy RNG not reproducible"
    print("  ✅ set_seed makes torch + numpy reproducible")


def test_seed_changes_initialisation():
    """Different seeds must give different initial weights."""
    bt.set_seed(0)
    m0 = torch.nn.Linear(8, 8).weight.detach().clone()
    bt.set_seed(1)
    m1 = torch.nn.Linear(8, 8).weight.detach().clone()
    assert not torch.equal(m0, m1), "seed is not affecting initialisation"
    print("  ✅ different seed gives different init")


# ---------------------------------------------------------------------------
# Synthetic dry-run for the four real experts (AGENTs.md section 11)
# ---------------------------------------------------------------------------

def test_synthetic_dry_run_all_four_experts():
    """1 forward + 1 backward per expert, driven by the real config files."""
    from scripts.config import TrainingConfig
    from scripts.trainers import build_trainer

    counts = np.array([500] + [5] * 99, dtype=np.int64)
    configs = {
        'CE': 'ce.yaml',
        'LA': 'lal.yaml',
        'BS': 'balanced_softmax.yaml',
        'Mixup': 'mixup.yaml',
    }

    g = torch.Generator().manual_seed(0)
    images = torch.randn(4, 3, 32, 32, generator=g)
    targets = torch.randint(0, 100, (4,), generator=g)
    loader = DataLoader(TensorDataset(images, targets), batch_size=4)

    for name, fname in configs.items():
        cfg_path = os.path.join(_proj_root, 'configs', fname)
        cfg = TrainingConfig.from_file(cfg_path).replace(
            epochs=3,
            checkpoint_dir=tempfile.mkdtemp(prefix=f'dsh_dry_{name}_'),
            device='cpu',
        )
        t = build_trainer(cfg, counts, device='cpu')
        hist = t.train(loader)
        assert len(hist) == t.epochs, f"{name}: history length {len(hist)}"
        missing = [n for n, p in t.model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not missing, f"{name}: {len(missing)} params without finite grad: {missing[:5]}"
        assert np.isfinite(hist[-1]['train_loss']), f"{name}: non-finite loss"
        print(f"  ✅ {name}: loss={hist[-1]['train_loss']:.3f} lr={hist[-1]['lr']:.5f} "
              f"grads populated")


def test_seed_controls_model_initialisation():
    """Two trainers built from the same config must start from identical weights.

    Regression test for a real bug: the model was constructed in
    ``ConfigDrivenTrainer.__init__`` while ``set_seed`` was only called inside
    ``train()`` — after the weights already existed. Every run therefore started
    from an uncontrolled random initialisation, so ``--seed`` did not make a run
    reproducible and the "3 seeds" were three arbitrary runs.
    """
    from scripts.config import TrainingConfig
    from scripts.trainers import build_trainer

    cfg = TrainingConfig.from_file(
        os.path.join(_proj_root, 'configs', 'ce.yaml')).replace(device='cpu')
    counts = np.array([500] + [5] * 99)

    def first_matrix(model):
        for k, v in model.state_dict().items():
            if v.ndim >= 2:
                return k, v
        raise AssertionError("no matrix parameter found")

    k1, w1 = first_matrix(build_trainer(cfg, counts, device='cpu').model)
    k2, w2 = first_matrix(build_trainer(cfg, counts, device='cpu').model)
    assert torch.equal(w1, w2), (
        f"same seed gave different initial weights for {k1} "
        f"(max |Δ|={float((w1 - w2).abs().max()):.6f}) — the seed does not cover "
        f"model initialisation"
    )
    print(f"  ✅ identical init under the same seed ({k1})")


def test_same_seed_gives_an_identical_run():
    """End-to-end: same seed must reproduce the whole training history."""
    from scripts.config import TrainingConfig
    from scripts.trainers import build_trainer

    counts = np.array([500] + [5] * 99)
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 3, 32, 32, generator=g),
                      torch.randint(0, 100, (16,), generator=g)),
        batch_size=8, shuffle=True,   # data order must be seeded too
    )

    histories = []
    for tag in ('a', 'b'):
        cfg = TrainingConfig.from_file(
            os.path.join(_proj_root, 'configs', 'ce.yaml')).replace(
            device='cpu', epochs=3, checkpoint_dir=_tmp_ckpt(f'same_{tag}'))
        histories.append(build_trainer(cfg, counts, device='cpu').train(loader))

    losses_a = [h['train_loss'] for h in histories[0]]
    losses_b = [h['train_loss'] for h in histories[1]]
    assert losses_a == losses_b, (
        f"same seed produced different runs:\n  {losses_a}\n  {losses_b}"
    )
    print(f"  ✅ same seed reproduces the run exactly: {losses_a}")


def test_different_seeds_give_different_inits():
    """The seed must actually matter — not be ignorable."""
    from scripts.config import TrainingConfig
    from scripts.trainers import build_trainer

    counts = np.array([500] + [5] * 99)
    path = os.path.join(_proj_root, 'configs', 'ce.yaml')

    def first_matrix(seed):
        cfg = TrainingConfig.from_file(path).replace(device='cpu', seed=seed)
        for _, v in build_trainer(cfg, counts, device='cpu').model.state_dict().items():
            if v.ndim >= 2:
                return v

    assert not torch.equal(first_matrix(78), first_matrix(88)), \
        "different seeds produced identical initial weights"
    print("  ✅ different seeds give different inits")


TESTS = [
    ("LR schedule matches reference", test_lr_schedule_matches_reference),
    ("LR schedule scales with base_lr", test_lr_schedule_scales_with_base_lr),
    ("Defaults are the published recipe", test_defaults_are_the_published_recipe),
    ("Optimiser matches reference", test_optimiser_matches_reference),
    ("No cosine scheduler", test_no_cosine_scheduler),
    ("train() takes only a train loader", test_train_accepts_only_train_loader),
    ("No validation machinery", test_no_validation_machinery_on_trainer),
    ("History has no validation metrics", test_history_has_no_validation_metrics),
    ("Scripts use only canonical artifact", test_training_scripts_reference_only_canonical_artifact),
    ("Only the final checkpoint is written", test_only_the_final_checkpoint_is_written),
    ("No milestone-checkpoint knobs", test_no_milestone_checkpoint_knobs_remain),
    ("Checkpoint names carry expert+seed", test_checkpoint_names_include_expert_and_seed),
    ("Final checkpoint self-describing", test_final_checkpoint_is_self_describing),
    ("NaN loss raises", test_nan_loss_raises),
    ("Non-finite logits raise", test_nonfinite_logits_raise),
    ("set_seed reproducible", test_seed_makes_initialisation_reproducible),
    ("Different seed changes init", test_seed_changes_initialisation),
    ("Synthetic dry-run: 4 experts", test_synthetic_dry_run_all_four_experts),
    ("Seed controls model init", test_seed_controls_model_initialisation),
    ("Same seed reproduces the run", test_same_seed_gives_an_identical_run),
    ("Different seeds differ", test_different_seeds_give_different_inits),
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


if __name__ == "__main__":
    sys.exit(main())
