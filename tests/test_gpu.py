"""
GPU verification tests.

This machine's GPU exists **for verification only**: the full 12-run training
sweep happens on Kaggle. These tests therefore exercise the CUDA code path
cheaply — forward, backward, device placement, memory headroom, and CPU/GPU
numerical agreement — and skip cleanly on a CPU-only machine so the suite stays
meaningful anywhere.

Run directly:
    python tests/test_gpu.py
"""

import os
import sys
import tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.base_trainer import set_seed
from scripts.config import TrainingConfig
from scripts.trainers import build_trainer

CUDA = torch.cuda.is_available()
CONFIG_DIR = os.path.join(_proj_root, 'configs')

EXPERT_CONFIGS = {
    'CE': 'ce.yaml',
    'LA': 'lal.yaml',
    'BS': 'balanced_softmax.yaml',
    'Mixup': 'mixup.yaml',
}


class _SkipTest(Exception):
    """Raised when a check cannot run on this machine.

    A distinct type, not a plain return: a test that returns normally is counted
    as a pass by the runner, which is how eight CUDA checks came to report green
    on a CPU-only box even though AGENTs.md section 11 relies on this file to
    prove the GPU path.
    """


def _skip(reason: str) -> None:
    """Skip the current test. The runner reports it separately from a pass."""
    raise _SkipTest(reason)


def _counts() -> np.ndarray:
    return np.array([500] + [5] * 99, dtype=np.int64)


def _tiny_loader(n=32, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=g)
    targets = torch.randint(0, 100, (n,), generator=g)
    return DataLoader(TensorDataset(images, targets), batch_size=batch)


def _config(fname, **overrides):
    base = {
        'epochs': 1, 'device': 'cuda',
        'checkpoint_dir': tempfile.mkdtemp(prefix='dsh_gputest_'),
    }
    base.update(overrides)
    return TrainingConfig.from_file(os.path.join(CONFIG_DIR, fname)).replace(**base)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def test_cuda_is_available():
    """A usable CUDA device must be visible."""
    if not CUDA:
        _skip('no CUDA device on this machine')
        return
    assert torch.version.cuda is not None, "torch is a CPU-only build"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"  ✅ CUDA {torch.version.cuda} on {name} "
          f"(sm_{props.major}{props.minor}, {props.total_memory / 1e9:.1f} GB)")


def test_device_auto_resolves_to_cuda():
    """`device: auto` in the configs must pick the GPU when one exists."""
    if not CUDA:
        _skip('no CUDA device')
        return
    cfg = TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    assert cfg.device == 'auto', f"config device is '{cfg.device}', expected 'auto'"
    assert cfg.resolved_device == 'cuda', f"auto resolved to '{cfg.resolved_device}'"
    print(f"  ✅ device 'auto' -> {cfg.resolved_device}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_set_seed_pins_cudnn():
    """set_seed must pin cuDNN so a seed reproduces on GPU."""
    set_seed(0)
    assert torch.backends.cudnn.deterministic is True, \
        "cudnn.deterministic is False — GPU runs are not reproducible"
    assert torch.backends.cudnn.benchmark is False, \
        "cudnn.benchmark is True — cuDNN may pick different algorithms per run"
    print("  ✅ set_seed pins cudnn.deterministic=True, benchmark=False")


def test_set_seed_reproducible_initialisation_on_gpu():
    """Same seed must give identical initial weights on the GPU."""
    if not CUDA:
        _skip('no CUDA device')
        return
    from models.resnet32 import ResNet32
    set_seed(7)
    a = ResNet32(num_classes=100).state_dict()
    set_seed(7)
    b = ResNet32(num_classes=100).state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), f"parameter '{k}' differs across identical seeds"
    print("  ✅ identical init under the same seed")


# ---------------------------------------------------------------------------
# Real training path on GPU
# ---------------------------------------------------------------------------

def test_all_four_experts_train_on_gpu():
    """Every expert must complete a step on CUDA with finite grads on-device."""
    if not CUDA:
        _skip('no CUDA device')
        return
    loader = _tiny_loader()
    for name, fname in EXPERT_CONFIGS.items():
        trainer = build_trainer(_config(fname), _counts(), device='cuda')
        off_device = [n for n, p in trainer.model.named_parameters() if not p.is_cuda]
        assert not off_device, f"{name}: {len(off_device)} params left on CPU"
        hist = trainer.train(loader)
        bad = [n for n, p in trainer.model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not bad, f"{name}: {len(bad)} params with missing/non-finite grad"
        assert np.isfinite(hist[-1]['train_loss']), f"{name}: non-finite loss"
        print(f"  ✅ {name}: on GPU, loss={hist[-1]['train_loss']:.4f}, grads finite")


def test_real_batch_size_fits_in_vram():
    """One full-size batch (batch_size from config) must fit without OOM."""
    if not CUDA:
        _skip('no CUDA device')
        return
    cfg = TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    bs = cfg.data.batch_size
    trainer = build_trainer(_config('ce.yaml'), _counts(), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    trainer.train(_tiny_loader(n=bs, batch=bs))
    peak = torch.cuda.max_memory_allocated() / 1e6
    total = torch.cuda.get_device_properties(0).total_memory / 1e6
    assert peak < total, f"peak {peak:.0f} MB exceeded total {total:.0f} MB"
    print(f"  ✅ batch={bs} fits: peak {peak:.0f} MB of {total:.0f} MB VRAM")


def test_nonfinite_logits_raise_on_gpu():
    """The NaN/Inf guard must fire on the CUDA path too, not just on CPU."""
    if not CUDA:
        _skip('no CUDA device')
        return
    from scripts.base_trainer import BaseTrainer
    import torch.nn.functional as F

    class _NaN(BaseTrainer):
        def __init__(self, **kw):
            super().__init__(model=torch.nn.Linear(3 * 32 * 32, 100),
                             expert_name='NaNGpu', **kw)

        def _compute_loss(self, images, targets, weights=None):
            logits = self.model(images.flatten(1))
            logits = logits.clone()
            logits[0, 0] = float('inf')
            return F.cross_entropy(logits, targets), logits, {}

    t = _NaN(device='cuda', epochs=1,
             checkpoint_dir=tempfile.mkdtemp(prefix='dsh_gputest_nan_'))
    try:
        t.train(_tiny_loader())
    except FloatingPointError as e:
        print(f"  ✅ GPU NaN guard fired: {str(e)[:70]}")
        return
    raise AssertionError("non-finite logits did not raise on the GPU path")


# ---------------------------------------------------------------------------
# Numerical agreement CPU vs GPU
# ---------------------------------------------------------------------------

def test_forward_agrees_between_cpu_and_gpu():
    """The CUDA path must compute what the CPU path computes."""
    if not CUDA:
        _skip('no CUDA device')
        return
    from models.resnet32 import ResNet32

    set_seed(0)
    cpu_model = ResNet32(num_classes=100).eval()
    gpu_model = ResNet32(num_classes=100).eval()
    gpu_model.load_state_dict(cpu_model.state_dict())
    gpu_model = gpu_model.cuda()

    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 3, 32, 32, generator=g)
    with torch.no_grad():
        out_cpu = cpu_model(x)
        out_gpu = gpu_model(x.cuda()).cpu()

    assert out_cpu.shape == out_gpu.shape, f"{out_cpu.shape} vs {out_gpu.shape}"
    max_diff = float((out_cpu - out_gpu).abs().max())
    assert max_diff < 1e-3, f"CPU/GPU logits diverge by {max_diff:.2e}"
    agree = (out_cpu.argmax(1) == out_gpu.argmax(1)).float().mean().item()
    print(f"  ✅ CPU/GPU logits agree (max |Δ|={max_diff:.2e}, argmax match {agree:.0%})")


# ---------------------------------------------------------------------------
# Local-GPU guard (advisory)
# ---------------------------------------------------------------------------

def test_reportable_run_on_local_gpu_is_flagged():
    """A non-dry run that would use the GPU must be flagged as local training."""
    if not CUDA:
        _skip('no CUDA device')
        return
    from scripts import train as train_mod
    cfg = TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    assert train_mod.full_run_on_local_gpu(cfg, max_batches=None) is True, \
        "a reportable run on the local GPU was not flagged"
    print("  ✅ reportable run on local GPU is flagged")


def test_dry_run_is_not_flagged():
    """A --max-batches dry run must not be flagged, even on the GPU."""
    from scripts import train as train_mod
    cfg = TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    assert train_mod.full_run_on_local_gpu(cfg, max_batches=2) is False, \
        "a dry run was incorrectly flagged"
    print("  ✅ dry run is not flagged")


def test_cpu_config_is_not_flagged():
    """An explicit cpu device must never be flagged."""
    from scripts import train as train_mod
    cfg = TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml')).replace(device='cpu')
    assert train_mod.full_run_on_local_gpu(cfg, max_batches=None) is False, \
        "a cpu run was incorrectly flagged"
    print("  ✅ explicit cpu run is not flagged")


def test_kaggle_environment_is_detected():
    """A Kaggle kernel must be recognised, so the guard stays silent there."""
    from scripts import train as train_mod
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Batch'
    try:
        assert train_mod.is_kaggle_environment() is True, \
            "Kaggle kernel not detected from KAGGLE_KERNEL_RUN_TYPE"
    finally:
        del os.environ['KAGGLE_KERNEL_RUN_TYPE']
    os.environ['KAGGLE_URL_BASE'] = 'https://www.kaggle.com'
    try:
        assert train_mod.is_kaggle_environment() is True, \
            "Kaggle kernel not detected from KAGGLE_URL_BASE"
    finally:
        del os.environ['KAGGLE_URL_BASE']
    assert train_mod.is_kaggle_environment() is False, \
        "the workspace machine was misdetected as Kaggle"
    print("  ✅ Kaggle environment detected (and workspace is not Kaggle)")


def test_no_warning_when_running_on_kaggle():
    """The full run belongs on Kaggle — the guard must not nag there."""
    from scripts import train as train_mod
    cfg = TrainingConfig.from_file(os.path.join(CONFIG_DIR, 'ce.yaml'))
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Batch'
    try:
        flagged = train_mod.full_run_on_local_gpu(cfg, max_batches=None)
    finally:
        del os.environ['KAGGLE_KERNEL_RUN_TYPE']
    assert flagged is False, (
        "a Kaggle run was flagged as 'local GPU' — the banner told the user "
        "full training belongs on Kaggle while they were already on Kaggle"
    )
    print("  ✅ no banner when running on Kaggle")


TESTS = [
    ("CUDA available", test_cuda_is_available),
    ("device auto -> cuda", test_device_auto_resolves_to_cuda),
    ("set_seed pins cuDNN", test_set_seed_pins_cudnn),
    ("Seeded init reproducible on GPU", test_set_seed_reproducible_initialisation_on_gpu),
    ("All four experts train on GPU", test_all_four_experts_train_on_gpu),
    ("Real batch size fits in VRAM", test_real_batch_size_fits_in_vram),
    ("NaN guard fires on GPU", test_nonfinite_logits_raise_on_gpu),
    ("CPU/GPU forward agreement", test_forward_agrees_between_cpu_and_gpu),
    ("Kaggle environment detected", test_kaggle_environment_is_detected),
    ("No banner on Kaggle", test_no_warning_when_running_on_kaggle),
    ("Reportable GPU run flagged", test_reportable_run_on_local_gpu_is_flagged),
    ("Dry run not flagged", test_dry_run_is_not_flagged),
    ("CPU run not flagged", test_cpu_config_is_not_flagged),
]


def main() -> int:
    if not CUDA:
        print("No CUDA device detected — GPU checks will be skipped.")
    passed = failed = skipped = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except _SkipTest as e:
            print(f"  ⊘ skipped {name} ({e})")
            skipped += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 - surface the real cause
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"  {len(TESTS)} tests: {passed} passed, {skipped} skipped, {failed} failed")
    if skipped:
        print(f"  NOTE: {skipped} check(s) did not run — that part of the GPU path "
              f"is UNVERIFIED on this machine, not passing.")
    return 1 if failed else 0


if __name__ == '__main__':
    if not CUDA:
        print("No CUDA device detected — GPU checks will be skipped.")
    sys.exit(main())
