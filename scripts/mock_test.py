#!/usr/bin/env python3
"""
Mock test for Stage-1 expert training.

Creates synthetic random data to verify forward/backward passes,
metric shapes, and logging functionality for all three expert types
without needing the actual CIFAR-100 dataset.

Usage:
    python scripts/mock_test.py
"""

import os
import shutil
import sys
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── synthetic datasets ───────────────────────────────────────────────

class DummyCIFAR100(Dataset):
    """Synthetic CIFAR-100-shaped data: single view."""

    def __init__(self, n_per_class: int = 10, n_classes: int = 100):
        self.data = torch.randn(n_per_class * n_classes, 3, 32, 32)
        self.labels = torch.arange(n_classes).repeat_interleave(n_per_class)
        self.n_classes = n_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()


class DummyCIFAR100TwoView(Dataset):
    """Synthetic CIFAR-100-shaped data: two augmented views (for PaCo)."""

    def __init__(self, n_per_class: int = 10, n_classes: int = 100):
        self.data = torch.randn(n_per_class * n_classes, 3, 32, 32)
        self.labels = torch.arange(n_classes).repeat_interleave(n_per_class)
        self.n_classes = n_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return two independently random views
        view1 = self.data[idx] + 0.02 * torch.randn_like(self.data[idx])
        view2 = self.data[idx] + 0.02 * torch.randn_like(self.data[idx])
        return [view1, view2], self.labels[idx].item()


def get_class_counts(dataset) -> np.ndarray:
    counts = np.zeros(dataset.n_classes, dtype=np.int64)
    for _, label in dataset:
        counts[label] += 1
    return counts


# ── test runner ──────────────────────────────────────────────────────

def test_expert(trainer_class, expert_name: str, extra_kwargs: dict = None,
                two_view: bool = False, queue_batch_align: bool = False):
    """Run a short training loop on dummy data and verify outputs."""
    print(f"\n{'='*60}")
    print(f"Testing: {expert_name}")
    print(f"{'='*60}")

    if two_view:
        train_data = DummyCIFAR100TwoView(n_per_class=10, n_classes=100)
    else:
        train_data = DummyCIFAR100(n_per_class=10, n_classes=100)
    val_data = DummyCIFAR100(n_per_class=5, n_classes=100)

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True,
        drop_last=queue_batch_align,
    )
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    class_counts = get_class_counts(train_data)

    kwargs = dict(
        device='cpu',
        lr=0.1,
        batch_size=64,
        epochs=3,
        checkpoint_dir='./mock_checkpoints',
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        trainer = trainer_class(**kwargs)

        # Set class weights for PaCo loss
        if hasattr(trainer, 'loss_fn') and hasattr(trainer.loss_fn, 'set_class_weight'):
            trainer.loss_fn.set_class_weight(class_counts.tolist())

        history = trainer.train(train_loader, val_loader,
                                class_counts=class_counts)

        # ── verify history structure ──
        last = history[-1]
        required_keys = [
            'epoch', 'lr', 'train_loss', 'val_loss',
            'val_ba', 'grad_norm',
            'val_acc_head', 'val_acc_medium', 'val_acc_tail',
        ]
        for key in required_keys:
            assert key in last, f"Missing key in history: {key}"
            assert np.isfinite(last[key]), f"Non-finite value for {key}: {last[key]}"

        # ── verify metrics are in reasonable ranges ──
        assert 0.0 <= last['val_ba'] <= 1.0, \
            f"BA out of range: {last['val_ba']}"
        assert last['train_loss'] > 0.0, \
            f"Loss not positive: {last['train_loss']}"
        assert last['grad_norm'] > 0.0, \
            f"Gradient norm is zero"

        # ── verify checkpoint files exist ──
        best_path = f'./mock_checkpoints/{expert_name}_best.pt'
        latest_path = f'./mock_checkpoints/{expert_name}_latest.pt'
        assert os.path.exists(best_path), f"Best checkpoint not found: {best_path}"
        assert os.path.exists(latest_path), f"Latest checkpoint not found: {latest_path}"

        # ── verify checkpoint loads correctly ──
        state = torch.load(best_path, map_location='cpu', weights_only=False)
        assert 'model_state_dict' in state
        assert 'epoch' in state
        assert state['expert_name'] == expert_name

        print(f"  ✓ All checks passed for {expert_name}")
        print(f"  ✓ Final Val BA: {last['val_ba']:.4f}")
        print(f"  ✓ Checkpoints saved and verified")
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MOCK TEST: Stage 1 Expert Training")
    print("=" * 60)
    print(f"PyTorch {torch.__version__}, "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    from scripts.train_mixup import MixupTrainer
    from scripts.train_lal import LALTrainer
    from scripts.train_paco import PaCoTrainer

    results = {}

    # ── Mixup+CE expert (standard, single view + Mixup augmentation) ──
    results['Mixup'] = test_expert(MixupTrainer, 'Mixup')

    # ── LAL expert (standard, single view) ──
    results['LAL'] = test_expert(LALTrainer, 'LAL')

    # ── PaCo expert (two views, queue, momentum encoder) ──
    results['PaCo'] = test_expert(
        PaCoTrainer, 'PaCo',
        extra_kwargs={
            'alpha': 0.5,
            'temperature': 0.07,
            'K': 64,          # small queue for CPU test
        },
        two_view=True,
        queue_batch_align=True,
    )

    # ── summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:10s}: {status}")
        all_pass = all_pass and passed

    if all_pass:
        print(f"\n  All experts passed the mock test! 🎉")
    else:
        print(f"\n  Some experts failed. See errors above.")
        sys.exit(1)

    # clean up
    shutil.rmtree('./mock_checkpoints', ignore_errors=True)
    print("  (mock_checkpoints cleaned up)")


if __name__ == '__main__':
    main()
