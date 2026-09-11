"""
Base trainer implementing the published CIFAR-LT training protocol.

Recipe (Cao et al., LDAM-DRW `cifar_train.py` — the reference CIFAR-LT setup):

    epochs       200          lr            0.1
    batch        128          weight decay  2e-4
    optimiser    SGD, momentum 0.9, nesterov=False
    warmup       linear over the first 5 epochs
    schedule     base_lr for epochs <=160, x0.01 for 161..180, x0.0001 for 181..200

There is **no validation split and no early stopping**. Experts train on the full
long-tailed training set and are evaluated on the balanced 10K test set. The
final-epoch model is the reported model; checkpoints at every 20th epoch from
epoch 160 exist for inspection only and must never be selected on test accuracy.

Subclasses set `self.model` / `self.loss_fn` and implement `_compute_loss`, which
returns `(loss, logits, aux)`.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG the training path uses (AGENTs.md section 10).

    Also pins the CUDA backend so that a seed actually reproduces a run:

    * ``cudnn.deterministic`` — cuDNN picks deterministic kernels
    * ``cudnn.benchmark``     — no per-run algorithm autotuning
    * TF32 disabled           — ``cudnn.allow_tf32`` defaults to True on Ampere
      and newer, which perturbs convolution results (measured: CPU/GPU logits
      diverged by 1.2e-01 with TF32 on versus 2.7e-04 with it off). Kaggle's T4
      is Turing and has no TF32, so leaving it on would make local GPU
      verification numerically unrepresentative of the reported Kaggle runs.

    The cost is throughput on Ampere-or-newer GPUs; determinism was chosen
    deliberately over speed (see docs/specs/gpu-verification.md).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def step_lr(
    epoch: int,
    base_lr: float = 0.1,
    warmup_epochs: int = 5,
    decay_epochs: tuple[int, int] = (160, 180),
    decay_factors: tuple[float, float] = (0.01, 0.0001),
) -> float:
    """Learning rate for a 1-indexed epoch, matching LDAM-DRW exactly.

    LDAM-DRW's ``adjust_learning_rate`` reads::

        epoch = epoch + 1
        if epoch <= 5:     lr = args.lr * epoch / 5     # linear warmup
        elif epoch > 180:  lr = args.lr * 0.0001
        elif epoch > 160:  lr = args.lr * 0.01
        else:              lr = args.lr

    This function takes an already 1-indexed epoch, so the boundaries are
    161 and 181.
    """
    if epoch < 1:
        raise ValueError(f"epoch must be >= 1, got {epoch}")
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if epoch > decay_epochs[1]:
        return base_lr * decay_factors[1]
    if epoch > decay_epochs[0]:
        return base_lr * decay_factors[0]
    return base_lr


# ---------------------------------------------------------------------------
# Head / Medium / Tail split helpers
# ---------------------------------------------------------------------------

def compute_class_groups(
    class_counts: np.ndarray,
    many_thresh: int = 100,
    few_thresh: int = 20,
) -> dict:
    """Return indices for Head (>=many), Medium, and Tail (<few) classes."""
    return {
        'head':   np.where(class_counts >= many_thresh)[0],
        'medium': np.where((class_counts >= few_thresh)
                           & (class_counts < many_thresh))[0],
        'tail':   np.where(class_counts < few_thresh)[0],
    }


def balanced_accuracy(all_targets: np.ndarray, all_preds: np.ndarray
                      ) -> tuple[float, dict[int, float]]:
    """Mean per-class recall (Balanced Accuracy)."""
    classes = sorted(set(all_targets.tolist()))
    per_class = {}
    for c in classes:
        mask = all_targets == c
        per_class[c] = (all_preds[mask] == c).sum() / max(mask.sum(), 1)
    ba = float(np.mean(list(per_class.values())))
    return ba, per_class


def group_accuracies(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-group accuracy: fraction of correctly classified samples."""
    result = {}
    for name, cls_list in groups.items():
        mask = np.isin(all_targets, cls_list)
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = (all_preds[mask] == all_targets[mask]).sum() / mask.sum()
    return result


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------

class BaseTrainer:
    """Shared training infrastructure for the four experts.

    Subclasses must set ``self.model``, ``self.expert_name`` and implement
    ``_compute_loss`` returning ``(loss, logits, aux)``.
    """

    #: Whether the (logits, targets) pair yields a meaningful training accuracy.
    #: Mixup sets this to False, since its logits come from mixed inputs.
    reports_train_accuracy: bool = True

    def __init__(
        self,
        model: nn.Module,
        expert_name: str,
        loss_fn: nn.Module | None = None,
        class_counts: np.ndarray | None = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.1,
        weight_decay: float = 2e-4,
        momentum: float = 0.9,
        batch_size: int = 128,
        epochs: int = 200,
        warmup_epochs: int = 5,
        decay_epochs: tuple[int, int] = (160, 180),
        decay_factors: tuple[float, float] = (0.01, 0.0001),
        checkpoint_dir: str = './checkpoints',
        seed: int = 0,
    ):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if loss_fn is not None else None
        self.expert_name = expert_name
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = tuple(decay_epochs)
        self.decay_factors = tuple(decay_factors)
        self.batch_size = batch_size
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.class_groups = None
        if class_counts is not None:
            self.class_groups = compute_class_groups(np.asarray(class_counts))

        # No nesterov: the reference CIFAR-LT recipe uses plain momentum SGD.
        self.optimiser = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=False,
        )

        self.epoch = 0
        self.history: list[dict] = []

    # ── to be overridden by subclasses ─────────────────────────────────

    def _compute_loss(
        self, images: torch.Tensor, targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass and loss.

        Args:
            images:  (B, 3, 32, 32) on self.device
            targets: (B,) on self.device
            weights: (B,) optional per-sample loss weights on self.device.

        Returns:
            (loss, logits, aux) where loss is a scalar tensor, logits is (B, C),
            and aux is a dict of extra scalars for logging.
        """
        raise NotImplementedError

    # ── training ──────────────────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        grad_norm_sum = 0.0
        correct = 0
        seen = 0
        n_batches = 0
        aux_acc: dict[str, float] = {}

        for batch in loader:
            if len(batch) == 3:
                images, targets, weights = batch
                weights = weights.to(self.device)
            else:
                images, targets = batch
                weights = None

            images = images.to(self.device)
            targets = targets.to(self.device)

            loss, logits, aux = self._compute_loss(images, targets, weights=weights)

            if not torch.isfinite(logits).all():
                n_bad = int((~torch.isfinite(logits)).sum())
                raise FloatingPointError(
                    f"[{self.expert_name}] epoch {self.epoch}: {n_bad} non-finite "
                    f"value(s) in logits — aborting instead of continuing a "
                    f"diverged run"
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[{self.expert_name}] epoch {self.epoch}: non-finite loss "
                    f"({loss.item()}) — aborting instead of continuing a "
                    f"diverged run"
                )

            self.optimiser.zero_grad()
            loss.backward()

            total_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.norm().item() ** 2
            grad_norm = total_norm_sq ** 0.5

            self.optimiser.step()

            total_loss += loss.item()
            grad_norm_sum += grad_norm
            n_batches += 1

            if self.reports_train_accuracy:
                correct += int((logits.argmax(dim=1) == targets).sum())
                seen += targets.numel()

            for k, v in aux.items():
                aux_acc[k] = aux_acc.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        if n_batches == 0:
            raise RuntimeError(
                f"[{self.expert_name}] training loader produced no batches"
            )

        metrics = {
            'loss': total_loss / n_batches,
            'grad_norm': grad_norm_sum / n_batches,
        }
        if self.reports_train_accuracy:
            metrics['acc'] = correct / max(seen, 1)
        for k, v in aux_acc.items():
            metrics[k] = v / n_batches
        return metrics

    # ── checkpointing ─────────────────────────────────────────────────

    def _save_checkpoint(self, log: dict, is_final: bool = True) -> Path:
        """Write the checkpoint. Only the final-epoch model is ever saved."""
        tag = 'final' if is_final else f'epoch{self.epoch}'
        path = self.checkpoint_dir / f'{self.expert_name}_seed{self.seed}_{tag}.pt'
        state = {
            'epoch': self.epoch,
            'seed': self.seed,
            'expert_name': self.expert_name,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'is_final': is_final,
            'log': log,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimiser.load_state_dict(state['optimiser_state_dict'])
        self.epoch = state['epoch']
        self.seed = state.get('seed', self.seed)
        print(f"Loaded checkpoint from {path} (epoch {self.epoch}, seed {self.seed})")

    def save_history(self, path: str | None = None) -> None:
        if path is None:
            path = self.checkpoint_dir / f'{self.expert_name}_seed{self.seed}_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")

    # ── public training loop ──────────────────────────────────────────

    def train(self, train_loader: DataLoader) -> list[dict]:
        """Train on the full long-tailed training set for a fixed budget.

        No validation set is used and no checkpoint is selected: the final-epoch
        model is the reported model, and it is the only checkpoint written.

        Returns:
            history: list of per-epoch log dicts.
        """
        set_seed(self.seed)
        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            epoch_start = time.time()

            current_lr = step_lr(
                epoch,
                base_lr=self.lr,
                warmup_epochs=self.warmup_epochs,
                decay_epochs=self.decay_epochs,
                decay_factors=self.decay_factors,
            )
            for pg in self.optimiser.param_groups:
                pg['lr'] = current_lr

            metrics = self._train_one_epoch(train_loader)

            log = {
                'epoch': epoch,
                'lr': current_lr,
                'time_s': time.time() - epoch_start,
                'train_loss': metrics['loss'],
                'train_acc': metrics.get('acc'),
                'grad_norm': metrics['grad_norm'],
            }
            for k, v in metrics.items():
                if k not in ('loss', 'grad_norm', 'acc'):
                    log[f'train_{k}'] = v

            self.history.append(log)

            if epoch == self.epochs:
                path = self._save_checkpoint(log, is_final=True)
                print(f"[{self.expert_name}] checkpoint -> {path.name}")

            if epoch == 1 or epoch % 10 == 0 or epoch == self.epochs:
                acc_str = (f"Train Acc {log['train_acc']:.2%} | "
                           if log['train_acc'] is not None else "")
                print(
                    f"[{self.expert_name} seed={self.seed}] Epoch {epoch:3d}/{self.epochs} | "
                    f"LR {current_lr:.6f} | "
                    f"Train Loss {log['train_loss']:.4f} | "
                    f"{acc_str}"
                    f"GradNorm {log['grad_norm']:.2f}"
                )

        total_time = time.time() - total_start
        print(f"[{self.expert_name} seed={self.seed}] done in {total_time:.0f}s "
              f"({self.epochs} epochs, final-epoch model is the reported model)")
        return self.history
