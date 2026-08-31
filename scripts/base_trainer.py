"""
Base trainer that implements the common training loop, metrics logging,
and checkpointing logic.  Specific trainers (CE, LAL, PaCo) subclass this
and override `_compute_loss()`.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Head / Medium / Tail split helpers
# ---------------------------------------------------------------------------

def compute_class_groups(
    class_counts: np.ndarray,
    many_thresh: int = 100,
    few_thresh: int = 20,
) -> dict:
    """Return indices for Head (≥many), Medium, and Tail (<few) classes."""
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
    """
    Shared training infrastructure.

    Subclasses must set:
        self.model
        self.loss_fn    (can be None if _compute_loss handles everything)
        self.expert_name
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None,
        expert_name: str,
        class_counts: np.ndarray | None = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        batch_size: int = 128,
        epochs: int = 200,
        warmup_epochs: int = 5,
        checkpoint_dir: str = './checkpoints',
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if loss_fn is not None else None
        self.expert_name = expert_name
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # class groups for head/medium/tail reporting
        self.class_groups = None
        if class_counts is not None:
            self.class_groups = compute_class_groups(class_counts)

        # optimiser & scheduler
        self.optimiser = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=epochs)

        # tracking
        self.best_metric_val = -1e9
        self.epoch = 0
        self.history: list[dict] = []

    # ── to be overridden by subclasses ─────────────────────────────────

    def _compute_loss(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass + loss computation.

        Args:
            images:  (B, 3, 32, 32) on self.device
            targets: (B,) on self.device

        Returns:
            loss:       scalar tensor (already on device, ready for backward)
            aux:        dict of auxiliary scalars for logging (e.g. component losses)
        """
        raise NotImplementedError

    def _forward_for_eval(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used during validation / checkpointing.
        Returns logits of shape (B, C).
        """
        raise NotImplementedError

    # ── training ──────────────────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        grad_norm_sum = 0.0
        n_batches = 0
        aux_acc: dict[str, float] = {}

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            loss, aux = self._compute_loss(images, targets)

            self.optimiser.zero_grad()
            loss.backward()

            # gradient norm (before clipping)
            total_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.norm().item() ** 2
            grad_norm = total_norm_sq ** 0.5

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimiser.step()

            total_loss += loss.item()
            grad_norm_sum += grad_norm
            n_batches += 1

            for k, v in aux.items():
                aux_acc[k] = aux_acc.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        metrics = {
            'loss': total_loss / n_batches,
            'grad_norm': grad_norm_sum / n_batches,
        }
        for k, v in aux_acc.items():
            metrics[k] = v / n_batches
        return metrics

    def validate(self, loader: DataLoader) -> dict:
        """Compute loss and accuracy metrics on a validation set."""
        self.model.eval()
        total_loss = 0.0
        all_targets, all_preds = [], []
        n_batches = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self._forward_for_eval(images)
                loss = self.loss_fn(logits, targets) if self.loss_fn is not None else torch.tensor(0.0)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                n_batches += 1

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        ba, _ = balanced_accuracy(all_targets, all_preds)

        metrics = {
            'loss': total_loss / n_batches,
            'ba': ba,
        }
        if self.class_groups is not None:
            grp = group_accuracies(all_targets, all_preds, self.class_groups)
            for name, acc in grp.items():
                metrics[f'acc_{name}'] = acc

        return metrics

    # ── public training loop ──────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Full training loop.

        Args:
            train_loader: long-tailed CIFAR-100 training set.
            val_loader:   balanced CIFAR-100 validation set.
            class_counts: per-class sample counts in the training set
                          (used to compute training-set BA).

        Returns:
            history: list of per-epoch log dicts.
        """
        # update class groups if provided here
        if class_counts is not None and self.class_groups is None:
            self.class_groups = compute_class_groups(class_counts)

        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            epoch_start = time.time()

            # warmup LR (linear from 0 → self.lr)
            if epoch <= self.warmup_epochs:
                warmup_lr = self.lr * epoch / self.warmup_epochs
                for pg in self.optimiser.param_groups:
                    pg['lr'] = warmup_lr

            train_metrics = self._train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # scheduler step (only after warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimiser.param_groups[0]['lr']

            # compile epoch log
            log = {
                'epoch': epoch,
                'lr': current_lr,
                'time_s': time.time() - epoch_start,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_ba': val_metrics['ba'],
            }
            # optional train BA (expensive on long-tailed data — compute
            # only if `class_counts` was provided, which enables it)
            if 'train_ba' in train_metrics:
                log['train_ba'] = train_metrics['ba']

            for prefix, src in [('train', train_metrics), ('val', val_metrics)]:
                for key in ('acc_head', 'acc_medium', 'acc_tail'):
                    if key in src:
                        log[f'{prefix}_{key}'] = src[key]

            # gradient norm
            log['grad_norm'] = train_metrics.get('grad_norm', 0.0)

            # extra aux losses
            for k, v in train_metrics.items():
                if k not in ('loss', 'grad_norm', 'ba'):
                    log[f'train_{k}'] = v

            self.history.append(log)

            # ── checkpoint by validation BA (strict 0.1% improvement) ──
            current_val_ba = val_metrics.get('ba', 0.0)
            if current_val_ba > self.best_metric_val + 1e-3:
                self.best_metric_val = current_val_ba
                self._save_checkpoint(log, is_best=True)

            # print every 10 epochs + first/last
            if epoch == 1 or epoch % 10 == 0 or epoch == self.epochs:
                h, m, t = (val_metrics.get('acc_head', 0.0),
                           val_metrics.get('acc_medium', 0.0),
                           val_metrics.get('acc_tail', 0.0))
                print(
                    f"[{self.expert_name}] Epoch {epoch:3d}/{self.epochs} | "
                    f"LR {current_lr:.4f} | "
                    f"Train Loss {log['train_loss']:.4f} | "
                    f"Val Loss {log['val_loss']:.4f} | "
                    f"Val BA {log['val_ba']:.2%} | "
                    f"H {h:.1%} M {m:.1%} T {t:.1%} | "
                    f"GradNorm {log['grad_norm']:.2f}"
                )

        total_time = time.time() - total_start
        print(f"[{self.expert_name}] ✓ Done in {total_time:.0f}s. "
              f"Best Val BA = {self.best_metric_val:.2%}")
        return self.history

    # ── checkpointing ─────────────────────────────────────────────────

    def _save_checkpoint(self, log: dict, is_best: bool = False):
        tag = 'best' if is_best else f'epoch_{self.epoch}'
        path = self.checkpoint_dir / f'{self.expert_name}_{tag}.pt'
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric_val': self.best_metric_val,
            'log': log,
            'expert_name': self.expert_name,
        }
        torch.save(state, path)

        # also overwrite latest
        latest = self.checkpoint_dir / f'{self.expert_name}_latest.pt'
        torch.save(state, latest)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimiser.load_state_dict(state['optimiser_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.best_metric_val = state['best_metric_val']
        self.epoch = state['epoch']
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def save_history(self, path: str | None = None):
        if path is None:
            path = self.checkpoint_dir / f'{self.expert_name}_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")
