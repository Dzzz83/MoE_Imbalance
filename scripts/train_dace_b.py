#!/usr/bin/env python3
"""
DACE Step 2: Train Expert B — Medium-class specialist with cascade routing.

Expert B uses ResNet32WithRouting (classifier + routing head) trained with:
  - CE + Mixup (α=1.0) for classification
  - Contrastive routing loss: KL(A || B) agreement signal
  - Cascade: Expert A is frozen, B's routing head learns to encode
    "how much does my Mixup distribution diverge from Expert A's LAL
    distribution on this sample?"
  - PartitionedDataset: 80% Med classes + 20% all classes
  - 200 epochs, SGD, cosine LR schedule

Usage:
    python scripts/train_dace_b.py [--data-root DATA] [--checkpoint-dir CKPT]
                                   [--dace-a-ckpt PATH] [--batch-size 128]
                                   [--epochs 200] [--lr 0.1]
                                   [--lambda-routing 0.1] [--kl-threshold 0.1]
"""

import os
import sys
import warnings

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32, ResNet32WithRouting
from losses.ce_loss import CELoss
from losses.kl_divergence import kl_divergence, compute_agreement_label
from losses.contrastive_routing_loss import ContrastiveRoutingLoss
from scripts.base_trainer import BaseTrainer
from data.cifar_lt import LongTailCIFAR100
from data.partitioned_dataset import PartitionedDataset, PartitionedSampler


# ── Medium class indices (20 <= training count < 100) ────────────────────
MED_THRESH_LOW = 20
MED_THRESH_HIGH = 100


def get_med_classes(class_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Return indices of Med classes (20-100 training samples)."""
    med_idx = np.where((class_counts >= MED_THRESH_LOW)
                       & (class_counts < MED_THRESH_HIGH))[0]
    return {'med': med_idx}


# ── Mixup helpers ────────────────────────────────────────────────────────

def mixup_data(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def mixup_criterion(
    criterion: nn.Module, pred: torch.Tensor,
    targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, targets_a) + (1.0 - lam) * criterion(pred, targets_b)


# ---------------------------------------------------------------------------
# DACE-B Trainer  (ResNet32WithRouting + Mixup + contrastive routing)
# ---------------------------------------------------------------------------

class DACEATrainer(BaseTrainer):
    """Minimal loader for Expert A (frozen during B's training)."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.model = ResNet32(num_classes=100)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class DACEBTrainer(BaseTrainer):
    """
    Trainer for DACE Expert B (Med specialist, Mixup + contrastive routing).

    Expert A is frozen.  For each batch:
      1. Forward A → probs_a
      2. Forward B (Mixup) → logits_b, probs_b, embeddings_b
      3. KL(probs_a || probs_b) → agreement labels
      4. L_contrastive pulls together embeddings with same agreement label
      5. L_total = L_cls + λ * L_contrastive
    """

    def __init__(
        self,
        expert_a: DACEATrainer,
        lambda_routing: float = 0.1,
        kl_threshold: float = 0.1,
        mixup_alpha: float = 1.0,
        **kwargs,
    ):
        self.model = ResNet32WithRouting(num_classes=100, routing_dim=32)
        self.expert_a = expert_a
        self.lambda_routing = lambda_routing
        self.kl_threshold = kl_threshold
        self.mixup_alpha = mixup_alpha

        self.loss_ce = CELoss()
        self.loss_contrastive = ContrastiveRoutingLoss(temperature=0.5)

        super().__init__(
            model=self.model,
            loss_fn=CELoss(),  # used by BaseTrainer.validate()
            expert_name='DACE_B',
            **kwargs,
        )

    def _compute_loss(self, images, targets, weights=None):
        batch_size = images.size(0)

        # 1. Forward Expert A (frozen, no grad)
        with torch.no_grad():
            logits_a = self.expert_a.forward_logits(images)
            probs_a = F.softmax(logits_a, dim=1)

        # 2. Mixup + Forward Expert B
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, targets, alpha=self.mixup_alpha,
        )
        logits_b, routing_emb = self.model(mixed_images)
        probs_b = F.softmax(logits_b, dim=1)

        # 3. Classification loss (Mixup CE)
        loss_cls = mixup_criterion(
            self.loss_ce, logits_b, targets_a, targets_b, lam,
        )

        # 4. Compute KL divergence and agreement labels
        # Use the NON-mixed images for KL (A vs B on the SAME input)
        # But B's features are from mixed images... We need B's probs on
        # the original (non-mixed) images for a fair KL comparison.
        # However, B is trained with Mixup, so its forward on non-mixed
        # images during training is valid (it's just a forward pass).
        # Re-forward B on original images for KL computation.
        logits_b_original, routing_emb_original = self.model(images)
        probs_b_original = F.softmax(logits_b_original, dim=1)

        kl_labels = compute_agreement_label(
            logits_a, logits_b_original,
            threshold=self.kl_threshold, metric='kl',
        )  # (B,) long: 0=agree, 1=disagree

        # 5. Contrastive routing loss on the routing embeddings
        loss_contrastive, aux = self.loss_contrastive(routing_emb_original, kl_labels)

        # 6. Total loss
        loss = loss_cls + self.lambda_routing * loss_contrastive

        if weights is not None:
            # `loss` is already a scalar here, so (loss * weights).mean() is just
            # loss * weights.mean() -- a global rescale, not per-sample weighting.
            # Per-sample weighting requires an unreduced classification term (as
            # train_dace_a does); silent pseudo-weighting is worse than a warning.
            warnings.warn(
                "per-sample weights are ignored by DACE-B: the loss is already "
                "reduced, so weighting it can only rescale the batch loss",
                RuntimeWarning, stacklevel=2,
            )

        return loss, {
            'loss_cls': loss_cls.detach(),
            'loss_contrastive': loss_contrastive.detach(),
            'kl_agree_ratio': (kl_labels == 0).float().mean().detach(),
        }

    def _forward_for_eval(self, images):
        """Return only logits (no routing head) for validation."""
        return self.model.forward_logits(images)

    def validate(self, loader):
        """Override validate to also log routing embedding stats."""
        metrics = super().validate(loader)

        # Also log routing embedding statistics on validation set
        self.model.eval()
        all_embs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                _, routing_emb = self.model(images)
                all_embs.append(routing_emb.cpu())

        if all_embs:
            all_embs = torch.cat(all_embs, dim=0)
            metrics['routing_emb_std'] = all_embs.std().item()
            metrics['routing_emb_mean_norm'] = all_embs.norm(dim=1).mean().item()

        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DACE Step 2: Train Expert B (Med)')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--dace-a-ckpt', default=None,
                        help='Path to DACE_A_best.pt checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--full-ratio', type=float, default=0.2,
                        help='Fraction of batches from ALL classes')
    parser.add_argument('--lambda-routing', type=float, default=0.1,
                        help='Weight for contrastive routing loss')
    parser.add_argument('--kl-threshold', type=float, default=0.1,
                        help='KL threshold for agree/disagree labeling')
    parser.add_argument('--mixup-alpha', type=float, default=1.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── resolve checkpoint paths ──────────────────────────────────────
    if args.dace_a_ckpt is None:
        args.dace_a_ckpt = os.path.join(args.checkpoint_dir, 'DACE_A_best.pt')
    if not os.path.exists(args.dace_a_ckpt):
        print(f"[Error] Expert A checkpoint not found: {args.dace_a_ckpt}")
        print("  Train Expert A first: python scripts/train_dace_a.py")
        sys.exit(1)

    # ── data ──────────────────────────────────────────────────────────
    train_idx = np.load(f'{args.data_root}/processed/lt_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/lt_val_indices.npy')

    base_train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=train_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
        already_subsampled=True,
    )
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        already_subsampled=True,
    )

    class_counts = base_train_set.get_class_counts()
    med_groups = get_med_classes(class_counts)

    # Partitioned training set: bias toward Med classes
    train_set = PartitionedDataset(base_train_set)
    train_sampler = PartitionedSampler(
        base_train_set,
        primary_groups=med_groups,
        full_ratio=args.full_ratio,
        num_samples=len(base_train_set),
        primary_name='Med',
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── load frozen Expert A ──────────────────────────────────────────
    print(f"\nLoading frozen Expert A from {args.dace_a_ckpt}...")
    expert_a = DACEATrainer(args.dace_a_ckpt, device=args.device)
    print(f"  Expert A loaded ({sum(p.numel() for p in expert_a.model.parameters()):,} params, frozen)")

    # ── train ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DACE Step 2: Training Expert B (Med specialist)")
    print(f"  Med classes: {med_groups['med'].tolist()}")
    print(f"  λ_routing: {args.lambda_routing}")
    print(f"  KL threshold: {args.kl_threshold}")
    print(f"  Mixup alpha: {args.mixup_alpha}")
    print(f"  Full ratio: {args.full_ratio}")
    print(f"{'='*60}\n")

    trainer = DACEBTrainer(
        expert_a=expert_a,
        lambda_routing=args.lambda_routing,
        kl_threshold=args.kl_threshold,
        mixup_alpha=args.mixup_alpha,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,   # BaseTrainer.train() reseeds; without this every
                          # --seed run was identical (it defaulted to 0)
    )
    trainer.train(train_loader, val_loader, class_counts=class_counts)
    trainer.save_history()
    print(f"  Expert B checkpoint saved to {args.checkpoint_dir}/DACE_B_best.pt")


if __name__ == '__main__':
    main()
