#!/usr/bin/env python3
"""
DACE Step 3: Train Expert C — Tail-class specialist with cascade routing.

Expert C uses ResNet32WithRouting (classifier + routing head) trained with:
  - Cross-Entropy for classification (with two-view augmentations for contrastive robustness)
  - Contrastive routing loss: KL(avg(A,B) || C) agreement signal
  - Cascade: Experts A and B are frozen, C's routing head learns to encode
    "how much does my distribution diverge from the A+B ensemble average?"
  - PartitionedDataset: 80% Tail classes + 20% all classes
  - 400 epochs (PaCo-standard length), SGD, step LR schedule

Usage:
    python scripts/train_dace_c.py [--data-root DATA] [--checkpoint-dir CKPT]
                                   [--dace-a-ckpt PATH] [--dace-b-ckpt PATH]
                                   [--batch-size 128] [--epochs 400] [--lr 0.05]
                                   [--lambda-routing 0.05] [--kl-threshold 0.1]
"""

import os
import sys

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


# ── Tail class indices (training count < 20) ────────────────────────────
TAIL_THRESH = 20


def get_tail_classes(class_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Return indices of Tail classes (< 20 training samples)."""
    tail_idx = np.where(class_counts < TAIL_THRESH)[0]
    return {'tail': tail_idx}


# ---------------------------------------------------------------------------
# Frozen expert loaders
# ---------------------------------------------------------------------------

class FrozenExpert:
    """Loads an expert model and freezes it for cascade training."""

    def __init__(self, checkpoint_path: str, has_routing: bool = False,
                 device: str = 'cpu'):
        self.has_routing = has_routing
        if has_routing:
            self.model = ResNet32WithRouting(num_classes=100, routing_dim=32)
        else:
            self.model = ResNet32(num_classes=100)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Load with strict=False to handle routing head mismatch
        missing, unexpected = self.model.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        if missing:
            print(f"  [FrozenExpert] Missing keys (expected if routing head): {missing}")
        if unexpected:
            print(f"  [FrozenExpert] Unexpected keys: {unexpected}")

        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.has_routing:
            logits, _ = self.model(images)
            return logits
        return self.model(images)


# ---------------------------------------------------------------------------
# DACE-C Trainer  (ResNet32WithRouting + CE + contrastive routing)
# ---------------------------------------------------------------------------

class DACECTrainer(BaseTrainer):
    """
    Trainer for DACE Expert C (Tail specialist, CE + contrastive routing).

    Experts A and B are frozen.  For each batch:
      1. Forward A → probs_a
      2. Forward B → probs_b
      3. avg_probs = (probs_a + probs_b) / 2
      4. Forward C → logits_c, probs_c, embeddings_c
      5. KL(avg_probs || probs_c) → agreement labels
      6. L_contrastive pulls together embeddings with same agreement label
      7. L_total = L_cls + λ * L_contrastive
    """

    def __init__(
        self,
        expert_a: FrozenExpert,
        expert_b: FrozenExpert,
        lambda_routing: float = 0.05,
        kl_threshold: float = 0.1,
        **kwargs,
    ):
        self.model = ResNet32WithRouting(num_classes=100, routing_dim=32)
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.lambda_routing = lambda_routing
        self.kl_threshold = kl_threshold

        self.loss_ce = CELoss()
        self.loss_contrastive = ContrastiveRoutingLoss(temperature=0.5)

        super().__init__(
            model=self.model,
            loss_fn=CELoss(),  # used by BaseTrainer.validate()
            expert_name='DACE_C',
            **kwargs,
        )

    def _compute_loss(self, images, targets, weights=None):
        batch_size = images.size(0)

        # 1. Forward Expert A (frozen)
        with torch.no_grad():
            logits_a = self.expert_a.forward_logits(images)
            probs_a = F.softmax(logits_a, dim=1)

            # 2. Forward Expert B (frozen)
            logits_b = self.expert_b.forward_logits(images)
            probs_b = F.softmax(logits_b, dim=1)

            # 3. Ensemble average
            avg_probs = (probs_a + probs_b) / 2.0
            # Convert back to logits for KL computation
            avg_logits = torch.log(avg_probs + 1e-12)

        # 4. Forward Expert C
        logits_c, routing_emb = self.model(images)
        probs_c = F.softmax(logits_c, dim=1)

        # 5. Classification loss (standard CE)
        loss_cls = self.loss_ce(logits_c, targets)

        # 6. Compute KL divergence and agreement labels
        # KL(avg(A,B) || C)
        kl_labels = compute_agreement_label(
            avg_logits, logits_c,
            threshold=self.kl_threshold, metric='kl',
        )

        # 7. Contrastive routing loss
        loss_contrastive, aux = self.loss_contrastive(routing_emb, kl_labels)

        # 8. Total loss
        loss = loss_cls + self.lambda_routing * loss_contrastive

        if weights is not None:
            loss = (loss * weights).mean()

        return loss, {
            'loss_cls': loss_cls.detach(),
            'loss_contrastive': loss_contrastive.detach(),
            'kl_agree_ratio': (kl_labels == 0).float().mean().detach(),
        }

    def _forward_for_eval(self, images):
        return self.model.forward_logits(images)

    def validate(self, loader):
        metrics = super().validate(loader)
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

    def train(self, train_loader, val_loader, class_counts=None):
        # Use step LR schedule (like PaCo) instead of cosine for longer training
        # Override the scheduler set in __init__
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimiser, milestones=[320, 360], gamma=0.1,
        )
        return super().train(train_loader, val_loader, class_counts=class_counts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DACE Step 3: Train Expert C (Tail)')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--dace-a-ckpt', default=None,
                        help='Path to DACE_A_best.pt checkpoint')
    parser.add_argument('--dace-b-ckpt', default=None,
                        help='Path to DACE_B_best.pt checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--full-ratio', type=float, default=0.2)
    parser.add_argument('--lambda-routing', type=float, default=0.05,
                        help='Weight for contrastive routing loss')
    parser.add_argument('--kl-threshold', type=float, default=0.1,
                        help='KL threshold for agree/disagree labeling')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── resolve checkpoint paths ──────────────────────────────────────
    ckpt_dir = args.checkpoint_dir
    if args.dace_a_ckpt is None:
        args.dace_a_ckpt = os.path.join(ckpt_dir, 'DACE_A_best.pt')
    if args.dace_b_ckpt is None:
        args.dace_b_ckpt = os.path.join(ckpt_dir, 'DACE_B_best.pt')

    for name, path in [('A', args.dace_a_ckpt), ('B', args.dace_b_ckpt)]:
        if not os.path.exists(path):
            print(f"[Error] Expert {name} checkpoint not found: {path}")
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
    tail_groups = get_tail_classes(class_counts)

    train_set = PartitionedDataset(base_train_set)
    train_sampler = PartitionedSampler(
        base_train_set,
        primary_groups=tail_groups,
        full_ratio=args.full_ratio,
        num_samples=len(base_train_set),
        primary_name='Tail',
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

    # ── load frozen experts ───────────────────────────────────────────
    print(f"\nLoading frozen Expert A from {args.dace_a_ckpt}...")
    expert_a = FrozenExpert(args.dace_a_ckpt, has_routing=False, device=args.device)

    print(f"Loading frozen Expert B from {args.dace_b_ckpt}...")
    expert_b = FrozenExpert(args.dace_b_ckpt, has_routing=True, device=args.device)

    # ── train ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DACE Step 3: Training Expert C (Tail specialist)")
    print(f"  Tail classes: {tail_groups['tail'].tolist()}")
    print(f"  λ_routing: {args.lambda_routing}")
    print(f"  KL threshold: {args.kl_threshold}")
    print(f"  Full ratio: {args.full_ratio}")
    print(f"{'='*60}\n")

    trainer = DACECTrainer(
        expert_a=expert_a,
        expert_b=expert_b,
        lambda_routing=args.lambda_routing,
        kl_threshold=args.kl_threshold,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train(train_loader, val_loader, class_counts=class_counts)
    trainer.save_history()
    print(f"  Expert C checkpoint saved to {args.checkpoint_dir}/DACE_C_best.pt")


if __name__ == '__main__':
    main()
