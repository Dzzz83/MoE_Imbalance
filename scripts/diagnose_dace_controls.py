#!/usr/bin/env python3
"""
DACE diagnostic #2 — two control experiments to resolve a contradiction.

Contradiction to resolve:
  (a) The recorded L_contrastive sat at ~4.80-4.84 for the whole run, which
      looks like "the routing head never learned anything".
  (b) A linear probe on the routing embeddings separates agree/disagree with
      AUROC 0.96 (Expert B) — which looks like "it learned a lot".

Both cannot be interpreted naively.  Two confounds are tested here:

  CONTROL 1 — Is 4.84 actually the chance level, or the FLOOR?
      With ~96% of samples in one class, the contrastive loss is compressed:
      even PERFECT clustering cannot push it far below ln(batch-1).  Compute
      the achievable range (random vs perfectly-clustered) at the real label
      ratio and compare with the observed trajectory.

  CONTROL 2 — Does the probe measure routing-head learning, or sample identity?
      Replace the TRAINED routing head with a RANDOM one on the SAME frozen
      backbone features and re-run the identical probe.  If AUROC stays high,
      the probe was reading sample identity already present in the shared
      backbone features, not any contrastive structure the head learned.

Read-only. Uses the long-tailed validation split for measurement only.
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32, ResNet32WithRouting, RoutingHead
from losses.contrastive_routing_loss import ContrastiveRoutingLoss
from data.cifar_lt import LongTailCIFAR100
from scripts.diagnose_dace import auroc, kl_per_sample, load_expert


def probe_auroc(X, y, seed=0, ridge=1e-2):
    """2-fold linear probe AUROC."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    half = len(y) // 2
    tr, te = idx[:half], idx[half:]
    w = np.linalg.solve(X[tr].T @ X[tr] + ridge * np.eye(X.shape[1]),
                        X[tr].T @ y[tr])
    return auroc(X[te] @ w, y[te])


def main():
    dev = 'cpu'
    ck = './checkpoints'
    torch.manual_seed(42)

    ea = load_expert(os.path.join(ck, 'DACE_A_best.pt'), False, dev)
    eb = load_expert(os.path.join(ck, 'DACE_B_best.pt'), True, dev)
    ec = load_expert(os.path.join(ck, 'DACE_C_best.pt'), True, dev)

    val_idx = np.load('data/processed/lt_val_indices.npy')
    val_set = LongTailCIFAR100(root='./data', base_train_indices=val_idx,
                               imbalance_ratio=100.0, train=False,
                               download=False, already_subsampled=True)
    loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    LA, LB, LC = [], [], []
    EBtr, ECtr, FEATb = [], [], []
    for images, _ in loader:
        with torch.no_grad():
            LA.append(ea(images)); 
            lb, ebv = eb(images); LB.append(lb); EBtr.append(ebv)
            lc, ecv = ec(images); LC.append(lc); ECtr.append(ecv)
            # backbone features before the routing head (shared with classifier)
            FEATb.append(eb.backbone(images))
    LA, LB, LC = torch.cat(LA), torch.cat(LB), torch.cat(LC)
    EBtr, ECtr = torch.cat(EBtr), torch.cat(ECtr)
    FEATb = torch.cat(FEATb)

    kl_ab = kl_per_sample(LA, LB)
    PA, PB = F.softmax(LA, 1), F.softmax(LB, 1)
    kl_ac = kl_per_sample(torch.log((PA + PB) / 2 + 1e-12), LC)

    print("=" * 72)
    print("  CONTROL 1: IS 4.84 THE CHANCE LEVEL OR THE FLOOR?")
    print("=" * 72)

    loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    for name, kl in (('B', kl_ab), ('C', kl_ac)):
        for tau in (0.1, 1.0):
            labels = (kl > tau).long()
            pos_frac = float((labels == 1).float().mean())
            print(f"\n  Expert {name}, tau={tau}: disagree fraction = {pos_frac:.4f}")

            # (i) random embeddings = chance
            Ls_rand, Ls_perf = [], []
            for _ in range(5):
                emb = torch.randn(len(labels), 32)
                l, _ = loss_fn(emb, labels)
                Ls_rand.append(l.item())
                # (ii) perfectly clustered: one point per class
                emb2 = torch.zeros(len(labels), 32)
                emb2[labels == 1, 0] = 1.0
                emb2[labels == 0, 1] = 1.0
                emb2 = emb2 + 1e-4 * torch.randn_like(emb2)  # break exact ties
                l2, _ = loss_fn(emb2, labels)
                Ls_perf.append(l2.item())
            print(f"    chance level (random embeddings)   : {np.mean(Ls_rand):.4f}")
            print(f"    FLOOR (perfectly clustered)        : {np.mean(Ls_perf):.4f}")
            print(f"    achievable range                   : "
                  f"{np.mean(Ls_perf):.4f} .. {np.mean(Ls_rand):.4f} "
                  f"(width {np.mean(Ls_rand)-np.mean(Ls_perf):.4f})")

    # Observed trajectory
    print("\n  Observed training trajectory:")
    for nm in ('DACE_B', 'DACE_C'):
        h = json.load(open(f'checkpoints/{nm}_history.json'))
        L = [e['train_loss_contrastive'] for e in h]
        print(f"    {nm}: start {L[0]:.4f} -> end {L[-1]:.4f} "
              f"(change {L[0]-L[-1]:+.4f}, min {min(L):.4f})")

    print("\n" + "=" * 72)
    print("  CONTROL 2: DOES THE PROBE MEASURE THE HEAD, OR SAMPLE IDENTITY?")
    print("=" * 72)
    print("  Same frozen backbone features; trained head vs RANDOM head.\n")

    for name, kl, feat, head in (
            ('B', kl_ab, FEATb, eb.routing_head),
            ('C', kl_ac, None, ec.routing_head)):
        if feat is None:
            continue
        for tau in (0.1, 1.0):
            y = (kl > tau).float().numpy()
            if y.sum() < 10 or (1 - y).sum() < 10:
                continue
            emb_trained = head(feat).detach()
            Xtr = torch.cat([emb_trained, torch.ones(len(emb_trained), 1)], 1).numpy()
            a_trained = probe_auroc(Xtr, y)

            # random head of identical shape on the SAME features
            rand_head = RoutingHead(64, 32)
            torch.manual_seed(123)
            for p in rand_head.parameters():
                torch.nn.init.normal_(p, 0, 0.1)
            emb_rand = rand_head(feat).detach()
            Xr = torch.cat([emb_rand, torch.ones(len(emb_rand), 1)], 1).numpy()
            a_rand = probe_auroc(Xr, y)

            print(f"  Expert {name}, tau={tau}:")
            print(f"    probe AUROC, TRAINED routing head : {a_trained:.4f}")
            print(f"    probe AUROC, RANDOM  routing head : {a_rand:.4f}")
            print(f"    -> head-specific contribution     : "
                  f"{a_trained - a_rand:+.4f}")

    # Sanity: can a probe read the agree label straight off the shared
    # backbone features alone (no routing head at all)?
    print("\n  Reference — probe on the SHARED BACKBONE features directly")
    print("  (no routing head involved; both classes of the task share it):")
    for tau in (0.1, 1.0):
        y = (kl_ab > tau).float().numpy()
        if y.sum() < 10 or (1 - y).sum() < 10:
            continue
        X = torch.cat([FEATb, torch.ones(len(FEATb), 1)], 1).numpy()
        print(f"    tau={tau}: backbone-only probe AUROC = {probe_auroc(X, y):.4f}")


if __name__ == '__main__':
    main()
