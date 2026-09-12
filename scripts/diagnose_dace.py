#!/usr/bin/env python3
"""
DACE diagnostic — WHY did routing collapse to "always Expert B"?

Read-only. Loads the trained DACE checkpoints and answers five questions with
evidence instead of speculation:

  Q1  What are the actual KL magnitudes?  (is tau=0.1 sane?)
  Q2  Do the routing embeddings encode the agree/disagree label at all?
      (linear probe AUROC — 0.5 means the head learned nothing)
  Q3  Is the routing score predictive of expert correctness?
      (the fundamental question for whether this routing signal can work)
  Q4  Are score_a / score_b / score_c on a comparable scale?
      (argmax across incomparable scales is arbitrary)
  Q5  What is the oracle headroom, and what does uniform vs routing capture?

Uses the long-tailed VALIDATION split for measurement (permitted by the DACE
plan for post-hoc analysis; never used for training).  No training, no writes.

Usage:
    python scripts/diagnose_dace.py [--data-root ./data] [--checkpoint-dir ./checkpoints]
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32, ResNet32WithRouting
from data.cifar_lt import LongTailCIFAR100


def load_expert(path, has_routing, device):
    model = (ResNet32WithRouting(100, 32) if has_routing else ResNet32(100))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device).eval()


@torch.no_grad()
def collect(expert_a, expert_b, expert_c, loader, device):
    """Collect logits + routing embeddings + labels for the whole split."""
    A, B, C, EB, EC, Y = [], [], [], [], [], []
    for images, targets in loader:
        images = images.to(device)
        la = expert_a(images)
        lb, eb = expert_b(images)
        lc, ec = expert_c(images)
        A.append(la.cpu()); B.append(lb.cpu()); C.append(lc.cpu())
        EB.append(eb.cpu()); EC.append(ec.cpu()); Y.append(targets)
    return (torch.cat(A), torch.cat(B), torch.cat(C),
            torch.cat(EB), torch.cat(EC), torch.cat(Y))


def kl_per_sample(logits_p, logits_q):
    """KL(softmax(p) || softmax(q)) per sample."""
    lp = F.log_softmax(logits_p, dim=1)
    lq = F.log_softmax(logits_q, dim=1)
    return (lp.exp() * (lp - lq)).sum(dim=1)


def auroc(scores, binary):
    """Rank-based AUROC with tie-corrected average ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    binary = np.asarray(binary, dtype=bool)
    n_pos, n_neg = int(binary.sum()), int((~binary).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # Average ranks for tied scores
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = (csum - counts + csum + 1) / 2.0
    ranks = avg[inv]
    return (ranks[binary].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--checkpoint-dir', default='./checkpoints')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-batches', type=int, default=0,
                    help='0 = use whole split')
    args = ap.parse_args()

    dev = args.device
    ck = args.checkpoint_dir

    print("=" * 72)
    print("  DACE DIAGNOSTIC")
    print("=" * 72)

    ea = load_expert(os.path.join(ck, 'DACE_A_best.pt'), False, dev)
    eb = load_expert(os.path.join(ck, 'DACE_B_best.pt'), True, dev)
    ec = load_expert(os.path.join(ck, 'DACE_C_best.pt'), True, dev)

    val_idx = np.load(f'{args.data_root}/processed/lt_val_indices.npy')
    val_set = LongTailCIFAR100(root=args.data_root, base_train_indices=val_idx,
                               imbalance_ratio=100.0, train=False,
                               download=False, already_subsampled=True)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    LA, LB, LC, EB_e, EC_e, Y = collect(ea, eb, ec, loader, dev)
    n = len(Y)
    PA, PB, PC = (F.softmax(LA, 1), F.softmax(LB, 1), F.softmax(LC, 1))
    pred_a, pred_b, pred_c = LA.argmax(1), LB.argmax(1), LC.argmax(1)
    cor_a, cor_b, cor_c = (pred_a == Y), (pred_b == Y), (pred_c == Y)

    print(f"\nValidation samples: {n}")

    # ── Q1: KL magnitudes ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Q1: ACTUAL KL MAGNITUDES  (training used tau=0.1)")
    print("=" * 72)
    kl_ab = kl_per_sample(LA, LB)
    kl_ac = kl_per_sample(torch.log((PA + PB) / 2 + 1e-12), LC)
    for name, kl in (('KL(A||B)', kl_ab), ('KL(avg(A,B)||C)', kl_ac)):
        q = np.percentile(kl.numpy(), [1, 5, 25, 50, 75, 95, 99])
        print(f"\n  {name}")
        print(f"    mean {kl.mean():.4f}   median {kl.median():.4f}   "
              f"max {kl.max():.4f}")
        print(f"    p1={q[0]:.4f} p5={q[1]:.4f} p25={q[2]:.4f} p50={q[3]:.4f} "
              f"p75={q[4]:.4f} p95={q[5]:.4f} p99={q[6]:.4f}")
        for tau in [0.1, 0.3, 0.5, 1.0, 2.0]:
            frac = (kl > tau).float().mean().item()
            print(f"    tau={tau:<4} -> disagree {100*frac:5.1f}%  "
                  f"agree {100*(1-frac):5.1f}%")

    # ── Q2: do embeddings encode the label? ──────────────────────────────
    print("\n" + "=" * 72)
    print("  Q2: DO ROUTING EMBEDDINGS ENCODE AGREE/DISAGREE?")
    print("      (linear probe AUROC; 0.50 = no information)")
    print("=" * 72)
    torch.manual_seed(0)
    for tau_probe in [0.1, 0.5, 1.0, 2.0]:
        lab_b = (kl_ab > tau_probe).float()
        lab_c = (kl_ac > tau_probe).float()
        for nm, emb, lab in (('B', EB_e, lab_b), ('C', EC_e, lab_c)):
            if lab.sum() < 10 or (1 - lab).sum() < 10:
                print(f"  tau={tau_probe:<4} expert {nm}: "
                      f"degenerate labels (pos={int(lab.sum())}) -> skipped")
                continue
            # simple ridge-regularised least squares probe, 2-fold
            X = torch.cat([emb, torch.ones(len(emb), 1)], 1).numpy()
            y = lab.numpy()
            # seeded: the probe split is a reported result, not noise
            idx = np.arange(len(y)); np.random.default_rng(0).shuffle(idx)
            half = len(y) // 2
            tr, te = idx[:half], idx[half:]
            w = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]),
                                X[tr].T @ y[tr])
            s = X[te] @ w
            auc = auroc(s, y[te])
            print(f"  tau={tau_probe:<4} expert {nm}: probe AUROC = {auc:.4f}")

    # ── Q3: does the score predict correctness? ──────────────────────────
    print("\n" + "=" * 72)
    print("  Q3: IS THE ROUTING SCORE PREDICTIVE OF CORRECTNESS?")
    print("      (AUROC of score vs 'this expert is correct'; 0.50 = useless)")
    print("=" * 72)
    def proto(emb, lab):
        lab = lab.bool()
        if lab.sum() > 0 and (~lab).sum() > 0:
            return emb[~lab].mean(0), emb[lab].mean(0)
        m = emb.mean(0)
        return m, m
    for tau_p in [0.1, 1.0]:
        pa_ag, pa_dis = proto(EB_e, kl_ab > tau_p)
        pc_ag, pc_dis = proto(EC_e, kl_ac > tau_p)
        nb, nc = F.normalize(EB_e, 1), F.normalize(EC_e, 1)
        s_b = (F.cosine_similarity(nb, F.normalize(pa_dis.unsqueeze(0), 1))
               - F.cosine_similarity(nb, F.normalize(pa_ag.unsqueeze(0), 1)))
        s_c = (F.cosine_similarity(nc, F.normalize(pc_dis.unsqueeze(0), 1))
               - F.cosine_similarity(nc, F.normalize(pc_ag.unsqueeze(0), 1)))
        print(f"\n  tau={tau_p}")
        print(f"    score_B vs B-correct : AUROC {auroc(s_b.numpy(), cor_b.numpy()):.4f}")
        print(f"    score_C vs C-correct : AUROC {auroc(s_c.numpy(), cor_c.numpy()):.4f}")
        print(f"    score_B vs B-WRONG   : AUROC {auroc(s_b.numpy(), (~cor_b).numpy()):.4f}")

    # ── Q4: score scale comparability ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Q4: ARE THE THREE SCORES ON A COMPARABLE SCALE?")
    print("=" * 72)
    pa_ag, pa_dis = proto(EB_e, kl_ab > 0.1)
    pc_ag, pc_dis = proto(EC_e, kl_ac > 0.1)
    nb, nc = F.normalize(EB_e, 1), F.normalize(EC_e, 1)
    s_b = (F.cosine_similarity(nb, F.normalize(pa_dis.unsqueeze(0), 1))
           - F.cosine_similarity(nb, F.normalize(pa_ag.unsqueeze(0), 1)))
    s_c = (F.cosine_similarity(nc, F.normalize(pc_dis.unsqueeze(0), 1))
           - F.cosine_similarity(nc, F.normalize(pc_ag.unsqueeze(0), 1)))
    s_a = -(s_b + s_c) / 2.0
    for nm, s in (('score_a (derived)', s_a), ('score_b', s_b), ('score_c', s_c)):
        print(f"  {nm:18s} mean {s.mean():+.4f}  std {s.std():.4f}  "
              f"min {s.min():+.4f}  max {s.max():+.4f}")
    stack = torch.stack([s_a, s_b, s_c], 1)
    win = stack.argmax(1)
    for i, nm in enumerate(['A', 'B', 'C']):
        print(f"  argmax picks {nm}: {100*(win==i).float().mean():5.1f}%")

    # cosine similarity between B and C embeddings (the fallback gate)
    sim_bc = F.cosine_similarity(nb, nc)
    print(f"\n  emb_sim(B,C): mean {sim_bc.mean():+.4f} std {sim_bc.std():.4f} "
          f"max {sim_bc.max():.4f}")
    for th in [0.3, 0.5, 0.7]:
        print(f"    fraction > {th} (uniform fallback would fire): "
              f"{100*(sim_bc>th).float().mean():.2f}%")

    # ── Q5: headroom accounting ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Q5: HEADROOM ACCOUNTING (validation split)")
    print("=" * 72)
    uni = (LA + LB + LC) / 3.0
    pu = uni.argmax(1)
    def ba(pred):
        accs = []
        for c in torch.unique(Y):
            m = (Y == c)
            accs.append((pred[m] == c).float().mean().item())
        return float(np.mean(accs))
    print(f"  BA  A {ba(pred_a):.4f}   B {ba(pred_b):.4f}   C {ba(pred_c):.4f}")
    print(f"  BA  uniform {ba(pu):.4f}")
    any_correct = cor_a | cor_b | cor_c
    print(f"  BA  oracle (>=1 correct exists) {ba(torch.where(any_correct, Y, pred_a)):.4f}")
    print(f"  all-wrong fraction: {100*(~any_correct).float().mean():.2f}%")
    print(f"\n  correct-count distribution:")
    cnt = cor_a.long() + cor_b.long() + cor_c.long()
    for k in range(4):
        print(f"    exactly {k} correct: {int((cnt==k).sum()):5d} "
              f"({100*(cnt==k).float().mean():5.2f}%)")


if __name__ == '__main__':
    main()
