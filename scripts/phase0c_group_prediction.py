#!/usr/bin/env python3
"""
PHASE 0c — is GROUP PREDICTION the real bottleneck?

Phase 0 established:
  * direct routing candidates (calibrated correctness, group-conditional with
    PREDICTED groups) do not beat uniform beyond noise;
  * the group ORACLE (TRUE class group) passes both criteria comfortably:
    +2.23pp BA, +10.53pp Tail.

That difference isolates the bottleneck: not the combination rule, but the
ability to tell which class group a sample belongs to.  This script measures
that directly.

It caches the forward pass to data/processed/dace_val_cache.npz so subsequent
Phase 0 analyses do not repeat it.

Validation only; test set never touched.
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cifar_lt import LongTailCIFAR100
from scripts.diagnose_dace import load_expert
from scripts.phase0_ceiling_probe import (
    fit_logreg, apply_logreg, balanced_accuracy, group_accuracy,
)

CACHE = 'data/processed/dace_val_cache.npz'


def get_val_arrays():
    """Forward pass over validation, cached to disk."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print(f"  (loaded cached activations from {CACHE})")
        return {k: z[k] for k in z.files}

    dev = 'cpu'
    ea = load_expert('./checkpoints/DACE_A_best.pt', False, dev)
    eb = load_expert('./checkpoints/DACE_B_best.pt', True, dev)
    ec = load_expert('./checkpoints/DACE_C_best.pt', True, dev)

    val_idx = np.load('data/processed/lt_val_indices.npy')
    val_set = LongTailCIFAR100(root='./data', base_train_indices=val_idx,
                               imbalance_ratio=100.0, train=False,
                               download=False, already_subsampled=True)
    loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    LA, LB, LC, FA, FB, FC, EBv, ECv, Y = [], [], [], [], [], [], [], [], []
    for images, targets in loader:
        with torch.no_grad():
            LA.append(ea(images))
            lb, ebv = eb(images); LB.append(lb); EBv.append(ebv)
            lc, ecv = ec(images); LC.append(lc); ECv.append(ecv)
            FA.append(ea.backbone(images)); FB.append(eb.backbone(images))
            FC.append(ec.backbone(images)); Y.append(targets)

    out = {'LA': torch.cat(LA).numpy(), 'LB': torch.cat(LB).numpy(),
           'LC': torch.cat(LC).numpy(), 'FA': torch.cat(FA).numpy(),
           'FB': torch.cat(FB).numpy(), 'FC': torch.cat(FC).numpy(),
           'EB': torch.cat(EBv).numpy(), 'EC': torch.cat(ECv).numpy(),
           'Y': torch.cat(Y).numpy()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"  (cached activations to {CACHE})")
    return out


def main():
    print("=" * 78)
    print("  PHASE 0c — IS GROUP PREDICTION THE BOTTLENECK?")
    print("=" * 78)

    A = get_val_arrays()
    LA, LB, LC, Yn = A['LA'], A['LB'], A['LC'], A['Y']
    n = len(Yn)
    LOGITS = np.stack([LA, LB, LC], 1)

    tr = np.load('data/processed/lt_train_indices.npy')
    from torchvision import datasets
    full = datasets.CIFAR100(root='./data', train=True, download=False)
    tgt = np.asarray(full.targets)[tr]
    counts = np.array([(tgt == c).sum() for c in range(100)])
    G = {'Head': np.where(counts >= 100)[0],
         'Med': np.where((counts >= 20) & (counts < 100))[0],
         'Tail': np.where(counts < 20)[0]}
    gnames = ['Head', 'Med', 'Tail']

    # true group per validation sample
    true_g = np.full(n, -1)
    for gi, g in enumerate(gnames):
        true_g[np.isin(Yn, G[g])] = gi
    assert (true_g >= 0).all()

    probs = {k: F.softmax(torch.tensor(LOGITS[:, i]), 1).numpy()
             for i, k in enumerate('ABC')}
    ens_p = (probs['A'] + probs['B'] + probs['C']) / 3.0
    group_p = np.stack([ens_p[:, G[g]].sum(1) for g in gnames], 1)
    pred_g_ens = group_p.argmax(1)

    print("\n  [A] Group prediction accuracy from the ENSEMBLE probabilities")
    acc = (pred_g_ens == true_g).mean()
    print(f"      overall accuracy: {acc:.4f}")
    print("\n      confusion (rows = true group, cols = predicted):")
    print(f"      {'':<8}" + "".join(f"{g:>10}" for g in gnames) + f"{'recall':>10}")
    for gi, g in enumerate(gnames):
        row = [(pred_g_ens[true_g == gi] == gj).sum() for gj in range(3)]
        tot = (true_g == gi).sum()
        cells = "".join(f"{c/max(tot,1):>10.3f}" for c in row)
        print(f"      {g:<8}{cells}{(row[gi]/max(tot,1)):>10.3f}")

    # how often is the true group even in the top-2?
    top2 = np.argsort(-group_p, 1)[:, :2]
    in_top2 = np.array([true_g[i] in top2[i] for i in range(n)]).mean()
    print(f"\n      true group in top-2 of predicted group probs: {in_top2:.4f}")

    # ── competence table + routing, fitted on half, evaluated on half ──
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    fit, ev = perm[:n // 2], perm[n // 2:]
    P = {k: LOGITS[:, i].argmax(1) for i, k in enumerate('ABC')}

    comp = np.zeros((3, 3))
    for i, k in enumerate('ABC'):
        for gi, g in enumerate(gnames):
            accs = [(P[k][fit][Yn[fit] == c] == c).mean()
                    for c in G[g] if (Yn[fit] == c).sum() > 0]
            comp[i, gi] = float(np.mean(accs)) if accs else 0.0
    print("\n  [B] Competence table (per-class recall, fitted on half):")
    print(f"      {'expert':<10}" + "".join(f"{g:>10}" for g in gnames))
    for i, k in enumerate('ABC'):
        print(f"      {k:<10}" + "".join(f"{comp[i, gi]:>10.4f}" for gi in range(3)))

    def combine(w, idx):
        return np.einsum('ne,nec->nc', w[idx], LOGITS[idx])

    uni = np.full((n, 3), 1 / 3.0)
    ba_u = balanced_accuracy(Yn[ev], combine(uni, ev).argmax(1))
    tl_u = group_accuracy(Yn[ev], combine(uni, ev).argmax(1), G['Tail'])

    # predicted-group routing
    w_pred = (group_p @ comp.T)
    w_pred /= np.clip(w_pred.sum(1, keepdims=True), 1e-9, None)
    p_pred = combine(w_pred, ev).argmax(1)
    # oracle-group routing
    onehot = np.zeros((n, 3)); onehot[np.arange(n), true_g] = 1.0
    w_or = (onehot @ comp.T)
    w_or /= np.clip(w_or.sum(1, keepdims=True), 1e-9, None)
    p_or = combine(w_or, ev).argmax(1)

    print("\n  [C] Routing with the competence table (eval half, n=%d)" % len(ev))
    print(f"      {'variant':<34}{'BA':>9}{'Tail':>9}")
    print(f"      {'uniform':<34}{ba_u:>9.4f}{tl_u:>9.4f}")
    print(f"      {'PREDICTED group':<34}{balanced_accuracy(Yn[ev], p_pred):>9.4f}"
          f"{group_accuracy(Yn[ev], p_pred, G['Tail']):>9.4f}")
    print(f"      {'ORACLE group':<34}{balanced_accuracy(Yn[ev], p_or):>9.4f}"
          f"{group_accuracy(Yn[ev], p_or, G['Tail']):>9.4f}")

    # what if group prediction were perfect on the samples it gets right?
    perfect_on_correct = true_g.copy()
    swapped = pred_g_ens.copy()
    p_hyb = np.zeros((n, 3))
    ok = (pred_g_ens == true_g)
    p_hyb[ok] = group_p[ok]
    p_hyb[~ok] = onehot[~ok]
    w_hyb = (p_hyb @ comp.T)
    w_hyb /= np.clip(w_hyb.sum(1, keepdims=True), 1e-9, None)
    p_h = combine(w_hyb, ev).argmax(1)
    print(f"      {'predicted ORACLE-fixed group':<34}"
          f"{balanced_accuracy(Yn[ev], p_h):>9.4f}"
          f"{group_accuracy(Yn[ev], p_h, G['Tail']):>9.4f}")

    # ── direct group classifier on expert logits+embeddings ────────────
    Ftr = np.hstack([A['FA'], A['FB'], A['FC'], A['EB'], A['EC'],
                     LOGITS.reshape(n, -1)])
    best = None
    for gi in range(3):
        y = (true_g == gi).astype(float)
        w_, mu_, sd_ = fit_logreg(Ftr[fit], y[fit], l2=1.0)
        s = apply_logreg(w_, mu_, sd_, Ftr)
        best = s[:, None] if best is None else np.hstack([best, s[:, None]])
    acc_clf = (best.argmax(1) == true_g).mean()
    print("\n  [D] Dedicated 3-way group classifier (logits + embeddings)")
    print(f"      accuracy: {acc_clf:.4f}   (ensemble-probability rule: {acc:.4f})")

    print("\n" + "=" * 78)
    print("  INTERPRETATION")
    print("=" * 78)
    tail_recall = (pred_g_ens[true_g == 2] == 2).mean()
    head_recall = (pred_g_ens[true_g == 0] == 0).mean()
    med_recall = (pred_g_ens[true_g == 1] == 1).mean()
    gap = (balanced_accuracy(Yn[ev], p_or) - balanced_accuracy(Yn[ev], p_pred))
    print(f"  Per-group recall: Head {head_recall:.3f}, Med {med_recall:.3f}, "
          f"Tail {tail_recall:.3f}")
    print(f"  BA lost purely to group-prediction error: {gap:+.4f}")
    print()
    print("  NOTE: overall group accuracy is misleading here. Head dominates the")
    print("  split, so a rule can score well while being useless where it matters.")
    if tail_recall < 0.50:
        print(f"  -> TAIL recall is only {tail_recall:.3f}: the router cannot tell a")
        print("     tail sample from a head sample, which is exactly the regime where")
        print("     routing would have to act. Group prediction IS the binding")
        print("     constraint, and it is hard because recognising tail samples IS")
        print("     the long-tail problem the experts already fail at.")
    else:
        print("  -> Group prediction is usable across all three groups.")
    print()
    print("  Competence table read: check whether the intended specialisation")
    print("  (A=Head, B=Med, C=Tail) actually materialised.")
    for gi, g in enumerate(gnames):
        col = comp[:, gi]
        best_e = 'ABC'[int(col.argmax())]
        print(f"    best expert on {g:<5}: {best_e} ({col.max():.4f})   "
              f"all: A={col[0]:.3f} B={col[1]:.3f} C={col[2]:.3f}")
    print("=" * 78)


if __name__ == '__main__':
    main()
