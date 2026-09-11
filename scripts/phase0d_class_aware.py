#!/usr/bin/env python3
"""
PHASE 0d — CLASS-AWARE ensemble weighting (per-CLASS weights, not per-sample).

Motivation
----------
Every routing method tested in this project used SCALAR per-expert weights
(one number per expert for all classes).  docs/results.md section 5.3 flags the
untested alternative:

    "PaCo dominates Head/Med (72%/43%) but is weak on Tail (12%).  LAL/BS are
     weaker on Head (60%) but much stronger on Tail (23%).  A class-aware or
     confidence-based router could exploit this."

Phase 0c showed that PER-SAMPLE group routing cannot exploit it, because tail
recall is 0.114 -- the router cannot identify a tail sample.

This script tests the alternative that avoids that failure entirely.  Instead of
deciding "which group is this sample in?", we weight each expert's score FOR
EACH CLASS by how competent that expert is ON THAT CLASS:

    z[c] = sum_e  w[e][c] * logits_e[c]

The class index c is always known, so NO group prediction is required.  This is
a static, class-aware combination rule (a generalisation of "optimal fixed
weights", which used a single scalar per expert).

Estimator
---------
w[:, c] is built from per-class recall measured on held-out data, with
empirical-Bayes shrinkage toward the group-level competence so that rare classes
(which have 0-2 held-out samples) are not driven by noise:

    recall_shrunk[e][c] = (n_c * recall[e][c] + k * recall_group[e][g(c)]) / (n_c + k)

then  w[:, c] = softmax_e( recall_shrunk[e][c] / T ).
T -> inf gives uniform weights, i.e. EXACTLY the uniform baseline, so the
baseline is inside the search space.  Both k (shrinkage) and T (sharpness) are
selected on a TUNE split and reported on an unseen EVAL split.

Protocol
--------
Validation only; the 10K test set is never touched.  Production would estimate
the weights on routing_dev (carved from train) instead of validation.

Reproduce:
    python scripts/phase0d_class_aware.py
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
from models.resnet32 import ResNet32, PaCoResNet32
from scripts.diagnose_dace import load_expert
from scripts.phase0_ceiling_probe import balanced_accuracy, group_accuracy

CK = './checkpoints'
CACHE_DIR = 'data/processed'

POOLS = {
    'DACE (partitioned)': [
        ('DACE_A_best.pt', 'plain'),
        ('DACE_B_best.pt', 'routing'),
        ('DACE_C_best.pt', 'routing'),
    ],
    'ORIGINAL (LAL+PaCo+Mixup)': [
        ('LAL_best.pt', 'plain'),
        ('PaCo_best.pt', 'paco'),
        ('Mixup_best.pt', 'plain'),
    ],
}


def load_model(path, kind, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get('model_state_dict', ck)
    if kind == 'routing':
        m = load_expert(path, True, device)
        return m, kind
    if kind == 'paco':
        m = PaCoResNet32(100, dim=32, K=1024, m=0.999, mlp=True)
    else:
        m = ResNet32(100)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m, kind


def forward_logits(model, kind, x):
    if kind == 'routing':
        return model(x)[0]
    return model(x)


def get_pool_logits(pool_name, specs):
    """Validation logits for a pool of experts, cached to disk."""
    tag = pool_name.split()[0].lower()
    cache = os.path.join(CACHE_DIR, f'phase0d_{tag}_val.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"    (cache hit: {cache})")
        return [z[f'L{i}'] for i in range(len(specs))], z['Y']

    val_idx = np.load('data/processed/lt_val_indices.npy')
    val_set = LongTailCIFAR100(root='./data', base_train_indices=val_idx,
                               imbalance_ratio=100.0, train=False,
                               download=False, already_subsampled=True)
    loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    models = [load_model(os.path.join(CK, p), k) for p, k in specs]
    bufs = [[] for _ in specs]
    Y = []
    for images, targets in loader:
        with torch.no_grad():
            for i, (m, k) in enumerate(models):
                bufs[i].append(forward_logits(m, k, images))
            Y.append(targets)
    Ls = [torch.cat(b).numpy() for b in bufs]
    Y = torch.cat(Y).numpy()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, Y=Y, **{f'L{i}': Ls[i] for i in range(len(Ls))})
    print(f"    (cached: {cache})")
    return Ls, Y


def class_weights(recall, group_of_class, comp_group, n_c, k, T):
    """Build w[e][c] from shrunk per-class recall.  Shape (E, C).

    Classes absent from the fit split (n_c == 0) fall back entirely to the
    group-level competence; without this, k=0 produces 0/0 = NaN weights, which
    silently poisons the hyperparameter selection.
    """
    E, C = recall.shape
    fallback = comp_group[:, group_of_class]
    num = n_c[None, :] * recall + k * fallback
    den = n_c[None, :] + k
    shrunk = np.where(den > 0, num / np.maximum(den, 1e-9), fallback)
    logits = shrunk / T
    logits = logits - logits.max(0, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(0, keepdims=True)


def main():
    print("=" * 82)
    print("  PHASE 0d — CLASS-AWARE (per-class) ENSEMBLE WEIGHTING")
    print("=" * 82)

    # class groups from TRAIN counts (immutable definition)
    tr = np.load('data/processed/lt_train_indices.npy')
    from torchvision import datasets
    full = datasets.CIFAR100(root='./data', train=True, download=False)
    tgt = np.asarray(full.targets)[tr]
    counts = np.array([(tgt == c).sum() for c in range(100)])
    gnames = ['Head', 'Med', 'Tail']
    G = {'Head': np.where(counts >= 100)[0],
         'Med': np.where((counts >= 20) & (counts < 100))[0],
         'Tail': np.where(counts < 20)[0]}
    group_of_class = np.zeros(100, dtype=int)
    for gi, g in enumerate(gnames):
        group_of_class[G[g]] = gi

    KS = [0, 1, 2, 5, 10, 20]
    TS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 1e6]
    NREP = 5

    for pool_name, specs in POOLS.items():
        print(f"\n{'='*82}\n  POOL: {pool_name}\n{'='*82}")
        Ls, Yn = get_pool_logits(pool_name, specs)
        E = len(Ls)
        n = len(Yn)
        LOGITS = np.stack(Ls, 1)                      # (n, E, 100)
        P = {i: LOGITS[:, i].argmax(1) for i in range(E)}

        ref, cls_ba, cls_tl, cls_hd, cls_md = [], [], [], [], []
        for rep in range(NREP):
            rng = np.random.default_rng(1000 + rep)
            perm = rng.permutation(n)
            a, b = int(.50 * n), int(.75 * n)
            fit, tune, ev = perm[:a], perm[a:b], perm[b:]

            uni = np.full((n, E), 1.0 / E)
            pu = np.einsum('ne,nec->nc', uni[ev], LOGITS[ev]).argmax(1)
            ref.append((balanced_accuracy(Yn[ev], pu),
                        group_accuracy(Yn[ev], pu, G['Tail']),
                        group_accuracy(Yn[ev], pu, G['Head']),
                        group_accuracy(Yn[ev], pu, G['Med'])))

            # --- per-class recall + group competence, fitted on FIT only ----
            n_c = np.array([(Yn[fit] == c).sum() for c in range(100)], dtype=float)
            recall = np.zeros((E, 100))
            for i in range(E):
                for c in range(100):
                    m = Yn[fit] == c
                    recall[i, c] = (P[i][fit][m] == c).mean() if m.sum() else 0.0
            comp_group = np.zeros((E, 3))
            for i in range(E):
                for gi, g in enumerate(gnames):
                    accs = [(P[i][fit][Yn[fit] == c] == c).mean()
                            for c in G[g] if (Yn[fit] == c).sum() > 0]
                    comp_group[i, gi] = float(np.mean(accs)) if accs else 0.0

            # --- select (k, T) on TUNE -------------------------------------
            best, best_cfg = -1, None
            for k in KS:
                for T in TS:
                    w = class_weights(recall, group_of_class, comp_group, n_c, k, T)
                    pu_t = np.einsum('ec,nec->nc', w, LOGITS[tune]).argmax(1)
                    ba = balanced_accuracy(Yn[tune], pu_t)
                    if ba > best:
                        best, best_cfg = ba, (k, T)
            k, T = best_cfg
            w = class_weights(recall, group_of_class, comp_group, n_c, k, T)
            p = np.einsum('ec,nec->nc', w, LOGITS[ev]).argmax(1)
            cls_ba.append(balanced_accuracy(Yn[ev], p))
            cls_tl.append(group_accuracy(Yn[ev], p, G['Tail']))
            cls_hd.append(group_accuracy(Yn[ev], p, G['Head']))
            cls_md.append(group_accuracy(Yn[ev], p, G['Med']))

        def m_(v):
            return float(np.mean(v)), float(np.std(v))
        r = np.array(ref)
        print(f"\n  {'method':<28}{'BA':>16}{'Head':>16}{'Med':>16}{'Tail':>16}")
        print("  " + "-" * 92)
        print(f"  {'uniform (reference)':<28}"
              f"{m_(r[:,0])[0]:>8.4f}+-{m_(r[:,0])[1]:.4f}"
              f"{m_(r[:,2])[0]:>8.4f}+-{m_(r[:,2])[1]:.4f}"
              f"{m_(r[:,3])[0]:>8.4f}+-{m_(r[:,3])[1]:.4f}"
              f"{m_(r[:,1])[0]:>8.4f}+-{m_(r[:,1])[1]:.4f}")
        cb, ct = np.array(cls_ba), np.array(cls_tl)
        print(f"  {'class-aware (per-class)':<28}"
              f"{m_(cb)[0]:>8.4f}+-{m_(cb)[1]:.4f}"
              f"{m_(cls_hd)[0]:>8.4f}+-{m_(cls_hd)[1]:.4f}"
              f"{m_(cls_md)[0]:>8.4f}+-{m_(cls_md)[1]:.4f}"
              f"{m_(ct)[0]:>8.4f}+-{m_(ct)[1]:.4f}")

        dba, dtl = cb - r[:, 0], ct - r[:, 1]
        print(f"\n  Paired gain over uniform:  dBA {dba.mean():+.4f}+-{dba.std():.4f}"
              f"   dTail {dtl.mean():+.4f}+-{dtl.std():.4f}")
        ok_ba = dba.mean() > 0 and dba.mean() > dba.std()
        ok_tl = dtl.mean() >= 0
        verdict = ('PASSES BOTH' if ok_ba and ok_tl else
                   'BA gain within noise' if dba.mean() > 0 else 'WORSE on BA')
        print(f"  verdict: {verdict}")
        print(f"  (selected hyperparameters on TUNE: k={k}, T={T})")

    print("\n" + "=" * 82)
    print("  NOTE: weights here are fitted on validation for measurement.  Production")
    print("  would fit them on routing_dev (carved from the training set) instead.")
    print("=" * 82)


if __name__ == '__main__':
    main()
