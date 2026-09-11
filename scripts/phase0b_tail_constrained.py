#!/usr/bin/env python3
"""
PHASE 0b — does a TAIL-CONSTRAINED variant satisfy both halves of the gate?

Phase 0 result: both routing candidates beat uniform on BA but LOSE on Tail.
The recorded acceptance criteria require BA *and* Tail above baseline
simultaneously (AGENTs.md section 6), so neither passes.

Diagnosis of why: routing shifts weight toward the Med specialist, whose Tail
accuracy is exactly 0.0000, while uniform preserves tail performance through
experts A and C.  The BA gain comes from Med; the Tail loss is collateral.

This script tests whether blending the routed weights back toward uniform can
satisfy BOTH constraints:

    w_final = (1 - alpha) * uniform + alpha * w_routed

alpha = 0 is exactly uniform (so the baseline is inside the search space).
alpha is selected on the TUNE portion subject to Tail >= uniform Tail; the
result is reported on the EVAL portion the selection never saw.

If no alpha satisfies both, the routing path cannot meet the recorded bar on
this expert pool and Phase 5 (expert strengthening) becomes necessary.

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


def main():
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
    LA, LB, LC = torch.cat(LA), torch.cat(LB), torch.cat(LC)
    FA, FB, FC = torch.cat(FA), torch.cat(FB), torch.cat(FC)
    EBv, ECv = torch.cat(EBv), torch.cat(ECv)
    Yn = torch.cat(Y).numpy()
    n = len(Yn)
    LOGITS = np.stack([LA.numpy(), LB.numpy(), LC.numpy()], 1)   # (n, 3, 100)

    def feats(logits, emb):
        p = F.softmax(logits, 1)
        t2 = p.topk(2, dim=1).values
        ent = -(p * torch.log(p + 1e-12)).sum(1)
        return torch.cat([emb, t2[:, 0:1], (t2[:, 0] - t2[:, 1])[:, None],
                          ent[:, None]], 1).numpy()

    X = {'A': feats(LA, FA), 'B': feats(LB, EBv), 'C': feats(LC, ECv)}
    P = {k: LOGITS[:, i].argmax(1) for i, k in enumerate('ABC')}
    correct = {k: (P[k] == Yn) for k in P}

    tr = np.load('data/processed/lt_train_indices.npy')
    from torchvision import datasets
    full = datasets.CIFAR100(root='./data', train=True, download=False)
    tgt = np.asarray(full.targets)[tr]
    counts = np.array([(tgt == c).sum() for c in range(100)])
    G = {'Head': np.where(counts >= 100)[0],
         'Med': np.where((counts >= 20) & (counts < 100))[0],
         'Tail': np.where(counts < 20)[0]}

    ens = (F.softmax(LA, 1) + F.softmax(LB, 1) + F.softmax(LC, 1)).div(3).numpy()
    Gp = np.stack([ens[:, G[g]].sum(1) for g in ('Head', 'Med', 'Tail')], 1)

    TEMPS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 1e6]
    ALPHAS = np.linspace(0.0, 1.0, 11)
    NREP = 5
    out = {}

    def combine(w, idx):
        """w: (n, 3) weights over ALL samples; idx: the subset to evaluate."""
        return np.einsum('ne,nec->nc', w[idx], LOGITS[idx])

    def record(tag, alpha, idx, y):
        w = out[tag]['w'][alpha]          # full-length weight matrix
        pred = combine(w, idx).argmax(1)
        return (balanced_accuracy(y, pred),
                group_accuracy(y, pred, G['Tail']),
                group_accuracy(y, pred, G['Head']),
                group_accuracy(y, pred, G['Med']))

    for rep in range(NREP):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(n)
        a, b = int(.50 * n), int(.75 * n)
        fit, tune, ev = perm[:a], perm[a:b], perm[b:]

        # --- routed weight sources, fitted on FIT only -------------------
        # (ii) group-conditional competence
        comp = np.zeros((3, 3))
        for i, k in enumerate('ABC'):
            for gi, g in enumerate(('Head', 'Med', 'Tail')):
                accs = [(P[k][fit][Yn[fit] == c] == c).mean()
                        for c in G[g] if (Yn[fit] == c).sum() > 0]
                comp[i, gi] = float(np.mean(accs)) if accs else 0.0
        wg = Gp @ comp.T
        wg /= np.clip(wg.sum(1, keepdims=True), 1e-9, None)

        # (i) calibrated correctness
        lg = np.stack([apply_logreg(*fit_logreg(X[k][fit],
                                                correct[k][fit].astype(float)),
                                    X[k]) for k in 'ABC'], 1)

        for tag in ('group', 'calib'):
            out.setdefault(tag, {'w': {}})
        Tsel = {}
        for T in TEMPS:
            L = lg / T; L = L - L.max(1, keepdims=True)
            w = np.exp(L); w /= w.sum(1, keepdims=True)
            ba = balanced_accuracy(Yn[tune], combine(w, tune).argmax(1))
            if T not in Tsel or ba > Tsel[T]:
                Tsel[T] = ba
        bestT = max(Tsel, key=Tsel.get)
        L = lg / bestT; L = L - L.max(1, keepdims=True)
        wc = np.exp(L); wc /= wc.sum(1, keepdims=True)

        for alpha in ALPHAS:
            out['group']['w'][alpha] = (1 - alpha) / 3.0 + alpha * wg
            out['calib']['w'][alpha] = (1 - alpha) / 3.0 + alpha * wc
        out['group'].setdefault('sel', []); out['calib'].setdefault('sel', [])
        # selection on TUNE: maximise BA subject to Tail >= uniform Tail
        uni_w = np.full((n, 3), 1 / 3.0)
        up = combine(uni_w, tune).argmax(1)
        uni_tail_tune = group_accuracy(Yn[tune], up, G['Tail'])
        for tag in ('group', 'calib'):
            feas, best = [], None
            for alpha in ALPHAS:
                w = out[tag]['w'][alpha]        # full-length, combine() subsets it
                pred = combine(w, tune).argmax(1)
                ba = balanced_accuracy(Yn[tune], pred)
                tl = group_accuracy(Yn[tune], pred, G['Tail'])
                if tl >= uni_tail_tune:
                    feas.append((ba, alpha))
            best = max(feas)[1] if feas else None
            out[tag]['sel'].append(best)

    # ── report ─────────────────────────────────────────────────────────
    print("=" * 84)
    print("  PHASE 0b — TAIL-CONSTRAINED BLEND  (alpha=0 is exactly uniform)")
    print("=" * 84)
    print(f"\n  {NREP} reps of 50/25/25.  alpha chosen on TUNE with constraint")
    print("  Tail >= uniform Tail, then measured on unseen EVAL.\n")
    print(f"  {'method / alpha':<26}{'BA':>13}{'Head':>13}{'Med':>13}{'Tail':>13}")

    # uniform reference
    uba, uh, um, ut = [], [], [], []
    for rep in range(NREP):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(n)
        ev = perm[int(.75 * n):]
        pred = combine(np.full((n, 3), 1 / 3.0), ev).argmax(1)
        uba.append(balanced_accuracy(Yn[ev], pred))
        uh.append(group_accuracy(Yn[ev], pred, G['Head']))
        um.append(group_accuracy(Yn[ev], pred, G['Med']))
        ut.append(group_accuracy(Yn[ev], pred, G['Tail']))
    print(f"  {'uniform (reference)':<26}{np.mean(uba):>13.4f}{np.mean(uh):>13.4f}"
          f"{np.mean(um):>13.4f}{np.mean(ut):>13.4f}")

    for tag, label in (('group', 'group-conditional'), ('calib', 'calibrated corr.')):
        print(f"\n  -- {label} --")
        for alpha in ALPHAS:
            ba, tl, hd, md = [], [], [], []
            for rep in range(NREP):
                rng = np.random.default_rng(1000 + rep)
                perm = rng.permutation(n)
                ev = perm[int(.75 * n):]
                r = record(tag, alpha, ev, Yn[ev])
                ba.append(r[0]); tl.append(r[1]); hd.append(r[2]); md.append(r[3])
            mark = ''
            if np.mean(ba) > np.mean(uba) and np.mean(tl) >= np.mean(ut):
                mark = '  <== PASSES BOTH'
            print(f"  alpha={alpha:<4.1f}{'':<14}{np.mean(ba):>13.4f}"
                  f"{np.mean(hd):>13.4f}{np.mean(md):>13.4f}{np.mean(tl):>13.4f}{mark}")
        sels = [s for s in out[tag]['sel'] if s is not None]
        print(f"  alpha selected on TUNE under the Tail constraint: "
              f"{sels if sels else 'NONE feasible'}")

    print("\n" + "=" * 84)
    print("  VERDICT")
    print("=" * 84)
    print(f"  uniform reference: BA {np.mean(uba):.4f}, Tail {np.mean(ut):.4f}")
    any_pass = False
    for tag in ('group', 'calib'):
        for alpha in ALPHAS:
            ba, tl = [], []
            for rep in range(NREP):
                rng = np.random.default_rng(1000 + rep)
                perm = rng.permutation(n)
                ev = perm[int(.75 * n):]
                r = record(tag, alpha, ev, Yn[ev])
                ba.append(r[0]); tl.append(r[1])
            if np.mean(ba) > np.mean(uba) and np.mean(tl) >= np.mean(ut):
                any_pass = True
                print(f"  PASS: {tag}, alpha={alpha:.1f} -> BA {np.mean(ba):.4f} "
                      f"(+{np.mean(ba)-np.mean(uba):.4f}), Tail {np.mean(tl):.4f} "
                      f"({np.mean(tl)-np.mean(ut):+.4f})")
    if not any_pass:
        print("  NO configuration satisfies BA > uniform AND Tail >= uniform.")
        print("  -> On this expert pool the routing path cannot meet the recorded")
        print("     bar. Phase 5 (strengthen experts, esp. give B tail ability)")
        print("     is required before routing can pass.")
    print("=" * 84)


if __name__ == '__main__':
    main()
