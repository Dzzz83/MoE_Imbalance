#!/usr/bin/env python3
"""
PHASE 0 — GATE G0: can ANY decision rule beat uniform on this expert pool?

This is the cheapest decisive experiment in the plan.  It runs on the EXISTING
checkpoints, needs no training, and decides whether the routing path
(Phases 1-4) or the expert path (Phase 5) is the right investment.

Methods compared
----------------
  (iii) uniform                      reference; the thing to beat
        best single expert
  (i)   calibrated correctness       per-expert logistic P(correct), combined by
                                     w = softmax(logit / T), T tuned on held-out
                                     data.  T -> inf is exactly uniform.
  (ii)  group-conditional            exploit the existing partitioning: weight each
                                     expert by its measured competence on the class
                                     group the ensemble predicts.  9 numbers total,
                                     so almost no variance.
        group ORACLE                 true class group (ceiling of idea ii)
        expert ORACLE                true correctness (absolute ceiling)

Honesty / protocol
------------------
  * Validation split ONLY.  The 10K test set is never touched.
  * Every fitted quantity (probes, competence table, temperature T) is fitted on a
    separate portion of the split and evaluated on a portion it never saw.
  * 5 independent random 50/25/25 (fit/tune/eval) repetitions; mean +/- std reported,
    so nothing rests on one lucky partition.
  * Forward passes are computed once and cached; only the small fits are repeated.

CAVEAT recorded honestly: lt_val is long-tailed and several tail classes have ZERO
validation samples, so Tail accuracy measured here is noisy and is indicative only.
The final Tail claim must come from the balanced 10K test set.

Usage:
    python scripts/phase0_ceiling_probe.py
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


# ── small, dependency-free logistic regression (IRLS) ───────────────────────

def fit_logreg(X, y, l2=1.0, iters=60):
    """L2-regularised logistic regression via IRLS. Returns (w, mu, sd)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        z = Xs @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        XtWX = Xs.T @ (Xs * W[:, None]) + l2 * np.eye(Xs.shape[1])
        XtWz = Xs.T @ (W * (z + (y - p) / W))
        try:
            w = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break
    return w, mu, sd


def apply_logreg(w, mu, sd, X):
    X = np.asarray(X, dtype=np.float64)
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Xs @ w            # logit


def balanced_accuracy(y, pred, classes=None):
    classes = np.unique(y) if classes is None else classes
    accs = [(pred[y == c] == c).mean() for c in classes if (y == c).sum() > 0]
    return float(np.mean(accs)) if accs else float('nan')


def group_accuracy(y, pred, class_list):
    m = np.isin(y, class_list)
    return float((pred[m] == y[m]).mean()) if m.sum() else float('nan')


# ── main ────────────────────────────────────────────────────────────────────

def main():
    dev = 'cpu'
    ck = './checkpoints'

    print("=" * 78)
    print("  PHASE 0 / GATE G0 — can any decision rule beat uniform here?")
    print("=" * 78)

    ea = load_expert(os.path.join(ck, 'DACE_A_best.pt'), False, dev)
    eb = load_expert(os.path.join(ck, 'DACE_B_best.pt'), True, dev)
    ec = load_expert(os.path.join(ck, 'DACE_C_best.pt'), True, dev)

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
            FA.append(ea.backbone(images))
            FB.append(eb.backbone(images))
            FC.append(ec.backbone(images))
            Y.append(targets)

    LA, LB, LC = torch.cat(LA), torch.cat(LB), torch.cat(LC)
    FA, FB, FC = torch.cat(FA), torch.cat(FB), torch.cat(FC)
    EBv, ECv = torch.cat(EBv), torch.cat(ECv)
    Yt = torch.cat(Y)
    Yn = Yt.numpy()
    n = len(Yn)

    def feats(logits, emb):
        p = F.softmax(logits, 1)
        top2 = p.topk(2, dim=1).values
        conf = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        ent = -(p * torch.log(p + 1e-12)).sum(1)
        return torch.cat([emb, conf[:, None], margin[:, None], ent[:, None]], 1).numpy()

    X = {'A': feats(LA, FA), 'B': feats(LB, EBv), 'C': feats(LC, ECv)}
    P = {'A': LA.argmax(1).numpy(), 'B': LB.argmax(1).numpy(), 'C': LC.argmax(1).numpy()}
    correct = {k: (P[k] == Yn) for k in P}

    # class groups from TRAIN counts (immutable definition per AGENTS.md)
    tr_idx = np.load('data/processed/lt_train_indices.npy')
    from torchvision import datasets
    full = datasets.CIFAR100(root='./data', train=True, download=False)
    tr_targets = np.asarray(full.targets)[tr_idx]
    counts = np.array([(tr_targets == c).sum() for c in range(100)])
    G = {'Head': np.where(counts >= 100)[0],
         'Med': np.where((counts >= 20) & (counts < 100))[0],
         'Tail': np.where(counts < 20)[0]}

    ens_probs = ((F.softmax(LA, 1) + F.softmax(LB, 1) + F.softmax(LC, 1)) / 3).numpy()
    group_probs = {g: ens_probs[:, cls].sum(1) for g, cls in G.items()}

    # Uniform reference = LOGIT averaging, matching scripts/evaluate_dace.py and
    # the project's recorded "uniform avg" convention.  (Averaging softmax
    # probabilities instead is a different operator and gives a different
    # baseline, which would make the gain-vs-uniform comparison invalid.)
    uni_logits_np = ((LA + LB + LC) / 3.0).numpy()

    TEMPS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 1e6]
    NREP = 5
    res = {}

    def rec(k, ba, tail, head, med):
        res.setdefault(k, {'ba': [], 'tail': [], 'head': [], 'med': []})
        res[k]['ba'].append(ba); res[k]['tail'].append(tail)
        res[k]['head'].append(head); res[k]['med'].append(med)

    for rep in range(NREP):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(n)
        a, b = int(0.50 * n), int(0.75 * n)
        fit, tune, ev = perm[:a], perm[a:b], perm[b:]

        # ---- (iii) uniform + best single expert -------------------------
        uni_pred = uni_logits_np[ev].argmax(1)
        rec('uniform', balanced_accuracy(Yn[ev], uni_pred),
            group_accuracy(Yn[ev], uni_pred, G['Tail']),
            group_accuracy(Yn[ev], uni_pred, G['Head']),
            group_accuracy(Yn[ev], uni_pred, G['Med']))
        for k in 'ABC':
            rec(f'expert {k}', balanced_accuracy(Yn[ev], P[k][ev]),
                group_accuracy(Yn[ev], P[k][ev], G['Tail']),
                group_accuracy(Yn[ev], P[k][ev], G['Head']),
                group_accuracy(Yn[ev], P[k][ev], G['Med']))

        # ---- (i) calibrated correctness, temperature-combined -----------
        W, logit = {}, {}
        for k in 'ABC':
            w_, mu_, sd_ = fit_logreg(X[k][fit], correct[k][fit].astype(float))
            W[k] = (w_, mu_, sd_)
            logit[k] = {name: apply_logreg(w_, mu_, sd_, X[k][idx])
                        for name, idx in (('tune', tune), ('ev', ev))}

        best_T, best_ba = None, -1
        for T in TEMPS:
            L = np.stack([logit[k]['tune'] for k in 'ABC'], 1) / T
            L = L - L.max(1, keepdims=True)
            wgt = np.exp(L); wgt /= wgt.sum(1, keepdims=True)
            comb = sum(wgt[:, i][:, None] * [LA, LB, LC][i].numpy()[tune]
                       for i in range(3))
            ba = balanced_accuracy(Yn[tune], comb.argmax(1))
            if ba > best_ba:
                best_ba, best_T = ba, T
        L = np.stack([logit[k]['ev'] for k in 'ABC'], 1) / best_T
        L = L - L.max(1, keepdims=True)
        wgt = np.exp(L); wgt /= wgt.sum(1, keepdims=True)
        comb = sum(wgt[:, i][:, None] * [LA, LB, LC][i].numpy()[ev] for i in range(3))
        pred_cal = comb.argmax(1)
        rec('calibrated correctness', balanced_accuracy(Yn[ev], pred_cal),
            group_accuracy(Yn[ev], pred_cal, G['Tail']),
            group_accuracy(Yn[ev], pred_cal, G['Head']),
            group_accuracy(Yn[ev], pred_cal, G['Med']))

        # ---- (ii) group-conditional competence table --------------------
        comp = {}
        for k in 'ABC':
            comp[k] = []
            for g, cls in G.items():
                accs = [(P[k][fit][Yn[fit] == c] == c).mean()
                        for c in cls if (Yn[fit] == c).sum() > 0]
                comp[k].append(float(np.mean(accs)) if accs else 0.0)
        comp = np.array([comp[k] for k in 'ABC'])          # (3 experts, 3 groups)
        Gp = np.stack([group_probs[g][ev] for g in ('Head', 'Med', 'Tail')], 1)
        wgt = Gp @ comp.T                                   # (n_eval, 3 experts)
        s = wgt.sum(1, keepdims=True); s[s == 0] = 1.0
        wgt = wgt / s
        comb = sum(wgt[:, i][:, None] * [LA, LB, LC][i].numpy()[ev] for i in range(3))
        pred_grp = comb.argmax(1)
        rec('group-conditional', balanced_accuracy(Yn[ev], pred_grp),
            group_accuracy(Yn[ev], pred_grp, G['Tail']),
            group_accuracy(Yn[ev], pred_grp, G['Head']),
            group_accuracy(Yn[ev], pred_grp, G['Med']))

        # ---- oracles ----------------------------------------------------
        true_g = np.full(len(ev), 2)
        for gi, g in enumerate(('Head', 'Med', 'Tail')):
            true_g[np.isin(Yn[ev], G[g])] = gi
        Gtrue = np.zeros((len(ev), 3)); Gtrue[np.arange(len(ev)), true_g] = 1.0
        wgt = Gtrue @ comp.T
        s = wgt.sum(1, keepdims=True); s[s == 0] = 1.0
        wgt = wgt / s
        comb = sum(wgt[:, i][:, None] * [LA, LB, LC][i].numpy()[ev] for i in range(3))
        pred_go = comb.argmax(1)
        rec('group ORACLE (true group)', balanced_accuracy(Yn[ev], pred_go),
            group_accuracy(Yn[ev], pred_go, G['Tail']),
            group_accuracy(Yn[ev], pred_go, G['Head']),
            group_accuracy(Yn[ev], pred_go, G['Med']))

        cor = np.stack([correct[k][ev] for k in 'ABC'], 1)
        any_ok = cor.any(1)
        pred_or = np.where(any_ok, Yn[ev], P['A'][ev])
        rec('expert ORACLE', balanced_accuracy(Yn[ev], pred_or),
            group_accuracy(Yn[ev], pred_or, G['Tail']),
            group_accuracy(Yn[ev], pred_or, G['Head']),
            group_accuracy(Yn[ev], pred_or, G['Med']))

    # ── report ─────────────────────────────────────────────────────────
    print(f"\n  {NREP} repetitions of 50/25/25 (fit/tune/eval) on validation "
          f"(n={n}); mean +/- std over reps\n")
    print(f"  {'method':<30}{'BA':>15}{'Head':>15}{'Med':>15}{'Tail':>15}")
    print("  " + "-" * 90)
    order = ['uniform', 'expert A', 'expert B', 'expert C',
             'calibrated correctness', 'group-conditional',
             'group ORACLE (true group)', 'expert ORACLE']
    for k in order:
        d = res[k]
        def ms(v):
            return f"{np.mean(v):.4f}+-{np.std(v):.4f}"
        print(f"  {k:<30}{ms(d['ba']):>15}{ms(d['head']):>15}"
              f"{ms(d['med']):>15}{ms(d['tail']):>15}")

    uni_ba = np.array(res['uniform']['ba'])
    uni_tail = np.array(res['uniform']['tail'])
    print("\n  Paired gain over uniform (same splits each rep), mean +/- std :")
    print(f"  {'method':<30}{'dBA':>18}{'dTail':>18}   verdict")
    gate = {}
    for k in order:
        if k == 'uniform':
            continue
        dba = np.array(res[k]['ba']) - uni_ba
        dtl = np.array(res[k]['tail']) - uni_tail
        sig_ba = dba.mean() > dba.std() and dba.mean() > 0
        ok_tail = dtl.mean() >= 0
        if sig_ba and ok_tail:
            v = 'PASSES BOTH'
        elif dba.mean() > 0 and not ok_tail:
            v = 'BA only, FAILS Tail'
        elif dba.mean() > 0:
            v = 'BA gain within noise'
        else:
            v = 'worse'
        gate[k] = v
        print(f"  {k:<30}{dba.mean():+.4f}+-{dba.std():.4f}"
              f"{dtl.mean():+.4f}+-{dtl.std():.4f}   {v}")

    print("\n  CAVEAT: lt_val is long-tailed and several tail classes have 0 samples,")
    print("  so the Tail column here is noisy — indicative only. The final Tail claim")
    print("  must be measured on the balanced 10K test set.")

    print("\n" + "=" * 78)
    print("  GATE G0 VERDICT")
    print("=" * 78)
    print(f"  uniform reference: BA {uni_ba.mean():.4f}, Tail {uni_tail.mean():.4f}")
    print("  Gate requires BA gain AND Tail not worse (AGENTs.md section 6).\n")
    want = ['calibrated correctness', 'group-conditional']
    winners = [k for k in want if gate.get(k, '').startswith('PASSES BOTH')]
    ba_only = [k for k in want if 'BA only' in gate.get(k, '')
               or 'within noise' in gate.get(k, '')]
    for k in want:
        print(f"    {k:<28} {gate.get(k)}")
    if winners:
        best = max(winners, key=lambda k: np.mean(res[k]['ba']))
        print(f"\n  -> G0 PASSES. Proceed on the ROUTING path; primary candidate: {best}")
        print("     Confirm on the balanced test set before committing to 3 seeds.")
    elif ba_only:
        print("\n  -> G0 PARTIAL: BA gains exist but the Tail requirement fails or is")
        print("     within noise. Routing alone cannot meet the recorded bar on this")
        print("     expert pool (Expert B has Tail accuracy 0.0000, so shifting weight")
        print("     toward it destroys what uniform preserves via A and C).")
        print("     -> Run Phase 0b (tail-constrained blend); if that also fails,")
        print("        Phase 5 (give the specialists some non-target competence)")
        print("        is required BEFORE routing can pass.")
    else:
        print("\n  -> G0 FAILS: no candidate beats uniform. Routing is not the lever")
        print("     on this pool. Pivot to Phase 5 (strengthen the experts) and report")
        print("     routing as a negative result.")
    print("=" * 78)


if __name__ == '__main__':
    main()
