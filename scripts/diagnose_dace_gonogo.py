#!/usr/bin/env python3
"""
DACE diagnostic #3 — THE GO/NO-GO TEST.

The routing signal currently in use (disagreement affinity) is ANTI-predictive of
correctness (AUROC 0.327).  Before designing any fix we must know whether ANY
signal available to a router can predict per-expert correctness.

If per-expert correctness is unpredictable (AUROC ~0.50), no routing architecture
can beat uniform and the fix must target the experts instead.  If it IS
predictable, we know the achievable routing BA and can design against it.

Protocol (strictly non-cheating):
  - Validation split only.  The 10K test set is never touched.
  - Validation is split in half: FIT half to train the correctness probe,
    EVAL half to measure.  No sample is used for both.
  - Nothing here is used for training any model; this is measurement only.

Measures, per expert:
  1. AUROC(max-softmax confidence  -> expert correct)   [classic trust meter]
  2. AUROC(entropy                 -> expert correct)
  3. AUROC(embedding probe         -> expert correct)   [what a router could learn]
Then the decision-relevant quantity:
  4. BA when routing by argmax predicted-correctness (fit on FIT half,
     applied to EVAL half), versus uniform / best-single-expert / oracle.
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
from scripts.diagnose_dace import load_expert, auroc


def ridge_fit_predict(Xtr, ytr, Xte, ridge=1.0):
    """Ridge regression on a binary target -> real-valued score for ranking."""
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    w = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    return Xte @ w


def balanced_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in classes]))


def main():
    dev = 'cpu'
    ck = './checkpoints'

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
    Y = torch.cat(Y)
    n = len(Y)

    print("=" * 74)
    print("  GO/NO-GO: IS PER-EXPERT CORRECTNESS PREDICTABLE?")
    print("  (validation only; fit-half / eval-half split; test set untouched)")
    print("=" * 74)

    experts = {
        'A': {'logits': LA, 'emb': FA},          # no routing head -> backbone feats
        'B': {'logits': LB, 'emb': EBv},
        'C': {'logits': LC, 'emb': ECv},
    }

    # deterministic half split
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    fit_idx, eval_idx = perm[:n // 2], perm[n // 2:]
    print(f"\n  fit half: {len(fit_idx)}   eval half: {len(eval_idx)}")

    correct = {}
    for nm, d in experts.items():
        correct[nm] = (d['logits'].argmax(1) == Y).numpy()

    # ── 1&2: confidence / entropy trust meters (no fitting needed) ──────
    print("\n  [1-2] Confidence / entropy trust meters (single feature, no fit)")
    print(f"  {'expert':<8}{'conf->correct':>16}{'entropy->correct':>18}")
    for nm, d in experts.items():
        p = F.softmax(d['logits'], 1)
        conf = p.max(1).values.numpy()
        ent = -(p * torch.log(p + 1e-12)).sum(1).numpy()
        a_conf = auroc(conf, correct[nm])
        a_ent = auroc(-ent, correct[nm])   # lower entropy -> more correct
        print(f"  {nm:<8}{a_conf:>16.4f}{a_ent:>18.4f}")

    # ── 3: embedding probe (fit on FIT half, score EVAL half) ───────────
    print("\n  [3] Embedding probe -> correctness (fit on FIT half, eval on EVAL half)")
    print(f"  {'expert':<8}{'feat dim':>10}{'AUROC':>10}   {'caveat':<30}")
    scores = {}
    for nm, d in experts.items():
        X = d['emb'].numpy()
        y = correct[nm].astype(float)
        s = ridge_fit_predict(X[fit_idx], y[fit_idx], X[eval_idx])
        a = auroc(s, correct[nm][eval_idx])
        caveat = 'backbone feats (no rout. head)' if nm == 'A' else 'routing head emb'
        print(f"  {nm:<8}{X.shape[1]:>10}{a:>10.4f}   {caveat:<30}")
        scores[nm] = s

    # ── 4: routing by predicted correctness ────────────────────────────
    print("\n  [4] DECISION-RELEVANT: route by argmax predicted-correctness")
    S = np.stack([scores['A'], scores['B'], scores['C']], 1)
    pick = S.argmax(1)
    pred_routed = np.stack([LA.argmax(1).numpy()[eval_idx],
                            LB.argmax(1).numpy()[eval_idx],
                            LC.argmax(1).numpy()[eval_idx]], 1)
    y_eval = Y.numpy()[eval_idx]
    routed_preds = pred_routed[np.arange(len(eval_idx)), pick]

    ba_routed = balanced_accuracy(y_eval, routed_preds)
    uni_preds = (LA + LB + LC).argmax(1).numpy()[eval_idx]
    ba_uniform = balanced_accuracy(y_eval, uni_preds)

    per_exp = {}
    for i, nm in enumerate(['A', 'B', 'C']):
        per_exp[nm] = balanced_accuracy(y_eval, pred_routed[:, i])

    cor = np.stack([correct['A'][eval_idx], correct['B'][eval_idx],
                    correct['C'][eval_idx]], 1)
    any_ok = cor.any(1)
    oracle_preds = np.where(any_ok, y_eval, pred_routed[:, 0])
    ba_oracle = balanced_accuracy(y_eval, oracle_preds)

    print(f"\n    BA uniform averaging        : {ba_uniform:.4f}")
    print(f"    BA best single expert       : {max(per_exp.values()):.4f} "
          f"({max(per_exp, key=per_exp.get)})")
    print(f"    BA routed (pred-correctness): {ba_routed:.4f}")
    print(f"    BA oracle (>=1 correct)     : {ba_oracle:.4f}")
    print(f"    -> routed gain over uniform : {ba_routed - ba_uniform:+.4f}")
    print(f"    -> oracle gain over uniform : {ba_oracle - ba_uniform:+.4f}")
    frac = (ba_routed - ba_uniform) / max(ba_oracle - ba_uniform, 1e-9)
    print(f"    -> fraction of oracle headroom captured: {100*frac:.1f}%")

    # ── verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    best_probe = max(auroc(ridge_fit_predict(
        experts[nm]['emb'].numpy()[fit_idx], correct[nm].astype(float)[fit_idx],
        experts[nm]['emb'].numpy()[eval_idx]), correct[nm][eval_idx])
        for nm in ['A', 'B', 'C'])
    print(f"  Best correctness-probe AUROC : {best_probe:.4f}")
    if best_probe < 0.55:
        print("  VERDICT: correctness is NOT predictable from these features.")
        print("           Routing cannot beat uniform; fix the EXPERTS instead.")
    else:
        print("  VERDICT: correctness IS predictable to a useful degree.")
        print("           Re-aiming the routing target is viable.")
    print("=" * 74)


if __name__ == '__main__':
    main()
