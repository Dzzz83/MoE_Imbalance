#!/usr/bin/env python3
"""
Comprehensive diversity and health analysis for the three trained experts.

Evaluates:
  1. Individual expert performance (BA, Head/Medium/Tail acc)
  2. Pairwise diversity (Cohen's κ, class-accuracy correlation, unique contribution)
  3. Calibration (confidence when correct / when wrong, ECE)
  4. Oracle accuracy (at least one expert correct)
  5. Training history convergence check
  6. Routing potential assessment
"""

import os, sys, json, math
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from models.resnet32 import ResNet32, PaCoResNet32
from losses.lal_loss import LALLoss
from losses.paco_loss import PaCoLoss
from scripts.base_trainer import balanced_accuracy, group_accuracies, compute_class_groups
from data.cifar_lt import LongTailCIFAR100, CIFAR100_MEAN, CIFAR100_STD
from torchvision import transforms

# ── helpers ──────────────────────────────────────────────────────────────

def softmax_logits(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits (stably)."""
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict]]:
    """Compute ECE — lower is better (well-calibrated → 0)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    bin_info = []
    for i, (lo, hi) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        gap = abs(avg_acc - avg_conf)
        ece += gap * prop_in_bin
        bin_info.append({
            'bin': i, 'lo': lo, 'hi': hi,
            'count': in_bin.sum(), 'acc': avg_acc, 'conf': avg_conf, 'gap': gap,
        })
    return ece, bin_info


# ── expert loaders ───────────────────────────────────────────────────────

def load_expert(ckpt_path: str, device: str):
    """Load a checkpoint and return (model, expert_name, epoch, val_ba)."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    name = state['expert_name']
    epoch = state['epoch']
    val_ba = state.get('best_metric_val', state.get('log', {}).get('val_ba', 0.0))

    if name == 'PaCo':
        model = PaCoResNet32(num_classes=100, dim=32, K=2048, m=0.999, mlp=True)
    else:
        model = ResNet32(num_classes=100)

    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    return model, name, epoch, val_ba


@torch.no_grad()
def predict(model, loader, device, expert_name: str):
    """Run inference; return dict of arrays."""
    all_logits, all_probs, all_preds, all_targets = [], [], [], []
    for images, targets in loader:
        if expert_name == 'PaCo':
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
        else:
            images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        probs = softmax_logits(logits.cpu().numpy())
        preds = logits.argmax(dim=1).cpu().numpy()
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(targets.cpu().numpy())
    return {
        'logits': np.concatenate(all_logits),
        'probs':  np.concatenate(all_probs),
        'preds':  np.concatenate(all_preds),
        'targets': np.concatenate(all_targets),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = './data'
    ckpt_dir = './checkpoints'

    print("=" * 72)
    print("EXPERT DIVERSITY & HEALTH ANALYSIS")
    print(f"Device: {device}")
    print("=" * 72)

    # ── 1. Load validation data ──────────────────────────────────────────
    print("\n[1] Loading validation set...")
    val_idx = np.load(f'{data_root}/processed/balanced_val_indices.npy')
    val_set = LongTailCIFAR100(
        root=data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        skip_longtail=True,
    )
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Also load a small train subset for training-set BA sanity check
    base_idx = np.load(f'{data_root}/processed/base_train_indices.npy')
    train_subset = LongTailCIFAR100(
        root=data_root,
        base_train_indices=base_idx,
        imbalance_ratio=100.0,
        train=False, download=False,   # no aug – cleaner metric
    )
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

    class_counts = train_subset.get_class_counts()
    groups = compute_class_groups(class_counts)
    n_classes = 100

    # ── 2. Load all three experts ────────────────────────────────────────
    print("\n[2] Loading experts...")
    experts = {}
    for name in ['LAL', 'PaCo', 'Mixup']:
        ckpt = f'{ckpt_dir}/{name}_best.pt'
        if not os.path.exists(ckpt):
            # fallback to latest
            ckpt = f'{ckpt_dir}/{name}_latest.pt'
        model, ename, epoch, ba = load_expert(ckpt, device)
        experts[name] = {'model': model, 'epoch': epoch, 'ckpt_ba': ba}
        print(f"  {name}: loaded from {ckpt}  (epoch={epoch}, best_val_BA={ba:.2%})")

    # ── 3. Run inference ─────────────────────────────────────────────────
    print("\n[3] Running inference on validation set (5K samples)...")
    results = {}
    for name in ['LAL', 'PaCo', 'Mixup']:
        results[name] = predict(experts[name]['model'], val_loader, device, name)
        n = len(results[name]['targets'])
        print(f"  {name}: {n} samples processed")

    # ── 4. Individual performance ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION A: INDIVIDUAL EXPERT PERFORMANCE")
    print("=" * 72)

    perf_rows = []
    for name in ['LAL', 'PaCo', 'Mixup']:
        r = results[name]
        ba, per_class = balanced_accuracy(r['targets'], r['preds'])
        grp = group_accuracies(r['targets'], r['preds'], groups)
        correct = (r['preds'] == r['targets'])
        avg_conf = r['probs'].max(axis=1).mean()
        conf_when_right = r['probs'].max(axis=1)[correct].mean() if correct.sum() > 0 else 0.0
        conf_when_wrong = r['probs'].max(axis=1)[~correct].mean() if (~correct).sum() > 0 else 0.0
        ece, _ = expected_calibration_error(r['probs'].max(axis=1), correct)
        perf_rows.append({
            'expert': name,
            'ba': ba, 'head': grp['head'], 'medium': grp['medium'], 'tail': grp['tail'],
            'avg_conf': avg_conf, 'conf_right': conf_when_right,
            'conf_wrong': conf_when_wrong, 'ece': ece,
        })

    # Print table
    header = f"{'Expert':>8} | {'BA':>8} | {'Head':>8} | {'Med':>8} | {'Tail':>8} | {'AvgConf':>8} | {'Conf✓':>8} | {'Conf✗':>8} | {'ECE':>8}"
    sep = "-" * len(header)
    print(f"\n  {header}")
    print(f"  {sep}")
    for row in perf_rows:
        print(f"  {row['expert']:>8} | {row['ba']:>7.2%} | {row['head']:>7.2%} | {row['medium']:>7.2%} | {row['tail']:>7.2%} | "
              f"{row['avg_conf']:>7.2%} | {row['conf_right']:>7.2%} | {row['conf_wrong']:>7.2%} | {row['ece']:>7.4f}")

    # ── 5. Check training history for convergence ────────────────────────
    print("\n" + "=" * 72)
    print("SECTION B: TRAINING HISTORY — CONVERGENCE CHECK")
    print("=" * 72)
    for name in ['LAL', 'PaCo', 'Mixup']:
        hist_path = f'{ckpt_dir}/{name}_history.json'
        if not os.path.exists(hist_path):
            print(f"  {name}: no history file found")
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        val_bas = [h['val_ba'] for h in hist]
        last_10 = np.mean(val_bas[-10:]) if len(val_bas) >= 10 else np.mean(val_bas)
        best_epoch = int(np.argmax(val_bas)) + 1
        best_ba = max(val_bas)
        final_ba = val_bas[-1]
        tail_loss = [h.get('val_loss', 0) for h in hist]
        print(f"  {name}:")
        print(f"    Epochs trained    : {len(hist)}")
        print(f"    Best Val BA       : {best_ba:.2%}  (epoch {best_epoch})")
        print(f"    Final Val BA      : {final_ba:.2%}")
        print(f"    Last-10 avg BA    : {last_10:.2%}")
        # Check for overfitting: val loss going up while train loss goes down
        train_losses = [h.get('train_loss', 0) for h in hist]
        val_losses = [h.get('val_loss', 0) for h in hist]
        if len(val_losses) > 20:
            early_val = np.mean(val_losses[:10])
            late_val = np.mean(val_losses[-10:])
            early_train = np.mean(train_losses[:10])
            late_train = np.mean(train_losses[-10:])
            if late_val > early_val * 1.05 and late_train < early_train * 0.95:
                print(f"    ⚠️  Possible overfitting: val loss ↑ ({early_val:.3f}→{late_val:.3f}) while train loss ↓ ({early_train:.3f}→{late_train:.3f})")
            else:
                print(f"    ✅ No overfitting sign (val loss: {early_val:.3f}→{late_val:.3f}, train: {early_train:.3f}→{late_train:.3f})")

    # ── 6. Diversity analysis ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION C: DIVERSITY ANALYSIS")
    print("=" * 72)

    names = ['LAL', 'PaCo', 'Mixup']
    # 6a. Pairwise Cohen's κ
    print("\n  --- Pairwise Cohen's κ (prediction agreement) ---")
    kappa_matrix = {}
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            k = cohen_kappa_score(results[n1]['preds'], results[n2]['preds'])
            kappa_matrix[f'{n1}↔{n2}'] = k
            verdict = '✅ Diverse' if k < 0.80 else ('⚠️ Borderline' if k < 0.90 else '❌ Too similar')
            print(f"  {n1:8s} ↔ {n2:8s}: κ = {k:.4f}  → {verdict}")

    # 6b. Per-class accuracy correlation
    print("\n  --- Per-class accuracy correlation ---")
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            _, ac1 = balanced_accuracy(results[n1]['targets'], results[n1]['preds'])
            _, ac2 = balanced_accuracy(results[n2]['targets'], results[n2]['preds'])
            classes = sorted(set(results[n1]['targets'].tolist()))
            accs1 = [ac1[c] for c in classes]
            accs2 = [ac2[c] for c in classes]
            r = np.corrcoef(accs1, accs2)[0, 1]
            pair = f'{n1}↔{n2}'
            print(f"  {n1:8s} ↔ {n2:8s}: per-class acc r = {r:.4f}")

    # 6c. Unique contribution (samples only this expert gets right)
    print("\n  --- Unique contribution (samples only ONE expert gets correct) ---")
    correct_masks = {}
    for name in names:
        correct_masks[name] = (results[name]['preds'] == results[name]['targets'])
    all_correct = np.stack([correct_masks[n] for n in names], axis=1)  # (N, 3)
    for i, name in enumerate(names):
        only_this = all_correct[:, i] & (~np.any(all_correct[:, [j for j in range(3) if j != i]], axis=1))
        pct = only_this.mean() * 100
        # head/medium/tail breakdown
        targets = results[name]['targets']
        head_mask = np.isin(targets, groups['head'])
        med_mask = np.isin(targets, groups['medium'])
        tail_mask = np.isin(targets, groups['tail'])
        h_pct = only_this[head_mask].sum() / max(head_mask.sum(), 1) * 100
        m_pct = only_this[med_mask].sum() / max(med_mask.sum(), 1) * 100
        t_pct = only_this[tail_mask].sum() / max(tail_mask.sum(), 1) * 100
        print(f"  {name:8s}: {pct:.2f}% unique correct (H={h_pct:.1f}%  M={m_pct:.1f}%  T={t_pct:.1f}%)")

    # ── 7. Oracle accuracy ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION D: ORACLE ACCURACY")
    print("=" * 72)
    oracle_correct = np.any(all_correct, axis=1)
    oracle_acc = oracle_correct.mean()
    print(f"\n  Oracle (at least one expert correct): {oracle_acc:.2%}")
    # Best single expert
    best_single = max(perf_rows, key=lambda r: r['ba'])
    print(f"  Best single expert              : {best_single['expert']} @ {best_single['ba']:.2%}")
    oracle_gap = oracle_acc - best_single['ba']
    print(f"  Oracle gap to best single       : {oracle_gap:.2%}")
    # Oracle group breakdown
    head_oracle = oracle_correct[np.isin(results['LAL']['targets'], groups['head'])].mean()
    med_oracle  = oracle_correct[np.isin(results['LAL']['targets'], groups['medium'])].mean()
    tail_oracle = oracle_correct[np.isin(results['LAL']['targets'], groups['tail'])].mean()
    print(f"  Oracle by group: Head={head_oracle:.2%}  Medium={med_oracle:.2%}  Tail={tail_oracle:.2%}")

    # Oracle with best 2
    print("\n  --- 2-expert oracles ---")
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            pair_correct = correct_masks[n1] | correct_masks[n2]
            pair_oracle = pair_correct.mean()
            pair_gap = pair_oracle - best_single['ba']
            print(f"  {n1}+{n2:>8}: Oracle={pair_oracle:.2%}  (gap to best single: {pair_gap:+.2%})")

    # ── 8. Calibration deep-dive ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION E: CALIBRATION — OVERCONFIDENCE CHECK")
    print("=" * 72)
    for name in names:
        r = results[name]
        correct = (r['preds'] == r['targets'])
        probs = r['probs']
        max_confs = probs.max(axis=1)

        # Top-10 most confident wrong predictions
        wrong_mask = ~correct
        wrong_conf = max_confs[wrong_mask]
        if len(wrong_conf) > 0:
            top10_wrong_idx = np.argsort(wrong_conf)[-10:][::-1]
            top10_conf = wrong_conf[top10_wrong_idx]
            print(f"\n  {name}: Top-10 confidence on WRONG predictions (mean={max_confs[wrong_mask].mean():.4f}):")
            print(f"    Confidences: {[f'{c:.4f}' for c in top10_conf]}")
            print(f"    Any at p=1.0? {'⚠️  YES — overconfident!' if any(c >= 0.9999 for c in top10_conf) else '✅ None at p=1.0'}")
        else:
            print(f"\n  {name}: No wrong predictions — perfect accuracy")

        # ECE per group
        for grp_name, cls_list in groups.items():
            mask = np.isin(r['targets'], cls_list)
            if mask.sum() == 0:
                continue
            grp_ece, _ = expected_calibration_error(max_confs[mask], correct[mask])
            grp_avg_conf = max_confs[mask].mean()
            grp_acc = correct[mask].mean()
            print(f"    {grp_name:>6s}: ECE={grp_ece:.4f}  avg_conf={grp_avg_conf:.2%}  acc={grp_acc:.2%}")

    # ── 9. Routing potential & ensemble comparisons ──────────────────────
    print("\n" + "=" * 72)
    print("SECTION F: ENSEMBLE & ROUTING POTENTIAL")
    print("=" * 72)

    # Uniform average ensemble
    print("\n  --- Uniform Average Ensemble ---")
    avg_probs = sum(results[n]['probs'] for n in names) / 3
    avg_preds = avg_probs.argmax(axis=1)
    targets = results[names[0]]['targets']
    avg_ba, _ = balanced_accuracy(targets, avg_preds)
    avg_grp = group_accuracies(targets, avg_preds, groups)
    print(f"  Average Ensemble BA       : {avg_ba:.2%}  (H={avg_grp['head']:.2%} M={avg_grp['medium']:.2%} T={avg_grp['tail']:.2%})")

    # Majority vote ensemble
    print("\n  --- Majority Vote Ensemble ---")
    all_preds_stack = np.stack([results[n]['preds'] for n in names], axis=0)
    mv_preds, _ = mode_stats(all_preds_stack, axis=0)
    mv_ba, _ = balanced_accuracy(targets, mv_preds)
    mv_grp = group_accuracies(targets, mv_preds, groups)
    print(f"  Majority Vote BA          : {mv_ba:.2%}  (H={mv_grp['head']:.2%} M={mv_grp['medium']:.2%} T={mv_grp['tail']:.2%})")

    # Confidence-based routing (pick expert with highest max softmax)
    print("\n  --- Confidence-based Routing ---")
    max_confs_all = np.stack([results[n]['probs'].max(axis=1) for n in names], axis=1)
    best_expert_idx = max_confs_all.argmax(axis=1)
    conf_preds = np.array([results[names[i]]['preds'][j] for j, i in enumerate(best_expert_idx)])
    conf_ba, _ = balanced_accuracy(targets, conf_preds)
    conf_grp = group_accuracies(targets, conf_preds, groups)
    print(f"  Confidence Routing BA     : {conf_ba:.2%}  (H={conf_grp['head']:.2%} M={conf_grp['medium']:.2%} T={conf_grp['tail']:.2%})")

    # Best expert selection rate
    for i, name in enumerate(names):
        rate = (best_expert_idx == i).mean()
        print(f"    {name:8s} selected: {rate:.2%}")

    # ── 10. Summary verdicts ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY VERDICTS")
    print("=" * 72)

    # Diversity check
    all_diverse = all(v < 0.80 for v in kappa_matrix.values())
    print(f"\n  Diversity (κ < 0.80 for all pairs): {'✅ PASS' if all_diverse else '❌ FAIL'}")

    # Overconfidence check
    overconf_experts = []
    for row in perf_rows:
        if row['conf_wrong'] > 0.50:
            overconf_experts.append(row['expert'])
    if not overconf_experts:
        print(f"  Overconfidence (conf_wrong < 0.50): ✅ PASS")
    else:
        print(f"  Overconfidence (conf_wrong < 0.50): ⚠️  {overconf_experts} have conf_wrong > 0.50")

    # Oracle gap
    print(f"  Oracle gap > 10% (routing headroom)  : {'✅ PASS' if oracle_gap > 0.10 else '⚠️  Below 10%'} — gap = {oracle_gap:.2%}")

    # Training convergence
    print(f"  Training convergence checked         : ✅ See Section B")

    # Print final table again for quick reference
    print(f"\n{'─'*72}")
    print("FINAL METRICS SUMMARY")
    print(f"{'─'*72}")
    header = f"{'Expert':>8} | {'BA':>8} | {'Head':>8} | {'Med':>8} | {'Tail':>8} | {'Conf✗':>8} | {'ECE':>8}"
    print(f"  {header}")
    print(f"  {'─'*len(header)}")
    for row in perf_rows:
        print(f"  {row['expert']:>8} | {row['ba']:>7.2%} | {row['head']:>7.2%} | {row['medium']:>7.2%} | {row['tail']:>7.2%} | "
              f"{row['conf_wrong']:>7.2%} | {row['ece']:>7.4f}")
    print(f"\n  Oracle               : {oracle_acc:.2%}")
    print(f"  Avg Ensemble          : {avg_ba:.2%}")
    print(f"  Majority Vote         : {mv_ba:.2%}")
    print(f"  Confidence Routing    : {conf_ba:.2%}")
    print(f"{'─'*72}")
    print("Analysis complete.")


def mode_stats(arr: np.ndarray, axis: int = 0):
    """Compute mode (most frequent value) along axis."""
    from scipy import stats as sp_stats
    mode_result = sp_stats.mode(arr, axis=axis)
    return mode_result.mode, mode_result.count


if __name__ == '__main__':
    main()
