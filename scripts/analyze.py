#!/usr/bin/env python3
"""
Unified analysis script for expert diversity, root cause, and calibration.

Usage:
    python scripts/analyze.py --mode diversity
    python scripts/analyze.py --mode root_cause
    python scripts/analyze.py --mode calibration
    python scripts/analyze.py --mode all
"""

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import time

import numpy as np
import torch

from scripts.utils.data import (
    load_all_experts, create_cifar_loader, get_class_groups, print_data_info,
)
from scripts.utils.metrics import (
    balanced_accuracy, per_class_accuracy, group_accuracies, confidence_metrics, ece,
)
from scripts.utils.features import extract_all_experts, softmax


def main():
    parser = argparse.ArgumentParser(
        description='Analyze expert behavior on CIFAR-100-LT'
    )
    parser.add_argument('--mode', type=str, default='all',
                        choices=['diversity', 'root_cause', 'calibration', 'all'],
                        help='Analysis mode')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='test',
                        help='Dataset to analyze')
    args = parser.parse_args()

    print('=' * 60)
    print(f'  EXPERT ANALYSIS — mode={args.mode}')
    print('=' * 60)

    # ── Load experts and data ──
    print('\nLoading experts...')
    models = load_all_experts(device=args.device, checkpoint_dir=args.checkpoint_dir)
    expert_names = list(models.keys())

    loader, class_counts = create_cifar_loader(
        args.dataset, args.data_root, batch_size=args.batch_size,
    )
    print_data_info(class_counts)

    print('Extracting features...')
    t0 = time.time()
    extracted = extract_all_experts(models, loader, args.device, return_features=True)
    print(f'  Done in {time.time()-t0:.1f}s')

    logits = np.stack([extracted[n]['logits'] for n in expert_names], axis=1)
    probs = softmax(logits)
    preds = logits.argmax(axis=2)
    targets = extracted['targets']
    correct = (preds == targets[:, None])

    if args.mode in ('diversity', 'all'):
        _analyze_diversity(expert_names, preds, targets, correct, probs, class_counts)

    if args.mode in ('root_cause', 'all'):
        _analyze_root_cause(expert_names, preds, targets, correct, logits, class_counts)

    if args.mode in ('calibration', 'all'):
        _analyze_calibration(expert_names, preds, targets, correct, probs, logits)


def _analyze_diversity(
    expert_names: list[str],
    preds: np.ndarray,
    targets: np.ndarray,
    correct: np.ndarray,
    probs: np.ndarray,
    class_counts: np.ndarray,
):
    """Analyze expert diversity and agreement patterns."""
    print('\n' + '=' * 60)
    print('  DIVERSITY ANALYSIS')
    print('=' * 60)

    N = len(targets)
    num_experts = len(expert_names)
    num_correct = correct.sum(axis=1)

    # Correctness breakdown
    all_correct = (num_correct == num_experts)
    any_correct = (num_correct > 0)
    all_wrong = (num_correct == 0)

    print(f'\nCorrectness breakdown:')
    print(f'  All {num_experts} correct: {all_correct.sum():5d} ({all_correct.mean()*100:.1f}%)')
    print(f'  Exactly 2 correct:     {(num_correct == 2).sum():5d} ({(num_correct == 2).mean()*100:.1f}%)')
    print(f'  Exactly 1 correct:     {(num_correct == 1).sum():5d} ({(num_correct == 1).mean()*100:.1f}%)')
    print(f'  All wrong:             {all_wrong.sum():5d} ({all_wrong.mean()*100:.1f}%)')

    # Per-expert accuracy
    print(f'\nPer-expert balanced accuracy:')
    for i, name in enumerate(expert_names):
        ba = balanced_accuracy(targets, preds[:, i])
        acc = (preds[:, i] == targets).mean()
        print(f'  {name:<10} BA={ba:.2%}  Acc={acc:.2%}')

    # Pairwise agreement (Cohen's κ)
    print(f'\nPairwise agreement (Cohen\'s κ):')
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            agree = (preds[:, i] == preds[:, j]).mean()
            # Expected agreement by chance
            from collections import Counter
            cnt_i = Counter(preds[:, i])
            cnt_j = Counter(preds[:, j])
            expected = sum(cnt_i.get(c, 0) * cnt_j.get(c, 0) for c in set(list(cnt_i.keys()) + list(cnt_j.keys()))) / N**2
            kappa = (agree - expected) / (1 - expected) if expected < 1 else 0
            print(f'  {expert_names[i]:<8} vs {expert_names[j]:<8}: agree={agree:.2%}, κ={kappa:.4f}')

    # Per-class accuracy correlation
    print(f'\nPer-class accuracy correlation (Pearson r):')
    per_class = [per_class_accuracy(targets, preds[:, i]) for i in range(num_experts)]
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            accs_i = np.array([per_class[i][c] for c in range(100)])
            accs_j = np.array([per_class[j][c] for c in range(100)])
            r = np.corrcoef(accs_i, accs_j)[0, 1]
            print(f'  {expert_names[i]:<8} vs {expert_names[j]:<8}: r={r:.4f}')

    # Groups
    groups = get_class_groups(class_counts)
    print(f'\nGroup accuracy by expert:')
    print(f'  {"Expert":<10} {"Head":>8} {"Med":>8} {"Tail":>8}')
    for i, name in enumerate(expert_names):
        ga = group_accuracies(targets, preds[:, i], groups)
        print(f'  {name:<10} {ga.get("Head",0)*100:>7.2f}% {ga.get("Med",0)*100:>7.2f}% {ga.get("Tail",0)*100:>7.2f}%')

    # Oracle and all-wrong ceiling
    oracle_preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        c = correct[i]
        if c.any():
            oracle_preds[i] = preds[i, np.where(c)[0][0]]
        else:
            oracle_preds[i] = preds[i, 0]
    oracle_ba = balanced_accuracy(targets, oracle_preds)
    print(f'\nOracle BA: {oracle_ba:.2%}')
    print(f'All-wrong ceiling: {all_wrong.mean()*100:.1f}%')


def _analyze_root_cause(
    expert_names: list[str],
    preds: np.ndarray,
    targets: np.ndarray,
    correct: np.ndarray,
    logits: np.ndarray,
    class_counts: np.ndarray,
):
    """Analyze root causes of routing failures."""
    print('\n' + '=' * 60)
    print('  ROOT CAUSE ANALYSIS')
    print('=' * 60)

    N = len(targets)
    num_experts = len(expert_names)
    num_correct = correct.sum(axis=1)
    any_correct = (num_correct > 0)
    all_wrong = (num_correct == 0)

    # 1. Label ambiguity — how many samples have a unique best expert?
    # Best expert = the one with highest confidence among those that are correct
    probs = softmax(logits)
    conf = probs.max(axis=2)

    best_expert = conf.argmax(axis=1)
    best_is_correct = correct[np.arange(N), best_expert]

    # Count how many samples have exactly one expert that is correct and confident
    unique_best = np.zeros(N, dtype=bool)
    for i in range(N):
        if num_correct[i] == 0:
            continue
        # Among correct experts, is there one with strictly highest confidence?
        correct_conf = conf[i, correct[i]]
        if correct_conf.sum() > 0 and (correct_conf == correct_conf.max()).sum() == 1:
            unique_best[i] = True

    print(f'\nLabel ambiguity:')
    print(f'  Samples with unique best expert: {unique_best.sum():,} ({unique_best.mean()*100:.1f}%)')
    print(f'  Samples with tied/no best:       {N - unique_best.sum():,} ({(1-unique_best.mean())*100:.1f}%)')

    # 2. Confidence routing analysis
    print(f'\nConfidence routing:')
    conf_choices = conf.argmax(axis=1)
    conf_correct = correct[np.arange(N), conf_choices]
    conf_ba = balanced_accuracy(targets, preds[np.arange(N), conf_choices])
    print(f'  Picks correct expert: {conf_correct.mean()*100:.1f}%')
    print(f'  BA: {conf_ba:.2%}')

    # 3. Lone Dissenter Paradox
    # When exactly 2 experts agree and 1 dissents, is the dissenter more often correct?
    print(f'\nLone Dissenter Paradox:')
    lone_dissenter_correct = 0
    lone_dissenter_total = 0
    for i in range(N):
        if num_correct[i] != 2:
            continue
        # Find the two that agree (same prediction)
        pairs_agree = []
        for a in range(num_experts):
            for b in range(a + 1, num_experts):
                if preds[i, a] == preds[i, b]:
                    pairs_agree.append((a, b))
        if len(pairs_agree) == 1:
            a, b = pairs_agree[0]
            # The dissenter is the third expert
            dissenter = [e for e in range(num_experts) if e not in (a, b)][0]
            # Majority prediction
            majority_pred = preds[i, a]
            majority_correct = (majority_pred == targets[i])
            dissenter_correct = correct[i, dissenter]
            lone_dissenter_total += 1
            if dissenter_correct and not majority_correct:
                lone_dissenter_correct += 1

    if lone_dissenter_total > 0:
        print(f'  When exactly 2 agree, dissenter is correct: '
              f'{lone_dissenter_correct}/{lone_dissenter_total} '
              f'({lone_dissenter_correct/lone_dissenter_total*100:.1f}%)')

    # 4. Feature learning gap (oracle-weighted routing vs uniform)
    # Oracle-weighted: for each sample, use the best expert (oracle)
    # This tells us the maximum possible routing gain
    uniform_ba = balanced_accuracy(targets, logits.mean(axis=1).argmax(axis=1))

    # Oracle BA (already computed in diversity)
    oracle_preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        c = correct[i]
        if c.any():
            oracle_preds[i] = preds[i, np.where(c)[0][0]]
        else:
            oracle_preds[i] = preds[i, 0]
    oracle_ba = balanced_accuracy(targets, oracle_preds)

    print(f'\nRouting headroom:')
    print(f'  Uniform avg BA: {uniform_ba:.2%}')
    print(f'  Oracle BA:      {oracle_ba:.2%}')
    print(f'  Oracle gap:     {oracle_ba - uniform_ba:+.2%}')

    # 5. Per-class error overlap
    print(f'\nPer-class error overlap (all 3 experts wrong):')
    per_class_all_wrong = np.zeros(100)
    per_class_count = np.zeros(100)
    for c in range(100):
        mask = targets == c
        if mask.sum() > 0:
            per_class_all_wrong[c] = all_wrong[mask].mean()
            per_class_count[c] = mask.sum()
    for c in np.argsort(per_class_all_wrong)[-5:]:
        if per_class_count[c] > 0:
            print(f'  Class {c}: {per_class_all_wrong[c]*100:.1f}% all wrong ({per_class_count[c]:.0f} samples)')


def _analyze_calibration(
    expert_names: list[str],
    preds: np.ndarray,
    targets: np.ndarray,
    correct: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray,
):
    """Analyze expert calibration."""
    print('\n' + '=' * 60)
    print('  CALIBRATION ANALYSIS')
    print('=' * 60)

    print(f'\nPer-expert calibration:')
    print(f'  {"Expert":<10} {"ECE":>8} {"Avg Conf":>10} {"Avg Conf Corr":>14} {"Avg Conf Wrong":>14}')
    for i, name in enumerate(expert_names):
        conf = probs[:, i].max(axis=1)
        corr = correct[:, i]
        cm = confidence_metrics(conf, corr)
        print(f'  {name:<10} {cm["ece"]*100:>7.2f}% {conf.mean():>9.4f} '
              f'{cm["avg_conf_correct"]:>13.4f} {cm["avg_conf_wrong"]:>13.4f}')

    # Confidence distributions
    print(f'\nConfidence distribution (binned):')
    bins = np.linspace(0, 1, 11)
    for i, name in enumerate(expert_names):
        conf = probs[:, i].max(axis=1)
        hist, _ = np.histogram(conf, bins=bins)
        print(f'  {name:<10}: {hist}')


if __name__ == '__main__':
    main()
