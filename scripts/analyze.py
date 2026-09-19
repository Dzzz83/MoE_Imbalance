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

from scripts.utils.data import load_all_experts, create_cifar_loader, print_data_info
from scripts.utils.metrics import (
    balanced_accuracy, per_class_accuracy, confidence_metrics,
)
from scripts.utils.features import extract_all_experts, softmax
from scripts.utils.test_access import TestAccessError
from scripts.expert_diagnostics import ExpertDiagnostics


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
    diagnostics = ExpertDiagnostics(
        predictions=preds,
        labels=targets,
        class_counts=class_counts,
        expert_names=expert_names,
    )
    complementarity = diagnostics.complementarity()
    num_correct = np.asarray(complementarity['num_correct_experts'])

    # Correctness breakdown
    all_correct = (num_correct == num_experts)
    all_wrong = (num_correct == 0)

    print('\nCorrectness breakdown:')
    print(f'  All {num_experts} correct: {all_correct.sum():5d} ({all_correct.mean()*100:.1f}%)')
    for count in range(num_experts - 1, 0, -1):
        print(f'  Exactly {count} correct:     {(num_correct == count).sum():5d} '
              f'({(num_correct == count).mean()*100:.1f}%)')
    print(f'  All wrong:               {all_wrong.sum():5d} ({all_wrong.mean()*100:.1f}%)')

    # Per-expert accuracy
    print('\nPer-expert balanced accuracy:')
    for i, name in enumerate(expert_names):
        metrics = complementarity['per_expert'][name]
        ba = metrics['ba']
        acc = metrics['accuracy']
        print(f'  {name:<10} BA={ba:.2%}  Acc={acc:.2%}')

    # Pairwise agreement (Cohen's κ)
    print("\nPairwise agreement (Cohen's κ):")
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
    print('\nPer-class accuracy correlation (Pearson r):')
    per_class = [per_class_accuracy(targets, preds[:, i]) for i in range(num_experts)]
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            accs_i = np.array([per_class[i][c] for c in range(100)])
            accs_j = np.array([per_class[j][c] for c in range(100)])
            r = np.corrcoef(accs_i, accs_j)[0, 1]
            print(f'  {expert_names[i]:<8} vs {expert_names[j]:<8}: r={r:.4f}')

    # Groups
    print('\nGroup accuracy by expert:')
    print(f'  {"Expert":<10} {"Head":>8} {"Med":>8} {"Tail":>8}')
    for i, name in enumerate(expert_names):
        metrics = complementarity['per_expert'][name]
        print(f'  {name:<10} {metrics.get("head",0)*100:>7.2f}% '
              f'{metrics.get("medium",0)*100:>7.2f}% '
              f'{metrics.get("tail",0)*100:>7.2f}%')

    print('\nPrediction-agreement patterns:')
    for pattern, count in complementarity['agreement_patterns']['pattern_counts'].items():
        print(f'  {pattern:<10} {count:5d} ({count / N * 100:.1f}%)')

    # Oracle and all-wrong ceiling
    headroom = diagnostics.hard_routing_headroom()
    oracle_ba = headroom['hard_selection_oracle_balanced_accuracy']
    print(f'\nOracle BA: {oracle_ba:.2%}')
    print(f'All-experts-wrong fraction: '
          f'{headroom["all_experts_wrong_fraction"] * 100:.1f}%')


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
    all_wrong = ~correct.any(axis=1)

    diagnostics = ExpertDiagnostics(
        predictions=preds,
        logits=logits,
        labels=targets,
        class_counts=class_counts,
        expert_names=expert_names,
    )
    correctness_report = diagnostics.correctness_diagnostics()
    probs = softmax(logits)
    conf = probs.max(axis=2)

    print('\nCorrectness-label ambiguity (denominators are explicit):')
    for key in ('exactly_one_correct', 'multiple_correct', 'no_correct'):
        event = correctness_report[key]
        print(f'  {key}: {event["count"]:,}/{event["denominator"]:,} '
              f'({event["fraction"] * 100:.1f}%)')
    global_conf = correctness_report['globally_most_confident']
    print(f'  globally most-confident expert correct: '
          f'{global_conf["correct_count"]:,}/{global_conf["denominator"]:,} '
          f'({global_conf["correct_fraction"] * 100:.1f}%)')
    ranking = correctness_report['confidence_ranking_among_correct']
    print('  confidence rank of best correct expert, conditioned on any correct:')
    for rank, fraction in ranking['best_correct_rank_fractions'].items():
        print(f'    rank {rank}: {fraction * 100:.1f}% '
              f'({ranking["best_correct_rank_counts"][rank]}/'
              f'{ranking["denominator"]})')
    unique_conf = correctness_report['unique_highest_confidence_among_correct']
    print('  unique highest-confidence correct expert (same conditioning): '
          f'{unique_conf["count"]}/{unique_conf["denominator"]} '
          f'({unique_conf["fraction"] * 100:.1f}%)')

    # 2. Confidence routing analysis
    print('\nConfidence routing:')
    conf_choices = conf.argmax(axis=1)
    conf_correct = correct[np.arange(N), conf_choices]
    conf_ba = balanced_accuracy(targets, preds[np.arange(N), conf_choices])
    print(f'  Picks correct expert: {conf_correct.mean()*100:.1f}%')
    print(f'  BA: {conf_ba:.2%}')

    # 3. General agreement/dissenter diagnostic.  A 2-1-1 pattern is reported
    # as non-unique; it is never forced into this denominator.
    patterns = diagnostics.agreement_patterns()
    unique = patterns['unique_dissenter']
    print(f'\nUnique-dissenter correctness (only {unique["pattern"]} patterns):')
    print(f'  agreeing group correct: {unique["agreeing_group_correct_count"]}/'
          f'{unique["denominator"]} '
          f'({unique["agreeing_group_correct_fraction"] * 100:.1f}%)')
    print(f'  dissenting expert correct: {unique["dissenting_expert_correct_count"]}/'
          f'{unique["denominator"]} '
          f'({unique["dissenting_expert_correct_fraction"] * 100:.1f}%)')

    # 4. Feature learning gap (oracle-weighted routing vs uniform)
    # Oracle-weighted: for each sample, use the best expert (oracle)
    # This tells us the maximum possible routing gain
    uniform_ba = balanced_accuracy(targets, logits.mean(axis=1).argmax(axis=1))

    oracle_ba = diagnostics.hard_routing_headroom()['hard_selection_oracle_balanced_accuracy']

    print('\nRouting headroom:')
    print(f'  Uniform avg BA: {uniform_ba:.2%}')
    print(f'  Oracle BA:      {oracle_ba:.2%}')
    print(f'  Oracle gap:     {oracle_ba - uniform_ba:+.2%}')

    # 5. Per-class error overlap
    print(f'\nPer-class error overlap (all {num_experts} experts wrong):')
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

    print('\nPer-expert calibration:')
    print(f'  {"Expert":<10} {"ECE":>8} {"Avg Conf":>10} {"Avg Conf Corr":>14} {"Avg Conf Wrong":>14}')
    for i, name in enumerate(expert_names):
        conf = probs[:, i].max(axis=1)
        corr = correct[:, i]
        cm = confidence_metrics(conf, corr)
        print(f'  {name:<10} {cm["ece"]*100:>7.2f}% {conf.mean():>9.4f} '
              f'{cm["avg_conf_correct"]:>13.4f} {cm["avg_conf_wrong"]:>13.4f}')

    # Confidence distributions
    print('\nConfidence distribution (binned):')
    bins = np.linspace(0, 1, 11)
    for i, name in enumerate(expert_names):
        conf = probs[:, i].max(axis=1)
        hist, _ = np.histogram(conf, bins=bins)
        print(f'  {name:<10}: {hist}')


if __name__ == '__main__':
    try:
        main()
    except TestAccessError as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
