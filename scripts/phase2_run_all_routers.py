#!/usr/bin/env python3
"""
Phase 2: Run all routing strategies on cached data.

Loads cached val/test features (from Phase 1), trains each router on the
val set, evaluates on the test set, and produces a comparison table.

Usage:
    python scripts/phase2_run_all_routers.py
    python scripts/phase2_run_all_routers.py --routers uniform,confidence,correctness
"""

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import json
import time
import pickle
from collections import OrderedDict

import numpy as np

from scripts.utils.data import get_class_groups
from scripts.utils.metrics import balanced_accuracy, group_accuracies
from scripts.utils.features import softmax

from scripts.router import (
    UniformRouter, ConfidenceRouter, ProductRouter,
    CorrectnessRouter, PairwiseRouter, ClusterRouter,
    GateRouter, SelectiveRouter,
)

CACHE_DIR = os.path.join(_proj_root, 'checkpoints', 'cache')
EXPERT3_NAMES = ['LAL', 'PaCo', 'Mixup']

# ── Router registry ──
# Each entry: (name, router_class, kwargs, group)
ROUTER_CONFIGS = OrderedDict([
    # Non-learned baselines (implemented as routers for consistent interface)
    ('uniform',     (UniformRouter,     {}, 'baseline')),
    ('product',     (ProductRouter,     {}, 'baseline')),
    ('confidence',  (ConfidenceRouter,  {'calibrate': True}, 'simple')),
    ('confidence_raw', (ConfidenceRouter, {'calibrate': False}, 'simple')),

    # Learned routers (trained on val set)
    ('correctness_24d',  (CorrectnessRouter, {'use_92d': False, 'hidden_layer_sizes': ()}, 'learned')),
    ('correctness_89d',  (CorrectnessRouter, {'use_92d': False}, 'learned')),
    ('correctness_92d',  (CorrectnessRouter, {'use_92d': True}, 'learned')),
    ('pairwise',    (PairwiseRouter,    {}, 'learned')),
    ('gate',        (GateRouter,        {}, 'learned')),
    ('cluster',     (ClusterRouter,     {'variant': 'hard', 'n_clusters': 8}, 'learned')),
    ('cluster_soft', (ClusterRouter,    {'variant': 'soft', 'n_clusters': 8}, 'learned')),

    # Selective routers (wrap correctness_92d)
    ('selective_92d_15',  (lambda en: SelectiveRouter(
        CorrectnessRouter(en, use_92d=True), en, confidence_threshold=0.15), {}, 'selective')),
    ('selective_92d_30',  (lambda en: SelectiveRouter(
        CorrectnessRouter(en, use_92d=True), en, confidence_threshold=0.30), {}, 'selective')),
    ('selective_92d_40',  (lambda en: SelectiveRouter(
        CorrectnessRouter(en, use_92d=True), en, confidence_threshold=0.40), {}, 'selective')),
])


def load_cached(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Run all routing strategies on cached data'
    )
    parser.add_argument('--routers', type=str, default=None,
                        help='Comma-separated list of routers to run (default: all)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(CACHE_DIR, 'routing_results.json'),
                        help='Output path for results JSON')
    args = parser.parse_args()

    selected = args.routers.split(',') if args.routers else list(ROUTER_CONFIGS.keys())

    # ── Load cached data ──
    print('=' * 65)
    print('  PHASE 2: Routing Benchmark (All Strategies)')
    print('=' * 65)

    print('\n[1/4] Loading cached features...')
    val_extracted = load_cached(os.path.join(CACHE_DIR, 'extracted_val.pkl'))
    test_extracted = load_cached(os.path.join(CACHE_DIR, 'extracted_test.pkl'))
    print(f'  Val:   {len(val_extracted["targets"]):,} samples')
    print(f'  Test:  {len(test_extracted["targets"]):,} samples')

    # Build logits arrays
    val_logits = np.stack([val_extracted[n]['logits'] for n in EXPERT3_NAMES], axis=1)
    test_logits = np.stack([test_extracted[n]['logits'] for n in EXPERT3_NAMES], axis=1)
    val_labels = val_extracted['targets']
    test_labels = test_extracted['targets']

    # ── Load group info ──
    print('\n[2/4] Loading class groups...')
    with open(os.path.join(CACHE_DIR, 'group_info.json'), 'r') as f:
        group_info = json.load(f)
    # Reconstruct class_counts array from group info
    train_counts = np.zeros(100, dtype=int)
    for c, cnt in zip(group_info['head_classes'], group_info['head_counts']):
        train_counts[c] = cnt
    for c, cnt in zip(group_info['med_classes'], group_info['med_counts']):
        train_counts[c] = cnt
    for c, cnt in zip(group_info['tail_classes'], group_info['tail_counts']):
        train_counts[c] = cnt
    groups = get_class_groups(train_counts)
    print(f'  Head: {len(groups["Head"])} classes, Med: {len(groups["Med"])}, Tail: {len(groups["Tail"])}')

    # ── Load baselines for comparison ──
    print('\n[3/4] Loading baselines...')
    with open(os.path.join(CACHE_DIR, 'baselines.json'), 'r') as f:
        baselines = json.load(f)
    uniform_ba = baselines['uniform_avg']['ba']
    opt_fixed_ba = baselines['opt_fixed']['ba']
    oracle_ba = baselines['oracle']['ba']
    print(f'  Uniform avg:  {uniform_ba:.2%}')
    print(f'  Opt fixed:    {opt_fixed_ba:.2%}')
    print(f'  Oracle:       {oracle_ba:.2%}')

    # ── Run routers ──
    print('\n[4/4] Running routers...')
    print(f'\n{"=" * 95}')
    print(f'  {"Router":<22} {"BA":>8} {"Head":>8} {"Med":>8} {"Tail":>8} '
          f'{"vsUniform":>10} {"vsOptFixed":>11} {"OracleGap":>10} {"AllWrong":>9}')
    print(f'  {"-" * 22} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 8} '
          f'{"-" * 10} {"-" * 11} {"-" * 10} {"-" * 9}')

    results = OrderedDict()
    beats_uniform = []
    beats_opt = []
    best_name = None
    best_ba_val = 0.0

    for router_name in selected:
        if router_name not in ROUTER_CONFIGS:
            print(f'  ❌ Unknown router: {router_name}')
            continue

        router_cls_or_factory, kwargs, group = ROUTER_CONFIGS[router_name]

        # Handle factory functions (for selective routers)
        if callable(router_cls_or_factory) and not isinstance(router_cls_or_factory, type):
            try:
                router = router_cls_or_factory(EXPERT3_NAMES)
            except Exception as e:
                print(f'  ❌ {router_name:<22} factory error: {e}')
                continue
        else:
            router = router_cls_or_factory(EXPERT3_NAMES, **kwargs)

        try:
            t0 = time.time()
            router.train(val_logits, val_labels, val_extracted)

            # Use predict_class() for the final class prediction (handles
            # uniform avg, product, and selective fallback correctly).
            preds = router.predict_class(test_logits, test_extracted)
            ba = balanced_accuracy(test_labels, preds)
            ga = group_accuracies(test_labels, preds, groups)
            elapsed = time.time() - t0

            # Also get expert usage for analysis
            expert_idx = router.predict(test_logits, test_extracted)
            expert_usage = {
                EXPERT3_NAMES[e]: float((expert_idx == e).mean() * 100)
                for e in range(3)
            }

            vs_uniform = ba - uniform_ba
            vs_opt = ba - opt_fixed_ba
            oracle_gap = oracle_ba - ba

            print(f'  {router_name:<22} {ba*100:>7.2f}% '
                  f'{ga.get("Head",0)*100:>7.2f}% '
                  f'{ga.get("Med",0)*100:>7.2f}% '
                  f'{ga.get("Tail",0)*100:>7.2f}% '
                  f'{vs_uniform*100:>+9.2f}% '
                  f'{vs_opt*100:>+10.2f}% '
                  f'{oracle_gap*100:>9.2f}% '
                  f'{baselines["oracle"]["all_wrong_pct"]:>8.1f}%'
                  f'  ({elapsed:.1f}s)')

            results[router_name] = {
                'ba': float(ba),
                'head_acc': float(ga.get('Head', 0)),
                'med_acc': float(ga.get('Med', 0)),
                'tail_acc': float(ga.get('Tail', 0)),
                'vs_uniform': float(vs_uniform),
                'vs_opt_fixed': float(vs_opt),
                'oracle_gap': float(oracle_gap),
                'oracle_ba': float(oracle_ba),
                'all_wrong_pct': float(baselines["oracle"]["all_wrong_pct"]),
                'expert_usage': expert_usage,
                'time_s': float(elapsed),
            }

            if vs_uniform > 0:
                beats_uniform.append((router_name, ba))
            if vs_opt > 0:
                beats_opt.append((router_name, ba))
            if ba > best_ba_val:
                best_ba_val = ba
                best_name = router_name

        except Exception as e:
            print(f'  ❌ {router_name:<22} {e}')
            results[router_name] = {'error': str(e)}

    # ── Summary ──
    print(f'\n{"=" * 95}')
    print(f'\nSummary:')
    print(f'  Uniform avg BA:  {uniform_ba:.2%}')
    print(f'  Opt fixed BA:    {opt_fixed_ba:.2%}  '
          f'(weights={baselines["opt_fixed"]["weights"]})')
    print(f'  Oracle BA:       {oracle_ba:.2%}')
    print(f'  All-wrong:       {baselines["oracle"]["all_wrong_pct"]:.1f}%')

    # Find best routing method
    beats_uniform.sort(key=lambda x: x[1], reverse=True)
    beats_opt.sort(key=lambda x: x[1], reverse=True)

    if best_name:
        best = results[best_name]
        print(f'\n  ⭐ Best routing:  {best_name} = {best["ba"]:.2%}  '
              f'(vs uniform: {best["vs_uniform"]*100:+.2f}%, '
              f'vs opt fixed: {best["vs_opt_fixed"]*100:+.2f}%)')

        if beats_uniform:
            print(f'\n  Methods that beat uniform avg ({len(beats_uniform)}/{len(results)}):')
            for name, ba in beats_uniform[:5]:
                print(f'    ✓ {name:<22} {ba:.2%}')
        else:
            print(f'\n  ⚠️  No method beats uniform avg ({uniform_ba:.2%})')

        if beats_opt:
            print(f'\n  Methods that beat opt fixed ({len(beats_opt)}/{len(results)}):')
            for name, ba in beats_opt[:3]:
                print(f'    ✓ {name:<22} {ba:.2%}')

    # ── Save results ──
    output = {
        'baselines': baselines,
        'routing': results,
        'summary': {
            'uniform_ba': uniform_ba,
            'opt_fixed_ba': opt_fixed_ba,
            'oracle_ba': oracle_ba,
            'all_wrong_pct': baselines['oracle']['all_wrong_pct'],
            'best_router': best_name if best_name else None,
            'best_ba': best['ba'] if best_name else None,
            'n_methods_beat_uniform': len(beats_uniform),
            'n_methods_beat_opt_fixed': len(beats_opt),
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n✅ Results saved to {args.output}')
    print(f'{"=" * 65}')


if __name__ == '__main__':
    main()
