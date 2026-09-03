#!/usr/bin/env python3
"""
Run all routing methods and produce a comparison table.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --routers uniform,confidence,product
    python scripts/benchmark.py --dataset test --output results.json
"""

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import torch

from scripts.utils.data import (
    load_all_experts, create_cifar_loader, get_class_groups, print_data_info,
)
from scripts.utils.metrics import compute_routing_metrics, balanced_accuracy
from scripts.utils.features import extract_all_experts, softmax, compute_89d_features

from scripts.router import (
    UniformRouter, ConfidenceRouter, ProductRouter,
    CorrectnessRouter, PairwiseRouter, ClusterRouter,
    GateRouter,
)

# ── Router registry ──
ROUTER_REGISTRY = OrderedDict([
    ('uniform',     UniformRouter),
    ('confidence',  ConfidenceRouter),
    ('product',     ProductRouter),
    ('correctness', CorrectnessRouter),
    ('pairwise',    PairwiseRouter),
    ('cluster',     ClusterRouter),
    ('gate',        GateRouter),
])

DEFAULT_ROUTERS = list(ROUTER_REGISTRY.keys())


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark all routing methods on CIFAR-100-LT'
    )
    parser.add_argument('--routers', type=str, default=','.join(DEFAULT_ROUTERS),
                        help='Comma-separated list of routers to run')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split for evaluation')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    selected_routers = args.routers.split(',')
    device = args.device

    print('=' * 65)
    print('  EXPERT ROUTING BENCHMARK')
    print('=' * 65)

    # ── Step 1: Load experts ──
    print('\n[1/5] Loading expert models...')
    models = load_all_experts(
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    for name, model in models.items():
        ba = model._checkpoint_ba
        print(f'  {name}: epoch={model._checkpoint_epoch}, '
              f'val_ba={ba:.2%}' if ba else f'  {name}: loaded')

    # ── Step 2: Load validation set ──
    print('\n[2/5] Loading validation set...')
    val_loader, val_counts = create_cifar_loader(
        'val', args.data_root, batch_size=args.batch_size,
    )
    print_data_info(val_counts)

    # ── Step 3: Extract validation features ──
    print('\n[3/5] Extracting validation features...')
    t0 = time.time()
    val_extracted = extract_all_experts(
        models, val_loader, device, return_features=True
    )
    val_logits = np.stack([val_extracted[n]['logits'] for n in models], axis=1)
    val_labels = val_extracted['targets']
    print(f'  Validation: {len(val_labels)} samples, {time.time()-t0:.1f}s')

    # ── Step 4: Load evaluation set ──
    print(f'\n[4/5] Loading {args.dataset} set...')
    eval_loader, eval_counts = create_cifar_loader(
        args.dataset, args.data_root, batch_size=args.batch_size,
    )

    print('\n[5/5] Extracting evaluation features...')
    t0 = time.time()
    eval_extracted = extract_all_experts(
        models, eval_loader, device, return_features=True
    )
    eval_logits = np.stack([eval_extracted[n]['logits'] for n in models], axis=1)
    eval_labels = eval_extracted['targets']
    print(f'  Evaluation: {len(eval_labels)} samples, {time.time()-t0:.1f}s')

    # ── Run all routers ──
    results = OrderedDict()
    results['info'] = {
        'experts': list(models.keys()),
        'train_counts': val_counts.tolist(),  # placeholder — use actual train counts
        'n_val': len(val_labels),
        'n_eval': len(eval_labels),
    }

    print(f'\n{"="*65}')
    print(f'  ROUTING RESULTS ON {args.dataset.upper()} SET')
    print(f'{"="*65}')
    print(f'  {"Router":<20} {"BA":>8} {"Head":>8} {"Med":>8} {"Tail":>8} '
          f'{"OracleGap":>10} {"AllWrong":>10}')
    print(f'  {"-"*20} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*10} {"-"*10}')

    # Baselines
    baselines = _compute_baselines(eval_logits, eval_labels, list(models.keys()))
    for name, ba in baselines.items():
        print(f'  {name:<20} {ba*100:>8.2f}%')

    # Per-expert BA
    for e, name in enumerate(models):
        e_ba = balanced_accuracy(eval_labels, eval_logits[:, e].argmax(axis=1))
        print(f'  {name:<20} {e_ba*100:>8.2f}%')

    print(f'  {"-"*64}')

    for router_name in selected_routers:
        if router_name not in ROUTER_REGISTRY:
            print(f'  ❌ Unknown router: {router_name}')
            continue

        router_cls = ROUTER_REGISTRY[router_name]
        router = router_cls(list(models.keys()))

        try:
            t0 = time.time()
            router.train(val_logits, val_labels, val_extracted)
            metrics = router.evaluate(
                eval_logits, eval_labels,
                class_counts=val_counts,
                features=eval_extracted,
            )
            elapsed = time.time() - t0

            results[router_name] = metrics

            print(f'  {router.name:<20} {metrics["ba"]*100:>8.2f}% '
                  f'{metrics["head_acc"]*100:>8.2f}% '
                  f'{metrics["med_acc"]*100:>8.2f}% '
                  f'{metrics["tail_acc"]*100:>8.2f}% '
                  f'{metrics["oracle_gap"]*100:>10.2f}% '
                  f'{metrics["all_wrong_pct"]:>10.1f}% '
                  f'  ({elapsed:.1f}s)')
        except Exception as e:
            print(f'  {router.name:<20} ❌ {e}')
            results[router_name] = {'error': str(e)}

    # ── Oracle ──
    expert_preds = eval_logits.argmax(axis=2)
    oracle_preds = np.zeros(len(eval_labels), dtype=np.int64)
    for i in range(len(eval_labels)):
        correct = expert_preds[i] == eval_labels[i]
        if correct.any():
            oracle_preds[i] = expert_preds[i][np.where(correct)[0][0]]
        else:
            oracle_preds[i] = expert_preds[i][0]
    oracle_ba = balanced_accuracy(eval_labels, oracle_preds)
    print(f'  {"Oracle":<20} {oracle_ba*100:>8.2f}%')

    print(f'{"="*65}')

    # ── Save results ──
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'\nResults saved to {args.output}')

    print('\n✅ Benchmark complete.')


def _compute_baselines(
    logits: np.ndarray,
    labels: np.ndarray,
    expert_names: list[str],
) -> dict:
    """Compute non-learned baselines."""
    baselines = OrderedDict()

    # Uniform average
    avg_logits = logits.mean(axis=1)
    baselines['Uniform avg'] = balanced_accuracy(labels, avg_logits.argmax(axis=1))

    # Optimal fixed weights (grid search on eval set — optimistic baseline)
    probs = softmax(logits)
    best_ba = 0.0
    for w0 in np.linspace(0, 1, 21):
        for w1 in np.linspace(0, 1 - w0, 21):
            w2 = 1.0 - w0 - w1
            if w2 < 0:
                continue
            combined = w0 * probs[:, 0] + w1 * probs[:, 1] + w2 * probs[:, 2]
            ba = balanced_accuracy(labels, combined.argmax(axis=1))
            if ba > best_ba:
                best_ba = ba
    baselines['Opt fixed (oracle)'] = best_ba

    return baselines


if __name__ == '__main__':
    main()
