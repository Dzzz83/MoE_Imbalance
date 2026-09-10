#!/usr/bin/env python3
"""
Phase 1: Cache expert outputs + compute baselines.

Extracts logits, probabilities, and backbone features for the 3 experts
(LAL, PaCo, Mixup) on val and test sets, caches them to disk, and computes
all non-learned baselines (uniform avg, product, opt fixed weights, oracle).

Usage:
    python scripts/phase1_cache_and_baselines.py
"""

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import json
import time
import pickle
from collections import OrderedDict

import numpy as np
import torch

from scripts.utils.data import (
    load_all_experts, create_cifar_loader, get_class_groups,
)
from scripts.utils.metrics import (
    balanced_accuracy, group_accuracies,
)
from scripts.utils.features import extract_all_experts, softmax

CACHE_DIR = os.path.join(_proj_root, 'checkpoints', 'cache')
EXPERT3_NAMES = ['LAL', 'PaCo', 'Mixup']


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def extract_and_cache(experts, dataset_type, batch_size=128, device='cpu'):
    """Extract features for a dataset split and cache to disk."""
    cache_path = os.path.join(CACHE_DIR, f'extracted_{dataset_type}.pkl')

    if os.path.exists(cache_path):
        print(f'  Loading cached {dataset_type} features...')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f'  Extracting {dataset_type} features (batch_size={batch_size})...')
    loader, counts = create_cifar_loader(
        dataset_type, batch_size=batch_size, num_workers=0,
    )
    t0 = time.time()
    extracted = extract_all_experts(experts, loader, device, return_features=True)
    elapsed = time.time() - t0
    print(f'    Done in {elapsed:.1f}s ({len(extracted["targets"]):,} samples)')

    # Only keep the 3 experts we care about
    filtered = {'targets': extracted['targets']}
    for name in EXPERT3_NAMES:
        filtered[name] = extracted[name]

    with open(cache_path, 'wb') as f:
        pickle.dump(filtered, f)
    print(f'    Cached to {cache_path}')

    return filtered


def compute_baselines(val_extracted, test_extracted, train_counts):
    """Compute all non-learned baselines on the test set."""
    groups = get_class_groups(train_counts)
    print(f'\nClass groups (from training counts):')
    print(f'  Head: {len(groups["Head"])} classes')
    print(f'  Med:  {len(groups["Med"])} classes')
    print(f'  Tail: {len(groups["Tail"])} classes')

    # Build logits arrays (N, 3, 100)
    val_logits = np.stack([val_extracted[n]['logits'] for n in EXPERT3_NAMES], axis=1)
    test_logits = np.stack([test_extracted[n]['logits'] for n in EXPERT3_NAMES], axis=1)
    test_probs = softmax(test_logits)
    val_probs = softmax(val_logits)
    test_labels = test_extracted['targets']
    val_labels = val_extracted['targets']

    baselines = OrderedDict()

    # ── 1. Uniform average ──
    uniform_logits = test_logits.mean(axis=1)
    uniform_preds = uniform_logits.argmax(axis=1)
    uniform_ba = balanced_accuracy(test_labels, uniform_preds)
    uniform_ga = group_accuracies(test_labels, uniform_preds, groups)
    baselines['uniform_avg'] = {
        'ba': float(uniform_ba),
        'head': float(uniform_ga.get('Head', 0)),
        'med': float(uniform_ga.get('Med', 0)),
        'tail': float(uniform_ga.get('Tail', 0)),
    }
    print(f'\nBaselines on TEST set (10K samples):')
    print(f'  Uniform avg:     BA={uniform_ba:.2%}  '
          f'(H={uniform_ga.get("Head",0)*100:.1f}% '
          f'M={uniform_ga.get("Med",0)*100:.1f}% '
          f'T={uniform_ga.get("Tail",0)*100:.1f}%)')

    # ── 2. Product combination (geometric mean) ──
    log_probs = np.log(test_probs + 1e-12)
    product_probs = np.exp(log_probs.mean(axis=1))
    product_probs /= product_probs.sum(axis=1, keepdims=True)
    product_preds = product_probs.argmax(axis=1)
    product_ba = balanced_accuracy(test_labels, product_preds)
    product_ga = group_accuracies(test_labels, product_preds, groups)
    baselines['product'] = {
        'ba': float(product_ba),
        'head': float(product_ga.get('Head', 0)),
        'med': float(product_ga.get('Med', 0)),
        'tail': float(product_ga.get('Tail', 0)),
    }
    print(f'  Product:         BA={product_ba:.2%}  '
          f'(H={product_ga.get("Head",0)*100:.1f}% '
          f'M={product_ga.get("Med",0)*100:.1f}% '
          f'T={product_ga.get("Tail",0)*100:.1f}%)')

    # ── 3. Optimal fixed weights (grid search on VAL set, evaluate on TEST) ──
    step = 0.05
    best_ba_val = 0.0
    best_weights = None
    count = 0
    for w0 in np.arange(0, 1 + step, step):
        for w1 in np.arange(0, 1 - w0 + step, step):
            w2 = 1.0 - w0 - w1
            if w2 < 0:
                continue
            w = np.array([w0, w1, w2])
            combined_val = (val_probs * w[None, :, None]).sum(axis=1)
            ba = balanced_accuracy(val_labels, combined_val.argmax(axis=1))
            count += 1
            if ba > best_ba_val:
                best_ba_val = ba
                best_weights = w.copy()

    # Evaluate best weights on test set
    combined_test = (test_probs * best_weights[None, :, None]).sum(axis=1)
    opt_ba = balanced_accuracy(test_labels, combined_test.argmax(axis=1))
    opt_ga = group_accuracies(test_labels, combined_test.argmax(axis=1), groups)
    baselines['opt_fixed'] = {
        'ba': float(opt_ba),
        'head': float(opt_ga.get('Head', 0)),
        'med': float(opt_ga.get('Med', 0)),
        'tail': float(opt_ga.get('Tail', 0)),
        'weights': [float(w) for w in best_weights],
        'val_ba': float(best_ba_val),
    }
    print(f'  Opt fixed (val): BA={opt_ba:.2%}  '
          f'weights={np.round(best_weights, 3).tolist()}  '
          f'(H={opt_ga.get("Head",0)*100:.1f}% '
          f'M={opt_ga.get("Med",0)*100:.1f}% '
          f'T={opt_ga.get("Tail",0)*100:.1f}%)')
    print(f'    (grid search on VAL set: {count} combinations, '
          f'best val BA={best_ba_val:.2%})')

    # ── 4. Oracle (upper bound) ──
    test_preds = test_logits.argmax(axis=2)
    oracle_preds = np.zeros(len(test_labels), dtype=np.int64)
    correct = (test_preds == test_labels[:, None])
    for i in range(len(test_labels)):
        c = correct[i]
        if c.any():
            oracle_preds[i] = test_preds[i, np.where(c)[0][0]]
        else:
            oracle_preds[i] = test_preds[i, 0]
    oracle_ba = balanced_accuracy(test_labels, oracle_preds)
    all_wrong_pct = (correct.sum(axis=1) == 0).mean() * 100
    baselines['oracle'] = {
        'ba': float(oracle_ba),
        'all_wrong_pct': float(all_wrong_pct),
    }
    print(f'  Oracle:          BA={oracle_ba:.2%}')
    print(f'  All-wrong:       {all_wrong_pct:.1f}%')

    # ── 5. Per-expert BAs ──
    for i, name in enumerate(EXPERT3_NAMES):
        e_ba = balanced_accuracy(test_labels, test_preds[:, i])
        e_ga = group_accuracies(test_labels, test_preds[:, i], groups)
        baselines[f'expert_{name}'] = {
            'ba': float(e_ba),
            'head': float(e_ga.get('Head', 0)),
            'med': float(e_ga.get('Med', 0)),
            'tail': float(e_ga.get('Tail', 0)),
        }
        print(f'  {name:<16} BA={e_ba:.2%}  '
              f'(H={e_ga.get("Head",0)*100:.1f}% '
              f'M={e_ga.get("Med",0)*100:.1f}% '
              f'T={e_ga.get("Tail",0)*100:.1f}%)')

    return baselines


def main():
    device = 'cpu'
    batch_size = 128

    print('=' * 60)
    print('  PHASE 1: Cache Expert Outputs + Baselines')
    print('=' * 60)

    ensure_cache_dir()

    # ── Load 3 experts ──
    print('\n[1/4] Loading 3 experts (LAL, PaCo, Mixup)...')
    all_models = load_all_experts(device=device)
    experts = {name: all_models[name] for name in EXPERT3_NAMES}
    for name in experts:
        print(f'  {name}: loaded (epoch={experts[name]._checkpoint_epoch})')

    # ── Load training class counts ──
    print('\n[2/4] Loading training set for class counts...')
    train_loader, train_counts = create_cifar_loader(
        'train', batch_size=batch_size, num_workers=0,
    )
    print(f'  Training samples: {train_counts.sum():,} '
          f'(min={train_counts.min()}, max={train_counts.max()})')

    # ── Extract and cache val + test features ──
    print('\n[3/4] Extracting and caching features...')
    val_extracted = extract_and_cache(experts, 'val', batch_size, device)
    test_extracted = extract_and_cache(experts, 'test', batch_size, device)

    # ── Compute baselines ──
    print('\n[4/4] Computing baselines...')
    baselines = compute_baselines(val_extracted, test_extracted, train_counts)

    # ── Save baselines ──
    baseline_path = os.path.join(CACHE_DIR, 'baselines.json')
    with open(baseline_path, 'w') as f:
        json.dump(baselines, f, indent=2)
    print(f'\n✅ Baselines saved to {baseline_path}')

    # ── Also save group info ──
    groups = get_class_groups(train_counts)
    group_info = {
        'head_classes': [int(c) for c in groups['Head']],
        'med_classes': [int(c) for c in groups['Med']],
        'tail_classes': [int(c) for c in groups['Tail']],
        'head_counts': [int(train_counts[c]) for c in groups['Head']],
        'med_counts': [int(train_counts[c]) for c in groups['Med']],
        'tail_counts': [int(train_counts[c]) for c in groups['Tail']],
    }
    with open(os.path.join(CACHE_DIR, 'group_info.json'), 'w') as f:
        json.dump(group_info, f, indent=2)

    print(f'\n{"=" * 60}')
    print('  PHASE 1 COMPLETE')
    print(f'  Cache: {CACHE_DIR}/')
    print(f'  Files: extracted_val.pkl, extracted_test.pkl, baselines.json')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
