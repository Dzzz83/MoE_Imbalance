#!/usr/bin/env python3
"""
Thorough router evaluation: compare multiple routing strategies
with proper held-out evaluation on ALL samples.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from models.resnet32 import ResNet32, PaCoResNet32
from scripts.base_trainer import balanced_accuracy, group_accuracies, compute_class_groups
from data.cifar_lt import LongTailCIFAR100

device = 'cpu'
data_root = './data'
ckpt_dir = './checkpoints'

# ── Load data ────────────────────────────────────────────────────
val_idx = np.load(f'{data_root}/processed/balanced_val_indices.npy')
val_set = LongTailCIFAR100(root=data_root, base_train_indices=val_idx,
    imbalance_ratio=100.0, train=False, download=False, skip_longtail=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
base_idx = np.load(f'{data_root}/processed/base_train_indices.npy')
train_set = LongTailCIFAR100(root=data_root, base_train_indices=base_idx,
    imbalance_ratio=100.0, train=False, download=False)
class_counts = train_set.get_class_counts()
groups = compute_class_groups(class_counts)

# ── Load models ──────────────────────────────────────────────────
def load(ckpt):
    s = torch.load(ckpt, map_location='cpu', weights_only=False)
    if s['expert_name'] == 'PaCo':
        m = PaCoResNet32(num_classes=100, dim=32, K=2048)
    else:
        m = ResNet32(num_classes=100)
    m.load_state_dict(s['model_state_dict'])
    m.to(device).eval()
    return m

models = {name: load(f'{ckpt_dir}/{name}_best.pt') for name in ['LAL','PaCo','Mixup']}

# ── Extract features + logits ────────────────────────────────────
@torch.no_grad()
def extract(model, loader, name):
    all_feats, all_logits, all_preds, all_targets = [], [], [], []
    for images, targets in loader:
        if name == 'PaCo':
            imgs = images[0] if isinstance(images, (list,tuple)) else images
        else:
            imgs = images
        imgs, targets = imgs.to(device), targets.to(device)
        if name == 'PaCo':
            feats = model.encoder_q[0](imgs)
        else:
            feats = model.backbone(imgs)
        feats = feats.view(feats.size(0), -1)
        logits = model.linear_q(feats) if name == 'PaCo' else model.fc(feats)
        all_feats.append(feats.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    return {k: np.concatenate(v) for k,v in [
        ('feats',all_feats),('logits',all_logits),('preds',all_preds),('targets',all_targets)]}

results = {name: extract(models[name], val_loader, name) for name in ['LAL','PaCo','Mixup']}
targets = results['LAL']['targets']
N = len(targets)

# Feature matrices
X_feat = np.concatenate([results[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)  # 192-d

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

# Logits (300-d)
X_logit = np.concatenate([results[n]['logits'] for n in ['LAL','PaCo','Mixup']], axis=1)

# Softmax probabilities (300-d)
X_prob = np.concatenate([softmax(r['logits']) for r in [results[n] for n in ['LAL','PaCo','Mixup']]], axis=1)

# Confidence scores (3-d)
X_conf = np.stack([softmax(r['logits']).max(1) for r in [results[n] for n in ['LAL','PaCo','Mixup']]], axis=1)

# Correctness (3-d, binary)
correct = np.stack([results[n]['preds'] == targets for n in ['LAL','PaCo','Mixup']], axis=1).astype(float)

any_correct = correct.any(axis=1)
# Target: which expert is best (for trainable samples)
y = np.full(N, -1, dtype=np.int64)
confs = np.stack([softmax(r['logits']).max(1) for r in [results[n] for n in ['LAL','PaCo','Mixup']]], axis=1)
for i in range(N):
    if any_correct[i]:
        candidates = np.where(correct[i])[0]
        y[i] = candidates[confs[i, candidates].argmax()]

train_mask = any_correct
print(f'Total samples: {N}')
print(f'Trainable (>=1 expert correct): {train_mask.sum()} ({train_mask.mean()*100:.1f}%)')
print(f'Non-trainable (all wrong): {(~train_mask).sum()} ({(~train_mask).mean()*100:.1f}%)')

# ── Multiple random splits for statistical significance ──────────
N_TRIALS = 10
np.random.seed(42)
seeds = np.random.randint(0, 10000, N_TRIALS)

trial_results = {method: [] for method in [
    'best_single', 'avg_ensemble', 'majority_vote', 'conf_routing',
    'lr_feat', 'lr_logit', 'lr_prob', 'lr_conf',
    'mlp_feat', 'mlp_feat_fb',
]}

for trial in range(N_TRIALS):
    rng = np.random.RandomState(seeds[trial])
    all_idx = np.arange(N)
    rng.shuffle(all_idx)
    split_80 = int(N * 0.8)
    tr_all = all_idx[:split_80]
    te_all = all_idx[split_80:]

    te_targets = targets[te_all]

    # ── Baselines (computed on test set) ──────────────────────────
    # Best single
    best_single_ba = 0
    for name in ['LAL','PaCo','Mixup']:
        ba, _ = balanced_accuracy(te_targets, results[name]['preds'][te_all])
        best_single_ba = max(best_single_ba, ba)

    # Average ensemble
    avg_probs = sum(softmax(r['logits']) for r in [results[n] for n in ['LAL','PaCo','Mixup']]) / 3
    avg_ba, _ = balanced_accuracy(te_targets, avg_probs[te_all].argmax(1))

    # Majority vote
    all_preds_stack = np.stack([results[n]['preds'][te_all] for n in ['LAL','PaCo','Mixup']], axis=0)
    mv_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds_stack)
    mv_ba, _ = balanced_accuracy(te_targets, mv_preds)

    # Confidence routing
    confs_te = np.stack([softmax(r['logits'])[te_all].max(1) for r in [results[n] for n in ['LAL','PaCo','Mixup']]], axis=1)
    conf_best = confs_te.argmax(1)
    conf_preds = np.array([results[['LAL','PaCo','Mixup'][conf_best[i]]]['preds'][te_all[i]] for i in range(len(te_all))])
    conf_ba, _ = balanced_accuracy(te_targets, conf_preds)

    # ── Train classifiers on train split ─────────────────────────
    tr_mask = train_mask[tr_all]
    tr_idx = tr_all[tr_mask]

    Xtr_feat = X_feat[tr_idx]
    Xtr_logit = X_logit[tr_idx]
    Xtr_prob = X_prob[tr_idx]
    Xtr_conf = X_conf[tr_idx]
    ytr = y[tr_idx]

    # Logistic Regression classifiers
    for label, Xtr in [('lr_feat', Xtr_feat), ('lr_logit', Xtr_logit),
                       ('lr_prob', Xtr_prob), ('lr_conf', Xtr_conf)]:
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced',
     solver='lbfgs')
        # Handle cases where not all classes present
        classes_present = np.unique(ytr)
        if len(classes_present) < 3:
            trial_results[label].append(0.0)
            continue
        clf.fit(Xtr, ytr)
        # Predict on all test samples
        te_preds = clf.predict(X_feat[te_all] if label == 'lr_feat' else
                               X_logit[te_all] if label == 'lr_logit' else
                               X_prob[te_all] if label == 'lr_prob' else
                               X_conf[te_all])
        # Route
        routed_preds = np.array([results[['LAL','PaCo','Mixup'][te_preds[i]]]['preds'][te_all[i]] for i in range(len(te_all))])
        ba, _ = balanced_accuracy(te_targets, routed_preds)
        trial_results[label].append(ba)

    # MLP on features (with/without fallback)
    Xtr_mlp = torch.FloatTensor(Xtr_feat)
    ytr_mlp = torch.LongTensor(ytr)
    loader = DataLoader(TensorDataset(Xtr_mlp, ytr_mlp), batch_size=64, shuffle=True)

    mlp = nn.Sequential(
        nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 3),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    mlp.train()
    for epoch in range(200):
        for bx, by in loader:
            opt.zero_grad()
            F.cross_entropy(mlp(bx), by).backward()
            opt.step()

    # Evaluate MLP
    mlp.eval()
    with torch.no_grad():
        te_logits = mlp(torch.FloatTensor(X_feat[te_all]))
        te_probs = F.softmax(te_logits, dim=1).numpy()
        te_confs = te_probs.max(1)
        te_choices = te_probs.argmax(1)

    # Hard routing
    hard_preds = np.array([results[['LAL','PaCo','Mixup'][te_choices[i]]]['preds'][te_all[i]] for i in range(len(te_all))])
    hard_ba, _ = balanced_accuracy(te_targets, hard_preds)
    trial_results['mlp_feat'].append(hard_ba)

    # With fallback
    fb_preds = np.array([
        results[['LAL','PaCo','Mixup'][te_choices[i]]]['preds'][te_all[i]]
        if te_confs[i] >= 0.6 else avg_probs[te_all[i]].argmax()
        for i in range(len(te_all))
    ])
    fb_ba, _ = balanced_accuracy(te_targets, fb_preds)
    trial_results['mlp_feat_fb'].append(fb_ba)

    # Oracle
    oracle_te = correct[te_all].any(1).mean()
    trial_results.setdefault('oracle', []).append(oracle_te)

    # Store baselines
    trial_results['best_single'].append(best_single_ba)
    trial_results['avg_ensemble'].append(avg_ba)
    trial_results['majority_vote'].append(mv_ba)
    trial_results['conf_routing'].append(conf_ba)

# ── Results ──────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('ROUTER COMPARISON (10 random 80/20 splits, mean ± std)')
print('=' * 72)
print(f'\n  {"Method":30s} | {"BA (mean±std)":>20s} | {"vs AvgEns":>10s}')
print(f'  {"─"*30}─|{"─"*20}─|{"─"*10}')

results_order = [
    ('best_single', 'Best Single Expert'),
    ('avg_ensemble', 'Average Ensemble'),
    ('majority_vote', 'Majority Vote'),
    ('conf_routing', 'Confidence Routing'),
    ('lr_conf', 'LogReg (3-d confidences)'),
    ('lr_prob', 'LogReg (300-d probs)'),
    ('lr_logit', 'LogReg (300-d logits)'),
    ('lr_feat', 'LogReg (192-d features)'),
    ('mlp_feat', 'MLP (192-d features, hard)'),
    ('mlp_feat_fb', 'MLP + Fallback (τ=0.6)'),
    ('oracle', 'Oracle (upper bound)'),
]

avg_ens_mean = np.mean(trial_results['avg_ensemble']) * 100

for key, label in results_order:
    vals = trial_results[key]
    mean = np.mean(vals) * 100
    std = np.std(vals) * 100
    vs_ens = mean - avg_ens_mean
    sig = '★' if vs_ens > 0.5 else (' ' if vs_ens > -0.5 else '↓')
    print(f'  {label:30s} | {mean:7.2f}% ± {std:4.2f}   | {vs_ens:+6.2f}% {sig}')

print(f'\n★ = beats average ensemble by >0.5%  ↓ = loses to average ensemble by >0.5%')
print(f'\n── Key Findings ──')
best_method = max([k for k in trial_results if k != 'oracle'], key=lambda k: np.mean(trial_results[k]))
best_mean = np.mean(trial_results[best_method]) * 100
avg_ens_mean_final = np.mean(trial_results['avg_ensemble']) * 100
conf_routing_mean = np.mean(trial_results['conf_routing']) * 100
conf_hurt = conf_routing_mean < avg_ens_mean_final
oracle_mean = np.mean(trial_results['oracle']) * 100

print(f'  Best method: {best_method} @ {best_mean:.2f}%')
print(f'  Average Ensemble: {avg_ens_mean_final:.2f}%')
best_beats_avg = best_mean > avg_ens_mean_final + 0.5
if best_beats_avg:
    print('  Can we beat Average Ensemble? YES')
else:
    print('  Can we beat Average Ensemble? NO - averaging is already optimal')
if conf_hurt:
    print('  Is LAL overconfidence hurting confidence routing? Yes')
else:
    print('  Is LAL overconfidence hurting confidence routing? No')
print(f'  Oracle gap: {oracle_mean:.2f}% - {avg_ens_mean_final:.2f}% = {oracle_mean - avg_ens_mean_final:.2f}%')