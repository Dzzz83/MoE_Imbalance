#!/usr/bin/env python3
"""
Systematic debug: why does routing underperform average ensemble?
Tests each hypothesis and isolates the root cause.
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

from models.resnet32 import ResNet32, PaCoResNet32
from scripts.base_trainer import balanced_accuracy, group_accuracies, compute_class_groups
from data.cifar_lt import LongTailCIFAR100
from torchvision import datasets, transforms

device = 'cpu'
data_root = './data'
ckpt_dir = './checkpoints'
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

# ── Load models ──────────────────────────────────────────────────
def load(ckpt):
    s = torch.load(ckpt, map_location='cpu', weights_only=False)
    if s['expert_name'] == 'PaCo':
        m = PaCoResNet32(num_classes=100, dim=32, K=2048)
    else:
        m = ResNet32(num_classes=100)
    m.load_state_dict(s['model_state_dict'])
    m.to(device).eval()
    return m, s['expert_name'], s.get('best_metric_val', 0)

models = {}
for name in ['LAL','PaCo','Mixup']:
    m, en, ba = load(f'{ckpt_dir}/{name}_best.pt')
    models[name] = m

# ── Load val data ────────────────────────────────────────────────
val_idx = np.load(f'{data_root}/processed/balanced_val_indices.npy')
base_idx = np.load(f'{data_root}/processed/base_train_indices.npy')
full = datasets.CIFAR100(root=data_root, train=True, download=False)

class Subset(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.images = full.data[indices]
        self.targets = np.array(full.targets)[indices]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return val_transform(self.images[idx]), self.targets[idx]

val_set = Subset(val_idx)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

# ── Feature extraction ───────────────────────────────────────────
@torch.no_grad()
def extract(model, loader, name):
    feats, logits, preds, tgts = [], [], [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        if name == 'PaCo':
            f = model.encoder_q[0](imgs).view(imgs.size(0), -1)
            lg = model.linear_q(f)
        else:
            f = model.backbone(imgs)
            lg = model.fc(f)
        feats.append(f.cpu().numpy())
        logits.append(lg.cpu().numpy())
        preds.append(lg.argmax(1).cpu().numpy())
        tgts.append(targets.numpy())
    return {k: np.concatenate(v) for k,v in [
        ('feats',feats),('logits',logits),('preds',preds),('targets',tgts)]}

results = {}
for name in ['LAL','PaCo','Mixup']:
    results[name] = extract(models[name], val_loader, name)
    ba, _ = balanced_accuracy(results[name]['targets'], results[name]['preds'])
    print(f'{name}: BA={ba*100:.2f}%')

targets = results['LAL']['targets']
N = len(targets)

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

print('\n' + '=' * 72)
print('DEBUG 1: VERIFY BASELINES')
print('=' * 72)

# Average ensemble
avg_probs = sum(softmax(results[n]['logits']) for n in ['LAL','PaCo','Mixup']) / 3
avg_preds = avg_probs.argmax(1)
avg_ba, avg_pc = balanced_accuracy(targets, avg_preds)
print(f'Average Ensemble BA: {avg_ba*100:.2f}%')

# Best single
for name in ['LAL','PaCo','Mixup']:
    ba, _ = balanced_accuracy(targets, results[name]['preds'])
    print(f'{name} BA: {ba*100:.2f}%')

# Oracle
correct = np.stack([results[n]['preds']==targets for n in ['LAL','PaCo','Mixup']], axis=1)
oracle = correct.any(1).mean()
print(f'Oracle: {oracle*100:.2f}%')
exactly_one = correct.sum(1) == 1
print(f'Exactly one expert correct: {exactly_one.mean()*100:.2f}%')
exactly_one_correct = correct[exactly_one]
print(f'  Of those, which expert: LAL={exactly_one_correct[:,0].mean()*100:.1f}% '
      f'PaCo={exactly_one_correct[:,1].mean()*100:.1f}% Mixup={exactly_one_correct[:,2].mean()*100:.1f}%')

print('\n' + '=' * 72)
print('DEBUG 2: ROUTING TARGET ANALYSIS')
print('=' * 72)

# Check how often multiple experts are correct
n_correct = correct.sum(1)
for k in range(4):
    pct = (n_correct == k).mean() * 100
    print(f'Samples with exactly {k} correct: {pct:.1f}%')

# When multiple experts are correct, does it matter which one we pick?
# Check: when 2 experts are correct, what is the gap between their confidences?
two_correct_mask = n_correct == 2
print(f'\nWhen 2 experts are correct ({two_correct_mask.sum()} samples):')
confs_arr = np.stack([softmax(results[n]['logits']).max(1) for n in ['LAL','PaCo','Mixup']], axis=1)
for i in range(two_correct_mask.sum()):
    pass  # will analyze below

# For samples with exactly 2 correct, check if the higher-confidence one is
# actually the better choice (spoiler: they're both correct, so it doesn't matter)
# But for the router training target: if both are correct, which label do we assign?
# We assign the higher-confidence one. This is ARBITRARY but not harmful (both are correct).

# The real problem: for samples with exactly 1 correct, does the target make sense?
# Target = that one correct expert. Router should learn to route to it.

# What about samples where ZERO experts are correct? No routing can help.
# These contribute to the irreducible error.

print(f'\nSamples where confidence routing picks a WRONG expert:')
confs = np.stack([softmax(r['logits']).max(1) for r in [results[n] for n in ['LAL','PaCo','Mixup']]], axis=1)
best_exp = confs.argmax(1)
conf_correct = np.array([results[['LAL','PaCo','Mixup'][best_exp[i]]]['preds'][i]==targets[i] for i in range(N)])
conf_failures = ~conf_correct
print(f'  Total failures: {conf_failures.sum()} / {N}')

# In these failures, COULD another expert have been correct?
for i in range(3):
    other_correct = correct[conf_failures][:, [j for j in range(3) if j != i]].any(1)
    expert_failures = (best_exp[conf_failures] == i)
    could_rescue = expert_failures & other_correct
    ename = ['LAL','PaCo','Mixup'][i]
    print(f'  Failures where {ename} was chosen but another was correct: '
          f'{could_rescue.sum()} / {expert_failures.sum()} failures')

# Key insight: when confidence routing fails, COULD the router have done better?
# Check if there exists ANY expert that is correct on those failures
any_correct_on_failures = correct[conf_failures].any(1)
print(f'  Failures where NO expert is correct: {(~any_correct_on_failures).sum()} '
      f'({(~any_correct_on_failures).sum()/max(conf_failures.sum(),1)*100:.1f}% of failures)')
print(f'  Failures where ANOTHER expert is correct: {any_correct_on_failures.sum()} '
      f'({any_correct_on_failures.sum()/max(conf_failures.sum(),1)*100:.1f}% of failures)')
print(f'  => These {any_correct_on_failures.sum()} samples are the MAXIMUM a perfect router could recover')

print('\n' + '=' * 72)
print('DEBUG 3: CAN FEATURES PREDICT THE CLASS LABEL?')
print('=' * 72)

# If we train a classifier on 192-d features to predict class labels,
# does it beat average ensemble?
X = np.concatenate([results[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)
print(f'Feature matrix: {X.shape}')

# Use a subset of training data for classifier training
rng = np.random.RandomState(42)
train_subset_idx = rng.choice(base_idx, 10000, replace=False)
train_subset = Subset(train_subset_idx)
train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0)

@torch.no_grad()
def extract_features(model, loader, name):
    all_f, all_t = [], []
    for imgs, tgts in loader:
        imgs = imgs.to(device)
        if name == 'PaCo':
            f = model.encoder_q[0](imgs).view(imgs.size(0), -1)
        else:
            f = model.backbone(imgs)
        all_f.append(f.cpu().numpy())
        all_t.append(tgts.numpy())
    return np.concatenate(all_f), np.concatenate(all_t)

print('Extracting features from training subset...')
X_train_list, y_train_list = [], []
for name in ['LAL','PaCo','Mixup']:
    f, t = extract_features(models[name], train_loader, name)
    X_train_list.append(f)
    y_train_list.append(t)
X_train = np.concatenate(X_train_list, axis=1)
y_train = y_train_list[0]

# Verify labels match across experts
assert all((y_train_list[i] == y_train).all() for i in range(3)), 'Label mismatch!'

print(f'Train features: {X_train.shape}')
print(f'Val features: {X.shape}')

# Train LR on combined features
print('\nTraining Logistic Regression on 192-d combined features...')
lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
lr.fit(X_train, y_train)
lr_preds = lr.predict(X)
lr_ba, lr_pc = balanced_accuracy(targets, lr_preds)
print(f'LR on 192-d combined features BA: {lr_ba*100:.2f}%')
print(f'Average Ensemble BA:              {avg_ba*100:.2f}%')
if lr_ba > avg_ba:
    print('✅ Features contain strong class signal! Routing problem is in the "which expert" framing.')
else:
    print(f'⚠️  Features limited? LR = {lr_ba*100:.2f}% vs AvgEns = {avg_ba*100:.2f}%')

# Also train LR on each individual backbone features
print('\nTraining LR on individual backbone features (64-d):')
for name in ['LAL','PaCo','Mixup']:
    Xtr_single = X_train_list[['LAL','PaCo','Mixup'].index(name)]
    Xval_single = np.concatenate([results[n]['feats'] for n in [name]], axis=1)
    lr_s = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    lr_s.fit(Xtr_single, y_train)
    ba_s, _ = balanced_accuracy(targets, lr_s.predict(Xval_single))
    print(f'  {name}: {ba_s*100:.2f}%')

# Also train on just the predicted logits/probs
print('\nTraining LR on softmax probabilities (300-d):')
X_probs = np.concatenate([softmax(results[n]['logits']) for n in ['LAL','PaCo','Mixup']], axis=1)
X_probs_train = np.concatenate([softmax(r) for r in [
    LogisticRegression(max_iter=200, C=1.0, solver='lbfgs').fit(
        np.zeros((len(y_train), 1)), y_train
    ).predict_proba(np.zeros((len(y_train), 1)))  # dummy
]], axis=1)  # This won't work, let me just use a simpler approach
# Actually let me just train on logits
print('  (skipped - would need logits from training set too)')

print('\n' + '=' * 72)
print('DEBUG 4: DOES CONFIDENCE HAVE ANY ROUTING POWER?')
print('=' * 72)

# Check: when confidence chooses the CORRECT expert, how confident is it?
# When it chooses WRONG, how confident?
conf_correct_mask = conf_correct
conf_wrong_mask = ~conf_correct

print(f'When confidence routing is CORRECT (n={conf_correct_mask.sum()}):')
print(f'  Mean confidence of chosen expert: {confs[conf_correct_mask, best_exp[conf_correct_mask]].mean():.4f}')
print(f'When confidence routing is WRONG (n={conf_wrong_mask.sum()}):')
print(f'  Mean confidence of chosen expert: {confs[conf_wrong_mask, best_exp[conf_wrong_mask]].mean():.4f}')

# Is there a threshold that would improve things?
print(f'\nConfidence threshold analysis:')
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = confs.max(1) >= thresh
    if mask.sum() == 0:
        continue
    # On these samples, use confidence routing
    sub_correct = conf_correct[mask]
    # On samples below threshold, use average ensemble
    ensemble_correct = (avg_preds[~mask] == targets[~mask])
    total_correct = sub_correct.sum() + ensemble_correct.sum()
    total_ba = total_correct / N
    print(f'  Conf≥{thresh:.1f} ({mask.mean()*100:.0f}% samples): BA={total_ba*100:.2f}%')

# What if we ONLY use confidence routing on samples where exactly one expert is correct?
# (Ideal scenario for routing)
one_correct_mask = exactly_one
one_correct_routed = conf_correct[one_correct_mask]
print(f'\nOn samples with exactly ONE correct expert ({one_correct_mask.sum()} samples):')
print(f'  Confidence routing picks the correct expert: {one_correct_routed.mean()*100:.1f}%')

# What about samples where confidence routing picks wrong but another expert is correct?
# This is the recoverable failure mode
recoverable = conf_failures & correct.any(1)
print(f'\nRecoverable failures (conf routing wrong, but some expert correct):')
print(f'  Count: {recoverable.sum()} samples')
# In these cases, which expert SHOULD have been chosen?
for i, name in enumerate(['LAL','PaCo','Mixup']):
    would_help = recoverable & correct[:, i] & (best_exp != i)
    print(f'  Choosing {name} would help on: {would_help.sum()} samples')

print('\n' + '=' * 72)
print('DEBUG 5: CALIBRATION CHECK')
print('=' * 72)

def compute_ece(confs, correct_bool, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = correct_bool[mask].mean()
        ece += abs(avg_acc - avg_conf) * mask.mean()
    return ece

for name in ['LAL','PaCo','Mixup']:
    probs = softmax(results[name]['logits'])
    confs_i = probs.max(1)
    correct_i = results[name]['preds'] == targets
    ece = compute_ece(confs_i, correct_i)
    avg_conf = confs_i.mean()
    avg_acc = correct_i.mean()
    print(f'{name}: avg_conf={avg_conf:.4f}, avg_acc={avg_acc:.4f}, ECE={ece:.4f}, '
          f'conf_wrong={confs_i[~correct_i].mean():.4f}')

# Temperature scaling for LAL
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

def temperature_loss(T):
    if T <= 0:
        return 1e10
    scaled = results['LAL']['logits'] / T
    probs_scaled = softmax(scaled)
    # NLL
    nll = 0
    for i in range(N):
        nll -= np.log(probs_scaled[i, targets[i]] + 1e-12)
    return nll / N

print('\nFinding optimal temperature for LAL...')
result = minimize_scalar(temperature_loss, bounds=(0.1, 10), method='bounded')
T_lal = result.x
print(f'Optimal T for LAL: {T_lal:.4f}')

# After calibration
lal_calibrated = results['LAL']['logits'] / T_lal
lal_cal_probs = softmax(lal_calibrated)
lal_cal_conf = lal_cal_probs.max(1)
lal_cal_ece = compute_ece(lal_cal_conf, results['LAL']['preds'] == targets)
print(f'After calibration: avg_conf={lal_cal_conf.mean():.4f}, ECE={lal_cal_ece:.4f}')

# Now re-evaluate confidence routing with CALIBRATED LAL
cal_confs = np.stack([
    lal_cal_conf,  # calibrated LAL
    softmax(results['PaCo']['logits']).max(1),  # PaCo (maybe also calibrate?)
    softmax(results['Mixup']['logits']).max(1),  # Mixup
], axis=1)
cal_best_exp = cal_confs.argmax(1)
cal_correct = np.array([results[['LAL','PaCo','Mixup'][cal_best_exp[i]]]['preds'][i]==targets[i] for i in range(N)])
cal_ba, _ = balanced_accuracy(targets, 
    np.array([results[['LAL','PaCo','Mixup'][cal_best_exp[i]]]['preds'][i] for i in range(N)]))
print(f'\nConfidence routing with CALIBRATED LAL:')
print(f'  BA: {cal_ba*100:.2f}%  (vs original {conf_correct.mean()*100:.2f}%)')
print(f'  Selection: LAL={(cal_best_exp==0).mean()*100:.1f}%  '
      f'PaCo={(cal_best_exp==1).mean()*100:.1f}%  Mixup={(cal_best_exp==2).mean()*100:.1f}%')
cal_recoverable = ~cal_correct & correct.any(1)
print(f'  Recoverable failures remaining: {cal_recoverable.sum()}')

# Also calibrate PaCo and Mixup
print('\nCalibrating all experts...')
temperatures = {}
for name in ['LAL','PaCo','Mixup']:
    def loss_fn(T):
        if T <= 0: return 1e10
        scaled = results[name]['logits'] / T
        probs_scaled = softmax(scaled)
        nll = sum(-np.log(probs_scaled[i, targets[i]] + 1e-12) for i in range(N)) / N
        return nll
    res = minimize_scalar(loss_fn, bounds=(0.1, 10), method='bounded')
    temperatures[name] = res.x
    print(f'  {name}: T={res.x:.4f}')

# Re-evaluate with all calibrated
all_cal_confs = np.stack([
    softmax(results['LAL']['logits'] / temperatures['LAL']).max(1),
    softmax(results['PaCo']['logits'] / temperatures['PaCo']).max(1),
    softmax(results['Mixup']['logits'] / temperatures['Mixup']).max(1),
], axis=1)
all_cal_best = all_cal_confs.argmax(1)
all_cal_preds = np.array([results[['LAL','PaCo','Mixup'][all_cal_best[i]]]['preds'][i] for i in range(N)])
all_cal_ba, _ = balanced_accuracy(targets, all_cal_preds)
print(f'\nConfidence routing with ALL calibrated: BA={all_cal_ba*100:.2f}%')
print(f'  Selection: LAL={(all_cal_best==0).mean()*100:.1f}%  '
      f'PaCo={(all_cal_best==1).mean()*100:.1f}%  Mixup={(all_cal_best==2).mean()*100:.1f}%')

print('\n' + '=' * 72)
print('SUMMARY')
print('=' * 72)
print(f'Average Ensemble:     {avg_ba*100:.2f}%')
print(f'Confidence (raw):     {conf_correct.mean()*100:.2f}%')
print(f'Confidence (cal LAL): {cal_ba*100:.2f}%')
print(f'Confidence (all cal): {all_cal_ba*100:.2f}%')
print(f'LR on features:       {lr_ba*100:.2f}%')
print(f'Oracle:               {oracle*100:.2f}%')
