#!/usr/bin/env python3
"""
Verify unverified routing hypotheses:
  H8:  Soft routing (MoE gating) with end-to-end classification loss
  H8b: Hard router with end-to-end classification loss (straight-through)
  H8c: What is the best input representation for the gate?

All experiments train on the SAME long-tailed data the experts used (fair).
"""
import os, sys, warnings, time
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

device = 'cpu'
data_root = './data'
ckpt_dir = './checkpoints'

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
models = {n: load(f'{ckpt_dir}/{n}_best.pt') for n in ['LAL','PaCo','Mixup']}

# ── Load training and validation data ────────────────────────────
base_idx = np.load(f'{data_root}/processed/base_train_indices.npy')
val_idx = np.load(f'{data_root}/processed/balanced_val_indices.npy')

# LT training set (same as experts used)
lt_train = LongTailCIFAR100(root=data_root, base_train_indices=base_idx,
    imbalance_ratio=100.0, train=False, download=False)
val_set = LongTailCIFAR100(root=data_root, base_train_indices=val_idx,
    imbalance_ratio=100.0, train=False, download=False, skip_longtail=True)

lt_loader = DataLoader(lt_train, batch_size=256, shuffle=False, num_workers=0)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

lt_class_counts = lt_train.get_class_counts()
groups = compute_class_groups(lt_class_counts)

print(f'LT train: {len(lt_train)} samples')
print(f'Val: {len(val_set)} samples')

# ── Extract features, logits, probs for all samples ─────────────
def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

@torch.no_grad()
def extract_all(model, loader, name):
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
    return {'feats': np.concatenate(feats), 'logits': np.concatenate(logits),
            'probs': softmax_np(np.concatenate(logits)),
            'preds': np.concatenate(preds), 'targets': np.concatenate(tgts)}

print('Extracting features from LT training set...')
t0 = time.time()
train_data = {}
for name in ['LAL','PaCo','Mixup']:
    train_data[name] = extract_all(models[name], lt_loader, name)
    print(f'  {name}: {time.time()-t0:.0f}s')

print('Extracting features from validation set...')
val_data = {}
for name in ['LAL','PaCo','Mixup']:
    val_data[name] = extract_all(models[name], val_loader, name)

# ── Build training matrices ──────────────────────────────────────
# Input representations to test
X_forms = {}

# 192-d concatenated backbone features
X_forms['feat192'] = np.concatenate([train_data[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)

# 300-d concatenated softmax probabilities
X_forms['prob300'] = np.concatenate([train_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=1)

# 300-d concatenated logits
X_forms['logit300'] = np.concatenate([train_data[n]['logits'] for n in ['LAL','PaCo','Mixup']], axis=1)

# 3-d max confidences
X_forms['conf3'] = np.stack([train_data[n]['probs'].max(1) for n in ['LAL','PaCo','Mixup']], axis=1)

# 3-d correctness (oracle feature — only for upper bound analysis)
# This one is cheating, just to see what's possible

y_train = train_data['LAL']['targets']  # true labels, same for all experts

# Validation matrices
X_val_forms = {
    'feat192': np.concatenate([val_data[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1),
    'prob300': np.concatenate([val_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=1),
    'logit300': np.concatenate([val_data[n]['logits'] for n in ['LAL','PaCo','Mixup']], axis=1),
    'conf3': np.stack([val_data[n]['probs'].max(1) for n in ['LAL','PaCo','Mixup']], axis=1),
}
y_val = val_data['LAL']['targets']

# Expert softmax probs on val set (needed for weighted combination)
val_expert_probs = np.stack([val_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=1)  # (5000, 3, 100)
train_expert_probs = np.stack([train_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=1)  # (9754, 3, 100)

print(f'\nTraining data: {X_forms["feat192"].shape}')
print(f'Val data: {X_val_forms["feat192"].shape}')

# ── Baselines ────────────────────────────────────────────────────
print(f'\n{"="*72}')
print('BASELINES')
print('='*72)

# Average ensemble
avg_probs = np.mean([val_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=0)
avg_ba, _ = balanced_accuracy(y_val, avg_probs.argmax(1))
avg_grp = group_accuracies(y_val, avg_probs.argmax(1), groups)
print(f'Average Ensemble: {avg_ba*100:.2f}%  (H={avg_grp["head"]*100:.1f} M={avg_grp["medium"]*100:.1f} T={avg_grp["tail"]*100:.1f})')

# Best single
for name in ['LAL','PaCo','Mixup']:
    ba, _ = balanced_accuracy(y_val, val_data[name]['preds'])
    grp = group_accuracies(y_val, val_data[name]['preds'], groups)
    print(f'{name}: {ba*100:.2f}%  (H={grp["head"]*100:.1f} M={grp["medium"]*100:.1f} T={grp["tail"]*100:.1f})')

# Oracle
correct = np.stack([val_data[n]['preds']==y_val for n in ['LAL','PaCo','Mixup']], axis=1)
oracle = correct.any(1).mean()
print(f'Oracle: {oracle*100:.2f}%')

# ── Experiment 1: Soft MoE Gating ────────────────────────────────
print(f'\n{"="*72}')
print('EXPERIMENT 1: SOFT MoE GATING (end-to-end classification loss)')
print('='*72)
print('Gate: MLP(input) → [w1,w2,w3] → weighted sum of expert probs → CrossEntropy(true_label)')

class SoftGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),
        )
    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

def train_soft_gate(input_key, hidden_dim=128, lr=1e-3, epochs=200, label=''):
    Xtr = torch.FloatTensor(X_forms[input_key])
    Xval = torch.FloatTensor(X_val_forms[input_key])
    ytr = torch.LongTensor(y_train)
    yva = torch.LongTensor(y_val)
    
    # Expert probs for combination
    probs_tr = torch.FloatTensor(train_expert_probs)  # (N, 3, 100)
    probs_va = torch.FloatTensor(val_expert_probs)     # (5000, 3, 100)
    
    gate = SoftGate(Xtr.size(1), hidden_dim)
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-4)
    
    dataset = TensorDataset(Xtr, ytr)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    gate.train()
    for epoch in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            weights = gate(bx)  # (B, 3)
            # Weighted combination of expert probs
            # probs_tr shape: (N, 3, 100), weights shape: (B, 3)
            # For each sample in batch, get its expert probs
            # This is tricky because we need to index into probs_tr
            # Let's use the batch indices to select
            # Actually, since we're using TensorDataset, let's get probs for this batch
            batch_indices = torch.tensor([i for i in range(len(dataset))])  # This won't work with shuffling
            # Simpler approach: store probs aligned with dataset order
            pass  # Will fix below
    
    # Simpler approach below
    return None

# Simpler approach: train without mini-batching (full batch since data fits in memory)
print('\nTraining soft gates (full-batch optimization)...')

results = {}

for input_key in ['feat192', 'prob300', 'logit300', 'conf3']:
    for hidden_dim in [0, 64, 128]:  # 0 = linear (no hidden)
        label = f'soft_{input_key}_h{hidden_dim}'
        
        Xtr = torch.FloatTensor(X_forms[input_key])
        Xval = torch.FloatTensor(X_val_forms[input_key])
        ytr = torch.LongTensor(y_train)
        yva = torch.LongTensor(y_val)
        
        # Get expert probs aligned with training data
        # We need to ensure proper alignment. Since we're not using DataLoader shuffling
        # during extraction, the order matches the LT dataset order.
        probs_tr = torch.FloatTensor(train_expert_probs)  # (9754, 3, 100)
        probs_va = torch.FloatTensor(val_expert_probs)     # (5000, 3, 100)
        
        if hidden_dim == 0:
            gate = nn.Linear(Xtr.size(1), 3)
        else:
            gate = nn.Sequential(
                nn.Linear(Xtr.size(1), hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 3),
            )
        
        opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Full-batch training
        gate.train()
        for epoch in range(500):
            opt.zero_grad()
            weights = F.softmax(gate(Xtr), dim=1)  # (N, 3)
            # Weighted combination: (N, 3) @ (N, 3, 100) -> (N, 100)
            # weights: (N, 3), probs_tr: (N, 3, 100)
            # We need: sum over k of weights[:,k] * probs_tr[:,k,:]
            combined = torch.einsum('nk,nkc->nc', weights, probs_tr)
            loss = F.cross_entropy(combined, ytr)
            loss.backward()
            opt.step()
        
        # Evaluate
        gate.eval()
        with torch.no_grad():
            val_weights = F.softmax(gate(Xval), dim=1)
            val_combined = torch.einsum('nk,nkc->nc', val_weights, probs_va)
            val_preds = val_combined.argmax(1).numpy()
        
        ba, _ = balanced_accuracy(y_val, val_preds)
        grp = group_accuracies(y_val, val_preds, groups)
        avg_weight = val_weights.mean(0).numpy()
        
        results[label] = {'ba': ba, 'grp': grp, 'avg_weight': avg_weight}
        
        vs_avg = (ba - avg_ba) * 100
        marker = '✅' if ba > avg_ba else ' '
        print(f'  {label:30s}: BA={ba*100:6.2f}% ({vs_avg:+.2f}% vs avg) '
              f'weights=[{avg_weight[0]:.2f},{avg_weight[1]:.2f},{avg_weight[2]:.2f}] {marker}')

# ── Experiment 2: What if gate sees oracle correctness? (Upper bound) ──
print(f'\n{"="*72}')
print('EXPERIMENT 2: ORACLE UPPER BOUND — gate sees which expert is correct')
print('='*72)

# Give the gate the correctness vector as input (this is cheating — just to see upper bound)
train_correct = np.stack([train_data[n]['preds']==y_train for n in ['LAL','PaCo','Mixup']], axis=1).astype(float)
val_correct = np.stack([val_data[n]['preds']==y_val for n in ['LAL','PaCo','Mixup']], axis=1).astype(float)

Xtr_oracle = torch.FloatTensor(np.concatenate([X_forms['feat192'], train_correct], axis=1))
Xval_oracle = torch.FloatTensor(np.concatenate([X_val_forms['feat192'], val_correct], axis=1))

gate = nn.Sequential(
    nn.Linear(195, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3),
)
opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
probs_tr = torch.FloatTensor(train_expert_probs)
probs_va = torch.FloatTensor(val_expert_probs)

gate.train()
for epoch in range(500):
    opt.zero_grad()
    weights = F.softmax(gate(Xtr_oracle), dim=1)
    combined = torch.einsum('nk,nkc->nc', weights, probs_tr)
    loss = F.cross_entropy(combined, torch.LongTensor(y_train))
    loss.backward()
    opt.step()

gate.eval()
with torch.no_grad():
    val_weights = F.softmax(gate(Xval_oracle), dim=1)
    val_combined = torch.einsum('nk,nkc->nc', val_weights, probs_va)
    val_preds = val_combined.argmax(1).numpy()

oracle_ba, _ = balanced_accuracy(y_val, val_preds)
oracle_grp = group_accuracies(y_val, val_preds, groups)
avg_w = val_weights.mean(0).numpy()
print(f'  Oracle-gated MoE          : BA={oracle_ba*100:.2f}%  weights=[{avg_w[0]:.2f},{avg_w[1]:.2f},{avg_w[2]:.2f}]')
print(f'  (This tells us the UPPER BOUND if the gate knew which expert is correct)')

# ── Experiment 3: Hard router trained end-to-end ─────────────────
print(f'\n{"="*72}')
print('EXPERIMENT 3: HARD ROUTER — trained with Gumbel-Softmax straight-through')
print('='*72)

class HardRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),
        )
    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)
        if hard:
            # Gumbel-Softmax straight-through
            return F.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            return F.softmax(logits / tau, dim=1)

for input_key in ['feat192', 'prob300', 'logit300', 'conf3']:
    label = f'hard_{input_key}'
    
    Xtr = torch.FloatTensor(X_forms[input_key])
    Xval = torch.FloatTensor(X_val_forms[input_key])
    
    router = HardRouter(Xtr.size(1), hidden_dim=128)
    opt = torch.optim.Adam(router.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Train with straight-through Gumbel-Softmax
    router.train()
    for epoch in range(500):
        opt.zero_grad()
        # Anneal temperature
        tau = max(0.5, 1.0 * (1 - epoch / 500))
        one_hot = router(Xtr, tau=tau, hard=True)  # (N, 3) one-hot
        # Combine: (N, 3) one-hot @ (N, 3, 100) -> (N, 100)
        combined = torch.einsum('nk,nkc->nc', one_hot, probs_tr)
        loss = F.cross_entropy(combined, torch.LongTensor(y_train))
        loss.backward()
        opt.step()
    
    # Evaluate (hard selection)
    router.eval()
    with torch.no_grad():
        one_hot = router(Xval, tau=0.5, hard=True)
        combined = torch.einsum('nk,nkc->nc', one_hot, probs_va)
        val_preds = combined.argmax(1).numpy()
        selected = one_hot.argmax(1).numpy()
    
    ba, _ = balanced_accuracy(y_val, val_preds)
    grp = group_accuracies(y_val, val_preds, groups)
    sel_rates = [(selected==i).mean() for i in range(3)]
    
    vs_avg = (ba - avg_ba) * 100
    marker = '✅' if ba > avg_ba else ' '
    print(f'  {label:30s}: BA={ba*100:6.2f}% ({vs_avg:+.2f}% vs avg) '
          f'sel=[{sel_rates[0]:.2f},{sel_rates[1]:.2f},{sel_rates[2]:.2f}] {marker}')

# ── Experiment 4: What if we use the PROPER training distribution? ──
# The soft gate was trained on LT data (imbalanced). What if we train on BALANCED data?
print(f'\n{"="*72}')
print('EXPERIMENT 4: SOFT GATE — trained on BALANCED data vs LT data')
print('='*72)

# Create a balanced subset from the base pool
rng = np.random.RandomState(42)
full = __import__('torchvision').datasets.CIFAR100(root=data_root, train=True, download=False)
all_targets = np.array(full.targets)

# Sample 50 per class from base pool (balanced)
balanced_idx = []
for c in range(100):
    candidates = np.where((all_targets[base_idx] == c))[0]  # indices within base_idx
    chosen = rng.choice(candidates, min(50, len(candidates)), replace=False)
    balanced_idx.extend(base_idx[chosen].tolist())
balanced_idx = np.array(balanced_idx)

# Extract features for balanced training set
from torchvision import transforms as T
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
val_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

class Subset:
    def __init__(self, indices):
        self.images = full.data[indices]
        self.targets = np.array(full.targets)[indices]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return val_transform(self.images[idx]), self.targets[idx]

bal_set = Subset(balanced_idx)
bal_loader = DataLoader(bal_set, batch_size=256, shuffle=False, num_workers=0)
print(f'Balanced training set: {len(bal_set)} samples')

# Extract features
bal_data = {}
for name in ['LAL','PaCo','Mixup']:
    bal_data[name] = extract_all(models[name], bal_loader, name)

# Build balanced training matrices
X_bal = {
    'feat192': np.concatenate([bal_data[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1),
    'prob300': np.concatenate([bal_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=1),
}
y_bal = bal_data['LAL']['targets']
bal_expert_probs = np.stack([bal_data[n]['probs'] for n in ['LAL','PaCo','Mixup']], axis=1)

# Train soft gate on balanced data
for input_key in ['feat192', 'prob300']:
    label = f'soft_{input_key}_h128_BALANCED'
    
    Xtr = torch.FloatTensor(X_bal[input_key])
    probs_tr = torch.FloatTensor(bal_expert_probs)
    
    gate = nn.Sequential(
        nn.Linear(Xtr.size(1), 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3),
    )
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    
    gate.train()
    for epoch in range(500):
        opt.zero_grad()
        weights = F.softmax(gate(Xtr), dim=1)
        combined = torch.einsum('nk,nkc->nc', weights, probs_tr)
        loss = F.cross_entropy(combined, torch.LongTensor(y_bal))
        loss.backward()
        opt.step()
    
    # Evaluate on VAL set (balanced, unseen)
    gate.eval()
    Xval = torch.FloatTensor(X_val_forms[input_key])
    with torch.no_grad():
        val_weights = F.softmax(gate(Xval), dim=1)
        val_combined = torch.einsum('nk,nkc->nc', val_weights, probs_va)
        val_preds = val_combined.argmax(1).numpy()
    
    ba, _ = balanced_accuracy(y_val, val_preds)
    grp = group_accuracies(y_val, val_preds, groups)
    avg_weight = val_weights.mean(0).numpy()
    vs_avg = (ba - avg_ba) * 100
    marker = '✅' if ba > avg_ba else ' '
    print(f'  {label:35s}: BA={ba*100:6.2f}% ({vs_avg:+.2f}% vs avg) '
          f'weights=[{avg_weight[0]:.2f},{avg_weight[1]:.2f},{avg_weight[2]:.2f}] {marker}')

# ── Summary ──────────────────────────────────────────────────────
print(f'\n{"="*72}')
print('SUMMARY')
print('='*72)
print(f'{"Method":45s} | {"BA":>8} | {"vsAvg":>8} | Notes')
print(f'{"─"*45}─|{"─"*8}─|{"─"*8}─|{"─"*20}')
print(f'{"Average Ensemble":45s} | {avg_ba*100:7.2f}% | {"":>8} | Baseline')
for label, r in sorted(results.items(), key=lambda x: x[1]['ba'], reverse=True):
    vs = (r['ba'] - avg_ba) * 100
    w = r['avg_weight']
    print(f'{label:45s} | {r["ba"]*100:7.2f}% | {vs:+7.2f}% | w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]')

# Best soft gate
if results:
    best = max(results.values(), key=lambda r: r['ba'])
    print(f'\nBest soft gate: {best["ba"]*100:.2f}% (vs avg ensemble {avg_ba*100:.2f}%)')
    if best['ba'] > avg_ba:
        print(f'✅ Soft gating BEATS averaging by +{(best["ba"]-avg_ba)*100:.2f}%')
    else:
        print(f'❌ Soft gating does NOT beat averaging')

print(f'\n── Key Insights ──')
print(f'• Oracle-gated MoE upper bound: {oracle_ba*100:.2f}%')
print(f'  (If gate knew which expert is correct, this is what it would achieve)')
print(f'• Gap from best soft gate to oracle: {oracle_ba*100 - best["ba"]*100:.2f}%' if results else '')
print(f'• This gap represents how much signal the features LOSE about expert correctness')
