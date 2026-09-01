#!/usr/bin/env python3
"""
Kaggle-compatible root cause analysis.
Run this on Kaggle with GPU enabled.

Instructions:
1. Upload the project folder as a Kaggle dataset or copy files manually
2. Set DATA_DIR and CKPT_DIR to point to the right locations
3. Run with Python 3 + GPU
"""
import os, sys, warnings, json, time, gc
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────
DATA_DIR = './data'       # Path to data/ folder (contains cifar-100-python/, processed/)
CKPT_DIR = './checkpoints'  # Path to checkpoints/
OUTPUT_FILE = './root_cause_results.json'
# ─────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root if needed
_proj_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from models.resnet32 import ResNet32, PaCoResNet32
from scripts.base_trainer import balanced_accuracy
from data.cifar_lt import LongTailCIFAR100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

def load_model(name):
    ckpt = f'{CKPT_DIR}/{name}_best.pt'
    s = torch.load(ckpt, map_location='cpu', weights_only=False)
    if s['expert_name'] == 'PaCo':
        m = PaCoResNet32(num_classes=100, dim=32, K=2048)
    else:
        m = ResNet32(num_classes=100)
    m.load_state_dict(s['model_state_dict'])
    m.to(device).eval()
    return m

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

class CIFAR100Subset:
    """Memory-efficient CIFAR-100 subset."""
    def __init__(self, indices):
        full = datasets.CIFAR100(root=DATA_DIR, train=True, download=False)
        self.images = full.data[indices]
        self.targets = np.array(full.targets)[indices]
        self.transform = val_transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.targets[idx]

# ── Load indices ─────────────────────────────────────────────────
base_idx = np.load(f'{DATA_DIR}/processed/base_train_indices.npy')
val_idx = np.load(f'{DATA_DIR}/processed/balanced_val_indices.npy')

# Validation set (5K)
val_subset = CIFAR100Subset(val_idx)
val_loader = DataLoader(val_subset, batch_size=512, shuffle=False, num_workers=2)

# Training subset: sample 3000 from the LT training set to keep memory low
# First get the actual LT training indices
lt_train = LongTailCIFAR100(root=DATA_DIR, base_train_indices=base_idx,
    imbalance_ratio=100.0, train=False, download=False)
# Get the sample_indices from the LT dataset
lt_sample_indices = lt_train.sample_indices  # indices into the full 50K training set

rng = np.random.RandomState(42)
subset_indices = rng.choice(lt_sample_indices, min(3000, len(lt_sample_indices)), replace=False)
train_subset = CIFAR100Subset(subset_indices)
train_loader = DataLoader(train_subset, batch_size=512, shuffle=False, num_workers=2)

N_train = len(train_subset)
N_val = len(val_subset)
print(f'Train: {N_train}, Val: {N_val}')

# ── Extract features (one model at a time) ───────────────────────
def extract_all(model, loader, name):
    feats, probs, targets = [], [], []
    for imgs, tgts in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            if name == 'PaCo':
                f = model.encoder_q[0](imgs).view(imgs.size(0), -1)
                lg = model.linear_q(f)
            else:
                f = model.backbone(imgs)
                lg = model.fc(f)
        feats.append(f.detach().cpu().numpy())
        probs.append(softmax_np(lg.detach().cpu().numpy()))
        targets.append(tgts.numpy())
    return {
        'feats': np.concatenate(feats).astype(np.float32),
        'probs': np.concatenate(probs).astype(np.float32),
        'targets': np.concatenate(targets),
    }

results = {}
t0 = time.time()

for name in ['LAL', 'PaCo', 'Mixup']:
    print(f'\nProcessing {name}...')
    m = load_model(name)
    tr = extract_all(m, train_loader, name)
    va = extract_all(m, val_loader, name)
    del m; gc.collect()
    results[name] = {'train': tr, 'val': va}
    print(f'  Done in {time.time()-t0:.0f}s')

# ── Build combined matrices ──────────────────────────────────────
print('\nBuilding combined matrices...')
X_tr = np.concatenate([results[n]['train']['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)
P_tr = np.stack([results[n]['train']['probs'] for n in ['LAL','PaCo','Mixup']], axis=1)
y_tr = results['LAL']['train']['targets']

X_va = np.concatenate([results[n]['val']['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)
P_va = np.stack([results[n]['val']['probs'] for n in ['LAL','PaCo','Mixup']], axis=1)
y_va = results['LAL']['val']['targets']

# Free individual arrays to save memory
for n in ['LAL','PaCo','Mixup']:
    del results[n]
gc.collect()

print(f'Train: {X_tr.shape}, Val: {X_va.shape}')

# ── COMPUTE ALL METRICS ──────────────────────────────────────────
print('\n' + '='*60)
print('COMPUTING METRICS')
print('='*60)

# 1. Uniform averaging
UBA, _ = balanced_accuracy(y_va, P_va.mean(1).argmax(1))
print(f'1. Uniform averaging:                    {UBA*100:.2f}%')

# 2. Oracle
val_correct = np.stack([P_va[:,i].argmax(1)==y_va for i in range(3)], axis=1)
oracle = val_correct.any(1).mean()
print(f'2. Oracle (any correct):                 {oracle*100:.2f}%')
irrecoverable = (1 - oracle) * 100
print(f'   Irrecoverable samples:                {irrecoverable:.1f}%')

# 3. Oracle-weighted (perfect soft routing)
sc = val_correct.sum(1, keepdims=True)
ow = np.where(sc > 0, val_correct/sc, np.ones(3)/3)
OBA, _ = balanced_accuracy(y_va, (P_va*ow.reshape(-1,3,1)).sum(1).argmax(1))
print(f'3. Oracle-weighted (perfect soft):       {OBA*100:.2f}%')
print(f'   Gain over uniform:                    +{(OBA-UBA)*100:.2f}%')

# 4. Loss-based oracle (perfect hard routing)
losses = np.stack([-np.log(P_va[:,i,y_va]+1e-12) for i in range(3)], axis=1)
LBA, _ = balanced_accuracy(y_va, P_va[np.arange(N_val), losses.argmin(1)].argmax(1))
print(f'4. Loss-based oracle (perfect hard):     {LBA*100:.2f}%')

# 5. Calibrated confidence routing
from scipy.optimize import minimize_scalar
temps = {}
for i, name in enumerate(['LAL','PaCo','Mixup']):
    logits_i = np.log(P_va[:,i,:] + 1e-12)  # approximate inverse softmax
    # Actually, we stored probs not logits. Let me compute temperature on probs directly
    def loss_fn(T):
        if T <= 0: return 1e10
        scaled = P_va[:,i,:] ** (1.0/T)
        scaled = scaled / scaled.sum(1, keepdims=True)
        return -np.log(scaled[np.arange(N_val), y_va] + 1e-12).mean()
    res = minimize_scalar(loss_fn, bounds=(0.1, 5), method='bounded')
    temps[name] = res.x

cal_p = np.stack([
    (P_va[:,0,:] ** (1.0/temps['LAL'])) / (P_va[:,0,:] ** (1.0/temps['LAL'])).sum(1, keepdims=True),
    (P_va[:,1,:] ** (1.0/temps['PaCo'])) / (P_va[:,1,:] ** (1.0/temps['PaCo'])).sum(1, keepdims=True),
    (P_va[:,2,:] ** (1.0/temps['Mixup'])) / (P_va[:,2,:] ** (1.0/temps['Mixup'])).sum(1, keepdims=True),
], axis=1)
cal_conf = cal_p.max(2)
CBA, _ = balanced_accuracy(y_va, cal_p[np.arange(N_val), cal_conf.argmax(1)].argmax(1))
print(f'5. Calibrated confidence:                {CBA*100:.2f}%')
print(f'   Temps: LAL={temps["LAL"]:.3f}, PaCo={temps["PaCo"]:.3f}, Mixup={temps["Mixup"]:.3f}')

# Selection rates
cal_sel = [(cal_conf.argmax(1)==i).mean() for i in range(3)]
print(f'   Selection: LAL={cal_sel[0]*100:.1f}% PaCo={cal_sel[1]*100:.1f}% Mixup={cal_sel[2]*100:.1f}%')

# 6. Learned soft gate (linear, fast)
print(f'\n6. Learned soft gate (training)...')
Xtr_t = torch.FloatTensor(X_tr)
Ptr_t = torch.FloatTensor(P_tr)
gate = nn.Linear(192, 3)
opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
for ep in range(500):
    opt.zero_grad()
    w = F.softmax(gate(Xtr_t), dim=1)
    loss = F.cross_entropy(torch.einsum('nk,nkc->nc', w, Ptr_t), torch.LongTensor(y_tr))
    loss.backward()
    opt.step()
gate.eval()
with torch.no_grad():
    w_va = F.softmax(gate(torch.FloatTensor(X_va)), dim=1)
    GBA, _ = balanced_accuracy(y_va, torch.einsum('nk,nkc->nc', w_va, torch.FloatTensor(P_va)).argmax(1).numpy())
    aw = w_va.mean(0).numpy()
print(f'   BA: {GBA*100:.2f}%  avg_weights=[{aw[0]:.3f},{aw[1]:.3f},{aw[2]:.3f}]')

# 7. Routing opportunity analysis
uniform_preds = P_va.mean(1).argmax(1)
uniform_wrong = uniform_preds != y_va
savable = uniform_wrong & val_correct.any(1)
all_wrong = uniform_wrong & (~val_correct.any(1))
print(f'\n7. Routing opportunity:')
print(f'   Uniform wrong:         {uniform_wrong.sum()} / {N_val} ({(uniform_wrong.sum()/N_val*100):.1f}%)')
print(f'   Could be saved:        {savable.sum()} / {N_val} ({(savable.sum()/N_val*100):.1f}%)')
print(f'   Irrecoverable:         {all_wrong.sum()} / {N_val} ({(all_wrong.sum()/N_val*100):.1f}%)')

# Best possible if we route savable samples perfectly
saved_preds = uniform_preds.copy()
savable_indices = np.where(savable)[0]
for idx in savable_indices:
    correct_experts = np.where(val_correct[idx])[0]
    saved_preds[idx] = P_va[idx, correct_experts[0]].argmax()
best_possible_ba, _ = balanced_accuracy(y_va, saved_preds)

# ── COMPUTE GAPS ──────────────────────────────────────────────────
print('\n' + '='*60)
print('ROOT CAUSE GAPS')
print('='*60)

gaps = {
    'oracle': float(oracle),
    'uniform_ba': float(UBA),
    'oracle_weighted_ba': float(OBA),
    'loss_oracle_ba': float(LBA),
    'calibrated_conf_ba': float(CBA),
    'learned_gate_ba': float(GBA),
    'best_possible_ba': float(best_possible_ba),
    'irrecoverable_pct': float(irrecoverable),
    'savable_count': int(savable.sum()),
    'model_gap_calibration': float(LBA - CBA),
    'model_gap_soft_vs_hard': float(LBA - OBA),
    'model_gap_feature_learning': float(OBA - GBA),
    'model_gap_uniform_to_best': float(best_possible_ba - UBA),
    'times': {
        'extraction_mins': (time.time() - t0) / 60,
    },
    'selection_rates': {
        'LAL': float(cal_sel[0]),
        'PaCo': float(cal_sel[1]),
        'Mixup': float(cal_sel[2]),
    },
    'temperatures': {k: float(v) for k, v in temps.items()},
    'learned_gate_weights': [float(aw[0]), float(aw[1]), float(aw[2])],
}

print(f'\n  Oracle (upper bound):                 {oracle*100:.2f}%')
print(f'  Uniform averaging:                    {UBA*100:.2f}%')
print(f'  Loss-based oracle (best possible):    {LBA*100:.2f}%')
print(f'  Oracle-weighted (perfect soft):       {OBA*100:.2f}%')
print(f'  Calibrated confidence:               {CBA*100:.2f}%')
print(f'  Learned soft gate:                    {GBA*100:.2f}%')
print(f'  Best possible (perfect on savable):   {best_possible_ba*100:.2f}%')
print()
print(f'  Gaps:')
print(f'    Miscalibration:          {gaps["model_gap_calibration"]*100:.2f}%')
print(f'    Hard vs soft routing:    {gaps["model_gap_soft_vs_hard"]*100:.2f}%')
print(f'    Feature learning:        {gaps["model_gap_feature_learning"]*100:.2f}%  ← MAIN BOTTLENECK')
print(f'    Total routing headroom:  {(LBA-UBA)*100:.2f}%')

# ── SAVE RESULTS ──────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump(gaps, f, indent=2)
print(f'\nResults saved to {OUTPUT_FILE}')
print(f'\nDone in {(time.time()-t0)/60:.1f} minutes.')
