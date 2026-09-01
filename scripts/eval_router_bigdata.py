#!/usr/bin/env python3
"""
Train router on ALL 45K base training images (instead of just 5K val set).
Evaluate on the 5K held-out validation set.
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

# ── Load 5K val set (for FINAL evaluation only) ──────────────────
val_idx = np.load(f'{data_root}/processed/balanced_val_indices.npy')
# We need the ORIGINAL CIFAR-100 training images, NOT the long-tailed subset.
# Create a dataset that returns ALL training images with their transforms.
from torchvision import datasets, transforms
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# Load all 50K training images
full_train = datasets.CIFAR100(root=data_root, train=True, download=False)
all_images = full_train.data  # ndarray (50000, 32, 32, 3)
all_targets = np.array(full_train.targets)  # (50000,)

# Split into base train (45K) and val (5K)
base_train_indices = np.load(f'{data_root}/processed/base_train_indices.npy')
val_indices = np.load(f'{data_root}/processed/balanced_val_indices.npy')

# ── Load models ──────────────────────────────────────────────────
def load(ckpt):
    s = torch.load(ckpt, map_location='cpu', weights_only=False)
    if s['expert_name'] == 'PaCo':
        m = PaCoResNet32(num_classes=100, dim=32, K=2048)
    else:
        m = ResNet32(num_classes=100)
    m.load_state_dict(s['model_state_dict'])
    m.to(device).eval()
    return m, s['expert_name']

models = {}
for name in ['LAL','PaCo','Mixup']:
    m, _ = load(f'{ckpt_dir}/{name}_best.pt')
    models[name] = m

# ── Inference helper ─────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

@torch.no_grad()
def infer_on_indices(model, name, indices, batch_size=256):
    """Run inference on specific indices from the full CIFAR-100 training set."""
    n = len(indices)
    all_feats = np.zeros((n, 64), dtype=np.float32)
    all_logits = np.zeros((n, 100), dtype=np.float32)
    all_preds = np.zeros(n, dtype=np.int64)
    all_targets = np.zeros(n, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        # Load raw images and convert to tensor
        batch_imgs = []
        for idx in batch_idx:
            img_pil = datasets.CIFAR100(root=data_root, train=True, download=False).data[idx]
            img_t = val_transform(img_pil)
            batch_imgs.append(img_t)
        imgs = torch.stack(batch_imgs).to(device)
        tgts = torch.tensor(all_targets[batch_idx].tolist() if False else [all_targets[i] for i in batch_idx]).to(device)
        # Actually just get targets directly
        # Hmm, let me simplify using a simple approach
    return None  # placeholder

# Because of the complexity above, let me use a simpler approach:
# Create a proper dataset

class CIFAR100Subset(torch.utils.data.Dataset):
    """A subset of CIFAR-100 by index."""
    def __init__(self, indices, transform=None):
        full = datasets.CIFAR100(root=data_root, train=True, download=False)
        self.images = full.data[indices]
        self.targets = np.array(full.targets)[indices]
        self.transform = transform or val_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        return img, self.targets[idx]

# Create datasets
base_train_set = CIFAR100Subset(base_train_indices)
val_set = CIFAR100Subset(val_indices)

base_loader = DataLoader(base_train_set, batch_size=256, shuffle=False, num_workers=0)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

print(f'Base train: {len(base_train_set)} samples')
print(f'Val: {len(val_set)} samples')

# ── Run inference on ALL 45K base train images ──────────────────
print('\nRunning inference on 45K base train images...')
@torch.no_grad()
def extract_all(model, loader, name):
    all_feats, all_logits, all_preds, all_targets = [], [], [], []
    for images, targets in loader:
        if name == 'PaCo':
            # PaCo expects two-view input during training, but for inference
            # we pass single image through the backbone directly
            feats = model.encoder_q[0](images.to(device))
            feats = feats.view(feats.size(0), -1)
            logits = model.linear_q(feats)
        else:
            feats = model.backbone(images.to(device))
            logits = model.fc(feats)
        all_feats.append(feats.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(targets.numpy())
    return {k: np.concatenate(v) for k,v in [
        ('feats',all_feats),('logits',all_logits),('preds',all_preds),('targets',all_targets)]}

t0 = time.time()
base_results = {}
for name in ['LAL','PaCo','Mixup']:
    print(f'  {name}...', end=' ', flush=True)
    base_results[name] = extract_all(models[name], base_loader, name)
    print(f'done ({time.time()-t0:.0f}s)')

print(f'Inference on 45K complete in {time.time()-t0:.0f}s')

# ── Also run on val set ──────────────────────────────────────────
print('\nRunning inference on 5K val set...')
val_results = {}
for name in ['LAL','PaCo','Mixup']:
    val_results[name] = extract_all(models[name], val_loader, name)

val_targets = val_results['LAL']['targets']

# ── Check oracle on base train ──────────────────────────────────
base_targets = base_results['LAL']['targets']
base_correct = np.stack([base_results[n]['preds'] == base_targets for n in ['LAL','PaCo','Mixup']], axis=1)
base_any_correct = base_correct.any(axis=1)
base_oracle = base_any_correct.mean()
print(f'\nBase train oracle: {base_oracle*100:.2f}%')
print(f'Base train usable samples: {base_any_correct.sum()} / {len(base_targets)}')

# ── Build training data for router ──────────────────────────────
X_base = np.concatenate([base_results[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)  # (45000, 192)
X_val = np.concatenate([val_results[n]['feats'] for n in ['LAL','PaCo','Mixup']], axis=1)  # (5000, 192)

# Routing target (use conf for tie-breaking)
base_confs = np.stack([softmax(base_results[n]['logits']).max(1) for n in ['LAL','PaCo','Mixup']], axis=1)
y_base = np.full(len(base_targets), -1, dtype=np.int64)
for i in range(len(base_targets)):
    if base_any_correct[i]:
        candidates = np.where(base_correct[i])[0]
        y_base[i] = candidates[base_confs[i, candidates].argmax()]

# Only use trainable samples for training
train_mask = base_any_correct
X_tr = X_base[train_mask]
y_tr = y_base[train_mask]
print(f'Router training samples: {len(X_tr)}')

# ── Train MLP Router on BIG data ────────────────────────────────
X_tr_t = torch.FloatTensor(X_tr)
y_tr_t = torch.LongTensor(y_tr)
loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

mlp = nn.Sequential(
    nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 3),
)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

mlp.train()
for epoch in range(100):
    for bx, by in loader:
        opt.zero_grad()
        F.cross_entropy(mlp(bx), by).backward()
        opt.step()

# ── Evaluate on VAL set (completely unseen during router training) ──
mlp.eval()
with torch.no_grad():
    val_logits = mlp(torch.FloatTensor(X_val))
    val_probs = F.softmax(val_logits, dim=1).numpy()
    val_confs = val_probs.max(1)
    val_choices = val_probs.argmax(1)

names = ['LAL','PaCo','Mixup']

# Hard routing
hard_preds = np.array([val_results[names[val_choices[i]]]['preds'][i] for i in range(len(val_targets))])
hard_ba, _ = balanced_accuracy(val_targets, hard_preds)
# Compute class groups from LONG-TAIL training set distribution
from data.cifar_lt import LongTailCIFAR100
lt_train = LongTailCIFAR100(root=data_root, base_train_indices=base_train_indices,
    imbalance_ratio=100.0, train=False, download=False)
lt_class_counts = lt_train.get_class_counts()
groups = compute_class_groups(lt_class_counts)
hard_grp = group_accuracies(val_targets, hard_preds, groups)

# With fallback
avg_probs = sum(softmax(vr['logits']) for vr in [val_results[n] for n in ['LAL','PaCo','Mixup']]) / 3
fb_preds = np.array([
    val_results[names[val_choices[i]]]['preds'][i] if val_confs[i] >= 0.6 else avg_probs[i].argmax()
    for i in range(len(val_targets))
])
fb_ba, _ = balanced_accuracy(val_targets, fb_preds)
fb_grp = group_accuracies(val_targets, fb_preds, groups)

# ── Baselines on val set ─────────────────────────────────────────
# Average ensemble
avg_ba, _ = balanced_accuracy(val_targets, avg_probs.argmax(1))
# Confidence routing
val_confs_all = np.stack([softmax(vr['logits']).max(1) for vr in [val_results[n] for n in ['LAL','PaCo','Mixup']]], axis=1)
conf_best = val_confs_all.argmax(1)
conf_preds = np.array([val_results[['LAL','PaCo','Mixup'][conf_best[i]]]['preds'][i] for i in range(len(val_targets))])
conf_ba, _ = balanced_accuracy(val_targets, conf_preds)
# Best single
best_ba = 0
for name in ['LAL','PaCo','Mixup']:
    ba, _ = balanced_accuracy(val_targets, val_results[name]['preds'])
    best_ba = max(best_ba, ba)
# Oracle
oracle = np.stack([vr['preds']==val_targets for vr in [val_results[n] for n in ['LAL','PaCo','Mixup']]], axis=1).any(1).mean()

# ── Also try Logistic Regression on the big data ─────────────────
# No class_weight to avoid sklearn issues
lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
lr.fit(X_tr, y_tr)
lr_preds_val = lr.predict(X_val)
lr_hard_preds = np.array([val_results[names[lr_preds_val[i]]]['preds'][i] for i in range(len(val_targets))])
lr_ba, _ = balanced_accuracy(val_targets, lr_hard_preds)

# ── Print comparison ─────────────────────────────────────────────
print('\n' + '=' * 72)
print('ROUTER EVALUATION — Trained on 45K, Tested on 5K Val')
print('=' * 72)

# Get class groups properly
base_cts = np.load(f'{data_root}/processed/base_train_indices.npy')
# We need class counts for grouping. Use the val targets.
# All groups computed from LT class counts above
avg_grp = group_accuracies(val_targets, avg_probs.argmax(1), groups)
conf_grp = group_accuracies(val_targets, conf_preds, groups)

print(f'\n  {"Method":35s} | {"BA":>8} | {"Head":>8} | {"Med":>8} | {"Tail":>8}')
print(f'  {"─"*35}─|{"─"*8}─|{"─"*8}─|{"─"*8}─|{"─"*8}')
methods = [
    ('Best Single', best_ba, {}),
    ('Average Ensemble', avg_ba, avg_grp),
    ('Confidence Routing', conf_ba, conf_grp),
    ('LogReg (192-d)', lr_ba, {}),
    ('MLP Router (hard)', hard_ba, hard_grp),
    ('MLP + Fallback', fb_ba, fb_grp),
    ('Oracle', oracle, {}),
]
for label, ba, grp in methods:
    h = grp.get('head', 0.0)
    m = grp.get('medium', 0.0)
    t = grp.get('tail', 0.0)
    print(f'  {label:35s} | {ba*100:7.2f}% | {h*100:7.2f}% | {m*100:7.2f}% | {t*100:7.2f}%')

# Selection rates
sel = [(val_choices == i).mean() for i in range(3)]
print(f'\n  MLP Selection: LAL={sel[0]*100:.1f}%  PaCo={sel[1]*100:.1f}%  Mixup={sel[2]*100:.1f}%')
correct_when = [
    (val_results[n]['preds'][val_choices==i] == val_targets[val_choices==i]).mean()
    if (val_choices==i).sum() > 0 else 0.0
    for i, n in enumerate(names)
]
print(f'  Correct when selected: LAL={correct_when[0]*100:.1f}%  PaCo={correct_when[1]*100:.1f}%  Mixup={correct_when[2]*100:.1f}%')

print(f'\n  Improvement over Avg Ensemble: +{fb_ba*100 - avg_ba*100:.2f}%' if fb_ba > avg_ba else f'\n  Still below Avg Ensemble: {fb_ba*100 - avg_ba*100:.2f}%')
print(f'  Oracle gap captured: {(max(fb_ba, lr_ba, hard_ba) - best_ba) / (oracle - best_ba) * 100:.0f}%')
