#!/usr/bin/env python3
"""Lightweight diagnostic: why is loss-based oracle only 2%?"""
import os, sys
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path: sys.path.insert(0, _proj_root)
import numpy as np
from scripts.base_trainer import balanced_accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# Quick check: load a small batch and compute
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

# Load just 500 samples
val_idx = np.load('./data/processed/balanced_val_indices.npy')[:200]  # just 200 samples
full = datasets.CIFAR100(root='./data', train=True, download=False)
class Subset:
    def __init__(self, idx):
        self.images = full.data[idx]; self.targets = np.array(full.targets)[idx]
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return val_transform(self.images[i]), self.targets[i]

loader = DataLoader(Subset(val_idx), batch_size=256, shuffle=False, num_workers=0)

# Load models
from models.resnet32 import ResNet32, PaCoResNet32
device = 'cpu'
def load(name):
    s = torch.load(f'./checkpoints/{name}_best.pt', map_location='cpu', weights_only=False)
    m = PaCoResNet32(num_classes=100, dim=32, K=2048) if s['expert_name']=='PaCo' else ResNet32(num_classes=100)
    m.load_state_dict(s['model_state_dict']); m.to(device).eval(); return m

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

# Extract (one model at a time, free memory)
probs_list = []
for name in ['LAL','PaCo','Mixup']:
    m = load(name)
    p, t = [], []
    for imgs, tgts in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            if name=='PaCo':
                f = m.encoder_q[0](imgs).view(imgs.size(0),-1)
                lg = m.linear_q(f)
            else:
                f = m.backbone(imgs); lg = m.fc(f)
        p.append(softmax_np(lg.cpu().numpy()))
        t.append(tgts.numpy())
    probs_list.append(np.concatenate(p))
    targets = np.concatenate(t)
    del m, p, t
    print(f'  {name} done')

P = np.stack(probs_list, axis=1)  # (500, 3, 100)
y = targets
N = len(y)

print(f'Samples: {N}')

# 1. Uniform baseline
uba, _ = balanced_accuracy(y, P.mean(1).argmax(1))
print(f'Uniform: {uba*100:.2f}%')

# 2. Accuracy oracle (any correct)
correct = np.stack([P[:,i].argmax(1)==y for i in range(3)], axis=1)
oracle = correct.any(1).mean()
print(f'Oracle: {oracle*100:.2f}%')

# 3. Loss-based oracle: pick expert with min -log(P[true_class])
tc_probs = np.stack([P[np.arange(N), i, y] for i in range(3)], axis=1)  # (N, 3) true-class probs
losses = -np.log(tc_probs + 1e-12)  # (N, 3)
best_exp = losses.argmin(1)  # expert with lowest loss

# For each sample, what is the best expert's prediction?
best_preds = P[np.arange(N), best_exp].argmax(1)
lba, _ = balanced_accuracy(y, best_preds)
print(f'Loss-based oracle: {lba*100:.2f}%')

# Debug: on samples where uniform is correct, what does loss-oracle do?
# On samples where uniform is WRONG, what does loss-oracle do?
uniform_preds = P.mean(1).argmax(1)

# Check: what fraction of the time does the best-expert (by loss) match
# a correct expert?
correct_by_loss = correct[np.arange(N), best_exp]
print(f'  Best-expert-by-loss is CORRECT: {correct_by_loss.mean()*100:.2f}%')

# Per-expert true-class probabilities
for i, name in enumerate(['LAL','PaCo','Mixup']):
    print(f'  {name}: avg true-class prob = {tc_probs[:,i].mean():.4f}, '
          f'corr={correct[:,i].mean()*100:.2f}%')

# Check: how often does LAL have the highest true-class prob when it's WRONG?
lal_wrong = ~correct[:,0]
lal_highest_tc = tc_probs.argmax(1) == 0
lal_wrong_but_highest = lal_wrong & lal_highest_tc
print(f'\nLAL is wrong AND has highest true-class prob: {lal_wrong_but_highest.sum()}/{N} ({lal_wrong_but_highest.mean()*100:.1f}%)')

# When LAL is wrong, what's its avg true-class prob vs PaCo's?
for i, name in enumerate(['LAL','PaCo','Mixup']):
    if lal_wrong.sum() > 0:
        print(f'  When LAL wrong: {name} avg true-class prob = {tc_probs[lal_wrong,i].mean():.4f}')

# Check: on samples where expert i is correct, what's their true-class prob?
for i, name in enumerate(['LAL','PaCo','Mixup']):
    mask = correct[:,i]
    if mask.sum() > 0:
        print(f'  When {name} CORRECT: avg true-class prob = {tc_probs[mask,i].mean():.4f}')
    mask_w = ~correct[:,i]
    if mask_w.sum() > 0:
        print(f'  When {name} WRONG:   avg true-class prob = {tc_probs[mask_w,i].mean():.4f}')

# The simple question: is the expert with the highest true-class prob
# also a correct expert?
correct_any = correct.any(1)
highest_is_correct = correct[np.arange(N), tc_probs.argmax(1)]
print(f'\nExpert with highest true-class prob IS correct: {highest_is_correct.mean()*100:.2f}%')
print(f'Expert with highest true-class prob IS correct (only where any correct): '
      f'{highest_is_correct[correct_any].mean()*100:.2f}%')

# What if we do "pick expert i where correct[i]"? That's the accuracy oracle.
# The loss oracle approximates this. How often does it match?
for i in range(3):
    match = (best_exp == i) & correct[:,i]
    print(f'  Loss oracle picks {["LAL","PaCo","Mixup"][i]} AND it is correct: {match.sum()}/{N}')
