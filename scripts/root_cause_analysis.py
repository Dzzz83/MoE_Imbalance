#!/usr/bin/env python3
"""
Root cause analysis: measure each gap in the routing pipeline to pinpoint
exactly where the signal is lost.
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
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from scripts.base_trainer import balanced_accuracy
from data.cifar_lt import LongTailCIFAR100
from models.resnet32 import ResNet32, PaCoResNet32

device = 'cpu'
data_root = './data'
ckpt_dir = './checkpoints'

# ── Load ─────────────────────────────────────────────────────────
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

base_idx = np.load(f'{data_root}/processed/base_train_indices.npy')
val_idx = np.load(f'{data_root}/processed/balanced_val_indices.npy')
lt_train = LongTailCIFAR100(root=data_root, base_train_indices=base_idx,
    imbalance_ratio=100.0, train=False, download=False)
val_set = LongTailCIFAR100(root=data_root, base_train_indices=val_idx,
    imbalance_ratio=100.0, train=False, download=False, skip_longtail=True)
lt_loader = DataLoader(lt_train, batch_size=256, shuffle=False, num_workers=0)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

@torch.no_grad()
def extract(model, loader, name):
    l, t = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        if name == 'PaCo':
            f = model.encoder_q[0](imgs).view(imgs.size(0),-1)
            lg = model.linear_q(f)
        else:
            f = model.backbone(imgs)
            lg = model.fc(f)
        l.append(lg.cpu().numpy())
        t.append(targets.numpy())
    return np.concatenate(l), np.concatenate(t)

names = ['LAL','PaCo','Mixup']
tr_logits, tr_t = {}, None
va_logits, va_t = {}, None
for n in names:
    l,t = extract(models[n], lt_loader, n)
    tr_logits[n] = l
    if tr_t is None: tr_t = t
    l2,t2 = extract(models[n], val_loader, n)
    va_logits[n] = l2
    if va_t is None: va_t = t2

ptr = np.stack([softmax_np(tr_logits[n]) for n in names], axis=1)  # (N_tr, 3, 100)
pva = np.stack([softmax_np(va_logits[n]) for n in names], axis=1)  # (N_va, 3, 100)

@torch.no_grad()
def get_feats(model, loader, name):
    f,t = [], []
    for imgs, tgts in loader:
        imgs = imgs.to(device)
        if name == 'PaCo':
            feat = model.encoder_q[0](imgs).view(imgs.size(0),-1)
        else:
            feat = model.backbone(imgs)
        f.append(feat.cpu().numpy())
        t.append(tgts.numpy())
    return np.concatenate(f), np.concatenate(t)

tr_feats = np.concatenate([get_feats(models[n], lt_loader, n)[0] for n in names], axis=1)
va_feats = np.concatenate([get_feats(models[n], val_loader, n)[0] for n in names], axis=1)

# ── PIPELINE GAP ANALYSIS ───────────────────────────────────────
print('=' * 65)
print('ROUTING PIPELINE — GAP ANALYSIS')
print('=' * 65)

# GAP 0: What is the best possible?
# If we knew exactly which expert is correct, we'd achieve oracle
n_va = len(va_t)
val_correct = np.stack([pva[:,i].argmax(1)==va_t for i in range(3)], axis=1)
oracle_acc = val_correct.any(1).mean()
print(f'\nGAP 0: Perfect oracle:                              {oracle_acc*100:.2f}%')

# GAP 1: Uniform averaging
uba, _ = balanced_accuracy(va_t, pva.mean(1).argmax(1))
print(f'GAP 1: Uniform averaging:                           {uba*100:.2f}%')
print(f'       Gap from oracle:                             {oracle_acc*100 - uba*100:.2f}%')

# GAP 2: Optimal FIXED weights (no per-sample routing)
bba, bw = 0, None
for w1 in np.linspace(0,1,21):
    for w2 in np.linspace(0,1-w1,21):
        w3 = 1-w1-w2
        if w3 < 0: continue
        w = np.array([w1,w2,w3])
        ba, _ = balanced_accuracy(va_t, (pva*w.reshape(1,3,1)).sum(1).argmax(1))
        if ba > bba:
            bba, bw = ba, w
print(f'\nGAP 2: Optimal fixed weights [{bw[0]:.3f},{bw[1]:.3f},{bw[2]:.3f}]:   {bba*100:.2f}%')
print(f'       Improvement from uniform:                    +{(bba-uba)*100:.2f}%')
print(f'       Residual gap to oracle:                      {oracle_acc*100 - bba*100:.2f}%')
print(f'       => This is the MAXIMUM gain from per-sample routing')

# GAP 3: Oracle-weighted (perfect routing — but soft weights)
sc = val_correct.sum(1, keepdims=True)
ow = np.where(sc > 0, val_correct/sc, np.ones(3)/3)
oba, _ = balanced_accuracy(va_t, (pva*ow.reshape(-1,3,1)).sum(1).argmax(1))
print(f'\nGAP 3: Oracle-weighted (perfect soft routing):      {oba*100:.2f}%')
print(f'       Gain from per-sample routing:                 +{(oba-bba)*100:.2f}%')
print(f'       Gap to oracle:                                {oracle_acc*100 - oba*100:.2f}%')

# GAP 4: Loss-based oracle (pick expert with lowest CE loss)
losses = np.stack([-np.log(pva[:,i,va_t]+1e-12) for i in range(3)], axis=1)
lba, _ = balanced_accuracy(va_t, pva[np.arange(n_va), losses.argmin(1)].argmax(1))
print(f'\nGAP 4: Loss-based oracle (perfect hard routing):    {lba*100:.2f}%')
print(f'       Gain from hard routing over soft:             +{(lba-oba)*100:.2f}%')

# GAP 5: What if we had PERFECT confidence (no miscalibration)?
# Use the correct label's probability as "confidence" — perfect calibration
oracle_confs = np.stack([pva[:,i,va_t] for i in range(3)], axis=1)  # prob of TRUE class
best_oracle_conf = oracle_confs.argmax(1)
oca, _ = balanced_accuracy(va_t, pva[np.arange(n_va), best_oracle_conf].argmax(1))
print(f'\nGAP 5: Perfect-confidence routing (no miscalib):    {oca*100:.2f}%')
print(f'       (Uses true-class probability as confidence)')
print(f'       Loss from miscalibration:                     +{(lba-oca)*100:.2f}%' if lba > oca else f'       (unexpected)')

# GAP 6: Calibrated confidence routing
temps = {'LAL':1.8528, 'PaCo':1.3335, 'Mixup':1.2574}
cal_p = np.stack([softmax_np(va_logits[n]/temps[n]) for n in names], axis=1)
cal_c = cal_p.max(2)
cba, _ = balanced_accuracy(va_t, cal_p[np.arange(n_va), cal_c.argmax(1)].argmax(1))
print(f'\nGAP 6: Calibrated confidence routing:               {cba*100:.2f}%')
print(f'       Gain from calibration:                        +{(cba-uba)*100:.2f}%' if cba > uba else f'       Below uniform: {cba*100:.2f}%')
print(f'       Residual gap to loss-oracle:                  +{(lba-cba)*100:.2f}%')

# GAP 7: Learned soft gate (linear, KL regularized)
Xtr = torch.FloatTensor(tr_feats)
Ptr = torch.FloatTensor(ptr)
gate = nn.Linear(192, 3)
opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
for ep in range(1000):
    opt.zero_grad()
    w = F.softmax(gate(Xtr), dim=1)
    comb = torch.einsum('nk,nkc->nc', w, Ptr)
    loss = F.cross_entropy(comb, torch.LongTensor(tr_t))
    kl = (w * torch.log(w*3+1e-12)).mean()
    (loss + 0.01*kl).backward()
    opt.step()
gate.eval()
with torch.no_grad():
    w = F.softmax(gate(torch.FloatTensor(va_feats)), dim=1)
    sba, _ = balanced_accuracy(va_t, torch.einsum('nk,nkc->nc', w, torch.FloatTensor(pva)).argmax(1).numpy())
    aw = w.mean(0).numpy()
print(f'\nGAP 7: Learned soft gate (linear, KL):              {sba*100:.2f}%  w=[{aw[0]:.3f},{aw[1]:.3f},{aw[2]:.3f}]')
print(f'       Gap to oracle-weighted:                       +{(oba-sba)*100:.2f}%')
print(f'       => This is the signal LOST by learning from features vs knowing correctness')

# GAP 8: LR 3-way classifier
tc = np.stack([ptr[:,i].argmax(1)==tr_t for i in range(3)], axis=1)
ty = np.full(len(tr_t), -1, dtype=int)
for i in range(len(tr_t)):
    c = tc[i]
    if c.sum() == 0: ty[i] = -1
    elif c.sum() == 1: ty[i] = np.where(c)[0][0]
    else: ty[i] = np.where(c)[0][ptr[i,c].max(1).argmax()]
u = ty >= 0
lr3 = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
lr3.fit(tr_feats[u], ty[u])
vp = lr3.predict(va_feats)
vr = np.array([pva[i,vp[i]].argmax() if vp[i]>=0 else pva[i].mean(0).argmax() for i in range(n_va)])
l3ba, _ = balanced_accuracy(va_t, vr)
print(f'GAP 8: LR 3-way (predict best expert):              {l3ba*100:.2f}%')

# ── CRITICAL QUESTION: What is the routing opportunity? ─────────
print('\n' + '=' * 65)
print('CRITICAL: WHERE DOES THE SIGNAL GO?')
print('=' * 65)

# On samples where uniform averaging is WRONG, how many could routing save?
uniform_preds = pva.mean(1).argmax(1)
uniform_wrong = uniform_preds != va_t

# Among those, how many have at least one correct expert?
can_be_saved = uniform_wrong & val_correct.any(1)
print(f'\nSamples where uniform averaging is wrong:          {uniform_wrong.sum()} / {n_va}')
print(f'Samples that COULD be saved by better routing:     {can_be_saved.sum()} / {n_va}')
print(f'(Already optimal: pick a correct expert instead of the average)')

# Among those savable, what is the distribution of correct experts?
for i, n in enumerate(names):
    count = (can_be_saved & val_correct[:,i]).sum()
    print(f'  Correct expert available: {n:8s} → {count} samples')

# What about samples where ALL experts are wrong AND uniform is wrong?
all_wrong_and_uniform_wrong = uniform_wrong & (~val_correct.any(1))
print(f'\nSamples where uniform is wrong AND all experts wrong: {all_wrong_and_uniform_wrong.sum()}')
print(f'No routing can help these — irreducible error.')

print(f'\nThus the MAXIMUM routing gain over uniform:')
print(f'  Best case: uniform + correctly route all {can_be_saved.sum()} savable samples')
can_be_saved_indices = np.where(can_be_saved)[0]
# For each savable sample, find a correct expert
saved_preds = uniform_preds.copy()
for idx in can_be_saved_indices:
    correct_experts = np.where(val_correct[idx])[0]
    if len(correct_experts) > 0:
        # Pick the best expert for this sample
        saved_preds[idx] = pva[idx, correct_experts[0]].argmax()
best_possible_ba, _ = balanced_accuracy(va_t, saved_preds)
print(f'  Upper bound (if router is perfect on savable):    {best_possible_ba*100:.2f}%')
print(f'  Gain over uniform:                                +{(best_possible_ba-uba)*100:.2f}%')
print(f'  Gain over oracle-weighted:                        +{(best_possible_ba-oba)*100:.2f}%' if best_possible_ba>oba else '')

# ── FINAL SUMMARY ───────────────────────────────────────────────
print('\n' + '=' * 65)
print('ROOT CAUSE SUMMARY')
print('=' * 65)
loss_gap = lba - oba
miscalib_gap = lba - cba if lba > cba else 0.0
feature_gap = oba - max(sba, l3ba)
print(f'')
print(f'Upper bound (oracle):                  {oracle_acc*100:.2f}%')
print(f'├─ Limitation: 37% samples all wrong    {100-oracle_acc*100:.1f}% of data is irrecoverable')
print(f'│')
print(f'└─ Best possible routing ceiling:       {lba*100:.2f}%  (loss-based oracle)')
print(f'   ├─ Loss from HARD vs SOFT routing:   {(lba-oba)*100:.2f}%')
print(f'   ├─ Loss from MISCALIBRATION:         {miscalib_gap*100:.2f}%')
print(f'   └─ Loss from FEATURE LEARNING:       {max(0, oba-max(sba,l3ba,cba))*100:.2f}%  ← main bottleneck')
print(f'      (This is the signal the features LOSE about which expert is correct)')
print(f'')
print(f'Current best learned method:           {max(sba,l3ba,cba)*100:.2f}%')
print(f'Uniform averaging:                     {uba*100:.2f}%')
print(f'')
print(f'If we could close the feature gap:     {oba*100:.2f}% would be achievable')
