import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility: patchify images -> tokens
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        # x: B,C,H,W -> tokens B, N, D
        x = self.norm(self.proj(x))
        B,D,H,W = x.shape
        x = x.flatten(2).transpose(1,2) # B, N, D
        return x

# -----------------------------
# Retention block (attention replacement)
# Simplified: depthwise causal conv with exponential kernel per channel (EMA-like)
# -----------------------------
class RetentionBlock(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.0, kernel_size=9):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size-1, bias=False)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_mult*dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_mult*dim, dim), nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)
        self._init_exp_kernel(kernel_size)
    def _init_exp_kernel(self, k):
        # initialize DW kernel as decaying exponential (causal)
        with torch.no_grad():
            for g in range(self.dwconv.weight.shape[0]):
                alpha = 0.8
                ker = torch.tensor([alpha**i for i in range(k)], dtype=self.dwconv.weight.dtype)
                ker = ker.flip(0) # causal tail at the end
                self.dwconv.weight[g,0,:] = ker
    def forward(self, x):
        # x: B, N, D
        h = self.norm1(x)
        y = h.transpose(1,2)  # B,D,N
        y = self.pw(self.dwconv(y))
        y = y[:,:,:x.size(1)]  # trim padding to sequence len
        y = y.transpose(1,2)
        x = x + self.drop(y)
        x = x + self.ff(x)
        return x

# -----------------------------
# Adaptive RetNet backbone
# -----------------------------
class RetNetTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=4, patch=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch = patch
        self.tokenizer = PatchEmbed(3, embed_dim, patch)
        self.blocks = nn.ModuleList([RetentionBlock(embed_dim, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    # ----- width/depth growth helpers -----
    def add_block(self):
        self.blocks.append(RetentionBlock(self.embed_dim))
    def widen_all(self, ex_k: int):
        new_dim = self.embed_dim + ex_k
        # tokenizer
        new_tok = PatchEmbed(3, new_dim, self.patch)
        copy_conv2d(self.tokenizer.proj, new_tok.proj)
        copy_bn2d(self.tokenizer.norm, new_tok.norm)
        self.tokenizer = new_tok
        # blocks
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = RetentionBlock(new_dim)
            transplant_block_retention(b, nb)
            new_blocks.append(nb)
        self.blocks = new_blocks
        # norm + head
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features)
        copy_linear_overlap(self.head, new_head)
        self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        x = self.tokenizer(x)           # B,N,D
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        x = x.mean(1)                   # global average over tokens
        return self.head(x)

# -----------------------------
# Weight copy utilities (overlap copy for growth)
# -----------------------------
@torch.no_grad()
def copy_conv2d(old: nn.Conv2d, new: nn.Conv2d):
    new.weight.zero_()
    oh, ow = old.weight.shape[:2]
    kh, kw = min(new.weight.shape[2], old.weight.shape[2]), min(new.weight.shape[3], old.weight.shape[3])
    new.weight[:oh,:ow,:kh,:kw].copy_(old.weight[:oh,:ow,:kh,:kw])
    if old.bias is not None and new.bias is not None:
        new.bias.zero_(); new.bias[:old.bias.numel()].copy_(old.bias)

@torch.no_grad()
def copy_bn2d(old: nn.BatchNorm2d, new: nn.BatchNorm2d):
    c = min(old.num_features, new.num_features)
    new.weight[:c].copy_(old.weight[:c]); new.bias[:c].copy_(old.bias[:c])
    new.running_mean[:c].copy_(old.running_mean[:c])
    new.running_var[:c].copy_(old.running_var[:c])

@torch.no_grad()
def copy_linear_overlap(old: nn.Linear, new: nn.Linear):
    out = min(old.out_features, new.out_features)
    inn = min(old.in_features, new.in_features)
    new.weight[:out,:inn].copy_(old.weight[:out,:inn])
    if old.bias is not None and new.bias is not None:
        new.bias[:out].copy_(old.bias[:out])

@torch.no_grad()
def transplant_block_retention(old: RetentionBlock, new: RetentionBlock):
    # copy depthwise + pw convs channel-overlap
    oc = min(old.dwconv.weight.shape[0], new.dwconv.weight.shape[0])
    k = min(old.dwconv.weight.shape[-1], new.dwconv.weight.shape[-1])
    new.dwconv.weight[:oc,0,-k:].copy_(old.dwconv.weight[:oc,0,-k:])
    new.pw.weight[:oc,:oc,0].copy_(old.pw.weight[:oc,:oc,0])
    if old.pw.bias is not None and new.pw.bias is not None:
        new.pw.bias[:oc].copy_(old.pw.bias[:oc])
    # FFN layers
    for (ol, nl) in zip(old.ff, new.ff):
        if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear):
            copy_linear_overlap(ol, nl)

# -----------------------------
# Training utilities
# -----------------------------
@dataclass
class TrainCfg:
    lr: float = 1e-3
    wd: float = 1e-4
    epochs: int = 30
    batch_size: int = 256
    patience: int = 5
    delta: float = 0.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class EarlyStopper:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience; self.delta = delta
        self.best = float('inf'); self.count = 0
        self.best_state = None
    def update(self, loss, model):
        improved = loss < self.best - self.delta
        if improved:
            self.best = loss; self.count = 0
            self.best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.count += 1
        return improved
    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def evaluate(model, loader, device):
    model.eval(); loss_sum=0; n=0; correct=0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_sum += loss.item()*x.size(0); n += x.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
    return loss_sum/n, correct/n

def train_inner(model, train_loader, val_loader, cfg: TrainCfg):
    device = cfg.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopper(cfg.patience, cfg.delta)
    for epoch in range(cfg.epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss, _ = evaluate(model, val_loader, device)
        es.update(val_loss, model)
        if es.count >= cfg.patience:
            break
    es.restore_best(model)
    return evaluate(model, val_loader, device)

# -----------------------------
# ADP strategies (6 variants)
# width_to_depth, depth_to_width, alt_depth, alt_width, depth_only, width_only
# -----------------------------
@dataclass
class ADPCfg:
    init_width: int = 64
    init_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 24
    trials_width: int = 2
    trials_depth: int = 2


def adp_search(model, train_loader, val_loader, train_cfg: TrainCfg, adp: ADPCfg, mode: str):
    # Initialize
    best_loss, best_acc = train_inner(model, train_loader, val_loader, train_cfg)
    baseline_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    def try_widen():
        pre_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        model.widen_all(adp.ex_k)
        loss, acc = train_inner(model, train_loader, val_loader, train_cfg)
        improved = loss < best[0] - train_cfg.delta
        if improved:
            return (loss, acc), True, None
        else:
            model.load_state_dict(pre_state); return (loss, acc), False, pre_state

    def try_deepen():
        pre_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        model.add_block()
        loss, acc = train_inner(model, train_loader, val_loader, train_cfg)
        improved = loss < best[0] - train_cfg.delta
        if improved:
            return (loss, acc), True, None
        else:
            model.load_state_dict(pre_state); return (loss, acc), False, pre_state

    best = (best_loss, best_acc)

    if mode == 'width_to_depth':
        w_trials = 0
        while model.embed_dim < adp.max_width and w_trials < adp.trials_width:
            _, ok, _ = try_widen(); w_trials += 1
            if ok:
                best = evaluate(model, val_loader, train_cfg.device)
                d_trials = 0
                while len(model.blocks) < adp.max_depth and d_trials < adp.trials_depth:
                    _, dok, _ = try_deepen(); d_trials += 1
                    if dok:
                        best = evaluate(model, val_loader, train_cfg.device)
                    else:
                        break
            else:
                break

    elif mode == 'depth_to_width':
        d_trials = 0
        while len(model.blocks) < adp.max_depth and d_trials < adp.trials_depth:
            _, ok, _ = try_deepen(); d_trials += 1
            if ok:
                best = evaluate(model, val_loader, train_cfg.device)
                w_trials = 0
                while model.embed_dim < adp.max_width and w_trials < adp.trials_width:
                    _, wok, _ = try_widen(); w_trials += 1
                    if wok:
                        best = evaluate(model, val_loader, train_cfg.device)
                    else:
                        break
            else:
                break

    elif mode == 'alt_depth':
        while True:
            _, dok, _ = try_deepen()
            did_any = dok
            if dok: best = evaluate(model, val_loader, train_cfg.device)
            _, wok, _ = try_widen()
            did_any = did_any or wok
            if wok: best = evaluate(model, val_loader, train_cfg.device)
            if not did_any:
                break

    elif mode == 'alt_width':
        while True:
            _, wok, _ = try_widen()
            did_any = wok
            if wok: best = evaluate(model, val_loader, train_cfg.device)
            _, dok, _ = try_deepen()
            did_any = did_any or dok
            if dok: best = evaluate(model, val_loader, train_cfg.device)
            if not did_any:
                break

    elif mode == 'depth_only':
        while len(model.blocks) < adp.max_depth:
            _, dok, _ = try_deepen()
            if not dok: break
            best = evaluate(model, val_loader, train_cfg.device)

    elif mode == 'width_only':
        while model.embed_dim < adp.max_width:
            _, wok, _ = try_widen()
            if not wok: break
            best = evaluate(model, val_loader, train_cfg.device)

    return best

# -----------------------------
# Factory
# -----------------------------

def build_retnet(num_classes=10, init_width=64, init_depth=2, patch=4):
    return RetNetTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch=patch)
