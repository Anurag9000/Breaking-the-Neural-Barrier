# =============================
# File: adp_transformer_pretext.py  (MODEL)
# Single-model Vision Transformer for pretext SSL + 6 ADP policies
# Part A(7): RotNet / Jigsaw / Colorization / DAE (denoising)
# =============================

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Patchify helpers
# -----------------

def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size; w = W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*patch_size*patch_size)
    return x

# -----------------
# Transformer blocks
# -----------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))
    def forward(self, x):
        h = self.ln1(x); a,_ = self.attn(h,h,h); x = x + a
        x = x + self.mlp(self.ln2(x)); return x

# -----------------
# Config & Model
# -----------------
@dataclass
class PretextCfg:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0

    variant: str = "rotnet"   # {rotnet, jigsaw, color, dae}
    jigsaw_grid: int = 3       # g x g patches for jigsaw classification
    jigsaw_k: int = 30         # number of fixed permutations (classes)

    # ADP caps
    max_dim: int = 1024
    max_depth: int = 24

class AdaptiveViTPretext(nn.Module):
    def __init__(self, cfg: PretextCfg):
        super().__init__()
        self.cfg = cfg
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Linear(cfg.in_chans * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.dim))
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)

        # Heads
        self.cls_head_rotnet = nn.Linear(cfg.dim, 4)
        self.cls_head_jigsaw = nn.Linear(cfg.dim, cfg.jigsaw_k)
        self.pix_head = nn.Linear(cfg.dim, cfg.in_chans * cfg.patch_size * cfg.patch_size)

        # Precompute jigsaw permutations (fixed)
        self.register_buffer("jigsaw_perms", self._make_perms(cfg))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ---- inspectors
    @property
    def width(self): return self.cfg.dim
    @property
    def total_neurons(self): return self.cfg.dim * (self.cfg.depth * 2)

    # ---- ADP ops
    def widen(self, ex_k: int):
        new_dim = min(self.cfg.dim + ex_k, self.cfg.max_dim)
        if new_dim == self.cfg.dim: return False
        self.patch_embed = nn.Linear(self.patch_embed.in_features, new_dim)
        new_pos = nn.Parameter(torch.zeros(1, self.num_patches, new_dim))
        self.pos_embed = new_pos
        self.blocks = nn.ModuleList([Block(new_dim, max(1, new_dim//32), 4.0) for _ in range(self.cfg.depth)])
        self.norm = nn.LayerNorm(new_dim)
        self.cls_head_rotnet = nn.Linear(new_dim, 4)
        self.cls_head_jigsaw = nn.Linear(new_dim, self.cfg.jigsaw_k)
        self.pix_head = nn.Linear(new_dim, self.pix_head.out_features)
        self.cfg.dim = new_dim
        return True

    def deepen(self):
        if self.cfg.depth >= self.cfg.max_depth: return False
        self.blocks.append(Block(self.cfg.dim, max(1, self.cfg.dim//32), 4.0))
        self.cfg.depth += 1
        return True

    # ---- enc
    def encode(self, x):
        patches = patchify(x, self.cfg.patch_size)
        tok = self.patch_embed(patches) + self.pos_embed
        for b in self.blocks: tok = b(tok)
        tok = self.norm(tok)
        return tok.mean(dim=1), patches  # pooled token, raw patch targets

    # ---- jigsaw perms
    def _make_perms(self, cfg: PretextCfg):
        g = cfg.jigsaw_grid
        n = g*g
        base = torch.arange(n)
        perms = []
        random.seed(7)
        seen = set()
        while len(perms) < cfg.jigsaw_k:
            p = base[torch.randperm(n)]
            tup = tuple(p.tolist())
            if tup in seen: continue
            seen.add(tup); perms.append(p)
        return torch.stack(perms, 0)  # (K, n)

    # ---- objectives
    def forward_rotnet(self, x):
        # choose random rotation for each sample and classify
        B = x.size(0)
        labels = torch.randint(0,4,(B,), device=x.device)
        # apply rotation (nearest neighbor by tensor ops)
        def rot(img, k):
            return torch.rot90(img, k=int(k), dims=(2,3))
        x_aug = torch.stack([rot(x[i], int(labels[i])) for i in range(B)], dim=0)
        h,_ = self.encode(x_aug)
        logits = self.cls_head_rotnet(h)
        loss = F.cross_entropy(logits, labels)
        return loss, {"labels": labels}

    def forward_jigsaw(self, x):
        B, C, H, W = x.shape
        g = self.cfg.jigsaw_grid
        ps = H // g  # assume square image divisible by g
        # cut to grid
        tiles = []
        for i in range(g):
            for j in range(g):
                tiles.append(x[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps])
        tiles = torch.stack(tiles, dim=2)  # (B, C, n, ps, ps) with n=g*g in dim=2
        n = g*g
        idx = torch.randint(0, self.cfg.jigsaw_k, (B,), device=x.device)
        perm = self.jigsaw_perms[idx]  # (B, n)
        # permute tiles per sample and reassemble image
        x_perm = []
        for b in range(B):
            order = perm[b]
            pieces = [tiles[b, :, k] for k in order]
            # recompose
            rows = []
            for r in range(g):
                row = torch.cat(pieces[r*g:(r+1)*g], dim=-1)
                rows.append(row)
            img = torch.cat(rows, dim=-2)
            x_perm.append(img)
        x_perm = torch.stack(x_perm, dim=0)
        h,_ = self.encode(x_perm)
        logits = self.cls_head_jigsaw(h)
        loss = F.cross_entropy(logits, idx)
        return loss, {"perm_idx": idx}

    def forward_color(self, x):
        # grayscale -> predict RGB (simple colorization proxy)
        gray = (0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]).repeat(1,3,1,1)
        h, patches = self.encode(gray)
        pred = self.pix_head(self.patch_embed(patchify(gray, self.cfg.patch_size)) + self.pos_embed)
        loss = F.l1_loss(pred, patchify(x, self.cfg.patch_size))
        return loss, {"pred": pred}

    def forward_dae(self, x):
        noise = torch.randn_like(x) * 0.1
        noisy = x + noise
        _, patches = self.encode(noisy)
        pred = self.pix_head(self.patch_embed(patchify(noisy, self.cfg.patch_size)) + self.pos_embed)
        target = patchify(x, self.cfg.patch_size)
        loss = F.mse_loss(pred, target)
        return loss, {"pred": pred}

    def forward(self, x):
        v = self.cfg.variant
        if v == 'rotnet': return self.forward_rotnet(x)
        if v == 'jigsaw': return self.forward_jigsaw(x)
        if v == 'color': return self.forward_color(x)
        if v == 'dae': return self.forward_dae(x)
        raise ValueError(v)

# -----------------
# ADP search (6 policies)
# -----------------
@dataclass
class SearchCfg:
    algo: str = 'wd'
    ex_k: int = 32
    delta: float = 1e-3
    trials_width: int = 2
    trials_depth: int = 2
    max_neurons: int = 4_000_000

@torch.no_grad()
def _clone_state(model: nn.Module):
    return {k:v.clone() for k,v in model.state_dict().items()}

def _load_state(model: nn.Module, st: Dict[str,torch.Tensor]):
    model.load_state_dict(st, strict=False)


def train_inner(model: AdaptiveViTPretext, images: torch.Tensor, steps: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best=float('inf'); best_state=_clone_state(model)
    for _ in range(steps):
        loss,_ = model(images)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        v=float(loss.item())
        if v<best: best=v; best_state=_clone_state(model)
    _load_state(model,best_state)
    return best


def adp_search(model: AdaptiveViTPretext, images: torch.Tensor, s: SearchCfg, lr: float):
    def accept(base,new): return new < (base - s.delta)
    base = train_inner(model, images, 5, lr)
    wf=df=0
    def width_series():
        nonlocal base, wf
        ok=False; wf=0
        while wf < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, images, 5, lr)
            if accept(base,new): base=new; ok=True; wf=0
            else: wf+=1
        return ok
    def depth_series():
        nonlocal base, df
        ok=False; df=0
        while df < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, images, 5, lr)
            if accept(base,new): base=new; ok=True; df=0
            else: df+=1
        return ok
    if s.algo=='depth_only':
        while df < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, images, 5, lr)
            if accept(base,new): base=new; df=0
            else: df+=1
        return {"best": base}
    if s.algo=='width_only':
        while wf < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, images, 5, lr)
            if accept(base,new): base=new; wf=0
            else: wf+=1
        return {"best": base}
    if s.algo=='wd':
        if width_series(): depth_series(); return {"best": base}
        return {"best": base}
    if s.algo=='dw':
        if depth_series(): width_series(); return {"best": base}
        return {"best": base}
    if s.algo=='alt_d':
        while True:
            d = depth_series(); w = width_series()
            if not d and not w: break
        return {"best": base}
    if s.algo=='alt_w':
        while True:
            w = width_series(); d = depth_series()
            if not w and not d: break
        return {"best": base}
    raise ValueError(s.algo)
