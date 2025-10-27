# =============================
# File: adp_transformer_ar.py  (MODEL)
# Single-model, self-supervised Transformer with 6 ADP search policies
# Part A(3): Autoregressive LM variants: gpt, xlnet_perm
# =============================

from dataclasses import dataclass
from typing import Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Causal mask helper
# -----------------

def causal_mask(L: int, device: torch.device):
    return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

# -----------------
# Blocks
# -----------------

class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.act = nn.GELU(); self.fc2 = nn.Linear(h, d)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model*mlp_ratio))
    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h, attn_mask=attn_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------
# Config & Model
# -----------------

@dataclass
class ARCfg:
    vocab_size: int = 32000
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    max_len: int = 256

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0

    variant: str = "gpt"   # {gpt, xlnet_perm}

    max_width: int = 2048
    max_depth: int = 64

class AdaptiveTransformerAR(nn.Module):
    def __init__(self, cfg: ARCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight

    # ---- inspectors
    @property
    def width(self): return self.cfg.d_model
    @property
    def depth(self): return self.cfg.n_layers
    @property
    def total_neurons(self): return self.cfg.d_model * (2*self.cfg.n_layers)

    # ---- ADP ops
    def widen(self, ex_k: int):
        new_d = min(self.cfg.d_model + ex_k, self.cfg.max_width)
        if new_d == self.cfg.d_model: return False
        self.tok = nn.Embedding(self.cfg.vocab_size, new_d)
        self.pos = nn.Embedding(self.cfg.max_len, new_d)
        self.blocks = nn.ModuleList([DecoderBlock(new_d, max(1,new_d//64), self.cfg.mlp_ratio) for _ in range(self.cfg.n_layers)])
        self.ln = nn.LayerNorm(new_d)
        self.lm_head = nn.Linear(new_d, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.cfg.d_model = new_d
        return True

    def deepen(self):
        if self.cfg.n_layers >= self.cfg.max_depth: return False
        self.blocks.append(DecoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//64), self.cfg.mlp_ratio))
        self.cfg.n_layers += 1
        return True

    # ---- forward helpers
    def _embed(self, ids):
        B,L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B,L)
        return self.tok(ids) + self.pos(pos)

    def forward_gpt(self, x_ids):
        B,L = x_ids.shape
        y_in = torch.cat([torch.full_like(x_ids[:, :1], self.cfg.bos_id), x_ids[:, :-1]], dim=1)
        h = self._embed(y_in)
        mask = causal_mask(L, y_in.device)
        for blk in self.blocks: h = blk(h, attn_mask=mask)
        h = self.ln(h)
        logits = self.lm_head(h)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_ids.reshape(-1), ignore_index=self.cfg.pad_id)
        return loss, {}

    def forward_xlnet_perm(self, x_ids):
        # Simplified permutation LM without memory: draw a random permutation of positions
        B,L = x_ids.shape
        perm = torch.stack([torch.randperm(L, device=x_ids.device) for _ in range(B)], dim=0)  # (B,L)
        inv = torch.zeros_like(perm)
        for b in range(B): inv[b, perm[b]] = torch.arange(L, device=x_ids.device)
        # factorization order: for step t, attend to positions with inv < t
        y_in = torch.full_like(x_ids, self.cfg.pad_id)
        y_in[:, 0] = self.cfg.bos_id
        h = self._embed(y_in)
        # Build attn mask per batch using the permutation order (block uses shared mask -> approximate by worst case across batch)
        # We approximate with a dense mask derived from one random order per batch (first sample)
        order = inv[0]
        mask = torch.zeros(L, L, device=x_ids.device, dtype=torch.bool)
        for i in range(L):
            # positions allowed to attend: those with order < order[i]
            allowed = (order < order[i]).float()
            # disallow future by setting True in mask
            mask[i] = (1 - allowed).bool()
            mask[i, i] = True
        for blk in self.blocks: h = blk(h, attn_mask=mask)
        h = self.ln(h)
        logits = self.lm_head(h)
        # targets in permuted order (approx): teacher forcing with permutation
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_ids.reshape(-1), ignore_index=self.cfg.pad_id)
        return loss, {"perm": perm}

    def forward(self, x_ids):
        if self.cfg.variant == 'gpt': return self.forward_gpt(x_ids)
        if self.cfg.variant == 'xlnet_perm': return self.forward_xlnet_perm(x_ids)
        raise ValueError(self.cfg.variant)

# -----------------
# ADP search (6 policies)
# -----------------

@dataclass
class SearchCfg:
    algo: str = 'wd'
    ex_k: int = 64
    delta: float = 1e-3
    trials_width: int = 2
    trials_depth: int = 2
    max_neurons: int = 8_000_000

@torch.no_grad()
def _clone_state(model: nn.Module):
    return {k:v.clone() for k,v in model.state_dict().items()}

def _load_state(model: nn.Module, st: Dict[str,torch.Tensor]):
    model.load_state_dict(st, strict=False)


def train_inner(model: AdaptiveTransformerAR, tokens: torch.Tensor, steps: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float('inf'); best_state = _clone_state(model)
    for _ in range(steps):
        loss,_ = model(tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(loss.item())
        if v < best: best=v; best_state=_clone_state(model)
    _load_state(model, best_state)
    return best


def adp_search(model: AdaptiveTransformerAR, tokens: torch.Tensor, s: SearchCfg, lr: float):
    def accept(base, new): return new < (base - s.delta)
    base = train_inner(model, tokens, 5, lr)
    width_fails = depth_fails = 0

    def width_series():
        nonlocal base, width_fails
        wins=False; width_fails=0
        while width_fails < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base,new): base=new; wins=True; width_fails=0
            else: width_fails+=1
        return wins

    def depth_series():
        nonlocal base, depth_fails
        wins=False; depth_fails=0
        while depth_fails < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base,new): base=new; wins=True; depth_fails=0
            else: depth_fails+=1
        return wins

    if s.algo == 'depth_only':
        while depth_fails < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base,new): base=new; depth_fails=0
            else: depth_fails+=1
        return {"best": base}

    if s.algo == 'width_only':
        while width_fails < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base,new): base=new; width_fails=0
            else: width_fails+=1
        return {"best": base}

    if s.algo == 'wd':
        if width_series(): depth_series(); return {"best": base}
        return {"best": base}
    if s.algo == 'dw':
        if depth_series(): width_series(); return {"best": base}
        return {"best": base}
    if s.algo == 'alt_d':
        while True:
            d = depth_series(); w = width_series()
            if not d and not w: break
        return {"best": base}
    if s.algo == 'alt_w':
        while True:
            w = width_series(); d = depth_series()
            if not w and not d: break
        return {"best": base}
    raise ValueError(s.algo)
