# =============================
# File: adp_transformer_contrastive.py  (MODEL)
# Single-model, self-supervised Transformer with 6 ADP search policies
# Part A(4): Contrastive methods — SimCLR, CPC, QuickThought, SimCSE
# =============================

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# -----------------
# Projection head utility
# -----------------
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# -----------------
# Transformer encoder backbone
# -----------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(d_model*mlp_ratio)), nn.GELU(), nn.Linear(int(d_model*mlp_ratio), d_model))
    def forward(self, x):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------
# Config and Model
# -----------------
@dataclass
class ContrastiveCfg:
    vocab_size: int = 32000
    max_len: int = 128
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    projection_dim: int = 128
    variant: str = "simclr"  # {simclr,cpc,quickthought,simcse}
    max_width: int = 2048
    max_depth: int = 64

class AdaptiveTransformerContrastive(nn.Module):
    def __init__(self, cfg: ContrastiveCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.proj = ProjectionHead(cfg.d_model, cfg.projection_dim)

    @property
    def width(self): return self.cfg.d_model
    @property
    def depth(self): return self.cfg.n_layers
    @property
    def total_neurons(self): return self.cfg.d_model * 2 * self.cfg.n_layers

    def widen(self, ex_k: int):
        new_d = min(self.cfg.d_model + ex_k, self.cfg.max_width)
        if new_d == self.cfg.d_model: return False
        self.tok = nn.Embedding(self.cfg.vocab_size, new_d)
        self.pos = nn.Embedding(self.cfg.max_len, new_d)
        self.blocks = nn.ModuleList([EncoderBlock(new_d, max(1,new_d//64), self.cfg.mlp_ratio) for _ in range(self.cfg.n_layers)])
        self.ln = nn.LayerNorm(new_d)
        self.proj = ProjectionHead(new_d, self.cfg.projection_dim)
        self.cfg.d_model = new_d
        return True

    def deepen(self):
        if self.cfg.n_layers >= self.cfg.max_depth: return False
        self.blocks.append(EncoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//64), self.cfg.mlp_ratio))
        self.cfg.n_layers += 1
        return True

    def encode(self, ids):
        B,L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B,L)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks: x = blk(x)
        x = self.ln(x)
        return x.mean(dim=1)

    # -----------------
    # Contrastive objectives
    # -----------------
    def forward_simclr(self, a_ids, b_ids):
        z1 = F.normalize(self.proj(self.encode(a_ids)), dim=-1)
        z2 = F.normalize(self.proj(self.encode(b_ids)), dim=-1)
        logits = z1 @ z2.T / 0.1
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss, {}

    def forward_simcse(self, a_ids, b_ids):
        return self.forward_simclr(a_ids, b_ids)

    def forward_quickthought(self, a_ids, b_ids):
        # sentence-level contrast: predict if next sentence is true pair or not
        z1 = F.normalize(self.encode(a_ids), dim=-1)
        z2 = F.normalize(self.encode(b_ids), dim=-1)
        logits = z1 @ z2.T / 0.1
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss, {}

    def forward_cpc(self, seqs):
        # seqs: (B, L)
        z = self.encode(seqs)
        # shift by one: predict next rep (dummy)
        pos = z[:, 1:, :]
        pred = z[:, :-1, :]
        pred = F.normalize(pred, dim=-1)
        pos = F.normalize(pos, dim=-1)
        logits = torch.bmm(pred, pos.transpose(1,2)) / 0.1
        labels = torch.arange(pred.size(1), device=z.device)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.repeat(pred.size(0)))
        return loss, {}

    def forward(self, *args):
        v = self.cfg.variant
        if v == 'simclr': return self.forward_simclr(*args)
        if v == 'simcse': return self.forward_simcse(*args)
        if v == 'quickthought': return self.forward_quickthought(*args)
        if v == 'cpc': return self.forward_cpc(*args)
        raise ValueError(v)

# -----------------
# ADP search policies (6)
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

def train_inner(model: AdaptiveTransformerContrastive, tokens_a, tokens_b, steps: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best=float('inf'); best_state=_clone_state(model)
    for _ in range(steps):
        loss,_ = model(tokens_a, tokens_b)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        v=float(loss.item())
        if v<best: best=v; best_state=_clone_state(model)
    _load_state(model,best_state)
    return best

def adp_search(model: AdaptiveTransformerContrastive, t1, t2, s: SearchCfg, lr: float):
    def accept(base,new): return new<(base-s.delta)
    base=train_inner(model,t1,t2,5,lr)
    wf=df=0
    def width_series():
        nonlocal base,wf
        ok=False; wf=0
        while wf<s.trials_width and model.total_neurons<s.max_neurons:
            if not model.widen(s.ex_k): break
            new=train_inner(model,t1,t2,5,lr)
            if accept(base,new): base=new; ok=True; wf=0
            else: wf+=1
        return ok
    def depth_series():
        nonlocal base,df
        ok=False; df=0
        while df<s.trials_depth and model.total_neurons<s.max_neurons:
            if not model.deepen(): break
            new=train_inner(model,t1,t2,5,lr)
            if accept(base,new): base=new; ok=True; df=0
            else: df+=1
        return ok
    if s.algo=='depth_only':
        while df<s.trials_depth and model.total_neurons<s.max_neurons:
            if not model.deepen(): break
            new=train_inner(model,t1,t2,5,lr)
            if accept(base,new): base=new; df=0
            else: df+=1
        return {"best":base}
    if s.algo=='width_only':
        while wf<s.trials_width and model.total_neurons<s.max_neurons:
            if not model.widen(s.ex_k): break
            new=train_inner(model,t1,t2,5,lr)
            if accept(base,new): base=new; wf=0
            else: wf+=1
        return {"best":base}
    if s.algo=='wd':
        if width_series(): depth_series(); return {"best":base}
        return {"best":base}
    if s.algo=='dw':
        if depth_series(): width_series(); return {"best":base}
        return {"best":base}
    if s.algo=='alt_d':
        while True:
            d=depth_series(); w=width_series()
            if not d and not w: break
        return {"best":base}
    if s.algo=='alt_w':
        while True:
            w=width_series(); d=depth_series()
            if not w and not d: break
        return {"best":base}
    raise ValueError(s.algo)
