# =============================
# File: adp_transformer_swav.py  (MODEL)
# Single-model Transformer with SwaV-style online prototypes + 6 ADP policies
# Part A(6): Online clustering / prototype learning (SwaV)
# =============================

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Transformer encoder block
# -----------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model*mlp_ratio)), nn.GELU(), nn.Linear(int(d_model*mlp_ratio), d_model)
        )
    def forward(self, x):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------
# Sinkhorn-Knopp (online) for assignment
# -----------------
@torch.no_grad()
def sinkhorn(Q: torch.Tensor, n_iters: int = 3, eps: float = 1e-12):
    # Q: (B, K) non-negative (logits already softmaxed)
    Q = Q.t()  # (K, B)
    K, B = Q.shape
    Q = Q / (Q.sum())
    r = torch.ones(K, device=Q.device) / K
    c = torch.ones(B, device=Q.device) / B
    for _ in range(n_iters):
        u = Q.sum(dim=1)
        Q *= (r / (u + eps)).unsqueeze(1)
        v = Q.sum(dim=0)
        Q *= (c / (v + eps)).unsqueeze(0)
    return (Q / (Q.sum(dim=0, keepdim=True) + eps)).t()  # (B, K)

# -----------------
# Config & Model
# -----------------
@dataclass
class SwAVCfg:
    vocab_size: int = 32000
    max_len: int = 128
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    proj_dim: int = 128
    n_prototypes: int = 300
    temperature: float = 0.1

    # ADP caps
    max_width: int = 2048
    max_depth: int = 64

class AdaptiveTransformerSwAV(nn.Module):
    def __init__(self, cfg: SwAVCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        # projection head
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.BatchNorm1d(cfg.d_model), nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.proj_dim)
        )
        # prototype matrix (normalized rows)
        self.prototypes = nn.Parameter(torch.randn(cfg.n_prototypes, cfg.proj_dim) * 0.02)

    # inspectors
    @property
    def width(self): return self.cfg.d_model
    @property
    def depth(self): return self.cfg.n_layers
    @property
    def total_neurons(self): return self.cfg.d_model * 2 * self.cfg.n_layers

    # ADP ops
    def widen(self, ex_k: int):
        new_d = min(self.cfg.d_model + ex_k, self.cfg.max_width)
        if new_d == self.cfg.d_model: return False
        self.tok = nn.Embedding(self.cfg.vocab_size, new_d)
        self.pos = nn.Embedding(self.cfg.max_len, new_d)
        self.blocks = nn.ModuleList([EncoderBlock(new_d, max(1,new_d//64), self.cfg.mlp_ratio) for _ in range(self.cfg.n_layers)])
        self.ln = nn.LayerNorm(new_d)
        self.proj = nn.Sequential(
            nn.Linear(new_d, new_d), nn.BatchNorm1d(new_d), nn.ReLU(inplace=True),
            nn.Linear(new_d, self.cfg.proj_dim)
        )
        self.cfg.d_model = new_d
        return True

    def deepen(self):
        if self.cfg.n_layers >= self.cfg.max_depth: return False
        self.blocks.append(EncoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//64), self.cfg.mlp_ratio))
        self.cfg.n_layers += 1
        return True

    # encode mean-pooled sentence representation
    def encode(self, ids):
        B,L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B,L)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks: x = blk(x)
        x = self.ln(x).mean(dim=1)
        return x

    def _project(self, x):
        z = self.proj(x)
        z = F.normalize(z, dim=-1)
        return z

    def _prototype_logits(self, z):
        # normalize prototypes
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        return z @ self.prototypes.t() / self.cfg.temperature  # (B, K)

    def forward(self, a_ids, b_ids):
        # two views (augmentations) already tokenized
        z1 = self._project(self.encode(a_ids))
        z2 = self._project(self.encode(b_ids))
        logits1 = self._prototype_logits(z1)
        logits2 = self._prototype_logits(z2)
        # compute assignments via Sinkhorn on softmaxed logits
        with torch.no_grad():
            q1 = sinkhorn(F.softmax(logits1, dim=-1))  # (B,K)
            q2 = sinkhorn(F.softmax(logits2, dim=-1))
        # swapped prediction
        loss = - ( (q1 * F.log_softmax(logits2, dim=-1)).sum(dim=-1).mean() \
                 + (q2 * F.log_softmax(logits1, dim=-1)).sum(dim=-1).mean() ) / 2.0
        return loss, {"z1": z1, "z2": z2}

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


def train_inner(model: AdaptiveTransformerSwAV, a, b, steps: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best=float('inf'); best_state=_clone_state(model)
    for _ in range(steps):
        loss,_ = model(a,b)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        v=float(loss.item())
        if v<best: best=v; best_state=_clone_state(model)
    _load_state(model,best_state)
    return best


def adp_search(model: AdaptiveTransformerSwAV, t1, t2, s: SearchCfg, lr: float):
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
