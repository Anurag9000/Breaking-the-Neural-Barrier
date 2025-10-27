# =============================
# File: adp_transformer_noncontrastive.py  (MODEL)
# Single-model Transformer for non-contrastive SSL with 6 ADP search policies
# Part A(5): Redundancy-Reduction / Non-contrastive — Barlow Twins, VICReg, SimSiam
# =============================

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Projection + (optional) predictor
# -----------------
class ProjectionHead(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in), nn.BatchNorm1d(d_in), nn.ReLU(inplace=True),
            nn.Linear(d_in, d_out)
        )
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
            nn.Linear(d, d)
        )
    def forward(self, x):
        return self.mlp(x)

# -----------------
# Encoder block (Transformer encoder)
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
# Config & Model
# -----------------
@dataclass
class NCfg:
    vocab_size: int = 32000
    max_len: int = 128
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    proj_dim: int = 2048  # typical large proj for Barlow/VICReg
    var_eps: float = 1e-4
    variant: str = "barlow"  # {barlow, vicreg, simsiam}
    # ADP caps
    max_width: int = 2048
    max_depth: int = 64

class AdaptiveTransformerNonContrastive(nn.Module):
    def __init__(self, cfg: NCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.proj = ProjectionHead(cfg.d_model, cfg.proj_dim)
        self.pred = Predictor(cfg.proj_dim)  # used in SimSiam; harmless for others

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
        self.proj = ProjectionHead(new_d, self.cfg.proj_dim)
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
        x = self.ln(x).mean(dim=1)
        return x

    # -----------------
    # Losses
    # -----------------
    def _barlow_loss(self, z1, z2, lambd=5e-3):
        # z1,z2: (B, D) normalized
        N,D = z1.size()
        c = (z1.T @ z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        return on_diag + lambd * off_diag

    def _vicreg_loss(self, z1, z2, sim_w=25.0, var_w=25.0, cov_w=1.0):
        # invariance
        sim = F.mse_loss(z1, z2)
        # variance
        def _var(z):
            std = torch.sqrt(z.var(dim=0) + 1e-4)
            return torch.mean(F.relu(1.0 - std))
        v = _var(z1) + _var(z2)
        # covariance
        z1c = z1 - z1.mean(dim=0)
        z2c = z2 - z2.mean(dim=0)
        c1 = (z1c.T @ z1c) / (z1.size(0)-1)
        c2 = (z2c.T @ z2c) / (z2.size(0)-1)
        cov = (c1.fill_diagonal_(0).pow(2).sum()/z1.size(1)) + (c2.fill_diagonal_(0).pow(2).sum()/z2.size(1))
        return sim_w*sim + var_w*v + cov_w*cov

    def _simsiam_loss(self, p, z):
        # stop-grad on z
        z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return - (p * z).sum(dim=-1).mean()

    # -----------------
    # Forwards (expects two augmented views tokenized separately)
    # -----------------
    def forward_barlow(self, a_ids, b_ids):
        z1 = F.normalize(self.proj(self.encode(a_ids)), dim=-1)
        z2 = F.normalize(self.proj(self.encode(b_ids)), dim=-1)
        loss = self._barlow_loss(z1, z2)
        return loss, {}

    def forward_vicreg(self, a_ids, b_ids):
        z1 = self.proj(self.encode(a_ids))
        z2 = self.proj(self.encode(b_ids))
        loss = self._vicreg_loss(z1, z2)
        return loss, {}

    def forward_simsiam(self, a_ids, b_ids):
        z1 = self.proj(self.encode(a_ids)); z2 = self.proj(self.encode(b_ids))
        p1 = self.pred(z1); p2 = self.pred(z2)
        loss = self._simsiam_loss(p1, z2) * 0.5 + self._simsiam_loss(p2, z1) * 0.5
        return loss, {}

    def forward(self, *args):
        v = self.cfg.variant
        if v == 'barlow': return self.forward_barlow(*args)
        if v == 'vicreg': return self.forward_vicreg(*args)
        if v == 'simsiam': return self.forward_simsiam(*args)
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


def train_inner(model: AdaptiveTransformerNonContrastive, a, b, steps: int, lr: float):
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


def adp_search(model: AdaptiveTransformerNonContrastive, t1, t2, s: SearchCfg, lr: float):
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
