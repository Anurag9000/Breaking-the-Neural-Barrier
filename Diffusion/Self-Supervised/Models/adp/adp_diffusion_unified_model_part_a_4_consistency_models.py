"""
ADP Diffusion – Unified Model (Part A4: Consistency Models)
Six ADP growth algorithms (w2d, d2w, alt_w, alt_d, width_only, depth_only)
wrapped around a width/depth‑editable UNet trained with a **self‑consistency** objective.

BNB rule: single‑model only (no teacher / EMA). Objective is teacher‑free.

Loss =  λ_c * || f(x_t1, t1) − f(x_t2, t2) ||^2  +  λ_a * || f(x_t, t) − x0 ||^2
- Consistency term encourages time‑agnostic reconstruction across two random noise levels.
- Anchor term prevents degenerate solutions and ties predictions to data.

Implementation notes:
- Time t ∈ (0,1]; noise scale σ(t) = σ_min * (σ_max/σ_min)^t (VE‑like mapping)
- f predicts **denoised image** x̂₀.
- Width = base channel count; Depth = blocks_per_stage (one ResBlock per stage per depth unit).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Overlap‑copy transplant utils
# -----------------------------

def _copy_overlap_(dst: torch.nn.Parameter, src: torch.nn.Parameter):
    with torch.no_grad():
        sd, ss = list(dst.shape), list(src.shape)
        sl = [min(a, b) for a, b in zip(sd, ss)]
        slices = tuple(slice(0, n) for n in sl)
        dst.zero_(); dst[slices].copy_(src[slices])

def transplant_module(dst: nn.Module, src: nn.Module):
    with torch.no_grad():
        for (dn, dpar), (sn, spar) in zip(dst.named_parameters(), src.named_parameters()):
            try: _copy_overlap_(dpar, spar)
            except Exception: pass
        for (dn, db), (sn, sb) in zip(dst.named_buffers(), src.named_buffers()):
            if db.shape == sb.shape: db.copy_(sb)

# -----------------------------
# UNet backbone (width/depth editable)
# -----------------------------

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(); self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False); self.bn = nn.BatchNorm2d(cout); self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c1 = ConvBNAct(c, c); self.c2 = ConvBNAct(c, c)
    def forward(self, x): return x + self.c2(self.c1(x))

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.conv = ConvBNAct(cin, cout, 3, 2, 1)
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.up = nn.ConvTranspose2d(cin, cout, 2, 2); self.post = ConvBNAct(cout, cout)
    def forward(self, x): return self.post(self.up(x))

class TimeEmbed(nn.Module):
    def __init__(self, c):
        super().__init__(); self.fc1 = nn.Linear(1, c); self.fc2 = nn.Linear(c, c); self.act = nn.SiLU()
    def forward(self, t):
        h = t.view(-1,1); h = self.act(self.fc1(h)); return self.fc2(h)

class UNetCM(nn.Module):
    """UNet predicting x0; supports width/depth growth with weight transplant."""
    def __init__(self, in_channels=3, base=32, stages=3, blocks_per_stage=1):
        super().__init__()
        self.in_channels, self.base, self.stages, self.blocks_per_stage = in_channels, base, stages, blocks_per_stage
        self.time_emb = TimeEmbed(base)
        self.in_conv = ConvBNAct(in_channels, base)
        chans = [base*(2**i) for i in range(stages)]
        self.downs = nn.ModuleList(); self.res_down = nn.ModuleList(); c = base
        for i in range(stages):
            self.res_down.append(nn.Sequential(*[ResBlock(c) for _ in range(blocks_per_stage)]))
            if i < stages-1: self.downs.append(Down(c, c*2)); c *= 2
            else: self.downs.append(nn.Identity())
        self.ups = nn.ModuleList(); self.res_up = nn.ModuleList()
        for i in reversed(range(stages)):
            self.res_up.append(nn.Sequential(*[ResBlock(chans[i]) for _ in range(blocks_per_stage)]))
            if i>0: self.ups.append(Up(chans[i], chans[i-1]))
            else: self.ups.append(nn.Identity())
        self.out = nn.Conv2d(base, in_channels, 1)

    # growth ops
    def append_depth(self):
        new = UNetCM(self.in_channels, self.base, self.stages, self.blocks_per_stage+1)
        transplant_module(new, self); return new
    def widen_all(self, delta: int):
        new = UNetCM(self.in_channels, self.base+delta, self.stages, self.blocks_per_stage)
        transplant_module(new, self); return new

    def forward(self, x, t):
        emb = self.time_emb(t)
        h = self.in_conv(x)
        h = h + emb.view(emb.size(0), -1, 1, 1)
        skips = []; c = h
        for i in range(self.stages):
            c = self.res_down[i](c); skips.append(c); c = self.downs[i](c)
        for i in range(self.stages):
            j = self.stages-1-i
            c = self.res_up[i](c + skips[j]); c = self.ups[i](c)
        return self.out(c)

# -----------------------------
# Consistency objective (teacher‑free)
# -----------------------------

@dataclass
class CMCfg:
    sigma_min: float = 0.01
    sigma_max: float = 50.0
    lambda_consistency: float = 1.0
    lambda_anchor: float = 0.1
    device: str = 'cuda'

    def sigma(self, t: torch.Tensor):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t


def cm_loss(model: UNetCM, x0: torch.Tensor, cfg: CMCfg, rng: torch.Generator):
    B = x0.size(0); device = x0.device
    # two random times
    t1 = torch.rand(B, device=device, generator=rng) * 0.95 + 0.05
    t2 = torch.rand(B, device=device, generator=rng) * 0.95 + 0.05
    s1, s2 = cfg.sigma(t1), cfg.sigma(t2)
    n1 = torch.randn_like(x0, generator=rng); n2 = torch.randn_like(x0, generator=rng)
    x1 = x0 + n1 * s1.view(-1,1,1,1)
    x2 = x0 + n2 * s2.view(-1,1,1,1)
    y1 = model(x1, t1)
    y2 = model(x2, t2)
    # consistency: predictions match across times
    l_c = F.mse_loss(y1, y2)
    # anchor: occasionally push to x0 at a random time
    t = torch.rand(B, device=device, generator=rng) * 0.95 + 0.05
    s = cfg.sigma(t); n = torch.randn_like(x0, generator=rng)
    xt = x0 + n * s.view(-1,1,1,1)
    y = model(xt, t)
    l_a = F.mse_loss(y, x0)
    return cfg.lambda_consistency * l_c + cfg.lambda_anchor * l_a

# -----------------------------
# Trainer + Early stop
# -----------------------------

@dataclass
class TrainCfg:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5
    grad_clip: float = 1.0
    device: str = 'cuda'
    seed: int = 42

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience; self.best = math.inf; self.bad = 0
        self.chk: Optional[Dict[str, torch.Tensor]] = None
    def step(self, value: float, model: nn.Module):
        if value < self.best:
            self.best = value; self.bad = 0
            self.chk = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
    def should_stop(self): return self.bad > self.patience


def train_one(model: UNetCM, train_loader, val_loader, tcfg: TrainCfg, ccfg: CMCfg):
    torch.manual_seed(tcfg.seed)
    device = torch.device(tcfg.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    rng = torch.Generator(device=device); rng.manual_seed(tcfg.seed + 456)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    stopper = EarlyStopper(tcfg.patience)
    for epoch in range(tcfg.max_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss = cm_loss(model, x, ccfg, rng)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
        # val
        model.eval(); vloss = 0.0; n=0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                vloss += cm_loss(model, x, ccfg, rng).item() * x.size(0)
                n += x.size(0)
        vloss /= max(n,1); stopper.step(vloss, model)
        if stopper.should_stop(): break
    if stopper.chk is not None: model.load_state_dict(stopper.chk)
    return stopper.best, {k: v.clone() for k, v in model.state_dict().items()}

# -----------------------------
# ADP search (six algorithms)
# -----------------------------

@dataclass
class SearchCfg:
    max_depth: int = 8
    max_base: int = 192
    trials_width: int = 2
    trials_depth: int = 2
    delta: float = 1e-4

def accept(new_val, old_val, delta): return new_val < (old_val - delta)

def _run(prop, tr, va, tcfg, ccfg): return train_one(prop, tr, va, tcfg, ccfg)

def adp_width_to_depth(model, tr, va, tcfg, ccfg, scfg):
    best_val, best_sd = _run(model, tr, va, tcfg, ccfg)
    w_fail = d_fail = 0
    while True:
        improved=False
        while w_fail < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop=model.widen_all(1); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; improved=True
            else: w_fail+=1
        if not improved: break
        improved=False
        while d_fail < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop=model.append_depth(); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; improved=True
            else: d_fail+=1
        if not improved: break
    model.load_state_dict(best_sd); return model, best_val

def adp_depth_to_width(model, tr, va, tcfg, ccfg, scfg):
    best_val, best_sd = _run(model, tr, va, tcfg, ccfg)
    d_fail = w_fail = 0
    while True:
        improved=False
        while d_fail < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop=model.append_depth(); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; improved=True
            else: d_fail+=1
        if not improved: break
        improved=False
        while w_fail < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop=model.widen_all(1); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; improved=True
            else: w_fail+=1
        if not improved: break
    model.load_state_dict(best_sd); return model, best_val

def adp_alt_depth(model, tr, va, tcfg, ccfg, scfg):
    best_val, best_sd = _run(model, tr, va, tcfg, ccfg)
    while True:
        any_acc=False
        f=0
        while f < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop=model.append_depth(); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        f=0
        while f < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop=model.widen_all(1); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        if not any_acc: break
    model.load_state_dict(best_sd); return model, best_val

def adp_alt_width(model, tr, va, tcfg, ccfg, scfg):
    best_val, best_sd = _run(model, tr, va, tcfg, ccfg)
    while True:
        any_acc=False
        f=0
        while f < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop=model.widen_all(1); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        f=0
        while f < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop=model.append_depth(); v,sd=_run(prop,tr,va,tcfg,ccfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        if not any_acc: break
    model.load_state_dict(best_sd); return model, best_val

def adp_depth_only(model, tr, va, tcfg, ccfg, scfg):
    best_val, best_sd = _run(model, tr, va, tcfg, ccfg)
    f=0
    while f < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
        prop=model.append_depth(); v,sd=_run(prop,tr,va,tcfg,ccfg)
        if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd
        else: f+=1
    model.load_state_dict(best_sd); return model, best_val

def adp_width_only(model, tr, va, tcfg, ccfg, scfg):
    best_val, best_sd = _run(model, tr, va, tcfg, ccfg)
    f=0
    while f < scfg.trials_width and model.base + 1 <= scfg.max_base:
        prop=model.widen_all(1); v,sd=_run(prop,tr,va,tcfg,ccfg)
        if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd
        else: f+=1
    model.load_state_dict(best_sd); return model, best_val

ADP_REGISTRY = {
    'w2d': adp_width_to_depth,
    'd2w': adp_depth_to_width,
    'alt_d': adp_alt_depth,
    'alt_w': adp_alt_width,
    'depth_only': adp_depth_only,
    'width_only': adp_width_only,
}


def build_model(in_channels=3, base=32, stages=3, blocks=1) -> UNetCM:
    return UNetCM(in_channels, base, stages, blocks)
