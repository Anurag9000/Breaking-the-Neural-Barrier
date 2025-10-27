"""
ADP Diffusion – Unified Model (Part A2: DDPM/DDIM)
Six ADP growth algorithms over a width/depth‑editable UNet trained with
DDPM‑style denoising objective. Optionally switch parameterization (eps/x0/v).

CLI is provided in the paired run file.
Design rules (BNB): single‑model, end‑to‑end; no EMA, no teacher/student.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Weight‑preserving transplant utils
# -----------------------------

def _copy_overlap_(dst: torch.nn.Parameter, src: torch.nn.Parameter):
    with torch.no_grad():
        sd, ss = list(dst.shape), list(src.shape)
        sl = [min(a, b) for a, b in zip(sd, ss)]
        slices = tuple(slice(0, n) for n in sl)
        dst.zero_()
        dst[slices].copy_(src[slices])

def transplant_module(dst: nn.Module, src: nn.Module):
    with torch.no_grad():
        for (dn, dpar), (sn, spar) in zip(dst.named_parameters(), src.named_parameters()):
            try: _copy_overlap_(dpar, spar)
            except Exception: pass
        for (dn, db), (sn, sb) in zip(dst.named_buffers(), src.named_buffers()):
            if db.shape == sb.shape:
                db.copy_(sb)

# -----------------------------
# UNet backbone (same growth ops as Part A1)
# -----------------------------

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBNAct(c, c)
        self.c2 = ConvBNAct(c, c)
    def forward(self, x):
        return x + self.c2(self.c1(x))

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ConvBNAct(cin, cout, k=3, s=2, p=1)
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.post = ConvBNAct(cout, cout)
    def forward(self, x):
        return self.post(self.up(x))

class TimeEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, c)
        self.act = nn.SiLU()
    def forward(self, t):
        h = t.view(-1,1)
        h = self.act(self.fc1(h))
        return self.fc2(h)

class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=3, base=32, stages=3, blocks_per_stage=1, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.base = base
        self.stages = stages
        self.blocks_per_stage = blocks_per_stage
        self.out_channels = out_channels or in_channels

        self.time_emb = TimeEmbed(base)
        self.in_conv = ConvBNAct(in_channels, base)

        chans = [base*(2**i) for i in range(stages)]
        self.downs = nn.ModuleList()
        self.res_down = nn.ModuleList()
        c = base
        for i in range(stages):
            self.res_down.append(nn.Sequential(*[ResBlock(c) for _ in range(blocks_per_stage)]))
            if i < stages - 1:
                self.downs.append(Down(c, c*2))
                c *= 2
            else:
                self.downs.append(nn.Identity())

        self.ups = nn.ModuleList()
        self.res_up = nn.ModuleList()
        for i in reversed(range(stages)):
            self.res_up.append(nn.Sequential(*[ResBlock(chans[i]) for _ in range(blocks_per_stage)]))
            if i > 0:
                self.ups.append(Up(chans[i], chans[i-1]))
            else:
                self.ups.append(nn.Identity())
        self.out = nn.Conv2d(base, self.out_channels, 1)

    # growth
    def append_depth(self):
        new = UNetDenoiser(self.in_channels, self.base, self.stages, self.blocks_per_stage+1, self.out_channels)
        transplant_module(new, self)
        return new
    def widen_all(self, delta: int):
        new = UNetDenoiser(self.in_channels, self.base+delta, self.stages, self.blocks_per_stage, self.out_channels)
        transplant_module(new, self)
        return new

    def forward(self, x, t):
        emb = self.time_emb(t)
        h = self.in_conv(x)
        h = h + emb.view(emb.size(0), -1, 1, 1)
        skips = []
        c = h
        for i in range(self.stages):
            c = self.res_down[i](c)
            skips.append(c)
            c = self.downs[i](c)
        for i in range(self.stages):
            j = self.stages-1-i
            c = self.res_up[i](c + skips[j])
            c = self.ups[i](c)
        return self.out(c)

# -----------------------------
# DDPM schedules + objective
# -----------------------------

@dataclass
class DDPMCfg:
    T: int = 1000
    schedule: str = 'cosine'  # 'linear'|'cosine'|'sqrt'
    param: str = 'eps'         # 'eps'|'x0'|'v'
    device: str = 'cuda'

    def build(self):
        if self.schedule == 'linear':
            beta_start, beta_end = 1e-4, 2e-2
            betas = torch.linspace(beta_start, beta_end, self.T)
        elif self.schedule == 'sqrt':
            betas = torch.linspace(0, 1, self.T)
            betas = (betas**2) * 0.02 + 1e-4
        else:  # cosine
            s = 0.008
            t = torch.linspace(0, self.T, self.T+1) / self.T
            alphas_cum = torch.cos((t + s) / (1 + s) * math.pi/2) ** 2
            alphas_cum = alphas_cum / alphas_cum[0]
            betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
            betas = betas.clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return betas.to(self.device), alphas.to(self.device), alphas_cumprod.to(self.device)


def q_sample(x0: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor, eps: torch.Tensor):
    ac = alphas_cumprod[t]
    return torch.sqrt(ac).view(-1,1,1,1) * x0 + torch.sqrt(1-ac).view(-1,1,1,1) * eps


def target_from_param(param: str, x0, eps, t, alphas_cumprod):
    ac = alphas_cumprod[t].view(-1,1,1,1)
    if param == 'eps':
        return eps
    elif param == 'x0':
        return x0
    else:  # v parameterization
        return torch.sqrt(ac) * eps - torch.sqrt(1-ac) * x0


def ddpm_loss(model: UNetDenoiser, x0: torch.Tensor, cfg: DDPMCfg, betas, alphas, alphas_cumprod, rng: torch.Generator):
    B = x0.size(0)
    t = torch.randint(0, cfg.T, (B,), device=x0.device, generator=rng)
    eps = torch.randn_like(x0, generator=rng)
    xt = q_sample(x0, t, alphas_cumprod, eps)
    tt = (t.float() + 1) / cfg.T  # scale to (0,1]
    pred = model(xt, tt)
    target = target_from_param(cfg.param, x0, eps, t, alphas_cumprod)
    return F.mse_loss(pred, target)

# -----------------------------
# Trainer + Early stop (shared with Part A1 style)
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
        self.patience = patience
        self.best = math.inf
        self.bad = 0
        self.chk: Optional[Dict[str, torch.Tensor]] = None
    def step(self, value: float, model: nn.Module):
        if value < self.best:
            self.best = value
            self.bad = 0
            self.chk = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.bad += 1
            return False
    def should_stop(self):
        return self.bad > self.patience


def train_one(model: UNetDenoiser, train_loader, val_loader, tcfg: TrainCfg, dcfg: DDPMCfg):
    torch.manual_seed(tcfg.seed)
    device = torch.device(tcfg.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(tcfg.seed + 123)

    betas, alphas, ac = dcfg.build()

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    stopper = EarlyStopper(tcfg.patience)
    for epoch in range(tcfg.max_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss = ddpm_loss(model, x, dcfg, betas, alphas, ac, rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
        # val
        model.eval(); vloss = 0.0; n=0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                vloss += ddpm_loss(model, x, dcfg, betas, alphas, ac, rng).item() * x.size(0)
                n += x.size(0)
        vloss /= max(n,1)
        stopper.step(vloss, model)
        if stopper.should_stop():
            break
    if stopper.chk is not None:
        model.load_state_dict(stopper.chk)
    return stopper.best, {k: v.clone() for k,v in model.state_dict().items()}

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


def accept(new_val, old_val, delta):
    return new_val < (old_val - delta)


def _run_train(prop, tr, va, tcfg, dcfg):
    return train_one(prop, tr, va, tcfg, dcfg)


def adp_width_to_depth(model, tr, va, tcfg, dcfg, scfg):
    best_val, best_sd = _run_train(model, tr, va, tcfg, dcfg)
    w_fail = d_fail = 0
    while True:
        improved = False
        while w_fail < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop = model.widen_all(1)
            v, sd = _run_train(prop, tr, va, tcfg, dcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd; improved=True
            else: w_fail += 1
        if not improved: break
        improved=False
        while d_fail < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop = model.append_depth()
            v, sd = _run_train(prop, tr, va, tcfg, dcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd; improved=True
            else: d_fail += 1
        if not improved: break
    model.load_state_dict(best_sd); return model, best_val


def adp_depth_to_width(model, tr, va, tcfg, dcfg, scfg):
    best_val, best_sd = _run_train(model, tr, va, tcfg, dcfg)
    d_fail = w_fail = 0
    while True:
        improved=False
        while d_fail < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop = model.append_depth(); v, sd = _run_train(prop, tr, va, tcfg, dcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd; improved=True
            else: d_fail += 1
        if not improved: break
        improved=False
        while w_fail < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop = model.widen_all(1); v, sd = _run_train(prop, tr, va, tcfg, dcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd; improved=True
            else: w_fail += 1
        if not improved: break
    model.load_state_dict(best_sd); return model, best_val


def adp_alt_depth(model, tr, va, tcfg, dcfg, scfg):
    best_val, best_sd = _run_train(model, tr, va, tcfg, dcfg)
    while True:
        any_acc=False
        # depth phase
        f=0
        while f < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop=model.append_depth(); v,sd=_run_train(prop,tr,va,tcfg,dcfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        # width phase
        f=0
        while f < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop=model.widen_all(1); v,sd=_run_train(prop,tr,va,tcfg,dcfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        if not any_acc: break
    model.load_state_dict(best_sd); return model, best_val


def adp_alt_width(model, tr, va, tcfg, dcfg, scfg):
    best_val, best_sd = _run_train(model, tr, va, tcfg, dcfg)
    while True:
        any_acc=False
        # width
        f=0
        while f < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop=model.widen_all(1); v,sd=_run_train(prop,tr,va,tcfg,dcfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        # depth
        f=0
        while f < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop=model.append_depth(); v,sd=_run_train(prop,tr,va,tcfg,dcfg)
            if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd; any_acc=True
            else: f+=1
        if not any_acc: break
    model.load_state_dict(best_sd); return model, best_val


def adp_depth_only(model, tr, va, tcfg, dcfg, scfg):
    best_val, best_sd = _run_train(model, tr, va, tcfg, dcfg)
    f=0
    while f < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
        prop=model.append_depth(); v,sd=_run_train(prop,tr,va,tcfg,dcfg)
        if accept(v,best_val,scfg.delta): model,best_val,best_sd=prop,v,sd
        else: f+=1
    model.load_state_dict(best_sd); return model, best_val


def adp_width_only(model, tr, va, tcfg, dcfg, scfg):
    best_val, best_sd = _run_train(model, tr, va, tcfg, dcfg)
    f=0
    while f < scfg.trials_width and model.base + 1 <= scfg.max_base:
        prop=model.widen_all(1); v,sd=_run_train(prop,tr,va,tcfg,dcfg)
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


def build_model(in_channels=3, base=32, stages=3, blocks=1, out_channels=None) -> UNetDenoiser:
    return UNetDenoiser(in_channels, base, stages, blocks, out_channels)
