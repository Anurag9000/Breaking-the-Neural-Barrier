"""
ADP Diffusion – Unified Model (Part A1: Score/SDE)
Single file that implements SIX ADP growth algorithms over a UNet score network
for self‑supervised diffusion training using continuous SDE score matching (VP/VE).

Select with flags (see run file):
  --family score_sde            # (this file currently implements Part A1)
  --adp {w2d,d2w,alt_w,alt_d,depth_only,width_only}

Design notes (BNB standing rule):
- Single‑model end‑to‑end; NO teacher/student, NO EMA.
- Self‑supervised objective: denoising score matching under SDE (VP/VE)
- ADP growth ops preserve weights via overlap‑copy (conv/bn/linear)
- Width = channel multiplier of all stages; Depth = number of residual blocks
  per stage (shared across stages for simplicity)
"""
from __future__ import annotations
import math
import argparse
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------

def _copy_overlap_(dst: torch.nn.Parameter, src: torch.nn.Parameter):
    with torch.no_grad():
        sd = list(dst.shape)
        ss = list(src.shape)
        sl = [min(a, b) for a, b in zip(sd, ss)]
        slices = tuple(slice(0, n) for n in sl)
        dst.zero_()
        dst[slices].copy_(src[slices])


def transplant_module(dst: nn.Module, src: nn.Module):
    """Overlap‑copy for Conv/BN/Linear recursively where shapes permit."""
    for (dn, dparam), (sn, sparam) in zip(dst.named_parameters(), src.named_parameters()):
        try:
            _copy_overlap_(dparam, sparam)
        except Exception:
            pass
    for (dn, dbuf), (sn, sbuf) in zip(dst.named_buffers(), src.named_buffers()):
        if dbuf.shape == sbuf.shape:
            try:
                dbuf.copy_(sbuf)
            except Exception:
                pass

# -----------------------------
# Time / SDE setups
# -----------------------------

@dataclass
class SDECfg:
    kind: str  # 'vp' or 've'
    t_min: float = 1e-4
    t_max: float = 1.0

    # VP: beta(t) in [beta_min, beta_max]
    beta_min: float = 0.1
    beta_max: float = 20.0

    # VE: sigma(t) in [sigma_min, sigma_max]
    sigma_min: float = 0.01
    sigma_max: float = 50.0

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        if self.kind == 'vp':
            # sigma(t) for VP SDE (variance‑preserving)
            log_mean_coeff = -0.25 * (self.beta_max - self.beta_min) * t**2 - 0.5 * self.beta_min * t
            return torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        else:
            # VE SDE: sigma(t) grows exponentially between sigma_min and sigma_max
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def perturb(self, x0: torch.Tensor, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t ~ U[t_min, t_max], return (t, x_noisy, noise_scale)."""
        B = x0.size(0)
        t = torch.rand(B, device=x0.device, generator=rng) * (self.t_max - self.t_min) + self.t_min
        sigma = self.marginal_std(t)
        eps = torch.randn_like(x0, generator=rng)
        x = x0 + eps * sigma.view(-1, 1, 1, 1)
        return t, x, sigma

# -----------------------------
# UNet building blocks (compact, weight‑preserving under width/depth edits)
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
        # t in [0,1]; shape (B,)
        h = t.view(-1, 1)
        h = self.act(self.fc1(h))
        return self.fc2(h)

class UNetScore(nn.Module):
    """UNet whose width and depth are editable. Predicts noise (score proxy)."""
    def __init__(self, in_channels=3, base=32, stages=3, blocks_per_stage=1):
        super().__init__()
        self.in_channels = in_channels
        self.base = base
        self.stages = stages
        self.blocks_per_stage = blocks_per_stage

        self.time_emb = TimeEmbed(base)
        self.in_conv = ConvBNAct(in_channels, base)

        # Down path
        chans = [base * (2**i) for i in range(stages)]
        self.downs = nn.ModuleList()
        self.res_down = nn.ModuleList()
        c = base
        for i in range(stages):
            self.res_down.append(nn.Sequential(*[ResBlock(c) for _ in range(blocks_per_stage)]))
            if i < stages - 1:
                self.downs.append(Down(c, c * 2))
                c *= 2
            else:
                self.downs.append(nn.Identity())

        # Up path
        self.ups = nn.ModuleList()
        self.res_up = nn.ModuleList()
        for i in reversed(range(stages)):
            self.res_up.append(nn.Sequential(*[ResBlock(chans[i]) for _ in range(blocks_per_stage)]))
            if i > 0:
                self.ups.append(Up(chans[i], chans[i - 1]))
            else:
                self.ups.append(nn.Identity())

        self.out = nn.Conv2d(base, in_channels, 1)

    # --- growth ops ---
    def append_depth(self):
        """Add one ResBlock per stage (depth +1)."""
        new = UNetScore(self.in_channels, self.base, self.stages, self.blocks_per_stage + 1)
        transplant_module(new, self)
        return new

    def widen_all(self, delta: int):
        """Increase base channels by delta (applies to all stages due to powers of 2)."""
        new = UNetScore(self.in_channels, self.base + delta, self.stages, self.blocks_per_stage)
        transplant_module(new, self)
        return new

    # --- forward ---
    def forward(self, x, t):
        emb = self.time_emb(t)
        h = self.in_conv(x)
        # inject time via FiLM‑like scaling (simple add)
        h = h + emb.view(emb.size(0), -1, 1, 1)

        skips = []
        c = h
        for i in range(self.stages):
            c = self.res_down[i](c)
            skips.append(c)
            c = self.downs[i](c)

        for i in range(self.stages):
            j = self.stages - 1 - i
            c = self.res_up[i](c + skips[j])
            c = self.ups[i](c)

        return self.out(c)

# -----------------------------
# Training: SDE score matching (single‑model)
# -----------------------------

@dataclass
class TrainCfg:
    sde: SDECfg
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5
    grad_clip: float = 1.0
    device: str = 'cuda'
    seed: int = 42


def dsm_loss(model: UNetScore, x0: torch.Tensor, rng: torch.Generator, sde: SDECfg) -> torch.Tensor:
    t, xt, sigma = sde.perturb(x0, rng)
    pred = model(xt, t)
    target = (xt - x0) / sigma.view(-1, 1, 1, 1)
    return F.mse_loss(pred, target)


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


def train_one(model: UNetScore, train_loader, val_loader, cfg: TrainCfg) -> Tuple[float, Dict[str, torch.Tensor]]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience)
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed + 1)

    for epoch in range(cfg.max_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss = dsm_loss(model, x, rng, cfg.sde)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        # val
        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                vloss += dsm_loss(model, x, rng, cfg.sde).item() * x.size(0)
                n += x.size(0)
        vloss /= max(1, n)
        stopper.step(vloss, model)
        if stopper.should_stop():
            break
    # restore best
    if stopper.chk is not None:
        model.load_state_dict(stopper.chk)
    return stopper.best, {k: v.clone() for k, v in model.state_dict().items()}

# -----------------------------
# ADP search (6 algorithms)
# -----------------------------

@dataclass
class SearchCfg:
    max_depth: int = 8
    max_base: int = 192
    trials_width: int = 2
    trials_depth: int = 2
    delta: float = 1e-4  # minimum improvement


def neurons(model: UNetScore) -> int:
    return model.base * (2**model.stages - 1)  # coarse proxy


def accept(new_val, old_val, delta):
    return new_val < (old_val - delta)


def adp_width_to_depth(model, train_loader, val_loader, tcfg: TrainCfg, scfg: SearchCfg):
    best_val, best_sd = train_one(model, train_loader, val_loader, tcfg)
    width_fails = 0
    depth_fails = 0
    while True:
        # WIDTH series
        improved = False
        while width_fails < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop = model.widen_all(1)
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                improved = True
            else:
                width_fails += 1
        if not improved:
            break
        # DEPTH series
        improved = False
        while depth_fails < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop = model.append_depth()
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                improved = True
            else:
                depth_fails += 1
        if not improved:
            break
    model.load_state_dict(best_sd)
    return model, best_val


def adp_depth_to_width(model, train_loader, val_loader, tcfg: TrainCfg, scfg: SearchCfg):
    best_val, best_sd = train_one(model, train_loader, val_loader, tcfg)
    depth_fails = 0
    width_fails = 0
    while True:
        improved = False
        while depth_fails < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop = model.append_depth()
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                improved = True
            else:
                depth_fails += 1
        if not improved:
            break
        improved = False
        while width_fails < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop = model.widen_all(1)
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                improved = True
            else:
                width_fails += 1
        if not improved:
            break
    model.load_state_dict(best_sd)
    return model, best_val


def adp_alt_depth(model, train_loader, val_loader, tcfg: TrainCfg, scfg: SearchCfg):
    best_val, best_sd = train_one(model, train_loader, val_loader, tcfg)
    while True:
        any_accepted = False
        # depth phase
        accepted = False
        fails = 0
        while fails < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop = model.append_depth()
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                accepted = True
                any_accepted = True
            else:
                fails += 1
        # width phase
        accepted = False
        fails = 0
        while fails < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop = model.widen_all(1)
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                accepted = True
                any_accepted = True
            else:
                fails += 1
        if not any_accepted:
            break
    model.load_state_dict(best_sd)
    return model, best_val


def adp_alt_width(model, train_loader, val_loader, tcfg: TrainCfg, scfg: SearchCfg):
    best_val, best_sd = train_one(model, train_loader, val_loader, tcfg)
    while True:
        any_accepted = False
        # width phase
        fails = 0
        while fails < scfg.trials_width and model.base + 1 <= scfg.max_base:
            prop = model.widen_all(1)
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                any_accepted = True
            else:
                fails += 1
        # depth phase
        fails = 0
        while fails < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
            prop = model.append_depth()
            v, sd = train_one(prop, train_loader, val_loader, tcfg)
            if accept(v, best_val, scfg.delta):
                model, best_val, best_sd = prop, v, sd
                any_accepted = True
            else:
                fails += 1
        if not any_accepted:
            break
    model.load_state_dict(best_sd)
    return model, best_val


def adp_depth_only(model, train_loader, val_loader, tcfg: TrainCfg, scfg: SearchCfg):
    best_val, best_sd = train_one(model, train_loader, val_loader, tcfg)
    fails = 0
    while fails < scfg.trials_depth and model.blocks_per_stage + 1 <= scfg.max_depth:
        prop = model.append_depth()
        v, sd = train_one(prop, train_loader, val_loader, tcfg)
        if accept(v, best_val, scfg.delta):
            model, best_val, best_sd = prop, v, sd
        else:
            fails += 1
    model.load_state_dict(best_sd)
    return model, best_val


def adp_width_only(model, train_loader, val_loader, tcfg: TrainCfg, scfg: SearchCfg):
    best_val, best_sd = train_one(model, train_loader, val_loader, tcfg)
    fails = 0
    while fails < scfg.trials_width and model.base + 1 <= scfg.max_base:
        prop = model.widen_all(1)
        v, sd = train_one(prop, train_loader, val_loader, tcfg)
        if accept(v, best_val, scfg.delta):
            model, best_val, best_sd = prop, v, sd
        else:
            fails += 1
    model.load_state_dict(best_sd)
    return model, best_val


ADP_REGISTRY = {
    'w2d': adp_width_to_depth,
    'd2w': adp_depth_to_width,
    'alt_d': adp_alt_depth,
    'alt_w': adp_alt_width,
    'depth_only': adp_depth_only,
    'width_only': adp_width_only,
}


def build_model(in_channels=3, base=32, stages=3, blocks=1) -> UNetScore:
    return UNetScore(in_channels, base, stages, blocks)

