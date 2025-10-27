# ============================================================
# File: adp_diff_sde.py  (MODEL)
# Single-model, score-based diffusion (VP-SDE) with integrated
# 6x ADP search policies (depth→width, width→depth, alt-depth,
# alt-width, depth-only, width-only). No EMA/teacher.
# ============================================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# VP-SDE schedule utilities
# -----------------------------

@torch.jit.script
def vp_alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """ᾱ(t) = exp(-(β_min t + 0.5 (β_max-β_min) t^2)) with t in [0,1]."""
    return torch.exp(-(beta_min * t + 0.5 * (beta_max - beta_min) * t * t))

@torch.jit.script
def vp_beta_t(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """β(t) = β_min + t(β_max-β_min)."""
    return beta_min + t * (beta_max - beta_min)


# -----------------------------
# Blocks and adaptive U-Net backbone (width/depthatable)
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class AdaptiveUNet(nn.Module):
    """Minimal U-Net-like denoiser whose stage widths and depth can grow on the fly."""
    def __init__(self, in_ch: int, widths: List[int], time_ch: int = 128):
        super().__init__()
        self.widths = list(widths)
        self.time_ch = time_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_ch), nn.SiLU(), nn.Linear(time_ch, time_ch)
        )
        self._build(in_ch)

    # --- architecture (re)builder
    def _build(self, in_ch: int):
        wlist = self.widths
        self.num_stages = len(wlist)
        downs, pools = [], []
        ch = in_ch
        for w in wlist:
            downs += [ConvBNReLU(ch + self.time_ch, w), ConvBNReLU(w, w)]
            pools += [nn.AvgPool2d(2)]
            ch = w
        self.down_blocks = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)

        self.mid1 = ConvBNReLU(ch + self.time_ch, ch)
        self.mid2 = ConvBNReLU(ch, ch)

        ups, upsamp = [], []
        up_ch = ch
        for w in reversed(wlist):
            ups += [ConvBNReLU(up_ch + w + self.time_ch, w), ConvBNReLU(w, w)]
            upsamp += [nn.ConvTranspose2d(up_ch, up_ch, 4, 2, 1)]
            up_ch = w
        self.up_blocks = nn.ModuleList(ups)
        self.upsample = nn.ModuleList(upsamp)

        self.out_conv = nn.Conv2d(up_ch, in_ch, 1)

    # --- helpers
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return self.time_mlp(t)

    def _cat_time(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        tmap = t_emb[:, :, None, None].expand(B, t_emb.size(1), H, W)
        return torch.cat([x, tmap], dim=1)

    # --- forward: outputs the SCORE field (∂/∂x log p_t(x))
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self._time_embed(t)
        feats = []
        cur = x
        for i in range(self.num_stages):
            cur = self.down_blocks[2*i](self._cat_time(cur, t_emb))
            cur = self.down_blocks[2*i + 1](cur)
            feats.append(cur)
            cur = self.pools[i](cur)
        cur = self.mid1(self._cat_time(cur, t_emb))
        cur = self.mid2(cur)
        for i in range(self.num_stages):
            skip = feats[-(i+1)]
            cur = self.upsample[i](cur)
            if cur.size(-1) != skip.size(-1):
                cur = F.interpolate(cur, size=skip.shape[-2:], mode='nearest')
            cur = torch.cat([cur, skip], dim=1)
            cur = self.up_blocks[2*i](self._cat_time(cur, t_emb))
            cur = self.up_blocks[2*i + 1](cur)
        return self.out_conv(cur)

    # --- ADP API
    def neurons(self) -> int:
        return int(sum(self.widths))
    def snapshot_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
    def restore_state(self, snap: Dict[str, torch.Tensor]):
        self.load_state_dict(snap, strict=True)
    def append_depth(self):
        last_w = self.widths[-1]
        self.widths.append(last_w)
        self._build(in_ch=self.out_conv.out_channels)
    def widen_all(self, ex_k: int):
        self.widths = [w + ex_k for w in self.widths]
        # transpose overlapping weights by reconstructing and copying
        old = self.state_dict()
        self._build(in_ch=self.out_conv.out_channels)
        new = self.state_dict()
        for k in new.keys():
            if k in old:
                src, dst = old[k], new[k]
                common = tuple(min(a, b) for a, b in zip(src.shape, dst.shape))
                slices = tuple(slice(0, c) for c in common)
                dst[slices] = src[slices]
        self.load_state_dict(new, strict=False)


# -----------------------------
# Single-model SDE (score parameterization)
# -----------------------------

class ScoreSDESingleModel(nn.Module):
    def __init__(self,
                 img_ch: int = 3,
                 widths: List[int] = [32, 64, 96],
                 beta_min: float = 0.1,
                 beta_max: float = 20.0):
        super().__init__()
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.denoiser = AdaptiveUNet(in_ch=img_ch, widths=widths, time_ch=128)

    # ---- perturb x0 -> x_t and true score target
    def perturb(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns x_t, eps, sigma(t) with x_t = sqrt(ᾱ) x0 + σ ε;  σ = sqrt(1-ᾱ).
        """
        alpha_bar = vp_alpha_bar(t, self.beta_min, self.beta_max)
        sigma = torch.sqrt(1.0 - alpha_bar)
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt()[:, None, None, None] * x0 + sigma[:, None, None, None] * eps
        return xt, eps, sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        t = torch.rand(B, device=device)
        xt, eps, sigma = self.perturb(x, t)
        target_score = -eps / sigma[:, None, None, None]
        pred_score = self.denoiser(xt, t)
        return 0.5 * F.mse_loss(pred_score, target_score)

    # ---- simple reverse-SDE sampler for visualization
    @torch.no_grad()
    def sample(self, B: int, img_ch: int, H: int, W: int, steps: int = 1000, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        x = torch.randn(B, img_ch, H, W, device=device)
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = t_grid[i].expand(B)
            dt = t_grid[i+1] - t_grid[i]  # negative
            beta_t = vp_beta_t(t, self.beta_min, self.beta_max)[:, None, None, None]
            score = self.denoiser(x, t)
            drift = -0.5 * beta_t * x - beta_t * score
            diffusion = torch.sqrt(torch.clamp(beta_t * (-dt), 1e-12))  # |dt|
            noise = torch.randn_like(x) if i < steps-1 else torch.zeros_like(x)
            x = x + drift * dt + diffusion * noise
        return torch.clamp(x, -1, 1)

    # ---- ADP pass-through
    def neurons(self) -> int:
        return self.denoiser.neurons()
    def snapshot_state(self):
        return {'denoiser': self.denoiser.snapshot_state()}
    def restore_state(self, snap):
        self.denoiser.restore_state(snap['denoiser'])
    def append_depth(self):
        self.denoiser.append_depth()
    def widen_all(self, ex_k: int):
        self.denoiser.widen_all(ex_k)


# -----------------------------
# Early-stopping trainer (shared)
# -----------------------------

@dataclass
class TrainCfg:
    lr: float = 2e-4
    max_epochs: int = 50
    es_patience: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_model(model: ScoreSDESingleModel, train_loader, val_loader, cfg: TrainCfg) -> Tuple[float, Dict]:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best = float('inf')
    best_snap = None
    bad = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss = model(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        # val
        model.eval()
        n, tot = 0, 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                l = model(x)
                tot += float(l.item()) * x.size(0)
                n += x.size(0)
        val = tot / max(1, n)
        if val + 1e-8 < best:
            best = val
            best_snap = {'model': model.state_dict()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.es_patience:
            break
    if best_snap is not None:
        model.load_state_dict(best_snap['model'])
    return best, best_snap


# -----------------------------
# ADP policies (6 variants)
# -----------------------------

@dataclass
class SearchCfg:
    delta: float = 0.0
    trials_width: int = 50
    trials_depth: int = 50
    ex_k: int = 8
    max_neurons: Optional[int] = None


def _accept(v_improved: float, v_base: float, delta: float) -> bool:
    return v_improved < (v_base - delta)


def adp_depth_then_width(model: ScoreSDESingleModel, train_loader, val_loader, tr_cfg: TrainCfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
    # depth series
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; d_fail = 0
        else:
            model.restore_state(pre); d_fail += 1
            if d_fail >= 2: break
    # width series
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; w_fail = 0
        else:
            model.restore_state(pre); w_fail += 1
            if w_fail >= 2: break
    return base


def adp_width_then_depth(model, train_loader, val_loader, tr_cfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; w_fail = 0
        else:
            model.restore_state(pre); w_fail += 1
            if w_fail >= 2: break
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; d_fail = 0
        else:
            model.restore_state(pre); d_fail += 1
            if d_fail >= 2: break
    return base


def adp_alt_depth_first(model, train_loader, val_loader, tr_cfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
    while True:
        improved = False
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        if not improved: break
    return base


def adp_alt_width_first(model, train_loader, val_loader, tr_cfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
    while True:
        improved = False
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        if not improved: break
    return base


def adp_depth_only(model, train_loader, val_loader, tr_cfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
    fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; fail = 0
        else:
            model.restore_state(pre); fail += 1
            if fail >= 2: break
    return base


def adp_width_only(model, train_loader, val_loader, tr_cfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
    fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr_cfg)
        if _accept(v, base, s.delta):
            base = v; fail = 0
        else:
            model.restore_state(pre); fail += 1
            if fail >= 2: break
    return base


POLICIES = {
    'depth2width': adp_depth_then_width,
    'width2depth': adp_width_then_depth,
    'alt_depth': adp_alt_depth_first,
    'alt_width': adp_alt_width_first,
    'depth_only': adp_depth_only,
    'width_only': adp_width_only,
}

# ============================================================
# End of adp_diff_sde.py
# ============================================================
