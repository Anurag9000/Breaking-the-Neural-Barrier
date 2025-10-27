# ============================================================
# File: adp_diff_domains.py  (MODEL)
# Single-model DDPM (ε-pred) generalized across domains:
#   --domain {1d,2d,3d,video,audio}
# A single adaptive U-Net backbone implemented in N-D (1D/2D/3D),
# enabling all 6 ADP policies (depth→width, width→depth, alt-depth,
# alt-width, depth-only, width-only). No EMA/teacher.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    a_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999)


def nd_layers(ndims: int):
    assert ndims in (1,2,3)
    Conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[ndims-1]
    TConv = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[ndims-1]
    BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[ndims-1]
    Pool = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[ndims-1]
    return Conv, TConv, BN, Pool


def up_kernel(ndims: int):
    return (4,) * ndims, (2,) * ndims, (1,) * ndims


class TimeMLP(nn.Module):
    def __init__(self, time_ch=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, time_ch), nn.SiLU(), nn.Linear(time_ch, time_ch))
        self.time_ch = time_ch
    def forward(self, t: torch.Tensor):
        if t.dim() == 1:
            t = t[:, None]
        return self.net(t)


def expand_time_nd(t_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    B = x.size(0); T = t_emb.size(1)
    # Expand across spatial dims
    expand_shape = [B, T] + [1]*(x.dim()-2)
    tile = t_emb.view(*expand_shape)
    mul = []
    for d in x.shape[2:]:
        mul.append(d)
    return tile.expand(B, T, *x.shape[2:])


class ConvBNActND(nn.Module):
    def __init__(self, ndims: int, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act=nn.SiLU):
        super().__init__()
        Conv, _, BN, _ = nd_layers(ndims)
        self.conv = Conv(in_ch, out_ch, k, s, p)
        self.bn = BN(out_ch)
        self.act = act(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AdaptiveUNetND(nn.Module):
    """U-Net-like denoiser generalized to 1D/2D/3D.
       Width growth = per-stage channels; Depth growth = #stages.
    """
    def __init__(self, ndims: int, in_ch: int, widths: List[int], time_ch: int = 128):
        super().__init__()
        self.ndims = ndims
        self.widths = list(widths)
        self.time = TimeMLP(time_ch)
        self._build(in_ch)

    def _build(self, in_ch: int):
        Conv, TConv, _, Pool = nd_layers(self.ndims)
        k_up, s_up, p_up = up_kernel(self.ndims)
        w = self.widths; T = self.time.time_ch
        self.num = len(w)
        self.down, self.pool = nn.ModuleList([]), nn.ModuleList([])
        ch = in_ch
        for wi in w:
            self.down += [ConvBNActND(self.ndims, ch + T, wi), ConvBNActND(self.ndims, wi, wi)]
            self.pool += [Pool(2)]
            ch = wi
        self.mid1, self.mid2 = ConvBNActND(self.ndims, ch + T, ch), ConvBNActND(self.ndims, ch, ch)
        self.up, self.upx = nn.ModuleList([]), nn.ModuleList([])
        up_ch = ch
        for wi in reversed(w):
            self.upx += [TConv(up_ch, up_ch, k_up, s_up, p_up)]
            self.up += [ConvBNActND(self.ndims, up_ch + wi + T, wi), ConvBNActND(self.ndims, wi, wi)]
            up_ch = wi
        ConvOut, _, _, _ = nd_layers(self.ndims)
        self.head = ConvOut(up_ch, in_ch, 1)

    def _inj(self, x: torch.Tensor, t_emb: torch.Tensor):
        return torch.cat([x, expand_time_nd(t_emb, x)], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time(t)
        feats = []
        cur = x
        for i in range(self.num):
            cur = self.down[2*i](self._inj(cur, te)); cur = self.down[2*i+1](cur)
            feats.append(cur); cur = self.pool[i](cur)
        cur = self.mid1(self._inj(cur, te)); cur = self.mid2(cur)
        for i in range(self.num):
            skip = feats[-(i+1)]
            cur = self.upx[i](cur)
            # simple size fix for odd shapes
            if cur.shape[2:] != skip.shape[2:]:
                cur = F.interpolate(cur, size=skip.shape[2:], mode='nearest')
            cur = torch.cat([cur, skip], dim=1)
            cur = self.up[2*i](self._inj(cur, te)); cur = self.up[2*i+1](cur)
        return self.head(cur)

    # ADP API
    def neurons(self) -> int:
        return int(sum(self.widths))
    def snapshot_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
    def restore_state(self, snap: Dict[str, torch.Tensor]):
        self.load_state_dict(snap, strict=True)
    def append_depth(self):
        self.widths.append(self.widths[-1])
        self._build(self.head.out_channels)
    def widen_all(self, ex_k: int):
        old = self.state_dict()
        self.widths = [w + ex_k for w in self.widths]
        self._build(self.head.out_channels)
        new = self.state_dict()
        for k in new:
            if k in old:
                src, dst = old[k], new[k]
                common = tuple(min(a,b) for a,b in zip(src.shape, dst.shape))
                sl = tuple(slice(0,c) for c in common)
                dst[sl] = src[sl]
        self.load_state_dict(new, strict=False)


# -----------------------------
# Domain map
# -----------------------------

DOMAIN_CFG = {
    '1d':   {'ndims': 1, 'in_ch': 1},         # e.g., time-series
    '2d':   {'ndims': 2, 'in_ch': 3},         # images
    '3d':   {'ndims': 3, 'in_ch': 1},         # volumes (grayscale by default)
    'video':{'ndims': 3, 'in_ch': 3},         # (C,T,H,W) → treat T as a spatial axis for 3D convs
    'audio':{'ndims': 2, 'in_ch': 1},         # spectrograms (2D)
}


class EpsDDPMND(nn.Module):
    def __init__(self, domain: str = '2d', widths: List[int] = [32,64,96], T: int = 1000):
        super().__init__()
        assert domain in DOMAIN_CFG, f"Unknown domain {domain}"
        self.domain = domain
        self.cfg = DOMAIN_CFG[domain]
        self.ndims = self.cfg['ndims']
        self.in_ch = self.cfg['in_ch']
        self.T = int(T)
        self.register_buffer('betas', cosine_beta_schedule(T))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.net = AdaptiveUNetND(self.ndims, self.in_ch, widths, time_ch=128)

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None: eps = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t_idx]
        while a_bar.dim() < x0.dim():
            a_bar = a_bar.unsqueeze(-1)
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps
        return x_t, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0); dev = x.device
        t_idx = torch.randint(0, self.T, (B,), device=dev, dtype=torch.long)
        t_norm = (t_idx.float() + 0.5) / self.T
        x_t, eps = self.q_sample(x, t_idx)
        eps_pred = self.net(x_t, t_norm)
        return 0.5 * F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, B: int, shape: Tuple[int, ...], steps: Optional[int] = None, device: Optional[torch.device] = None):
        """shape excludes batch; must match domain channel+spatial dims.
        1d:   (C, L)
        2d:   (C, H, W)
        3d:   (C, D, H, W)
        video:(C, T, H, W)  (treated as 3D)
        audio:(C, H, W)
        """
        if device is None: device = next(self.parameters()).device
        if steps is None: steps = self.T
        x = torch.randn(B, *shape, device=device)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = (t.float() + 0.5) / self.T
            eps = self.net(x, t_norm)
            beta = self.betas[i]; alpha = self.alphas[i]; a_bar = self.alphas_cumprod[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1.0/torch.sqrt(alpha)) * (x - beta/torch.sqrt(1.0 - a_bar) * eps) + torch.sqrt(beta) * noise
        return x.clamp(-1, 1)

    # ADP passthrough
    def neurons(self) -> int: return self.net.neurons()
    def snapshot_state(self): return {'net': self.net.state_dict()}
    def restore_state(self, snap): self.net.load_state_dict(snap['net'])
    def append_depth(self): self.net.append_depth()
    def widen_all(self, ex_k: int): self.net.widen_all(ex_k)


# -----------------------------
# Early-stopping trainer & ADP policies (shared)
# -----------------------------

@dataclass
class TrainCfg:
    lr: float = 2e-4
    max_epochs: int = 50
    es_patience: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_model(model: EpsDDPMND, train_loader, val_loader, cfg: TrainCfg):
    device = torch.device(cfg.device); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best = float('inf'); best_snap = None; bad = 0
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
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                l = model(x)
                tot += float(l.item()) * x.size(0); n += x.size(0)
        val = tot / max(1, n)
        if val + 1e-8 < best:
            best = val; best_snap = {'model': model.state_dict()}; bad = 0
        else:
            bad += 1
        if bad >= cfg.es_patience:
            break
    if best_snap is not None:
        model.load_state_dict(best_snap['model'])
    return best, best_snap


@dataclass
class SearchCfg:
    delta: float = 0.0
    trials_width: int = 50
    trials_depth: int = 50
    ex_k: int = 8
    max_neurons: Optional[int] = None


def _accept(v_improved: float, v_base: float, d: float) -> bool:
    return v_improved < (v_base - d)


def adp_depth_then_width(model: EpsDDPMND, train_loader, val_loader, tr: TrainCfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; d_fail = 0
        else:
            model.restore_state(pre); d_fail += 1
            if d_fail >= 2: break
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; w_fail = 0
        else:
            model.restore_state(pre); w_fail += 1
            if w_fail >= 2: break
    return base


def adp_width_then_depth(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; w_fail = 0
        else:
            model.restore_state(pre); w_fail += 1
            if w_fail >= 2: break
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; d_fail = 0
        else:
            model.restore_state(pre); d_fail += 1
            if d_fail >= 2: break
    return base


def adp_alt_depth_first(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    while True:
        improved = False
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; improved = True
        else: model.restore_state(pre)
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; improved = True
        else: model.restore_state(pre)
        if not improved: break
    return base


def adp_alt_width_first(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    while True:
        improved = False
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; improved = True
        else: model.restore_state(pre)
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; improved = True
        else: model.restore_state(pre)
        if not improved: break
    return base


def adp_depth_only(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; fail = 0
        else: model.restore_state(pre); fail += 1
        if fail >= 2: break
    return base


def adp_width_only(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; fail = 0
        else: model.restore_state(pre); fail += 1
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
# End of adp_diff_domains.py
# ============================================================
