# ============================================================
# File: adp_diff_tasks.py  (MODEL)
# Single-model DDPM (ε-pred) with multi-task supervision:
#   --task {inpaint, sr, control, translate, segcond, regress}
# Conditioning is concatenated as extra channels (learned-free),
# all while supporting the 6 ADP policies over the denoiser.
# No EMA/teacher. Compact reference implementation.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Cosine schedule
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    a_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999)


# -----------------------------
# Adaptive U-Net with time injection and variable input channels
# -----------------------------

class TimeMLP(nn.Module):
    def __init__(self, time_ch=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, time_ch), nn.SiLU(), nn.Linear(time_ch, time_ch))
        self.time_ch = time_ch
    def forward(self, t: torch.Tensor):
        if t.dim() == 1:
            t = t[:, None]
        return self.net(t)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class AdaptiveUNetCond(nn.Module):
    """U-Net-like denoiser whose stage widths & depth can mutate (ADP).
       Accepts extra conditioning channels via input concatenation.
    """
    def __init__(self, in_ch: int, cond_ch: int, widths: List[int], time_ch: int = 128):
        super().__init__()
        self.widths = list(widths)
        self.cond_ch = int(cond_ch)
        self.time = TimeMLP(time_ch)
        self._build(in_ch + cond_ch)

    def _build(self, total_in: int):
        w = self.widths; T = self.time.time_ch
        self.num = len(w)
        self.down, self.pool = nn.ModuleList([]), nn.ModuleList([])
        ch = total_in
        for wi in w:
            self.down += [ConvBNAct(ch + T, wi), ConvBNAct(wi, wi)]
            self.pool += [nn.AvgPool2d(2)]
            ch = wi
        self.mid1, self.mid2 = ConvBNAct(ch + T, ch), ConvBNAct(ch, ch)
        self.up, self.upx = nn.ModuleList([]), nn.ModuleList([])
        up_ch = ch
        for wi in reversed(w):
            self.upx += [nn.ConvTranspose2d(up_ch, up_ch, 4, 2, 1)]
            self.up += [ConvBNAct(up_ch + wi + T, wi), ConvBNAct(wi, wi)]
            up_ch = wi
        self.head = nn.Conv2d(up_ch, 3, 1)  # predict eps in RGB space by default

    def _inj(self, x, t_emb):
        B, C, H, W = x.shape
        return torch.cat([x, t_emb[:, :, None, None].expand(B, t_emb.size(1), H, W)], dim=1)

    def forward(self, x_and_cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time(t)
        feats = []
        cur = x_and_cond
        for i in range(self.num):
            cur = self.down[2*i](self._inj(cur, te)); cur = self.down[2*i+1](cur)
            feats.append(cur); cur = self.pool[i](cur)
        cur = self.mid1(self._inj(cur, te)); cur = self.mid2(cur)
        for i in range(self.num):
            skip = feats[-(i+1)]
            cur = self.upx[i](cur)
            if cur.size(-1) != skip.size(-1):
                cur = F.interpolate(cur, size=skip.shape[-2:], mode='nearest')
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
        last_w = self.widths[-1]
        self.widths.append(last_w)
        # rebuild with same input channel count
        in_total = self.down[0].conv.in_channels - self.time.time_ch
        self._build(in_total)
    def widen_all(self, ex_k: int):
        old = self.state_dict(); self.widths = [w + ex_k for w in self.widths]
        in_total = self.down[0].conv.in_channels - self.time.time_ch
        self._build(in_total)
        new = self.state_dict()
        for k in new:
            if k in old:
                src, dst = old[k], new[k]
                com = tuple(min(a,b) for a,b in zip(src.shape, dst.shape))
                sl = tuple(slice(0,c) for c in com)
                dst[sl] = src[sl]
        self.load_state_dict(new, strict=False)


# -----------------------------
# Multi-Task DDPM (ε-pred) with concatenated conditioning
# -----------------------------

TASKS = {'inpaint','sr','control','translate','segcond','regress'}

class TaskDDPMSingleModel(nn.Module):
    def __init__(self,
                 task: str = 'inpaint',
                 img_ch: int = 3,
                 cond_ch: int = 4,  # default placeholder; runner will compute per-task
                 widths: List[int] = [32,64,96],
                 T: int = 1000,
                 aux_reg_dim: int = 0,      # >0 enables an auxiliary regression head
                 lambda_aux: float = 0.1):
        super().__init__()
        assert task in TASKS
        self.task = task
        self.T = int(T)
        self.aux_reg_dim = int(aux_reg_dim)
        self.lambda_aux = float(lambda_aux)
        self.register_buffer('betas', cosine_beta_schedule(T))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.net = AdaptiveUNetCond(in_ch=img_ch, cond_ch=cond_ch, widths=widths, time_ch=128)
        if aux_reg_dim > 0:
            # simple global reg head over mid activations: we tap via a conv on input
            self.reg = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(widths[-1], max(64, widths[-1]//2)), nn.SiLU(),
                nn.Linear(max(64, widths[-1]//2), aux_reg_dim)
            )
        else:
            self.reg = None

    # diffusion helpers
    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None: eps = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t_idx][:, None, None, None]
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps
        return x_t, eps

    def _pack_input(self, x_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_t, cond], dim=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, aux_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0); dev = x.device
        t_idx = torch.randint(0, self.T, (B,), device=dev, dtype=torch.long)
        t_norm = (t_idx.float() + 0.5) / self.T
        x_t, eps = self.q_sample(x, t_idx)
        x_in = self._pack_input(x_t, cond)
        eps_pred = self.net(x_in, t_norm)
        loss = 0.5 * F.mse_loss(eps_pred, eps)
        if self.reg is not None and aux_target is not None:
            # crude mid-feature: reuse eps_pred as a proxy (lighter than tapping internal blocks)
            reg_pred = self.reg(eps_pred)
            loss = loss + self.lambda_aux * 0.5 * F.mse_loss(reg_pred, aux_target)
        return loss

    @torch.no_grad()
    def sample(self, B: int, img_ch: int, H: int, W: int, cond: torch.Tensor,
               steps: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None: device = next(self.parameters()).device
        if steps is None: steps = self.T
        x = torch.randn(B, img_ch, H, W, device=device)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = (t.float() + 0.5) / self.T
            x_in = self._pack_input(x, cond)
            eps = self.net(x_in, t_norm)
            beta = self.betas[i]; alpha = self.alphas[i]; a_bar = self.alphas_cumprod[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1.0/torch.sqrt(alpha)) * (x - beta/torch.sqrt(1.0 - a_bar) * eps) + torch.sqrt(beta) * noise
        return x.clamp(-1, 1)

    # ADP passthrough
    def neurons(self) -> int: return self.net.neurons()
    def snapshot_state(self): return {'net': self.net.snapshot_state(), 'task': self.task}
    def restore_state(self, snap): self.net.restore_state(snap['net'])
    def append_depth(self): self.net.append_depth()
    def widen_all(self, ex_k: int): self.net.widen_all(ex_k)


# -----------------------------
# Early-stopping trainer and ADP search policies
# -----------------------------

@dataclass
class TrainCfg:
    lr: float = 2e-4
    max_epochs: int = 50
    es_patience: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_model(model: TaskDDPMSingleModel, train_loader, val_loader, cfg: TrainCfg) -> Tuple[float, Dict]:
    device = torch.device(cfg.device); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best = float('inf'); best_snap = None; bad = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                x, cond, aux = batch
                aux = aux.to(device)
            else:
                x, cond = batch; aux = None
            x = x.to(device); cond = cond.to(device)
            loss = model(x, cond, aux)
            opt.zero_grad(set_to_none=True); loss.backward()
            if cfg.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        # val
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, cond, aux = batch; aux = aux.to(device)
                else:
                    x, cond = batch; aux = None
                x = x.to(device); cond = cond.to(device)
                l = model(x, cond, aux)
                tot += float(l.item()) * x.size(0); n += x.size(0)
        val = tot / max(1, n)
        if val + 1e-8 < best:
            best = val; best_snap = {'model': model.state_dict()}; bad = 0
        else:
            bad += 1
        if bad >= cfg.es_patience: break
    if best_snap is not None: model.load_state_dict(best_snap['model'])
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


def adp_depth_then_width(model: TaskDDPMSingleModel, train_loader, val_loader, tr: TrainCfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; d_fail = 0
        else: model.restore_state(pre); d_fail += 1; if d_fail >= 2: break
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; w_fail = 0
        else: model.restore_state(pre); w_fail += 1; if w_fail >= 2: break
    return base


def adp_width_then_depth(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; w_fail = 0
        else: model.restore_state(pre); w_fail += 1; if w_fail >= 2: break
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; d_fail = 0
        else: model.restore_state(pre); d_fail += 1; if d_fail >= 2: break
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
        else: model.restore_state(pre); fail += 1; if fail >= 2: break
    return base


def adp_width_only(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons: model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta): base = v; fail = 0
        else: model.restore_state(pre); fail += 1; if fail >= 2: break
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
# End of adp_diff_tasks.py
# ============================================================
