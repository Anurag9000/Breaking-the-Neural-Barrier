# ============================================================
# File: adp_diff_diffae.py  (MODEL)
# Single-model Diffusion–Autoencoder hybrid:
#  • One network containing: image encoder, latent denoiser (U-Net), image decoder
#  • Joint objective = image reconstruction + latent diffusion (ε-pred) loss
#  • No EMA/teacher, no external VAE — encoder/decoder are part of the same model
#  • Integrated 6x ADP policies over the latent denoiser (width & depth growth)
# ============================================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Cosine schedule (DDPM-style)
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    a_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999)


# -----------------------------
# Building blocks
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -----------------------------
# Internal Autoencoder (encoder+decoder)
# -----------------------------

class InternalAE(nn.Module):
    """A compact stride-2 encoder and symmetric decoder trained jointly."""
    def __init__(self, in_ch: int = 3, latent_ch: int = 4, levels: int = 2):
        super().__init__()
        enc, ch = [], in_ch
        for _ in range(levels):
            enc += [ConvBNReLU(ch, latent_ch, k=3, s=2, p=1)]
            ch = latent_ch
        self.encoder = nn.Sequential(*enc)

        dec, ch = [], latent_ch
        for _ in range(levels):
            dec += [nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.SiLU(inplace=True)]
        self.decoder = nn.Sequential(*dec)
        self.to_rgb = nn.Conv2d(latent_ch, in_ch, 1)

    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        x = self.decoder(z)
        return self.to_rgb(x)


# -----------------------------
# Adaptive U-Net operating in LATENT space (ADP-mutable)
# -----------------------------

class AdaptiveUNet(nn.Module):
    def __init__(self, in_ch: int, widths: List[int], time_ch: int = 128):
        super().__init__()
        self.widths = list(widths)
        self.time_ch = time_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_ch), nn.SiLU(), nn.Linear(time_ch, time_ch)
        )
        self._build(in_ch)

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
        self.out_conv = nn.Conv2d(up_ch, in_ch, 1)  # predicts eps in latent space

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return self.time_mlp(t)

    def _cat_t(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        tmap = te[:, :, None, None].expand(B, te.size(1), H, W)
        return torch.cat([x, tmap], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self._t_embed(t)
        feats = []
        cur = x
        for i in range(self.num_stages):
            cur = self.down_blocks[2*i](self._cat_t(cur, te))
            cur = self.down_blocks[2*i + 1](cur)
            feats.append(cur)
            cur = self.pools[i](cur)
        cur = self.mid1(self._cat_t(cur, te))
        cur = self.mid2(cur)
        for i in range(self.num_stages):
            skip = feats[-(i+1)]
            cur = self.upsample[i](cur)
            if cur.size(-1) != skip.size(-1):
                cur = F.interpolate(cur, size=skip.shape[-2:], mode='nearest')
            cur = torch.cat([cur, skip], dim=1)
            cur = self.up_blocks[2*i](self._cat_t(cur, te))
            cur = self.up_blocks[2*i + 1](cur)
        return self.out_conv(cur)

    # ADP ops
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
        old = self.state_dict()
        self._build(in_ch=self.out_conv.out_channels)
        new = self.state_dict()
        for k in new.keys():
            if k in old:
                src, dst = old[k], new[k]
                common = tuple(min(a, b) for a, b in zip(src.shape, dst.shape))
                sl = tuple(slice(0, c) for c in common)
                dst[sl] = src[sl]
        self.load_state_dict(new, strict=False)


# -----------------------------
# Single-model Diffusion–AE (latent ε-pred + recon)
# -----------------------------

class DiffAESingleModel(nn.Module):
    def __init__(self,
                 img_ch: int = 3,
                 latent_ch: int = 4,
                 latent_levels: int = 2,
                 widths: List[int] = [32, 64, 96],
                 T: int = 1000,
                 lambda_recon: float = 1.0):
        super().__init__()
        self.T = int(T)
        self.lambda_recon = float(lambda_recon)
        self.register_buffer('betas', cosine_beta_schedule(T))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        self.ae = InternalAE(in_ch=img_ch, latent_ch=latent_ch, levels=latent_levels)
        self.latent_net = AdaptiveUNet(in_ch=latent_ch, widths=widths, time_ch=128)

    # diffusion helpers (latent space)
    def q_sample(self, z0: torch.Tensor, t_idx: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(z0)
        a_bar = self.alphas_cumprod[t_idx][:, None, None, None]
        zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * eps
        return zt, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        # encode to latent
        z0 = self.ae.encode(x)
        # recon path (teacher-free)
        x_rec = self.ae.decode(z0)
        loss_recon = 0.5 * F.mse_loss(x_rec, x)

        # latent diffusion ε-loss
        t_idx = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        t_norm = (t_idx.float() + 0.5) / self.T
        zt, eps = self.q_sample(z0, t_idx)
        eps_pred = self.latent_net(zt, t_norm)
        loss_diff = 0.5 * F.mse_loss(eps_pred, eps)

        return self.lambda_recon * loss_recon + loss_diff

    @torch.no_grad()
    def sample(self, B: int, img_ch: int, H: int, W: int, device: Optional[torch.device] = None, steps: Optional[int] = None):
        if device is None:
            device = next(self.parameters()).device
        if steps is None:
            steps = self.T
        # sample latent grid size given AE levels (H/2^L, W/2^L)
        L = len(self.ae.encoder)
        # each level adds one ConvBNReLU; but strides are 2 exactly per level
        lat_H, lat_W = H // (2 ** (len(self.ae.encoder))), W // (2 ** (len(self.ae.encoder)))
        z = torch.randn(B, self.latent_net.out_conv.in_channels, lat_H, lat_W, device=device)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = (t.float() + 0.5) / self.T
            eps = self.latent_net(z, t_norm)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            noise = torch.randn_like(z) if i > 0 else torch.zeros_like(z)
            z = (1.0 / torch.sqrt(alpha)) * (z - beta / torch.sqrt(1.0 - alpha_bar) * eps) + torch.sqrt(beta) * noise
        x = self.ae.decode(z)
        return torch.clamp(x, -1, 1)

    # ADP passthrough (over latent_net)
    def neurons(self) -> int:
        return self.latent_net.neurons()
    def snapshot_state(self):
        return {
            'ae': self.ae.state_dict(),
            'latent': self.latent_net.snapshot_state()
        }
    def restore_state(self, snap):
        self.ae.load_state_dict(snap['ae'])
        self.latent_net.restore_state(snap['latent'])
    def append_depth(self):
        self.latent_net.append_depth()
    def widen_all(self, ex_k: int):
        self.latent_net.widen_all(ex_k)


# -----------------------------
# Early-stopping trainer (joint loss)
# -----------------------------

@dataclass
class TrainCfg:
    lr: float = 2e-4
    max_epochs: int = 50
    es_patience: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_model(model: DiffAESingleModel, train_loader, val_loader, cfg: TrainCfg) -> Tuple[float, Dict]:
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
            best = val; best_snap = {'model': model.state_dict()}; bad = 0
        else:
            bad += 1
        if bad >= cfg.es_patience:
            break
    if best_snap is not None:
        model.load_state_dict(best_snap['model'])
    return best, best_snap


# -----------------------------
# ADP search policies (6 variants)
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


def adp_depth_then_width(model: DiffAESingleModel, train_loader, val_loader, tr: TrainCfg, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    d_fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; d_fail = 0
        else:
            model.restore_state(pre); d_fail += 1
            if d_fail >= 2: break
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; w_fail = 0
        else:
            model.restore_state(pre); w_fail += 1
            if w_fail >= 2: break
    return base


def adp_width_then_depth(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    w_fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
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
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; d_fail = 0
        else:
            model.restore_state(pre); d_fail += 1
            if d_fail >= 2: break
    return base


def adp_alt_depth_first(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    while True:
        improved = False
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        if not improved: break
    return base


def adp_alt_width_first(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    while True:
        improved = False
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; improved = True
        else:
            model.restore_state(pre)
        if not improved: break
    return base


def adp_depth_only(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    fail = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
        if _accept(v, base, s.delta):
            base = v; fail = 0
        else:
            model.restore_state(pre); fail += 1
            if fail >= 2: break
    return base


def adp_width_only(model, train_loader, val_loader, tr, s: SearchCfg):
    base, _ = train_one_model(model, train_loader, val_loader, tr)
    fail = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, tr)
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
# End of adp_diff_diffae.py
# ============================================================
