# =============================================
# File: adp_diff_latent.py
# Single-model latent-space diffusion with integrated 6x ADP policies
# (depth→width, width→depth, alternating-depth-first, alternating-width-first,
#  depth-only, width-only)
#
# Design rules (per "Breaking Neural Barrier" project):
# • Single-model only: no EMA teacher, no external pre-trained VAE; the latent encoder/decoder
#   lives inside this model and is trained end-to-end with the denoiser.
# • ADP modifies the SAME model in-place via weight transplantation (no multi-model interaction).
# • Self-contained, minimal dependencies (PyTorch only).
# =============================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility: simple cosine beta schedule (DDPM-compatible)
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-6, 0.999)


# -----------------------------
# Building blocks (Conv → BN → ReLU)
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -----------------------------
# Internal Latent Adapter (encoder+decoder), trained jointly
# -----------------------------

class InternalLatentAdapter(nn.Module):
    """
    A simple stride-2 encoder + symmetric decoder to create a lower-resolution latent grid.
    This is NOT a second model; it is part of the same network. The denoiser works in latent space.
    """
    def __init__(self, in_ch: int = 3, latent_ch: int = 4, levels: int = 2):
        super().__init__()
        enc = []
        ch = in_ch
        for _ in range(levels):
            enc += [ConvBNReLU(ch, latent_ch, k=3, s=2, p=1)]
            ch = latent_ch
        self.encoder = nn.Sequential(*enc)

        dec = []
        ch = latent_ch
        for _ in range(levels):
            dec += [nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*dec)
        self.to_rgb = nn.Conv2d(latent_ch, in_ch, 1)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder(z)
        return self.to_rgb(x)


# -----------------------------
# Adaptive U-Net-ish backbone (width = channels per stage; depth = #stages)
# -----------------------------

class AdaptiveUNet(nn.Module):
    """
    A lean U-Net-like backbone whose width and depth can be mutated at runtime.
    Width: out_channels of each stage; Depth: number of down/up stages.
    """
    def __init__(self, in_ch: int, base_widths: List[int], num_stages: int, time_ch: int = 128):
        super().__init__()
        assert num_stages == len(base_widths), "num_stages must equal len(base_widths)"

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_ch), nn.SiLU(), nn.Linear(time_ch, time_ch)
        )

        self.widths: List[int] = list(base_widths)
        self.num_stages = num_stages
        self.time_ch = time_ch

        # Down path
        downs, pools = [], []
        ch = in_ch
        for w in self.widths:
            downs += [ConvBNReLU(ch + time_ch, w), ConvBNReLU(w, w)]
            pools += [nn.AvgPool2d(2)]
            ch = w
        self.down_blocks = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)

        # Bottleneck
        self.mid1 = ConvBNReLU(ch + time_ch, ch)
        self.mid2 = ConvBNReLU(ch, ch)

        # Up path
        ups = []
        for w in reversed(self.widths):
            ups += [ConvBNReLU(ch + w + time_ch, w), ConvBNReLU(w, w)]
            ch = w
        self.up_blocks = nn.ModuleList(ups)
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(w, w, 4, 2, 1) for w in reversed(self.widths)])

        self.out_conv = nn.Conv2d(ch, in_ch, 1)

    # ---------- helpers ----------
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        # t should be in [0, 1]
        if t.dim() == 1:
            t = t[:, None]
        return self.time_mlp(t)

    def _inject_time(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t_map = t_emb[:, :, None, None].expand(B, t_emb.size(1), H, W)
        return torch.cat([x, t_map], dim=1)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self._time_embed(t)

        feats = []
        cur = x
        # down
        for i in range(self.num_stages):
            cur = self.down_blocks[2*i](self._inject_time(cur, t_emb))
            cur = self.down_blocks[2*i + 1](cur)
            feats.append(cur)
            cur = self.pools[i](cur)

        # mid
        cur = self.mid1(self._inject_time(cur, t_emb))
        cur = self.mid2(cur)

        # up
        for i in range(self.num_stages):
            skip = feats[-(i+1)]
            cur = self.upsample[i](cur)
            # align shapes
            if cur.size(-1) != skip.size(-1):
                cur = F.interpolate(cur, size=skip.shape[-2:], mode='nearest')
            cur = torch.cat([cur, skip], dim=1)
            cur = self.up_blocks[2*i](self._inject_time(cur, t_emb))
            cur = self.up_blocks[2*i + 1](cur)

        return self.out_conv(cur)

    # ---------- ADP ops ----------
    def snapshot_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def restore_state(self, snap: Dict[str, torch.Tensor]):
        self.load_state_dict(snap, strict=True)

    def neurons(self) -> int:
        # proxy capacity = sum of stage widths
        return int(sum(self.widths))

    def append_depth(self):
        """Add one stage at the bottom (down+up) copying last width."""
        last_w = self.widths[-1]
        # Down append
        self.widths.append(last_w)
        self.num_stages += 1

        self.down_blocks.extend([
            ConvBNReLU(last_w + self.time_ch, last_w),
            ConvBNReLU(last_w, last_w)
        ])
        self.pools.append(nn.AvgPool2d(2))

        # Update bottleneck remains same shape (last_w)
        # Up path grows in front (because we iterate in order)
        self.up_blocks = nn.ModuleList(
            [ConvBNReLU(last_w + last_w + self.time_ch, last_w), ConvBNReLU(last_w, last_w)] + list(self.up_blocks)
        )
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(last_w, last_w, 4, 2, 1)] + list(self.upsample))
        # out_conv remains compatible (last layer already emits in_ch)

    def widen_all(self, ex_k: int):
        """Increase each stage width by ex_k, transplanting overlapping weights."""
        new_widths = [w + ex_k for w in self.widths]
        self._resize_to(new_widths)

    def _resize_to(self, new_widths: List[int]):
        assert len(new_widths) == self.num_stages
        old = self.state_dict()
        old_widths = self.widths
        self.widths = list(new_widths)

        # Rebuild modules with new sizes
        in_ch = self.out_conv.in_channels  # this equals the last up stage width
        time_ch = self.time_ch
        # Recreate everything from scratch then transplant
        new = AdaptiveUNet(self.out_conv.out_channels, self.widths, self.num_stages, time_ch)
        new_sd = new.state_dict()

        def copy_param(dst_key, src_key):
            if src_key in old and dst_key in new_sd:
                src = old[src_key]
                dst = new_sd[dst_key]
                # overlap copy
                common = tuple(min(a, b) for a, b in zip(src.shape, dst.shape))
                slices = tuple(slice(0, c) for c in common)
                dst[slices] = src[slices]
                new_sd[dst_key] = dst

        # Map keys by names (same architecture layout up to channel sizes)
        for k in new_sd.keys():
            copy_param(k, k)

        # Load transplanted
        new.load_state_dict(new_sd, strict=False)
        # Replace self modules with new's
        self.__dict__.update(new.__dict__)


# -----------------------------
# Full single-model Latent Diffusion (epsilon prediction)
# -----------------------------

class LatentDiffusionSingleModel(nn.Module):
    def __init__(self,
                 img_ch: int = 3,
                 latent_ch: int = 4,
                 latent_levels: int = 2,
                 widths: List[int] = [32, 64, 96],
                 T: int = 1000):
        super().__init__()
        self.T = T
        self.betas = cosine_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.adapter = InternalLatentAdapter(img_ch, latent_ch, levels=latent_levels)
        self.denoiser = AdaptiveUNet(in_ch=latent_ch,
                                     base_widths=widths,
                                     num_stages=len(widths),
                                     time_ch=128)

    # -------- diffusion helpers ---------
    def _to_device(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

    def q_sample(self, x0_latent, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0_latent)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x0_latent + sqrt_one_minus * noise, noise

    # -------- forward (training) ---------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        self._to_device(device)
        B = x.size(0)
        t = torch.randint(0, self.T, (B,), device=device)
        t_norm = (t.float() + 0.5) / self.T

        z0 = self.adapter.encode(x)               # latent clean
        zt, eps = self.q_sample(z0, t)            # noisy latent + ground-truth noise
        eps_pred = self.denoiser(zt, t_norm)      # predict noise
        loss = F.mse_loss(eps_pred, eps)
        return loss

    # -------- sampling (for validation visuals) ---------
    @torch.no_grad()
    def sample(self, B: int, img_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        self._to_device(device)
        H, W = img_size
        z = torch.randn(B, self.denoiser.out_conv.in_channels, H // (2 ** 2), W // (2 ** 2), device=device)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = (t.float() + 0.5) / self.T
            eps_pred = self.denoiser(z, t_norm)
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_cum = self.alphas_cumprod[i]
            if i > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = (1 / torch.sqrt(alpha_t)) * (z - beta_t / torch.sqrt(1 - alpha_cum) * eps_pred) + torch.sqrt(beta_t) * noise
        x = self.adapter.decode(z)
        return torch.clamp(x, -1, 1)

    # -------- ADP passthrough ---------
    def snapshot_state(self):
        return {
            'adapter': self.adapter.state_dict(),
            'denoiser': self.denoiser.snapshot_state(),
        }

    def restore_state(self, snap):
        self.adapter.load_state_dict(snap['adapter'])
        self.denoiser.restore_state(snap['denoiser'])

    def append_depth(self):
        self.denoiser.append_depth()

    def widen_all(self, ex_k: int):
        self.denoiser.widen_all(ex_k)

    def neurons(self):
        # proxy capacity (adapter is small, focus on denoiser)
        return self.denoiser.neurons()


# -----------------------------
# Lightweight early-stopping trainer (single model)
# -----------------------------

@dataclass
class TrainCfg:
    lr: float = 2e-4
    max_epochs: int = 50
    es_patience: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_model(model: LatentDiffusionSingleModel,
                    train_loader,
                    val_loader,
                    cfg: TrainCfg) -> Tuple[float, Dict]:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float('inf')
    best_snap = None
    bad = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for x,_ in train_loader:
            x = x.to(device)
            loss = model(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        # val
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x,_ in val_loader:
                x = x.to(device)
                loss = model(x)
                val_loss += loss.item() * x.size(0)
                n += x.size(0)
        val_loss /= max(1, n)
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_snap = {
                'model': model.state_dict(),
            }
            bad = 0
        else:
            bad += 1
        if bad >= cfg.es_patience:
            break
    if best_snap is not None:
        model.load_state_dict(best_snap['model'])
    return best_val, best_snap


# -----------------------------
# ADP search policies (6 variants)
# -----------------------------

@dataclass
class SearchCfg:
    delta: float = 0.0            # required improvement
    trials_width: int = 50
    trials_depth: int = 50
    ex_k: int = 8                 # widen increment
    max_neurons: Optional[int] = None


def _accept(improved: float, baseline: float, delta: float) -> bool:
    return improved < (baseline - delta)


def adp_depth_then_width(model: LatentDiffusionSingleModel, train_loader, val_loader, train_cfg: TrainCfg, s: SearchCfg):
    base_val, _ = train_one_model(model, train_loader, val_loader, train_cfg)
    # Depth series
    d_fails = 0
    for _ in range(s.trials_depth):
        pre_snap = model.snapshot_state()
        model.append_depth()
        # capacity guard
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre_snap)
            break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v
            d_fails = 0
        else:
            model.restore_state(pre_snap)
            d_fails += 1
            if d_fails >= 2:
                break
    # Width series
    w_fails = 0
    for _ in range(s.trials_width):
        pre_snap = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre_snap)
            break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v
            w_fails = 0
        else:
            model.restore_state(pre_snap)
            w_fails += 1
            if w_fails >= 2:
                break
    return base_val


def adp_width_then_depth(model, train_loader, val_loader, train_cfg, s: SearchCfg):
    base_val, _ = train_one_model(model, train_loader, val_loader, train_cfg)
    # Width series
    w_fails = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre)
            break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v
            w_fails = 0
        else:
            model.restore_state(pre)
            w_fails += 1
            if w_fails >= 2:
                break
    # Depth series
    d_fails = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre)
            break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v
            d_fails = 0
        else:
            model.restore_state(pre)
            d_fails += 1
            if d_fails >= 2:
                break
    return base_val


def adp_alt_depth_first(model, train_loader, val_loader, train_cfg, s: SearchCfg):
    base_val, _ = train_one_model(model, train_loader, val_loader, train_cfg)
    while True:
        improved = False
        # depth phase
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v; improved = True
        else:
            model.restore_state(pre)
        # width phase
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v; improved = True
        else:
            model.restore_state(pre)
        if not improved:
            break
    return base_val


def adp_alt_width_first(model, train_loader, val_loader, train_cfg, s: SearchCfg):
    base_val, _ = train_one_model(model, train_loader, val_loader, train_cfg)
    while True:
        improved = False
        # width
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v; improved = True
        else:
            model.restore_state(pre)
        # depth
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre); break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v; improved = True
        else:
            model.restore_state(pre)
        if not improved:
            break
    return base_val


def adp_depth_only(model, train_loader, val_loader, train_cfg, s: SearchCfg):
    base_val, _ = train_one_model(model, train_loader, val_loader, train_cfg)
    fails = 0
    for _ in range(s.trials_depth):
        pre = model.snapshot_state()
        model.append_depth()
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre)
            break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v; fails = 0
        else:
            model.restore_state(pre); fails += 1
            if fails >= 2:
                break
    return base_val


def adp_width_only(model, train_loader, val_loader, train_cfg, s: SearchCfg):
    base_val, _ = train_one_model(model, train_loader, val_loader, train_cfg)
    fails = 0
    for _ in range(s.trials_width):
        pre = model.snapshot_state()
        model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons() > s.max_neurons:
            model.restore_state(pre)
            break
        v, _ = train_one_model(model, train_loader, val_loader, train_cfg)
        if _accept(v, base_val, s.delta):
            base_val = v; fails = 0
        else:
            model.restore_state(pre); fails += 1
            if fails >= 2:
                break
    return base_val


POLICIES = {
    'depth2width': adp_depth_then_width,
    'width2depth': adp_width_then_depth,
    'alt_depth': adp_alt_depth_first,
    'alt_width': adp_alt_width_first,
    'depth_only': adp_depth_only,
    'width_only': adp_width_only,
}


# =============================================
# End of adp_diff_latent.py
# =============================================


# =============================================
# File: run_adp_diff.py
# Unified runner for Part A families and 6 ADP policies
#   • In this first release, Part A implemented: 'latent' (LatentDiffusionSingleModel)
#   • Future parts can be added behind --part {latent, sde, flow, ...}
# =============================================

import argparse
from functools import partial

import torch
import torchvision as tv
import torchvision.transforms as T

# Local imports when split into files:
# from adp_diff_latent import LatentDiffusionSingleModel, TrainCfg, SearchCfg, POLICIES


def make_loaders(data_root: str, img_size: int, batch: int, val_split: float = 0.2):
    tfm_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=4),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    tfm_val = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    ds_train_full = tv.datasets.CIFAR10(data_root, train=True, download=True, transform=tfm_train)
    ds_val_full   = tv.datasets.CIFAR10(data_root, train=True, download=True, transform=tfm_val)

    n = len(ds_train_full)
    n_val = int(n * val_split)
    idx = torch.randperm(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train_full, train_idx.tolist()),
                                               batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_val_full, val_idx.tolist()),
                                               batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser(description='Unified ADP-Diffusion Runner (Part A families + 6 ADP policies)')

    # Part selector (in this drop only 'latent' is implemented)
    p.add_argument('--part', type=str, default='latent', choices=['latent'],
                   help='Which Part-A family to use (this step: latent).')

    # ADP policy selector
    p.add_argument('--adp', type=str, default='depth2width',
                   choices=['depth2width','width2depth','alt_depth','alt_width','depth_only','width_only'],
                   help='ADP policy for architecture growth.')

    # Data & train
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--img-size', type=int, default=32)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.2)

    # Train cfg
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-epochs', type=int, default=30)
    p.add_argument('--es-patience', type=int, default=7)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Search cfg
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--trials-depth', type=int, default=20)
    p.add_argument('--trials-width', type=int, default=20)
    p.add_argument('--ex-k', type=int, default=8)
    p.add_argument('--max-neurons', type=int, default=0, help='0 disables capacity limit')

    # Model hyperparams
    p.add_argument('--latent-ch', type=int, default=4)
    p.add_argument('--latent-levels', type=int, default=2)
    p.add_argument('--widths', type=int, nargs='+', default=[32, 64, 96])
    p.add_argument('--T', type=int, default=1000)

    args = p.parse_args()

    train_loader, val_loader = make_loaders(args.data, args.img_size, args.batch, args.val_split)

    # Instantiate Part A model (latent)
    if args.part == 'latent':
        model = LatentDiffusionSingleModel(img_ch=3,
                                           latent_ch=args.latent_ch,
                                           latent_levels=args.latent_levels,
                                           widths=args.widths,
                                           T=args.T)
    else:
        raise NotImplementedError('Only --part latent is implemented in this drop.')

    train_cfg = TrainCfg(lr=args.lr, max_epochs=args.max_epochs, es_patience=args.es_patience,
                         grad_clip=args.grad_clip, device=args.device)
    maxN = None if args.max_neurons == 0 else args.max_neurons
    search_cfg = SearchCfg(delta=args.delta, trials_width=args.trials_width, trials_depth=args.trials_depth,
                           ex_k=args.ex_k, max_neurons=maxN)

    policy_fn = POLICIES[args.adp]
    best = policy_fn(model, train_loader, val_loader, train_cfg, search_cfg)

    print(f"Finished. Best val loss = {best:.4f}. Final neurons = {model.neurons()}.")


if __name__ == '__main__':
    main()

# =============================================
# End of run_adp_diff.py
# =============================================
