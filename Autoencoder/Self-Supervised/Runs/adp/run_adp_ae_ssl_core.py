# adp_ae_ssl_core.py
# Unified Single-Model Self-Supervised Autoencoder Core (Non-VAE, No Teacher/EMA)
# Supports many AE variants via algo switches (see SUPPORTED_ALGOS).
# Author: ADP / Breaking Neural Barrier

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Configuration
# ----------------------------

@dataclass
class AEConfig:
    in_channels: int = 3
    base_channels: int = 64
    depth: int = 4                      # encoder depth (decoder mirrors)
    bottleneck_dim: int = 256
    use_unet: bool = True               # if True, adds skip connections
    norm: str = "bn"                    # 'bn' | 'gn' | 'ln' | 'none'
    act: str = "relu"                   # 'relu' | 'gelu' | 'silu'
    # Regularizer weights (kept 0 by default unless algo needs them)
    w_sparse_l1: float = 0.0            # L1 on latent
    w_group_sparse: float = 0.0         # group l2 on latent groups
    group_size: int = 16
    w_contractive: float = 0.0          # Frobenius norm of d(h)/d(x)
    w_entropy: float = 0.0              # encourage higher latent entropy
    w_tv: float = 0.0                   # total variation on recon
    w_whiten: float = 0.0               # whiten latent covariance
    # Robust recon loss
    recon_loss: str = "mse"             # 'mse' | 'l1' | 'huber'
    huber_delta: float = 1.0
    # Masking / corruption
    mask_ratio: float = 0.6             # for masked / dropblock / half etc.
    block_size: int = 16                # dropblock side
    # Colorization
    grayscale_weighted: bool = True
    # Rotation / jigsaw heads
    rot_head: bool = True               # enables rotation classification when algo='rotation'
    jigsaw_head: bool = True            # enables 2x2 jigsaw classification when algo='jigsaw'
    # Device placement (used by helper ops that create tensors)
    device: Optional[str] = None

# ----------------------------
# Small layers and utilities
# ----------------------------

def _norm(nc: int, norm: str):
    if norm == "bn":
        return nn.BatchNorm2d(nc)
    if norm == "gn":
        return nn.GroupNorm(num_groups=max(1, nc // 16), num_channels=nc)
    if norm == "ln":
        # LayerNorm over CxHxW -> use GroupNorm(1, C) as LN over channels
        return nn.GroupNorm(1, nc)
    return nn.Identity()

def _act(act: str):
    return {"relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU()}[act]

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, cfg: AEConfig = AEConfig()):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.norm = _norm(c_out, cfg.norm)
        self.act  = _act(cfg.act)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, cfg: AEConfig):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvBNAct(c_in, c_out, cfg=cfg)
        self.conv2 = ConvBNAct(c_out, c_out, cfg=cfg)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.conv2(x)
        return x

# ----------------------------
# Encoder / Decoder
# ----------------------------

class Encoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        C = cfg.base_channels
        layers = []
        in_c = cfg.in_channels
        chs = []
        for d in range(cfg.depth):
            layers.append(ConvBNAct(in_c, C, cfg=cfg))
            layers.append(ConvBNAct(C, C, cfg=cfg))
            chs.append(C)
            if d < cfg.depth - 1:
                layers.append(nn.MaxPool2d(2))
            in_c = C
            C = C * 2
        self.body = nn.Sequential(*layers)
        self.out_channels = chs[-1]
        self.chs = chs

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats = []
        cur = x
        for m in self.body:
            cur = m(cur)
            if isinstance(m, ConvBNAct):
                feats.append(cur)
        return cur, feats  # final feature map + intermediate conv outputs

class Decoder(nn.Module):
    def __init__(self, enc: Encoder, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        chs = enc.chs
        # mirror channels descending
        ups = []
        C = chs[-1]
        for i in range(len(chs)-2, -1, -1):
            ups.append(UpBlock(C, chs[i], cfg))
            C = chs[i]
        self.ups = nn.ModuleList(ups)
        self.head = nn.Sequential(
            nn.Conv2d(chs[0], cfg.in_channels, kernel_size=1, bias=True)
        )

    def forward(self, z, skips: Optional[List[torch.Tensor]] = None):
        x = z
        if self.cfg.use_unet and skips is not None:
            # Take last conv features per stage (stride-2 between stages)
            # Collect corresponding skip feats (approx every 3 modules in Encoder)
            # We'll greedily concatenate the latest spatially matching features.
            skip_cursor = len(skips) - 1
            for up in self.ups:
                x = up(x)
                # Find a reasonably sized skip (closest spatial size)
                while skip_cursor >= 0 and skips[skip_cursor].shape[-1] > x.shape[-1]:
                    skip_cursor -= 1
                if skip_cursor >= 0 and skips[skip_cursor].shape[-1] == x.shape[-1]:
                    x = torch.cat([x, skips[skip_cursor]], dim=1)
                    # fuse back to desired channels
                    x = ConvBNAct(x.shape[1], x.shape[1] // 2, cfg=self.cfg).to(x.device)(x)
                    skip_cursor -= 1
        else:
            for up in self.ups:
                x = up(x)
        return self.head(x)

# ----------------------------
# Heads for auxiliary SSL tasks
# ----------------------------

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=[2,3])

class RotHead(nn.Module):
    def __init__(self, in_c: int, cfg: AEConfig):
        super().__init__()
        self.pool = GlobalAvgPool()
        self.fc = nn.Linear(in_c, 4)  # 0, 90, 180, 270
    def forward(self, feat):
        z = self.pool(feat)
        return self.fc(z)

class JigsawHead(nn.Module):
    """
    2x2 jigsaw on a downsampled conv feature; we classify among 4 fixed permutations to keep it simple.
    """
    def __init__(self, in_c: int, cfg: AEConfig):
        super().__init__()
        self.pool = GlobalAvgPool()
        self.fc = nn.Linear(in_c, 4)
    def forward(self, feat):
        return self.fc(self.pool(feat))

# ----------------------------
# Main unified AE
# ----------------------------

SUPPORTED_ALGOS = {
    # Classical / robust / regularized
    "plain", "sparse", "contractive", "robust_l1", "robust_huber", "group_sparse",
    "entropy", "whiten",
    # Masking / missing-data
    "masked", "dropblock", "half", "patch_remove", "inpaint", "context", "blindspot",
    # Colorization
    "colorize",
    # Spatial reasoning
    "rotation", "jigsaw",
    # Distortion / recovery
    "self_distortion", "blur_sharp", "color_jitter_recover",
    # Frequency masking (2D)
    "freq_mask",
    # Latent consistency
    "latent_cycle", "split_latent"
}

class SelfSupervisedAE(nn.Module):
    """
    Single-model, non-VAE self-supervised Autoencoder core.
    Use forward_train(x, algo=..., **kwargs) to compute loss dict for a given pretext.
    """
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(self.encoder, cfg)
        # Aux heads (remain unused unless algo requires them)
        self.rot_head = RotHead(self.encoder.out_channels, cfg) if cfg.rot_head else None
        self.jigsaw_head = JigsawHead(self.encoder.out_channels, cfg) if cfg.jigsaw_head else None

    # ------------------------
    # Corruptions / masks
    # ------------------------
    def _to_device(self, t: torch.Tensor):
        if self.cfg.device is None:
            return t.to(next(self.parameters()).device)
        return t.to(self.cfg.device)

    def make_random_mask(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        B, C, H, W = x.shape
        m = torch.rand(B, 1, H, W, device=x.device)
        return (m > ratio).float()  # 1 = keep, 0 = mask

    def make_dropblock_mask(self, x: torch.Tensor, block: int, ratio: float) -> torch.Tensor:
        B, C, H, W = x.shape
        mask = torch.ones(B, 1, H, W, device=x.device)
        num_blocks = max(1, int((H*W*ratio) / (block*block)))
        for b in range(B):
            for _ in range(num_blocks):
                y = random.randrange(0, max(1, H - block + 1))
                z = random.randrange(0, max(1, W - block + 1))
                mask[b, :, y:y+block, z:z+block] = 0.0
        return mask

    def make_half_mask(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        # Drop a contiguous half stripe along width or height depending on ratio
        B, C, H, W = x.shape
        mask = torch.ones(B, 1, H, W, device=x.device)
        if random.random() < 0.5:
            cut = int(W * ratio)
            if random.random() < 0.5:
                mask[:, :, :, :cut] = 0.0
            else:
                mask[:, :, :, W-cut:] = 0.0
        else:
            cut = int(H * ratio)
            if random.random() < 0.5:
                mask[:, :, :cut, :] = 0.0
            else:
                mask[:, :, H-cut:, :] = 0.0
        return mask

    def make_patch_remove_mask(self, x: torch.Tensor, patch: int, ratio: float) -> torch.Tensor:
        # remove several random patches (like Cutout)
        return self.make_dropblock_mask(x, patch, ratio)

    def freq_mask(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        # Mask bands along H or W (SpecAugment-style)
        B, C, H, W = x.shape
        mask = torch.ones(B, 1, H, W, device=x.device)
        bands = max(1, int((H if random.random() < 0.5 else W) * ratio))
        if random.random() < 0.5:
            # horizontal band
            y = random.randrange(0, max(1, H - bands + 1))
            mask[:, :, y:y+bands, :] = 0.0
        else:
            # vertical band
            z = random.randrange(0, max(1, W - bands + 1))
            mask[:, :, :, z:z+bands] = 0.0
        return mask

    # ------------------------
    # Loss helpers
    # ------------------------
    def recon_loss(self, recon, target):
        if self.cfg.recon_loss == "mse":
            return F.mse_loss(recon, target)
        if self.cfg.recon_loss == "l1":
            return F.l1_loss(recon, target)
        if self.cfg.recon_loss == "huber":
            return F.huber_loss(recon, target, delta=self.cfg.huber_delta)
        raise ValueError("Unknown recon loss")

    def total_variation(self, x):
        # TV on reconstruction to encourage local smoothness (useful for inpainting/context)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return (dx.abs().mean() + dy.abs().mean())

    def contractive_penalty(self, h, x):
        # Approximate ||∂h/∂x||_F^2 using autograd on mean of h
        # NOTE: For efficiency, we backprop a scalar head through encoder activations.
        ones = torch.ones_like(h)
        (g,) = torch.autograd.grad(
            outputs=(h * ones).sum(),
            inputs=x,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )
        if g is None:
            return x.new_tensor(0.0)
        return (g ** 2).mean()

    def whiten_penalty(self, h):
        # Whiten latent (per batch): covariance ≈ I
        z = h.flatten(2).mean(-1)  # global average over spatial dims -> (B, C)
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (z.shape[0] - 1 + 1e-6)
        I = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return ((cov - I) ** 2).mean()

    def group_sparse_penalty(self, h, group: int):
        z = h.flatten(2).mean(-1)  # (B, C)
        B, C = z.shape
        G = max(1, C // group)
        loss = 0.0
        for g in range(G):
            sl = slice(g*group, min((g+1)*group, C))
            loss = loss + torch.sqrt(1e-6 + (z[:, sl] ** 2).sum(dim=1)).mean()
        return loss / G

    # ------------------------
    # Forward passes
    # ------------------------
    def encode(self, x):
        z, feats = self.encoder(x)
        return z, feats

    def decode(self, z, feats):
        return self.decoder(z, feats if self.cfg.use_unet else None)

    def forward(self, x):
        # plain reconstruction
        z, feats = self.encode(x)
        recon = self.decode(z, feats)
        return recon

    # ------------------------
    # Algorithm-specific training step
    # ------------------------
    def forward_train(self, x: torch.Tensor, algo: str, **kwargs) -> Dict[str, Any]:
        """
        Returns dict with:
          loss: scalar
          logs: dict of scalars
          recon: reconstruction (when applicable)
          aux: optional logits/targets for classification pretexts
        """
        assert algo in SUPPORTED_ALGOS, f"Unsupported algo '{algo}'. Supported: {sorted(SUPPORTED_ALGOS)}"
        cfg = self.cfg
        logs: Dict[str, float] = {}
        aux: Dict[str, torch.Tensor] = {}

        if algo in {"plain", "sparse", "contractive", "robust_l1", "robust_huber", "group_sparse", "entropy", "whiten"}:
            with torch.set_grad_enabled(True):
                x_in = x
                z, feats = self.encode(x_in)
                recon = self.decode(z, feats)

                # base reconstruction loss
                if algo == "robust_l1":
                    base = F.l1_loss(recon, x)
                elif algo == "robust_huber":
                    base = F.huber_loss(recon, x, delta=cfg.huber_delta)
                else:
                    base = self.recon_loss(recon, x)

                loss = base
                logs["recon"] = base.item()

                # regularizers depending on algo or cfg weights
                if algo == "sparse" or cfg.w_sparse_l1 > 0:
                    z_lat = z.flatten(2).mean(-1)
                    l_sparse = z_lat.abs().mean()
                    w = cfg.w_sparse_l1 if algo != "sparse" else max(cfg.w_sparse_l1, 1e-3)
                    loss = loss + w * l_sparse
                    logs["sparse_l1"] = l_sparse.item()

                if algo == "group_sparse" or cfg.w_group_sparse > 0:
                    l_gs = self.group_sparse_penalty(z, cfg.group_size)
                    w = cfg.w_group_sparse if algo != "group_sparse" else max(cfg.w_group_sparse, 1e-3)
                    loss = loss + w * l_gs
                    logs["group_sparse"] = l_gs.item()

                if algo == "contractive" or cfg.w_contractive > 0:
                    l_con = self.contractive_penalty(z, x)
                    w = cfg.w_contractive if algo != "contractive" else max(cfg.w_contractive, 1e-4)
                    loss = loss + w * l_con
                    logs["contractive"] = l_con.item()

                if algo == "entropy" or cfg.w_entropy > 0:
                    z_lat = z.flatten(2).mean(-1)
                    # maximize entropy -> minimize negative entropy ~ mean of log variance proxy
                    std = (1e-6 + z_lat.var(dim=0)).sqrt()
                    l_ent = (-torch.log(std + 1e-6)).mean()
                    w = cfg.w_entropy if algo != "entropy" else max(cfg.w_entropy, 1e-4)
                    loss = loss + w * l_ent
                    logs["neg_entropy"] = l_ent.item()

                if algo == "whiten" or cfg.w_whiten > 0:
                    l_w = self.whiten_penalty(z)
                    w = cfg.w_whiten if algo != "whiten" else max(cfg.w_whiten, 1e-4)
                    loss = loss + w * l_w
                    logs["whiten"] = l_w.item()

                if cfg.w_tv > 0:
                    l_tv = self.total_variation(recon)
                    loss = loss + cfg.w_tv * l_tv
                    logs["tv"] = l_tv.item()

                return {"loss": loss, "logs": logs, "recon": recon, "aux": aux}

        # ---------- Masking / Missing Data ----------
        if algo in {"masked", "dropblock", "half", "patch_remove", "inpaint", "context", "blindspot", "freq_mask"}:
            if algo == "masked":
                mask = self.make_random_mask(x, cfg.mask_ratio)
            elif algo == "dropblock":
                mask = self.make_dropblock_mask(x, cfg.block_size, cfg.mask_ratio)
            elif algo == "half":
                mask = self.make_half_mask(x, cfg.mask_ratio)
            elif algo == "patch_remove":
                mask = self.make_patch_remove_mask(x, cfg.block_size, cfg.mask_ratio)
            elif algo == "freq_mask":
                mask = self.freq_mask(x, cfg.mask_ratio)
            else:
                # inpaint/context/blindspot: just use dropblock-like holes (context differs by loss weighting)
                mask = self.make_dropblock_mask(x, cfg.block_size, cfg.mask_ratio)

            corrupted = x * mask
            z, feats = self.encode(corrupted)
            recon = self.decode(z, feats)

            # Loss only on masked (inpainting/context) or on full?
            if algo in {"inpaint", "context", "blindspot"}:
                # emphasize masked regions (1 - mask)
                weight = (1.0 - mask)
                # to avoid zero division, normalize by mean weight
                denom = weight.mean().clamp_min(1e-6)
                if cfg.recon_loss == "mse":
                    base = ((recon - x) ** 2 * weight).sum() / (x.shape[0] * x.shape[1] * denom * x.shape[2] * x.shape[3])
                elif cfg.recon_loss == "l1":
                    base = ((recon - x).abs() * weight).sum() / (x.shape[0] * x.shape[1] * denom * x.shape[2] * x.shape[3])
                else:
                    diff = (recon - x).abs()
                    base = F.huber_loss(recon, x, delta=cfg.huber_delta, reduction="none")
                    base = (base * weight).sum() / (x.shape[0] * denom * x.shape[2] * x.shape[3])
            else:
                base = self.recon_loss(recon, x)

            loss = base
            logs = {"recon": base.item(), "masked_ratio": cfg.mask_ratio}

            if cfg.w_tv > 0:
                l_tv = self.total_variation(recon)
                loss = loss + cfg.w_tv * l_tv
                logs["tv"] = l_tv.item()

            return {"loss": loss, "logs": logs, "recon": recon, "aux": {"mask": mask}}

        # ---------- Colorization ----------
        if algo == "colorize":
            # Input: grayscale; Target: original color image
            if x.shape[1] == 1:
                gray = x
                target = x.repeat(1, 3, 1, 1)  # degenerate case
            else:
                if cfg.grayscale_weighted:
                    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1,3,1,1)
                    gray = (x * w).sum(dim=1, keepdim=True)
                else:
                    gray = x.mean(dim=1, keepdim=True)
                target = x

            z, feats = self.encode(gray.repeat(1, 3, 1, 1))  # keep 3 channels through encoder if built for RGB
            recon = self.decode(z, feats)
            loss = self.recon_loss(recon, target)
            logs = {"recon": loss.item()}
            return {"loss": loss, "logs": logs, "recon": recon, "aux": {"gray": gray}}

        # ---------- Spatial reasoning ----------
        if algo == "rotation":
            assert self.rot_head is not None, "Rotation head disabled in config."
            # Apply a random rotation to the input; classify rotation; reconstruct original (optional)
            k = random.choice([0, 1, 2, 3])  # 0,90,180,270
            x_rot = torch.rot90(x, k, dims=[-2, -1])
            z, feats = self.encode(x_rot)
            logits = self.rot_head(z)
            target = torch.full((x.shape[0],), k, device=x.device, dtype=torch.long)
            cls_loss = F.cross_entropy(logits, target)
            # Optional reconstruction of unrotated image (useful auxiliary)
            recon = self.decode(z, feats)
            # Re-rotate back before recon loss:
            recon_back = torch.rot90(recon, (4-k) % 4, dims=[-2, -1])
            rec_loss = self.recon_loss(recon_back, x)
            loss = cls_loss + rec_loss
            logs = {"rot_ce": cls_loss.item(), "recon": rec_loss.item()}
            return {"loss": loss, "logs": logs, "recon": recon_back, "aux": {"logits": logits, "target": target}}

        if algo == "jigsaw":
            assert self.jigsaw_head is not None, "Jigsaw head disabled in config."
            # 2x2 grid; choose among 4 fixed permutations (identity + 3 swaps)
            B, C, H, W = x.shape
            h2, w2 = H // 2, W // 2
            patches = [
                x[:, :, :h2, :w2],   # A
                x[:, :, :h2, w2:],   # B
                x[:, :, h2:, :w2],   # C
                x[:, :, h2:, w2:],   # D
            ]
            perms = [
                (0,1,2,3),           # identity
                (1,0,2,3),           # swap A,B
                (0,2,1,3),           # swap B,C
                (3,1,2,0),           # rotate blocks
            ]
            pid = random.randrange(0, 4)
            order = perms[pid]
            top = torch.cat([patches[order[0]], patches[order[1]]], dim=-1)
            bot = torch.cat([patches[order[2]], patches[order[3]]], dim=-1)
            x_perm = torch.cat([top, bot], dim=-2)

            z, feats = self.encode(x_perm)
            logits = self.jigsaw_head(z)
            target = torch.full((x.shape[0],), pid, device=x.device, dtype=torch.long)
            cls_loss = F.cross_entropy(logits, target)

            # Optional: reconstruct original ordering
            recon = self.decode(z, feats)
            rec_loss = self.recon_loss(recon, x_perm)  # keep self-consistent
            loss = cls_loss + rec_loss
            logs = {"jigsaw_ce": cls_loss.item(), "recon": rec_loss.item()}
            return {"loss": loss, "logs": logs, "recon": recon, "aux": {"logits": logits, "target": target}}

        # ---------- Distortion / recovery ----------
        if algo in {"self_distortion", "blur_sharp", "color_jitter_recover"}:
            corrupted = x
            if algo in {"self_distortion", "blur_sharp"}:
                # apply random gaussian blur
                k = random.choice([3,5])
                sigma = random.uniform(0.5, 1.2)
                pad = k // 2
                # build separable gaussian kernel
                coords = torch.arange(k, device=x.device) - (k-1)/2
                g = torch.exp(-0.5 * (coords/sigma)**2)
                g = (g / g.sum()).view(1,1,k,1)
                gx = g
                gy = g.transpose(-2,-1)
                blurred = F.conv2d(F.pad(x, (pad,pad,pad,pad), mode="reflect"), gx.expand(x.shape[1],1,k,1), groups=x.shape[1])
                blurred = F.conv2d(F.pad(blurred, (0,0,0,0)), gy.expand(x.shape[1],1,1,k), groups=x.shape[1])
                corrupted = blurred
            if algo in {"self_distortion", "color_jitter_recover"}:
                # simple brightness/contrast jitter
                alpha = random.uniform(0.8, 1.2)
                beta  = random.uniform(-0.1, 0.1)
                corrupted = corrupted * alpha + beta

            z, feats = self.encode(corrupted)
            recon = self.decode(z, feats)
            loss = self.recon_loss(recon, x)
            logs = {"recon": loss.item()}
            return {"loss": loss, "logs": logs, "recon": recon, "aux": {"corrupted": corrupted}}

        # ---------- Latent consistency ----------
        if algo in {"latent_cycle", "split_latent"}:
            z, feats = self.encode(x)
            recon = self.decode(z, feats)
            base = self.recon_loss(recon, x)
            loss = base
            logs = {"recon": base.item()}

            if algo == "latent_cycle":
                z2, _ = self.encode(recon.detach())
                cyc = F.mse_loss(z2, z.detach())
                loss = loss + 0.1 * cyc
                logs["latent_cycle"] = cyc.item()

            if algo == "split_latent":
                # split channels and predict one half from the other (linear self-prediction in latent space)
                z_lat = z.flatten(2).mean(-1)  # (B, C)
                C = z_lat.shape[1]
                C2 = C // 2
                a, b = z_lat[:, :C2], z_lat[:, C2:]
                # simple linear predictor (no extra params): minimize ||a - stop(b)|| and ||b - stop(a)||
                spa = F.mse_loss(a, b.detach())
                spb = F.mse_loss(b, a.detach())
                sp = 0.5 * (spa + spb)
                loss = loss + 0.1 * sp
                logs["split_latent"] = sp.item()

            if cfg.w_tv > 0:
                l_tv = self.total_variation(recon)
                loss = loss + cfg.w_tv * l_tv
                logs["tv"] = l_tv.item()

            return {"loss": loss, "logs": logs, "recon": recon, "aux": {}}

        raise RuntimeError("Code path not reached; algo unhandled.")

# ----------------------------
# Factory
# ----------------------------

def build_model(cfg: Optional[AEConfig] = None) -> SelfSupervisedAE:
    cfg = cfg or AEConfig()
    model = SelfSupervisedAE(cfg)
    return model

# ----------------------------
# Quick self-test (optional)
# ----------------------------
if __name__ == "__main__":
    # Simple smoke test on random data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AEConfig(in_channels=3, base_channels=32, depth=3, bottleneck_dim=128, device=device)
    net = build_model(cfg).to(device)
    x = torch.randn(4, 3, 64, 64, device=device)

    for algo in ["plain", "masked", "dropblock", "half", "inpaint", "context",
                 "colorize", "rotation", "jigsaw", "self_distortion",
                 "freq_mask", "latent_cycle", "split_latent"]:
        out = net.forward_train(x, algo=algo)
        print(f"{algo}: loss={out['loss'].item():.4f} logs={ {k:round(v,4) for k,v in out['logs'].items()} }")
