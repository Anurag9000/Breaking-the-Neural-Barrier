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

        if algo :
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

                # ---------- 89) Perceptual-Self AE ----------
                if algo == "perceptual_self":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    # use the model’s own mid-level features as "perceptual" targets
                    f_orig = feats[-1] if isinstance(feats, (list,tuple)) else z
                    f_rec, _ = self.encode(recon.detach())  # detach path through recon image
                    f_rec = f_rec if f_rec.shape == f_orig.shape else F.interpolate(f_rec, size=f_orig.shape[-2:])
                    perc = F.l1_loss(f_rec, f_orig.detach())
                    loss = rec + 0.1 * perc
                    logs = {"recon": rec.item(), "self_perc": perc.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 113) Pyramid Reconstruction ----------
                if algo == "pyr_recon":
                    # decode to multiple scales and match targets at each scale
                    z, feats = self.encode(x)
                    r_full = self.decode(z, feats)
                    # build targets at 1/2 and 1/4
                    t2 = _down(x, 2); t4 = _down(x, 4)
                    # reuse decoder by downscaling the full recon (simple & stable)
                    r2 = _down(r_full, 2); r4 = _down(r_full, 4)
                    L = self.recon_loss(r_full, x) + self.recon_loss(r2, t2) + self.recon_loss(r4, t4)
                    logs = {"pyr_loss": L.item()}
                    return {"loss": L, "logs": logs, "recon": r_full}

                # ---------- 114) Laplacian Pyramid Matching ----------
                if algo == "laplacian_pyr":
                    Lx, Gx = _build_laplacian_pyr(x, levels=3)
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    Lr, Gr = _build_laplacian_pyr(r, levels=3)
                    # band-wise match + small coarsest Gaussian match
                    band = sum(self.recon_loss(a, b) for a,b in zip(Lr, Lx))
                    coarse = self.recon_loss(Gr[0], Gx[0])
                    loss = band + 0.5*coarse
                    logs = {"band": band.item(), "coarse": coarse.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 115) Super-Resolution ×4 ----------
                if algo == "superres_x4":
                    # degrade by ×4 area downsample then nearest upsample; reconstruct sharp original
                    x_lr = _down(x, 4, "area")
                    x_up = _up(x_lr, 4, "nearest")
                    z, feats = self.encode(x_up); r = self.decode(z, feats)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 116) Progressive Resize Curriculum ----------
                if algo == "prog_resize":
                    # pick a random scale in {1/4,1/3,1/2,1} → upsample to native → reconstruct native
                    scales = [4,3,2,1]
                    s = random.choice(scales)
                    xs = _up(_down(x, s, "area"), s, "bilinear") if s>1 else x
                    z, feats = self.encode(xs); r = self.decode(z, feats)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item(), "scale": 1.0/s}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 117) Scale-Invariant Latent ----------
                if algo == "scale_invariant":
                    # two scales of same image; latent (pooled) should agree; reconstruct canonical (native) view
                    x_small = _up(_down(x, 2, "area"), 2, "bilinear")
                    z0, f0 = self.encode(x);      r0 = self.decode(z0, f0)
                    z1, f1 = self.encode(x_small)
                    h0 = F.normalize(z0.flatten(2).mean(-1), dim=1)
                    h1 = F.normalize(z1.flatten(2).mean(-1), dim=1)
                    inv = F.mse_loss(h0, h1.detach())
                    rec = self.recon_loss(r0, x)
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 118) Multi-Scale Self-Perceptual ----------
                if algo == "ms_perceptual":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    # build two additional scales and compare encoder features there too
                    x2, r2 = _down(x,2), _down(r,2)
                    x4, r4 = _down(x,4), _down(r,4)
                    f_x , _ = self.encode(x);  f_r , _ = self.encode(r.detach())
                    f_x2, _ = self.encode(x2); f_r2, _ = self.encode(r2.detach())
                    f_x4, _ = self.encode(x4); f_r4, _ = self.encode(r4.detach())
                    # perceptual via L1 on top features (shape-safe via interp if needed)
                    def match(a,b):
                        if a.shape != b.shape: b = F.interpolate(b, size=a.shape[-2:])
                        return F.l1_loss(b, a.detach())
                    perc = match(f_x, f_r) + match(f_x2, f_r2) + match(f_x4, f_r4)
                    loss = rec + 0.05*perc
                    logs = {"recon": rec.item(), "ms_perc": perc.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 119) Anti-Alias Consistency ----------
                if algo == "anti_alias":
                    # After low-pass then down+up, image should return close to original ↔ decoder should produce anti-aliased detail
                    aa = _gauss_blur(x, k=5, sigma=1.0)
                    da = _up(_down(aa, 2, "area"), 2, "bilinear")
                    z, f = self.encode(da); r = self.decode(z, f)
                    # penalize residual high-freq mismatch more than low-freq
                    _, rh = _fft_split(r); _, xh = _fft_split(x)
                    hf = F.l1_loss(rh, xh)
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.05*hf
                    logs = {"recon": rec.item(), "hf": hf.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 120) Cross-Scale Cycle (low↔high) ----------
                if algo == "cross_scale_cycle":
                    # low → high (super-res) and high → low (downscale recon) should be cycle-consistent
                    xl = _down(x, 2, "area")                 # low-res target proxy
                    # path A: upsample low, reconstruct high, then down it back (cycle to low)
                    xu = _up(xl, 2, "nearest")
                    zA, fA = self.encode(xu); rA = self.decode(zA, fA)      # high prediction
                    cyc_lo = _down(rA, 2, "area")
                    L_cyc = self.recon_loss(cyc_lo, xl)
                    # path B: take hi → recon hi, then compare to hi
                    zB, fB = self.encode(x);  rB = self.decode(zB, fB)
                    L_hi = self.recon_loss(rB, x)
                    loss = L_hi + 0.5*L_cyc
                    logs = {"hi": L_hi.item(), "cycle_lo": L_cyc.item()}
                    return {"loss": loss, "logs": logs, "recon": rA}

                # ---------- 97) Symmetry Consistency ----------
                if algo == "symmetry_consistency":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    # symmetry maps of input and recon should agree
                    ms_x = _symmetry_map(x)
                    ms_r = _symmetry_map(r)
                    sym = F.l1_loss(ms_r, ms_x.detach())
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.1 * sym
                    logs = {"recon": rec.item(), "sym": sym.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 98) Keypoint Consistency (unsup soft-argmax) ----------
                if algo == "kp_consistency":
                    fA, _ = self.encode(x)
                    # mildly jittered view
                    k = random.choice([0,1,2,3])
                    xB = torch.rot90(x, k, dims=[-2,-1])
                    fB, _ = self.encode(xB)
                    kpA = _spatial_softargmax(fA)
                    kpB = _spatial_softargmax(fB)
                    # rotate kps from B back to A's frame
                    if k == 1:  kpB = kpB[:, [1,0]] * torch.tensor([1,-1], device=x.device)  # rough rotation mapping
                    if k == 2:  kpB = -kpB
                    if k == 3:  kpB = kpB[:, [1,0]] * torch.tensor([-1,1], device=x.device)
                    kpl = F.mse_loss(kpA, kpB.detach())
                    # recon tether
                    r = self.decode(fA, _) if isinstance(_, list) else self.decode(fA, fA)
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.1 * kpl
                    logs = {"recon": rec.item(), "kp": kpl.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 99) Saliency Preserve (input-grad agreement) ----------
                if algo == "saliency_preserve":
                    x = x.clone().detach().requires_grad_(True)
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    (rec).backward(retain_graph=True)
                    sal_x = x.grad.detach().abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
                    x.grad = None
                    # saliency of recon wrt recon pixels
                    r = r.clone().detach().requires_grad_(True)
                    rec2 = self.recon_loss(r, r)  # identity; grads ~ saliency emphasis
                    (rec2).backward()
                    sal_r = r.grad.detach().abs().mean(dim=1, keepdim=True)
                    # match normalized saliency maps
                    def _norm(m): return m / (m.amax(dim=(-2,-1), keepdim=True) + 1e-6)
                    sp = F.l1_loss(_norm(sal_x), _norm(sal_r))
                    loss = rec.detach() + 0.05 * sp
                    logs = {"recon": float(rec.detach().item()), "sal": sp.item()}
                    return {"loss": loss, "logs": logs, "recon": r.detach()}

                # ---------- 100) Proto-Seg Consistency ----------
                if algo == "proto_seg":
                    f, _ = self.encode(x)
                    z_lat = f.mean(dim=[2,3])          # (B,C)
                    C, a = _kmeans_one_step(z_lat, K=min(6, z_lat.shape[0]))
                    # encourage latent to be close to its proto center
                    z_hat = a @ C                       # (B,D)
                    cons = F.mse_loss(z_lat, z_hat.detach())
                    r = self.decode(f, _) if isinstance(_, list) else self.decode(f, f)
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.001 * cons
                    logs = {"recon": rec.item(), "proto": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 105) MC-Dropout Consistency ----------
                if algo == "mc_dropout":
                    # Two dropout passes on the *same* input; recon should be consistent and close to x
                    x_in = x
                    # encode once -> feature; apply dropout before decode in two ways
                    z, feats = self.encode(x_in)                 # feats can be a list or tensor
                    def _drop_feat(F):
                        if isinstance(F, (list,tuple)):
                            G = list(F); G[-1] = F.dropout2d(G[-1], p=0.3, training=True) if hasattr(F, "dropout2d") else F[-1]
                            return G
                        else:
                            return F.dropout2d(F, p=0.3, training=True) if hasattr(F, "dropout2d") else F
                    f1 = _drop_feat(feats); f2 = _drop_feat(feats)
                    r1 = self.decode(z, f1); r2 = self.decode(z, f2)
                    rec = 0.5*(self.recon_loss(r1, x_in) + self.recon_loss(r2, x_in))
                    cons = F.mse_loss(r1, r2.detach())
                    loss = rec + 0.1 * cons
                    logs = {"recon": rec.item(), "mc_cons": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r1}

                # ---------- 106) AugMix Reconstruction ----------
                if algo == "augmix_recon":
                    xmix, (a1, a2, a3), w = _augmix_three(x)
                    z_m, f_m = self.encode(xmix); r_m = self.decode(z_m, f_m)
                    # decode each aug (teacher-free): encourage mixture of individual recons ≈ recon of mixture
                    z1,f1 = self.encode(a1); r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(a2); r2 = self.decode(z2,f2)
                    z3,f3 = self.encode(a3); r3 = self.decode(z3,f3)
                    mix_of_recons = (w[0]*r1 + w[1]*r2 + w[2]*r3).clamp(0,1)
                    recon_cons = F.mse_loss(r_m, mix_of_recons.detach())
                    # also tie to clean x for stability
                    rec = self.recon_loss(r_m, x)
                    loss = rec + 0.1 * recon_cons
                    logs = {"recon": rec.item(), "mix_cons": recon_cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r_m}

                # ---------- 107) Noise-Ramp (sigma curriculum) ----------
                if algo == "noise_ramp":
                    # sample sigma from a skewed beta to emulate a "ramp" of noise levels
                    a,b = 2.0, 5.0
                    u = torch.distributions.Beta(a,b).sample().to(x.device)
                    sigma = float(0.02 + 0.28*u)  # [0.02, 0.30]
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    z, f = self.encode(xn); r = self.decode(z, f)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item(), "sigma": sigma}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 108) Randomized Smoothing (K-noise ensemble) ----------
                if algo == "rand_smooth":
                    K = 4
                    sig = 0.15
                    recons = []
                    var_pen = 0.0
                    for _ in range(K):
                        xn = (x + sig*torch.randn_like(x)).clamp(0,1)
                        z, f = self.encode(xn); recons.append(self.decode(z, f))
                    R = torch.stack(recons, dim=0)             # (K,B,C,H,W)
                    r_mean = R.mean(dim=0)
                    # variance penalty across ensemble (reduce sensitivity)
                    var_pen = R.var(dim=0).mean()
                    rec = self.recon_loss(r_mean, x)
                    loss = rec + 0.05 * var_pen
                    logs = {"recon": rec.item(), "var": float(var_pen)}
                    return {"loss": loss, "logs": logs, "recon": r_mean}

                # ---------- 109) Stochastic-Depth Invariance ----------
                if algo == "stoch_depth_inv":
                    # one pass with a randomly skipped top encoder block; one normal pass
                    x0 = x
                    # normal
                    zN, fN = self.encode(x0); rN = self.decode(zN, fN)
                    # "skip" by zeroing the last feature map (proxy to dropping a block)
                    if isinstance(fN, (list,tuple)):
                        fS = list(fN); fS[-1] = torch.zeros_like(fS[-1])
                    else:
                        fS = torch.zeros_like(fN)
                    rS = self.decode(zN, fS)
                    rec = 0.5*(self.recon_loss(rN, x0) + self.recon_loss(rS, x0))
                    inv = F.mse_loss(rS, rN.detach())
                    loss = rec + 0.1 * inv
                    logs = {"recon": rec.item(), "sd_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": rN}

                # ---------- 110) Sigma-Mix Consistency ----------
                if algo == "sigma_mix":
                    s1, s2 = 0.05, 0.25
                    x1 = (x + s1*torch.randn_like(x)).clamp(0,1)
                    x2 = (x + s2*torch.randn_like(x)).clamp(0,1)
                    z1,f1 = self.encode(x1); r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(x2); r2 = self.decode(z2,f2)
                    rec = 0.5*(self.recon_loss(r1, x) + self.recon_loss(r2, x))
                    cons = F.mse_loss(r1, r2.detach())
                    loss = rec + 0.1 * cons
                    logs = {"recon": rec.item(), "sigma_cons": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r1}

                # ---------- 111) Latent-Noise Robustness ----------
                if algo == "latent_noise_robust":
                    z, f = self.encode(x)
                    # add noise in latent before decode
                    z_lat = z
                    if z_lat.dim() == 4:
                        eps = 0.05*torch.randn_like(z_lat)
                    else:
                        eps = 0.05*torch.randn_like(z_lat)
                    r_clean = self.decode(z_lat, f)
                    r_noisy = self.decode(z_lat + eps, f)
                    rec = self.recon_loss(r_clean, x)
                    cons = F.mse_loss(r_noisy, r_clean.detach())
                    loss = rec + 0.1 * cons
                    logs = {"recon": rec.item(), "latent_cons": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r_noisy}

                # ---------- 112) Feature-Channel Dropout (encoder) ----------
                if algo == "feature_ch_drop":
                    # zero random encoder channels; reconstruct clean target
                    z, f = self.encode(x)
                    if isinstance(f, (list,tuple)):
                        Fenc = list(f)
                        t = Fenc[-1]
                        B,C,H,W = t.shape
                        drop_ratio = 0.25
                        k = int(C * drop_ratio)
                        idx = torch.randperm(C, device=x.device)[:k]
                        t_mod = t.clone(); t_mod[:, idx, :, :] = 0.0
                        Fenc[-1] = t_mod
                        recon = self.decode(z, Fenc)
                    else:
                        t = f; B,C,H,W = t.shape
                        idx = torch.randperm(C, device=x.device)[:int(0.25*C)]
                        t_mod = t.clone(); t_mod[:, idx, :, :] = 0.0
                        recon = self.decode(z, t_mod)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "drop_ratio": 0.25}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 101) Centroid Consistency (edge COM) ----------
                if algo == "centroid_consistency":
                    edges = _sobel_edges(x)                    # (B,1,H,W) — you already added _sobel_edges earlier in K
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    edges_r = _sobel_edges(r)
                    c_x = _center_of_mass(edges)
                    c_r = _center_of_mass(edges_r)
                    com = F.mse_loss(c_r, c_x.detach())
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.05 * com
                    logs = {"recon": rec.item(), "com": com.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 102) Thin-Edge Reconstruction ----------
                if algo == "thin_edge_recon":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    e_x = _sobel_edges(x); e_r = _sobel_edges(r)
                    # promote thin edges: L1 on edges + TV on edge map
                    el = F.l1_loss(e_r, e_x.detach())
                    tv = _total_variation(e_r)
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.2*el + 0.001*tv
                    logs = {"recon": rec.item(), "edge": el.item(), "tv_edge": tv.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 103) Color-Space Consistency (YUV) ----------
                if algo == "colorspace_consistency":
                    yuv = _rgb_to_yuv(x)
                    z, feats = self.encode(yuv); r_yuv = self.decode(z, feats)
                    # RGB reconstruction from predicted YUV
                    r_rgb = _yuv_to_rgb(r_yuv).clamp(0,1)
                    # losses in both spaces
                    L_rgb = self.recon_loss(r_rgb, x)
                    L_yuv = self.recon_loss(r_yuv, yuv)
                    loss = L_rgb + 0.2 * L_yuv
                    logs = {"recon_rgb": L_rgb.item(), "recon_yuv": L_yuv.item()}
                    return {"loss": loss, "logs": logs, "recon": r_rgb}

                # ---------- 104) BG/FG Separation (self mask) ----------
                if algo == "bg_fg_sep":
                    # self-derive FG mask from edges + symmetry cue, then weight recon on FG
                    e = _sobel_edges(x)
                    s = _symmetry_map(x)
                    m = (e + 0.3*s)
                    m = m / (m.amax(dim=(-2,-1), keepdim=True) + 1e-6)
                    m = (m > 0.25).float()  # coarse FG
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    # emphasize FG pixels
                    w = m
                    denom = w.mean().clamp_min(1e-6)
                    if self.cfg.recon_loss == "mse":
                        fg = ((r - x)**2 * w).sum() / (x.numel()*denom)
                    elif self.cfg.recon_loss == "l1":
                        fg = ((r - x).abs() * w).sum() / (x.numel()*denom)
                    else:
                        hub = F.huber_loss(r, x, delta=self.cfg.huber_delta, reduction="none")
                        fg = (hub * w).sum() / (x.shape[0]*x.shape[2]*x.shape[3]*denom)
                    # small BG term to avoid drift
                    bg = self.recon_loss(r*(1-w), x*(1-w))
                    loss = fg + 0.1*bg
                    logs = {"fg": fg.item(), "bg": bg.item(), "fg_ratio": float(denom)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 90) MS-SSIM Reconstruction ----------
                if algo == "ms_ssim_recon":
                    z, feats = self.encode(x); recon = self.decode(z, feats)
                    # loss = (1 - MS-SSIM) + small MSE tether
                    ssim_val = _msssim(recon.clamp(0,1), x.clamp(0,1), levels=3)
                    loss = (1.0 - ssim_val) + 0.1 * F.mse_loss(recon, x)
                    logs = {"msssim": float(ssim_val.item())}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 91) Edge-Preserving AE ----------
                if algo == "edge_preserve":
                    z, feats = self.encode(x); recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    e_tgt = _sobel_edges(x); e_rec = _sobel_edges(recon)
                    e_loss = F.l1_loss(e_rec, e_tgt)
                    tv = _total_variation(recon)
                    loss = rec + 0.2*e_loss + 0.001*tv
                    logs = {"recon": rec.item(), "edge": e_loss.item(), "tv": tv.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 92) Histogram-Matching AE (soft) ----------
                if algo == "hist_match":
                    z, feats = self.encode(x); recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    h_x  = _soft_hist(x.clamp(0,1), bins=32, sigma=0.02)
                    h_r  = _soft_hist(recon.clamp(0,1), bins=32, sigma=0.02)
                    h_loss = F.mse_loss(h_r, h_x.detach())
                    loss = rec + 0.05 * h_loss
                    logs = {"recon": rec.item(), "hist": h_loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 93) Frequency-Band AE ----------
                if algo == "freq_band":
                    # split target into low/high and force recon to match both
                    z, feats = self.encode(x); recon = self.decode(z, feats)
                    xl, xh = _fft_split(x)
                    rl, rh = _fft_split(recon)
                    L = self.recon_loss(rl, xl) + self.recon_loss(rh, xh)
                    logs = {"freq_loss": L.item()}
                    return {"loss": L, "logs": logs, "recon": recon}

                # ---------- 94) Perceptual-Contrast (recon↔orig in-batch InfoNCE) ----------
                if algo == "perceptual_contrast":
                    z_o, feats_o = self.encode(x); r = self.decode(z_o, feats_o)
                    z_r, feats_r = self.encode(r.detach())
                    h_o = F.normalize(z_o.flatten(2).mean(-1), dim=1)
                    h_r = F.normalize(z_r.flatten(2).mean(-1), dim=1)
                    tau = 0.2
                    logits = (h_r @ h_o.t()) / tau
                    targets = torch.arange(x.shape[0], device=x.device)
                    nce = F.cross_entropy(logits, targets)
                    rec = self.recon_loss(r, x)
                    loss = nce + 0.1 * rec
                    logs = {"nce": nce.item(), "recon": rec.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 95) Multi-Objective (recon + MS-SSIM + self-perceptual) ----------
                if algo == "multi_objective":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    ssim_val = _msssim(r.clamp(0,1), x.clamp(0,1))
                    f_orig = feats[-1] if isinstance(feats, (list,tuple)) else z
                    f_rec, _ = self.encode(r.detach())
                    f_rec = f_rec if f_rec.shape == f_orig.shape else F.interpolate(f_rec, size=f_orig.shape[-2:])
                    perc = F.l1_loss(f_rec, f_orig.detach())
                    # cosine schedule for weights (epoch not available here; use batch-noise proxy)
                    # simple static mix that works well:
                    loss = rec + 0.5*(1.0 - ssim_val) + 0.1*perc
                    logs = {"recon": rec.item(), "msssim": float(ssim_val.item()), "self_perc": perc.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 96) Anti-Artifact (anti-blocking/deringing) ----------
                if algo == "anti_artifact":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    # discourage seam energy on 8×8 boundaries in recon
                    seam = _block_seam_penalty(r, bs=8)
                    # also discourage extra high-freq ringing vs target
                    _, rh = _fft_split(r); _, xh = _fft_split(x)
                    ring = F.l1_loss(rh, xh)
                    loss = rec + 0.01*seam + 0.05*ring
                    logs = {"recon": rec.item(), "seam": seam.item(), "ring": ring.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

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


                # ---------- Spectral AE (FFT masking/reconstruction) ----------
                if algo == "spectral":
                    # mask in frequency domain; reconstruct original spatial image
                    X = torch.fft.rfft2(x, norm="ortho")
                    B, C, Hf, Wf = X.shape
                    ratio = self.cfg.mask_ratio
                    # rectangular notch
                    mh = max(1, int(Hf * ratio * 0.5)); mw = max(1, int(Wf * ratio * 0.5))
                    y0 = torch.randint(0, max(1, Hf - mh), (1,), device=x.device).item()
                    x0 = torch.randint(0, max(1, Wf - mw), (1,), device=x.device).item()
                    M = torch.ones_like(X)
                    M[..., y0:y0+mh, x0:x0+mw] = 0.0
                    Xc = X * M
                    x_cor = torch.fft.irfft2(Xc, s=x.shape[-2:], norm="ortho").real
                    z, feats = self.encode(x_cor)
                    recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "mask_ratio": ratio}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 73) Adversarially-Robust Reconstruction ----------
                if algo == "adv_recon":
                    # create an adversarially corrupted input (PGD-like on input) and train to reconstruct clean x
                    x_adv = x.clone().detach().requires_grad_(True)
                    eps = 4/255.0
                    steps = 3
                    step = 2/255.0
                    for _ in range(steps):
                        z, feats = self.encode(x_adv)
                        recon = self.decode(z, feats)
                        loss_adv = self.recon_loss(recon, x)  # maximize this w.r.t. x_adv
                        (loss_adv).backward()
                        with torch.no_grad():
                            x_adv = x_adv + step * x_adv.grad.sign()
                            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                            x_adv = x_adv.clamp(0,1).detach().requires_grad_(True)
                        x_adv.grad = None
                    # now train the model on (x_adv -> x)
                    z, feats = self.encode(x_adv)
                    recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "eps": float(eps)}
                    return {"loss": loss, "logs": logs, "recon": recon, "aux": {"x_adv": x_adv.detach()}}

            # ---------- 81) Global-Contrast AE (single-encoder InfoNCE) ----------
            if algo == "global_contrast":
                # two independent augs; invariance in latent with in-batch negatives
                def aug(img):
                    k = random.choice([0,1,2,3])
                    j = img * random.uniform(0.8,1.2) + random.uniform(-0.1,0.1)
                    if random.random()<0.5: j = torch.flip(j, dims=[-1])
                    return torch.rot90(j.clamp(0,1), k, dims=[-2,-1])

                x1, x2 = aug(x), aug(x)
                z1, f1 = self.encode(x1); z2, f2 = self.encode(x2)
                h1 = F.normalize(z1.flatten(2).mean(-1), dim=1)   # (B,D)
                h2 = F.normalize(z2.flatten(2).mean(-1), dim=1)   # (B,D)
                tau = 0.2
                logits = (h1 @ h2.t()) / tau
                targets = torch.arange(x.shape[0], device=x.device)
                nce = F.cross_entropy(logits, targets)
                # small recon tether
                r1 = self.decode(z1, f1)
                rec = self.recon_loss(r1, x1)
                loss = nce + 0.1 * rec
                logs = {"nce": nce.item(), "recon": rec.item()}
                return {"loss": loss, "logs": logs, "recon": r1}

            # ---------- 82) MixUp-Reconstruction (predict both sources) ----------
            if algo == "mixup_recon":
                B = x.size(0)
                perm = torch.randperm(B, device=x.device)
                x2 = x[perm]
                lam = random.uniform(0.3, 0.7)
                xm = lam * x + (1-lam) * x2
                z, f = self.encode(xm)
                # lazily create two tiny heads to predict source1/source2 from the same decoded feature map
                if not hasattr(self, "_mixup_head1"):
                    Cdec = f.shape[1]
                    self._mixup_head1 = nn.Conv2d(Cdec, x.shape[1], 1).to(x.device)
                    self._mixup_head2 = nn.Conv2d(Cdec, x.shape[1], 1).to(x.device)
                feat = f  # use top encoder feature as carrier for both heads
                # decode once to restore spatial details, then 1x1 to RGB
                base = self.decode(z, f)
                s1 = self._mixup_head1(f) + 0.0*base  # keep graph; base helps gradients flow through decoder path
                s2 = self._mixup_head2(f) + 0.0*base
                # reconstruct both originals; optionally weight by mix coeffs
                L1 = self.recon_loss(s1, x)
                L2 = self.recon_loss(s2, x2)
                loss = lam*L1 + (1-lam)*L2
                logs = {"src1": L1.item(), "src2": L2.item(), "lam": lam}
                # show sum as a quick preview of separation quality
                recon = (lam*s1 + (1-lam)*s2).clamp(0,1)
                return {"loss": loss, "logs": logs, "recon": recon}

            # ---------- 83) CutMix-Reconstruction (anchor restoration) ----------
            if algo == "cutmix_recon":
                B, C, H, W = x.shape
                perm = torch.randperm(B, device=x.device)
                x2 = x[perm]
                # rectangle region from x2 into x
                rh, rw = H//4, W//4
                y0 = random.randint(0, max(0, H-rh)); x0 = random.randint(0, max(0, W-rw))
                mask = torch.ones(B,1,H,W, device=x.device)
                mask[:,:,y0:y0+rh, x0:x0+rw] = 0.0
                xm = x*mask + x2*(1-mask)
                z, f = self.encode(xm); rec = self.decode(z, f)
                # emphasize the replaced region (we want to bring back the anchor x)
                w = 1.0 - mask
                denom = w.mean().clamp_min(1e-6)
                if self.cfg.recon_loss == "mse":
                    loss = ((rec - x)**2 * w).sum() / (x.numel() * denom)
                elif self.cfg.recon_loss == "l1":
                    loss = ((rec - x).abs() * w).sum() / (x.numel() * denom)
                else:
                    hub = F.huber_loss(rec, x, delta=self.cfg.huber_delta, reduction="none")
                    loss = (hub * w).sum() / (x.shape[0]*x.shape[2]*x.shape[3]*denom)
                logs = {"recon_masked": loss.item()}
                return {"loss": loss, "logs": logs, "recon": rec}

            # ---------- 84) Cross-Jigsaw (swap K patches across two images) ----------
            if algo == "cross_jigsaw":
                B,C,H,W = x.shape; G = 3
                ph, pw = H//G, W//G
                perm = torch.randperm(B, device=x.device)
                x2 = x[perm]
                # sample K swap indices
                K = random.randint(1, G) * G // 2
                idxs = random.sample(range(G*G), K)
                xm1 = x.clone(); xm2 = x2.clone()
                for k in idxs:
                    r, c = divmod(k, G)
                    r0, r1 = r*ph, (r+1)*ph; c0, c1 = c*pw, (c+1)*pw
                    tmp = xm1[:,:,r0:r1,c0:c1].clone()
                    xm1[:,:,r0:r1,c0:c1] = xm2[:,:,r0:r1,c0:c1]
                    xm2[:,:,r0:r1,c0:c1] = tmp
                # train to reconstruct both originals
                z1, f1 = self.encode(xm1); r1 = self.decode(z1, f1)
                z2, f2 = self.encode(xm2); r2 = self.decode(z2, f2)
                L1 = self.recon_loss(r1, x)
                L2 = self.recon_loss(r2, x2)
                loss = 0.5*(L1+L2)
                logs = {"rec1": L1.item(), "rec2": L2.item(), "K": K}
                return {"loss": loss, "logs": logs, "recon": r1}

            # ---------- 85) Style-Swap Recon (AdaIN: style B on A → reconstruct A) ----------
            if algo == "style_swap_recon":
                B = x.size(0)
                perm = torch.randperm(B, device=x.device)
                x_style = x[perm]
                fA, _ = self.encode(x)           # content features (top encoder feat)
                fB, _ = self.encode(x_style)     # style features
                f_mix = _adain(fA, fB)
                # decode from style-swapped features
                rec = self.decode(f_mix, [f_mix])  # reuse as carrier
                # target is the *content* image x; small Gram consistency to encourage style learning
                Lpix = self.recon_loss(rec, x)
                Gt = _gram(fB.detach()); Gp = _gram(self.encode(rec)[0])
                Lsty = (Gp - Gt).abs().mean()
                loss = Lpix + 0.02 * Lsty
                logs = {"recon": Lpix.item(), "gram": Lsty.item()}
                return {"loss": loss, "logs": logs, "recon": rec}

            # ---------- 86) Cross-Color Recon (Y from A, chroma from B → reconstruct A) ----------
            if algo == "cross_color_recon":
                Bch, Gch, Rch = x[:,2:3], x[:,1:2], x[:,0:1]  # keep order explicit
                Ya = 0.299*Rch + 0.587*Gch + 0.114*Bch        # luminance of A
                perm = torch.randperm(x.size(0), device=x.device)
                xb = x[perm]
                Rb, Gb, Bb = xb[:,0:1], xb[:,1:2], xb[:,2:3]
                Yb = 0.299*Rb + 0.587*Gb + 0.114*Bb
                Cb = Bb - Yb; Cr = Rb - Yb
                xmix = torch.cat([Ya + Cr, Ya + (0*Cb), Ya + Cb], dim=1).clamp(0,1)
                z, f = self.encode(xmix); rec = self.decode(z, f)
                loss = self.recon_loss(rec, x)
                logs = {"recon": loss.item()}
                return {"loss": loss, "logs": logs, "recon": rec}

            # ---------- 87) Multi-Crop Consistency (global+local crops) ----------
            if algo == "multi_crop_consistency":
                def rand_crop(img, size_ratio=(0.5, 1.0)):
                    H,W = img.shape[-2:]
                    s = random.uniform(*size_ratio)
                    h = max(8, int(H*s)); w = max(8, int(W*s))
                    y0 = random.randint(0, H-h); x0 = random.randint(0, W-w)
                    out = torch.zeros_like(img); out[:,:,y0:y0+h,x0:x0+w] = img[:,:,y0:y0+h,x0:x0+w]
                    return out
                xg1, xg2 = rand_crop(x,(0.7,1.0)), rand_crop(x,(0.7,1.0))
                xl = [rand_crop(x,(0.3,0.5)) for _ in range(2)]
                zs, hs, rec_loss = [], [], 0.0
                for img in [xg1, xg2] + xl:
                    z,f = self.encode(img); hs.append(F.normalize(z.flatten(2).mean(-1), dim=1))
                    rec_loss = rec_loss + self.recon_loss(self.decode(z,f), img)
                # invariance across all crops (average pairwise MSE)
                inv = 0.0; M = len(hs)
                for i in range(M):
                    for j in range(i+1, M):
                        inv = inv + F.mse_loss(hs[i], hs[j].detach())
                inv = inv * (2.0/(M*(M-1)))
                loss = rec_loss/M + 0.1 * inv
                logs = {"recon": (rec_loss/M).item(), "latent_inv": inv.item()}
                return {"loss": loss, "logs": logs, "recon": self.decode(*self.encode(xg1))}

            # ---------- 88) Neighbor-Patch Consistency ----------
            if algo == "neighbor_patch_consistency":
                G = 3
                patches = []
                B,C,H,W = x.shape; ph, pw = H//G, W//G
                for r in range(G):
                    for c in range(G):
                        patches.append(x[:,:,r*ph:(r+1)*ph, c*pw:(c+1)*pw])
                K = len(patches)
                i = random.randrange(K)
                # pick a spatial neighbor of i (if possible)
                ri, ci = divmod(i, G)
                cand = []
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if abs(dr)+abs(dc)==1:
                            rr, cc = ri+dr, ci+dc
                            if 0<=rr<G and 0<=cc<G:
                                cand.append(rr*G+cc)
                j = random.choice(cand) if cand else (i+1) % K
                xi, xj = patches[i], patches[j]
                _, zi = self.encode(xi); _, zj = self.encode(xj)
                ei = F.normalize(zi.flatten(2).mean(-1), dim=1)
                ej = F.normalize(zj.flatten(2).mean(-1), dim=1)
                # neighbor pairs should be closer than non-neighbors sampled from batch
                perm = torch.randperm(B, device=x.device)
                xk = patches[random.randrange(K)][perm]
                _, zk = self.encode(xk)
                ek = F.normalize(zk.flatten(2).mean(-1), dim=1)
                margin = 0.1
                trip = F.relu(margin + (ei - ej).pow(2).sum(dim=1) - (ei - ek).pow(2).sum(dim=1)).mean()
                # small recon tether on xi
                fi,_ = self.encode(xi); rec = self.recon_loss(self.decode(fi,_), xi)
                loss = trip + 0.1 * rec
                logs = {"triplet": trip.item(), "recon": rec.item()}

                # ---------- 74) JPEG/Block-Artifact Recovery (approximate) ----------
                if algo == "jpeg_recover":
                    # cheap JPEG-ish corruption: block-average + quantize + upsample (introduces blocking/ringing)
                    B, C, H, W = x.shape
                    bs = 8
                    pool = nn.AvgPool2d(bs, bs, ceil_mode=False)
                    unpool = nn.Upsample(scale_factor=bs, mode="nearest")
                    q = 32  # quantization levels
                    x_blk = unpool(pool(x))
                    x_q = torch.round(x_blk * q) / q
                    z, feats = self.encode(x_q)
                    recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "block": bs, "q": q}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 75) Super-Resolution ×2 ----------
                if algo == "superres_x2":
                    # downsample by 2 (avg), upsample (nearest), then reconstruct original
                    x_lr = F.avg_pool2d(x, kernel_size=2, stride=2)
                    x_up = F.interpolate(x_lr, scale_factor=2, mode="nearest")
                    z, feats = self.encode(x_up)
                    recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recon, "aux": {"lr": x_lr}}

                # ---------- 76) Scale-Consistency (random scale → canonical) ----------
                if algo == "scale_consistency":
                    s = random.uniform(0.6, 1.4)
                    H, W = x.shape[-2:]
                    xs = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
                    xs = F.interpolate(xs, size=(H, W), mode="bilinear", align_corners=False)
                    z, feats = self.encode(xs); recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "scale": s}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 77) Bit-Depth Dequantization ----------
                if algo == "bit_dequant":
                    # quantize to n bits then reconstruct 8-bit space
                    nbits = 4  # 4-bit → 8-bit
                    levels = (2**nbits) - 1
                    xq = torch.round(x * levels) / levels
                    z, feats = self.encode(xq); recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "nbits": nbits}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 78) Chroma Subsample / Upsample ----------
                if algo == "chroma_upsample":
                    # crude YCbCr-like split: Y = luminance; Cb/Cr ~= (B-R) and (G-R) surrogates
                    R, G, Bc = x[:,0:1], x[:,1:2], x[:,2:3]
                    Y  = 0.299*R + 0.587*G + 0.114*Bc
                    Cb = Bc - Y
                    Cr = R  - Y
                    # subsample chroma 2× (like 4:2:0), then upsample
                    Cb_s = F.avg_pool2d(Cb, 2, 2)
                    Cr_s = F.avg_pool2d(Cr, 2, 2)
                    Cb_u = F.interpolate(Cb_s, scale_factor=2, mode="bilinear", align_corners=False)
                    Cr_u = F.interpolate(Cr_s, scale_factor=2, mode="bilinear", align_corners=False)
                    x_chr = torch.cat([Y + Cr_u, Y + (0*Cb_u), Y + Cb_u], dim=1).clamp(0,1)
                    z, feats = self.encode(x_chr); recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 79) OOD Score via Reconstruction Error ----------
                if algo == "ood_score":
                    # standard recon; also return a per-sample MSE to be used as an OOD score
                    z, feats = self.encode(x); recon = self.decode(z, feats)
                    per = ((recon - x)**2).flatten(1).mean(1)  # (B,)
                    loss = per.mean()
                    logs = {"recon": loss.item(), "ood_avg": per.mean().item()}
                    return {"loss": loss, "logs": logs, "recon": recon, "aux": {"ood_per_sample": per.detach()}}

                # ---------- 80) Motion-Deblur ----------
                if algo == "motion_deblur":
                    # apply a small horizontal/vertical blur and reconstruct sharp
                    B, C, H, W = x.shape
                    k = random.choice([3,5,7])
                    ker = torch.zeros(1,1,k,k, device=x.device)
                    if random.random() < 0.5:
                        ker[0,0,k//2,:] = 1.0 / k  # horizontal
                    else:
                        ker[0,0,:,k//2] = 1.0 / k  # vertical
                    x_blur = F.conv2d(x, ker.expand(C,1,k,k), padding=k//2, groups=C)
                    z, feats = self.encode(x_blur); recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item(), "k": k}
                    return {"loss": loss, "logs": logs, "recon": recon, "aux": {"blur": x_blur}}

                # ---------- 66) Transform-Equivariant AE ----------
                if algo == "equivariant":
                    # pick a random geometric transform T (rotation {0,90,180,270} + optional flip)
                    k = random.choice([0,1,2,3])
                    do_flip = random.random() < 0.5
                    xT = torch.rot90(x, k, dims=[-2, -1])
                    if do_flip: xT = torch.flip(xT, dims=[-1])

                    # encode both views
                    z0, f0 = self.encode(x); zT, fT = self.encode(xT)

                    # decode transformed, then "undo" T before comparing to original
                    recT = self.decode(zT, fT)
                    recT_back = torch.rot90(recT, (4 - k) % 4, dims=[-2, -1])
                    if do_flip: recT_back = torch.flip(recT_back, dims=[-1])

                    # reconstruction equivariance + latent consistency
                    rloss = self.recon_loss(recT_back, x)
                    # global-average latent vectors
                    h0 = z0.flatten(2).mean(-1); hT = zT.flatten(2).mean(-1)
                    # we want hT to *encode* the same content (equivariance ≈ invariance in pooled latent)
                    h_loss = F.mse_loss(hT, h0.detach())
                    loss = rloss + 0.1 * h_loss
                    logs = {"recon": rloss.item(), "latent_eq": h_loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recT_back}

                # ---------- 67) Augmentation-Consistency AE ----------
                if algo == "aug_consistency":
                    # two independent strong augs; encourage latent agreement + self recon
                    def aug(img):
                        # lightweight on-the-fly augs (no torchvision dependency here)
                        # jitters + small rotate
                        alpha = random.uniform(0.8, 1.2); beta = random.uniform(-0.1, 0.1)
                        j = (img * alpha + beta).clamp(0,1)
                        k = random.choice([0,1,2,3])
                        if random.random() < 0.5: j = torch.flip(j, dims=[-1])
                        return torch.rot90(j, k, dims=[-2, -1])

                    x1, x2 = aug(x), aug(x)
                    z1, f1 = self.encode(x1); r1 = self.decode(z1, f1)
                    z2, f2 = self.encode(x2); r2 = self.decode(z2, f2)

                    # latent agreement (invariance) + per-view recon
                    h1 = z1.flatten(2).mean(-1); h2 = z2.flatten(2).mean(-1)
                    inv = F.mse_loss(h1, h2.detach())
                    rec = 0.5 * (self.recon_loss(r1, x1) + self.recon_loss(r2, x2))
                    loss = rec + 0.1 * inv
                    logs = {"recon": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r1}

                # ---------- 69) Occlusion-Recovery AE (irregular) ----------
                if algo == "occlusion_recover":
                    # draw several irregular blob occluders
                    B, C, H, W = x.shape
                    mask = torch.ones(B, 1, H, W, device=x.device)
                    num_blobs = random.randint(2, 5)
                    for _ in range(num_blobs):
                        cy = random.randint(0, H-1); cx = random.randint(0, W-1)
                        ry = random.randint(H//16, H//6); rx = random.randint(W//16, W//6)
                        yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
                        blob = (((yy - cy).float()/ry)**2 + ((xx - cx).float()/rx)**2 <= 1.0).float()
                        mask = mask * (1.0 - blob.view(1,1,H,W))
                    xm = x * mask.clamp(min=0.0)
                    z, f = self.encode(xm); rec = self.decode(z, f)

                    # emphasize the occluded area
                    w = 1.0 - mask
                    denom = w.mean().clamp_min(1e-6)
                    if self.cfg.recon_loss == "mse":
                        base = ((rec - x)**2 * w).sum() / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*denom)
                    elif self.cfg.recon_loss == "l1":
                        base = ((rec - x).abs() * w).sum() / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*denom)
                    else:
                        hub = F.huber_loss(rec, x, delta=self.cfg.huber_delta, reduction="none")
                        base = (hub * w).sum() / (x.shape[0]*x.shape[2]*x.shape[3]*denom)
                    loss = base
                    logs = {"recon_masked": base.item()}
                    return {"loss": loss, "logs": logs, "recon": rec, "aux": {"mask": mask}}

                # ---------- 70) Cropped-Reconstruction AE ----------
                if algo == "crop_reconstruct":
                    B, C, H, W = x.shape
                    # choose a random crop (30–60% area)
                    s = random.uniform(0.3, 0.6)
                    h = int(H * math.sqrt(s)); w = int(W * math.sqrt(s))
                    y0 = random.randint(0, max(0, H-h)); x0 = random.randint(0, max(0, W-w))
                    crop = x[:, :, y0:y0+h, x0:x0+w]
                    # place crop on zero canvas at same location (only local view)
                    xm = torch.zeros_like(x)
                    xm[:, :, y0:y0+h, x0:x0+w] = crop
                    z, f = self.encode(xm); rec = self.decode(z, f)
                    # emphasize invisible region (1 where zero originally)
                    mask = torch.zeros(B,1,H,W, device=x.device)
                    mask[:, :, y0:y0+h, x0:x0+w] = 1.0
                    wgt = 1.0 - mask
                    denom = wgt.mean().clamp_min(1e-6)
                    if self.cfg.recon_loss == "mse":
                        base = ((rec - x)**2 * wgt).sum() / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*denom)
                    elif self.cfg.recon_loss == "l1":
                        base = ((rec - x).abs() * wgt).sum() / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*denom)
                    else:
                        hub = F.huber_loss(rec, x, delta=self.cfg.huber_delta, reduction="none")
                        base = (hub * wgt).sum() / (x.shape[0]*x.shape[2]*x.shape[3]*denom)
                    loss = base
                    logs = {"recon_invisible": base.item(), "crop_ratio": s}
                    return {"loss": loss, "logs": logs, "recon": rec}

                # ---------- 71) Geometric-Warp (unwarp) AE ----------
                if algo == "unwarp":
                    # random small perspective-like warp via a flow field, then reconstruct canonical
                    B, C, H, W = x.shape
                    yy, xx = torch.meshgrid(
                        torch.linspace(-1, 1, H, device=x.device),
                        torch.linspace(-1, 1, W, device=x.device),
                        indexing="ij"
                    )
                    # create a gentle swirl/warp field
                    r = torch.sqrt(xx**2 + yy**2)
                    theta = torch.atan2(yy, xx) + 0.15 * torch.sin(3 * r * math.pi)
                    gx = r * torch.cos(theta); gy = r * torch.sin(theta)
                    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(B,1,1,1)  # (B,H,W,2)
                    xw = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
                    z, f = self.encode(xw); rec = self.decode(z, f)
                    loss = self.recon_loss(rec, x)  # unwarp to canonical x
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": rec}

                # ---------- 56) Self-Predictive AE ----------
                if algo == "self_predictive":
                    # Predict a random subset of latent dims from the remaining dims (single frame)
                    z, feats = self.encode(x)                   # z: (B,C,h,w)
                    z_lat = z.flatten(2).mean(-1)               # (B,D)
                    B, D = z_lat.shape
                    D2 = max(1, D // 2)
                    idx = torch.randperm(D, device=x.device)
                    A, Bp = z_lat[:, idx[:D2]], z_lat[:, idx[D2:]]  # A predicts Bp
                    # linear predictor (parameter-free, stop-grad on target)
                    pred = A @ torch.pinverse(A.detach()) @ Bp.detach()
                    sp = F.mse_loss(pred, Bp.detach())
                    # standard reconstruction to tether
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    loss = rec + 0.1 * sp
                    logs = {"recon": rec.item(), "self_pred": sp.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 59) Centered AE (mean(z)≈0) ----------
                if algo == "centered":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    z_lat = z.flatten(2).mean(-1)
                    center = (z_lat.mean(dim=0) ** 2).mean()
                    loss = rec + 0.01 * center
                    logs = {"recon": rec.item(), "center": center.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 60) Sparse-Code AE (L1 on latent + linear decoder tie) ----------
                if algo == "sparse_code":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    z_lat = z.flatten(2).mean(-1)
                    l1 = z_lat.abs().mean()
                    loss = rec + 1e-3 * l1
                    logs = {"recon": rec.item(), "l1": l1.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 61) Gaussianization AE (match N(0,1) moments) ----------
                if algo == "gaussianize":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    z_lat = z.flatten(2).mean(-1)
                    m  = z_lat.mean(dim=0)
                    v  = z_lat.var(dim=0)
                    # target mean=0, var=1; penalize skew/kurtosis slightly (batch moments)
                    skew = ((z_lat - m).pow(3).mean(dim=0) / (v.sqrt()+1e-6)).abs().mean()
                    kurt = (((z_lat - m)**4).mean(dim=0) / (v**2 + 1e-6) - 3.0).abs().mean()
                    m_pen = (m**2).mean() + ((v - 1.0)**2).mean()
                    loss = rec + 0.01 * m_pen + 0.001 * (skew + kurt)
                    logs = {"recon": rec.item(), "moments": m_pen.item(), "skewkurt": (skew+kurt).item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 62) Orthogonal-Subspace AE (decorrelate groups) ----------
                if algo == "orth_subspace":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    z_lat = z.flatten(2).mean(-1)    # (B,D)
                    D = z_lat.shape[1]; G = 2        # two subspaces for simplicity
                    Dg = D // G
                    a, b = z_lat[:, :Dg], z_lat[:, Dg:]
                    # cross-covariance near zero
                    a0 = a - a.mean(0, keepdim=True); b0 = b - b.mean(0, keepdim=True)
                    C = (a0.T @ b0) / (a.shape[0]-1+1e-6)
                    ort = (C**2).mean()
                    loss = rec + 1e-3 * ort
                    logs = {"recon": rec.item(), "crosscov": ort.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 63) Cluster-Consistency AE (in-batch k-means step) ----------
                if algo == "cluster_consistency":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    z_lat = z.flatten(2).mean(-1)              # (B,D)
                    K = min(8, z_lat.shape[0])                 # small in-batch K
                    # init centroids as random samples
                    idx = torch.randperm(z_lat.shape[0], device=x.device)[:K]
                    C = z_lat[idx].clone()                     # (K,D)
                    # 1 Lloyd iteration
                    d2 = ((z_lat.unsqueeze(1) - C.unsqueeze(0))**2).sum(-1)  # (B,K)
                    a  = d2.argmin(dim=1)                                     # (B,)
                    for k in range(K):
                        mask = (a==k)
                        if mask.any():
                            C[k] = z_lat[mask].mean(dim=0)
                    # assignment consistency penalty: ||z - C_a||^2
                    cons = ((z_lat - C[a])**2).mean()
                    loss = rec + 1e-3 * cons
                    logs = {"recon": rec.item(), "cluster": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 64) Deterministic Information Bottleneck (IB surrogate) ----------
                if algo == "ib_det":
                    z, feats = self.encode(x)
                    recon = self.decode(z, feats)
                    rec = self.recon_loss(recon, x)
                    # IB surrogate: shrink latent energy + jacobian smoothness (lightweight)
                    z_lat = z.flatten(2).mean(-1)
                    energy = (z_lat**2).mean()
                    # optional tiny contractive term
                    ones = torch.ones_like(z)
                    (g,) = torch.autograd.grad(outputs=(z*ones).sum(), inputs=x,
                                            retain_graph=True, create_graph=True, allow_unused=True)
                    jac = (g**2).mean() if g is not None else x.new_tensor(0.0)
                    loss = rec + 1e-3 * energy + 1e-4 * jac
                    logs = {"recon": rec.item(), "energy": energy.item(), "jac": float(jac)}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- 65) Rank-Preserving AE (triplet ranking) ----------
                if algo == "rank_preserve":
                    # Preserve relative distances from input-space features to latent space
                    f, z = self.encode(x)                       # f used to form input-space distances
                    recon = self.decode(z, f)                   # (note: decoder expects feats list; we pass f as carrier)
                    rec = self.recon_loss(recon, x)
                    zx = f.mean(dim=[2,3])                      # (B,C) input-proxy embedding
                    z_lat = z.flatten(2).mean(-1)               # (B,D) latent
                    B = x.shape[0]
                    # Sample triplets (i, j, k): Dx(i,j) < Dx(i,k) should imply Dz(i,j) < Dz(i,k)
                    def pdist(U): G = U @ U.t(); n2 = U.pow(2).sum(1,keepdim=True); return (n2 + n2.t() - 2*G).clamp_min(0)
                    Dx = pdist(zx); Dz = pdist(z_lat)
                    # build a few triplets
                    margin = 0.01
                    Lr = x.new_tensor(0.0)
                    T = min(16, B)  # up to 16 triplets
                    for _ in range(T):
                        i, j, k = torch.randint(0, B, (3,), device=x.device)
                        if Dx[i, j] < Dx[i, k]:
                            Lr = Lr + F.relu(margin + Dz[i, j] - Dz[i, k])
                        else:
                            Lr = Lr + F.relu(margin + Dz[i, k] - Dz[i, j])
                    Lr = Lr / T
                    loss = rec + 0.1 * Lr
                    logs = {"recon": rec.item(), "rank": Lr.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- Phase AE (predict phase from magnitude) ----------
                if algo == "phase_pred":
                    X = torch.fft.rfft2(x, norm="ortho")
                    mag = torch.abs(X)
                    phase = torch.angle(X)
                    # feed magnitude (3->3; re-use encoder by tiling/normalizing)
                    m_in = (mag / (mag.amax(dim=(-2,-1), keepdim=True) + 1e-6)).float()
                    z, feats = self.encode(m_in)
                    ph_hat = self.decode(z, feats)           # predict phase image in [-pi, pi]
                    # wrap loss (circular): use 1 - cos delta
                    phase_target = phase
                    wrap_loss = (1 - torch.cos(ph_hat - phase_target)).mean()
                    # reconstruct spatial from predicted (mag + phase)
                    X_hat = mag * torch.exp(1j * ph_hat)
                    x_hat = torch.fft.irfft2(X_hat, s=x.shape[-2:], norm="ortho").real
                    rec = self.recon_loss(x_hat, x)
                    loss = rec + 0.1 * wrap_loss
                    logs = {"recon": rec.item(), "phase": wrap_loss.item()}
                    return {"loss": loss, "logs": logs, "recon": x_hat}

                # ---------- Texture-Inpainting (mask + texture statistics) ----------
                if algo == "texture_inpaint":
                    # cutout-style mask + texture statistic match using *model's* own mid-level feats (no external VGG)
                    mask = self.make_dropblock_mask(x, self.cfg.block_size, self.cfg.mask_ratio)
                    xm = x * mask
                    # encode original and masked for feature stats
                    f_full, _ = self.encode(x)
                    f_mask, feats = self.encode(xm)
                    recon = self.decode(f_mask, feats)


                    return {"loss": loss, "logs": logs, "recon": recon, "aux": aux}


                # ---------- Quantized AE (QAE) ----------
                if algo == "qae":
                    # encode to a conv feature we will quantize (use last encoder feature map)
                    f, feats = self.encode(x)            # f: (B,C,h,w)
                    # project to latent D for codebook
                    D = min(128, f.shape[1])
                    proj = nn.Conv2d(f.shape[1], D, 1, bias=False).to(f.device)
                    z_local = proj(f)
                    # codebook (instantiate lazily once)
                    if not hasattr(self, "_vq"):
                        self._vq = VQCodebook(K=256, D=D, beta=0.25).to(f.device)
                    zq, vq_loss, _ = self._vq(z_local)
                    # lift back to decoder channels
                    up = nn.Conv2d(D, f.shape[1], 1, bias=False).to(f.device)
                    f_q = up(zq)
                    recon = self.decode(f_q, feats)
                    rec = self.recon_loss(recon, x)
                    loss = rec + vq_loss
                    logs = {"recon": rec.item(), "vq": vq_loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

                # ---------- Noise-to-Noise (N2N) ----------
                if algo == "n2n":
                    # make two *independent* noisy realizations; target is the *other* noisy view
                    sigma = 0.1
                    n1 = (x + sigma * torch.randn_like(x)).clamp(0,1)
                    n2 = (x + sigma * torch.randn_like(x)).clamp(0,1)
                    z, feats = self.encode(n1)
                    recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, n2)    # no clean target needed
                    logs = {"recon": loss.item(), "sigma": sigma}
                    return {"loss": loss, "logs": logs, "recon": recon, "aux": {"src": n1, "tgt": n2}}

                # ---------- Missing-Modality (channels) ----------
                if algo == "modality_mask":
                    # randomly zero 1 channel (e.g., RGB -> drop R/G/B)
                    B, C, H, W = x.shape
                    drop = torch.randint(low=0, high=C, size=(B,), device=x.device)
                    xm = x.clone()
                    for b in range(B):
                        xm[b, drop[b], :, :] = 0.0
                    z, feats = self.encode(xm)
                    recon = self.decode(z, feats)
                    loss = self.recon_loss(recon, x)
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recon}

        def gram(F):  # (B,C,h,w) -> (B,C,C)
            B,C,h,w = F.shape
            Fm = F.view(B, C, h*w)
            G = Fm @ Fm.transpose(1,2) / (h*w)
            return G
        # texture (Gram) on masked region only: approximate by weighting feature maps
        tex_full = gram(f_full)
        tex_rec  = gram(self.encode(recon)[0].detach())
        # L1 on masked pixels + Gram discrepancy
        pix = ((recon - x).abs() * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)
        sty = (tex_rec - tex_full).abs().mean()
        loss = pix + 0.05 * sty
        logs = {"pix_masked": float(pix), "gram": float(sty)}

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

class VQCodebook(nn.Module):
    def __init__(self, K=256, D=128, beta=0.25):
        super().__init__()
        self.emb = nn.Embedding(K, D)
        nn.init.uniform_(self.emb.weight, -1/ K, 1/ K)
        self.beta = beta  # commitment weight

    def forward(self, z):
        # z: (B, D, h, w) -> quantize per location
        B, D, h, w = z.shape
        zf = z.permute(0,2,3,1).contiguous().view(-1, D)  # (B*h*w, D)
        # distances to codebook
        e = self.emb.weight                               # (K, D)
        d2 = (zf.pow(2).sum(1, keepdim=True)
              - 2 * zf @ e.t()
              + e.pow(2).sum(1).unsqueeze(0))            # (N, K)
        idx = d2.argmin(dim=1)                            # nearest code index
        zq = self.emb(idx).view(B, h, w, D).permute(0,3,1,2).contiguous()
        # VQ losses (no EMA, single-model)
        # straight-through estimator
        zq_st = z + (zq - z).detach()
        # commitment + codebook move
        commit = self.beta * (z.detach() - zq).pow(2).mean()
        codebook = (z - zq.detach()).pow(2).mean()
        return zq_st, commit + codebook, idx.view(B, h, w)

def _gram(F: torch.Tensor) -> torch.Tensor:
    # F: (B,C,h,w) -> Gram (B,C,C)
    B,C,h,w = F.shape
    Fm = F.view(B, C, h*w)
    return (Fm @ Fm.transpose(1,2)) / (h*w)

def _adain(content: torch.Tensor, style: torch.Tensor, eps=1e-5) -> torch.Tensor:
    # content, style: (B,C,H,W)
    cm, cs = content.mean([2,3], keepdim=True), content.var([2,3], keepdim=True)
    sm, ss = style.mean([2,3], keepdim=True),  style.var([2,3], keepdim=True)
    c_norm = (content - cm) / (cs + eps).sqrt()
    return c_norm * (ss + eps).sqrt() + sm

import math

def _gauss_kernel(ch, k=5, sigma=1.0, device="cpu", dtype=torch.float32):
    ax = torch.arange(k, device=device, dtype=dtype) - (k-1)/2
    g1d = torch.exp(-0.5*(ax/sigma)**2); g1d = g1d / g1d.sum()
    k2d = (g1d[:,None] @ g1d[None,:]).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    return k2d.expand(ch,1,k,k).contiguous()

def _blur(x, k=5, sigma=1.0):
    C = x.shape[1]
    ker = _gauss_kernel(C, k=k, sigma=sigma, device=x.device, dtype=x.dtype)
    return F.conv2d(x, ker, padding=k//2, groups=C)

def _ssim_one_scale(x, y, k=5, sigma=1.5, C1=0.01**2, C2=0.03**2):
    mu_x = _blur(x, k, sigma);    mu_y = _blur(y, k, sigma)
    x_mu = x - mu_x;              y_mu = y - mu_y
    sigma_x = _blur(x_mu*x_mu, k, sigma)
    sigma_y = _blur(y_mu*y_mu, k, sigma)
    sigma_xy= _blur(x_mu*y_mu, k, sigma)
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x.pow(2)+mu_y.pow(2)+C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()

def _msssim(x, y, levels=3):
    # simple MS-SSIM: average SSIM over progressively blurred/downsampled scales
    vals=[]; x_s, y_s = x, y
    for l in range(levels):
        vals.append(_ssim_one_scale(x_s, y_s, k=5, sigma=1.0+0.5*l))
        if l < levels-1:
            x_s = F.avg_pool2d(x_s, 2, 2); y_s = F.avg_pool2d(y_s, 2, 2)
    return sum(vals)/len(vals)

def _sobel_edges(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = kx.transpose(-1,-2)
    C = x.shape[1]
    gx = F.conv2d(x, kx.expand(C,1,3,3), padding=1, groups=C)
    gy = F.conv2d(x, ky.expand(C,1,3,3), padding=1, groups=C)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return mag.max(dim=1, keepdim=True).values  # (B,1,H,W)

def _total_variation(x):
    return (x[...,1:,:]-x[...,:-1,:]).abs().mean() + (x[...,:,1:]-x[...,:,:-1]).abs().mean()

def _soft_hist(x, bins=32, minv=0.0, maxv=1.0, sigma=0.01):
    # x: (B,C,H,W) in [0,1]; returns (B,C,bins) soft histogram
    B,C,H,W = x.shape
    centers = torch.linspace(minv, maxv, bins, device=x.device, dtype=x.dtype).view(1,1,bins,1,1)
    xv = x.unsqueeze(2)  # (B,C,1,H,W)
    d = (xv - centers)/sigma
    w = torch.exp(-0.5*d*d)
    w = w / (w.sum(dim=2, keepdim=True)+1e-6)
    hist = w.sum(dim=[3,4]) / (H*W)  # normalize
    return hist  # (B,C,bins)

def _fft_split(x, ratio=0.25):
    X = torch.fft.rfft2(x, norm="ortho")
    B,C,Hf,Wf = X.shape
    cy, cx = Hf//2, Wf//2
    yy = torch.arange(Hf, device=x.device).view(-1,1)
    xx = torch.arange(Wf, device=x.device).view(1,-1)
    dist = torch.sqrt((yy-cy).float()**2 + (xx-cx).float()**2)
    r = ratio * dist.max()
    low_m = (dist <= r).float()
    high_m= 1.0 - low_m
    Xl = X * low_m
    Xh = X * high_m
    xl = torch.fft.irfft2(Xl, s=x.shape[-2:], norm="ortho").real
    xh = torch.fft.irfft2(Xh, s=x.shape[-2:], norm="ortho").real
    return xl, xh

def _block_seam_penalty(x, bs=8):
    # penalize power concentrated on 8×8 grid seams (blocking/deringing)
    B,C,H,W = x.shape
    loss = 0.0
    # vertical seams
    for c in range(bs, W, bs):
        loss = loss + (x[:,:,:,c:c+1] - x[:,:,:,c-1:c]).abs().mean()
    # horizontal seams
    for r in range(bs, H, bs):
        loss = loss + (x[:,:,r:r+1,:] - x[:,:,r-1:r,:]).abs().mean()
    return loss

import math

def _symmetry_map(x):
    # horizontal flip similarity map in [0,1]
    xf = torch.flip(x, dims=[-1])
    d = (x - xf).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
    m = 1.0 - (d / (d.amax(dim=(-2,-1), keepdim=True) + 1e-6))
    return m  # higher -> more symmetric

def _spatial_softargmax(feat, temperature=0.1):
    # feat: (B,C,H,W) -> per-channel keypoint coords in normalized [-1,1]
    B,C,H,W = feat.shape
    a = feat.pow(2).sum(dim=1, keepdim=True)  # (B,1,H,W) energy
    a = a / (a.amax(dim=(-2,-1), keepdim=True) + 1e-6)
    a = (a / temperature).flatten(2)
    a = torch.softmax(a, dim=-1).view(B,1,H,W)
    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=feat.device),
        torch.linspace(-1,1,W,device=feat.device),
        indexing="ij"
    )
    x_exp = (a * xx).sum(dim=[2,3], keepdim=False)  # (B,1)
    y_exp = (a * yy).sum(dim=[2,3], keepdim=False)  # (B,1)
    return torch.cat([x_exp, y_exp], dim=1)         # (B,2)

def _rgb_to_yuv(x):
    R, G, B = x[:,0:1], x[:,1:2], x[:,2:3]
    Y  = 0.299*R + 0.587*G + 0.114*B
    U  = 0.492*(B - Y)
    V  = 0.877*(R - Y)
    return torch.cat([Y,U,V], dim=1)

def _yuv_to_rgb(yuv):
    Y,U,V = yuv[:,0:1], yuv[:,1:2], yuv[:,2:3]
    R = Y + 1.140*V
    G = Y - 0.395*U - 0.581*V
    B = Y + 2.032*U
    return torch.cat([R,G,B], dim=1)

def _kmeans_one_step(Z, K=4):
    # Z: (B,D) -> soft one-step kmeans assignments, returns centers and assignment
    B,D = Z.shape
    idx = torch.randperm(B, device=Z.device)[:K]
    C = Z[idx].clone()  # (K,D)
    d2 = ((Z.unsqueeze(1) - C.unsqueeze(0))**2).sum(-1)  # (B,K)
    a  = torch.softmax(-d2, dim=1)                        # soft assign
    C  = (a.t() @ Z) / (a.sum(0, keepdim=True).t() + 1e-6)  # (K,D)
    return C, a                                           # centers, soft assign

def _center_of_mass(weight_map):
    # weight_map: (B,1,H,W) non-negative -> (B,2) in [-1,1]
    B,_,H,W = weight_map.shape
    weight = weight_map / (weight_map.sum(dim=[2,3], keepdim=True) + 1e-6)
    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=weight_map.device),
        torch.linspace(-1,1,W,device=weight_map.device),
        indexing="ij"
    )
    x_exp = (weight*xx).sum(dim=[2,3]); y_exp = (weight*yy).sum(dim=[2,3])
    return torch.stack([x_exp, y_exp], dim=1)  # (B,2)

def _strong_aug(x):
    # lightweight, fully differentiable aug (no external torchvision need)
    k = random.choice([0,1,2,3])
    y = x * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
    if random.random() < 0.5: y = torch.flip(y, dims=[-1])
    return torch.rot90(y.clamp(0,1), k, dims=[-2, -1])

def _augmix_three(x):
    a1, a2, a3 = _strong_aug(x), _strong_aug(x), _strong_aug(x)
    w = torch.rand(3, device=x.device); w = w / (w.sum() + 1e-6)
    mix = w[0]*a1 + w[1]*a2 + w[2]*a3
    return mix.clamp(0,1), (a1, a2, a3), w

def _down(x, s=2, mode="area"):
    if mode == "area":
        return F.adaptive_avg_pool2d(x, (x.shape[-2]//s, x.shape[-1]//s))
    return F.interpolate(x, scale_factor=1.0/s, mode="bilinear", align_corners=False)

def _up(x, s=2, mode="nearest"):
    return F.interpolate(x, scale_factor=s, mode=mode, align_corners=False if mode=="bilinear" else None)

def _gauss_blur(x, k=5, sigma=1.0):
    C = x.shape[1]
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k-1)/2
    g1 = torch.exp(-0.5*(ax/sigma)**2); g1 = g1/g1.sum()
    k2 = (g1[:,None]@g1[None,:]).unsqueeze(0).unsqueeze(0).expand(C,1,k,k).contiguous()
    return F.conv2d(x, k2, padding=k//2, groups=C)

def _build_laplacian_pyr(x, levels=3):
    pyrG, pyrL = [], []
    cur = x
    for l in range(levels-1):
        low = _down(cur, 2, "area")
        up  = _up(low, 2, "bilinear")
        lap = cur - up
        pyrL.append(lap)
        cur = low
    pyrG.append(cur)     # coarsest Gaussian
    return pyrL, pyrG    # list of Laplacian bands, list with 1 Gaussian (coarsest)

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
