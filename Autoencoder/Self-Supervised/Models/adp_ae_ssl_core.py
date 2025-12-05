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

                # ---------- 137) Heteroscedastic Reconstruction ----------
                if algo == "hetero_recon":
                    z, feats = self.encode(x)
                    r = self.decode(z, feats)
                    # lazily add a variance head (log σ^2)
                    if not hasattr(self, "_var_head"):
                        Cdec = feats[0].shape[1] if isinstance(feats, (list,tuple)) else z.shape[1]
                        self._var_head = nn.Conv2d(Cdec, 1, 1).to(x.device)
                    # use top encoder feat as carrier for σ
                    f_top = feats[-1] if isinstance(feats, (list,tuple)) else z
                    logvar = self._var_head(f_top).clamp(-8.0, 4.0)  # σ in ~[e^-4, e^2]
                    nll = _gauss_nll(x, r, logvar).mean()
                    # small prior on σ to stop collapse
                    prior = 0.001 * (logvar.exp().mean() + (logvar**2).mean())
                    loss = nll + prior
                    logs = {"nll": float(nll), "sigma_mean": float(logvar.exp().mean())}
                    return {"loss": loss, "logs": logs, "recon": r, "aux": {"logvar": logvar}}

                # ---------- 138) Aleatoric Consistency ----------
                if algo == "aleatoric_consistency":
                    # two different augs; residual^2 should agree with predicted variance
                    def aug(img):
                        s = random.uniform(0.9, 1.1); b = random.uniform(-0.05, 0.05)
                        j = (img*s + b).clamp(0,1)
                        k = random.choice([0,1,2,3])
                        return torch.rot90(j, k, dims=[-2,-1])
                    x1, x2 = aug(x), aug(x)
                    z1,f1 = self.encode(x1); r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(x2); r2 = self.decode(z2,f2)
                    if not hasattr(self, "_var_head"):
                        Cdec = f1[-1].shape[1] if isinstance(f1, (list,tuple)) else z1.shape[1]
                        self._var_head = nn.Conv2d(Cdec, 1, 1).to(x.device)
                    logv1 = self._var_head(f1[-1] if isinstance(f1,(list,tuple)) else z1)
                    logv2 = self._var_head(f2[-1] if isinstance(f2,(list,tuple)) else z2)
                    # agreement: E[(res^2 - σ^2)^2]
                    res1 = (r1 - x1); res2 = (r2 - x2)
                    cons = ((res1.pow(2) - logv1.exp())**2).mean() + ((res2.pow(2) - logv2.exp())**2).mean()
                    rec  = 0.5*(self.recon_loss(r1, x1) + self.recon_loss(r2, x2))
                    loss = rec + 0.05 * cons
                    logs = {"recon": rec.item(), "var_cons": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r1}

                # ---------- 139) Temperature-Scaled Latent ----------
                if algo == "temp_scale_latent":
                    # learn a global temperature T on latent to calibrate residuals
                    z, feats = self.encode(x)
                    if not hasattr(self, "_logT"):
                        self._logT = nn.Parameter(torch.zeros(1, device=x.device))
                    T = self._logT.exp().clamp(0.25, 4.0)
                    zt = z / T
                    r = self.decode(zt, feats)
                    rec = self.recon_loss(r, x)
                    # mild regularizer to keep T near 1
                    reg = (T - 1.0).abs()
                    loss = rec + 0.001 * reg
                    logs = {"recon": rec.item(), "T": float(T.detach())}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 140) Conformal-Quantile Recon (τ-quantile of |error|) ----------
                if algo == "conformal_quantile":
                    tau = 0.9
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    # predict per-pixel quantile qτ for |residual|
                    if not hasattr(self, "_q_head"):
                        Cdec = feats[-1].shape[1] if isinstance(feats,(list,tuple)) else z.shape[1]
                        self._q_head = nn.Conv2d(Cdec, 1, 1).to(x.device)
                    f_top = feats[-1] if isinstance(feats,(list,tuple)) else z
                    q = F.softplus(self._q_head(f_top))  # qτ >= 0
                    err = (r - x).abs()
                    pin = _pinball(err, q, tau).mean()
                    # add recon tether for stability
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.5 * pin
                    logs = {"recon": rec.item(), "pinball": pin.item()}
                    return {"loss": loss, "logs": logs, "recon": r, "aux": {"q_tau": q}}

                # ---------- 141) Selective Reconstruction (confidence with coverage) ----------
                if algo == "selective_recon":
                    target_cov = 0.7
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    if not hasattr(self, "_conf_head"):
                        Cdec = feats[-1].shape[1] if isinstance(feats,(list,tuple)) else z.shape[1]
                        self._conf_head = nn.Conv2d(Cdec, 1, 1).to(x.device)
                    conf = torch.sigmoid(self._conf_head(feats[-1] if isinstance(feats,(list,tuple)) else z))  # (B,1,H,W)
                    # selective loss: weight recon by conf; penalize coverage drift
                    wrec = (conf * (r - x).pow(2)).mean()
                    cov  = conf.mean()
                    cov_pen = (cov - target_cov).abs()
                    # encourage conf ≈ low error (rank correlation surrogate)
                    corr = F.mse_loss(conf, (1.0/(1e-3 + (r - x).abs())).detach())
                    loss = wrec + 0.1*cov_pen + 0.01*corr
                    logs = {"wrec": wrec.item(), "coverage": cov.item()}
                    return {"loss": loss, "logs": logs, "recon": r, "aux": {"conf": conf}}

                # ---------- 142) Uncertainty-Guided Recon (WLS) ----------
                if algo == "uncert_guided":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    if not hasattr(self, "_var_head"):
                        Cdec = feats[-1].shape[1] if isinstance(feats,(list,tuple)) else z.shape[1]
                        self._var_head = nn.Conv2d(Cdec, 1, 1).to(x.device)
                    logv = self._var_head(feats[-1] if isinstance(feats,(list,tuple)) else z).clamp(-8, 4)
                    w = (logv.exp() + 1e-6).reciprocal()       # 1/σ^2
                    w = w / (w.mean() + 1e-8)
                    wls = (w * (r - x).pow(2)).mean()
                    # tiny smoothness on σ map
                    tv = 0.0005 * _total_variation(logv)
                    loss = wls + tv
                    logs = {"wls": wls.item(), "tv_sigma": float(tv)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 143) Robust Worst-of-K ----------
                if algo == "robust_worstofk":
                    # small perturbations within a budget; minimize the worst loss
                    eps = 2/255.0
                    def make_view(img):
                        y = img + eps*torch.randn_like(img)
                        k = random.choice([0,1,2,3])
                        return torch.rot90(y.clamp(0,1), k, dims=[-2,-1])
                    views = _worstofk(x, make_view, K=4)
                    losses, recons = [], []
                    for v in views:
                        z,f = self.encode(v); r = self.decode(z,f)
                        losses.append(self.recon_loss(r, x))
                        recons.append(r)
                    idx = torch.argmax(torch.stack([l.detach() for l in losses]))
                    loss = losses[int(idx)]
                    logs = {"worst_loss": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": recons[int(idx)]}

                # ---------- 144) Error-Calibration Curve ----------
                if algo == "error_calibration":
                    # compare predicted σ to empirical |error| across soft bins
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    if not hasattr(self, "_var_head"):
                        Cdec = feats[-1].shape[1] if isinstance(feats,(list,tuple)) else z.shape[1]
                        self._var_head = nn.Conv2d(Cdec, 1, 1).to(x.device)
                    logv = self._var_head(feats[-1] if isinstance(feats,(list,tuple)) else z)
                    sigma = (logv*0.5).exp().clamp(1e-3, 2.5)  # σ
                    err = (r - x).abs().detach()
                    # build soft histogram bins on predicted σ and compare average errors per bin
                    edges = torch.linspace(0.0, 1.0, 8, device=x.device)
                    wb = _soft_bins(sigma, edges)  # (B,1,H,W,M)
                    # per-bin expected error ≈ sum(w*err)/sum(w)
                    num = (wb * err.unsqueeze(-1)).sum(dim=[1,2,3])   # (B,M)
                    den = (wb.sum(dim=[1,2,3]) + 1e-6)                # (B,M)
                    e_bin = (num/den).mean(dim=0)                     # (M,)
                    s_bin = (wb * sigma.unsqueeze(-1)).sum(dim=[1,2,3]).mean(dim=0) / den.mean(dim=0)
                    # calibration: e_bin ≈ c * s_bin (c≈1); fit c on the fly (least squares)
                    c = ( (e_bin*s_bin).sum() / (s_bin.pow(2).sum() + 1e-8) ).detach()
                    cal = ((e_bin - c*s_bin)**2).mean()
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.1 * cal
                    logs = {"recon": rec.item(), "cal": cal.item(), "c": float(c)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 145) Rate–Distortion (entropy proxy) ----------
                if algo == "rd_lagrange":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    D = self.recon_loss(r, x)
                    R = _entropy_proxy_lat(z)
                    lam = 5e-3  # start small; tune per dataset
                    loss = D + lam * R
                    logs = {"D": D.item(), "R_proxy": R.item(), "lambda": lam}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 146) Latent Quantization (STE) ----------
                if algo == "latent_quant_ste":
                    z, feats = self.encode(x)
                    # quantize latent with learnable step (optional); here fixed Δ=1/16
                    scale = 16.0
                    zq = _round_ste(z * scale) / scale
                    r = self.decode(zq, feats)
                    # tether to clean and to pre-quant recon for stability
                    rc = self.decode(z, feats).detach()
                    rec = self.recon_loss(r, x)
                    ste = F.mse_loss(r, rc)
                    loss = rec + 0.1 * ste
                    logs = {"recon": rec.item(), "ste_cons": ste.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 147) Gate-Prune L1 (channel gating) ----------
                if algo == "gate_prune_l1":
                    z, feats = self.encode(x)
                    # lazy gate vector on top feature channels
                    if isinstance(feats, (list,tuple)):
                        Ftop = feats[-1]
                    else:
                        Ftop = z
                    B,C,H,W = Ftop.shape
                    if not hasattr(self, "_gate"):
                        self._gate = nn.Parameter(torch.ones(C, device=x.device))
                    g = torch.sigmoid(self._gate).view(1,C,1,1)
                    Fg = Ftop * g
                    if isinstance(feats, (list,tuple)):
                        feats_mod = list(feats); feats_mod[-1] = Fg
                    else:
                        feats_mod = Fg
                    r = self.decode(z, feats_mod)
                    rec = self.recon_loss(r, x)
                    spars = g.mean()
                    loss = rec + 1e-3 * spars  # L1-like via sigmoid mean
                    logs = {"recon": rec.item(), "gate_mean": spars.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 148) Distortion-Balance (learnable weights) ----------
                if algo == "distortion_balance":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    if not hasattr(self, "_w_mse"):  # log-weights to keep positivity
                        self._w_mse  = nn.Parameter(torch.tensor(0.0, device=x.device))
                        self._w_ssim = nn.Parameter(torch.tensor(0.0, device=x.device))
                        self._w_perc = nn.Parameter(torch.tensor(0.0, device=x.device))
                    # components
                    Lmse  = F.mse_loss(r, x)
                    Lssim = 1.0 - _msssim(r.clamp(0,1), x.clamp(0,1))
                    f_orig = feats[-1] if isinstance(feats,(list,tuple)) else z
                    f_rec, _ = self.encode(r.detach()); 
                    if f_rec.shape != f_orig.shape: f_rec = F.interpolate(f_rec, size=f_orig.shape[-2:])
                    Lperc = F.l1_loss(f_rec, f_orig.detach())
                    w_mse  = F.softplus(self._w_mse )
                    w_ssim = F.softplus(self._w_ssim)
                    w_perc = F.softplus(self._w_perc)
                    loss = w_mse*Lmse + w_ssim*Lssim + w_perc*Lperc
                    logs = {"Lmse": Lmse.item(), "Lssim": float(Lssim.item()), "Lperc": Lperc.item(),
                            "w_mse": float(w_mse), "w_ssim": float(w_ssim), "w_perc": float(w_perc)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 149) 8×8 DCT Masking Recon ----------
                if algo == "block_dct_mask":
                    # mask high-freq DCT coeffs per 8×8 block; reconstruct original spatial x
                    bs = 8
                    # pad to multiple just in case
                    H,W = x.shape[-2:]
                    H2 = (H//bs)*bs; W2=(W//bs)*bs
                    if H2!=H or W2!=W:
                        x = x[..., :H2, :W2]
                    D, ctx = _block_dct2(x, bs=bs)      # (BC, 64, nblk)
                    # keep a low-freq triangle (quality proxy)
                    keep = 20
                    idx = torch.arange(D.shape[1], device=x.device)
                    mask = (idx < keep).float().view(1,-1,1)
                    Dm = D * mask
                    xm = _block_idct2(Dm, ctx)
                    z, feats = self.encode(xm); r = self.decode(z, feats)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item(), "keep_coeffs": keep}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 150) Bitrate Temperature Control ----------
                if algo == "bitrate_temp_ctrl":
                    target = 0.08  # target |z| mean (rate proxy)
                    z, feats = self.encode(x)
                    if not hasattr(self, "_logT_lat"):
                        self._logT_lat = nn.Parameter(torch.zeros(1, device=x.device))
                    T = self._logT_lat.exp().clamp(0.25, 4.0)
                    zt = z / T
                    r = self.decode(zt, feats)
                    rec = self.recon_loss(r, x)
                    rate = _entropy_proxy_lat(zt)
                    ctrl = _target_match(rate, target)
                    loss = rec + 1e-2 * ctrl
                    logs = {"recon": rec.item(), "rate_proxy": rate.item(), "T": float(T.detach())}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 151) Checkerboard Suppression ----------
                if algo == "checker_suppress":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    ali = _alias_penalty(r)
                    loss = rec + 0.02 * ali
                    logs = {"recon": rec.item(), "alias": ali.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 152) JPEG Round-Trip Consistency ----------
                if algo == "jpeg_roundtrip":
                    # approximate JPEG: block DCT → quantize → dequant → iDCT
                    bs = 8; Q = 16.0
                    H,W = x.shape[-2:]
                    H2 = (H//bs)*bs; W2=(W//bs)*bs
                    if H2!=H or W2!=W:
                        x = x[..., :H2, :W2]
                    D, ctx = _block_dct2(x, bs=bs)
                    Dq = _round_ste(D / Q) * Q
                    xj = _block_idct2(Dq, ctx).clamp(0,1)
                    # train AE to reconstruct clean x from JPEG-corrupted xj,
                    # and also make roundtrip close: r -> jpeg(r) ≈ jpeg(x)
                    z, f = self.encode(xj); r = self.decode(z, f)
                    Dj, ctx2 = _block_dct2(r.clamp(0,1), bs=bs)
                    rj = _block_idct2(_round_ste(Dj / Q)*Q, ctx2).clamp(0,1)
                    Lclean = self.recon_loss(r, x)
                    Lrt    = F.mse_loss(rj, xj)
                    loss = Lclean + 0.5 * Lrt
                    logs = {"clean": Lclean.item(), "roundtrip": Lrt.item(), "Q": Q}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 153) Gamma Sweep ----------
                if algo == "gamma_sweep":
                    g = random.uniform(0.6, 1.6)
                    xg = _gamma_corr(x, g)
                    z0,f0 = self.encode(x);  r0 = self.decode(z0,f0)   # canonical
                    z1,f1 = self.encode(xg); r1 = self.decode(z1,f1)   # gamma-view
                    rec = self.recon_loss(r0, x)
                    h0 = F.normalize(z0.flatten(2).mean(-1), dim=1)
                    h1 = F.normalize(z1.flatten(2).mean(-1), dim=1)
                    inv = F.mse_loss(h0, h1.detach())
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "gamma": g, "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 154) Photometric Affine Consistency ----------
                if algo == "photo_affine_cons":
                    xa = _photo_affine(x)
                    z0,f0 = self.encode(x);  r0 = self.decode(z0,f0)
                    z1,f1 = self.encode(xa); r1 = self.decode(z1,f1)
                    rec = self.recon_loss(r0, x)
                    inv = F.mse_loss(z0.flatten(2).mean(-1), z1.flatten(2).mean(-1).detach())
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 155) RandConv Invariance ----------
                if algo == "randconv_invariance":
                    xr = _rand_depthwise_conv3(x)
                    z0,f0 = self.encode(x);  r0 = self.decode(z0,f0)
                    z1,f1 = self.encode(xr); r1 = self.decode(z1,f1)
                    rec = self.recon_loss(r0, x)
                    cons = F.mse_loss(r0, r1.detach())
                    loss = rec + 0.1*cons
                    logs = {"recon": rec.item(), "recon_cons": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 156) Haze/Fog Recover ----------
                if algo == "haze_fog_recover":
                    Ih, t, A = _synth_haze(x)          # hazy view
                    z,f = self.encode(Ih); r = self.decode(z,f)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item(), "t_mean": float(t.mean()), "A_mean": float(A.mean())}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 157) Background Randomize → Reconstruct ----------
                if algo == "bg_randomize_recon":
                    m = _fg_mask_coarse(x)                   # (B,1,H,W)
                    noise_bg = torch.rand_like(x)
                    xb = x*m + noise_bg*(1-m)
                    z,f = self.encode(xb); r = self.decode(z,f)
                    # emphasize FG reconstruction but keep small BG tether
                    denom = m.mean().clamp_min(1e-6)
                    if self.cfg.recon_loss == "mse":
                        fg = ((r-x).pow(2) * m).sum() / (x.numel()*denom)
                    elif self.cfg.recon_loss == "l1":
                        fg = ((r-x).abs() * m).sum() / (x.numel()*denom)
                    else:
                        hub = F.huber_loss(r, x, delta=self.cfg.huber_delta, reduction="none")
                        fg = (hub*m).sum() / (x.shape[0]*x.shape[2]*x.shape[3]*denom)
                    bg = 0.1 * self.recon_loss(r*(1-m), x*(1-m))
                    loss = fg + bg
                    logs = {"fg": fg.item(), "bg": bg.item(), "fg_ratio": float(denom)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 158) Gamut Consistency (sRGB↔linear) ----------
                if algo == "gamut_consistency":
                    # sRGB ≈ gamma 2.2; compare latent/recon across domains
                    x_lin  = x.clamp(0,1).pow(2.2)          # pseudo-linear
                    x_srgb = x_lin.clamp(0,1).pow(1/2.2)
                    z1,f1 = self.encode(x_lin);  r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(x_srgb); r2 = self.decode(z2,f2)
                    # train in linear domain as canonical
                    rec = self.recon_loss(r1, x_lin)
                    inv = F.mse_loss(z1.flatten(2).mean(-1), z2.flatten(2).mean(-1).detach())
                    loss = rec + 0.1*inv
                    logs = {"recon_lin": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r1.clamp(0,1).pow(1/2.2)}  # show sRGB preview

                # ---------- 159) Histogram-Equalization Consistency ----------
                if algo == "hist_eq_consistency":
                    xe = _soft_equalize(x)
                    z0,f0 = self.encode(x);  r0 = self.decode(z0,f0)
                    z1,f1 = self.encode(xe); r1 = self.decode(z1,f1)
                    rec = self.recon_loss(r0, x)
                    inv = F.mse_loss(z0.flatten(2).mean(-1), z1.flatten(2).mean(-1).detach())
                    # also encourage hist of r0 to match hist of x (soft)
                    hx  = _soft_hist(x.clamp(0,1), bins=32, sigma=0.02)
                    hr0 = _soft_hist(r0.clamp(0,1), bins=32, sigma=0.02)
                    hL = F.mse_loss(hr0, hx.detach())
                    loss = rec + 0.05*inv + 0.02*hL
                    logs = {"recon": rec.item(), "latent_inv": inv.item(), "hist": hL.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 160) AdaBN-Sim Invariance (channel-stats shift) ----------
                if algo == "adabn_sim_invariance":
                    xs = _channel_stats_shift(x)
                    z0,f0 = self.encode(x);  r0 = self.decode(z0,f0)
                    z1,f1 = self.encode(xs); r1 = self.decode(z1,f1)
                    rec = self.recon_loss(r0, x)
                    inv = F.mse_loss(z0.flatten(2).mean(-1), z1.flatten(2).mean(-1).detach())
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 161) Homography Cycle ----------
                if algo == "homography_cycle":
                    B = x.size(0)
                    Hm = _rand_homography(B, device=x.device, dtype=x.dtype)
                    x_w = _warp_homography(x, Hm)
                    z,f = self.encode(x_w); r_w = self.decode(z,f)
                    # unwarp reconstruction with inverse H
                    Hinv = torch.inverse(Hm)
                    r_un = _warp_homography(r_w, Hinv)
                    # cycle to canonical + small recon tether in warped space
                    Lcyc = self.recon_loss(r_un, x)
                    Lt   = 0.3 * self.recon_loss(r_w, x_w)
                    loss = Lcyc + Lt
                    logs = {"cycle": Lcyc.item(), "tether": Lt.item()}
                    return {"loss": loss, "logs": logs, "recon": r_un}

                # ---------- 162) Perspective Invariance ----------
                if algo == "perspective_invariance":
                    B = x.size(0)
                    H1 = _rand_homography(B, device=x.device, dtype=x.dtype)
                    H2 = _rand_homography(B, device=x.device, dtype=x.dtype)
                    x1 = _warp_homography(x, H1); x2 = _warp_homography(x, H2)
                    z1,f1 = self.encode(x1); r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(x2); r2 = self.decode(z2,f2)
                    h1 = F.normalize(z1.flatten(2).mean(-1), dim=1)
                    h2 = F.normalize(z2.flatten(2).mean(-1), dim=1)
                    inv = F.mse_loss(h1, h2.detach())
                    # small reconstruction tethers in each view
                    rec = 0.5*(self.recon_loss(r1, x1) + self.recon_loss(r2, x2))
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r1}

                # ---------- 163) Rectify Recon (affine tilts/shears) ----------
                if algo == "rectify_recon":
                    xa = _rand_affine(x)
                    z,f = self.encode(xa); r = self.decode(z,f)
                    loss = self.recon_loss(r, x)  # canonical target = original
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 164) Vanishing-Orientation Consistency ----------
                if algo == "vanish_consistency":
                    z,f = self.encode(x); r = self.decode(z,f)
                    hx = _grad_orientation_hist(x); hr = _grad_orientation_hist(r.clamp(0,1))
                    hL = F.mse_loss(hr, hx.detach())
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.05*hL
                    logs = {"recon": rec.item(), "ori_hist": hL.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 165) Principal-Point Shift (crop/pad) ----------
                if algo == "ppoint_shift":
                    B,C,H,W = x.shape
                    # crop a random window, then resize back (simulate principal point shift)
                    dh = int(0.1*H); dw = int(0.1*W)
                    y0 = random.randint(0, dh); x0 = random.randint(0, dw)
                    y1 = H - (dh - y0); x1 = W - (dw - x0)
                    xc = F.interpolate(x[..., y0:y1, x0:x1], size=(H,W), mode="bilinear", align_corners=False)
                    z0,f0 = self.encode(x);  r0 = self.decode(z0,f0)
                    zc,fc = self.encode(xc); rc = self.decode(zc,fc)
                    rec = self.recon_loss(r0, x)
                    inv = F.mse_loss(z0.flatten(2).mean(-1), zc.flatten(2).mean(-1).detach())
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "latent_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 166) Radial De-warp ----------
                if algo == "radial_dewarp":
                    xd = _radial_distort(x)
                    z,f = self.encode(xd); r = self.decode(z,f)
                    loss = self.recon_loss(r, x)  # train to undo distortion
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 167) Overlap Stitch Consistency ----------
                if algo == "overlap_stitch":
                    # two overlapping crops should reconstruct consistently on the overlap
                    B,C,H,W = x.shape
                    oh = H//3; ow = W//3
                    y0 = random.randint(0, H-oh-1); x0 = random.randint(0, W-ow-1)
                    y1 = random.randint(0, H-oh-1); x1 = random.randint(0, W-ow-1)
                    A = x[..., y0:y0+2*oh, x0:x0+2*ow]
                    Bv= x[..., y1:y1+2*oh, x1:x1+2*ow]
                    # define overlap in full coords
                    Ay0, Ay1 = y0+oh//2, y0+3*oh//2
                    Ax0, Ax1 = x0+ow//2, x0+3*ow//2
                    By0, By1 = y1, y1+2*oh
                    Bx0, Bx1 = x1, x1+2*ow
                    # encode-decode both (resized back to native for loss)
                    Ar = F.interpolate(self.decode(*self.encode(A)), size=(H,W), mode="bilinear", align_corners=False)
                    Br = F.interpolate(self.decode(*self.encode(Bv)), size=(H,W), mode="bilinear", align_corners=False)
                    # overlap region (intersection)
                    oy0, oy1 = max(Ay0, By0), min(Ay1, By1)
                    ox0, ox1 = max(Ax0, Bx0), min(Ax1, Bx1)
                    if oy1<=oy0 or ox1<=ox0:
                        # degenerate overlap; fall back to full recon tether
                        z,f = self.encode(x); r = self.decode(z,f)
                        loss = self.recon_loss(r, x)
                        logs = {"recon": loss.item(), "overlap": 0}
                        return {"loss": loss, "logs": logs, "recon": r}
                    overlapA = Ar[..., oy0:oy1, ox0:ox1]
                    overlapB = Br[..., oy0:oy1, ox0:ox1]
                    cons = F.mse_loss(overlapA, overlapB.detach())
                    # small global tether
                    glob = 0.5*(self.recon_loss(Ar, x) + self.recon_loss(Br, x))
                    loss = glob + 0.1*cons
                    logs = {"glob": glob.item(), "overlap_cons": cons.item(), "overlap_px": (oy1-oy0)*(ox1-ox0)}
                    return {"loss": loss, "logs": logs, "recon": Ar}

                # ---------- 168) Homography Keypoint Alignment ----------
                if algo == "homo_kp_alignment":
                    B = x.size(0)
                    Hm = _rand_homography(B, device=x.device, dtype=x.dtype)
                    xw = _warp_homography(x, Hm)
                    # pseudo keypoints (sparse mask) on both
                    kp_x  = _corner_kp_map(x)
                    kp_xw = _corner_kp_map(xw)
                    # warp back kp_xw into original frame and compare maps
                    Hinv = torch.inverse(Hm)
                    kp_xw_back = _warp_homography(kp_xw, Hinv)
                    # loss: kp maps agree + recon tether on warped view to avoid trivialities
                    z,f = self.encode(xw); rw = self.decode(z,f)
                    kpL = F.mse_loss(kp_xw_back, kp_x.detach())
                    rec = self.recon_loss(rw, xw)
                    loss = rec + 0.1*kpL
                    logs = {"recon": rec.item(), "kp_align": kpL.item()}
                    return {"loss": loss, "logs": logs, "recon": _warp_homography(rw, Hinv)}

                # ---------- 169) BYOL-StopGrad (single-net) ----------
                if algo == "byol_stopgrad":
                    # two augs; predict stop-grad target from the other view; no EMA/teacher
                    x1, x2 = _simple_aug(x), _simple_aug(x)
                    z1, f1 = self.encode(x1); z2, f2 = self.encode(x2)
                    h1 = _pool_lat(z1); h2 = _pool_lat(z2)          # (B,D)
                    # tiny predictor head (shared) on top features
                    if not hasattr(self, "_pred_head"):
                        D = h1.shape[1]
                        self._pred_head = nn.Sequential(nn.Linear(D, D, bias=False)).to(x.device)
                    p1 = F.normalize(self._pred_head(h1), dim=1)
                    t2 = F.normalize(h2.detach(), dim=1)
                    inv12 = 2 - 2*(p1*t2).sum(dim=1).mean()        # 2*(1 - cosine)
                    # symmetric branch
                    p2 = F.normalize(self._pred_head(h2), dim=1)
                    t1 = F.normalize(h1.detach(), dim=1)
                    inv21 = 2 - 2*(p2*t1).sum(dim=1).mean()
                    # reconstruction tether on one view
                    r = self.decode(z1, f1)
                    rec = self.recon_loss(r, x1)
                    loss = 0.5*(inv12 + inv21) + 0.1*rec
                    logs = {"inv": 0.5*(inv12.item()+inv21.item()), "recon": rec.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 170) VICReg-AE ----------
                if algo == "vicreg_ae":
                    x1, x2 = _simple_aug(x), _simple_aug(x)
                    z1, f1 = self.encode(x1); z2, f2 = self.encode(x2)
                    h1, h2 = _pool_lat(z1), _pool_lat(z2)
                    # invariance
                    iL = F.mse_loss(h1, h2.detach())
                    # variance (avoid collapse)
                    def var_term(h): 
                        s = h.std(dim=0) + 1e-6
                        return F.relu(1.0 - s).mean()
                    vL = var_term(h1) + var_term(h2)
                    # covariance (decorrelate)
                    C = _cov(h1); covL = (_offdiag(C).pow(2).mean() + _offdiag(_cov(h2)).pow(2).mean())
                    # recon tether
                    r = self.decode(z1, f1); rec = self.recon_loss(r, x1)
                    loss = iL + 0.1*vL + 0.05*covL + 0.1*rec
                    logs = {"inv": iL.item(), "var": vL.item(), "cov": covL.item(), "recon": rec.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 171) Barlow-Single ----------
                if algo == "barlow_single":
                    x1, x2 = _simple_aug(x), _simple_aug(x)
                    z1, _ = self.encode(x1); z2, _ = self.encode(x2)
                    h1, h2 = _pool_lat(z1), _pool_lat(z2)
                    C = _corr_matrix(h1, h2)                      # (D,D)
                    on  = (C.diag() - 1).pow(2).mean()
                    off = _offdiag(C).pow(2).mean()
                    r = self.decode(z1, _); rec = self.recon_loss(r, x1)
                    loss = on + 0.01*off + 0.1*rec
                    logs = {"on": on.item(), "off": off.item(), "recon": rec.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 172) Decoder Feature Decor ----------
                if algo == "decoder_decor":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    Ftop = feats[-1] if isinstance(feats,(list,tuple)) else z
                    B,C,H,W = Ftop.shape
                    Fflat = Ftop.permute(0,2,3,1).reshape(-1, C)      # (B*H*W, C)
                    Cc = _cov(Fflat)
                    decor = _offdiag(Cc).pow(2).mean()
                    loss = rec + 0.01*decor
                    logs = {"recon": rec.item(), "decor": decor.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 173) Filter Ortho ----------
                if algo == "filter_ortho":
                    # Orthonormalize conv filters (last encoder block) via penalty
                    rec_loss = 0.0
                    ortho = 0.0
                    # call a forward to get recon/tether
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec_loss = self.recon_loss(r, x)
                    # iterate parameters once; penalize conv weight correlations
                    for m in self.modules():
                        if isinstance(m, nn.Conv2d):
                            W = m.weight  # (Cout,Cin,kh,kw)
                            Cout = W.shape[0]
                            M = W.view(Cout, -1)                     # filters flattened
                            G = (M @ M.t())                          # (Cout,Cout)
                            I = torch.eye(Cout, device=W.device, dtype=W.dtype)
                            ortho = ortho + ((G - I).pow(2).mean())
                    loss = rec_loss + 1e-4*ortho
                    logs = {"recon": rec_loss.item(), "ortho": float(ortho)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 174) Activation Balance ----------
                if algo == "act_balance":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    Ftop = feats[-1] if isinstance(feats,(list,tuple)) else z
                    m, s = _channel_stats(Ftop)
                    # want means ~0 and stds ~ shared median
                    m_pen = m.abs().mean()
                    s_med = s.median()
                    s_pen = (s - s_med).abs().mean()
                    loss = rec + 0.01*m_pen + 0.01*s_pen
                    logs = {"recon": rec.item(), "m_pen": m_pen.item(), "s_pen": s_pen.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 175) Target Sparsity ----------
                if algo == "target_sparsity":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    l1, pen, frac = _soft_sparsity(z, target=0.2)
                    loss = rec + 1e-3*l1 + 0.05*pen
                    logs = {"recon": rec.item(), "l1": l1.item(), "sparse_frac": float(frac)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 176) Latent Whiten Unit ----------
                if algo == "latent_whiten_unit":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    h = _pool_lat(z)
                    C = _cov(h)
                    white = ((C - torch.eye(C.shape[0], device=C.device, dtype=C.dtype))**2).mean()
                    loss = rec + 0.01*white
                    logs = {"recon": rec.item(), "white": white.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 177) Patch-Graph Contrast ----------
                if algo == "patch_graph_contrast":
                    ps, stride = 16, 16
                    B,C,H,W = x.shape
                    P, Gh, Gw = _unfold_patches(x, ps, stride)
                    h, zraw = _patch_latents(self.encode, P)       # (B,n,D)
                    # same-image positives: each patch ↔ its immediate neighbors
                    n = h.shape[1]
                    # pick one neighbor per patch (4-neighborhood)
                    pos_idx = []
                    for r in range(Gh):
                        for c in range(Gw):
                            i = r*Gw+c
                            nbrs = []
                            if r>0: nbrs.append((r-1)*Gw+c)
                            if r< Gh-1: nbrs.append((r+1)*Gw+c)
                            if c>0: nbrs.append(r*Gw+c-1)
                            if c< Gw-1: nbrs.append(r*Gw+c+1)
                            pos_idx.append(random.choice(nbrs))
                    pos_idx = torch.tensor(pos_idx, device=x.device)
                    h_flat = h.reshape(B*n, -1)
                    h_pos  = h.reshape(B*n, -1)[pos_idx.repeat(B) % n]
                    # in-batch negatives
                    logits = (h_flat @ h_pos.t()) / 0.2
                    targets = torch.arange(B*n, device=x.device)
                    nce = F.cross_entropy(logits, targets)
                    # small recon tether on the full image
                    z, f = self.encode(x); r = self.decode(z,f); rec = self.recon_loss(r, x)
                    loss = nce + 0.1*rec
                    logs = {"nce": nce.item(), "recon": rec.item(), "Gh": Gh, "Gw": Gw}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 178) Patch-Order Consistency ----------
                if algo == "patch_order_consistency":
                    ps, stride = 16, 16
                    B,C,H,W = x.shape
                    P, Gh, Gw = _unfold_patches(x, ps, stride)          # (B,n,C,ps,ps)
                    n = P.shape[1]
                    # shuffle patches spatially
                    perm = torch.stack([torch.randperm(n, device=x.device) for _ in range(B)], dim=0)
                    P_shuf = P[torch.arange(B).unsqueeze(1), perm]
                    x_shuf = _fold_patches(P_shuf, H, W, ps, stride)
                    # train to reconstruct canonical x
                    z, f = self.encode(x_shuf); r = self.decode(z,f)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item(), "n_patches": n}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 179) Region-Prototype Consistency ----------
                if algo == "region_proto_consistency":
                    ps, stride = 16, 16
                    B,C,H,W = x.shape
                    P, Gh, Gw = _unfold_patches(x, ps, stride)
                    h, _ = _patch_latents(self.encode, P)                # (B,n,D)
                    # per-image soft K-means → assignments should be sharp & consistent
                    K = min(6, h.shape[1])
                    Ctr = []
                    Assign = []
                    for b in range(B):
                        Cb, ab = _kmeans_one_step(h[b], K=K)            # (K,D), (n,K)
                        Ctr.append(Cb); Assign.append(ab)
                    Ctr = torch.stack(Ctr,0); ab = torch.stack(Assign,0)  # (B,K,D), (B,n,K)
                    recon_loss = self.recon_loss(self.decode(*self.encode(x)), x)
                    # encourage entropy low (confident patch-region assignment)
                    ent = -(ab.clamp_min(1e-6).log() * ab).sum(dim=-1).mean()
                    # encourage proto coverage (avoid collapse): mean assignment ~ uniform
                    cov = F.mse_loss(ab.mean(dim=1), torch.full_like(ab.mean(1), 1.0/K))
                    loss = recon_loss + 0.01*ent + 0.01*cov
                    logs = {"recon": recon_loss.item(), "entropy": ent.item(), "cover": cov.item()}
                    return {"loss": loss, "logs": logs, "recon": self.decode(*self.encode(x))}

                # ---------- 180) Nonlocal Self-Recon ----------
                if algo == "nonlocal_self_recon":
                    ps, stride = 16, 16
                    B,C,H,W = x.shape
                    P, Gh, Gw = _unfold_patches(x, ps, stride)          # (B,n,C,ps,ps)
                    h, _ = _patch_latents(self.encode, P)               # (B,n,D)
                    Bn, n, D = h.shape
                    # for each patch, find a similar patch (nonlocal) within same image and swap a subset
                    P_sw = P.clone()
                    for b in range(Bn):
                        sim = h[b] @ h[b].t()                           # (n,n)
                        nn = sim.topk(k=3, dim=1).indices[:,1]          # 1st is itself; take 2nd best
                        idx = torch.randperm(n, device=x.device)[: n//4]
                        P_sw[b, idx] = P[b, nn[idx]]
                    x_sw = _fold_patches(P_sw, H, W, ps, stride)
                    z,f = self.encode(x_sw); r = self.decode(z,f)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item(), "Gh": Gh, "Gw": Gw}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 181) Patch Cycle Consistency ----------
                if algo == "patch_cycle_consistency":
                    ps = 32
                    B,C,H,W = x.shape
                    y0 = random.randint(0, H-ps); x0 = random.randint(0, W-ps)
                    crop = x[..., y0:y0+ps, x0:x0+ps]
                    # recon full then crop same region from recon; require cycle closeness on the crop
                    z,f = self.encode(x); r = self.decode(z,f)
                    r_crop = r[..., y0:y0+ps, x0:x0+ps]
                    cyc = self.recon_loss(r_crop, crop)
                    # also tether full recon a bit
                    rec = 0.5*self.recon_loss(r, x)
                    loss = cyc + rec
                    logs = {"cycle_crop": cyc.item(), "recon": rec.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 182) CutBlur Recon ----------
                if algo == "cutblur_recon":
                    B,C,H,W = x.shape
                    # pick region and replace with blurred version
                    rh, rw = H//3, W//3
                    y0 = random.randint(0, H-rh); x0 = random.randint(0, W-rw)
                    xb = _gauss_blur(x, k=7, sigma=1.2)
                    xm = x.clone(); xm[..., y0:y0+rh, x0:x0+rw] = xb[..., y0:y0+rh, x0:x0+rw]
                    z,f = self.encode(xm); r = self.decode(z,f)
                    # emphasize the blurred window to be sharpened back
                    w = torch.zeros(B,1,H,W, device=x.device, dtype=x.dtype)
                    w[..., y0:y0+rh, x0:x0+rw] = 1.0
                    denom = w.mean().clamp_min(1e-6)
                    Lw = ((r - x).pow(2) * w).sum() / (x.numel()*denom)
                    Lg = 0.2*self.recon_loss(r, x)
                    loss = Lw + Lg
                    logs = {"win": Lw.item(), "global": Lg.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 183) Set Invariance (multi-patch pools) ----------
                if algo == "set_invariance_pools":
                    ps, stride = 24, 24
                    B,C,H,W = x.shape
                    P, Gh, Gw = _unfold_patches(x, ps, stride)      # (B,n,C,ps,ps)
                    # sample two disjoint sets of patches from same image
                    n = P.shape[1]
                    idxA = torch.randperm(n, device=x.device)[: n//4]
                    idxB = torch.randperm(n, device=x.device)[: n//4]
                    hA, _ = _patch_latents(self.encode, P[:, idxA])
                    hB, _ = _patch_latents(self.encode, P[:, idxB])
                    pA = F.normalize(hA.mean(dim=1), dim=1)         # set pooling
                    pB = F.normalize(hB.mean(dim=1), dim=1)
                    inv = F.mse_loss(pA, pB.detach())
                    # recon tether on full image
                    r = self.decode(*self.encode(x))
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "set_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 185) Contractive Latent (||∂z/∂x||) ----------
                if algo == "contractive_latent":
                    x_req = x.clone().detach().requires_grad_(True)
                    z, feats = self.encode(x_req)
                    # pooled z for Jacobian wrt x
                    if z.dim()==4: zp = z.flatten(2).mean(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x_req)
                    else:          zp = z.unsqueeze(-1).unsqueeze(-1).expand_as(x_req)
                    r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    jac = _hutch_trace_jacobian_norm(zp, x_req)     # ≈ ||∂z/∂x||_F^2
                    loss = rec + 1e-3 * jac
                    logs = {"recon": rec.item(), "jac_x_to_z": float(jac)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 186) Decoder Smoothness (||∂r/∂z||) ----------
                if algo == "decoder_smoothness":
                    z, feats = self.encode(x)
                    z_req = z.clone().detach().requires_grad_(True)
                    r = self.decode(z_req, feats)
                    rec = self.recon_loss(r, x)
                    # shape-match r to z_req by pooling r to a pseudo z-shape map
                    if z_req.dim()==4:
                        r_for_hutch = F.adaptive_avg_pool2d(r, z_req.shape[-2:])
                    else:
                        r_for_hutch = r.flatten(1)[:,:z_req.shape[-1]].view_as(z_req)
                    jac = _hutch_trace_jacobian_norm(r_for_hutch, z_req)   # ≈ ||∂r/∂z||_F^2
                    loss = rec + 1e-4 * jac
                    logs = {"recon": rec.item(), "jac_z_to_r": float(jac)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 187) Latent Lipschitz (Δz vs Δx) ----------
                if algo == "latent_lipschitz":
                    # second, lightly augmented view
                    def aug(img):
                        s = random.uniform(0.9, 1.1); b = random.uniform(-0.05, 0.05)
                        k = random.choice([0,1,2,3]); j = (img*s + b).clamp(0,1)
                        return torch.rot90(j, k, dims=[-2,-1])
                    x2 = aug(x)
                    z1,_ = self.encode(x); z2,_ = self.encode(x2)
                    h1 = z1.flatten(2).mean(-1) if z1.dim()==4 else z1
                    h2 = z2.flatten(2).mean(-1) if z2.dim()==4 else z2
                    dx = (x - x2).flatten(1).norm(dim=1) + 1e-6
                    dz = (h1 - h2).norm(dim=1)
                    kappa = 5.0
                    viol = F.relu(dz - kappa*dx).mean()              # penalize if ||Δz|| > κ||Δx||
                    r = self.decode(*self.encode(x))
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.1*viol
                    logs = {"recon": rec.item(), "lip_viol": viol.item(), "kappa": kappa}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 188) Isometry Pairwise ----------
                if algo == "isometry_pairwise":
                    z, f = self.encode(x); r = self.decode(z,f)
                    rec = self.recon_loss(r, x)
                    h = z.flatten(2).mean(-1) if z.dim()==4 else z
                    D_lat = _pairwise_dists(F.normalize(h, dim=1))
                    D_pix = _pairwise_dists(F.normalize(_downsample_for_metric(x), dim=1))
                    # match after normalizing scales
                    D_lat = D_lat / (D_lat.mean() + 1e-6); D_pix = D_pix / (D_pix.mean() + 1e-6)
                    iso = F.mse_loss(D_lat, D_pix.detach())
                    loss = rec + 0.05*iso
                    logs = {"recon": rec.item(), "iso": iso.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 189) Interp Linearity (image space) ----------
                if algo == "interp_linearity":
                    # sample pairs within batch
                    B = x.size(0); idx = torch.randperm(B, device=x.device)
                    xA, xB = x, x[idx]
                    zA,fA = self.encode(xA); rA = self.decode(zA,fA)
                    zB,fB = self.encode(xB); rB = self.decode(zB,fB)
                    t = 0.5
                    zM = (1-t)*zA + t*zB
                    rM = self.decode(zM, fA)   # reuse skip‐structure from A (works for plain too)
                    lin = self.recon_loss(rM, (1-t)*rA.detach() + t*rB.detach())
                    rec = 0.5*(self.recon_loss(rA, xA) + self.recon_loss(rB, xB))
                    loss = rec + 0.1*lin
                    logs = {"recon": rec.item(), "interp": lin.item()}
                    return {"loss": loss, "logs": logs, "recon": rM}

                # ---------- 190) Interp Mid-Cycle (latent t=0.5) ----------
                if algo == "interp_midcycle":
                    B = x.size(0); idx = torch.randperm(B, device=x.device)
                    zA,_ = self.encode(x); zB,_ = self.encode(x[idx])
                    t = 0.5
                    zM = (1-t)*zA + t*zB
                    rM = self.decode(zM, _ if not isinstance(_, (list,tuple)) else _)
                    # re-encode and match mid latent (cycle)
                    zC,_2 = self.encode(rM.detach())
                    hM = zM.flatten(2).mean(-1) if zM.dim()==4 else zM
                    hC = zC.flatten(2).mean(-1) if zC.dim()==4 else zC
                    cyc = F.mse_loss(hC, hM.detach())
                    # small recon tether from A
                    rec = self.recon_loss(self.decode(zA, _), x)
                    loss = rec + 0.1*cyc
                    logs = {"recon": rec.item(), "mid_cycle": cyc.item()}
                    return {"loss": loss, "logs": logs, "recon": rM}

                # ---------- 191) Curvature Control (second order along v) ----------
                if algo == "curvature_control":
                    z, f = self.encode(x)
                    eps = 0.05
                    if z.dim()==4:
                        v = F.normalize(torch.randn_like(z), dim=1)
                    else:
                        v = F.normalize(torch.randn_like(z), dim=1)
                    r0 = self.decode(z, f)
                    rP = self.decode(z + eps*v, f)
                    rN = self.decode(z - eps*v, f)
                    # discrete second derivative ∝ rP + rN - 2*r0
                    curv = (rP + rN - 2*r0).pow(2).mean()
                    rec = self.recon_loss(r0, x)
                    loss = rec + 0.05*curv
                    logs = {"recon": rec.item(), "curv": curv.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 192) Orthant Balance ----------
                if algo == "orthant_balance":
                    z, f = self.encode(x); r = self.decode(z, f)
                    rec = self.recon_loss(r, x)
                    h = z.flatten(2).mean(-1) if z.dim()==4 else z        # (B,D)
                    # encourage balanced sign usage and similar per-dim scales
                    sign_bal = (h.sign().mean(dim=0).abs()).mean()        # want ~0
                    scale_bal= (h.std(dim=0) - h.std()).abs().mean()
                    loss = rec + 0.01*sign_bal + 0.01*scale_bal
                    logs = {"recon": rec.item(), "sign": sign_bal.item(), "scale": scale_bal.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 193) MixUp Reconstruction ----------
                if algo == "mixup_recon":
                    xm, idx, lam = _mixup(x)
                    zM,fM = self.encode(xm); rM = self.decode(zM,fM)
                    z1,f1 = self.encode(x);  r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(x[idx]); r2 = self.decode(z2,f2)
                    mix_cons = self.recon_loss(rM, (lam*r1 + (1-lam)*r2).detach())
                    rec = self.recon_loss(r1, x)
                    loss = 0.1*rec + mix_cons
                    logs = {"mix_cons": mix_cons.item(), "lam": lam, "recon": rec.item()}
                    return {"loss": loss, "logs": logs, "recon": rM}

                # ---------- 194) CutMix Infill (hole-fill to original) ----------
                if algo == "cutmix_infill":
                    xm, idx, m = _cutmix(x)               # m=1 keep original; 0 → replaced
                    z,f = self.encode(xm); r = self.decode(z,f)
                    # emphasize the replaced region to match the original x (not the donor)
                    w = (1-m)
                    denom = w.mean().clamp_min(1e-6)
                    Lw = ((r - x).pow(2) * w).sum() / (x.numel()*denom)
                    Lg = 0.2 * self.recon_loss(r, x)
                    loss = Lw + Lg
                    logs = {"win": Lw.item(), "global": Lg.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 195) FMix Recon (Fourier mask mix) ----------
                if algo == "fmix_recon":
                    B,C,H,W = x.shape
                    idx = torch.randperm(B, device=x.device)
                    m = _rand_fmask((B,C,H,W))    # (B,1,H,W)
                    xm = x*m + x[idx]*(1-m)
                    zM,fM = self.encode(xm); rM = self.decode(zM,fM)
                    r1 = self.decode(*self.encode(x))
                    r2 = self.decode(*self.encode(x[idx]))
                    mix_cons = self.recon_loss(rM, (m*r1 + (1-m)*r2).detach())
                    loss = mix_cons
                    logs = {"mix_cons": mix_cons.item()}
                    return {"loss": loss, "logs": logs, "recon": rM}

                # ---------- 196) GridMask Recon ----------
                if algo == "gridmask_recon":
                    xg, grid = _gridmask(x)
                    z,f = self.encode(xg); r = self.decode(z,f)
                    # focus on dropped stripes
                    w = (1-grid)
                    denom = w.mean().clamp_min(1e-6)
                    Lw = ((r - x).pow(2) * w).sum() / (x.numel()*denom)
                    Lg = 0.2 * self.recon_loss(r, x)
                    loss = Lw + Lg
                    logs = {"win": Lw.item(), "global": Lg.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 197) Mosaic 2×2 Recon ----------
                if algo == "mosaic_quad_recon":
                    xm = _mosaic2x2(x)
                    z,f = self.encode(xm); r = self.decode(z,f)
                    # canonical recon to original x + encourage downsampled consistency
                    rec = self.recon_loss(r, x)
                    r_ds = F.avg_pool2d(r, 2, 2)
                    x_ds = F.avg_pool2d(x, 2, 2)
                    ds = self.recon_loss(r_ds, x_ds)
                    loss = rec + 0.1*ds
                    logs = {"recon": rec.item(), "down_cons": ds.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 198) MixStyle (feature-statistics mix) ----------
                if algo == "mixstyle_stats":
                    # encode twice; mix top feature statistics; enforce content invariance + recon
                    zA, fA = self.encode(x)
                    idx = torch.randperm(x.size(0), device=x.device)
                    zB, fB = self.encode(x[idx])
                    fA_top = fA[-1] if isinstance(fA,(list,tuple)) else zA
                    fB_top = fB[-1] if isinstance(fB,(list,tuple)) else zB
                    f_mix = _mixstyle_feats(fA_top, fB_top, p=random.uniform(0.3,0.7))
                    if isinstance(fA,(list,tuple)):
                        Fmod = list(fA); Fmod[-1] = f_mix
                        r = self.decode(zA, Fmod)
                    else:
                        r = self.decode(zA, f_mix)
                    rec = self.recon_loss(r, x)
                    # invariance on pooled latent
                    hA = zA.flatten(2).mean(-1) if zA.dim()==4 else zA
                    zMix,_ = self.encode(r.detach())
                    hM = zMix.flatten(2).mean(-1) if zMix.dim()==4 else zMix
                    inv = F.mse_loss(hA, hM.detach())
                    loss = rec + 0.1*inv
                    logs = {"recon": rec.item(), "inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 199) PatchMix Jigsaw ----------
                if algo == "patchmix_jigsaw":
                    xm = _patchmix_jigsaw(x, ps=16)
                    z,f = self.encode(xm); r = self.decode(z,f)
                    loss = self.recon_loss(r, x)
                    logs = {"recon": loss.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 200) Policy Rand-Consensus ----------
                if algo == "policy_randcons":
                    # draw K random policies (mix of gamma/affine/blur/flip/rot), average their recons
                    def _rand_policy(img):
                        y = img
                        if random.random()<0.7:  y = y.clamp(0,1).pow(random.uniform(0.6,1.6))  # gamma
                        if random.random()<0.7:  y = (y*random.uniform(0.8,1.2)+random.uniform(-0.1,0.1)).clamp(0,1) # affine
                        if random.random()<0.5:  y = _gauss_blur(y, k=5, sigma=random.uniform(0.5,1.5))
                        if random.random()<0.5:  y = torch.flip(y, dims=[-1])
                        if random.random()<0.5:  y = torch.rot90(y, random.choice([0,1,2,3]), dims=[-2,-1])
                        return y.clamp(0,1)
                    K = 4
                    Rs = []
                    for _ in range(K):
                        xk = _rand_policy(x)
                        z,f = self.encode(xk); Rs.append(self.decode(z,f))
                    Rmean = torch.stack(Rs, 0).mean(0)
                    # consensus recon should match clean; individual recons should agree with mean
                    rec = self.recon_loss(Rmean, x)
                    var = torch.stack([self.recon_loss(rk, Rmean.detach()) for rk in Rs]).mean()
                    loss = rec + 0.05*var
                    logs = {"recon": rec.item(), "cons_var": float(var)}
                    return {"loss": loss, "logs": logs, "recon": Rmean}

                # ---------- 201) Self-Paced Reconstruction (pixel-level) ----------
                if algo == "self_paced_recon":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    # compute per-pixel squared error
                    err = (r - x).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
                    # focus ratio p(t): start small → grow to 1.0
                    t = _train_progress(self)
                    p = 0.1 + 0.9*t  # from 10% → 100%
                    # threshold at p-quantile per-image
                    q = torch.quantile(err.flatten(2), 1.0 - p, dim=-1, keepdim=True).unsqueeze(-1)  # (B,1,1,1)
                    w = (err >= q).float()
                    denom = w.mean().clamp_min(1e-6)
                    Lp = ((r - x).pow(2) * w).sum() / (x.numel() * denom)
                    loss = Lp
                    logs = {"p_focus": float(p), "loss": float(Lp)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 202) Batch-Hard Reconstruction (sample-level) ----------
                if algo == "batch_hard_recon":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    # per-sample loss
                    per = (r - x).pow(2).flatten(1).mean(dim=1)  # (B,)
                    # keep top-k(t) samples; start small then grow
                    t = _train_progress(self)
                    B = x.size(0); k = max(1, int((0.2 + 0.8*t) * B))
                    topk = torch.topk(per, k=k, largest=True).indices
                    L = per[topk].mean()
                    logs = {"k": k, "mean_topk": float(L)}
                    return {"loss": L, "logs": logs, "recon": r[topk] if r.ndim==4 else r}

                # ---------- 203) Focal Reconstruction ----------
                if algo == "focal_recon":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    gamma = 1.5  # can tune
                    e = (r - x).abs() + 1e-6
                    # focal MSE ≈ e^{2} * (e / e.mean)^γ (normalized)
                    scale = (e / (e.mean(dim=[1,2,3], keepdim=True) + 1e-6)).pow(gamma)
                    L = (scale * (r - x).pow(2)).mean()
                    logs = {"gamma": gamma, "loss": float(L)}
                    return {"loss": L, "logs": logs, "recon": r}

                # ---------- 204) Cosine Noise Curriculum ----------
                if algo == "cosine_noise_curr":
                    t = _train_progress(self)
                    # σ anneals from 0.25 → 0.02
                    sigma = _cos_interp(0.25, 0.02, t)
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    z, feats = self.encode(xn); r = self.decode(z, feats)
                    L = self.recon_loss(r, x)
                    logs = {"sigma": float(sigma), "recon": L.item()}
                    return {"loss": L, "logs": logs, "recon": r}

                # ---------- 205) Occlusion Growth (cutout) ----------
                if algo == "occlusion_growth":
                    t = _train_progress(self)
                    frac = 0.05 + 0.35*t  # area fraction 5%→40%
                    B,C,H,W = x.shape
                    m = _cutout_mask(B, H, W, frac).to(x.device).type_as(x)  # 1=keep, 0=occlude
                    xo = x*m
                    z, feats = self.encode(xo); r = self.decode(z, feats)
                    # emphasize occluded region
                    w = (1-m)
                    denom = w.mean().clamp_min(1e-6)
                    Lw = ((r - x).pow(2) * w).sum() / (x.numel() * denom)
                    Lg = 0.2 * self.recon_loss(r, x)
                    loss = Lw + Lg
                    logs = {"frac": float(frac), "win": Lw.item(), "global": Lg.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 206) Resolution Growth ----------
                if algo == "res_growth":
                    t = _train_progress(self)
                    # downscale factor decays from 4→1 with cosine schedule
                    s = int(round(_cos_interp(4.0, 1.0, t)))
                    if s < 1: s = 1
                    if s > 1:
                        xs = _up_to(_down_to(x, (x.shape[-2]//s, x.shape[-1]//s)), (x.shape[-2], x.shape[-1]))
                    else:
                        xs = x
                    z, feats = self.encode(xs); r = self.decode(z, feats)
                    L = self.recon_loss(r, x)
                    logs = {"scale": s, "recon": L.item()}
                    return {"loss": L, "logs": logs, "recon": r}

                # ---------- 207) Blur Decay ----------
                if algo == "blur_decay":
                    t = _train_progress(self)
                    sigma = _cos_interp(2.0, 0.0, t)  # heavy→none
                    xb = _gauss_blur(x, k=7, sigma=max(0.0, sigma)) if sigma > 1e-6 else x
                    z, feats = self.encode(xb); r = self.decode(z, feats)
                    L = self.recon_loss(r, x)
                    logs = {"sigma": float(sigma), "recon": L.item()}
                    return {"loss": L, "logs": logs, "recon": r}

                # ---------- 209) Latent Centering (EMA) ----------
                if algo == "latent_centering":
                    z, f = self.encode(x); r = self.decode(z, f)
                    rec = self.recon_loss(r, x)
                    h = z.flatten(2).mean(-1) if z.dim()==4 else z  # (B,D)
                    if not hasattr(self, "_z_mean"):
                        self._z_mean = None
                    self._z_mean = _ema_update(self._z_mean, h.mean(0), m=0.99)
                    cen = (h.mean(0) - self._z_mean.detach()).pow(2).mean()  # keep batch mean near EMA 0
                    loss = rec + 0.01*cen
                    logs = {"recon": rec.item(), "center": cen.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 210) Gray-World Color Constancy ----------
                if algo == "color_constancy":
                    z, f = self.encode(x); r = self.decode(z, f).clamp(0,1)
                    rec = self.recon_loss(r, x)
                    # gray-world: channel means similar (on recon)
                    m, _ = _channel_mean_var(r)
                    cc = ((m - m.mean()).abs()).mean()
                    loss = rec + 0.01*cc
                    logs = {"recon": rec.item(), "grayworld": cc.item(), "means": m.detach().tolist()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 211) FFT Low-Freq Phase Preservation ----------
                if algo == "fft_phase_preserve":
                    z, f = self.encode(x); r = self.decode(z, f).clamp(0,1)
                    rec = self.recon_loss(r, x)
                    B,C,H,W = x.shape
                    mx, phx = _fft_split_mag_phase(x)
                    mr, phr = _fft_split_mag_phase(r)
                    mask = _lowfreq_mask(H, W, frac=0.2, device=x.device)
                    phaseL = ((torch.sin(phr - phx) * mask).pow(2)).mean()  # small angle dist ~ squared diff
                    # also keep low-freq magnitude close (light)
                    magL = (((mr - mx) * mask).pow(2)).mean()
                    loss = rec + 0.02*phaseL + 0.01*magL
                    logs = {"recon": rec.item(), "phaseL": phaseL.item(), "magL": magL.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 212) Saturation Guard ----------
                if algo == "saturation_guard":
                    z, f = self.encode(x); r = self.decode(z, f).clamp(0,1)
                    rec = self.recon_loss(r, x)
                    def sat_frac(t):
                        lo = (t < 0.01).float().mean()
                        hi = (t > 0.99).float().mean()
                        return lo + hi
                    s_in  = sat_frac(x)
                    s_out = sat_frac(r)
                    guard = F.relu(s_out - s_in).mean()   # penalize extra saturation only
                    loss = rec + 0.05*guard
                    logs = {"recon": rec.item(), "sat_excess": guard.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 213) Moment Match (μ, σ²) ----------
                if algo == "moment_match_12":
                    z, f = self.encode(x); r = self.decode(z, f).clamp(0,1)
                    rec = self.recon_loss(r, x)
                    mx, vx = _channel_mean_var(x); mr, vr = _channel_mean_var(r)
                    mL = (mr - mx.detach()).pow(2).mean()
                    vL = (vr - vx.detach()).abs().mean()
                    loss = rec + 0.05*mL + 0.02*vL
                    logs = {"recon": rec.item(), "meanL": mL.item(), "varL": vL.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 214) TV on Residual ----------
                if algo == "tv_residual":
                    z, f = self.encode(x); r = self.decode(z, f)
                    rec = self.recon_loss(r, x)
                    tv = _tv2d((r - x).clamp(-1,1))
                    loss = rec + 0.01*tv
                    logs = {"recon": rec.item(), "tv_res": float(tv)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 215) Edge-Preserve Ratio ----------
                if algo == "edge_preserve_ratio":
                    r = self.decode(*self.encode(x)).clamp(0,1)
                    gx = _sobel_edges(x).mean(dim=1, keepdim=True) + 1e-6
                    gr = _sobel_edges(r).mean(dim=1, keepdim=True) + 1e-6
                    ratio = (gr / gx).clamp(0, 3.0)
                    # want ratio ≈ 1 across pixels
                    er = (ratio - 1.0).abs().mean()
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.02*er
                    logs = {"recon": rec.item(), "edge_ratio_err": er.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 216) Tone-Curve Consistency (invertible S-curve) ----------
                if algo == "tonecurve_consistency":
                    # Learn a tiny parametric S-curve and its inverse; recon should be consistent across both domains
                    if not hasattr(self, "_tone_alpha"):
                        # α>0 sharpness; β in (0,1) mid-point (use logit param)
                        self._tone_alpha = nn.Parameter(torch.tensor(1.0))
                        self._tone_betaL = nn.Parameter(torch.tensor(0.0))  # beta = sigmoid(betaL)
                    alpha = F.softplus(self._tone_alpha) + 1e-3
                    beta  = torch.sigmoid(self._tone_betaL)
                    def _tone(y):
                        # y ∈ [0,1]; logistic S-curve centered at β with slope α
                        return (1 / (1 + torch.exp(-alpha*(y - beta)))).clamp(0,1)
                    def _inv_tone(y):
                        # approximate inverse via safe logit
                        y = y.clamp(1e-4, 1-1e-4)
                        return (torch.log(y/(1-y))/alpha + beta).clamp(0,1)
                    xt = _tone(x)
                    z1,f1 = self.encode(xt); r1 = self.decode(z1,f1)
                    # map back through inverse and compare to original
                    r_back = _inv_tone(r1)
                    cyc = self.recon_loss(r_back, x)
                    # small tether in tone domain too
                    teth = 0.3 * self.recon_loss(r1, xt)
                    loss = cyc + teth
                    logs = {"cycle": cyc.item(), "tether": teth.item(), "alpha": float(alpha), "beta": float(beta)}
                    return {"loss": loss, "logs": logs, "recon": r_back}

                # ---------- 208) Mix Curriculum (rotate family by phase) ----------
                if algo == "mix_curriculum":
                    t = _train_progress(self)
                    phase = int((t * 4) % 4)  # 0,1,2,3 cycles
                    if phase == 0:
                        # noise phase
                        sigma = _cos_interp(0.25, 0.05, t)
                        x_aug = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    elif phase == 1:
                        # occlusion phase
                        frac = 0.1 + 0.3*t
                        B,C,H,W = x.shape
                        m = _cutout_mask(B, H, W, frac).to(x.device).type_as(x)
                        x_aug = x*m
                    elif phase == 2:
                        # resolution phase
                        s = int(round(_cos_interp(3.0, 1.0, t))); s = max(1, s)
                        x_aug = _up_to(_down_to(x, (x.shape[-2]//s, x.shape[-1]//s)), (x.shape[-2], x.shape[-1]))
                    else:
                        # blur phase
                        sigma = _cos_interp(1.5, 0.1, t)
                        x_aug = _gauss_blur(x, k=5, sigma=sigma)
                    z, feats = self.encode(x_aug); r = self.decode(z, feats)
                    # canonical target is clean x
                    L = self.recon_loss(r, x)
                    logs = {"phase": phase, "recon": L.item()}
                    return {"loss": L, "logs": logs, "recon": r}

                # ---------- 184) Patch Link Prediction (adjacency) ----------
                if algo == "patch_link_prediction":
                    ps, stride = 16, 16
                    B,C,H,W = x.shape
                    P, Gh, Gw = _unfold_patches(x, ps, stride)      # (B,n,C,ps,ps)
                    h, _ = _patch_latents(self.encode, P)           # (B,n,D)
                    # build simple positive/negative pairs: neighbors = positive; far = negative
                    pairs, labels = [], []
                    for r in range(Gh):
                        for c in range(Gw):
                            i = r*Gw+c
                            nbrs = []
                            if r>0: nbrs.append((r-1)*Gw+c)
                            if c>0: nbrs.append(r*Gw+c-1)
                            if r< Gh-1: nbrs.append((r+1)*Gw+c)
                            if c< Gw-1: nbrs.append(r*Gw+c+1)
                            if nbrs:
                                j = random.choice(nbrs); pairs.append((i,j)); labels.append(1.0)
                                # sample a non-neighbor
                                k = random.randrange(Gh*Gw)
                                while k in nbrs or k==i: k = random.randrange(Gh*Gw)
                                pairs.append((i,k)); labels.append(0.0)
                    pairs = torch.tensor(pairs, device=x.device)
                    y = torch.tensor(labels, device=x.device).float()
                    i, j = pairs[:,0], pairs[:,1]
                    # use cosine sim as link score; BCE with targets
                    sim = (h.reshape(B*Gh*Gw, -1)[i] * h.reshape(B*Gh*Gw, -1)[j]).sum(dim=1)
                    sim = (sim / (1e-6 + h.reshape(B*Gh*Gw, -1)[i].norm(dim=1) * h.reshape(B*Gh*Gw, -1)[j].norm(dim=1))).clamp(-1,1)
                    score = (sim+1)/2
                    # small recon tether
                    rec = self.recon_loss(self.decode(*self.encode(x)), x)
                    link = F.binary_cross_entropy(score, y)
                    loss = rec + 0.1*link
                    logs = {"recon": rec.item(), "link": link.item(), "Gh": Gh, "Gw": Gw}
                    return {"loss": loss, "logs": logs, "recon": self.decode(*self.encode(x))}

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

                # ---------- 129) Energy-Contrast AE ----------
                if algo == "energy_contrast":
                    # energy E(x) = ||r(x) - x||^2 ; minimize on data, maximize on corrupted
                    z, f = self.encode(x); r = self.decode(z, f)
                    E_pos = ((r - x)**2).mean()

                    sigma = _rand_sigma()
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    zn, fn = self.encode(xn); rn = self.decode(zn, fn)
                    E_neg = ((rn - xn)**2).mean()

                    loss = E_pos + 0.25 * _hinge(0.05 - (E_neg - E_pos))  # push neg energy up
                    logs = {"E_pos": float(E_pos), "E_neg": float(E_neg), "sigma": sigma}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 130) Noise-Conditional DAE (multi-σ + cross-σ consistency) ----------
                if algo == "noise_cond_dae":
                    s1, s2 = _rand_sigma(), _rand_sigma()
                    x1 = (x + s1*torch.randn_like(x)).clamp(0,1)
                    x2 = (x + s2*torch.randn_like(x)).clamp(0,1)
                    z1,f1 = self.encode(x1); r1 = self.decode(z1,f1)
                    z2,f2 = self.encode(x2); r2 = self.decode(z2,f2)
                    Ld = 0.5*(self.recon_loss(r1, x) + self.recon_loss(r2, x))
                    # cross-σ score consistency
                    sθ1 = _dae_score(x1, r1, s1); sθ2 = _dae_score(x2, r2, s2)
                    Lc = F.mse_loss(sθ1, sθ2.detach())
                    loss = Ld + 0.05*Lc
                    logs = {"denoise": Ld.item(), "cross_sigma": Lc.item(), "s1": s1, "s2": s2}
                    return {"loss": loss, "logs": logs, "recon": r1}

                # ---------- 131) Sliced Score Matching (Hutchinson) ----------
                if algo == "sliced_score_match":
                    sigma = _rand_sigma()
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    xn = xn.detach().requires_grad_(True)
                    z,f = self.encode(xn); r = self.decode(z,f)
                    sθ = _dae_score(xn, r, sigma)
                    # sliced score matching: 0.5||s||^2 + div s
                    v = torch.randn_like(xn)  # or Rademacher
                    sm = 0.5 * (sθ**2).mean() + _divergence_sliced(sθ, xn, v)
                    # tiny recon tether (stabilize)
                    rec = 0.1 * self.recon_loss(r, x)
                    loss = sm + rec
                    logs = {"ssm": float(sm.detach()), "recon": float(rec.detach()), "sigma": sigma}
                    return {"loss": loss, "logs": logs, "recon": r.detach()}

                # ---------- 132) Langevin Consistency ----------
                if algo == "langevin_consistency":
                    sigma = _rand_sigma()
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    z,f = self.encode(xn); r = self.decode(z,f)
                    sθ = _dae_score(xn, r, sigma)
                    step = 0.5 * (sigma**2)
                    x_step = (xn + step * sθ).clamp(0,1)

                    # after one Langevin-like step, recon should move closer to clean
                    z2,f2 = self.encode(x_step); r2 = self.decode(z2,f2)
                    d0 = F.mse_loss(r,  x); d1 = F.mse_loss(r2, x)
                    improve = _hinge(d1 - d0)  # want d1 <= d0
                    loss = d1 + 0.1*improve
                    logs = {"d_before": d0.item(), "d_after": d1.item(), "sigma": sigma}
                    return {"loss": loss, "logs": logs, "recon": r2}

                # ---------- 133) Adversarial-Energy (hard negatives) ----------
                if algo == "adv_energy":
                    # PGD on input to *increase* energy; then train to lower energy on clean
                    eps = 4/255.0; step = 2/255.0; iters = 3
                    x_adv = x.clone().detach().requires_grad_(True)
                    for _ in range(iters):
                        z,f = self.encode(x_adv); r = self.decode(z,f)
                        E = ((r - x_adv)**2).mean()
                        (-E).backward()  # ascend energy
                        with torch.no_grad():
                            x_adv = (x_adv + step * x_adv.grad.sign())
                            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0,1).detach().requires_grad_(True)
                        x_adv.grad = None
                    # contrast: high energy on x_adv, low on x
                    zc,fc = self.encode(x); rc = self.decode(zc,fc)
                    Ec = ((rc - x)**2).mean()
                    za,fa = self.encode(x_adv); ra = self.decode(za,fa)
                    Ea = ((ra - x_adv)**2).mean()
                    loss = Ec + 0.25*_hinge(0.1 - (Ea - Ec))
                    logs = {"E_clean": float(Ec), "E_adv": float(Ea)}
                    return {"loss": loss, "logs": logs, "recon": rc}

                # ---------- 134) Multi-σ Path (σ1→σ2 monotone improvement) ----------
                if algo == "multi_sigma_path":
                    s1, s2 = sorted([_rand_sigma(), _rand_sigma()], reverse=True)  # s1 > s2
                    x1 = (x + s1*torch.randn_like(x)).clamp(0,1)
                    z1,f1 = self.encode(x1); r1 = self.decode(z1,f1)
                    # denoise again starting from x1 but assuming smaller σ2
                    sθ1 = _dae_score(x1, r1, s1)
                    x2 = (x1 + 0.5*(s1**2) * sθ1).clamp(0,1)
                    z2,f2 = self.encode(x2); r2 = self.decode(z2,f2)

                    Ld = self.recon_loss(r2, x)
                    # ensure r2 is closer than r1
                    imp = _hinge(F.mse_loss(r2, x) - F.mse_loss(r1, x))
                    loss = Ld + 0.1*imp
                    logs = {"denoise": Ld.item(), "improve": imp.item(), "s1": s1, "s2": s2}
                    return {"loss": loss, "logs": logs, "recon": r2}

                # ---------- 135) Residual-Denoise (predict ε) ----------
                if algo == "residual_denoise":
                    sigma = _rand_sigma()
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    z,f = self.encode(xn)
                    base = self.decode(z,f)       # same decoder; use it to predict residual ε̂
                    eps_hat = base - xn           # residual branch (no new head needed)
                    x_hat = (xn + eps_hat).clamp(0,1)
                    Lres = F.mse_loss(eps_hat, x - xn)  # residual supervision
                    Lrec = self.recon_loss(x_hat, x)
                    loss = Lres + 0.5*Lrec
                    logs = {"res": Lres.item(), "recon": Lrec.item(), "sigma": sigma}
                    return {"loss": loss, "logs": logs, "recon": x_hat}

                # ---------- 136) Score-Norm Balance ----------
                if algo == "score_norm_balance":
                    sigma = _rand_sigma()
                    xn = (x + sigma*torch.randn_like(x)).clamp(0,1)
                    z,f = self.encode(xn); r = self.decode(z,f)
                    sθ = _dae_score(xn, r, sigma)
                    # control score magnitude: target batch norm roughly ||(x - xn)|| / σ^2
                    target = ((x - xn).flatten(1).norm(dim=1) / (sigma**2 + 1e-8)).mean()
                    snorm = sθ.flatten(1).norm(dim=1).mean()
                    reg = (snorm - target.detach()).abs()
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.01*reg
                    logs = {"recon": rec.item(), "score_norm": float(snorm), "target": float(target)}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 121) HSIC Disentangle ----------
                if algo == "hsic_disentangle":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    h = _flat_lat(z)
                    D = h.shape[1]; D2 = D//2
                    A, Bp = h[:, :D2], h[:, D2:]
                    hsic = _hsic_lin(A, Bp)         # lower is better (independence)
                    loss = rec + 0.01 * hsic
                    logs = {"recon": rec.item(), "hsic": hsic.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 122) Total-Correlation Min ----------
                if algo == "total_corr_min":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    tc = _tc_offdiag(_flat_lat(z))
                    loss = rec + 0.01 * tc
                    logs = {"recon": rec.item(), "tc": tc.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 123) Jacobian-Ortho (rand-direction deltas) ----------
                if algo == "jacobian_ortho":
                    z, feats = self.encode(x)
                    r0 = self.decode(z, feats)
                    eps = 0.05
                    v1, v2 = _two_rand_dirs_like(z)
                    r1 = self.decode(z + eps*v1, feats)
                    r2 = self.decode(z + eps*v2, feats)
                    d1 = (r1 - r0) / eps
                    d2 = (r2 - r0) / eps
                    cos = _cos_sim(d1, d2).mean().abs()        # want near-orthogonal sensitivities
                    rec = self.recon_loss(r0, x)
                    loss = rec + 0.05 * cos
                    logs = {"recon": rec.item(), "jac_ortho": cos.item()}
                    return {"loss": loss, "logs": logs, "recon": r0}

                # ---------- 124) Style-Invariant Split (AdaIN cue) ----------
                if algo == "style_invariant_split":
                    fA, _ = self.encode(x)
                    # style-perturbed view via AdaIN with shuffled style
                    perm = torch.randperm(x.size(0), device=x.device)
                    fB, _ = self.encode(x[perm])
                    f_mix = _apply_AdaIN_feat(fA, fB)
                    # split latent: first half = "content"; enforce invariance across style swap
                    zA = _flat_lat(fA); zMix = _flat_lat(f_mix)
                    D = zA.shape[1]; D2 = D//2
                    inv = F.mse_loss(zA[:, :D2], zMix[:, :D2].detach())
                    # decode content features (use mixed as carrier to avoid trivial)
                    r = self.decode(f_mix, [f_mix])
                    rec = self.recon_loss(r, x)
                    loss = rec + 0.1 * inv
                    logs = {"recon": rec.item(), "content_inv": inv.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 125) Color/Shape Split ----------
                if algo == "color_shape_split":
                    # gray vs. color view; gray should use 'shape' half, color the 'color' half
                    gray = (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]).repeat(1,3,1,1)
                    zc, fc = self.encode(x); rg = self.decode(zc, fc)       # color path
                    zg, fg = self.encode(gray); rg2 = self.decode(zg, fg)    # gray path
                    h = _flat_lat(zc); D = h.shape[1]; D2 = D//2
                    # encourage gray to live in shape half (first D2): shrink color-half energy for gray
                    hg = _flat_lat(zg)
                    sup = (hg[:, D2:]**2).mean()
                    # recon both
                    rec = 0.5*(self.recon_loss(rg, x) + self.recon_loss(rg2, gray))
                    loss = rec + 1e-3 * sup
                    logs = {"recon": rec.item(), "gray_suppress": sup.item()}
                    return {"loss": loss, "logs": logs, "recon": rg}

                # ---------- 126) Latent-Dropout Recon ----------
                if algo == "latent_dropout_recon":
                    z, f = self.encode(x)
                    h = z
                    if h.dim()==4:
                        B,C,H,W = h.shape
                        keep = int(C * 0.75)
                        idx = torch.randperm(C, device=x.device)[:keep]
                        m = torch.zeros(C, device=x.device); m[idx]=1.0
                        m = m.view(1,C,1,1)
                        r_drop = self.decode(h*m, f)
                    else:
                        B,D = h.shape
                        keep = int(D * 0.75)
                        idx = torch.randperm(D, device=x.device)[:keep]
                        m = torch.zeros(D, device=x.device); m[idx]=1.0
                        r_drop = self.decode(h*m, f)
                    r_full = self.decode(h, f)
                    rec = self.recon_loss(r_full, x)
                    cons = F.mse_loss(r_drop, r_full.detach())
                    loss = rec + 0.1 * cons
                    logs = {"recon": rec.item(), "latent_cons": cons.item()}
                    return {"loss": loss, "logs": logs, "recon": r_full}

                # ---------- 127) kNN Geodesic / Diffusion Consistency ----------
                if algo == "knn_geodesic":
                    f, z = self.encode(x)
                    r = self.decode(z, f)
                    rec = self.recon_loss(r, x)
                    zx = f.mean(dim=[2,3])              # input-proxy embedding
                    zl = _flat_lat(z)
                    Px = _diffusion_affinity(zx, k=5)   # row-stochastic
                    Pz = _diffusion_affinity(zl, k=5)
                    # align one-step diffusion (or 2-step for stability)
                    Df = F.mse_loss(Pz, Px.detach())
                    loss = rec + 0.1 * Df
                    logs = {"recon": rec.item(), "diff_align": Df.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

                # ---------- 128) Subspace Independence (decorrelate groups + whiten) ----------
                if algo == "subspace_independence":
                    z, feats = self.encode(x); r = self.decode(z, feats)
                    rec = self.recon_loss(r, x)
                    h = _flat_lat(z); D = h.shape[1]; D2 = D//2
                    a, b = h[:, :D2], h[:, D2:]
                    cross = _hsic_lin(a, b)
                    white = 0.5*(_whiten_penalty(a) + _whiten_penalty(b))
                    loss = rec + 0.01*cross + 0.001*white
                    logs = {"recon": rec.item(), "cross": cross.item(), "white": white.item()}
                    return {"loss": loss, "logs": logs, "recon": r}

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

def _flat_lat(z):
    # pool spatial if needed → (B,D)
    return z.flatten(2).mean(-1) if z.dim()==4 else z

def _cov(z):  # z: (B,D) -> (D,D)
    z0 = z - z.mean(0, keepdim=True)
    return (z0.T @ z0) / (z0.shape[0]-1 + 1e-6)

def _tc_offdiag(z):
    C = _cov(z)
    off = C - torch.diag(torch.diag(C))
    return off.abs().mean()

def _whiten_penalty(z):
    C = _cov(z)
    I = torch.eye(C.shape[0], device=z.device, dtype=z.dtype)
    return ((C - I)**2).mean()

def _hsic_lin(x, y):
    # linear-kernel HSIC ~ ||Cov(x,y)||_F^2
    x0 = x - x.mean(0, keepdim=True); y0 = y - y.mean(0, keepdim=True)
    Cxy = (x0.T @ y0) / (x.shape[0]-1 + 1e-6)
    return (Cxy**2).mean()

def _two_rand_dirs_like(z):
    # produce two orthonormal random directions with same shape as z (per-sample)
    if z.dim()==4:
        B,C,H,W = z.shape
        v1 = torch.randn_like(z); v2 = torch.randn_like(z)
        def norm(v): return v / (v.flatten(1).norm(dim=1, keepdim=True).view(B,1,1,1)+1e-6)
        v1 = norm(v1); v2 = norm(v2 - (v1*v2).sum(dim=(1,2,3),keepdim=True)*v1)
        return v1, norm(v2)
    else:
        B,D = z.shape
        v1 = torch.randn_like(z); v2 = torch.randn_like(z)
        def norm(v): return v / (v.norm(dim=1, keepdim=True)+1e-6)
        v1 = norm(v1); v2 = norm(v2 - (v1*v2).sum(dim=1,keepdim=True)*v1)
        return v1, norm(v2)

def _apply_AdaIN_feat(fA, fB, eps=1e-5):
    mA, vA = fA.mean([2,3], keepdim=True), fA.var([2,3], keepdim=True)
    mB, vB = fB.mean([2,3], keepdim=True), fB.var([2,3], keepdim=True)
    fn = (fA - mA) / (vA+eps).sqrt()
    return fn * (vB+eps).sqrt() + mB

def _pdist(U):
    G = U @ U.t(); n2 = U.pow(2).sum(1, keepdim=True); return (n2 + n2.t() - 2*G).clamp_min(0)

def _diffusion_affinity(U, k=5):
    # build kNN heat kernel W; return normalized diffusion matrix P
    D2 = _pdist(U)
    # sigma via median of nonzero dists
    m = torch.median(D2[D2>0]).clamp_min(1e-6)
    W = torch.exp(-D2/(2*m))
    # keep only kNN per row (sparsify softly)
    B = W.shape[0]
    topk = torch.topk(W, k=min(k, B), dim=1).values[:, -1:].clamp_min(1e-9)
    W = W * (W >= topk)  # soft mask
    d = W.sum(1, keepdim=True) + 1e-6
    P = W / d
    return P  # row-stochastic diffusion

def _cos_sim(a, b):  # both flattened
    a = a.flatten(1); b = b.flatten(1)
    return (a*b).sum(1) / (a.norm(dim=1)+1e-6) / (b.norm(dim=1)+1e-6)

def _dae_score(x_noisy, recon, sigma):
    # DAE identity: r*(x) ≈ x + σ^2 ∇ log p(x)  → score ≈ (recon - x_noisy) / σ^2
    return (recon - x_noisy) / (sigma**2 + 1e-8)

def _rand_sigma(lo=0.05, hi=0.25):
    return float(random.uniform(lo, hi))

def _hinge(a):  # ReLU
    return F.relu(a)

def _divergence_sliced(s, x, v):
    # s: score(x) same shape as x ; v: Rademacher/normal noise same shape as x
    # Hutchinson trick: E_v[ v^T (∂s/∂x) v ] = tr(J_s) = div s
    g = (s * v).sum()
    (dv,) = torch.autograd.grad(g, x, retain_graph=True, create_graph=True)
    return (dv * v).sum() / v.numel()

def _gauss_nll(x, mu, logvar):
    # per-pixel Gaussian NLL: 0.5*(log(2π) + log σ^2 + (x-μ)^2 / σ^2)
    return 0.5*(math.log(2*math.pi) + logvar + ((x-mu)**2) / (logvar.exp() + 1e-8))

def _soft_bins(v, edges):
    # v: (B,...) in R+, edges: 1D tensor sorted length M+1
    # returns (B,...,M) soft assignments using triangular kernels
    M = edges.numel() - 1
    ctr = 0.5*(edges[:-1] + edges[1:])
    w = torch.clamp(1.0 - (v.unsqueeze(-1) - ctr)**2 / (0.25*(edges[1:]-edges[:-1])**2 + 1e-8), min=0.0)
    w = w / (w.sum(dim=-1, keepdim=True)+1e-8)
    return w  # (..., M)

def _pinball(y, q, tau):
    # y, q >=0 ; pinball loss for |residual| quantile
    d = y - q
    return torch.maximum(tau*d, (tau-1)*d)

def _worstofk(x, make_view, K=4):
    cand = []
    for _ in range(K):
        cand.append(make_view(x))
    return cand  # list length K

# --- DCT basis ---
def _dct_mat(n, device, dtype):
    k = torch.arange(n, device=device, dtype=dtype).view(-1,1)
    i = torch.arange(n, device=device, dtype=dtype).view(1,-1)
    M = torch.cos(math.pi*(i+0.5)*k/n)
    M[0,:] = M[0,:] / math.sqrt(2.0)
    return M * math.sqrt(2.0/n)

def _block_dct2(x, bs=8):
    # x: (B,C,H,W) in [0,1]
    B,C,H,W = x.shape
    assert H%bs==0 and W%bs==0, "H,W must be multiples of block size"
    M = _dct_mat(bs, x.device, x.dtype)             # (bs,bs)
    xt = x.view(B*C, 1, H, W)
    # unfold blocks
    u = F.unfold(xt, bs, stride=bs)                 # (BC*1, bs*bs, nblk)
    u = u.transpose(1,2).contiguous().view(-1,1,bs,bs)
    # 2D DCT: M*u*M^T
    d = torch.einsum('ab,nbcd->nacd', M, u.squeeze(1))
    d = torch.einsum('nbcd,ab->nbca', d, M.t()).unsqueeze(1)  # (N,1,bs,bs)
    d = d.view(B*C, -1, bs*bs).transpose(1,2)      # (BC, bs*bs, nblk)
    return d, (B,C,H,W,bs,M)

def _block_idct2(D, ctx):
    B,C,H,W,bs,M = ctx
    D = D.transpose(1,2).contiguous().view(-1,1,bs,bs)   # (BC*nblk,1,bs,bs)
    u = torch.einsum('ab,nbcd->nacd', M.t(), D.squeeze(1))
    u = torch.einsum('nbcd,ab->nbca', u, M).unsqueeze(1) # (N,1,bs,bs)
    u = u.view(B*C, bs*bs, -1).contiguous()
    x = F.fold(u, output_size=(H,W), kernel_size=bs, stride=bs)
    return x.view(B,C,H,W)

def _round_ste(z):
    # straight-through rounding
    return (z - z.detach()) + torch.round(z.detach())

def _alias_penalty(img):
    # penalize checkerboard/aliasing via down-up mismatch (odd/even grids)
    d1 = F.avg_pool2d(img, 2, 2)
    u1 = F.interpolate(d1, scale_factor=2, mode="nearest")
    d2 = F.avg_pool2d(img[:,:,1:,1:], 2, 2)  # shifted grid
    u2 = F.interpolate(d2, scale_factor=2, mode="nearest")
    return F.l1_loss(u1, u2)

def _entropy_proxy_lat(z):
    # Laplace-like proxy: E|z| encourages compressibility
    h = z.flatten(2).mean(-1) if z.dim()==4 else z
    return h.abs().mean()

def _target_match(val, target):
    return (val - target).abs()

def _gamma_corr(x, gamma):
    # x in [0,1]
    return x.clamp(0,1).pow(gamma)

def _photo_affine(x, alpha=None, beta=None):
    # y = alpha * x + beta, clamp to [0,1]
    if alpha is None: alpha = random.uniform(0.7, 1.4)
    if beta  is None: beta  = random.uniform(-0.2, 0.2)
    return (alpha*x + beta).clamp(0,1)

def _rand_depthwise_conv3(x):
    # Random positive 3×3 depthwise kernel (anti/blur-ish depending on weights)
    B,C,H,W = x.shape
    k = torch.rand(C,1,3,3, device=x.device, dtype=x.dtype)
    k = k / (k.sum(dim=(2,3), keepdim=True) + 1e-6)
    return F.conv2d(x, k, padding=1, groups=C)

def _synth_haze(x, beta=None, A=None):
    # Simple atmospheric scattering: I = J * t + A*(1-t), t=exp(-β d)
    # Use luminance as depth surrogate ∈ [0,1]
    if beta is None: beta = random.uniform(0.6, 1.8)
    Y = (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]).clamp(0,1)
    d = (1.0 - Y)                    # closer (bright) → smaller depth
    t = torch.exp(-beta * d)         # transmission
    if A is None:
        A = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)*0.3 + 0.7  # [0.7,1.0]
    I = x * t + A * (1.0 - t)
    return I.clamp(0,1), t, A

def _fg_mask_coarse(x):
    # reuse edges + symmetry (from earlier parts) to get a crude FG mask
    e = _sobel_edges(x)
    s = _symmetry_map(x) if ' _symmetry_map' in globals() else 0.0*e
    m = (e + 0.3*s)
    m = m / (m.amax(dim=(-2,-1), keepdim=True) + 1e-6)
    return (m > 0.25).float()

def _soft_equalize(x, bins=64, sigma=0.02):
    # differentiable per-channel histogram equalization using soft hist/CDF
    # returns x_eq in [0,1]
    B,C,H,W = x.shape
    # soft histogram
    hist = _soft_hist(x.clamp(0,1), bins=bins, sigma=sigma)            # (B,C,bins)
    cdf  = torch.cumsum(hist, dim=-1)
    # bin centers
    ctr = torch.linspace(0,1,bins, device=x.device, dtype=x.dtype).view(1,1,bins)
    # map each pixel by soft assignment to CDF( value )
    xv = x.unsqueeze(2)                                                # (B,C,1,H,W)
    # soft bin weights (same as _soft_hist internal, but re-compute lightweight)
    d = (xv - ctr.view(1,1,bins,1,1))/sigma
    w = torch.exp(-0.5*d*d); w = w / (w.sum(dim=2, keepdim=True) + 1e-6)  # (B,C,bins,H,W)
    cdf_img = (w * cdf.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)             # (B,C,H,W)
    return cdf_img.clamp(0,1)

def _channel_stats_shift(x):
    # Simulate BN stats shift: normalize per-channel, then re-standardize with random μ,σ
    B,C,H,W = x.shape
    mu  = x.mean(dim=[2,3], keepdim=True)
    var = x.var(dim=[2,3], keepdim=True)
    xn = (x - mu) / (var + 1e-6).sqrt()
    mu2  = torch.empty_like(mu).uniform_(-0.3, 0.3)
    std2 = torch.empty_like(var).uniform_(0.7, 1.5).sqrt()
    y = (xn * std2 + mu2).clamp(-2, 2)  # allow small out-of-[0,1] before clamp
    return y.clamp(0,1)

def _meshgrid(B, H, W, device, dtype):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    g = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (H,W,3)
    return g.unsqueeze(0).repeat(B,1,1,1)  # (B,H,W,3)

def _rand_homography(B, mag=0.15, tilt=0.10, persp=0.02, device="cpu", dtype=torch.float32):
    # small rotation/scale/translation + mild tilt + perspective
    Hs = []
    for _ in range(B):
        ang = (torch.rand(1)*2-1)*mag*math.pi  # ~[-0.15π,0.15π]
        s   = 1.0 + (torch.rand(1)*2-1)*0.1
        tx  = (torch.rand(1)*2-1)*0.2
        ty  = (torch.rand(1)*2-1)*0.2
        shx = (torch.rand(1)*2-1)*tilt
        shy = (torch.rand(1)*2-1)*tilt
        p1  = (torch.rand(1)*2-1)*persp
        p2  = (torch.rand(1)*2-1)*persp
        ca, sa = torch.cos(ang), torch.sin(ang)
        R = torch.tensor([[ca, -sa, 0.0],
                          [sa,  ca, 0.0],
                          [0.0, 0.0, 1.0]], dtype=dtype, device=device)
        S = torch.tensor([[s, 0, 0],[0, s, 0],[0,0,1]], dtype=dtype, device=device)
        Sh= torch.tensor([[1, shx, 0],[shy, 1, 0],[0,0,1]], dtype=dtype, device=device)
        T = torch.tensor([[1,0,tx],[0,1,ty],[0,0,1]], dtype=dtype, device=device)
        P = torch.tensor([[1,0,0],[0,1,0],[p1,p2,1]], dtype=dtype, device=device)
        Hm = (T @ Sh @ R @ S) @ P
        Hs.append(Hm)
    return torch.stack(Hs, dim=0)  # (B,3,3)

def _warp_homography(x, H):
    # x: (B,C,H,W), H: (B,3,3) mapping NDC→NDC
    B,C,Hh,Wh = x.shape
    g = _meshgrid(B, Hh, Wh, x.device, x.dtype)   # (B,H,W,3)
    Hg = torch.einsum("bij,bhwj->bhwi", H, g)     # (B,H,W,3)
    xn = Hg[...,:2] / (Hg[...,2:3].clamp_min(1e-6))
    # grid_sample expects (y,x) order; we have (x,y) in xn[...,0],xn[...,1]
    grid = torch.stack([xn[...,0], xn[...,1]], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", align_corners=True)

def _rand_affine(x):
    B,_,H,W = x.shape
    ang = (torch.rand(B, device=x.device)-0.5)*0.35*math.pi
    shx = (torch.rand(B, device=x.device)-0.5)*0.25
    shy = (torch.rand(B, device=x.device)-0.5)*0.25
    sc  = 1.0 + (torch.rand(B, device=x.device)-0.5)*0.2
    tx  = (torch.rand(B, device=x.device)-0.5)*0.3
    ty  = (torch.rand(B, device=x.device)-0.5)*0.3
    M = []
    for i in range(B):
        ca, sa = torch.cos(ang[i]), torch.sin(ang[i])
        A = torch.tensor([[ca, -sa],[sa, ca]], device=x.device, dtype=x.dtype) * sc[i]
        A = A @ torch.tensor([[1, shx[i]],[shy[i], 1]], device=x.device, dtype=x.dtype)
        t = torch.tensor([tx[i], ty[i]], device=x.device, dtype=x.dtype)
        M.append(torch.cat([A, t.view(2,1)], dim=1))
    M = torch.stack(M, dim=0)  # (B,2,3)
    grid = F.affine_grid(M, size=x.size(), align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", align_corners=True)

def _radial_distort(x, k1=None, k2=None):
    # simple barrel/pincushion via r' = r*(1 + k1 r^2 + k2 r^4)
    B,C,H,W = x.shape
    if k1 is None: k1 = (torch.rand(B, device=x.device)-0.5)*0.4  # [-0.2,0.2]
    if k2 is None: k2 = (torch.rand(B, device=x.device)-0.5)*0.2
    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=x.device),
        torch.linspace(-1,1,W,device=x.device),
        indexing="ij"
    )
    rr2 = xx**2 + yy**2
    k1v = k1.view(B,1,1); k2v = k2.view(B,1,1)
    s = 1 + k1v*rr2 + k2v*rr2*rr2
    xg = xx*s; yg = yy*s
    grid = torch.stack([xg, yg], dim=-1).unsqueeze(0).repeat(B,1,1,1)
    return F.grid_sample(x, grid, mode="bilinear", align_corners=True)

def _grad_orientation_hist(x, bins=36):
    # orientation histogram from Sobel (per-image, grayscale); returns (B,bins) normalized
    g = _sobel_edges(x)                    # (B,1,H,W) magnitude
    # quick orientations using finite diffs (reuse Sobel gx/gy approximations)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = kx.transpose(-1,-2)
    gx = F.conv2d(x.mean(1, keepdim=True), kx, padding=1)
    gy = F.conv2d(x.mean(1, keepdim=True), ky, padding=1)
    ang = torch.atan2(gy, gx)  # [-π, π]
    ang = (ang + math.pi) / (2*math.pi)  # [0,1)
    # soft binning by triangular kernels around centers
    B,_,H,W = ang.shape
    centers = torch.linspace(0,1, bins+1, device=x.device)[:-1] + 0.5/bins
    angv = ang.view(B,1,H,W)
    dif = torch.abs(angv - centers.view(1,bins,1,1))
    w = (1 - dif*bins).clamp_min(0.0) * g  # weight by magnitude
    h = w.view(B, bins, -1).sum(-1)
    return (h / (h.sum(-1, keepdim=True) + 1e-6))  # (B,bins)

def _corner_kp_map(x, k=5):
    # pseudo keypoints via Harris-like cornerness from gradients; returns (B,1,H,W)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = kx.transpose(-1,-2)
    g = x.mean(1, keepdim=True)
    gx = F.conv2d(g, kx, padding=1); gy = F.conv2d(g, ky, padding=1)
    R = (gx*gx * gy*gy - (gx*gy)**2) - 0.04*(gx*gx + gy*gy)**2
    R = torch.relu(R)
    # non-max suppression via max-pool
    Rmax = F.max_pool2d(R, k, stride=1, padding=k//2)
    M = (R == Rmax).float() * (R > R.mean(dim=[2,3], keepdim=True)).float()
    return M  # sparse mask of keypoints

def _simple_aug(x):
    k = random.choice([0,1,2,3])
    y = x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)
    if random.random() < 0.5: y = torch.flip(y, dims=[-1])
    return torch.rot90(y.clamp(0,1), k, dims=[-2, -1])

def _pool_lat(z):
    return z.flatten(2).mean(-1) if z.dim()==4 else z  # (B,D)

def _offdiag(M):
    return M - torch.diag(torch.diag(M))

def _corr_matrix(A, B=None, eps=1e-6):
    # A,B: (B,D) zero-mean -> (D,D) correlation
    if B is None: B = A
    A0 = A - A.mean(0, keepdim=True); B0 = B - B.mean(0, keepdim=True)
    As = A0 / (A0.std(0, keepdim=True) + eps)
    Bs = B0 / (B0.std(0, keepdim=True) + eps)
    N = A.shape[0]
    return (As.T @ Bs) / (N + eps)

def _channel_stats(x):
    # x: (B,C,H,W) -> per-channel mean/std over spatial & batch
    B,C,H,W = x.shape
    m = x.mean(dim=[0,2,3])                # (C,)
    s = x.std(dim=[0,2,3]) + 1e-6          # (C,)
    return m, s

def _soft_sparsity(z, target=0.2):
    # encourage fraction of (approx) zeros ~ target, via L1 & mean-activation control
    h = _pool_lat(z)                        # (B,D)
    l1 = h.abs().mean()
    mean_act = h.abs().gt(1e-3).float().mean()
    pen = (mean_act - target)**2
    return l1, pen, mean_act

def _unfold_patches(x, ps=16, stride=None):
    # x: (B,C,H,W) -> (B, n, C, ps, ps), grid (Gh,Gw)
    if stride is None: stride = ps
    B,C,H,W = x.shape
    Gh, Gw = (H - ps)//stride + 1, (W - ps)//stride + 1
    P = F.unfold(x, ps, stride=stride)                      # (B, C*ps*ps, n)
    n = P.shape[-1]
    P = P.transpose(1,2).contiguous().view(B, n, C, ps, ps) # (B,n,C,ps,ps)
    return P, Gh, Gw

def _fold_patches(P, H, W, ps=16, stride=None):
    # P: (B,n,C,ps,ps)
    if stride is None: stride = ps
    B,n,C,ps,ps = P.shape
    P2 = P.view(B, n, C*ps*ps).transpose(1,2).contiguous()  # (B,C*ps*ps,n)
    out = F.fold(P2, output_size=(H,W), kernel_size=ps, stride=stride)
    # recompose normalizer for overlap
    ones = torch.ones(B,1,H,W, device=P.device, dtype=P.dtype)
    norm = F.fold(F.unfold(ones, ps, stride=stride), (H,W), ps, stride=stride)
    return out / (norm + 1e-6)

def _patch_latents(encoder, patches):
    # patches: (B,n,C,ps,ps) -> (B,n,D) pooled latents
    B,n,C,ps,ps = patches.shape
    P = patches.view(B*n, C, ps, ps)
    z, _ = encoder(P)
    h = F.normalize(z.flatten(2).mean(-1), dim=1) if z.dim()==4 else F.normalize(z, dim=1)
    return h.view(B, n, -1), z  # (B,n,D), raw z (B*n, ...)

def _hutch_trace_jacobian_norm(f_of_x, x):
    """
    Approximates E_v ||J||_F^2 via Hutchinson: E[(∂f/∂x v)^2] with Rademacher v.
    x must require_grad.
    Returns scalar estimate.
    """
    v = torch.randint_like(x, low=0, high=2).float()*2 - 1  # Rademacher
    y = (f_of_x * v).sum()
    (gy,) = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)
    return (gy**2).mean()

def _pairwise_dists(A):
    # A: (B,D) -> (B,B) squared Euclidean
    g = A @ A.t()
    n2 = A.pow(2).sum(1, keepdim=True)
    return (n2 + n2.t() - 2*g).clamp_min(0)

def _downsample_for_metric(x, size=16):
    return F.interpolate(x, (size,size), mode="area").flatten(1)

def _mixup(x):
    B = x.size(0)
    lam = torch.distributions.Beta(0.4, 0.4).sample().to(x.device)
    idx = torch.randperm(B, device=x.device)
    xm = lam*x + (1-lam)*x[idx]
    return xm.clamp(0,1), idx, float(lam)

def _cutmix(x):
    B,C,H,W = x.shape
    idx = torch.randperm(B, device=x.device)
    # box
    rx, ry = random.uniform(0.3,0.7), random.uniform(0.3,0.7)
    rw, rh = int(W*rx), int(H*ry)
    x0 = random.randint(0, W-rw); y0 = random.randint(0, H-rh)
    m = torch.ones(B,1,H,W, device=x.device, dtype=x.dtype)
    m[..., y0:y0+rh, x0:x0+rw] = 0.0
    xm = x*m + x[idx]*(1-m)
    return xm.clamp(0,1), idx, m

def _rand_fmask(shape, decay=3):
    # FMix-style mask in Fourier domain
    B,C,H,W = shape
    freqs = torch.randn(B,1,H,W, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
    Freq = torch.fft.rfftn(freqs, dim=(2,3))
    ky = torch.arange(Freq.shape[2], device=Freq.device).view(1,1,-1,1)
    kx = torch.arange(Freq.shape[3], device=Freq.device).view(1,1,1,-1)
    rad = (ky**2 + kx**2).float().pow(-decay/2).clamp_max(1e3)
    Freq = Freq * rad
    mask = torch.fft.irfftn(Freq, s=(H,W)).real
    mask = (mask - mask.amin(dim=(2,3), keepdim=True)) / (mask.amax(dim=(2,3), keepdim=True)-mask.amin(dim=(2,3), keepdim=True) + 1e-6)
    thr = torch.rand(B,1,1,1, device=mask.device)
    m = (mask > thr).float()
    return m  # (B,1,H,W) in {0,1}

def _gridmask(x, ratio=(0.3,0.6), rotate=True):
    B,C,H,W = x.shape
    d = random.uniform(int(min(H,W)*ratio[0]), int(min(H,W)*ratio[1]))
    l = int(d*random.uniform(0.45, 0.65))
    grid = torch.ones(B,1,H,W, device=x.device, dtype=x.dtype)
    for i in range(0, H, d):
        grid[..., i:i+l, :] = 0.0
    for j in range(0, W, d):
        grid[..., :, j:j+l] = 0.0
    if rotate:
        k = random.choice([0,1,2,3])
        grid = torch.rot90(grid, k, dims=[-2,-1])
    return x*grid, grid

def _mosaic2x2(x):
    B,C,H,W = x.shape
    idx1 = torch.randperm(B, device=x.device)
    idx2 = torch.randperm(B, device=x.device)
    idx3 = torch.randperm(B, device=x.device)
    h2, w2 = H//2, W//2
    top = torch.cat([F.interpolate(x, (h2,w2)), F.interpolate(x[idx1], (h2,w2))], dim=-1)
    bot = torch.cat([F.interpolate(x[idx2], (h2,w2)), F.interpolate(x[idx3], (h2,w2))], dim=-1)
    xm = torch.cat([top, bot], dim=-2)
    return xm

def _mixstyle_feats(fA, fB, p=0.5):
    # AdaIN-like blend of statistics; random blend p
    mA, vA = fA.mean([2,3], keepdim=True), fA.var([2,3], keepdim=True)
    mB, vB = fB.mean([2,3], keepdim=True), fB.var([2,3], keepdim=True)
    p = torch.tensor(p, device=fA.device, dtype=fA.dtype)
    m = (1-p)*mA + p*mB
    s = (1-p)*vA + p*vB
    fn = (fA - mA) / (vA+1e-5).sqrt()
    return fn * (s+1e-5).sqrt() + m

def _patchmix_jigsaw(x, ps=16):
    B,C,H,W = x.shape
    Gh, Gw = H//ps, W//ps
    # carve grid
    P = F.unfold(x, ps, stride=ps)  # (B, C*ps*ps, n)
    n = P.shape[-1]
    idx = torch.stack([torch.randperm(n, device=x.device) for _ in range(B)], dim=0)
    Psh = P.clone()
    for b in range(B):
        Psh[b] = P[b,:,idx[b]]
    xmix = F.fold(Psh, (H,W), ps, stride=ps)
    return xmix

def _train_progress(self):
    # normalized progress in [0,1]; tries cfg.max_steps/epochs if present, else soft fallback
    if not hasattr(self, "_y_steps"): self._y_steps = 0
    self._y_steps += 1
    # prefer explicit max steps if you have it in cfg; else assume ~100k iters
    T = getattr(self.cfg, "max_steps", None)
    if T is None:
        # try epochs*len_loader heuristic if user set it; else fallback
        T = getattr(self.cfg, "epochs", 100) * getattr(self.cfg, "approx_steps_per_epoch", 1000)
    return min(1.0, float(self._y_steps) / max(1, int(T)))

def _cos_interp(a, b, t):
    # cosine anneal from a→b over t∈[0,1]
    ct = 0.5*(1+math.cos(math.pi*t))
    return b + (a - b) * ct  # t=0 -> a, t=1 -> b

def _cutout_mask(B, H, W, frac):
    # square cutout occupying `frac` of area (clipped)
    frac = float(max(0.01, min(0.9, frac)))
    side = int((frac**0.5) * min(H, W))
    y0 = torch.randint(0, max(1, H-side+1), (B,), device='cuda' if torch.cuda.is_available() else 'cpu')
    x0 = torch.randint(0, max(1, W-side+1), (B,), device=y0.device)
    m = torch.ones(B,1,H,W, device=y0.device)
    for b in range(B):
        m[b,:, y0[b]:y0[b]+side, x0[b]:x0[b]+side] = 0.0
    return m

def _down_to(x, target_hw):
    return F.interpolate(x, size=target_hw, mode="area")

def _up_to(x, target_hw):
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

def _ema_update(buf, val, m=0.99):
    if buf is None:
        return val.detach()
    return m*buf + (1-m)*val.detach()

def _channel_mean_var(img):
    # img: (B,C,H,W) -> (C,) means, (C,) vars (batch+spatial)
    m = img.mean(dim=[0,2,3])
    v = img.var(dim=[0,2,3])
    return m, v

def _fft_split_mag_phase(x):
    # x: (B,C,H,W) in [0,1]
    X = torch.fft.rfftn(x, dim=(2,3))
    mag = torch.abs(X)
    ph  = torch.angle(X)
    return mag, ph

def _lowfreq_mask(h, w, frac=0.15, device="cpu"):
    kh = int(max(1, round(h*frac)))
    kw = int(max(1, round((w//2+1)*frac)))
    mask = torch.zeros(1,1,h,(w//2)+1, device=device)
    mask[..., :kh, :kw] = 1.0
    return mask

def _tv2d(x):
    dx = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dy = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dx + dy

# ----------------------------
# Quick self-test (optional)
# ----------------------------
# ----------------------------
# ADP INTEGRATION (Added during refactor)
# ----------------------------

import copy

@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    algo: str = "plain"  # The SSL algo to train with
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 8  # For base_channels expansion
    max_width: int = 1024  # max base_channels
    max_depth: int = 12
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 30
    batch_size: int = 32

def _overlap_copy_(dst, src):
    # Copy overlapping sub-region from src to dst (in-place)
    dims = [min(a, b) for a, b in zip(dst.shape, src.shape)]
    if not dims: return
    slices = tuple(slice(0, d) for d in dims)
    dst[slices].copy_(src[slices])

def _struct_copy(dst_model, src_model):
    # Copy weights from src to dst by name, handling shape mismatches via overlap
    src_params = dict(src_model.named_parameters())
    for name, dst_p in dst_model.named_parameters():
        if name in src_params:
            _overlap_copy_(dst_p.data, src_params[name].data)
    src_bufs = dict(src_model.named_buffers())
    for name, dst_b in dst_model.named_buffers():
        if name in src_bufs:
            _overlap_copy_(dst_b.data, src_bufs[name].data)

def snapshot_arch_and_state(model, state_dict=None):
    # We snapshot the CONFIG and the state.
    # If state_dict is provided (e.g. from best_state), use it.
    # Otherwise use model.state_dict().
    sd = state_dict if state_dict is not None else model.state_dict()
    # We need to deepcopy the config to preserve architecture definition
    return {
        "cfg": copy.deepcopy(model.cfg),
        "state": copy.deepcopy(sd)
    }

def restore_arch_and_state(model, snap, device):
    # Reconstruct model from snap config
    new_model = SelfSupervisedAE(snap["cfg"]).to(device)
    new_model.load_state_dict(snap["state"])
    return new_model

def expand_width(model, ex_k, max_width):
    # Expand base_channels
    if model.cfg.base_channels + ex_k > max_width:
        return None
    new_cfg = copy.deepcopy(model.cfg)
    new_cfg.base_channels += ex_k
    
    # Check max neurons constraint roughly? (Optional, skip for now or use dummy calc)
    # Rebuild
    device = next(model.parameters()).device
    new_model = SelfSupervisedAE(new_cfg).to(device)
    _struct_copy(new_model, model)
    return new_model

def expand_depth(model, max_depth):
    # Expand depth
    if model.cfg.depth + 1 > max_depth:
        return None
    new_cfg = copy.deepcopy(model.cfg)
    new_cfg.depth += 1
    
    device = next(model.parameters()).device
    new_model = SelfSupervisedAE(new_cfg).to(device)
    _struct_copy(new_model, model)
    return new_model

def total_neurons(model):
    # Rough estimate: count parameters
    return sum(p.numel() for p in model.parameters())

def train_with_early_stopping(model, data, acfg: ADPConfig, device) -> Tuple[float, Dict[str, Any]]:
    # Unpack data
    train_loader, val_loader = data
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(acfg.max_epochs):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
            # Some algos might need cleaner checks, but assuming simple x input for now
            # If batch is (img, target), we only use img for SSL typically
            
            opt.zero_grad(set_to_none=True)
            out = model.forward_train(x, algo=acfg.algo)
            loss = out["loss"]
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
                out = model.forward_train(x, algo=acfg.algo)
                val_loss += out["loss"].item()
                steps += 1
        val_loss /= max(1, steps)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if es_counter >= acfg.patience:
            break

    return best_val, best_state

def adp_search(model, data, acfg: ADPConfig, device):
    # Initial Baseline
    best_val, best_state = train_with_early_stopping(model, data, acfg, device)
    model.load_state_dict(best_state)
    
    global_best_val = best_val
    global_best_snap = snapshot_arch_and_state(model, best_state)

    # Helpers for the loops
    def can_widen(m):
        return (m.cfg.base_channels + acfg.ex_k <= acfg.max_width) and (total_neurons(m) < acfg.max_neurons)
    
    def can_deepen(m):
        return (m.cfg.depth + 1 <= acfg.max_depth) and (total_neurons(m) < acfg.max_neurons) # Approximate check

    def optimize_width_at_fixed_depth(curr_model):
        local_val, local_state = train_with_early_stopping(curr_model, data, acfg, device)
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        width_fails = 0
        while width_fails < acfg.trials_width:
            if not can_widen(curr_model):
                break
            
            next_model = expand_width(curr_model, acfg.ex_k, acfg.max_width)
            if next_model is None: break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, data, acfg, device)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_fails = 0
            else:
                width_fails += 1
                # Forward-only: NO ROLLBACK per-expansion
        
        # Restore local best for this context
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    def optimize_depth_at_fixed_width(curr_model):
        local_val, local_state = train_with_early_stopping(curr_model, data, acfg, device)
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        depth_fails = 0
        while depth_fails < acfg.trials_depth:
            if not can_deepen(curr_model):
                break
                
            next_model = expand_depth(curr_model, acfg.max_depth)
            if next_model is None: break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, data, acfg, device)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_fails = 0
            else:
                depth_fails += 1
        
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    mode = acfg.adp_mode
    if mode in ["width_only", "width"]:
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
        
    elif mode in ["depth_only", "depth"]:
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
        
    elif mode == "depth_to_width":
        model, base_val, base_snap = optimize_width_at_fixed_depth(model)
        global_best_val = base_val
        global_best_snap = base_snap
        
        depth_fails = 0
        while depth_fails < acfg.trials_depth:
            if not can_deepen(model): break
            nm = expand_depth(model, acfg.max_depth)
            if nm is None: break
            model = nm
            
            model, val_d, snap_d = optimize_width_at_fixed_depth(model)
            if val_d < global_best_val - acfg.delta:
                global_best_val = val_d
                global_best_snap = snap_d
                depth_fails = 0
            else:
                depth_fails += 1
        model = restore_arch_and_state(model, global_best_snap, device)

    elif mode == "width_to_depth":
        model, base_val, base_snap = optimize_depth_at_fixed_width(model)
        global_best_val = base_val
        global_best_snap = base_snap
        
        width_fails = 0
        while width_fails < acfg.trials_width:
            if not can_widen(model): break
            nm = expand_width(model, acfg.ex_k, acfg.max_width)
            if nm is None: break
            model = nm
            
            model, val_w, snap_w = optimize_depth_at_fixed_width(model)
            if val_w < global_best_val - acfg.delta:
                global_best_val = val_w
                global_best_snap = snap_w
                width_fails = 0
            else:
                width_fails += 1
        model = restore_arch_and_state(model, global_best_snap, device)

    elif mode in ["alt_width", "alt_depth"]:
        depth_sat, width_sat = False, False
        phase = "width" if mode == "alt_width" else "depth"
        
        while not (depth_sat and width_sat):
            improved = False
            if phase == "width":
                model, val, snap = optimize_width_at_fixed_depth(model)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                width_sat = not improved
                model = restore_arch_and_state(model, global_best_snap, device)
                phase = "depth"
            else:
                model, val, snap = optimize_depth_at_fixed_width(model)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                depth_sat = not improved
                model = restore_arch_and_state(model, global_best_snap, device)
                phase = "width"
                
        model = restore_arch_and_state(model, global_best_snap, device)

    return global_best_val, model

def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP Unified SSL Core")
    p.add_argument("--algo", type=str, default="plain", help="SSL algorithm name")
    p.add_argument("--adp-mode", type=str, default="width_to_depth")
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=8, help="Expand base_channels by this")
    p.add_argument("--max-width", type=int, default=128, help="Max base_channels")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-neurons", type=int, default=1_000_000)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data-root", type=str, default=".")
    
    # Model init args
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    
    args, unknown = p.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Synthetic Data for "Refactoring" demo (since we can't depend on external data presence)
    # Creating a simple dataset
    class SyntheticData(torch.utils.data.Dataset):
        def __init__(self, N=100):
            self.data = torch.randn(N, 3, 32, 32)
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx], 0 # dummy target
        
    train_ds = SyntheticData(100)
    val_ds = SyntheticData(50)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    cfg = AEConfig(
        base_channels=args.base_channels,
        depth=args.depth,
        bottleneck_dim=args.bottleneck_dim,
        device=device
    )
    model = SelfSupervisedAE(cfg).to(device)
    
    acfg = ADPConfig(
        adp_mode=args.adp_mode,
        algo=args.algo,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size
    )
    
    print(f"[ADP Unified Core] Starting search: mode={acfg.adp_mode} algo={acfg.algo} init_width={model.cfg.base_channels}")
    best_val, best_model = adp_search(model, (train_dl, val_dl), acfg, device)
    print(f"[ADP Unified Core] DONE. Best Val={best_val:.6f} Width={best_model.cfg.base_channels} Depth={best_model.cfg.depth}")

if __name__ == "__main__":
    main()

