# adp_ae_temporal_core.py
# Unified single-model temporal self-supervised AE core (non-VAE, no teacher/EMA).
# Handles 7 predictive/temporal algos via the 'algo' argument:
#   'temporal_predictive', 'order_pred', 'future_latent', 'cycle_temporal',
#   'autoregressive_patches', 'motion_residual', 'latent_transition'
#
# Inputs: frames generated from static images by small random temporal transforms
#         OR real video windows prepared by the runner (B, T, C, H, W).
# Author: ADP / Breaking Neural Barrier

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------

@dataclass
class TAEConfig:
    in_channels: int = 3
    base_channels: int = 64
    depth: int = 4
    latent_dim: int = 256
    norm: str = "bn"      # 'bn'|'gn'|'ln'|'none'
    act: str = "relu"     # 'relu'|'gelu'|'silu'
    use_unet: bool = False
    patch_grid: int = 4   # for autoregressive patches
    # losses
    recon_loss: str = "mse"  # 'mse'|'l1'|'huber'
    huber_delta: float = 1.0
    # heads
    order_num_classes: int = 3  # identity, reversed, swap_inner
    # device
    device: Optional[str] = None

# ----------------------------
# Small helpers
# ----------------------------

def _norm(nc, kind):
    if kind == "bn":  return nn.BatchNorm2d(nc)
    if kind == "gn":  return nn.GroupNorm(max(1, nc // 16), nc)
    if kind == "ln":  return nn.GroupNorm(1, nc)
    return nn.Identity()

def _act(kind):
    return {"relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU()}[kind]

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, cfg: TAEConfig):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.norm = _norm(c_out, cfg.norm)
        self.act  = _act(cfg.act)
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, cfg: TAEConfig):
        super().__init__()
        self.conv1 = ConvBNAct(c_in, c_out, cfg)
        self.conv2 = ConvBNAct(c_out, c_out, cfg)
        self.pool  = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.pool(x); return x

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, cfg: TAEConfig):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvBNAct(c_in, c_out, cfg)
        self.conv2= ConvBNAct(c_out, c_out, cfg)
    def forward(self, x):
        x = self.up(x); x = self.conv(x); x = self.conv2(x); return x

# ----------------------------
# Frame encoder/decoder (2D per frame) + temporal pooling
# ----------------------------

class FrameEncoder(nn.Module):
    def __init__(self, cfg: TAEConfig):
        super().__init__()
        C = cfg.base_channels
        layers = [ConvBNAct(cfg.in_channels, C, cfg)]
        for d in range(1, cfg.depth):
            layers += [DownBlock(C, C*2, cfg)]
            C *= 2
        self.body = nn.Sequential(*layers)
        self.out_channels = C
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Linear(C, cfg.latent_dim)

    def forward(self, x):  # x: (B,C,H,W)
        f = self.body(x)
        g = self.gap(f).flatten(1)          # (B, C)
        h = self.to_latent(g)               # (B, D)
        return f, h

class FrameDecoder(nn.Module):
    def __init__(self, enc: FrameEncoder, cfg: TAEConfig):
        super().__init__()
        C = enc.out_channels
        ups = []
        for d in range(cfg.depth - 1):
            ups.append(UpBlock(C, C//2, cfg)); C//=2
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(C, cfg.in_channels, 1)

    def forward(self, f):  # f: (B,C,h,w)
        x = f
        for up in self.ups: x = up(x)
        return self.head(x)

# ----------------------------
# Temporal AE (single model)
# ----------------------------

SUPPORTED_TEMPORAL = {
    "temporal_predictive",    # 21
    "order_pred",             # 22
    "future_latent",          # 23
    "cycle_temporal",         # 24
    "autoregressive_patches", # 25
    "motion_residual",        # 26
    "latent_transition"       # 27
}

class TemporalAE(nn.Module):
    def __init__(self, cfg: TAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = FrameEncoder(cfg)
        self.dec = FrameDecoder(self.enc, cfg)
        # light heads
        self.next_latent = nn.Linear(cfg.latent_dim, cfg.latent_dim)         # for 21/23/27
        self.order_head  = nn.Linear(cfg.latent_dim, cfg.order_num_classes)  # for 22

    # ---- utils ----
    def _recon_loss(self, x, y):
        if self.cfg.recon_loss == "mse":   return F.mse_loss(x, y)
        if self.cfg.recon_loss == "l1":    return F.l1_loss(x, y)
        return F.huber_loss(x, y, delta=self.cfg.huber_delta)

    def encode_frame(self, x):     # (B,C,H,W) -> (f,h)
        return self.enc(x)

    def decode_frame(self, f):     # (B,C,h,w) -> (B,C,H,W)
        return self.dec(f)

    # ---- main entry: (B,T,C,H,W) ----
    def forward_train(self, seq: torch.Tensor, algo: str) -> Dict[str, Any]:
        assert algo in SUPPORTED_TEMPORAL, f"Unsupported algo {algo}"
        B, T, C, H, W = seq.shape
        logs: Dict[str, float] = {}

        # helper to get per-frame features/latents
        feats, lats = [], []
        for t in range(T):
            f, h = self.encode_frame(seq[:, t])
            feats.append(f); lats.append(h)
        # stack
        Fstack = torch.stack(feats, dim=1)   # (B,T,C,h,w)
        Hstack = torch.stack(lats,  dim=1)   # (B,T,D)

        # 21) temporal_predictive: predict x_{t+1} from x_t
        if algo == "temporal_predictive":
            assert T >= 2, "Need at least 2 frames"
            f_t = feats[0]
            h_t = lats[0]
            h_next_hat = self.next_latent(h_t)
            # decode from predicted latent by lifting to feature map: simple broadcast
            # (use current feature map magnitude as carrier)
            scale = (f_t.norm(p=2, dim=1, keepdim=True) + 1e-6)
            f_hat = f_t * (h_next_hat.norm(p=2, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) / scale)
            x_next_hat = self.decode_frame(f_hat)
            loss = self._recon_loss(x_next_hat, seq[:, 1])
            logs["recon_next"] = loss.item()
            return {"loss": loss, "logs": logs, "recon": x_next_hat}

        # 22) order_pred: classify whether order is identity, reversed, or inner-swap
        if algo == "order_pred":
            assert T >= 3, "Need T>=3"
            # choose permutation id uniformly
            pid = random.randint(0, self.cfg.order_num_classes - 1)
            if pid == 0:      order = [0,1,2]               # identity
            elif pid == 1:    order = [2,1,0]               # reversed
            else:             order = [0,2,1]               # swap inner
            # aggregate sequence latent (mean)
            z = Hstack[:, order].mean(dim=1)                # (B,D)
            logits = self.order_head(z)
            target = torch.full((B,), pid, device=seq.device, dtype=torch.long)
            cls_loss = F.cross_entropy(logits, target)
            logs["order_ce"] = cls_loss.item()
            # optional reconstruction of the last permuted frame
            x_perm_last = seq[:, order[-1]]
            f_last, _ = self.encode_frame(x_perm_last)
            rec_last = self.decode_frame(f_last)
            rec_loss = self._recon_loss(rec_last, x_perm_last)
            logs["recon"] = rec_loss.item()
            loss = cls_loss + rec_loss
            return {"loss": loss, "logs": logs, "recon": rec_last}

        # 23) future_latent: consistency h_{t+1} vs Linear(h_t)
        if algo == "future_latent":
            assert T >= 2, "Need T>=2"
            h_t = Hstack[:, 0]
            h_tp1 = Hstack[:, 1].detach()
            h_hat = self.next_latent(h_t)
            lat_loss = F.mse_loss(h_hat, h_tp1)
            # reconstruct next frame from predicted latent (same trick as above)
            f_t = Fstack[:, 0]
            scale = (f_t.norm(p=2, dim=1, keepdim=True) + 1e-6)
            f_hat = f_t * (h_hat.norm(p=2, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) / scale)
            x_hat = self.decode_frame(f_hat)
            rec_loss = self._recon_loss(x_hat, seq[:, 1])
            loss = rec_loss + 0.1 * lat_loss
            logs["recon_next"] = rec_loss.item(); logs["latent_mse"] = lat_loss.item()
            return {"loss": loss, "logs": logs, "recon": x_hat}

        # 24) cycle_temporal: encode→decode→encode cycle on each frame, match latents
        if algo == "cycle_temporal":
            cyc = 0.0; rec = 0.0
            for t in range(T):
                f, h = feats[t], lats[t]
                x_rec = self.decode_frame(f)
                rec += self._recon_loss(x_rec, seq[:, t])
                _, h2 = self.encode_frame(x_rec.detach())
                cyc += F.mse_loss(h2, h.detach())
            rec = rec / T; cyc = cyc / T
            loss = rec + 0.1 * cyc
            logs["recon"] = rec.item(); logs["latent_cycle"] = cyc.item()
            return {"loss": loss, "logs": logs, "recon": self.decode_frame(feats[-1])}

        # 25) autoregressive_patches: predict next patch from previous ones (single frame)
        if algo == "autoregressive_patches":
            # choose a frame, select a patch index k to predict, mask future patches
            t = 0
            x = seq[:, t]
            B, C, H, W = x.shape
            G = self.cfg.patch_grid
            ph, pw = H // G, W // G
            total = G * G
            k = random.randint(1, total-1)  # predict patch k given 0..k-1
            mask = torch.zeros(B, 1, H, W, device=x.device)
            # reveal 0..k-1 patches in raster order
            for idx in range(k):
                r, c = divmod(idx, G)
                mask[:, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw] = 1.0
            x_vis = x * mask
            f, _ = self.encode_frame(x_vis)
            x_rec = self.decode_frame(f)
            # loss only on patch k
            r, c = divmod(k, G)
            patch_mask = torch.zeros_like(mask)
            patch_mask[:, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw] = 1.0
            # weighted loss
            diff = (x_rec - x)
            if self.cfg.recon_loss == "mse":
                num = (diff**2 * patch_mask).sum()
            elif self.cfg.recon_loss == "l1":
                num = (diff.abs() * patch_mask).sum()
            else:
                hub = F.huber_loss(x_rec, x, delta=self.cfg.huber_delta, reduction="none")
                num = (hub * patch_mask).sum()
            denom = patch_mask.sum().clamp_min(1.0)
            loss = num / denom
            logs["patch_k"] = float(k); logs["patch_loss"] = loss.item()
            return {"loss": loss, "logs": logs, "recon": x_rec}

        # 26) motion_residual: predict residual r such that x_t + r ≈ x_{t+1}
        if algo == "motion_residual":
            assert T >= 2, "Need T>=2"
            xt = seq[:, 0]; xt1 = seq[:, 1]
            # encode xt; decode a residual image
            f_t, _ = self.encode_frame(xt)
            r = self.decode_frame(f_t)     # residual predictor
            x_next = xt + r
            loss = self._recon_loss(x_next, xt1)
            logs["recon_next"] = loss.item()
            return {"loss": loss, "logs": logs, "recon": x_next}

        # 27) latent_transition: learn A: h_t -> h_{t+1}, and decode predicted next frame
        if algo == "latent_transition":
            assert T >= 2, "Need T>=2"
            h_t = Hstack[:, 0]
            h_tp1 = Hstack[:, 1].detach()
            h_hat = self.next_latent(h_t)
            lat_loss = F.mse_loss(h_hat, h_tp1)
            # decode via feature scaling trick (like 21/23)
            f_t = Fstack[:, 0]
            scale = (f_t.norm(p=2, dim=1, keepdim=True) + 1e-6)
            f_hat = f_t * (h_hat.norm(p=2, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) / scale)
            x_hat = self.decode_frame(f_hat)
            rec_loss = self._recon_loss(x_hat, seq[:, 1])
            loss = rec_loss + 0.1 * lat_loss
            logs["recon_next"] = rec_loss.item(); logs["latent_mse"] = lat_loss.item()
            return {"loss": loss, "logs": logs, "recon": x_hat}

        raise RuntimeError("Unhandled branch.")
        
def build_model(cfg: Optional[TAEConfig] = None) -> TemporalAE:
    cfg = cfg or TAEConfig()
    return TemporalAE(cfg)
