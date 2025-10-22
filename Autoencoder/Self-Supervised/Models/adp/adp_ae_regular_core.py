# adp_ae_regular_core.py
# Single-model Autoencoder core for "E. Regularization & Representation Geometry" (36–45).
# Implements: laplacian, manifold, tangent_prop, entropy_reg, mi_surrogate,
# orthogonal, low_rank, tied_weights, normalized, whitening
# Author: ADP / Breaking Neural Barrier

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------
@dataclass
class RAEConfig:
    in_channels: int = 3
    base_channels: int = 64
    depth: int = 4
    latent_dim: int = 256
    norm: str = "bn"            # 'bn'|'gn'|'ln'|'none'
    act: str = "relu"           # 'relu'|'gelu'|'silu'
    recon_loss: str = "mse"     # 'mse'|'l1'|'huber'
    huber_delta: float = 1.0
    # Regularizer weights (you can also just select via algo)
    w_laplacian: float = 1.0
    w_manifold: float = 1.0
    w_tangent: float = 0.1
    w_entropy: float = 0.001
    w_mi: float = 0.001
    w_orth: float = 1e-4
    w_lowrank: float = 1e-4
    w_normalize: float = 0.01
    w_whiten: float = 1e-4
    # Tangent-Prop perturbation scale
    tangent_eps: float = 0.03
    # Device (for helper tensors)
    device: Optional[str] = None

# ----------------------------
# Blocks
# ----------------------------
def _norm(c, kind):
    if kind=="bn": return nn.BatchNorm2d(c)
    if kind=="gn": return nn.GroupNorm(max(1, c//16), c)
    if kind=="ln": return nn.GroupNorm(1, c)
    return nn.Identity()

def _act(a): return {"relu":nn.ReLU(inplace=True),"gelu":nn.GELU(),"silu":nn.SiLU()}[a]

class ConvBNAct(nn.Module):
    def __init__(self, ci, co, cfg: RAEConfig):
        super().__init__()
        self.c = nn.Conv2d(ci, co, 3, 1, 1, bias=False)
        self.n = _norm(co, cfg.norm)
        self.a = _act(cfg.act)
    def forward(self, x): return self.a(self.n(self.c(x)))

class Down(nn.Module):
    def __init__(self, ci, co, cfg: RAEConfig):
        super().__init__()
        self.b1 = ConvBNAct(ci, co, cfg); self.b2 = ConvBNAct(co, co, cfg)
        self.p = nn.MaxPool2d(2)
    def forward(self, x):
        x=self.b1(x); x=self.b2(x); return self.p(x)

class Up(nn.Module):
    def __init__(self, ci, co, cfg: RAEConfig):
        super().__init__()
        self.u = nn.Upsample(scale_factor=2, mode="nearest")
        self.b1 = ConvBNAct(ci, co, cfg); self.b2 = ConvBNAct(co, co, cfg)
    def forward(self,x):
        x=self.u(x); x=self.b1(x); return self.b2(x)

# ----------------------------
# Encoder / Decoder with tied FC option
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, cfg: RAEConfig):
        super().__init__()
        C = cfg.base_channels
        layers = [ConvBNAct(cfg.in_channels, C, cfg)]
        for _ in range(1, cfg.depth):
            layers.append(Down(C, C*2, cfg)); C*=2
        self.body = nn.Sequential(*layers)
        self.out_c = C
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, cfg.latent_dim, bias=True)  # will be tied (transpose) on demand

    def forward(self, x):
        f = self.body(x)
        z = self.fc(self.gap(f).flatten(1))
        return f, z

class Decoder(nn.Module):
    def __init__(self, enc: Encoder, cfg: RAEConfig, use_tied: bool):
        super().__init__()
        self.use_tied = use_tied
        self.enc_ref = enc if use_tied else None
        C = enc.out_c
        ups=[]
        for _ in range(cfg.depth-1):
            ups.append(Up(C, C//2, cfg)); C//=2
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(C, cfg.in_channels, 1)
        # linear "unpool" from latent back to conv features (channel-scale carrier)
        self.fc_proj = nn.Linear(cfg.latent_dim, enc.out_c, bias=False)
        if use_tied:
            # weight will be set from enc.fc.weight.T at forward
            with torch.no_grad():
                self.fc_proj.weight.copy_(enc.fc.weight.T)

    def forward(self, f, z):
        # if tied, sync weights each call
        if self.use_tied and (self.fc_proj.weight.shape == self.enc_ref.fc.weight.T.shape):
            self.fc_proj.weight = nn.Parameter(self.enc_ref.fc.weight.T, requires_grad=False)
        # scale base feature map with latent projection (simple carrier)
        scale = self.fc_proj(z).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        base = f * (scale / (f.norm(p=2, dim=1, keepdim=True) + 1e-6))
        x = base
        for u in self.ups: x = u(x)
        return self.head(x)

# ----------------------------
# Main model
# ----------------------------
SUPPORTED_REGULAR = {
    "laplacian",        # 36
    "manifold",         # 37
    "tangent_prop",     # 38
    "entropy_reg",      # 39
    "mi_surrogate",     # 40 (variance+whitening as MI proxy; single-model)
    "orthogonal",       # 41
    "low_rank",         # 42
    "tied_weights",     # 43
    "normalized",       # 44
    "whitening"         # 45
}

class RegularAE(nn.Module):
    def __init__(self, cfg: RAEConfig, algo: str):
        super().__init__()
        assert algo in SUPPORTED_REGULAR
        self.cfg = cfg
        self.algo = algo
        self.enc = Encoder(cfg)
        self.dec = Decoder(self.enc, cfg, use_tied=(algo=="tied_weights"))

    # ---------- losses ----------
    def _recon_loss(self, y, x):
        if self.cfg.recon_loss == "mse":   return F.mse_loss(y, x)
        if self.cfg.recon_loss == "l1":    return F.l1_loss(y, x)
        return F.huber_loss(y, x, delta=self.cfg.huber_delta)

    def _latent_cov(self, z):  # (B,D) -> (D,D)
        z0 = z - z.mean(dim=0, keepdim=True)
        return (z0.T @ z0) / (z0.shape[0] - 1 + 1e-6)

    def _whiten_penalty(self, z):
        cov = self._latent_cov(z)
        I = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return ((cov - I) ** 2).mean()

    def _entropy_proxy(self, z):
        # maximize entropy ~ maximize variance: minimize negative log std
        std = (z.var(dim=0) + 1e-6).sqrt()
        return (-torch.log(std)).mean()

    def _pairwise_dists(self, U):  # (B,D) -> (B,B) sq distances
        # ||u_i - u_j||^2 = ||u_i||^2 + ||u_j||^2 - 2 u_i.u_j
        G = U @ U.t()
        n2 = U.pow(2).sum(dim=1, keepdim=True)
        D = n2 + n2.t() - 2*G
        return D.clamp_min(0)

    def _laplacian_penalty(self, zx, z):  # zx: (B,Dx) — "input space" embedding; z: (B,D) latent
        # Build affinity W from zx with Gaussian kernel using median heuristic for sigma.
        with torch.no_grad():
            Dx = self._pairwise_dists(zx)
            # median of off-diagonal distances
            m = torch.median(Dx[Dx>0]).clamp_min(1e-6)
            sigma2 = m
            W = torch.exp(-Dx / (2*sigma2))
            W.fill_diagonal_(0.0)
            Dg = torch.diag(W.sum(dim=1))
            L = Dg - W
        # Laplacian loss: sum over channels z^T L z
        return torch.trace(z.t() @ L @ z) / (z.shape[0] + 1e-6)

    def _manifold_penalty(self, zx, z):
        Dx = self._pairwise_dists(zx)
        Dz = self._pairwise_dists(z)
        Dx = Dx / (Dx.mean() + 1e-6)
        Dz = Dz / (Dz.mean() + 1e-6)
        return F.mse_loss(Dz, Dx)

    def _tangent_prop(self, x):
        # Encourage output to be locally invariant to small input perturbations.
        # Finite-difference: || f(x + eps) - f(x) ||^2
        eps = self.cfg.tangent_eps
        noise = torch.randn_like(x) * eps
        f0, z0 = self.enc(x)
        y0 = self.dec(f0, z0)
        f1, z1 = self.enc((x + noise).clamp(0,1))
        y1 = self.dec(f1, z1)
        return F.mse_loss(y1, y0.detach())

    def _orth_penalty(self):
        # Encourage conv/linear weights to be orthogonal
        loss = 0.0
        def pen(W):
            if W.dim() > 2:
                Wm = W.view(W.shape[0], -1)
            else:
                Wm = W
            G = Wm @ Wm.t()
            I = torch.eye(G.shape[0], device=W.device, dtype=W.dtype)
            return ((G - I) ** 2).mean()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                loss = loss + pen(m.weight)
        return loss

    def _nuclear_norm(self, Z):  # nuclear norm of (B,D)
        # Note: this encourages lower-rank Z when minimized
        U, S, Vh = torch.linalg.svd(Z - Z.mean(dim=0, keepdim=True), full_matrices=False)
        return S.sum()

    def _normalize_penalty(self, z):
        # Encourage ||z||_2 ≈ 1
        n2 = (z.pow(2).sum(dim=1) + 1e-6)
        return ((n2 - 1.0) ** 2).mean()

    # ---------- training entry ----------
    def forward_train(self, x: torch.Tensor, algo: str) -> Dict[str, Any]:
        assert algo == self.algo, "Model was built for a specific algo"
        f, z = self.enc(x)
        recon = self.dec(f, z)
        base = self._recon_loss(recon, x)
        loss = base
        logs: Dict[str, float] = {"recon": base.item()}

        # Build an "input-space" embedding zx (pre-latent pooled features)
        with torch.no_grad():
            zx = f.mean(dim=[2,3])  # (B, C)

        if algo == "laplacian":
            lp = self._laplacian_penalty(zx, z)
            loss = loss + self.cfg.w_laplacian * lp
            logs["laplacian"] = lp.item()

        elif algo == "manifold":
            mp = self._manifold_penalty(zx, z)
            loss = loss + self.cfg.w_manifold * mp
            logs["manifold"] = mp.item()

        elif algo == "tangent_prop":
            tp = self._tangent_prop(x)
            loss = loss + self.cfg.w_tangent * tp
            logs["tangent"] = tp.item()

        elif algo == "entropy_reg":
            ent = self._entropy_proxy(z)
            loss = loss + self.cfg.w_entropy * ent
            logs["neg_entropy"] = ent.item()

        elif algo == "mi_surrogate":
            # Simple MI surrogate: maximize latent entropy and decorrelate dimensions
            ent = self._entropy_proxy(z)
            white = self._whiten_penalty(z)
            loss = loss + self.cfg.w_mi * (ent + white)
            logs["neg_entropy"] = ent.item(); logs["whiten"] = white.item()

        elif algo == "orthogonal":
            ort = self._orth_penalty()
            loss = loss + self.cfg.w_orth * ort
            logs["orth"] = ort.item()

        elif algo == "low_rank":
            nuc = self._nuclear_norm(z)
            loss = loss + self.cfg.w_lowrank * nuc
            logs["nuclear"] = nuc.item()

        elif algo == "tied_weights":
            # already using tied fc weights; optionally add small tie penalty (almost zero)
            # here base recon is the principal objective
            pass

        elif algo == "normalized":
            nm = self._normalize_penalty(z)
            loss = loss + self.cfg.w_normalize * nm
            logs["norm_unit"] = nm.item()

        elif algo == "whitening":
            wh = self._whiten_penalty(z)
            loss = loss + self.cfg.w_whiten * wh
            logs["whiten"] = wh.item()

        else:
            raise RuntimeError("Unhandled algo")

        return {"loss": loss, "logs": logs, "recon": recon}

def build_model(cfg: Optional[RAEConfig] = None, algo: str = "laplacian") -> RegularAE:
    cfg = cfg or RAEConfig()
    return RegularAE(cfg, algo)
