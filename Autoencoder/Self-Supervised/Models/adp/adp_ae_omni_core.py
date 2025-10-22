# adp_ae_omni_core.py
# Omni-dispatcher for ALL AE variants across Parts A–I (single model per core).
# It picks the correct core (ssl / temporal / spatial / regular) for the requested --algo,
# builds the model with the appropriate config, and exposes a uniform forward_train().

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

# --- Import your existing cores ---
# (These are assumed to be in the same directory.)
from adp_ae_ssl_core import AEConfig as SSLConfig, build_model as build_ssl, SUPPORTED_ALGOS as SSL_SET
from adp_ae_temporal_core import TAEConfig as TEMPConfig, build_model as build_temp, SUPPORTED_TEMPORAL as TEMP_SET
from adp_ae_spatial_core import SAEConfig as SPConfig, build_model as build_sp, SUPPORTED_SPATIAL as SP_SET
from adp_ae_regular_core import RAEConfig as REGConfig, build_model as build_reg, SUPPORTED_REGULAR as REG_SET

# registry (strings only; actual sets come from cores)
OMNI_SETS = {
    "ssl": set(SSL_SET),            # Parts A, B, F, G, H, I (and any patched algos)
    "temporal": set(TEMP_SET),      # Part C
    "spatial": set(SP_SET),         # Part D
    "regular": set(REG_SET),        # Part E
}

def _find_family(algo: str) -> str:
    for fam, aset in OMNI_SETS.items():
        if algo in aset:
            return fam
    raise ValueError(f"Unknown algo '{algo}'. Not found in SSL/Temporal/Spatial/Regular registries.")

@dataclass
class OmniConfig:
    # common
    device: Optional[str] = None
    # ---- SSL core ----
    ssl_base_ch: int = 64
    ssl_depth: int = 4
    ssl_unet: bool = False
    ssl_norm: str = "bn"
    ssl_act: str = "relu"
    ssl_recon_loss: str = "mse"
    ssl_huber_delta: float = 1.0
    ssl_mask_ratio: float = 0.6
    ssl_block_size: int = 16
    # selected regularizers in ssl core (if used)
    w_sparse_l1: float = 0.0
    w_group_sparse: float = 0.0
    group_size: int = 16
    w_contractive: float = 0.0
    w_entropy: float = 0.0
    w_tv: float = 0.0
    w_whiten: float = 0.0

    # ---- Temporal core ----
    temp_base_ch: int = 64
    temp_depth: int = 4
    temp_latent_dim: int = 256
    temp_norm: str = "bn"
    temp_act: str = "relu"
    temp_patch_grid: int = 4
    temp_recon_loss: str = "mse"
    temp_huber_delta: float = 1.0

    # ---- Spatial core ----
    sp_base_ch: int = 64
    sp_depth: int = 4
    sp_latent_dim: int = 256
    sp_norm: str = "bn"
    sp_act: str = "relu"
    sp_patch_grid: int = 3
    sp_jigsaw_classes: int = 6
    sp_scale_bins: int = 3
    sp_trans_max_px: int = 8
    sp_recon_loss: str = "mse"
    sp_huber_delta: float = 1.0

    # ---- Regular core ----
    reg_base_ch: int = 64
    reg_depth: int = 4
    reg_latent_dim: int = 256
    reg_norm: str = "bn"
    reg_act: str = "relu"
    reg_recon_loss: str = "mse"
    reg_huber_delta: float = 1.0
    # weights (can be changed from runner)
    w_laplacian: float = 1.0
    w_manifold: float = 1.0
    w_tangent: float = 0.1
    w_reg_entropy: float = 0.001
    w_mi: float = 0.001
    w_orth: float = 1e-4
    w_lowrank: float = 1e-4
    w_normalize: float = 0.01
    w_reg_whiten: float = 1e-4
    tangent_eps: float = 0.03

class OmniAE(nn.Module):
    def __init__(self, algo: str, cfg: OmniConfig):
        super().__init__()
        self.algo = algo
        self.family = _find_family(algo)
        self.cfg = cfg

        if self.family == "ssl":
            sc = SSLConfig(
                in_channels=3,
                base_channels=cfg.ssl_base_ch,
                depth=cfg.ssl_depth,
                use_unet=cfg.ssl_unet,
                norm=cfg.ssl_norm,
                act=cfg.ssl_act,
                w_sparse_l1=cfg.w_sparse_l1,
                w_group_sparse=cfg.w_group_sparse,
                group_size=cfg.group_size,
                w_contractive=cfg.w_contractive,
                w_entropy=cfg.w_entropy,
                w_tv=cfg.w_tv,
                w_whiten=cfg.w_whiten,
                recon_loss=cfg.ssl_recon_loss,
                huber_delta=cfg.ssl_huber_delta,
                mask_ratio=cfg.ssl_mask_ratio,
                block_size=cfg.ssl_block_size,
                device=cfg.device,
            )
            self.core = build_ssl(sc)

        elif self.family == "temporal":
            tc = TEMPConfig(
                in_channels=3,
                base_channels=cfg.temp_base_ch,
                depth=cfg.temp_depth,
                latent_dim=cfg.temp_latent_dim,
                norm=cfg.temp_norm,
                act=cfg.temp_act,
                patch_grid=cfg.temp_patch_grid,
                recon_loss=cfg.temp_recon_loss,
                huber_delta=cfg.temp_huber_delta,
                device=cfg.device,
            )
            self.core = build_temp(tc)

        elif self.family == "spatial":
            spc = SPConfig(
                in_channels=3,
                base_channels=cfg.sp_base_ch,
                depth=cfg.sp_depth,
                latent_dim=cfg.sp_latent_dim,
                norm=cfg.sp_norm,
                act=cfg.sp_act,
                patch_grid=cfg.sp_patch_grid,
                jigsaw_classes=cfg.sp_jigsaw_classes,
                scale_bins=cfg.sp_scale_bins,
                trans_max_px=cfg.sp_trans_max_px,
                recon_loss=cfg.sp_recon_loss,
                huber_delta=cfg.sp_huber_delta,
                device=cfg.device,
            )
            self.core = build_sp(spc)

        else:  # "regular"
            rc = REGConfig(
                in_channels=3,
                base_channels=cfg.reg_base_ch,
                depth=cfg.reg_depth,
                latent_dim=cfg.reg_latent_dim,
                norm=cfg.reg_norm,
                act=cfg.reg_act,
                recon_loss=cfg.reg_recon_loss,
                huber_delta=cfg.reg_huber_delta,
                w_laplacian=cfg.w_laplacian,
                w_manifold=cfg.w_manifold,
                w_tangent=cfg.w_tangent,
                w_entropy=cfg.w_reg_entropy,
                w_mi=cfg.w_mi,
                w_orth=cfg.w_orth,
                w_lowrank=cfg.w_lowrank,
                w_normalize=cfg.w_normalize,
                w_whiten=cfg.w_reg_whiten,
                tangent_eps=cfg.tangent_eps,
                device=cfg.device,
            )
            # build_model requires algo for regular core
            self.core = build_reg(rc, algo=algo)

    def forward_train(self, x, algo: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # delegate to the selected core
        a = algo or self.algo
        if self.family == "regular":
            # regular core is "bound" to specific algo; forward_train requires same string
            return self.core.forward_train(x, a)
        return self.core.forward_train(x, a)

def build_model(algo: str, cfg: Optional[OmniConfig] = None) -> OmniAE:
    return OmniAE(algo, cfg or OmniConfig())
