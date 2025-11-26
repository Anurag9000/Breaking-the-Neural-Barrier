"""Profile registry for ae_contractive variant."""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict

from Autoencoder.ae_contractive.Models.variant_impl import AEContractiveModel


def _builder(meta, args, **overrides):
    cfg = {
        "latent_dim": getattr(args, "latent_dim", 128),
        "hidden": getattr(args, "hidden", 256),
        "layers": getattr(args, "layers", 3),
        "gconv": getattr(args, "gconv", "gcn2"),
        "proj_newton_steps": getattr(args, "proj_newton_steps", 1),
    }
    cfg.update(overrides)
    return AEContractiveModel(meta, **cfg)


PROFILE_REGISTRY: Dict[str, Callable] = {
    name: partial(_builder)
    for name in [
        "adp_base_2head",
        "adp_base_4head",
        "adp_den_2head",
        "adp_den_2head_depth_only",
        "adp_den_2_head_width_only",
        "adp_den_4head_depth_only",
        "adp_den_4head_expandtillplateuing",
        "adp_den_4_head_width_only",
        "adp_den_alt_depth_1_head",
        "adp_den_alt_depth_2_head",
        "adp_den_alt_depth_4_head",
        "adp_den_alt_width_1_head",
        "adp_den_alt_width_2_head",
        "adp_den_alt_width_4_head",
        "adp_den_depth_only",
        "adp_den_expandtillplateuing",
        "adp_den_width_only",
        "adp_depth",
        "adp_depth_2head",
        "adp_depth_4head",
        "adp_width",
        "adp_width_2head",
        "adp_width_4head",
        "dnn_stl",
    ]
}
