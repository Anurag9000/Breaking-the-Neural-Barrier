"""Profile registry for TCN AE variant."""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Tuple

from Autoencoder.ae_tcn_stl.Models.variant_impl import AETCNModel


def _parse_dilations(args) -> Tuple[int, ...]:
    if isinstance(args.tcn_dilations, (list, tuple)):
        return tuple(int(d) for d in args.tcn_dilations)
    return tuple(int(x) for x in str(args.tcn_dilations).split(','))


def _builder(meta, args, **overrides):
    cfg = {
        "temporal_len": getattr(args, "temporal_len", 1),
        "tcn_channels": getattr(args, "tcn_channels", 256),
        "tcn_dilations": _parse_dilations(args),
        "hidden": getattr(args, "hidden", 256),
        "latent_dim": getattr(args, "latent_dim", 128),
        "proj_newton_steps": getattr(args, "proj_newton_steps", 1),
    }
    cfg.update(overrides)
    return AETCNModel(meta, **cfg)


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
