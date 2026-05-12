"""
ADP wrapper for medical U-Net DAE + segmentation.

Reuses the U-Net DAE supervised ADP implementation.
"""

from .dae_unet_conv_sup_stl_adp_width_to_depth import (  # noqa: F401
    ADPConfig,
    adp_search,
    make_loaders,
)
from .dae_unet_conv_sup_stl import SupDAEUNetConv as ModelClass  # type: ignore # noqa: F401
