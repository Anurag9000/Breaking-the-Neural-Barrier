"""
Medical-image U-Net DAE + segmentation head.

This is aliased to the existing U-Net DAE supervised model, which is already
set up for segmentation/classification on CIFAR-style inputs.
"""

from .dae_unet_conv_sup_stl import SupDAEUNetConv as SupDAEMedUNet  # type: ignore # noqa: F401
from .dae_unet_conv_sup_stl import sup_dae_total_neurons as sup_dae_medunet_total_neurons  # noqa: F401

