"""
ADP wrapper for inverse-problem DAE.

Reuses the Gaussian Conv DAE supervised ADP implementation.
"""

from .dae_gaussian_conv_sup_stl_adp_width_to_depth import (  # noqa: F401
    ADPConfig,
    adp_search,
    make_loaders,
)
from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv as ModelClass  # noqa: F401

