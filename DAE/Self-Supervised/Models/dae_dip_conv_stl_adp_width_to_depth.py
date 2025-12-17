from .dae_dip_conv_stl import DAEDIPConv as ModelClass  # noqa: F401
from .dae_gaussian_conv_stl_adp_width_to_depth import (  # noqa: F401
    ADPConfig,
    adp_search,
    make_loaders,
)

# ADP for DIP reuses the Gaussian Conv DAE search utilities, but is conceptually
# a per-image decoder optimised from noise (see STL runner). Here we keep the
# same width/depth expansion logic.

