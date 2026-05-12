"""
ADP wrapper for DAE-regularized GAN discriminator.

Reuses the DAE-regularized ResNet supervised ADP machinery.
"""

from .dae_resnet_reg_sup_stl_adp_width_to_depth import (  # noqa: F401
    ADPConfig,
    adp_search,
)
from CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet as ModelClass  # noqa: F401
