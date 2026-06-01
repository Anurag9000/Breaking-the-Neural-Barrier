"""
DAE-regularized GAN discriminator for classification.

This is implemented by aliasing to the DAE-regularized ResNet classifier
backbone, which already combines a ResNet classifier with an auxiliary
reconstruction head.
"""

from .dae_resnet_reg_sup_stl import SupDAEResNet as SupDAEGANReg  # noqa: F401
from .dae_resnet_reg_sup_stl import sup_dae_resnet_total_neurons as sup_dae_ganreg_total_neurons  # noqa: F401

