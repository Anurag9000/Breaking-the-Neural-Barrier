"""
Super-resolution DAE: reconstruct HR from LR+noise.

This uses the Gaussian Conv DAE encoder + classifier backbone, interpreted as
operating on upsampled CIFAR inputs. For ADP we reuse the existing Gaussian
Conv DAE supervised machinery.
"""

from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv as SupDAESuperRes  # noqa: F401
from .dae_gaussian_conv_sup_stl import sup_dae_total_neurons as sup_dae_superres_total_neurons  # noqa: F401

