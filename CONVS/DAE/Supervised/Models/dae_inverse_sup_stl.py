"""
Inverse-problem DAE (deblurring / deconvolution + supervised head).

This is aliased to the Gaussian Conv DAE encoder + classifier model; training
scripts can provide appropriate corrupted inputs to emulate deblurring.
"""

from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv as SupDAEInverse  # noqa: F401
from .dae_gaussian_conv_sup_stl import sup_dae_total_neurons as sup_dae_inverse_total_neurons  # noqa: F401

