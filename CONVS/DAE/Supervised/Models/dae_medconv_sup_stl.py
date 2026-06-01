"""
Medical-image Conv DAE + diagnosis classifier.

This is aliased to the Gaussian Conv DAE encoder + classifier implementation,
so it inherits the same STL and ADP behaviour while conceptually targeting a
medical-style image classification task.
"""

from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv as SupDAEMedConv  # noqa: F401
from .dae_gaussian_conv_sup_stl import sup_dae_total_neurons as sup_dae_medconv_total_neurons  # noqa: F401

