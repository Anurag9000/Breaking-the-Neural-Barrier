"""
Student-teacher DAE: student denoiser mimics clean teacher + labels.

Implemented as an alias to the Gaussian Conv DAE encoder + classifier; training
scripts can supply a fixed teacher model if desired.
"""

from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv as SupDAEStudent  # noqa: F401
from .dae_gaussian_conv_sup_stl import sup_dae_total_neurons as sup_dae_student_total_neurons  # noqa: F401

