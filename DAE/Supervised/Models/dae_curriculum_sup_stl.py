"""
Curriculum DAE (noise level schedule + classifier head).

We alias to the Gaussian Conv DAE encoder + classifier; curriculum behaviour
is controlled by the STL/ADP runners via their noise/noise-schedule knobs.
"""

from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv as SupDAECurriculum  # noqa: F401
from .dae_gaussian_conv_sup_stl import sup_dae_total_neurons as sup_dae_curriculum_total_neurons  # noqa: F401

