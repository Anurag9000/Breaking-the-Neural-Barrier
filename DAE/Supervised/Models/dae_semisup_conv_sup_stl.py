import torch
from typing import Tuple

from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv, sup_dae_total_neurons


class SupDAESemiConv(SupDAEGaussianConv):
    """
    Semi-supervised conv DAE encoder + classifier.

    Architecturally this is identical to SupDAEGaussianConv; the difference
    between STL and semi-supervised variants lives in the training script
    (which splits labeled vs unlabeled data and uses a combined loss).
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x)


def sup_dae_semisup_total_neurons(width: int, depth: int, num_classes: int) -> int:
    return sup_dae_total_neurons(width, depth, num_classes)

