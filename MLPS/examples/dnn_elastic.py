"""
dnn_elastic.py
==============

Elastic-Net (L1 + L2) fully connected neural network.

Goal:
    Combine L1 sparsity and L2 weight-decay effects via explicit penalties.

Architecture:
    Identical to `dnn_l2.py` but with both l1_penalty() and l2_penalty() methods,
    and hyperparameters lambda1, lambda2 to weight each term.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DNN_Elastic(nn.Module):
    """Fully connected DNN with explicit L1 + L2 regularization for OPF prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the DNN_Elastic model.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            hidden_dim (Optional[int]): # units in hidden layers; defaults to 4 * input_dim.
            lambda1 (float): Coefficient for L1 penalty (default=1.0).
            lambda2 (float): Coefficient for L2 penalty (default=1.0).
            use_bounds (bool): Whether to apply bounded output activation.
            bounds_low (Tensor): Lower bounds per output node.
            bounds_high (Tensor): Upper bounds per output node.
            mask (Tensor): Binary mask indicating which outputs use bounds.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        logger.debug(
            f"Initializing DNN_Elastic(input={input_dim}, hidden={hidden_dim}, "
            f"output={output_dim}, λ1={self.lambda1}, λ2={self.lambda2})"
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("Bounds or mask must be provided when use_bounds=True")
            if not (len(bounds_low) == len(bounds_high) == len(mask) == output_dim):
                raise ValueError(
                    f"Mismatch: bounds/mask length must equal output_dim ({output_dim}), "
                    f"got bounds_low={len(bounds_low)}, bounds_high={len(bounds_high)}, mask={len(mask)}"
                )
            self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
        else:
            self.bound_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): shape (batch_size, input_dim).

        Returns:
            Tensor: shape (batch_size, output_dim).
        """
        out = self.net(x)
        return self.bound_layer(out)

    def get_all_shared_weights(self) -> list[torch.Tensor]:
        """
        Collect all linear-layer weights.

        Returns:
            List[Tensor]: weight tensors.
        """
        weights = [layer.weight for layer in self.net if isinstance(layer, nn.Linear)]
        logger.debug(f"Collected weights from {len(weights)} linear layers.")
        return weights

    def l1_penalty(self) -> torch.Tensor:
        """
        Compute L1 penalty: sum of absolute values of all weights.
        """
        l1 = torch.stack([w.abs().sum() for w in self.get_all_shared_weights()]).sum()
        logger.debug(f"L1 penalty: {l1.item()}")
        return l1

    def l2_penalty(self) -> torch.Tensor:
        """
        Compute L2 penalty: sum of squares of all weights.
        """
        l2 = torch.stack([(w ** 2).sum() for w in self.get_all_shared_weights()]).sum()
        logger.debug(f"L2 penalty: {l2.item()}")
        return l2

    def elastic_penalty(self) -> torch.Tensor:
        """
        Combined Elastic-Net penalty: λ1 * L1 + λ2 * L2.
        """
        penalty = self.lambda1 * self.l1_penalty() + self.lambda2 * self.l2_penalty()
        logger.debug(f"Elastic-Net penalty: {penalty.item()}")
        return penalty
