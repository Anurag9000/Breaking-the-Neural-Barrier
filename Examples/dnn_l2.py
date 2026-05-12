"""
dnn_l2.py
=========

L2-Regularized fully connected neural network (weight decay).

Goal:
    Encourage parameter sparsity and improve generalization through L2 penalty.

Architecture:
    Identical to `dnn_stl.py` but intended for training with L2 regularization (via optimizer).

Paper Context:
    Forms an intermediate baseline between STL and advanced continual learning methods.
    Helps assess whether simple weight decay reduces forgetting.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DNN_L2(nn.Module):
    """Fully connected DNN with L2 regularization for OPF prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the DNN_L2 model.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            hidden_dim (Optional[int]): Number of hidden units in each hidden layer;
                defaults to 4 * input_dim if not provided.
            use_bounds (bool): Whether to apply bounded output activation.
            bounds_low (torch.Tensor): Lower bounds for each output node.
            bounds_high (torch.Tensor): Upper bounds for each output node.
            mask (torch.Tensor): Binary mask indicating which outputs use bounds.
        """
        super().__init__()
        # — default hidden_dim to 4×input_dim if not overridden —
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        logger.debug(
            f"Initializing DNN_L2 with input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}"
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
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        output = self.net(x)
        output = self.bound_layer(output)
        return output

    def get_all_shared_weights(self) -> list[torch.Tensor]:
        """
        Get weights of all shared fully connected layers.

        Returns:
            list[torch.Tensor]: List of weight tensors for each shared layer.
        """
        weights = [layer.weight for layer in self.net if isinstance(layer, nn.Linear)]
        logger.debug(f"Collected weights from {len(weights)} shared layers.")
        return weights
