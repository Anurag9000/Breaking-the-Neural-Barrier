"""
dnn_stl.py
==========

Standard fully connected neural network (STL - Single Task Learning).

Purpose:
    Baseline model that trains a separate MLP for each task without knowledge transfer.

Structure:
    - Two hidden layers with ReLU activations
    - One output layer
    - No regularization or multi-task components

Role in Paper:
    Serves as the "lower bound" comparison in continual learning experiments.
    Demonstrates catastrophic forgetting when trained sequentially.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional
import sys
from pathlib import Path

# Add the parent directory of Dyn_DNN4OPF to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))



from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FullyConnectedNet(nn.Module):
    """
    Fully connected feedforward neural network with two hidden layers.
    """

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
        Initialize the FullyConnectedNet.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output units.
            hidden_dim (Optional[int]): Number of neurons in hidden layers;
                defaults to 4 * input_dim if not provided.
            use_bounds (bool): Whether to apply bounded activation to the output.
            bounds_low (torch.Tensor): Lower bounds for each output node.
            bounds_high (torch.Tensor): Upper bounds for each output node.
            mask (torch.Tensor): Binary mask indicating which outputs use bounds.
        """
        super().__init__()
        # — default hidden_dim to 4×input_dim if not overridden —
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        logger.debug(
            f"Initializing FullyConnectedNet with input_dim={input_dim}, "
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
                    f"got {len(bounds_low)}, {len(bounds_high)}, {len(mask)})"
                )
            self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
        else:
            self.bound_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

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
