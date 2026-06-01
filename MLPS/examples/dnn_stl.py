"""
dnn_stl.py
==========

Standard fully connected neural network (STL - Single Task Learning).

Purpose:
    Baseline model that trains a separate MLP for each task without knowledge transfer.

Structure:
    - **Configurable** number of hidden layers (*depth*)  
    - **Configurable** neurons per hidden layer (*width*)  
    - ReLU activations
    - Optional bounded activation on the output layer

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
    Fully connected feed-forward neural network with a configurable
    number of hidden layers (*depth*) and neurons per hidden layer (*width*).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: Optional[int] = None,
        depth: int = 2,
        # -------- legacy arg kept for backward-compat ----------
        hidden_dim: Optional[int] = None,
        # --------------------------------------------------------
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
            width (Optional[int]): Neurons in each hidden layer; defaults to 4 * input_dim.
            depth (int): Number of hidden layers (≥ 1).
            hidden_dim (Optional[int]): Deprecated alias for *width* (kept for compatibility).
            use_bounds (bool): Whether to apply bounded activation to the output.
            bounds_low / bounds_high (torch.Tensor): Per-output lower / upper bounds.
            mask (torch.Tensor): Binary mask indicating which outputs use bounds.
        """
        super().__init__()

        # --------------------------------------------------
        # Resolve width, preserving legacy `hidden_dim` flag
        # --------------------------------------------------
        if width is None:
            width = hidden_dim if hidden_dim is not None else 4 * input_dim
        if depth < 1:
            raise ValueError("`depth` must be ≥ 1")

        logger.debug(
            f"Initializing FullyConnectedNet with input_dim={input_dim}, "
            f"width={width}, depth={depth}, output_dim={output_dim}"
        )

        # -----------------------------
        # Dynamically build the network
        # -----------------------------
        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

        # ---------------------------
        # Optional bounded activation
        # ---------------------------
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

    # ------------- forward & helpers stay unchanged -------------
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
