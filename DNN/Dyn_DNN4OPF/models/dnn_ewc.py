"""
===============================================================================
DNN-EWC: Elastic Weight Consolidation Model Definition
===============================================================================

This module defines the feedforward neural network architecture for the 
EWC (Elastic Weight Consolidation) continual learning framework, as described in 
Kirkpatrick et al., 2017 ("Overcoming Catastrophic Forgetting in Neural Networks").

--------------------------------------------------------------------------------
Core Objective:
    To mitigate catastrophic forgetting by constraining important weights using
    a task-specific Fisher information-based quadratic penalty.

--------------------------------------------------------------------------------
Model Architecture:
    - Standard fully connected MLP with:
        • Two hidden layers (ReLU activation)
        • Task-specific output heads (single or multiple depending on use)
    - Model weights are stored after training each task for penalty computation.

--------------------------------------------------------------------------------
Workflow Context:
    This module defines the backbone used during sequential training. It provides:
        - The MLP used for prediction
        - A model whose parameters are subjected to Fisher-based penalties
        - Compatibility with task-specific heads if needed for modularity

--------------------------------------------------------------------------------
Interactions:
    ▸ Called in:
        - `trainer.py` → during training loop (train_one_task)
        - `ewc_utils.py` → for snapshotting parameters and computing penalties
    ▸ Paired with:
        - EWC class (Fisher computation, penalty)
        - Dataset: OPF batches B1–B4 (for training)

--------------------------------------------------------------------------------
Use in Pipeline:
    1. Initialized and trained on task 0
    2. Parameters snapshotted post-training
    3. Fisher matrix computed on training data
    4. On task t>0:
        - Model loaded
        - Trained with MSE + EWC loss from previous Fisher matrices

--------------------------------------------------------------------------------
Relevant Paper:
    Kirkpatrick et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks.
    Proceedings of the National Academy of Sciences, USA.
"""
import logging
import torch
import torch.nn as nn
from typing import Optional
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DNN_EWC(nn.Module):
    """
    Feedforward neural network model designed for Elastic Weight Consolidation (EWC).
    
    This model consists of two shared hidden layers followed by a task output layer.
    The shared layers are explicitly stored to allow access for importance weight computation
    and regularization methods like EWC.
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
        Initialize the DNN_EWC model architecture.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output layer.
            hidden_dim (Optional[int]): Number of hidden units in each shared hidden layer;
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

        self.shared_layers: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        ])

        self.output_layer: nn.Linear = nn.Linear(hidden_dim, output_dim)

        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("Bounds or mask must be provided when use_bounds=True")
            if not (len(bounds_low) == len(bounds_high) == len(mask) == output_dim):
                raise ValueError(
                    f"Bounds/mask size mismatch with output_dim={output_dim} → "
                    f"got bounds_low={len(bounds_low)}, bounds_high={len(bounds_high)}, mask={len(mask)}"
                )
            self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
        else:
            self.bound_layer = nn.Identity()

        logger.info(
            f"DNN_EWC initialized with input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for layer in self.shared_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.bound_layer(x)
        return x

    def get_all_shared_weights(self) -> list[torch.Tensor]:
        """
        Retrieve weight matrices from all shared linear layers.

        Returns:
            list[torch.Tensor]: List of shared layer weight tensors.
        """
        return [layer[0].weight for layer in self.shared_layers]
