"""
dnn_mtl.py
==========

Multi-Task Learning (MTL) model with shared trunk and task-specific heads.

Purpose:
    Trains a shared representation across tasks with per-task output heads.

Architecture:
    - Shared: 2-layer fully connected with ReLU
    - Task-specific: hidden + output layers

Methodology Role:
    Evaluates performance trade-offs in multi-task vs. continual task learning.
    Forms a point of comparison against progressive/DEN models with task isolation.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import torch
import torch.nn as nn
import logging
from typing import Optional, List, Union
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DNN_MTL(nn.Module):
    """
    Multi-task learning model with a shared feature extractor and dynamic per-task heads.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: Union[int, List[int]],
        shared_hidden: Optional[int] = None,
        task_hidden: Optional[int] = None,
        *,
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # Default hidden sizes
        if shared_hidden is None:
            shared_hidden = 4 * input_dim
        if task_hidden is None:
            task_hidden = 4 * input_dim

        # Normalize output_dims to a list
        if isinstance(output_dims, int):
            output_dims = [output_dims]
        self.output_dims: List[int] = output_dims
        self.task_hidden = task_hidden

        logger.debug(
            f"Initializing DNN_MTL with input_dim={input_dim}, "
            f"shared_hidden={shared_hidden}, task_hidden={task_hidden}, output_dims={self.output_dims}"
        )

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.ReLU(),
        )

        # One head per task
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_hidden, task_hidden),
                nn.ReLU(),
                nn.Linear(task_hidden, od)
            )
            for od in self.output_dims
        ])

        # Bounded activation or identity
        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("Bounds or mask must be provided when use_bounds=True")
            if not (len(bounds_low) == len(bounds_high) == len(mask) == self.output_dims[0] if len(self.output_dims)==1 else all(d == len(bounds_low) for d in self.output_dims)):
                raise ValueError(
                    f"Mismatch: bounds/mask length must equal each output_dim, "
                    f"got bounds_low={len(bounds_low)}, bounds_high={len(bounds_high)}, mask={len(mask)}"
                )
            self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
        else:
            self.bound_layer = nn.Identity()

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """
        Forward pass through shared trunk and specified task head.

        Args:
            x (torch.Tensor): Input of shape (batch_size, input_dim).
            task_id (int): Index of the task head to use.

        Returns:
            torch.Tensor: Output of shape (batch_size, output_dims[task_id]).
        """
        shared_rep = self.shared(x)
        output = self.heads[task_id](shared_rep)
        output = self.bound_layer(output)
        return output

    def add_task_head(self, output_dim: int) -> None:
        """
        Dynamically add a new task-specific head.

        Args:
            output_dim (int): Number of outputs for the new head.
        """
        head = nn.Sequential(
            nn.Linear(self.shared[-2].out_features, self.task_hidden),
            nn.ReLU(),
            nn.Linear(self.task_hidden, output_dim)
        )
        self.heads.append(head)
        self.output_dims.append(output_dim)
        logger.debug(f"Added new task head with output_dim={output_dim}")

    def get_all_shared_weights(self) -> List[torch.Tensor]:
        """
        Get weights of all shared fully connected layers.

        Returns:
            List[torch.Tensor]: Weights of shared Linear layers.
        """
        weights = [layer.weight for layer in self.shared if isinstance(layer, nn.Linear)]
        logger.debug(f"Collected weights from {len(weights)} shared layers.")
        return weights

    def freeze_head(self, task_id: int) -> None:
        """Freeze parameters of the specified task-head."""
        for p in self.heads[task_id].parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, task_id: int) -> None:
        """Un-freeze parameters of the specified task-head."""
        for p in self.heads[task_id].parameters():
            p.requires_grad_(True)