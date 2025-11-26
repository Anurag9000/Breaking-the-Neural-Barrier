"""
===============================================================================
EWC Utility Class: Fisher Estimation and Penalty Computation
===============================================================================

This module defines the core EWC class that encapsulates:
    - Fisher information matrix estimation (diagonal approximation)
    - Weight snapshotting
    - Computation of the EWC loss penalty for continual learning

--------------------------------------------------------------------------------
Core Objective:
    Enforce retention of important parameters learned in previous tasks by 
    penalizing deviation from stored parameter values using the Fisher matrix.

--------------------------------------------------------------------------------
Methodology Overview:
    - Fisher is estimated by squaring the gradients of the MSE loss per parameter.
    - A snapshot of parameters is stored immediately after each task.
    - A penalty term is added to future losses using:
        Σ_i 0.5 * F_i * (θ_i - θ_i*)²

--------------------------------------------------------------------------------
Key Methods:
    ▸ __init__(model, dataloader, device):
        - Stores snapshot of weights after task t
        - Calls `_compute_fisher()` to estimate Fisher matrix from dataloader

    ▸ _compute_fisher(dataloader):
        - Approximates diagonal Fisher by accumulating squared gradients per weight
        - Uses MSE loss over multiple minibatches for empirical average
        - Returns a dictionary of weight importance scores (F_i)

    ▸ penalty(model):
        - Computes weighted L2 penalty between current and stored parameters
        - Returns scalar penalty to be added to current task loss

--------------------------------------------------------------------------------
Interactions:
    ▸ Called by:
        - `trainer.py` in `train_one_task()` after each task is completed

    ▸ Stores:
        - Fisher matrix per task
        - Parameter snapshot per task

    ▸ Used during:
        - Subsequent task training via additive regularization term

--------------------------------------------------------------------------------
Mathematical Foundation:
    EWC loss is derived from Bayesian posterior approximation:
        log p(θ | D_A, D_B) ≈ log p(D_B | θ) + log p(θ | D_A)
    The second term is modeled via Laplace approximation centered at θ_A* with 
    diagonal precision given by the Fisher matrix.

--------------------------------------------------------------------------------
Relevant Paper:
    Kirkpatrick et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks.
    PNAS.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EWC:
    """
    Elastic Weight Consolidation (EWC) helper class to compute Fisher information matrix
    and calculate penalty to prevent catastrophic forgetting.
    """

    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device: Union[str, torch.device] = None, task_id: int = 0) -> None:
        """
        Initialize EWC with model parameters and compute Fisher information.

        Args:
            model (nn.Module): The neural network model.
            dataloader (torch.utils.data.DataLoader): DataLoader for computing Fisher information.
            device (str, optional): Device to perform computations on. Defaults to 'cpu'.
        """
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.task_id = task_id
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        logger.debug(f"Computing Fisher information matrix on device {device}")
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute the Fisher information matrix for all trainable parameters.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader to compute Fisher.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping parameter names to Fisher matrices.
        """
        epsilon = 1  # Fisher damping constant
        # initialize only for parameters that require grad
        fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.eval()
        sigma_sq = 1.0  # assumed variance for Gaussian likelihood
        total_samples = 0

        for xb, yb, _ in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.model.zero_grad()

            # route through correct forward signature
            if hasattr(self.model, "columns"):
                pred = self.model(xb, self.task_id)
            else:
                pred = self.model(xb)

            # compute per-sample loss
            loss = nn.functional.mse_loss(pred, yb, reduction='mean') / (2 * sigma_sq)

            # only differentiate w.r.t. the trainable params
            named_params = [
                (n, p) for n, p in self.model.named_parameters() if p.requires_grad
            ]
            params = [p for _, p in named_params]
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=False,
                create_graph=False
            )

            # accumulate squared gradients into fisher
            for (n, _), g in zip(named_params, grads):
                fisher[n] += g.detach().pow(2) * xb.size(0)

            total_samples += xb.size(0)

        # normalize and add damping
        for n in fisher:
            fisher[n] /= total_samples
            fisher[n] += epsilon

        logger.debug("Fisher information matrix computed.")
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty for the current model parameters.

        Args:
            model (nn.Module): The current model to penalize.

        Returns:
            torch.Tensor: Scalar tensor representing the EWC penalty.
        """
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += 0.5 * (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        logger.debug(f"EWC penalty computed: {loss.item() if loss.numel() == 1 else loss}")
        return loss

    def accumulate_fisher(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Add Fisher information from a new task to the current Fisher matrix.
        """
        new_fisher = self._compute_fisher(dataloader)
        for n in self.fisher:
            self.fisher[n] += new_fisher[n]
