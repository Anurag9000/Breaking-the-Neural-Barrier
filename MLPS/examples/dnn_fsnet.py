import torch
import torch.nn as nn
from Dyn_DNN4OPF.utils.fsnet_utils import hybrid_lbfgs_solve, lbfgs_solve

class FSNet(nn.Module):
    """
    Feasibility-Seeking Neural Network (FSNet) for OPF problems.
    
    Combines a multi-layer perceptron to predict an initial solution,
    followed by a hybrid L-BFGS solver for constraint feasibility refinement.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        fs_config: dict = None
    ):
        super(FSNet, self).__init__()
        # Build MLP backbone
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        # Output layer (Sigmoid to ensure outputs in [0,1])
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        # L-BFGS solver configuration (with sensible defaults)
        default_fs = {
            "max_diff_iter": 50,
            "max_iter": 100,
            "memory": 50,
            "scale": 10,
        }
        # Merge user-provided fs_config over defaults
        self.fs_config = {**default_fs, **(fs_config or {})}

    def forward(
        self,
        x: torch.Tensor,
        data: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through FSNet.

        Args:
            x: Tensor of shape (B, input_dim) -- problem parameters, scaled.
            data: Dict[str, Tensor] -- problem data (e.g., matrices, bounds).

        Returns:
            y_pred: Raw MLP prediction, shape (B, output_dim), in [0,1].
            y_refined: Feasibility-refined solution via fully differentiable L-BFGS.
        """
        y_pred = self.mlp(x)
        # 2) Rescale & clamp initial guess into true variable bounds
        lo = data.bounds_lo.to(x)
        hi = data.bounds_hi.to(x)
        y0 = y_pred * (hi - lo) + lo
        y0 = torch.clamp(y0, lo, hi)

        # 3) Hybrid L-BFGS refinement (truncated backprop, scaled objective)
        with torch.enable_grad():
            y_refined = hybrid_lbfgs_solve(
                x,
                y0,
                data,
                max_diff_iter=self.fs_config["max_diff_iter"],
                memory=self.fs_config["memory"],
                scale=self.fs_config["scale"]
            )

        return y_pred, y_refined