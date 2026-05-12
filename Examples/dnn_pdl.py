import torch
import torch.nn as nn
import torch.nn.init as init
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct


class PrimalNet(nn.Module):
    """
    Fully connected primal network predicting OPF decision variables y.
    Architecture: two hidden layers with ReLU, always followed by bounded activation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bounds_low: torch.Tensor,
        bounds_high: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        super().__init__()
        # Hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        # Output head
        self.head = nn.Linear(hidden_dim, output_dim)

        # Bounded activation: always enforce bounds
        if bounds_low is None or bounds_high is None:
            raise ValueError("bounds_low and bounds_high must be provided for PrimalNet")
        if mask is None:
            # default: no dimensions hard-clamped (allow PDL soft penalties to enforce bounds)
            mask = torch.zeros(output_dim, dtype=torch.bool, device=bounds_low.device)
        # Validate lengths
        if not (len(bounds_low) == len(bounds_high) == len(mask) == output_dim):
            raise ValueError(f"bounds_low, bounds_high, and mask must have length output_dim={output_dim}")
        # Wrap in BoundedAct (only dims with mask=True will be hard-clamped)
        self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
        self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        out = self.head(h2)
        return self.bound_layer(out)


class DualNet(nn.Module):
    """
    Dual network predicting Lagrange multipliers (μ for inequalities, λ for equalities).
    Shared hidden MLP followed by separate heads, zero-initialized to start at 0.
    """
    def __init__(
        self,
        input_dim: int,
        num_ineq: int,
        num_eq: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        # Build shared MLP
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for in_d, out_d in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)
        # Heads for μ and λ
        self.mu_head = nn.Linear(dims[-1], num_ineq)
        self.lam_head = nn.Linear(dims[-1], num_eq)

        # Zero-initialize heads to ensure Dφ(x)=0 at k=0
        init.zeros_(self.mu_head.weight)
        init.zeros_(self.mu_head.bias)
        init.zeros_(self.lam_head.weight)
        init.zeros_(self.lam_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        mu = torch.relu(self.mu_head(h))       # ensure μ ≥ 0
        lam = self.lam_head(h)
        return mu, lam


class DNNPDL(nn.Module):
    """
    Primal–Dual DNN model for OPF: outputs both decision variables y and dual multipliers.
    """
    def __init__(self, cfg):
        super().__init__()
        # Primal network: enforce bounds via cfg.model.bounds_* and mask
        self.primal = PrimalNet(
            input_dim  = cfg.model.input_dim,
            hidden_dim = cfg.model.hidden_primal,
            output_dim = cfg.model.output_dim,
            bounds_low = cfg.model.bounds_low,
            bounds_high= cfg.model.bounds_high,
            mask        = getattr(cfg.model, "mask", None),
        )
        # Dual network
        self.dual = DualNet(
            input_dim  = cfg.model.input_dim,
            num_ineq   = cfg.model.num_g,
            num_eq     = cfg.model.num_h,
            hidden_dims= cfg.model.hidden_dual,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_pred      = self.primal(x)
        mu_pred, lam_pred = self.dual(x)
        return y_pred, mu_pred, lam_pred
