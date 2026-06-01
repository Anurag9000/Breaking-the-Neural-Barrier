import logging
from typing import Any, Optional
import torch
import torch.nn as nn
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DNN_DC3(nn.Module):
    """
    DC3 model: predicts partial variables z, completes equalities, 
    and applies gradient-based correction to enforce inequalities.
    """
    def __init__(
        self,
        data: Any,
        hidden_dim: Optional[int] = None,
        *,
        corr_steps_train: int = 50,
        corr_steps_test: int = 50,
        corr_lr: float = 1e-3,
        corr_eps: float = 1e-3,
        soft_weight: float = 1.0,
        soft_eq_frac: float,
        use_partial: bool = False,
        use_bounds: bool = True,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.data = data
        # Hyperparameters
        self.corr_steps_train = corr_steps_train
        self.corr_steps_test = corr_steps_test
        self.corr_lr = corr_lr
        self.corr_eps = corr_eps
        self.soft_weight = soft_weight
        self.soft_eq_frac = soft_eq_frac
        self.use_partial = use_partial

        # Dimensions: predict only free vars z
        in_dim = data.xdim
        # output dimension = ydim - knowns; if partial, also omit eq dims
        if use_partial:
            out_dim = data.ydim - data.nknowns
        else:
            out_dim = data.ydim
        if hidden_dim is None:
            hidden_dim = 4 * in_dim

        # Build MLP: two hidden layers
        layers = []
        for a,b in zip([in_dim, hidden_dim], [hidden_dim, hidden_dim]):
            layers += [nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        # Initialize
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        self.net = nn.Sequential(*layers).to(data.device)

        # Optional output bounds
        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("Bounds and mask must be provided when use_bounds=True")
            self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
        else:
            self.bound_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.use_partial:
            y = self.data.complete_partial(x, z)
        else:
            y = z
        y = self.bound_layer(y)
        return y

    def _grad_correct(self, x: torch.Tensor, y: torch.Tensor, train: bool) -> tuple[torch.Tensor,int]:
        """
        Perform DC3 correction steps on y to reduce gₓ(y) violations.
        If train=True, runs fixed corr_steps_train with gradient tracking.
        If train=False, runs until ≤ corr_eps or corr_steps_test, without grad.
        """
        steps = 0
        y_new = y
        lr = self.corr_lr
        eps = self.corr_eps
        max_steps = self.corr_steps_train if train else self.corr_steps_test
        momentum = 0.9
        old_step = 0
        context = torch.enable_grad()
        with context:
            while steps < max_steps and (
                train or (torch.max(self.data.ineq_dist(x,y_new)) > eps)):
                grad_y = self.data.ineq_grad(x, y_new)
                if self.use_partial:
                    # slice ∇g to the free-variable block  (z-space step)
                    start = self.data.nknowns + self.data.neq           # KNOWN | EQ |
                    grad_z = grad_y[:, start:]
                    z      = y_new[:, start:] - lr * grad_z
                    y_new = self.data.complete_equalities(x, z)

                else:
                    # full-space update
                    y_new = y_new - lr * grad_y

                # project back to physical bounds (Eq. 4)
                y_new = self.bound_layer(y_new)
                steps += 1
        return y_new, steps

def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    DC3 composite loss  f_OPF + λ·(‖h‖² + ‖g⁺‖²)  [Eq. (2)].
    `soft_eq_frac` splits λ between equality/inequality terms.
    """
    obj = self.data.obj_fn(y)                                  # f(y)

    h      = self.data.eq_resid(x, y)                          # shape [B, neq]
    g_pos  = torch.clamp(self.data.ineq_dist(x, y), min=0.0)   # g⁺

    eq_pen   = (h      ** 2).sum(dim=1)                        # ‖h‖²
    ineq_pen = (g_pos  ** 2).sum(dim=1)                        # ‖g⁺‖²

    loss = obj + self.soft_weight * (
        self.soft_eq_frac * eq_pen + (1.0 - self.soft_eq_frac) * ineq_pen
    )
    return loss
