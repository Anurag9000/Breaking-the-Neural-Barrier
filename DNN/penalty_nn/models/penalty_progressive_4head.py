import math
import torch
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_progressive_4head import DNN_Progressive_4HEAD
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


class PenaltyDNNProgressive_4HEAD(DNN_Progressive_4HEAD):
    """
    Progressive DNN with physics-aware penalty (MSE + equality + inequality).
    Inherits shared trunk and multiple heads from DNN_Progressive.
    Heads (task_id): 0→Pg, 1→Qg, 2→Va, 3→Vm
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        bounds_low: torch.Tensor,
        bounds_high: torch.Tensor,
        mask: torch.Tensor,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        case_name: str,
        clip_test: bool = False,
    ):
        # penalty weights
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq   = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

        # initialize base Progressive network FIRST
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=mask,
        )

        # ------------------------------------------------------------------
        # Physics bounds as non-trainable buffers (GPU-first)
        # ------------------------------------------------------------------
        dev = bounds_low.device
        const = load_case_bounds(case_name)
        # generator limits (require presence)
        p_min = const["p_min"].to(dev)
        p_max = const["p_max"].to(dev)
        q_min = const["q_min"].to(dev)
        q_max = const["q_max"].to(dev)
        # bus limits
        v_min = const["v_min"].to(dev)
        v_max = const["v_max"].to(dev)
        # voltage angle bounds (±π if not provided)
        va_min = const.get("va_min", torch.full_like(v_min, -math.pi, device=dev))
        va_max = const.get("va_max", torch.full_like(v_min,  math.pi, device=dev))

        self.register_buffer("p_min",  p_min)
        self.register_buffer("p_max",  p_max)
        self.register_buffer("q_min",  q_min)
        self.register_buffer("q_max",  q_max)
        self.register_buffer("v_min",  v_min)
        self.register_buffer("v_max",  v_max)
        self.register_buffer("va_min", va_min)
        self.register_buffer("va_max", va_max)

        # apply test-time clipping to all head bound_layers
        for col in self.columns:
            col.bound_layer.apply_bounds.fill_(clip_test)

    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        task_id: int,
        *,
        metadata=None,
    ) -> torch.Tensor:
        """
        Composite loss: lambda_loss * MSE + lambda_eq * equality_norm + lambda_ineq * inequality_norm

        Equality term uses the full state [Pg,Qg,Va,Vm]; to avoid using stale/partial
        predictions we compute it on the Vm pass (task_id == 3).
        """
        # forward pass through selected head
        y_pred = self(x, task_id)

        # 1) MSE term
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # 2) equality residuals norm (only on Vm pass with full vector)
        if task_id == 3:
            # move any tensor metadata to the correct device
            if isinstance(metadata, dict):
                metadata = {
                    k: (v.to(y_pred.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in metadata.items()
                }
            with torch.no_grad():
                y_pg = self(x, 0)
                y_qg = self(x, 1)
                y_va = self(x, 2)
            # ordering: [Pg, Qg, Va, Vm]
            y_full = torch.cat([y_pg, y_qg, y_va, y_pred], dim=1)
            eq_out = power_balance_residuals(y_full, metadata or {})
            # support either a single tensor or (real, imag)
            if isinstance(eq_out, (tuple, list)) and len(eq_out) == 2:
                res_r, res_i = eq_out
                eq_norm = (res_r.pow(2) + res_i.pow(2)).mean().sqrt()
            else:
                eq_norm = eq_out.pow(2).mean().sqrt()
        else:
            eq_norm = torch.zeros((), device=y_pred.device)

        # 3) inequality violations norm (per-head bounds)
        if task_id == 0:   # Pg
            lower, upper = self.p_min, self.p_max
        elif task_id == 1: # Qg
            lower, upper = self.q_min, self.q_max
        elif task_id == 2: # Va
            lower, upper = self.va_min, self.va_max
        else:              # Vm
            lower, upper = self.v_min, self.v_max

        lo_v  = F.relu(lower - y_pred)
        hi_v  = F.relu(y_pred - upper)
        ineq  = torch.cat([lo_v, hi_v], dim=1)
        ineq_norm = ineq.pow(2).mean().sqrt()

        # weighted sum
        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )
