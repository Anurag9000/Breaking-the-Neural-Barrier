import math
import torch
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_progressive_2head import DNN_Progressive2Head
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


class PenaltyDNNProgressive_2Head(DNN_Progressive2Head):
    """
    Progressive DNN with physics-aware penalty (MSE + equality + inequality).
    Inherits shared trunk and two heads from DNN_Progressive2Head:
      - task_id == 0 → Pg_Qg head
      - task_id == 1 → Va_Vm head
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

        # initialize base Progressive network FIRST (ensures nn.Module state)
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
        # generator limits (require presence; avoids silent wrong bounds)
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
        Two-head semantics:
          task_id == 0 → Pg_Qg head (inequality vs p/q limits)
          task_id == 1 → Va_Vm head (inequality vs va/vm limits)
        Equality term requires the *full* state [Pg,Qg,Va,Vm]; we compute it
        only when both heads are available (on the Va_Vm pass) to avoid
        using stale/partial predictions.
        """
        # forward pass through selected head (kept on-device)
        y_pred = self(x, task_id)

        # 1) MSE term
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # 2) equality residual norm (use full vector when we have both heads)
        if task_id == 1:  # Va_Vm pass → build full prediction [Pg,Qg,Va,Vm]
            # ensure any tensor metadata is on the same device (GPU-first)
            if isinstance(metadata, dict):
                metadata = {
                    k: (v.to(y_pred.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in metadata.items()
                }
            with torch.no_grad():
                y_pgqg = self(x, 0)  # current frozen/earlier head
            # NOTE: y_full ordering matches model heads: [Pg,Qg,Va,Vm]
            y_full = torch.cat([y_pgqg, y_pred], dim=1)
            eq_out = power_balance_residuals(y_full, metadata or {})
            # support either a single tensor or (real, imag)
            if isinstance(eq_out, (tuple, list)) and len(eq_out) == 2:
                res_real, res_imag = eq_out
                eq_norm = (res_real.pow(2) + res_imag.pow(2)).mean().sqrt()
            else:
                eq_norm = eq_out.pow(2).mean().sqrt()
        else:
            # defer equality penalty until Va_Vm head is evaluated
            eq_norm = torch.zeros((), device=y_pred.device)

        # 3) inequality violations norm – per-head bounds
        if task_id == 0:  # Pg_Qg
            lower = torch.cat([self.p_min, self.q_min])
            upper = torch.cat([self.p_max, self.q_max])
            lo_v  = F.relu(lower - y_pred)
            hi_v  = F.relu(y_pred - upper)
        else:  # Va_Vm (angles bounded by ±π, magnitudes by v_min/v_max)
            lower = torch.cat([self.va_min, self.v_min])
            upper = torch.cat([self.va_max, self.v_max])
            lo_v  = F.relu(lower - y_pred)
            hi_v  = F.relu(y_pred - upper)

        ineq = torch.cat([lo_v, hi_v], dim=1)
        ineq_norm = ineq.pow(2).mean().sqrt()

        # weighted sum
        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )
