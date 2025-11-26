import torch
from Dyn_DNN4OPF.models.dnn_fsnet import FSNet
from Dyn_DNN4OPF.utils.fsnet_utils import _create_objective_function

class PenaltyFSNet(FSNet):
    """Penalty version of FSNet with composite loss

    Loss:
        L = λ₁ · baseline_loss + λ₂ · ‖h‖₂ + λ₃ · ‖g⁺‖₂
    where
        • *baseline_loss* replicates the original FSNet objective
          (quadratic generator cost + projection‑gap),
        • *h* are AC power‑balance residuals (equalities),
        • *g⁺* are the positive parts of inequality residuals.
    ``baseline_loss`` is kept exactly as in the vanilla trainer so that
    diagnostics, logging, and early‑stopping heuristics remain compatible.
    """

    def __init__(
        self,
        *fsnet_args,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        **fsnet_kwargs,
    ) -> None:
        super().__init__(*fsnet_args, **fsnet_kwargs)
        self.lambda_loss = lambda_loss
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq

    # ------------------------------------------------------------------ #
    #  Custom composite loss ‑‑ trainer calls   model.loss_fn(x, y, data)
    # ------------------------------------------------------------------ #
    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor | None = None,
        data_dict=None,
    ) -> torch.Tensor:
        """Compute composite penalty loss.

        Args
        -----
        x : torch.Tensor
            Batch of problem parameters (B, input_dim).
        y_true : torch.Tensor | None
            Ground‑truth target (unused, kept for API compatibility).
        data_dict : SimpleNamespace
            Must expose ``eq_resid`` and ``ineq_resid`` as in the baseline
            run script.
        """
        device = x.device
        # 1) Forward pass --------------------------------------------------
        y_pred, y_refined = self.forward(x, data_dict)

        # 2) Baseline FSNet loss (quadratic cost + projection gap) ---------
        obj_fn = _create_objective_function(x, data_dict, scale=1.0)
        gap = torch.nn.functional.mse_loss(y_pred, y_refined, reduction="mean")
        baseline_loss = obj_fn(y_refined) + gap

        # 3) Equality penalty ‖h‖₂ ----------------------------------------
        eq_r, eq_i = data_dict.eq_resid(x, y_refined)              # (B, n_bus)
        eq_pen = torch.linalg.vector_norm(
            torch.stack([eq_r, eq_i], dim=0), ord=2, dim=(0, 2)
        ).mean()  # mean over batch

        # 4) Inequality penalty ‖g⁺‖₂ --------------------------------------
        ineq_raw = data_dict.ineq_resid(x, y_refined)              # signed dist.
        ineq_pos = torch.clamp_min(ineq_raw, 0.0)
        ineq_pen = torch.linalg.vector_norm(ineq_pos, ord=2, dim=1).mean()

        # 5) Composite loss -----------------------------------------------
        return (
            self.lambda_loss * baseline_loss
            + self.lambda_eq * eq_pen
            + self.lambda_ineq * ineq_pen
        )
