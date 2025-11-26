"""
================================================================================
Physics‑Aware Penalty DEN Model Definition
================================================================================

Extends the Dynamically Expandable Network (DEN) to include a composite physics‑aware
penalty loss:
    L = λ₁·MSE(y_pred, y_true)
      + λ₂·‖power_balance_residuals(y_pred)‖₂
      + λ₃·‖inequality_violations(y_pred)‖₂

All model parameters, buffers, and computations are pinned on GPU at construction,
avoiding any host↔device transfers during forward or loss computation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from types import SimpleNamespace
from Dyn_DNN4OPF.models.dnn_den import DEN
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.utils.logger_plotter import compute_inequality_violation_per_sample
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

__all__ = ["PenaltyDEN"]


class PenaltyDEN(DEN):
    """
    DEN with composite physics-aware penalty loss. Inherits all dynamic expansion,
    gating, and pruning behavior from the base DEN implementation.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object supplying dimensions, tolerances, and clipping flags.
    lambda_loss : float
        Weight λ₁ on the plain MSE term.
    lambda_eq : float
        Weight λ₂ on the equality (power‑balance) residual norm.
    lambda_ineq : float
        Weight λ₃ on the inequality violation norm.
    case_name : str | None
        OPF case identifier for loading physical bounds. Defaults to cfg.case_name.
    clip_test : bool
        Placeholder flag; clipping for DEN is applied externally via a separate BoundedAct layer.
    """

    def __init__(
        self,
        cfg: SimpleNamespace,
        *,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        case_name: str | None = None,
        clip_test: bool = False,
    ) -> None:
        # ── Initialize base DEN (builds layers, masks, timestamps) ─────────────
        super().__init__(cfg)

        # ── Capture penalty weights ──────────────────────────────────────────
        self.lambda_loss  = float(lambda_loss)
        self.lambda_eq    = float(lambda_eq)
        self.lambda_ineq  = float(lambda_ineq)

        # ── Determine device once (inherited DEN sets self.device) ────────────
        dev: torch.device = self.device
        # Move all submodules and buffers to device
        self.to(dev)

        # ── Load and register physical bounds as buffers ─────────────────────
        case = case_name or getattr(cfg, "case_name", None)
        const = load_case_bounds(case)
        # register each bound vector on GPU
        for key in ("p_min", "p_max", "q_min", "q_max", "v_min", "v_max"):
            buf = const[key].view(1, -1).to(dev, non_blocking=True)
            # persistent=False avoids saving these intermediate buffers in checkpoints
            self.register_buffer(key, buf, persistent=False)

        # ── Also keep bus indices and Y-bus on device for equality residuals ──
        # indices as long buffers
        self.register_buffer("gen_bus_idx", const["gen_buses"].to(dev, non_blocking=True, dtype=torch.long), persistent=False)
        self.register_buffer("load_bus_idx", const["load_buses"].to(dev, non_blocking=True, dtype=torch.long), persistent=False)
        # y_bus may be a sparse tensor or a (vals, rows, cols, shape) tuple
        yb = const.get("y_bus")
        if torch.is_tensor(yb):
            self.register_buffer("y_bus_sparse", yb.to(dev, non_blocking=True), persistent=False)
            self._ybus_kind = "sparse"
        else:
            vals, rows, cols, shape = yb
            self.register_buffer("ybus_vals", vals.to(dev, non_blocking=True), persistent=False)
            self.register_buffer("ybus_rows", rows.to(dev, non_blocking=True, dtype=torch.long), persistent=False)
            self.register_buffer("ybus_cols", cols.to(dev, non_blocking=True, dtype=torch.long), persistent=False)
            self._ybus_shape = shape
            self._ybus_kind  = "tuple"

        # ── DEN has no internal bound layer; external clipping remains separate ──
        # (clip_test provided here for API consistency)

    # ---- helpers --------------------------------------------------------------
    def _ybus(self):
        if getattr(self, "_ybus_kind", None) == "sparse":
            return self.y_bus_sparse
        return (self.ybus_vals, self.ybus_rows, self.ybus_cols, self._ybus_shape)

    def _split_pred(self, y: torch.Tensor):
        """Split DEN output [PG | QG | Va | Vm] using bound sizes."""
        ng = self.p_min.size(1)
        nb = self.v_min.size(1)
        pg = y[:, :ng]
        qg = y[:, ng: 2*ng]
        va = y[:, 2*ng: 2*ng + nb]
        vm = y[:, 2*ng + nb: 2*ng + 2*nb]
        return pg, qg, va, vm

    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        metadata: dict | None = None,
        y_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Composite penalty loss:
        L = λ_loss·MSE + λ_eq·||h||^2 + λ_ineq·||g⁺||^2

        y_pred passthrough avoids double forward when the caller already computed it.
        """
        x = x.to(self.device, non_blocking=True)
        y_true = y_true.to(self.device, non_blocking=True)
        if y_pred is None:
            y_pred = self(x)
        else:
            y_pred = y_pred.to(self.device, non_blocking=True)

        # split predictions
        # expected head ordering: [PG, QG, Va, Vm]
        n_gen = self.p_min.shape[1]  # buffers are [1, n]
        n_bus = self.v_min.shape[1]
        pg = y_pred[:, :n_gen]
        qg = y_pred[:, n_gen:2*n_gen]
        va = y_pred[:, 2*n_gen:2*n_gen + n_bus]
        vm = y_pred[:, 2*n_gen + n_bus:]

        # PD/QD: from metadata if present; else from features x = [PD|QD]
        if metadata is not None and "pd" in metadata and "qd" in metadata:
            pd = metadata["pd"].to(self.device, non_blocking=True)
            qd = metadata["qd"].to(self.device, non_blocking=True)
        else:
            pd = x[:, :n_bus]
            qd = x[:, n_bus:2*n_bus]

        # equality residuals
        dp, dq = power_balance_residuals(
            pg, qg, pd, qd, vm, va, self.y_bus, self.gen_bus_idx, self.load_bus_idx, n_bus=n_bus
        )
        eq_norm = (dp.pow(2) + dq.pow(2)).mean()

        # inequality hinge (on PG, QG, Vm; Va unbounded)
        def sq_hinge_below(z, lo):  # (lo - z)+
            return F.relu(lo - z).pow(2)
        def sq_hinge_above(z, hi):  # (z - hi)+
            return F.relu(z - hi).pow(2)

        ineq_pg = (sq_hinge_below(pg, self.p_min) + sq_hinge_above(pg, self.p_max)).mean()
        ineq_qg = (sq_hinge_below(qg, self.q_min) + sq_hinge_above(qg, self.q_max)).mean()
        ineq_vm = (sq_hinge_below(vm, self.v_min) + sq_hinge_above(vm, self.v_max)).mean()
        ineq_norm = ineq_pg + ineq_qg + ineq_vm

        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        return (self.lambda_loss * mse
                + self.lambda_eq   * eq_norm
                + self.lambda_ineq * ineq_norm)

    def _train_one_epoch_penalty(self: DEN, loader, optimizer) -> float:
        """
        Train over one epoch using self.loss_fn (MSE + physics penalties).
        """
        self.train()
        total_loss = 0.0
        for xb, yb, *_ in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            # composite physics-aware loss
            loss = self.loss_fn(xb, yb, metadata=getattr(self, "_den_constraints", None))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints: dict,
        *,
        max_epochs: int | None = None,
        delta: float | None    = None
    ) -> float:
        """
        Like DEN.fit_task but uses self.loss_fn; validates on *pure MSE* with
        element-normalized aggregation.
        """
        device = self.device
        self._den_constraints = constraints  # if you use it internally

        optim, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        max_epochs = max_epochs or int(self.max_epochs)
        min_delta  = float(delta) if delta is not None else float(self.loss_thr)
        best_val   = float("inf")
        patience_left = int(self.patience)
        best_state = None

        # optional warmup
        for _ in range(int(getattr(self, "warm_epochs", 0))):
            self._train_one_epoch_penalty(train_loader, optim)
            self.on_epoch_end()

        epoch = 0
        while epoch < max_epochs and patience_left > 0:
            # ---- train one epoch on penalty loss ----
            self._train_one_epoch_penalty(train_loader, optim)
            self.on_epoch_end()

            # ---- validate on *MSE* (element-normalized) ----
            self.eval()
            val_sum, n_val = 0.0, 0
            with torch.no_grad():
                for xb, yb, *_ in val_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    yp = self(xb)
                    val_sum += F.mse_loss(yp, yb, reduction="sum").item()
                    n_val   += yb.numel()
            val_mse = val_sum / max(1, n_val)

            if val_mse + min_delta < best_val:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                patience_left = int(self.patience)
            else:
                patience_left -= 1

            epoch += 1

        if best_state is not None:
            self.load_state_dict(best_state)
        return best_val
