import os
import logging
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case, CASE
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.models.adp_den_2head import ADP_DEN_2Head

# Configure module-level logger
debug_logger = logging.getLogger(__name__)
debug_logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Minimal resize helpers (match semantics used elsewhere in ADP code)
# ──────────────────────────────────────────────────────────────────────────────
def _resize_linear(old: torch.nn.Linear, new_out: int, new_in: int | None = None):
    """Return a new Linear(new_in, new_out) with overlapping weights copied."""
    if new_in is None:
        new_in = old.in_features
    new = torch.nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _resize_head(old: torch.nn.Linear, new_in: int):
    """Resize only the input dimension of the output head."""
    return _resize_linear(old, old.out_features, new_in)


class PenaltyADP_DEN_2Head(ADP_DEN_2Head):
    """
    Plateau ADP-DEN (2-head) with physics-informed penalty loss **and patience on width expansion**.

    Acceptance rule (unchanged): accept iff best_val_mse < pre_exp_val_mse - delta.
    `delta=None` is treated as 0.0.

    Patience on width:
      - Read `trials_width = cfg.get('trials_width', 5)`.
      - Maintain `_width_failures` counter.
      - On failure within patience → keep capacity (no rollback) and try another widening.
      - If failures ≥ trials_width → rollback to the pre-series snapshot and stop expanding.
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__(cfg)

        # Patience knob (dict-style access with default) + counter
        try:
            self.trials_width = cfg.get('trials_width', 5)
        except AttributeError:
            self.trials_width = getattr(cfg, 'trials_width', 5) if hasattr(cfg, 'trials_width') else 5
        self._width_failures = 0

        # Register physics constraints as buffers (auto-moved to GPU)
        init_from_case(cfg.case_name)
        dev = self.device
        for name in ("p_min", "p_max", "q_min", "q_max", "v_min", "v_max",
                     "y_bus", "gen_bus_idx", "load_bus_idx"):
            buf = getattr(CASE, name)
            if not isinstance(buf, torch.Tensor):
                # index arrays must be long; others float
                if name in ("gen_bus_idx", "load_bus_idx"):
                    buf = torch.as_tensor(buf, dtype=torch.long, device=dev)
                else:
                    buf = torch.as_tensor(buf, dtype=torch.float32, device=dev)
            else:
                buf = buf.to(dev, non_blocking=True)
            self.register_buffer(name, buf)
        # ensure module + newly registered buffers are on the same device
        self.to(dev)

        # Penalty weights
        self.lambda_loss = getattr(cfg, "lambda_loss", 1.0)
        self.lambda_eq   = getattr(cfg, "lambda_eq",   1.0)
        self.lambda_ineq = getattr(cfg, "lambda_ineq", 1.0)

    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor, *, metadata=None) -> torch.Tensor:
        """
        Compute composite loss: MSE + equality residuals + inequality violations.
        """
        # move to device once
        x = x.to(self.device, non_blocking=True)
        y_true = y_true.to(self.device, non_blocking=True)

        # Data term
        y_pred = self(x)
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # Split predictions and inputs
        ng = self.p_min.shape[-1]
        nb = self.v_min.shape[-1]
        pg = y_pred[:, : ng]
        qg = y_pred[:, ng : 2 * ng]
        va = y_pred[:, 2 * ng : 2 * ng + nb]
        vm = y_pred[:, 2 * ng + nb : 2 * ng + 2 * nb]
        pd = x[:, : nb]
        qd = x[:, nb : 2 * nb]

        # Equality (power balance) residuals
        res_P, res_Q = power_balance_residuals(
            pg, qg, pd, qd, vm, va,
            y_bus=self.y_bus,
            gen_bus_idx=self.gen_bus_idx,
            load_bus_idx=self.load_bus_idx,
            n_bus=nb
        )
        eq_norm = (res_P.pow(2) + res_Q.pow(2)).mean()

        # Inequality violations (generator P limits as example)
        violation_p = torch.clamp_min(pg - self.p_max, 0) + torch.clamp_min(self.p_min - pg, 0)
        ineq_norm = violation_p.pow(2).mean()

        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helper to shrink architecture back to a previous width
    # ──────────────────────────────────────────────────────────────────────
    def _shrink_to(self, fc1_out: int, pq_fc2_out: int, vm_fc2_out: int) -> None:
        """Resize hidden layers (and heads' input dims) back to provided widths."""
        self.fc1    = _resize_linear(self.fc1,    fc1_out,    self.fc1.in_features)
        self.pq_fc2 = _resize_linear(self.pq_fc2, pq_fc2_out, self.pq_fc2.in_features)
        self.vm_fc2 = _resize_linear(self.vm_fc2, vm_fc2_out, self.vm_fc2.in_features)
        self.head_pq = _resize_head(self.head_pq, pq_fc2_out)
        self.head_vm = _resize_head(self.head_vm, vm_fc2_out)
        self.to(self.device)  # keep on same device

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None
    ) -> float:
        """
        Plateau-based training loop using composite loss_fn for backprop.
        Adds **patience on width expansion** compared to the base.

        Returns best validation MSE (for the last accepted architecture).
        """
        device        = self.device
        patience_orig = self.patience
        ex_k          = self.ex_k
        # Acceptance threshold
        Δ             = self.delta if delta is None else delta
        delta_val     = 0.0 if Δ is None else float(Δ)
        max_neurons   = self.config.max_neurons

        # Optimizer & scheduler for current architecture
        opt, sched = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        # Global epoch tracking
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Track last accepted baseline
        accepted_state   = self.snapshot()
        accepted_val_mse = float("inf")

        stop_all = False
        self._width_failures = 0  # reset at task start

        while not stop_all:
            # ───── inner early-stopping on validation MSE ─────
            best_state   = self.snapshot()
            best_val_mse = float("inf")
            patience     = patience_orig

            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_all = True
                    break

                # ---- Training epoch (penalty loss) ----
                self.train()
                for xb, yb, *_ in train_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    loss = self.loss_fn(xb, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    sched.step()
                except Exception:
                    pass

                # ---- Validation epoch (MSE) ----
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        val_loss += F.mse_loss(self(xb), yb).item()
                val_loss /= len(val_loader)

                debug_logger.info(
                    f"[Task {self.current_task}] epoch {self.global_epoch_count:4d}  val-MSE {val_loss:.6f}  patience {patience}"
                )

                if val_loss < best_val_mse - delta_val:
                    best_val_mse = val_loss
                    best_state   = self.snapshot()
                    patience     = patience_orig
                else:
                    patience -= 1

            if stop_all:
                self._restore(best_state)
                accepted_state = best_state
                accepted_val_mse = min(accepted_val_mse, best_val_mse)
                break

            # Update accepted baseline for this plateau
            self._restore(best_state)
            accepted_state   = best_state
            accepted_val_mse = best_val_mse

            # ───── Start a WIDTH expansion *series* with patience ─────
            pre_exp_state = accepted_state
            pre_exp_val   = accepted_val_mse
            pre_dims      = (self.fc1.out_features, self.pq_fc2.out_features, self.vm_fc2.out_features)
            self._width_failures = 0

            debug_logger.info(
                f"Starting width-expansion series from baseline val-MSE {pre_exp_val:.6f}; trials_width={int(self.trials_width)}."
            )

            self._restore(pre_exp_state)

            while True:
                # Capacity guards: block if current or next expansion exceeds limits
                current_fc1 = self.fc1.out_features
                current_pq  = self.pq_fc2.out_features
                current_vm  = self.vm_fc2.out_features
                if (
                    current_fc1 >= max_neurons
                    or current_pq >= max_neurons
                    or current_vm >= max_neurons
                    or current_fc1 + ex_k > max_neurons
                    or current_pq + ex_k > max_neurons
                    or current_vm + ex_k > max_neurons
                ):
                    debug_logger.info("Hit max_neurons guard; stopping width expansions.")
                    # rollback to series baseline if no acceptance happened
                    self._shrink_to(*pre_dims)
                    self._restore(pre_exp_state)
                    accepted_state   = pre_exp_state
                    accepted_val_mse = pre_exp_val
                    stop_all = True
                    break

                # 1) Expand width by +ex_k across fc1 and both branch fc2
                self.expand_layer(self.fc1, ex_k)
                self.expand_layer(self.pq_fc2, ex_k)
                self.expand_layer(self.vm_fc2, ex_k)
                # Resize heads to match widened fc2 inputs (safe even if unchanged)
                if self.head_pq.in_features != self.pq_fc2.out_features:
                    self.head_pq = _resize_head(self.head_pq, self.pq_fc2.out_features).to(device)
                if self.head_vm.in_features != self.vm_fc2.out_features:
                    self.head_vm = _resize_head(self.head_vm, self.vm_fc2.out_features).to(device)

                debug_logger.info("[WIDTH] Expanded fc1 + branch-fc2 by %d neurons.", ex_k)

                # 2) New optimizer/scheduler for the widened architecture
                opt, sched = get_optimizer_scheduler(
                    self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
                )

                # 3) Train widened model to early stop
                widen_best  = float("inf")
                widen_state = self.snapshot()
                patience = patience_orig

                while patience > 0 and not stop_all:
                    self.global_epoch_count += 1
                    if self.global_epoch_count >= max_epochs:
                        stop_all = True
                        break

                    # train
                    self.train()
                    for xb, yb, *_ in train_loader:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        loss = self.loss_fn(xb, yb)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    try:
                        sched.step()
                    except Exception:
                        pass

                    # validate
                    self.eval()
                    vtot = 0.0
                    with torch.no_grad():
                        for xb, yb, *_ in val_loader:
                            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                            vtot += F.mse_loss(self(xb), yb).item()
                    vtot /= len(val_loader)

                    debug_logger.info(
                        f"[Task {self.current_task}] (widened) epoch {self.global_epoch_count:4d}  val-MSE {vtot:.6f}  patience {patience}"
                    )

                    if vtot < widen_best - delta_val:
                        widen_best  = vtot
                        widen_state = self.snapshot()
                        patience    = patience_orig
                    else:
                        patience -= 1

                if stop_all:
                    # Choose better between widened and pre-series baseline
                    if widen_best < pre_exp_val - delta_val:
                        self._restore(widen_state)
                        accepted_state   = widen_state
                        accepted_val_mse = widen_best
                    else:
                        self._shrink_to(*pre_dims)
                        self._restore(pre_exp_state)
                        accepted_state   = pre_exp_state
                        accepted_val_mse = pre_exp_val
                    break

                # 4) Decide with *patience on expansion*
                if widen_best < pre_exp_val - delta_val:
                    debug_logger.info(
                        "Width expansion accepted; counter reset. Improved %.6f → %.6f (delta=%.6f)",
                        pre_exp_val, widen_best, delta_val,
                    )
                    self._restore(widen_state)
                    accepted_state   = widen_state
                    accepted_val_mse = widen_best
                    self._width_failures = 0
                    break  # leave width-series; continue outer loop to retrain at this width
                else:
                    self._width_failures += 1
                    if self._width_failures >= int(self.trials_width):
                        debug_logger.info(
                            "Width expansions without improvement reached %d; rolling back and stopping.",
                            int(self.trials_width),
                        )
                        # Roll back to pre-series baseline and stop expanding
                        self._shrink_to(*pre_dims)
                        self._restore(pre_exp_state)
                        accepted_state   = pre_exp_state
                        accepted_val_mse = pre_exp_val
                        stop_all = True
                        break
                    else:
                        debug_logger.info(
                            "No val improvement after width expansion (trial %d/%d); trying another expansion.",
                            self._width_failures, int(self.trials_width),
                        )
                        # Keep widened capacity/weights and immediately try another widening
                        self._restore(widen_state)
                        continue

            if stop_all:
                break

            # After an accepted width expansion, re-init optimizer/scheduler and continue
            opt, sched = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )
            # outer loop continues to next plateau cycle

        # Final restore
        self._restore(accepted_state)
        return accepted_val_mse
