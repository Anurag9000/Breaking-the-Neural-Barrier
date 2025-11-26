import os
import logging
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case, CASE
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.models.adp_den_expandtillplateuing import ADP_DEN

# Configure module-level logger
debug_logger = logging.getLogger(__name__)
debug_logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Minimal resize helpers to allow reverting a failed expansion (width shrink)
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


class PenaltyADP_DEN_4head(ADP_DEN):
    """
    Plateau ADP-DEN with physics-informed penalty loss **and patience on width expansion**.

    The expansion is accepted iff best_val_mse < pre_exp_val_mse - delta, where
    delta comes from config (or fit_task arg) and None is treated as 0.0.

    Patience (width-only here):
        trials_width = cfg.get('trials_width', 5)
        - Allow up to N consecutive failed width expansions before rolling back
          to the pre-series baseline and stopping expansion.
        - On a failure within patience: keep the added capacity (no rollback) and
          immediately try another width expansion (sometimes two+ increments are
          needed to escape a local optimum).
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
        for name in (
            "p_min",
            "p_max",
            "q_min",
            "q_max",
            "v_min",
            "v_max",
            "y_bus",
            "gen_bus_idx",
            "load_bus_idx",
        ):
            buf = getattr(CASE, name)
            if not isinstance(buf, torch.Tensor):
                # index arrays must be long; others float
                if name in ("gen_bus_idx", "load_bus_idx"):
                    buf = torch.tensor(buf, dtype=torch.long, device=dev)
                else:
                    buf = torch.tensor(buf, dtype=torch.float32, device=dev)
            else:
                buf = buf.to(dev, non_blocking=True)
            self.register_buffer(name, buf)

        # Penalty weights
        self.lambda_loss = getattr(cfg, "lambda_loss", 1.0)
        self.lambda_eq = getattr(cfg, "lambda_eq", 1.0)
        self.lambda_ineq = getattr(cfg, "lambda_ineq", 1.0)

    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor, *, metadata=None) -> torch.Tensor:
        """
        Composite loss: MSE + equality residuals + inequality violations.
        """
        # ensure inputs on the correct device
        x = x.to(self.device, non_blocking=True)
        y_true = y_true.to(self.device, non_blocking=True)

        # Data term
        y_pred = self(x)
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # Split predictions and inputs using bounds to infer sizes
        ng = self.p_min.shape[-1]
        nb = self.v_min.shape[-1]
        pg = y_pred[:, :ng]
        qg = y_pred[:, ng : 2 * ng]
        va = y_pred[:, 2 * ng : 2 * ng + nb]
        vm = y_pred[:, 2 * ng + nb : 2 * ng + 2 * nb]
        pd = x[:, :nb]
        qd = x[:, nb : 2 * nb]

        # Equality (power balance) residuals
        res_P, res_Q = power_balance_residuals(
            pg,
            qg,
            pd,
            qd,
            vm,
            va,
            y_bus=self.y_bus,
            gen_bus_idx=self.gen_bus_idx,
            load_bus_idx=self.load_bus_idx,
            n_bus=nb,
        )
        eq_norm = (res_P.pow(2) + res_Q.pow(2)).mean()

        # Inequality violations (generator P limits as example)
        violation_p = torch.clamp_min(pg - self.p_max, 0) + torch.clamp_min(self.p_min - pg, 0)
        ineq_norm = violation_p.pow(2).mean()

        return self.lambda_loss * mse + self.lambda_eq * eq_norm + self.lambda_ineq * ineq_norm

    # ───────────────────────── Series baseline helpers ─────────────────────────
    def _capture_widths(self):
        """Capture current widths for fc1 and each branch fc2 for later revert."""
        return (
            self.fc1.out_features,
            {br: getattr(self, f"{br}_fc2").out_features for br in self.branches},
        )

    def _shrink_to(self, widths_tuple) -> None:
        """
        Shrink layers back to `widths_tuple` = (fc1_out, {br: fc2_out})
        and adjust each head's input dim accordingly.
        """
        fc1_out, br_outs = widths_tuple
        device = self.device

        # fc1
        if self.fc1.out_features != fc1_out:
            self.fc1 = _resize_linear(self.fc1, fc1_out, self.fc1.in_features).to(device)

        # each branch fc2 and head
        for br, w in br_outs.items():
            fc2_name = f"{br}_fc2"
            head_name = f"head_{br}"
            fc2 = getattr(self, fc2_name)
            if fc2.out_features != w:
                setattr(self, fc2_name, _resize_linear(fc2, w, fc2.in_features).to(device))
            if hasattr(self, head_name):
                head = getattr(self, head_name)
                if head.in_features != w:
                    setattr(self, head_name, _resize_head(head, w).to(device))

        self.to(device)

    # ────────────────────────────────────────────────────────────────────────
    # Training with *patience on expansion* (width-only)
    # ────────────────────────────────────────────────────────────────────────
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None,
    ) -> float:
        """
        Plateau-based training loop using composite loss_fn for backprop.
        Adds *patience on expansion* for width increases.

        Returns best validation MSE (last accepted baseline).
        """
        device = self.device
        patience_orig = self.patience
        ex_k = self.ex_k
        # Acceptance threshold
        Δ = self.delta if delta is None else delta
        delta_val = 0.0 if Δ is None else float(Δ)
        max_neurons = self.config.max_neurons

        # Optimizer & scheduler for current architecture
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        # Global epoch tracking
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Failure counter (width) for this task
        self._width_failures = 0

        # Track accepted baseline
        accepted_state = self.snapshot()
        accepted_val_mse = float("inf")

        stop_all = False

        while not stop_all:
            # ───── inner early-stopping on current architecture ─────
            best_state = self.snapshot()
            best_val_mse = float("inf")
            patience = patience_orig

            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_all = True
                    break

                # Train epoch with penalty loss
                self.train()
                for xb, yb, *_ in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    loss = self.loss_fn(xb, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    sched.step()
                except Exception:
                    pass

                # Validate
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        val_loss += F.mse_loss(self(xb), yb).item()
                val_loss /= len(val_loader)

                debug_logger.info(
                    f"[Task {self.current_task}] epoch {self.global_epoch_count:4d}  val-MSE {val_loss:.6f}  patience {patience}"
                )

                if val_loss < best_val_mse - delta_val:
                    best_val_mse = val_loss
                    best_state = self.snapshot()
                    patience = patience_orig
                else:
                    patience -= 1

            if stop_all:
                # restore best and exit
                self._restore(best_state)
                accepted_state = best_state
                accepted_val_mse = min(accepted_val_mse, best_val_mse)
                break

            # Lock in accepted baseline for this cycle
            self._restore(best_state)
            accepted_state = best_state
            accepted_val_mse = best_val_mse

            # ───── Start a WIDTH expansion *series* with patience ─────
            pre_exp_state = accepted_state  # series baseline
            pre_exp_val = accepted_val_mse
            pre_sizes = self._capture_widths()
            self._width_failures = 0

            debug_logger.info(
                f"Starting width-expansion series from baseline val-MSE {pre_exp_val:.6f}; trials_width={int(self.trials_width)}."
            )

            # Start expansions from the accepted baseline
            self._restore(pre_exp_state)

            while True:
                # Capacity guards: current or next expansion would exceed limits
                current_fc1 = self.fc1.out_features
                current_fc2_max = max(getattr(self, f"{br}_fc2").out_features for br in self.branches)
                if (
                    current_fc1 >= max_neurons
                    or any(getattr(self, f"{br}_fc2").out_features >= max_neurons for br in self.branches)
                    or current_fc1 + ex_k > max_neurons
                    or any(getattr(self, f"{br}_fc2").out_features + ex_k > max_neurons for br in self.branches)
                ):
                    debug_logger.info("Hit max_neurons guard; stopping width expansions.")
                    # rollback to series baseline if no acceptance happened
                    self._shrink_to(pre_sizes)
                    self._restore(pre_exp_state)
                    accepted_state = pre_exp_state
                    accepted_val_mse = pre_exp_val
                    stop_all = True
                    break

                # 1) Expand width by +ex_k across fc1 and all branch fc2
                self.expand_layer(self.fc1, ex_k)
                for br in self.branches:
                    self.expand_layer(getattr(self, f"{br}_fc2"), ex_k)
                # Resize heads to match widened inputs (safe even if unchanged)
                for br in self.branches:
                    head_name = f"head_{br}"
                    if hasattr(self, head_name):
                        head = getattr(self, head_name)
                        fc2 = getattr(self, f"{br}_fc2")
                        if head.in_features != fc2.out_features:
                            setattr(self, head_name, _resize_head(head, fc2.out_features).to(device))

                debug_logger.info("[WIDTH] Expanded fc1 + branch-fc2 by %d neurons.", ex_k)

                # 2) New optimizer/scheduler for the widened architecture
                opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

                # 3) Train widened model to early stop
                widen_best = float("inf")
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
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
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
                            xb = xb.to(device, non_blocking=True)
                            yb = yb.to(device, non_blocking=True)
                            vtot += F.mse_loss(self(xb), yb).item()
                    vtot /= len(val_loader)

                    debug_logger.info(
                        f"[Task {self.current_task}] (widened) epoch {self.global_epoch_count:4d}  val-MSE {vtot:.6f}  patience {patience}"
                    )

                    if vtot < widen_best - delta_val:
                        widen_best = vtot
                        widen_state = self.snapshot()
                        patience = patience_orig
                    else:
                        patience -= 1

                if stop_all:
                    # Pick the better between widened best and pre-series baseline
                    if widen_best < pre_exp_val - delta_val:
                        self._restore(widen_state)
                        accepted_state = widen_state
                        accepted_val_mse = widen_best
                    else:
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_state)
                        accepted_state = pre_exp_state
                        accepted_val_mse = pre_exp_val
                    break

                # 4) Decide with *patience on expansion*
                if widen_best < pre_exp_val - delta_val:
                    debug_logger.info(
                        "Width expansion accepted; resetting failure counter. Improved %.6f → %.6f (delta=%.6f)",
                        pre_exp_val, widen_best, delta_val,
                    )
                    self._restore(widen_state)
                    accepted_state = widen_state
                    accepted_val_mse = widen_best
                    self._width_failures = 0
                    break  # leave width-series; continue outer loop to retrain at this width
                else:
                    self._width_failures += 1
                    if self._width_failures >= int(self.trials_width):
                        debug_logger.info(
                            f"Expansions without improvement reached {int(self.trials_width)}; rolling back and stopping."
                        )
                        # Roll back to pre-series baseline and stop expanding
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_state)
                        accepted_state = pre_exp_state
                        accepted_val_mse = pre_exp_val
                        stop_all = True
                        break
                    else:
                        debug_logger.info(
                            f"No val improvement after width expansion (trial {self._width_failures}/{int(self.trials_width)}); trying another expansion."
                        )
                        # Keep the current widened capacity; try another widening
                        self._restore(widen_state)
                        continue

            if stop_all:
                break

            # After an accepted width expansion, re-init optimizer/scheduler and continue
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
            # outer loop continues to next plateau cycle
            continue

        # Final restore
        self._restore(accepted_state)
        return accepted_val_mse
