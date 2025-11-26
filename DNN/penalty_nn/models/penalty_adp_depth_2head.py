import copy
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case, CASE
from Dyn_DNN4OPF.models.dnn_den import power_balance_residuals
from Dyn_DNN4OPF.models.adp_depth_2head import ADPDepth_2Head, expand_depth, expand_width


def train_early_stop(
    model: ADPDepth_2Head,
    train_loader,
    val_loader,
    *,
    patience: int,
    delta: float,
    max_epochs: int,
) -> float:
    """
    Early-stopping on validation MSE, replacing MSE-loss with penalty loss_fn.
    Creates a fresh optimizer/scheduler each call (after any arch change) via config if present.
    Restores best-epoch weights before returning, and returns the best validation MSE.
    """
    device = next(model.parameters()).device
    opt, sch = torch.optim.Adam(model.parameters(), lr=model.lr), None
    try:
        # try scheduler if available
        from config import SCHEDULER_PARAMS
        from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
        opt, sch = get_optimizer_scheduler(model.parameters(), lr=model.lr, **SCHEDULER_PARAMS)
    except Exception:
        pass

    best_val = float("inf")
    counter = patience
    best_state = model.snapshot()

    while model.global_epoch < max_epochs and counter > 0:
        # Train
        model.global_epoch += 1
        model.train()
        for xb, yb, *_ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = model.loss_fn(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if sch:
            try:
                sch.step()
            except Exception:
                pass

        # Validate on pure MSE for patience
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb, *_ in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                preds = model(xb)
                tot += F.mse_loss(preds, yb, reduction='sum').item()
                n += yb.shape[0]
        val_mse = tot / max(1, n)

        if val_mse < best_val - delta:
            best_val = val_mse
            best_state = model.snapshot()
            counter = patience
        else:
            counter -= 1

    model._restore(best_state)
    return best_val


class PenaltyADPDepth_2Head(ADPDepth_2Head):
    """
    Depth-first ADP (2-head) with physics-informed penalty loss and
    *patience on expansion* for both depth (outer series) and width (inner series).

    Patience knobs (dict-style access with defaults):
        trials_depth = cfg.get('trials_depth', 5)
        trials_width = cfg.get('trials_width', 5)

    Acceptance rule:
        Accept an expansion iff best_val < pre_series_val - delta (delta None→0.0).
    """
    def __init__(self, cfg: SimpleNamespace):
        super().__init__(cfg)
        init_from_case(cfg.case_name)
        # bounds as float
        for name in ("p_min", "p_max", "q_min", "q_max", "v_min", "v_max"):
            buf = getattr(CASE, name)
            if not isinstance(buf, torch.Tensor):
                buf = torch.tensor(buf, dtype=torch.float32)
            self.register_buffer(name, buf)
        # indices as long
        for name in ("gen_bus_idx", "load_bus_idx"):
            buf = getattr(CASE, name)
            if not isinstance(buf, torch.Tensor):
                buf = torch.tensor(buf, dtype=torch.long)
            elif buf.dtype != torch.long:
                buf = buf.to(torch.long)
            self.register_buffer(name, buf)
        # y_bus preserve dtype (may be complex)
        yb = getattr(CASE, "y_bus")
        if not isinstance(yb, torch.Tensor):
            try:
                yb = torch.tensor(yb)
            except Exception:
                yb = torch.tensor(yb, dtype=torch.float32)
        self.register_buffer("y_bus", yb)
        # Penalty weights
        self.lambda_loss = getattr(cfg, "lambda_loss", 1.0)
        self.lambda_eq   = getattr(cfg, "lambda_eq",   1.0)
        self.lambda_ineq = getattr(cfg, "lambda_ineq", 1.0)

        # Patience knobs (prefer dict-like .get; fallback to attributes)
        try:
            self.trials_depth = cfg.get('trials_depth', 5)
        except AttributeError:
            self.trials_depth = getattr(cfg, 'trials_depth', 5) if hasattr(cfg, 'trials_depth') else 5
        try:
            self.trials_width = cfg.get('trials_width', 5)
        except AttributeError:
            self.trials_width = getattr(cfg, 'trials_width', 5) if hasattr(cfg, 'trials_width') else 5

        # Failure counters
        self._depth_failures = 0
        self._width_failures = 0

    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)
        y_pred = self(x)
        mse = F.mse_loss(y_pred, y_true, reduction="mean")
        pg, qg, pd, qd, vm, va = self._split_pred(x, y_pred)
        res_P, res_Q = power_balance_residuals(
            pg, qg, pd, qd, vm, va,
            y_bus=self.y_bus,
            gen_bus_idx=self.gen_bus_idx,
            load_bus_idx=self.load_bus_idx,
            n_bus=self.v_min.shape[-1]
        )
        eq_norm = (res_P.pow(2) + res_Q.pow(2)).mean()
        relu = torch.relu
        viol_p = relu(pg - self.p_max) + relu(self.p_min - pg)
        viol_q = relu(qg - self.q_max) + relu(self.q_min - qg)
        viol_v = relu(vm - self.v_max) + relu(self.v_min - vm)
        ineq_norm   = (viol_p.pow(2).mean() + viol_q.pow(2).mean() + viol_v.pow(2).mean()) / 3.0
        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints=None,
        *,
        max_epochs: int = 10000,
        delta: float | None = None
    ) -> float:
        """
        Depth-first expansion with *patience on expansion*:
        - Outer series: DEPTH (add +1 hidden layer) — uses trials_depth / _depth_failures.
        - Inner series: WIDTH (add +ex_k neurons)  — uses trials_width / _width_failures.

        Acceptance rule: accept iff new_best < pre_series_best - delta. (delta None→0.0)
        """
        # Threshold
        Δ = self.delta if delta is None else delta
        delta_thr = 0.0 if Δ is None else float(Δ)

        P = self.patience
        inc = self.ex_k

        # Initialize counters for this task
        self._depth_failures = 0
        self._width_failures = 0

        # Track globally accepted model
        accepted_model = self.snapshot()
        accepted_val = float('inf')

        stop_outer = False

        while not stop_outer and len(self.hidden_layers) < self.max_depth:
            # 1) Train current architecture to plateau (baseline before depth series)
            base_val = train_early_stop(
                self, train_loader, val_loader,
                patience=P, delta=delta_thr, max_epochs=max_epochs
            )
            accepted_model = self.snapshot()
            accepted_val = base_val

            # 2) DEPTH expansion series with patience
            preD_model = accepted_model
            preD_val = accepted_val
            self._depth_failures = 0

            while True:
                # Depth guard
                if len(self.hidden_layers) >= self.max_depth:
                    # Roll back to pre-series baseline (no acceptance in this series)
                    self._restore(preD_model)
                    accepted_model = preD_model
                    accepted_val = preD_val
                    stop_outer = True
                    break

                # Expand depth by +1 and train to early stop
                expand_depth(self)
                d_val = train_early_stop(
                    self, train_loader, val_loader,
                    patience=P, delta=delta_thr, max_epochs=max_epochs
                )

                if d_val < preD_val - delta_thr:
                    # ACCEPT depth: reset counter and update series baseline
                    self._depth_failures = 0
                    preD_model = self.snapshot()
                    preD_val = d_val
                    accepted_model = self.snapshot()
                    accepted_val = d_val

                    # 3) WIDTH expansion series from this accepted depth
                    preW_model = self.snapshot()
                    preW_val = d_val
                    self._width_failures = 0

                    while True:
                        # Width capacity guard
                        total_neurons = (
                            sum(l.out_features for l in self.hidden_layers)
                            + sum(getattr(self, f"{br}_fc2").out_features for br in self.branches)
                        )
                        curr_width = self.hidden_layers[-1].out_features
                        if (
                            total_neurons + inc * (len(self.hidden_layers) + len(self.branches)) > self.max_neurons
                            or curr_width + inc > self.max_width
                        ):
                            # Roll back to pre-width baseline (if no acceptance happened)
                            self._restore(preW_model)
                            accepted_model = preW_model
                            accepted_val = preW_val
                            break

                        # Expand width and train to early stop
                        expand_width(self, inc)
                        w_val = train_early_stop(
                            self, train_loader, val_loader,
                            patience=P, delta=delta_thr, max_epochs=max_epochs
                        )

                        if w_val < preW_val - delta_thr:
                            # ACCEPT width: reset counter and update baseline, keep widening
                            self._width_failures = 0
                            preW_model = self.snapshot()
                            preW_val = w_val
                            accepted_model = self.snapshot()
                            accepted_val = w_val
                            continue
                        else:
                            self._width_failures += 1
                            if self._width_failures >= int(self.trials_width):
                                # Roll back to pre-width baseline
                                self._restore(preW_model)
                                accepted_model = preW_model
                                accepted_val = preW_val
                                break
                            else:
                                # Keep added capacity; try another widening
                                continue

                    # After finishing width series, continue depth series (may add more layers)
                    self._restore(preD_model)  # continue from last accepted depth
                    continue

                else:
                    # Depth FAIL
                    self._depth_failures += 1
                    if self._depth_failures >= int(self.trials_depth):
                        # Roll back to pre-depth-series baseline
                        self._restore(preD_model)
                        accepted_model = preD_model
                        accepted_val = preD_val
                        stop_outer = True
                        break
                    else:
                        # Keep added layer; try another layer in next loop
                        continue

            if stop_outer:
                break

            # Prepare for next outer cycle (if capacity remains)
            self._restore(accepted_model)
            # loop continues until guards/patience stop

        # Final restore
        self._restore(accepted_model)
        return accepted_val
