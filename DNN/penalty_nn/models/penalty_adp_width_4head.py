import copy
import torch
import torch.nn.functional as F
import logging
from types import SimpleNamespace
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case, CASE
from Dyn_DNN4OPF.models.dnn_den import power_balance_residuals
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.models.adp_width import ADPWidth, expand_width, expand_depth

logger = logging.getLogger(__name__)


def train_early_stop(
    model: ADPWidth,
    train_loader,
    val_loader,
    *,
    patience: int,
    delta: float,
    max_epochs: int,
) -> float:
    """
    Early-stopping on validation MSE, using penalty loss_fn for training.
    Restores the model to the best epoch snapshot before returning.
    """
    device = next(model.parameters()).device
    opt, sch = get_optimizer_scheduler(model.parameters(), lr=model.lr, **SCHEDULER_PARAMS)
    best_val = float("inf")
    counter = patience
    best_state = copy.deepcopy(model)

    while model.global_epoch < max_epochs and counter > 0:
        # Train one epoch with penalty loss
        model.global_epoch += 1
        model.train()
        for xb, yb, *_ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = model.loss_fn(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        try:
            sch.step()
        except Exception:
            pass

        # Validate on pure MSE for early-stop
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb, *_ in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                preds = model(xb)
                tot += F.mse_loss(preds, yb, reduction="sum").item()
                n += yb.shape[0]
        val_mse = tot / max(1, n)

        if val_mse < best_val - delta:
            best_val = val_mse
            best_state = copy.deepcopy(model)
            counter = patience
        else:
            counter -= 1

    # restore best
    model._restore(best_state)
    return best_val


class PenaltyADPWidth_4Head(ADPWidth):
    """
    Width-first ADP with physics-informed penalty loss and *patience on expansion*
    for both width (outer loop) and depth (inner loop).
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__(cfg)

        # Register physics constraints as buffers (moved to GPU with .to(device))
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
        self.lambda_eq = getattr(cfg, "lambda_eq", 1.0)
        self.lambda_ineq = getattr(cfg, "lambda_ineq", 1.0)

        # Patience knobs (dict-style access with defaults; fall back if cfg lacks .get)
        try:
            self.trials_width = cfg.get("trials_width", 5)
        except AttributeError:
            self.trials_width = getattr(cfg, "trials_width", 5) if hasattr(cfg, "trials_width") else 5
        try:
            self.trials_depth = cfg.get("trials_depth", 5)
        except AttributeError:
            self.trials_depth = getattr(cfg, "trials_depth", 5) if hasattr(cfg, "trials_depth") else 5

        # Failure counters
        self._width_failures = 0
        self._depth_failures = 0

    def _split_pred(self, x: torch.Tensor, y_pred: torch.Tensor):
        """
        Return (pg, qg, pd, qd, vm, va) with outputs ordered [PG|QG|VA|VM]
        and inputs beginning [PD|QD|...].
        """
        nG, nB = self.n_gen, self.n_bus
        pg = y_pred[:, :nG]
        qg = y_pred[:, nG : 2 * nG]
        va = y_pred[:, 2 * nG : 2 * nG + nB]
        vm = y_pred[:, 2 * nG + nB : 2 * (nG + nB)]
        pd = x[:, :nB]
        qd = x[:, nB : 2 * nB]
        return pg, qg, pd, qd, vm, va

    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)
        y_pred = self(x)
        # 1) data term
        mse = F.mse_loss(y_pred, y_true, reduction="mean")
        # 2) equality
        pg, qg, pd, qd, vm, va = self._split_pred(x, y_pred)
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
            n_bus=self.v_min.shape[-1],
        )
        eq_norm = (res_P.pow(2) + res_Q.pow(2)).mean()
        # 3) inequality (two-sided P/Q/V)
        relu = torch.relu
        violation_p = relu(pg - self.p_max) + relu(self.p_min - pg)
        violation_q = relu(qg - self.q_max) + relu(self.q_min - qg)
        violation_v = relu(vm - self.v_max) + relu(self.v_min - vm)
        ineq_norm = (
            violation_p.pow(2).mean() + violation_q.pow(2).mean() + violation_v.pow(2).mean()
        ) / 3.0
        return self.lambda_loss * mse + self.lambda_eq * eq_norm + self.lambda_ineq * ineq_norm

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints=None,
        *,
        max_epochs: int = 10000,
        delta: float | None = None,
    ) -> float:
        """
        Dual-loop with *patience on expansion*:
        - Outer: width expansions by +ex_k neurons (uses trials_width / _width_failures)
        - Inner: depth expansions by +1 layer       (uses trials_depth / _depth_failures)

        Acceptance rule (unchanged): accept an expansion iff
            best_val < pre_exp_val - delta
        where delta=None is treated as 0.0.
        """
        # Acceptance threshold
        Δ = self.delta if delta is None else delta
        delta_thr = 0.0 if Δ is None else float(Δ)

        P = self.patience
        inc = self.ex_k

        # Track the globally accepted baseline across iterations
        accepted_model = copy.deepcopy(self)
        accepted_val = float("inf")

        # (Re)initialize counters at the start of the task
        self._width_failures = 0
        self._depth_failures = 0

        stop_outer = False

        while not stop_outer:
            # 1) Train current architecture to plateau (establish baseline for series)
            base_val = train_early_stop(
                self,
                train_loader,
                val_loader,
                patience=P,
                delta=delta_thr,
                max_epochs=max_epochs,
            )
            accepted_model = copy.deepcopy(self)
            accepted_val = base_val

            # 2) Start a WIDTH expansion series with patience
            preW_model = copy.deepcopy(accepted_model)
            preW_val = accepted_val
            self._width_failures = 0

            while True:  # width-series loop
                # Capacity guards (respect project limits exactly as existing code)
                total_neurons = (
                    sum(l.out_features for l in self.hidden_layers)
                    + sum(getattr(self, f"{br}_fc2").out_features for br in self.branches)
                )
                curr_width = self.hidden_layers[-1].out_features
                if (
                    total_neurons + inc * (len(self.hidden_layers) + len(self.branches)) > self.max_neurons
                    or curr_width + inc > self.max_width
                ):
                    logger.info("Hit max_neurons or max_width; stop width expansions.")
                    stop_outer = True
                    # Roll back to last accepted width baseline if we haven't accepted in this series
                    self._restore(preW_model)
                    accepted_model = copy.deepcopy(preW_model)
                    accepted_val = preW_val
                    break

                # Expand width by +ex_k and train to early stop
                expand_width(self, inc)
                logger.info("[WIDTH] Expanded by %+d neurons (fc1 & branch fc2).", inc)
                w_val = train_early_stop(
                    self,
                    train_loader,
                    val_loader,
                    patience=P,
                    delta=delta_thr,
                    max_epochs=max_epochs,
                )

                # Decide accept/fail with patience on expansion
                if w_val < preW_val - delta_thr:
                    # ACCEPT: keep, reset counter, refresh accepted baseline and exit width-series
                    logger.info(
                        "[WIDTH] Expansion accepted; counter reset. %.6f → %.6f (delta=%.6f)",
                        preW_val,
                        w_val,
                        delta_thr,
                    )
                    self._width_failures = 0
                    accepted_model = copy.deepcopy(self)
                    accepted_val = w_val
                    break  # go attempt DEPTH series from this accepted width
                else:
                    # FAIL: increment counter and either retry (no rollback) or rollback if exhausted
                    self._width_failures += 1
                    if self._width_failures >= int(self.trials_width):
                        logger.info(
                            "[WIDTH] Expansions without improvement reached %d; rolling back and stopping.",
                            int(self.trials_width),
                        )
                        # Rollback to pre-series baseline
                        self._restore(preW_model)
                        accepted_model = copy.deepcopy(preW_model)
                        accepted_val = preW_val
                        stop_outer = True  # stop further width growth
                        break
                    else:
                        logger.info(
                            "[WIDTH] No val improvement after expansion (trial %d/%d); trying another expansion.",
                            self._width_failures,
                            int(self.trials_width),
                        )
                        # Keep widened capacity and attempt *another* widening in the next loop
                        continue

            if stop_outer:
                break

            # 3) From accepted width, start a DEPTH expansion series with patience
            preD_model = copy.deepcopy(accepted_model)
            preD_val = accepted_val
            self._restore(accepted_model)  # ensure we start from the accepted width state
            self._depth_failures = 0

            while True:  # depth-series loop
                # Depth guard
                if len(self.hidden_layers) >= self.max_depth:
                    logger.info("Hit max_depth; stop depth expansions.")
                    # Roll back to last accepted depth baseline (pre-series)
                    self._restore(preD_model)
                    accepted_model = copy.deepcopy(preD_model)
                    accepted_val = preD_val
                    break

                # Expand depth by +1 hidden layer and train to early stop
                expand_depth(self)
                logger.info("[DEPTH] Added +1 hidden layer.")
                d_val = train_early_stop(
                    self,
                    train_loader,
                    val_loader,
                    patience=P,
                    delta=delta_thr,
                    max_epochs=max_epochs,
                )

                # Decide accept/fail with patience on expansion
                if d_val < preD_val - delta_thr:
                    logger.info(
                        "[DEPTH] Expansion accepted; counter reset. %.6f → %.6f (delta=%.6f)",
                        preD_val,
                        d_val,
                        delta_thr,
                    )
                    self._depth_failures = 0
                    preD_model = copy.deepcopy(self)  # update series baseline to the newly accepted depth
                    preD_val = d_val
                    accepted_model = copy.deepcopy(self)
                    accepted_val = d_val
                    # Continue depth-series to see if more layers help
                    continue
                else:
                    self._depth_failures += 1
                    if self._depth_failures >= int(self.trials_depth):
                        logger.info(
                            "[DEPTH] Expansions without improvement reached %d; rolling back and stopping.",
                            int(self.trials_depth),
                        )
                        # Roll back to the last accepted depth within this series
                        self._restore(preD_model)
                        accepted_model = copy.deepcopy(preD_model)
                        accepted_val = preD_val
                        break  # stop depth expansion series; return to outer loop for another width try
                    else:
                        logger.info(
                            "[DEPTH] No val improvement after expansion (trial %d/%d); trying another expansion.",
                            self._depth_failures,
                            int(self.trials_depth),
                        )
                        # Keep added capacity and try adding another layer
                        continue

            # after finishing depth series, loop back to another WIDTH series if capacity remains
            # (stop_outer may be set earlier by width guard/patience)
            self._restore(accepted_model)
            # continue to outer while not stop_outer

        # Final restore of globally accepted model
        self._restore(accepted_model)
        return accepted_val
