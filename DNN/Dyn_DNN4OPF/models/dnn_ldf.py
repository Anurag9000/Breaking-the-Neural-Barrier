# dnn_ldf.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from Dyn_DNN4OPF.utils.constraint_losses import (
    power_balance_residuals,
    mean_constraint_violation
)
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DELTA = 1e-6

class LDF(nn.Module):
    """
    Lagrangian Dual Penalty Network with vanilla early stopping on pure-MSE,
    """
    def __init__(self, config):
        super().__init__()
        self.config            = config
        self.in_dim, self.h1_dim, self.h2_dim = config.dims
        self.n_classes         = config.n_classes
        self.n_bus = self.in_dim // 2
        self.n_gen = self.n_classes // 2 - self.n_bus
        self.lr                = config.lr
        self.epochs            = getattr(config, "epochs", 10000)
        self.kickin            = getattr(config, "kickin", 0)
        self.step_size         = getattr(config, "step_size", 1e-5)
        self.update_freq       = getattr(config, "update_freq", 500)
        self.divide_by_counter = getattr(config, "divide_by_counter", True)
        self.patience          = getattr(config, "patience", 100)
        # network layers
        self.fc1               = nn.Linear(self.in_dim, self.h1_dim)
        self.fc2               = nn.Linear(self.h1_dim, self.h2_dim)
        self.head              = nn.Linear(self.h2_dim, self.n_classes)
        self.delta             = getattr(config, "delta", 1e-6)
        # dual multipliers
        self.register_buffer("duals", torch.zeros(5))
        self.update_count      = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints,
        *,
        max_epochs: int = 10000,
        delta: float    = DELTA
    ) -> float:
        """
        Train with:
          • primal-dual updates unchanged,
          • pure-MSE validation logging,
          • vanilla early stopping on best val-MSE + delta + patience,
        """

        device        = next(self.parameters()).device
        delta = self.delta if delta is None else DELTA
        optimizer, scheduler = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        # vanilla early stopping setup
        patience = getattr(self, "patience", 100)
        best_val_mse = float("inf")
        counter = patience
        previous_best = None

        for epoch in range(max_epochs):

            # — primal-dual training step unchanged —  
            self.train()
            total_base_mse = 0.0
            for Xb, Yb, *_ in train_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                preds = self(Xb)

                # base MSE loss
                base_loss = F.mse_loss(preds, Yb)
                total_base_mse += base_loss.item()

                # constraint residuals
                pd, qd = Xb[:, :self.n_bus], Xb[:, self.n_bus:2*self.n_bus]
                pg = preds[:, :self.n_gen]
                qg = preds[:, self.n_gen:2*self.n_gen]
                va = preds[:, 2*self.n_gen:2*self.n_gen + self.n_bus]
                vm = preds[:, 2*self.n_gen + self.n_bus:2*(self.n_gen + self.n_bus)]
                res_P, res_Q = power_balance_residuals(
                    pg=pg, qg=qg, pd=pd, qd=qd, vm=vm, va=va,
                    y_bus=constraints["eq"]["y_bus"],
                    gen_bus_idx=constraints["eq"]["gen_bus_idx"],
                    load_bus_idx=constraints["eq"]["load_bus_idx"],
                    n_bus=self.n_bus
                )
                viol_P, viol_Q, viol_pg, viol_qg, viol_vm = mean_constraint_violation(
                    Y_pred=preds, res_real=res_P, res_imag=res_Q,
                    bounds=constraints["ineq"],
                    num_gens=self.n_gen, num_buses=self.n_bus
                )
                violation_vec = torch.tensor(
                    [viol_P, viol_Q, viol_pg, viol_qg, viol_vm],
                    device=device
                )
                penalty = (self.duals * violation_vec).sum()

                # backward & step
                (base_loss + penalty).backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

                # dual updates unchanged
                self.update_count += 1
                if self.update_count > self.kickin and self.update_count % self.update_freq == 0:
                    with torch.no_grad():
                        delta_dual = self.step_size * violation_vec
                        if self.divide_by_counter:
                            u = self.update_count // self.update_freq
                            if u > 0:
                                delta_dual /= u
                        self.duals = torch.clamp(self.duals + delta_dual, min=0)

            # — validation pass: pure-MSE only —  
            self.eval()
            with torch.no_grad():
                val_sum = 0.0
                for Xv, Yv, *_ in val_loader:
                    Xv, Yv = Xv.to(device), Yv.to(device)
                    val_sum += F.mse_loss(self(Xv), Yv).item()
                val_mse = val_sum / len(val_loader)

            logger.info(f"[LDF] Epoch {epoch} | Val MSE: {val_mse:.6f}")   # pure-MSE log

            # — vanilla early stopping —  
            if val_mse < best_val_mse - delta:
                best_val_mse = val_mse
                counter = patience
            else:
                counter -= 1
                logger.info(f"[LDF] No improvement (Δ<{delta}); counter → {counter}")
                if counter == 0:
                    logger.info(f"[LDF] Early stopping at epoch {epoch}")
                    break

        # — test evaluation: pure-MSE only —  
        self.eval()
        test_sum = 0.0
        with torch.no_grad():
            for Xt, Yt in test_loader:
                Xt, Yt = Xt.to(device), Yt.to(device)
                test_sum += F.mse_loss(self(Xt), Yt).item()
        test_mse = test_sum / len(test_loader)
        logger.info(f"[LDF] Final Test MSE: {test_mse:.6f}")

        return test_mse
