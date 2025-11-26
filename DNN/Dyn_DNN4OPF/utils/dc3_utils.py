# ── Dyn_DNN4OPF/utils/dc3_utils.py ──────────────────────────────────────────
"""
Physics-aware utilities for DNN_DC3.

Exposes callbacks:
    • complete_partial
    • eq_resid / eq_grad
    • ineq_dist / ineq_grad / ineq_partial_grad

Accepts both “flat” and “nested” constraint dictionaries.
All numeric data live on the global CUDA device from the moment they are
constructed, so no extra .to() calls are needed during training.
"""
from __future__ import annotations
from typing import Callable, Tuple, Dict, Union, Optional

import torch
from torch import Tensor
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals

# ── single source of truth for device placement ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _tensorize(arr, *, dtype=None):
    """Convert *arr* to a CUDA tensor (no-op if it already is one)."""
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=dtype) if dtype else arr.to(device)
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _violation(v: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """Element-wise distance outside the closed interval [lo, hi] (CUDA)."""
    return torch.clamp(v - hi, min=0) + torch.clamp(lo - v, min=0)


def _move_ybus(y_obj):
    """
    Recursively move Y-bus to CUDA.

    * If given as a sparse tensor → .to(device)
    * If given as 4-tuple (values,row,col,shape) → convert each field
      (shape is turned into LongTensor[2] if it is a Python tuple)
    """
    if torch.is_tensor(y_obj):
        return y_obj.to(device)

    if isinstance(y_obj, (tuple, list)) and len(y_obj) == 4:
        vals, rows, cols, shp = y_obj
        vals = _tensorize(vals, dtype=torch.complex64)
        rows = _tensorize(rows, dtype=torch.long)
        cols = _tensorize(cols, dtype=torch.long)
        if not torch.is_tensor(shp):
            shp = torch.tensor(shp, dtype=torch.long, device=device)
        return (vals, rows, cols, shp)

    return y_obj  # should not happen


# --------------------------------------------------------------------------- #
# Main helper                                                                 #
# --------------------------------------------------------------------------- #
class DC3Helper:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        train_loader,
        valid_loader,
        test_loader,
        constraints: dict,
        xdim: int,
        ydim: int,
        nknowns: int,
        n_bus: int,
        n_gen: int,
        use_partial: bool,
        obj_fn: Callable[[Tensor], Tensor],
    ):
        self.obj_fn = obj_fn

        # 1) Accept both FLAT and NESTED constraint dicts -------------------
        if {"ineq", "eq"} <= constraints.keys():
            ineq = constraints["ineq"]
            eq   = constraints["eq"]
            flat = {**ineq, **eq}
        else:                               # already flat
            ineq = eq = flat = constraints

        # 2) Public attributes expected by DNN_DC3 --------------------------
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader

        self.constraints  = flat            # merged, CUDA-ready later
        self.xdim         = xdim
        self.ydim         = ydim
        self.nknowns      = nknowns
        self.n_bus        = n_bus
        self.n_gen        = n_gen
        self.neq          = 2 * n_bus
        self.use_partial  = use_partial

        # 3) Inequality bounds → CUDA tensors ------------------------------
        self.p_min = _tensorize(ineq["p_min"])
        self.p_max = _tensorize(ineq["p_max"])
        self.q_min = _tensorize(ineq["q_min"])
        self.q_max = _tensorize(ineq["q_max"])
        self.v_min = _tensorize(ineq["v_min"])
        self.v_max = _tensorize(ineq["v_max"])

        # 4) Equality-side tensors → CUDA ----------------------------------
        self.constraints["y_bus"]        = _move_ybus(eq["y_bus"])
        self.constraints["gen_bus_idx"]  = _tensorize(eq["gen_bus_idx"], dtype=torch.long)
        self.constraints["load_bus_idx"] = _tensorize(eq["load_bus_idx"], dtype=torch.long)
        if "s_max" in flat:  # thermal limits (optional)
            self.constraints["s_max"] = _tensorize(flat["s_max"])

    # --------------------------------------------------------------------- #
    # Callbacks                                                             #
    # --------------------------------------------------------------------- #
    def complete_partial(self, x: Tensor, z: Tensor) -> Tensor:
        """Re-assemble full y if the network predicts only the free block."""
        return (
            torch.cat([x[:, : self.nknowns], z], dim=1).to(device)
            if self.use_partial
            else z
        )

    # ------------------------------------------------------------------ #
    def complete_equalities(
        self,
        x: Tensor,
        z: Tensor,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> Tensor:
        """
        Newton iterations (Alg. 1, DC3 paper) to satisfy equality constraints.
        All maths on CUDA.
        """
        B = z.size(0)
        va = torch.zeros(B, self.n_bus, device=device)
        vm = torch.ones(B, self.n_bus, device=device)

        y = torch.cat([x[:, : self.nknowns], z, vm, va], dim=1).to(device)
        y.requires_grad_(True)

        step = torch.tensor(1e-1, dtype=torch.float32, device=device)
        tol_t = torch.tensor(tol, dtype=torch.float32, device=device)

        for _ in range(max_iter):
            h = self.eq_resid(x, y)              # (B, neq)
            if h.abs().max() < tol_t:
                break
            grad, = torch.autograd.grad((h ** 2).sum(dim=1).mean(), y, create_graph=False)
            y = (y - step * grad).clone().requires_grad_(True)

        return y.detach()

    # ------------------------------------------------------------------ #
    def eq_resid(self, x: Tensor, y: Tensor) -> Tensor:
        pd = x[:, : self.n_bus]
        qd = x[:, self.n_bus : 2 * self.n_bus]

        pg = y[:, : self.n_gen]
        qg = y[:, self.n_gen : 2 * self.n_gen]
        va = y[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = y[:, 2 * self.n_gen + self.n_bus :]

        res_P, res_Q = power_balance_residuals(
            pg,
            qg,
            pd,
            qd,
            vm,
            va,
            y_bus=self.constraints["y_bus"],
            gen_bus_idx=self.constraints["gen_bus_idx"],
            load_bus_idx=self.constraints["load_bus_idx"],
            n_bus=self.n_bus,
        )
        return torch.cat([res_P, res_Q], dim=1)

    # ------------------------------------------------------------------ #
    def eq_grad(self, x: Tensor, y: Tensor) -> Tensor:
        y_req = y.clone().detach().requires_grad_(True)
        loss   = (self.eq_resid(x, y_req) ** 2).sum(dim=1).mean()
        grad,  = torch.autograd.grad(loss, y_req, create_graph=False)
        return grad

    # ------------------------------------------------------------------ #
    def ineq_dist(self, x: Tensor, y: Tensor) -> Tensor:
        pg = y[:, : self.n_gen]
        qg = y[:, self.n_gen : 2 * self.n_gen]
        vm = y[:, 2 * self.n_gen + self.n_bus :]

        d_pg = _violation(pg, self.p_min, self.p_max)
        d_qg = _violation(qg, self.q_min, self.q_max)
        d_vm = _violation(vm, self.v_min, self.v_max)

        if "s_max" in self.constraints:                       # branch limits
            flows = self._branch_flows(y)                     # (B, E)
            s_max = self.constraints["s_max"]                 # CUDA tensor
            d_br  = torch.clamp(flows - s_max, min=0)
            return torch.cat([d_pg, d_qg, d_vm, d_br], dim=1)

        return torch.cat([d_pg, d_qg, d_vm], dim=1)

    # ------------------------------------------------------------------ #
    def _branch_flows(self, y: Tensor) -> Tensor:
        """Crude upper-bound on |S_ij|; ignores phase-shifts for speed."""
        pg = y[:, : self.n_gen]
        qg = y[:, self.n_gen : 2 * self.n_gen]
        va = y[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = y[:, 2 * self.n_gen + self.n_bus :]

        V = vm * torch.exp(1j * va)                     # (B, n_bus)
        Y = (
            self.constraints["y_bus"]
            if torch.is_tensor(self.constraints["y_bus"])
            else self.constraints["y_bus"][0]  # sparse tuple → values
        )
        Y = Y.to_dense() if Y.is_sparse else Y          # ensure dense
        S = V @ Y.T * torch.conj(V)                     # nodal injections
        return S.abs()                                  # crude ≥ line flows

    # ------------------------------------------------------------------ #
    def ineq_grad(self, x: Tensor, y: Tensor) -> Tensor:
        y_req = y.clone().detach().requires_grad_(True)
        loss   = (self.ineq_dist(x, y_req) ** 2).sum(dim=1).mean()
        grad,  = torch.autograd.grad(loss, y_req, create_graph=False)
        return grad

    def ineq_partial_grad(self, x: Tensor, y: Tensor) -> Tensor:
        g = self.ineq_grad(x, y)
        return g[:, self.nknowns :] if self.use_partial else g
