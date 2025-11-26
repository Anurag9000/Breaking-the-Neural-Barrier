"""
Global constraint helpers for PDLTrainer
========================================
Vectorised helpers that operate purely on y_pred + case constants.
All numeric data live on CUDA from the moment init_from_case(raw) is called.
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional, Tuple
import math
import torch
from torch import Tensor

from Dyn_DNN4OPF.utils.constraint_losses import (
    power_balance_residuals,
    objective as _obj,
)

# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASE: Optional[SimpleNamespace] = None
# --------------------------------------------------------------------------- #


def init_from_case(raw: dict) -> None:
    """
    Convert a bounds/metadata dict into GPU-resident tensors for fast reuse.
    Call **once** before training.
    """
    global CASE

    # helper: always return CUDA tensor
    def tensor(x, *, dtype=torch.float32):
        return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.tensor(
            x, dtype=dtype, device=device
        )

    n_gen, n_bus = len(raw["p_max"]), len(raw["v_max"])

    # --- move / convert everything ----------------------------------------
    y_bus_raw = raw["y_bus"]
    if torch.is_tensor(y_bus_raw):                     # sparse tensor
        y_bus_gpu = y_bus_raw.to(device)
    else:                                              # 4-tuple (val,row,col,shape)
        v, r, c, shp = y_bus_raw
        y_bus_gpu = (
            v.to(device),
            r.to(device),
            c.to(device),
            torch.as_tensor(shp, dtype=torch.long, device=device)
            if not torch.is_tensor(shp)
            else shp.to(device),
        )

    CASE = SimpleNamespace(
        # meta
        n_gen=n_gen,
        n_bus=n_bus,
        # inequality bounds
        v_max=tensor(raw["v_max"]).view(1, n_bus),
        v_min=tensor(raw["v_min"]).view(1, n_bus),
        p_max=tensor(raw["p_max"]).view(1, n_gen),
        p_min=tensor(raw["p_min"]).view(1, n_gen),
        q_max=tensor(raw["q_max"]).view(1, n_gen),
        q_min=tensor(raw["q_min"]).view(1, n_gen),
        # equality data
        pd=tensor(raw["pd"]).view(1, -1),
        qd=tensor(raw["qd"]).view(1, -1),
        y_bus=y_bus_gpu,  # all components already on CUDA
        gen_bus_idx=tensor(raw["gen_buses"], dtype=torch.long),
        load_bus_idx=tensor(raw["load_buses"], dtype=torch.long),
        # cost coeffs
        cost_q=tensor(raw["cost_q"]),
        cost_l=tensor(raw["cost_l"]),
        cost_c=tensor(raw["cost_c"]),
    )


# --------------------------------------------------------------------------- #
def _split_y(y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Assumes canonical order: [Pg | Qg | Va | Vm]."""
    assert CASE is not None, "call init_from_case(raw) first"
    n_g, n_b = CASE.n_gen, CASE.n_bus
    pg = y[:, :n_g]
    qg = y[:, n_g : 2 * n_g]
    vm = y[:, 2 * n_g : 2 * n_g + n_b]
    va = y[:, 2 * n_g + n_b :]
    return pg, qg, va, vm


# --------------------------------------------------------------------------- #
# Public helpers used by PDLTrainer
# --------------------------------------------------------------------------- #
def compute_g(y_pred: Tensor) -> Tensor:
    """Inequality residuals g ≥ 0 (violations)."""
    assert CASE is not None, "call init_from_case(raw) first"
    n_g, n_b = CASE.n_gen, CASE.n_bus
    dev      = y_pred.device

    lo = torch.cat(
        (
            CASE.p_min.squeeze(0).to(dev),
            CASE.q_min.squeeze(0).to(dev),
            torch.full((n_b,), -math.pi, device=dev),
            CASE.v_min.squeeze(0).to(dev),
        )
    )
    hi = torch.cat(
        (
            CASE.p_max.squeeze(0).to(dev),
            CASE.q_max.squeeze(0).to(dev),
            torch.full((n_b,), math.pi, device=dev),
            CASE.v_max.squeeze(0).to(dev),
        )
    )
    return torch.maximum(y_pred - hi, lo - y_pred)  # (B, 2(n_g+n_b))


def compute_h(y_pred: Tensor) -> Tensor:
    """Equality residuals (real|imag) per bus → shape (B, 2·n_bus)."""
    assert CASE is not None, "call init_from_case(raw) first"
    dev = y_pred.device

    pd, qd       = CASE.pd.to(dev), CASE.qd.to(dev)
    g_idx        = CASE.gen_bus_idx.to(dev)
    l_idx        = CASE.load_bus_idx.to(dev)
    pg, qg, va, vm = _split_y(y_pred)

    # rebuild Ybus tuple on this device
    y_bus = CASE.y_bus
    if torch.is_tensor(y_bus):               # sparse tensor
        y_bus_dev = y_bus.to(dev)
    else:                                    # 4-tuple
        v, r, c, shp = y_bus
        y_bus_dev = (v.to(dev), r.to(dev), c.to(dev), shp.to(dev))

    real, imag = power_balance_residuals(
        pg, qg, pd, qd, vm, va,
        y_bus=y_bus_dev,
        gen_bus_idx=g_idx,
        load_bus_idx=l_idx,
        n_bus=CASE.n_bus,
    )
    return torch.cat([real, imag], dim=1)


def objective(y_pred: Tensor) -> Tensor:
    """Quadratic generation cost (scalar per sample)."""
    assert CASE is not None, "call init_from_case(raw) first"
    pg, *_ = _split_y(y_pred)
    cq, cl, cc = CASE.cost_q.to(pg.device), CASE.cost_l.to(pg.device), CASE.cost_c.to(pg.device)
    return _obj(pg, cq, cl, cc).squeeze(1)
