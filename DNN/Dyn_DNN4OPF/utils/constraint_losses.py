"""
constraint_losses.py
=====================

Implements AC-OPF constraint loss terms for integration into the Dyn_DNN4OPF pipeline.

Includes:
    - 7 Inequality Constraints:
        1. Voltage magnitude upper bound
        2. Voltage magnitude lower bound
        3. Generator real power upper bound
        4. Generator real power lower bound
        5. Generator reactive power upper bound
        6. Generator reactive power lower bound
        7. Thermal limit violations (|S_ij| ≤ rate_a)

    - 2 Equality Constraints:
        1. Real power balance (Σ PG - PD = Re{V * conj(YV)})
        2. Reactive power balance (Σ QG - QD = Im{V * conj(YV)})

Each constraint loss can be weighted by a corresponding lambda and included in total training loss.
"""

from typing import Tuple
import torch
from torch import Tensor
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, Union
import sys
from pathlib import Path
import json
import torch
from torch import Tensor
from typing import Tuple, Optional

# Add the project root directory (one level above Dyn_DNN4OPF) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

DEFAULT_LAMBDAS = {
    "lambda_vu": 1, "lambda_vl": 1,
    "lambda_pu": 1, "lambda_pl": 1,
    "lambda_qu": 1, "lambda_ql": 1,
    "lambda_th": 1,
    "lambda_real": 1, "lambda_imag": 1
}

# def voltage_upper_bound_loss(vm: Tensor, v_max: Tensor, lambda_vu: float = 1) -> Tensor:
#     return lambda_vu * F.relu(vm - v_max)

# def voltage_lower_bound_loss(vm: Tensor, v_min: Tensor, lambda_vl: float = 1) -> Tensor:
#     return lambda_vl * F.relu(v_min - vm)

# def generator_real_upper_bound_loss(pg: Tensor, p_max: Tensor, lambda_pu: float = 1) -> Tensor:
#     return lambda_pu * F.relu(pg - p_max)

# def generator_real_lower_bound_loss(pg: Tensor, p_min: Tensor, lambda_pl: float = 1) -> Tensor:
#     return lambda_pl * F.relu(p_min - pg)

# def generator_reactive_upper_bound_loss(qg: Tensor, q_max: Tensor, lambda_qu: float = 1) -> Tensor:
#     return lambda_qu * F.relu(qg - q_max)

# def generator_reactive_lower_bound_loss(qg: Tensor, q_min: Tensor, lambda_ql: float = 1) -> Tensor:
#     return lambda_ql * F.relu(q_min - qg)

def _scatter_to_bus(complex_power: Tensor, bus_idx: Tensor, n_bus: int) -> Tensor:
    """
    Scatters complex power values into their corresponding bus indices.
    
    Args:
        complex_power: Tensor of shape [N, L] where L is number of loads.
        bus_idx: LongTensor of shape [L] with bus indices for each load.
        n_bus: Total number of buses.

    Returns:
        Tensor of shape [N, n_bus] with complex power aggregated per bus.
    """
    N, L = complex_power.shape
    if L == n_bus:
        return complex_power.clone()

    assert L == len(bus_idx)
    out = torch.zeros((N, n_bus), dtype=torch.cfloat, device=complex_power.device)
    for i in range(L):
        out[:, bus_idx[i]] += complex_power[:, i]
    return out

def power_balance_residuals(
    pg: Tensor, qg: Tensor,
    pd: Tensor, qd: Tensor,
    vm: Tensor, va: Tensor,
    y_bus: Union[torch.sparse.Tensor, Tuple[Tensor,Tensor,Tensor,Tuple[int,int]]],
    gen_bus_idx: Tensor,
    load_bus_idx: Tensor,
    n_bus: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Same maths as equality_power_balance_loss, but returns the *per-bus*
    residual vectors (real & imag) **without** squaring / averaging.
    """
    # 1) Rebuild sparse
    if isinstance(y_bus, tuple):
        y_val, y_row, y_col, y_shape = y_bus
        idx = torch.stack([y_row, y_col], dim=0)
        y_bus = torch.sparse_coo_tensor(idx, y_val, size=y_shape)

    # 2) Move indices & Ybus to vm’s device
    device = vm.device
    if y_bus.device != device:         y_bus      = y_bus.to(device)
    if gen_bus_idx.device != device:   gen_bus_idx = gen_bus_idx.to(device)
    if load_bus_idx.device != device:  load_bus_idx= load_bus_idx.to(device)
    if pd.device != device: pd = pd.to(device)
    if qd.device != device: qd = qd.to(device)
    # 3) Build V, I, S always
    V = vm * torch.exp(1j * va)                     # (B, n_bus)
    Y = y_bus.to_dense() if y_bus.is_sparse else y_bus
    I = V @ Y.T                                     # (B, n_bus)
    S = V * torch.conj(I)                           # (B, n_bus)

    # 4) Number of buses
    bus_count = V.size(1) if n_bus is None else n_bus

    # 5) Scatter gen & load
    generation = _scatter_to_bus(pg + 1j*qg, gen_bus_idx, bus_count)
    load       = _scatter_to_bus(pd + 1j*qd, load_bus_idx,  bus_count)

    # 6) Residual per bus
    resid = generation - load - S

    return resid.real, resid.imag

def objective(pg: Tensor, cost_q: Tensor, cost_l: Tensor, cost_c: Tensor) -> Tensor:
    """
    Computes total generation cost using fixed cost coefficients across a batch.

    Args:
        pg (Tensor): Power generation for each generator, shape (N, G)
        cost_q (Tensor): Quadratic coefficients, shape (G,) — same for all samples
        cost_l (Tensor): Linear coefficients, shape (G,)
        cost_c (Tensor): Constant terms, shape (G,)

    Returns:
        Tensor: Shape (N, 1) total generation cost per sample
    """
    # 1) ensure all cost vectors live on the same device as pg
    device = pg.device
    cost_q = cost_q.to(device)
    cost_l = cost_l.to(device)
    cost_c = cost_c.to(device)

    # 2) expand them from (G,) → (N, G)
    cost_q = cost_q.unsqueeze(0).expand_as(pg)
    cost_l = cost_l.unsqueeze(0).expand_as(pg)
    cost_c = cost_c.unsqueeze(0).expand_as(pg)

    # 3) compute per-generator cost and sum
    individual_costs = cost_q * pg**2 + cost_l * pg + cost_c
    total_costs = individual_costs.sum(dim=1, keepdim=True)
    return total_costs

def gap_objective(objective: torch.Tensor, metadata_file: Union[str, Path]) -> torch.Tensor:
    """
    Computes the gap in the objective function.

    Args:
        objective (Tensor): Scalar tensor representing the objective function value.
        metadata_file (str or Path): Path to the JSON file containing metadata.

    Returns:
        Tensor: Scalar tensor representing the gap in the objective function (percentage).
    """
    # Load metadata from JSON file
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    with open(metadata_file, "r") as file:
        metadata_list = json.load(file)

    # Convert flattened metadata to a tensor
    metadata = torch.tensor(metadata_list, dtype=torch.float32)

    if metadata.device != objective.device:
        metadata = metadata.to(objective.device)

    # Compute the gap
    gap = abs((metadata - objective) / metadata)
    return gap * 100

def mean_constraint_violation(
    Y_pred: Tensor,
    res_real: Tensor,
    res_imag: Tensor,
    bounds: Dict[str, Tensor],
    num_gens: int,
    num_buses: int
    ) -> Tuple[float,float,float,float,float]:
    """
    Returns:
        mean_real: [N, 1] mean real power residual (ΔP) per sample
        mean_reac: [N, 1] mean reactive power residual (ΔQ) per sample
        mean_pg_viol: [N, 1] mean PG inequality violation per sample
        mean_qg_viol: [N, 1] mean QG inequality violation per sample
        mean_vm_viol: [N, 1] mean VM inequality violation per sample
    """
    N = Y_pred.shape[0]

    # --- Equality constraints ---
    mean_real = res_real.abs().mean(dim=1, keepdim=True)   # shape [N,1]
    mean_reac = res_imag.abs().mean(dim=1, keepdim=True)   # shape [N,1]

    # --- Inequality constraints ---
    pg = Y_pred[:, :num_gens]
    qg = Y_pred[:, num_gens:2*num_gens]
    vm = Y_pred[:, 2*num_gens + num_buses : 2*num_gens + 2*num_buses]

    def compute_violation(values: torch.Tensor,
                        lo:     torch.Tensor,
                        hi:     torch.Tensor) -> torch.Tensor:
        # keep every operand on the device where the predictions live
        lo, hi = lo.to(values.device), hi.to(values.device)
        return torch.clamp(values - hi, min=0) + torch.clamp(lo - values, min=0)

    pg_viol = compute_violation(pg, bounds["p_min"], bounds["p_max"])  # [N, G]
    qg_viol = compute_violation(qg, bounds["q_min"], bounds["q_max"])  # [N, G]
    vm_viol = compute_violation(vm, bounds["v_min"], bounds["v_max"])  # [N, B]

    # per‐sample mean violation over gens/buses
    mean_pg_viol = pg_viol.abs().mean(dim=1, keepdim=True)   # [N, 1]
    mean_qg_viol = qg_viol.abs().mean(dim=1, keepdim=True)   # [N, 1]
    mean_vm_viol = vm_viol.abs().mean(dim=1, keepdim=True)   # [N, 1]

    me_mean_real=mean_real.mean()
    me_mean_reac=mean_reac.mean()
    me_mean_pg_viol=mean_pg_viol.mean()
    me_mean_qg_viol=mean_qg_viol.mean()
    me_mean_vm_viol=mean_vm_viol.mean()

    return me_mean_real,me_mean_reac,me_mean_pg_viol,me_mean_qg_viol,me_mean_vm_viol

def inequality_residuals(g):
    return F.relu(g)           # max(0, g)

# constraint_losses.py

def compute_violation(g: torch.Tensor,
                      h: torch.Tensor,
                      mu: torch.Tensor,
                      rho: float) -> torch.Tensor:
    """
    Computes the maximum scalar constraint violation as in Eq. (8) of the PDL paper:
        v = max_{i ∈ batch} max( ||h_i||_∞, ||σ_i||_∞ ),
    where σ = max{ g, -µ / ρ } applies elementwise for the inequality residuals.

    Args:
        g   (Tensor[N, nineq]): inequality residuals g_x(y)
        h   (Tensor[N, neq])  : equality residuals h_x(y)
        mu  (Tensor[N, nineq]): current inequality dual estimates µ_k
        rho (float)           : penalty coefficient ρ

    Returns:
        Tensor: a scalar tensor equal to the maximum violation over the batch.
    """
    # σ_x,j(y) = max{ g_x,j(y), –µ_k,j / ρ }
    sigma = torch.max(g, -mu / rho)

    # per-sample ∞-norm of inequality and equality violations
    v_g = sigma.abs().max(dim=1)[0]  # shape: [N]
    v_h = h.abs().max(dim=1)[0]      # shape: [N]

    # per-sample violation, then maximum over all samples
    v_sample = torch.max(v_g, v_h)
    return v_sample.max()
