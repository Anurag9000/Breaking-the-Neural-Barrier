# utils/lbfgs_solver.py  (GPU-first version)
# --------------------------------------------------------------------------- #
# Differentiable and non-differentiable batch L-BFGS solvers for OPF
# All scalars (eps, step sizes, line-search constants, etc.) are tensors that
# live on the same CUDA device as the optimisation variables.
# --------------------------------------------------------------------------- #
from __future__ import annotations
from typing import Optional, Callable

import torch

# Single source of truth for device placement
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# JIT-compatible helpers (two-loop recursion & gamma)                         #
# --------------------------------------------------------------------------- #
@torch.jit.script
def _search_direction(
    g: torch.Tensor,           # (B, n)
    S: torch.Tensor,           # (m, B, n)
    Y: torch.Tensor,           # (m, B, n)
    gamma: torch.Tensor        # (B, 1)
) -> torch.Tensor:
    """
    Two-loop recursion: d_k = –H_k⁻¹ g_k   (batch version).
    All tensors already on CUDA; eps is also a CUDA tensor.
    """
    m = S.size(0)
    eps = g.new_tensor(1e-6)                       # 0-D CUDA tensor
    rho = 1.0 / ((S * Y).sum(-1, keepdim=True) + eps)  # (m, B, 1)

    q = g.clone()
    alphas: list[torch.Tensor] = []
    for i in range(m - 1, -1, -1):
        alpha_i = rho[i] * (S[i] * q).sum(-1, keepdim=True)  # (B,1)
        alphas.append(alpha_i)
        q = q - alpha_i * Y[i]

    r = gamma * q                                           # apply H₀
    alphas = alphas[::-1]
    for i in range(m):
        beta = rho[i] * (Y[i] * r).sum(-1, keepdim=True)
        r = r + S[i] * (alphas[i] - beta)

    return -r                                               # descent dir.


@torch.jit.script
def compute_gamma(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Initial Hessian scaling  γ_k = (sᵀy)/(yᵀy)   (batch version).
    """
    eps = S.new_tensor(1e-6)
    s_dot_y = (S[-1] * Y[-1]).sum(-1, keepdim=True)
    y_dot_y = (Y[-1] * Y[-1]).sum(-1, keepdim=True) + eps
    return s_dot_y / y_dot_y


# --------------------------------------------------------------------------- #
# Config object (kept as before, but values used as tensors when needed)      #
# --------------------------------------------------------------------------- #
class LBFGSConfig:
    """Hyper-parameters for L-BFGS."""
    def __init__(
        self,
        max_iter: int = 20,
        memory: int = 20,
        val_tol: float = 1e-6,
        grad_tol: float = 1e-6,
        scale: float = 10.0,
        c: float = 1e-4,
        rho_ls: float = 0.5,
        max_ls_iter: int = 10,
        verbose: bool = False,
    ):
        self.max_iter   = max_iter
        self.memory     = memory
        self.val_tol    = val_tol
        self.grad_tol   = grad_tol
        self.scale      = scale
        self.c          = c
        self.rho_ls     = rho_ls
        self.max_ls_iter = max_ls_iter
        self.verbose    = verbose


# --------------------------------------------------------------------------- #
# Objective factory (Eq. 5, FSNet paper)                                     #
# --------------------------------------------------------------------------- #
def _create_objective_function(
    x: torch.Tensor,
    data,
    scale: float | torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns  F(x,y) = f_OPF(y) + λ_F (‖h_x(y)‖² + ‖g⁺_x(y)‖²)
    All maths happen on the device of *x* / *y*.
    """
    λ_F = torch.as_tensor(scale, dtype=x.dtype, device=x.device)  # CUDA scalar

    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq_r, eq_i = data.eq_resid(x, y)              # (B, n_bus)
        ineq_raw   = data.ineq_resid(x, y)            # signed distances
        ineq_pos   = torch.clamp_min(ineq_raw, 0.0)

        eq_pen   = (eq_r**2 + eq_i**2).sum(-1)        # (B,)
        ineq_pen = (ineq_pos**2).sum(-1)              # (B,)
        cost     = data.obj_fn(y)                     # (B,)

        return cost.mean() + λ_F * (eq_pen + ineq_pen).mean()
    return _obj


# --------------------------------------------------------------------------- #
# Convergence check (tolerances as CUDA tensors)                              #
# --------------------------------------------------------------------------- #
def _check_convergence(
    f_val: torch.Tensor,
    g: torch.Tensor,
    config: LBFGSConfig,
) -> torch.Tensor:
    val_tol  = f_val.new_tensor(config.val_tol)
    grad_tol = g.new_tensor(config.grad_tol)

    val_ok  = f_val / config.scale < val_tol
    grad_ok = g.norm(dim=1) < grad_tol
    return val_ok | grad_ok


# --------------------------------------------------------------------------- #
# Back-tracking line search (scalars → CUDA)                                  #
# --------------------------------------------------------------------------- #
def _backtracking_line_search(
    y: torch.Tensor,
    d: torch.Tensor,
    g: torch.Tensor,
    f_val: torch.Tensor,
    obj_func: Callable[[torch.Tensor], torch.Tensor],
    config: LBFGSConfig,
) -> torch.Tensor:          # returns CUDA scalar tensor
    step     = y.new_tensor(1.0)                      # α₀
    c        = y.new_tensor(config.c)
    rho_ls   = y.new_tensor(config.rho_ls)
    dir_der  = (g * d).sum()

    with torch.no_grad():
        for _ in range(config.max_ls_iter):
            y_trial = y + step * d
            f_trial = obj_func(y_trial)
            if (f_trial <= f_val + c * step * dir_der).all():
                break
            step = step * rho_ls
    return step


# --------------------------------------------------------------------------- #
# Vectorised, differentiable L-BFGS                                           #
# --------------------------------------------------------------------------- #
def lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LBFGSConfig] = None,
    **kwargs,
) -> torch.Tensor:
    if config is None:
        config = LBFGSConfig(**kwargs)

    y        = y_init.clone().requires_grad_(True)
    B, n     = y.shape
    S_hist   = torch.zeros(config.memory, B, n, device=_DEVICE, dtype=y.dtype)
    Y_hist   = torch.zeros_like(S_hist)

    hist_len = 0
    hist_ptr = 0
    x        = x.to(_DEVICE)
    obj_func = _create_objective_function(x, data, config.scale)

    for k in range(config.max_iter):
        f_val = obj_func(y)
        g     = torch.autograd.grad(f_val, y, create_graph=True)[0]

        if _check_convergence(f_val, g, config).all():
            if config.verbose:
                print(f"[LBFGS] converged at iter {k}")
            break

        if hist_len > 0:
            idx   = (hist_ptr - hist_len + torch.arange(hist_len, device=_DEVICE)) % config.memory
            S     = S_hist[idx]
            Y     = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d     = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g                                  # steepest-descent

        step = _backtracking_line_search(y, d, g, f_val, obj_func, config)
        y_next            = y + step * d
        S_hist[hist_ptr]  = y_next - y
        Y_hist[hist_ptr]  = g.detach()  # y_next grad reused in next iter
        hist_ptr          = (hist_ptr + 1) % config.memory
        hist_len          = min(hist_len + 1, config.memory)
        y = y_next.detach().requires_grad_(True)

    return y


# --------------------------------------------------------------------------- #
# Non-differentiable variant (graph not kept)                                 #
# --------------------------------------------------------------------------- #
def nondiff_lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LBFGSConfig] = None,
    S_hist: Optional[torch.Tensor] = None,
    Y_hist: Optional[torch.Tensor] = None,
    hist_len: int = 0,
    hist_ptr: int = 0,
    **kwargs,
) -> torch.Tensor:
    if config is None:
        config = LBFGSConfig(**kwargs)

    y = y_init.clone().requires_grad_(True)
    B, n = y.shape
    if S_hist is None:
        S_hist = torch.zeros(config.memory, B, n, device=_DEVICE, dtype=y.dtype)
        Y_hist = torch.zeros_like(S_hist)

    x = x.to(_DEVICE)
    obj_func = _create_objective_function(x, data, config.scale)

    for k in range(config.max_iter):
        f_val = obj_func(y)
        g     = torch.autograd.grad(f_val, y, create_graph=True)[0]
        y.requires_grad_(False)

        if _check_convergence(f_val, g, config).all():
            if config.verbose:
                print(f"[LBFGS/nd] converged at iter {k}")
            break

        if hist_len > 0:
            idx   = (hist_ptr - hist_len + torch.arange(hist_len, device=_DEVICE)) % config.memory
            S     = S_hist[idx]
            Y     = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d     = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g

        step    = _backtracking_line_search(y, d, g, f_val, obj_func, config)
        y_next  = y + step * d
        y_next.requires_grad_(True)

        S_hist[hist_ptr] = y_next - y
        Y_hist[hist_ptr] = g.detach()
        hist_ptr         = (hist_ptr + 1) % config.memory
        hist_len         = min(hist_len + 1, config.memory)
        y = y_next

    return y


# --------------------------------------------------------------------------- #
# Hybrid variant (differentiable + truncated BPTT)                            #
# --------------------------------------------------------------------------- #
def hybrid_lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_diff_iter: int = 20,
    config: Optional[LBFGSConfig] = None,
    **kwargs,
) -> torch.Tensor:
    if config is None:
        config = LBFGSConfig(**kwargs)

    diff_conf = LBFGSConfig(
        max_iter=max_diff_iter,
        memory=config.memory,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        c=config.c,
        rho_ls=config.rho_ls,
        max_ls_iter=config.max_ls_iter,
        verbose=config.verbose,
    )

    y        = y_init.clone().requires_grad_(True)
    B, n     = y.shape
    S_hist   = torch.zeros(config.memory, B, n, device=_DEVICE, dtype=y.dtype)
    Y_hist   = torch.zeros_like(S_hist)

    hist_len = 0
    hist_ptr = 0
    x        = x.to(_DEVICE)
    obj_func = _create_objective_function(x, data, config.scale)

    for k in range(max_diff_iter):
        f_val = obj_func(y)
        g     = torch.autograd.grad(f_val, y, create_graph=True)[0]

        if _check_convergence(f_val, g, diff_conf).all():
            if config.verbose:
                print(f"[LBFGS/hybrid] diff-phase converged at iter {k}")
            return y

        if hist_len > 0:
            idx   = (hist_ptr - hist_len + torch.arange(hist_len, device=_DEVICE)) % config.memory
            S     = S_hist[idx]
            Y     = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d     = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g

        step              = _backtracking_line_search(y, d, g, f_val, obj_func, diff_conf)
        y_next            = y + step * d
        S_hist[hist_ptr]  = y_next - y
        Y_hist[hist_ptr]  = g.detach()
        hist_ptr          = (hist_ptr + 1) % config.memory
        hist_len          = min(hist_len + 1, config.memory)
        y = y_next.detach().requires_grad_(True)

    # Switch to non-differentiable tail -------------------------------------
    remain_conf = LBFGSConfig(
        max_iter=config.max_iter - max_diff_iter,
        memory=config.memory,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        c=config.c,
        rho_ls=config.rho_ls,
        max_ls_iter=config.max_ls_iter,
        verbose=config.verbose,
    )
    y_tail = nondiff_lbfgs_solve(
        x,
        y,
        data,
        remain_conf,
        S_hist=S_hist,
        Y_hist=Y_hist,
        hist_len=hist_len,
        hist_ptr=hist_ptr,
    )
    return y + (y_tail - y)                   # keep graph for diff phase only
