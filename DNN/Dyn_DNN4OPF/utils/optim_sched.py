"""
Optimizer + Scheduler factory for Dyn_DNN4OPF.

Provides a uniform Adam optimizer together with a
CosineAnnealingWarmRestarts learning‐rate schedule.
"""

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Iterable, Tuple

def get_optimizer_scheduler(
    params: Iterable,
    lr: float,
    T_0: int,
    T_mult: int = 1,
    eta_min: float = 0.0
) -> Tuple[Adam, CosineAnnealingWarmRestarts]:
    """
    Instantiate Adam + CosineAnnealingWarmRestarts.

    Args:
        params: model.parameters() or any iterable of parameters.
        lr: initial learning rate for Adam.
        T_0: number of epochs (or iterations) before first restart.
        T_mult: factor to increase T_i after each restart.
        eta_min: minimum learning rate.

    Returns:
        optimizer: torch.optim.Adam
        scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """
    optimizer = Adam(params, lr=lr)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )
    return optimizer, scheduler
