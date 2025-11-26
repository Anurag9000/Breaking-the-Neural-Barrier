import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional, Union, List, Dict, Any
import copy
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


DELTA = 0

def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    task_id: Optional[int] = None,
    label: str = None
) -> float:
    """
    Run **one** epoch and return average loss, logging per-batch MSE.

    Args:
        train: If True do a training epoch, otherwise evaluation.
        label: Optional tag like 'Test' to branch behavior.
    """
    total = 0.0
    model.train() if train else model.eval()
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=1e-3,
        **SCHEDULER_PARAMS
    )

    # Special test-only loop (no logging here)
    if label == 'Test':
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = (
                model(xb)
                if model.__class__.__name__ == "DNN_EWC"
                else model(xb, task_id)
                if task_id is not None
                else model(xb)
            )
            total += loss_fn(pred, yb).item()
        return total / len(loader)

    # Standard train/val loop, with per-batch MSE logging
    for batch_idx, (xb, yb, *_) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        # forward
        if model.__class__.__name__ == "DNN_EWC":
            pred = model(xb)
        elif task_id is not None:
            pred = model(xb, task_id)
        else:
            pred = model(xb)

        # loss
        loss = loss_fn(pred, yb)

        # log this batch’s MSE
        logger.debug(
            f"[{'Train' if train else 'Val '}] "
            f"Batch {batch_idx:03d} → MSE = {loss.item():.6f}"
        )

        # backward & step (if training)
        if train:
            assert optimizer is not None, "Optimizer required for train=True"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        total += loss.item()

    return total / len(loader)

def _to_device(model: nn.Module,
               device: Optional[torch.device] = None) -> torch.device:
    """Move model to `device` (CUDA if available by default) and return it."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return device

def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 10000,
    lr: float,
    patience: int = 100,
    save_path: str,
    max_epochs: int = 10000,
    optimizer_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Generic supervised-regression loop with:
      • pure-MSE logging only
      • vanilla early stopping
    """
    
    device = _to_device(model, device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(), lr=lr, **SCHEDULER_PARAMS, **(optimizer_kwargs or {})
    )
    loss_fn = nn.MSELoss()
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    depth = max(len(linear_layers) - 1, 0)              # exclude output layer
    width = linear_layers[0].out_features if depth > 0 else 0
    logs: List[Dict[str, Union[int, float, str]]] = []
    best_val = float("inf")
    counter = patience

    for epoch in range(max_epochs):

        # training epoch
        train_mse = _epoch_loss(model, train_loader, loss_fn, device,
                                train=True, optimizer=optimizer)
        scheduler.step()

        # validation epoch
        val_mse = _epoch_loss(model, val_loader, loss_fn, device, train=False)
        logger.info(
            f"Epoch {epoch:03d} | depth={depth} | width={width} | "
            f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}"
        )
        logs.append({"epoch": epoch, "train_loss": train_mse, "val_loss": val_mse})

        # vanilla early stopping
        if val_mse < best_val - DELTA:
            best_val = val_mse
            torch.save(model.state_dict(), save_path)
            counter = patience
        else:
            counter -= 1
            if counter == 0:
                logger.info(f"No improvement for {patience} epochs; stopping.")
                break

    return logs

def _average_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
    *,
    label: str,
    task_id: Optional[int] = None,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute mean loss of `model` over `loader` (in eval mode),
    using the provided loss_fn (default: MSE).
    """
    # move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    total = 0.0
    with torch.no_grad():
        for batch in loader:
            xb, yb, *rest = batch
            xb, yb = xb.to(device), yb.to(device)
            # forward pass
            pred = model(xb, task_id) if task_id is not None else model(xb)
            total += loss_fn(pred, yb).item()

    return total / len(loader)

def _train_model_mae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 10000,
    lr: float,
    patience: int = 100,
    save_path: str,
    max_epochs: int = 10000,
    optimizer_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Generic supervised-regression loop with:
      • pure-MSE logging only
      • vanilla early stopping
    """
    
    device = _to_device(model, device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(), lr=lr, **SCHEDULER_PARAMS, **(optimizer_kwargs or {})
    )
    loss_fn = nn.L1Loss()
    logs: List[Dict[str, Union[int, float, str]]] = []
    best_val = float("inf")
    counter = patience

    for epoch in range(max_epochs):

        # training epoch
        train_mse = _epoch_loss(model, train_loader, loss_fn, device,
                                train=True, optimizer=optimizer)
        scheduler.step()

        # validation epoch
        val_mse = _epoch_loss(model, val_loader, loss_fn, device, train=False)

        logs.append({"epoch": epoch, "train_loss": train_mse, "val_loss": val_mse})

        # vanilla early stopping
        if val_mse < best_val - DELTA:
            best_val = val_mse
            torch.save(model.state_dict(), save_path)
            counter = patience
        else:
            counter -= 1
            if counter == 0:
                logger.info(f"No improvement for {patience} epochs; stopping.")
                break

    return logs

def _train_epoch_progressive(
    model: nn.Module,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total = 0.0
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=1e-3,
        **SCHEDULER_PARAMS
    )

    for xb, yb,_ in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb, task_id)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)

def _validate_progressive(
    model: nn.Module,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    """Evaluate model on `loader` and return mean loss."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, task_id)
            total += loss_fn(pred, yb).item()
    return total / len(loader)

def _epoch_loss_fixed(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    train: bool,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    task_id: int,
) -> float:
    """Run one epoch for head *task_id*; keep all tensors on GPU."""
    total = 0.0
    model.train() if train else model.eval()

    for xb, yb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb, task_id)
        loss = loss_fn(pred, yb)
        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / len(loader)

def _train_one_progressive_task(
    model: nn.Module,
    task_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Train *one* fixed head (Pg/Qg/Va/Vm) with early stopping on val MSE."""
    device = _to_device(model)

    # freeze all params first, then unfreeze column + adapters of this head
    for col_id, col in enumerate(model.columns):
        req = col_id == task_id
        for p in col.parameters():
            p.requires_grad = req
    if task_id > 0:
        ad1, ad2 = model.adapters[task_id]
        for block in ad1 + ad2:
            for p in block.parameters():
                p.requires_grad = True
    # build optimiser over trainable params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = get_optimizer_scheduler(trainable, lr=cfg.get("lr", 1e-3), **SCHEDULER_PARAMS)
    loss_fn = nn.MSELoss()

    logs: List[Dict[str, Any]] = []
    patience = int(cfg.get("patience", 100))
    counter = patience
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.get("max_epochs", 10000)):
        tr_loss = _epoch_loss_fixed(model, train_loader, loss_fn, device, True,
                                    optimizer=optimizer, task_id=task_id)
        scheduler.step()
        val_loss = _epoch_loss_fixed(model, val_loader, loss_fn, device, False,
                                     task_id=task_id)
        logs.append({"epoch": epoch, "task": task_id,
                     "train_loss": tr_loss, "val_loss": val_loss})
        logger.info(f"[ProgTask {task_id}] Ep{epoch:03d} tr={tr_loss:.6f} val={val_loss:.6f} cnt={counter}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = patience
        else:
            counter -= 1
            if counter == 0:
                logger.info(f"Early stop head {task_id} at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return logs
