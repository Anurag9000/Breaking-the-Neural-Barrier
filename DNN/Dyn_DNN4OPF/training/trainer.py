import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Union
from typing import Optional
from training.training_helpers import _average_loss, _train_model
from Dyn_DNN4OPF.models.dnn_den        import DEN
from Dyn_DNN4OPF.models.dnn_den_2head  import DEN_2Heads
from Dyn_DNN4OPF.models.dnn_den_4head  import DEN_4Heads
from types import SimpleNamespace
import inspect
import torch.nn.functional as F
from typing import Optional, Any, Tuple
import logging
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.training.ewc_utils import EWC
from pathlib import Path
from Dyn_DNN4OPF.training.training_helpers import (
    _to_device,
    _epoch_loss,
    _train_model,
    _average_loss,
    _train_one_progressive_task,
    _train_model_mae,
)
import torch.nn.functional as F
from Dyn_DNN4OPF.models.dnn_fsnet import FSNet
from Dyn_DNN4OPF.utils.fsnet_utils import _create_objective_function
import copy
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

logger = logging.getLogger(__name__)
if not logger.handlers:  # don't add duplicate handlers in notebooks/multiple imports
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


DELTA = 0

def _build_model(config_obj: SimpleNamespace, device: torch.device) -> nn.Module:
    """
    Small factory to build the correct model from config_obj.model.
    Falls back to single-head DEN if unspecified.
    """
    name = (getattr(config_obj, "model", "DEN") or "DEN").upper()
    if name in {"DEN-2HEAD", "DEN_2HEAD", "DEN2HEAD"}:
        # DEN_2Heads constructor uses explicit dims
        in_dim     = config_obj.dims[0]
        out_dim    = config_obj.n_classes
        hidden_dim = getattr(config_obj, "h1_dim", None)
        mask       = getattr(config_obj, "mask", None)
        m = DEN_2Heads(
            input_dim=in_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            use_bounds=False,     # keep hard bounds external (clip layer at eval)
            mask=mask,
        ).to(device)
        return m
    if name in {"DEN-4HEAD", "DEN_4HEAD", "DEN4HEAD"}:
        # DEN_4Heads takes a config namespace
        return DEN_4Heads(config_obj).to(device)
    # Default: single-head DEN
    return DEN(config_obj).to(device)

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10000,
    lr: float = 1e-3,
    patience: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = "best_model.pth",
    max_epochs: int = 10000
) -> List[Dict[str, Union[int, float]]]:
    """
    Plain MSE training with Adam (no regularisation), with:
      • pure-MSE logging only
      • vanilla early stopping
    """
    return _train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_path=save_path,
        max_epochs=max_epochs,
        optimizer_kwargs={},
        device=device,
    )

def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    label: str,
    task_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate `model` on `loader` with optional `task_id`.
    Backward-compatible for models with or without task_id/predict_all.
    Also syncs `model.current_task` buffer when present.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # If the model uses a current_task buffer for gating, set it
    if task_id is not None and hasattr(model, "current_task") and isinstance(getattr(model, "current_task"), torch.Tensor):
        try:
            model.current_task.data.fill_(int(task_id))
        except Exception:
            pass

    # # FSNet special case (kept as in repo)
    # if isinstance(model, FSNet):
    #     total_loss = 0.0
    #     total_count = 0
    #     with torch.no_grad():
    #         for xb, y_true in loader:
    #             xb, y_true = xb.to(device), y_true.to(device)
    #             # FSNet returns (raw, refined)
    #             _, y_refined = model(xb, model.data_dict)
    #             total_loss += F.mse_loss(y_refined, y_true, reduction="sum").item()
    #             total_count += y_true.numel()
    #     return total_loss / max(1, total_count)

    # General fallback
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # backward-compatible forward
            if task_id is not None:
                try:
                    pred = model(xb, task_id)
                except TypeError:
                    pred = model(xb)
            elif hasattr(model, "predict_all"):
                pred = model.predict_all(xb)
            else:
                pred = model(xb)

            total += F.mse_loss(pred, yb, reduction="sum").item()
            count += yb.numel()

    return total / max(1, count)

def train_den_tasks(
    config: Dict[str, Any],
    task_loaders: List[Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]],
    *,
    max_epochs: int = 10000,
    delta: float = DELTA
) -> nn.Module:
    """
    Orchestrate sequential training across tasks:
      1) Build the correct model class from config["model"].
      2) For each task: call the model's own fit_task(...).
      3) Evaluate on test set with explicit task_id.
      4) Early-stop across tasks if test MSE doesn't improve by `delta`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_obj = SimpleNamespace(**config)
    model = _build_model(config_obj, device)

    previous_best_test_mse: Optional[float] = None
    all_logs: List[Dict[str, Any]] = []

    for task_id, (train_dl, val_dl, test_dl, constraints) in enumerate(task_loaders, start=1):
        logger.info(f"--- Starting Task {task_id} ---")
        # Make gating explicit when present
        if hasattr(model, "current_task") and isinstance(getattr(model, "current_task"), torch.Tensor):
            model.current_task.data.fill_(int(task_id))

        # Call the model's fit_task with a signature-safe argument set
        sig = inspect.signature(model.fit_task)
        kwargs = dict(train_loader=train_dl, val_loader=val_dl, max_epochs=max_epochs, delta=delta)
        if "test_loader" in sig.parameters:  kwargs["test_loader"] = test_dl
        if "constraints" in sig.parameters:  kwargs["constraints"] = constraints

        best_val_mse = model.fit_task(**kwargs)
        logger.info(f"[Task {task_id}] Best validation MSE during training: {float(best_val_mse):.6f}")

        # Evaluate on test set; always pass task_id for multi-head/gated models
        test_mse = evaluate(
            model,
            test_dl,
            label="DEN_test",
            task_id=task_id,
            device=device,
        )
        logger.info(f"[Task {task_id}] Test MSE: {float(test_mse):.6f}")

        # Log architecture / metrics (use h1_dim/h2_dim if exposed)
        h1_dim = getattr(model, "h1_dim", getattr(model, "fc1").out_features if hasattr(model, "fc1") else None)
        h2_dim = getattr(model, "h2_dim", None)
        all_logs.append({
            "task":           int(task_id),
            "train_val_mse":  float(best_val_mse),
            "test_mse":       float(test_mse),
            "hidden1_dim":    int(h1_dim) if h1_dim is not None else None,
            "hidden2_dim":    int(h2_dim) if h2_dim is not None else None,
        })

        # Early-stop across tasks (vanilla)
        if previous_best_test_mse is None or (test_mse <= previous_best_test_mse - delta):
            previous_best_test_mse = float(test_mse)
            logger.info(f"Continuing; best cross-task test MSE now {previous_best_test_mse:.6f}")
        else:
            logger.info(
                f"No improvement on Task {task_id} "
                f"(test_mse={float(test_mse):.6f} ≥ {previous_best_test_mse:.6f}−{delta}); terminating."
            )
            break

    # Persist logs & generate plot (existing utilities)
    save_logs_to_csv(all_logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        config["log_file"].replace(".csv", "_plot.png"),
        test_plot_name="den_test_plot.png"
    )
    return model

def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

def _freeze_all_heads_except(model: nn.Module, task_id: int) -> None:
    """Freeze every head **except** the selected one; keep shared layers trainable.
    Works for models exposing `heads: ModuleDict|ModuleList` and `id2name` mapping.
    """
    if not hasattr(model, "heads"):
        return  # nothing to do

    # Resolve the current head name (ModuleDict) or index (ModuleList)
    current_key = None
    if hasattr(model, "id2name") and isinstance(model.id2name, dict):  # type: ignore[attr-defined]
        current_key = model.id2name[task_id]  # e.g., "pg", "qg", "va", "vm" or "pg_qg"

    if isinstance(model.heads, nn.ModuleDict):
        for name, head in model.heads.items():
            for p in head.parameters():
                p.requires_grad_(name == current_key)
    elif isinstance(model.heads, nn.ModuleList):
        for i, head in enumerate(model.heads):
            for p in head.parameters():
                p.requires_grad_(i == task_id)


def _unfreeze_shared_layers(model: nn.Module) -> None:
    for p in model.parameters():
        # if it belongs to shared parts, it will already be True; heads get handled above
        if not p.requires_grad:
            # do not force-enable parameters that belong to frozen heads
            pass

def _trainable_params(model: nn.Module):
    """Prefer model-provided selector (e.g., get_all_shared_weights) if present."""
    if hasattr(model, "get_all_shared_weights"):
        return model.get_all_shared_weights()
    return (p for p in model.parameters() if p.requires_grad)


def _unpack_batch(batch):
    """Supports (X, Y) or (X, Y, metadata[, ...])."""
    xb, yb = batch[:2]
    meta = batch[2] if len(batch) > 2 else None
    return xb, yb, meta

def current_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Normalize and return a torch.device object.

    Args:
        device: Can be None, a string like 'cuda' or 'cpu', or a torch.device object.

    Returns:
        torch.device: A valid device object.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device

def train_mae(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10000,
    lr: float = 1e-3,
    patience: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = "best_model_mae.pth",
    max_epochs: int = 10000
) -> List[Dict[str, Union[int, float]]]:
    """
    Same as `train`, but uses MAE (L1Loss) instead of MSE.
    """
    return _train_model_mae(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_path=save_path,
        max_epochs=max_epochs,
        device=device,
    )

def evaluate_mae(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    label: str,
    task_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Same as `evaluate`, but computes mean absolute error.
    """
    return _average_loss(model, loader,
                         label=label, task_id=task_id,
                         device=device, loss_fn=nn.L1Loss())

def train_one_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_id: int = 0,
    ewc_list: Optional[List[Any]] = None,
    lambda_ewc: float = 1000,
    epochs: int = 10000,
    lr: float = 1e-3,
    patience: int = 100,
    device: Optional[torch.device] = None,
    max_epochs: int = 10000
) -> Tuple[List[Dict[str, Union[int, float, str]]], float]:
    """
    Train a single fixed-head task with EWC.
    """
    # Move model to device once
    device = _to_device(model, device)

    # Optimizer & scheduler setup
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(), lr=lr, **SCHEDULER_PARAMS
    )
    loss_fn = nn.MSELoss()
    logs: List[Dict[str, Union[int, float, str]]] = []

    best_val_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    counter = patience

    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # <-- use task_id in forward
            base = loss_fn(model(xb, task_id), yb)
            reg  = sum(lambda_ewc * e.penalty(model) for e in (ewc_list or []))
            (base + reg).backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # compute *pure* MSE using task_id
        pure_train_mse = _epoch_loss(
            model, train_loader, nn.MSELoss(), device,
            train=False, task_id=task_id
        )
        pure_val_mse = _epoch_loss(
            model, val_loader, nn.MSELoss(), device,
            train=False, task_id=task_id
        )

        logs.append({
            "epoch": epoch,
            "train_loss": pure_train_mse,
            "val_loss":   pure_val_mse
        })
        logger.info(
            f"[EWC] Task {task_id} | Epoch {epoch:03d} | "
            f"train={pure_train_mse:.6f} | val={pure_val_mse:.6f} | "
            f"counter={counter}/{patience}"
        )

        # early stopping on pure_val_mse
        if pure_val_mse < best_val_mse - DELTA:
            best_val_mse = pure_val_mse
            best_state   = copy.deepcopy(model.state_dict())
            counter = patience
        else:
            counter -= 1
            if counter == 0:
                logger.info(f"No improvement for {patience} epochs; rolling back.")
                model.load_state_dict(best_state)
                break

    return logs, best_val_mse


def train_task_sequential(
    model: nn.Module,
    *,
    task_train_loaders: List[DataLoader],
    task_val_loaders:   List[DataLoader],
    epochs:     int                   = 10000,
    lr:         float                 = 1e-3,
    patience:   int                   = 100,
    lambda_ewc: float                 = 1e3,
    device:     Optional[torch.device] = None,
    log_file:   Optional[str]         = None,
    delta:      float                 = 1e-6,
    csv_writer: Optional[callable]    = None,  # e.g., Dyn_DNN4OPF.utils.logger_plotter.save_logs_to_csv
) -> List[Dict[str, float]]:
    """Generic **sequential** EWC trainer (works for 4‑head and 2‑head backbones).

    Args
    ----
    model: backbone exposing `forward(x, task_id)` and `heads`/`id2name`.
    task_train_loaders / task_val_loaders: one DataLoader per task head, **in order**.
    epochs, lr, patience: training hyper‑params with early stopping on val MSE.
    lambda_ewc: regularization weight for accumulated EWC penalties from past tasks.
    device: explicit device; if None, uses model's current device.
    log_file: unused here; pass `csv_writer` to emit CSV externally.
    delta: improvement margin for early stopping.
    csv_writer: optional callable: `csv_writer(logs, log_file)` to persist logs.
    """
    dev = device or _model_device(model)
    model.to(dev)

    num_tasks = len(task_train_loaders)
    logs: List[Dict[str, float]] = []
    ewc_history: List[EWC] = []

    for task_id in range(num_tasks):
        tr_loader = task_train_loaders[task_id]
        va_loader = task_val_loaders[task_id]

        # Freeze non‑current heads; keep current head + shared layers trainable
        _freeze_all_heads_except(model, task_id)
        _unfreeze_shared_layers(model)

        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

        best_val = float("inf")
        counter  = patience
        best_state = None

        for epoch in range(epochs):
            # --------------------- train --------------------- #
            model.train()
            train_loss_sum = 0.0
            train_mse_sum  = 0.0
            seen = 0
            for batch in tr_loader:
                xb, yb = batch[:2]
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                preds = model(xb, task_id)
                mse   = F.mse_loss(preds, yb, reduction="mean")
                # Accumulate EWC penalties from all previous tasks
                penalty = 0.0
                for ewc in ewc_history:
                    penalty = penalty + ewc.penalty(model)
                loss = mse + lambda_ewc * penalty
                loss.backward()
                optimizer.step()

                bs = yb.size(0)
                train_loss_sum += loss.item() * bs
                train_mse_sum  += mse.item() * bs
                seen += bs

            avg_train_mse  = train_mse_sum / max(seen, 1)

            # -------------------- validate ------------------- #
            model.eval()
            val_sum = 0.0
            val_seen = 0
            with torch.no_grad():
                for batch in va_loader:
                    xb, yb = batch[:2]
                    xb = xb.to(dev, non_blocking=True)
                    yb = yb.to(dev, non_blocking=True)
                    preds = model(xb, task_id)
                    val_sum += F.mse_loss(preds, yb, reduction="sum").item()
                    val_seen += yb.numel() if yb.ndim > 1 else yb.size(0)
            # keep per‑sample MSE consistent with training average
            avg_val_mse = val_sum / (val_seen / yb.shape[-1]) if yb.ndim > 1 else val_sum / max(val_seen, 1)

            logs.append({
                "task": task_id,
                "epoch": epoch,
                "train_loss": avg_train_mse,  # report pure MSE (matches your previous behavior)
                "val_loss":   avg_val_mse,
            })
            if csv_writer and log_file:
                try:
                    csv_writer(logs, log_file)
                except Exception:
                    pass

            # Early stopping on validation MSE
            if avg_val_mse < best_val - delta:
                best_val = avg_val_mse
                counter = patience
                best_state = copy.deepcopy(model.state_dict())
                logger.info(f"[EWC] Task{task_id} Ep{epoch:03d} best val={best_val:.6f}")
            else:
                counter -= 1
                if counter == 0:
                    logger.info(f"[EWC] Early stop Task{task_id} at epoch {epoch}")
                    break

        # restore best weights for this task
        if best_state is not None:
            model.load_state_dict(best_state)

        # snapshot Fisher for the completed task and append to history
        logger.info(f"[EWC] Snapshot Fisher for task {task_id}")
        ewc_history.append(EWC(model, tr_loader, device=dev, task_id=task_id))

        # finally, freeze the finished head (keeps shared layers trainable for next task)
        _freeze_all_heads_except(model, -1)  # freeze all heads; next loop will unfreeze current one

    return logs

def train_mtl(
    model: nn.Module,
    task_loaders: List[DataLoader],
    val_loaders:  List[DataLoader],
    epochs: int = 10000,
    lr: float = 1e-3,
    patience: int = 100,
    delta: float = DELTA,
    label: str = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    max_epochs: int = 10000
) -> List[Dict[str, float]]:
    """
    Jointly train shared trunk + all heads every epoch, with

    Args:
        delta: minimal improvement threshold.
        save_path: where to dump the best checkpoint (if provided).
    """
    # device setup
    device = current_device(device)
    model.to(device)

    # optimizer & scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=lr,
        **SCHEDULER_PARAMS
    )
    loss_fn = nn.MSELoss()

    best_val_loss: float = float("inf")
    counter: int = patience
    logs: List[Dict[str, float]] = []

    total_train_batches = sum(len(loader) for loader in task_loaders)
    total_val_batches   = sum(len(loader) for loader in val_loaders)

    for epoch in range(epochs):
        # --- training ---
        model.train()
        train_mse_accum = 0.0
        for task_id, loader in enumerate(task_loaders):
            for xb, yb, _ in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb, task_id)
                mse_loss = loss_fn(preds, yb)
                mse_loss.backward()
                train_mse_accum += mse_loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        avg_train_mse = train_mse_accum / total_train_batches

        # --- validation ---
        model.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            for task_id, loader in enumerate(val_loaders):
                for xb, yb, _ in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_mse_accum += loss_fn(model(xb, task_id), yb).item()
        avg_val_mse = val_mse_accum / total_val_batches

        # log pure MSE only
        logs.append({
            "epoch":      float(epoch),
            "train_loss": avg_train_mse,
            "val_loss":   avg_val_mse
        })
        logger.info(f"[MTL] Epoch {epoch:03d} | train_loss={avg_train_mse:.6f} | val_loss={avg_val_mse:.6f}")

        # vanilla early stopping
        if avg_val_mse < best_val_loss - delta:
            best_val_loss = avg_val_mse
            if save_path:
                torch.save(model.state_dict(), save_path)
                logger.info(f"[MTL] Saved best model (epoch {epoch}) to {save_path}")
            counter = patience
        else:
            counter -= 1
            logger.info(f"[MTL] No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"[MTL] Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

    return logs

def train_with_l2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = "best_model_l2.pth",
    max_epochs: int = 10000
) -> List[Dict[str, Union[int, float]]]:
    """
    Training with L2 (weight-decay) regularisation, with:
      • pure-MSE logging only
      • vanilla early stopping
    """

    return _train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        max_epochs=max_epochs,
        lr=lr,
        patience=patience,
        save_path=save_path,
        optimizer_kwargs={"weight_decay": weight_decay},
        device=device,
    )

def train_with_l1(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int               = 10000,
    lr: float                 = 1e-3,
    l1_coeff: float           = 1e-4,
    patience: int             = 100,
    device: Optional[torch.device] = None,
    save_path: str            = "best_model_l1.pth",
    max_epochs: int           = 10000,
    delta: float              = DELTA
) -> List[Dict[str, float]]:
    """
    Train a model with explicit L1 regularization but:
      • log only pure MSE (no L1 terms)
      • vanilla early stopping on val MSE
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=lr,
        **SCHEDULER_PARAMS
    )
    loss_fn = nn.MSELoss()

    best_val_mse: float = float("inf")
    counter: int = patience
    logs: List[Dict[str, float]] = []

    for epoch in range(epochs):
        # —— training epoch: accumulate pure MSE ——  
        model.train()
        train_mse_accum = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            mse_loss = loss_fn(preds, yb)
            full_loss = mse_loss + l1_coeff * model.l1_penalty()
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()
            scheduler.step()
            train_mse_accum += mse_loss.item()
        avg_train_mse = train_mse_accum / len(train_loader)

        # —— validation epoch: pure MSE ——  
        model.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_mse_accum += loss_fn(model(xb), yb).item()
        avg_val_mse = val_mse_accum / len(val_loader)

        # log pure MSE only
        logs.append({"epoch": float(epoch), "train_loss": avg_train_mse, "val_loss": avg_val_mse})
        logger.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")

        # —— vanilla early stopping ——  
        if avg_val_mse < best_val_mse - delta:
            best_val_mse = avg_val_mse
            torch.save(model.state_dict(), save_path)
            logger.info(f"[L1] Saved best model (epoch {epoch}) to {save_path}")
            counter = patience
        else:
            counter -= 1
            logger.info(f"[L1] No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"[L1] Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

    return logs

def train_with_elastic(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int               = 10000,
    lr: float                 = 1e-3,
    lambda1: float            = 1.0,
    lambda2: float            = 1.0,
    patience: int             = 100,
    device: Optional[torch.device] = None,
    save_path: str            = "best_model_elastic.pth",
    max_epochs: int           = 10000,
    delta: float              = DELTA
) -> List[Dict[str, float]]:
    """
    Train with Elastic-Net but:
      • log only pure MSE (no L1/L2 terms)
      • vanilla early stopping on val MSE
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=lr,
        **SCHEDULER_PARAMS
    )
    loss_fn = nn.MSELoss()

    best_val_mse: float = float("inf")
    counter: int = patience
    logs: List[Dict[str, float]] = []

    for epoch in range(epochs):
        # —— training epoch: accumulate pure MSE ——  
        model.train()
        train_mse_accum = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            mse_loss = loss_fn(preds, yb)
            full_loss = mse_loss + lambda1 * model.l1_penalty() + lambda2 * model.l2_penalty()
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()
            scheduler.step()
            train_mse_accum += mse_loss.item()
        avg_train_mse = train_mse_accum / len(train_loader)

        # —— validation epoch: pure MSE ——  
        model.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_mse_accum += loss_fn(model(xb), yb).item()
        avg_val_mse = val_mse_accum / len(val_loader)

        # log pure MSE only
        logs.append({"epoch": float(epoch), "train_loss": avg_train_mse, "val_loss": avg_val_mse})
        logger.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")

        # —— vanilla early stopping ——  
        if avg_val_mse < best_val_mse - delta:
            best_val_mse = avg_val_mse
            torch.save(model.state_dict(), save_path)
            logger.info(f"[Elastic] Saved best model (epoch {epoch}) to {save_path}")
            counter = patience
        else:
            counter -= 1
            logger.info(f"[Elastic] No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"[Elastic] Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

    return logs

def train_dc3(
    model: torch.nn.Module,
    data,
    args: dict,
    save_dir: str,
    *,
    max_epochs: int = 10000,
    delta: float = DELTA,
):
    """
    Train a DC-3 model with:
      • pure-MSE logging (model.loss mean)
      • vanilla early stopping on validation MSE
    """
    # — Device —
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.autograd.set_detect_anomaly(True)
    # — Early-stopping bookkeeping —
    patience      = args.get("patience", 100)
    counter       = patience
    best_val_mse  = None

    # — Hyper-params —
    epochs            = args["epochs"]        # user-cfg limit
    batch_size        = args["batch_size"]
    lr                = args["lr"]
    results_save_freq = args.get("resultsSaveFreq", 1)

    # — DataLoaders (already prepared by run_script) —
    train_loader = data.train_loader
    valid_loader = data.valid_loader

    # — Optimiser & scheduler —
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(), lr=lr, **SCHEDULER_PARAMS
    )

    # — Training loop —
    for epoch in range(epochs):        # respect cfg['epochs']
        # —— train phase ——
        model.train()
        train_losses = []
        for X, *_ in train_loader:                      # unpack (X, Y, OBJ)
            X = X.to(device)
            y_tilde, _      = model(X), None
            y_corr_train, _ = model._grad_correct(X, y_tilde, train=True)
            loss = model.loss(X, y_corr_train).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_mse = sum(train_losses) / len(train_losses)
        logger.info(f"Epoch {epoch} — Avg-train loss: {avg_train_mse:.6f}")

        # —— validation + early-stopping ——
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for X, *_ in valid_loader:                  # unpack (X, Y, OBJ)
                X = X.to(device)
                y_tilde, _ = model(X), None
                y_corr, _  = model._grad_correct(X, y_tilde, train=False)
                valid_losses.append(model.loss(X, y_corr).mean().item())

        avg_valid_mse = sum(valid_losses) / len(valid_losses)
        logger.info(f"Epoch {epoch} — Validation loss: {avg_valid_mse:.6f}")

        # —— vanilla early stopping ——
        if best_val_mse is None or avg_valid_mse < best_val_mse - delta:
            best_val_mse = avg_valid_mse
            counter = patience
        else:
            counter -= 1
            logger.info(f"[DC3] No improvement (Δ < {delta}); counter → {counter}")
            if counter == 0:
                logger.info(
                    f"[DC3] Early stopping at epoch {epoch} "
                    f"after {patience} epochs without improvement."
                )
                break

        # —— checkpoint ——
        if epoch % results_save_freq == 0:
            ckpt = Path(save_dir) / f"dc3_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")

    # — Final save —
    final_ckpt = Path(save_dir) / "dc3_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    logger.info(f"Training complete — final model at {final_ckpt}")

def train_fsnet(
    model: nn.Module,
    train_loader: DataLoader,
    data_dict: dict,
    val_loader:   DataLoader,
    epochs:       int                  = 10000,
    lr:           float                = 1e-3,
    lambda_fs:    float                = 1,
    patience:     int                  = 100,
    device:       Optional[torch.device] = None,
    save_path:    Optional[str]        = "best_model_fsnet.pth",
    max_epochs:   int                  = 10000,
    delta:        float                = DELTA
) -> List[Dict[str, Union[int, float]]]:
    """
    Train an FSNet model with feature-selection regularization, now with:
      • pure-MSE (projection-gap) logging only
      • vanilla early stopping on gap MSE
    """
    # normalize device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=lr,
        **SCHEDULER_PARAMS
    )

    best_val_mse = float("inf")
    counter = patience
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(epochs):
        # —— training epoch: compute pure projection-gap MSE ——  
        model.train()
        train_gap_accum = 0.0
        for batch in train_loader:
            xb = batch[0].to(device)
            y_pred, y_refined = model(xb, data_dict)
            gap = ((y_pred - y_refined) ** 2).mean()
            # backward on original loss for learning
            obj_fn = _create_objective_function(xb, data_dict, scale=lambda_fs)
            loss = obj_fn(y_refined) + lambda_fs * gap
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_gap_accum += gap.item()
        avg_train_mse = train_gap_accum / len(train_loader)
        logger.info("[FSNet] Epoch %3d/%3d — Train MSE: %.6f",
                        epoch, epochs, avg_train_mse)  

        # —— validation epoch: compute pure projection-gap MSE ——  
        model.eval()
        val_gap_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(device)
                y_pred, y_refined = model(xb, data_dict)
                gap = ((y_pred - y_refined) ** 2).mean()
                val_gap_accum += gap.item()
        avg_val_mse = val_gap_accum / len(val_loader)
        logger.info("[FSNet] Epoch %3d/%3d —  Val MSE: %.6f",
                    epoch+1, epochs, avg_val_mse)  
        # log pure MSE only
        logs.append({"epoch": epoch, "train_loss": avg_train_mse, "val_loss": avg_val_mse})
        logger.info(f"[FSNet] Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")

        # —— vanilla early stopping ——  
        if avg_val_mse < best_val_mse - delta:
            best_val_mse = avg_val_mse
            torch.save(model.state_dict(), save_path)
            logger.info(f"[FSNet] Saved best model (epoch {epoch}) to {save_path}")
            counter = patience
        else:
            counter -= 1
            logger.info(f"[FSNet] No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"[FSNet] Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

    return logs

def train_mtl_incremental(
    model: nn.Module,
    task_loaders: List[DataLoader],
    val_loaders:  List[DataLoader],
    *,
    epochs:     int                    = 10000,
    lr:         float                  = 1e-3,
    patience:   int                    = 100,
    device:     Optional[torch.device] = None,
    save_path:  Optional[str]          = None,
    max_epochs: int                    = 10000,  # kept for API compatibility; unused
    delta:      float                  = 1e-12,
) -> List[Dict[str, float]]:
    """
    Sequential (incremental) training of a fixed-head MTL model with early stopping.

    Works for **both** 4-head (Pg, Qg, Va, Vm) *and* 2-head (Pg_Qg, Va_Vm)
    implementations, as long as the model exposes:
      • `model(x, task_id)` forward
      • `model.heads` (ModuleList)
      • `freeze_head(task_id)` / `unfreeze_head(task_id)`
      • optional `loss_fn(x, y, task_id, metadata=None)` (penalty variants)
    """
    # ---- device -----------------------------------------------------------
    device = device or _model_device(model)
    model.to(device)

    logs: List[Dict[str, float]] = []

    # number of heads (2 or 4). If neither, still trains generically
    num_tasks = len(getattr(model, "heads", []))
    if num_tasks == 0:
        raise ValueError("Model must expose a non-empty `heads` ModuleList for incremental training.")

    # ---- iterate tasks ----------------------------------------------------
    for task_id in range(num_tasks):
        # Unfreeze this head, freeze the rest (GPU-first)
        if hasattr(model, "unfreeze_head") and hasattr(model, "freeze_head"):
            model.unfreeze_head(task_id)
            for other in range(num_tasks):
                if other != task_id:
                    model.freeze_head(other)

        optimizer = torch.optim.Adam(_trainable_params(model), lr=float(lr))

        best_val = float("inf")
        wait = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        tr_loader = task_loaders[task_id]
        va_loader = val_loaders[task_id]

        for epoch in range(int(epochs)):
            # ---------------- TRAIN ----------------
            model.train()
            train_sum = 0.0
            train_cnt = 0

            for batch in tr_loader:
                xb, yb, meta = _unpack_batch(batch)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Keep metadata tensors on device
                if isinstance(meta, dict):
                    meta = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                            for k, v in meta.items()}

                optimizer.zero_grad(set_to_none=True)
                if hasattr(model, "loss_fn"):
                    loss = model.loss_fn(xb, yb, task_id, metadata=meta)
                else:
                    preds = model(xb, task_id)
                    loss = F.mse_loss(preds, yb, reduction="mean")
                loss.backward()
                optimizer.step()

                train_sum += loss.detach().item() * yb.numel()
                train_cnt += yb.numel()

            train_loss = train_sum / max(train_cnt, 1)

            # ---------------- VALID ----------------
            model.eval()
            val_sum = 0.0
            val_cnt = 0
            with torch.no_grad():
                for batch in va_loader:
                    xb, yb, meta = _unpack_batch(batch)
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    if isinstance(meta, dict):
                        meta = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                                for k, v in meta.items()}
                    if hasattr(model, "loss_fn"):
                        vloss = model.loss_fn(xb, yb, task_id, metadata=meta)
                    else:
                        preds = model(xb, task_id)
                        vloss = F.mse_loss(preds, yb, reduction="mean")
                    val_sum += vloss.item() * yb.numel()
                    val_cnt += yb.numel()

            val_loss = val_sum / max(val_cnt, 1)

            logs.append({
                "task": float(task_id),
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })

            # ------------- EARLY STOP -------------
            if val_loss < best_val - float(delta):
                best_val = val_loss
                wait = 0
                # snapshot strictly on-device to avoid host transfer
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= int(patience):
                    break

        # Restore best weights for this head before moving on
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # Optionally freeze this head permanently (keeps final config consistent)
        if hasattr(model, "freeze_head"):
            model.freeze_head(task_id)

    if save_path:
        torch.save(model.state_dict(), save_path)

    return logs

def train_progressive_fixed(
    model: nn.Module,
    task_loaders: List[DataLoader],
    val_loaders:  List[DataLoader],
    task_names:   List[str],
    *,
    epochs: int,
    lr: float,
    patience: int,
    log_file: str = "train_progressive_fixed.csv",
) -> List[Dict[str, Union[int, float, str]]]:
    """
    One trainer for BOTH variants (4-head and 2-head). It matches your run scripts:
      train_progressive_fixed(model, task_loaders, val_loaders, task_names,
                              epochs=..., lr=..., patience=..., log_file=...)
    GPU-first (non_blocking transfers, no host↔device ping-pong), and it works
    with vanilla and penalty models (uses model.loss_fn if present).

    Per-head early stopping is used; we also keep a simple cross-task counter
    like in your snippet, but fixed to use the correct log key ("Val Loss").
    """
    # ---- sanity ------------------------------------------------------------
    assert len(task_loaders) == len(val_loaders) == len(task_names), \
        "task_loaders, val_loaders, task_names must have the same length"
    device = next(model.parameters()).device

    def _trainable_params(m: nn.Module):
        return m.get_all_shared_weights() if hasattr(m, "get_all_shared_weights") \
               else (p for p in m.parameters() if p.requires_grad)

    def _unpack(batch):
        xb, yb = batch[:2]
        meta = batch[2] if len(batch) > 2 else None
        return xb, yb, meta

    def _loss(m: nn.Module, xb, yb, task_id: int, meta):
        if hasattr(m, "loss_fn"):
            if isinstance(meta, dict):
                meta = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                        for k, v in meta.items()}
            return m.loss_fn(xb, yb, task_id, metadata=meta)
        preds = m(xb, task_id)
        return F.mse_loss(preds, yb, reduction="mean")

    # ---- training loop across tasks ---------------------------------------
    all_logs: List[Dict[str, Union[int, float, str]]] = []
    global_patience = int(patience)
    counter = global_patience
    prev_val: Optional[float] = None

    num_tasks = len(task_names)  # works for 2-head and 4-head (len(model.columns) also ok)
    for task_id in range(num_tasks):
        name = task_names[task_id]
        tr_loader = task_loaders[task_id]
        va_loader = val_loaders[task_id]

        logger.info(f"--- Fixed Prog Training {name} (ID={task_id}) ---")
        opt = torch.optim.Adam(_trainable_params(model), lr=float(lr))

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = float("inf")
        wait = 0

        for epoch in range(1, int(epochs) + 1):
            # ---- train ----------------------------------------------------
            model.train()
            train_sum = 0.0
            train_cnt = 0
            for batch in tr_loader:
                xb, yb, meta = _unpack(batch)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                loss = _loss(model, xb, yb, task_id, meta)
                loss.backward()
                opt.step()

                train_sum += loss.detach().item() * yb.numel()
                train_cnt += yb.numel()
            train_loss = train_sum / max(train_cnt, 1)

            # ---- validation ----------------------------------------------
            model.eval()
            val_sum = 0.0
            val_cnt = 0
            with torch.no_grad():
                for batch in va_loader:
                    xb, yb, meta = _unpack(batch)
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    vloss = _loss(model, xb, yb, task_id, meta)
                    val_sum += vloss.item() * yb.numel()
                    val_cnt += yb.numel()
            val_loss = val_sum / max(val_cnt, 1)

            all_logs.append({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Val Loss": val_loss,   # <-- consistent key
                "Task": str(name),
                "Task ID": int(task_id),
            })

            # per-task early stopping
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # restore best weights for this task before moving on
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # optional cross-task early stop like your original snippet (fixed key)
        curr_val = best_val
        if prev_val is not None and curr_val >= prev_val:
            counter -= 1
            logger.info(f"Val did not improve across tasks ({prev_val:.6f} → {curr_val:.6f}), cnt→{counter}")
            if counter == 0:
                logger.info(f"Stopped fixed-progressive at task {task_id}")
                break
        else:
            counter = global_patience
        prev_val = curr_val

    save_logs_to_csv(all_logs, log_file)
    return all_logs
