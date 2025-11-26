import os,sys
from pathlib import Path
import math
import copy
sys.path.append(str(Path(__file__).resolve().parents[2]))
from Dyn_DNN4OPF.training.trainer import DELTA
import torch.nn.functional as F
import torch.nn as nn
import json
from torch.utils.data import DataLoader
import logging
import torch
from typing import List, Dict, Any, Union, Optional
from Dyn_DNN4OPF.training.ewc_utils import EWC
from Dyn_DNN4OPF.training.training_helpers import _to_device
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
lr = 1e-3
from typing import Tuple

current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_penalty(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    device:       torch.device,
    cfg:          dict,
    save_path:    str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> None:
    """
    Train a Penalty STL model using its custom loss_fn, with:
      • pure-MSE logging only
      • vanilla early stopping on val MSE
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=cfg.get("lr", lr),
        **SCHEDULER_PARAMS
    )

    best_val = float("inf")
    patience = cfg.get("patience", 100)
    counter  = patience
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(max_epochs):
        # —— training epoch ——  
        model.train()
        total_train_loss = 0.0
        for x, y, meta in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.loss_fn(x, y, metadata=meta)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # accumulate *pure* MSE
            preds = model(x)
            total_train_loss += F.mse_loss(preds, y).item()
        train_loss = total_train_loss / len(train_loader.dataset)

        # —— validation epoch ——  
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y, meta in val_loader:
                x, y = x.to(device), y.to(device)
                # log *pure* MSE on validation
                preds = model(x)
                total_val_loss += F.mse_loss(preds, y).item()
        val_loss = total_val_loss / len(val_loader.dataset)

        # —— log & persist ——  
        logger.info(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_logs_to_csv(logs, cfg.get("log_file", "penalty_log.csv"))

        # —— vanilla early stopping ——  
        if val_loss < best_val - delta:
            best_val = val_loss
            counter = patience
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best model (epoch {epoch}) to {save_path}")
        else:
            counter -= 1
            logger.info(f"No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"Early stopping at epoch {epoch}")
                break

def train_penalty_mae(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    device:       torch.device,
    cfg:          dict,
    save_path:    str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> None:
    """
    Train a Penalty MAE model using its custom loss_fn, with:
      • pure-MAE logging only
      • vanilla early stopping on val MAE
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=cfg.get("lr", lr),
        **SCHEDULER_PARAMS
    )

    best_val = float("inf")
    patience = cfg.get("patience", 100)
    counter  = patience
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(max_epochs):
        # —— training epoch ——  
        model.train()
        total_train_loss = 0.0
        for x, y, meta in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.loss_fn(x, y, metadata=meta)
            loss.backward()
            optimizer.step()
            scheduler.step()
            preds = model(x)
            total_train_loss += F.l1_loss(preds, y).item()
        train_loss = total_train_loss / len(train_loader.dataset)

        # —— validation epoch ——  
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y, meta in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                total_val_loss += F.l1_loss(preds, y).item()
        val_loss = total_val_loss / len(val_loader.dataset)

        # —— log & persist ——  
        logger.info(f"Epoch {epoch:03d} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_logs_to_csv(logs, cfg.get("log_file", "train_penalty_mae.csv"))

        # —— vanilla early stopping ——  
        if val_loss < best_val - delta:
            best_val = val_loss
            counter = patience
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best model (epoch {epoch}) to {save_path}")
        else:
            counter -= 1
            logger.info(f"No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"Early stopping at epoch {epoch}")
                break

def train_penalty_ewc(
    model: torch.nn.Module,
    task_train_loaders: List[DataLoader],
    task_val_loaders:   List[DataLoader],
    device:             torch.device,
    cfg:                Dict[str, Any],
    save_dir:           str,
    *,
    max_epochs: int     = 10000,
    delta: float        = DELTA
) -> None:
    """
    Sequentially train a Penalty-EWC model with:
      • pure-MSE logging only,
      • vanilla best-val + Δ + patience early stopping,
      • task-plateau spawn & stop logic preserved.
    """
    model.to(device)
    ewc_list: List[EWC] = []
    previous_best_loss: Optional[float] = None
    task_id = 1

    # prepare output dirs
    models_dir = os.path.join(save_dir, "models")
    logs_dir   = os.path.join(save_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)
    log_file = os.path.join(logs_dir, cfg.get("log_file", "train_penalty_ewc.csv"))

    all_logs: List[Dict[str, Union[int, float]]] = []

    # outer patience‐spawn loop
    while task_id <= len(task_train_loaders):
        train_loader = task_train_loaders[task_id-1]
        val_loader   = task_val_loaders[task_id-1]

        # snapshot before task
        prev_state = copy.deepcopy(model.state_dict())
        optimizer, scheduler = get_optimizer_scheduler(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            **SCHEDULER_PARAMS
        )

        best_val_mse = float("inf")
        best_state   = copy.deepcopy(model.state_dict())
        patience_ctr = 0
        patience     = cfg.get("patience", 100)

        for epoch in range(max_epochs):
            # —— training epoch with EWC penalty (unchanged) ——  
            model.train()
            for x, y, meta in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = model.loss_fn(x, y, meta)
                for ewc in ewc_list:
                    loss = loss + cfg.get("lambda_ewc", 0.0) * ewc.penalty(model)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # —— compute & log pure MSE ——  
            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_sum += F.mse_loss(model(x), y).item()
            val_mse = val_sum / len(val_loader.dataset)

            logger.info(f"Task {task_id} Ep{epoch:03d} | Val MSE: {val_mse:.6f}")
            all_logs.append({"task": task_id, "epoch": epoch, "val_loss": val_mse})

            # vanilla early stopping on pure MSE
            if val_mse < best_val_mse - delta:
                best_val_mse = val_mse
                best_state   = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                torch.save(best_state, os.path.join(models_dir, f"task{task_id}_best.pth"))
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        # restore best of this task
        model.load_state_dict(best_state)

        # spawn vs stop decision
        if previous_best_loss is None or best_val_mse < previous_best_loss - delta:
            # ✔ improvement → record Fisher & next task
            ewc_list.append(EWC(model, train_loader, device=device))
            previous_best_loss = best_val_mse
            task_id += 1
            continue
        else:
            # ✖ no improvement → rollback & exit
            logger.info(
                f"[Penalty-EWC] Task {task_id} val={best_val_mse:.6f} "
                f"did not beat prev={previous_best_loss:.6f}−{delta}. Stopping."
            )
            model.load_state_dict(prev_state)
            break

    # save logs
    save_logs_to_csv(all_logs, log_file)

def train_penalty_l2(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    cfg:          Dict[str, Any],
    save_path:    str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> None:
    """
    Train a PenaltyL2 model using its custom loss_fn, but:
      • log only pure MSE each epoch,
      • vanilla early stopping on validation MSE.
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=cfg.get("lr", lr),
        **SCHEDULER_PARAMS
    )
    mse_fn = nn.MSELoss()

    best_val_mse = float("inf")
    patience     = cfg.get("patience", 100)
    counter      = patience
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(max_epochs):

        # —— training epoch: pure MSE accumulation ——  
        model.train()
        train_accum = 0.0
        for x, y, meta in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            full_loss = model.loss_fn(x, y, meta)
            full_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                preds = model(x)
                train_accum += mse_fn(preds, y).item()
        avg_train_mse = train_accum / len(train_loader.dataset)

        # —— validation epoch: pure MSE only ——  
        model.eval()
        val_accum = 0.0
        with torch.no_grad():
            for x, y, meta in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_accum += mse_fn(preds, y).item()
        avg_val_mse = val_accum / len(val_loader.dataset)

        # log & save to CSV
        logger.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")
        logs.append({"epoch": epoch, "train_loss": avg_train_mse, "val_loss": avg_val_mse})
        save_logs_to_csv(logs, cfg.get("log_file", "train_penalty_l2.csv"))

        # vanilla early stopping on pure-Val MSE
        if avg_val_mse < best_val_mse - delta:
            best_val_mse = avg_val_mse
            counter = patience
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PenaltyL2 model (epoch {epoch}) to {save_path}")
        else:
            counter -= 1
            logger.info(f"No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"Early stopping PenaltyL2 at epoch {epoch}")
                break

def train_penalty_l1(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    save_path: str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> None:
    """
    Train a PenaltyL1 model using its custom loss_fn + L1 penalty, but with:
      • pure-MSE logging only
      • vanilla best-val + Δ + patience early stopping
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        **SCHEDULER_PARAMS
    )
    mse_fn   = nn.MSELoss()
    best_val = float("inf")
    patience = cfg.get("patience", 100)
    counter  = patience
    epochs   = cfg.get("epochs", 10000)
    log_csv  = cfg.get("log_file", "train_penalty_l1.csv")
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(max_epochs):
        # —— training epoch: accumulate pure MSE ——  
        model.train()
        train_accum = 0.0
        for xb, yb, meta in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # backprop on full custom loss
            full_loss = model.loss_fn(xb, yb, meta) + cfg.get("lambda_l1", 0) * model.l1_penalty()
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()
            scheduler.step()
            # accumulate pure MSE
            with torch.no_grad():
                preds = model(xb)
                train_accum += mse_fn(preds, yb).item()
        avg_train_mse = train_accum / len(train_loader.dataset)

        # —— validation epoch: pure MSE only ——  
        model.eval()
        val_accum = 0.0
        with torch.no_grad():
            for xb, yb, meta in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_accum += mse_fn(preds, yb).item()
        avg_val_mse = val_accum / len(val_loader.dataset)

        # log & persist
        logger.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")
        logs.append({"epoch": epoch, "train_loss": avg_train_mse, "val_loss": avg_val_mse})
        save_logs_to_csv(logs, log_csv)

        # vanilla early stopping
        if avg_val_mse < best_val - delta:
            best_val = avg_val_mse
            counter  = patience
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PenaltyL1 model (epoch {epoch}) to {save_path}")
        else:
            counter -= 1
            logger.info(f"No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"Early stopping PenaltyL1 at epoch {epoch}")
                break

def train_penalty_elastic(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    save_path: str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> None:
    """
    Train PenaltyElastic model with:
      • pure-MSE logging only
      • vanilla best-val + Δ + patience early stopping
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        **SCHEDULER_PARAMS
    )
    mse_fn   = nn.MSELoss()
    best_val = float("inf")
    patience = cfg.get("patience", 100)
    counter  = patience
    epochs   = cfg.get("epochs", 10000)
    log_csv  = cfg.get("log_file", "train_penalty_elastic.csv")
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(max_epochs):
        # —— training epoch: run full penalty update, accumulate pure MSE ——  
        model.train()
        train_accum = 0.0
        for xb, yb, meta in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            full_loss = model.loss_fn(xb, yb, meta)
            full_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                preds = model(xb)
                train_accum += mse_fn(preds, yb).item()
        avg_train_mse = train_accum / len(train_loader.dataset)

        # —— validation epoch: pure MSE only ——  
        model.eval()
        val_accum = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_accum += mse_fn(preds, yb).item()
        avg_val_mse = val_accum / len(val_loader.dataset)

        # log & persist
        logger.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")
        logs.append({"epoch": epoch, "train_loss": avg_train_mse, "val_loss": avg_val_mse})
        save_logs_to_csv(logs, log_csv)

        # vanilla early stopping on pure MSE
        if avg_val_mse < best_val - delta:
            best_val = avg_val_mse
            counter  = patience
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PenaltyElastic model (epoch {epoch}) to {save_path}")
        else:
            counter -= 1
            logger.info(f"No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"Early stopping PenaltyElastic at epoch {epoch}")
                break

def train_penalty_fsnet(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    data_dict: dict,
    val_loader:   torch.utils.data.DataLoader,
    device:       torch.device,
    cfg:          dict,
    save_path:    str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> None:
    """
    Train a PenaltyFSNet model using its custom loss_fn, but with:
      • pure‐MSE logging only (projection‐gap MSE)
      • vanilla best‐val + Δ + patience early stopping
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        **SCHEDULER_PARAMS
    )

    mse_fn = nn.MSELoss()
    best_val = float("inf")
    patience = cfg.get("patience", 100)
    counter  = patience
    epochs   = cfg.get("epochs", 10000)
    logs: List[Dict[str, Union[int, float]]] = []

    for epoch in range(max_epochs):
        # —— training step (unchanged) ——  
        model.train()
        for batch in train_loader:
            X = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
            optimizer.zero_grad()
            loss = model.loss_fn(X, None, data_dict)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # —— compute pure projection‐gap MSE on train ——  
        model.eval()
        train_sum = 0.0
        with torch.no_grad():
            for batch in train_loader:
                X = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
                y_pred, y_refined = model(X, data_dict)
                train_sum += mse_fn(y_pred, y_refined).item()
        train_mse = train_sum / len(train_loader.dataset)

        # —— compute pure projection‐gap MSE on validation ——  
        val_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
                y_pred, y_refined = model(X, data_dict)
                val_sum += mse_fn(y_pred, y_refined).item()
        val_mse = val_sum / len(val_loader.dataset)

        # log & persist
        logger.info(f"Epoch {epoch:03d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        logs.append({"epoch": epoch, "train_loss": train_mse, "val_loss": val_mse})
        save_logs_to_csv(logs, cfg.get("log_file", "penalty_fsnet_log.csv"))

        # vanilla early stopping
        if val_mse < best_val - delta:
            best_val = val_mse
            counter  = patience
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PenaltyFSNet model (epoch {epoch}) to {save_path}")
        else:
            counter -= 1
            logger.info(f"No improvement (Δ<{delta}); counter → {counter}")
            if counter == 0:
                logger.info(f"Early stopping PenaltyFSNet at epoch {epoch}")
                break

def train_penalty_progressive_patience_spawn(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    eval_loader:  torch.utils.data.DataLoader,
    cfg:          Dict[str, Any],
    save_path:    str,
    log_file:     str = "train_penalty_progressive_ps.csv",
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> List[Dict[str, Any]]:
    """
    Same flow as `train_progressive_patience_spawn`, but with:
      • pure‐MSE logging only
      • vanilla best‐val + Δ + patience early stopping
    """
    device     = _to_device(model)
    all_logs: List[Dict[str, Any]] = []
    prev_best  = float("inf")
    task_id    = 0
    fisher_list: List[Any] = []

    for epoch in range(max_epochs):
        logger.info(f"── Penalty‐Prog Column {task_id} ──")
        optimizer, scheduler = get_optimizer_scheduler(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            **SCHEDULER_PARAMS
        )

        best_val = float("inf")
        patience = cfg.get("patience", 100)
        counter  = patience
        epoch    = 0
        task_logs: List[Dict[str, Any]] = []

        # inner vanilla early‐stop loop
        for epoch in range (max_epochs):
            # —— training epoch ——  
            model.train()
            train_sum = 0.0
            for x, y, *meta in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = model.loss_fn(x, y, meta[0] if meta else None, task_id=task_id)
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    train_sum += F.mse_loss(model(x, task_id), y).item()
            train_mse = train_sum / len(train_loader.dataset)

            # —— validation epoch ——  
            model.eval()
            with torch.no_grad():
                val_mse = F.mse_loss(
                    model(val_loader.dataset.tensors[0].to(device), task_id),
                    val_loader.dataset.tensors[1].to(device),
                ).item()

            task_logs.append({
                "task":       task_id,
                "epoch":      epoch,
                "train_loss": train_mse,
                "val_loss":   val_mse
            })

            # vanilla early stopping
            if val_mse < best_val - delta:
                best_val   = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                counter    = patience
            else:
                counter -= 1
                if counter == 0:
                    break

            epoch += 1

        # rollback to best
        model.load_state_dict(best_state)
        all_logs.extend(task_logs)
        fisher_list.append(EWC(model, train_loader, device))

        # spawn vs stop
        if task_id == 0 or (prev_best - best_val) > delta:
            prev_best = best_val
            model.add_column(output_dim=train_loader.dataset.tensors[1].shape[1])
            task_id += 1
            continue
        else:
            logger.info("⚑  Improvement ≤ Δ – training stops.")
            break

    save_logs_to_csv(all_logs, log_file)
    torch.save(model.state_dict(), save_path)
    return all_logs

def train_penalty_mtl_incremental(
    model: torch.nn.Module,
    task_loaders: List[DataLoader],
    val_loaders:   List[DataLoader],
    device:        torch.device,
    cfg:           Dict[str, Any],
    save_path:     str,
    *,
    max_epochs: int   = 10000,
    delta:      float = DELTA
) -> List[Dict[str, Any]]:
    """
    Penalty‐MTL incremental‐head growth with:
      • pure‐MSE logging only
      • vanilla best‐val + Δ + patience early stopping
    """
    model.to(device)
    logs: List[Dict[str, Any]] = []
    global_best = float("inf")
    task_id     = 0
    lr          = cfg.get("lr", 1e-3)
    epochs      = cfg.get("epochs", 10000)
    patience    = cfg.get("patience", 100)

    while task_id < len(task_loaders):

        tr_loader = task_loaders[task_id]
        va_loader = val_loaders[task_id]
        if task_id > 0:
            model.freeze_head(task_id - 1)

        optimizer, scheduler = get_optimizer_scheduler(
            model.parameters(),
            lr=lr,
            **SCHEDULER_PARAMS
        )

        best_val = float("inf")
        counter  = patience

        for epoch in range(max_epochs):
            # —— training epoch ——  
            model.train()
            tr_sum = 0.0
            for x, y_full, meta in tr_loader:
                x, y_full, meta = x.to(device), y_full.to(device), meta.to(device)
                optimizer.zero_grad()
                loss = model.loss_fn(x, torch.split(y_full, model.output_dims, 1), metadata=meta)
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    preds = model.predict_all(x)
                    tr_sum += F.mse_loss(preds, y_full).item()
            avg_tr_mse = tr_sum / len(tr_loader.dataset)

            # —— validation epoch ——  
            model.eval()
            va_sum = 0.0
            with torch.no_grad():
                for x, y_full, _ in va_loader:
                    x, y_full = x.to(device), y_full.to(device)
                    va_sum += F.mse_loss(model.predict_all(x), y_full).item()
            avg_va_mse = va_sum / len(va_loader.dataset)

            logs.append({
                "task":       task_id,
                "epoch":      epoch,
                "train_loss": avg_tr_mse,
                "val_loss":   avg_va_mse
            })
            save_logs_to_csv(logs, cfg.get("log_file", "train_penalty_mtl.csv"))
            logger.info(f"[Penalty Inc‐MTL] Task{task_id} Ep{epoch:03d} "
                        f"train={avg_tr_mse:.6f} val={avg_va_mse:.6f}")

            # vanilla early stopping on this head
            if avg_va_mse < best_val - delta:
                best_val = avg_va_mse
                best_state = copy.deepcopy(model.state_dict())
                counter = patience
            else:
                counter -= 1
                if counter == 0:
                    break

        # rollback to best head
        if 'best_state' in locals():
            model.load_state_dict(best_state)

        # spawn vs stop
        if best_val < global_best - delta:
            global_best = best_val
            model.freeze_head(task_id)
            nxt = task_id + 1
            if nxt < len(task_loaders):
                od = next(iter(task_loaders[nxt]))[1].shape[1]
                model.add_task_head(od)
                task_id += 1
                continue
        break

    if save_path:
        torch.save(model.state_dict(), save_path)
    return logs

def train_penalty_den_tasks(
    config: Dict[str, Any],
    task_loaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, Any]]],
    *,
    max_epochs: Optional[int] = None,
    delta: Optional[float]    = None,
) -> torch.nn.Module:
    """
    Adapter that constructs the right DEN variant and trains it on a single task.
    Returns the trained model (restored to best val).
    """
    device = current_device
    model_name = str(config.get("model", "")).lower()
    # ---- construct model (dispatch) ------------------------------------------
    if "4head" in model_name:
        from penalty_nn.models.penalty_den_4head import PenaltyDEN_4head as ModelCls
        model = ModelCls(
            cfg=config,
            lambda_loss=config["lambda_loss"],
            lambda_eq=config["lambda_eq"],
            lambda_ineq=config["lambda_ineq"],
            case_name=config.get("case_name"),
            clip_test=config.get("clip_test", False),
        ).to(device)
    elif "2head" in model_name:
        from penalty_nn.models.penalty_den_2head import PenaltyDEN as ModelCls
        model = ModelCls(
            cfg=config,
            lambda_loss=config["lambda_loss"],
            lambda_eq=config["lambda_eq"],
            lambda_ineq=config["lambda_ineq"],
            case_name=config.get("case_name"),
            clip_test=config.get("clip_test", False),
        ).to(device)
    else:
        # single-head variant uses lambda_mse kw name
        from penalty_nn.models.penalty_den import PenaltyDEN as ModelCls
        model = ModelCls(
            config,
            lambda_mse=config["lambda_loss"],
            lambda_eq=config["lambda_eq"],
            lambda_ineq=config["lambda_ineq"],
            clip_test=config.get("clip_test", False),
        ).to(device)

    # ---- train on the (single) provided task --------------------------------
    assert len(task_loaders) == 1, "train_penalty_den_tasks expects a single task."
    train_loader, val_loader, test_loader, constraints = task_loaders[0]

    # For single-head, register case constants (bounds & Y-bus) once
    if "2head" not in model_name and "4head" not in model_name and hasattr(model, "register_case_constants"):
        ineq, eq = constraints["ineq"], constraints["eq"]
        const = {
            "p_min": ineq["p_min"], "p_max": ineq["p_max"],
            "q_min": ineq["q_min"], "q_max": ineq["q_max"],
            "v_min": ineq["v_min"], "v_max": ineq["v_max"],
            "y_bus": eq["y_bus"],
            "gen_bus_idx": eq["gen_bus_idx"], "load_bus_idx": eq["load_bus_idx"],
        }
        model.register_case_constants(const)

    save_path = str(Path(config.get("log_file", "penalty_log.csv")).with_suffix(".pth"))
    train_penalty(
        model, train_loader, val_loader, device, config, save_path,
        max_epochs=(max_epochs if max_epochs is not None else int(config.get("max_epochs", 10000))),
        delta=(delta if delta is not None else float(config.get("loss_thr", 1e-4))),
    )
    # restore best & return
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model
