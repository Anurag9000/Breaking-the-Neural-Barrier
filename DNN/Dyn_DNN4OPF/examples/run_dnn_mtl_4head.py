"""
Multi-Task Learning (MTL) model with shared trunk and task-specific heads.

Purpose:
    Trains a shared representation across tasks with per-task output heads.

Architecture:
    - Shared: 2-layer fully connected with ReLU
    - Task-specific: hidden + output layers

Methodology Role:
    Evaluates performance trade-offs in multi-task vs. continual task learning.
"""

import logging
from typing import Dict, Any, List
import torch
import sys
from pathlib import Path

from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
sys.path.append(str(Path(__file__).resolve().parents[2]))
from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
from Dyn_DNN4OPF.models.dnn_mtl_4head import DNN_MTL_4HEAD, TASK_PG, TASK_QG, TASK_VA, TASK_VM

from Dyn_DNN4OPF.training.trainer import train_mtl_incremental, evaluate, DELTA
from Dyn_DNN4OPF.utils.logger_plotter import (
    save_logs_to_csv,
    plot_losses_from_csv,
)
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs, get_param_names,
    plot_aggregate, plot_per_output,
)
from Dyn_DNN4OPF.utils.repro import set_determinism
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility,
)
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Logger setup & reproducibility
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
set_determinism()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int, device = device)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

# --------------------------------------------------------------------------- #
#  Default hyper-parameters (overridable via Optuna / CLI)
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 10000,
    "patience": 100,
    "max_epochs": 10000,     # forwarded verbatim to trainer
    "log_file": "train_mtl.csv",
    "shared_hidden": 112,
    "task_hidden": 64,
    "mask": None,
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": 27000,
    "val_samples": 1500,
    "test_samples": 1500,
    "batches": None,
    "model": "MTL",
    "clip_test": False,      # ⟶ enable bounded clipping only at Test time
}

# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing DNN-MTL training pipeline")

    # --------------------------------------------------------------------- #
    #  Merge Optuna-tuned hyper-parameters
    # --------------------------------------------------------------------- #
    optuna_params_file = "best_hyperparameters_dnn_mtl.txt"
    optuna_params      = load_optuna_best_params(optuna_params_file)
    config: Dict[str, Any] = {**DEFAULT_CONFIG, **optuna_params}
    logger.info(f"Using training config: {config}")

    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"],
    )

    # Results directory scaffold
    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]
    model_name = config["model"]
    case_name  = config["case_name"]
    output_dir = start_dir / "Results" / f"{model_name}_{case_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "logs", "plots", "diagnostics"):
        (output_dir / sub).mkdir(exist_ok=True)
    config["log_file"] = str(output_dir / "logs" / config["log_file"])

    # --------------------------------------------------------------------- #
    #  I/O dimensions, bounds, masks
    # --------------------------------------------------------------------- #
    input_dim, single_output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = single_output_dim // 2 - n_bus

    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)
    bounds_low, bounds_high = load_output_bounds(case_name)
    check_bounds_compatibility(bounds_low, bounds_high,
                               config["mask"], single_output_dim)
    mask_full = default_mask(n_gen, n_bus)  
    # assume mask_full is List[bool] of length `single_output_dim`
    mask = [
        torch.tensor(mask_full[               : n_gen], dtype=torch.bool, device=device),
        torch.tensor(mask_full[n_gen        : 2*n_gen], dtype=torch.bool, device=device),
        torch.tensor(mask_full[2*n_gen     : 2*n_gen + n_bus], dtype=torch.bool, device=device),
        torch.tensor(mask_full[2*n_gen+n_bus:           ], dtype=torch.bool, device=device),
    ]
    config["mask"] = mask
    # --------------------------------------------------------------------- #
    #  Model instantiation
    #    • Bounds applied **only** at Test time via clip layer
    # --------------------------------------------------------------------- #
    model = DNN_MTL_4HEAD(
        input_dim=input_dim,
        n_gen=n_gen,
        n_bus=n_bus,
        hidden_dim=config.get("shared_hidden"),
        activation=torch.nn.ReLU,
        device=device,
        use_bounds=False,        # still no clipping during Train/Val
        bounds_low=bounds_low,   # full-case lists of length 4
        bounds_high=bounds_high, # ditto
        mask=config["mask"] # see next section for how to build this
    )
    model.to(device)

    # --------------------------------------------------------------------- #
    #  Training (incremental API – currently one head)
    # --------------------------------------------------------------------- #
    X_tr, Y_tr, _ = train_loader.dataset.tensors
    X_va, Y_va, _ = val_loader.dataset.tensors

    # slice them into per-head targets
    pg_tr = Y_tr[:,               :n_gen]
    qg_tr = Y_tr[:,   n_gen      :2*n_gen]
    va_tr = Y_tr[:, 2*n_gen      :2*n_gen + n_bus]
    vm_tr = Y_tr[:, 2*n_gen+n_bus:]

    pg_va = Y_va[:,               :n_gen]
    qg_va = Y_va[:,   n_gen      :2*n_gen]
    va_va = Y_va[:, 2*n_gen      :2*n_gen + n_bus]
    vm_va = Y_va[:, 2*n_gen+n_bus:]

    X_tr = X_tr.to(device)
    pg_tr = pg_tr.to(device)
    qg_tr = qg_tr.to(device)
    va_tr = va_tr.to(device)
    vm_tr = vm_tr.to(device)

    X_va = X_va.to(device)
    pg_va = pg_va.to(device)
    qg_va = qg_va.to(device)
    va_va = va_va.to(device)
    vm_va = vm_va.to(device)

    # rebuild DataLoader lists
    batch = config["batch_size"]
    train_loaders = [
        DataLoader(TensorDataset(X_tr, pg_tr), batch_size=batch, shuffle=True, pin_memory=True),
        DataLoader(TensorDataset(X_tr, qg_tr), batch_size=batch, shuffle=True, pin_memory=True),
        DataLoader(TensorDataset(X_tr, va_tr), batch_size=batch, shuffle=True, pin_memory=True),
        DataLoader(TensorDataset(X_tr, vm_tr), batch_size=batch, shuffle=True, pin_memory=True),
    ]
    val_loaders = [
        DataLoader(TensorDataset(X_va, pg_va), batch_size=batch, shuffle=False, pin_memory=True),
        DataLoader(TensorDataset(X_va, qg_va), batch_size=batch, shuffle=False, pin_memory=True),
        DataLoader(TensorDataset(X_va, va_va), batch_size=batch, shuffle=False, pin_memory=True),
        DataLoader(TensorDataset(X_va, vm_va), batch_size=batch, shuffle=False, pin_memory=True),
    ]

    # call your new fixed-head trainer
    logs = train_mtl_incremental(
        model       = model,
        task_loaders= train_loaders,
        val_loaders = val_loaders,
        epochs      = config["epochs"],
        lr          = config["lr"],
        patience    = config["patience"],
        device      = device,
        save_path   = None,
        delta       = DELTA,
    )
    # --------------------------------------------------------------------- #
    #  Persist checkpoint
    # --------------------------------------------------------------------- #
    ckpt_path = output_dir / "models" / f"best_model_MTL_{case_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved MTL checkpoint to {ckpt_path}")

    # --------------------------------------------------------------------- #
    #  Logs → CSV & training curves
    # --------------------------------------------------------------------- #
    save_logs_to_csv(logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        str(output_dir / "plots" / "train_mtl_plot.png"),
        str(output_dir / "plots" / "train_mtl_test_plot.png"),
    )

    # --------------------------------------------------------------------- #
    #  Optional Test-time clipping mask
    # --------------------------------------------------------------------- #
    if config["clip_test"]:
        clip_layer = BoundedAct(
            bounds_low.to(device),
            bounds_high.to(device),
            torch.tensor(config["mask"], dtype=torch.bool, device=device),
        )
        clip_layer.apply_bounds.fill_(True)   # enforce clipping
        test_model = torch.nn.Sequential(model, clip_layer).to(device)
    else:
        test_model = model

    # --------------------------------------------------------------------- #
    #  Final Test evaluation (pure MSE)
    # --------------------------------------------------------------------- #
    test_loss = evaluate(test_model, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    # --------------------------------------------------------------------- #
    #  Metadata + aggregate/per-output plots
    # --------------------------------------------------------------------- #
    save_metadata_to_json(OBJ_test, output_dir / "logs" / "metadata.json")

    df = load_logs(Path(config["log_file"]))
    df = df.rename(columns={"Epoch": "epoch",
                            "Train Loss": "train_loss",
                            "Val Loss": "val_loss"})
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    plot_aggregate(df, model_name, output_dir / "plots" / model_name)
    plot_per_output(df, model_name,
                    output_dir / "plots" / model_name,
                    get_param_names(train_loader))

    # --------------------------------------------------------------------- #
    #  Diagnostics (MSE-only) – uses wrapped model if clipping enabled
    # --------------------------------------------------------------------- #
    with torch.no_grad():
        X_tr, Y_tr, _ = train_loader.dataset.tensors
        X_va, Y_va, _ = val_loader.dataset.tensors
        X_te, Y_te    = test_loader.dataset.tensors

    generate_all_diagnostics(
        model=test_model,
        datasets={
            "Train":      (X_tr, Y_tr),
            "Validation": (X_va, Y_va),
            "Test":       (X_te, Y_te),
        },
        device=device,
        case_json=Path("data") / f"sample_{case_name.split('_')[2]}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=model_name,
        mse_only=True,      # ⟶ suppress non-MSE plots/metrics
    )

    logger.info(f"Completed DNN-MTL training pipeline. Test loss: {test_loss:.6f}")


# --------------------------------------------------------------------------- #
#  Entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
