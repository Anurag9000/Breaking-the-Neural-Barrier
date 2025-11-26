"""
dnn_l2.py
=========

L2-Regularized fully connected neural network (weight decay).

Goal:
    Encourage parameter sparsity and improve generalization through L2 penalty.

Architecture:
    Identical to `dnn_stl.py` but intended for training with L2 regularization (via optimizer).

Paper Context:
    Forms an intermediate baseline between STL and advanced continual learning methods.
    Helps assess whether simple weight decay reduces forgetting.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
import sys

import torch

# Add the project root directory (one level above Dyn_DNN4OPF) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
from penalty_nn.models.penalty_l2 import PenaltyDNN_L2
from Dyn_DNN4OPF.training.trainer import train_with_l2, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs, get_param_names,
    plot_aggregate, plot_per_output,
)
from Dyn_DNN4OPF.utils.repro import set_determinism
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility
)

# ──────────────────────────── setup ────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
set_determinism()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int, device = device)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 10000,
    "patience": 10000,
    "max_epochs": 10000,          # forwarded verbatim to trainer
    "hidden_dim": 112,
    "log_file": "train_l2.csv",
    "mask": None,
    "weight_decay": 1e-4,
    # ── penalty-loss weights ─────────────────────────────────
    "lambda_loss": 1.0,      # λ₁ on baseline (MSE) term
    "lambda_eq":   1.0,      # λ₂ on equality residuals
    "lambda_ineq": 1.0,      # λ₃ on inequality violations
    "model": "PenaltyL2",

    # ── Case-agnostic loader keys ─────────────────────────────
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": 27000,   # None = use full OPF train split
    "val_samples":   1500,    # None = use full OPF val split
    "test_samples":  1500,    # None = use full OPF test split
    "batches":       None,    # None = default splits; or list of batch indices

    # ── Test-time clipping ───────────────────────────────────
    "clip_test": False,       # if True → clip *only* on Test/diagnostics
}

# ────────────────────────── pipeline ───────────────────────────
def run_pipeline(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing DNN-L2 training pipeline")

    # -------- hyper-parameter resolution ---------------------
    optuna_params_file = "best_hyperparameters_dnn_l2.txt"
    optuna_params = load_optuna_best_params(optuna_params_file)
    config: Dict[str, Any] = {**DEFAULT_CONFIG, **optuna_params}
    logger.info(f"Using training config: {config}")

    # -------- data loaders -----------------------------------
    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config.get("train_samples"),
        config.get("val_samples"),
        config.get("test_samples"),
        config.get("batches"),
    )

    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]  # move two levels up if launched from examples/

    model_name = config["model"]
    case_name  = config["case_name"]
    output_dir = start_dir / "Results" / f"{model_name}_{case_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "logs", "plots", "diagnostics"):
        (output_dir / sub).mkdir(exist_ok=True)

    # -------- log file path ----------------------------------
    config["log_file"] = str(output_dir / "logs" / config["log_file"])

    # -------- dimensions & bounds ----------------------------
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus

    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)

    bounds_low, bounds_high = load_output_bounds(case_name=config["case_name"])
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    # -------- model ------------------------------------------
    hidden_dim = config.get("hidden_dim") or (4 * input_dim)
    model = PenaltyDNN_L2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bounds=True,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=config["mask"],
        lambda_loss=config["lambda_loss"],
        lambda_eq=config["lambda_eq"],
        lambda_ineq=config["lambda_ineq"],
    )
    logger.info(
        f"Initialized DNN_L2(input={input_dim}, hidden={hidden_dim}, "
        f"output={output_dim}, bounds=True)"
    )
    model.to(device)

    # -------- training ---------------------------------------
    logs = train_with_l2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        epochs=config["epochs"],
        patience=config["patience"],
        max_epochs=config["max_epochs"],
        device=device,
    )

    # -------- checkpoint save --------------------------------
    ckpt_dir = output_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_model_L2_{config['case_name']}.pth"
    ckpt_path = ckpt_dir / ckpt_name
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved L2 checkpoint to {ckpt_path}")

    # -------- logs to CSV & plots ----------------------------
    save_logs_to_csv(logs, config["log_file"])
    p_plots = Path(config["log_file"]).parent.parent / "plots"
    plot_losses_from_csv(
        config["log_file"],
        str(p_plots / "train_l2_plot.png"),
        str(p_plots / "train_l2_test_plot.png"),
    )

    # -------- reload best weights ----------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ─── optional test-time clipping ──────────────────────────
    if config.get("clip_test", False):
        clip_layer = BoundedAct(
            bounds_low.to(device),
            bounds_high.to(device),
            torch.tensor(config["mask"], dtype=torch.bool, device=device),
        )
        clip_layer.apply_bounds.fill_(True)
        eval_model = torch.nn.Sequential(model, clip_layer).to(device)
    else:
        eval_model = model

    # -------- evaluate on Test set (MSE only) ----------------
    test_loss = evaluate(eval_model, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    # -------- metadata & plots -------------------------------
    save_metadata_to_json(OBJ_test, output_dir / "logs" / "metadata.json")

    df = load_logs(Path(config["log_file"]))
    df = df.rename(columns={
        "Epoch":      "epoch",
        "Train Loss": "train_loss",
        "Val Loss":   "val_loss",
    })
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    out_dir_plots = output_dir / "plots" / model_name
    plot_aggregate(df, model_name, out_dir_plots)
    param_names = get_param_names(train_loader)
    plot_per_output(df, model_name, out_dir_plots, param_names)

    # -------- diagnostics ------------------------------------
    with torch.no_grad():
        X_train, Y_train, _ = train_loader.dataset.tensors
        X_val,   Y_val, _   = val_loader.dataset.tensors
        X_test,  Y_test     = test_loader.dataset.tensors

    diagnostic_inputs = {
        "Train":      (X_train, Y_train),
        "Validation": (X_val,   Y_val),
        "Test":       (X_test,  Y_test),
    }

    case = case_name.split("_")[2]
    generate_all_diagnostics(
        model=eval_model,
        datasets=diagnostic_inputs,
        device=device,
        case_json=Path("data") / f"sample_{case}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

    logger.info(f"Completed DNN-L2 training pipeline. Test loss: {test_loss:.6f}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
