"""
dnn_stl.py
==========

Standard fully connected neural network (STL - Single Task Learning).

Purpose:
    Baseline model that trains a separate MLP for each task without knowledge transfer.

Structure:
    - Two hidden layers with ReLU activations
    - One output layer
    - No regularization or multi-task components

Role in Paper:
    Serves as the "lower bound" comparison in continual learning experiments.
    Demonstrates catastrophic forgetting when trained sequentially.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import logging
from typing import Dict, Any, List
import torch
from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
from Dyn_DNN4OPF.models.dnn_stl import FullyConnectedNet
from Dyn_DNN4OPF.training.trainer import train, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from Dyn_DNN4OPF.utils.config import (
    default_mask,
    check_bounds_compatibility,
    get_io_dims_from_loader
)
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.utils.repro import set_determinism
set_determinism()
from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs, get_param_names,
)

def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 100000,
    "patience": 100,
    "hidden_dim": 270,
    "width": 270,
    "depth": 1,
    "log_file": "train_stl.csv",
    "mask": None,
    "model": "STL",
    "case_name": "pglib_opf_case118_ieee",
    "train_samples": 27000,
    "val_samples": 1500,
    "test_samples": 1500,
    "batches": None,
    "clip_test": False,
    "max_epochs": 1e10,
}

def run_pipeline(cfg: Dict[str, Any]) -> None:
    config: Dict[str, Any] = {**DEFAULT_CONFIG, **cfg}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")
    logger.info("Initializing DNN-STL training pipeline")

    # Load Optuna hyperparameters if available
    optuna_params_file = "best_hyperparameters_dnn_stl.txt"
    optuna_params = load_optuna_best_params(optuna_params_file)

    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config.get("train_samples", 27000),
        config.get("val_samples", 1500),
        config.get("test_samples", 1500),
        config.get("batches", None)
    )

    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]  # Move two levels up if 'examples' is in the path

    model_name = config["model"]
    case_name = config["case_name"]
    output_dir = start_dir / "Results" / f"{model_name}_{case_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "logs", "plots", "diagnostics"):
        (output_dir / sub).mkdir(exist_ok=True)

    # override default log filename
    config["log_file"] = str(output_dir / "logs" / config["log_file"])

    case_name = config.get("case_name", "pglib_opf_case118_ieee")  # default if not supplied
    bounds_low, bounds_high = load_output_bounds(case_name=case_name)
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus

    # Use default mask if not provided
    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Verify bounds and mask dimensions match output_dim
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    hidden_dim = config.get("hidden_dim")
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    model = FullyConnectedNet(
        input_dim=input_dim,
        output_dim=output_dim,
        width=config["width"],
        depth=config["depth"],
        use_bounds=True,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=config["mask"]
    )
    logger.info(
        f"Initialized FullyConnectedNet(input={input_dim}, width={config['width']}, depth={config['depth']}, "
        f"output={output_dim}, use_bounds=True)"
    )
    model.to(device)

    logs = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["lr"],
        epochs=config["epochs"],
        patience=config["patience"],
        max_epochs=config["max_epochs"],
        device=device,
        save_path=output_dir / "models" / f"best_model_{model_name}.pth",
    )

    # build the directory where you want to stash your STL model
    out_dir = output_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # match the load filename
    ckpt_name = f"best_model_STL_{config['case_name']}.pth"
    ckpt_path = out_dir / ckpt_name

    # save your trained model
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved STL checkpoint to {ckpt_path}")

    save_logs_to_csv(logs, config["log_file"])

    # ─── save & load checkpoint from models/
    ckpt_save = output_dir / "models" / f"best_model_{model_name}_{case_name}.pth"
    model.load_state_dict(torch.load(ckpt_save, map_location=device))
    logger.info(f"Loaded best model weights from {ckpt_save}")
    model.eval()

    # ─── plot losses into plots/
    p = output_dir / "plots"
    plot_losses_from_csv(
        config["log_file"],
        str(p / "train_stl_plot.png"),
        str(p / "train_stl_train_plot.png"),
    )

    csv_path = output_dir / "logs" / config["log_file"]
    df = load_logs(csv_path)
    df = df.rename(columns={
        "Epoch": "epoch",
        "Train Loss": "train_loss",
        "Val Loss": "val_loss"
    })
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    out_dir = output_dir / "plots" / model_name

    param_names = get_param_names(train_loader)

    with torch.no_grad():
        X_train, Y_train, _ = train_loader.dataset.tensors
        X_val, Y_val, _ = val_loader.dataset.tensors
        X_test, Y_test = test_loader.dataset.tensors

    # ─── optional test-time clipping ────────────────────────────────────────────
    model_for_test = model
    if config.get("clip_test", False):
        clip_layer = BoundedAct(
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=config["mask"],
        )
        model_for_test = torch.nn.Sequential(model, clip_layer).to(device)

    test_loss = evaluate(model_for_test, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    m = OBJ_test

    save_metadata_to_json(m, output_dir / "logs" / "metadata.json")
    diagnostic_inputs = {
        "Train": (X_train, Y_train),
        "Validation": (X_val, Y_val),
        "Test": (X_test, Y_test),
    }

    case = case_name.split("_")[2]

    generate_all_diagnostics(
        model=model_for_test,
        datasets=diagnostic_inputs,
        device=device,
        case_json=Path("data") / f"sample_{case}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

    logger.info(f"Completed DNN-STL training pipeline. Test loss: {test_loss:.6f}")

if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
