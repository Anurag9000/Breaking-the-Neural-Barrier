from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs, get_param_names,
)
"""
run_dnn_mae.py
==========

Standard fully connected neural network (MAE - Single Task Learning).

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
from Dyn_DNN4OPF.models.dnn_mae import FullyConnectedNet
from Dyn_DNN4OPF.training.trainer import train, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from Dyn_DNN4OPF.utils.config import (
    default_mask,
    check_bounds_compatibility,
    get_io_dims_from_loader,
)
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.utils.repro import set_determinism
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

set_determinism()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int, device = device)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size":     1024,
    "lr":             1e-3,
    "epochs":         10000,
    "patience":       100,
    "max_epochs":     10000,      # forwarded verbatim to trainer
    "hidden_dim":     270,
    "log_file":       "train_mae.csv",
    "mask":           None,
    "model":          "MAE",
    "case_name":      "pglib_opf_case118_ieee",
    "train_samples":  27000,
    "val_samples":    1500,
    "test_samples":   1500,
    "batches":        None,
    "clip_test":      False,      # ⇢ wrap in BoundedAct only during Test if True
}

def run_pipeline(cfg: Dict[str, Any]) -> None:
    config: Dict[str, Any] = {**DEFAULT_CONFIG, **cfg}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")
    logger.info("Initializing DNN-MAE training pipeline")

    # Load Optuna hyperparameters if available
    optuna_params_file = "best_hyperparameters_dnn_mae.txt"
    optuna_params = load_optuna_best_params(optuna_params_file)

    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config.get("train_samples", 27000),
        config.get("val_samples", 1500),
        config.get("test_samples", 1500),
        config.get("batches", None),
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

    bounds_low, bounds_high = load_output_bounds(case_name=case_name)
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus

    # Use default mask if not provided
    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Verify bounds and mask dimensions match output_dim
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    hidden_dim = config.get("hidden_dim") or 4 * input_dim

    model = FullyConnectedNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bounds=True,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=config["mask"],
    )

    # Disable clipping during training
    if hasattr(model, "bound_layer"):
        model.bound_layer.apply_bounds.fill_(False)

    logger.info(
        f"Initialized FullyConnectedNet(input={input_dim}, hidden={hidden_dim}, "
        f"output={output_dim}, use_bounds=True)"
    )
    model.to(device)

    # -------------------- training --------------------
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

    # Save checkpoint
    ckpt_path = output_dir / "models" / f"best_model_{model_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved MAE checkpoint to {ckpt_path}")

    # Persist training logs
    save_logs_to_csv(logs, config["log_file"])

    # Reload best weights (optional)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # -------------------- test-time clipping wrapper --------------------
    if config["clip_test"]:
        # keep internal layer OFF, add external bounded layer
        if hasattr(model, "bound_layer"):
            model.bound_layer.apply_bounds.fill_(False)
        clip_layer = BoundedAct(
            bounds_low.to(device),
            bounds_high.to(device),
            torch.tensor(config["mask"], dtype=torch.bool, device=device),
        )
        clip_layer.apply_bounds.fill_(True)
        test_model = torch.nn.Sequential(model, clip_layer).to(device)
    else:
        # enable internal clipping for evaluation if desired
        if hasattr(model, "bound_layer"):
            model.bound_layer.apply_bounds.fill_(True)
        test_model = model

    test_model.eval()

    # -------------------- plots --------------------
    p = output_dir / "plots"
    plot_losses_from_csv(
        config["log_file"],
        str(p / "train_mae_plot.png"),
        str(p / "train_mae_train_plot.png"),
    )

    csv_path = output_dir / "logs" / Path(config["log_file"]).name
    df = load_logs(csv_path)
    df = df.rename(
        columns={
            "Epoch": "epoch",
            "Train Loss": "train_loss",
            "Val Loss": "val_loss",
        }
    )
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )  # snake_case

    param_names = get_param_names(train_loader)
    # plot_aggregate(df, model_name, output_dir / "plots" / model_name)
    # plot_per_output(df, model_name, output_dir / "plots" / model_name, param_names)

    # -------------------- evaluation --------------------
    with torch.no_grad():
        X_train, Y_train, _ = train_loader.dataset.tensors
        X_val, Y_val, _ = val_loader.dataset.tensors
        X_test, Y_test = test_loader.dataset.tensors

    test_loss = evaluate(test_model, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    save_metadata_to_json(OBJ_test, output_dir / "logs" / "metadata.json")

    diagnostic_inputs = {
        "Train": (X_train, Y_train),
        "Validation": (X_val, Y_val),
        "Test": (X_test, Y_test),
    }

    case = case_name.split("_")[2]

    generate_all_diagnostics(
        model=test_model,
        datasets=diagnostic_inputs,
        device=device,
        case_json=Path("data") / f"sample_{case}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

    logger.info(f"Completed DNN-MAE training pipeline. Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
