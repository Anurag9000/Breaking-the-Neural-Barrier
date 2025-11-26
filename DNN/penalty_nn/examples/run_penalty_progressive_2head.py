"""
===============================================================================
Progressive Neural Network (PNN) Implementation
===============================================================================

This module implements the Progressive Neural Network architecture, as introduced by 
Rusu et al., 2016, designed for continual learning by **adding a new network column** per 
task and leveraging lateral connections for transfer while preventing forgetting.

"""
import logging
from typing import Dict, Any
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
from typing import List
from penalty_nn.models.penalty_progressive_2head import PenaltyDNNProgressive_2Head
from penalty_nn.training.penalty_trainer import train_penalty_progressive, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility,
)
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from pathlib import Path
from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs,
    get_param_names,
    plot_aggregate,
    plot_per_output,
)
from Dyn_DNN4OPF.utils.repro import set_determinism

set_determinism()
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct  # ← added for test-time clipping
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 10000,
    "patience": 100,
    "max_epochs": 10000,          # ← forward verbatim to trainer
    "hidden_dim": 112,
    "log_file": "train_progressive.csv",
    "mask": None,
    "model": "PenaltyProgressive",
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": 27000,
    "val_samples": 1500,
    "test_samples": 1500,
    "batches": None,
    "clip_test": False,           # ← new flag: enable test-only clipping
    "lambda_loss": 1.0,           # ← weight for baseline MSE
    "lambda_eq":   1.0,           # ← weight for equality residuals
    "lambda_ineq": 1.0,           # ← weight for inequality violations
}

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# relu_mask = torch.tensor([...])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int, device = device)


def run_pipeline(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Progressive Network training pipeline")

    optuna_params_file = "best_hyperparameters_dnn_mtl.txt"
    optuna_params = load_optuna_best_params(optuna_params_file)

    config: Dict[str, Any] = {**DEFAULT_CONFIG, **optuna_params, **cfg}
    logger.info(f"Using training config: {config}")

    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[2]  # Move two levels up if 'examples' is in the path

    model_name = config["model"]
    case_name = config["case_name"]
    output_dir = start_dir / "Results" / f"{model_name}_{case_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "logs", "plots", "diagnostics"):
        (output_dir / sub).mkdir(exist_ok=True)

    # Override log-file path inside config (store absolute path once)
    config["log_file"] = str(output_dir / "logs" / config["log_file"])
    log_file_path = Path(config["log_file"])  # canonical path object

    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"],
    )

    bounds_low, bounds_high = load_output_bounds(case_name=case_name)
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus

    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    hidden_dim = config["hidden_dim"] or 4 * input_dim

    model = PenaltyDNNProgressive_2Head(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=_to_tensor_mask(config["mask"]),
        lambda_loss=config["lambda_loss"],
        lambda_eq=config["lambda_eq"],
        lambda_ineq=config["lambda_ineq"],
        case_name=config["case_name"],
        clip_test=config["clip_test"],
    ).to(device)
    logger.info(
        f"Initialized PenaltyDNNProgressive(input={input_dim}, hidden={hidden_dim}, use_bounds=True)"
    )

    logger.info("Starting DNN-Progressive training pipeline…")
    logs = train_penalty_progressive(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        log_file=log_file_path,  # avoid double-joining directories
        save_path=output_dir / "models" / f"best_model_{model_name}_{case_name}.pth",
        max_epochs=config["max_epochs"],
    )

    # Save trained checkpoint
    ckpt_path = output_dir / "models" / f"best_model_Progressive_{case_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved Progressive checkpoint to {ckpt_path}")

    save_logs_to_csv(logs, log_file_path)

    # Plot losses (MSE-only)
    pdir = output_dir / "plots"
    plot_losses_from_csv(
        log_file_path,
        str(pdir / "train_progressive_plot.png"),
        str(pdir / "train_progressive_test_plot.png"),
    )

    # Diagnostics: aggregate / per-output
    df = load_logs(log_file_path)
    df = df.rename(
        columns={"Epoch": "epoch", "Train Loss": "train_loss", "Val Loss": "val_loss"}
    )
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    plot_aggregate(df, model_name, pdir)
    plot_per_output(df, model_name, pdir, get_param_names(train_loader))

    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ── optional test-time clipping ──────────────────────────────────────────
    # NOTE: PenaltyDNNProgressive includes per-column bound layers already.
    # Wrapping with nn.Sequential would break (x, task_id) signatures;
    # keep the model as-is for evaluation.
    model_test = model

    # Evaluate Test MSE
    test_loss = evaluate(model_test, test_loader, label="Test", task_id=0, device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    # Save metadata
    save_metadata_to_json(OBJ_test, output_dir / "logs" / "metadata.json")

    # Diagnostics across splits
    with torch.no_grad():
        X_train, Y_train, _ = train_loader.dataset.tensors
        X_val, Y_val, _ = val_loader.dataset.tensors
        X_test, Y_test, _ = test_loader.dataset.tensors  # ← ensure 3-tuple

    diagnostic_inputs = {
        "Train": (X_train, Y_train),
        "Validation": (X_val, Y_val),
        "Test": (X_test, Y_test),
    }
    case_short = case_name.split("_")[2]

    generate_all_diagnostics(
        model=model_test,
        datasets=diagnostic_inputs,
        device=device,
        case_json=Path("data") / f"sample_{case_short}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        task_id=0,
        model_name=config["model"],
    )

    logger.info("Completed DNN-Progressive training pipeline.")


if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
