"""
run_dnn_elastic.py
==================

Elastic-Net (L1 + L2) DNN training pipeline.

Fixes applied
-------------
1. **Test-only clipping** – controlled by `clip_test` in the run-file config.  
2. **All train/val hyper-parameters** (epochs, max_epochs, patience, etc.) are read
   directly from the run-file config – nothing is hard-coded.  
3. **Pure-MSE logging only** – unchanged, but verified.  
Everything else is kept byte-for-byte identical.

"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
from penalty_nn.models.penalty_elastic import PenaltyDNNElastic
from Dyn_DNN4OPF.training.trainer import train_with_elastic, evaluate
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from Dyn_DNN4OPF.utils.logger_plotter import (
    save_logs_to_csv,
    plot_losses_from_csv,
    generate_all_diagnostics,
)
from Dyn_DNN4OPF.utils.plot_utils import (
    save_metadata_to_json,
    load_logs,
    get_param_names,
    plot_aggregate,
    plot_per_output,
)
from Dyn_DNN4OPF.utils.repro import set_determinism
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds

set_determinism()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

# ── central, editable run-file hyper-parameters ───────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # Optimiser / training
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 10000,
    "max_epochs": 10000,
    "patience": 100,
    # Network
    "hidden_dim": 112,
    "lambda1": 1.0,
    "lambda2": 1.0,
    "mask": None,
    # Penalty weights
    "lambda_loss": 1.0,
    "lambda_eq": 1.0,
    "lambda_ineq": 1.0,
    # Extras
    "clip_test": False,          # ← NEW: clip only at test if True
    "log_file": "train_elastic.csv",
    "model": "Elastic",
    # Data split
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": 27000,
    "val_samples": 1500,
    "test_samples": 1500,
    "batches": None,
}


# ── pipeline ───────────────────────────────────────────────────────────────
def run_pipeline(cfg: Dict[str, Any]) -> None:
    # merge defaults ← user cfg ← optuna
    optuna_params = load_optuna_best_params("best_hyperparameters_dnn_elastic.txt")
    config = {**DEFAULT_CONFIG, **cfg}
    config.update(optuna_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # data
    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"],
    )

    # output dirs
    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]
    out_root = start_dir / "Results" / f"{config['model']}_{config['case_name']}"
    for sub in ("models", "logs", "plots", "diagnostics"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    config["log_file"] = str(out_root / "logs" / config["log_file"])

    # I/O dims & bounds
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus
    bounds_low, bounds_high = load_output_bounds(config["case_name"])
    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    # model
    hidden_dim = config["hidden_dim"] or 4 * input_dim
    model = PenaltyDNNElastic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lambda1=config["lambda1"],
        lambda2=config["lambda2"],
        lambda_loss=config["lambda_loss"],
        lambda_eq=config["lambda_eq"],
        lambda_ineq=config["lambda_ineq"],
        use_bounds=True,              # BoundedAct created but disabled (apply_bounds=False)
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=config["mask"],
    ).to(device)

    # train (pure-MSE; patience & max_epochs from config)
    logs = train_with_elastic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        lambda1=config["lambda1"],
        lambda2=config["lambda2"],
        patience=config["patience"],
        device=device,
        save_path=str(out_root / "models" / f"best_model_Elastic_{config['case_name']}.pth"),
        max_epochs=config["max_epochs"],
    )

    save_logs_to_csv(logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        str(out_root / "plots" / "train_elastic_plot.png"),
        str(out_root / "plots" / "train_elastic_test_plot.png"),
    )

    # reload best checkpoint
    ckpt = out_root / "models" / f"best_model_Elastic_{config['case_name']}.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ── test-only clipping ────────────────────────────────────────────────
    model_for_test = model
    if config["clip_test"]:
        clip_layer = BoundedAct(bounds_low, bounds_high, config["mask"])
        clip_layer.apply_bounds.fill_(True)   # enable clipping
        model_for_test = torch.nn.Sequential(model, clip_layer).to(device)
        logger.info("Test-time clipping enabled via clip_test=True")

    test_loss = evaluate(model_for_test, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    # metadata + diagnostics
    save_metadata_to_json(OBJ_test, out_root / "logs" / "metadata.json")

    df = load_logs(Path(config["log_file"]))
    df = df.rename(columns={"Epoch": "epoch", "Train Loss": "train_loss", "Val Loss": "val_loss"})
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    plot_aggregate(df, config["model"], out_root / "plots" / config["model"])
    param_names = get_param_names(train_loader)
    plot_per_output(df, config["model"], out_root / "plots" / config["model"], param_names)

    X_tr, Y_tr, _ = train_loader.dataset.tensors
    X_va, Y_va, _ = val_loader.dataset.tensors
    X_te, Y_te = test_loader.dataset.tensors

    generate_all_diagnostics(
        model=model_for_test,   # respects test-time clipping
        datasets={
            "Train": (X_tr, Y_tr),
            "Validation": (X_va, Y_va),
            "Test": (X_te, Y_te),
        },
        device=device,
        case_json=Path("data") / f"sample_{config['case_name'].split('_')[2]}.json",
        output_dir=str(out_root),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

    logger.info("Pipeline complete. Test MSE: %.6f", test_loss)


if __name__ == "__main__":
    run_pipeline({})
