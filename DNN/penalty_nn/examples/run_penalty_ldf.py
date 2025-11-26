from Dyn_DNN4OPF.training.trainer import evaluate
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics, save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.repro import set_determinism
set_determinism()
from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
from Dyn_DNN4OPF.utils.config import get_io_dims_from_loader, default_mask
from penalty_nn.models.penalty_ldf import PenaltyLDF
from types import SimpleNamespace
import logging
import torch
from pathlib import Path
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

# Default configuration for LDF (Lagrangian Dual Penalty Network)
DEFAULT_CONFIG = {
    # Training hyperparameters
    "batch_size":        1024,
    "lr":                1e-3,
    "epochs":            10000,
    "delta":             1e-6,
    "patience":          100,
    "step_size":         1e-5,
    "kickin":            0,
    "update_freq":       500,
    "divide_by_counter": True,
    "exclude_keys":      [],
    "clip_test":         False,
    # ───────── Penalty-loss weights (λ) ─────────
    "l_loss":            1.0,   # λ₁ – baseline MSE term
    "l_eq":              1.0,   # λ₂ – equality residual term
    "l_ineq":            1.0,   # λ₃ – inequality violation term
    # Network architecture
    "h1_dim":            100,
    "h2_dim":            100,

    # Logging & checkpointing
    "log_file":          "train_ldf.csv",

    # Identifiers
    "model":             "LDF",
    "case_name":         "pglib_opf_case118_ieee",
    "train_samples":     27000,
    "val_samples":       1500,
    "test_samples":      1500,
    "batches":           None,

    # Base-loss and optimizer choices
    "loss":              "mse",
    "optimizer":         "adam",
    "activation":        "relu",
    "boundrepair":       "none",
    "weight_init_seed":  42,

    # Equality‐constraint tolerances
    "p_tol":             1e-1,
    "q_tol":             1e-1,

    # Constraint margin thresholds
    "constraint_thresholds": {
        "voltage_upper":  1e-3,
        "voltage_lower":  1e-3,
        "gen_real_upper": 1e-3,
        "gen_real_lower": 1e-3,
        "gen_reac_upper": 1e-3,
        "gen_reac_lower": 1e-3,
    },
}

def run_pipeline(cfg: dict) -> None:
    """
    Entry point for LDF training pipeline.
    """
    # Merge defaults, Optuna params, and user cfg
    optuna_params = load_optuna_best_params("best_hyperparameters_ldf.txt")
    config = {**DEFAULT_CONFIG, **optuna_params, **cfg}
    logger = logging.getLogger(__name__)
    logger.info(f"Using training config: {config}")

    # Prepare output directories
    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]
    model_name = config["model"]
    case_name  = config["case_name"]
    output_dir = start_dir / "Results" / f"{model_name}_{case_name}"
    for sub in ("models", "logs", "plots", "diagnostics"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    config["log_file"] = str(output_dir / "logs" / config["log_file"])

    # Load data
    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        case_name,
        config.get("train_samples"),
        config.get("val_samples"),
        config.get("test_samples"),
        config.get("batches"),
    )

    # I/O dims & mask
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus
    if config.get("mask") is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Pack dims & classes
    config["dims"]     = (input_dim, config["h1_dim"], config["h2_dim"])
    config["n_classes"] = output_dim

    # Build soft-inequality & equality constraints
    raw = load_case_bounds(case_name)
    ths = config["constraint_thresholds"]
    soft_ineq = {
        "v_max": raw["v_max"] + ths["voltage_upper"],
        "v_min": raw["v_min"] - ths["voltage_lower"],
        "p_max": raw["p_max"] + ths["gen_real_upper"],
        "p_min": raw["p_min"] - ths["gen_real_lower"],
        "q_max": raw["q_max"] + ths["gen_reac_upper"],
        "q_min": raw["q_min"] - ths["gen_reac_lower"],
    }
    eq = {
        "y_bus":        raw["y_bus"],
        "gen_bus_idx":  torch.tensor(raw["gen_buses"],  dtype=torch.long, device = device),
        "load_bus_idx": torch.tensor(raw["load_buses"], dtype=torch.long, device = device),
        "p_tol":        config["p_tol"],
        "q_tol":        config["q_tol"],
    }
    constraints = {"ineq": soft_ineq, "eq": eq}

    # Single-task loader
    task_loaders = [(train_loader, val_loader, test_loader, constraints)]

    # Instantiate and run LDF
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PenaltyLDF(SimpleNamespace(**config)).to(device)
    all_logs = []

    for task_id, (tr, va, te, cons) in enumerate(task_loaders, start=1):
        logger.info(f"--- Starting Task {task_id} ---")
        model.current_task = task_id
        test_perf = model.fit_task(tr, va, te, cons, max_epochs=config["epochs"])
        logger.info(f"[Task {task_id}] Test performance: {test_perf:.6f}")
        all_logs.append({
            "epoch":       task_id,
            "test_perf":   test_perf,
            "hidden1_dim": config["h1_dim"],
            "hidden2_dim": config["h2_dim"],
        })

    # Save & plot training losses
    save_logs_to_csv(all_logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        str(output_dir / "plots" / f"{model_name}_plot.png"),
        test_plot_name = f"{model_name}_plotting.png",
    )

    # Diagnostics: unpack OBJ_test and save metadata.json
    X_tr,  Y_tr,  _        = train_loader.dataset.tensors
    X_va,  Y_va,  _        = val_loader.dataset.tensors
    X_te,  Y_te            = test_loader.dataset.tensors

    # write out metadata.json for gap_objective
    logs_dir = output_dir / "logs"
    save_metadata_to_json(OBJ_test, logs_dir / "metadata.json")

    # now generate diagnostics
    generate_all_diagnostics(
        model=model,
        datasets={
            "Train":      (X_tr,  Y_tr),
            "Validation": (X_va,  Y_va),
            "Test":       (X_te,  Y_te),
        },
        device=device,
        case_json=Path('data')/f"sample_{case_name.split('_')[2]}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=model_name,
    )
    model_for_test = model
    if config.get("clip_test", False):
        bounds_lo, bounds_hi = load_output_bounds(case_name)
        clip_layer = BoundedAct(
            bounds_low=bounds_lo,
            bounds_high=bounds_hi,
            mask=config["mask"],
        )
        model_for_test = torch.nn.Sequential(model, clip_layer).to(device)
    # Final evaluation
    test_loss = evaluate(model_for_test, test_loader, label="Test", device=device)
    logger.info(f"Completed LDF training pipeline. Test loss: {test_loss:.6f}")

if __name__ == "__main__":
    run_pipeline({})
