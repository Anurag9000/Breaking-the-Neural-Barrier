from Dyn_DNN4OPF.training.trainer import evaluate
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
from Dyn_DNN4OPF.utils.plot_utils import (
    save_metadata_to_json, load_logs, get_param_names,
    plot_aggregate, plot_per_output,
)
from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
import logging
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds, load_output_bounds
from utils.bounded_act import BoundedAct 
from Dyn_DNN4OPF.utils.config import get_io_dims_from_loader, default_mask, check_bounds_compatibility
from Dyn_DNN4OPF.utils.logger_plotter import plot_losses_from_csv,generate_all_diagnostics
from Dyn_DNN4OPF.training.trainer import train_den_tasks
from Dyn_DNN4OPF.utils.repro import set_determinism
set_determinism()

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

# Default configuration for DEN (Dynamic Expansion Network)
DEFAULT_CONFIG = {
    # ─── Training & model hyperparameters ──────────────────────────────────────
    "batch_size":     1024,
    "lr":             1e-3,
    "l1_lambda":      1e-4,
    "l2_lambda":      1e-4,
    "gl_lambda":      1e-4,
    "regular_lambda": 1e-2,
    "loss_thr":       1e-4,    # MSE threshold before triggering expansion
    "clip_test":      False,
    # ── Raw violation thresholds (replacing pct_… keys) ────────────────────────
    "dp_thr":  1e-2,   # ΔP mean‐violation limit
    "dq_thr":  1e-2,   # ΔQ mean‐violation limit
    "pg_thr":  1e-2,   # PG mean‐violation limit
    "qg_thr":  1e-2,   # QG mean‐violation limit
    "vm_thr":  1e-2,   # VM mean‐violation limit
    "gap_thr": 5e-2,   # objective‐gap limit

    # ─── Expansion & drift hyperparameters ────────────────────────────────────
    "spl_thr":        0.25,    # semantic-drift splitting threshold (L2 drift)
    "warmup_epochs":  100,      # unconditional training epochs
    "patience":       100,      # additional epochs before checking for expansion
    "max_epochs":     10000,
    "ex_k":           10,       # neurons to add on each expansion
    "h1_dim":         10,
    "h2_dim":         10,
    "max_total_neurons": 1000,


    # ─── Logging & checkpointing ───────────────────────────────────────────────
    "log_file":       "train_den.csv",

    # ─── Data / model identifiers ─────────────────────────────────────────────
    "model":          "DEN-4HEAD",
    "case_name":      "pglib_opf_case118_ieee",
    "train_samples":  27000,
    "val_samples":    1500,
    "test_samples":   1500,
    "batches":        None,   # for sequential-task splits, if any

    # ─── Equality‐constraint tolerances ────────────────────────────────────────
    "p_tol":    1e-1,  # per‐bus ΔP tolerance
    "q_tol":    1e-1,  # per‐bus ΔQ tolerance

    # ──── Constraint margin thresholds ________________________________________ 

    "constraint_thresholds": {
        "voltage_upper":   1e-3,
        "voltage_lower":   1e-3,
        "gen_real_upper":  1e-3,
        "gen_real_lower":  1e-3,
        "gen_reac_upper":  1e-3,
        "gen_reac_lower":  1e-3,
    }
}

def run_pipeline(cfg: dict) -> None:
    """
    Entry point for DEN training pipeline.
    """
    # Merge defaults, Optuna params, and user cfg
    optuna_params = load_optuna_best_params("best_hyperparameters_den.txt")
    global test_set
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

    # Load Data
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

    # Bounds compatibility
    bounds_low, bounds_high = load_output_bounds(case_name=case_name)
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)
    clip_layer = BoundedAct(
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=torch.tensor(config["mask"], dtype=torch.bool, device=device),
    )  # apply_bounds default = False (training & val stay unclipped)

    # Final config packing
    config["dims"]      = (input_dim, config["h1_dim"], config["h2_dim"])
    config["n_classes"] = output_dim

    # Build soft inequality bounds & equality data
    raw = load_case_bounds(case_name)
    ths = config["constraint_thresholds"] = DEFAULT_CONFIG["constraint_thresholds"]
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
        "p_tol": config["p_tol"],
        "q_tol": config["q_tol"],
    }
    constraints = {"ineq": soft_ineq, "eq": eq}

    # Single-task loader list
    task_loaders = [(train_loader, val_loader, test_loader, constraints)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_den_tasks(
        config, task_loaders,
        max_epochs=config["max_epochs"]
    )
    if config.get("clip_test", True):
        clip_layer.apply_bounds.fill_(True)
        model_eval = torch.nn.Sequential(model, clip_layer).to(device)
    else:
        model_eval = model
    clip_layer.apply_bounds.fill_(True)

    plot_losses_from_csv(
        config["log_file"],
        str(output_dir / "plots" / "train_den_plot.png"),
        test_plot_name = "denplotting.png",
    )

    # 7) Evaluation & diagnostics
    model.eval()
    with torch.no_grad():
        X_tr, Y_tr, _ = train_loader.dataset.tensors
        X_va, Y_va, _ = val_loader.dataset.tensors
        X_te, Y_te    = test_loader.dataset.tensors
    model.current_task = 1
    test_loss = evaluate(model_eval, test_loader, label="Test", task_id=1, device=device)

    logger.info("Final Test MSE: %.6f", test_loss)

    # Metadata & save
    meta = OBJ_test
    save_metadata_to_json(meta, output_dir/'logs'/'metadata.json')

    # Aggregated & per-output plots
    df = load_logs(Path(config['log_file']))
    df = df.rename(columns={"Epoch": "epoch", "Train Loss": "train_loss", "Val Loss": "val_loss"})
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    plot_aggregate(df, config['model'], output_dir/'plots'/config['model'])
    plot_per_output(df, config['model'], output_dir/'plots'/config['model'], get_param_names(train_loader))

    # Central diagnostics
    generate_all_diagnostics(
        model=model_eval,
        datasets={
            "Train":       (X_tr, Y_tr),
            "Validation":  (X_va, Y_va),
            "Test":        (X_te, Y_te),
        },
        device=device,
        case_json=Path('data')/f"sample_{config['case_name'].split('_')[2]}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config['model'],
    )
    final_fc1_neurons = model.fc1.out_features
    final_fc2_neurons = model.fc2.out_features
    logger.info(f"Final neuron counts — FC1: {final_fc1_neurons}, FC2: {final_fc2_neurons}")

    logger.info(f"Completed DNN-DEN training pipeline. Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    run_pipeline({})
