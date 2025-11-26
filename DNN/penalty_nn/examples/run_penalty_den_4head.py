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
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from Dyn_DNN4OPF.utils.config import get_io_dims_from_loader, default_mask, check_bounds_compatibility
from Dyn_DNN4OPF.utils.logger_plotter import plot_losses_from_csv,generate_all_diagnostics
from penalty_nn.training.penalty_trainer import train_penalty_den_tasks
from penalty_nn.models.penalty_den import PenaltyDEN
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
    # ─── Composite-loss weights (λ₁, λ₂, λ₃) ────────────────────────────────────
    "l_loss":  1.0,   # weight on MSE
    "l_eq":    1.0,   # weight on equality residuals ‖h‖₂
    "l_ineq":  1.0,   # weight on inequality violations ‖g⁺‖₂
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
    "model":          "PenaltyDEN-4Head",
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
    },
    "lambda_loss":  1,    # will be set from l_loss
    "lambda_eq":    1,    # will be set from l_eq
    "lambda_ineq":  1,    # will be set from l_ineq
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_pipeline(cfg: dict) -> None:
    optuna_params = load_optuna_best_params("best_hyperparameters_den.txt")
    config = {**DEFAULT_CONFIG, **optuna_params, **cfg}
    config["lambda_loss"] = config.pop("l_loss", config["lambda_loss"])
    config["lambda_eq"]   = config.pop("l_eq",   config["lambda_eq"])
    config["lambda_ineq"] = config.pop("l_ineq", config["lambda_ineq"])
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    case_name = config["case_name"]
    train_loader, val_loader, test_loader = get_data_loaders(
        case_name, batch_size=config["batch_size"]
    )
    n_gen, n_bus, in_dim, out_dim = get_io_dims_from_loader(train_loader)

    raw = load_case_bounds(case_name)
    ths = config.setdefault("constraint_thresholds", DEFAULT_CONFIG["constraint_thresholds"])
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
        "gen_bus_idx":  torch.tensor(raw["gen_buses"],  dtype=torch.long, device=device),
        "load_bus_idx": torch.tensor(raw["load_buses"], dtype=torch.long, device=device),
        "p_tol": config["p_tol"],
        "q_tol": config["q_tol"],
    }
    constraints = {"ineq": soft_ineq, "eq": eq}

    task_loaders = [(train_loader, val_loader, test_loader, constraints)]
    model = train_penalty_den_tasks(
        config, task_loaders,
        max_epochs=int(config["max_epochs"]),
        delta=config.get("delta", None),
    )

    clip_layer = BoundedAct(
        p_min=soft_ineq["p_min"], p_max=soft_ineq["p_max"],
        q_min=soft_ineq["q_min"], q_max=soft_ineq["q_max"],
        v_min=soft_ineq["v_min"], v_max=soft_ineq["v_max"],
        apply_bounds=False,
    ).to(device)
    if config.get("clip_test", True):
        clip_layer.apply_bounds.fill_(True)
        model_eval = torch.nn.Sequential(model, clip_layer).to(device)
    else:
        model_eval = model

    plot_losses_from_csv(
        config["log_file"],
        str(output_dir / "plots" / "train_den_plot.png"),
        test_plot_name="denplotting.png",
    )

    model_eval.eval()
    test_loss, *_ = evaluate(model_eval, test_loader, device)
    generate_all_diagnostics(
        model_eval, test_loader,
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config['model'],
    )
    logging.getLogger(__name__).info(f"Completed penalty DEN (4-head). Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    run_pipeline({})
