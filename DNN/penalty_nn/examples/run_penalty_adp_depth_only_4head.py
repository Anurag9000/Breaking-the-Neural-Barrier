import logging
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from Dyn_DNN4OPF.data.opf_loader import (
    get_data_loaders,
    load_case_bounds,
    load_output_bounds,
)
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility,
)
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json, load_logs
from Dyn_DNN4OPF.training.trainer import evaluate
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
from penalty_nn.models.adp_penalty_4head_depth_only import PenaltyADP_DEN_4Head
from Dyn_DNN4OPF.utils.repro import set_determinism

set_determinism(42)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG = {
    # ─── Training & model hyperparameters ─────────────────────────────────────
    "delta": 0,  # tolerance for decreasing patience counter
    "batch_size": 1024,
    "lr": 1e-3,
    "l1_lambda": 1e-4,
    "trials_depth": 5,
    "trials_width": 5,
    "l2_lambda": 1e-4,
    "gl_lambda": 1e-4,
    "regular_lambda": 1e-2,
    "loss_thr": 1e-3,
    "lambda_loss": 1.0,   # weight on data‐MSE term
    "lambda_eq":   1.0,   # weight on equality residuals
    "lambda_ineq": 1.0,   # weight on inequality violations
    # ── Raw violation thresholds ──────────────────────────────────────────────
    "dp_thr": 1e-2,  # ΔP mean-violation limit
    "dq_thr": 1e-2,  # ΔQ mean-violation limit
    "pg_thr": 1e-2,  # PG mean-violation limit
    "qg_thr": 1e-2,  # QG mean-violation limit
    "vm_thr": 1e-2,  # VM mean-violation limit
    "gap_thr": 5e-2,  # objective-gap limit
    # ── Constraint margin thresholds ─────────────────────────────────────────
    "constraint_thresholds": {
        "voltage_upper": 1e-3,
        "voltage_lower": 1e-3,
        "gen_real_upper": 1e-3,
        "gen_real_lower": 1e-3,
        "gen_reac_upper": 1e-3,
        "gen_reac_lower": 1e-3,
    },
    # ─── Expansion & drift hyperparameters ───────────────────────────────────
    "spl_thr": 0.25,
    "warmup_epochs": 100,
    "patience": 100,
    "max_epochs": 100000,
    "ex_k": 10,
    "init_width": 10,
    "init_depth": 2,
    "max_neurons": 100000,
    # ─── Optional test-time clipping (test-set only) ─────────────────────────
    "clip_test": False,
    # ─── Logging & checkpoints ───────────────────────────────────────────────
    "log_file": "train_adp_den.csv",
    # ─── Data / model identifiers ────────────────────────────────────────────
    "model": "ADP-DEN-Depth-Only-4Head-penalty",
    "case_name": "pglib_opf_case118_ieee",
    "train_samples": 5000,
    "val_samples": 1500,
    "test_samples": 1500,
    "batches": None,
    # ─── Equality-constraint tolerances ──────────────────────────────────────
    "p_tol": 1e-1,
    "q_tol": 1e-1,
}


def run_pipeline(cfg: dict) -> None:
    # Merge defaults + user overrides
    config = {**DEFAULT_CONFIG, **cfg}
    logger = logging.getLogger(__name__)
    logger.info(f"Using config: {config}")

    # Prepare output dirs
    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]
    out_dir = start_dir / "Results" / f"{config['model']}_{config['case_name']}"
    for sub in ("models", "logs", "plots", "diagnostics"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    config["log_file"] = str(out_dir / "logs" / config["log_file"])

    # Load data
    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"],
    )

    # Infer dims & build mask
    in_dim, out_dim = get_io_dims_from_loader(train_loader)
    n_bus = in_dim // 2
    n_gen = out_dim // 2 - n_bus
    if config.get("mask") is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Check bounds
    bounds_low, bounds_high = load_output_bounds(config["case_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], out_dim)
    config["bounds_low"] = bounds_low.to(device)
    config["bounds_high"] = bounds_high.to(device)
    # Final config fields
    config["dims"] = tuple([in_dim] + [config["init_width"]] *
                       config["init_depth"])
    config["n_classes"] = out_dim

    # Build soft constraints
    raw = load_case_bounds(config["case_name"])
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
        "y_bus": raw["y_bus"],
        "gen_bus_idx": torch.tensor(raw["gen_buses"], dtype=torch.long),
        "load_bus_idx": torch.tensor(raw["load_buses"], dtype=torch.long),
        "p_tol": config["p_tol"],
        "q_tol": config["q_tol"],
    }
    constraints = {"ineq": soft_ineq, "eq": eq}

    # Single-task loader
    task_loaders = [(train_loader, val_loader, test_loader, constraints)]

    # Instantiate & train
    model = PenaltyADP_DEN_4Head(SimpleNamespace(**config)).to(device)
    if hasattr(model, "bound_layer"):
        # Disable clipping for training / validation
        model.bound_layer.apply_bounds.fill_(False)

    all_logs = []
    for task_id, (tr, va, te, cons) in enumerate(task_loaders, start=1):
        model.current_task = task_id
        logger.info(f"--- Task {task_id} ---")
        test_perf = model.fit_task(
            tr,
            va,
            te,
            cons,
            max_epochs=config["max_epochs"],
        )
        logger.info(f"[Task {task_id}] Test perf: {test_perf:.6f}")

        all_logs.append(
            {
                "task": task_id,
                "test_perf": test_perf,
                "h1_dim": model.layers[0].out_features, 
                "h2_dim": model.layers[-1].out_features,
            }
        )

    # Save & plot
    save_logs_to_csv(all_logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        str(out_dir / "plots" / f"{config['model']}_loss.png"),
        test_plot_name=f"{config['model']}_testplot.png",
    )

    # Enable (optional) test-time clipping
    if hasattr(model, "bound_layer"):
        model.bound_layer.apply_bounds.fill_(config.get("clip_test", False))

    # Final evaluation & diagnostics
    model.eval()
    with torch.no_grad():
        X_tr, Y_tr, _ = train_loader.dataset.tensors
        X_va, Y_va, _ = val_loader.dataset.tensors
        X_te, Y_te = test_loader.dataset.tensors
    test_loss = evaluate(model, test_loader, label="Test", device=device)
    logger.info(f"Final Test MSE: {test_loss:.6f}")

    # Metadata & diagnostics
    save_metadata_to_json(OBJ_test, out_dir / "logs" / "metadata.json")
    df = load_logs(Path(config["log_file"]))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    generate_all_diagnostics(
        model=model,
        datasets={"Train": (X_tr, Y_tr), "Validation": (X_va, Y_va), "Test": (X_te, Y_te)},
        device=device,
        case_json=Path("data") / f"sample_{config['case_name'].split('_')[2]}.json",
        output_dir=str(out_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

    logger.info(f"Completed pipeline. Final Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    run_pipeline({})
