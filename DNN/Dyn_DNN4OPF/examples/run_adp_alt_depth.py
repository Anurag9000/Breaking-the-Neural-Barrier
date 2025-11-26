"""
CONTROL init_width and init_depth
The number of hidden layers = init_depth
Each hidden layer width = init_width
(Initializer only; ADP grows the net.)

Drop‑in runner for **Alternating ADP (Depth→Width), 1‑Head**.
Mirrors the stock run_adp_depth.py flow: data → constraints → model → train → logs/plots/diagnostics.
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
from torch import nn

# Repo root on path
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
from Dyn_DNN4OPF.utils.repro import set_determinism

# Import the Alternating ADP (Depth→Width) controller (1‑Head)
from Dyn_DNN4OPF.models.adp_den_alt_depth_1head import ADP_ALT_DEPTH_1Head

set_determinism(42)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict = {
    # ─── Training & model hyperparameters ─────────────────────────────────────
    "delta": 0.0,
    "trials_depth": 10,
    "trials_width": 10,
    "batch_size": 1024,
    "lr": 1e-3,
    "l1_lambda": 1e-4,
    "l2_lambda": 1e-4,
    "gl_lambda": 1e-4,
    "regular_lambda": 1e-2,
    "loss_thr": 1e-3,
    # ── Raw violation thresholds ──────────────────────────────────────────────
    "dp_thr": 1e-2,
    "dq_thr": 1e-2,
    "pg_thr": 1e-2,
    "qg_thr": 1e-2,
    "vm_thr": 1e-2,
    "gap_thr": 5e-2,
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
    "drift_lambda": 1e-2,
    "max_grad": 1.0,
    "ex_k": 8,
    "max_depth": 16,
    "max_neurons": 4096,
    "patience": 20,
    "warmup_epochs": 0,
    "max_epochs": 500,
    # ─── Init arch & logging ─────────────────────────────────────────────────
    "init_width": 64,
    "init_depth": 1,
    "log_file": "adp_alt_depth.csv",
    # ─── Data / model identifiers ────────────────────────────────────────────
    "model": "ADP-ALT-DEPTH",
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": 5000,
    "val_samples": 1500,
    "test_samples": 1500,
    "batches": None,
    # ─── Equality-constraint tolerances ──────────────────────────────────────
    "p_tol": 1e-1,
    "q_tol": 1e-1,
}


def run_pipeline(cfg: Dict) -> None:
    # Merge defaults + overrides
    config = {**DEFAULT_CONFIG, **cfg}
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
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
    in_dim, out_dim = get_io_dims_from_loader(train_loader)
    n_bus = in_dim // 2
    n_gen = out_dim // 2 - n_bus
    config.setdefault("mask", default_mask(n_gen, n_bus))

    # Check bounds
    bounds_low, bounds_high = load_output_bounds(config["case_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], out_dim)
    config["bounds_low"], config["bounds_high"] = bounds_low.to(device), bounds_high.to(device)
    config["dims"] = tuple([in_dim] + [config["init_width"]] * config["init_depth"])  # init arch
    config["n_classes"] = out_dim

    # Build soft constraints for possible diagnostics & future use
    raw = load_case_bounds(config["case_name"])  # includes y_bus, limits, indices
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
    # Move to device
    for k in list(soft_ineq.keys()): soft_ineq[k] = soft_ineq[k].to(device)
    for k in ("y_bus", "gen_bus_idx", "load_bus_idx"):
        eq[k] = eq[k].to(device)

    constraints = {"ineq": soft_ineq, "eq": eq}

    # Single-task loader bundle
    task_loaders = [(train_loader, val_loader, test_loader, constraints)]

    # Instantiate controller + inner model
    controller = ADP_ALT_DEPTH_1Head(
        in_dim=in_dim,
        out_dim=out_dim,
        width=config["init_width"],
        depth=config["init_depth"],
        ex_k=config["ex_k"],
        max_neurons=config["max_neurons"],
        max_depth=config["max_depth"],
        delta=config["delta"],
        patience_width=config["trials_width"],
        patience_depth=config["trials_depth"],
        device=device,
    )
    net = controller.model  # underlying nn.Module

    # Disable clipping on bound layers if present
    if hasattr(net, "bound_layer"):
        net.bound_layer.apply_bounds.fill_(False)

    # Train on single task
    all_logs = []
    for task_id, (tr, va, te, cons) in enumerate(task_loaders, start=1):
        logger.info(f"--- Task {task_id} ---")
        _ = controller.fit(
            tr,
            va,
            loss_fn=nn.MSELoss(),
            physics_metric=None,  # 1‑Head uses MSE‑first acceptance
            max_global_epochs=config["max_epochs"],
        )
        test_loss, _ = evaluate(net, te, nn.MSELoss(), device=device)
        logger.info(f"[Task {task_id}] Test MSE: {test_loss:.6f}")
        total_hidden = controller.model.fc2.depth * controller.model.width
        all_logs.append({"task": task_id, "test_perf": float(test_loss), "total_hidden": int(total_hidden)})

    # Save & plot
    save_logs_to_csv(all_logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        str(out_dir / "plots" / f"{config['model']}_loss.png"),
        test_plot_name=f"{config['model']}_testplot.png",
    )

    # Metadata
    X_tr, Y_tr = next(iter(train_loader))
    X_va, Y_va = next(iter(val_loader))
    X_te, Y_te = next(iter(test_loader))
    save_metadata_to_json(
        {
            "model": config["model"],
            "case": config["case_name"],
            "in_dim": in_dim,
            "out_dim": out_dim,
            "n_bus": n_bus,
            "n_gen": n_gen,
        },
        str(out_dir / "plots" / f"{config['model']}_metadata.json"),
    )

    # Diagnostics (uses the inner nn.Module)
    generate_all_diagnostics(
        model=net,
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
