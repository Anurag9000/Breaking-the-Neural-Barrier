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
from Dyn_DNN4OPF.utils.logger_plotter import (
    save_logs_to_csv,
    plot_training_curves,
    generate_all_diagnostics,
)
from Dyn_DNN4OPF.models.adp_den_2head_width_only import ADP_DEN_2Head
from Dyn_DNN4OPF.utils.repro import set_determinism

set_determinism(42)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG = {
    # ─── Training & model hyperparameters ─────────────────────────────────────
    "delta": 0,
    "patience": 10,           # inner early-stopping patience
    "trials_depth": 10,       # outer patience (expansion failures)
    "warmup_epochs": 100,
    "spl_thr": 0.25,
    "ex_k": 10,
    "init_width": 10,
    "init_depth": 2,
    "max_neurons": 100000,
    "lr": 5e-4,
    "max_epochs": 100000,
    # ─── Dataset / batching ──────────────────────────────────────────────────
    "case_name": "GOC_118_10",
    "batches": 50,
    "train_samples": 10000,
    "val_samples": 2000,
    "test_samples": 2000,
    # ─── Logging / paths ─────────────────────────────────────────────────────
    "model": "ADP_DEN_2Head_width_only",
    "log_file": str(Path("Results") / "adp_den_2head_width_only_log.csv"),
    # ─── Constraint thresholds (diagnostics) ─────────────────────────────────
    "pg_thr": 1e-2,
    "qg_thr": 1e-2,
    "vm_thr": 1e-2,
    "gap_thr": 5e-2,
    "constraint_thresholds": {
        "voltage_upper": 1e-3,
        "voltage_lower": 1e-3,
        "gen_real_upper": 1e-3,
        "gen_real_lower": 1e-3,
        "gen_reac_upper": 1e-3,
        "gen_reac_lower": 1e-3,
    },
}


def run_pipeline(overrides: dict | None = None):
    config = DEFAULT_CONFIG | (overrides or {})
    logger = logging.getLogger("run_adp_den_2head_width_only")
    logger.setLevel(logging.INFO)

    # Data
    train_loader, val_loader, test_loader, task_loaders = get_data_loaders(
        case_name=config["case_name"],
        train_samples=config["train_samples"],
        val_samples=config["val_samples"],
        test_samples=config["test_samples"],
        batches=config["batches"],
    )

    # Infer dims & build mask
    in_dim, out_dim = get_io_dims_from_loader(train_loader)
    n_bus = in_dim // 2
    n_gen = out_dim // 2 - n_bus
    if config.get("mask") is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Bounds to device
    bounds_low, bounds_high = load_output_bounds(config["case_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], out_dim)
    config["bounds_low"] = bounds_low.to(device)
    config["bounds_high"] = bounds_high.to(device)

    # Final config dims
    config["dims"] = tuple([in_dim] + [config["init_width"]] * config["init_depth"])
    config["n_classes"] = out_dim

    # Soft constraints
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
    constraints = {
        "ineq": soft_ineq,
        "y_bus": raw["y_bus"],
        "gen_bus_idx": raw["gen_bus_idx"],
        "load_bus_idx": raw["load_bus_idx"],
    }

    # Model
    mcfg = SimpleNamespace(**config)
    model = ADP_DEN_2Head(mcfg).to(device)
    model.patience = config["patience"]
    model.lr = config["lr"]
    model.n_bus = n_bus
    model.n_gen = n_gen
    model.bound_layer.apply_bounds.fill_(False)

    # Train per task
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
                "h1_dim": model.fc1.out_features,
                "h2_dim": max(model.pq_fc2.out_features, model.vm_fc2.out_features),
            }
        )

    # Save & plot
    save_logs_to_csv(all_logs, config["log_file"])
    plot_training_curves(all_logs, title=config["model"])

    # Final diagnostics on last task
    out_dir = Path("Results") / f"{config['model']}_{config['case_name']}"
    X_tr, Y_tr = next(iter(train_loader))
    X_va, Y_va = next(iter(val_loader))
    X_te, Y_te = next(iter(test_loader))
    test_loss = generate_all_diagnostics(
        model,
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
