"""
run_dnn_dc3.py: DC3 training pipeline for Dyn_DNN4OPF.
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import json
from Dyn_DNN4OPF.utils.repro import set_determinism

set_determinism(42)

from Dyn_DNN4OPF.training.trainer import train_dc3, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics, save_logs_to_csv
from Dyn_DNN4OPF.utils.plot_utils import (
    save_metadata_to_json,
    load_logs,
    get_param_names,
    plot_aggregate,
    plot_per_output,
)
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case, objective

from Dyn_DNN4OPF.data.opf_loader import (
    load_case_bounds,
    load_output_bounds,
    get_data_loaders,
    load_cost_coeff,
    DATASET_ROOT,
)
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility,
)
from penalty_nn.models.penalty_dc3 import PenaltyDNN_DC3
from Dyn_DNN4OPF.utils.dc3_utils import DC3Helper

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

# ─── Default configuration for DC3 ───────────────────────────────────────────
DEFAULT_CONFIG = {
    # Training hyperparameters
    "epochs": 10000,
    "batch_size": 1024,
    "lr": 1e-3,
    "hidden_size": 64,
    "use_bounds": True,
    "patience": 100,
    "max_epochs": 10000,
    # DC3 correction hyperparameters
    "corrTrainSteps": 5,
    "corrTestMaxSteps": 50,
    "corrLr": 1e-3,
    "corrEps": 1e-3,
    "corrMomentum": 0.9,
    "softWeight": 1.0,
    "softWeightEqFrac": 0.5,
    "use_partial": True,
    "lambda_loss": 1.0,   # weight on baseline_loss
    "lambda_eq":   1.0,   # weight on ‖h‖₂
    "lambda_ineq": 1.0,   # weight on ‖g⁺‖₂
    # Logging & checkpoints
    "log_file": "train_dnn_dc3.csv",
    "clip_test": False,
    # Data identifiers
    "case_name": "pglib_opf_case30_ieee",
    "train_samples": None,
    "val_samples": None,
    "test_samples": None,
    "batches": None,
    # Constraint margin thresholds for diagnostics
    "dp_thr": 1e-2,
    "dq_thr": 1e-2,
    "pg_thr": 1e-2,
    "qg_thr": 1e-2,
    "vm_thr": 1e-2,
    "gap_thr": 5e-2,
    # Constraint tolerances (equality)
    "p_tol": 1e-1,
    "q_tol": 1e-1,
    # Threshold margins
    "constraint_thresholds": {
        "voltage_upper": 1e-3,
        "voltage_lower": 1e-3,
        "gen_real_upper": 1e-3,
        "gen_real_lower": 1e-3,
        "gen_reac_upper": 1e-3,
        "gen_reac_lower": 1e-3,
    },
}

# Append project root to path if needed
start_dir = Path.cwd()
if start_dir.name == "examples":
    start_dir = start_dir.parent
sys.path.append(str(start_dir))


def run_pipeline(cfg: dict) -> None:
    """
    Entry point for DC3 training and evaluation.
    """
    # Merge default config with user overrides
    config = {**DEFAULT_CONFIG, **cfg}
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Using config: {config}")

    # Prepare output directories
    model_name = "DC3"
    case_name = config["case_name"]
    output_dir = start_dir / "Results" / f"{model_name}_{case_name}"
    for sub in ("models", "logs", "plots", "diagnostics"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    config["log_file"] = str(output_dir / "logs" / config["log_file"])

    # Load data loaders
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
    if not config.get("mask"):
        config["mask"] = default_mask(n_gen, n_bus)

    # Check output bounds compatibility
    bounds_low, bounds_high = load_output_bounds(case_name)
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    # ─── Build & register PDL “raw” case ────────────────────────────────────
    raw = load_case_bounds(case_name)

    # augment with per-bus loads and cost coeffs
    sample_file = DATASET_ROOT / f"sample_{case_name.split('_')[2]}.json"
    data_json = json.loads(sample_file.read_text())

    # per-bus real/reactive loads
    pd = torch.zeros(len(data_json["grid"]["nodes"]["bus"]), device = device)
    qd = torch.zeros_like(pd, device = device)
    receivers = data_json["grid"]["edges"]["load_link"]["receivers"]
    loads = data_json["grid"]["nodes"]["load"]
    for idx, bus in enumerate(receivers):
        pd[bus] += loads[idx][0]
        qd[bus] += loads[idx][1]
    raw["pd"], raw["qd"] = pd, qd

    # cost coeffs
    raw.update(load_cost_coeff(data_json))
    init_from_case(raw)

    # ─── Build DC3 constraint dict ──────────────────────────────────────────
    ths = config["constraint_thresholds"]
    soft_ineq = {
        "v_max": raw["v_max"] + ths["voltage_upper"],
        "v_min": raw["v_min"] - ths["voltage_lower"],
        "p_max": raw["p_max"] + ths["gen_real_upper"],
        "p_min": raw["p_min"] - ths["gen_real_lower"],
        "q_max": raw["q_max"] + ths["gen_reac_upper"],
        "q_min": raw["q_min"] - ths["gen_reac_lower"],
    }
    eq_constraints = {
        "y_bus": raw["y_bus"],
        "gen_bus_idx": torch.tensor(raw["gen_buses"], dtype=torch.long, device = device),
        "load_bus_idx": torch.tensor(raw["load_buses"], dtype=torch.long, device = device),
        "p_tol": config["p_tol"],
        "q_tol": config["q_tol"],
    }
    constraints = {"ineq": soft_ineq, "eq": eq_constraints}

    # Build helper + model
    helper = DC3Helper(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        constraints=constraints,
        xdim=input_dim,
        ydim=output_dim,
        nknowns=2 * n_bus,
        n_bus=n_bus,
        n_gen=n_gen,
        use_partial=config["use_partial"],
        obj_fn=objective,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PenaltyDNN_DC3(
        data=helper,
        hidden_dim=config["hidden_size"],
        corr_steps_train=config["corrTrainSteps"],
        corr_steps_test=config["corrTestMaxSteps"],
        corr_lr=config["corrLr"],
        corr_eps=config["corrEps"],
        soft_weight=config["softWeight"],
        soft_eq_frac=config["softWeightEqFrac"],
        use_partial=config["use_partial"],
        use_bounds=config["use_bounds"],
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=config["mask"],
        lambda_loss=config["lambda_loss"],
        lambda_eq=config["lambda_eq"],
        lambda_ineq=config["lambda_ineq"],
    ).to(device)

    # ─── Train ──────────────────────────────────────────────────────────────
    train_dc3(
        model,
        model.data,
        config,
        str(output_dir / "models"),
        max_epochs=config["max_epochs"],
    )

    # ─── optional test-time clipping ────────────────────────────────────────
    model_for_test = model
    if config.get("clip_test", False) and hasattr(model, "bound_layer"):
        if hasattr(model.bound_layer, "apply_bounds"):
            model.bound_layer.apply_bounds.data = torch.tensor(True)
        else:
            from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

            clip_layer = BoundedAct(bounds_low, bounds_high, config["mask"])
            model_for_test = torch.nn.Sequential(model, clip_layer).to(device)

    model_for_test.eval()
    test_loss = evaluate(model_for_test, test_loader, label="Test", device=device)
    logger.info(f"Final Test MSE: {test_loss:.6f}")

    # ─── Diagnostics & plotting ─────────────────────────────────────────────
    X_tr, Y_tr, _ = train_loader.dataset.tensors
    X_va, Y_va, _ = val_loader.dataset.tensors
    X_te, Y_te = test_loader.dataset.tensors

    save_metadata_to_json(OBJ_test, output_dir / "logs" / "metadata.json")
    save_logs_to_csv([{"test_loss": test_loss}], config["log_file"])

    df = load_logs(Path(config["log_file"]))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    generate_all_diagnostics(
        model=model_for_test,
        datasets={
            "Train": (X_tr, Y_tr),
            "Validation": (X_va, Y_va),
            "Test": (X_te, Y_te),
        },
        device=device,
        case_json=start_dir / "data" / f"sample_{case_name.split('_')[2]}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=model_name,
    )

    logger.info(f"Completed DC3 training pipeline. Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    run_pipeline({})
