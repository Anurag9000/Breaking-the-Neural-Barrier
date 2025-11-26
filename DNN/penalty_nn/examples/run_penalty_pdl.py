"""
===============================================================================
DNN-PDL: Primal-Dual Learning Run Script
===============================================================================

This script trains and evaluates a *Primal–Dual Learning* (PDL) network for
AC-OPF emulation, following the methodology of Vuffray et al., 2021
(“DC3: Constraint-Compliant Deep Control for Physical Systems”) but adapted to
the PDL setting.

───────────────────────────────────────────────────────────────────────────────
Key additions in this revision
───────────────────────────────────────────────────────────────────────────────
1. **`clip_test` flag** (default False) in `DEFAULT_CONFIG`  
   – When True, the trained *primal* network is wrapped in  
   `torch.nn.Sequential(primal_net, BoundedAct)` **only for test-set
   evaluation / diagnostics** so that selected outputs are hard-clipped to their
   physical limits.

2. **Patience & max_epochs** are still read verbatim from
   `DEFAULT_CONFIG["pdl"]` and forwarded unchanged to `PDLTrainer`.

3. **MSE-only logging** – The script continues to compute, log and CSV-store
   *only* the final test-set mean-squared error; no extra metrics are added.

All other behaviour remains byte-for-byte identical to the original file.
===============================================================================
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
import torch
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case
from torch.utils.data import DataLoader
from penalty_nn.models.penalty_pdl import PenaltyDNNPDL
from penalty_nn.training import penalty_pdl_trainer
from Dyn_DNN4OPF.data.opf_loader import get_data_loaders, load_output_bounds
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility,
)
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv
from Dyn_DNN4OPF.utils.repro import set_determinism
from Dyn_DNN4OPF.data.opf_loader import (
    load_case_bounds,
    DATASET_ROOT,
    load_cost_coeff,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct  # ← NEW import

set_determinism()
import json

# -----------------------------------------------------------------------------
# Default configuration for Primal-Dual Learning (PDL)
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Data
    "batch_size": 1024,
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": None,
    "val_samples": None,
    "test_samples": None,
    "batches": None,
    "clip_test": False,  # ← NEW flag ─ clip only on Test set
    # Penalty-loss weights (independent of vanilla model)
    "l_loss": 1.0,   # λ₁ – baseline objective
    "l_eq":   1.0,   # λ₂ – equality residuals
    "l_ineq": 1.0,   # λ₃ – inequality violations
    # Model
    "hidden_primal": 64,
    "hidden_dual": [64, 64],

    # Optimizer
    "optimizer": {
        "lr_primal": 1e-3,
        "lr_dual": 1e-3,
    },

    # PDL hyperparams
    "pdl": {
        "rho_init": 1.0,
        "rho_max": 1e4,
        "alpha": 10,
        "tau": 0.5,
        "outer_iters": 10,
        "inner_iters": 5,
        "max_epochs": 10000,
        "patience": 100,
    },

    # Logging & saving
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_file": "train_pdl.csv",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _add_load_and_cost(raw: dict, case_name: str) -> dict:
    """Augment `raw` with pd/qd and cost_q/l/c straight from the sample JSON."""
    sample = DATASET_ROOT / f"sample_{case_name.split('_')[2]}.json"
    data = json.loads(Path(sample).read_text())

    # ---------------- PD / QD per BUS -----------------
    n_bus = len(data["grid"]["nodes"]["bus"])
    pd, qd = torch.zeros(n_bus, device = device), torch.zeros(n_bus, device = device)
    receivers = data["grid"]["edges"]["load_link"]["receivers"]
    loads = data["grid"]["nodes"]["load"]
    for load_idx, bus in enumerate(receivers):
        pd[bus] += loads[load_idx][0]
        qd[bus] += loads[load_idx][1]
    raw["pd"], raw["qd"] = pd, qd

    # ---------------- Gen-cost coefficients -----------
    raw.update(load_cost_coeff(data))  # adds cost_q, cost_l, cost_c
    return raw


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def evaluate_primal(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Compute mean MSE loss of the primal network over loader.
    """
    mse_loss = torch.nn.MSELoss()
    model.to(device).eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += mse_loss(pred, yb).item() * xb.size(0)
            count += xb.size(0)
    return total / count if count > 0 else 0.0


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_pipeline(cfg: dict) -> None:
    # Merge defaults
    config = {**DEFAULT_CONFIG, **cfg}
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using config: {config}")

    # Prepare output dirs
    start_dir = Path.cwd()
    output_dir = start_dir / "Results" / f"PDL_{config['case_name']}"
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"],
    )

    # Dimensions and mask
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus
    if "mask" not in config or config.get("mask") is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Bounds check
    bounds_low, bounds_high = load_output_bounds(case_name=config["case_name"])
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    raw = _add_load_and_cost(load_case_bounds(config["case_name"]), config["case_name"])
    init_from_case(raw)

    # Pack model config
    model_cfg = SimpleNamespace(
        model=SimpleNamespace(
            input_dim=input_dim,
            hidden_primal=config["hidden_primal"],
            hidden_dual=config["hidden_dual"],
            output_dim=output_dim,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=torch.tensor(config["mask"], dtype=torch.bool, device = device),
            num_g=output_dim,  # inequality dims = output dims
            num_h=2 * n_bus,  # equality dims = 2 * n_bus (P/Q balance)
        ),
        optimizer=SimpleNamespace(**config["optimizer"]),
        pdl=SimpleNamespace(**config["pdl"]),
        device=config["device"],
        l_loss=config["l_loss"],
        l_eq=config["l_eq"],
        l_ineq=config["l_ineq"],
    )

    # Initialize model and trainer
    device = torch.device(model_cfg.device)
    model = PenaltyDNNPDL(model_cfg).to(device)
    trainer = penalty_pdl_trainer(model, train_loader, val_loader, model_cfg)

    # Train
    primal_net, dual_net = trainer.train()

    # Save trained networks
    torch.save(primal_net.state_dict(), output_dir / "models" / "primal_net.pth")
    torch.save(dual_net.state_dict(), output_dir / "models" / "dual_net.pth")

    # ──────────────── optional Test-set hard-clipping ───────────────────────
    if config["clip_test"]:
        clip_layer = BoundedAct(
            bounds_low.to(device),
            bounds_high.to(device),
            torch.tensor(config["mask"], dtype=torch.bool, device=device),
        )
        clip_layer.apply_bounds.fill_(True)
        eval_model = torch.nn.Sequential(primal_net, clip_layer).to(device)
    else:
        eval_model = primal_net

    # Evaluate primal on test set (MSE only)
    mse_test = evaluate_primal(eval_model, test_loader, device)
    logger.info(f"Primal Test MSE: {mse_test:.6f}")

    # Log and plot
    logs = [{"mse_test": mse_test}]
    log_csv = output_dir / "logs" / config["log_file"]
    save_logs_to_csv(logs, str(log_csv))

    logger.info(f"PDL pipeline completed. Results in {output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDL run script")
    parser.add_argument("--config", type=Path, help="Optional JSON config file")
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE overrides")
    args = parser.parse_args()

    # Load JSON if provided
    cfg: dict = {}
    if args.config and args.config.exists():
        cfg = json.loads(args.config.read_text())

    # Apply overrides
    for ov in args.overrides or []:
        k, v = ov.split("=", 1)
        try:
            cfg[k] = eval(v)
        except Exception:
            cfg[k] = v

    run_pipeline(cfg)
