from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.training.trainer import train_fsnet, evaluate
from penalty_nn.models.penalty_fsnet import PenaltyFSNet
from Dyn_DNN4OPF.data.opf_loader import (
    load_optuna_best_params,
    get_data_loaders,
    load_case_bounds,
    load_output_bounds,
)
from Dyn_DNN4OPF.utils.config import get_io_dims_from_loader, default_mask, check_bounds_compatibility
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, generate_all_diagnostics
from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs, get_param_names, plot_aggregate, plot_per_output
)
from Dyn_DNN4OPF.utils.repro import set_determinism, worker_init_fn
import logging
from pathlib import Path
from Dyn_DNN4OPF.data.opf_loader import DATASET_ROOT, load_cost_coeff
import torch
import torch.nn.functional as F
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
import json
from Dyn_DNN4OPF.utils.pdl_constraints import init_from_case, compute_g, compute_h
# Ensure reproducibility
set_determinism(42)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

# ─── Default config for FSNet ───────────────────────────────────────────────
DEFAULT_CONFIG = {
    "batch_size":   1024,
    "lr":           1e-3,
    "epochs":       10000,
    "max_epochs":   10000,   # forwarded verbatim to trainer
    "lambda_fs":    1e-4,    # weight on feasibility-distance term
    "patience":     100,     # early stop on val loss
    "save_path":    "best_fsnet.pth",
    "log_file":     "train_fsnet.csv",

    "model":        "FSNet",
    "case_name":    "pglib_opf_case118_ieee",
    "train_samples":27000,
    "val_samples":  1500,
    "test_samples": 1500,
    "batches":      None,
    "clip_test":    False,   # apply BoundedAct only during TEST/diagnostics
    "lambda_loss":  1.0,     # weight on baseline loss
    "lambda_eq":    1.0,     # weight on equality residuals
    "lambda_ineq":  1.0,     # weight on inequality violations
}

def run_pipeline(cfg: dict) -> None:
    # Merge defaults, Optuna, and overrides
    optuna_params = load_optuna_best_params("best_hyperparameters_fsnet.txt")
    config = {**DEFAULT_CONFIG, **optuna_params, **cfg}
    logger = logging.getLogger(__name__)
    logger.info(f"[FSNet] Using config: {config}")

    # Prepare output dirs
    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]
    output_dir = start_dir / "Results" / f"{config['model']}_{config['case_name']}"
    for d in ("models","logs","plots","diagnostics"):
        (output_dir / d).mkdir(parents=True, exist_ok=True)
    config["log_file"] = str(output_dir / "logs" / config["log_file"])

    # Load data
    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"],
    )

    # Determine I/O dims & mask
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus
    if config.get("mask") is None:
        config["mask"] = default_mask(n_gen, n_bus)

    # Check bounds compatibility
    bounds_low, bounds_high = load_output_bounds(case_name=config["case_name"])
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    # Pack constraint data (used by FSNet’s forward through lbfgs)
    raw = load_case_bounds(case_name=config["case_name"])
    raw["gen_buses"]  = torch.tensor(raw["gen_buses"],  dtype=torch.long, device = device)
    raw["load_buses"] = torch.tensor(raw["load_buses"], dtype=torch.long, device = device)

    sample_file = DATASET_ROOT / f"sample_{config['case_name'].split('_')[2]}.json"
    data        = json.loads(sample_file.read_text())

    # per-bus real/reactive loads
    pd = torch.zeros(len(data["grid"]["nodes"]["bus"]), device = device)
    qd = torch.zeros_like(pd, device = device)
    receivers = data["grid"]["edges"]["load_link"]["receivers"]
    loads     = data["grid"]["nodes"]["load"]
    for idx, bus in enumerate(receivers):
        pd[bus] += loads[idx][0]
        qd[bus] += loads[idx][1]
    raw["pd"], raw["qd"] = pd, qd

    # quadratic, linear, constant cost coeffs
    raw.update(load_cost_coeff(data))

    # now register for objective, eq/ineq helpers
    init_from_case(raw)
    from types import SimpleNamespace
    data = SimpleNamespace(**raw)
    data.bounds_lo = bounds_low    # lower output bounds (PG|QG|VA|VM)
    data.bounds_hi = bounds_high   # upper output bounds (PG|QG|VA|VM)
    data.eq_resid   = lambda x, y: power_balance_residuals(
        pg            = y[:, :n_gen],
        qg            = y[:, n_gen:2*n_gen],
        pd            = raw["pd"].unsqueeze(0).expand(y.size(0), -1),
        qd            = raw["qd"].unsqueeze(0).expand(y.size(0), -1),
        vm            = y[:, 2*n_gen:2*n_gen+n_bus],
        va            = y[:, 2*n_gen+n_bus:],
        y_bus         = raw["y_bus"],
        gen_bus_idx   = raw["gen_buses"],
        load_bus_idx  = raw["load_buses"],
        n_bus         = n_bus
    )
    data.ineq_resid = lambda x, y: compute_g(y)

    data_dict = data

    # Instantiate FSNet
    model = PenaltyFSNet(
        input_dim=input_dim,
        hidden_dim=config.get("hidden_dim", 128),
        output_dim=output_dim,
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.1),
        fs_config=config.get("fs_config", {}),
        lambda_loss=config["lambda_loss"],
        lambda_eq=config["lambda_eq"],
        lambda_ineq=config["lambda_ineq"]
    )
    model.to(device)

    # Train
    logs = train_fsnet(
        model=model,
        train_loader=train_loader,
        data_dict=data_dict,
        val_loader=val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        lambda_fs=config["lambda_fs"],
        patience=config["patience"],
        device=None,
        save_path=str(output_dir / config["save_path"]),
        max_epochs=config["max_epochs"],   # ← forwarded verbatim
    )
    save_logs_to_csv(logs, config["log_file"])

    # Plot training curves
    df = load_logs(config["log_file"])
    plot_aggregate(df, config["model"], output_dir/"plots"/config["model"])
    plot_per_output(df, config["model"], output_dir/"plots"/config["model"], get_param_names(train_loader))

    # Evaluate on test set  (apply optional clipping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(output_dir / config["save_path"], map_location=device))
    model.to(device).eval()
    model.data_dict = data_dict  # ensure forward can fetch dict

    clip_layer = BoundedAct(bounds_low, bounds_high, torch.tensor(config["mask"], dtype=torch.bool, device = device))
    clip_layer.apply_bounds.fill_(True)

    def _maybe_clip(y: torch.Tensor) -> torch.Tensor:
        return clip_layer(y) if config["clip_test"] else y

    total_mse, total_elems = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, y_refined = model(xb, data_dict)
            y_refined = _maybe_clip(y_refined)
            total_mse += F.mse_loss(y_refined, yb, reduction='sum').item()
            total_elems += yb.numel()
    test_loss = total_mse / total_elems
    logger.info("[FSNet] Final Test MSE: %.6f", test_loss)

    # Diagnostics – wrap model so generate_all_diagnostics sees clipped outputs
    class _DiagWrapper(torch.nn.Module):
        def __init__(self, base, clip, data_dict):
            super().__init__()
            self.base = base
            self.clip = clip
            self.data = data_dict
        def forward(self, x):
            _, y_refined = self.base(x, self.data)
            return self.clip(y_refined) if config["clip_test"] else y_refined

    diag_model = _DiagWrapper(model, clip_layer, data_dict)

    generate_all_diagnostics(
        model=diag_model,
        datasets={
            "Train":       (train_loader.dataset.tensors[0], train_loader.dataset.tensors[1]),
            "Validation":  (val_loader.dataset.tensors[0], val_loader.dataset.tensors[1]),
            "Test":        (test_loader.dataset.tensors[0], test_loader.dataset.tensors[1]),
        },
        device=device,
        case_json=Path("data")/f"sample_{config['case_name'].split('_')[2]}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

if __name__ == "__main__":
    run_pipeline({})
