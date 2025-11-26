"""
run_dnn_l1.py
=============

L1-Regularized DNN training pipeline. Mirrors run_dnn_l2.py but uses explicit
L1 penalty (via train_with_l1) instead of optimizer weight decay.
"""
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
import logging
from typing import Dict, Any, List
from pathlib import Path
from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
import torch
from Dyn_DNN4OPF.utils.repro import set_determinism
set_determinism()
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from Dyn_DNN4OPF.models.dnn_l1 import DNN_L1
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
from Dyn_DNN4OPF.training.trainer import train_with_l1, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv, plot_losses_from_csv
from Dyn_DNN4OPF.utils.plot_utils import (
    save_metadata_to_json,
    load_logs,
    get_param_names,
    plot_aggregate,
    plot_per_output
)
from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct   # ➜ added

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int, device = device)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size":    1024,
    "lr":            1e-3,
    "epochs":        10000,
    "max_epochs":    10000,   # ➜ forwarded verbatim to trainer
    "patience":      100,
    "hidden_dim":    112,
    "log_file":      "train_l1.csv",
    "mask":          None,
    "l1_coeff":      1e-4,
    "model":         "L1",

    # ── Data / case settings ─────────────────────────────
    "case_name":     "pglib_opf_case14_ieee",
    "train_samples": 27000,
    "val_samples":   1500,
    "test_samples":  1500,
    "batches":       None,

    # ── Optional test-time clipping ──────────────────────
    "clip_test":     False,   # wrap model in BoundedAct only for Test set
}

def run_pipeline(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting DNN-L1 training pipeline")

    # load hyperparams if tuned
    optuna_file = "best_hyperparameters_dnn_l1.txt"
    optuna_params = load_optuna_best_params(optuna_file)
    config = {**DEFAULT_CONFIG, **optuna_params}
    logger.info(f"Training config: {config}")

    # prepare data
    train_loader, val_loader, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config["train_samples"],
        config["val_samples"],
        config["test_samples"],
        config["batches"]
    )

    # prepare output directories
    start_dir = Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]
    model_name = config["model"]
    case_name  = config["case_name"]
    out_root   = start_dir / "Results" / f"{model_name}_{case_name}"
    for sub in ("models","logs","plots","diagnostics"):
        (out_root/sub).mkdir(parents=True, exist_ok=True)
    config["log_file"] = str(out_root/"logs"/config["log_file"])

    # determine dims & bounds
    input_dim, output_dim = get_io_dims_from_loader(train_loader)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus

    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)
    bounds_low, bounds_high = load_output_bounds(case_name=case_name)
    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    hidden_dim = config.get("hidden_dim")
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    model = DNN_L1(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bounds=False,                    # ➜ no clipping during train/val
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=config["mask"]
    )
    logger.info(
        f"Initialized DNN_L1(input={input_dim}, hidden={hidden_dim}, "
        f"output={output_dim}, bounds=False)"
    )
    model.to(device)

    # train
    logs = train_with_l1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["max_epochs"],         # ➜ verbatim forward
        lr=config["lr"],
        l1_coeff=config["l1_coeff"],
        patience=config["patience"]
    )

    # save best checkpoint
    ckpt_dir = out_root/"models"
    ckpt_name = f"best_model_L1_{case_name}.pth"
    torch.save(model.state_dict(), ckpt_dir/ckpt_name)
    logger.info(f"Saved L1 checkpoint to {ckpt_dir/ckpt_name}")

    # save logs & plots (only MSE logged)
    save_logs_to_csv(logs, config["log_file"])
    plot_losses_from_csv(
        config["log_file"],
        str(out_root/"plots"/"train_l1_plot.png"),
        str(out_root/"plots"/"train_l1_test_plot.png"),
    )

    # load best for evaluation
    model.load_state_dict(torch.load(ckpt_dir/ckpt_name, map_location=device))
    model.eval()

    # ─── optional test-time clipping ─────────────────────
    if config.get("clip_test", False):
        clip_layer = BoundedAct(
            bounds_low.to(device),
            bounds_high.to(device),
            torch.tensor(config["mask"], dtype=torch.bool, device=device),
        )
        clip_layer.apply_bounds.fill_(True)
        test_model = torch.nn.Sequential(model, clip_layer).to(device)
    else:
        test_model = model

    # evaluate
    test_loss = evaluate(test_model, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)
    m = OBJ_test
    save_metadata_to_json(m, out_root/"logs"/"metadata.json")

    # per-output diagnostics
    df = load_logs(Path(config["log_file"]))
    df = df.rename(columns={"Epoch":"epoch","Train Loss":"train_loss","Val Loss":"val_loss"})
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    plot_aggregate(df, model_name, out_root/"plots"/model_name)
    param_names = get_param_names(train_loader)
    plot_per_output(df, model_name, out_root/"plots"/model_name, param_names)

    # centralized diagnostics
    X_tr, Y_tr,_ = train_loader.dataset.tensors
    X_va, Y_va,_ = val_loader.dataset.tensors
    X_te, Y_te   = test_loader.dataset.tensors
    diagnostic_inputs = {
        "Train":      (X_tr, Y_tr),
        "Validation": (X_va, Y_va),
        "Test":       (X_te, Y_te),
    }
    case = case_name.split("_")[2]
    generate_all_diagnostics(
        model=test_model,                    # ➜ diagnostics use same wrapper
        datasets=diagnostic_inputs,
        device=device,
        case_json=Path("data")/f"sample_{case}.json",
        output_dir=str(out_root),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=model_name,
    )

    logger.info(f"Completed DNN-L1 pipeline. Test loss: {test_loss:.6f}")

if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
