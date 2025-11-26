"""
===============================================================================
DNN-EWC: Elastic Weight Consolidation Model Definition
===============================================================================

This module defines the feedforward neural network architecture for the 
EWC (Elastic Weight Consolidation) continual learning framework, as described in 
Kirkpatrick et al., 2017 ("Overcoming Catastrophic Forgetting in Neural Networks").

--------------------------------------------------------------------------------
Core Objective:
    To mitigate catastrophic forgetting by constraining important weights using
    a task-specific Fisher information-based quadratic penalty.

--------------------------------------------------------------------------------
Model Architecture:
    - Standard fully connected MLP with:
        • Two hidden layers (ReLU activation)
        • Task-specific output heads (single or multiple depending on use)
    - Model weights are stored after training each task for penalty computation.

--------------------------------------------------------------------------------
Workflow Context:
    This module defines the backbone used during sequential training. It provides:
        - The MLP used for prediction
        - A model whose parameters are subjected to Fisher-based penalties
        - Compatibility with task-specific heads if needed for modularity

--------------------------------------------------------------------------------
Interactions:
    ▸ Called in:
        - `trainer.py` → during training loop (train_one_task)
        - `ewc_utils.py` → for snapshotting parameters and computing penalties
    ▸ Paired with:
        - EWC class (Fisher computation, penalty)

--------------------------------------------------------------------------------
Use in Pipeline:
    1. Initialized and trained on task 0
    2. Parameters snapshotted post-training
    3. Fisher matrix computed on training data
    4. On task t>0:
        - Model loaded
        - Trained with MSE + EWC loss from previous Fisher matrices

--------------------------------------------------------------------------------
Relevant Paper:
    Kirkpatrick et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks.
    Proceedings of the National Academy of Sciences, USA.
"""
import logging
from typing import Dict, Any, List
import sys
import torch
from pathlib import Path
from Dyn_DNN4OPF.data.opf_loader import load_optuna_best_params, get_data_loaders
sys.path.append(str(Path(__file__).resolve().parents[2]))
from penalty_nn.models.penalty_ewc_2head import PenaltyDNN_EWC_2Head
from Dyn_DNN4OPF.utils.plot_utils import save_metadata_to_json
from penalty_nn.training.penalty_trainer import train_penalty_task_sequential, evaluate
from Dyn_DNN4OPF.utils.logger_plotter import plot_losses_from_csv
from pathlib import Path
from Dyn_DNN4OPF.utils.plot_utils import (
    load_logs, get_param_names,
    plot_aggregate, plot_per_output,
)
from typing import List
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from Dyn_DNN4OPF.data.opf_loader import load_output_bounds
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from Dyn_DNN4OPF.utils.logger_plotter import generate_all_diagnostics
# relu_mask = torch.tensor([
#     1, 1, 1, 1, 1,     # Pg (5)
#     1, 1, 1, 1, 1,     # Qg (5)
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Va (14) → no clipping
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # Vm (14)
# ])

from Dyn_DNN4OPF.utils.config import (
    get_io_dims_from_loader,
    default_mask,
    check_bounds_compatibility
)
from Dyn_DNN4OPF.utils.repro import set_determinism
set_determinism()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _to_tensor_mask(m: List[int]) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.int, device = device)

__all__ = ["DEFAULT_CONFIG", "run_pipeline"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 10000,
    "patience": 100,
    "hidden_dim": 112,
    "lambda_ewc":   1e3,  
    "lambda_loss":  1.0,
    "lambda_eq":    1.0,
    "lambda_ineq":  1.0,  
    "log_file": "train_ewc.csv",
    "mask": None,
    "model" : "PenaltyEWC",

    # ── Case-agnostic loader keys ───────────────────────
    "case_name": "pglib_opf_case14_ieee",
    "train_samples": 27000,   # None = use full OPF train split
    "val_samples":   1500,   # None = use full OPF val split
    "test_samples":  1500,   # None = use full OPF test split
    "batches":       None,   # None = default splits; or list of batch indices
    "clip_test":     False,  # default ⟶ *no* clipping on Test set
}

def run_pipeline(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing DNN-EWC training pipeline")

    optuna_params_file = "best_hyperparameters_dnn_ewc.txt"
    optuna_params = load_optuna_best_params(optuna_params_file)

    config: Dict[str, Any] = {**DEFAULT_CONFIG, **optuna_params}
    logger.info(f"Using training config: {config}")
    task_train_loaders, task_val_loaders, test_loader, OBJ_test = get_data_loaders(
        config["batch_size"],
        config["case_name"],
        config.get("train_samples", 27000),
        config.get("val_samples", 1500),
        config.get("test_samples", 1500),
        config.get("batches", None)
    )
    start_dir= Path.cwd()
    if "examples" in str(start_dir):
        start_dir = start_dir.parents[1]  # Move two levels up if 'examples' is in the path

    model_name = config["model"]
    case_name  = config["case_name"]
    output_dir = start_dir/"Results"/f"{model_name}_{case_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models","logs","plots","diagnostics"):
        (output_dir/sub).mkdir(exist_ok=True)

    # override log path so CSV → logs/
    config["log_file"] = str(output_dir/"logs"/config["log_file"])

    case_name = config.get("case_name", "pglib_opf_case14_ieee")
    bounds_low, bounds_high = load_output_bounds(case_name=case_name)
    input_dim, output_dim = get_io_dims_from_loader(task_train_loaders)
    n_bus = input_dim // 2
    n_gen = output_dim // 2 - n_bus

    if config["mask"] is None:
        config["mask"] = default_mask(n_gen, n_bus)
    case_name  = config.get("case_name", "pglib_opf_case14_ieee")
    bounds_low, bounds_high = load_output_bounds(case_name=case_name)

    check_bounds_compatibility(bounds_low, bounds_high, config["mask"], output_dim)

    hidden_dim = config.get("hidden_dim")
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    model = PenaltyDNN_EWC_2Head(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bounds=True,  # enable bound layers for potential clipping
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        mask=torch.tensor(config["mask"], dtype=torch.bool, device=device),  # mask on-device
        lambda_loss=config["lambda_loss"],
        lambda_eq=config["lambda_eq"],
        lambda_ineq=config["lambda_ineq"],
        case_name=config["case_name"],  # load and register physics bounds
        clip_test=config["clip_test"],  # toggle test-time clipping
        device=device  # construct model directly on correct device
    )

    logger.info(
        f"Initialized DNN_EWC(input={input_dim}, hidden={hidden_dim}, "
        f"output={output_dim}, use_bounds=True)"
    )
    model.to(device)

    train_penalty_task_sequential(
        model=model,
        task_train_loaders=task_train_loaders,
        task_val_loaders=task_val_loaders,
        eval_loader=task_val_loaders,
        task_name="ewc",
        ewc_list=[],
        lambda_ewc=config["lambda_ewc"],
        epochs=config["epochs"],
        lr=config["lr"],
        patience=config["patience"],
        log_file=config["log_file"],
        max_epochs=config["epochs"],
        device = device,
    )
    clip_layer = BoundedAct(
        bounds_low, bounds_high,
        torch.tensor(config["mask"], dtype=torch.bool, device = device)
    )
    clip_layer.apply_bounds.fill_(True)
    test_model = torch.nn.Sequential(model, clip_layer).to(device) if config["clip_test"] else model
    test_loss = evaluate(test_model, test_loader, label="Test", device=device)
    logger.info("Final Test MSE: %.6f", test_loss)

    p = Path(config["log_file"]).parent.parent/"plots"
    plot_losses_from_csv(
        config["log_file"],
        str(p/"train_ewc_plot.png"),
        str(p/"train_ewc_test_plot.png"),
    )
    m = OBJ_test
    save_metadata_to_json(m, output_dir/"logs"/"metadata.json")

    # build the directory where you want to stash your model
    out_dir    = output_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # match the load filename
    ckpt_name = f"best_model_EWC_{config['case_name']}.pth"
    ckpt_path = out_dir / ckpt_name

    # save your trained model
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved EWC checkpoint to {ckpt_path}")

    csv_path   = Path(config["log_file"])
    df         = load_logs(csv_path)
    df = df.rename(columns={
        "Epoch":      "epoch",
        "Train Loss": "train_loss",
        "Val Loss":   "val_loss"
    })
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    model_name = config["model"]
    out_dir    = output_dir/"plots"/model_name

    plot_aggregate(df, model_name, out_dir)

    param_names = get_param_names(task_train_loaders)
    plot_per_output(df, model_name, out_dir, param_names)

    # After training, reload best weights by model’s class name:
    ckpt = output_dir/"models"/f"best_model_{model_name}_{case_name}.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Run all diagnostics for task 0
    with torch.no_grad():
        X_train, Y_train,_ = task_train_loaders.dataset.tensors
        X_val,   Y_val,_   = task_val_loaders.dataset.tensors
        X_test,  Y_test  = test_loader.dataset.tensors

    if device.type == "cuda":
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        X_val,   Y_val   = X_val.to(device),   Y_val.to(device)
        X_test,  Y_test  = X_test.to(device),  Y_test.to(device)

    diagnostic_inputs = {
        "Train":          (X_train, Y_train),
        "Validation":     (X_val,   Y_val),
        "Test":           (X_test,  Y_test),
    }

    case = case_name.split("_")[2]
    generate_all_diagnostics(
        model=test_model,
        datasets=diagnostic_inputs,
        device=device,
        case_json=Path("data")/f"sample_{case}.json",
        output_dir=str(output_dir),
        num_gens=n_gen,
        num_buses=n_bus,
        model_name=config["model"],
    )

if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)