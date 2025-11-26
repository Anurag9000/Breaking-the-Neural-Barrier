"""
===============================================================================
Logging and Plotting Utilities for Dyn_DNN4OPF
===============================================================================

This module implements:
    - Structured logging of training metrics to CSV files.
    - Appending loss entries for later analysis.
    - Generation of timestamped plots for training, validation, and test losses.

--------------------------------------------------------------------------------
Core Functionalities:
    - save_logs_to_csv(logs, csv_name):
        Saves list of training log dicts or tuples to CSV for persistent record..
    - plot_losses_from_csv(csv_name, train_val_plot_name):
        Reads CSV logs and generates PNG loss plots with unique timestamps.
    - plot_losses(csv_name, task_type):
        Wrapper generating standard filenames and calling plot function.

--------------------------------------------------------------------------------
Usage:
    - Integrated into training pipelines after each epoch or task.
    - Facilitates easy visualization of training dynamics.
    - Useful for diagnosing overfitting and monitoring generalization.

--------------------------------------------------------------------------------
Design Notes:
    - Uses pandas for CSV I/O and matplotlib for plotting.
    - Includes warnings for missing data columns.
    - Generates plots with clear legends and titles.

--------------------------------------------------------------------------------
Example:
    logger_plotter.save_logs_to_csv(logs, "train_log.csv")
    logger_plotter.plot_losses("train_log.csv", "ewc_task")
"""

import logging
from typing import List, Dict, Union, Tuple
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
from typing import Optional
import os
from torch import Tensor
import torch.nn as nn
import numpy as np
import csv
from Dyn_DNN4OPF.utils.constraint_losses import mean_constraint_violation
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import os
import json
import torch
from typing import Dict, Tuple

from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, objective, gap_objective
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds, extract_opf_constraints, load_cost_coeff

import json
from typing import Dict, Tuple

from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.data.opf_loader import extract_opf_constraints, load_case_bounds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TEST_COL = "Test Loss"

def save_logs_to_csv(logs: List[Union[Dict[str, Union[int, float, str]], Tuple]], csv_name: str) -> None:
    """
    Save training logs to a CSV file.
    Supports both list of dicts and list of 3-tuples (epoch, train_loss, val_loss).
    Normalizes column names for compatibility with plotters.
    """
    if not logs:
        logger.warning("Empty logs received; skipping CSV write.")
        return

    if isinstance(logs[0], dict):
        df = pd.DataFrame(logs)
    elif isinstance(logs[0], (list, tuple)) and len(logs[0]) == 3:
        df = pd.DataFrame(logs, columns=["epoch", "train_loss", "val_loss"])
    else:
        raise ValueError("Logs must be a list of dicts or 3-element tuples/lists.")

    # Normalize column names to match plotter expectations
    df.rename(columns=lambda c: c.strip().lower().title().replace("_", " "), inplace=True)
    for c in ("Train Loss", "Val Loss", TEST_COL):
        if c not in df.columns:
            df[c] = ""
    df.to_csv(csv_name, index=False)
    logger.info(f"Saved training logs to {csv_name}")

def plot_losses_from_csv(csv_name: str, train_val_plot_name: str, test_plot_name: str) -> None:
    """
    Plot training, validation, and test losses from CSV and save plots as timestamped images.
    """
    df = pd.read_csv(csv_name)

    # Timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    train_val_plot_name = train_val_plot_name.replace(".png", f"_{timestamp}.png")
    test_plot_name = test_plot_name.replace(".png", f"_{timestamp}.png")

    # --- Plot Train + Val Losses (excluding Generalization row) ---
    if all(col in df.columns for col in ['Epoch', 'Train Loss', 'Val Loss']):
        df_clean = df[df["Epoch"].apply(lambda v: str(v).isdigit())].copy()
        df_clean["Epoch"] = pd.to_numeric(df_clean["Epoch"], errors="coerce")
        df_clean = df_clean.dropna(subset=["Epoch", "Train Loss", "Val Loss"])
        df_clean["Epoch"] = df_clean["Epoch"].astype(int)

        x = df_clean["Epoch"]
        y_train = df_clean["Train Loss"]
        y_val = df_clean["Val Loss"]

        # Side-by-side subplots: linear and log scale
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Linear Scale Plot
        axes[0].plot(x, y_train, label="Train Loss", linestyle='-')
        axes[0].plot(x, y_val, label="Validation Loss", linestyle='--')
        axes[0].set_title("Train & Validation Loss (Linear Scale)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Log Scale Plot
        axes[1].plot(x, y_train, label="Train Loss", linestyle='-')
        axes[1].plot(x, y_val, label="Validation Loss", linestyle='--')
        axes[1].set_title("Train & Validation Loss (Log Scale)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, which='both')

        plt.tight_layout()
        plt.savefig(train_val_plot_name, bbox_inches='tight')
        logger.info(f"Saved Train & Validation plot as {train_val_plot_name}")
        plt.close()
    else:
        logger.warning("Required columns missing for Train/Val plot. Skipping train_val plot.")


        plt.close()

def plot_mse_per_output(Y_true: torch.Tensor, Y_pred: torch.Tensor, name: str, num_outputs: int = 38, save_dir: Optional[str] = None) -> float:
    """
    Plot MSE for each individual output variable over all test samples.

    This function calculates and plots the MSE distribution for each output variable.

    Args:
        Y_true (torch.Tensor): Ground truth output tensor of shape [N, num_outputs].
        Y_pred (torch.Tensor): Predicted output tensor of same shape as Y_true.
        name (str): Dataset label for the plot titles and filenames (e.g., "Test").
        num_outputs (int): Total number of output variables to calculate MSE for.
        save_dir (Optional[str]): Directory to save the output plots. Created if it doesn't exist.

    Output:
        Saves one histogram plot for each output variable, showing MSE over all test samples.
    Returns:
        float: Mean MSE across all outputs and samples.
    """
    os.makedirs(save_dir, exist_ok=True)

    Y_true = Y_true.cpu().detach()
    Y_pred = Y_pred.cpu().detach()

    # Compute MSE for each sample and output
    mse_per_output = (Y_true - Y_pred) ** 2  # Shape: [N, num_outputs]

    # # Generate plots for each output variable
    # for i in range(num_outputs):
    #     mse_values = mse_per_output[:, i].numpy()  # Extract MSE values for the i-th output

    #     plt.figure(figsize=(8, 6))
    #     plt.hist(mse_values, bins=20, alpha=0.7, color="blue",edgecolor='black')
    #     plt.title(f"{name} - MSE Distribution for Output {i}")
    #     plt.xlabel("MSE")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     pdf_path = os.path.join(save_dir, f"{name.lower()}_output_{i}_mse.pdf")
    #     plt.savefig(pdf_path, format="pdf")
    #     plt.close()

    # Compute mean MSE across all outputs and samples
    mean_mse = mse_per_output.mean().item()

    # logger.info(f"Saved MSE plots for outputs as {pdf_path}")
    return mean_mse

#--------------------------Equality Constraints-----------------------------
#--------------------------Power Balance Residuals-----------------------------

def compute_power_balance_stats_per_sample(
    res_real: torch.Tensor,  # shape: (N, n_buses)
    res_imag: torch.Tensor   # shape: (N, n_buses)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each sample, compute:
        - Mean of signed real and reactive residuals over buses.
        - Max of absolute real and reactive residuals over buses.

    Args:
        res_real (torch.Tensor): Real power residuals, shape (N, n_buses).
        res_imag (torch.Tensor): Reactive power residuals, shape (N, n_buses).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - mean_real_per_sample: shape (N,)
            - mean_imag_per_sample: shape (N,)
            - max_abs_real_per_sample: shape (N,)
            - max_abs_imag_per_sample: shape (N,)
    """
    # Ensure on CPU and detached from graph
    res_real = res_real.cpu().detach()
    res_imag = res_imag.cpu().detach()

    # Mean over buses (abs)
    mean_real = res_real.abs().mean(dim=1)
    mean_reac = res_imag.abs().mean(dim=1)   # shape [N,1]

    # Max of absolute values over buses
    max_real = res_real.abs().max(dim=1).values  # shape (N,)
    max_reac = res_imag.abs().max(dim=1).values  # shape (N,)

    return mean_real, mean_reac, max_real, max_reac

def compute_inequality_violation_per_sample(
    y_pred: Tensor,
    bounds: Dict[str, Tensor],
    num_gens: int,
    num_buses: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute per-sample mean inequality constraint violations for PG, QG, and VM.

    Args:
        y_pred (Tensor): Predictions of shape [N, 2*num_gens + 2*num_buses].
        bounds (Dict[str, Tensor]): Dictionary with min/max bounds for p, q, v.
        num_gens (int): Number of generators.
        num_buses (int): Number of buses.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (pg_mean, qg_mean, vm_mean), each of shape [N, 1]
    """
    y_pred = y_pred.detach().cpu()

    # Split predicted outputs
    pg = y_pred[:, :num_gens]
    qg = y_pred[:, num_gens:2 * num_gens]
    vm = y_pred[:, 2 * num_gens + num_buses:2 * num_gens + 2 * num_buses]

    def violation(values: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        return torch.clamp(values - hi, min=0) + torch.clamp(lo - values, min=0)

    # Compute [N, d] violation matrices
    pg_viol = violation(pg, bounds["p_min"], bounds["p_max"])
    qg_viol = violation(qg, bounds["q_min"], bounds["q_max"])
    vm_viol = violation(vm, bounds["v_min"], bounds["v_max"])

    # Compute per-sample mean [N, 1]
    pg_mean = pg_viol.abs().mean(dim=1)
    qg_mean = qg_viol.abs().mean(dim=1)
    vm_mean = vm_viol.abs().mean(dim=1)

    #Compute per-sample max [N,1]
    pg_max = pg_viol.abs().max(dim=1).values
    qg_max = qg_viol.abs().max(dim=1).values
    vm_max = vm_viol.abs().max(dim=1).values

    return pg_mean, qg_mean, vm_mean, pg_max, qg_max, vm_max

def generate_summary_csv(
    model_name: str,
    case_name: str,
    train_samples: int,
    test_samples: int,
    mse_mean: float,
    gap_obj_mean: float,
    real_max_max: float,
    real_max_mean:float,
    real_mean_max:float,
    real_mean_mean: float,
    reactive_max_max: float,
    reactive_max_mean:float,
    reactive_mean_max:float,
    reactive_mean_mean: float,
    pg_max_max: float,
    pg_max_mean:float,
    pg_mean_max:float,
    pg_mean_mean: float,
    qg_max_max: float,
    qg_max_mean:float,
    qg_mean_max:float,
    qg_mean_mean: float,
    vm_max_max: float,
    vm_max_mean:float,
    vm_mean_max:float,
    vm_mean_mean: float,
    output_dir: str,
    total_neurons:int =None
    ):
    """
    Save summary metrics into a CSV file.

    Args:
        split_name (str): Name of the data split (e.g., "Train", "Test").
        case_name (str): Name of the case.
        train_samples (int): Number of training samples.
        test_samples (int): Number of test samples.
        mse_mean (float): Mean MSE across outputs and samples.
        gap_obj_mean (float): L2 norm of the gap objective.
        l2_real_max (float): Maximum L2 norm of real power residuals.
        l2_real_mean (float): Mean L2 norm of real power residuals.
        l2_reactive_max (float): Maximum L2 norm of reactive power residuals.
        l2_reactive_mean (float): Mean L2 norm of reactive power residuals.
        l2_pg_max (float): Maximum L2 norm of inequality PG constraints.
        l2_pg_mean (float): Mean L2 norm of inequality PG constraints.
        l2_qg_max (float): Maximum L2 norm of inequality QG constraints.
        l2_qg_mean (float): Mean L2 norm of inequality QG constraints.
        l2_vm_max (float): Maximum L2 norm of inequality VM constraints.
        l2_vm_mean (float): Mean L2 norm of inequality VM constraints.
        output_dir (str): Directory to save the CSV file.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "SUMMARY.csv"

    # Prepare data
    row = {
        "Model Name": model_name,
        "Case Name": case_name,
        "Time Taken": "",
        "Total_Neurons": total_neurons,
        "Train Samples": train_samples,
        "Test Samples": test_samples,
        "Mean MSE": mse_mean,
        "Mean Gap Objective %": gap_obj_mean,
        "Max_Max Real Power Residual": real_max_max,
        "Max_Mean Real Power Residual": real_max_mean,
        "Mean_Max Real Power Residual": real_mean_max,
        "Mean_Mean Real Power Residual ": real_mean_mean,
        "Max_Max Reactive Power Residual": reactive_max_max,
        "Max_Mean Reactive Power Residual": reactive_max_mean,
        "Mean_Max Reactive Power Residual": reactive_mean_max,
        "Mean_Mean Reactive Power Residual ": reactive_mean_mean,
        "Max_Max Inequality PG": pg_max_max,
        "Max_Mean Inequality PG": pg_max_mean,
        "Mean_Max Inequality PG": pg_mean_max,
        "Mean_Mean Inequality PG": pg_mean_mean,
        "Max_Max Inequality QG": qg_max_max,
        "Max_Mean Inequality QG": qg_max_mean,
        "Mean_Max Inequality QG": qg_mean_max,
        "Mean_Mean Inequality QG": qg_mean_mean,
        "Max_Max Inequality VM": vm_max_max,
        "Max_Mean Inequality VM": vm_max_mean,
        "Mean_Max Inequality VM": vm_mean_max,
        "Mean_Mean Inequality VM": vm_mean_mean,
    }

    # Save to CSV
    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

    print(f"Summary row added to {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MASTER DIAGNOSTIC WRAPPER
# ──────────────────────────────────────────────────────────────────────────────
pg_test = None

@torch.no_grad()
def generate_all_diagnostics(
    model: torch.nn.Module,
    datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],  # {"Train": (X,Y), …}
    *,
    device: torch.device | str = "cpu",
    case_json: str ,
    output_dir: str = "Diagnostics",
    num_gens: int = 5,
    num_buses: int = 14,
    model_name: str, 
    task_id: int | None = None,
    ) -> None:
    """
    Produce *every* diagnostic plot currently created in the run-scripts
    (distributions, per-var histograms, constraint scatter / hist, power balance).
    """

    # ── prepare paths --------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir_d = output_dir/'diagnostics'
    paths = {
        "indiv":              output_dir_d / "PerVariable",
        "Sg_vector":          output_dir_d / "SgVector",
        "constraint_scatter": output_dir_d / "ConstraintScatter",
        "power_balance":      output_dir_d/ "PowerBalance",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir/'logs'
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir/'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    

    case_stem = Path(case_json).stem.removeprefix("sample_")
    case_suffix_map = {
    'case14': 'ieee',
    'case30': 'ieee',
    'case57': 'ieee',
    'case118': 'ieee',
    'case500': 'goc',
    'case2000': 'goc',
    'case6470': 'rte',
    'case4661': 'sdetpglib_opf_case10000_goc',
    'case13659': 'pegase'}

    suffix = case_suffix_map[case_stem]
    case_name = f"pglib_opf_{case_stem}_{suffix}"
    bounds = load_case_bounds(case_name)

    base_dir = output_dir.parent.parent
    case_json = base_dir/"data"/f"sample_{case_stem}.json"
    opf_json = json.loads(Path(case_json).read_text())
    opf_data = extract_opf_constraints(opf_json)

    y_val, y_row, y_col, y_shape = opf_data["y_bus"]
    Y_bus = torch.sparse_coo_tensor(
        torch.stack([y_row, y_col]), y_val, size=y_shape
    )

    gen_bus_idx = torch.tensor(
        [opf_data["bus_map"][str(b)] for b in opf_data["gen_buses"]]
    )
    load_bus_idx = torch.tensor(
        [opf_data["bus_map"][str(b)] for b in opf_data["load_buses"]]
    )
    cost=load_cost_coeff(opf_json)
    cost_q=cost['cost_q'].to(device)
    cost_l=cost['cost_l'].to(device)
    cost_c=cost['cost_c'].to(device)

    model.eval()
    model.to(device)

    if task_id is None and hasattr(model, "columns"):
        task_id = len(model.columns) - 1

    for split_name, data in datasets.items():
        if len(data) == 3:
            X, Y_true, meta = data
        else:
            X, Y_true = data
            meta = None
            
            X = X.to(device)
            Y_true = Y_true.to(device)

            with torch.no_grad():
                if task_id is not None:
                    Y_pred = model(X, task_id)
                else:
                    Y_pred = model(X)
            if(split_name == 'Test'):
                pd = X[:, :num_buses]
                qd = X[:, num_buses : 2 * num_buses]
                pg = Y_pred[:, :num_gens]
                qg = Y_pred[:, num_gens : 2 * num_gens]
                va = Y_pred[:, 2 * num_gens : 2 * num_gens + num_buses]
                vm = Y_pred[:, 2 * num_gens + num_buses :]
                # pg = Y_true[:, :num_gens]
                # qg = Y_true[:, num_gens : 2 * num_gens]
                # va = Y_true[:, 2 * num_gens : 2 * num_gens + num_buses]
                # vm = Y_true[:, 2 * num_gens + num_buses :]

                o=objective(pg=pg,cost_q=cost_q,cost_l=cost_l,cost_c=cost_c)
                g = gap_objective(objective=o, metadata_file=csv_dir/'metadata.json')
                gap_obj_mean = g.mean().item()

                gap_json_file = csv_dir/'gap_obj.json'
                try:
                    # Convert tensor to list for JSON compatibility
                    gap_list = g.tolist() if isinstance(g, torch.Tensor) else g
                    with open(gap_json_file, "w") as f:
                        json.dump(gap_list, f, indent=4)
                    print("Gap objective saved to output\\gap_objective.json")
                except Exception as e:
                    print(f"Failed to save gap objective to JSON: {e}")

                #mse
                mean_mse = plot_mse_per_output(
                    Y_true=Y_true, Y_pred=Y_pred, name=split_name,num_outputs=2 * num_gens + 2 * num_buses, save_dir=paths["indiv"])

                #plot equality constraints
                res_P, res_Q = power_balance_residuals(
                    pg = pg, qg = qg, pd = pd, qd = qd, vm = vm, va = va,
                    y_bus = Y_bus, gen_bus_idx = gen_bus_idx, load_bus_idx = load_bus_idx,n_bus=num_buses
                )  

                # plot_power_balance(
                #     res_P, res_Q, split_name, paths["power_balance"]
                # )

                # save_l2_power_residuals_per_bus_to_csv(res_real=res_P, res_imag=res_Q, output_dir=csv_dir)

                # l2_real_max, l2_real_mean, l2_reactive_max, l2_reactive_mean = save_l2_power_residuals_per_sample_to_csv(
                #     res_real=res_P, res_imag=res_Q, output_dir=csv_dir
                # )

                # plot_l2_power_residuals_per_sample_from_csv_pdf(
                #     csv_path=csv_dir/'l2_power_residuals_per_sample.csv',
                #     output_dir=plot_dir
                # )

                # plot_constraint_scatter(
                #     Y_pred, bounds, split_name, paths["constraint_scatter"]
                # )

                me_real,me_reac,max_real, max_reac = compute_power_balance_stats_per_sample(res_real=res_P, res_imag=res_Q)

                mean_mean_real= me_real.mean().item()
                mean_mean_reac= me_reac.mean().item()

                max_mean_real=me_real.max().item()
                max_mean_reac=me_reac.max().item()

                mean_max_real=max_real.mean().item()
                mean_max_reac=max_reac.mean().item()

                max_max_real = max_real.max().item()
                max_max_reac = max_reac.max().item()

                # plot_power_balance_histograms(mean_real=me_real, mean_imag=me_reac,
                #                               max_real=max_real, max_imag=max_reac,
                #                               directory=plot_dir)
                
                #inequality constraints

                # compute_and_save_inequality_constraints(
                #     y_pred=Y_pred,
                #     bounds=bounds,
                #     num_gens=num_gens,
                #     num_buses=num_buses,
                #     out_dir=csv_dir
                # )

                pg_mean, qg_mean, vm_mean, pg_max,qg_max,vm_max= compute_inequality_violation_per_sample(y_pred=Y_pred, bounds=bounds,num_gens=num_gens,num_buses=num_buses)

                pg_mean_mean=pg_mean.mean().item()
                qg_mean_mean=qg_mean.mean().item()
                vm_mean_mean=vm_mean.mean().item()

                pg_max_mean=pg_mean.max().item()
                qg_max_mean=qg_mean.max().item()
                vm_max_mean=vm_mean.max().item()

                pg_mean_max=pg_max.mean().item()
                qg_mean_max=qg_max.mean().item()
                vm_mean_max=vm_max.mean().item()

                pg_max_max=pg_max.max().item()
                qg_max_max=qg_max.max().item()
                vm_max_max=vm_max.max().item()



                # l2_pg, l2_qg, l2_vm = save_l2_inequality_constraints_per_sample(
                #     y_pred=Y_pred,
                #     bounds=bounds,
                #     output_dir=csv_dir
                # )
                # plot_l2_inequality_constraints_from_csv(
                #     csv_path=csv_dir/'Inequality_constraints_per_sample_l2.csv',
                #     output_dir=plot_dir
                # )          

                # plot_sg_vector_deviation(
                #     Y_true=Y_true,
                #     Y_pred=Y_pred,
                #     name=split_name,
                #     num_gens=num_gens,
                #     save_dir=paths["Sg_vector"]
                # )     
                #plot gap objective     

                # plot_gap_objective(json_file=csv_dir/'gap_obj.json',save_path=plot_dir/'Gap_Objective.pdf')

                if model_name=='DEN' or model_name =='ADP-DEN' or model_name=='ADP_DEN' :
                    total_hidden_neurons = sum(l.out_features for l in model.layers)
                elif model_name=='STL':
                    total_hidden_neurons = sum(
                        layer.out_features for i, layer in enumerate(model.net)
                        if isinstance(layer, nn.Linear) and i != len(model.net) - 1
                    )

                else:
                    total_hidden_neurons=None


                generate_summary_csv(
                    model_name = model_name,
                    case_name=case_name,
                    total_neurons=total_hidden_neurons,
                    train_samples=len(datasets['Train'][0]),
                    test_samples=len(datasets['Test'][0]),
                    mse_mean=mean_mse,
                    gap_obj_mean=gap_obj_mean,
                    real_max_max=max_max_real,
                    real_max_mean=max_mean_real,
                    real_mean_max=mean_max_real,
                    real_mean_mean=mean_mean_real,
                    reactive_max_max=max_max_reac,
                    reactive_max_mean=max_mean_reac,
                    reactive_mean_max=mean_max_reac,
                    reactive_mean_mean=mean_mean_reac,
                    pg_max_max = pg_max_max,
                    pg_max_mean=pg_max_mean,
                    pg_mean_max=pg_mean_max,
                    pg_mean_mean=pg_mean_mean,
                    qg_max_max=qg_max_max,
                    qg_max_mean=qg_max_mean,
                    qg_mean_max=qg_mean_max,
                    qg_mean_mean=qg_mean_mean,
                    vm_max_max=vm_max_max,
                    vm_max_mean=vm_max_mean,
                    vm_mean_max=vm_mean_max,
                    vm_mean_mean=vm_mean_mean,
                    output_dir=output_dir.parent      
                )

                y=mean_constraint_violation(Y_pred=Y_pred,res_real=res_P,res_imag=res_Q, bounds=bounds,num_gens= num_gens,num_buses=num_buses)

                print(f"Found values as {y}")

                print(f"Diagnostics saved to “{output_dir}”")