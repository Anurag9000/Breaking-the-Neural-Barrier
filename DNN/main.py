"""
Unified CLI launcher for Dyn_DNN4OPF: dispatches to the correct run_*.py pipeline.

Overview
-------------------------------------------------------------------------------
This script allows you to run any model pipeline (STL, MTL, EWC, etc.) using a unified interface.
All training, evaluation, data loading, case selection, and hyperparameter configuration can be
controlled via command-line arguments or optional config files.

Usage Examples
-------------------------------------------------------------------------------

▶ Basic usage (defaults to STL on case14):
    $ python main.py

▶ Run a specific model on default full training set:
    $ python main.py --model progressive

▶ Select a different OPF case:
    $ python main.py --model l1 --case_name pglib_opf_case14_ieee

▶ Train on a subset of samples from the full training set:
    $ python main.py --model stl train_samples=50000

▶ Use custom validation and test sample counts:
    $ python main.py --model stl val_samples=5000 test_samples=5000

▶ Load only specific batches from training data:
    $ python main.py --model mtl --batches="[1,2,3]"

▶ Load from config file + override specific keys:
    $ python main.py --model ewc --config cfgs/ewc_case14.json epochs=50 lr=1e-4

▶ Override constraint loss weights or optimizer settings:
    $ python main.py --model mtl lambda_th=0.5 lr=1e-3

▶ Enable output bounding with a custom mask:
    $ python main.py --model stl use_bounds=True mask="[1]*10 + [0]*28"

▶ Evaluate-only mode (skip training):
    $ python main.py --model l1 --config best_model_cfg.yaml evaluate_only=True

▶ Progressive model with adapter pruning:
    $ python main.py --model progressive prune_adapters=True alpha_threshold=1e-2

Data Selection Options
-------------------------------------------------------------------------------

• case_name
    - Format: full PGLib OPF name, e.g., "pglib_opf_case14_ieee"
    - Default: "pglib_opf_case14_ieee"

• train_samples, val_samples, test_samples
    - Integer values specifying how many examples to load from each split
    - If None (default), loads:
        → All batches for training (∼270,000 points)
        → 1 full batch (∼15,000 points) for val/test

• batches
    - List[int] format (quoted Python list), e.g. "[1, 3, 7]"
    - Selects specific batches from training split only

Masking & Bounded Output Control
-------------------------------------------------------------------------------

• use_bounds
    - Enables bounded ReLU output using specified or default mask

• mask
    - Custom list of 0s and 1s to apply bounding mask
    - Default: dynamically inferred based on case (all zeros of correct size)
    - Order: [Pg] * ngen + [Qg] * ngen + [Va] * nbus + [Vm] * nbus

Training Parameters
-------------------------------------------------------------------------------

• epochs, batch_size, lr, patience, hidden_dim
    - Standard training loop hyperparameters

Constraint Loss Weights
-------------------------------------------------------------------------------

• lambda_vu, lambda_vl, lambda_pl, lambda_ql, lambda_th
• lambda_real, lambda_imag

Model-Specific Keys
-------------------------------------------------------------------------------

• lambda_ewc         (for EWC)
• alpha_threshold    (for progressive)
• prune_adapters     (for progressive)

Evaluation / Tuning
-------------------------------------------------------------------------------

• evaluate_only       = bool   (skip training and evaluate only)
• optuna_trials       = int    (if tuning enabled)
• optuna_sampler      = str    (e.g., "TPESampler")
• optuna_metric       = str    (e.g., "gen_loss", "val_loss")

Overrides and Config File Merging
-------------------------------------------------------------------------------

• CLI inline overrides can specify any key=value (e.g., batch_size=2048)
• Quoted Python expressions are supported (e.g., mask="[1]*10 + [0]*28")
• Final config resolution order:
      CLI overrides > config file > run_* default config

Available Models
-------------------------------------------------------------------------------

• stl
• adp
• adp_depth
• adp_width

"""

from __future__ import annotations
import argparse, importlib, pathlib, sys, os
from typing import Dict, Any
from pathlib import Path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ast
from Dyn_DNN4OPF.utils.repro import set_determinism
from Dyn_DNN4OPF.utils.config import infer_case_sizes
import time
import logging
import re
import pandas as pd

set_determinism()

MODEL_MODULES: dict[str, str] = {
    # Total entries: 66 (33 Dyn_DNN4OPF + 33 penalty_nn)

    # ----------------- Dyn_DNN4OPF -----------------
    # STL
    "stl":                          "Dyn_DNN4OPF.examples.run_dnn_stl",

    # DNN (den variants)
    "dnn_den_4head":                "Dyn_DNN4OPF.examples.run_dnn_den_4head",
    "dnn_den_2head":                "Dyn_DNN4OPF.examples.run_dnn_den_2head",
    "dnn_den":                      "Dyn_DNN4OPF.examples.run_dnn_den",

    # ADP (den)
    "adp":                          "Dyn_DNN4OPF.examples.run_adp_den",
    "adp_4head":                    "Dyn_DNN4OPF.examples.run_adp_den_4head",
    "adp_2head":                    "Dyn_DNN4OPF.examples.run_adp_den_2head",

    # ADP (den, depth_only)
    "adp_depth_only":               "Dyn_DNN4OPF.examples.run_adp_den_depth_only",
    "adp_4head_depth_only":         "Dyn_DNN4OPF.examples.run_adp_den_4head_depth_only",
    "adp_2head_depth_only":         "Dyn_DNN4OPF.examples.run_adp_den_2head_depth_only",

    # ADP (den, width_only)
    "adp_width_only":               "Dyn_DNN4OPF.examples.run_adp_den_width_only",
    "adp_4head_width_only":         "Dyn_DNN4OPF.examples.run_adp_den_4head_width_only",
    "adp_2head_width_only":         "Dyn_DNN4OPF.examples.run_adp_den_2head_width_only",

    # ADP (alt — alternative architecture variants)
    "adp_alt_depth":                "Dyn_DNN4OPF.examples.run_adp_alt_depth",
    "adp_alt_depth_2head":          "Dyn_DNN4OPF.examples.run_adp_alt_depth_2_head",
    "adp_alt_depth_4head":          "Dyn_DNN4OPF.examples.run_adp_alt_depth_4_head",
    "adp_alt_width":                "Dyn_DNN4OPF.examples.run_adp_alt_width",
    "adp_alt_width_2head":          "Dyn_DNN4OPF.examples.run_adp_alt_width_2_head",
    "adp_alt_width_4head":          "Dyn_DNN4OPF.examples.run_adp_alt_width_4_head",

    # ADP (depth sweep)
    "adp_depth":                    "Dyn_DNN4OPF.examples.run_adp_depth",
    "adp_depth_4head":              "Dyn_DNN4OPF.examples.run_adp_depth_4head",
    "adp_depth_2head":              "Dyn_DNN4OPF.examples.run_adp_depth_2head",

    # ADP (width sweep)
    "adp_width":                    "Dyn_DNN4OPF.examples.run_adp_width",
    "adp_width_4head":              "Dyn_DNN4OPF.examples.run_adp_width_4head",
    "adp_width_2head":              "Dyn_DNN4OPF.examples.run_adp_width_2head",

    # DNN (other heads/losses/regularizers)
    "dnn_mtl_4head":                "Dyn_DNN4OPF.examples.run_dnn_mtl_4head",
    "dnn_mtl_2head":                "Dyn_DNN4OPF.examples.run_dnn_mtl_2head",
    "dnn_mtl":                      "Dyn_DNN4OPF.examples.run_dnn_mtl",
    "dnn_l1":                       "Dyn_DNN4OPF.examples.run_dnn_l1",
    "dnn_l2":                       "Dyn_DNN4OPF.examples.run_dnn_l2",
    "dnn_ldf":                      "Dyn_DNN4OPF.examples.run_dnn_ldf",
    "dnn_elastic":                  "Dyn_DNN4OPF.examples.run_dnn_elastic",
    "dnn_mae":                      "Dyn_DNN4OPF.examples.run_dnn_mae",
    "dnn_fsnet":                    "Dyn_DNN4OPF.examples.run_dnn_fsnet",
    "dnn_pdl":                      "Dyn_DNN4OPF.examples.run_dnn_pdl",
    "dnn_dc3":                      "Dyn_DNN4OPF.examples.run_dnn_dc3",

    # DNN (progressive)
    "dnn_progressive_4head":        "Dyn_DNN4OPF.examples.run_dnn_progressive_4head",
    "dnn_progressive_2head":        "Dyn_DNN4OPF.examples.run_dnn_progressive_2head",
    "dnn_progressive":              "Dyn_DNN4OPF.examples.run_dnn_progressive",

    # DNN (EWC)
    "dnn_ewc_4head":                "Dyn_DNN4OPF.examples.run_dnn_ewc_4head",
    "dnn_ewc_2head":                "Dyn_DNN4OPF.examples.run_dnn_ewc_2head",
    "dnn_ewc":                      "Dyn_DNN4OPF.examples.run_dnn_ewc",

    # ----------------- penalty_nn -----------------
    # STL
    "penalty_stl":                  "penalty_nn.examples.run_penalty_stl",

    # penalty ADP (den)
    "penalty_adp_4head":            "penalty_nn.examples.run_penalty_adp_4head",
    "penalty_adp_2head":            "penalty_nn.examples.run_penalty_adp_2head",
    "penalty_adp":                  "penalty_nn.examples.run_penalty_adp_den",

    # penalty ADP (depth)
    "penalty_adp_depth":            "penalty_nn.examples.run_penalty_adp_depth",
    "penalty_adp_depth_4head":      "penalty_nn.examples.run_penalty_adp_depth_4head",
    "penalty_adp_depth_2head":      "penalty_nn.examples.run_penalty_adp_depth_2head",

    # penalty ADP (depth_only)
    "penalty_adp_depth_only":       "penalty_nn.examples.run_penalty_adp_depth_only",
    "penalty_adp_depth_only_4head": "penalty_nn.examples.run_penalty_adp_depth_only_4head",
    "penalty_adp_2head_depth_only": "penalty_nn.examples.run_penalty_adp_2head_depth_only",

    # penalty ADP (width_only)
    "penalty_adp_width_only":       "penalty_nn.examples.run_penalty_adp_width_only",
    "penalty_adp_width_only_4head": "penalty_nn.examples.run_penalty_adp_width_only_4head",
    "penalty_adp_2head_width_only": "penalty_nn.examples.run_penalty_adp_2head_width_only",

    # penalty ADP (width)
    "penalty_adp_width":            "penalty_nn.examples.run_penalty_adp_width",
    "penalty_adp_width_4head":      "penalty_nn.examples.run_penalty_adp_width_4head",
    "penalty_adp_width_2head":      "penalty_nn.examples.run_penalty_adp_width_2head",

    # penalty DEN
    "penalty_den_4head":            "penalty_nn.examples.run_penalty_den_4head",
    "penalty_den_2head":            "penalty_nn.examples.run_penalty_den_2head",
    "penalty_den":                  "penalty_nn.examples.run_penalty_den",

    # penalty ADP (alt — alternative architecture variants)
    "penalty_adp_alt_depth":        "penalty_nn.examples.run_penalty_adp_alt_depth",
    "penalty_adp_alt_depth_2head":  "penalty_nn.examples.run_penalty_adp_alt_depth_2_head",
    "penalty_adp_alt_depth_4head":  "penalty_nn.examples.run_penalty_adp_alt_depth_4_head",
    "penalty_adp_alt_width":        "penalty_nn.examples.run_penalty_adp_alt_width",
    "penalty_adp_alt_width_2head":  "penalty_nn.examples.run_penalty_adp_alt_width_2_head",
    "penalty_adp_alt_width_4head":  "penalty_nn.examples.run_penalty_adp_alt_width_4_head",

    # penalty (other heads/losses/regularizers)
    "penalty_ewc_4head":            "penalty_nn.examples.run_penalty_ewc_4head",
    "penalty_ewc_2head":            "penalty_nn.examples.run_penalty_ewc_2head",
    "penalty_ewc":                  "penalty_nn.examples.run_penalty_ewc",
    "penalty_mtl_4head":            "penalty_nn.examples.run_penalty_mtl_4head",
    "penalty_mtl_2head":            "penalty_nn.examples.run_penalty_mtl_2head",
    "penalty_mtl":                  "penalty_nn.examples.run_penalty_mtl",
    "penalty_progressive_4head":    "penalty_nn.examples.run_penalty_progressive_4head",
    "penalty_progressive_2head":    "penalty_nn.examples.run_penalty_progressive_2head",
    "penalty_progressive":          "penalty_nn.examples.run_penalty_progressive",
    "penalty_l1":                   "penalty_nn.examples.run_penalty_l1",
    "penalty_l2":                   "penalty_nn.examples.run_penalty_l2",
    "penalty_ldf":                  "penalty_nn.examples.run_penalty_ldf",
    "penalty_elastic":              "penalty_nn.examples.run_penalty_elastic",
    "penalty_mae":                  "penalty_nn.examples.run_penalty_mae",
    "penalty_fsnet":                "penalty_nn.examples.run_penalty_fsnet",
    "penalty_pdl":                  "penalty_nn.examples.run_penalty_pdl",
    "penalty_dc3":                  "penalty_nn.examples.run_penalty_dc3",
}


# Right now all neurons are unmasked
# To mask a neuron put the value 1
# Or one can manually specicify an entire list containing 38 collections of ones and zeros
# To find the most optimal solution maybe try with different permutes to be precise 38 factorial permutes of ones and zeroes
# The masking tensor should be filled in the order [5*Pg, 5*Qg, 14*Va, 14*Vm] in a contigous array

class Tee:
    def __init__(self, log_file):
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.__stdout__  # raw terminal
        self.log = open(log_file, "w", encoding="utf-8")  # "w" to overwrite each run

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_default_mask(case_name: str) -> list[int]:
    """Build a 0-mask dynamically for Pg, Qg, Va, Vm based on case_name."""
    sizes = infer_case_sizes(case_name)
    n_gen, n_bus = sizes["n_gen"], sizes["n_bus"]
    return [0] * (2 * n_gen + 2 * n_bus)

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Dyn_DNN4OPF launcher",
        description="Run any model pipeline from a single entry-point.")
    
    # Required model selection
    p.add_argument("-m", "--model", default="stl", choices=MODEL_MODULES,
                   help="Which model pipeline to invoke (default: stl)")
    
    p.add_argument("--hidden_dims",   type=int, default=10,
                   help="Number of hidden neurons in each layer")
    
    p.add_argument("--num_runs",      type=int, default=5,
                                    help="How many times to repeat each run")

    # Optional config file
    p.add_argument("-c", "--config", type=str,
                   help="Optional JSON/YAML file with parameter overrides")
    
    # Inline overrides (key=value)
    p.add_argument("overrides", nargs="*",
                   help="Extra KEY=VALUE pairs that override defaults")
    
    # Case selection
    p.add_argument("--case_name", "-k",
                   default="pglib_opf_case14_ieee",
                   help="Any PGLib OPF case (e.g. pglib_opf_case14_ieee)")

    # Sample sizes and batch selection
    p.add_argument("--train_samples", type=int, default=2000,
                   help="Number of training samples to load (default: 20000)")
    p.add_argument("--val_samples", type=int, default=None,
                   help="Number of validation samples to load (default:1500 )")
    p.add_argument("--test_samples", type=int, default=1500,
                   help="Number of test samples to load (default: 1500)")
    p.add_argument("--batches", type=str, default=None,
                   help='List of batches to load, e.g., "[1,2,3]"')
    
    return p

def _decode_value(v: str) -> Any:
    """Best-effort cast from CLI string to bool / int / float / str"""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:                      # try int/float first
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        pass
    if v.strip().startswith(("[", "(")):
        return ast.literal_eval(v)
    return v

def _parse_keyvals(pairs: list[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not of form KEY=VALUE")
        k, v = pair.split("=", 1)
        cfg[k] = _decode_value(v)
    return cfg

def main() -> None:
    args = _build_parser().parse_args()
    
    num_runs = args.num_runs

    if args.val_samples is None:
        args.val_samples = args.train_samples // 18

    if args.test_samples is None:
        args.test_samples = 1500
        
    # ——— dynamic logging per run configuration ———
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{args.model}_{args.case_name}_" \
            f"{args.train_samples}samples_{args.hidden_dims}neurons_{num_runs}runs.txt"
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    # Interactive input prompts if required arguments are not provided
    models = list(MODEL_MODULES.keys())
    print("\nAvailable models:")
    for i, name in enumerate(models, 1):
        print(f"  {i:2d}. {name}")
    default_model = models[0]  # expected 'stl'
    while True:
        choice = input(f"Select model by number (default: 1 = '{default_model}'): ").strip()
        if choice == "":
            args.model = default_model
            break
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                args.model = models[idx - 1]
                break
            else:
                print(f"Invalid number. Enter a value between 1 and {len(models)}.")
        except ValueError:
            print("Please enter a valid integer.")

    args.case_name = input("Enter the OPF case name (default: pglib_opf_case14_ieee): ") or "pglib_opf_case14_ieee"
    args.hidden_dims = int(input("Enter the no of hidden dimensions (default: 10): ") or 10 )
    args.train_samples = int(input("Enter the number of training samples (max value : 270000 and def : 20000): ") or 20000)
    args.val_samples = args.train_samples // 18
    args.test_samples = int(input("Enter the number of test samples (max value : 15000 and def: 1500): ") or 1500)
    num_runs = int(input("Enter how many times to run the model (default: 5): ") or 5)

    # # Interactive threshold inputs
    # loss_thr = input("Enter MSE loss threshold for expansion (default: 1e-3): ") or "1e-3"
    # dp_thr   = input("Enter ΔP mean-violation threshold (default: 0.005): ") or "0.005"
    # dq_thr   = input("Enter ΔQ mean-violation threshold (default: 0.003): ") or "0.003"
    # pg_thr   = input("Enter PG mean-violation threshold (default: 0.002): ") or "0.002"
    # qg_thr   = input("Enter QG mean-violation threshold (default: 0.0015): ") or "0.0015"
    # vm_thr   = input("Enter VM mean-violation threshold (default: 0.025): ") or "0.025"

    mod_name = MODEL_MODULES[args.model]
    mod = importlib.import_module(mod_name)

    # 2. start from that script’s DEFAULT_CONFIG 
    try:
        config = mod.DEFAULT_CONFIG
    except AttributeError as e:
        sys.exit(f"[FATAL] {mod_name} must expose DEFAULT_CONFIG dict – {e}")

    # Update the DEFAULT_CONFIG with user inputs
    config["case_name"] = args.case_name
    config["train_samples"] = args.train_samples
    config["val_samples"] = args.val_samples
    config["test_samples"] = args.test_samples

    config['hidden_dim'] = args.hidden_dims
    config['h1_dim'] = args.hidden_dims
    config['h2_dim'] = args.hidden_dims
    
    # #Assign to config
    # config["loss_thr"] = float(loss_thr)
    # config["dp_thr"]   = float(dp_thr)
    # config["dq_thr"]   = float(dq_thr)
    # config["pg_thr"]   = float(pg_thr)
    # config["qg_thr"]   = float(qg_thr)
    # config["vm_thr"]   = float(vm_thr)

    for run_idx in range(num_runs):

        # ─── log model‐size & sample‐size ───────────────────────────────────────────
        neurons = config['hidden_dim']  # you could also compute total neurons if you have more layers
        logging.info(
            f"🚀 Starting runs: model={args.model!r}, hidden_dim={neurons}, "
            f"train_samples={config['train_samples']}, "
            f"val_samples={config['val_samples']}, "
            f"test_samples={config['test_samples']}"
        )
        print(
            f"🔧 Configuration: model={args.model}, hidden_dim={neurons}, "
            f"train={config['train_samples']}, val={config['val_samples']}, "
            f"test={config['test_samples']}"
        )

        print(f"\n🔁 Running model iteration {run_idx + 1}/{num_runs}")

        start_time = time.time()
        mod.run_pipeline(config)
        end_time = time.time()
        elapsed = end_time - start_time

        # Log the elapsed time
        logging.info(f"✅ Run {run_idx + 1}: Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

        summary_csv_path = pathlib.Path("Results/SUMMARY.csv")

        if summary_csv_path.exists():
            df = pd.read_csv(summary_csv_path)

            # Format as 3m 2.45s
            minutes = int(elapsed // 60)
            seconds = round(elapsed % 60, 2)
            format_time = f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

            # ✅ Update the last row's "Time Taken" column safely
            if "Time Taken" in df.columns:
                df.loc[df.index[-1], "Time Taken"] = format_time
                df.to_csv(summary_csv_path, index=False)
                logging.info(f"✅ Time Taken updated in last row of {summary_csv_path}")
            else:
                logging.warning(f"⚠️ 'Time Taken' column missing in SUMMARY.csv. Not updated.")
        else:
            logging.warning(f"⚠️ SUMMARY.csv not found — couldn't update with time.")

    # def process_summary_data(summary_csv_path: str):
    #     if not Path(summary_csv_path).exists():
    #         print(f"SUMMARY.csv not found at {summary_csv_path}")
    #         return

    #     # Load the SUMMARY.csv file
    #     summary_data = pd.read_csv(summary_csv_path)

    #     # Define paths for min_data and mean_data
    #     min_data_path = summary_csv_path.replace("SUMMARY.csv", "min_data.csv")
    #     mean_data_path = summary_csv_path.replace("SUMMARY.csv", "mean_data.csv")

    #     # Process Min Data
    #     min_data = summary_data.loc[
    #         summary_data.groupby(['Model Name', 'Case Name', 'Train Samples','Test Samples','Total_Neurons'])['Mean MSE'].idxmin()
    #     ]
    #     if Path(min_data_path).exists():
    #         existing_min_data = pd.read_csv(min_data_path)
    #         min_data = pd.concat([existing_min_data, min_data]).drop_duplicates(
    #             subset=['Model Name', 'Case Name', 'Train Samples', 'Test Samples','Total_Neurons']
    #         )
    #     min_data.to_csv(min_data_path, index=False)
    #     print(f"Min Data saved to {min_data_path}")

    #     # Process Mean Data
    #     mean_data = summary_data.groupby(
    #         ['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'], as_index=False
    #     ).mean(numeric_only=True)

    #     # Retain all non-numerical columns
    #     non_numeric_data = summary_data[
    #         ['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons']
    #     ].drop_duplicates()
    #     mean_data = pd.merge(non_numeric_data, mean_data, on=['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'], how='inner')

    #     if Path(mean_data_path).exists():
    #         existing_mean_data = pd.read_csv(mean_data_path)
    #         mean_data = pd.concat([existing_mean_data, mean_data]).drop_duplicates(
    #             subset=['Model Name', 'Case Name', 'Train Samples', 'Test Samples','Total_Neurons']
    #         )
    #     mean_data.to_csv(mean_data_path, index=False)
    #     print(f"Mean Data saved to {mean_data_path}")

    # ── helpers ─────────────────────────────────────────────────────────────────────
    _time_pat = re.compile(r'(?:(\d+)m\s*)?([\d.]+)s')

    def _time_to_secs(t: str | float | int | None) -> float | None:
        """
        Convert '3m 2.45s' → 182.45 (float seconds).
        Returns None if the value is missing or cannot be parsed.
        """
        if pd.isna(t):
            return None
        m = _time_pat.match(str(t).strip())
        if not m:
            return None
        mins = int(m.group(1)) if m.group(1) else 0
        secs = float(m.group(2))
        return mins * 60 + secs


    def _secs_to_hms(elapsed: float | None) -> str:
        """
        Convert seconds back to the original string format.
        182.45 → '3m 2.45s', 45.32 → '45.32s'
        """
        if pd.isna(elapsed):
            return ""
        minutes = int(elapsed // 60)
        seconds = round(elapsed % 60, 2)
        return f"{minutes}m {seconds:.2f}s" if minutes else f"{seconds:.2f}s"


    def process_summary_data(summary_csv_path: str) -> None:
        path = Path(summary_csv_path)

        if not path.exists():
            logging.warning(f"SUMMARY.csv not found at {summary_csv_path}")
            return

        summary_data = pd.read_csv(path)

        if "Time Taken" not in summary_data.columns:
            logging.warning("'Time Taken' column missing in SUMMARY.csv – skipping time aggregation.")
            summary_data["Time Taken"] = pd.NA

        summary_data["Time Taken (s)"] = summary_data["Time Taken"].apply(_time_to_secs)

        adp_data = summary_data[summary_data["Model Name"] == "ADP-DEN"]
        other_data = summary_data[summary_data["Model Name"] != "ADP-DEN"]

        adp_depth_data = summary_data[summary_data["Model Name"] == "ADP-DEPTH"]
        other_depth_data = summary_data[summary_data["Model Name"] != "ADP-DEPTH"]

        adp_width_data  = summary_data[summary_data["Model Name"] == "ADP-WIDTH"]
        other_width_data= summary_data[summary_data["Model Name"] != "ADP-WIDTH"]

        # MIN DATA
        min_adp = adp_data.loc[
            adp_data.groupby(
                ["Model Name", "Case Name", "Train Samples", "Test Samples"]
            )["Mean MSE"].idxmin()
        ]
        min_depth = adp_depth_data.loc[
            adp_depth_data.groupby(
                ["Model Name", "Case Name", "Train Samples", "Test Samples"]
            )["Mean MSE"].idxmin()
        ]
        min_other_adp = other_data.loc[
            other_data.groupby(
                ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
            )["Mean MSE"].idxmin()
        ]
        min_other_depth = other_depth_data.loc[
            other_depth_data.groupby(
                ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
            )["Mean MSE"].idxmin()
        ]
        min_width = adp_width_data.loc[
            adp_width_data.groupby(
                ["Model Name", "Case Name", "Train Samples", "Test Samples"]
            )["Mean MSE"].idxmin()
        ]
        min_other_width = other_width_data.loc[
            other_width_data.groupby(
                ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"]
            )["Mean MSE"].idxmin()
        ]
        min_data = pd.concat([min_adp, min_other_adp], ignore_index=True)
        min_data_depth = pd.concat([min_depth, min_other_depth], ignore_index=True)
        min_data_width  = pd.concat([min_width,  min_other_width ], ignore_index=True)

        min_path = path.with_name("min_data.csv")
        if min_path.exists():
            existing = pd.read_csv(min_path)
            min_data = pd.concat([existing, min_data]).drop_duplicates(
                subset=["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
                keep="last"
            )
        min_data.to_csv(min_path, index=False)
        logging.info(f"Min Data saved to {min_path}")

        min_path_depth = path.with_name("min_data_depth.csv")
        if min_path_depth.exists():
            existing_depth = pd.read_csv(min_path_depth)
            min_data_depth = pd.concat([existing_depth, min_data_depth]).drop_duplicates(
                subset=["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
                keep="last"
            )
        min_data_depth.to_csv(min_path_depth, index=False)
        logging.info(f"Min Depth Data saved to {min_path_depth}")

        min_path_width = path.with_name("min_data_width.csv")
        if min_path_width.exists():
            existing_w = pd.read_csv(min_path_width)
            min_data_width = pd.concat([existing_w, min_data_width]).drop_duplicates(
                subset=["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
                keep="last"
            )
        min_data_width.to_csv(min_path_width, index=False)
        logging.info(f"Min Width Data saved to {min_path_width}")

        mean_adp = adp_data.groupby(
            ["Model Name", "Case Name", "Train Samples", "Test Samples"],
            as_index=False
        ).mean(numeric_only=True)

        mean_other = other_data.groupby(
            ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
            as_index=False
        ).mean(numeric_only=True)

        for df in (mean_adp, mean_other):
            if "Time Taken (s)" in df.columns:
                df["Time Taken"] = df["Time Taken (s)"].apply(_secs_to_hms)
                df.drop(columns=["Time Taken (s)"], inplace=True)

        mean_data = pd.concat([mean_adp, mean_other], ignore_index=True)

        mean_path = path.with_name("mean_data.csv")
        if mean_path.exists():
            existing = pd.read_csv(mean_path)
            mean_data = pd.concat([existing, mean_data]).drop_duplicates(
                subset=["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
                keep="last"
            )
        if "Time Taken" in mean_data.columns:
            cols = mean_data.columns.tolist()
            cols.remove("Time Taken")
            mean_data = mean_data[cols[:2] + ["Time Taken"] + cols[2:]]
        mean_data.to_csv(mean_path, index=False)
        logging.info(f"Mean Data saved to {mean_path}")

        mean_depth_adp = adp_depth_data.groupby(
            ["Model Name", "Case Name", "Train Samples", "Test Samples"],
            as_index=False
        ).mean(numeric_only=True)

        mean_depth_other = other_depth_data.groupby(
            ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
            as_index=False
        ).mean(numeric_only=True)

        mean_width_adp = adp_width_data.groupby(
            ["Model Name", "Case Name", "Train Samples", "Test Samples"],
            as_index=False
        ).mean(numeric_only=True)
        mean_width_other = other_width_data.groupby(
            ["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
            as_index=False
        ).mean(numeric_only=True)
        for df in (mean_width_adp, mean_width_other):
            if "Time Taken (s)" in df.columns:
                df["Time Taken"] = df["Time Taken (s)"].apply(_secs_to_hms)
                df.drop(columns=["Time Taken (s)"], inplace=True)

        for df in (mean_depth_adp, mean_depth_other):
            if "Time Taken (s)" in df.columns:
                df["Time Taken"] = df["Time Taken (s)"].apply(_secs_to_hms)
                df.drop(columns=["Time Taken (s)"], inplace=True)

        mean_data_depth = pd.concat([mean_depth_adp, mean_depth_other], ignore_index=True)
        mean_data_width = pd.concat([mean_width_adp,  mean_width_other ], ignore_index=True)
        mean_path_depth = path.with_name("mean_data_depth.csv")
        if mean_path_depth.exists():
            existing = pd.read_csv(mean_path_depth)
            mean_data_depth = pd.concat([existing, mean_data_depth]).drop_duplicates(
                subset=["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
                keep="last"
            )
        if "Time Taken" in mean_data_depth.columns:
            cols = mean_data_depth.columns.tolist()
            cols.remove("Time Taken")
            mean_data_depth = mean_data_depth[cols[:2] + ["Time Taken"] + cols[2:]]
        mean_data_depth.to_csv(mean_path_depth, index=False)
        logging.info(f"Mean Depth Data saved to {mean_path_depth}")

        mean_path_width = path.with_name("mean_data_width.csv")
        if mean_path_width.exists():
            existing_w = pd.read_csv(mean_path_width)
            mean_data_width = pd.concat([existing_w, mean_data_width]).drop_duplicates(
                subset=["Model Name", "Case Name", "Train Samples", "Test Samples", "Total_Neurons"],
                keep="last"
            )
        if "Time Taken" in mean_data_width.columns:
            cols = mean_data_width.columns.tolist()
            cols.remove("Time Taken")
            mean_data_width = mean_data_width[cols[:2] + ["Time Taken"] + cols[2:]]
        mean_data_width.to_csv(mean_path_width, index=False)
        logging.info(f"Mean Width Data saved to {mean_path_width}")

        # Run the function
        summary_csv_path = "Results/SUMMARY.csv"
        process_summary_data(summary_csv_path)

if __name__ == "__main__":
    main()
