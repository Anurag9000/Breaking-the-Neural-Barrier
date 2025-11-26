import argparse
import torch
from torch_geometric.datasets import OPFDataset
from Dyn_DNN4OPF.data.opf_loader import flatten_heterodata
import yaml
def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--case_name", type=str, default="pglib_opf_case14_ieee",
                        help="Which OPF case (must be one of VALID_CASES).")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_outputs", action="store_true")
    parser.add_argument("--lambda_eq", type=float, default=0.5)
    parser.add_argument("--lambda_ineq", type=float, default=0.5)
    parser.add_argument("--dataset", type=str, default="case14")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--config", type=str, default="")
    return parser

def load_config_file(config_path: str) -> dict:
    if config_path:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def merge_configs(defaults: dict, file_cfg: dict, cli_cfg: dict) -> dict:
    cfg = defaults.copy()
    cfg.update(file_cfg)
    cfg.update({k: v for k, v in cli_cfg.items() if v is not None})
    return cfg

DEFAULT_CFG = {
    "lr": 0.001,
    "epochs": 100,
    "batch_size": 64,
    "clip_outputs": False,
    "lambda_eq": 0.5,
    "lambda_ineq": 0.5,
    "dataset": "case14",
    "task_id": 0
}

def infer_case_sizes(case_name: str, root: str = "data") -> dict[str, int]:
    """
    Load one sample from the given OPF 'case_name' and return:
      - n_bus   : number of buses
      - n_gen   : number of generators
      - in_dim  : flattened input dimension  (2 * n_bus)
      - out_dim : flattened output dimension (2 * (n_gen + n_bus))
    """
    sample = OPFDataset(root=root, case_name=case_name, num_groups=1, split="train")[0]
    # count buses/gens:
    n_bus = sample["bus"].num_nodes
    n_gen = sample["generator"].num_nodes

    # flatten to x and y tensors:
    x_flat, y_flat = flatten_heterodata(sample)
    return {
        "n_bus":   n_bus,
        "n_gen":   n_gen,
        "in_dim":  x_flat.numel(),
        "out_dim": y_flat.numel(),
    }

def get_io_dims_from_loader(loader: torch.utils.data.DataLoader) -> tuple[int, int]:
    x, y,_ = loader.dataset.tensors
    return x.shape[1], y.shape[1]

def default_mask(n_gen: int, n_bus: int) -> torch.Tensor:
    # 2*n_gen (for PG, QG) + 2*n_bus (for VA, VM)
    return torch.ones(2 * n_gen + 2 * n_bus, dtype=torch.int)

def check_bounds_compatibility(
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
    mask: torch.Tensor,
    output_dim: int
) -> None:
    if not (len(bounds_low) == len(bounds_high) == len(mask) == output_dim):
        raise ValueError(
            f"Bounds/mask mismatch: Expected length {output_dim}, got "
            f"{len(bounds_low)}, {len(bounds_high)}, {len(mask)}"
        )

__all__ = [
    "build_arg_parser",
    "load_config_file",
    "merge_configs",
    "infer_case_sizes",
    "get_io_dims_from_loader",
    "default_mask",
    "check_bounds_compatibility",
]
