import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger

BASE_PATH = Path(__file__).with_name("ae_ortho_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_ORTHO_STL = baseline_module.AE_ORTHO_STL  # type: ignore
orthogonality_penalty = baseline_module.orthogonality_penalty  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore of arch/state.


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 16
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    lam_ortho: float = 1e-3


def rebuild_model(width: int, depth: int, pool_after: List[int]) -> AE_ORTHO_STL:
    return AE_ORTHO_STL(in_channels=3, width=width, depth=depth, pool_after=pool_after)


def widen_model(model: AE_ORTHO_STL, ex_k: int, max_width: int):
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return
    new_model = rebuild_model(new_w, model.depth, list(model.pool_after))
    new_model.load_state_dict(model.state_dict(), strict=False)
    return new_model


def deepen_model(model: AE_ORTHO_STL):
    new_model = rebuild_model(model.width, model.depth + 1, list(model.pool_after))
    new_model.load_state_dict(model.state_dict(), strict=False)
    return new_model


def total_neurons(model: AE_ORTHO_STL) -> int:
    return model.width * (model.depth + 1)


def snapshot_arch_and_state(model: AE_ORTHO_STL, state_dict=None):
    state = state_dict if state_dict is not None else model.state_dict()
    return {"width": model.width, "depth": model.depth, "pool_after": list(model.pool_after), "state": copy.deepcopy(state)}


def restore_arch_and_state(model: AE_ORTHO_STL, snap, device):
    restored = rebuild_model(snap["width"], snap["depth"], list(snap["pool_after"])).to(device)
    restored.load_state_dict(snap["state"])
    return restored


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    sys.path.append(str(Path(__file__).resolve().parents[1] / "Runs"))
    from _common_real_image import make_real_image_loaders
    dl_train, dl_val, _ = make_real_image_loaders(
        data_root="./data",
        batch_size=batch_size,
        val_ratio=val_split,
        num_workers=4,
        image_size=224,
    )
    return dl_train, dl_val

def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Orthogonal AE (Supervised) width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool-after", type=int, nargs="*", default=[])
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lam-ortho", type=float, default=1e-3)
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_ORTHO_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after).to(device)
    acfg = ADPConfig(
        adp_mode=args.adp_mode,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        max_epochs=args.max_epochs,
        lam_ortho=args.lam_ortho,
    )
    results_dir = Path(f"results_{BASE_PATH.stem}")
    # Initialize Logger
    logger = ContinuousLogger(results_dir, "ae_ortho_stl", args.adp_mode)
    
    best_val, model, width, depth = adp_search(model, dl_train, dl_val, acfg, device, logger=logger, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    logger.log_console(f"Done. Best val={best_val} w={width} d={depth}")
    logger.close()
    print(f"[ADP Ortho AE STL] mode={args.adp_mode} best_val={best_val:.6f} width={width} depth={depth}")
