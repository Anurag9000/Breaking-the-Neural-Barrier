import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

BASE_PATH = Path(__file__).with_name("ae_graph_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_GRAPH_STL = baseline_module.AE_GRAPH_STL  # type: ignore
ae_graph_total_neurons = baseline_module.ae_graph_total_neurons  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 32
    max_width: int = 512
    max_depth: int = 12
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 20
    patch_size: int = 4


def rebuild_model(dim: int, depth: int, patch_size: int) -> AE_GRAPH_STL:
    return AE_GRAPH_STL(in_channels=3, patch_size=patch_size, dim=dim, depth=depth)


def widen_model(model: AE_GRAPH_STL, ex_k: int, max_width: int):
    new_dim = min(max_width, model.dim + ex_k)
    if new_dim == model.dim:
        return model
    new_model = rebuild_model(new_dim, model.depth, model.ps)
    new_model.load_state_dict(model.state_dict(), strict=False)
    return new_model


def deepen_model(model: AE_GRAPH_STL):
    new_model = rebuild_model(model.dim, model.depth + 1, model.ps)
    new_model.load_state_dict(model.state_dict(), strict=False)
    return new_model


def total_neurons(model: AE_GRAPH_STL) -> int:
    return ae_graph_total_neurons(model.dim, model.depth)


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def train_with_patience(model: AE_GRAPH_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.MSELoss()
    best = float("inf")
    best_state = None
    pat = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            x_rec, _ = model(x)
            loss = crit(x_rec, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = 0.0
            n = 0
            for x, _ in dl_val:
                x = x.to(device)
                x_rec, _ = model(x)
                l = crit(x_rec, x)
                val += l.item() * x.size(0)
                n += x.size(0)
            val = val / max(n, 1)
        history.append(val)
        if val < best - acfg.delta:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            pat = acfg.patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: AE_GRAPH_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    val_history = []
    improvements = []
    def can_widen():
        return (model.dim + acfg.ex_k) <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen():
        return (model.depth + 1) <= acfg.max_depth and (total_neurons(model) + model.dim) <= acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
    best_val, best_state = inner_val, copy.deepcopy(model.state_dict())
    improvements.append((total_neurons(model), inner_val))
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre_state = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                pre_dim = model.dim
                model = widen_model(model, acfg.ex_k, acfg.max_width).to(device)
                v = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    improvements.append((total_neurons(model), inner_val))
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre_state)
                    model.dim = pre_dim
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre_state = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                pre_d = model.depth
                model = deepen_model(model).to(device)
                v = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
                    improvements.append((total_neurons(model), inner_val))
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre_state)
                    model.depth = pre_d
                    pd -= 1
            if mode == "depth_only":
                continue
        if mode == "width_to_depth" and not improved:
            mode = "depth"
            pd = acfg.trials_depth
            improved = True
        elif mode == "depth_to_width" and not improved:
            mode = "width"
            pw = acfg.trials_width
            improved = True
        elif mode in ("alt_width", "alt_depth"):
            mode = "depth" if mode == "alt_width" else "width"
            improved = True
    model.load_state_dict(best_state)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val


def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Graph AE (Supervised) width/depth search")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"],
    )
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_GRAPH_STL(in_channels=3, patch_size=args.patch_size, dim=args.dim, depth=args.depth).to(device)
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
        patch_size=args.patch_size,
    )
    results_dir = Path(f"results_{BASE_PATH.stem}")
    best = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    print(f"[ADP Graph AE STL] mode={args.adp_mode} best_val={best:.6f} dim={model.dim} depth={model.depth}")


if __name__ == "__main__":
    main()
