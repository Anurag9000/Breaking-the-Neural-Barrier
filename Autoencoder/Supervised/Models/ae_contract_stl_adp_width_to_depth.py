import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_contract_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_CONTRACT_STL = baseline_module.AE_CONTRACT_STL  # type: ignore
ConvBlock = baseline_module.ConvBlock  # type: ignore
DeconvBlock = baseline_module.DeconvBlock  # type: ignore
contractive_penalty_hutchinson = baseline_module.contractive_penalty_hutchinson  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 16
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 20
    lam_contractive: float = 1e-3
    hutch_iters: int = 1


def rebuild_model(width: int, depth: int, pool_after: List[int]) -> AE_CONTRACT_STL:
    return AE_CONTRACT_STL(in_channels=3, width=width, depth=depth, pool_after=pool_after)


def widen_model(model: AE_CONTRACT_STL, ex_k: int, max_width: int):
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return
    new_model = rebuild_model(new_w, model.depth, list(model.pool_after))
    new_model.load_state_dict(model.state_dict(), strict=False)
    return new_model


def deepen_model(model: AE_CONTRACT_STL):
    new_d = model.depth + 1
    new_model = rebuild_model(model.width, new_d, list(model.pool_after))
    new_model.load_state_dict(model.state_dict(), strict=False)
    return new_model


def total_neurons(model: AE_CONTRACT_STL) -> int:
    return model.width * (model.depth + 1)


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def train_with_patience(model: AE_CONTRACT_STL, dl_train, dl_val, acfg: ADPConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.MSELoss()
    best = float("inf")
    best_state = None
    pat = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            x_rec, z = model(x)
            rec_loss = crit(x_rec, x)
            pen = acfg.lam_contractive * contractive_penalty_hutchinson(model.encoder, x, acfg.hutch_iters)
            loss = rec_loss + pen
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


def adp_search(model: AE_CONTRACT_STL, dl_train, dl_val, acfg: ADPConfig, device):
    def can_widen():
        return (model.width + acfg.ex_k) <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen():
        return (model.depth + 1) <= acfg.max_depth and (total_neurons(model) + model.width) <= acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device)
    best_val, best_state = inner_val, copy.deepcopy(model.state_dict())
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre_state = copy.deepcopy(model.state_dict())
                pre_w = model.width
                pre_val = inner_val
                new_model = widen_model(model, acfg.ex_k, acfg.max_width)
                if new_model is not None:
                    model = new_model.to(device)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre_state)
                    model.width = pre_w
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre_state = copy.deepcopy(model.state_dict())
                pre_d = model.depth
                pre_val = inner_val
                model = deepen_model(model).to(device)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
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
    return best_val


def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Contractive AE (Supervised) width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool-after", type=int, nargs="*", default=[])
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
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lam-contract", type=float, default=1e-3)
    p.add_argument("--hutch-iters", type=int, default=1)
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_CONTRACT_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after).to(device)
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
        lam_contractive=args.lam_contract,
        hutch_iters=args.hutch_iters,
    )
    best = adp_search(model, dl_train, dl_val, acfg, device)
    print(f"[ADP Contractive AE STL] mode={args.adp_mode} best_val={best:.6f} width={model.width} depth={model.depth}")


if __name__ == "__main__":
    main()
