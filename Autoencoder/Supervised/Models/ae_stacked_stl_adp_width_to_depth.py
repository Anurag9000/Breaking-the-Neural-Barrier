import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_stacked_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_STACKED_STL = baseline_module.AE_STACKED_STL  # type: ignore
ConvBlock = baseline_module.ConvBlock  # type: ignore
Mix1x1 = baseline_module.Mix1x1  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 24
    max_neurons: int = 8_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 20
    pool_after: List[int] = None
    mix_every: int = 0


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(new_state, old_state):
    merged = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(model: AE_STACKED_STL, width: int, depth: int, device, pool_after: List[int], mix_every: int) -> AE_STACKED_STL:
    new_model = AE_STACKED_STL(in_channels=model.in_channels, width=width, depth=depth, pool_after=pool_after, mix_every=mix_every).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))


def train_with_patience(model: AE_STACKED_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf"); best_state=None; pat=acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            rec, _ = model(x)
            loss = F.mse_loss(rec, x)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = 0.0; n = 0
            for x, _ in dl_val:
                x = x.to(device)
                rec, _ = model(x)
                l = F.mse_loss(rec, x)
                val += l.item(); n += 1
            val = val / max(n,1)
        history.append(val)
        if val < best - acfg.delta:
            best = val; best_state = copy.deepcopy(model.state_dict()); pat = acfg.patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: AE_STACKED_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_stacked_stl")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    def can_widen(width: int, depth: int):
        new_w = min(acfg.max_width, width + acfg.ex_k)
        return new_w > width and total_neurons(new_w, depth) <= acfg.max_neurons

    def can_deepen(width: int, depth: int):
        return depth + 1 <= acfg.max_depth and total_neurons(width, depth + 1) <= acfg.max_neurons

    cur_w, cur_d = model.width, model.depth
    best_val = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
    best_state = copy.deepcopy(model.state_dict())
    improvements.append((total_neurons(cur_w, cur_d), best_val))

    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only","width","width_to_depth","alt_width"):
            if can_widen(cur_w, cur_d) and pw > 0:
                pre_state = copy.deepcopy(model.state_dict()); pre_w = cur_w; pre_val = best_val
                cur_w = min(acfg.max_width, cur_w + acfg.ex_k)
                model = rebuild_model(model, cur_w, cur_d, device, list(model.pool_after), model.mix_every)
                v = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
                if v < pre_val - acfg.delta:
                    best_val = v; pw = acfg.trials_width; improved=True
                    best_state = copy.deepcopy(model.state_dict())
                    improvements.append((total_neurons(cur_w, cur_d), best_val))
                else:
                    model.load_state_dict(pre_state); cur_w = pre_w; pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only","depth","depth_to_width","alt_depth"):
            if can_deepen(cur_w, cur_d) and pd > 0:
                pre_state = copy.deepcopy(model.state_dict()); pre_d = cur_d; pre_val = best_val
                cur_d += 1
                model = rebuild_model(model, cur_w, cur_d, device, list(model.pool_after), model.mix_every)
                v = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
                if v < pre_val - acfg.delta:
                    best_val = v; pd = acfg.trials_depth; improved=True
                    best_state = copy.deepcopy(model.state_dict())
                    improvements.append((total_neurons(cur_w, cur_d), best_val))
                else:
                    model.load_state_dict(pre_state); cur_d = pre_d; pd -= 1
            if mode == "depth_only":
                continue
        if mode == "width_to_depth" and not improved:
            mode = "depth"; pd = acfg.trials_depth; improved=True
        elif mode == "depth_to_width" and not improved:
            mode = "width"; pw = acfg.trials_width; improved=True
        elif mode in ("alt_width","alt_depth"):
            mode = "depth" if mode=="alt_width" else "width"; improved=True
    model.load_state_dict(best_state)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val, model, cur_w, cur_d


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP Stacked AE width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--pool-after", type=int, nargs="*", default=[])
    p.add_argument("--mix-every", type=int, default=0)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=24)
    p.add_argument("--max-neurons", type=int, default=8_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_stacked_stl"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_STACKED_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after, mix_every=args.mix_every).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, pool_after=args.pool_after, mix_every=args.mix_every)
    best, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP Stacked AE] mode={args.adp_mode} best_val={best:.6f} width={w} depth={d}")


if __name__ == "__main__":
    main()
