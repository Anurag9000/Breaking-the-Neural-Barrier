import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_qaware_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_QAWARE_STL = baseline_module.AE_QAWARE_STL  # type: ignore
STEQuant = baseline_module.STEQuant  # type: ignore
ae_qaware_total_neurons = baseline_module.ae_qaware_total_neurons  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 12
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 20
    n_bits: int = 8
    per_channel: bool = False
    quant_everywhere: bool = False
    pool_after: Optional[List[int]] = None


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    tgt_slices = tuple(slice(0, c) for c in common)
    src_slices = tuple(slice(0, c) for c in common)
    tgt[tgt_slices] = src[src_slices]
    return tgt


def _copy_over(new: Dict[str, torch.Tensor], old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in new.items():
        if k in old:
            if old[k].shape == v.shape:
                out[k] = old[k]
            else:
                out[k] = _resize_tensor(v.shape, old[k])
        else:
            out[k] = v
    return out


def rebuild_model(model: AE_QAWARE_STL, width: int, depth: int, device, acfg: ADPConfig) -> AE_QAWARE_STL:
    pool_after = list(model.pool_after)
    new_model = AE_QAWARE_STL(
        in_channels=model.in_channels,
        width=width,
        depth=depth,
        pool_after=pool_after,
        n_bits=acfg.n_bits,
        per_channel=acfg.per_channel,
        quant_everywhere=acfg.quant_everywhere,
    ).to(device)
    old_sd = model.state_dict()
    new_sd = new_model.state_dict()
    merged = _copy_over(new_sd, old_sd)
    new_model.load_state_dict(merged, strict=False)
    return new_model


def train_with_patience(model: AE_QAWARE_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list):
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
            val = 0.0; n=0
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


def adp_search(model: AE_QAWARE_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_qaware")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    def can_widen(width: int, depth: int) -> bool:
        new_w = min(acfg.max_width, width + acfg.ex_k)
        return ae_qaware_total_neurons(new_w, depth) <= acfg.max_neurons and new_w > width

    def can_deepen(width: int, depth: int) -> bool:
        return depth + 1 <= acfg.max_depth and ae_qaware_total_neurons(width, depth + 1) <= acfg.max_neurons

    cur_width, cur_depth = model.width, model.depth
    best_val = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
    best_state = copy.deepcopy(model.state_dict())
    improvements.append((ae_qaware_total_neurons(cur_width, cur_depth), best_val))

    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only","width","width_to_depth","alt_width"):
            if can_widen(cur_width, cur_depth) and pw>0:
                pre_state = copy.deepcopy(model.state_dict()); pre_w = cur_width; pre_val = best_val
                cur_width = min(acfg.max_width, cur_width + acfg.ex_k)
                model = rebuild_model(model, cur_width, cur_depth, device, acfg)
                v = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
                if v < pre_val - acfg.delta:
                    best_val = v; pw = acfg.trials_width; improved=True
                    best_state = copy.deepcopy(model.state_dict())
                    improvements.append((ae_qaware_total_neurons(cur_width, cur_depth), best_val))
                else:
                    model.load_state_dict(pre_state); cur_width = pre_w; pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only","depth","depth_to_width","alt_depth"):
            if can_deepen(cur_width, cur_depth) and pd>0:
                pre_state = copy.deepcopy(model.state_dict()); pre_d = cur_depth; pre_val = best_val
                cur_depth += 1
                model = rebuild_model(model, cur_width, cur_depth, device, acfg)
                v = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
                if v < pre_val - acfg.delta:
                    best_val = v; pd = acfg.trials_depth; improved=True
                    best_state = copy.deepcopy(model.state_dict())
                    improvements.append((ae_qaware_total_neurons(cur_width, cur_depth), best_val))
                else:
                    model.load_state_dict(pre_state); cur_depth = pre_d; pd -= 1
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
    return best_val, model, cur_width, cur_depth


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
    p = argparse.ArgumentParser(description="ADP Quantization-Aware AE width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool-after", type=int, nargs="*", default=[])
    p.add_argument("--n-bits", type=int, default=8)
    p.add_argument("--per-channel", action="store_true", default=False)
    p.add_argument("--quant-everywhere", action="store_true", default=False)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_qaware"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_QAWARE_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after,
                          n_bits=args.n_bits, per_channel=args.per_channel, quant_everywhere=args.quant_everywhere).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, n_bits=args.n_bits,
                     per_channel=args.per_channel, quant_everywhere=args.quant_everywhere, pool_after=args.pool_after)
    best, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP QAware AE] mode={args.adp_mode} best_val={best:.6f} width={w} depth={d}")


if __name__ == "__main__":
    main()
