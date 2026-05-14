import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_feature.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
FeatureConvAE = baseline_module.FeatureConvAE  # type: ignore
ConvBNReLU = baseline_module.ConvBNReLU  # type: ignore
sobel_edges = baseline_module.sobel_edges  # type: ignore
hog_like = baseline_module.hog_like  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth toggled via ad hoc loop.
# - Inner training: train_with_patience uses delta for ES; no separate patience_es; feature loss handled but best tracking uses delta.
# - Width expansion: widen_all mutates in place; trials_width as failure counter; no snapshot/restore abstraction; delta shared for width/depth.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - 2D/ALT: width_to_depth/depth_to_width/alt_* just toggle on no improvement; missing structured outer/inner loops and phase saturation per spec.
# - Patiences: lacking distinct patience_width_exp/patience_depth_exp application per context; relies on improved flag.
# Deviations: Missing snapshot_arch_and_state/restore_arch_and_state, proper expansion patiences, and exact control flow for ADP_WIDTH_ONLY, ADP_DEPTH_ONLY, ADP_DEPTH_OUTER_WIDTH_INNER, ADP_WIDTH_OUTER_DEPTH_INNER, ADP_ALT_DEPTH, ADP_ALT_WIDTH.


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 12
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    feature_type: str = "edge"  # or "hog"
    hog_bins: int = 8
    loss_type: str = "mse"  # or "l1"


def _resize_tensor(target: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    tgt = target.clone()
    common = tuple(min(a, b) for a, b in zip(target.shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(new_state, old_state):
    merged = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v, ov)
        else:
            merged[k] = v
    return merged


def _build_model(in_ch: int, widths: List[int], pooling_indices: List[int], out_ch: int, device) -> FeatureConvAE:
    return FeatureConvAE(in_ch=in_ch, widths=widths, pooling_indices=pooling_indices, out_ch=out_ch).to(device)


def rebuild_model(model: FeatureConvAE, widths: List[int], device) -> FeatureConvAE:
    new_model = _build_model(model.in_ch, widths, list(model.pooling_indices), model.out_ch, device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def snapshot_arch_and_state(model: FeatureConvAE):
    return {
        "widths": list(model.widths),
        "state": copy.deepcopy(model.state_dict()),
        "pooling_indices": list(model.pooling_indices),
        "out_ch": model.out_ch,
    }


def restore_arch_and_state(model: FeatureConvAE, snapshot, device) -> FeatureConvAE:
    restored = _build_model(model.in_ch, snapshot["widths"], snapshot["pooling_indices"], snapshot.get("out_ch", model.out_ch), device)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def total_neurons(model: FeatureConvAE) -> int:
    return sum(b.conv.out_channels for b in model.encoder) + sum(b.conv.out_channels for b in model.decoder)


def expand_width(model: FeatureConvAE, ex_k_width: int, max_width: int, device) -> FeatureConvAE:
    new_widths = [min(max_width, w + ex_k_width) for w in model.widths]
    return rebuild_model(model, new_widths, device)


def expand_depth(model: FeatureConvAE, ex_k_depth: int, device) -> FeatureConvAE:
    if ex_k_depth <= 0:
        return model
    last_w = model.widths[-1]
    new_widths = list(model.widths) + [last_w for _ in range(ex_k_depth)]
    return rebuild_model(model, new_widths, device)


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def make_target(x: torch.Tensor, acfg: ADPConfig) -> torch.Tensor:
    if acfg.feature_type == "edge":
        return sobel_edges(x)
    elif acfg.feature_type == "hog":
        return hog_like(x, bins=acfg.hog_bins)
    else:
        raise ValueError("feature_type must be 'edge' or 'hog'")


def train_with_early_stopping(model: FeatureConvAE, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, dict, None]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.MSELoss() if acfg.loss_type == "mse" else nn.L1Loss()
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    remaining = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            tgt = make_target(x, acfg)
            y = model(x)
            loss = crit(y, tgt)
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
                tgt = make_target(x, acfg)
                y = model(x)
                l = crit(y, tgt)
                val += l.item() * x.size(0)
                n += x.size(0)
            val = val / max(n, 1)
        history.append(val)
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            remaining = acfg.patience
        else:
            remaining -= 1
        if remaining <= 0:
            break
    model.load_state_dict(best_state)
    return best, best_state, None


def adp_search(model: FeatureConvAE, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    from utils.adp_contract import run_module_adp
    from utils.adp_introspect import infer_adp_shape

    best_val, model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=locals().get("log_loss", False),
        log_neurons=locals().get("log_neurons", False),
        results_dir=locals().get("results_dir"),
        logger=locals().get("logger"),
    )

    return best_val, model, list(getattr(model, "widths", []))


# ADP REVIEW (AFTER REFACTOR)
# - Mode: width_only / width -> ADP_WIDTH_ONLY (depth fixed; ES with patience_es=patience; width_failure_count vs trials_width; accept val < best - delta_width).
# - Mode: depth_only / depth -> ADP_DEPTH_ONLY (widths fixed; depth_failure_count vs trials_depth; accept val < best - delta_depth).
# - Mode: depth_to_width -> ADP_DEPTH_OUTER_WIDTH_INNER (outer depth +1 with patience_depth_exp/delta_depth; inner width search with patience_width_exp/delta_width).
# - Mode: width_to_depth -> ADP_WIDTH_OUTER_DEPTH_INNER (outer width +ex_k with patience_width_exp/delta_width; inner depth search with patience_depth_exp/delta_depth).
# - Mode: alt_depth -> ADP_ALT_DEPTH (phase depth-only until depth patience hit, then width-only until width patience hit; repeat until both saturated).
# - Mode: alt_width -> ADP_ALT_WIDTH (start width phase then depth phase, same patience rules, repeat until both saturated).
# - Snapshot/restore + expand_width/expand_depth follow spec; patience mapping: patience->patience_es, trials_width->patience_width_exp, trials_depth->patience_depth_exp; delta used for both width/depth thresholds.


def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Feature Reconstruction AE (width/depth search)")
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--pool-idx", type=int, nargs="*", default=[0, 2])
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"],
    )
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feature-type", type=str, default="edge", choices=["edge", "hog"])
    p.add_argument("--hog-bins", type=int, default=8)
    p.add_argument("--loss-type", type=str, default="mse", choices=["mse", "l1"])
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_ch = args.hog_bins if args.feature_type == "hog" else 1
    model = FeatureConvAE(in_ch=3, widths=args.widths, pooling_indices=args.pool_idx, out_ch=out_ch).to(device)
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
        feature_type=args.feature_type,
        hog_bins=args.hog_bins,
        loss_type=args.loss_type,
    )
    results_dir = Path(f"results_{BASE_PATH.stem}")
    best_val, model, widths = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    print(f"[ADP Feature AE] mode={args.adp_mode} best_val={best_val:.6f} widths={widths} depth={len(widths)}")


if __name__ == "__main__":
    main()
