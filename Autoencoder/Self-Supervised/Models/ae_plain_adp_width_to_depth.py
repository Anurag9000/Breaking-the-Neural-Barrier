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
BASE_PATH = Path(__file__).with_name("ae_plain.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
PlainConvAE = baseline_module.PlainConvAE  # type: ignore
ConvBNReLU = baseline_module.ConvBNReLU  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth handled via ad hoc while-loop that switches modes.
# - Inner training: train_with_patience uses delta as ES threshold and single patience; no dedicated patience_es; no phys metric.
# - Width expansion: widen_all mutates layers in place; acceptance uses delta, trials_width as counter; no snapshot/restore abstraction; no explicit delta_width vs delta_depth.
# - Depth expansion: append_depth mutates encoder; acceptance uses delta, trials_depth as counter; rollback via state_dict only, not architecture snapshot.
# - 2D/ALT algorithms: width_to_depth/depth_to_width and alt_* simply toggle modes on no improvement; lack outer/inner or phase-based patience per ADP spec.
# - Stopping: relies on "improved" flag rather than patience_width_exp/patience_depth_exp in proper contexts; total_neurons checks limited.
# Deviations: Missing snapshot_arch_and_state/restore_arch_and_state, structured expansion patiences, and exact control flow for ADP_WIDTH_ONLY, ADP_DEPTH_ONLY, ADP_DEPTH_OUTER_WIDTH_INNER, ADP_WIDTH_OUTER_DEPTH_INNER, ADP_ALT_DEPTH, ADP_ALT_WIDTH.


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
    grad_clip: float = 1.0
    max_epochs: int = 20


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


def _build_model(in_ch: int, widths: List[int], pooling_indices: List[int], device) -> PlainConvAE:
    return PlainConvAE(in_ch=in_ch, widths=widths, pooling_indices=pooling_indices).to(device)


def rebuild_model(model: PlainConvAE, widths: List[int], device) -> PlainConvAE:
    new_model = _build_model(model.in_ch, widths, list(model.pooling_indices), device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def snapshot_arch_and_state(model: PlainConvAE):
    return {
        "widths": list(model.widths),
        "state": copy.deepcopy(model.state_dict()),
        "pooling_indices": list(model.pooling_indices),
    }


def restore_arch_and_state(model: PlainConvAE, snapshot, device) -> PlainConvAE:
    restored = _build_model(model.in_ch, snapshot["widths"], snapshot["pooling_indices"], device)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def total_neurons(model: PlainConvAE) -> int:
    return model.total_neurons()


def expand_width(model: PlainConvAE, ex_k_width: int, max_width: int, device) -> PlainConvAE:
    new_widths = [min(max_width, w + ex_k_width) for w in model.widths]
    return rebuild_model(model, new_widths, device)


def expand_depth(model: PlainConvAE, ex_k_depth: int, device) -> PlainConvAE:
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


def train_with_early_stopping(model: PlainConvAE, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, dict, None]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.MSELoss()
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    remaining = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            y = model(x)
            loss = crit(y, x)
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
                y = model(x)
                l = crit(y, x)
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


def adp_search(model: PlainConvAE, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    delta_width = acfg.delta
    delta_depth = acfg.delta
    patience_width_exp = acfg.trials_width
    patience_depth_exp = acfg.trials_depth
    ex_k_width = acfg.ex_k
    ex_k_depth = 1

    def can_widen(widths: List[int], base_model: PlainConvAE) -> bool:
        proposed = [min(acfg.max_width, w + ex_k_width) for w in widths]
        if max(proposed) > acfg.max_width:
            return False
        new_model = rebuild_model(base_model, proposed, device)
        return total_neurons(new_model) <= acfg.max_neurons

    def can_deepen(widths: List[int], base_model: PlainConvAE) -> bool:
        new_depth = len(widths) + ex_k_depth
        if new_depth > acfg.max_depth:
            return False
        proposed = list(widths) + [widths[-1] for _ in range(ex_k_depth)]
        new_model = rebuild_model(base_model, proposed, device)
        return total_neurons(new_model) <= acfg.max_neurons

    # Initial training
    best_val, best_state, _ = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history)
    best_widths = list(model.widths)
    model.load_state_dict(best_state)
    improvements.append((total_neurons(model), best_val))

    def width_search(local_model: PlainConvAE, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_widths = list(local_model.widths)
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
        local_model.load_state_dict(local_best_state)
        width_failure_count = 0
        while width_failure_count < patience_width_exp and can_widen(local_model.widths, local_model):
            snap = snapshot_arch_and_state(local_model)
            local_model = expand_width(local_model, ex_k_width, acfg.max_width, device)
            val, state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
            if val < local_best_val - delta_width:
                local_best_val = val
                local_best_state = state
                local_best_widths = list(local_model.widths)
                width_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
                local_model.load_state_dict(local_best_state)
            else:
                width_failure_count += 1
                local_model = restore_arch_and_state(local_model, snap, device)
                local_model.load_state_dict(local_best_state)
        local_model = rebuild_model(local_model, local_best_widths, device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_widths

    def depth_search(local_model: PlainConvAE, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_widths = list(local_model.widths)
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
        local_model.load_state_dict(local_best_state)
        depth_failure_count = 0
        while depth_failure_count < patience_depth_exp and can_deepen(local_model.widths, local_model):
            snap = snapshot_arch_and_state(local_model)
            local_model = expand_depth(local_model, ex_k_depth, device)
            val, state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
            if val < local_best_val - delta_depth:
                local_best_val = val
                local_best_state = state
                local_best_widths = list(local_model.widths)
                depth_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
                local_model.load_state_dict(local_best_state)
            else:
                depth_failure_count += 1
                local_model = restore_arch_and_state(local_model, snap, device)
                local_model.load_state_dict(local_best_state)
        local_model = rebuild_model(local_model, local_best_widths, device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_widths

    mode = acfg.adp_mode
    if mode in ("width_only", "width"):
        model, best_val, best_state, best_widths = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
    elif mode in ("depth_only", "depth"):
        model, best_val, best_state, best_widths = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
    elif mode == "depth_to_width":  # ADP_DEPTH_OUTER_WIDTH_INNER
        model, best_val, best_state, best_widths = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        depth_failure_count = 0
        while depth_failure_count < patience_depth_exp and can_deepen(best_widths, model):
            saved_snap = snapshot_arch_and_state(model)
            model = expand_depth(model, ex_k_depth, device)
            cand_model, cand_val, cand_state, cand_widths = width_search(model, log_improvement=False)
            if cand_val < best_val - delta_depth:
                best_val = cand_val
                best_state = cand_state
                best_widths = cand_widths
                depth_failure_count = 0
                model = cand_model
                model.load_state_dict(best_state)
                improvements.append((total_neurons(model), best_val))
            else:
                depth_failure_count += 1
                model = restore_arch_and_state(model, saved_snap, device)
                model.load_state_dict(best_state)
    elif mode == "width_to_depth":  # ADP_WIDTH_OUTER_DEPTH_INNER
        model, best_val, best_state, best_widths = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        width_failure_count = 0
        while width_failure_count < patience_width_exp and can_widen(best_widths, model):
            saved_snap = snapshot_arch_and_state(model)
            model = expand_width(model, ex_k_width, acfg.max_width, device)
            cand_model, cand_val, cand_state, cand_widths = depth_search(model, log_improvement=False)
            if cand_val < best_val - delta_width:
                best_val = cand_val
                best_state = cand_state
                best_widths = cand_widths
                width_failure_count = 0
                model = cand_model
                model.load_state_dict(best_state)
                improvements.append((total_neurons(model), best_val))
            else:
                width_failure_count += 1
                model = restore_arch_and_state(model, saved_snap, device)
                model.load_state_dict(best_state)
    elif mode == "alt_depth":
        depth_saturated = False
        width_saturated = False
        phase = "depth"
        while not (depth_saturated and width_saturated):
            if phase == "depth":
                model, phase_val, phase_state, phase_widths = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_widths = phase_widths
                    depth_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(model, best_widths, device)
                model.load_state_dict(best_state)
                phase = "width"
            else:
                model, phase_val, phase_state, phase_widths = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_widths = phase_widths
                    width_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(model, best_widths, device)
                model.load_state_dict(best_state)
                phase = "depth"
    elif mode == "alt_width":
        depth_saturated = False
        width_saturated = False
        phase = "width"
        while not (depth_saturated and width_saturated):
            if phase == "width":
                model, phase_val, phase_state, phase_widths = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_widths = phase_widths
                    width_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(model, best_widths, device)
                model.load_state_dict(best_state)
                phase = "depth"
            else:
                model, phase_val, phase_state, phase_widths = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_widths = phase_widths
                    depth_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(model, best_widths, device)
                model.load_state_dict(best_state)
                phase = "width"
    else:
        raise ValueError(f"Unsupported ADP mode: {mode}")

    model = rebuild_model(model, best_widths, device)
    model.load_state_dict(best_state)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val, model, best_widths


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

    p = argparse.ArgumentParser(description="ADP Plain AE (width/depth search)")
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--pool-idx", type=int, nargs="*", default=[0, 2])
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
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlainConvAE(in_ch=3, widths=args.widths, pooling_indices=args.pool_idx).to(device)
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
    )
    results_dir = Path(f"results_{BASE_PATH.stem}")
    best_val, model, widths = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    print(f"[ADP Plain AE] mode={args.adp_mode} best_val={best_val:.6f} widths={widths} depth={len(widths)}")


if __name__ == "__main__":
    main()
