import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_tcn_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_TCN_STL = baseline_module.AE_TCN_STL  # type: ignore
TCNBlock = baseline_module.TCNBlock  # type: ignore
ae_tcn_total_neurons = baseline_module.ae_tcn_total_neurons  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Supported modes in this core:
#     * width_only/width      -> Single while-loop widening with trials_width counter; no structured width-expansion patience.
#     * depth_only/depth      -> Single while-loop deepening with trials_depth counter; no dedicated depth-expansion patience.
#     * width_to_depth        -> Starts width search, then flips to depth when no improvement; width inner/outer separation missing.
#     * depth_to_width        -> Starts depth search, then flips to width when no improvement; lacks width-outer/depth-inner structure.
#     * alt_width / alt_depth -> Alternates single expansions between width and depth regardless of saturation definition.
# - Inner training: train_with_patience uses single delta/patience for ES; accepts improvements > delta only, resets patience; no phys metric.
# - Expansions: rebuild_model increments width by ex_k or depth by +1 but merges state shallowly; no explicit snapshot/restore helpers.
# - Acceptance criteria: single delta used for both width/depth; failure counters are trials_width/trials_depth applied ad hoc, not per context.
# - Deviations vs ADP_algorithms.md:
#     * Missing distinct patience_es / patience_width_exp / patience_depth_exp and delta_width / delta_depth handling.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
#     * 2D searches (depth_outer_width_inner, width_outer_depth_inner) and ALT phases do not follow specified outer/inner or phase saturation rules.
#     * Stopping conditions rely on a generic 'improved' flag rather than spec-defined patience counters per dimension/phase.


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
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 100_000_000
    seq_len: int = 32
    pool_every: int = 0


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


def _normalize_dilations(model: AE_TCN_STL, target_depth: int) -> List[int]:
    dils = list(getattr(model, "dilations", [1] * getattr(model, "depth", target_depth)))
    if not dils:
        dils = [1] * target_depth
    if len(dils) < target_depth:
        dils = dils + [dils[-1]] * (target_depth - len(dils))
    else:
        dils = dils[:target_depth]
    return dils


def _build_model(in_channels: int, width: int, depth: int, dilations: List[int], device, pool_every: int) -> AE_TCN_STL:
    kwargs = dict(in_channels=in_channels, width=width, depth=depth, dilations=dilations)
    try:
        new_model = AE_TCN_STL(**kwargs, pool_every=pool_every)  # type: ignore[arg-type]
    except TypeError:
        new_model = AE_TCN_STL(**kwargs)  # type: ignore[arg-type]
    return new_model.to(device)


def rebuild_model(model: AE_TCN_STL, width: int, depth: int, device, pool_every: int) -> AE_TCN_STL:
    dilations = _normalize_dilations(model, depth)
    new_model = _build_model(model.in_channels, width, depth, dilations, device, pool_every)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def snapshot_arch_and_state(model: AE_TCN_STL):
    return {
        "width": model.width,
        "depth": model.depth,
        "state": copy.deepcopy(model.state_dict()),
        "dilations": copy.deepcopy(getattr(model, "dilations", [1] * model.depth)),
    }


def restore_arch_and_state(model: AE_TCN_STL, snapshot, device, pool_every: int) -> AE_TCN_STL:
    width = snapshot["width"]
    depth = snapshot["depth"]
    dilations = snapshot.get("dilations", [1] * depth)
    dilations = list(dilations) if dilations else [1] * depth
    dilations = (dilations + [dilations[-1]] * (depth - len(dilations))) if len(dilations) < depth else dilations[:depth]
    restored = _build_model(model.in_channels, width, depth, dilations, device, pool_every)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def expand_width(model: AE_TCN_STL, ex_k_width: int, device, pool_every: int) -> AE_TCN_STL:
    new_w = model.width + ex_k_width
    return rebuild_model(model, new_w, model.depth, device, pool_every)


def expand_depth(model: AE_TCN_STL, ex_k_depth: int, device, pool_every: int) -> AE_TCN_STL:
    new_d = model.depth + ex_k_depth
    return rebuild_model(model, model.width, new_d, device, pool_every)


def total_neurons(width: int, depth: int) -> int:
    return int(ae_tcn_total_neurons(width, depth))


def train_with_early_stopping(model: AE_TCN_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_patience = acfg.patience
    remaining = es_patience
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
        
        # Log to console and text file
        msg = f"  Epoch {_+1}/{acfg.max_epochs} | Device: {device} | Val Loss: {val:.6f} | Best: {best:.6f} | Pat: {pat}/{acfg.patience}"
        if verbose and logger:
            logger.log_console(msg)
        elif verbose:
            # print(msg) # optional, keep silent if desired, but logger is preferred
            pass
        
        # Log to CSV immediately
        if logger:
            logger.log_epoch_stats({
                "epoch": len(history),
                "width": model.width,
                "depth": model.depth,
                "neurons": total_neurons(model),
                "val_loss": val,
                "best_val": best,
                "es_counter": acfg.patience - pat, # approx
                "improved": (val < best - acfg.delta)
            })
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            remaining = es_patience
        else:
            remaining -= 1
        if remaining <= 0:
            break
    model.load_state_dict(best_state)
    return best_val, best_state, None


# ADP REVIEW (AFTER REFACTOR)
# - Mode: width_only / width -> Implements ADP_WIDTH_ONLY (depth fixed, ES with patience_es=patience; width expansions ex_k with acceptance val < best - delta_width; width_failure_count vs trials_width).
# - Mode: depth_only / depth -> Implements ADP_DEPTH_ONLY (width fixed; depth expansions of +1 with acceptance val < best - delta_depth; depth_failure_count vs trials_depth).
# - Mode: depth_to_width -> Implements ADP_DEPTH_OUTER_WIDTH_INNER (outer depth steps gated by delta_depth/patience_depth_exp; inner width search with delta_width/patience_width_exp).
# - Mode: width_to_depth -> Implements ADP_WIDTH_OUTER_DEPTH_INNER (outer width steps gated by delta_width/patience_width_exp; inner depth search with delta_depth/patience_depth_exp).
# - Mode: alt_depth -> Implements ADP_ALT_DEPTH (phases: depth-only until depth_failure_count hits patience_depth_exp; then width-only until patience_width_exp; repeat until both saturated).
# - Mode: alt_width -> Implements ADP_ALT_WIDTH (phases start with width-only then depth-only, same patience logic; repeat until both saturated).
# - Snapshot/restore + expand_width/expand_depth follow ADP_algorithms.md; patience mappings: patience->patience_es, trials_width->patience_width_exp, trials_depth->patience_depth_exp; delta used for both width/depth thresholds.


def adp_search(model: AE_TCN_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_tcn_stl")):
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

    return best_val, model, *infer_adp_shape(model)


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
    p = argparse.ArgumentParser(description="ADP TCN AE width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool-every", type=int, default=0)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"])
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
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_tcn_stl"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AE_TCN_STL(in_channels=3, width=args.width, depth=args.depth, pool_every=args.pool_every).to(device)
    except TypeError:
        model = AE_TCN_STL(in_channels=3, width=args.width, depth=args.depth).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, pool_every=args.pool_every)
    best, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP TCN AE] mode={args.adp_mode} best_val={best:.6f} width={w} depth={d}")
