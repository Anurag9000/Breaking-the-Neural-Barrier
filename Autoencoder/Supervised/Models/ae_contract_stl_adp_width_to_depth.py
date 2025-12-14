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

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_contract_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_CONTRACT_STL = baseline_module.AE_CONTRACT_STL  # type: ignore
ConvBlock = baseline_module.ConvBlock  # type: ignore
DeconvBlock = baseline_module.DeconvBlock  # type: ignore
contractive_penalty_hutchinson = baseline_module.contractive_penalty_hutchinson  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth via ad hoc loop with per-expansion rollback.
# - Inner training: train_with_patience ties ES to delta; no separate patience_es.
# - Expansions: widen/deepen with rollback on failure; single delta for width/depth.
# - 2D/ALT: toggles modes on no improvement; lacks forward-only expansion and context-end restore per updated spec.
# - Missing snapshot/restore abstractions and forward-only patience handling.


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


def train_with_patience(model: AE_CONTRACT_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list, logger: Optional[ContinuousLogger] = None, verbose: bool = True):
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
    return best, best_state


def adp_search(model: AE_CONTRACT_STL, dl_train, dl_val, acfg: ADPConfig, device, logger: ContinuousLogger, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path(\"results_adp\")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history = []
    improvements: List[tuple] = []

    def can_widen(width: int, depth: int):
        return (width + acfg.ex_k) <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen(width: int, depth: int):
        return (depth + 1) <= acfg.max_depth and (total_neurons(model) + width) <= acfg.max_neurons

    logger.log_console('[INITIAL TRAINING]')
    best_val, best_state = train_with_patience(model, dl_train, dl_val, acfg, device, val_history, logger=logger)
    best_width = model.width
    best_depth = model.depth
    improvements.append((total_neurons(model), best_val))
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode

    def width_search(local_model: AE_CONTRACT_STL, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_width = local_model.width
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        width_failure_count = 0
        while width_failure_count < pw and can_widen(local_model.width, local_model.depth):
            widened = widen_model(local_model, acfg.ex_k, acfg.max_width)
            if widened is not None:
                local_model = widened.to(device)
            val, state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if val < local_best_val - acfg.delta:
                local_best_val = val
                local_best_state = state
                local_best_width = local_model.width
                width_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
                    logger.log_console(f"[WIDTH OPT] ✓ IMPROVEMENT: New best: {val:.6f}")
                    if log_loss: plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
                    if log_neurons: plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
            else:
                width_failure_count += 1
                logger.log_console(f'[WIDTH OPT] ✗ No improvement')
        local_model = rebuild_model(local_best_width, local_model.depth, list(local_model.pool_after)).to(device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_width

    def depth_search(local_model: AE_CONTRACT_STL, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_depth = local_model.depth
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        depth_failure_count = 0
        while depth_failure_count < pd and can_deepen(local_model.width, local_model.depth):
            local_model = deepen_model(local_model).to(device)
            val, state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if val < local_best_val - acfg.delta:
                local_best_val = val
                local_best_state = state
                local_best_depth = local_model.depth
                depth_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
                    logger.log_console(f"[WIDTH OPT] ✓ IMPROVEMENT: New best: {val:.6f}")
                    if log_loss: plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
                    if log_neurons: plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
            else:
                depth_failure_count += 1
                logger.log_console(f'[DEPTH OPT] ✗ No improvement')
        local_model = rebuild_model(local_model.width, local_best_depth, list(local_model.pool_after)).to(device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_depth

    if mode in ("width_only", "width"):
        model, best_val, best_state, best_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_depth = model.depth
    elif mode in ("depth_only", "depth"):
        model, best_val, best_state, best_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_width = model.width
    elif mode == "depth_to_width":
        model, best_val, best_state, best_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_depth = model.depth
        depth_failure_count = 0
        while depth_failure_count < pd and can_deepen(best_width, best_depth):
            model = deepen_model(model).to(device)
            cand_model, cand_val, cand_state, cand_width = width_search(model, log_improvement=False)
            if cand_val < best_val - acfg.delta:
                best_val = cand_val; best_state = cand_state; best_depth = model.depth; best_width = cand_width; depth_failure_count = 0; model = cand_model; model.load_state_dict(best_state); improvements.append((total_neurons(model), best_val))
            else:
                depth_failure_count += 1
                logger.log_console(f'[DEPTH OPT] ✗ No improvement')
    elif mode == "width_to_depth":
        model, best_val, best_state, best_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_width = model.width
        width_failure_count = 0
        while width_failure_count < pw and can_widen(best_width, best_depth):
            widened = widen_model(model, acfg.ex_k, acfg.max_width)
            if widened is not None:
                model = widened.to(device)
            cand_model, cand_val, cand_state, cand_depth = depth_search(model, log_improvement=False)
            if cand_val < best_val - acfg.delta:
                best_val = cand_val; best_state = cand_state; best_width = model.width; best_depth = cand_depth; width_failure_count = 0; model = cand_model; model.load_state_dict(best_state); improvements.append((total_neurons(model), best_val))
            else:
                width_failure_count += 1
                logger.log_console(f'[WIDTH OPT] ✗ No improvement')
    elif mode == "alt_depth":
        depth_saturated = False; width_saturated = False; phase = "depth"; best_width = model.width; best_depth = model.depth
        while not (depth_saturated and width_saturated):
            if phase == "depth":
                model, phase_val, phase_state, phase_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_depth = phase_depth; best_width = model.width; depth_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(best_width, best_depth, list(model.pool_after)).to(device); model.load_state_dict(best_state); phase = "width"
            else:
                model, phase_val, phase_state, phase_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_width = phase_width; width_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(best_width, best_depth, list(model.pool_after)).to(device); model.load_state_dict(best_state); phase = "depth"
    elif mode == "alt_width":
        depth_saturated = False; width_saturated = False; phase = "width"; best_width = model.width; best_depth = model.depth
        while not (depth_saturated and width_saturated):
            if phase == "width":
                model, phase_val, phase_state, phase_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_width = phase_width; width_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(best_width, best_depth, list(model.pool_after)).to(device); model.load_state_dict(best_state); phase = "depth"
            else:
                model, phase_val, phase_state, phase_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_depth = phase_depth; depth_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(best_width, best_depth, list(model.pool_after)).to(device); model.load_state_dict(best_state); phase = "width"
    else:
        raise ValueError(f"Unsupported ADP mode: {mode}")

    model = rebuild_model(best_width, best_depth, list(model.pool_after)).to(device)
    model.load_state_dict(best_state)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val, model, best_width, best_depth


# ADP REVIEW (AFTER REFACTOR)
# - Modes implement forward-only ADP spec (no per-expansion rollback; restore global best at context end) across width_only/depth_only, width_to_depth, depth_to_width, alt_width, alt_depth.


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
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")
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
    results_dir = Path(f"results_{BASE_PATH.stem}")
    # Initialize Logger
    logger = ContinuousLogger(results_dir, "ae_contract_stl", args.adp_mode)
    
    best_val, model, width, depth = adp_search(model, dl_train, dl_val, acfg, device, logger=logger, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    logger.log_console(f"Done. Best val={best_val} w={width} d={depth}")
    logger.close()
    print(f"[ADP Contractive AE STL] mode={args.adp_mode} best_val={best_val:.6f} width={width} depth={depth}")


if __name__ == "__main__":
    main()
