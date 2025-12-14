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

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_predictive.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
PredictiveSeqAE = baseline_module.PredictiveSeqAE  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth toggled via ad hoc loop.
# - Inner training: train_with_patience ties ES to delta; no separate patience_es; rollback per failure.
# - Expansions: widen_model/deepen_model mutate in place; rollback on fail; delta shared for width/depth.
# - 2D/ALT: toggle modes on no improvement; lacks forward-only expansion and context-end restore per updated spec.
# - Missing snapshot/restore abstractions and forward-only patience application.


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 32
    max_width: int = 512
    max_depth: int = 6
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    dropout: float = 0.1


def resize_gru(gru: nn.GRU, hidden_new: int, num_layers: int = None) -> nn.GRU:
    """Rebuild GRU with larger hidden size or depth, copying overlapping weights."""
    num_layers = gru.num_layers if num_layers is None else num_layers
    device = gru.weight_ih_l0.device
    g_new = nn.GRU(
        input_size=gru.input_size,
        hidden_size=hidden_new,
        num_layers=num_layers,
        batch_first=True,
        dropout=gru.dropout if num_layers > 1 else 0.0,
    ).to(device)
    min_h = min(hidden_new, gru.hidden_size)
    min_l = min(num_layers, gru.num_layers)
    for l in range(min_l):
        for suffix in ["ih", "hh"]:
            w_old = getattr(gru, f"weight_{suffix}_l{l}")
            w_new = getattr(g_new, f"weight_{suffix}_l{l}")
            with torch.no_grad():
                w_new[:min_h, : w_old.size(1)] = w_old[:min_h]
        for suffix in ["ih", "hh", "ih", "hh"]:
            b_old = getattr(gru, f"bias_{suffix}_l{l}")
            b_new = getattr(g_new, f"bias_{suffix}_l{l}")
            with torch.no_grad():
                b_new[:min_h] = b_old[:min_h]
    return g_new


def resize_out(out: nn.Linear, hidden_new: int) -> nn.Linear:
    new = nn.Linear(hidden_new, out.out_features).to(out.weight.device)
    with torch.no_grad():
        r = min(hidden_new, out.in_features)
        new.weight[:, :r] = out.weight[:, :r]
        new.bias = out.bias.clone()
    return new


def widen_model(model: PredictiveSeqAE, ex_k: int, max_width: int):
    new_h = min(max_width, model.hidden_size + ex_k)
    model.rnn = resize_gru(model.rnn, new_h, model.num_layers)
    model.out_proj = resize_out(model.out_proj, new_h)
    model.hidden_size = new_h


def deepen_model(model: PredictiveSeqAE):
    new_layers = model.num_layers + 1
    model.rnn = resize_gru(model.rnn, model.hidden_size, new_layers)
    model.num_layers = new_layers


def total_neurons(model: PredictiveSeqAE) -> int:
    return model.hidden_size * model.num_layers


def snapshot_arch_and_state(model: PredictiveSeqAE):
    return {
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "state": copy.deepcopy(model.state_dict()),
    }


def restore_arch_and_state(model: PredictiveSeqAE, snapshot, device) -> PredictiveSeqAE:
    restored = PredictiveSeqAE(
        feature_dim=model.feature_dim,
        hidden_size=snapshot["hidden_size"],
        num_layers=snapshot["num_layers"],
        dropout=model.dropout,
    ).to(device)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def make_seq_loaders(batch_size: int = 128, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, int]:
    """
    Turn CIFAR10 images into sequences by flattening rows: T=32, F=3*32=96.
    """
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    def collate(batch):
        xs = []
        for x, _ in batch:
            x = x.view(3, 32, 32).permute(1, 2, 0).reshape(32, 96)  # (T,F)
            xs.append(x)
        x_stack = torch.stack(xs, dim=0)
        return x_stack, torch.empty(x_stack.size(0))

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)
    return dl_train, dl_val, 96


def seq_loss(model: PredictiveSeqAE, x: torch.Tensor, crit):
    # x: (B,T,F). Predict x_{t+1}. Last timestep has no target; ignore it.
    y = model(x)
    tgt = x[:, 1:, :]
    pred = y[:, :-1, :]
    loss = crit(pred, tgt)
    return loss


def train_with_patience(model: PredictiveSeqAE, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.MSELoss()
    best = float("inf")
    best_state = None
    pat = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            loss = seq_loss(model, x, crit)
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
                l = seq_loss(model, x, crit)
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
    return best, best_state


# ADP REVIEW (AFTER REFACTOR)
# - Modes map to ADP_WIDTH_ONLY / ADP_DEPTH_ONLY / ADP_DEPTH_OUTER_WIDTH_INNER / ADP_WIDTH_OUTER_DEPTH_INNER / ADP_ALT_DEPTH / ADP_ALT_WIDTH with forward-only expansions (no per-step rollback; restore global best at context end).


def adp_search(model: PredictiveSeqAE, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history = []
    improvements = []

    def can_widen(hidden_size: int, num_layers: int):
        return (hidden_size + acfg.ex_k) <= acfg.max_width and (hidden_size + acfg.ex_k) * num_layers <= acfg.max_neurons

    def can_deepen(hidden_size: int, num_layers: int):
        return (num_layers + 1) <= acfg.max_depth and (num_layers + 1) * hidden_size <= acfg.max_neurons

    best_val, best_state = train_with_patience(model, dl_train, dl_val, acfg, device, val_history)
    best_hidden = model.hidden_size
    best_layers = model.num_layers
    improvements.append((total_neurons(model), best_val))
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode

    def width_search(local_model: PredictiveSeqAE, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_hidden = local_model.hidden_size
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history)
        width_failure_count = 0
        while width_failure_count < pw and can_widen(local_model.hidden_size, local_model.num_layers):
            widen_model(local_model, acfg.ex_k, acfg.max_width)
            val, state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history)
            if val < local_best_val - acfg.delta:
                local_best_val = val
                local_best_state = state
                local_best_hidden = local_model.hidden_size
                width_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
            else:
                width_failure_count += 1
        local_model = rebuild_model(local_model, local_best_hidden, local_model.num_layers, device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_hidden

    def depth_search(local_model: PredictiveSeqAE, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_layers = local_model.num_layers
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history)
        depth_failure_count = 0
        while depth_failure_count < pd and can_deepen(local_model.hidden_size, local_model.num_layers):
            deepen_model(local_model)
            val, state = train_with_patience(local_model, dl_train, dl_val, acfg, device, val_history)
            if val < local_best_val - acfg.delta:
                local_best_val = val
                local_best_state = state
                local_best_layers = local_model.num_layers
                depth_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
            else:
                depth_failure_count += 1
        local_model = rebuild_model(local_model, local_model.hidden_size, local_best_layers, device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_layers

    if mode in ("width_only", "width"):
        model, best_val, best_state, best_hidden = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_layers = model.num_layers
    elif mode in ("depth_only", "depth"):
        model, best_val, best_state, best_layers = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_hidden = model.hidden_size
    elif mode == "depth_to_width":
        model, best_val, best_state, best_hidden = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_layers = model.num_layers
        depth_failure_count = 0
        while depth_failure_count < pd and can_deepen(best_hidden, best_layers):
            model = expand_depth(model, 1, device) if False else deepen_model(model) or model  # maintain structure
            cand_model, cand_val, cand_state, cand_hidden = width_search(model, log_improvement=False)
            if cand_val < best_val - acfg.delta:
                best_val = cand_val; best_state = cand_state; best_hidden = cand_hidden; best_layers = model.num_layers; depth_failure_count = 0; model = cand_model; model.load_state_dict(best_state); improvements.append((total_neurons(model), best_val))
            else:
                depth_failure_count += 1
    elif mode == "width_to_depth":
        model, best_val, best_state, best_layers = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_hidden = model.hidden_size
        width_failure_count = 0
        while width_failure_count < pw and can_widen(best_hidden, best_layers):
            widen_model(model, acfg.ex_k, acfg.max_width)
            cand_model, cand_val, cand_state, cand_layers = depth_search(model, log_improvement=False)
            if cand_val < best_val - acfg.delta:
                best_val = cand_val; best_state = cand_state; best_hidden = model.hidden_size; best_layers = cand_layers; width_failure_count = 0; model = cand_model; model.load_state_dict(best_state); improvements.append((total_neurons(model), best_val))
            else:
                width_failure_count += 1
    elif mode == "alt_depth":
        depth_saturated = False; width_saturated = False; phase = "depth"
        while not (depth_saturated and width_saturated):
            if phase == "depth":
                model, phase_val, phase_state, phase_layers = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_layers = phase_layers; best_hidden = model.hidden_size; depth_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(model, best_hidden, best_layers, device); model.load_state_dict(best_state); phase = "width"
            else:
                model, phase_val, phase_state, phase_hidden = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_hidden = phase_hidden; width_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(model, best_hidden, best_layers, device); model.load_state_dict(best_state); phase = "depth"
    elif mode == "alt_width":
        depth_saturated = False; width_saturated = False; phase = "width"
        while not (depth_saturated and width_saturated):
            if phase == "width":
                model, phase_val, phase_state, phase_hidden = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_hidden = phase_hidden; width_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(model, best_hidden, best_layers, device); model.load_state_dict(best_state); phase = "depth"
            else:
                model, phase_val, phase_state, phase_layers = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val; best_state = phase_state; best_layers = phase_layers; depth_saturated = False; improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(model, best_hidden, best_layers, device); model.load_state_dict(best_state); phase = "width"
    else:
        raise ValueError(f"Unsupported ADP mode: {mode}")

    model = rebuild_model(model, best_hidden, best_layers, device)
    model.load_state_dict(best_state)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val, model, best_hidden, best_layers


def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Predictive Seq AE (width/depth search)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
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
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")
    args = p.parse_args()

    dl_train, dl_val, feat_dim = make_seq_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredictiveSeqAE(feature_dim=feat_dim, hidden_size=args.hidden, num_layers=args.num_layers, dropout=args.dropout).to(device)
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
        dropout=args.dropout,
    )
    results_dir = Path(f"results_{BASE_PATH.stem}")
    best_val, model, hidden, layers = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    print(f"[ADP Predictive SeqAE] mode={args.adp_mode} best_val={best_val:.6f} hidden={hidden} layers={layers}")


if __name__ == "__main__":
    main()
