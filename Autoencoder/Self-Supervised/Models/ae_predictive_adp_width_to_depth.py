import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_predictive.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
PredictiveSeqAE = baseline_module.PredictiveSeqAE  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 32
    max_width: int = 512
    max_depth: int = 6
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 20
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


def train_with_patience(model: PredictiveSeqAE, dl_train, dl_val, acfg: ADPConfig, device):
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


def adp_search(model: PredictiveSeqAE, dl_train, dl_val, acfg: ADPConfig, device):
    def can_widen():
        return (model.hidden_size + acfg.ex_k) <= acfg.max_width and (model.hidden_size + acfg.ex_k) * model.num_layers <= acfg.max_neurons

    def can_deepen():
        return (model.num_layers + 1) <= acfg.max_depth and (model.num_layers + 1) * model.hidden_size <= acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device)
    best_val, best_state = inner_val, copy.deepcopy(model.state_dict())
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_h = model.hidden_size
                pre_val = inner_val
                widen_model(model, acfg.ex_k, acfg.max_width)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    # rollback
                    model.hidden_size = pre_h
                    model.rnn = resize_gru(model.rnn, pre_h, model.num_layers)
                    model.out_proj = resize_out(model.out_proj, pre_h)
                    model.load_state_dict(pre)
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_layers = model.num_layers
                pre_val = inner_val
                deepen_model(model)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.rnn = resize_gru(model.rnn, model.hidden_size, pre_layers)
                    model.num_layers = pre_layers
                    model.load_state_dict(pre)
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
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
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
    best = adp_search(model, dl_train, dl_val, acfg, device)
    print(f"[ADP Predictive SeqAE] mode={args.adp_mode} best_val={best:.6f} hidden={model.hidden_size} layers={model.num_layers}")


if __name__ == "__main__":
    main()
