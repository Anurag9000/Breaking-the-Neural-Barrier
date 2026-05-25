import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_seq.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
SeqAE = baseline_module.SeqAE  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth via ad hoc loop.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - 2D/ALT: toggle modes on no improvement; lacks forward-only expansion and context-end restore per updated spec.
# - Missing snapshot/restore abstractions and forward-only patience application.


class ImageRowsAsSequence(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        c, h, w = x.shape
        seq = x.permute(1, 0, 2).contiguous().view(h, c * w)
        return seq, y


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
    bidirectional: bool = False


def resize_gru(gru: nn.GRU, hidden_new: int, num_layers: int = None, bidirectional: bool = None) -> nn.GRU:
    num_layers = gru.num_layers if num_layers is None else num_layers
    bidirectional = gru.bidirectional if bidirectional is None else bidirectional
    device = gru.weight_ih_l0.device
    g_new = nn.GRU(
        input_size=gru.input_size,
        hidden_size=hidden_new,
        num_layers=num_layers,
        batch_first=True,
        dropout=gru.dropout if num_layers > 1 else 0.0,
        bidirectional=bidirectional,
    ).to(device)
    min_h = min(hidden_new, gru.hidden_size)
    min_l = min(num_layers, gru.num_layers)
    dir_mult_old = 2 if gru.bidirectional else 1
    dir_mult_new = 2 if bidirectional else 1
    for l in range(min_l * min(dir_mult_old, dir_mult_new)):
        for suffix in ["ih", "hh"]:
            w_old = getattr(gru, f"weight_{suffix}_l{l}")
            w_new = getattr(g_new, f"weight_{suffix}_l{l}")
            with torch.no_grad():
                r_out = min(w_new.size(0), w_old.size(0))
                r_in = min(w_new.size(1), w_old.size(1))
                w_new[:r_out, :r_in] = w_old[:r_out, :r_in]
        for suffix in ["ih", "hh"]:
            b_old = getattr(gru, f"bias_{suffix}_l{l}")
            b_new = getattr(g_new, f"bias_{suffix}_l{l}")
            with torch.no_grad():
                r = min(b_new.size(0), b_old.size(0))
                b_new[:r] = b_old[:r]
    return g_new


def resize_linear(out: nn.Linear, in_new: int, out_new: int) -> nn.Linear:
    new = nn.Linear(in_new, out_new, bias=out.bias is not None).to(out.weight.device)
    with torch.no_grad():
        r_out = min(out_new, out.out_features)
        r_in = min(in_new, out.in_features)
        new.weight[:r_out, :r_in] = out.weight[:r_out, :r_in]
        if out.bias is not None and new.bias is not None:
            new.bias[:r_out] = out.bias[:r_out]
    return new


def widen_model(model: SeqAE, ex_k: int, max_width: int):
    new_h = min(max_width, model.hidden_size + ex_k)
    if new_h == model.hidden_size:
        return
    model.encoder = resize_gru(model.encoder, new_h, model.num_layers, model.bidirectional)
    model.decoder = resize_gru(model.decoder, new_h, model.num_layers, False)
    # adjust latent projection/out_proj sizes
    model.to_latent = resize_linear(model.to_latent, new_h * model.dir_mult * model.num_layers, new_h)
    model.out_proj = resize_linear(model.out_proj, new_h, model.feature_dim)
    model.hidden_size = new_h


def deepen_model(model: SeqAE):
    new_layers = model.num_layers + 1
    model.encoder = resize_gru(model.encoder, model.hidden_size, new_layers, model.bidirectional)
    model.decoder = resize_gru(model.decoder, model.hidden_size, new_layers, False)
    model.to_latent = resize_linear(model.to_latent, model.hidden_size * model.dir_mult * new_layers, model.hidden_size)
    model.num_layers = new_layers


def total_neurons(model: SeqAE) -> int:
    return model.total_neurons()


def snapshot_arch_and_state(model: SeqAE):
    return {
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "bidirectional": model.bidirectional,
        "state": copy.deepcopy(model.state_dict()),
    }


def restore_arch_and_state(model: SeqAE, snapshot, device) -> SeqAE:
    restored = SeqAE(
        feature_dim=model.feature_dim,
        hidden_size=snapshot["hidden_size"],
        num_layers=snapshot["num_layers"],
        dropout=model.dropout,
        bidirectional=snapshot.get("bidirectional", model.bidirectional),
    ).to(device)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def make_seq_loaders(batch_size: int = 128, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, int]:
    """
    Turn real images into sequences by flattening rows: T=32, F=3*32=96.
    """
    sys.path.append(str(Path(__file__).resolve().parents[1] / "Runs"))
    from _common_real_image import make_real_image_loaders
    base_train, base_val, _ = make_real_image_loaders("./data", batch_size=batch_size, val_ratio=val_split, num_workers=0, image_size=32)
    train_set = ImageRowsAsSequence(base_train.dataset)
    val_set = ImageRowsAsSequence(base_val.dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, 3 * 32


def recon_loss(model: SeqAE, x: torch.Tensor, crit):
    # x: (B,T,F)
    y = model(x)
    return crit(y, x)


def train_with_patience(model: SeqAE, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.MSELoss()
    best = float("inf")
    best_state = None
    pat = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            loss = recon_loss(model, x, crit)
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
                l = recon_loss(model, x, crit)
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


def adp_search(model: SeqAE, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
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


def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Sequence AE (GRU) width/depth search")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--bidirectional", action="store_true", default=False)
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "width_to_depth"],
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
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_seq"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val, feat_dim = make_seq_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqAE(feature_dim=feat_dim, hidden_size=args.hidden, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
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
        bidirectional=args.bidirectional,
    )
    best_val, model, hidden, layers = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP SeqAE] mode={args.adp_mode} best_val={best_val:.6f} hidden={hidden} layers={layers}")
