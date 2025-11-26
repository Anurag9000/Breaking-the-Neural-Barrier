import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load baseline
BASELINE_PATH = Path(__file__).with_name("mlp_ae_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASELINE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
MLPAutoencoder = baseline_module.MLPAutoencoder  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 128
    max_width: int = 4096
    max_depth: int = 10
    max_neurons: int = 10_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 128
    val_split: float = 0.1
    max_epochs: int = 50


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _rebuild_mlp_ae(model: MLPAutoencoder, hidden_widths: List[int]):
    """Rebuild encoder/decoder with given hidden widths, transplanting weights where possible."""
    device = next(model.parameters()).device
    in_dim = model.in_dim
    bottleneck = model.bottleneck
    use_bn = model.use_bn
    act = model.output_activation

    # encoder
    enc_layers = []
    prev = in_dim
    old_enc = list(model.enc)
    for w in hidden_widths:
        block = baseline_module.MLPBlock(prev, w, use_bn).to(device)  # type: ignore
        # overlap copy
        if old_enc:
            old_block = old_enc.pop(0)
            _resize_linear(old_block.linear, w, prev)
            block.linear = _resize_linear(old_block.linear, w, prev)
        enc_layers.append(block)
        prev = w
    model.enc = nn.Sequential(*enc_layers)
    model.hidden_widths = hidden_widths

    # bottleneck
    model.fc_mu = _resize_linear(model.fc_mu, bottleneck, prev)

    # decoder
    dec_layers = []
    prev_dec = bottleneck
    old_dec = list(model.dec)
    for w in reversed(hidden_widths):
        block = baseline_module.MLPBlock(prev_dec, w, use_bn).to(device)  # type: ignore
        if old_dec:
            old_block = old_dec.pop(0)
            block.linear = _resize_linear(old_block.linear, w, prev_dec)
        dec_layers.append(block)
        prev_dec = w
    model.dec = nn.Sequential(*dec_layers)
    model.out = _resize_linear(model.out, model.out.out_features, prev_dec)


def widen_all(model: MLPAutoencoder, ex_k: int, max_width: int):
    new_h = [min(max_width, w + ex_k) for w in model.hidden_widths]
    _rebuild_mlp_ae(model, new_h)


def append_depth(model: MLPAutoencoder):
    # append a layer with same width as last hidden to both encoder and decoder mirror
    new_h = model.hidden_widths + [model.hidden_widths[-1]]
    _rebuild_mlp_ae(model, new_h)


def total_neurons(model: MLPAutoencoder) -> int:
    enc = sum(model.hidden_widths)
    dec = sum(model.hidden_widths)
    return enc + dec + model.bottleneck


def make_loaders(batch_size: int, val_split: float):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def train_epoch(model: MLPAutoencoder, dl, opt, device):
    model.train()
    total, n = 0.0, 0
    for x, _ in dl:
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        xr = model(x)
        loss = F.mse_loss(xr, x)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def val_epoch(model: MLPAutoencoder, dl, device):
    model.eval()
    total, n = 0.0, 0
    for x, _ in dl:
        x = x.to(device)
        xr = model(x)
        loss = F.mse_loss(xr, x)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_with_patience(model: MLPAutoencoder, dl_train, dl_val, acfg: ADPConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf")
    best_state = None
    pat = acfg.patience
    for _ in range(acfg.max_epochs):
        train_epoch(model, dl_train, opt, device)
        val = val_epoch(model, dl_val, device)
        if val < best - 1e-9:
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


def adp_search(model: MLPAutoencoder, dl_train, dl_val, acfg: ADPConfig, device):
    def can_widen():
        return max(model.hidden_widths) + acfg.ex_k <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen():
        return len(model.hidden_widths) + 1 <= acfg.max_depth and (total_neurons(model) + model.hidden_widths[-1]) <= acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device)
    best_val = inner_val
    best_state = copy.deepcopy(model.state_dict())

    pw = acfg.trials_width
    pd = acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                widen_all(model, acfg.ex_k, acfg.max_width)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    if v < best_val:
                        best_val = v
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre)
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                append_depth(model)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
                    if v < best_val:
                        best_val = v
                        best_state = copy.deepcopy(model.state_dict())
                else:
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
    p = argparse.ArgumentParser(description="ADP MLP Autoencoder (width/depth search)")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=128)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    in_dim = 3 * 32 * 32
    model = MLPAutoencoder(in_dim, hidden_widths=args.hidden, bottleneck=args.bottleneck)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
    best = adp_search(model.to(device), dl_train, dl_val, acfg, device)
    print(f\"[ADP MLP AE] mode={args.adp_mode} best_val={best:.6f} hidden={model.hidden_widths} depth={len(model.hidden_widths)+1}\")


if __name__ == "__main__":
    main()
