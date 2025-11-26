import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_blindspot.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
BlindSpotConvAE = baseline_module.BlindSpotConvAE  # type: ignore
ConvBNReLU = baseline_module.ConvBNReLU  # type: ignore


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


def _resize_conv(conv: nn.Conv2d, in_ch: int, out_ch: int) -> nn.Conv2d:
    k_h, k_w = conv.kernel_size
    new = nn.Conv2d(in_ch, out_ch, kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, bias=conv.bias is not None, dilation=conv.dilation).to(conv.weight.device)
    with torch.no_grad():
        r = min(out_ch, conv.out_channels)
        c = min(in_ch, conv.in_channels)
        new.weight[:r, :c] = conv.weight[:r, :c]
        if conv.bias is not None and new.bias is not None:
            new.bias[:r] = conv.bias[:r]
    return new


def _resize_bn(bn: nn.BatchNorm2d, out_ch: int) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(out_ch).to(bn.weight.device)
    with torch.no_grad():
        r = min(out_ch, bn.weight.numel())
        new.weight[:r] = bn.weight[:r]
        new.bias[:r] = bn.bias[:r]
        new.running_mean[:r] = bn.running_mean[:r]
        new.running_var[:r] = bn.running_var[:r]
    return new


def _resize_block(block: ConvBNReLU, in_ch: int, out_ch: int) -> ConvBNReLU:
    block.conv = _resize_conv(block.conv, in_ch, out_ch)
    block.bn = _resize_bn(block.bn, out_ch)
    return block


def rebuild_decoder(model: BlindSpotConvAE, widths: List[int]):
    rev = list(reversed(widths))
    dec = []
    ch = rev[0]
    old_dec = list(model.decoder)
    for w in rev[1:]:
        blk = ConvBNReLU(ch, w)
        if old_dec:
            old_blk = old_dec.pop(0)
            blk = _resize_block(blk, ch, w)
        dec.append(blk)
        ch = w
    model.decoder = nn.ModuleList(dec)
    model.head = nn.Conv2d(ch, model.in_ch, kernel_size=1, stride=1, padding=0)


def widen_all(model: BlindSpotConvAE, ex_k: int, max_width: int):
    new_widths = []
    prev = model.in_ch
    for blk in model.encoder:
        out = min(max_width, blk.conv.out_channels + ex_k)
        _resize_block(blk, prev, out)
        new_widths.append(out)
        prev = out
    model.widths = new_widths
    rebuild_decoder(model, new_widths)


def append_depth(model: BlindSpotConvAE):
    last_w = model.encoder[-1].conv.out_channels
    new_blk = ConvBNReLU(last_w, last_w)
    model.encoder.append(new_blk)
    model.widths.append(last_w)
    rebuild_decoder(model, model.widths)


def total_neurons(model: BlindSpotConvAE) -> int:
    return sum(b.conv.out_channels for b in model.encoder) + sum(b.conv.out_channels for b in model.decoder)


def train_with_patience(model: BlindSpotConvAE, dl_train, dl_val, acfg: ADPConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf"); best_state=None; pat=acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            mask = torch.rand_like(x[:, :1]) < 0.1
            x_in = x * (1 - mask)
            opt.zero_grad(set_to_none=True)
            y = model(x_in)
            loss = ((y - x) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = 0.0; n = 0
            for x, _ in dl_val:
                x = x.to(device)
                mask = torch.rand_like(x[:, :1]) < 0.1
                x_in = x * (1 - mask)
                y = model(x_in)
                l = ((y - x) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                val += l.item(); n += 1
            val = val / max(n,1)
        if val < best - acfg.delta:
            best = val; best_state = copy.deepcopy(model.state_dict()); pat = acfg.patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: BlindSpotConvAE, dl_train, dl_val, acfg: ADPConfig, device):
    def can_widen():
        return max(model.widths) + acfg.ex_k <= acfg.max_width and total_neurons(model) < acfg.max_neurons
    def can_deepen():
        return len(model.widths) + 1 <= acfg.max_depth and (total_neurons(model) + model.widths[-1]) <= acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device)
    best_val = inner_val; best_state = copy.deepcopy(model.state_dict())
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only","width","width_to_depth","alt_width"):
            if can_widen() and pw>0:
                pre = copy.deepcopy(model.state_dict()); pre_val = inner_val
                widen_all(model, acfg.ex_k, acfg.max_width)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v; pw = acfg.trials_width; improved=True
                    if v < best_val: best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre); pw -= 1
            if mode == "width_only": continue
        if mode in ("depth_only","depth","depth_to_width","alt_depth"):
            if can_deepen() and pd>0:
                pre = copy.deepcopy(model.state_dict()); pre_val = inner_val
                append_depth(model)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v; pd = acfg.trials_depth; improved=True
                    if v < best_val: best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre); pd -= 1
            if mode == "depth_only": continue
        if mode == "width_to_depth" and not improved:
            mode = "depth"; pd = acfg.trials_depth; improved=True
        elif mode == "depth_to_width" and not improved:
            mode = "width"; pw = acfg.trials_width; improved=True
        elif mode in ("alt_width","alt_depth"):
            mode = "depth" if mode=="alt_width" else "width"; improved=True
    model.load_state_dict(best_state)
    return best_val


def make_loaders(batch_size: int = 64, val_split: float = 0.1):
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
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    p = argparse.ArgumentParser(description="ADP Blindspot AE (width/depth search)")
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--pool-idx", type=int, nargs="*", default=[1])
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
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * 0.1)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlindSpotConvAE(in_ch=3, widths=args.widths, pooling_indices=args.pool_idx).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs)
    best = adp_search(model, dl_train, dl_val, acfg, device)
    print(f"[ADP Blindspot AE] mode={args.adp_mode} best_val={best:.6f} widths={model.widths} depth={len(model.widths)}")


if __name__ == "__main__":
    main()
