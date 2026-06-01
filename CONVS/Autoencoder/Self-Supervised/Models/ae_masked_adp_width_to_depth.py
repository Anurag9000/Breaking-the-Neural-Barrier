import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_masked.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
MaskedConvAE = baseline_module.MaskedConvAE  # type: ignore
ConvBNReLU = baseline_module.ConvBNReLU  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Inner training: train_with_patience ties ES to delta; no separate patience_es; no phys metric.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - 2D/ALT loops: toggle modes on no improvement; lack forward-only expansion and context-end restore per updated spec.
# - Missing snapshot/restore abstractions and distinct width/depth failure handling aligned to new forward-only rule.


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
    mask_ratio: float = 0.6
    patch_size: int = 4


def _resize_conv(conv: nn.Conv2d, in_ch: int, out_ch: int) -> nn.Conv2d:
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


def rebuild_decoder(model: MaskedConvAE, widths: List[int]):
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


def widen_all(model: MaskedConvAE, ex_k: int, max_width: int):
    new_widths = []
    prev = model.in_ch
    for blk in model.encoder:
        out = min(max_width, blk.conv.out_channels + ex_k)
        _resize_block(blk, prev, out)
        new_widths.append(out)
        prev = out
    model.widths = new_widths
    rebuild_decoder(model, new_widths)


def append_depth(model: MaskedConvAE):
    last_w = model.encoder[-1].conv.out_channels
    model.encoder.append(ConvBNReLU(last_w, last_w))
    model.widths.append(last_w)
    rebuild_decoder(model, model.widths)


def total_neurons(model: MaskedConvAE) -> int:
    return sum(b.conv.out_channels for b in model.encoder) + sum(b.conv.out_channels for b in model.decoder)


def snapshot_arch_and_state(model: MaskedConvAE):
    return {
        "widths": list(model.widths),
        "state": copy.deepcopy(model.state_dict()),
        "pooling_indices": list(model.pooling_indices),
    }


def restore_arch_and_state(model: MaskedConvAE, snapshot, device) -> MaskedConvAE:
    restored = MaskedConvAE(in_ch=model.in_ch, widths=snapshot["widths"], pooling_indices=snapshot["pooling_indices"]).to(device)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    tf = transforms.Compose([transforms.ToTensor()])
    sys.path.append(str(Path(__file__).resolve().parents[1] / "Runs"))
    from _common_real_image import make_real_image_loaders
    dl_train, dl_val, _ = make_real_image_loaders("./data", batch_size=8, image_size=32, num_workers=0)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def mask_patches(x: torch.Tensor, ratio: float, patch_size: int):
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    gh, gw = H // patch_size, W // patch_size
    patch_mask = (torch.rand(B, 1, gh, gw, device=x.device) < ratio).float()
    mask = F.interpolate(patch_mask, size=(H, W), mode="nearest")
    return mask


def train_with_patience(model: MaskedConvAE, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf"); best_state=None; pat=acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            mask = mask_patches(x, acfg.mask_ratio, acfg.patch_size)
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
            val = 0.0; n=0
            for x, _ in dl_val:
                x = x.to(device)
                mask = mask_patches(x, acfg.mask_ratio, acfg.patch_size)
                x_in = x * (1 - mask)
                y = model(x_in)
                l = ((y - x) ** 2 * mask).sum() / (mask.sum() + 1e-8)
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


def adp_search(model: MaskedConvAE, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
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


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP Masked AE width/depth search")
    p.add_argument("--widths", type=int, nargs="+", default=[32,64,128])
    p.add_argument("--pool-idx", type=int, nargs="*", default=[0,2])
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
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
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_masked"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedConvAE(in_ch=3, widths=args.widths, pooling_indices=args.pool_idx).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience,
                     trials_width=args.trials_width, trials_depth=args.trials_depth,
                     ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs,
                     mask_ratio=args.mask_ratio, patch_size=args.patch_size)
    best_val, model, widths = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP Masked AE] mode={args.adp_mode} best_val={best_val:.6f} widths={widths} depth={len(widths)}")
