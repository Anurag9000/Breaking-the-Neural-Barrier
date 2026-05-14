# run_adp_ae_ssl.py
# Runner for the unified Single-Model Self-Supervised AE core (non-VAE).
# Usage example:
#   python run_adp_ae_ssl.py --algo masked --dataset imagefolder --data-root /path/to/images

import os
import math
import time
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision.utils import save_image, make_grid

from adp_ae_ssl_core import AEConfig, build_model, SUPPORTED_ALGOS
from _common_real_image import make_real_image_loaders

# -----------------------
# Utils
# -----------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def psnr(x, y):
    # x,y in [0,1] range recommended
    mse = torch.mean((x - y) ** 2)
    if mse.item() == 0:
        return torch.tensor(99.0, device=x.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# -----------------------
# Data
# -----------------------

def load_dataset(name: str, data_root: str, img_size: int, val_split: float, download: bool, seed: int):
    del name, download, seed
    train_loader, val_loader, test_loader = make_real_image_loaders(
        data_root=data_root,
        batch_size=1,
        val_ratio=val_split,
        num_workers=0,
        image_size=img_size,
    )
    return train_loader.dataset, val_loader.dataset, test_loader.dataset

# -----------------------
# Training / Eval
# -----------------------

def save_samples(x, recon, out_dir: str, tag: str, max_n: int = 16):
    ensure_dir(out_dir)
    with torch.no_grad():
        x = x[:max_n].clamp(0,1)
        recon = recon[:max_n].clamp(0,1)
        grid = make_grid(torch.cat([x, recon], dim=0), nrow=max_n)
        save_image(grid, os.path.join(out_dir, f"samples_{tag}.png"))

def train_one_epoch(model, loader, optimizer, scaler, device, algo, grad_clip, log_interval):
    model.train()
    total, total_loss = 0, 0.0
    t0 = time.time()
    for it, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model.forward_train(x, algo=algo)
            loss = out["loss"]

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += x.size(0)
        total_loss += loss.item() * x.size(0)

        if (it + 1) % log_interval == 0:
            dt = time.time() - t0
            print(f"  [train] it={it+1}/{len(loader)}  loss={total_loss/total:.4f}  {dt:.1f}s")
            t0 = time.time()

    return total_loss / max(1, total)

@torch.no_grad()
def evaluate(model, loader, device, algo, save_preview_dir=None, preview_tag="val"):
    model.eval()
    total, total_loss, total_psnr = 0, 0.0, 0.0
    preview_done = False

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        out = model.forward_train(x, algo=algo)
        loss = out["loss"].item()
        recon = out.get("recon", None)

        total += x.size(0)
        total_loss += loss * x.size(0)
        if recon is not None:
            total_psnr += psnr(recon.clamp(0,1), x.clamp(0,1)).item() * x.size(0)

        # one preview image
        if (not preview_done) and save_preview_dir and recon is not None:
            save_samples(x, recon, save_preview_dir, preview_tag)
            preview_done = True

    avg_loss = total_loss / max(1, total)
    avg_psnr = (total_psnr / max(1, total)) if total_psnr > 0 else float('nan')
    return {"loss": avg_loss, "psnr": avg_psnr}

# -----------------------
# Main
# -----------------------

def main():
    p = argparse.ArgumentParser(description="Unified runner for single-model self-supervised AEs (non-VAE)")
    # Data
    p.add_argument("--dataset", type=str, default="imagefolder", choices=["imagefolder"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--download", action="store_true")
    # Model
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--unet", action="store_true")
    p.add_argument("--norm", type=str, default="bn", choices=["bn","gn","ln","none"])
    p.add_argument("--act", type=str, default="relu", choices=["relu","gelu","silu"])
    # Algo / SSL
    p.add_argument("--algo", type=str, required=True, choices=sorted(list(SUPPORTED_ALGOS)))
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--recon-loss", type=str, default="mse", choices=["mse","l1","huber"])
    p.add_argument("--huber-delta", type=float, default=1.0)
    # Regularizers
    p.add_argument("--w-sparse-l1", type=float, default=0.0)
    p.add_argument("--w-group-sparse", type=float, default=0.0)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--w-contractive", type=float, default=0.0)
    p.add_argument("--w-entropy", type=float, default=0.0)
    p.add_argument("--w-tv", type=float, default=0.0)
    p.add_argument("--w-whiten", type=float, default=0.0)
    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # Misc
    p.add_argument("--out-dir", type=str, default="./results_ae_ssl")
    p.add_argument("--save-every", type=int, default=0, help="Optional: save checkpoint every N epochs (0=only best)")
    p.add_argument("--log-interval", type=int, default=50)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "ckpts")
    img_dir  = os.path.join(args.out_dir, "images")
    ensure_dir(ckpt_dir)
    ensure_dir(img_dir)

    # Data
    train_set, val_set, test_set = load_dataset(
        args.dataset, args.data_root, args.img_size, args.val_split, args.download, args.seed
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    test_loader  = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    # Model
    cfg = AEConfig(
        in_channels=3,
        base_channels=args.base_ch,
        depth=args.depth,
        use_unet=args.unet,
        norm=args.norm,
        act=args.act,
        w_sparse_l1=args.w_sparse_l1,
        w_group_sparse=args.w_group_sparse,
        group_size=args.group_size,
        w_contractive=args.w_contractive,
        w_entropy=args.w_entropy,
        w_tv=args.w_tv,
        w_whiten=args.w_whiten,
        recon_loss=args.recon_loss,
        huber_delta=args.huber_delta,
        mask_ratio=args.mask_ratio,
        block_size=args.block_size,
        device=str(device),
    )

    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params/1e6:.3f}M | Algo: {args.algo} | Dataset: {args.dataset}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Train loop with early stopping on val loss
    best_val = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.algo, args.grad_clip, args.log_interval)
        val_metrics = evaluate(model, val_loader, device, args.algo, save_preview_dir=img_dir, preview_tag=f"val_ep{epoch}")
        print(f"[val] loss={val_metrics['loss']:.4f}  psnr={val_metrics['psnr']:.2f} dB")

        history.append({"epoch": epoch, "train_loss": tr_loss, **val_metrics})

        # checkpointing
        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"ADP_AE_{args.algo}_{args.dataset}_best.pth")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "val": best_val}, best_path)
            print(f"  ✓ Saved BEST checkpoint: {best_path}")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            path = os.path.join(ckpt_dir, f"ADP_AE_{args.algo}_{args.dataset}_ep{epoch}.pth")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "val": val_metrics["loss"]}, path)
            print(f"  • Saved checkpoint: {path}")

        # early stopping
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping: best epoch={best_epoch}, best val={best_val:.4f}")
            break

    # Final eval on test
    print("\n=== Testing on held-out set ===")
    test_metrics = evaluate(model, test_loader, device, args.algo, save_preview_dir=img_dir, preview_tag="test")
    print(f"[test] loss={test_metrics['loss']:.4f}  psnr={test_metrics['psnr']:.2f} dB")

    # Save history
    hist_path = os.path.join(args.out_dir, f"history_{args.algo}_{args.dataset}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {hist_path}")

if __name__ == "__main__":
    main()
