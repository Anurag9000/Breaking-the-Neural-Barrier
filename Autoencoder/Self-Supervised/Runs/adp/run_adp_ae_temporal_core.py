# run_adp_ae_temporal_core.py
# Core runner for temporal AE variants. Builds synthetic "temporal" windows from images
# by applying small random transforms over time (works with CIFAR10/STL10/ImageFolder).
# Use thin wrappers to select algo (see the 7 run_* wrappers).
#
# Author: ADP / Breaking Neural Barrier

import os, json, random, argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

from adp_ae_temporal_core import TAEConfig, build_model, SUPPORTED_TEMPORAL

# ---------- utils ----------
def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def psnr(x,y):
    mse = ((x-y)**2).mean()
    return (20*torch.log10(1.0/torch.sqrt(mse))).item() if mse.item()>0 else 99.0

# ---------- temporal dataset wrapper ----------
class TemporalFromImages(Dataset):
    """
    Turns a still-image dataset into short sequences (T frames) by applying
    small, *correlated* augmentations over time to the SAME image.
    """
    def __init__(self, base_ds, T=2, size=128):
        self.ds = base_ds
        self.T = T
        self.size = size
        # base canonical transform
        self.base = T = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.ToTensor(),
        ])
        # small temporal jitter
        self.temporal_jitter = torchvision.transforms.RandomAffine(
            degrees=2, translate=(0.02,0.02), scale=(0.98,1.02), shear=2
        )

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        x0 = self.base(img)
        seq = [x0]
        for t in range(1, self.T):
            xt = self.temporal_jitter(x0)
            seq.append(xt)
        x = torch.stack(seq, dim=0)   # (T,C,H,W)
        return x, 0

# ---------- data loading ----------
def load_dataset(name, root, img_size, T, val_split, download, seed):
    name = name.lower()
    if name == "cifar10":
        full_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download)
        test_set   = torchvision.datasets.CIFAR10(root=root, train=False, download=download)
    elif name == "stl10":
        full_train = torchvision.datasets.STL10(root=root, split="train", download=download)
        test_set   = torchvision.datasets.STL10(root=root, split="test", download=download)
    elif name == "imagefolder":
        full_train = torchvision.datasets.ImageFolder(os.path.join(root, "train") if os.path.isdir(os.path.join(root,"train")) else root)
        test_set   = torchvision.datasets.ImageFolder(os.path.join(root, "test")  if os.path.isdir(os.path.join(root,"test"))  else root)
    else:
        raise ValueError("Unknown dataset")

    n = len(full_train); n_val = max(1, int(val_split * n)); n_tr = n - n_val
    gen = torch.Generator().manual_seed(seed)
    tr_base, va_base = random_split(full_train, [n_tr, n_val], generator=gen)

    train = TemporalFromImages(tr_base, T=T, size=img_size)
    val   = TemporalFromImages(va_base, T=T, size=img_size)
    test  = TemporalFromImages(test_set, T=T, size=img_size)
    return train, val, test

# ---------- training / eval ----------
def save_preview(seq, recon, out_dir, tag):
    ensure_dir(out_dir)
    # show t=0 and t=1 (if exist)
    x0 = seq[:,0].clamp(0,1)
    xr = recon.clamp(0,1)
    grid = make_grid(torch.cat([x0, xr], dim=0), nrow=x0.size(0))
    save_image(grid, os.path.join(out_dir, f"samples_{tag}.png"))

def train_epoch(model, loader, opt, device, algo, log_int):
    model.train(); tot = 0; loss_sum = 0.0
    for it, (seq, _) in enumerate(loader):
        seq = seq.to(device)  # (B,T,C,H,W)
        opt.zero_grad(set_to_none=True)
        out = model.forward_train(seq, algo=algo)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * seq.size(0); tot += seq.size(0)
        if (it+1) % log_int == 0:
            print(f"  it {it+1}/{len(loader)}  loss={loss_sum/tot:.4f}")
    return loss_sum / max(1, tot)

@torch.no_grad()
def evaluate(model, loader, device, algo, out_img, tag):
    model.eval(); tot=0; L=0.0; P=0.0; preview=False
    for seq, _ in loader:
        seq = seq.to(device)
        out = model.forward_train(seq, algo=algo)
        loss = out["loss"].item()
        recon = out.get("recon", None)
        tot += seq.size(0); L += loss * seq.size(0)
        if recon is not None:
            # PSNR between recon and target used in that algo: we use last 'recon' vs seq[:,1] if exists else seq[:,0]
            tgt = seq[:,1] if seq.size(1) > 1 else seq[:,0]
            P += psnr(recon.clamp(0,1), tgt.clamp(0,1)) * seq.size(0)
            if not preview:
                save_preview(seq, recon, out_img, tag); preview=True
    return {"loss": L/max(1,tot), "psnr": (P/max(1,tot)) if P>0 else float("nan")}

# ---------- main ----------
def main(default_algo=None, default_T=2):
    ap = argparse.ArgumentParser("Temporal AE Runner (core)")
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","stl10","imagefolder"])
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--T", type=int, default=default_T)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--download", action="store_true")
    # model
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--norm", type=str, default="bn", choices=["bn","gn","ln","none"])
    ap.add_argument("--act", type=str, default="relu", choices=["relu","gelu","silu"])
    ap.add_argument("--patch-grid", type=int, default=4)
    ap.add_argument("--recon-loss", type=str, default="mse", choices=["mse","l1","huber"])
    ap.add_argument("--huber-delta", type=float, default=1.0)
    # algo
    ap.add_argument("--algo", type=str, default=default_algo, choices=sorted(list(SUPPORTED_TEMPORAL)))
    # train
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="./results_temporal")
    args = ap.parse_args()

    assert args.algo in SUPPORTED_TEMPORAL, "Choose a temporal algo"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train, val, test = load_dataset(args.dataset, args.data_root, args.img_size, args.T, args.val_split, args.download, args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    val_loader   = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    test_loader  = DataLoader(test, batch_size=args.batch_size*2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    # model
    cfg = TAEConfig(in_channels=3, base_channels=args.base_ch, depth=args.depth,
                    latent_dim=args.latent_dim, norm=args.norm, act=args.act,
                    patch_grid=args.patch_grid, recon_loss=args.recon_loss, huber_delta=args.huber_delta,
                    device=str(device))
    model = build_model(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_ck = os.path.join(args.out_dir, "ckpts"); out_img = os.path.join(args.out_dir, "images")
    ensure_dir(out_ck); ensure_dir(out_img)

    best = 1e9; best_ep=-1; hist=[]
    for ep in range(1, args.epochs+1):
        print(f"\n=== Epoch {ep}/{args.epochs} | Algo={args.algo} | T={args.T} ===")
        tr = train_epoch(model, train_loader, opt, device, args.algo, args.log_interval)
        va = evaluate(model, val_loader, device, args.algo, out_img, tag=f"val_ep{ep}")
        print(f"[val] loss={va['loss']:.4f}  psnr={va['psnr']:.2f} dB")
        hist.append({"epoch": ep, "train_loss": tr, **va})
        if va["loss"] < best - 1e-6:
            best, best_ep = va["loss"], ep
            path = os.path.join(out_ck, f"TEMP_AE_{args.algo}_{args.dataset}_best.pth")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": ep, "val": best}, path)
            print(f"  ✓ Saved BEST: {path}")

    te = evaluate(model, test_loader, device, args.algo, out_img, tag="test")
    print(f"\n[test] loss={te['loss']:.4f}  psnr={te['psnr']:.2f} dB")
    with open(os.path.join(args.out_dir, f"history_{args.algo}_{args.dataset}.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    main()
