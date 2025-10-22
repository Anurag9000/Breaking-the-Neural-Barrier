# run_adp_ae_regular_core.py
# Runner for Part E regularization/geometry AEs. Saves best ckpt & previews.

import os, json, random, argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

from adp_ae_regular_core import RAEConfig, build_model, SUPPORTED_REGULAR

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def build_transforms(img_size, train):
    if train:
        return T.Compose([T.Resize(img_size), T.RandomCrop(img_size, padding=4),
                          T.RandomHorizontalFlip(), T.ToTensor()])
    else:
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])

def load_dataset(name, root, img_size, val_split, download, seed):
    tr = build_transforms(img_size, True)
    te = build_transforms(img_size, False)
    name = name.lower()
    if name=="cifar10":
        full = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=tr)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=te)
    elif name=="stl10":
        full = torchvision.datasets.STL10(root=root, split="train", download=download, transform=tr)
        test = torchvision.datasets.STL10(root=root, split="test",  download=download, transform=te)
    elif name=="imagefolder":
        train_dir = os.path.join(root, "train"); test_dir = os.path.join(root, "test")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            full = torchvision.datasets.ImageFolder(train_dir, transform=tr)
            test = torchvision.datasets.ImageFolder(test_dir,  transform=te)
        else:
            full = torchvision.datasets.ImageFolder(root, transform=tr)
            test = torchvision.datasets.ImageFolder(root, transform=te)
    else:
        raise ValueError("Unknown dataset")
    n = len(full); nv = max(1, int(val_split*n)); nt = n - nv
    gen = torch.Generator().manual_seed(seed)
    train, val = random_split(full, [nt, nv], generator=gen)
    return train, val, test

@torch.no_grad()
def psnr(x,y):
    mse = ((x-y)**2).mean()
    return 20*torch.log10(1.0/torch.sqrt(mse)) if mse.item()>0 else torch.tensor(99.0, device=x.device)

def save_preview(x, recon, out_dir, tag, n=8):
    ensure_dir(out_dir)
    x = x[:n].clamp(0,1); r = recon[:n].clamp(0,1)
    grid = make_grid(torch.cat([x, r], dim=0), nrow=n)
    save_image(grid, os.path.join(out_dir, f"samples_{tag}.png"))

def train_epoch(model, loader, opt, device, algo, log_int):
    model.train(); tot=0; L=0.0
    for it,(x,_) in enumerate(loader):
        x=x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model.forward_train(x, algo)
        loss = out["loss"]; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += x.size(0); L += loss.item()*x.size(0)
        if (it+1)%log_int==0:
            print(f"  it {it+1}/{len(loader)} loss={L/tot:.4f}")
    return L/max(1,tot)

@torch.no_grad()
def evaluate(model, loader, device, algo, out_img, tag):
    model.eval(); tot=0; L=0.0; P=0.0; preview=False
    for x,_ in loader:
        x=x.to(device, non_blocking=True)
        out = model.forward_train(x, algo)
        loss = out["loss"].item(); recon = out.get("recon", None)
        tot += x.size(0); L += loss*x.size(0)
        if recon is not None:
            P += psnr(recon.clamp(0,1), x.clamp(0,1)).item()*x.size(0)
            if not preview:
                save_preview(x, recon, out_img, tag); preview=True
    return {"loss": L/max(1,tot), "psnr": (P/max(1,tot)) if P>0 else float('nan')}

def main(default_algo=None):
    ap = argparse.ArgumentParser("Regularization/Geometry AE Runner (Part E)")
    # data
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","stl10","imagefolder"])
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--download", action="store_true")
    # model
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--norm", type=str, default="bn", choices=["bn","gn","ln","none"])
    ap.add_argument("--act", type=str, default="relu", choices=["relu","gelu","silu"])
    ap.add_argument("--recon-loss", type=str, default="mse", choices=["mse","l1","huber"])
    ap.add_argument("--huber-delta", type=float, default=1.0)
    # regularizer weights (optional fine-tuning)
    ap.add_argument("--w-laplacian", type=float, default=1.0)
    ap.add_argument("--w-manifold", type=float, default=1.0)
    ap.add_argument("--w-tangent", type=float, default=0.1)
    ap.add_argument("--w-entropy", type=float, default=0.001)
    ap.add_argument("--w-mi", type=float, default=0.001)
    ap.add_argument("--w-orth", type=float, default=1e-4)
    ap.add_argument("--w-lowrank", type=float, default=1e-4)
    ap.add_argument("--w-normalize", type=float, default=0.01)
    ap.add_argument("--w-whiten", type=float, default=1e-4)
    ap.add_argument("--tangent-eps", type=float, default=0.03)
    # algo
    ap.add_argument("--algo", type=str, default=default_algo, choices=sorted(list(SUPPORTED_REGULAR)))
    # train
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="./results_regular")
    args = ap.parse_args()

    assert args.algo in SUPPORTED_REGULAR
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = load_dataset(args.dataset, args.data_root, args.img_size, args.val_split, args.download, args.seed)
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    te = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    cfg = RAEConfig(
        in_channels=3, base_channels=args.base_ch, depth=args.depth, latent_dim=args.latent_dim,
        norm=args.norm, act=args.act, recon_loss=args.recon_loss, huber_delta=args.huber_delta,
        w_laplacian=args.w_laplacian, w_manifold=args.w_manifold, w_tangent=args.w_tangent,
        w_entropy=args.w_entropy, w_mi=args.w_mi, w_orth=args.w_orth, w_lowrank=args.w_lowrank,
        w_normalize=args.w_normalize, w_whiten=args.w_whiten, tangent_eps=args.tangent_eps,
        device=str(device),
    )
    model = build_model(cfg, algo=args.algo).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ck = os.path.join(args.out_dir, "ckpts"); im = os.path.join(args.out_dir, "images")
    Path(ck).mkdir(parents=True, exist_ok=True); Path(im).mkdir(parents=True, exist_ok=True)

    best=1e9; hist=[]
    for ep in range(1, args.epochs+1):
        print(f"\n=== Epoch {ep}/{args.epochs} | Algo={args.algo} ===")
        tr_loss = train_epoch(model, tr, opt, device, args.algo, args.log_interval)
        va_m = evaluate(model, va, device, args.algo, im, tag=f"val_ep{ep}")
        print(f"[val] loss={va_m['loss']:.4f}  psnr={va_m['psnr']:.2f} dB")
        hist.append({"epoch": ep, "train_loss": tr_loss, **va_m})
        if va_m["loss"] < best - 1e-6:
            best = va_m["loss"]
            p = os.path.join(ck, f"REG_AE_{args.algo}_{args.dataset}_best.pth")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": ep, "val": best}, p)
            print(f"  ✓ Saved BEST: {p}")

    te_m = evaluate(model, te, device, args.algo, im, tag="test")
    print(f"\n[test] loss={te_m['loss']:.4f}  psnr={te_m['psnr']:.2f} dB")
    with open(os.path.join(args.out_dir, f"history_{args.algo}_{args.dataset}.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    main()
