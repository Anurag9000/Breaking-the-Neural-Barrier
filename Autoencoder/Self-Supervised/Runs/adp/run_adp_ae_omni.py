# run_adp_ae_omni.py
# One runner to rule them all (A–I).
# Auto-picks the right dataset wrapper (temporal vs. image) based on --algo family.

import os, json, random, argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

from adp_ae_omni_core import build_model, OmniConfig, OMNI_SETS

# ------------ utils ------------
def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def psnr(x,y):
    mse = ((x-y)**2).mean()
    return 20*torch.log10(1.0/torch.sqrt(mse)) if mse.item()>0 else torch.tensor(99.0, device=x.device)

def save_preview(inp, recon, out_dir, tag, n=8, is_temporal=False):
    ensure_dir(out_dir)
    if is_temporal:
        x = inp[:,0].clamp(0,1)  # show t=0 reference
    else:
        x = inp.clamp(0,1)
    x = x[:n]; r = recon[:n].clamp(0,1)
    grid = make_grid(torch.cat([x, r], dim=0), nrow=n)
    save_image(grid, os.path.join(out_dir, f"samples_{tag}.png"))

# ------------ temporal dataset wrapper ------------
class TemporalFromImages(Dataset):
    def __init__(self, base_ds, T=2, size=128):
        self.ds = base_ds; self.T=T; self.size=size
        self.base = Tform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
        self.T = Tform
        self.temporal_jitter = torchvision.transforms.RandomAffine(
            degrees=2, translate=(0.02,0.02), scale=(0.98,1.02), shear=2
        )
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        x0 = self.T(img)
        seq = [x0]
        for _ in range(1, self.T):
            seq.append(self.temporal_jitter(x0))
        x = torch.stack(seq, dim=0)
        return x, 0

# ------------ data loading ------------
def build_transforms(img_size, train):
    if train:
        return T.Compose([T.Resize(img_size), T.RandomCrop(img_size, padding=4),
                          T.RandomHorizontalFlip(), T.ToTensor()])
    else:
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])

def load_images(name, root, img_size, val_split, download, seed):
    tr = build_transforms(img_size, True)
    te = build_transforms(img_size, False)
    name=name.lower()
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
    n = len(full); nv = max(1, int(val_split*n)); nt = n-nv
    gen = torch.Generator().manual_seed(seed)
    train, val = random_split(full, [nt, nv], generator=gen)
    return train, val, test

def load_temporal(name, root, img_size, val_split, download, seed, Tlen):
    tr_base, va_base, test_base = load_images(name, root, img_size, val_split, download, seed)
    train = TemporalFromImages(tr_base, T=Tlen, size=img_size)
    val   = TemporalFromImages(va_base, T=Tlen, size=img_size)
    test  = TemporalFromImages(test_base, T=Tlen, size=img_size)
    return train, val, test

# ------------ training / eval ------------
def train_epoch(model, loader, opt, device, algo, is_temporal, log_int):
    model.train(); tot=0; L=0.0
    for it, (xb, _) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model.forward_train(xb, algo=algo)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += xb.size(0); L += loss.item()*xb.size(0)
        if (it+1) % log_int == 0:
            print(f"  it {it+1}/{len(loader)}  loss={L/tot:.4f}")
    return L/max(1,tot)

@torch.no_grad()
def evaluate(model, loader, device, algo, is_temporal, out_img, tag):
    model.eval(); tot=0; L=0.0; P=0.0; preview=False
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        out = model.forward_train(xb, algo=algo)
        loss = out["loss"].item()
        recon = out.get("recon", None)
        tot += xb.size(0); L += loss*xb.size(0)
        if recon is not None:
            # PSNR vs canonical target:
            tgt = xb[:,1] if (is_temporal and xb.size(1)>1) else (xb if not is_temporal else xb[:,0])
            P += psnr(recon.clamp(0,1), tgt.clamp(0,1)).item()*xb.size(0)
            if not preview:
                save_preview(xb, recon, out_img, tag, is_temporal=is_temporal); preview=True
    return {"loss": L/max(1,tot), "psnr": (P/max(1,tot)) if P>0 else float('nan')}

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser("ADP Omni AE Runner (Parts A–I)")
    # data
    ap.add_argument("--algo", type=str, required=True, help="Any supported algorithm across A–I")
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","stl10","imagefolder"])
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--download", action="store_true")
    # temporal length (only used if algo is in temporal set)
    ap.add_argument("--T", type=int, default=2)

    # model (SSL core knobs)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--unet", action="store_true")
    ap.add_argument("--norm", type=str, default="bn", choices=["bn","gn","ln","none"])
    ap.add_argument("--act", type=str, default="relu", choices=["relu","gelu","silu"])
    ap.add_argument("--recon-loss", type=str, default="mse", choices=["mse","l1","huber"])
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--block-size", type=int, default=16)
    # ssl regularizers (optional)
    ap.add_argument("--w-sparse-l1", type=float, default=0.0)
    ap.add_argument("--w-group-sparse", type=float, default=0.0)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--w-contractive", type=float, default=0.0)
    ap.add_argument("--w-entropy", type=float, default=0.0)
    ap.add_argument("--w-tv", type=float, default=0.0)
    ap.add_argument("--w-whiten", type=float, default=0.0)

    # spatial knobs
    ap.add_argument("--patch-grid", type=int, default=3)
    ap.add_argument("--jigsaw-classes", type=int, default=6)
    ap.add_argument("--scale-bins", type=int, default=3)
    ap.add_argument("--trans-max-px", type=int, default=8)

    # temporal knobs
    ap.add_argument("--latent-dim", type=int, default=256)  # shared for spatial/temporal
    ap.add_argument("--temp-patch-grid", type=int, default=4)

    # regular knobs/weights
    ap.add_argument("--w-laplacian", type=float, default=1.0)
    ap.add_argument("--w-manifold", type=float, default=1.0)
    ap.add_argument("--w-tangent", type=float, default=0.1)
    ap.add_argument("--w-reg-entropy", type=float, default=0.001)
    ap.add_argument("--w-mi", type=float, default=0.001)
    ap.add_argument("--w-orth", type=float, default=1e-4)
    ap.add_argument("--w-lowrank", type=float, default=1e-4)
    ap.add_argument("--w-normalize", type=float, default=0.01)
    ap.add_argument("--w-reg-whiten", type=float, default=1e-4)
    ap.add_argument("--tangent-eps", type=float, default=0.03)

    # train
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="./results_omni")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # figure out family and pick dataset wrapper
    algo = args.algo
    if   algo in OMNI_SETS["temporal"]: is_temporal = True
    elif algo in OMNI_SETS["ssl"]     : is_temporal = False
    elif algo in OMNI_SETS["spatial"] : is_temporal = False
    elif algo in OMNI_SETS["regular"] : is_temporal = False
    else:
        raise ValueError(f"Unknown algo '{algo}'. Not in any registry.")

    if is_temporal:
        train_ds, val_ds, test_ds = load_temporal(args.dataset, args.data_root, args.img_size,
                                                  args.val_split, args.download, args.seed, args.T)
    else:
        train_ds, val_ds, test_ds = load_images(args.dataset, args.data_root, args.img_size,
                                                args.val_split, args.download, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    # Build Omni config
    ocfg = OmniConfig(
        device=str(device),
        # SSL
        ssl_base_ch=args.base_ch, ssl_depth=args.depth, ssl_unet=args.unet, ssl_norm=args.norm, ssl_act=args.act,
        ssl_recon_loss=args.recon_loss, ssl_huber_delta=args.huber_delta,
        ssl_mask_ratio=args.mask_ratio, ssl_block_size=args.block_size,
        w_sparse_l1=args.w_sparse_l1, w_group_sparse=args.w_group_sparse, group_size=args.group_size,
        w_contractive=args.w_contractive, w_entropy=args.w_entropy, w_tv=args.w_tv, w_whiten=args.w_whiten,
        # Temporal
        temp_base_ch=args.base_ch, temp_depth=args.depth, temp_latent_dim=args.latent_dim,
        temp_norm=args.norm, temp_act=args.act, temp_patch_grid=args.temp_patch_grid,
        temp_recon_loss=args.recon_loss, temp_huber_delta=args.huber_delta,
        # Spatial
        sp_base_ch=args.base_ch, sp_depth=args.depth, sp_latent_dim=args.latent_dim,
        sp_norm=args.norm, sp_act=args.act, sp_patch_grid=args.patch_grid,
        sp_jigsaw_classes=args.jigsaw_classes, sp_scale_bins=args.scale_bins,
        sp_trans_max_px=args.trans_max_px, sp_recon_loss=args.recon_loss, sp_huber_delta=args.huber_delta,
        # Regular
        reg_base_ch=args.base_ch, reg_depth=args.depth, reg_latent_dim=args.latent_dim,
        reg_norm=args.norm, reg_act=args.act, reg_recon_loss=args.recon_loss, reg_huber_delta=args.huber_delta,
        w_laplacian=args.w_laplacian, w_manifold=args.w_manifold, w_tangent=args.w_tangent,
        w_reg_entropy=args.w_reg_entropy, w_mi=args.w_mi, w_orth=args.w_orth, w_lowrank=args.w_lowrank,
        w_normalize=args.w_normalize, w_reg_whiten=args.w_reg_whiten, tangent_eps=args.tangent_eps,
    )

    model = build_model(algo=algo, cfg=ocfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ck = os.path.join(args.out_dir, "ckpts"); im = os.path.join(args.out_dir, "images")
    ensure_dir(ck); ensure_dir(im)

    best=1e9; hist=[]
    for ep in range(1, args.epochs+1):
        print(f"\n=== Epoch {ep}/{args.epochs} | Algo={algo} ===")
        tr = train_epoch(model, train_loader, opt, device, algo, is_temporal, args.log_interval)
        va = evaluate(model, val_loader, device, algo, is_temporal, im, tag=f"val_ep{ep}")
        print(f"[val] loss={va['loss']:.4f}  psnr={va['psnr']:.2f} dB")
        hist.append({"epoch": ep, "train_loss": tr, **va})
        if va["loss"] < best - 1e-6:
            best = va["loss"]
            p = os.path.join(ck, f"OMNI_{algo}_{args.dataset}_best.pth")
            torch.save({"model": model.state_dict(), "epoch": ep, "val": best, "algo": algo}, p)
            print(f"  ✓ Saved BEST: {p}")

    te = evaluate(model, test_loader, device, algo, is_temporal, im, tag="test")
    print(f"\n[test] loss={te['loss']:.4f}  psnr={te['psnr']:.2f} dB")
    with open(os.path.join(args.out_dir, f"history_{algo}_{args.dataset}.json"), "w") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    main()
