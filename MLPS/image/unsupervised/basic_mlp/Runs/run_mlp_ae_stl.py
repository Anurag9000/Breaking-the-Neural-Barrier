
import argparse, math, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mlp_ae_stl import MLPAutoencoder

def get_dataset(name: str, data_dir: str, img_size=(32, 32), train=True):
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # [0,1]
    ])
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(data_dir, train=train, download=True, transform=tfm)
    if name == "cifar100":
        return datasets.CIFAR100(data_dir, train=train, download=True, transform=tfm)
    raise ValueError(f"Unsupported dataset: {name}. Use cifar10 or cifar100.")

def build_autoencoder(input_shape, hidden_widths, bottleneck, output_activation):
    C, H, W = input_shape
    in_dim = C * H * W
    return MLPAutoencoder(in_dim, hidden_widths, bottleneck, use_bn=True, output_activation=output_activation)

def train_epoch(model, loader, optimizer, device, denoise_std=0.0):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        clean = imgs
        if denoise_std > 0:
            noisy = torch.clamp(imgs + denoise_std * torch.randn_like(imgs), 0.0, 1.0)
        else:
            noisy = imgs
        optimizer.zero_grad()
        recon = model(noisy)
        loss = loss_fn(recon, clean.view(clean.size(0), -1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total / max(count, 1)

@torch.no_grad()
def eval_epoch(model, loader, device, denoise_std=0.0):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        clean = imgs
        if denoise_std > 0:
            noisy = torch.clamp(imgs + denoise_std * torch.randn_like(imgs), 0.0, 1.0)
        else:
            noisy = imgs
        recon = model(noisy)
        loss = loss_fn(recon, clean.view(clean.size(0), -1))
        total += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total / max(count, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--img_size", type=int, nargs=2, default=[32,32])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--output_activation", type=str, default="sigmoid", choices=["sigmoid","tanh"])
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--denoise_std", type=float, default=0.0, help="Set >0.0 to train a denoising AE")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms/loader builder convenience variables
    tf_ = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor()])
    if args.dataset.lower() == "cifar10":
        train_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf_)
        C=3
    elif args.dataset.lower() == "cifar100":
        train_full = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=tf_)
        C=3
    else:
        raise ValueError("Unsupported dataset. Use cifar10 or cifar100.")

    H, W = args.img_size
    in_shape = (C, H, W)

    val_len = int(len(train_full) * args.val_split)
    train_len = len(train_full) - val_len
    train_set, val_set = random_split(train_full, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_autoencoder(in_shape, args.hidden, args.bottleneck, args.output_activation).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, opt, device, denoise_std=args.denoise_std)
        va = eval_epoch(model, val_loader, device, denoise_std=args.denoise_std)
        if va < best_val:
            best_val = va
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"[{epoch:03d}] train_mse={tr:.6f} val_mse={va:.6f} | hidden={args.hidden} bottleneck={args.bottleneck}")

    if best_state is not None:
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", f"mlp_ae_stl_{args.dataset}.pt")
        torch.save({"model": best_state, "val_mse": best_val, "config": vars(args)}, ckpt_path)
        print(f"Saved best checkpoint to: {ckpt_path} (val_mse={best_val:.6f})")

if __name__ == "__main__":
    try:
        import os as _os, sys as _sys
        if _os.name == "posix" and _sys.platform.startswith("linux"):
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)
        elif _os.name == "nt":
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetProcessWorkingSetSize(_ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass
    main()
