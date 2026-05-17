
import argparse, os, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mlp_ssl_stl import MLPSSL, nt_xent_loss

class TwoCropTransform:
    def __init__(self, base_transform):
        self.base = base_transform
    def __call__(self, x):
        return self.base(x), self.base(x)

def build_ssl_transforms(img_size):
    blur_kernel = int(0.1*min(img_size))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])
    return TwoCropTransform(augmentation)

def build_data(dataset, data_dir, img_size, val_split):
    ssl_tfm = build_ssl_transforms(img_size)
    dataset = dataset.lower()
    if dataset == "cifar10":
        ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=ssl_tfm)
        C=3
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=ssl_tfm)
        C=3
    else:
        raise ValueError("Unsupported dataset. Use cifar10 or cifar100.")
    val_len = int(len(ds) * val_split)
    tr_len = len(ds) - val_len
    tr, va = random_split(ds, [tr_len, val_len])
    return tr, va, (C, img_size[0], img_size[1])

def train_epoch(model, loader, device, temperature):
    model.train()
    total, n = 0.0, 0
    for (x1, x2), _ in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        _, p1 = model(x1)
        _, p2 = model(x2)
        loss = nt_xent_loss(p1, p2, temperature=temperature)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x1.size(0)
        n += x1.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, device, temperature):
    model.eval()
    total, n = 0.0, 0
    for (x1, x2), _ in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        _, p1 = model(x1)
        _, p2 = model(x2)
        loss = nt_xent_loss(p1, p2, temperature=temperature)
        total += loss.item() * x1.size(0)
        n += x1.size(0)
    return total / max(n, 1)

def main():
    global optimizer
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--img_size", type=int, nargs=2, default=[32,32])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024])
    p.add_argument("--rep_dim", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, shape = build_data(args.dataset, args.data_dir, args.img_size, args.val_split)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    in_dim = shape[0]*shape[1]*shape[2]
    model = MLPSSL(in_dim, args.hidden, args.rep_dim, args.proj_dim, use_bn=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, trl, device, args.temperature)
        va_loss = eval_epoch(model, val, device, args.temperature)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[{ep:03d}] train_ntxent={tr_loss:.6f} val_ntxent={va_loss:.6f} | hidden={args.hidden} rep={args.rep_dim} proj={args.proj_dim}")

    if best_state is not None:
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", f"mlp_ssl_stl_{args.dataset}.pt")
        torch.save({"model": best_state, "val_ntxent": best_val, "config": vars(args)}, path)
        print(f"Saved best checkpoint to: {path} (val_ntxent={best_val:.6f})")

if __name__ == "__main__":
    main()
