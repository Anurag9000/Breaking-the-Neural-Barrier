"""Runner: VICRegL-ViT"""
import argparse
import torch
import torch.optim as optim
from torchvision import transforms

from TRANSFORMERS.Transformer.Supervised.Runs._common_real_image import make_real_image_loaders
from vicregl_vit import VICRegLConfig, VICRegLViT


def _make_tensor_aug(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
    ])


def _make_two_views(imgs, aug):
    x1 = imgs
    x2 = torch.stack([aug(img) for img in imgs])
    return x1, x2


def train_epoch(model, loader, device, opt, aug):
    model.train()
    total = 0.0
    for imgs, _ in loader:
        x1, x2 = _make_two_views(imgs.to(device), aug)
        loss, _ = model(x1, x2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_obj(model, loader, device, aug):
    model.eval()
    total = 0.0
    for imgs, _ in loader:
        x1, x2 = _make_two_views(imgs.to(device), aug)
        loss, _ = model(x1, x2)
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--proj_hidden", type=int, default=2048)
    ap.add_argument("--proj_out", type=int, default=1024)
    ap.add_argument("--w_inv", type=float, default=25.0)
    ap.add_argument("--w_var", type=float, default=25.0)
    ap.add_argument("--w_cov", type=float, default=1.0)
    ap.add_argument("--w_local", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--save", default="vicregl_vit_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_real_image_loaders(args.data_dir, batch_size=args.batch_size, image_size=args.img)
    aug = _make_tensor_aug(args.img)

    cfg = VICRegLConfig(
        args.img,
        args.patch,
        args.dim,
        args.depth,
        args.heads,
        args.ratio,
        args.proj_hidden,
        args.proj_out,
        args.w_inv,
        args.w_var,
        args.w_cov,
        args.w_local,
        args.gamma,
    )
    model = VICRegLViT(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(1, args.epochs + 1):
        trl = train_epoch(model, tr, device, opt, aug)
        val = eval_obj(model, va, device, aug)
        print(f"epoch {ep:03d} | train {trl:.4f} | val {val:.4f}")
        if val + 1e-6 < best:
            best = val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, args.save)
        print(f"Saved {args.save} (val {best:.4f})")
    test = eval_obj(model, te, device, aug)
    print(f"Test objective: {test:.4f}")


if __name__ == "__main__":
    main()
