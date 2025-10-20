import argparse, os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from vae_discretized_logistic_model import VAE_DiscretizedLogistic


def get_loaders(root, bs, val_split=5000):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    tr_len = len(train) - val_split
    tr, va = random_split(train, [tr_len, val_split], generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, bs, True, num_workers=2, pin_memory=True),
            DataLoader(va, bs, False, num_workers=2, pin_memory=True),
            DataLoader(test, bs, False, num_workers=2, pin_memory=True))


def train_epoch(model, loader, opt, device):
    model.train(); total = 0.0
    for x,_ in loader:
        x = x.to(device)
        loss = model(x)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()*x.size(0)
    return total/len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            loss = model(x)
            total += loss.item()*x.size(0)
    return total/len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='data')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--z-dim', type=int, default=32)
    ap.add_argument('--outdir', type=str, default='results_vae/discretized_logistic')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr, va, te = get_loaders(args.data_root, args.batch_size)
    model = VAE_DiscretizedLogistic(in_ch=1, out_ch=1, z_dim=args.z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best=float('inf'); best_state=None; bad=0
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, tr, opt, device)
        va_loss = eval_epoch(model, va, device)
        print(f"epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss + 1e-9 < best:
            best = va_loss; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad += 1
            if bad >= args.patience:
                print('Early stopping.'); break

    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss = eval_epoch(model, te, device)
    print(f'TEST (ELBO DiscLogit): {te_loss:.4f}')

    torch.save(model.state_dict(), os.path.join(args.outdir, 'vae_disc_logit.pt'))

if __name__ == '__main__':
    main()
