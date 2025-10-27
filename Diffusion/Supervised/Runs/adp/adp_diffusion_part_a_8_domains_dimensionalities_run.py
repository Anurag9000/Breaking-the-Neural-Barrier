# ============================================================
# File: run_adp_diff_domains.py  (RUN)
# Runner for generalized N-D DDPM (ε-pred) with 6 ADP policies
# across domains: --domain {1d,2d,3d,video,audio}
# ============================================================

import argparse

import torch
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from adp_diff_domains import EpsDDPMND, TrainCfg, SearchCfg, POLICIES


# -----------------------------
# Minimal domain-aware datasets
# -----------------------------

class RandomND(Dataset):
    def __init__(self, shape, n=5000):
        self.shape = shape; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.randn(*self.shape)
        x = x.clamp(-1,1)
        return x, 0


def make_loaders(domain: str, data_root: str, img_size: int, batch: int, val_split: float,
                 length_1d: int, vol_d: int, vid_t: int):
    if domain == '2d':
        tfm_train = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(img_size, padding=4),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tfm_val = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        ds_train_full = tv.datasets.CIFAR10(data_root, train=True, download=True, transform=tfm_train)
        ds_val_full   = tv.datasets.CIFAR10(data_root, train=True, download=True, transform=tfm_val)
        n = len(ds_train_full); n_val = int(n * val_split)
        idx = torch.randperm(n); val_idx = idx[:n_val]; train_idx = idx[n_val:]
        train = torch.utils.data.Subset(ds_train_full, train_idx.tolist())
        val   = torch.utils.data.Subset(ds_val_full, val_idx.tolist())
        train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    # synthetic fallbacks for other domains
    if domain == '1d':
        shape = (1, length_1d)
    elif domain == '3d':
        shape = (1, vol_d, img_size, img_size)
    elif domain == 'video':
        shape = (3, vid_t, img_size, img_size)
    elif domain == 'audio':
        shape = (1, img_size, img_size)  # spectrogram-like
    else:
        raise ValueError('Unknown domain')

    N = 5000
    ds = RandomND(shape, n=N)
    n_val = int(N * val_split)
    train_loader = DataLoader(torch.utils.data.Subset(ds, list(range(n_val, N))), batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(torch.utils.data.Subset(ds, list(range(0, n_val))), batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description='N-D DDPM (ε-pred) with ADP across domains')

    p.add_argument('--domain', type=str, default='2d', choices=['1d','2d','3d','video','audio'])
    p.add_argument('--adp', type=str, default='depth2width',
                   choices=['depth2width','width2depth','alt_depth','alt_width','depth_only','width_only'])

    # Data
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--img-size', type=int, default=32)
    p.add_argument('--length-1d', type=int, default=1024)
    p.add_argument('--vol-d', type=int, default=16)
    p.add_argument('--vid-t', type=int, default=8)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.2)

    # Train cfg
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-epochs', type=int, default=30)
    p.add_argument('--es-patience', type=int, default=7)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Search cfg
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--trials-depth', type=int, default=20)
    p.add_argument('--trials-width', type=int, default=20)
    p.add_argument('--ex-k', type=int, default=8)
    p.add_argument('--max-neurons', type=int, default=0, help='0 disables capacity limit')

    # Model hyperparams
    p.add_argument('--widths', type=int, nargs='+', default=[32,64,96])
    p.add_argument('--T', type=int, default=1000)

    args = p.parse_args()

    train_loader, val_loader = make_loaders(args.domain, args.data, args.img_size, args.batch, args.val_split,
                                            args.length_1d, args.vol_d, args.vid_t)

    model = EpsDDPMND(domain=args.domain, widths=args.widths, T=args.T)

    train_cfg = TrainCfg(lr=args.lr, max_epochs=args.max_epochs, es_patience=args.es_patience,
                         grad_clip=args.grad_clip, device=args.device)
    maxN = None if args.max_neurons == 0 else args.max_neurons
    search_cfg = SearchCfg(delta=args.delta, trials_width=args.trials_width, trials_depth=args.trrials_depth if hasattr(args,'trrials_depth') else args.trials_depth,
                           ex_k=args.ex_k, max_neurons=maxN)

    best = POLICIES[args.adp](model, train_loader, val_loader, train_cfg, search_cfg)
    print(f"[DOMAINS:{args.domain}] Best val loss = {best:.4f}. Final neurons = {model.neurons()}.")


if __name__ == '__main__':
    main()

# ============================================================
# End of run_adp_diff_domains.py
# ============================================================
