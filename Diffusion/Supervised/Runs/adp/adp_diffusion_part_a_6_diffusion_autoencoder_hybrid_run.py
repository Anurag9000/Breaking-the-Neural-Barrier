# ============================================================
# File: run_adp_diff_diffae.py  (RUN)
# Runner for single-model Diffusion–Autoencoder hybrid with 6 ADP policies
# ============================================================

import argparse

import torch
import torchvision as tv
import torchvision.transforms as T

from adp_diff_diffae import DiffAESingleModel, TrainCfg, SearchCfg, POLICIES


def make_loaders(data_root: str, img_size: int, batch: int, val_split: float = 0.2):
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

    n = len(ds_train_full)
    n_val = int(n * val_split)
    idx = torch.randperm(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train_full, train_idx.tolist()),
                                               batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_val_full, val_idx.tolist()),
                                               batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser(description='Diffusion–Autoencoder Hybrid (single model) with ADP')

    p.add_argument('--adp', type=str, default='depth2width',
                   choices=['depth2width','width2depth','alt_depth','alt_width','depth_only','width_only'])

    # Data
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--img-size', type=int, default=32)
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
    p.add_argument('--latent-ch', type=int, default=4)
    p.add_argument('--latent-levels', type=int, default=2)
    p.add_argument('--widths', type=int, nargs='+', default=[32,64,96])
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--lambda-recon', type=float, default=1.0)

    args = p.parse_args()

    train_loader, val_loader = make_loaders(args.data, args.img_size, args.batch, args.val_split)

    model = DiffAESingleModel(img_ch=3, latent_ch=args.latent_ch, latent_levels=args.latent_levels,
                              widths=args.widths, T=args.T, lambda_recon=args.lambda_recon)

    train_cfg = TrainCfg(lr=args.lr, max_epochs=args.max_epochs, es_patience=args.es_patience,
                         grad_clip=args.grad_clip, device=args.device)
    maxN = None if args.max_neurons == 0 else args.max_neurons
    search_cfg = SearchCfg(delta=args.delta, trials_width=args.trials_width, trials_depth=args.trials_depth,
                           ex_k=args.ex_k, max_neurons=maxN)

    best = POLICIES[args.adp](model, train_loader, val_loader, train_cfg, search_cfg)
    print(f"[DIFF-AE] Finished. Best val loss = {best:.4f}. Final neurons = {model.neurons()}.")


if __name__ == '__main__':
    main()

# ============================================================
# End of run_adp_diff_diffae.py
# ============================================================
