import math
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from rnn_dsae import GRUDenoisingAutoencoder

class ToySineDataset(Dataset):
    def __init__(self, n_samples=10000, T=64, D=8, noise_std=0.05, seed=42):
        rng = np.random.RandomState(seed)
        self.clean = []
        self.noisy = []
        for _ in range(n_samples):
            freq = rng.uniform(0.5, 2.0, size=D)
            phase = rng.uniform(0, 2*math.pi, size=D)
            t = np.linspace(0, 1, T)
            seq = np.stack([np.sin(2*math.pi*freq[d]*t + phase[d]) for d in range(D)], axis=1)
            clean = seq.astype(np.float32)
            corrupt = clean + rng.randn(T, D).astype(np.float32) * noise_std
            self.clean.append(clean)
            self.noisy.append(corrupt)
        self.clean = np.stack(self.clean, axis=0)
        self.noisy = np.stack(self.noisy, axis=0)

    def __len__(self):
        return self.clean.shape[0]

    def __getitem__(self, idx):
        c = torch.from_numpy(self.clean[idx])
        n = torch.from_numpy(self.noisy[idx])
        return c, n

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
    def step(self, val):
        if val < self.best - self.min_delta:
            self.best = val
            self.count = 0
            return True
        else:
            self.count += 1
            return False
    def should_stop(self):
        return self.count >= self.patience


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch):
    clean = torch.stack([b[0] for b in batch], dim=0)
    noisy = torch.stack([b[1] for b in batch], dim=0)
    return clean, noisy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--data_T', type=int, default=64)
    p.add_argument('--data_D', type=int, default=16)
    p.add_argument('--n_samples', type=int, default=20000)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', type=str, default='dsae_gru_best.pt')
    args = p.parse_args()

    set_seed(args.seed)

    ds = ToySineDataset(n_samples=args.n_samples, T=args.data_T, D=args.data_D, noise_std=0.1, seed=args.seed)
    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUDenoisingAutoencoder(input_dim=args.data_D, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, num_layers=args.layers, bidirectional=args.bidirectional).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    es = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    best = None
    for epoch in range(1, args.epochs+1):
        model.train()
        tr = 0.0
        for clean, noisy in train_loader:
            clean = clean.to(device)
            noisy = noisy.to(device)
            z, h0 = model(clean, noisy)
            dec_inp = torch.zeros_like(clean)
            dec_inp[:,1:,:] = clean[:,:-1,:]
            rec = model.decode(h0, dec_inp)
            loss = loss_fn(rec, clean)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * clean.size(0)
        tr /= len(train_loader.dataset)

        # val
        model.eval()
        va = 0.0
        with torch.no_grad():
            for clean, noisy in val_loader:
                clean = clean.to(device)
                noisy = noisy.to(device)
                z, h0 = model(clean, noisy)
                dec_inp = torch.zeros_like(clean)
                dec_inp[:,1:,:] = clean[:,:-1,:]
                rec = model.decode(h0, dec_inp)
                loss = loss_fn(rec, clean)
                va += loss.item() * clean.size(0)
        va /= len(val_loader.dataset)

        improved = es.step(va)
        if improved:
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | train {tr:.6f} | val {va:.6f} | best {es.best:.6f}")
        if es.should_stop():
            print('Early stopping.')
            break

    if best is not None:
        model.load_state_dict(best)
    torch.save(model.state_dict(), args.save)
    print(f"Saved best model to {args.save}")

if __name__ == '__main__':
    main()
