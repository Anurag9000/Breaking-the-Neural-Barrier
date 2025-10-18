import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from edgeconv_model import EdgeConvNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, ratios=(0.8, 0.1, 0.1), seed=42):
    torch.manual_seed(seed)
    n = len(dataset)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+n_val]
    test_idx = perm[n_train+n_val:]
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_corr, total = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        pred = out.argmax(dim=-1)
        total_loss += loss.item() * data.num_graphs
        total_corr += (pred == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, total_corr / total


def train_graph_level(args):
    dataset = TUDataset(root=os.path.join(args.data_root, args.dataset), name=args.dataset)
    in_channels = dataset.num_features
    out_channels = dataset.num_classes
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = args.device
    model = EdgeConvNet(in_channels, args.hidden, out_channels, num_layers=args.layers,
                        dropout=args.dropout, use_batchnorm=not args.no_bn).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    best_state = None
    patience = 0
    epochs, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        epochs.append(epoch); val_losses.append(val_loss)

        if val_loss < best_val - args.delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            print('Early stopping.'); break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"EdgeConv_{args.dataset}.pth"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='PROTEINS')
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='runs_edgeconv')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--no-bn', action='store_true')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=1000)
    p.add_argument('--log-interval', type=int, default=5)

    args = p.parse_args(); set_seed(args.seed); train_graph_level(args)

if __name__ == '__main__':
    main()
