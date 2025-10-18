import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt

from gin_model import GIN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, data, split='val'):
    model.eval()
    mask = getattr(data, f'{split}_mask')
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        pred = out.argmax(dim=-1)
        acc = (pred[mask] == data.y[mask]).float().mean().item()
    return loss, acc


def train_node_level(args):
    dataset = Planetoid(root=os.path.join(args.data_root, args.dataset), name=args.dataset,
                        transform=NormalizeFeatures())
    data = dataset[0].to(args.device)

    model = GIN(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
        use_batchnorm=not args.no_bn,
        train_eps=args.train_eps
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    best_state = None
    patience = 0
    epochs, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        val_loss, val_acc = evaluate(model, data, 'val')
        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch}] train={loss.item():.4f} val={val_loss:.4f} acc={val_acc:.4f}")
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

    test_loss, test_acc = evaluate(model, data, 'test')
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"GIN_{args.dataset}.pth"))
    plt.figure(); plt.semilogy(epochs, val_losses)
    plt.xlabel('epoch'); plt.ylabel('val_loss (log)'); plt.title('GIN val loss')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"GIN_{args.dataset}_val_loss.png"))
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'PubMed'])
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='runs_gin')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--no-bn', action='store_true')
    p.add_argument('--train-eps', dest='train_eps', action='store_true')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=2000)
    p.add_argument('--log-interval', type=int, default=10)

    args = p.parse_args()
    set_seed(args.seed)
    train_node_level(args)

if __name__ == '__main__':
    main()
