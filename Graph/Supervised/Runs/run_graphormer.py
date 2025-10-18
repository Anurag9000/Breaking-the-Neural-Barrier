import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

from graphormer_model import GraphTransformer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, data, split='val'):
    model.eval()
    mask = getattr(data, f'{split}_mask')
    with torch.no_grad():
        deg = degree(data.edge_index[0], data.num_nodes).long().to(data.x.device)
        out = model(data.x, data.edge_index, edge_attr=None, deg=deg)
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        pred = out.argmax(dim=-1)
        acc = (pred[mask] == data.y[mask]).float().mean().item()
    return loss, acc


def train_node_level(args):
    dataset = Planetoid(root=os.path.join(args.data_root, args.dataset), name=args.dataset,
                        transform=NormalizeFeatures())
    data = dataset[0].to(args.device)

    model = GraphTransformer(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        use_batchnorm=args.bn,
        use_degree_bias=not args.no_deg_bias,
        max_degree=args.max_degree
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    best_state = None
    patience = 0
    epochs, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        model.train(); opt.zero_grad()
        deg = degree(data.edge_index[0], data.num_nodes).long().to(data.x.device)
        out = model(data.x, data.edge_index, edge_attr=None, deg=deg)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"GraphTransformer_{args.dataset}.pth"))
    plt.figure(); plt.semilogy(epochs, val_losses)
    plt.xlabel('epoch'); plt.ylabel('val_loss (log)'); plt.title('GraphTransformer val loss')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"GraphTransformer_{args.dataset}_val_loss.png"))
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'PubMed'])
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='runs_graphormer')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--bn', action='store_true')
    p.add_argument('--no-deg-bias', dest='no_deg_bias', action='store_true')
    p.add_argument('--max-degree', dest='max_degree', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=2000)
    p.add_argument('--log-interval', type=int, default=10)

    args = p.parse_args(); set_seed(args.seed); train_node_level(args)

if __name__ == '__main__':
    main()
