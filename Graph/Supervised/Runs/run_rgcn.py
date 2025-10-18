import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import AIFBDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from rgcn_model import RGCNNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, graph_level=False):
    model.eval()
    total_loss, total_corr, total = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        if graph_level:
            out = model(data.x.float(), data.edge_index, data.edge_type, data.batch)
        else:
            out = model(data.x.float(), data.edge_index, data.edge_type)
        loss = F.cross_entropy(out, data.y)
        pred = out.argmax(dim=-1)
        total_loss += loss.item() * data.num_graphs
        total_corr += (pred == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, total_corr / total


def train_relational(args):
    dataset = AIFBDataset(root=os.path.join(args.data_root, args.dataset))
    data = dataset[0]
    device = args.device
    data = data.to(device)

    model = RGCNNet(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        num_relations=data.num_edge_types,
        num_layers=args.layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
        use_batchnorm=not args.no_bn,
        graph_level=False
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    best_state = None
    patience = 0
    epochs, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x.float(), data.edge_index, data.edge_type)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
        val_pred = out[data.val_mask].argmax(dim=-1)
        val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()

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

    test_loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask]).item()
    test_pred = out[data.test_mask].argmax(dim=-1)
    test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"RGCN_{args.dataset}.pth"))
    plt.figure(); plt.semilogy(epochs, val_losses)
    plt.xlabel('epoch'); plt.ylabel('val_loss (log)'); plt.title('R-GCN val loss')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"RGCN_{args.dataset}_val_loss.png"))
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='AIFB')
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='runs_rgcn')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--num-bases', dest='num_bases', type=int, default=None)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--no-bn', action='store_true')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=2000)
    p.add_argument('--log-interval', type=int, default=10)

    args = p.parse_args(); set_seed(args.seed); train_relational(args)

if __name__ == '__main__':
    main()
