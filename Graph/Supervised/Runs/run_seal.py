import argparse
import os
import random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from seal_model import SEALSubgraphGNN, extract_enclosing_subgraph


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_seal_dataset(data, num_pos=1000, num_neg=1000, num_hops=2, device='cpu'):
    pos_edges = data.edge_index.t().tolist()
    random.shuffle(pos_edges)
    pos_edges = pos_edges[:num_pos]

    neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes,
                                       num_neg_samples=num_neg)
    neg_edges = neg_edge_index.t().tolist()

    graphs = []
    labels = []
    for (u, v) in pos_edges:
        nodes, sub_edge_index, x_aug = extract_enclosing_subgraph(data, u, v, num_hops=num_hops)
        graphs.append(Data(x=x_aug, edge_index=sub_edge_index))
        labels.append(1)
    for (u, v) in neg_edges:
        nodes, sub_edge_index, x_aug = extract_enclosing_subgraph(data, u, v, num_hops=num_hops)
        graphs.append(Data(x=x_aug, edge_index=sub_edge_index))
        labels.append(0)

    for i, g in enumerate(graphs):
        g.y = torch.tensor(labels[i]).long()
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)  # single graph per Data
    return graphs


def collate_graphs(graphs, batch_size):
    # PyG DataLoader handles batching Data objects automatically
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    return loader


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_corr, total = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        pred = out.argmax(dim=-1)
        total_loss += loss.item() * batch.num_graphs
        total_corr += (pred == batch.y).sum().item()
        total += batch.num_graphs
    return total_loss / total, total_corr / total


def train_seal(args):
    dataset = Planetoid(root=os.path.join(args.data_root, args.dataset), name=args.dataset,
                        transform=NormalizeFeatures())
    data = dataset[0]

    graphs = build_seal_dataset(data, num_pos=args.num_pos, num_neg=args.num_neg,
                                num_hops=args.hops, device=args.device)

    # Simple train/val/test split over constructed subgraphs
    n = len(graphs); n_train = int(0.8*n); n_val = int(0.1*n)
    train_graphs = graphs[:n_train]; val_graphs = graphs[n_train:n_train+n_val]; test_graphs = graphs[n_train+n_val:]

    train_loader = collate_graphs(train_graphs, args.batch_size)
    val_loader = collate_graphs(val_graphs, args.batch_size)
    test_loader = collate_graphs(test_graphs, args.batch_size)

    device = args.device
    model = SEALSubgraphGNN(in_channels=dataset.num_node_features, hidden_channels=args.hidden,
                            num_layers=args.layers, dropout=args.dropout, use_batchnorm=not args.no_bn).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf'); best_state = None; patience = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val - args.delta:
            best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            print('Early stopping.'); break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"SEAL_{args.dataset}.pth"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Cora', choices=['Cora','Citeseer','PubMed'])
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='runs_seal')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--hops', type=int, default=2)
    p.add_argument('--num-pos', dest='num_pos', type=int, default=2000)
    p.add_argument('--num-neg', dest='num_neg', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--no-bn', action='store_true')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--log-interval', type=int, default=5)

    args = p.parse_args(); set_seed(args.seed); train_seal(args)

if __name__ == '__main__':
    main()
