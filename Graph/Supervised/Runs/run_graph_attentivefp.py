import argparse, torch, torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from graph_attentivefp import AttentiveFPNet

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total_loss=0; correct=0; n=0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.edge_attr if hasattr(data,'edge_attr') else None)
        loss = F.cross_entropy(out, data.y)
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(-1)
        correct += (pred == data.y).sum().item()
        n += data.num_graphs
    return total_loss/n, correct/n

def train(args):
    dataset = TUDataset(root=args.data_root, name=args.dataset, use_node_attr=True, transform=OneHotDegree(args.max_degree, cat=True))
    num_classes = dataset.num_classes
    in_dim = dataset.num_features
    # split
    torch.manual_seed(42)
    n = len(dataset); n_train = int(0.8*n); n_val = int(0.1*n); n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = args.device
    model = AttentiveFPNet(in_dim, args.hidden_dim, num_classes, args.num_layers, args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val=1e9; best=None; patience=args.patience
    for epoch in range(1, args.max_epochs+1):
        model.train()
        for data in train_loader:
            data = data.to(device); opt.zero_grad()
            out = model(data.x, data.edge_index, data.batch, data.edge_attr if hasattr(data,'edge_attr') else None)
            loss = F.cross_entropy(out, data.y)
            loss.backward(); opt.step()
        vloss, vacc = evaluate(model, val_loader, device)
        if vloss < best_val - 1e-6:
            best_val = vloss; best = {k:v.cpu() for k,v in model.state_dict().items()}; patience=args.patience
        else:
            patience -= 1
            if patience<=0: break
        if epoch % 10 == 0:
            tloss, tacc = evaluate(model, train_loader, device)
            print(f"{epoch:04d} | train {tacc:.3f} | val {vacc:.3f} | vloss {vloss:.4f}")
    if best is not None:
        model.load_state_dict({k:v.to(device) for k,v in best.items()})
    vloss, vacc = evaluate(model, val_loader, device)
    tloss, tacc = evaluate(model, train_loader, device)
    sloss, sacc = evaluate(model, test_loader, device)
    print(f"Final | train {tacc:.3f} | val {vacc:.3f} | test {sacc:.3f}")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='MUTAG')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--max_degree', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--max_epochs', type=int, default=500)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args(); train(args)
