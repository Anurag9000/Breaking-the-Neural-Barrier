import argparse, torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.optim import AdamW
from graph_transformerconv import TransformerConvNet

@torch.no_grad()
def evaluate(model, data):
    model.eval(); out = model(data.x, data.edge_index)
    pred = out.argmax(-1)
    val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
    tr = (pred[data.train_mask]==data.y[data.train_mask]).float().mean().item()
    va = (pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()
    te = (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
    return val_loss, tr, va, te

def train(args):
    dataset = Planetoid(root=args.data_root, name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(args.device)
    model = TransformerConvNet(dataset.num_features, args.hidden_dim, dataset.num_classes, args.num_layers, args.heads, args.dropout).to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val=float('inf'); best=None; patience=args.patience
    for epoch in range(1, args.max_epochs+1):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        val_loss, tr, va, te = evaluate(model, data)
        if val_loss < best_val - 1e-6:
            best_val = val_loss; best = {k:v.cpu() for k,v in model.state_dict().items()}; patience=args.patience
        else:
            patience -= 1
            if patience<=0: break
        if epoch % 50 == 0:
            print(f"{epoch:04d} | tr {tr:.3f} | va {va:.3f} | te {te:.3f} | vloss {val_loss:.4f}")
    if best is not None:
        model.load_state_dict({k:v.to(args.device) for k,v in best.items()})
    val_loss, tr, va, te = evaluate(model, data)
    print(f"Final | val {val_loss:.4f} | train {tr:.3f} | val {va:.3f} | test {te:.3f}")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Cora', choices=['Cora','CiteSeer','PubMed'])
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--max_epochs', type=int, default=2000)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args(); train(args)
