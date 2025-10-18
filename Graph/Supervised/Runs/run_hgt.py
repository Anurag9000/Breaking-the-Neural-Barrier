import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import IMDB
from torch_geometric.transforms import ToUndirected, AddSelfLoops

from hgt_model import HGTNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, data, target_type, split_mask_name='val_mask'):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, target_type)
        mask = data[target_type][split_mask_name]
        loss = F.cross_entropy(out[mask], data[target_type].y[mask]).item()
        pred = out.argmax(dim=-1)
        acc = (pred[mask] == data[target_type].y[mask]).float().mean().item()
    return loss, acc


def train_imdb(args):
    dataset = IMDB(root=os.path.join(args.data_root, 'IMDB'))
    data = dataset[0]

    data = ToUndirected()(data)
    data = AddSelfLoops()(data)
    device = args.device
    data = data.to(device)

    target_type = 'movie'
    in_channels_dict = {k: data[k].num_features for k in data.node_types}

    model = HGTNet(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_channels=args.hidden,
        out_channels=data[target_type].y.max().item() + 1,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf'); best_state = None; patience = 0
    epochs, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        model.train(); opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, target_type)
        loss = F.cross_entropy(out[data[target_type].train_mask], data[target_type].y[data[target_type].train_mask])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        val_loss, val_acc = evaluate(model, data, target_type, 'val_mask')
        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch}] train={loss.item():.4f} val={val_loss:.4f} acc={val_acc:.4f}")
        epochs.append(epoch); val_losses.append(val_loss)

        if val_loss < best_val - args.delta:
            best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            print('Early stopping.'); break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, data, target_type, 'test_mask')
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"HGT_IMDB.pth"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--out-dir', type=str, default='runs_hgt')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--max-epochs', type=int, default=2000)
    p.add_argument('--log-interval', type=int, default=10)

    args = p.parse_args(); set_seed(args.seed); train_imdb(args)

if __name__ == '__main__':
    main()
