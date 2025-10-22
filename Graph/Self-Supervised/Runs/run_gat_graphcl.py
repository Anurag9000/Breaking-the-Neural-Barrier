import argparse, torch, os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from adp_gat_graphcl import GraphCL_GAT, Config, set_seed

def train_loop(model, data, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, bad = 1e9, 0
    for ep in range(1, cfg.epochs+1):
        model.train(); opt.zero_grad()
        loss = model.contrastive_loss(data)
        loss.backward(); opt.step()
        if loss.item() < best - 1e-4:
            best, bad = loss.item(), 0
            torch.save(model.state_dict(), cfg.ckpt)
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"[GraphCL-GAT] epoch {ep} loss {loss.item():.4f} best {best:.4f} (pat:{bad}/{cfg.patience})")
        if bad >= cfg.patience: break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data")
    args = ap.parse_args()
    set_seed(42)
    ds = Planetoid(args.root, "Cora", transform=NormalizeFeatures())
    data = ds[0].to(Config.device)
    model = GraphCL_GAT(ds.num_features, Config()).to(Config.device)
    train_loop(model, data, model.cfg)
    print("Saved:", model.cfg.ckpt)

if __name__ == "__main__":
    main()
