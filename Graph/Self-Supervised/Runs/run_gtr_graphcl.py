import argparse, torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from adp_gtr_graphcl import GraphCL_GTR, Config

def train(model, data, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, bad = 1e9, 0
    for ep in range(1, cfg.epochs+1):
        model.train(); opt.zero_grad()
        loss = model.contrastive_loss(data)
        loss.backward(); opt.step()
        if loss.item() < best-1e-4:
            best, bad = loss.item(), 0
            torch.save(model.state_dict(), cfg.ckpt)
        else:
            bad += 1
        if bad >= cfg.patience: break
        if ep % 10 == 0: print(f"[GraphCL-GTR] ep {ep} loss {loss.item():.4f} best {best:.4f}")
    print("Saved:", cfg.ckpt)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=str, default="./data"); args = ap.parse_args()
    ds = Planetoid(args.root, "Cora", transform=NormalizeFeatures()); data = ds[0].to(Config.device)
    model = GraphCL_GTR(ds.num_features, Config()).to(Config.device)
    train(model, data, model.cfg)
if __name__ == "__main__": main()
