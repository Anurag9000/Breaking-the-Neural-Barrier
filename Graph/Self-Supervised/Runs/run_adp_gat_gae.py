import argparse, torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from adp_gat_gae import VGAE_GAT, Config

def train(m, data, cfg):
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    best, bad = 1e9, 0
    for ep in range(1, cfg.epochs+1):
        m.train(); opt.zero_grad()
        loss = m.loss(data); loss.backward(); opt.step()
        if loss.item()<best-1e-4: best,bad=loss.item(),0; torch.save(m.state_dict(), cfg.ckpt)
        else: bad+=1
        if ep%10==0: print(f"[VGAE-GAT] ep {ep} loss {loss.item():.4f} best {best:.4f}")
        if bad>=cfg.patience: break
    print("Saved:", cfg.ckpt)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=str, default="./data"); args=ap.parse_args()
    ds=Planetoid(args.root,"Cora",transform=NormalizeFeatures()); data=ds[0].to(Config.device)
    m=VGAE_GAT(ds.num_features, Config()).to(Config.device); train(m, data, m.cfg)
