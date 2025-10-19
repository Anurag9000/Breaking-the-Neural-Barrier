import argparse, torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from adp_gtr_graphmae import GraphMAE, Config, node_mask

def train(m, data, cfg):
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    best, bad = 1e9, 0
    for ep in range(1, cfg.epochs+1):
        m.train(); opt.zero_grad()
        mask = node_mask(data.x.size(0), cfg.mask_ratio, data.x.device)
        pred = m(data, mask); tgt = data.x[mask]
        loss = F.mse_loss(pred, tgt); loss.backward(); opt.step()
        if loss.item()<best-1e-4: best,bad=loss.item(),0; torch.save(m.state_dict(), cfg.ckpt)
        else: bad+=1
        if ep%10==0: print(f"[GraphMAE] ep {ep} loss {loss.item():.4f} best {best:.4f}")
        if bad>=cfg.patience: break
    print("Saved:", cfg.ckpt)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=str, default="./data"); args=ap.parse_args()
    ds=Planetoid(args.root,"Cora",transform=NormalizeFeatures()); data=ds[0].to(Config.device)
    m=GraphMAE(ds.num_features).to(Config.device); cfg=Config(); train(m,data,cfg)
