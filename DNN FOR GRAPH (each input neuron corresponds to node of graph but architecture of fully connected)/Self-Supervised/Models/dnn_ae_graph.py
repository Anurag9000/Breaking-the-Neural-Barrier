
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

def load_planetoid(name="Cora", root="./data"):
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root=root, name=name)
    data = ds[0]
    return (data.x, data.y, data.train_mask, data.val_mask, data.test_mask), ds.num_classes

class DNNNodeAE(nn.Module):
    def __init__(self, num_nodes, hidden=128, depth=2):
        super().__init__()
        self.num_nodes=num_nodes; self.hidden=hidden; self.depth=depth
        self.feature_proj=None; self.decoder_out=None
        self.in_lin=nn.Linear(num_nodes,hidden,bias=False)
        self.hiddens=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(depth-1)])
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.in_lin.weight)
        for l in self.hiddens: nn.init.kaiming_normal_(l.weight)
    def _ensure_feature_proj(self,F): 
        if self.feature_proj is None:
            self.feature_proj=nn.Parameter(torch.randn(F)*0.02)
    def _ensure_decoder(self,F):
        if self.decoder_out is None:
            self.decoder_out=nn.Linear(self.hidden,self.num_nodes*F,bias=True)
            nn.init.zeros_(self.decoder_out.bias)
            nn.init.kaiming_normal_(self.decoder_out.weight)
    def forward(self,X):
        N,F=X.shape; assert N==self.num_nodes
        self._ensure_feature_proj(F); self._ensure_decoder(F)
        z=(X@self.feature_proj).unsqueeze(0)
        z=F.relu(self.in_lin(z))
        for l in self.hiddens: z=F.relu(l(z))
        return self.decoder_out(z).view(N,F)

@dataclass
class TrainCfg:
    lr:float=1e-3; weight_decay:float=5e-4; max_epochs:int=2000; patience:int=100
    grad_clip:float=1.0; device:str="cuda" if torch.cuda.is_available() else "cpu"

def train_autoencoder(model,data,cfg):
    X,y,train_mask,val_mask,test_mask=data
    X=X.to(cfg.device); train_mask,val_mask,test_mask=[m.to(cfg.device) for m in (train_mask,val_mask,test_mask)]
    model=model.to(cfg.device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=float('inf'); best_state=None; pat=cfg.patience
    for e in range(cfg.max_epochs):
        model.train(); opt.zero_grad(); Xh=model(X); loss=F.mse_loss(Xh[train_mask],X[train_mask]); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step()
        model.eval(); 
        with torch.no_grad(): v=F.mse_loss(model(X)[val_mask],X[val_mask]).item()
        if v<best-1e-9: best=v; best_state={k:v.cpu() for k,v in model.state_dict().items()}; pat=cfg.patience
        else: pat-=1
        if pat<=0: break
    if best_state: model.load_state_dict(best_state)
    with torch.no_grad(): test=F.mse_loss(model(X)[test_mask],X[test_mask]).item()
    return best,test

def main():
    import argparse; p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=str,default="Cora")
    p.add_argument("--hidden",type=int,default=128); p.add_argument("--depth",type=int,default=3)
    a=p.parse_args()
    data,_=load_planetoid(a.dataset); X,_,_,_,_=data; N=X.size(0)
    m=DNNNodeAE(N,a.hidden,a.depth); cfg=TrainCfg(); v,t=train_autoencoder(m,data,cfg)
    print(f"[AE] {a.dataset} N={N} H={a.hidden} D={a.depth} Val={v:.6f} Test={t:.6f}")
