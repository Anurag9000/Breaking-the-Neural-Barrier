import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from rnn_dc import DCGRU

class ToySeq(Dataset):
    def __init__(self, n=60000, T=64, D=16, seed=42):
        rng=np.random.RandomState(seed)
        self.X=[]
        for _ in range(n):
            x=rng.randn(T,D).astype(np.float32)
            x=np.cumsum(x,axis=0)*0.05
            self.X.append(x)
        self.X=np.stack(self.X,0)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return torch.from_numpy(self.X[i])

# very small k-means (L2) for embeddings
@torch.no_grad()
def kmeans(x, k, iters=10):
    # x: (N, D)
    N,D=x.shape
    idx=torch.randperm(N, device=x.device)[:k]
    c=x[idx].clone()  # (k,D)
    for _ in range(iters):
        # assign
        dist=(x.unsqueeze(1)-c.unsqueeze(0)).pow(2).sum(-1)  # (N,k)
        a=dist.argmin(dim=1)
        # update
        for i in range(k):
            sel=(a==i)
            if sel.any():
                c[i]=x[sel].mean(0)
    return a, c

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta; self.best=float('inf'); self.count=0
    def step(self,v):
        if v<self.best-self.min_delta: self.best=v; self.count=0; return True
        self.count+=1; return False
    def should_stop(self): return self.count>=self.patience


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--cluster_epochs',type=int,default=5, help='epochs per clustering round')
    ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--batch',type=int,default=512)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--hidden',type=int,default=256)
    ap.add_argument('--proj',type=int,default=128)
    ap.add_argument('--layers',type=int,default=1)
    ap.add_argument('--clusters',type=int,default=200)
    ap.add_argument('--T',type=int,default=64)
    ap.add_argument('--D',type=int,default=16)
    ap.add_argument('--n',type=int,default=60000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--save',type=str,default='dc_gru_best.pt')
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ds=ToySeq(args.n,args.T,args.D,args.seed)
    loader=DataLoader(ds,batch_size=args.batch,shuffle=True,drop_last=True)

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=DCGRU(args.D,args.hidden,args.proj,args.layers,args.clusters).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr)

    best=None
    for r in range(args.rounds):
        # 1) extract features for all data
        feats=[]
        with torch.no_grad():
            for x in DataLoader(ds,batch_size=args.batch,shuffle=False):
                x=x.to(dev); z=net.features(x); feats.append(z.cpu())
        feats=torch.cat(feats,0)
        # 2) kmeans to produce pseudo labels
        y, _ = kmeans(feats.to(dev), args.clusters, iters=10)
        y=y.cpu()
        # 3) train classifier on pseudo-labels
        es=EarlyStopper(patience=3, min_delta=1e-4)
        for ep in range(1,args.cluster_epochs+1):
            net.train(); trl=0.0
            i0=0
            for x in loader:
                B=x.size(0)
                lab=y[i0:i0+B]; i0+=B
                x=x.to(dev); lab=lab.to(dev)
                logit=net(x)
                loss=torch.nn.functional.cross_entropy(logit,lab)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
                trl+=loss.item()*B
            trl/=len(ds)
            # simple running best on training loss
            es.step(trl)
            print(f'[Round {r+1}/{args.rounds}] Epoch {ep:02d} | train {trl:.6f}')
        best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

    if best is not None: net.load_state_dict(best)
    torch.save(net.state_dict(),args.save)
    print(f'Saved best model to {args.save}')

if __name__=='__main__':
    main()
