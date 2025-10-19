import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from rnn_npair import NPairGRU

class ToyAug(Dataset):
    def __init__(self, n=30000, T=64, D=16, seed=42):
        rng=np.random.RandomState(seed)
        self.X=[]
        for _ in range(n):
            x=rng.randn(T,D).astype(np.float32)
            x=np.cumsum(x,axis=0)*0.05
            self.X.append(x)
        self.X=np.stack(self.X,0)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return torch.from_numpy(self.X[i])

def augment(x,noise_std=0.05,drop_prob=0.1):
    x1=x+noise_std*torch.randn_like(x)
    x2=x+noise_std*torch.randn_like(x)
    return x1,x2

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta; self.best=float('inf'); self.count=0
    def step(self,v):
        if v<self.best-self.min_delta: self.best=v; self.count=0; return True
        self.count+=1; return False
    def should_stop(self): return self.count>=self.patience


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--hidden',type=int,default=256)
    ap.add_argument('--proj',type=int,default=128)
    ap.add_argument('--layers',type=int,default=1)
    ap.add_argument('--T',type=int,default=64)
    ap.add_argument('--D',type=int,default=16)
    ap.add_argument('--n',type=int,default=40000)
    ap.add_argument('--val_split',type=float,default=0.1)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--save',type=str,default='npair_gru_best.pt')
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ds=ToyAug(args.n,args.T,args.D,args.seed)
    n_val=int(len(ds)*args.val_split); n_tr=len(ds)-n_val
    tr_ds,va_ds=random_split(ds,[n_tr,n_val],generator=torch.Generator().manual_seed(args.seed))

    tr=DataLoader(tr_ds,batch_size=args.batch,shuffle=True,drop_last=True)
    va=DataLoader(va_ds,batch_size=args.batch,shuffle=False,drop_last=False)

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=NPairGRU(args.D,args.hidden,args.proj,args.layers).to(dev)

    opt=torch.optim.AdamW(net.parameters(),lr=args.lr)
    es=EarlyStopper(args.patience,1e-4)

    def nce(z1,z2,temp=0.2):
        z=torch.cat([z1,z2],0)
        sim=z@z.t()/temp
        B=z1.size(0)
        mask=torch.eye(2*B,device=z.device).bool()
        sim.masked_fill_(mask,-1e9)
        labels=torch.cat([torch.arange(B,2*B), torch.arange(0,B)]).to(z.device)
        return torch.nn.functional.cross_entropy(sim,labels)

    best=None
    for ep in range(1,args.epochs+1):
        net.train(); trl=0.0
        for x in tr:
            x=x.to(dev)
            x1,x2=augment(x)
            z1=net.encode(x1); z2=net.encode(x2)
            loss=nce(z1,z2,0.2)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            trl+=loss.item()*x.size(0)
        trl/=len(tr.dataset)

        net.eval(); val=0.0
        with torch.no_grad():
            for x in va:
                x=x.to(dev)
                x1,x2=augment(x)
                z1=net.encode(x1); z2=net.encode(x2)
                loss=nce(z1,z2,0.2)
                val+=loss.item()*x.size(0)
        val/=len(va.dataset)

        if es.step(val): best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        print(f'Epoch {ep:03d} | train {trl:.6f} | val {val:.6f} | best {es.best:.6f}')
        if es.should_stop(): print('Early stopping.'); break

    if best is not None: net.load_state_dict(best)
    torch.save(net.state_dict(),args.save)
    print(f'Saved best model to {args.save}')

if __name__=='__main__':
    main()
