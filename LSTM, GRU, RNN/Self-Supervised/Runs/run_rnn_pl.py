import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from rnn_pl import PLGRU

class ToyLabeled(Dataset):
    def __init__(self, n=20000, T=64, D=16, num_classes=5, seed=42):
        rng=np.random.RandomState(seed)
        self.X=[]; self.y=[]
        for _ in range(n):
            c=rng.randint(0,num_classes)
            x=rng.randn(T,D).astype(np.float32)
            # class-dependent drift
            x+= (c+1)*0.02*np.arange(T)[:,None]
            self.X.append(x); self.y.append(c)
        self.X=np.stack(self.X,0); self.y=np.array(self.y,np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class ToyUnlabeled(Dataset):
    def __init__(self, n=40000, T=64, D=16, seed=123):
        rng=np.random.RandomState(seed)
        self.X=[]
        for _ in range(n):
            x=rng.randn(T,D).astype(np.float32)
            x=np.cumsum(x,axis=0)*0.03
            self.X.append(x)
        self.X=np.stack(self.X,0)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return torch.from_numpy(self.X[i])

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta; self.best=float('inf'); self.count=0
    def step(self,v):
        if v<self.best-self.min_delta: self.best=v; self.count=0; return True
        self.count+=1; return False
    def should_stop(self): return self.count>=self.patience


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs_sup',type=int,default=20)
    ap.add_argument('--epochs_pl',type=int,default=40)
    ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--hidden',type=int,default=256)
    ap.add_argument('--layers',type=int,default=1)
    ap.add_argument('--classes',type=int,default=5)
    ap.add_argument('--T',type=int,default=64)
    ap.add_argument('--D',type=int,default=16)
    ap.add_argument('--n_lab',type=int,default=20000)
    ap.add_argument('--n_unlab',type=int,default=40000)
    ap.add_argument('--val_split',type=float,default=0.1)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--save',type=str,default='pl_gru_best.pt')
    ap.add_argument('--pl_conf',type=float,default=0.8)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    lab=ToyLabeled(args.n_lab,args.T,args.D,args.classes,args.seed)
    n_val=int(len(lab)*args.val_split); n_tr=len(lab)-n_val
    tr_lab,va_lab=random_split(lab,[n_tr,n_val],generator=torch.Generator().manual_seed(args.seed))
    unlab=ToyUnlabeled(args.n_unlab,args.T,args.D,args.seed+1)

    trL=DataLoader(tr_lab,batch_size=args.batch,shuffle=True,drop_last=True)
    vaL=DataLoader(va_lab,batch_size=args.batch,shuffle=False,drop_last=False)
    UL=DataLoader(unlab,batch_size=args.batch,shuffle=True,drop_last=True)

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=PLGRU(args.D,args.hidden,args.layers,args.classes).to(dev)

    opt=torch.optim.AdamW(net.parameters(),lr=args.lr)
    es=EarlyStopper(args.patience,1e-4)

    # supervised warmup
    for ep in range(1,args.epochs_sup+1):
        net.train(); trl=0.0
        for x,y in trL:
            x=x.to(dev); y=y.to(dev)
            logit=net(x)
            loss=torch.nn.functional.cross_entropy(logit,y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            trl+=loss.item()*x.size(0)
        trl/=len(trL.dataset)
        # quick val
        net.eval(); val=0.0
        with torch.no_grad():
            for x,y in vaL:
                x=x.to(dev); y=y.to(dev)
                logit=net(x)
                loss=torch.nn.functional.cross_entropy(logit,y)
                val+=loss.item()*x.size(0)
        val/=len(vaL.dataset)
        es.step(val)
        print(f'[Warmup] Epoch {ep:03d} | train {trl:.6f} | val {val:.6f}')

    # pseudo-label phase (same model generates and trains on its PLs)
    for ep in range(1,args.epochs_pl+1):
        net.train(); trl=0.0
        # supervised minibatches
        for x,y in trL:
            x=x.to(dev); y=y.to(dev)
            logit=net(x)
            loss=torch.nn.functional.cross_entropy(logit,y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            trl+=loss.item()*x.size(0)
        # unlabeled with PLs
        for x in UL:
            x=x.to(dev)
            with torch.no_grad():
                logits=net(x)
                prob=torch.softmax(logits,dim=-1)
                conf, pseudo=prob.max(dim=-1)
                mask=conf>=args.pl_conf
            if mask.any():
                logits=net(x[mask])
                loss=torch.nn.functional.cross_entropy(logits,pseudo[mask])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        # quick val
        net.eval(); val=0.0
        with torch.no_grad():
            for x,y in vaL:
                x=x.to(dev); y=y.to(dev)
                logit=net(x)
                loss=torch.nn.functional.cross_entropy(logit,y)
                val+=loss.item()*x.size(0)
        val/=len(vaL.dataset)
        print(f'[PL] Epoch {ep:03d} | train {trl/ max(1,len(trL.dataset)):.6f} | val {val:.6f}')

    torch.save(net.state_dict(),args.save)
    print(f'Saved best model to {args.save}')

if __name__=='__main__':
    main()
