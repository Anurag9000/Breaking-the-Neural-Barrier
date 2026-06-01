import os, json, argparse, random
import numpy as np
import torch
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt
from model_hubert_single import HuBERTSingle

@dataclass
class Config:
    sample_len: int = 32000
    batch_size: int = 8
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: str = './outs_hubert_single'
    seed: int = 42
    warmup_batches: int = 200


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synth_wavs(sample_len, n=1000):
    for _ in range(n):
        t = torch.linspace(0, 1, sample_len)
        wav = torch.sin(2*torch.pi*(220+random.random()*440)*t) + 0.1*torch.randn_like(t)
        yield wav.unsqueeze(0)


def batch_iter(cfg: Config):
    buf=[]
    for wav in synth_wavs(cfg.sample_len):
        buf.append(wav)
        if len(buf)==cfg.batch_size:
            yield torch.stack(buf,0)
            buf=[]


def save_plot(curve, title, path, semilogy=False):
    plt.figure(); plt.semilogy(curve) if semilogy else plt.plot(curve)
    plt.title(title); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight'); plt.close()


def train(cfg: Config):
    set_seed(cfg.seed)
    model = HuBERTSingle().to(cfg.device)

    # warmup codebook with random features
    feats=[]
    with torch.no_grad():
        for i,b in enumerate(batch_iter(cfg)):
            if i*cfg.batch_size > cfg.warmup_batches: break
            f = model.proj(model.feature(b.to(cfg.device))).reshape(-1, model.proj.out_features)
            feats.append(f)
    feats = torch.cat(feats,0)
    model.init_codebook(feats)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best=float('inf'); best_state=None; patience=cfg.patience
    tr_curve=[]
    for ep in range(cfg.epochs):
        model.train(); tr=0.0; n=0
        for batch in batch_iter(cfg):
            batch = batch.to(cfg.device)
            loss = model(batch)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr += loss.item(); n+=1
        tr = tr/max(n,1); tr_curve.append(tr)
        print(f'Epoch {ep+1}/{cfg.epochs} | train {tr:.4f}')
        if tr < best - 1e-4:
            best=tr; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; patience=cfg.patience
        else:
            patience-=1
            if patience==0:
                print('Early stopping.'); break

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_plot(tr_curve, 'HuBERT-single Train Loss', os.path.join(cfg.out_dir,'train_loss.png'), semilogy=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(cfg.out_dir,'best_state.pt'))
        with open(os.path.join(cfg.out_dir,'summary.json'),'w') as f:
            json.dump({'best_train': float(best)}, f, indent=2)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--sample_len', type=int, default=32000)
    a.add_argument('--batch_size', type=int, default=8)
    a.add_argument('--epochs', type=int, default=10)
    a.add_argument('--lr', type=float, default=3e-4)
    a.add_argument('--weight_decay', type=float, default=1e-4)
    a.add_argument('--patience', type=int, default=3)
    a.add_argument('--out_dir', type=str, default='./outs_hubert_single')
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--warmup_batches', type=int, default=200)
    cfg = Config(**vars(a.parse_args()))
    train(cfg)
