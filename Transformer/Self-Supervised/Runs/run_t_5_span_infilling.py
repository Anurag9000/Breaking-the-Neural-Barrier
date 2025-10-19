import os, json, argparse, random
import numpy as np
import torch
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt
from model_t5_span_infilling import T5SpanInfilling
from model_simcse_transformer import SimpleTokenizer

@dataclass
class Config:
    corpus_path: str = './corpus.txt'
    max_len: int = 256
    dim: int = 512
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: str = './outs_t5_span_infilling'
    seed: int = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sentences(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def make_batches(lines, tok, cfg):
    random.shuffle(lines)
    for i in range(0, len(lines), cfg.batch_size):
        chunk = lines[i:i+cfg.batch_size]
        X = torch.stack([tok.encode(s, cfg.max_len) for s in chunk])
        yield X


def save_plot(curve, title, path, semilogy=False):
    plt.figure(); plt.semilogy(curve) if semilogy else plt.plot(curve)
    plt.title(title); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight'); plt.close()


def train(cfg: Config):
    set_seed(cfg.seed)
    lines = load_sentences(cfg.corpus_path)
    tok = SimpleTokenizer(lines)

    model = T5SpanInfilling(len(tok.itos), cfg.dim, cfg.enc_depth, cfg.dec_depth, cfg.heads, cfg.mlp_ratio, cfg.max_len).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best=float('inf'); best_state=None; patience=cfg.patience
    tr_curve=[]
    for ep in range(cfg.epochs):
        model.train(); tr=0.0; n=0
        for X in make_batches(lines, tok, cfg):
            X=X.to(cfg.device)
            loss = model(X)
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
    save_plot(tr_curve, 'T5 Span-Infilling Train Loss', os.path.join(cfg.out_dir,'train_loss.png'), semilogy=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(cfg.out_dir,'best_state.pt'))
        with open(os.path.join(cfg.out_dir,'summary.json'),'w') as f:
            json.dump({'best_train': float(best)}, f, indent=2)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--corpus_path', type=str, default='./corpus.txt')
    a.add_argument('--max_len', type=int, default=256)
    a.add_argument('--dim', type=int, default=512)
    a.add_argument('--enc_depth', type=int, default=6)
    a.add_argument('--dec_depth', type=int, default=6)
    a.add_argument('--heads', type=int, default=8)
    a.add_argument('--mlp_ratio', type=float, default=4.0)
    a.add_argument('--batch_size', type=int, default=32)
    a.add_argument('--epochs', type=int, default=10)
    a.add_argument('--lr', type=float, default=3e-4)
    a.add_argument('--weight_decay', type=float, default=1e-4)
    a.add_argument('--patience', type=int, default=3)
    a.add_argument('--out_dir', type=str, default='./outs_t5_span_infilling')
    a.add_argument('--seed', type=int, default=42)
    cfg = Config(**vars(a.parse_args()))
    train(cfg)
