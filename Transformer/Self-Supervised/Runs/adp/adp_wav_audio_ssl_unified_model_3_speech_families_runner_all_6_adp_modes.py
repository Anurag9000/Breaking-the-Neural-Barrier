# ================================================
# File: adp_wav_ssl.py
# Purpose: Single-model Adaptive Audio (Speech) SSL backbone supporting
#          ALL 6 ADP modes and 3 SSL families:
#          {wav2vec2 (CPC), HuBERT (masked unit pred), WavLM (unified masked)}
#
# Single-model policy:
# - No momentum/EMA, no dual encoders. All views come from the SAME encoder.
# - HuBERT uses a fixed, non-learned unitizer proxy (random proj + argmax) or
#   offline codes; here we simulate with a frozen projection for toy runs.
# - wav2vec2 implemented as InfoNCE on predicted future latent frames.
# - WavLM as masked-frame regression with denoising noise.
# ================================================

from __future__ import annotations
import math, time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Configs
# ---------------------------

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    patience: int = 50
    max_epochs: int = 10_000_000
    clip_grad_norm: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SearchConfig:
    mode: str = 'width_to_depth'
    trials_width: int = 4
    trials_depth: int = 4
    ex_embed: int = 128
    ex_ff: int = 256
    ex_heads: int = 1
    max_embed: int = 1024
    max_ff: int = 4096
    max_heads: int = 16
    max_layers: int = 24

@dataclass
class ArchInit:
    sample_rate: int = 16000
    win_ms: int = 25
    hop_ms: int = 10
    in_chans: int = 1
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    ff_dim: int = 1024
    conv_downsample: int = 4    # total stride from conv frontend

@dataclass
class ObjectiveConfig:
    objective: str = 'wav2vec2'  # {'wav2vec2','hubert','wavlm'}
    temperature: float = 0.1     # InfoNCE
    future_k: int = 3            # predict next k steps (CPC)
    mask_prob: float = 0.065
    mask_span: int = 10
    unit_codebook: int = 8192    # HuBERT codebook size (proxy)

# ---------------------------
# Audio frontend & encoder
# ---------------------------

class ConvFeatureExtractor(nn.Module):
    """Simple 1D Conv stack to produce frame-level features."""
    def __init__(self, in_ch=1, embed=256, downsample=4):
        super().__init__()
        # three convs strided by 2 -> total stride 8 (approx); keep simple
        hid = embed // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=10, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(hid, hid, kernel_size=8, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(hid, embed, kernel_size=4, stride=max(1, downsample//4), padding=1), nn.GELU(),
        )
    def forward(self, x):  # x: (B,1,T)
        return self.conv(x)  # (B, D, T')

class AdaptiveAudioEncoder(nn.Module):
    def __init__(self, arch: ArchInit):
        super().__init__()
        self.embed_dim = arch.embed_dim
        self.ff_dim = arch.ff_dim
        self.depth = arch.depth
        self.num_heads = arch.num_heads
        self.frontend = ConvFeatureExtractor(arch.in_chans, arch.embed_dim, arch.conv_downsample)
        enc_layer = nn.TransformerEncoderLayer(d_model=arch.embed_dim, nhead=arch.num_heads,
                                               dim_feedforward=arch.ff_dim, dropout=0.1,
                                               activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=arch.depth)
        self.norm = nn.LayerNorm(arch.embed_dim)
        # heads
        self.cpc_proj = nn.Linear(arch.embed_dim, arch.embed_dim, bias=False)
        self.pred_proj = nn.ModuleDict({})  # future-step specific predictors
        self.unit_proj = None  # HuBERT classifier (lazy)
        self.recon = nn.Linear(arch.embed_dim, arch.embed_dim, bias=False)  # WavLM masked recon in latent space

    # ------- ADP plumbing -------
    def snapshot(self) -> Dict:
        return {
            'state_dict': {k: v.detach().cpu() for k,v in self.state_dict().items()},
            'arch': {'embed_dim': self.embed_dim, 'ff_dim': self.ff_dim, 'depth': self.depth, 'num_heads': self.num_heads}
        }
    def restore(self, snap: Dict):
        a = snap['arch']
        self._rebuild(int(a['embed_dim']), int(a['ff_dim']), int(a['depth']), int(a['num_heads']))
        self.load_state_dict(snap['state_dict'], strict=True)

    def _rebuild(self, embed, ff, depth, heads):
        old = {k: v.detach().cpu() for k,v in self.state_dict().items()}
        device = next(self.parameters()).device
        self.embed_dim, self.ff_dim, self.depth, self.num_heads = embed, ff, depth, heads
        self.frontend = ConvFeatureExtractor(1, embed).to(device)
        layer = nn.TransformerEncoderLayer(d_model=embed, nhead=heads, dim_feedforward=ff, dropout=0.1,
                                           activation='gelu', batch_first=True, norm_first=True).to(device)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth).to(device)
        self.norm = nn.LayerNorm(embed).to(device)
        self.cpc_proj = nn.Linear(embed, embed, bias=False).to(device)
        self.recon = nn.Linear(embed, embed, bias=False).to(device)
        new = self.state_dict()
        for k in new.keys():
            if k in old:
                o, n = old[k], new[k]
                sl = tuple(min(a,b) for a,b in zip(o.shape, n.shape))
                idx = tuple(slice(0,s) for s in sl)
                with torch.no_grad(): n[idx] = o[idx]
        self.load_state_dict(new, strict=False)

    def widen(self, sc: SearchConfig):
        d = min(self.embed_dim + sc.ex_embed, sc.max_embed)
        ff = min(self.ff_dim + sc.ex_ff, sc.max_ff)
        heads = min(self.num_heads + sc.ex_heads, sc.max_heads)
        if d % heads != 0: d = (d // heads) * heads
        self._rebuild(d, ff, self.depth, heads)

    def append_layer(self, sc: SearchConfig):
        depth = min(self.depth + 1, sc.max_layers)
        self._rebuild(self.embed_dim, self.ff_dim, depth, self.num_heads)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------- core encode -------
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B,1,T) -> feats: (B,T',D)
        z = self.frontend(wav)                 # (B,D,T')
        z = z.transpose(1,2)                   # (B,T',D)
        h = self.encoder(z)                    # (B,T',D)
        h = self.norm(h)
        return h

    # ------- objectives -------
    def loss_wav2vec2(self, batch, oc: ObjectiveConfig):
        wav = batch['wav']
        h = self.encode(wav)                   # (B,T,D)
        B,T,D = h.shape
        losses = []
        for k in range(1, oc.future_k+1):
            # predictor per horizon
            Wk = self.pred_proj.get(str(k))
            if Wk is None:
                Wk = nn.Linear(D, D, bias=False).to(h.device)
                self.pred_proj[str(k)] = Wk
            p = Wk(h[:, :-k, :])               # (B,T-k,D)
            t = h[:, k:, :].detach()           # (B,T-k,D)
            # InfoNCE with in-batch negatives
            p = F.normalize(p, dim=-1); t = F.normalize(t, dim=-1)
            p_flat = p.reshape(-1, D)
            t_flat = t.reshape(-1, D)
            logits = (p_flat @ t_flat.t()) / oc.temperature
            labels = torch.arange(logits.size(0), device=h.device)
            loss = F.cross_entropy(logits, labels)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        return loss, {'cpc_loss': loss.item()}

    def loss_hubert(self, batch, oc: ObjectiveConfig):
        wav = batch['wav']
        h = self.encode(wav)  # (B,T,D)
        B,T,D = h.shape
        # build mask
        num_mask = max(1, int(oc.mask_prob * T))
        perm = torch.rand(B, T, device=h.device).argsort(1)
        mask_idx = perm[:, :num_mask]
        keep_idx = perm[:, num_mask:]
        # unitizer proxy (frozen random codebook)
        codebook = getattr(self, 'unit_codebook', None)
        if codebook is None:
            self.unit_codebook = nn.Parameter(torch.randn(oc.unit_codebook, D) * 0.02, requires_grad=False).to(h.device)
            codebook = self.unit_codebook
        with torch.no_grad():
            logits_u = h @ codebook.t()        # (B,T,K)
            target_ids = logits_u.argmax(-1)   # (B,T)
        # classifier (lazy)
        if self.unit_proj is None:
            self.unit_proj = nn.Linear(D, oc.unit_codebook).to(h.device)
        masked_h = h.gather(1, mask_idx.unsqueeze(-1).expand(-1,-1,D))
        logits = self.unit_proj(masked_h)
        tgt = target_ids.gather(1, mask_idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return loss, {'hubert_ce': loss.item()}

    def loss_wavlm(self, batch, oc: ObjectiveConfig):
        wav = batch['wav']
        h = self.encode(wav)  # (B,T,D)
        B,T,D = h.shape
        num_mask = max(1, int(oc.mask_prob * T))
        perm = torch.rand(B, T, device=h.device).argsort(1)
        mask_idx = perm[:, :num_mask]
        keep_idx = perm[:, num_mask:]
        noisy = h + 0.05*torch.randn_like(h)
        pred = self.recon(noisy.gather(1, keep_idx.unsqueeze(-1).expand(-1,-1,D)))
        target = h.gather(1, mask_idx.unsqueeze(-1).expand(-1,-1,D))
        loss = F.mse_loss(pred, target)
        return loss, {'wavlm_mse': loss.item()}

    def objective_step(self, batch, oc: ObjectiveConfig):
        o = oc.objective.lower()
        if o == 'wav2vec2': return self.loss_wav2vec2(batch, oc)
        if o == 'hubert': return self.loss_hubert(batch, oc)
        if o == 'wavlm': return self.loss_wavlm(batch, oc)
        raise ValueError(f'Unknown objective {o}')

# ---------------------------
# Trainer with 6 ADP modes
# ---------------------------

class ADPTrainer:
    def __init__(self, model: AdaptiveAudioEncoder, tc: TrainConfig, sc: SearchConfig, oc: ObjectiveConfig):
        self.model = model.to(tc.device)
        self.tc, self.sc, self.oc = tc, sc, oc
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        self.best_val = float('inf')
        self.best_snap = None
        self.global_epoch = 0

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        tot, n = 0.0, 0
        meters: Dict[str, float] = {}
        for batch in loader:
            batch = {k:(v.to(self.tc.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
            if train: self.opt.zero_grad(set_to_none=True)
            loss, m = self.model.objective_step(batch, self.oc)
            if train:
                loss.backward()
                if self.tc.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.clip_grad_norm)
                self.opt.step()
            tot += float(loss.item()); n += 1
            for k,v in m.items(): meters[k] = meters.get(k,0.0) + float(v)
        return tot/max(n,1), {k:v/max(n,1) for k,v in meters.items()}

    def fit_es(self, tr, va):
        bad = 0
        while self.global_epoch < self.tc.max_epochs:
            trl,_ = self.run_epoch(tr, True)
            val,mm = self.run_epoch(va, False)
            self.global_epoch += 1
            if val < self.best_val:
                self.best_val = val; self.best_snap = self.model.snapshot(); bad = 0
            else:
                bad += 1
            print({'epoch':self.global_epoch,'val':round(val,4),'best':round(self.best_val,4),'bad':bad,
                   'arch':{'embed':self.model.embed_dim,'ff':self.model.ff_dim,'heads':self.model.num_heads,'depth':self.model.depth},
                   'params_m': round(self.model.total_params()/1e6,3)})
            if bad >= self.tc.patience: break
        if self.best_snap is not None:
            self.model.restore(self.best_snap)
        return self.best_val

    def search(self, tr, va):
        m = self.sc.mode
        if m == 'width_to_depth':
            return self._width_then_depth(tr, va)
        if m == 'depth_to_width':
            return self._depth_then_width(tr, va)
        if m == 'alt_depth':
            return self._alternate(tr, va, first='depth')
        if m == 'alt_width':
            return self._alternate(tr, va, first='width')
        if m == 'depth_only':
            return self._depth_only(tr, va)
        if m == 'width_only':
            return self._width_only(tr, va)
        raise ValueError('unknown adp mode')

    def _width_then_depth(self, tr, va):
        print('[ADP] Phase: Width then Depth')
        for _ in range(self.sc.trials_width):
            self.fit_es(tr, va); self.model.widen(self.sc)
        for _ in range(self.sc.trials_depth):
            self.fit_es(tr, va); self.model.append_layer(self.sc)
        return self.best_val

    def _depth_then_width(self, tr, va):
        print('[ADP] Phase: Depth then Width')
        for _ in range(self.sc.trials_depth):
            self.fit_es(tr, va); self.model.append_layer(self.sc)
        for _ in range(self.sc.trials_width):
            self.fit_es(tr, va); self.model.widen(self.sc)
        return self.best_val

    def _alternate(self, tr, va, first='depth'):
        order = ['depth','width'] if first=='depth' else ['width','depth']
        td = tw = 0
        while td < self.sc.trials_depth or tw < self.sc.trials_width:
            for ph in order:
                if ph=='depth' and td < self.sc.trials_depth:
                    self.fit_es(tr, va); self.model.append_layer(self.sc); td += 1
                if ph=='width' and tw < self.sc.trials_width:
                    self.fit_es(tr, va); self.model.widen(self.sc); tw += 1
                if td>=self.sc.trials_depth and tw>=self.sc.trials_width: break
        return self.best_val

    def _depth_only(self, tr, va):
        for _ in range(self.sc.trials_depth):
            self.fit_es(tr, va); self.model.append_layer(self.sc)
        return self.best_val

    def _width_only(self, tr, va):
        for _ in range(self.sc.trials_width):
            self.fit_es(tr, va); self.model.widen(self.sc)
        return self.best_val

# ================================================
# File: run_adp_wav_ssl.py
# Purpose: Unified runner for 3 speech SSL objectives & 6 ADP modes.
# ================================================

import argparse
from typing import Tuple

class ToyWaveDataset(torch.utils.data.Dataset):
    """Synthetic 1D audio dataset with simple augmentations.
    Replace with LibriSpeech/LS-960 or VoxCeleb loaders for real runs."""
    def __init__(self, n=4096, length=16000):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        # mix sinusoids + noise
        t = torch.linspace(0, 1, length)
        self.wav = []
        for i in range(n):
            f1, f2 = torch.randint(50, 400, (1,), generator=g).item(), torch.randint(400, 2000, (1,), generator=g).item()
            x = 0.5*torch.sin(2*math.pi*f1*t) + 0.5*torch.sin(2*math.pi*f2*t)
            x += 0.05*torch.randn_like(x)
            self.wav.append(x.unsqueeze(0))
        self.wav = torch.stack(self.wav, dim=0)  # (N,1,T)
        self.n = n
    def _aug(self, x):
        x = x + 0.02*torch.randn_like(x)
        # time masking
        T = x.size(-1)
        if T > 1000:
            s = torch.randint(0, T-400, (1,)).item(); e = s + torch.randint(50, 400, (1,)).item()
            x[..., s:e] = 0
        return x
    def __getitem__(self, idx):
        w = self.wav[idx]
        return {'wav': w, 'wav1': self._aug(w.clone()), 'wav2': self._aug(w.clone())}
    def __len__(self):
        return self.n


def build_loaders(args, oc: ObjectiveConfig):
    ds = ToyWaveDataset(n=args.train_size, length=args.length)
    n_val = max(64, int(0.1 * len(ds)))
    n_tr = len(ds) - n_val
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return tr_loader, va_loader


def main():
    p = argparse.ArgumentParser()
    # Arch
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--ff_dim', type=int, default=1024)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=4)

    # Objective
    p.add_argument('--objective', type=str, default='wav2vec2', choices=['wav2vec2','hubert','wavlm'])

    # Train
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--max_epochs', type=int, default=10_000_000)

    # ADP
    p.add_argument('--adp_mode', type=str, default='width_to_depth', choices=['width_to_depth','depth_to_width','alt_depth','alt_width','depth_only','width_only'])
    p.add_argument('--trials_width', type=int, default=2)
    p.add_argument('--trials_depth', type=int, default=2)
    p.add_argument('--ex_embed', type=int, default=128)
    p.add_argument('--ex_ff', type=int, default=256)
    p.add_argument('--ex_heads', type=int, default=1)
    p.add_argument('--max_embed', type=int, default=1024)
    p.add_argument('--max_ff', type=int, default=4096)
    p.add_argument('--max_heads', type=int, default=16)
    p.add_argument('--max_layers', type=int, default=24)

    # Data
    p.add_argument('--train_size', type=int, default=4096)
    p.add_argument('--length', type=int, default=16000)

    args = p.parse_args()

    arch = ArchInit(embed_dim=args.embed_dim, ff_dim=args.ff_dim, depth=args.depth, num_heads=args.num_heads)
    oc = ObjectiveConfig(objective=args.objective)
    tc = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, patience=args.patience, max_epochs=args.max_epochs)
    sc = SearchConfig(mode=args.adp_mode, trials_width=args.trials_width, trials_depth=args.trials_depth,
                      ex_embed=args.ex_embed, ex_ff=args.ex_ff, ex_heads=args.ex_heads,
                      max_embed=args.max_embed, max_ff=args.max_ff, max_heads=args.max_heads, max_layers=args.max_layers)

    print('[ARCH]', asdict(arch))
    print('[OBJ ]', asdict(oc))
    print('[ADP ]', asdict(sc))

    model = AdaptiveAudioEncoder(arch)
    trainer = ADPTrainer(model, tc, sc, oc)
    tr, va = build_loaders(args, oc)
    best = trainer.search(tr, va)
    print('[BEST_VAL]', best)

if __name__ == '__main__':
    main()
