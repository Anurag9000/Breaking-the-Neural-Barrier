# =============================
# File: adp_transformer_speech.py  (MODEL)
# Single-encoder Speech SSL (Wav2Vec2 / HuBERT / WavLM) + 6 ADP policies
# =============================

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Simple Conv feature extractor (raw waveform -> frame features)
# -----------------
class FeatExtractor(nn.Module):
    def __init__(self, in_ch=1, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=10, stride=5, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(128, feat_dim, kernel_size=4, stride=2, padding=1), nn.GELU(),
        )
    def forward(self, wav):
        # wav: (B, T) -> (B, F, L)
        x = wav.unsqueeze(1)
        return self.net(x)

# -----------------
# Transformer encoder block
# -----------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model*mlp_ratio)), nn.GELU(), nn.Linear(int(d_model*mlp_ratio), d_model)
        )
    def forward(self, x):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------
# Config
# -----------------
@dataclass
class SpeechCfg:
    sample_rate: int = 16000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0

    feat_dim: int = 128
    variant: str = "wav2vec2"  # {wav2vec2, hubert, wavlm}
    mask_prob: float = 0.065
    mask_length: int = 10

    # HuBERT/WavLM pseudo-label head
    codebook_size: int = 100

    # ADP caps
    max_width: int = 1024
    max_depth: int = 48

class AdaptiveSpeechSSL(nn.Module):
    def __init__(self, cfg: SpeechCfg):
        super().__init__()
        self.cfg = cfg
        self.feat = FeatExtractor(1, cfg.feat_dim)
        self.proj = nn.Linear(cfg.feat_dim, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, 4096, cfg.d_model))  # big enough max len
        self.blocks = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)

        # heads
        self.mlm_head = nn.Linear(cfg.d_model, cfg.feat_dim)              # wav2vec2-style feature regression
        self.codebook = nn.Parameter(torch.randn(cfg.codebook_size, cfg.d_model) * 0.02)  # HuBERT/WavLM codebook centers
        self.cls_head = nn.Linear(cfg.d_model, cfg.codebook_size)         # logits over clusters

        nn.init.trunc_normal_(self.pos, std=0.02)

    # ---- inspectors
    @property
    def width(self): return self.cfg.d_model
    @property
    def depth(self): return self.cfg.n_layers
    @property
    def total_neurons(self): return self.cfg.d_model * (2*self.cfg.n_layers)

    # ---- ADP ops
    def widen(self, ex_k: int):
        new_d = min(self.cfg.d_model + ex_k, self.cfg.max_width)
        if new_d == self.cfg.d_model: return False
        self.proj = nn.Linear(self.cfg.feat_dim, new_d)
        self.pos = nn.Parameter(torch.zeros(1, self.pos.size(1), new_d))
        self.blocks = nn.ModuleList([EncoderBlock(new_d, max(1,new_d//32), self.cfg.mlp_ratio) for _ in range(self.cfg.n_layers)])
        self.ln = nn.LayerNorm(new_d)
        self.mlm_head = nn.Linear(new_d, self.cfg.feat_dim)
        self.codebook = nn.Parameter(torch.randn(self.cfg.codebook_size, new_d) * 0.02)
        self.cls_head = nn.Linear(new_d, self.cfg.codebook_size)
        self.cfg.d_model = new_d
        return True

    def deepen(self):
        if self.cfg.n_layers >= self.cfg.max_depth: return False
        self.blocks.append(EncoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//32), self.cfg.mlp_ratio))
        self.cfg.n_layers += 1
        return True

    # ---- masking over time steps
    def _compute_mask(self, L: int, device: torch.device):
        mask = torch.zeros(L, dtype=torch.bool, device=device)
        num_mask = int(self.cfg.mask_prob * L / self.cfg.mask_length)
        for _ in range(num_mask):
            start = torch.randint(0, max(1, L - self.cfg.mask_length + 1), (1,), device=device).item()
            mask[start:start+self.cfg.mask_length] = True
        return mask

    # ---- forward backbone
    def _encode(self, wav: torch.Tensor):
        # wav: (B, T)
        x = self.feat(wav)           # (B, F, L)
        x_t = x.transpose(1, 2)      # (B, L, F)
        B, L, Fdim = x_t.shape
        h = self.proj(x_t) + self.pos[:, :L]
        for blk in self.blocks: h = blk(h)
        h = self.ln(h)               # (B, L, D)
        return h, x_t                # tokens + target features

    # ---- objectives
    def forward_wav2vec2(self, wav: torch.Tensor):
        h, feats = self._encode(wav)
        B, L, D = h.shape
        # mask positions
        loss = 0.0
        for b in range(B):
            m = self._compute_mask(L, wav.device)
            pred = self.mlm_head(h[b])
            loss += F.mse_loss(pred[m], feats[b][m]) if m.any() else pred.sum()*0
        return loss / B, {}

    @torch.no_grad()
    def _closest_code(self, h: torch.Tensor):
        # h: (B,L,D); codebook: (K,D)
        cb = F.normalize(self.codebook, dim=-1)
        q = F.normalize(h, dim=-1)
        # cosine distance argmax
        sim = torch.einsum('bld,kd->blk', q, cb)
        idx = sim.argmax(dim=-1)  # (B,L)
        return idx

    def forward_hubert(self, wav: torch.Tensor):
        h, _ = self._encode(wav)
        idx = self._closest_code(h).detach()
        logits = self.cls_head(h)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
        return loss, {}

    def forward_wavlm(self, wav: torch.Tensor):
        # multi-task: predict masked features + cluster ids
        h, feats = self._encode(wav)
        B, L, D = h.shape
        loss_rec = 0.0
        loss_cls = 0.0
        idx = self._closest_code(h).detach()
        for b in range(B):
            m = self._compute_mask(L, wav.device)
            pred = self.mlm_head(h[b])
            loss_rec += F.mse_loss(pred[m], feats[b][m]) if m.any() else pred.sum()*0
        logits = self.cls_head(h)
        loss_cls = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
        return (loss_rec/B + loss_cls), {}

    def forward(self, wav: torch.Tensor):
        v = self.cfg.variant
        if v == 'wav2vec2': return self.forward_wav2vec2(wav)
        if v == 'hubert': return self.forward_hubert(wav)
        if v == 'wavlm': return self.forward_wavlm(wav)
        raise ValueError(v)

# -----------------
# ADP search (6 policies)
# -----------------
@dataclass
class SearchCfg:
    algo: str = 'wd'
    ex_k: int = 32
    delta: float = 1e-3
    trials_width: int = 2
    trials_depth: int = 2
    max_neurons: int = 8_000_000

@torch.no_grad()
def _clone_state(model: nn.Module):
    return {k:v.clone() for k,v in model.state_dict().items()}

def _load_state(model: nn.Module, st: Dict[str,torch.Tensor]):
    model.load_state_dict(st, strict=False)


def train_inner(model: AdaptiveSpeechSSL, wav: torch.Tensor, steps: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best=float('inf'); best_state=_clone_state(model)
    for _ in range(steps):
        loss,_ = model(wav)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        v=float(loss.item())
        if v<best: best=v; best_state=_clone_state(model)
    _load_state(model,best_state)
    return best


def adp_search(model: AdaptiveSpeechSSL, wav: torch.Tensor, s: SearchCfg, lr: float):
    def accept(base,new): return new < (base - s.delta)
    base = train_inner(model, wav, 5, lr)
    wf=df=0
    def width_series():
        nonlocal base, wf
        ok=False; wf=0
        while wf < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, wav, 5, lr)
            if accept(base,new): base=new; ok=True; wf=0
            else: wf+=1
        return ok
    def depth_series():
        nonlocal base, df
        ok=False; df=0
        while df < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, wav, 5, lr)
            if accept(base,new): base=new; ok=True; df=0
            else: df+=1
        return ok
    if s.algo=='depth_only':
        while df < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, wav, 5, lr)
            if accept(base,new): base=new; df=0
            else: df+=1
        return {"best": base}
    if s.algo=='width_only':
        while wf < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, wav, 5, lr)
            if accept(base,new): base=new; wf=0
            else: wf+=1
        return {"best": base}
    if s.algo=='wd':
        if width_series(): depth_series(); return {"best": base}
        return {"best": base}
    if s.algo=='dw':
        if depth_series(): width_series(); return {"best": base}
        return {"best": base}
    if s.algo=='alt_d':
        while True:
            d = depth_series(); w = width_series()
            if not d and not w: break
        return {"best": base}
    if s.algo=='alt_w':
        while True:
            w = width_series(); d = depth_series()
            if not w and not d: break
        return {"best": base}
    raise ValueError(s.algo)
