# =============================
# File: adp_transformer_text.py  (MODEL)
# Single-model, self-supervised Transformer with 6 ADP search policies
# Part A(2): Text denoising / masked-token objectives
# Variants: mlm_bert, bart, t5_span, mass, prefixlm
# =============================

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Resizing helpers (copy-overlap)
# -----------------

def _overlap(dst: torch.nn.Parameter, src: torch.nn.Parameter):
    with torch.no_grad():
        d = dst.data; s = src.data
        slices = tuple(slice(0, min(d.size(i), s.size(i))) for i in range(d.dim()))
        d.zero_(); d[slices] = s[slices]

def _resize_linear(m: nn.Linear, in_f: int, out_f: int):
    n = nn.Linear(in_f, out_f, bias=(m.bias is not None))
    _overlap(n.weight, m.weight)
    if m.bias is not None: _overlap(n.bias, m.bias)
    return n

def _resize_layernorm(ln: nn.LayerNorm, hidden: int):
    n = nn.LayerNorm(hidden, eps=ln.eps, elementwise_affine=ln.elementwise_affine)
    if ln.elementwise_affine:
        _overlap(n.weight, ln.weight); _overlap(n.bias, ln.bias)
    return n

# -----------------
# Attention blocks
# -----------------

class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d); self.mlp = MLP(d, int(d*mlp_ratio))
    def forward(self, x):
        h = self.ln1(x); a,_ = self.attn(h,h,h); x = x + a
        x = x + self.mlp(self.ln2(x)); return x

class DecoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d); self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d); self.mlp = MLP(d, int(d*mlp_ratio))
    def forward(self, x, mem):
        h = self.ln1(x); a,_ = self.self_attn(h,h,h, attn_mask=self._causal_mask(x))
        x = x + a
        h = self.ln2(x); a,_ = self.cross_attn(h, mem, mem)
        x = x + a
        x = x + self.mlp(self.ln3(x)); return x
    def _causal_mask(self, x):
        L = x.size(1)
        return torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

# -----------------
# Config & Model
# -----------------

@dataclass
class ADPTextCfg:
    vocab_size: int = 32000
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    mask_id: int = 3
    max_len: int = 256

    d_model: int = 512
    n_layers_enc: int = 6
    n_layers_dec: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0

    variant: str = "mlm_bert"   # {mlm_bert,bart,t5_span,mass,prefixlm}

    # caps for ADP
    max_width: int = 2048
    max_depth: int = 48

class AdaptiveTransformerText(nn.Module):
    def __init__(self, cfg: ADPTextCfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.encoder = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers_enc)])
        self.enc_ln = nn.LayerNorm(cfg.d_model)

        # decoder is used for bart/t5/mass/prefixlm
        self.decoder = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers_dec)])
        self.dec_ln = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    # ---- inspectors
    @property
    def width(self): return self.cfg.d_model
    @property
    def depth(self):
        # use encoder depth if encoder-only; otherwise sum is informative
        if self.cfg.variant == "mlm_bert":
            return self.cfg.n_layers_enc
        return self.cfg.n_layers_enc + self.cfg.n_layers_dec
    @property
    def total_neurons(self):
        return self.cfg.d_model * (2*self.cfg.n_layers_enc + 3*self.cfg.n_layers_dec)

    # ---- ADP ops
    def widen(self, ex_k: int):
        new_d = min(self.cfg.d_model + ex_k, self.cfg.max_width)
        if new_d == self.cfg.d_model: return False
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, new_d, _weight=None)
        self.pos_emb = nn.Embedding(self.cfg.max_len, new_d, _weight=None)
        # rebuild blocks
        self.encoder = nn.ModuleList([EncoderBlock(new_d, max(1,new_d//64), self.cfg.mlp_ratio) for _ in range(self.cfg.n_layers_enc)])
        self.enc_ln = nn.LayerNorm(new_d)
        self.decoder = nn.ModuleList([DecoderBlock(new_d, max(1,new_d//64), self.cfg.mlp_ratio) for _ in range(self.cfg.n_layers_dec)])
        self.dec_ln = nn.LayerNorm(new_d)
        self.lm_head = nn.Linear(new_d, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.cfg.d_model = new_d
        return True

    def deepen(self):
        if self.depth >= self.cfg.max_depth: return False
        if self.cfg.variant == "mlm_bert":
            self.encoder.append(EncoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//64), self.cfg.mlp_ratio))
            self.cfg.n_layers_enc += 1
        else:
            # alternate: append to encoder, then decoder
            if self.cfg.n_layers_enc <= self.cfg.n_layers_dec:
                self.encoder.append(EncoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//64), self.cfg.mlp_ratio))
                self.cfg.n_layers_enc += 1
            else:
                self.decoder.append(DecoderBlock(self.cfg.d_model, max(1,self.cfg.d_model//64), self.cfg.mlp_ratio))
                self.cfg.n_layers_dec += 1
        return True

    # ---- enc/dec passes
    def encode(self, x_ids):
        B,L = x_ids.shape
        pos = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B,L)
        x = self.tok_emb(x_ids) + self.pos_emb(pos)
        for blk in self.encoder: x = blk(x)
        return self.enc_ln(x)

    def decode(self, y_ids, mem):
        B,L = y_ids.shape
        pos = torch.arange(L, device=y_ids.device).unsqueeze(0).expand(B,L)
        y = self.tok_emb(y_ids) + self.pos_emb(pos)
        for blk in self.decoder: y = blk(y, mem)
        return self.dec_ln(y)

    # -----------------
    # Objectives
    # -----------------
    def forward_mlm(self, x_ids, mlm_prob=0.15):
        B,L = x_ids.shape
        mask = (torch.rand_like(x_ids.float()) < mlm_prob) & (x_ids!=self.cfg.pad_id)
        inputs = x_ids.clone()
        inputs[mask] = self.cfg.mask_id
        enc = self.encode(inputs)
        logits = self.lm_head(enc)
        loss = F.cross_entropy(logits[mask], x_ids[mask]) if mask.any() else logits.sum()*0
        return loss, {"masked": mask}

    def _noise_bart(self, x_ids):
        # simple token masking noise
        noisy = x_ids.clone()
        prob = torch.rand_like(noisy.float())
        noisy[(prob<0.15) & (noisy!=self.cfg.pad_id)] = self.cfg.mask_id
        return noisy

    def forward_bart(self, x_ids):
        noisy = self._noise_bart(x_ids)
        mem = self.encode(noisy)
        # teacher forcing: shift right with BOS
        y_in = torch.cat([torch.full_like(x_ids[:, :1], self.cfg.bos_id), x_ids[:, :-1]], dim=1)
        dec = self.decode(y_in, mem)
        logits = self.lm_head(dec)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_ids.reshape(-1), ignore_index=self.cfg.pad_id)
        return loss, {}

    def _make_t5_span(self, x_ids, span_prob=0.15, mean_span=3):
        B,L = x_ids.shape
        encoder = x_ids.clone()
        targets = []
        for b in range(B):
            l = L
            i = 0
            enc_line = []
            tgt_line = [self.cfg.bos_id]
            sentinel = self.cfg.vocab_size-1  # last id as first sentinel
            while i<l:
                if random.random()<span_prob:
                    span = max(1, int(random.expovariate(1/mean_span)))
                    end = min(l, i+span)
                    enc_line.append(sentinel)
                    tgt_line.append(sentinel)
                    tgt_line.extend(x_ids[b, i:end].tolist())
                    sentinel -= 1
                    i = end
                else:
                    enc_line.append(int(x_ids[b, i]))
                    i+=1
            tgt_line.append(self.cfg.eos_id)
            # pad to L
            enc_line = enc_line[:L] + [self.cfg.pad_id]*max(0, L-len(enc_line))
            tgt_line = tgt_line[:L] + [self.cfg.pad_id]*max(0, L-len(tgt_line))
            encoder[b] = torch.tensor(enc_line, device=x_ids.device)
            targets.append(torch.tensor(tgt_line, device=x_ids.device))
        targets = torch.stack(targets, dim=0)
        return encoder, targets

    def forward_t5(self, x_ids):
        enc_in, tgt = self._make_t5_span(x_ids)
        mem = self.encode(enc_in)
        y_in = torch.cat([tgt[:, :1], tgt[:, :-1]], dim=1)
        dec = self.decode(y_in, mem)
        logits = self.lm_head(dec)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=self.cfg.pad_id)
        return loss, {}

    def _make_mass(self, x_ids, span_ratio=0.5):
        B,L = x_ids.shape
        encoder = x_ids.clone()
        target = torch.full_like(x_ids, self.cfg.pad_id)
        for b in range(B):
            span = max(1, int(L*span_ratio/2))
            start = random.randint(0, max(0, L-span))
            end = start+span
            target[b, :span] = x_ids[b, start:end]
            encoder[b, start:end] = self.cfg.mask_id
        # decoder input is BOS + target[:-1]
        y_in = torch.cat([torch.full_like(target[:, :1], self.cfg.bos_id), target[:, :-1]], dim=1)
        return encoder, y_in, target

    def forward_mass(self, x_ids):
        enc_in, y_in, tgt = self._make_mass(x_ids)
        mem = self.encode(enc_in)
        dec = self.decode(y_in, mem)
        logits = self.lm_head(dec)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=self.cfg.pad_id)
        return loss, {}

    def forward_prefixlm(self, x_ids, prefix_ratio=0.3):
        B,L = x_ids.shape
        pref = int(L*prefix_ratio)
        y_in = torch.cat([torch.full_like(x_ids[:, :1], self.cfg.bos_id), x_ids[:, :-1]], dim=1)
        # use decoder-only by letting mem be zeros of length=1
        mem = torch.zeros(B,1,self.cfg.d_model, device=x_ids.device)
        dec = self.decode(y_in, mem)
        logits = self.lm_head(dec)
        loss_mask = torch.zeros_like(x_ids).bool(); loss_mask[:, pref:] = True
        loss = F.cross_entropy(logits[loss_mask], x_ids[loss_mask]) if loss_mask.any() else logits.sum()*0
        return loss, {}

    def forward(self, x_ids):
        v = self.cfg.variant
        if v == "mlm_bert": return self.forward_mlm(x_ids)
        if v == "bart": return self.forward_bart(x_ids)
        if v == "t5_span": return self.forward_t5(x_ids)
        if v == "mass": return self.forward_mass(x_ids)
        if v == "prefixlm": return self.forward_prefixlm(x_ids)
        raise ValueError(v)

# -----------------
# ADP search (6 policies)
# -----------------

@dataclass
class SearchCfg:
    algo: str = "wd"  # {wd,dw,alt_d,alt_w,depth_only,width_only}
    ex_k: int = 64
    delta: float = 1e-3
    trials_width: int = 2
    trials_depth: int = 2
    max_neurons: int = 8_000_000

@torch.no_grad()
def _clone_state(model: nn.Module):
    return {k: v.clone() for k,v in model.state_dict().items()}

def _load_state(model: nn.Module, st: Dict[str,torch.Tensor]):
    model.load_state_dict(st, strict=False)


def train_inner(model: AdaptiveTransformerText, tokens: torch.Tensor, steps: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float('inf'); best_state = _clone_state(model)
    for _ in range(steps):
        loss,_ = model(tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        val = float(loss.item())
        if val < best:
            best = val; best_state = _clone_state(model)
    _load_state(model, best_state)
    return best


def adp_search(model: AdaptiveTransformerText, tokens: torch.Tensor, s: SearchCfg, lr: float):
    def accept(base, new): return new < (base - s.delta)
    base = train_inner(model, tokens, 5, lr)
    width_fails = depth_fails = 0

    def width_series():
        nonlocal base, width_fails
        wins = False; width_fails = 0
        while width_fails < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base, new): base = new; wins=True; width_fails=0
            else: width_fails+=1
        return wins

    def depth_series():
        nonlocal base, depth_fails
        wins = False; depth_fails = 0
        while depth_fails < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base, new): base = new; wins=True; depth_fails=0
            else: depth_fails+=1
        return wins

    if s.algo == 'depth_only':
        while depth_fails < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen(): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base, new): base = new; depth_fails=0
            else: depth_fails+=1
        return {"best": base}

    if s.algo == 'width_only':
        while width_fails < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k): break
            new = train_inner(model, tokens, 5, lr)
            if accept(base, new): base = new; width_fails=0
            else: width_fails+=1
        return {"best": base}

    if s.algo == 'wd':
        if width_series(): depth_series(); return {"best": base}
        return {"best": base}
    if s.algo == 'dw':
        if depth_series(): width_series(); return {"best": base}
        return {"best": base}
    if s.algo == 'alt_d':
        while True:
            d = depth_series(); w = width_series()
            if not d and not w: break
        return {"best": base}
    if s.algo == 'alt_w':
        while True:
            w = width_series(); d = depth_series()
            if not w and not d: break
        return {"best": base}
    raise ValueError(s.algo)
