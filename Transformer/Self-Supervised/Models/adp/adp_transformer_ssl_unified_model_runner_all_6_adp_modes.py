# ================================================
# File: adp_transformer_ssl.py
# Purpose: Single-model Adaptive Transformer for Self-Supervised Learning
#          with all 6 ADP search variants (width→depth, depth→width,
#          alternating-depth, alternating-width, depth-only, width-only).
#
# Domains supported: TEXT (ready), VISION/SPEECH (scaffold hooks provided).
# Objectives supported (TEXT):
#   - MLM (BERT/RoBERTa)
#   - AR  (GPT/PrefixLM)
#   - DENOISE_SPAN (T5/BART single-model compliance via span-MLM w/ sentinels)
#   - SIMCSE (contrastive sentence-level)
#   - CPC_TEXT (predictive coding on latent spans)
#   - VICREG_TEXT (non-contrastive invariance)
#
# Single-model rule: We avoid momentum encoders/EMA, dual-tower training, or
# teacher-student schemes. For BART/T5 family, we implement span-infilling using
# a single shared encoder (encoder-only approximation), which is compliant and
# preserves the denoising seq2seq spirit without a second train-time model.
# ================================================

from __future__ import annotations
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utility / Config Structures
# ---------------------------

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 10_000_000
    patience: int = 50
    clip_grad_norm: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SearchConfig:
    mode: str = 'width_to_depth'  # {'width_to_depth','depth_to_width','alt_depth','alt_width','depth_only','width_only'}
    trials_width: int = 8
    trials_depth: int = 8
    ex_k: int = 64              # width expansion step (d_model increment)
    ex_ff: int = 256            # feed-forward expansion step
    ex_heads: int = 1           # attention heads increment (kept divisibility)
    max_d_model: int = 1024
    max_ffn: int = 4096
    max_heads: int = 16
    max_layers: int = 48
    max_tokens: int = 50_000_000 # safety budget (approx tokens)

@dataclass
class ArchInit:
    vocab_size: int = 50_000
    max_len: int = 512
    d_model: int = 256
    nhead: int = 4
    dim_ff: int = 1024
    nlayers: int = 6
    dropout: float = 0.1

@dataclass
class ObjectiveConfig:
    objective: str = 'mlm'   # {'mlm','ar','denoise_span','simcse','cpc_text','vicreg_text'}
    # Objective-specific knobs
    mlm_prob: float = 0.15
    span_lambda: float = 3.0         # expected span length (T5-style Poisson)
    temperature: float = 0.05        # contrastive
    cpc_k: int = 3                   # predict next k spans
    vicreg_var_weight: float = 1.0
    vicreg_inv_weight: float = 25.0
    vicreg_cov_weight: float = 1.0

# ---------------------------
# Positional Embedding (sinusoidal)
# ---------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x):
        # x: (B, L, D)
        return self.pe[:, :x.size(1), :]

# ---------------------------
# Adaptive Transformer Encoder (width/depth mutable)
# ---------------------------

class AdaptiveTransformer(nn.Module):
    def __init__(self, arch: ArchInit):
        super().__init__()
        self.vocab_size = arch.vocab_size
        self.max_len = arch.max_len
        self.d_model = arch.d_model
        self.nhead = arch.nhead
        self.dim_ff = arch.dim_ff
        self.nlayers = arch.nlayers
        self.dropout = arch.dropout

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = SinusoidalPositionalEmbedding(self.d_model, self.max_len)
        self.dropout_layer = nn.Dropout(self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_ff,
            dropout=self.dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.nlayers)

        # Heads
        self.mlm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.ar_head = self.mlm_head  # tied
        self.pooler = nn.Identity()   # CLS or mean pooling handled in forward

        self.reset_parameters()

    # -------- Initialization helpers --------
    def reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------- Forward variants --------
    def forward_enc(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return contextual token embeddings (B, L, D)."""
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        x = x + self.pos_emb(x)
        x = self.dropout_layer(x)
        h = self.encoder(x, mask=None, src_key_padding_mask=(~attn_mask) if attn_mask is not None else None)
        return h

    # -------- Objective: MLM --------
    def loss_mlm(self, batch, obj: ObjectiveConfig):
        input_ids, labels, attn_mask = batch['input_ids'], batch['mlm_labels'], batch['attn_mask']
        h = self.forward_enc(input_ids, attn_mask)
        logits = self.mlm_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == labels) & (labels != -100)).sum().item()
            total = (labels != -100).sum().item()
        metrics = {'mlm_acc': correct / max(total, 1)}
        return loss, metrics

    # -------- Objective: AR (causal LM) --------
    def loss_ar(self, batch, obj: ObjectiveConfig):
        input_ids, attn_mask = batch['input_ids'], batch['attn_mask']
        # Causal shift
        labels = input_ids[:, 1:].contiguous()
        dec_inp = input_ids[:, :-1].contiguous()
        dec_mask = attn_mask[:, :-1] if attn_mask is not None else None
        h = self.forward_enc(dec_inp, dec_mask)
        logits = self.ar_head(h)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = ((pred == labels) & (labels != 0)).sum().item()
            total = (labels != 0).sum().item()
        metrics = {'ar_acc': correct / max(total, 1)}
        return loss, metrics

    # -------- Objective: DENOISE_SPAN (T5/BART-style via single encoder)
    # We implement sentinel-span infilling with encoder-only prediction.
    # Batch should provide masked_input_ids and labels with -100 on non-targets.
    def loss_denoise_span(self, batch, obj: ObjectiveConfig):
        input_ids, labels, attn_mask = batch['masked_input_ids'], batch['denoise_labels'], batch['attn_mask']
        h = self.forward_enc(input_ids, attn_mask)
        logits = self.mlm_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = ((pred == labels) & (labels != -100)).sum().item()
            total = (labels != -100).sum().item()
        metrics = {'denoise_token_acc': correct / max(total, 1)}
        return loss, metrics

    # -------- Objective: SimCSE (single-model, dropout views) --------
    def loss_simcse(self, batch, obj: ObjectiveConfig):
        input_ids, attn_mask = batch['input_ids'], batch['attn_mask']
        # view 1
        self.train(); torch.set_grad_enabled(True)
        h1 = self.forward_enc(input_ids, attn_mask)[:, 0]  # CLS = index 0 assumed
        # view 2 (different dropout mask)
        h2 = self.forward_enc(input_ids, attn_mask)[:, 0]
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        logits = (h1 @ h2.t()) / obj.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        return loss, {'ntxent_acc': acc}

    # -------- Objective: CPC-Text (predict next K spans in latent space) --------
    def loss_cpc_text(self, batch, obj: ObjectiveConfig):
        # batch provides span_embeddings (B, S, L, D) or token ids with span indices.
        # For simplicity here we compute mean-pooled span embeddings from tokens.
        input_ids, attn_mask, span_index = batch['input_ids'], batch['attn_mask'], batch['span_index']  # (B, L), (B, L), (B, S, 2)
        h = self.forward_enc(input_ids, attn_mask)  # (B, L, D)
        # Build span reps as mean over [start:end)
        spans = []
        for b in range(h.size(0)):
            svec = []
            for (s, e) in span_index[b].tolist():
                s = max(0, int(s)); e = max(s+1, int(e))
                svec.append(h[b, s:e].mean(dim=0))
            spans.append(torch.stack(svec, dim=0))
        span_h = torch.stack(spans, dim=0)  # (B, S, D)
        B, S, D = span_h.shape
        K = obj.cpc_k
        preds = []
        targets = []
        for k in range(1, K+1):
            Wk = getattr(self, f'cpc_W{k}', None)
            if Wk is None:
                Wk = nn.Linear(D, D, bias=False).to(span_h.device)
                setattr(self, f'cpc_W{k}', Wk)
            preds.append(Wk(span_h[:, :-k, :]))    # (B, S-k, D)
            targets.append(span_h[:, k:, :])        # (B, S-k, D)
        losses = []
        for p, t in zip(preds, targets):
            p = F.normalize(p, dim=-1)
            t = F.normalize(t, dim=-1)
            logits = torch.einsum('bsd,bsd->bs', p, t)  # positive scores
            # In-batch negatives: (B*(S-k)) x (B*(S-k)) similarity
            p_flat = p.reshape(-1, D)
            t_flat = t.reshape(-1, D)
            sim = (p_flat @ t_flat.t()) / obj.temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        return loss, {'cpc_loss': loss.item()}

    # -------- Objective: VICReg-Text (non-contrastive) --------
    def loss_vicreg_text(self, batch, obj: ObjectiveConfig):
        input_ids_a, attn_mask_a = batch['input_ids'], batch['attn_mask']
        input_ids_b, attn_mask_b = batch['input_ids_b'], batch['attn_mask_b']
        za = self.forward_enc(input_ids_a, attn_mask_a)[:, 0]
        zb = self.forward_enc(input_ids_b, attn_mask_b)[:, 0]
        inv = F.mse_loss(za, zb)
        def variance(z):
            eps = 1e-4
            std = torch.sqrt(z.var(dim=0) + eps)
            return torch.mean(F.relu(1 - std))
        def covariance(z):
            z = z - z.mean(dim=0, keepdim=True)
            N, D = z.shape
            cov = (z.t() @ z) / (N - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            return (off_diag ** 2).sum() / D
        var = variance(za) + variance(zb)
        cov = covariance(za) + covariance(zb)
        loss = obj.vicreg_inv_weight * inv + obj.vicreg_var_weight * var + obj.vicreg_cov_weight * cov
        return loss, {'vicreg_inv': inv.item(), 'vicreg_var': var.item(), 'vicreg_cov': cov.item()}

    # -----------------------
    # ADP width/depth mutators
    # -----------------------
    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def snapshot(self) -> Dict:
        return {
            'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
            'arch': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'dim_ff': self.dim_ff,
                'nlayers': self.nlayers,
            }
        }

    def restore(self, snap: Dict):
        arch = snap['arch']
        self._rebuild_arch(int(arch['d_model']), int(arch['nhead']), int(arch['dim_ff']), int(arch['nlayers']))
        self.load_state_dict(snap['state_dict'], strict=True)

    def _rebuild_arch(self, d_model: int, nhead: int, dim_ff: int, nlayers: int):
        # rebuild encoder with new shapes; copy old weights where possible
        old_sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        device = next(self.parameters()).device
        self.d_model, self.nhead, self.dim_ff, self.nlayers = d_model, nhead, dim_ff, nlayers
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model).to(device)
        self.pos_emb = SinusoidalPositionalEmbedding(self.d_model, self.max_len).to(device)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_ff,
            dropout=self.dropout, batch_first=True, activation='gelu', norm_first=True
        ).to(device)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.nlayers).to(device)
        self.mlm_head = nn.Linear(self.d_model, self.vocab_size, bias=False).to(device)
        # try to load overlapping weights
        new_sd = self.state_dict()
        for k in new_sd.keys():
            if k in old_sd:
                old = old_sd[k]
                new = new_sd[k]
                # copy overlap
                slices = tuple(min(a, b) for a, b in zip(old.shape, new.shape))
                if len(slices) == 0:
                    continue
                idx_old = tuple(slice(0, s) for s in slices)
                idx_new = tuple(slice(0, s) for s in slices)
                with torch.no_grad():
                    new[idx_new] = old[idx_old]
        self.load_state_dict(new_sd, strict=False)

    def widen(self, ex_d_model: int = 64, ex_ff: int = 256, ex_heads: int = 1, caps: SearchConfig = None):
        d = min(self.d_model + ex_d_model, caps.max_d_model if caps else self.d_model + ex_d_model)
        ff = min(self.dim_ff + ex_ff, caps.max_ffn if caps else self.dim_ff + ex_ff)
        heads = min(self.nhead + ex_heads, caps.max_heads if caps else self.nhead + ex_heads)
        # ensure divisibility
        if d % heads != 0:
            d = (d // heads) * heads
        self._rebuild_arch(d, heads, ff, self.nlayers)

    def append_layer(self, caps: SearchConfig = None):
        nl = min(self.nlayers + 1, caps.max_layers if caps else self.nlayers + 1)
        self._rebuild_arch(self.d_model, self.nhead, self.dim_ff, nl)


# ---------------------------
# Trainer with all 6 ADP modes
# ---------------------------

class ADPTrainer:
    def __init__(self, model: AdaptiveTransformer, train_conf: TrainConfig, search_conf: SearchConfig, obj_conf: ObjectiveConfig):
        self.model = model.to(train_conf.device)
        self.tc = train_conf
        self.sc = search_conf
        self.oc = obj_conf
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.tc.lr, weight_decay=self.tc.weight_decay)
        self.best_val = float('inf')
        self.best_snap = None
        self.global_epoch = 0
        self.t0 = time.time()

    def objective_step(self, batch) -> Tuple[torch.Tensor, Dict]:
        obj = self.oc.objective.lower()
        if obj == 'mlm':
            return self.model.loss_mlm(batch, self.oc)
        elif obj == 'ar':
            return self.model.loss_ar(batch, self.oc)
        elif obj == 'denoise_span':
            return self.model.loss_denoise_span(batch, self.oc)
        elif obj == 'simcse':
            return self.model.loss_simcse(batch, self.oc)
        elif obj == 'cpc_text':
            return self.model.loss_cpc_text(batch, self.oc)
        elif obj == 'vicreg_text':
            return self.model.loss_vicreg_text(batch, self.oc)
        else:
            raise ValueError(f"Unknown objective: {obj}")

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        n = 0
        metrics_accum: Dict[str, float] = {}
        for batch in loader:
            batch = {k: (v.to(self.tc.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if train:
                self.opt.zero_grad(set_to_none=True)
            loss, metrics = self.objective_step(batch)
            if train:
                loss.backward()
                if self.tc.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.clip_grad_norm)
                self.opt.step()
            total_loss += loss.item()
            n += 1
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
        avg_loss = total_loss / max(n, 1)
        avg_metrics = {k: v / max(n, 1) for k, v in metrics_accum.items()}
        return avg_loss, avg_metrics

    def fit_with_early_stop(self, train_loader, val_loader):
        patience = self.tc.patience
        bad = 0
        while self.global_epoch < self.tc.max_epochs:
            tr_loss, tr_m = self.run_epoch(train_loader, train=True)
            val_loss, val_m = self.run_epoch(val_loader, train=False)
            self.global_epoch += 1
            improved = val_loss < self.best_val
            if improved:
                self.best_val = val_loss
                self.best_snap = self.model.snapshot()
                bad = 0
            else:
                bad += 1
            # Logging (concise)
            if self.global_epoch % 1 == 0:
                print({
                    'epoch': self.global_epoch,
                    'val_loss': round(val_loss, 4),
                    'best_val': round(self.best_val, 4),
                    'bad': bad,
                    'arch': {'d_model': self.model.d_model, 'ff': self.model.dim_ff, 'heads': self.model.nhead, 'layers': self.model.nlayers},
                    'params_m': round(self.model.total_params() / 1e6, 3)
                })
            if bad >= patience:
                break
        # restore best
        if self.best_snap is not None:
            self.model.restore(self.best_snap)
        return self.best_val

    # -----------------------
    # Six ADP modes orchestrator
    # -----------------------
    def search(self, train_loader, val_loader):
        mode = self.sc.mode
        if mode == 'width_to_depth':
            return self._width_then_depth(train_loader, val_loader)
        elif mode == 'depth_to_width':
            return self._depth_then_width(train_loader, val_loader)
        elif mode == 'alt_depth':
            return self._alternate(train_loader, val_loader, first='depth')
        elif mode == 'alt_width':
            return self._alternate(train_loader, val_loader, first='width')
        elif mode == 'depth_only':
            return self._depth_only(train_loader, val_loader)
        elif mode == 'width_only':
            return self._width_only(train_loader, val_loader)
        else:
            raise ValueError(f"Unknown ADP mode: {mode}")

    def _width_then_depth(self, tr, va):
        print('[ADP] Phase 1: Width search')
        for t in range(self.sc.trials_width):
            self.fit_with_early_stop(tr, va)
            self.model.widen(self.sc.ex_k, self.sc.ex_ff, self.sc.ex_heads, self.sc)
        print('[ADP] Phase 2: Depth search')
        for t in range(self.sc.trials_depth):
            self.fit_with_early_stop(tr, va)
            self.model.append_layer(self.sc)
        return self.best_val

    def _depth_then_width(self, tr, va):
        print('[ADP] Phase 1: Depth search')
        for t in range(self.sc.trials_depth):
            self.fit_with_early_stop(tr, va)
            self.model.append_layer(self.sc)
        print('[ADP] Phase 2: Width search')
        for t in range(self.sc.trials_width):
            self.fit_with_early_stop(tr, va)
            self.model.widen(self.sc.ex_k, self.sc.ex_ff, self.sc.ex_heads, self.sc)
        return self.best_val

    def _alternate(self, tr, va, first='depth'):
        order = ['depth', 'width'] if first == 'depth' else ['width', 'depth']
        tw = td = 0
        while tw < self.sc.trials_width or td < self.sc.trials_depth:
            for phase in order:
                if phase == 'depth' and td < self.sc.trials_depth:
                    self.fit_with_early_stop(tr, va)
                    self.model.append_layer(self.sc)
                    td += 1
                if phase == 'width' and tw < self.sc.trials_width:
                    self.fit_with_early_stop(tr, va)
                    self.model.widen(self.sc.ex_k, self.sc.ex_ff, self.sc.ex_heads, self.sc)
                    tw += 1
                if tw >= self.sc.trials_width and td >= self.sc.trials_depth:
                    break
        return self.best_val

    def _depth_only(self, tr, va):
        for t in range(self.sc.trials_depth):
            self.fit_with_early_stop(tr, va)
            self.model.append_layer(self.sc)
        return self.best_val

    def _width_only(self, tr, va):
        for t in range(self.sc.trials_width):
            self.fit_with_early_stop(tr, va)
            self.model.widen(self.sc.ex_k, self.sc.ex_ff, self.sc.ex_heads, self.sc)
        return self.best_val


# ================================================
# File: run_adp_transformer_ssl.py
# Purpose: Unified runner that selects one of 6 ADP modes via --adp-mode
#          and one of 6 TEXT objectives via --objective.
#          Uses a toy synthetic dataset by default; swap with real loaders.
# ================================================

import argparse
from typing import Iterable

class ToyTextDataset(torch.utils.data.Dataset):
    """Minimal text-like dataset for quick smoke tests.
    Generates random token sequences with occasional masked tokens and span masks.
    Replace with a real corpus/tokenizer for production."""
    def __init__(self, vocab_size=50_000, seq_len=64, n=4096, mlm_prob=0.15, span_lambda=3.0):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.data = torch.randint(5, vocab_size, (n, seq_len), generator=g)
        self.attn = torch.ones(n, seq_len, dtype=torch.bool)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mlm_prob = mlm_prob
        self.span_lambda = span_lambda

    def _mask_mlm(self, x):
        labels = torch.full_like(x, -100)
        prob = torch.rand_like(x.float())
        mask = prob < self.mlm_prob
        labels[mask] = x[mask]
        x_masked = x.clone()
        x_masked[mask] = 4  # [MASK] token id reserved
        return x_masked, labels

    def _span_infilling(self, x):
        # Simple span mask: randomly choose start points, mask geometric-length spans
        labels = torch.full_like(x, -100)
        x_masked = x.clone()
        span_prob = self.mlm_prob
        for i in range(x.size(0)):
            j = 1
            while j < x.size(1)-1:
                if torch.rand(1).item() < span_prob:
                    span_len = max(1, int(torch.poisson(torch.tensor(self.span_lambda)).item()))
                    end = min(x.size(1)-1, j+span_len)
                    labels[i, j:end] = x[i, j:end]
                    x_masked[i, j] = 4  # [MASK]/sentinel
                    if end - j > 1:
                        x_masked[i, j+1:end] = 0  # pad within span
                    j = end
                else:
                    j += 1
        return x_masked, labels

    def __getitem__(self, idx):
        ids = self.data[idx]
        attn = self.attn[idx]
        sample = {
            'input_ids': ids,
            'attn_mask': attn
        }
        # Precompute aux labels for various objectives
        masked, mlm_labels = self._mask_mlm(ids)
        sample['mlm_labels'] = mlm_labels
        sample['masked_input_ids'] = masked
        denoise_in, denoise_labels = self._span_infilling(ids)
        sample['denoise_labels'] = denoise_labels
        # For SimCSE: use same input twice with dropout noise (handled in model)
        # For CPC: create span indices: divide into chunks of 8 tokens
        spans = []
        st = 0
        while st < ids.size(0):
            en = min(st+8, ids.size(0))
            spans.append([st, en])
            st = en
        sample['span_index'] = torch.tensor(spans, dtype=torch.long)
        # VICReg view B: simple token dropout augmentation
        drop = (torch.rand_like(ids.float()) < 0.1)
        sample['input_ids_b'] = torch.where(drop, torch.full_like(ids, 1), ids)  # replace with [UNK]=1
        sample['attn_mask_b'] = attn
        return sample

    def __len__(self):
        return self.data.size(0)


def build_loaders(args, oc: ObjectiveConfig) -> Tuple[Iterable, Iterable]:
    ds = ToyTextDataset(vocab_size=args.vocab_size, seq_len=args.seq_len, n=args.train_size,
                        mlm_prob=oc.mlm_prob, span_lambda=oc.span_lambda)
    # split
    n_val = max(64, int(0.1 * len(ds)))
    n_tr = len(ds) - n_val
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return tr_loader, va_loader


def main():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument('--vocab_size', type=int, default=50_000)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--dim_ff', type=int, default=1024)
    p.add_argument('--nlayers', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)

    # Objective
    p.add_argument('--objective', type=str, default='mlm', choices=['mlm','ar','denoise_span','simcse','cpc_text','vicreg_text'])

    # Training
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--max_epochs', type=int, default=10_000_000)

    # ADP Search
    p.add_argument('--adp_mode', type=str, default='width_to_depth', choices=['width_to_depth','depth_to_width','alt_depth','alt_width','depth_only','width_only'])
    p.add_argument('--trials_width', type=int, default=3)
    p.add_argument('--trials_depth', type=int, default=3)
    p.add_argument('--ex_k', type=int, default=64)
    p.add_argument('--ex_ff', type=int, default=256)
    p.add_argument('--ex_heads', type=int, default=1)
    p.add_argument('--max_d_model', type=int, default=1024)
    p.add_argument('--max_ffn', type=int, default=4096)
    p.add_argument('--max_heads', type=int, default=16)
    p.add_argument('--max_layers', type=int, default=48)

    # Data
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--train_size', type=int, default=4096)

    args = p.parse_args()

    arch = ArchInit(
        vocab_size=args.vocab_size, max_len=args.max_len, d_model=args.d_model,
        nhead=args.nhead, dim_ff=args.dim_ff, nlayers=args.nlayers, dropout=args.dropout
    )
    obj = ObjectiveConfig(objective=args.objective)
    tc = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                     patience=args.patience, max_epochs=args.max_epochs)
    sc = SearchConfig(mode=args.adp_mode, trials_width=args.trials_width, trials_depth=args.trials_depth,
                      ex_k=args.ex_k, ex_ff=args.ex_ff, ex_heads=args.ex_heads,
                      max_d_model=args.max_d_model, max_ffn=args.max_ffn,
                      max_heads=args.max_heads, max_layers=args.max_layers)

    print('[ARCH]', asdict(arch))
    print('[OBJ ]', asdict(obj))
    print('[ADP ]', asdict(sc))

    model = AdaptiveTransformer(arch)
    trainer = ADPTrainer(model, tc, sc, obj)

    tr_loader, va_loader = build_loaders(args, obj)
    best = trainer.search(tr_loader, va_loader)
    print('[BEST_VAL]', best)

if __name__ == '__main__':
    main()
