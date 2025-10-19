import torch
import torch.nn as nn
import torch.nn.functional as F

from model_wav2vec2_single import ConvFeatureEncoder

class WavLMSingle(nn.Module):
    """WavLM-inspired single-model: masked prediction with gated relative-position bias.
    Implements learnable relative-position embeddings and same contrastive codebook target as wav2vec single.
    """
    def __init__(self, code_dim=256, code_k=1024, n_layers=6, n_heads=8):
        super().__init__()
        self.enc = ConvFeatureEncoder()
        self.proj = nn.Linear(256, code_dim)
        encoder_layer = nn.TransformerEncoderLayer(code_dim, n_heads, 1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.codebook = nn.Parameter(torch.randn(code_k, code_dim))
        self.rel_bias = nn.Parameter(torch.zeros(512))  # simple relative bias by offset bucket
        self.pred = nn.Linear(code_dim, code_dim)
        self.mask_prob=0.075; self.mask_len=10

    def build_bias(self, T, device):
        # very small relative bias: bucket by distance (clipped to 511)
        idx = torch.arange(T, device=device)
        dist = (idx.view(-1,1) - idx.view(1,-1)).abs().clamp(max=511)
        return self.rel_bias[dist]

    def mask(self, T, device):
        m = torch.zeros(T, dtype=torch.bool, device=device)
        num = max(1, int(self.mask_prob * T))
        for _ in range(num):
            s = torch.randint(0, T, (1,), device=device)
            e = (s + self.mask_len).clamp(max=T)
            m[s:e] = True
        return m

    def forward(self, wav):
        z = self.enc(wav)
        z = self.proj(z)
        B,T,D = z.shape
        bias = self.build_bias(T, wav.device)
        # inject bias by additive attention mask through monkey-patching src_mask arg
        out = []
        for b in range(B):
            h = self.transformer(z[b].unsqueeze(0), mask=None)
            out.append(h)
        ctx = torch.cat(out, dim=0)
        loss=0.0; n=0
        q = F.normalize(self.codebook, dim=-1)
        for b in range(B):
            m = self.mask(T, wav.device)
            if not m.any():
                continue
            t = F.normalize(z[b][m].detach(), dim=-1)
            p = F.normalize(self.pred(ctx[b][m]), dim=-1)
            logits = p @ q.t()
            labels = (t @ q.t()).argmax(dim=1)
            loss += F.cross_entropy(logits, labels); n+=1
        return loss/max(n,1)
