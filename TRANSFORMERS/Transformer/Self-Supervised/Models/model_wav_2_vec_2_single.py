import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFeatureEncoder(nn.Module):
    def __init__(self, in_ch=1, dims=(64,128,256), strides=(5,2,2), kernels=(10,3,3)):
        super().__init__()
        layers=[]; c=in_ch
        for d,s,k in zip(dims, strides, kernels):
            layers+= [nn.Conv1d(c,d,k,s, padding=k//2), nn.GELU(), nn.GroupNorm(8,d)]
            c=d
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B,1,T)
        z = self.net(x)  # (B,C,T')
        return z.transpose(1,2)  # (B,T',C)

class Wav2Vec2Single(nn.Module):
    """Simplified wav2vec 2.0-style: mask time steps; contrastive prediction to a learnable codebook.
    Single model with in-model codebook (no EMA/teacher)."""
    def __init__(self, code_dim=256, code_k=1024, mask_prob=0.065, mask_len=10):
        super().__init__()
        self.enc = ConvFeatureEncoder()
        self.proj = nn.Linear(256, code_dim)
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(code_dim, 8, 1024, batch_first=True), 6)
        self.codebook = nn.Parameter(torch.randn(code_k, code_dim))
        self.mask_prob = mask_prob; self.mask_len = mask_len
        self.pred = nn.Linear(code_dim, code_dim)

    def mask(self, T, device):
        m = torch.zeros(T, dtype=torch.bool, device=device)
        num = max(1, int(self.mask_prob * T))
        for _ in range(num):
            s = torch.randint(0, T, (1,), device=device)
            e = (s + self.mask_len).clamp(max=T)
            m[s:e] = True
        return m

    def forward(self, wav):  # wav: (B,1,T)
        z = self.enc(wav)              # (B,T',C)
        z = self.proj(z)               # (B,T',D)
        B,T,D = z.shape
        ctx = self.context(z)          # (B,T,D)
        loss=0.0; n=0
        for b in range(B):
            m = self.mask(T, wav.device)
            masked = ctx[b][m]         # (M,D)
            if masked.size(0)==0: continue
            q = F.normalize(self.codebook, dim=-1)
            t = F.normalize(z[b][m].detach(), dim=-1)  # targets from features (no teacher)
            p = F.normalize(self.pred(masked), dim=-1)
            logits = p @ q.t()          # (M,K)
            labels = (t @ q.t()).argmax(dim=1)
            loss += F.cross_entropy(logits, labels); n+=1
        return loss/max(n,1)
