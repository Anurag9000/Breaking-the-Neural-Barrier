import torch
import torch.nn as nn

# ----------------------------
# Nyströmformer: landmark-based attention approximation (simplified)
# ----------------------------
class NystromSelfAttn(nn.Module):
    def __init__(self, dim, heads=8, landmarks=64):
        super().__init__(); self.h=heads; self.dim=dim; self.dk=dim//heads; self.L=landmarks
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        B,S,D = x.shape; H=self.h; dh=self.dk; L=min(self.L, S)
        x1 = self.ln1(x)
        q = self.q(x1).view(B,S,H,dh); k = self.k(x1).view(B,S,H,dh); v = self.v(x1).view(B,S,H,dh)
        # choose landmarks as uniform samples
        idx = torch.linspace(0,S-1,L,device=x.device).long()
        K_land = k[:, idx]
        # compute Nystrom approximation: A ≈ QK_L (K_L^T K_L)^-1 K_L^T
        QK = (q.unsqueeze(-2) @ K_land.transpose(-1,-2)) / (dh**0.5)  # B,S,H,1,L
        KLK = (K_land.unsqueeze(-2) @ K_land.transpose(-1,-2)) / (dh**0.5)  # B,H,1,L,L
        # add identity to stabilize pseudo-inverse
        I = torch.eye(L, device=x.device).view(1,1,L,L)
        KLK = KLK.squeeze(-3) + 0.1*I
        inv = torch.linalg.inv(KLK)
        A_approx = QK @ inv.unsqueeze(-3) @ K_land.transpose(-1,-2).unsqueeze(-3)
        out = (A_approx @ v.unsqueeze(-3)).squeeze(-3).view(B,S,D)
        x = x + self.proj(out)
        x = x + self.ffn(self.ln2(x))
        return x

class NystromformerEncoder(nn.Module):
    def __init__(self, vocab, num_classes, dim=256, depth=6, heads=8, landmarks=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([NystromSelfAttn(dim, heads, landmarks) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, num_classes)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:,0]))
