import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class Perceiver(nn.Module):
    """Basic Perceiver: latent array models input tokens via cross-attention in; pooled latents -> classifier."""
    def __init__(self, input_dim=64, latent_dim=256, latent_len=64, depth=6, heads=4, num_classes=10):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, latent_len, latent_dim))
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(input_dim, latent_dim)
        self.v = nn.Linear(input_dim, latent_dim)
        self.heads = heads; self.scale = (latent_dim//heads)**-0.5
        self.blocks = nn.ModuleList([SelfAttention(latent_dim, heads) for _ in range(depth)])
        self.cls = nn.Linear(latent_dim, num_classes)
    def forward(self, x):
        B,N,F = x.shape; H=self.heads; D=self.q.out_features
        lat = self.latents.expand(B,-1,-1)
        q = self.q(lat).view(B,-1,H,D//H)
        k = self.k(x).view(B,N,H,D//H)
        v = self.v(x).view(B,N,H,D//H)
        attn = (q.unsqueeze(3)*self.scale) @ k.transpose(2,3)
        attn = attn.softmax(-1)
        lat = (attn @ v).sum(3).view(B,-1,D)
        for blk in self.blocks: lat = blk(lat)
        z = lat.mean(dim=1)
        return self.cls(z)
