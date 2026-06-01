import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=4, dim=256):
        super().__init__()
        self.q = nn.Linear(q_dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads; self.scale = (dim//heads) ** -0.5
    def forward(self, queries, kv):
        # queries: (B, Q, q_dim); kv: (B, N, kv_dim)
        B,Q,_ = queries.shape; N = kv.size(1); H=self.heads; D=self.q.out_features
        q = self.q(queries).view(B,Q,H,D//H)
        k = self.k(kv).view(B,N,H,D//H)
        v = self.v(kv).view(B,N,H,D//H)
        attn = (q.unsqueeze(3) * self.scale) @ k.transpose(2,3)  # (B,Q,H,N)
        attn = attn.softmax(-1)
        out = (attn @ v).sum(3).view(B,Q,D)
        return self.proj(out)

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

class PerceiverIO(nn.Module):
    """Perceiver-IO: latent array with cross-attention in/out.
    This version maps a generic input token array to logits for classification.
    """
    def __init__(self, input_dim=64, latent_dim=256, latent_len=64, depth=6, heads=4, num_classes=10):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, latent_len, latent_dim))
        self.in_cross = CrossAttention(latent_dim, input_dim, heads, latent_dim)
        self.blocks = nn.ModuleList([SelfAttention(latent_dim, heads) for _ in range(depth)])
        self.out_cross = CrossAttention(num_classes, latent_dim, heads, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        B,N,F = x.shape
        lat = self.latents.expand(B, -1, -1)
        lat = lat + self.in_cross(lat, x)
        for blk in self.blocks:
            lat = blk(lat)
        # pool latents to a single vector
        z = lat.mean(dim=1)
        return self.classifier(z)
