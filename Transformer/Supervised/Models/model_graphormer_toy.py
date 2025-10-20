import torch
import torch.nn as nn

class GraphormerToy(nn.Module):
    """Simplified Graphormer: token per node + structural bias (shortest-path distance) added to attention weights."""
    def __init__(self, num_node_feats: int, num_classes: int, d_model: int = 256, nhead: int = 8, depth: int = 6):
        super().__init__()
        self.inp = nn.Linear(num_node_feats, d_model)
        self.bias_proj = nn.Linear(1, nhead)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, spd):
        # x: (B, N, F); spd: (B, N, N) shortest path distances (0 on diag)
        h = self.inp(x)
        B, N, D = h.shape
        # turn spd to attention bias mask
        bias = self.bias_proj(spd.unsqueeze(-1)).permute(0,3,1,2)  # (B, H, N, N)
        # flatten batch into sequence groups using custom attn mask through hooks
        # Use additive mask by summing over heads in-place via register_forward_pre_hook-like pattern
        # Here we approximate by folding bias into encoder via repeated layers with attn_mask (same for all heads)
        # Take mean across heads to a single mask for simplicity
        mean_bias = bias.mean(dim=1)  # (B, N, N)
        out = []
        for b in range(B):
            attn_mask = -1e9 * torch.ones(N, N, device=h.device)
            attn_mask = attn_mask + mean_bias[b]
            out.append(self.encoder(h[b:b+1], mask=attn_mask))
        z = torch.cat(out, dim=0)
        z = self.norm(z.mean(dim=1))
        return self.head(z)
