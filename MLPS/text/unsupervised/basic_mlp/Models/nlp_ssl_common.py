
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bn: bool=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None: x = self.bn(x)
        return self.act(x)

class TextAvgEmbed(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    def forward(self, token_ids, lengths):
        E = self.emb(token_ids)                              # (B,L,D)
        mask = (token_ids != 0).float().unsqueeze(-1)        # (B,L,1)
        summed = (E * mask).sum(dim=1)                       # (B,D)
        denom = lengths.clamp(min=1).float().unsqueeze(-1)   # (B,1)
        return summed / denom                                # (B,D)

def nt_xent_loss(z1, z2, temperature: float=0.05):
    """
    Symmetric NT-Xent loss for two views within a batch.
    z1, z2: (B, d) normalized features.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)                       # (2B, d)
    sim = reps @ reps.t() / temperature                      # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(mask, -1e9)                             # remove self-sim
    # positives: (i -> i+B) and (i+B -> i)
    targets = torch.arange(B, device=sim.device)
    loss_i = F.cross_entropy(sim[:B, B:], targets)           # z1 against z2
    loss_j = F.cross_entropy(sim[B:, :B], targets)           # z2 against z1
    return 0.5*(loss_i + loss_j)
