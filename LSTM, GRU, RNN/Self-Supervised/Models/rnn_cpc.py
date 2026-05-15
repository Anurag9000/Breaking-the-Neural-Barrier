import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.matmul(a, b.transpose(-1, -2))  # (B,Tq,Tk)

class CPCGRU(nn.Module):
    """
    CPC with a single GRU encoder shared for context and future.
    - Encode sequence to latent z_t
    - Context c_t is the GRU hidden output at t
    - Predict future z_{t+k} using linear predictors W_k, k=1..K
    - InfoNCE across batch-time negatives
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int = 128, K: int = 3, num_layers: int = 1):
        super().__init__()
        self.K = K
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.predictors = nn.ModuleList([nn.Linear(hidden_dim, proj_dim) for _ in range(K)])

    def forward(self, x):
        # x: (B,T,D)
        h, _ = self.encoder(x)   # (B,T,H)
        z = self.proj(h)         # (B,T,P)
        return h, z

    def nce_loss(self, h, z):
        # h: (B,T,H), z: (B,T,P)
        B,T,_ = z.shape
        total_loss = 0.0
        n_terms = 0
        for k, pred in enumerate(self.predictors, start=1):
            # predictor on context h_t -> pred for z_{t+k}
            c = pred(h[:, :-k, :])            # (B, T-k, P)
            target = z[:, k:, :]              # (B, T-k, P)
            # negatives: all other positions in batch
            # compute similarity matrix between c and all z positions at same future time indices
            # flatten batch/time for contrast within batch
            c_flat = c.reshape(-1, c.size(-1))        # (B*(T-k), P)
            tgt_flat = target.reshape(-1, target.size(-1))
            # scores to all targets in the batch-time set
            logits = torch.matmul(F.normalize(c_flat, dim=-1), F.normalize(tgt_flat, dim=-1).t())  # (N,N)
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss
            n_terms += 1
        return total_loss / max(n_terms, 1)
