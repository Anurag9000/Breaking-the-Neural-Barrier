
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
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_widths, rep_dim: int, use_bn: bool=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_widths = list(hidden_widths)
        self.rep_dim = int(rep_dim)
        self.use_bn = use_bn

        layers = []
        prev = in_dim
        for w in self.hidden_widths:
            layers.append(MLPBlock(prev, w, use_bn))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.rep = nn.Linear(prev, self.rep_dim)

    def forward(self, img):
        x = img.view(img.size(0), -1)
        h = self.backbone(x)
        z = self.rep(h)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        # 2-layer MLP projector
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, proj_dim)

    def forward(self, z):
        x = self.fc1(z)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MLPSSL(nn.Module):
    """
    Single-model SimCLR-style MLP for images (no CNN).
    """
    def __init__(self, in_dim: int, hidden_widths, rep_dim: int, proj_dim: int, use_bn: bool=True):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_widths, rep_dim, use_bn)
        self.projector = ProjectionHead(rep_dim, proj_dim)

    def forward(self, img):
        # returns representation and projection
        z = self.encoder(img)
        p = self.projector(z)
        return z, p

def nt_xent_loss(p_i, p_j, temperature: float=0.2):
    """
    SimCLR NT-Xent loss (single-encoder; two views pass through same weights).
    p_i, p_j: (N, D) projection vectors for two augmented views.
    """
    z_i = F.normalize(p_i, dim=1)
    z_j = F.normalize(p_j, dim=1)

    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

    # similarity matrix
    sim = torch.mm(z, z.t())  # (2N, 2N)
    # mask self-sim
    diag = torch.eye(2*N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(diag, -9e15)

    # positives: (i, i+N) and (i+N, i)
    pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    idx = torch.arange(0, 2*N).to(z.device)
    pos_sim = sim[idx, pos]  # (2N,)

    logits = sim / temperature
    labels = pos  # index of positive for each row

    loss = F.cross_entropy(logits, labels)
    return loss
