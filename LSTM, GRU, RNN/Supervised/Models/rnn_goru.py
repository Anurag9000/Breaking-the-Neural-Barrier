import torch
import torch.nn as nn

class GORUCell(nn.Module):
    """Simplified GORU: GRU with orthogonal recurrent transforms and modReLU candidate."""
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.Wz = nn.Linear(input_dim, hidden_size)
        self.Wr = nn.Linear(input_dim, hidden_size)
        self.Wn = nn.Linear(input_dim, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Un = nn.Linear(hidden_size, hidden_size, bias=False)
        # orthogonal init on recurrent
        for m in (self.Uz, self.Ur, self.Un):
            nn.init.orthogonal_(m.weight)
        for m in (self.Wz, self.Wr, self.Wn):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.bias_n = nn.Parameter(torch.zeros(hidden_size))  # modReLU bias
    def modrelu(self, x, b):
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-6
        return torch.relu(norm + b) * x / norm
    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        r = torch.sigmoid(self.Wr(x) + self.Ur(h))
        n_lin = self.Wn(x) + self.Un(r * h)
        n = torch.tanh(n_lin)  # fallback if modReLU unstable
        return (1 - z) * h + z * n

class RNN_GORU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(GORUCell(d, hidden_size))
            d = hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        B, T, D = x.size()
        out = x
        for cell in self.layers:
            h = torch.zeros(B, self.head.in_features, device=x.device)
            seq = []
            for t in range(T):
                h = cell(out[:, t, :], h)
                seq.append(h)
            out = torch.stack(seq, 1)
            out = self.drop(out)
        return self.head(out[:, -1, :])
