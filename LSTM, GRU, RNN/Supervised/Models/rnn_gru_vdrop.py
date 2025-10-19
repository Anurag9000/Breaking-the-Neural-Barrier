import torch
import torch.nn as nn

class RNN_GRU_VDrop(nn.Module):
    """GRU with variational (locked) dropout on hidden-to-hidden (Gal & Ghahramani style)."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.grucell = nn.GRUCell(input_dim, hidden_size)
        self.layers = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, T, D = x.size()
        # Locked dropout masks
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        mask = [self.dropout(torch.ones_like(h[0])) for _ in range(self.num_layers)]
        out = None
        for t in range(T):
            h[0] = self.grucell(x[:, t, :], h[0] * mask[0])
            for l, cell in enumerate(self.layers, start=1):
                h[l] = cell(h[l-1], h[l] * mask[l])
            out = h[-1]
        return self.head(out)
