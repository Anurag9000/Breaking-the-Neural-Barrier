import torch
import torch.nn as nn

class QRNNLayer(nn.Module):
    """Quasi-RNN fo-pooling layer with convolutional gates."""
    def __init__(self, input_dim, hidden_size, ksz=2):
        super().__init__()
        self.conv_z = nn.Conv1d(input_dim, hidden_size, ksz, padding=ksz-1)
        self.conv_f = nn.Conv1d(input_dim, hidden_size, ksz, padding=ksz-1)
        self.conv_o = nn.Conv1d(input_dim, hidden_size, ksz, padding=ksz-1)
        for m in (self.conv_z, self.conv_f, self.conv_o):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        # x: (B,T,D) -> (B,D,T)
        B,T,D = x.size()
        xin = x.transpose(1,2)
        z = torch.tanh(self.conv_z(xin)[:, :, :T])
        f = torch.sigmoid(self.conv_f(xin)[:, :, :T])
        o = torch.sigmoid(self.conv_o(xin)[:, :, :T])
        c = torch.zeros(B, z.size(1), T, device=x.device)
        for t in range(T):
            c[:,:,t] = f[:,:,t] * (c[:,:,t-1] if t>0 else 0.0) + (1 - f[:,:,t]) * z[:,:,t]
        h = o * c
        return h.transpose(1,2)

class RNN_QRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, ksz:int=2, dropout: float=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(QRNNLayer(d, hidden_size, ksz))
            d = hidden_size
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.drop(x)
        return self.head(x[:, -1, :])
