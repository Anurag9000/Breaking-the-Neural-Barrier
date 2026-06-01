import torch
import torch.nn as nn

class SelfAttention1D(nn.Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model // n_heads
        self.nh = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
    def forward(self, x):
        # x: (B,T,D)
        B,T,D = x.size()
        q = self.q(x).view(B,T,self.nh,self.d).transpose(1,2)  # (B,nh,T,d)
        k = self.k(x).view(B,T,self.nh,self.d).transpose(1,2)
        v = self.v(x).view(B,T,self.nh,self.d).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (self.d ** 0.5)
        w = att.softmax(-1)
        y = (w @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.o(y)

class RNN_GRU_AttnHead(nn.Module):
    """GRU encoder + intra-sequence self-attention over hidden states."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, heads:int=1, dropout: float=0.1, bidirectional: bool=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.attn = SelfAttention1D(hidden_size*mult, n_heads=heads)
        self.norm = nn.LayerNorm(hidden_size*mult)
        self.head = nn.Linear(hidden_size*mult, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        h, _ = self.gru(x)
        a = self.attn(h)
        y = self.norm(h + a)
        # pool over time (mean)
        y = y.mean(dim=1)
        return self.head(y)
