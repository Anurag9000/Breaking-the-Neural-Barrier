import torch
import torch.nn as nn

class TOVGRU(nn.Module):
    """
    Classify whether chunk order of a sequence is correct or permuted.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, num_chunks: int = 4):
        super().__init__()
        self.num_chunks = num_chunks
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        h,_ = self.encoder(x)
        h = h.mean(dim=1)
        return self.head(h)

if __name__ == '__main__':
    B,T,D=4,64,8
    net=TOVGRU(D,64)
    x=torch.randn(B,T,D)
    y=net(x)
    print(y.shape)
