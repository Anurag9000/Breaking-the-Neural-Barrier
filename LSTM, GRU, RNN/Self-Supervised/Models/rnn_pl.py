import torch
import torch.nn as nn
import torch.nn.functional as F

class PLGRU(nn.Module):
    """Single-model pseudo-labeling: same GRU encoder + classifier head."""
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int=1, num_classes:int=10):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_classes))
    def forward(self, x):
        h,_=self.encoder(x)
        h=h.mean(dim=1)
        return self.head(h)

if __name__=='__main__':
    B,T,D=8,64,16
    net=PLGRU(D,64,num_classes=5)
    x=torch.randn(B,T,D)
    print(net(x).shape)
