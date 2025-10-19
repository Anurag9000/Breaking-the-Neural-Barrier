import torch
import torch.nn as nn
import torch.nn.functional as F

class TIDGRU(nn.Module):
    """
    Temporal Instance Discrimination: single GRU encoder + linear classifier over instance IDs.
    For toy purposes, we subsample a manageable number of instances per epoch.
    """
    def __init__(self, input_dim:int, hidden_dim:int, proj_dim:int=128, num_layers:int=1, n_classes:int=1024):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.cls = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        h,_ = self.encoder(x)
        h = h.mean(dim=1)
        z = F.normalize(self.proj(h), dim=-1)
        return self.cls(z)

if __name__=='__main__':
    B,T,D=32,64,16
    net=TIDGRU(D,128,128,n_classes=256)
    x=torch.randn(B,T,D)
    print(net(x).shape)
