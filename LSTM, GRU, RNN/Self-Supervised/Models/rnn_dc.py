import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGRU(nn.Module):
    """DeepCluster for sequences: single GRU encoder + linear classifier trained on k-means pseudo-labels."""
    def __init__(self, input_dim:int, hidden_dim:int, proj_dim:int=128, num_layers:int=1, num_clusters:int=100):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.cls = nn.Linear(proj_dim, num_clusters)
    def features(self, x):
        h,_=self.encoder(x); h=h.mean(dim=1)
        return F.normalize(self.proj(h),dim=-1)
    def forward(self, x):
        z=self.features(x)
        return self.cls(z)

if __name__=='__main__':
    B,T,D=8,64,16
    net=DCGRU(D,128,64,num_clusters=10)
    x=torch.randn(B,T,D)
    print(net(x).shape)
