import torch, torch.nn as nn, torch.nn.functional as F, random
from dataclasses import dataclass
from adp_gat_graphcl import GATEncoder, aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph
AUGS=[aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

def sinkhorn(Q, iters=3, eps=1e-12):
    Q = torch.exp(Q); Q = Q / (Q.sum()); K,B = Q.size(0), Q.size(1)
    for _ in range(iters):
        Q /= (Q.sum(dim=1, keepdim=True)+eps); Q /= (Q.sum(dim=0, keepdim=True)+eps)
    return (Q * Q.size(0)).detach()

@dataclass
class Config:
    nprotos:int=100; temperature:float=0.1; lr:float=1e-3; epochs:int=400; patience:int=50; ckpt:str="ckpt_gat_proto.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class Proto_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GATEncoder(in_dim); self.protos = nn.Parameter(torch.randn(cfg.nprotos, self.enc.proj.out_features))
        nn.init.kaiming_normal_(self.protos); self.cfg=cfg
    def loss(self, data):
        a1, a2 = random.choice(AUGS), random.choice(AUGS)
        z1, z2 = self.enc(a1(data)), self.enc(a2(data))
        z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
        # assignments via prototypes
        p1 = torch.matmul(z1, F.normalize(self.protos, dim=-1).t())/self.cfg.temperature
        p2 = torch.matmul(z2, F.normalize(self.protos, dim=-1).t())/self.cfg.temperature
        # balanced codes (approx) – here use softmax as proxy
        q1 = F.softmax(p1, dim=1).detach(); q2 = F.softmax(p2, dim=1).detach()
        loss = -(torch.mean(torch.sum(q1*F.log_softmax(p2,dim=1),dim=1)) + torch.mean(torch.sum(q2*F.log_softmax(p1,dim=1),dim=1)))/2
        return loss
