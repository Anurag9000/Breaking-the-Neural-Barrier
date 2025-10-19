from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


@dataclass
class TreeLSTMConfig:
    vocab_size: int = 5000
    emb_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 3
    pad_idx: int = 0


class ChildSumTreeLSTM(nn.Module):
    """Child-Sum TreeLSTM (Tai et al., 2015) single-model classifier using root state.
    Tree is provided as (x, children) where children[i] = list of child indices of node i.
    """
    def __init__(self, cfg: TreeLSTMConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.W_iou = nn.Linear(cfg.emb_dim, 3*D)
        self.U_iou = nn.Linear(D, 3*D, bias=False)
        self.W_f = nn.Linear(cfg.emb_dim, D)
        self.U_f = nn.Linear(D, D, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(D, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in [self.W_iou, self.U_iou, self.W_f, self.U_f, self.fc]:
            if hasattr(m, 'weight'): nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

    def node(self, x_i, child_states: List[Tuple[torch.Tensor, torch.Tensor]]):
        if len(child_states) == 0:
            h_sum = x_i.new_zeros(x_i.size(0), self.cfg.hidden_dim)
        else:
            h_sum = torch.stack([h for (h, c) in child_states], dim=0).sum(dim=0)
        iou = self.W_iou(x_i) + self.U_iou(h_sum)
        i, o, u = iou.chunk(3, dim=-1)
        i = torch.sigmoid(i); o = torch.sigmoid(o); u = torch.tanh(u)
        if len(child_states) == 0:
            c = i * u
        else:
            f_list = []
            for (h_k, c_k) in child_states:
                f_k = torch.sigmoid(self.W_f(x_i) + self.U_f(h_k))
                f_list.append(f_k * c_k)
            c = i * u + torch.stack(f_list, dim=0).sum(dim=0)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, tokens: torch.LongTensor, children: List[List[List[int]]], roots: List[int]):
        # tokens: (B, N) node token ids per batch; children: B x N x variable children list; roots: B list root idx
        B, N = tokens.shape
        x = self.embed(tokens)  # (B,N,E)
        device = tokens.device
        D = self.cfg.hidden_dim
        outputs = []
        for b in range(B):
            # compute in topological order with post-order DFS
            ch = children[b]
            root = roots[b]
            # build order
            visited = set()
            order = []
            stack = [(root, 0, False)]  # (node, parent, expanded?)
            # Ensure we visit all nodes reachable from root
            while stack:
                u, p, expanded = stack.pop()
                if not expanded:
                    stack.append((u, p, True))
                    for v in ch[u]:
                        stack.append((v, u, False))
                else:
                    order.append(u)
            h = [x.new_zeros(D) for _ in range(N)]
            c = [x.new_zeros(D) for _ in range(N)]
            for u in order:
                child_states = [(h[v], c[v]) for v in ch[u]]
                h[u], c[u] = self.node(x[b, u], child_states)
            outputs.append(h[root])
        H = torch.stack(outputs, dim=0)
        H = self.dropout(H)
        return self.fc(H)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
