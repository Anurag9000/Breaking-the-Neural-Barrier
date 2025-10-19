import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

class EdgeGATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout=0.5, heads=4, concat=True):
        super().__init__(aggr='add')
        self.heads=heads; self.out_dim=out_dim; self.concat=concat; self.dropout=dropout
        self.lin_src = nn.Linear(in_dim, heads*out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads*out_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_dim))
        self.bias = nn.Parameter(torch.zeros(heads*out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.att)
        self.edge_proj = nn.Linear(1, heads, bias=False)  # scalar edge feature -> head-wise bias

    def forward(self, x, edge_index, edge_attr=None):
        H = self.heads; x_src = self.lin_src(x); x_dst = self.lin_dst(x)
        x_src = x_src.view(-1,H,self.out_dim); x_dst = x_dst.view(-1,H,self.out_dim)
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1),1), device=x.device)
        return self.propagate(edge_index, x=(x_src,x_dst), edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_index, edge_attr):
        # compute attention scores with edge bias per head
        a_input = torch.cat([x_i, x_j], dim=-1)  # [E, H, 2*D]
        alpha = (a_input * self.att).sum(dim=-1)  # [E,H]
        alpha = alpha + self.edge_proj(edge_attr)  # inject edge feature bias
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        m = x_j * alpha.unsqueeze(-1)
        return m

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = super().aggregate(inputs, index, ptr, dim_size)
        if self.concat:
            out = out.reshape(-1, self.heads*self.out_dim)
        else:
            out = out.mean(dim=1)
        out = out + self.bias
        return out

class EdgeGATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=7, heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers>=2
        self.dropout=dropout
        self.layers = nn.ModuleList()
        self.layers.append(EdgeGATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers-2):
            self.layers.append(EdgeGATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.out = EdgeGATConv(hidden_dim*heads, out_dim, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index, edge_attr)
