import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.sparse as dglsp


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_channels=64, num_heads=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, A, h, edge_weight):
        N = h.shape[0]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads) * self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        attn = dglsp.spmatrix(attn.indices(), torch.sigmoid(attn.val) * edge_weight, attn.shape)
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]
        return out.reshape(N, -1)


class GTLayer(nn.Module):
    def __init__(self, hidden_channels=64, num_heads=4, activation=F.relu, dropout=0.1):
        super().__init__()
        self.MHA = SparseMHA(hidden_channels, num_heads)
        self.norm = nn.BatchNorm1d(hidden_channels)
        self.FFN1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.FFN2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, h, edge_weight=None):
        h = self.norm(h + self.MHA(A, h, edge_weight))
        h = self.dropout(h)
        h = self.FFN2(self.dropout(self.activation(self.FFN1(h))))
        return h


class TimeEncode(nn.Module):
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        step = 10
        self.w = nn.Linear(1, dim, bias=False)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / step ** np.linspace(0, step - 1, self.dim, dtype=np.float32))).reshape(self.dim, -1), False)

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class DEGT(nn.Module):
    def __init__(
        self,
        in_channels,
        edge_channels,
        hidden_channels,
        out_channels,
        num_heads=16,
        num_layers=2,
        dropout=0.0
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GTLayer(hidden_channels, num_heads, F.gelu))
        self.node_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        ) if in_channels > 0 else None
        self.time_embed = TimeEncode(num_heads)
        self.edge_embed = nn.Embedding(edge_channels, num_heads) if edge_channels is not None else None
        self.dire_embed = nn.Embedding(2, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.aggregate = nn.Linear(hidden_channels, hidden_channels)
        self.predict = nn.Sequential(
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, edge_t, edge_d, return_h=False):
        x = self.node_embed(x) if self.node_embed is not None else torch.ones((x.shape[0], self.hidden_channels), device=x.device)
        if self.edge_embed is not None:
            edge_embed = self.edge_embed(edge_attr)
        time_embed = self.time_embed(edge_t)
        dire_embed = self.dire_embed(edge_d)
        A = dglsp.spmatrix(edge_index, shape=(x.shape[0], x.shape[0]))
        for conv in self.convs:
            x = conv(A, x, (edge_embed + dire_embed) * time_embed if self.edge_embed is not None else dire_embed * time_embed)
        x = self.aggregate(x)
        out = self.predict(x)
        if return_h:
            return out, x
        return out
