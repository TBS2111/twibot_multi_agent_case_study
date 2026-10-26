"""Signed GNN + Graph Transformer hybrid encoder (used by the GBT-CMA agent).

x (fused by cross-modal attention)
   -> SignedGNN layers      : local message passing, positive neighbours attract, negative repel
   -> Graph Transformer     : attention over neighbours, signed edge weights bias the scores
   -> gated fusion          : gate * gnn_out + (1 - gate) * transformer_out
   -> linear -> embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_transformer import GraphTransformerLayer


class SignedGNNLayer(nn.Module):
    """h_i' = LN( h_i + GELU( W_s h_i + W_p * mean_{j in N+(i)} h_j  -  W_n * mean_{j in N-(i)} h_j ) )
    Means are weighted by |edge weight| (balance-theory style signed GCN)."""

    def __init__(self, d, dropout):
        super().__init__()
        self.w_self = nn.Linear(d, d)
        self.w_pos = nn.Linear(d, d, bias=False)
        self.w_neg = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _agg(x, src, dst, w, n):
        deg = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, dst, w)
        out = torch.zeros_like(x).index_add_(0, dst, x[src] * w[:, None])
        return out / (deg[:, None] + 1e-8)

    def forward(self, x, ei, ew):
        n = x.size(0)
        src, dst = ei
        pos = ew > 0
        a_pos = self._agg(x, src[pos], dst[pos], ew[pos].abs(), n)
        a_neg = self._agg(x, src[~pos], dst[~pos], ew[~pos].abs(), n)
        h = self.w_self(x) + self.w_pos(a_pos) - self.w_neg(a_neg)
        return self.ln(x + self.drop(F.gelu(h)))


class GNNTransformerEncoder(nn.Module):
    def __init__(self, d, out_dim, heads, gnn_layers, gt_layers, dropout):
        super().__init__()
        self.gnn = nn.ModuleList(SignedGNNLayer(d, dropout) for _ in range(gnn_layers))
        self.gt = nn.ModuleList(GraphTransformerLayer(d, heads, dropout) for _ in range(gt_layers))
        self.gate = nn.Linear(2 * d, d)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x, ei, ew):
        n = x.size(0)
        h = x
        for layer in self.gnn:                       # local, signed message passing
            h = layer(h, ei, ew)
        loop = torch.arange(n, device=x.device)      # transformer attends with self-loops
        ei_l = torch.cat([ei, torch.stack([loop, loop])], 1)
        ew_l = torch.cat([ew, torch.ones(n, device=x.device)])
        t = h
        for layer in self.gt:                        # attention-based refinement
            t = layer(t, ei_l, ew_l)
        g = torch.sigmoid(self.gate(torch.cat([h, t], -1)))
        return self.out(g * h + (1 - g) * t)
