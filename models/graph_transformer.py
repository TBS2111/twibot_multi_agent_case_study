import math
import torch
import torch.nn as nn


def edge_softmax(score, dst, n):
    """softmax of score[E,H] over edges sharing the same destination node."""
    H = score.size(1)
    mx = torch.full((n, H), -1e30, device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, dst[:, None].expand_as(score), score, "amax", include_self=True)
    ex = torch.exp(score - mx[dst])
    den = torch.zeros(n, H, device=score.device, dtype=score.dtype).index_add_(0, dst, ex)
    return ex / (den[dst] + 1e-16)


class GraphTransformerLayer(nn.Module):
    """Sparse multi-head graph attention. The signed edge weight (a) biases the
    attention score and (b) its sign flips the message, so negative neighbours repel."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.edge_bias = nn.Linear(1, heads)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ei, ew):
        n = x.size(0)
        src, dst = ei
        q = self.q(x).view(n, self.h, self.dk)
        k = self.k(x).view(n, self.h, self.dk)
        v = self.v(x).view(n, self.h, self.dk)
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.dk) + self.edge_bias(ew[:, None])
        alpha = self.drop(edge_softmax(score, dst, n))
        msg = v[src] * alpha.unsqueeze(-1) * torch.sign(ew)[:, None, None]
        out = torch.zeros(n, self.h, self.dk, device=x.device, dtype=x.dtype).index_add_(0, dst, msg)
        x = self.ln1(x + self.drop(self.o(out.reshape(n, -1))))
        return self.ln2(x + self.drop(self.ffn(x)))


class GraphTransformer(nn.Module):
    def __init__(self, d, out_dim, heads, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(GraphTransformerLayer(d, heads, dropout) for _ in range(layers))
        self.out = nn.Linear(d, out_dim)

    def forward(self, x, ei, ew):
        n = x.size(0)
        loop = torch.arange(n, device=x.device)
        ei = torch.cat([ei, torch.stack([loop, loop])], 1)
        ew = torch.cat([ew, torch.ones(n, device=x.device)])
        for layer in self.layers:
            x = layer(x, ei, ew)
        return self.out(x)
