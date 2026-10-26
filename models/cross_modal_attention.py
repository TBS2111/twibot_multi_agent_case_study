import torch
import torch.nn as nn
from typing import Dict


class CrossModalAttention(nn.Module):
    """Each user has one token per modality (semantic / behavioural / trust+profile).
    Tokens attend to each other (cross-modal), then are pooled into one vector."""

    def __init__(self, dims: Dict[str, int], d: int, heads: int, dropout: float):
        super().__init__()
        self.names = list(dims)
        self.proj = nn.ModuleDict({n: nn.Sequential(nn.Linear(m, d), nn.GELU(), nn.LayerNorm(d))
                                   for n, m in dims.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(dims), d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.pool = nn.Linear(d, 1)

    def forward(self, mods: Dict[str, torch.Tensor]) -> torch.Tensor:
        toks = torch.stack([self.proj[n](mods[n]) for n in self.names], 1) + self.mod_emb  # N,M,d
        a, _ = self.attn(toks, toks, toks)
        toks = self.ln1(toks + a)
        toks = self.ln2(toks + self.ffn(toks))
        w = torch.softmax(self.pool(toks), dim=1)
        return (w * toks).sum(1)
