"""tspn_gcn_layer.py — GCN ablation layer: uniform mean aggregation, no attention.

Locked Plan §9.1 ablation 4 / Research Brief §8 Ablation 1: "replace GAT
with GCN (equal-weight mean aggregation over neighbors). Tests whether the
learned attention weights add value over treating all suppliers equally."

Keeps everything else about TSPNGATLayer's structure (per-head value
projection, multi-head concat, ELU activation, residual connection) so the
ONLY thing this ablation changes relative to the full model is how a
node's neighbors are weighted when aggregated -- uniform (1/in-degree, via
PyG's aggr="mean") instead of learned softmax attention. No query/key/edge
projections or attention parameters exist in this layer at all, since
there is no attention to compute.

last_alpha is always None here -- GCN has no attention weights, and the
loss function must skip the L1 attention-sparsity term for this ablation
rather than being handed a fabricated uniform value that would misrepresent
this ablation's own interpretability comparison in Phase 10.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class TSPNGCNLayer(MessagePassing):
    def __init__(
        self,
        in_dim: int = 128,
        edge_dim: int = 64,
        num_heads: int = 4,
        head_dim: int = 32,
    ):
        super().__init__(aggr="mean", node_dim=0)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim

        self.W_v = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_res = nn.Linear(in_dim, self.out_dim, bias=False)
        self.act = nn.ELU()

        self.last_alpha = None   # GCN has no attention weights (see module docstring)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        v = self.W_v(x).view(n, self.num_heads, self.head_dim)

        out = self.propagate(edge_index, v=v, size=(n, n))   # uniform mean over incoming edges
        out = out.reshape(n, self.out_dim)
        out = self.act(out)

        h_res = self.W_res(x)
        return out + h_res

    def message(self, v_j: torch.Tensor) -> torch.Tensor:
        return v_j
