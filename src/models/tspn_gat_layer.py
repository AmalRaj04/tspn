"""tspn_gat_layer.py — Custom edge-feature-aware multi-head GAT layer.

Locked Plan §7.2. NOT standard torch_geometric.nn.GATConv, which ignores
edge features in attention computation -- this layer includes them, which
is the mechanism that lets the shock signal (e[3]) modulate attention.

Per-head attention (locked formula), K=4 heads, head_dim=32:
    query_i  = W_q_k @ h_i
    key_j    = W_k_k @ h_j
    edge_kk  = W_e_k @ e_ij
    score_ij = a_k^T . LeakyReLU([query_i || key_j || edge_kk], slope=0.2)
    score_ij = score_ij / sqrt(head_dim)     -- CP30: scaling BEFORE softmax,
                                                 standard scaled dot-product
                                                 attention, prevents logit
                                                 explosion / attention collapse
    alpha_ij = softmax over all j in N(i) of score_ij
    alpha_ij = Dropout(alpha_ij, p=0.3)      -- training only (nn.Dropout,
                                                 auto-disabled by model.eval(),
                                                 CP28)

Per-head aggregation: value_j = W_v_k @ h_j; m_i^k = sum_j alpha_ij * value_j

Multi-head concat + residual:
    h_i_concat = concat([m_i^1..m_i^4])           -> dim 128
    h_i_out    = ELU(h_i_concat) + W_res @ h_i_input   (W_res: Linear(128,128), no bias)

Per-head weight matrices (W_q_k, W_k_k, W_v_k, W_e_k for k=1..4) are
implemented as one combined Linear(in, num_heads*head_dim) reshaped into
(num_heads, head_dim) -- mathematically identical to 4 independent
Linear(in, head_dim) modules (each output chunk has its own weight rows,
never shared across heads), just one matmul instead of four.

self.last_alpha (shape [num_edges, num_heads]) is stored on every forward
pass, WITHOUT detaching -- the training loss's L1 attention-sparsity term
(config.TRAINING["loss_weight_l1_attn"]) backprops through it, so it must
stay attached to the autograd graph. CP32: read it immediately after the
forward call, before any other forward pass overwrites it.

CP27: gat_layer1 and gat_layer2 in tspn.py MUST be two separate
TSPNGATLayer(...) constructor calls -- never `gat_layer2 = gat_layer1`,
which would alias the weights and silently collapse the model to 1 layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class TSPNGATLayer(MessagePassing):
    def __init__(
        self,
        in_dim: int = 128,
        edge_dim: int = 64,
        num_heads: int = 4,
        head_dim: int = 32,
        negative_slope: float = 0.2,
        dropout: float = 0.3,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(head_dim)

        self.W_q = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, self.out_dim, bias=False)
        self.W_res = nn.Linear(in_dim, self.out_dim, bias=False)

        self.attn = nn.Parameter(torch.empty(num_heads, 3 * head_dim))
        nn.init.xavier_uniform_(self.attn)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

        self.last_alpha: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q = self.W_q(x).view(n, self.num_heads, self.head_dim)
        k = self.W_k(x).view(n, self.num_heads, self.head_dim)
        v = self.W_v(x).view(n, self.num_heads, self.head_dim)
        e = self.W_e(edge_attr).view(-1, self.num_heads, self.head_dim)

        out = self.propagate(edge_index, q=q, k=k, v=v, e=e, size=(n, n))
        out = out.reshape(n, self.out_dim)
        out = self.act(out)

        h_res = self.W_res(x)
        return out + h_res

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        e: torch.Tensor,
        index: torch.Tensor,
        size_i: int,
    ) -> torch.Tensor:
        concat = torch.cat([q_i, k_j, e], dim=-1)          # (E, H, 3*head_dim)
        scored = self.leaky_relu(concat)
        score = (scored * self.attn.unsqueeze(0)).sum(dim=-1)   # (E, H)
        score = score / self.scale                          # CP30 scaling

        alpha = softmax(score, index, num_nodes=size_i)     # (E, H), softmax over incoming edges per target
        alpha = self.attn_dropout(alpha)
        self.last_alpha = alpha   # not detached -- L1 loss term needs gradients (see module docstring)

        return v_j * alpha.unsqueeze(-1)   # (E, H, head_dim)
