"""feature_embedding.py — Node and edge feature embedding layers.

Locked Plan §7.1. No weight sharing between the node and edge paths.
Kaiming uniform initialization is nn.Linear's PyTorch default, so no
custom init code is needed.

Node path: Linear(9, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1)
Edge path: Linear(6, 64)  -> ReLU -> Dropout(0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    def __init__(
        self,
        node_feat_in: int = 9,
        edge_feat_in: int = 6,
        node_embed_dim: int = 128,
        edge_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_path = nn.Sequential(
            nn.Linear(node_feat_in, node_embed_dim),
            nn.BatchNorm1d(node_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_path = nn.Sequential(
            nn.Linear(edge_feat_in, edge_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_embed = self.node_path(x)
        edge_embed = self.edge_path(edge_attr)
        return node_embed, edge_embed
