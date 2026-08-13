"""output_head.py — Multi-horizon prediction head.

Locked Plan §7.4. Three fully independent MLPs (no shared weights), one
per horizon:
    Linear(256, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 64) -> ReLU -> Linear(64, 1)

Output: (N_NODES, 3), column 0 = 3m, column 1 = 6m, column 2 = 12m.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_horizon_mlp(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


class MultiHorizonHead(nn.Module):
    def __init__(self, in_dim: int = 256, dropout: float = 0.2, num_heads: int = 3):
        super().__init__()
        self.mlp_3m = _make_horizon_mlp(in_dim, dropout)
        self.mlp_6m = _make_horizon_mlp(in_dim, dropout)
        self.mlp_12m = _make_horizon_mlp(in_dim, dropout)
        assert num_heads == 3, "MultiHorizonHead is locked to exactly 3 horizons (3m/6m/12m)"

    def forward(self, h_temporal: torch.Tensor) -> torch.Tensor:
        p3 = self.mlp_3m(h_temporal)
        p6 = self.mlp_6m(h_temporal)
        p12 = self.mlp_12m(h_temporal)
        return torch.cat([p3, p6, p12], dim=1)   # (N_NODES, 3)
