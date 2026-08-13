"""tspn_gru.py — Temporal GRU module.

Locked Plan §7.3. Input: list of 8 node representation matrices, each
(N_NODES, 128), one per quarterly snapshot. Stacked to (8, N_NODES, 128)
and processed by a single-layer, unidirectional GRU with the node
dimension acting as the batch dimension (batch_first=False expects
(seq_len, batch, input_size) -- here batch=N_NODES, all nodes share the
same GRU weights, processed in parallel).

Only the FINAL hidden state is used (shape (N_NODES, 256)) -- not the full
output sequence -- followed by Dropout(0.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TSPNTemporalGRU(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: list[torch.Tensor]) -> torch.Tensor:
        """sequence: list of 8 tensors, each (N_NODES, input_dim)."""
        seq_tensor = torch.stack(sequence, dim=0)   # (8, N_NODES, input_dim)
        _, h_n = self.gru(seq_tensor)                # h_n: (num_layers, N_NODES, hidden_dim)
        final_hidden = h_n[-1]                        # (N_NODES, hidden_dim) -- last layer's final state
        return self.dropout(final_hidden)
