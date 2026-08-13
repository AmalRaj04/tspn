"""tspn.py — Full TSPN model assembly.

Locked Plan §7.5. Forward pass sequence (locked, do not reorder):
    1. For each snapshot q in [0..7]:
       a. node_embed_q, edge_embed_q = feature_embedding(x_q, edge_attr_q)
       b. rep1_q = gat_layer1(node_embed_q, edge_index_q, edge_embed_q)
       c. rep2_q = gat_layer2(rep1_q, edge_index_q, edge_embed_q)
       d. append rep2_q to sequence list
    2. seq_tensor = stack(sequence list)        -> (8, N_NODES, 128)
    3. temporal = gru(seq_tensor)               -> (N_NODES, 256)
    4. predictions = output_head(temporal)      -> (N_NODES, 3)
    5. return predictions, gat_layer1.last_alpha

Resolved spec contradiction: the Locked Plan's "critical design rules" list
also states "edge_embed from snapshot 7 (event-time) is used for
gat_layer1/2 in all snapshots." That directly contradicts CP21 (Risk
Checkpoints, marked Critical): "e[3] (tariff_delta) must be 0.0 in
snapshots 0-6. If non-zero, the model sees the shock BEFORE it happens --
temporal data leakage." If every snapshot's GAT pass used snapshot 7's
edge features, e[3] would be nonzero in every snapshot by construction,
which is exactly the leakage CP21 exists to prevent -- and it would also
be inconsistent with how the PyG datasets were actually built in Phase 5
(each snapshot has its own edge_features file, shock isolated to q7 only,
verified by scripts/verify_phase5_pyg_datasets.py). This implementation
follows the explicit, numbered "do not reorder" algorithm above: each
snapshot uses ITS OWN edge_index/edge_attr from its own Data object. See
PROJECT_STATE.md for this documented resolution.

Critical design rules honored:
    - gat_layer1 and gat_layer2 are two separate TSPNGATLayer(...)
      constructor calls (CP27) -- never aliased.
    - output_head receives ONLY the GRU's final hidden state.
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
from torch_geometric.data import Data

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from feature_embedding import FeatureEmbedding  # noqa: E402
from tspn_gat_layer import TSPNGATLayer  # noqa: E402
from tspn_gru import TSPNTemporalGRU  # noqa: E402
from output_head import MultiHorizonHead  # noqa: E402


class TSPN(nn.Module):
    def __init__(self, model_config: dict | None = None):
        super().__init__()
        cfg = model_config or config.MODEL

        self.feature_embedding = FeatureEmbedding(
            node_feat_in=cfg["node_feat_in"],
            edge_feat_in=cfg["edge_feat_in"],
            node_embed_dim=cfg["node_embed_dim"],
            edge_embed_dim=cfg["edge_embed_dim"],
            dropout=cfg["node_embed_dropout"],
        )

        gat_kwargs = dict(
            in_dim=cfg["node_embed_dim"],
            edge_dim=cfg["edge_embed_dim"],
            num_heads=cfg["gat_num_heads"],
            head_dim=cfg["gat_head_dim"],
            negative_slope=cfg["gat_leaky_slope"],
            dropout=cfg["gat_attn_dropout"],
        )
        assert cfg["gat_num_layers"] == 2, "TSPN is locked to exactly 2 GAT layers"
        # CP27: two INDEPENDENT constructor calls -- never `gat_layer2 = gat_layer1`
        self.gat_layer1 = TSPNGATLayer(**gat_kwargs)
        self.gat_layer2 = TSPNGATLayer(**gat_kwargs)

        self.gru = TSPNTemporalGRU(
            input_dim=cfg["gru_input_dim"],
            hidden_dim=cfg["gru_hidden_dim"],
            num_layers=cfg["gru_num_layers"],
            dropout=cfg["gru_output_dropout"],
        )

        self.output_head = MultiHorizonHead(
            in_dim=cfg["gru_hidden_dim"],
            dropout=cfg["mlp_dropout"],
            num_heads=cfg["mlp_num_heads"],
        )

    def forward(self, temporal_sequence: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(temporal_sequence) == 8, f"expected 8 snapshots, got {len(temporal_sequence)}"

        rep_sequence: list[torch.Tensor] = []
        for data_q in temporal_sequence:
            node_embed_q, edge_embed_q = self.feature_embedding(data_q.x, data_q.edge_attr)
            rep1_q = self.gat_layer1(node_embed_q, data_q.edge_index, edge_embed_q)
            rep2_q = self.gat_layer2(rep1_q, data_q.edge_index, edge_embed_q)
            rep_sequence.append(rep2_q)

        temporal = self.gru(rep_sequence)
        predictions = self.output_head(temporal)
        return predictions, self.gat_layer1.last_alpha
