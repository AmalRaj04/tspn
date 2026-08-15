"""tspn_ablations.py — Ablation model variants (Locked Plan §9.1 / Research
Brief §8, run order items 4-7).

Each class mirrors TSPN's own forward pass as closely as possible so the
ONE intentional architectural difference per ablation is easy to spot by
diffing against tspn.py, rather than being buried in conditional branches
inside a single do-everything class. The core TSPN class in tspn.py is
left completely untouched -- it is the paper's main result and its
docstring is explicit that the forward pass is locked.

The no-shock ablation (Research Brief §8 Ablation 3 / Locked Plan item 6)
needs NO new model class at all: it only changes node_feat_in from 9 to 10
(one extra input dimension for the direct-tariff-exposure scalar), which
TSPN's existing constructor already supports via model_config. Its actual
work is a DATA transform (zero e[3] everywhere, add the extra node
feature) -- see src/training/ablation_data.py -- not an architecture
change, so it is built by calling `TSPN(model_config={**config.MODEL,
"node_feat_in": 10})` directly rather than duplicated here.

CP37 (Risk Checkpoints, Critical): every ablation must be trained from a
FRESH instance, never a copy of a partially-trained model or a shared
optimizer -- enforced at the training-loop level (src/training/
train_ablations.py constructs a new instance of the relevant class inside
the fold loop), not by anything in this file.
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
from tspn_gcn_layer import TSPNGCNLayer  # noqa: E402
from tspn_gru import TSPNTemporalGRU  # noqa: E402
from output_head import MultiHorizonHead  # noqa: E402


class TSPNGCNAblation(nn.Module):
    """Ablation 1 (Locked Plan item 4): TSPNGATLayer -> TSPNGCNLayer (uniform
    mean aggregation) for both layers. Everything else identical to TSPN.
    Tests whether learned attention adds value over treating all suppliers
    equally."""

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

        gcn_kwargs = dict(
            in_dim=cfg["node_embed_dim"],
            edge_dim=cfg["edge_embed_dim"],
            num_heads=cfg["gat_num_heads"],
            head_dim=cfg["gat_head_dim"],
        )
        self.gat_layer1 = TSPNGCNLayer(**gcn_kwargs)
        self.gat_layer2 = TSPNGCNLayer(**gcn_kwargs)

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

    def forward(self, temporal_sequence: list[Data]) -> tuple[torch.Tensor, None]:
        assert len(temporal_sequence) == 8, f"expected 8 snapshots, got {len(temporal_sequence)}"
        rep_sequence: list[torch.Tensor] = []
        for data_q in temporal_sequence:
            # feature_embedding still runs its edge_path for interface parity
            # with TSPN's embedding step, but TSPNGCNLayer's forward() never
            # reads edge_attr (a real GCN ignores edge features entirely --
            # that's the ablation). This means feature_embedding.edge_path's
            # parameters legitimately never receive a gradient during
            # training for THIS model -- expected here, not a CP29 violation
            # (verified: TSPN itself, MLP baseline, and the other 3 ablations
            # all have zero missing-grad parameters).
            node_embed_q, edge_embed_q = self.feature_embedding(data_q.x, data_q.edge_attr)
            rep1_q = self.gat_layer1(node_embed_q, data_q.edge_index, edge_embed_q)
            rep2_q = self.gat_layer2(rep1_q, data_q.edge_index, edge_embed_q)
            rep_sequence.append(rep2_q)
        temporal = self.gru(rep_sequence)
        predictions = self.output_head(temporal)
        return predictions, None   # no attention weights to return (GCN)


class TSPNNoTemporalAblation(nn.Module):
    """Ablation 2 (Locked Plan item 5): "set seq_len=1, use only snapshot 7."
    Structurally identical to TSPN (same GAT layers, same GRU, same output
    head) but only ever fed a length-1 temporal sequence, so the GRU's
    recurrence never accumulates multi-quarter history. Kept as a genuine
    length-1 GRU pass (not "no GRU at all") specifically so the output_head
    still receives a 256-dim input with no architecture-dimension confound
    -- the only thing removed is the ABILITY to use temporal history, not
    the GRU module itself. Tests whether lagged price dynamics add value
    over the event-quarter snapshot alone."""

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
        # Accepts either the full 8-snapshot sequence (uses only the last,
        # event-quarter snapshot) or an already-trimmed length-1 sequence.
        single = [temporal_sequence[-1]]
        data_q = single[0]
        node_embed_q, edge_embed_q = self.feature_embedding(data_q.x, data_q.edge_attr)
        rep1_q = self.gat_layer1(node_embed_q, data_q.edge_index, edge_embed_q)
        rep2_q = self.gat_layer2(rep1_q, data_q.edge_index, edge_embed_q)

        temporal = self.gru([rep2_q])   # length-1 sequence through the GRU
        predictions = self.output_head(temporal)
        return predictions, self.gat_layer1.last_alpha


class TSPN1LayerAblation(nn.Module):
    """Ablation 4 (Locked Plan item 7): remove the gat_layer2 call --
    genuinely only 1 GAT layer exists (not constructed, not just skipped),
    so the ablation's parameter count honestly reflects a 1-hop model.
    Tests whether capturing the 2-hop neighborhood (indirect suppliers)
    adds value over 1-hop (direct suppliers only)."""

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
        self.gat_layer1 = TSPNGATLayer(**gat_kwargs)   # only layer -- no gat_layer2 at all

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
            rep_sequence.append(rep1_q)
        temporal = self.gru(rep_sequence)
        predictions = self.output_head(temporal)
        return predictions, self.gat_layer1.last_alpha
