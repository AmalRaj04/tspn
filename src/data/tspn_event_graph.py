"""tspn_event_graph.py — Container class for one event's temporal graph.

Defined in its own stable module (not inline in build_pyg_dataset.py) so
that torch.load can resolve the class by its module path when later phases
(training, evaluation, the app) load data/pyg_datasets/{event}.pt.

Per Locked Plan §5.2, this is a plain container object, not a formal PyG
Dataset/InMemoryDataset subclass.

Loading note: PyTorch >= 2.1 defaults torch.load(weights_only=True), which
rejects arbitrary custom classes. Load these files with:
    torch.load(path, weights_only=False)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class TSPNEventGraph:
    temporal_sequence: list[Data]   # 8 PyG Data objects, snapshot 0 (oldest) .. 7 (event quarter)
    y: torch.Tensor                 # (N_NODES, 3) float32 — cols: delta_3m, delta_6m, delta_12m
    label_mask: torch.Tensor        # (N_NODES,) bool
    direct_hit_mask: torch.Tensor   # (N_NODES,) bool — True iff node is the tgt of a directly shocked edge
    event_name: str
    event_date: str
