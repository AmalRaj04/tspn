"""ablation_data.py — Data transform for the no-shock ablation.

Locked Plan §9.1 item 6 / Research Brief §8 Ablation 3: "instead of e[3]
(tariff_delta), represent the shock as a global node signal (add delta_tariff
contribution to the affected node's features directly)." Tests whether
edge-level shock injection (bilateral precision) matters over a coarser
node-level representation.

This is a pure data transform, not a model change -- the resulting 10-dim
node features (9 original + 1 direct-tariff-exposure scalar) are fed into
an ordinary TSPN instance constructed with node_feat_in=10 (see
train_ablations.py); no new model class is needed (see tspn_ablations.py's
module docstring for why).

Per snapshot, the added feature IS `compute_node_delta_tau` (reused from
the Leontief baseline, not reimplemented) applied to THAT snapshot's own
edge_attr -- which is already 0 for snapshots 0-6 and real only at the
event-quarter snapshot 7, so this preserves the same "shock only visible
at the event quarter" property as the edge-level signal it replaces,
without needing separate leakage bookkeeping.
"""

from __future__ import annotations

import copy
import os
import sys

import torch
from torch_geometric.data import Data

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_BASELINES = os.path.join(PROJECT_ROOT, "src", "baselines")
for p in (PROJECT_ROOT, SRC_BASELINES):
    if p not in sys.path:
        sys.path.insert(0, p)

from leontief_io import compute_node_delta_tau  # noqa: E402


def shock_as_node_feature(temporal_sequence: list[Data]) -> list[Data]:
    """Returns a new list of Data objects: x has an extra column (direct
    tariff exposure), edge_attr's e[3] column is zeroed everywhere. Never
    mutates the input (CP23 discipline, same as src/training/augmentation.py)."""
    new_sequence: list[Data] = []
    for data_q in temporal_sequence:
        d = copy.deepcopy(data_q)

        tau = compute_node_delta_tau(d)   # numpy, (N_NODES,) -- 0 pre-event, real at q7
        tau_tensor = torch.tensor(tau, dtype=d.x.dtype, device=d.x.device).unsqueeze(1)
        d.x = torch.cat([d.x, tau_tensor], dim=1)   # (N_NODES, 10)

        d.edge_attr = d.edge_attr.clone()
        d.edge_attr[:, 3] = 0.0   # remove the edge-level shock signal entirely

        new_sequence.append(d)
    return new_sequence
