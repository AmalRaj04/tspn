"""build_pyg_dataset.py — Assemble PyG temporal graph datasets for all 6 events.

Implements Locked Plan §5 (PyG Graph Dataset Construction):

- ONE fixed edge_index, built from edges_2014.parquet (the project's single
  frozen structural prior, used consistently since Phase 2's Comtrade
  extension design), sorted by (src_id, tgt_id). This exact same edge_index
  tensor is reused for every one of the 8 snapshots of every one of the 6
  events (CP22) — never rebuilt or reordered per-quarter.
- For a given quarter, an edge present in the canonical 2014 template but
  missing from that quarter's edge_features file (can happen since the
  edge-count threshold is reapplied after Comtrade rescaling — see
  PROJECT_STATE.md finding #17) gets all 6 features zero-filled rather than
  being dropped, per CP22's stated fallback.
- Each of the 8 snapshots is built as an independent PyG Data object with
  its own x/edge_attr tensors (never derived via list multiplication or
  in-place mutation of a shared template) — this structurally avoids the
  CP23 aliasing bug by construction, not by an explicit deepcopy step.
- e3 (tariff_delta) is verified zero in snapshots 0-6 and nonzero at
  snapshot 7 for every event (CP21) as a hard assertion before saving.
- direct_hit_mask is True only for nodes that are the TARGET of an edge
  with nonzero e3 at the event-quarter snapshot (q7) — not downstream or
  upstream nodes.

Output: data/pyg_datasets/{event_name}.pt (TSPNEventGraph, see
src/data/tspn_event_graph.py). Load with torch.load(path, weights_only=False).

Usage:
    python src/data/build_pyg_dataset.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _here not in sys.path:
    sys.path.insert(0, _here)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402

N_NODES = config.GRAPH["N_NODES"]
EDGE_COLS = ["e0", "e1", "e2", "e3", "e4", "e5"]
FEATURE_COLS = [f"f{i}" for i in range(9)]

EDGES_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"])
NF_QUARTERLY_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "node_features_quarterly")
EF_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "edge_features")
LABELS_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"])
OUT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"])


# ---------------------------------------------------------------------------
# 1. Fixed canonical edge_index (CP22)
# ---------------------------------------------------------------------------
def build_canonical_edge_template() -> tuple[torch.Tensor, pd.MultiIndex, pd.DataFrame]:
    """Fixed edge structure for ALL snapshots of ALL events, from edges_2014."""
    path = os.path.join(EDGES_DIR, "edges_2014.parquet")
    df = pd.read_parquet(path, columns=["src_id", "tgt_id", "src_country", "tgt_country"])
    df = df.sort_values(["src_id", "tgt_id"]).reset_index(drop=True)

    dup = df.duplicated(subset=["src_id", "tgt_id"]).sum()
    assert dup == 0, f"edges_2014 has {dup} duplicate (src_id, tgt_id) pairs — cannot build a clean template"

    edge_index = torch.tensor(
        np.stack([df["src_id"].values, df["tgt_id"].values]), dtype=torch.long
    )
    canonical_multiindex = pd.MultiIndex.from_frame(df[["src_id", "tgt_id"]])
    return edge_index, canonical_multiindex, df[["src_id", "tgt_id", "src_country", "tgt_country"]]


# ---------------------------------------------------------------------------
# 2. Per-snapshot node feature tensor
# ---------------------------------------------------------------------------
def build_node_tensor(nf_df: pd.DataFrame, snapshot_idx: int) -> torch.Tensor:
    snap = nf_df[nf_df["snapshot_idx"] == snapshot_idx].sort_values("node_id")
    assert len(snap) == N_NODES, f"snapshot {snapshot_idx}: {len(snap)} rows (expected {N_NODES})"
    assert list(snap["node_id"].values) == list(range(N_NODES)), \
        f"snapshot {snapshot_idx}: node_id not a contiguous 0..{N_NODES-1} range in sorted order"
    return torch.tensor(snap[FEATURE_COLS].values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. Per-snapshot edge feature tensor, aligned to the canonical template
# ---------------------------------------------------------------------------
def build_edge_tensor(
    event_name: str, snapshot_idx: int, canonical_index: pd.MultiIndex
) -> torch.Tensor:
    path = os.path.join(EF_DIR, f"edge_features_{event_name}_q{snapshot_idx}.parquet")
    ef = pd.read_parquet(path, columns=["src_id", "tgt_id"] + EDGE_COLS)
    ef = ef.set_index(["src_id", "tgt_id"])
    aligned = ef.reindex(canonical_index)[EDGE_COLS].fillna(0.0)
    return torch.tensor(aligned.values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Labels -> y / label_mask
# ---------------------------------------------------------------------------
def build_label_tensors(event_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = os.path.join(LABELS_DIR, f"labels_{event_name}.parquet")
    df = pd.read_parquet(path).sort_values("node_id")
    assert len(df) == N_NODES, f"{event_name}: labels has {len(df)} rows (expected {N_NODES})"
    assert list(df["node_id"].values) == list(range(N_NODES)), \
        f"{event_name}: label node_id not contiguous 0..{N_NODES-1}"

    y = torch.tensor(df[["delta_3m", "delta_6m", "delta_12m"]].values, dtype=torch.float32)
    label_mask = torch.tensor(df["has_label"].values, dtype=torch.bool)
    return y, label_mask


# ---------------------------------------------------------------------------
# 5. direct_hit_mask: target of a directly-shocked edge at the event quarter
# ---------------------------------------------------------------------------
def build_direct_hit_mask(event_name: str) -> torch.Tensor:
    path = os.path.join(EF_DIR, f"edge_features_{event_name}_q7.parquet")
    ef = pd.read_parquet(path, columns=["tgt_id", "e3"])
    hit_tgt_ids = set(ef.loc[ef["e3"] != 0.0, "tgt_id"].unique().tolist())
    mask = torch.zeros(N_NODES, dtype=torch.bool)
    if hit_tgt_ids:
        idx = torch.tensor(sorted(hit_tgt_ids), dtype=torch.long)
        mask[idx] = True
    return mask


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def build_event_graph(
    event: dict, edge_index: torch.Tensor, canonical_index: pd.MultiIndex
) -> TSPNEventGraph:
    event_name = event["name"]
    nf_path = os.path.join(NF_QUARTERLY_DIR, f"{event_name}.parquet")
    nf_df = pd.read_parquet(nf_path)

    temporal_sequence: list[Data] = []
    for idx in range(8):
        x = build_node_tensor(nf_df, idx)
        edge_attr = build_edge_tensor(event_name, idx, canonical_index)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        temporal_sequence.append(data)

    y, label_mask = build_label_tensors(event_name)
    direct_hit_mask = build_direct_hit_mask(event_name)

    return TSPNEventGraph(
        temporal_sequence=temporal_sequence,
        y=y,
        label_mask=label_mask,
        direct_hit_mask=direct_hit_mask,
        event_name=event_name,
        event_date=event["date"],
    )


def validate_event_graph(graph: TSPNEventGraph, edge_index: torch.Tensor) -> None:
    seq = graph.temporal_sequence
    assert len(seq) == 8, f"{graph.event_name}: expected 8 snapshots, got {len(seq)}"

    for q in range(8):
        assert seq[q].x.shape == (N_NODES, 9), f"{graph.event_name} q{q}: x shape {seq[q].x.shape}"
        assert not torch.isnan(seq[q].x).any(), f"{graph.event_name} q{q}: NaN in x"
        assert not torch.isnan(seq[q].edge_attr).any(), f"{graph.event_name} q{q}: NaN in edge_attr"
        assert torch.equal(seq[q].edge_index, edge_index), \
            f"{graph.event_name} q{q}: edge_index differs from the canonical template (CP22)"

    # CP21: shock isolated to snapshot 7 only
    for q in range(7):
        delta_sum = seq[q].edge_attr[:, 3].abs().sum().item()
        assert delta_sum == 0.0, f"{graph.event_name} snapshot {q}: shock leaks (sum={delta_sum})"
    assert seq[7].edge_attr[:, 3].abs().sum().item() > 0, \
        f"{graph.event_name} snapshot 7: no shock signal"

    # CP23: snapshots are distinct objects with distinct edge_attr (not aliased)
    assert id(seq[0]) != id(seq[7]), f"{graph.event_name}: snapshot 0 and 7 are the same object"
    assert not torch.equal(seq[0].edge_attr, seq[7].edge_attr), \
        f"{graph.event_name}: snapshot 0 and 7 have identical edge_attr"

    # CP24: no NaN among labeled nodes
    labeled_y = graph.y[graph.label_mask]
    assert not torch.isnan(labeled_y).any(), f"{graph.event_name}: NaN in labeled y (CP24)"

    n_direct = int(graph.direct_hit_mask.sum().item())
    print(f"    validated: {n_direct} direct-hit nodes, "
          f"{int(graph.label_mask.sum().item())}/{N_NODES} labeled")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building canonical edge_index from edges_2014.parquet (CP22)...")
    edge_index, canonical_index, canonical_df = build_canonical_edge_template()
    print(f"  Canonical edge count: {edge_index.shape[1]:,}")

    for event in config.EVENTS:
        event_name = event["name"]
        print(f"\nBuilding PyG dataset for {event_name} (date {event['date']})...")
        graph = build_event_graph(event, edge_index, canonical_index)
        validate_event_graph(graph, edge_index)

        out_path = os.path.join(OUT_DIR, f"{event_name}.pt")
        torch.save(graph, out_path)
        print(f"  Saved -> {out_path}")

    print(f"\nDone. {len(config.EVENTS)} PyG dataset files saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
