"""verify_phase5_pyg_datasets.py — Hard-gate verification of Phase 5 exit criteria.

Independently reloads data/pyg_datasets/{event}.pt from disk (not reusing
in-memory objects from build_pyg_dataset.py) and re-checks CP21-CP24 plus
the Locked Plan §5.3 dataset validation checklist. Run before starting
Phase 6.

Usage:
    python scripts/verify_phase5_pyg_datasets.py
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DATA not in sys.path:
    sys.path.insert(0, SRC_DATA)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401  (needed for unpickling)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")


N_NODES = config.GRAPH["N_NODES"]
PYG_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"])
EVENT_NAMES = [e["name"] for e in config.EVENTS]

_graphs: dict[str, TSPNEventGraph] = {}


def load_all():
    for name in EVENT_NAMES:
        path = os.path.join(PYG_DIR, f"{name}.pt")
        assert os.path.exists(path), f"missing {path}"
        _graphs[name] = torch.load(path, weights_only=False)
check("all 6 .pt files exist and load with torch.load(weights_only=False)", load_all)


def test_shapes():
    for name, g in _graphs.items():
        assert len(g.temporal_sequence) == 8, f"{name}: {len(g.temporal_sequence)} snapshots"
        for q, d in enumerate(g.temporal_sequence):
            assert d.x.shape == (N_NODES, 9), f"{name} q{q}: x shape {d.x.shape}"
            assert d.edge_attr.shape[1] == 6, f"{name} q{q}: edge_attr dim {d.edge_attr.shape[1]}"
            assert d.edge_index.shape[0] == 2, f"{name} q{q}: edge_index shape {d.edge_index.shape}"
        assert g.y.shape == (N_NODES, 3), f"{name}: y shape {g.y.shape}"
        assert g.label_mask.shape == (N_NODES,), f"{name}: label_mask shape {g.label_mask.shape}"
        assert g.direct_hit_mask.shape == (N_NODES,), f"{name}: direct_hit_mask shape {g.direct_hit_mask.shape}"
check("§5.3: shapes correct (x=[2408,9], edge_attr dim=6, y=[2408,3])", test_shapes)


def test_no_nan_in_x():
    for name, g in _graphs.items():
        for q, d in enumerate(g.temporal_sequence):
            assert not torch.isnan(d.x).any(), f"{name} q{q}: NaN in x"
check("§5.3: no NaN in any x tensor", test_no_nan_in_x)


def test_cp21_shock_isolation():
    for name, g in _graphs.items():
        for q in range(7):
            s = g.temporal_sequence[q].edge_attr[:, 3].abs().sum().item()
            assert s == 0.0, f"{name} q{q}: shock leaks into pre-event snapshot ({s})"
        s7 = g.temporal_sequence[7].edge_attr[:, 3].abs().sum().item()
        assert s7 > 0.0, f"{name} q7: no shock signal at event quarter"
check("CP21: shock (e3) zero in snapshots 0-6, nonzero only at snapshot 7", test_cp21_shock_isolation)


def test_cp22_fixed_edge_index():
    ref = None
    for name, g in _graphs.items():
        for q in range(8):
            ei = g.temporal_sequence[q].edge_index
            if ref is None:
                ref = ei
            assert torch.equal(ei, ref), f"{name} q{q}: edge_index differs from the global canonical template"
check("CP22: identical edge_index across all 8 snapshots of all 6 events", test_cp22_fixed_edge_index)


def test_cp23_no_aliasing():
    for name, g in _graphs.items():
        seq = g.temporal_sequence
        assert id(seq[0]) != id(seq[7]), f"{name}: snapshot 0 and 7 are the same Python object"
        assert not torch.equal(seq[0].edge_attr, seq[7].edge_attr), \
            f"{name}: snapshot 0 and 7 have identical edge_attr"
check("CP23: no Python object aliasing across snapshots (list-multiplication bug)", test_cp23_no_aliasing)


def test_cp24_no_nan_in_labeled_y():
    for name, g in _graphs.items():
        labeled_y = g.y[g.label_mask]
        assert not torch.isnan(labeled_y).any(), f"{name}: NaN in labeled y"
        unlabeled_y = g.y[~g.label_mask]
        assert torch.isnan(unlabeled_y).all(), f"{name}: unlabeled rows should be all-NaN"
check("CP24: no NaN in y[label_mask]; unlabeled rows are fully NaN", test_cp24_no_nan_in_labeled_y)


def test_label_coverage_60pct():
    for name, g in _graphs.items():
        cov = float(g.label_mask.float().mean().item())
        flag = "OK" if cov >= 0.60 else "below 60% (accepted per PROJECT_STATE.md decision 3)"
        print(f"    {name}: {cov:.1%} label coverage [{flag}]")
check("§5.3: label_mask coverage reported per event", test_label_coverage_60pct)


def test_direct_hit_reasonable():
    for name, g in _graphs.items():
        n_hit = int(g.direct_hit_mask.sum().item())
        assert 0 < n_hit < N_NODES, f"{name}: {n_hit} direct-hit nodes (suspicious)"
        print(f"    {name}: {n_hit} direct-hit nodes")
check("direct_hit_mask: nonzero and not implausibly large for every event", test_direct_hit_reasonable)


print(f"\n{'='*60}")
print(f"Phase 5 verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 5 complete, ready for Phase 6")
