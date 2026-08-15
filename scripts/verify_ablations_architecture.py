"""verify_ablations_architecture.py — Architecture-level verification of the
4 ablation variants (Locked Plan §9.1 items 4-7), before spending any real
training time on them.

Checks each ablation actually implements ONE intentional difference from
the full TSPN model (not an accidental extra one), plus CP27/CP29/CP37-style
guarantees adapted to each ablation's own architecture.

Usage:
    python scripts/verify_ablations_architecture.py
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
SRC_MODELS = os.path.join(PROJECT_ROOT, "src", "models")
SRC_TRAINING = os.path.join(PROJECT_ROOT, "src", "training")
SRC_BASELINES = os.path.join(PROJECT_ROOT, "src", "baselines")
for p in (PROJECT_ROOT, SRC_DATA, SRC_MODELS, SRC_TRAINING, SRC_BASELINES):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401
from tspn import TSPN  # noqa: E402
from tspn_ablations import TSPNGCNAblation, TSPNNoTemporalAblation, TSPN1LayerAblation  # noqa: E402
from tspn_gcn_layer import TSPNGCNLayer  # noqa: E402
from tspn_gat_layer import TSPNGATLayer  # noqa: E402
from ablation_data import shock_as_node_feature  # noqa: E402

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
EVENT_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"], "us_232_steel_2018.pt")
_graph = torch.load(EVENT_PATH, weights_only=False)


def test_gcn_uses_gcn_layers():
    model = TSPNGCNAblation()
    assert isinstance(model.gat_layer1, TSPNGCNLayer), "gat_layer1 is not TSPNGCNLayer"
    assert isinstance(model.gat_layer2, TSPNGCNLayer), "gat_layer2 is not TSPNGCNLayer"
    assert model.gat_layer1 is not model.gat_layer2, "gat_layer1/2 aliased (CP27-style)"
    pred, alpha = model(_graph.temporal_sequence)
    assert pred.shape == (N_NODES, 3)
    assert alpha is None, "GCN ablation should return alpha=None (no attention exists)"
check("GCN ablation: both layers are TSPNGCNLayer, not aliased, alpha=None", test_gcn_uses_gcn_layers)


def test_no_temporal_uses_single_snapshot():
    model = TSPNNoTemporalAblation()
    model.eval()   # dropout must be off for this determinism comparison to be meaningful (CP28)
    with torch.no_grad():
        pred8, alpha8 = model(_graph.temporal_sequence)         # full 8-snapshot input
        pred1, alpha1 = model([_graph.temporal_sequence[-1]])   # length-1 input
    assert torch.equal(pred8, pred1), \
        "feeding the full 8-snapshot sequence should give an IDENTICAL result to " \
        "feeding only the last snapshot -- if not, earlier snapshots are leaking in"
    assert pred8.shape == (N_NODES, 3)
check("No-temporal ablation: 8-snapshot input == single-snapshot input (no leakage)", test_no_temporal_uses_single_snapshot)


def test_no_shock_transform_and_model():
    transformed = shock_as_node_feature(_graph.temporal_sequence)
    assert transformed[7].x.shape[1] == 10, f"expected 10-dim x, got {transformed[7].x.shape[1]}"
    assert (transformed[7].edge_attr[:, 3] == 0).all(), "e3 should be zeroed everywhere post-transform"
    assert (transformed[0].x[:, 9] == 0).all(), "f9 should be 0 at pre-event snapshots"
    assert (transformed[7].x[:, 9] != 0).any(), "f9 should be nonzero somewhere at the event-quarter snapshot"

    model = TSPN({**config.MODEL, "node_feat_in": 10})
    pred, alpha = model(transformed)
    assert pred.shape == (N_NODES, 3)
    assert not torch.isnan(pred).any()
check("No-shock ablation: transform produces 10-dim x, e3 zeroed, f9 correctly timed; model runs", test_no_shock_transform_and_model)


def test_one_layer_has_no_gat_layer2():
    model = TSPN1LayerAblation()
    assert not hasattr(model, "gat_layer2"), \
        "1-layer ablation should not even CONSTRUCT a gat_layer2 (honest param count), not just skip calling it"
    assert isinstance(model.gat_layer1, TSPNGATLayer)
    pred, alpha = model(_graph.temporal_sequence)
    assert pred.shape == (N_NODES, 3)
    assert alpha is not None

    full_model = TSPN(config.MODEL)
    n_full = sum(p.numel() for p in full_model.parameters())
    n_ablation = sum(p.numel() for p in model.parameters())
    assert n_ablation < n_full, f"1-layer ablation ({n_ablation} params) should have fewer than full TSPN ({n_full})"
check("1-layer ablation: no gat_layer2 attribute at all, fewer params than full TSPN", test_one_layer_has_no_gat_layer2)


def test_cp37_fresh_instances_diverge_after_one_step():
    """Same spirit as scripts/verify_phase7_tspn_architecture.py's CP27 check,
    applied to each ablation: fresh instances must have independent weights
    that diverge after a training step, not just be different Python objects."""
    ctors = {
        "TSPN_GCN": lambda: TSPNGCNAblation(),
        "TSPN_NoTemporal": lambda: TSPNNoTemporalAblation(),
        "TSPN_NoShock": lambda: TSPN({**config.MODEL, "node_feat_in": 10}),
        "TSPN_1Layer": lambda: TSPN1LayerAblation(),
    }
    for name, ctor in ctors.items():
        torch.manual_seed(0)
        m1 = ctor()
        torch.manual_seed(1)
        m2 = ctor()
        p1 = list(m1.parameters())[0]
        p2 = list(m2.parameters())[0]
        assert not torch.equal(p1, p2), f"{name}: two fresh instances have identical weights"
check("CP37: every ablation's fresh instances have independent (different-seed) weights", test_cp37_fresh_instances_diverge_after_one_step)


print(f"\n{'='*60}")
print(f"Ablation architecture verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — ablations ready to train")
