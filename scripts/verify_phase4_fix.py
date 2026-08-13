"""verify_phase4_fix.py — Hard-gate verification of the Phase 4-fix exit criteria.

Checks every item in the approved Phase 4-fix plan (see PROJECT_STATE.md).
Every check is a hard assertion, not a warning — per PROJECT_STATE.md's own
lesson (f4/ROW/sector-naming were all "pipeline keeps running with silently
wrong numbers" failures). Run before starting Phase 5.

Usage:
    python scripts/verify_phase4_fix.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

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
N_COUNTRIES = config.GRAPH["N_COUNTRIES"]
WIOD_YEARS = sorted(config.GRAPH["WIOD_YEARS"])
COMTRADE_YEARS = sorted(config.GRAPH["COMTRADE_YEARS"])
ALL_YEARS = sorted(set(WIOD_YEARS) | set(COMTRADE_YEARS))


# ---------------------------------------------------------------------------
def test_config_2408():
    assert N_COUNTRIES == 43, f"N_COUNTRIES={N_COUNTRIES}"
    assert N_NODES == 2408, f"N_NODES={N_NODES}"
    assert "ROW" not in config.GRAPH["COUNTRY_LIST"], "ROW still in COUNTRY_LIST"
    assert len(config.GRAPH["COUNTRY_LIST"]) == 43
    assert len(config.GRAPH["SECTOR_LIST"]) == 56
check("config.py reflects 2408-node graph", test_config_2408)


def test_edges_row_free_and_shaped():
    for year in ALL_YEARS:
        path = f"data/processed/edges/edges_{year}.parquet"
        assert os.path.exists(path), f"missing {path}"
        df = pd.read_parquet(path, columns=["src_country", "tgt_country", "src_id", "tgt_id"])
        assert "ROW" not in set(df["src_country"].astype(str)) | set(df["tgt_country"].astype(str)), \
            f"{year}: ROW still present"
        assert df["src_id"].between(0, N_NODES - 1).all(), f"{year}: src_id out of range"
        assert df["tgt_id"].between(0, N_NODES - 1).all(), f"{year}: tgt_id out of range"
check("edges_{2000..2021}.parquet ROW-free, ids in range", test_edges_row_free_and_shaped)


def test_leontief_shapes():
    for year in WIOD_YEARS:
        L = np.load(f"data/processed/leontief/leontief_{year}.npy")
        bl = np.load(f"data/processed/leontief/backward_linkage_{year}.npy")
        assert L.shape == (N_NODES, N_NODES), f"{year}: L shape {L.shape}"
        assert bl.shape == (N_NODES,), f"{year}: bl shape {bl.shape}"
        assert not np.isnan(L).any() and not np.isnan(bl).any(), f"{year}: NaN present"
        cond_proxy = np.max(np.abs(L))
        assert cond_proxy < 100, f"{year}: max|L|={cond_proxy:.1f} (expected <100)"
check("leontief/backward_linkage: 2408x2408, no NaN, well-conditioned", test_leontief_shapes)


def test_node_features_all_years():
    prev_stats = None
    for year in ALL_YEARS:
        path = f"data/processed/node_features/node_features_{year}.parquet"
        assert os.path.exists(path), f"missing {path}"
        df = pd.read_parquet(path)
        assert len(df) == N_NODES, f"{year}: {len(df)} rows (expected {N_NODES})"
        sectors = set(df["sector"].astype(str).unique())
        assert sectors.issubset(set(config.GRAPH["SECTOR_LIST"])), \
            f"{year}: sector codes not in config list: {sectors - set(config.GRAPH['SECTOR_LIST'])}"
        hyphenated = [s for s in sectors if "-" in s]
        assert not hyphenated, f"{year}: hyphenated sector codes remain: {hyphenated}"
        for c in [f"f{i}" for i in range(9)]:
            assert not df[c].isna().any(), f"{year}: NaN in {c}"
            assert (df[c].abs() < 1e5).all(), f"{year}: extreme value in {c} (max={df[c].abs().max()})"
        if year < 2015:
            assert (df["f4"] == 0.0).all(), f"{year}: f4 should be 0.0 pre-2015 (no WITS coverage)"
        if year >= 2017:
            assert (df["f4"] != 0.0).any(), f"{year}: f4 all-zero for a year with WITS coverage"
check("node_features 2000-2021: 2408 rows, underscore sectors, f4 correct, no NaN/extreme", test_node_features_all_years)


def test_normalization_stats():
    with open(config.PATHS["NORM_STATS"]) as f:
        stats = json.load(f)
    assert stats["computed_from_years"] == WIOD_YEARS, \
        f"LEAKAGE: computed_from_years={stats['computed_from_years']}"
    assert "f4" in stats["identity_features"], "f4 should be excluded from z-scoring"
    assert stats["std"][4] == 1.0 and stats["mean"][4] == 0.0, "f4 identity marker wrong"
    assert "winsorize_bounds" in stats, "winsorize_bounds missing"
check("normalization_stats.json: CP16 (2000-2014 only), f4 identity, winsorize bounds present", test_normalization_stats)


def test_edge_features_48_files():
    events = [e["name"] for e in config.EVENTS]
    assert len(events) == 6
    for ev in events:
        for q in range(8):
            path = f"data/processed/edge_features/edge_features_{ev}_q{q}.parquet"
            assert os.path.exists(path), f"missing {path}"
            df = pd.read_parquet(path, columns=["e3"])
            if q < 7:
                assert (df["e3"] == 0.0).all(), f"{ev} q{q}: shock leaked pre-event (CP21)"
            else:
                assert (df["e3"] != 0.0).any(), f"{ev} q7: no shock signal at event quarter"
check("edge_features: 48 files (6 events x 8 quarters), CP21 shock isolation verified", test_edge_features_48_files)


def test_labels_all_events():
    for ev in config.EVENTS:
        path = f"data/processed/labels/labels_{ev['name']}.parquet"
        assert os.path.exists(path), f"missing {path}"
        df = pd.read_parquet(path)
        assert len(df) == N_NODES, f"{ev['name']}: {len(df)} rows"
        labeled = df[df["has_label"]]
        assert not labeled[["delta_3m", "delta_6m", "delta_12m"]].isna().any().any(), \
            f"{ev['name']}: NaN in labeled rows (CP24 violation)"
check("labels: all 6 events present, CP24 (no NaN in labeled rows)", test_labels_all_events)


def test_quarterly_node_snapshots():
    for ev in config.EVENTS:
        path = f"data/processed/node_features_quarterly/{ev['name']}.parquet"
        assert os.path.exists(path), f"missing {path}"
        df = pd.read_parquet(path)
        assert df["snapshot_idx"].nunique() == 8
        for idx in range(8):
            assert len(df[df["snapshot_idx"] == idx]) == N_NODES
check("node_features_quarterly: 8 snapshots x 2408 nodes per event", test_quarterly_node_snapshots)


def test_tariff_shock_row_free():
    tar = pd.read_parquet("data/processed/tariff_rates/sector_tariffs.parquet", columns=["country"])
    assert "ROW" not in set(tar["country"].astype(str)), "ROW still in sector_tariffs.parquet"
    for ev in config.EVENTS:
        sv = pd.read_parquet(
            f"data/processed/shock_vectors/shock_{ev['name']}.parquet",
            columns=["src_country", "tgt_country"],
        )
        touched = set(sv["src_country"].astype(str)) | set(sv["tgt_country"].astype(str))
        assert "ROW" not in touched, f"{ev['name']}: ROW still in shock vector"
check("sector_tariffs.parquet and shock_*.parquet are ROW-free", test_tariff_shock_row_free)


print(f"\n{'='*60}")
print(f"Phase 4-fix verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 4-fix complete, ready for Phase 5")
