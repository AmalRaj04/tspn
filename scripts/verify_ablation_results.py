"""verify_ablation_results.py — Verification of the trained ablation results.

Same spirit as scripts/verify_phase9_full_tspn.py, extended to the 4
ablations. Checks all 24 (ablation, fold) resume states genuinely
converged, checkpoints load, and results/tables are complete and clean.

Usage:
    python scripts/verify_ablation_results.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

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


CKPT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["MODEL_CHECKPOINTS"])
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
EVENT_NAMES = [e["name"] for e in config.EVENTS]
ABLATION_NAMES = ["TSPN_GCN", "TSPN_NoTemporal", "TSPN_NoShock", "TSPN_1Layer"]
HAS_ATTENTION = {"TSPN_GCN": False, "TSPN_NoTemporal": True, "TSPN_NoShock": True, "TSPN_1Layer": True}


def test_all_ablation_folds_converged():
    for ablation in ABLATION_NAMES:
        for fold_idx in range(6):
            path = os.path.join(CKPT_DIR, f"{ablation}_fold{fold_idx}_resume_state.json")
            assert os.path.exists(path), f"missing {path}"
            with open(path) as f:
                state = json.load(f)
            assert state["early_stopped"], f"{ablation} fold {fold_idx}: did not early-stop ({state})"
            assert state["patience_counter"] == config.TRAINING["early_stop_patience"], \
                f"{ablation} fold {fold_idx}: patience_counter={state['patience_counter']}"
            assert 0 < state["epoch"] < config.TRAINING["max_epochs"] - 1, \
                f"{ablation} fold {fold_idx}: epoch={state['epoch']} looks suspicious"
check("all 24 (ablation, fold) pairs genuinely early-stopped", test_all_ablation_folds_converged)


def test_checkpoints_loadable():
    for ablation in ABLATION_NAMES:
        for fold_idx in range(6):
            path = os.path.join(CKPT_DIR, f"{ablation}_fold{fold_idx}_best.pt")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert set(["epoch", "val_rmse_6m", "model_state"]).issubset(ckpt.keys())
            assert ckpt["val_rmse_6m"] > 0
check("all 24 checkpoints load cleanly with correct metadata", test_checkpoints_loadable)


def test_attention_files_match_design():
    for ablation in ABLATION_NAMES:
        for fold_idx in range(6):
            path = os.path.join(RESULTS_TABLES, f"attention_{ablation}_fold{fold_idx}.npy")
            exists = os.path.exists(path)
            expected = HAS_ATTENTION[ablation]
            assert exists == expected, \
                f"{ablation} fold {fold_idx}: attention file exists={exists}, expected={expected}"
            if exists:
                alpha = np.load(path)
                assert not np.isnan(alpha).any()
check("attention files present only where expected (GCN has none, by design)", test_attention_files_match_design)


def test_results_csv():
    path = os.path.join(RESULTS_TABLES, "all_results.csv")
    df = pd.read_csv(path)
    for model in ["TSPN"] + ABLATION_NAMES:
        sub = df[df["model_name"] == model]
        assert len(sub) == 6, f"{model}: expected 6 rows, got {len(sub)}"
        assert set(sub["val_event"]) == set(EVENT_NAMES)
        dupes = sub.groupby("fold").size()
        assert (dupes == 1).all(), f"{model}: duplicate fold rows"
    metric_cols = [c for c in df.columns if c.startswith(("RMSE_", "MAE_", "R2_", "DirAcc_"))]
    assert not df[metric_cols].isna().any().any(), "NaN in results"
check("all_results.csv: TSPN + 4 ablations x 6 folds, no NaN, no dupes", test_results_csv)


def test_baselines_csv_intact():
    path = os.path.join(RESULTS_TABLES, "baselines.csv")
    assert os.path.exists(path), f"missing {path}"
    df = pd.read_csv(path)
    for model in ["Leontief_IO", "MLP_no_graph", "Panel_VAR"]:
        assert model in df["model_name"].values, f"{model} missing from baselines.csv"
check("baselines.csv intact (survived this Colab<->local sync)", test_baselines_csv_intact)


print(f"\n{'='*60}")
print(f"Ablation results verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
