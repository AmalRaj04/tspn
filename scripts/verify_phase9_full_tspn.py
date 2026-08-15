"""verify_phase9_full_tspn.py — Verification of the full-TSPN Phase 9 run.

Checks structural correctness of the real (not smoke-test) TSPN training
results now that all 6 folds have actually converged (on Colab). Same
spirit as scripts/verify_phase8_training.py but pointed at the real run
and additionally checking early-stopping sanity (patience actually hit 20)
and that results/tables/baselines.csv survived the local<->Drive file sync.

Usage:
    python scripts/verify_phase9_full_tspn.py
"""

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


def test_all_folds_converged():
    for fold_idx in range(6):
        path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_resume_state.json")
        assert os.path.exists(path), f"missing {path}"
        import json
        with open(path) as f:
            state = json.load(f)
        assert state["early_stopped"], f"fold {fold_idx}: did not early-stop (state={state})"
        assert state["patience_counter"] == config.TRAINING["early_stop_patience"], \
            f"fold {fold_idx}: patience_counter={state['patience_counter']}, expected {config.TRAINING['early_stop_patience']}"
        assert 0 < state["epoch"] < config.TRAINING["max_epochs"] - 1, \
            f"fold {fold_idx}: epoch={state['epoch']} looks suspicious (too low or hit max_epochs)"
check("all 6 folds genuinely early-stopped (not hit max_epochs, not degenerate)", test_all_folds_converged)


def test_results_csv():
    path = os.path.join(RESULTS_TABLES, "all_results.csv")
    df = pd.read_csv(path)
    tspn_rows = df[df["model_name"] == "TSPN"]
    assert len(tspn_rows) == 6, f"expected 6 TSPN rows, got {len(tspn_rows)}"
    assert set(tspn_rows["val_event"]) == set(EVENT_NAMES)
    metric_cols = [c for c in df.columns if c.startswith(("RMSE_", "MAE_", "R2_", "DirAcc_"))]
    assert not tspn_rows[metric_cols].isna().any().any(), "NaN in TSPN results"
    dupes = tspn_rows.groupby("fold").size()
    assert (dupes == 1).all(), f"duplicate fold rows: {dupes[dupes > 1].to_dict()}"
check("all_results.csv: 6 TSPN rows, all 6 events, no NaN, no dupes", test_results_csv)


def test_baselines_csv_intact():
    path = os.path.join(RESULTS_TABLES, "baselines.csv")
    assert os.path.exists(path), f"missing {path} (wiped by the Colab<->local results/tables sync?)"
    df = pd.read_csv(path)
    for model in ["Leontief_IO", "MLP_no_graph", "Panel_VAR"]:
        assert model in df["model_name"].values, f"{model} missing from baselines.csv"
check("baselines.csv survived the Colab upload (restored from git if not)", test_baselines_csv_intact)


def test_checkpoints_loadable():
    for fold_idx in range(6):
        path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_best.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert set(["epoch", "val_rmse_6m", "model_state"]).issubset(ckpt.keys())
        assert ckpt["val_rmse_6m"] > 0
check("all 6 checkpoints load cleanly with correct metadata (cross-device, Colab -> local)", test_checkpoints_loadable)


def test_attention_files():
    for fold_idx in range(6):
        path = os.path.join(RESULTS_TABLES, f"attention_fold{fold_idx}.npy")
        alpha = np.load(path)
        assert alpha.shape[1] == config.MODEL["gat_num_heads"]
        assert not np.isnan(alpha).any()
check("attention_fold{0-5}.npy present, correct shape, no NaN", test_attention_files)


print(f"\n{'='*60}")
print(f"Phase 9 (full TSPN) verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
