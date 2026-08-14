"""verify_phase8_training.py — Hard-gate verification of Phase 8 exit criteria.

Run AFTER a training run (even a short smoke test) has produced checkpoints
for all 6 folds. Checks source-level invariants (CP34's model.eval()-first
ordering, CP35's clip-before-step ordering) plus artifact-level invariants
(checkpoint metadata, attention files, results table).

Usage:
    python scripts/verify_phase8_training.py
"""

import inspect
import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_TRAINING = os.path.join(PROJECT_ROOT, "src", "training")
for p in (PROJECT_ROOT, SRC_TRAINING):
    if p not in sys.path:
        sys.path.insert(0, p)

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
ALL_RESULTS_PATH = os.path.join(RESULTS_TABLES, "all_results.csv")
N_NODES = config.GRAPH["N_NODES"]


def test_checkpoints_exist_with_metadata():
    for fold_idx in range(6):
        path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_best.pt")
        assert os.path.exists(path), f"missing {path}"
        ckpt = torch.load(path, weights_only=False)
        assert set(["epoch", "val_rmse_6m", "model_state"]).issubset(ckpt.keys()), \
            f"fold {fold_idx}: checkpoint is a bare state_dict, not {{epoch, val_rmse_6m, model_state}} (CP36)"
        assert isinstance(ckpt["epoch"], int), f"fold {fold_idx}: epoch not an int"
        assert ckpt["val_rmse_6m"] >= 0, f"fold {fold_idx}: negative val_rmse_6m"
check("CP36: all 6 checkpoints exist with {epoch, val_rmse_6m, model_state} metadata", test_checkpoints_exist_with_metadata)


def test_attention_files():
    for fold_idx in range(6):
        path = os.path.join(RESULTS_TABLES, f"attention_fold{fold_idx}.npy")
        assert os.path.exists(path), f"missing {path}"
        alpha = np.load(path)
        assert alpha.ndim == 2 and alpha.shape[1] == config.MODEL["gat_num_heads"], \
            f"fold {fold_idx}: attention shape {alpha.shape}"
        assert not np.isnan(alpha).any(), f"fold {fold_idx}: NaN in attention weights"
check("attention_fold{0-5}.npy exist, correct shape, no NaN", test_attention_files)


def test_train_logs():
    for fold_idx in range(6):
        path = os.path.join(RESULTS_TABLES, f"train_log_fold{fold_idx}.csv")
        assert os.path.exists(path), f"missing {path}"
        df = pd.read_csv(path)
        assert len(df) >= 1, f"fold {fold_idx}: empty train log"
        assert set(["epoch", "train_loss", "val_rmse_6m", "lr"]).issubset(df.columns), \
            f"fold {fold_idx}: train log missing expected columns"
        assert not df["train_loss"].isna().any(), f"fold {fold_idx}: NaN in train_loss log (CP35)"
check("train_log_fold{0-5}.csv exist with loss-curve columns, no NaN", test_train_logs)


def test_results_table():
    assert os.path.exists(ALL_RESULTS_PATH), f"missing {ALL_RESULTS_PATH}"
    df = pd.read_csv(ALL_RESULTS_PATH)
    tspn_rows = df[df["model_name"] == "TSPN"]
    assert len(tspn_rows) == 6, f"expected 6 TSPN rows, got {len(tspn_rows)}"
    assert set(tspn_rows["val_event"]) == set(e["name"] for e in config.EVENTS), \
        "TSPN val_event set doesn't match the 6 configured events"
    dupes = tspn_rows.groupby("fold").size()
    assert (dupes == 1).all(), f"CP39: duplicate fold rows: {dupes[dupes > 1].to_dict()}"
    metric_cols = [c for c in df.columns if c.startswith(("RMSE_", "MAE_", "R2_", "DirAcc_"))]
    assert not tspn_rows[metric_cols].isna().any().any(), "NaN in TSPN results row"
check("results/tables/all_results.csv: 6 TSPN rows, one per event, no dupes (CP39), no NaN", test_results_table)


def test_cp34_validate_ordering():
    from train import validate
    src = inspect.getsource(validate)
    lines = [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith(("def ", '"""', "#"))]
    assert lines[0].startswith("model.eval()"), \
        f"CP34 VIOLATION: validate()'s first statement is not model.eval() -- got: {lines[0]!r}"
    assert "augment_" not in src, "CP34 VIOLATION: validate() references an augmentation function"
check("CP34: validate() has model.eval() as its literal first statement, no augmentation calls", test_cp34_validate_ordering)


def test_cp35_clip_before_step():
    from train import train_one_fold
    src = inspect.getsource(train_one_fold)
    clip_pos = src.find("clip_grad_norm_")
    step_pos = src.find("optimizer.step()")
    assert clip_pos != -1 and step_pos != -1, "could not locate clip_grad_norm_/optimizer.step() in source"
    assert clip_pos < step_pos, "CP35 VIOLATION: gradient clipping happens AFTER optimizer.step()"
check("CP35: gradient clipping happens before optimizer.step() (source order)", test_cp35_clip_before_step)


def test_cp33_contamination_guard_present():
    from train import train_one_fold
    src = inspect.getsource(train_one_fold)
    assert "CONTAMINATION" in src and "raise RuntimeError" in src, \
        "CP33: no hard RuntimeError contamination guard found in train_one_fold"
check("CP33: hard RuntimeError contamination guard present in source", test_cp33_contamination_guard_present)


def test_cp38_warmup_present():
    assert config.TRAINING.get("grad_clip_warmup_norm") is not None
    assert config.TRAINING.get("grad_clip_warmup_epochs") is not None
    from train import train_one_fold
    src = inspect.getsource(train_one_fold)
    assert "grad_clip_warmup_norm" in src and "grad_clip_warmup_epochs" in src
check("CP38: grad-clip warmup schedule present in config.py and used in train.py", test_cp38_warmup_present)


print(f"\n{'='*60}")
print(f"Phase 8 verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 8 training infrastructure complete, ready for Phase 9")
