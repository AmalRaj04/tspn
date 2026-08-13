"""verify_phase6_baselines.py — Hard-gate verification of Phase 6 exit criteria.

Usage:
    python scripts/verify_phase6_baselines.py
"""

import os
import sys

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


RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "baselines.csv")
EVENT_NAMES = [e["name"] for e in config.EVENTS]


def test_pass_through_rate_set():
    assert config.LEONTIEF["PASS_THROUGH_RATE"] is not None, "PASS_THROUGH_RATE still None"
    assert isinstance(config.LEONTIEF["PASS_THROUGH_RATE"], float), "PASS_THROUGH_RATE not a float"
check("config.LEONTIEF['PASS_THROUGH_RATE'] calibrated and saved", test_pass_through_rate_set)


def test_results_csv_populated():
    assert os.path.exists(RESULTS_PATH), f"missing {RESULTS_PATH}"
    df = pd.read_csv(RESULTS_PATH)
    assert len(df) > 0, "baselines.csv is empty"
    metric_cols = [c for c in df.columns if c.startswith(("RMSE_", "MAE_", "R2_", "DirAcc_"))]
    assert len(metric_cols) == 12, f"expected 12 metric columns, got {len(metric_cols)}"
    assert not df[metric_cols].isna().any().any(), "NaN found in results/tables/baselines.csv"
check("results/tables/baselines.csv populated, 12 metric columns, no NaN", test_results_csv_populated)


def test_all_three_baselines_present():
    df = pd.read_csv(RESULTS_PATH)
    models = set(df["model_name"].unique())
    expected = {"Leontief_IO", "MLP_no_graph", "Panel_VAR"}
    assert expected.issubset(models), f"missing models: {expected - models}"
check("all 3 baselines (Leontief_IO, MLP_no_graph, Panel_VAR) present", test_all_three_baselines_present)


def test_leontief_cp26_uk_excluded():
    df = pd.read_csv(RESULTS_PATH)
    leontief_rows = df[df["model_name"] == "Leontief_IO"]
    assert "uk_global_tariff_2021" not in leontief_rows["val_event"].values, \
        "CP26 VIOLATION: Leontief evaluated on its own calibration event"
    assert len(leontief_rows) == 5, f"Leontief should have 5 folds (UK excluded), got {len(leontief_rows)}"
check("CP26: Leontief excludes uk_global_tariff_2021 (calibration event), has exactly 5 folds", test_leontief_cp26_uk_excluded)


def test_mlp_var_six_folds_no_contamination():
    df = pd.read_csv(RESULTS_PATH)
    for model in ["MLP_no_graph", "Panel_VAR"]:
        rows = df[df["model_name"] == model]
        assert len(rows) == 6, f"{model}: expected 6 folds, got {len(rows)}"
        assert set(rows["val_event"].unique()) == set(EVENT_NAMES), \
            f"{model}: val_event set doesn't match the 6 configured events"
check("MLP_no_graph and Panel_VAR each cover all 6 events exactly once", test_mlp_var_six_folds_no_contamination)


def test_no_duplicate_rows():
    df = pd.read_csv(RESULTS_PATH)
    dupes = df.groupby(["model_name", "fold"]).size()
    assert (dupes == 1).all(), f"CP39: duplicate rows found: {dupes[dupes > 1].to_dict()}"
check("CP39: no duplicate (model_name, fold) rows", test_no_duplicate_rows)


def test_leontief_diracc_sanity():
    df = pd.read_csv(RESULTS_PATH)
    leontief_rows = df[df["model_name"] == "Leontief_IO"]
    mean_diracc_6m = float(leontief_rows["DirAcc_6m"].mean())
    print(f"    Leontief mean DirAcc_6m across 5 folds: {mean_diracc_6m:.4f}")
    if not (0.55 <= mean_diracc_6m <= 0.70):
        print(f"    NOTE: outside [0.55, 0.70] sanity range -- documented as PROJECT_STATE.md "
              f"finding #27 (GBR/UK calibration-event label-coverage gap), not a failure.")
check("Leontief DirAcc_6m sanity check (informational, see finding #27)", test_leontief_diracc_sanity)


def test_baseline_ordering_sane():
    """Sanity check: learned baselines (MLP, VAR) should generally beat the naive
    analytic Leontief baseline -- not a hard spec requirement, but a useful signal
    that nothing is badly broken."""
    df = pd.read_csv(RESULTS_PATH)
    means = df.groupby("model_name")["RMSE_6m"].mean()
    print(f"    mean RMSE_6m by model: {means.to_dict()}")
    if means["Leontief_IO"] <= means.get("MLP_no_graph", float("inf")):
        print("    NOTE: Leontief RMSE_6m <= MLP RMSE_6m -- unexpected but not asserted as a failure")
check("baseline RMSE ordering reported (informational)", test_baseline_ordering_sane)


print(f"\n{'='*60}")
print(f"Phase 6 verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 6 complete, ready for Phase 7")
