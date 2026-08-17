# TSPN — Deviation Reconciliation + Phase-by-Phase Completion Plan

## Context

TSPN (Tariff Shock Propagation Network) is a research project building a temporal GNN that predicts sector-level price cascades from tariff shocks, backed by a research paper + Streamlit dashboard. Development followed five planning documents (Masterplan, Locked Implementation Plan, Risk Checkpoints, Complete Implementation Guide, Research Brief), but real-world data availability forced many deviations during Phases 0–4 (WIOD only goes to 2014 not 2016, HS→ISIC needed a CPC21 intermediate step, sector counts changed, country lists changed, etc.). The user tracked some of these deviations manually but lost track of others, and asked for: (1) a reconciliation of what actually deviated from the plan, (2) a full read of the actual codebase to establish ground truth, and (3) a phase-by-phase plan to complete the project correctly from here, addressing any newly-discovered defects before building further.

An inventory agent read every processed file's actual schema/row-counts and every source file's actual logic. That inventory (not the five planning docs) is the source of truth below. Three real bugs were found in the current HEAD (commit `4c2c8ca "audites and node(not working)"`) that must be fixed before Phase 5 can start; the user has made the calls on how to fix them (see Decisions Locked, below).

**Nothing has been coded yet.** This plan is for review; implementation starts only after approval, phase by phase.

---

## Section A — Consolidated Deviation Log (replaces the user's lost tracking)

This is the single canonical list of every deviation from the original 5 planning docs, merging what the user already knew with what the inventory agent found. This should become `PROJECT_STATE.md` in the repo root (currently the repo has no such file — `README.md` is a single blank line, and `audit_report.txt` is a stale one-off dump).

### A1. Already known to the user (confirmed still true in current code/data)
| # | Deviation | Where |
|---|---|---|
| 1 | Sectors merged: `J58`, `J59_J60` collapsed from planned `J58-J60`/`J59-J60`/`J61`/`J62-J63` scheme → final 56-sector list uses `J58`, `J59_J60`, `J62_J63` etc. | `config.py GRAPH["SECTOR_LIST"]` |
| 2 | WIOD years restricted to 2000–2014 (not 2000–2016) — official WIOD 2016 release's `.xlsb` files only go to 2014 in what was actually downloadable | `config.py`: `WIOD_YEARS = list(range(2000, 2015))` |
| 3 | Sector codes standardized to underscore form (`J59_J60`, `J62_J63`, `M69_M70`, `M74_M75`, `R_S`) instead of hyphens, `RoW`→`ROW` | `config.py GRAPH["SECTOR_LIST"]`, `COUNTRY_LIST` |
| 4 | `ZAF` replaced with `HRV` in the 44-country list | `config.py GRAPH["COUNTRY_LIST"]` |
| 5 | HS→ISIC concordance needed an intermediate step: HS6 → **CPC21** → ISIC4 → WIOD56 (guide assumed a direct `hs2017_isic4.xlsx`, which doesn't exist as a clean download) | `src/data/build_concordance.py`, `data/raw/concordance/CPC21-HS2017.csv`, `cpc21-hs2012.txt`, `isic4-cpc21.txt` |
| 6 | Section 232 datasets generated directly from Proclamations 9704/9705 + HTSUS Rev 1.2, not from the guide's Federal Register PDF estimates. Real counts: Steel = 303 HTS-8 codes (guide estimated ~170), Aluminum = 36 (guide estimated ~60) | `data/raw/tariff_events/us_232_steel_2018.csv` (303 rows), `us_232_aluminum_2018.csv` (36 rows) — confirmed present |
| 7 | Two independent tariff data streams, not one: shock events (e[3], primary causal signal, from Federal Register/EU OJ/UKGT) vs. WITS schedules (e[2]/f[4], background context only) | Architecturally correct as implemented |
| 8 | Eurostat PPI: correct indicator is `PRC_PRR_DOM`, not `PRIN` as the guide stated. NACE B/C/D present; E and H not present in `sts_inppd_m` | `src/data/clean_ppi.py` — confirmed uses correct indicator |
| 9 | Graph extended 2015–2021 via Comtrade using **WIOD-2014** (not WIOD-2016) as the frozen structural prior — topology, sector shares, node IDs frozen; only bilateral flow *magnitudes* updated | `src/data/extend_with_comtrade.py` |
| 10 | Sector tariffs and shock deltas computed as **weighted averages** (Σweight·rate / Σweight), not summed — prevents inflation when many HS6 codes map to one sector | `src/data/compute_tariff_rates.py`, `build_shock_vectors.py` — confirmed |
| 11 | WITS coverage expanded to 20 reporter countries; final tariff coverage 78.2% observed / 15.0% interpolated / 6.8% missing | `data/processed/tariff_rates/sector_tariffs.parquet` — confirmed: 12,320 rows, observed=9632, interpolated=1848, missing=840 (6.8%) |

### A2. Newly discovered by this session's codebase inventory (user was not aware)
| # | Finding | Severity | Where |
|---|---|---|---|
| 12 | **ROW node has no features.** Leontief/technical-coefficient matrices are built 44×56=2464 (ROW included), but `node_features_*.parquet` is 43×56=2408 rows — ROW is silently absent because WIOD's socioeconomic accounts file has no ROW row. Any code assuming 2464 nodes uniformly will break. | Critical | `compute_node_features.py` vs `compute_leontief.py` |
| 13 | **Sector-code convention split.** `config.py`, `sector_tariffs.parquet`, and `shock_*.parquet` use underscores (`C10_C12`); `node_features_*.parquet` persists the **raw hyphenated** WIOD form (`C10-C12`) despite having an internal `normalize_sector()` shim — the shim is used for lookups but never applied to the persisted column. Any join on 2 of the 56 sector codes silently drops rows. | Critical | `compute_node_features.py` |
| 14 | **f4 (tariff_exposure) has zero temporal variation.** Exactly 354 unique values repeated identically across all 15 years 2000–2014, because the only tariff data available (WITS 2017–2021) is averaged once into a static per-(country,sector) dict and stamped onto every year. This is what the HEAD commit message ("not working") refers to. | Critical | `compute_node_features.py::build_tariff_lookup()` |
| 15 | **PPI-lag coverage is sparse.** `has_ppi_lags` true for only 12.3%–27.2% of node-years (73–88% default to 0.0), because `ppi_quarterly_all.parquet` only covers 29/44 countries (EU members + USA + WLD benchmark) and 25/56 sectors. 15 countries incl. CHN, JPN, GBR, CAN, KOR, IND, BRA, RUS, TUR have **zero** PPI coverage. | High | `data/processed/labels/ppi_quarterly_all.parquet` |
| 16 | **Node features only exist for 2000–2014.** No `node_features_2015..2021.parquet` exists at all — yet all 6 tariff events occur in 2018 or 2021, i.e. exactly the years with no node features. This blocks Phase 5 entirely until fixed. | Critical | `data/processed/node_features/` (only 15 files, 2000–2014) |
| 17 | **Comtrade-extension `import_pen_coeff` never recomputed for 2015–2021** despite the module's own docstring claiming it rescales. Verified byte-identical `import_pen_coeff` for the same edge across 2015/2018/2021. Only `flow_usd` for cross-border edges is rescaled; domestic flows are frozen at 2014 values (this part is by design and documented). | Medium | `src/data/extend_with_comtrade.py` |
| 18 | **No quarterly interpolation layer exists.** All processed data (edges, node features, Leontief) is annual, but the locked spec requires `SEQ_LEN=8` quarterly snapshots per event. Nothing in the repo currently produces quarterly graphs. | Critical (blocks Phase 5) | n/a — file doesn't exist |
| 19 | **Edge features are 89% incomplete.** `edges_*.parquet` has only 2 of the 6 config-defined edge dims (`flow_usd` raw, `import_pen_coeff`). `applied_tariff`, `tariff_delta`, `product_hhi`, `domestic_flag` are never materialized, and `log_trade_flow` is not actually log-transformed at rest (transform must happen at feature-build time). No `compute_edge_features.py` exists. | Critical (blocks Phase 5) | n/a — file doesn't exist |
| 20 | **No standalone label generator.** `generate_labels.py` doesn't exist; label logic is embedded inside `clean_ppi.py` and doesn't yet implement the event-relative 3m/6m/12m computation from Locked Plan §4.4 (CP17). | Critical (blocks Phase 5) | n/a — file doesn't exist |
| 21 | **`LEONTIEF["PASS_THROUGH_RATE"]` still `None`** — expected, since calibration is a Phase 6 step, but flagging so it isn't forgotten. | Low (expected) | `config.py` |
| 22 | **Phases 5–13 are 100% unimplemented.** `src/models/`, `src/baselines/`, `src/training/`, `src/analysis/`, `app/` contain only empty `__init__.py` files / empty dirs. `data/pyg_datasets/`, `models/checkpoints/`, `models/onnx/`, `results/` are all empty. This is not a deviation, just the honest current boundary of the project — Phase 4 (feature engineering) is where real work stops. | n/a | confirmed via repo-wide search |

---

## Section B — Decisions Locked This Session

1. **ROW handling**: Drop ROW entirely. Graph becomes **43 countries × 56 sectors = 2,408 nodes**. `config.py GRAPH` (`N_COUNTRIES`, `N_NODES`, `COUNTRY_LIST`) updated; Leontief/technical-coefficient matrices re-sliced (not reparsed — see Phase 4 plan) to drop ROW's row/col before re-inverting.
2. **Sector naming**: Underscore form (`C10_C12`) is the project standard everywhere. `compute_node_features.py` fixed to persist `normalize_sector()` output instead of raw WIOD labels.
3. **Label coverage**: Accept current 29/44-country PPI coverage as a documented, scoped limitation. Compute actual per-event label coverage in Phase 4 continuation; report honestly in the paper rather than delaying for more data sources.
4. **f4 (tariff_exposure) redesign**: `f4 = 0.0` for years with no WITS coverage (2000–2014, flagged `has_tariff_data=False`), and genuine trade-value-weighted tariff exposure for 2015–2021 where `sector_tariffs.parquet` actually has data.

---

## Section C — Phase-by-Phase Plan Going Forward

### Phase 4-fix — Repair and Complete Feature Engineering (next immediate work)
**Goal**: Fix the 3 critical bugs (§A2 #12–14), extend feature coverage into the event years (#16), and build the two missing Phase-4 scripts (edge features, labels) plus the quarterly interpolation layer (#18). Everything downstream depends on this being correct.

**C4.1 — `config.py` graph-size update**
- `N_COUNTRIES: 44→43`, `N_NODES: 2464→2408`, remove `"ROW"` from `COUNTRY_LIST`.
- Recompute `node_id()` mapping is unaffected (still `country_idx*56 + sector_idx`), but all persisted `.npy`/`.npz` matrices need re-slicing (next step).

**C4.2 — Re-slice technical coefficients & Leontief matrices to drop ROW**
- New/modified: `src/data/build_technical_coefficients.py`, `src/data/compute_leontief.py`.
- For each year 2000–2014: load existing `A_{year}.npz` (2464×2464 sparse), drop the 56 rows/cols belonging to ROW → 2408×2408. **Do not slice the existing Leontief inverse** (mathematically invalid); re-run `L = inv(I − A_sliced + εI)` on the reduced matrix and re-save `leontief_{year}.npy` (2408,2408) and `backward_linkage_{year}.npy` (2408,). This is cheap (fewer entries) — not a reparse of WIOD Excel.
- Also re-slice `edges_{year}.parquet` for all years (2000–2021) to drop rows/cols touching ROW, and remap `src_id`/`tgt_id` to the new 2408-node indexing.

**C4.3 — Fix `compute_node_features.py`**
- Persist `normalize_sector()` output (underscore form) as the `sector` column instead of raw WIOD labels — closes #13.
- Rewrite `build_tariff_lookup()`: return `(rate, has_tariff_data)` per (country,sector,year); for years without WITS coverage return `(0.0, False)`; only look up real `sector_tariffs.parquet` values for years where they exist — closes #14.
- **Extend the script to run for years 2015–2021**, not just 2000–2014 — closes #16. This requires:
  - Gross output / value added proxy for 2015–2021 (WIOD socioeconomic accounts stop at 2014): hold 2014 values with a documented note, OR scale by IMF World Economic Outlook nominal-GDP growth per country (light lookup, already-free data) — recommend the GDP-scaling approach since it's a small addition and materially better than a frozen constant; document the approximation explicitly (same spirit as the existing Comtrade-extension design decision).
  - Backward linkage for 2015–2021: reuse `leontief_2014.npy`'s backward linkage vector (structural prior frozen, consistent with `extend_with_comtrade.py`'s existing documented design) — no new inversion needed.
  - PPI lags for 2015–2021: pull directly from `ppi_quarterly_all.parquet` (already covers up to 2024).
- Output: `node_features_{2000..2021}.parquet`, 2,408 rows each, `sector` column underscore form, `f4` per the new rule, `has_tariff_data` and `has_ppi_lags` boolean flags both present.
- Recompute `normalization_stats.json` **strictly from 2000–2014** (training period only) — this is CP16 from the original Risk Checkpoints doc and is correctly scoped already; just re-verify after the above changes.

**C4.4 — Build `src/data/compute_edge_features.py`** (new file — closes #19)
- For each event × each of 8 quarterly snapshots (depends on C4.6 quarterly layer):
  - `e0 = log(flow_usd + 1)` (transform at build time, not at rest)
  - `e1 = import_pen_coeff` (carried through)
  - `e2 = applied_tariff` — join `sector_tariffs.parquet` on (src_country, tgt_country would not exist there — `sector_tariffs.parquet` is per-country not bilateral per the actual schema; use the importer/reporter country's rate as the applied tariff on that sector, consistent with how MFN tariffs are actually structured — reporter charges the same MFN rate to all partners unless in a bilateral shock)
  - `e3 = tariff_delta` — **zero for snapshots 0–6, injected only at snapshot 7** from the relevant `shock_{event}.parquet` (CP21 requirement)
  - `e4 = product_hhi` — Herfindahl index of trade concentration across HS6 codes per bilateral-sector pair, computed from raw Comtrade HS6-level files (needs a new aggregation step since current Comtrade parquet files are HS2-level per the inventory — verify HS6 granularity is available before committing to this; if only HS2 is on disk, document HHI as computed at HS2 granularity as a scoped limitation)
  - `e5 = domestic_flag` — trivial from src_country==tgt_country
- Output: `data/processed/edge_features/edge_features_{event_name}_q{0-7}.parquet`

**C4.5 — Build `src/data/generate_labels.py`** (new file — closes #20)
- Implements Locked Plan §4.4 exactly: for each event at date `d_e`, `delta_{3m,6m,12m} = (ppi[d_e+k] - ppi[d_e]) / ppi[d_e]`, quarter-aligned to the event's actual quarter (CP17 — the most commonly-missed checkpoint in the original risk doc).
- `has_label = False` unless all three horizons are non-null (CP24).
- After running: **compute and report actual label coverage per event** (Decision #3) — print `{event}: {coverage:.1%}` and flag any event below 60% in a short markdown table in `PROJECT_STATE.md`.

**C4.6 — Build quarterly interpolation layer** (new file, e.g. `src/data/build_quarterly_snapshots.py` — closes #18)
- Linear interpolation between adjacent annual node-feature/edge snapshots to produce the 8 quarterly graphs ending at each event's quarter, per Locked Plan §4.3/§5.2.
- This is a genuinely new piece of engineering not covered by any existing script — flagging it clearly since it's easy to underestimate.

**Phase 4-fix exit criteria**
- [ ] `config.py` reflects 2,408-node graph
- [ ] All `edges_*.parquet`, `leontief_*.npy`, `backward_linkage_*.npy` re-sliced/recomputed, ROW-free, consistent shapes
- [ ] `node_features_{2000..2021}.parquet` exist, 2,408 rows each, underscore sector codes, f4 fixed
- [ ] `edge_features_{event}_q{0-7}.parquet` exist for all 6 events × 8 quarters (48 files)
- [ ] `labels_{event}.parquet` exist for all 6 events with documented coverage %
- [ ] `normalization_stats.json` recomputed from 2000–2014 only, with `computed_from_years` key (CP16)
- [ ] Write `PROJECT_STATE.md` to repo root consolidating Section A + B of this plan, so this reconciliation is never lost again

---

### Phase 5 — PyG Graph Dataset Construction
**Goal**: `data/pyg_datasets/{event_name}.pt` for all 6 events (currently zero files exist).
- New: `src/data/build_pyg_dataset.py`.
- Fixed node/edge index (sorted by src_id,tgt_id) shared across all 8 snapshots of all 6 events (CP22).
- `copy.deepcopy()` per snapshot, never list multiplication (CP23 — a classic and easy bug given Python semantics).
- `direct_hit_mask` = target of a directly-shocked edge only, not downstream nodes (CP41-adjacent care).
- Validate: e3 zero in snapshots 0–6, non-zero only in 7, per event (CP21) — hard assertion, not a warning.

### Phase 6 — Baseline Models
- `src/baselines/leontief_io.py`: calibrate `LEONTIEF_PASS_THROUGH_RATE` on the UK Global Tariff 2021 event only (locked procedure), then apply uniformly; exclude that event from Leontief's own reported fold average (CP26).
- `src/baselines/panel_var.py`: statsmodels VAR, lag order 4, per (country,sector).
- `src/baselines/mlp_no_graph.py`: excludes f3 (backward linkage) since it's graph-derived (CP31); adds one scalar direct-tariff-exposure feature instead.

### Phase 7 — TSPN Architecture
- `src/models/feature_embedding.py`, `tspn_gat_layer.py` (custom MessagePassing subclass, edge features in attention, NOT standard GATConv), `tspn_gru.py`, `output_head.py`, `tspn.py` (full assembly) — all per Locked Plan §7, dimensions already frozen in `config.py MODEL`.
- Watch two classic bugs called out in Risk Checkpoints: gat_layer1/gat_layer2 must be separate constructor calls, not aliased (CP27); attention scores must be scaled by `1/sqrt(head_dim)` before softmax (CP30).

### Phase 8 — Training Infrastructure
- `src/training/losses.py`, `augmentation.py`, `train.py` (LOEO-CV, 6 folds), `evaluate.py`.
- Hard-fail assertion on LOEO contamination every fold (CP33) and on stochastic eval-mode output (CP34) — both marked ★ critical in the original risk doc for good reason.

### Phase 9 — Experiments
- Fixed run order: Leontief → VAR → MLP → GCN ablation → no-temporal ablation → no-shock ablation → 1-layer ablation → full TSPN.
- Re-init model + optimizer inside each ablation loop iteration (CP37).

### Phase 10 — Interpretability & Analysis
- `src/analysis/interpretability.py` (attention-vs-Leontief correlation — verify index order `L[src,tgt]` not transposed, CP40), `cascade_depth.py`, `amplifier_sectors.py`.

### Phase 11 — Paper Writing
- Fixed section order per Locked Plan §11 (Data → Model → Experiments → Interpretability → Related Work → Intro → Conclusion).

### Phase 12 — Product MVP (Streamlit dashboard)
- Currently 0 files under `app/`. Build per Locked Plan §12.2 day-by-day feature order once Phase 7–10 produce a trained model to export via ONNX.

### Phase 13 — Maintenance
- As originally scoped; not time-sensitive now.

---

## Immediate Next Step

Start with **Phase 4-fix** (C4.1–C4.6 above) since Phases 5–13 are all blocked on it. Recommend doing it as a sequence of small, independently-verifiable script changes (matching the project's existing audit-script culture — each fix should get a corresponding `scripts/audit_*.py` verification, consistent with how the rest of Phase 1–4 was built) rather than one large rewrite.

## Verification Approach
- Reuse the project's own pattern: every fix gets a paired `scripts/audit_*.py` (already exists for most Phase 4 concerns) or `scripts/check_*.py` that asserts the specific bug is closed (e.g. `audit_node_ids.py` should assert exactly 2408 rows and 0 hyphenated sector codes after C4.3).
- Before Phase 5 starts, re-run `scripts/audit_project_state.py` (already exists) and refresh `audit_report.txt` so it stops being stale.
- Adopt the original Risk Checkpoints doc's ★-critical checks (CP16, CP17, CP21, CP22, CP23) as hard `assert`/`raise` gates in the new scripts, not just print statements — several of the bugs found this session (f4, ROW, sector naming) are exactly the class of "silent wrong numbers, pipeline keeps running" failure that doc warns about.
