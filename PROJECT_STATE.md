# TSPN — Project State & Deviation Log

**This document is the source of truth whenever it conflicts with the original planning docs** (`TSPN_Implementation_Masterplan.md`, `TSPN_Locked_Implementation_Plan.md`, `TSPN_Risk_Checkpoints.md`, `TSPN_Complete_Implementation_Guide.md`, `TSPN_Research_Brief.md` — kept outside the repo). Those documents describe the *intended* design; real-world data availability forced deviations during implementation. This file tracks every deviation, why it happened, and the current, authoritative state of the pipeline. Update it whenever a design decision changes.

Last reconciled: 2026-08-13.

---

## 1. Deviations from the original plan (confirmed against actual code/data)

### 1.1 Known before this reconciliation

| # | Deviation | Reason | Where |
|---|---|---|---|
| 1 | Sectors merged: planned `J58-J60`/`J59-J60` scheme collapsed into final list `J58`, `J59_J60` | Matches actual WIOD sector breakdown available | `config.py GRAPH["SECTOR_LIST"]` |
| 2 | WIOD years restricted to **2000–2014**, not 2000–2016 | Official WIOD 2016-release `.xlsb` files actually available only go to 2014 | `config.py`: `WIOD_YEARS = list(range(2000, 2015))` |
| 3 | Sector codes standardized to **underscore** form (`J59_J60`, `M69_M70`, `M74_M75`, `R_S`) instead of hyphens; `RoW` renamed `ROW` | Consistency / string-safety | `config.py` |
| 4 | `ZAF` replaced with `HRV` in the 44-country list | Data availability | `config.py GRAPH["COUNTRY_LIST"]` (superseded — see §2, ROW now also dropped) |
| 5 | HS→ISIC concordance needed an intermediate step: **HS6 → CPC21 → ISIC4 → WIOD56** | No clean direct `hs2017_isic4.xlsx` download exists; UN Stats only publishes HS↔CPC and CPC↔ISIC tables | `src/data/build_concordance.py`, `data/raw/concordance/CPC21-HS2017.csv`, `cpc21-hs2012.txt`, `isic4-cpc21.txt` |
| 6 | Section 232 datasets built directly from Proclamations 9704/9705 + HTSUS Rev 1.2, not Federal Register PDF estimates. Real counts: Steel = **303** HTS-8 codes (guide estimated ~170), Aluminum = **36** (guide estimated ~60) | More authoritative source, exact scope | `data/raw/tariff_events/us_232_steel_2018.csv` (303 rows), `us_232_aluminum_2018.csv` (36 rows) |
| 7 | Two independent tariff data streams: shock **events** (e[3], primary causal signal, from Federal Register/EU OJ/UKGT) vs. **WITS schedules** (e[2]/f[4], background context only) | Architectural clarification, not really a deviation | Confirmed correct as implemented |
| 8 | Eurostat PPI: correct indicator is **`PRC_PRR_DOM`**, not `PRIN` as the guide stated. NACE B/C/D present in `sts_inppd_m`; E and H are not | Guide error, corrected via live API probing | `src/data/clean_ppi.py` |
| 9 | Graph extended 2015–2021 via Comtrade using **WIOD-2014** (not WIOD-2016) as the frozen structural prior — topology, sector shares, node IDs frozen; only bilateral flow *magnitudes* updated | 2014 is the last available WIOD year (see #2) | `src/data/extend_with_comtrade.py` |
| 10 | Sector tariffs and shock deltas computed as **weighted averages** (Σweight·rate / Σweight), never summed | Prevents tariff inflation when many HS6 codes map to one sector | `src/data/compute_tariff_rates.py`, `build_shock_vectors.py` |
| 11 | WITS coverage expanded to **20 reporter countries**; final tariff coverage 78.2% observed / 15.0% interpolated / 6.8% missing | Initial ~40 country-year coverage (77% missing) was unacceptable | `data/processed/tariff_rates/sector_tariffs.parquet` |

### 1.2 Discovered during the 2026-08-13 codebase inventory (previously untracked)

| # | Finding | Severity | Status |
|---|---|---|---|
| 12 | **ROW node had no features.** Leontief/technical-coefficient matrices were built 44×56=2464 (ROW included), but `node_features_*.parquet` was 43×56=2408 rows — ROW silently absent because WIOD socioeconomic accounts have no ROW row. | Critical | Resolved — ROW dropped project-wide (§2) |
| 13 | **Sector-code convention split.** `config.py` used underscores; `node_features_*.parquet` persisted raw hyphenated WIOD form despite an internal `normalize_sector()` shim never being applied to the output column. | Critical | Resolved — underscore is now enforced at persistence time |
| 14 | **f4 (tariff_exposure) had zero temporal variation** — 354 unique values repeated identically across all 15 years 2000–2014, because the only tariff data (WITS 2017–2021) was averaged once and stamped onto every year. This is what HEAD commit `4c2c8ca "audites and node(not working)"` refers to. | Critical | Resolved — f4=0.0 pre-2015 (flagged `has_tariff_data=False`), real weighted exposure 2015–2021 |
| 15 | **PPI-lag coverage sparse** — `has_ppi_lags` true for only 12.3%–27.2% of node-years, because `ppi_quarterly_all.parquet` covers only 29/44 countries and 25/56 sectors. 15 countries (CHN, JPN, GBR, CAN, KOR, IND, BRA, RUS, TUR, ...) have zero PPI coverage. | High | **Accepted as scoped limitation** — see §3 |
| 16 | **Node features only existed for 2000–2014**, but all 6 tariff events occur in 2018/2021. | Critical | Resolved — extended to 2015–2021 |
| 17 | **Comtrade-extension `import_pen_coeff` never recomputed for 2015–2021** despite the module docstring claiming it rescales; verified byte-identical across years. Domestic flows frozen at 2014 values is by design and documented; the coefficient non-recompute was not. | Medium | Open — tracked, not yet fixed (does not block Phase 5; revisit if it distorts edge feature e1) |
| 18 | **No quarterly interpolation layer existed** despite `SEQ_LEN=8` requiring 8 quarterly snapshots per event. | Critical | In progress — `src/data/build_quarterly_snapshots.py` |
| 19 | **Edge features 89% incomplete** — only 2 of 6 config-defined dims materialized (`flow_usd` raw, `import_pen_coeff`). | Critical | In progress — `src/data/compute_edge_features.py` |
| 20 | **No standalone label generator** — label logic embedded in `clean_ppi.py`, event-relative 3m/6m/12m computation not yet implemented per spec. | Critical | In progress — `src/data/generate_labels.py` |
| 21 | `LEONTIEF["PASS_THROUGH_RATE"]` still `None` | Low (expected) | Expected — set during Phase 6 calibration |
| 22 | Phases 5–13 (PyG dataset, model, baselines, training, analysis, paper, product) are 0% implemented | n/a | Expected — Phase 4 is the current frontier |

### 1.3 Discovered while implementing the Phase 4-fix itself (2026-08-13)

| # | Finding | Severity | Status |
|---|---|---|---|
| 23 | **f4 z-score normalization exploded to ~10^6.** Because f4=0.0 for every 2000–2014 row by design (decision #4), its training-period std is exactly 0.0. Z-scoring any real 2015–2021 tariff-exposure value against a ~0 std divided it by ~1e-8, producing normalized values like 1,341,726.6 instead of a small number. | Critical | Resolved — f4 excluded from z-scoring entirely, left in raw units (already bounded ~0–1); recorded explicitly via `zscore_features`/`identity_features` keys in `normalization_stats.json` |
| 24 | **f1 (import_penetration) and f2 (export_intensity) had catastrophic outliers, including in the original 2000–2014 training data**, not just the new 2015–2021 years. Raw-value percentiles measured p0=−174 / p100=+23 for f1 and p0=−0.32 / p100=+111 for f2 — nodes with near-zero gross output (small economies/niche sectors) blow up the `x / gross_output` formulas. Left unclipped, this risks NaN loss during GAT/GRU training (CP35). | Critical | Resolved — winsorized at [p1, p99] of the training period only (`[0.0, 1.59]` for f1, `[0.0, 1.26]` for f2), applied uniformly to all years; bounds recorded in `normalization_stats.json["winsorize_bounds"]` |
| 25 | **Edge feature e2/e3 sector-matching was ambiguous.** A WIOD edge is (src_country, src_sector) → (tgt_country, tgt_sector). The tariffed product's classification is `src_sector` (confirmed by reading `build_shock_vectors.py`, whose `sector` column is the tariffed product's WIOD sector, not the buyer's sector). e2 (applied_tariff) and e3 (tariff_delta) must therefore join `sector_tariffs.parquet`/`shock_*.parquet` on `(tgt_country, src_sector)`, not `(tgt_country, tgt_sector)` — joining on tgt_sector would apply tariffs to the wrong edges. | Medium (correctness) | Resolved in `compute_edge_features.py` — documented at the top of the file |
| 26 | **product_hhi (e4) cannot be computed as specified.** The spec calls for "HHI across HS6 codes in this bilateral-sector pair," but `data/raw/comtrade/comtrade_{ISO3}_{YEAR}.parquet` is HS2-level only (97 codes, verified) with no bilateral partner breakdown (`partner` column is always `None`). | Medium (scoped limitation, not a bug) | Approximated as the importer's own HS2-level import concentration within the HS2 codes mapped to the edge's sector (via `extend_with_comtrade.HS2_SECTOR_MAP`), applied uniformly across all source countries into that (tgt_country, sector) node. 0.0 for years <2015 (no Comtrade data) and for sectors with no HS2 mapping (most services). Document as a limitation in the paper. |

### 1.4 Discovered while building Phase 6 baselines (2026-08-13)

| # | Finding | Severity | Status |
|---|---|---|---|
| 27 | **The Leontief baseline's calibration event (UK Global Tariff 2021, locked choice) has almost no overlap between its own direct-hit nodes and the labeled dataset.** UK's `affected_importers` is `["GBR"]` only, so all 54 of its direct-hit nodes are GBR sectors — but GBR has **zero** PPI coverage (post-Brexit, excluded from Eurostat, not USA; confirmed: only 4/56 GBR sector nodes have `has_label=True`, evidently via the WLD commodity fallback, not real GBR data). The calibrated `PASS_THROUGH_RATE` (−5.42) is therefore fit almost entirely on indirect/cross-border correlation among the other 668 labeled UK-event nodes, not a genuine direct pass-through relationship, and transfers poorly: applying it to the other 5 events gives DirAcc_6m as low as 0.13 (steel) and a 5-event mean of 0.30 — below chance (0.50) and well outside the original [0.55, 0.70] sanity range. | High (real finding, not a code bug) | **Not "fixed"** — the locked calibration procedure (UK event, closed-form OLS on 6m RMSE) was implemented exactly as specified and verified correct (formula, tau direction, and OLS derivation all checked by hand); the poor transfer is a genuine consequence of GBR's absence from the PPI dataset, not an engineering defect. Documented for the paper's limitations section — and is itself a legitimate, useful finding: it's a concrete demonstration of why a naive static baseline is fragile, which is exactly the gap TSPN is meant to address. |
| 28 | **`statsmodels.tsa.api.VAR` cannot implement the Panel VAR baseline as literally specified.** The Locked Plan says "PPI as endogenous variable [singular] and tariff_rate as exogenous," but `VAR` requires >=2 endogenous columns and raises `ValueError: Only gave one variable to VAR` on a single-column endog (verified directly). | Low (necessary tool substitution) | Substituted `statsmodels.tsa.ar_model.AutoReg(ppi_series, lags=4, exog=tariff_series)` — the correct statsmodels tool for a single-endogenous AR(4)+exogenous model, which is what the spec actually describes. Same lag order (4), same PPI/tariff roles. Documented at the top of `panel_var.py`. |

---

## 2. Locked decisions (2026-08-13)

1. **ROW handling**: ROW dropped entirely. Graph is now **43 countries × 56 sectors = 2,408 nodes**. All persisted matrices (`A_{year}.npz`, `leontief_{year}.npy`, `backward_linkage_{year}.npy`, `edges_{year}.parquet`) re-sliced/rebuilt to exclude ROW; `src_id`/`tgt_id` remapped to the new 0–2407 indexing.
2. **Sector naming**: underscore form (`C10_C12`) is the project-wide standard. All persisted files now use it.
3. **Label coverage**: current 29/44-country PPI coverage accepted as a documented, scoped limitation rather than delaying for more data sources (IMF-IFS, etc.). Per-event coverage is computed and reported honestly; any event below the 60% target (CP19 in the original Risk Checkpoints doc) is flagged in the paper as a data-availability limitation, not hidden. **Measured result: all 6 events have identical 27.9% coverage (672/2,408 nodes)** — the same 29-country x 25-sector footprint every time, since every event's label set is drawn from the same `ppi_quarterly_all.parquet`. This is below the original 60% CP19 target; accepted per this decision, to be stated plainly in the paper's limitations section rather than pursued further at this stage.
4. **f4 (tariff_exposure) redesign**: `f4 = 0.0` for years with no WITS coverage (2000–2014, flagged `has_tariff_data=False`); genuine trade-value-weighted tariff exposure for 2015–2021 where `sector_tariffs.parquet` has real data.
5. **Gross output for 2015–2021** (WIOD socioeconomic accounts stop at 2014): scaled from the 2014 baseline using per-country nominal GDP growth (documented approximation, same spirit as the Comtrade-extension design).
6. **Backward linkage for 2015–2021**: reuses `leontief_2014.npy`'s backward-linkage vector (structural prior frozen), consistent with `extend_with_comtrade.py`'s existing design — no new matrix inversion needed for those years.

---

## 3. Current authoritative schemas (Phase 4, in progress)

Countries: 43 (ROW excluded). Sectors: 56 (underscore form). Nodes: 2,408. `node_id = country_idx * 56 + sector_idx`.

| File | Path pattern | Years | Rows | Notes |
|---|---|---|---|---|
| Edges | `data/processed/edges/edges_{YEAR}.parquet` | 2000–2021 | ~145–175K/yr | 2000–2014 = WIOD; 2015–2021 = Comtrade-extended (frozen WIOD-2014 topology) |
| Technical coefficients | `data/processed/technical_coefficients/A_{YEAR}.npz` | 2000–2014 | 2408×2408 sparse | ROW-sliced |
| Leontief inverse | `data/processed/leontief/leontief_{YEAR}.npy` | 2000–2014 | 2408×2408 | ROW-sliced, re-inverted (not sliced from old inverse) |
| Backward linkage | `data/processed/leontief/backward_linkage_{YEAR}.npy` | 2000–2014 | (2408,) | 2015–2021 reuses 2014 vector |
| Node features | `data/processed/node_features/node_features_{YEAR}.parquet` | 2000–2021 | 2,408/yr | underscore sectors, f4 per rule above, `has_tariff_data` + `has_ppi_lags` + `is_gdp_scaled` flags. f1/f2 winsorized (finding #24) |
| Normalization stats | `data/processed/node_features/normalization_stats.json` | computed from 2000–2014 only | — | `computed_from_years`, `zscore_features`/`identity_features` (f4 is identity, finding #23), `winsorize_bounds` for f1/f2 |
| GDP proxy | `data/raw/wb_gdp/gdp_current_usd.parquet` | 2000–2021 | 42/43 countries | World Bank `NY.GDP.MKTP.CD`, no API key; TWN has no WB series, falls back to flat 2014 gross_output |
| Tariff rates | `data/processed/tariff_rates/sector_tariffs.parquet` | 2015–2021 | 12,320 | weighted-average, `data_source` flag |
| Shock vectors | `data/processed/shock_vectors/shock_{event}.parquet` | event-specific | — | weighted-average deltas, `is_direct_hit` flag |
| Edge features | `data/processed/edge_features/edge_features_{event}_q{0-7}.parquet` | per event | ~132K/quarter | e3 nonzero only at q7 (CP21 verified); e2/e3 joined on src_sector not tgt_sector (finding #25); e4 is an HS2-level, non-bilateral approximation (finding #26) |
| Node features (quarterly) | `data/processed/node_features_quarterly/{event}.parquet` | per event | 2,408 x 8 | linear interpolation between adjacent annual node_features (see `quarterly_interpolation.py`) |
| Labels | `data/processed/labels/labels_{event}.parquet` | per event | 2,408 | 3m/6m/12m via compounding quarterly PPI changes, `has_label` requires all 4 forward quarters found (CP24). Measured coverage: 27.9% for all 6 events (see decision 3) |
| PyG datasets | `data/pyg_datasets/{event}.pt` | per event | 8 x 2,408-node snapshots | `TSPNEventGraph` (src/data/tspn_event_graph.py); load with `torch.load(path, weights_only=False)`. Fixed 157,838-edge canonical index from edges_2014 shared across every snapshot/event (CP22) |

---

## 4. Roadmap status

- **Phase 0–3**: Complete.
- **Phase 4 (Feature Engineering)**: **Complete** as of 2026-08-13. All findings in §1.2 and §1.3 fixed and verified (`scripts/verify_phase4_fix.py`, 9/9 checks pass). Node features, edge features, quarterly snapshots, and labels exist for all 6 events; `PROJECT_STATE.md` and `audit_report.txt` are current.
- **Phase 5 (PyG Graph Dataset Construction)**: **Complete** as of 2026-08-13. `data/pyg_datasets/{event}.pt` exists for all 6 events (~32MB each, 192MB total — within the spec's 200-400MB expectation), verified independently after reload (`scripts/verify_phase5_pyg_datasets.py`, 9/9 checks pass): CP21 (shock isolated to snapshot 7), CP22 (one fixed edge_index — 157,838 edges from `edges_2014.parquet` — shared across all 8 snapshots of all 6 events), CP23 (no Python object aliasing across snapshots), CP24 (no NaN in labeled y). direct_hit_mask ranges 33–524 nodes per event (eu_retaliation is the largest, consistent with its broader country/product scope — not a bug). Label coverage carries over the 27.9% figure from Phase 4 (decision 3).
- **Phase 6 (Baseline Models)**: **Complete** as of 2026-08-13. All 3 baselines (`Leontief_IO`, `MLP_no_graph`, `Panel_VAR`) produce predictions for all 6 events and are recorded to `results/tables/baselines.csv`, verified independently (`scripts/verify_phase6_baselines.py`, 8/8 checks pass): `LEONTIEF["PASS_THROUGH_RATE"]` calibrated and saved to `config.py` (−5.42, see finding #27 for why it's negative); CP26 (Leontief excludes its own UK calibration event, 5 folds not 6); CP31 (MLP excludes f3); CP39 (no duplicate rows). Baseline ordering is sane: mean RMSE_6m is Leontief 0.059, MLP 0.017, Panel_VAR 0.019 — both learned baselines clearly beat the naive analytic one, as expected.
- **Phase 7 (TSPN Architecture) through Phase 13 (Maintenance)**: Not started. Immediate next step is `src/models/feature_embedding.py`, `tspn_gat_layer.py` (custom edge-feature-aware attention, NOT standard GATConv), `tspn_gru.py`, `output_head.py`, `tspn.py`. Watch CP27 (gat_layer1/gat_layer2 must be separate constructor calls) and CP30 (scale attention scores by 1/sqrt(head_dim)). See the approved plan for the full Phase 7–13 outline.

### Phase 6 deliverables (this session)
- `src/baselines/metrics.py` — shared RMSE/MAE/R2/DirAcc + results-table writer (dedup on model+fold, CP39)
- `src/baselines/leontief_io.py` — analytic baseline, UK-calibrated pass-through rate
- `src/baselines/mlp_no_graph.py` — 9-dim (f3 excluded, CP31) feedforward baseline, LOEO-CV
- `src/baselines/panel_var.py` — per-(country,sector) AR(4)+exog baseline (statsmodels VAR substituted for AutoReg, see finding #28)
- `scripts/verify_phase6_baselines.py` — independent verification gate
- Modified: `config.py` (`LEONTIEF["PASS_THROUGH_RATE"]` set)

### Phase 5 deliverables (this session)
- `src/data/tspn_event_graph.py` — `TSPNEventGraph` container class (stable module path for `torch.load`)
- `src/data/build_pyg_dataset.py` — assembles the 6 `.pt` files
- `scripts/verify_phase5_pyg_datasets.py` — independent reload-and-verify gate

### Phase 4-fix deliverables
- `PROJECT_STATE.md` (this file)
- `scripts/migrate_drop_row.py` — one-time ROW-removal migration (edges, tariff_rates, shock_vectors)
- `src/data/download_wb_gdp.py` — World Bank GDP fetch for the 2015–2021 gross_output proxy
- `src/data/quarterly_interpolation.py` — shared annual→quarterly interpolation utility
- `src/data/build_quarterly_snapshots.py` — quarterly node-feature snapshots per event
- `src/data/compute_edge_features.py` — full 6-dim edge feature matrix per event x 8 quarters
- `src/data/generate_labels.py` — event-relative 3m/6m/12m label generation
- `scripts/verify_phase4_fix.py` — hard-gate verification of all Phase 4-fix exit criteria
- Modified: `config.py` (2,408-node graph), `src/data/compute_node_features.py` (rewritten), `src/data/extend_with_comtrade.py` (magic-number fix)
