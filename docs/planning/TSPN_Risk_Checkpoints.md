# TSPN — Risk Checkpoints and Validation Gates
### 48 Checkpoints Across 8 Phase Groups

> **How to use**: Before moving from one phase to the next, run every detection test in that phase's section. If any test fails, execute the action before proceeding. Critical checkpoints (marked ★) must pass — do not continue if they fire.

---

## GROUP 1 — Setup & Data (Phases 0–1)

### CP01 | Medium | Config.py values don't match locked spec
**Problem**: A wrong hyperparameter in config.py propagates silently — e.g. node_embed_dim=256 instead of 128 changes model size with no error thrown anywhere.
**Detection**: After creating config.py, manually cross-reference every single value against the Master Locked Parameters table in the implementation plan. Write a diff script comparing config values to a reference JSON.
**Action**: Correct the value in config.py. If any scripts already ran using the wrong value, rerun them from the point of divergence.

---

### CP02 ★ | Critical | WIOD Excel matrix offset wrong — entire graph corrupt
**Problem**: If WIOD_MATRIX_ROW_OFFSET or WIOD_MATRIX_COL_OFFSET are wrong, you parse the wrong cells. All edge weights are garbage. The pipeline continues without any error because wrong numbers are still numbers.
**Detection**:
```
After parsing 2016 WIOD:
  assert 55_000 <= edge_table.flow_usd.sum() / 1e9 <= 70_000   # total ~$60T in 2016
  assert edge_table[['src_country','src_sector']].drop_duplicates().shape[0] == 2464
  assert edge_table[['tgt_country','tgt_sector']].drop_duplicates().shape[0] == 2464
```
**Action**: Open the Excel file manually. Identify the exact row/column where the IO matrix begins (the cell at the intersection of sector-row and sector-column headers). Update WIOD_MATRIX_ROW_OFFSET and WIOD_MATRIX_COL_OFFSET in config.py. Reparse all 17 years.

---

### CP03 | High | HTS code extraction typos from Federal Register PDFs
**Problem**: Manual copy-paste introduces truncated or malformed codes (e.g. '7208.1' instead of '72081000'). Failed concordance lookups silently reduce shock coverage — fewer sector-country pairs are shocked than they should be.
**Detection**:
```
For each tariff event CSV:
  assert all HTS codes are exactly 8 or 10 digits (no dots, no spaces)
  assert row count matches the count stated in the Federal Register notice:
    us_232_steel: ~170 rows
    us_232_aluminum: ~60 rows
    us_301_list1: 818 rows
    us_301_list2: 284 rows
```
**Action**: Use Tabula (free PDF table extractor: https://tabula.technology) instead of manual copy-paste. Tabula handles the formatted tables in the Federal Register PDFs correctly. Re-extract the affected event file.

---

### CP04 | High | WITS missing bilateral rates default to 0.0 instead of MFN fallback
**Problem**: WITS doesn't have tariff data for all bilateral pairs. If missing bilateral rates are set to 0.0 rather than falling back to MFN rate, shocked edges appear unaffected. All shock magnitude predictions are biased downward.
**Detection**:
```
In tariff_rates.parquet after computation:
  assert count(data_source == 'missing') / total_rows < 0.05
  # Spot check: CHN→USA steel sector rate for 2015 must be ~0.00-0.02% (near-zero MFN, before Section 301)
  # If it shows 0.00 AND data_source='missing', the fallback failed
  usa_chn_steel = tariff_df[(tariff_df.src=='CHN') & (tariff_df.tgt=='USA') & (tariff_df.sector=='C24') & (tariff_df.year==2015)]
  assert usa_chn_steel.data_source.values[0] != 'missing'
```
**Action**: Fix fallback order in compute_tariff_rates.py: (1) bilateral rate from WITS, (2) MFN rate for the reporter country, (3) 0.0 only as absolute last resort. Never assign 0.0 before checking for the MFN rate.

---

### CP05 | Medium | Comtrade download incomplete due to API pagination errors
**Problem**: API timeout or rate limit causes some country-year Parquet files to be truncated. Trade flows for those reporters are underreported in the 2017-2021 extension, biasing those years' edge weights.
**Detection**:
```
After all downloads:
  for reporter in wiod_country_list:
    for year in [2017, 2018, 2019, 2020, 2021]:
      f = f'data/raw/comtrade/comtrade_{reporter}_{year}.parquet'
      assert os.path.exists(f), f"Missing: {f}"
      df = pd.read_parquet(f)
      assert len(df) >= 500, f"Too few rows: {reporter} {year} ({len(df)} rows)"
```
**Action**: Identify all (reporter, year) pairs with missing or insufficient files. Re-run the download for those specific pairs with a 15-second delay between calls instead of 7.5 seconds.

---

### CP06 | Medium | torch-scatter / torch-sparse version mismatch
**Problem**: Wrong wheel version causes either silent wrong results in scatter operations (used inside GATConv) or an ImportError that prevents any progress.
**Detection**:
```python
import torch
import torch_scatter
x = torch.tensor([1., 2., 3.])
idx = torch.tensor([0, 0, 1])
result = torch_scatter.scatter_add(x, idx)
assert torch.allclose(result, torch.tensor([3., 3.])), f"Wrong: {result}"
print("scatter OK")
```
**Action**: Install using the exact PyG wheel index for your torch version:
https://data.pyg.org/whl/torch-2.1.0+cpu.html (CPU) or the corresponding CUDA version. Uninstall and reinstall torch-scatter and torch-sparse.

---

## GROUP 2 — WIOD Processing (Phase 2)

### CP07 | High | import_pen_coeff exceeds 1.0 for re-export economies
**Problem**: Re-exports cause bilateral flows to exceed the buyer's gross output for entrepôt economies (Netherlands, Belgium). import_pen_coeff > 1.0 violates IO assumptions, corrupts f[1], and destabilizes the Leontief computation.
**Detection**:
```
assert edge_table.import_pen_coeff.max() <= 1.0
violation_count = (edge_table.import_pen_coeff > 0.99).sum()
print(f"{violation_count} edges exceed 0.99 import_pen_coeff")
# Expected: a few hundred edges for NLD, BEL, SGP (if in dataset) re-export flows
```
**Action**: Cap at 0.99:
```python
edge_table['import_pen_coeff'] = edge_table['import_pen_coeff'].clip(upper=0.99)
```
This is standard practice in IO analysis. Document in the paper as a data cleaning step for re-export economies.

---

### CP08 ★ | Critical | Leontief inverse near-singular — backward linkage values explode
**Problem**: If (I-A) is near-singular for some years, scipy.linalg.inv() either fails with a LinAlgError or returns a numerically unstable matrix with entries > 1000. Backward linkage feature f[3] then has extreme outliers that collapse the normalized feature distribution.
**Detection**:
```python
for year in range(2000, 2017):
    A = build_technical_coefficients(year)
    cond = np.linalg.cond(np.eye(2464) - A)
    L = np.linalg.inv(np.eye(2464) - A)
    assert cond < 1e6, f"Year {year}: near-singular, condition={cond:.2e}"
    assert np.max(np.abs(L)) < 100, f"Year {year}: L has extreme values, max={np.max(np.abs(L)):.1f}"
```
**Action**: Apply Tikhonov regularization:
```python
eps = 1e-4  # from config.py: LEONTIEF_REG_EPS
L = np.linalg.inv(np.eye(2464) - A + eps * np.eye(2464))
```
Add LEONTIEF_REG_EPS = 1e-4 to config.py. Recompute all 17 Leontief inverse files.

---

### CP09 ★ | Critical | Node count inconsistent across years — index misalignment
**Problem**: A country or sector missing from one year's WIOD breaks the 2464-node assumption. Node IDs assigned for that year don't align with other years in the temporal sequence. Message passing uses the wrong node features.
**Detection**:
```python
for year in range(2000, 2017):
    et = pd.read_parquet(f'data/processed/edges/edges_{year}.parquet')
    n_countries = et['src_country'].nunique()
    n_sectors = et['src_sector'].nunique()
    assert n_countries == 44, f"Year {year}: only {n_countries} countries"
    assert n_sectors == 56, f"Year {year}: only {n_sectors} sectors"
```
**Action**: For missing (country, sector) pairs in a given year: insert zero-flow placeholder rows:
```python
# These rows have flow_usd=0 and import_pen_coeff=0
# They are filtered out by the edge threshold, preserving the node mapping
missing_nodes = full_node_list - present_nodes
for node in missing_nodes:
    add_zero_placeholder_row(edge_table, year, node)
```

---

### CP10 | High | Edge count after filtering outside expected range
**Problem**: Too few edges (< 50k) means the graph is too sparse for multi-hop message passing. Too many (> 200k) causes slow PyG operations and possible OOM during training.
**Detection**:
```python
for year in range(2000, 2017):
    n = len(pd.read_parquet(f'data/processed/edges/edges_{year}.parquet'))
    assert 80_000 <= n <= 150_000, f"Year {year}: {n} edges (outside expected 80k-150k)"
```
**Action**: If too sparse: lower threshold to 0.0005 in config.py (GRAPH.edge_threshold). If too dense: raise to 0.002. **Either change requires a full pipeline restart from Phase 2.** Update config.py before rerunning.

---

### CP11 | Medium | Comtrade country codes don't match WIOD country list
**Problem**: Comtrade uses different ISO codes for some territories (e.g. 'TWN' vs 'TW' for Taiwan, separate EU member codes vs 'EU' aggregate). Mismatched codes are silently dropped, reducing trade flow coverage for those countries.
**Detection**:
```python
comtrade_countries = set(comtrade_edges['src_country'].unique())
wiod_countries = set(config.GRAPH.country_list)
missing = comtrade_countries - wiod_countries
assert len(missing) == 0, f"Unmatched Comtrade codes: {missing}"
```
**Action**: Add a harmonization dictionary mapping non-standard Comtrade codes to WIOD ISO3 codes. Any code not in the dictionary maps to 'RoW'. Apply this mapping before building the Comtrade edge tables.

---

## GROUP 3 — Tariff & Shocks (Phase 3)

### CP12 ★ | Critical | Shock vector direction reversed — wrong edges shocked
**Problem**: A tariff is charged by the IMPORTER on goods from the EXPORTER. The shock goes on edges where the target country is the importer. If built as (importer→exporter), every shock edge is backwards. All cascade predictions are structurally inverted.
**Detection**:
```python
shock = pd.read_parquet('data/processed/shock_vectors/shock_us_232_steel_2018.parquet')

# MUST be zero: any non-USA target with non-zero delta
assert shock[(shock.tgt_country != 'USA') & (shock.delta_tariff > 0)].empty, \
    "Section 232: shock found on non-USA target — direction reversed"

# MUST be non-zero: CHN→USA steel sector
assert shock[(shock.src_country == 'CHN') & (shock.tgt_country == 'USA') &
             (shock.isic_sector == 'C24') & (shock.delta_tariff > 0)].shape[0] > 0, \
    "Section 232: CHN→USA steel not shocked — coverage missing"
```
**Action**: Swap src_country and tgt_country assignment in build_shock_vectors.py. Rerun all 6 shock vector files.

---

### CP13 | High | Zero shock coverage for one or more events
**Problem**: If the HS→ISIC concordance lookup fails (often due to HS revision mismatch), an event has zero non-zero shock entries. The model trains on a null shock for one event per fold, distorting all fold results.
**Detection**:
```python
for event_name in config.EVENTS.names:
    shock = pd.read_parquet(f'data/processed/shock_vectors/shock_{event_name}.parquet')
    nonzero = shock[shock.delta_tariff != 0]
    assert len(nonzero) >= 50, f"{event_name}: only {len(nonzero)} non-zero shock entries"
    assert nonzero.isic_sector.nunique() >= 3, f"{event_name}: only {nonzero.isic_sector.nunique()} sectors affected"
```
**Action**: Check whether the event's HTS codes use HS 2012 vs HS 2017 revision. Section 301 events use pre-2017 HS codes. Switch to the hs2012_isic4.xlsx concordance for those events.

---

### CP14 | High | UK Global Tariff delta sign inverted
**Problem**: Formula UK_rate − EU_CET computes reductions as negative (correct). But EU_CET − UK_rate inverts the sign — many genuine tariff reductions appear as increases. Model learns wrong price direction for the UK event.
**Detection**:
```python
uk_shock = pd.read_parquet('data/processed/shock_vectors/shock_uk_global_tariff_2021.parquet')
neg_pct = (uk_shock.delta_tariff < 0).mean()
# Approximately 35-50% of UK Global Tariff changes are reductions vs EU CET
assert 0.25 <= neg_pct <= 0.60, f"UK shock: only {neg_pct:.1%} negative — may be inverted"
```
**Action**: Fix the subtraction order: `delta_tariff = uk_rate - eu_cet_rate` (UK minus EU, NOT EU minus UK). Rerun the UK shock vector file.

---

### CP15 | High | Missing pre-event WITS data creates false zero deltas
**Problem**: For bilateral pairs where WITS has no pre-event tariff data, delta_tariff is silently set to 0.0 even though a real shock occurred. The model underestimates shock magnitude for those pairs.
**Detection**:
```python
# For Section 232: manually verify major steel exporters are shocked
section232 = pd.read_parquet('data/processed/shock_vectors/shock_us_232_steel_2018.parquet')
for country in ['CHN', 'KOR', 'JPN', 'DEU', 'BRA']:
    row = section232[(section232.src_country == country) &
                     (section232.tgt_country == 'USA') &
                     (section232.isic_sector == 'C24')]
    assert len(row) > 0 and row.delta_tariff.values[0] > 0.20, \
        f"Section 232: {country}→USA steel delta is {row.delta_tariff.values[0] if len(row)>0 else 'MISSING'}"
```
**Action**: For bilateral pairs with missing WITS pre-event rates, use the statutory rate from the Federal Register notice directly as delta_tariff (the notice states "tariffs of X% will be imposed" — that X% IS the delta). This is more reliable than WITS for event-specific shocks.

---

## GROUP 4 — Feature Engineering (Phase 4)

### CP16 ★ | Critical | Normalization statistics computed from test-period data
**Problem**: THE MOST IMPORTANT CHECKPOINT. If normalization_stats.json uses 2017–2021 data (which includes the post-shock outcomes), test-period statistics contaminate training. RMSE numbers are artificially optimistic. All published research results are invalid.
**Detection**:
```python
import json
with open('data/processed/node_features/normalization_stats.json') as f:
    stats = json.load(f)
assert stats['computed_from_years'] == list(range(2000, 2017)), \
    f"LEAKAGE: normalization uses years {stats['computed_from_years']}"
# Also: if 'computed_from_years' key doesn't exist, add it immediately
```
**Action**: Delete normalization_stats.json. Recompute by filtering strictly to years 2000–2016 only. Recompute all normalized node feature files. Recompute edge features (Phase 4, Step 4.3). Recompute PyG datasets (Phase 5). This is a full Phase 4+5 restart.

---

### CP17 ★ | Critical | PPI label reference quarter misaligned with event date
**Problem**: Labels computed relative to the wrong quarter are systematically wrong. If the baseline PPI is taken from Q4 2017 instead of Q1 2018 for the Section 232 event, the "3-month change" label actually captures changes from Q4 2017 to Q2 2018 — a 6-month window mislabeled as 3-month.
**Detection**:
```python
# BLS PPI for steel (PCU3317XX3317XX) rose approximately 10-18% from Q1 2018 to Q3 2018
labels_232 = pd.read_parquet('data/processed/labels/labels_us_232_steel_2018.parquet')
usa_steel = labels_232[(labels_232.country == 'USA') & (labels_232.sector == 'C24')]
assert len(usa_steel) > 0, "No USA steel label for Section 232"
delta_6m = usa_steel.delta_6m.values[0]
assert 0.05 <= delta_6m <= 0.25, \
    f"USA steel 6m delta = {delta_6m:.3f} — expected 5-25% increase. Reference quarter may be wrong."
```
**Action**: Fix generate_labels.py: ppi_base must be taken from the same quarter as the event_date (Q1 2018 for March 8, 2018 Section 232). Recompute all 6 label files.

---

### CP18 | High | Backward linkage outliers collapse f[3] distribution after normalization
**Problem**: If the Leontief inverse has any instability, a few sectors have backward linkage > 50. After z-score normalization, these outliers compress the remaining 99% of nodes to near-zero — f[3] is effectively constant for most nodes.
**Detection**:
```python
bl_all = []
for year in range(2000, 2017):
    bl = pd.read_parquet(f'data/processed/edges/backward_linkage_{year}.parquet')
    bl_all.extend(bl.backward_linkage.tolist())
p99 = np.percentile(bl_all, 99)
assert p99 < 20, f"99th percentile backward linkage = {p99:.1f} (should be < 20)"

# After normalization:
nf = pd.read_parquet('data/processed/node_features/node_features_2016.parquet')
assert nf['f3'].std() > 0.3, f"f[3] std after normalization = {nf['f3'].std():.3f} (too small)"
```
**Action**: Winsorize before normalization:
```python
p99 = np.percentile(bl_array, 99)
bl_array = np.clip(bl_array, None, p99)
```
Add BL_WINSORIZE_PCT = 99 to config.py.

---

### CP19 | High | Label coverage below 60% — training signal too sparse
**Problem**: Fewer than 1,478 labeled nodes per event means training loss is computed from < 60% of the graph. The model has insufficient signal to learn the full cascade pattern and may degenerate to predicting zero everywhere.
**Detection**:
```python
for event_name in config.EVENTS.names:
    labels = pd.read_parquet(f'data/processed/labels/labels_{event_name}.parquet')
    coverage = labels.has_label.mean()
    assert coverage >= 0.60, \
        f"{event_name}: label coverage = {coverage:.1%} (below 60% threshold)"
    print(f"{event_name}: {coverage:.1%} coverage ({labels.has_label.sum()} / 2464 nodes)")
```
**Action**: (1) Add IMF-IFS producer price indices (free at data.imf.org) for additional countries. (2) Add World Bank Pink Sheet commodity proxies for basic metals, chemicals, agriculture sectors globally. (3) Document all proxy label sources in the paper. Target >= 70% coverage for each event.

---

### CP20 ★ | Critical | Shock signal from event A contaminates event B's edge features
**Problem**: If edge feature files are cached without resetting e[3]=0.0, one event's tariff delta leaks into another event's graph. The model trains with phantom shocks and learns spurious propagation patterns.
**Detection**:
```python
for event_a in config.EVENTS.names:
    for event_b in config.EVENTS.names:
        if event_a == event_b: continue
        ef_b = pd.read_parquet(f'data/processed/edge_features/edge_features_{event_b}_q7.parquet')
        shock_a = pd.read_parquet(f'data/processed/shock_vectors/shock_{event_a}.parquet')
        # No edge feature delta should match event_a's shock values when we're processing event_b
        assert ef_b['e3'].abs().sum() == expected_shock_b_total, \
            f"Event {event_b} edge features contain shock from {event_a}"
```
**Action**: In build_pyg_dataset.py: always start with e[3]=0.0 for all edges. Then inject only the current event's shock vector. Never load or reuse edge_attr tensors across events.

---

## GROUP 5 — PyG Dataset (Phase 5)

### CP21 ★ | Critical | Shock signal non-zero in pre-event snapshots — temporal data leakage
**Problem**: e[3] (tariff_delta) must be 0.0 in snapshots 0–6. If non-zero, the model sees the shock BEFORE it happens — temporal data leakage. The model trivially achieves low RMSE by memorizing the leakage rather than learning propagation dynamics.
**Detection**:
```python
for event_name in config.EVENTS.names:
    event = torch.load(f'data/pyg_datasets/{event_name}.pt')
    for q in range(7):  # snapshots 0-6 must have zero delta
        delta_sum = event.temporal_sequence[q].edge_attr[:, 3].abs().sum().item()
        assert delta_sum == 0.0, f"{event_name} snapshot {q}: shock leaks ({delta_sum:.4f})"
    # Snapshot 7 must have non-zero delta
    assert event.temporal_sequence[7].edge_attr[:, 3].abs().sum().item() > 0, \
        f"{event_name} snapshot 7: no shock signal"
```
**Action**: Fix build_pyg_dataset.py. Inject shock vector ONLY into snapshot index 7. For snapshots 0–6, e[3] = 0.0 always (even if pre-event tariff rates changed in those quarters — only the DELTA at event time goes in e[3]).

---

### CP22 ★ | Critical | Edge index ordering inconsistent across snapshots
**Problem**: Each snapshot's edge_index must list edges in the same order so that edge_attr[:,i] refers to the same edge across all time steps. Inconsistent ordering means edge features describe wrong edges in some snapshots — message passing is structurally corrupt.
**Detection**:
```python
for event_name in config.EVENTS.names:
    event = torch.load(f'data/pyg_datasets/{event_name}.pt')
    ref = event.temporal_sequence[0].edge_index
    for q in range(1, 8):
        assert torch.equal(event.temporal_sequence[q].edge_index, ref), \
            f"{event_name}: edge_index differs at snapshot {q}"
        assert event.temporal_sequence[q].edge_index.shape == ref.shape
```
**Action**: Build ONE fixed edge_index from WIOD 2016, sorted by (src_id, tgt_id). Use this same edge_index for ALL snapshots of ALL events. For edges that don't exist in a given year, set all their edge features to 0.0 rather than removing them.

---

### CP23 | High | Python reference sharing — all 8 snapshots point to the same object
**Problem**: `sequence = [base_graph] * 8` creates 8 references to ONE object. Injecting shock into sequence[7] modifies ALL 8 simultaneously — a silent variant of CP21 where shock leaks into all snapshots.
**Detection**:
```python
for event_name in config.EVENTS.names:
    event = torch.load(f'data/pyg_datasets/{event_name}.pt')
    assert id(event.temporal_sequence[0]) != id(event.temporal_sequence[7]), \
        f"{event_name}: snapshots 0 and 7 are the same Python object"
    # Also verify edge_attr actually differs
    assert not torch.equal(event.temporal_sequence[0].edge_attr,
                           event.temporal_sequence[7].edge_attr), \
        f"{event_name}: snapshots 0 and 7 have identical edge features"
```
**Action**: Use `copy.deepcopy(base_snapshot)` for each of the 8 slots when building the temporal sequence. Never use list multiplication (`* n`) with mutable objects.

---

### CP24 | High | NaN labels with all-True mask — NaN propagates into training loss
**Problem**: A node marked has_label=True but one of its 3 horizon PPI values is NaN (series ended before the observation window). NaN in the loss propagates NaN gradients → NaN parameters. Training is unrecoverable.
**Detection**:
```python
for event_name in config.EVENTS.names:
    event = torch.load(f'data/pyg_datasets/{event_name}.pt')
    labeled_y = event.y[event.label_mask]
    assert not torch.isnan(labeled_y).any(), \
        f"{event_name}: NaN found in labeled nodes' y tensor"
```
**Run this assertion BEFORE starting any training.** If it fires after training has begun, you must restart from a clean checkpoint.
**Action**: In generate_labels.py: set `has_label = False` for any node where ANY of delta_3m, delta_6m, delta_12m is NaN. All three horizons must be available for a node to have has_label=True.

---

### CP25 | High | Node ID mapping differs between training and product inference
**Problem**: Country sort order used in training (alphabetical from WIOD list) differs from what the app uses (perhaps API order, or different file). Predictions are shown under wrong country/sector labels.
**Detection**:
```python
# In the deployed app, trigger the Section 232 steel event.
# Top 5 predicted nodes by |delta_6m| must include:
# (USA, C24) and (USA, C25) among the top 10
# If random unrelated sectors dominate: mapping is wrong
```
**Action**: In `app/utils/inference.py`: load country_list and sector_list directly from config.py (or a config JSON exported at training time). Never derive the mapping from any other source.

---

## GROUP 6 — Architecture (Phases 6–7)

### CP26 ★ | Critical | Leontief baseline calibrated and evaluated on the same event
**Problem**: Calibrating LEONTIEF_PASS_THROUGH_RATE on UK Global Tariff then evaluating Leontief on that same event's LOEO fold creates circular evaluation. The Leontief baseline looks artificially perfect on that fold, inflating its reported mean RMSE.
**Detection**:
```python
# Check results CSV after Leontief evaluation
leontief_rows = results_df[results_df.model_name == 'Leontief_IO']
assert 'uk_global_tariff_2021' not in leontief_rows.val_event.values, \
    "Leontief was evaluated on its calibration event"
assert len(leontief_rows) == 5, "Leontief should have 5 folds, not 6"
```
**Action**: In Leontief evaluation loop: skip the fold where held_out_event == 'uk_global_tariff_2021'. Report Leontief metrics across 5 folds with a table footnote: "Leontief excludes the UK event (used for pass-through rate calibration)".

---

### CP27 ★ | Critical | gat_layer1 and gat_layer2 share weights — effectively 1-layer model
**Problem**: `self.gat_layer2 = self.gat_layer1` in Python creates a reference, not a copy. Both layers share the same weight tensors. The model has 1 GAT layer, not 2. The "TSPN 1-layer" ablation and the full model produce identical results — the 2-hop finding is scientifically invalid.
**Detection**:
```python
model = TSPN(config)
assert id(model.gat_layer1) != id(model.gat_layer2), "Layers share identity"

# After one training step, weights should diverge:
# (They start the same from random init but diverge after first gradient step)
before_1 = model.gat_layer1.W_q.weight.data.clone()
before_2 = model.gat_layer2.W_q.weight.data.clone()
# run one training step
assert not torch.equal(before_1, model.gat_layer1.W_q.weight.data) or \
       not torch.equal(before_2, model.gat_layer2.W_q.weight.data), \
    "Both layers updated identically — they share weights"
```
**Action**: `self.gat_layer1 = TSPNGATLayer(config.MODEL)` then `self.gat_layer2 = TSPNGATLayer(config.MODEL)` — two separate constructor calls.

---

### CP28 | High | Attention dropout active during validation — stochastic predictions
**Problem**: Dropout active in eval mode means predictions vary between identical forward passes. Early stopping targets a noisy metric. Best checkpoint is selected on noise, not true model quality.
**Detection**:
```python
model.eval()
event = torch.load('data/pyg_datasets/us_232_steel_2018.pt')
out1, _ = model(event.temporal_sequence)
out2, _ = model(event.temporal_sequence)
max_diff = (out1 - out2).abs().max().item()
assert max_diff < 1e-7, f"Model is stochastic in eval mode: max diff = {max_diff}"
```
**Action**: Ensure all dropout layers are `nn.Dropout(p=...)` — never `F.dropout(x, p=..., training=True)` with hardcoded `training=True`. `nn.Dropout` automatically deactivates when `model.eval()` is called.

---

### CP29 | High | Some parameters have None gradient after backward
**Problem**: Parameters disconnected from the computation graph are never optimized. They stay at random initialization. Common causes: in-place operations (`x += y`) in residual connections breaking autograd.
**Detection**:
```python
model = TSPN(config)
event = torch.load('data/pyg_datasets/us_232_steel_2018.pt')
pred, alpha = model(event.temporal_sequence)
loss = pred[event.label_mask].sum()  # dummy loss
loss.backward()
missing = [n for n, p in model.named_parameters() if p.grad is None]
assert len(missing) == 0, f"No grad for: {missing}"
```
**Action**: Replace all in-place operations in the model with out-of-place equivalents:
- `x += y` → `x = x + y`
- `x[:, i] = val` → build new tensor

---

### CP30 | High | GAT attention collapse — all probability on one neighbor
**Problem**: Without attention score scaling, large logit values cause softmax to collapse: one neighbor receives α=1, all others α≈0. The model effectively ignores graph topology. Attention-Leontief correlation will be near-zero.
**Detection**:
```python
# After 20 training epochs:
alpha = model.gat_layer1.last_alpha  # shape: (num_edges, num_heads)
alpha_mean = alpha.mean(dim=1)  # average across heads

# Compute attention entropy per target node
# Group edges by target node, compute -sum(alpha * log(alpha)) per node
entropy_per_node = compute_attention_entropy(alpha_mean, event.edge_index)
mean_entropy = entropy_per_node.mean().item()
assert mean_entropy > 0.5, f"Attention collapsed: mean entropy = {mean_entropy:.3f} (should be > 0.5)"
```
**Action**: Add attention score scaling before softmax:
```python
score_ij = score_ij / math.sqrt(self.head_dim)  # head_dim = 32, scale = 0.177
```
This is standard scaled dot-product attention. Prevents logit explosion.

---

### CP31 | High | MLP no-graph baseline uses Leontief-derived features — unfair comparison
**Problem**: f[3] (backward linkage) is derived from the full Leontief inverse, which encodes the complete graph structure. Including it in the MLP "no-graph" baseline gives that baseline indirect access to graph information. The comparison is not fair.
**Detection**:
```python
# Check mlp_no_graph.py:
# grep for "f3" or "backward_linkage" in the feature selection code
# If f[3] appears: the baseline has graph-derived information
```
**Action**: Remove f[3] from the MLP baseline feature vector. Use only: f[0], f[1], f[2], f[4], f[5], f[6], f[7], f[8], plus one additional feature for direct tariff exposure (total delta_tariff on incoming edges, trade-value weighted). Document in paper.

---

### CP32 | Medium | last_alpha attribute stale — wrong attention weights saved
**Problem**: `gat_layer1.last_alpha` is overwritten by every forward pass. If two forward passes occur before reading the attribute, last_alpha contains the wrong event's weights.
**Detection**: Ensure in validation loop that `alpha = model.gat_layer1.last_alpha.detach().clone()` is the FIRST line after `pred, _ = model(...)`. Never separate the forward call from the alpha read by another forward call.
**Action**: Restructure the validation loop:
```python
pred, _ = model(event.temporal_sequence)
alpha_saved = model.gat_layer1.last_alpha.detach().clone()  # ← immediately after
val_rmse = compute_rmse(pred, event.y, event.label_mask)
# Only NOW do anything else
```

---

## GROUP 7 — Training & Experiments (Phases 8–9)

### CP33 ★ | Critical | LOEO-CV contamination — held-out event seen during training
**Problem**: THE MOST CRITICAL EVALUATION CHECKPOINT. Any form of held-out event data leaking into the training set invalidates all RMSE numbers. The model memorizes rather than generalizes. Published results are fabricated.
**Detection**:
```python
# Add this assertion as the FIRST LINE of every training epoch loop
for fold_idx in range(6):
    held_out_name = config.EVENTS.names[fold_idx]
    train_events = [e for e in all_events if e.event_name != held_out_name]
    assert len(train_events) == 5, f"Fold {fold_idx}: wrong number of train events ({len(train_events)})"
    assert held_out_name not in [e.event_name for e in train_events], \
        f"CONTAMINATION: {held_out_name} in training set for fold {fold_idx}"
```
**Action**: This assertion must be a hard `raise RuntimeError` — not a warning, not a print. Never allow training to continue if this fires. Fix the split logic and restart.

---

### CP34 ★ | Critical | Augmentation active during validation — wrong checkpoint selected
**Problem**: Augmentation noise in validation makes RMSE vary across identical runs. Early stopping targets a noisy signal. The "best" checkpoint is selected based on luck, not model quality. All evaluation metrics have artificial variance.
**Detection**:
```python
model.eval()
val_event = torch.load(f'data/pyg_datasets/{held_out_name}.pt')
# Two identical forward passes must produce identical output
out1 = model(val_event.temporal_sequence)[0]
out2 = model(val_event.temporal_sequence)[0]
assert (out1 - out2).abs().max().item() < 1e-8, \
    "Validation is stochastic — augmentation still active"
```
**Action**: Create a `validate(model, event)` function where `model.eval()` is the literal first line. Pass clean (unaugmented) event data. Never reuse the training forward pass function for validation.

---

### CP35 | High | NaN loss mid-training — model unrecoverable
**Problem**: NaN loss produces NaN gradients, then NaN parameters. All subsequent outputs are NaN. If a checkpoint is saved after this point, the saved model is useless.
**Detection**:
```python
# After every training step:
if torch.isnan(loss):
    print(f"NaN loss at epoch {epoch}, event {event.event_name}")
    print(f"Max pred: {pred.abs().max():.4f}, Max label: {event.y[event.label_mask].abs().max():.4f}")
    raise RuntimeError("NaN loss — stopping before checkpoint is corrupted")
```
**Action**: Four root causes to check in order:
1. NaN labels not masked (CP24) — run CP24 check first
2. Gradient clipping applied AFTER optimizer.step() — must be BEFORE
3. Log transform: verify e[0] = log(flow + 1) not log(flow)
4. Leontief instability in f[3] — apply CP08 fix

---

### CP36 | High | Wrong checkpoint loaded for evaluation — wrong model measured
**Problem**: Loading the last-epoch checkpoint instead of the best-epoch checkpoint. Evaluation reports a degraded model's performance, understating TSPN's true quality.
**Detection**:
```python
# When saving:
torch.save({
    'epoch': epoch,
    'val_rmse_6m': val_rmse_6m,
    'model_state': model.state_dict()
}, f'models/checkpoints/tspn_fold{fold_idx}_best.pt')

# When loading for evaluation:
ckpt = torch.load(f'models/checkpoints/tspn_fold{fold_idx}_best.pt')
print(f"Loading fold {fold_idx} best checkpoint: epoch {ckpt['epoch']}, val_rmse_6m={ckpt['val_rmse_6m']:.4f}")
model.load_state_dict(ckpt['model_state'])
```
**Action**: Always save checkpoint metadata (epoch, val_rmse). Print and verify these values on load. Keep a separate `_final.pt` for the last epoch to avoid confusion with the `_best.pt`.

---

### CP37 ★ | Critical | Ablation experiments share model state — ablation table invalid
**Problem**: Sequential ablations that don't re-initialize the model start from prior weights. Results reflect partially-trained-on-prior-task weights, not a randomly initialized ablation variant. The ablation table draws wrong conclusions about each component's contribution.
**Detection**:
```python
for ablation_name, AblationClass in ablation_variants.items():
    model = AblationClass(config)  # ← must be INSIDE the loop
    # Verify random initialization (not inherited weights):
    init_norm = sum(p.abs().mean().item() for p in model.parameters()) / len(list(model.parameters()))
    assert 0.01 < init_norm < 1.0, \
        f"Ablation {ablation_name}: init norm {init_norm:.4f} — may have inherited weights"
```
**Action**: Call `model = AblationVariant(config)` and `optimizer = Adam(model.parameters(), ...)` as the FIRST lines of each ablation training loop, inside the loop. Never carry model or optimizer state across experiments.

---

### CP38 | Medium | Gradient clipping too aggressive in early training
**Problem**: GRU gradients are large in the first 20 epochs. Clipping at max_norm=1.0 from epoch 0 prevents meaningful updates — training stalls at initialization quality for many epochs.
**Detection**:
```python
# Log gradient norm BEFORE clipping:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
wandb.log({'grad_norm_before_clip': grad_norm})
# If mean grad_norm > 5.0 for 20+ consecutive epochs: clipping is suppressing learning
```
**Action**: Warmup: use max_norm=5.0 for epochs 0–30, then 1.0 afterward. Add to config.py:
```
GRAD_CLIP_MAX_NORM = 1.0
GRAD_CLIP_WARMUP_NORM = 5.0
GRAD_CLIP_WARMUP_EPOCHS = 30
```

---

### CP39 | Medium | Results CSV has duplicate rows from Colab restart
**Problem**: Session crash mid-fold and restart appends duplicate rows without removing the partial previous run. Mean RMSE is computed from duplicated folds, giving wrong statistics.
**Detection**:
```python
results = pd.read_csv('results/tables/all_results.csv')
dupes = results.groupby(['model_name', 'fold']).size()
assert (dupes == 1).all(), f"Duplicate rows found: {dupes[dupes > 1]}"
```
**Action**: Add deduplication to `record_all_metrics()`:
```python
results = pd.read_csv('results/tables/all_results.csv')
results = results[~((results.model_name == model_name) & (results.fold == fold_idx))]
results = pd.concat([results, new_row], ignore_index=True)
results.to_csv('results/tables/all_results.csv', index=False)
```

---

## GROUP 8 — Analysis & Product (Phases 10 & 12)

### CP40 ★ | Critical | Attention-Leontief correlation uses transposed index — main finding inverted
**Problem**: α_ij = weight that buyer j pays to supplier i. L[i,j] = input from i per unit output from j. These align: compare α_ij vs L[i,j]. If you accidentally compare α_ij vs L[j,i] (transpose), you get near-zero or negative correlation. The paper's core interpretability finding fails.
**Detection**:
```python
# Manual sanity check for a known high-linkage pair:
# DEU basic metals → DEU motor vehicles (C24 → C29)
c24_deu_id = config.GRAPH.node_id('DEU', 'C24')
c29_deu_id = config.GRAPH.node_id('DEU', 'C29')

alpha = np.load('results/tables/attention_fold0.npy').mean(axis=1)  # avg across heads
L = np.load('data/processed/edges/leontief_2018.npy')

# Find the edge (C24_DEU → C29_DEU) in the edge index
edge_pos = find_edge_position(edge_index, src=c24_deu_id, tgt=c29_deu_id)
alpha_val = alpha[edge_pos]
leontief_val = L[c24_deu_id, c29_deu_id]  # NOT L[c29, c24]

# Both should be relatively large for this economically important linkage
print(f"Alpha: {alpha_val:.4f}, Leontief: {leontief_val:.4f}")
assert alpha_val > 0.05, "Steel→Auto attention should be non-trivial"
assert leontief_val > 0.01, "Steel→Auto Leontief should be non-trivial"
```
**Action**: If correlation near zero: try `L.T` instead of `L` in the comparison. If correlation improves, you had the transpose. Fix index order and re-report all correlation values.

---

### CP41 | High | Cascade depth measured from indirect nodes — depth underreported
**Problem**: Using all high-|Δp| nodes as cascade origins (instead of is_direct_hit nodes) includes 1-hop-downstream nodes already, making the measured cascade appear shorter than it truly is.
**Detection**:
```python
event = torch.load('data/pyg_datasets/us_232_steel_2018.pt')
n_direct_hits = event.direct_hit_mask.sum().item()
# Section 232: approximately 40-50 directly hit nodes
# (all countries × 1 steel sector, minus exempted countries)
assert 30 <= n_direct_hits <= 70, \
    f"Section 232: {n_direct_hits} direct hits (expected 30-70)"
# If n_direct_hits > 200: direct_hit_mask includes downstream nodes
```
**Action**: Fix direct_hit_mask in build_pyg_dataset.py. A node is directly hit ONLY if it is the BUYER (target) in a directly shocked edge — not its downstream customers and not its upstream suppliers.

---

### CP42 | Medium | Paper figures saved at wrong DPI or format — submission rejected
**Problem**: Journals require vector PDF figures. Rasterized figures appear blurry when zoomed in the reviewer's PDF reader. This is caught at technical check before peer review.
**Detection**:
```python
import os
for fig_name in ['fig3_attention_leontief.pdf', 'fig4_cascade_depth.pdf',
                  'fig5_amplifiers.pdf', 'fig6_model_comparison.pdf']:
    path = f'results/figures/{fig_name}'
    assert os.path.exists(path), f"Missing: {path}"
    size_kb = os.path.getsize(path) / 1024
    assert size_kb > 50, f"{fig_name}: {size_kb:.0f}KB — likely rasterized (vector PDF > 50KB)"
```
**Action**: Save all figures with:
```python
plt.savefig('fig.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig.png', dpi=300, bbox_inches='tight')
```

---

### CP43 ★ | Critical | ONNX output differs from PyTorch — product shows wrong predictions
**Problem**: ONNX operator implementations sometimes differ from PyTorch in float precision. Product users see different numbers than the published research paper. Trust in the tool is damaged.
**Detection**:
```python
import onnxruntime as ort
import numpy as np

# Load PyTorch model and run
model.eval()
with torch.no_grad():
    pt_out = model(test_sequence)[0].numpy()

# Load ONNX and run same input
sess = ort.InferenceSession('models/onnx/tspn_best.onnx')
onnx_out = sess.run(None, prepare_onnx_inputs(test_sequence))[0]

max_diff = np.abs(pt_out - onnx_out).max()
assert max_diff < 1e-4, f"PyTorch vs ONNX max diff: {max_diff:.6f} (threshold: 1e-4)"
```
**Action**: If difference > 1e-4: (1) Try float64 export. (2) Use TorchScript instead: `torch.jit.script(model)` — numerically identical to PyTorch. Save as `.pts` and use `torch.jit.load` in the app.

---

### CP44 ★ | Critical | Normalization not applied at product inference — garbage predictions
**Problem**: Model trained on z-score normalized features. Raw features at inference produce outputs at the wrong scale — typically near-constant or extreme values. All product predictions are meaningless.
**Detection**:
```python
# In the app: trigger Section 232 steel shock.
# Expected: top predicted sectors include (USA, C24) and (USA, C25)
# Failure sign: top sectors are random service sectors (G47, I56, etc.)

# Also: compare raw vs normalized feature vectors
raw_f0 = np.log(gross_output + 1)  # e.g. 8.5 (billions USD)
norm_f0 = (raw_f0 - stats['mean'][0]) / stats['std'][0]  # should be near 0 for avg sector
print(f"Raw f0: {raw_f0:.2f}, Normalized: {norm_f0:.2f}")
```
**Action**: In `app/utils/inference.py`: add at the top of the inference function:
```python
with open('normalization_stats.json') as f:
    stats = json.load(f)
node_features = (node_features_raw - np.array(stats['mean'])) / np.array(stats['std'])
```
Ship `normalization_stats.json` with the deployed app. Never use raw features for inference.

---

### CP45 | High | App shock vector formula differs from training injection
**Problem**: The training pipeline uses trade-value-weighted tariff deltas per ISIC sector. If the app uses a simplified flat delta, the model receives a structurally different shock at inference — cascade predictions are wrong.
**Detection**:
```python
# Compute Section 232 shock vector using both formulas:
training_shock = build_shock_vector_training_formula('us_232_steel_2018')
app_shock = app_scenario_parser.compute_shock('USA', 'C24', 0.25, affected_exporters='all')

mean_diff = (training_shock - app_shock).abs().mean()
assert mean_diff < 0.05 * training_shock.abs().mean(), \
    f"App shock differs from training shock by {mean_diff:.4f} (>5% of mean)"
```
**Action**: In `app/utils/scenario_parser.py`, import and call `compute_shock_vector()` from `src/data/build_shock_vectors.py` directly. Do not re-implement it.

---

### CP46 | High | PyVis renders full 2464 nodes — app iframe crashes
**Problem**: Rendering hundreds of nodes and edges in PyVis causes 30+ second freeze and iframe crash. Product is unusable for the most economically significant large shocks.
**Detection**: Test with Section 301 List 3 (largest shock). Count nodes with |Δp(6m)| > 0.5%. If > 400 nodes, rendering will likely crash. Time the PyVis rendering call — assert < 5 seconds.
**Action**: Implement a hard node budget of 150:
1. Always include all is_direct_hit nodes
2. Add nodes by |Δp(6m)| descending until budget reached
3. Show "Displaying X of Y affected nodes" message
4. Add "Download full graph" button that exports PyVis HTML for local viewing

---

### CP47 | Medium | Partial cache key causes wrong cached result
**Problem**: If the cache key uses only (target_country, sector, magnitude) and excludes affected_exporters, two different scenarios get the same cache hit. User B's query returns User A's result.
**Detection**:
```python
# Submit two scenarios that differ only in affected_exporters:
# (1) Section 232: all countries
# (2) Section 232: China only (counterfactual)
# Results must differ — China-only should show smaller overall impact
result_all = get_cached_or_compute(tgt='USA', sector='C24', delta=0.25, src='all')
result_chn = get_cached_or_compute(tgt='USA', sector='C24', delta=0.25, src='CHN')
assert result_all != result_chn, "Cache collision: two different scenarios returned same result"
```
**Action**: Cache key = `hash(json.dumps({'tgt': tgt_country, 'sec': sector, 'delta': delta, 'src': sorted(src_countries), 'date': event_date}, sort_keys=True))`. All parameters must be in the key.

---

### CP48 | Medium | Model checkpoint file too large for free hosting
**Problem**: Streamlit Community Cloud has 1GB repo size limit. A wrong model dimension (e.g. accidentally 10× too large) creates a large ONNX file that can't be stored in the repo.
**Detection**:
```python
import os
onnx_size_mb = os.path.getsize('models/onnx/tspn_best.onnx') / (1024 * 1024)
assert onnx_size_mb < 100, f"ONNX model too large: {onnx_size_mb:.1f}MB (expected < 100MB)"
```
Expected size at specified dimensions: 15–50MB.
**Action**: If > 100MB: verify all model dimensions match config.py exactly (a single wrong dimension can make the model 10× larger). If dimensions are correct but file is still large: use Git LFS or store on Hugging Face Hub (free, unlimited model hosting) and download at app startup.

---

## SUMMARY — Checkpoints by Severity

| Severity | Count | Consequence if Missed |
|---|---|---|
| ★ Critical | 13 | Completely invalidates results or makes product non-functional |
| High | 22 | Significantly degrades results, hard to detect later |
| Medium | 13 | Degrades results or UX, fixable without full pipeline restart |
| **Total** | **48** | |

## Critical Checkpoints — Quick Reference
| ID | Phase | Title |
|---|---|---|
| CP02 | WIOD Processing | WIOD Excel matrix offset wrong |
| CP08 | WIOD Processing | Leontief inverse near-singular |
| CP09 | WIOD Processing | Node count inconsistent across years |
| CP12 | Tariff & Shocks | Shock vector direction reversed |
| CP16 | Feature Engineering | Normalization data leakage |
| CP17 | Feature Engineering | PPI label reference quarter wrong |
| CP20 | Feature Engineering | Shock signal contaminates other events |
| CP21 | PyG Dataset | Shock in wrong temporal snapshot |
| CP22 | PyG Dataset | Edge index inconsistent across snapshots |
| CP26 | Architecture | Leontief calibration/evaluation overlap |
| CP27 | Architecture | GAT layers share weights |
| CP33 | Training | LOEO-CV contamination |
| CP34 | Training | Augmentation during validation |
| CP37 | Experiments | Ablation model state shared |
| CP40 | Analysis | Attention-Leontief index transposed |
| CP43 | Product | ONNX numerical mismatch |
| CP44 | Product | Normalization not applied at inference |
