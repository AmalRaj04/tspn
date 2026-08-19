# Raw Data Redownload Guide

`data/raw/` (~1.4GB) is gitignored and was deleted from this machine on
2026-08-17 to free disk space. **Nothing about the project's current
functionality depends on it** — `data/processed/`, `data/pyg_datasets/`,
`models/checkpoints/`, and `results/` (the actual pipeline outputs Phase
4–10 produced and verified) are untouched and remain fully usable. This
guide exists only for the scenario where `data/raw/` genuinely needs to be
regenerated from scratch — e.g. re-deriving `data/processed/` after a bug
fix to an early-pipeline script.

Total original size and breakdown (for reference — re-downloading
everything reproduces roughly this):

| Directory | Size | Source |
|---|---|---|
| `data/raw/wiod/` | 890MB | Manual download |
| `data/raw/wits/` | 189MB | Script (`src/data/wits_download_cmd.txt`) |
| `data/raw/tariff_events/` | 358MB | Mixed manual + script |
| `data/raw/comtrade/` | 19MB | Script (`src/data/download_comtrade.py`) |
| `data/raw/eurostat_ppi/` | 2.7MB | Script (`src/data/download_eurostat.sh`) |
| `data/raw/commodity_prices/` | 756KB | Manual download |
| `data/raw/concordance/` | 432KB | Manual download |
| `data/raw/bls_ppi/` | 148KB | Script (`src/data/download_bls_ppi.py`) |
| `data/raw/wb_gdp/` | 12KB | Script (`src/data/download_wb_gdp.py`) |

---

## Scripted downloads (just run these)

```bash
# World Bank GDP (no API key needed)
python src/data/download_wb_gdp.py

# Eurostat PPI (no API key needed)
bash src/data/download_eurostat.sh

# WITS tariff schedules (no API key needed, SDMX API)
bash src/data/wits_download_cmd.txt
```

```bash
# BLS PPI -- needs a free API key from https://www.bls.gov/developers/
export BLS_API_KEY=<your-key>
python src/data/download_bls_ppi.py
```

```bash
# UN Comtrade -- needs a free API key from https://comtradeplus.un.org/
# (register, then generate a key under your account)
export COMTRADE_API_KEY=<your-key>
python src/data/download_comtrade.py
```

## Manual downloads

### WIOD (World Input-Output Database) — `data/raw/wiod/`
- Source: <https://www.rug.nl/ggdc/valuechain/wiod/> — the 2016 release.
- Files needed: `WIOT{YEAR}_Nov16_ROW.xlsb` for YEAR = 2000–2014 (15 files), plus `Socio_Economic_Accounts.xlsx` (the WIOD Socioeconomic Accounts file).
- Note: the site also has an October-2016 vintage in some places — this project used the **November 2016** vintage specifically (`_Nov16_`); match that filename pattern exactly, since `src/data/parse_wiod.py` expects it.

### World Bank Commodity Price Data (Pink Sheet) — `data/raw/commodity_prices/`
- Source: <https://www.worldbank.org/en/research/commodity-markets> — download the monthly "Pink Sheet" Excel workbook.
- Save as `data/raw/commodity_prices/wb_pink_sheet.xlsx`.

### HS/CPC/ISIC/NAICS concordance tables — `data/raw/concordance/`
- Source: UN Statistics Division correspondence tables, <https://unstats.un.org/unsd/classifications/Econ> (or search "UN Stats correspondence tables HS CPC ISIC").
- Files needed: `CPC21-HS2017.csv`, `cpc21-hs2012.txt`, `isic4-cpc21.txt`.
- `2017_NAICS_to_ISIC_4.xlsx`: US Census Bureau NAICS-ISIC concordance, <https://www.census.gov/naics/> (search "NAICS to ISIC concordance").
- Note: the project's actual concordance chain is HS6 → CPC21 → ISIC4 → WIOD56 (not a direct HS→ISIC file, which doesn't exist as a clean download) — see `PROJECT_STATE.md` finding for why, and `src/data/build_concordance.py` for the exact chain logic.

### Tariff event source documents — `data/raw/tariff_events/`
Mostly primary-source PDFs, each processed by a matching `src/data/extract_*.py` script:

- **Section 232 Steel/Aluminum** (`2018-05477.pdf`, `2018-05478.pdf`): Federal Register, Proclamations 9705 (steel) and 9704 (aluminum) — search by document number at <https://www.federalregister.gov/>.
- **Section 301 List 1/2** (`2018-13248.pdf`, `2018-17709.pdf`): Federal Register — search by document number at <https://www.federalregister.gov/>.
- **HTSUS Chapters 72/73/76** (`Chapter 72/73/76_2018HTSARevision1_2.pdf`): US Harmonized Tariff Schedule, Revision 1.2 (March 2018) — <https://hts.usitc.gov/> (archive/revision history).
- **EU Retaliation** (`CELEX_32018R0886_EN_TXT.pdf`): EU Official Journal, Implementing Regulation (EU) 2018/886 — search CELEX number `32018R0886` at <https://eur-lex.europa.eu/>.
- **Related EU regulation** (`CELEX_32020R1577_EN_TXT.pdf`): search CELEX number `32020R1577` at <https://eur-lex.europa.eu/>.
- **UK Global Tariff** (`uk-tariff-2021-01-01--v4.0.1527--measures-on-declarable-commodities.csv`, `...--commodities.csv`): UK government Trade Tariff bulk data export — <https://www.trade-tariff.service.gov.uk/> (look for their data/API bulk-download offering; these are large files, ~360MB total).

The already-*processed* event CSVs this project actually uses downstream
(`us_232_steel_2018.csv`, `us_232_aluminum_2018.csv`, `us_301_list1_2018.csv`,
`us_301_list2_2018.csv`, `eu_retaliation_2018.csv`, `uk_global_tariff_2021.csv`)
are regenerated from the above raw sources by running, in order:
`extract_232_tariffs.py`, `extract_301_list1.py`, `extract_301_list2.py`,
`extract_eu_retaliation_2018.py`, `extract_uk_global_tariff_2021.py`.

---

## After re-downloading

Re-run the Phase 4 pipeline scripts in `src/data/` in the order described
in `PROJECT_STATE.md` (§3, "Current authoritative schemas") to regenerate
`data/processed/` — but note **this should not normally be necessary**:
`data/processed/`, `data/pyg_datasets/`, and `models/checkpoints/` already
exist, are verified (see `scripts/verify_phase4_fix.py` and later phase
verification scripts), and are what every later phase actually depends on.
Only re-derive from raw if you specifically need to change something in
the Phase 4 feature-engineering logic itself.
