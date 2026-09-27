# Data Card — PJM Short-Term Load Forecasting (2020–2025)

**Phase 0 — documents current v1 reality + v2 requirements. No data modified.**

## 1. Dataset identity

- **Name (v1, existing):** `pjm_smart_grid_2020_2025.csv` in `dataset/processed/`
- **Name (v2, planned):** `pjm_smart_grid_2020_2025_v2.csv` + `pjm_weather_2020_2025.csv` + `events.parquet` + `test_origins.parquet` (not yet built)
- **Sources:** PJM Interconnection public Data Miner — `hrl_load_metered.csv`, `wind_gen.csv`, `solar_gen.csv`, per year 2020–2025 (`dataset/raw/pjm/`; 18 CSVs; see `results/data_validation/pjm_raw_audit_2020_2025.json`)
- **Target:** `pjm_load_mw` (RTO-total hourly load, MW)
- **Period (v1, verified):** 2020-01-01 05:00 UTC → 2026-01-01 04:00 UTC, hourly, **52,608 rows**, 0 gaps, 0 duplicates after merge
- **Target stats (v1):** mean ~179,950 MW; range 109,874–320,308 MW (`pjm_processed_2020_2025_report.json`)
- **Exogenous (v1):** RTO `wind_generation_mw` (mean ~3,383 MW), `solar_generation_mw` (mean ~1,350 MW); **no temperature / humidity / holiday** (`configs/pjm_data.yaml`: `weather.temperature: null`, `humidity: null`)
- **Renewable penetration:** 1.99% (2020) → 3.40% (2025); load–wind corr −0.148, load–solar +0.261 (`pjm_eda_metrics_2020_2025.json`)

## 2. Build pipeline (v1, as-coded in `src/data/pjm_integration.py`)

1. Load: group-by timestamp sum of `mw` across all valid load areas (30 areas, 22 zones).
2. Wind/solar: keep `area == "RTO"` only, drop timestamp duplicates, reindex to full hourly range, inner-join on `timestamp` (join row-loss reported in `merge_statistics`; hours with missing RTO renewables are dropped, not imputed).
3. Negatives: wind handful clipped, solar **24,245 cells clipped to 0** (~40% of night hours are negative metering noise) — no sensitivity study yet (required for v2).
4. Cleaning (`src/data/preprocessing.py`): regular-grid reindex per group, exogenous `time`-interpolation (limit 6, inside only), target gaps ≤ `max_target_gap: 3` interpolated else dropped with counts (`cleaning_report.json`).
5. Features (`src/data/features.py`, leakage-safe): calendar (hour, dow, dom, month, woy, is_weekend) + 6 cyclical sin/cos + `lags [1,24,168]` via `groupby.shift(lag)` + rolling mean/std `[24,168]` via `shift(1).rolling()` → 21 model features (`pjm_final_metadata.json`).
6. Scaling/splits: `StandardScaler` fit on train only; chronological 70/15/15 (train → 2024-03-14, val → 2025-02-06, test → 2026-01-01); 168 warmup rows dropped from train.

## 3. v2 requirements (planned, not built)

- **Weather (mandatory):** ERA5 (primary) or NOAA ISD fallback — hourly footprint-mean 2m temperature, humidity/dew-point, 10m wind speed, cloud cover; derived CDH/HDH, heat-index, solar-zenith proxy. Merge-asof with coverage/join-loss log (`weather_coverage.json`).
- **Calendar:** US federal + PJM holidays + bridge flags (`holidays` package), season, DST-transition flag.
- **Renewable framing:** `total_renewable`, `penetration`, `net_load = load − wind − solar` (operational target alongside load).
- **Events (`events.parquet`):** deterministic heat-wave (≥3d Tmax ≥ 35 °C), cold-wave (≥2d Tmin ≤ −10 °C), peak top-1% hours, normal.
- **Origins (`test_origins.parquet`):** all test origins with full 168h history + full horizon truth, shared by every model (PROTOCOL §6).

## 4. Splits

- **v1 (current):** 70/15/15 mid-month cuts (see §2.6). Usable for provenance only.
- **v2 (locked per PROTOCOL §3):** Train 2020-01–2023-12 / Val 2024 / Test 2025–Jan26 (calendar years) + 4 quarterly rolling origins through 2025. No random splits.

## 5. Known limitations (must appear in manuscript)

Single RTO aggregate (no zonal/second ISO); no weather in v1 (omitted-variable bias); RTO-only renewables at 2–3% penetration (weak signal); negative-clip policy unvalidated; inner-join silent-drop semantics; bogus "2026" annual EDA row (4 Jan-2026 hours labeled a year); SHAP files reference phantom `net_load`/`total_renewable_mw` absent from v1 pipeline.

## 6. Provenance pointers

`results/data_validation/pjm_raw_audit_2020_2025.json` · `pjm_processed_2020_2025_report.json` · `pjm_final_metadata.json` · `pjm_eda_metrics_2020_2025.json` · `src/data/pjm_integration.py:155` · `configs/pjm_data.yaml` · `Q1_Phase0_Research_Strategy.md §1–§5`.

## 7. Phase 1A verified inventory (read-only scan, 2026-09-24 UTC — no files moved/modified)

Full report: `data/raw/DATASET_INVENTORY.md`. Checksums: `data/checksums/sha256.txt` (31 files).

- **Files on disk:** 18 raw PJM CSVs (`dataset/raw/pjm/{2020..2025}/{hrl_load_metered,wind_gen,solar_gen}.csv`) + 13 processed CSVs (`dataset/processed/`). Raw counts match `pjm_raw_audit_2020_2025.json`.
- **Raw load files:** 8 cols (`datetime_beginning_utc, datetime_beginning_ept, nerc_region, mkt_region, zone, load_area, mw, is_verified`); 262,800–263,520 rows/yr; 0 missing in any column as stored; 0 exact full-row duplicates; timestamp duplicates structural (30 load-areas × hourly: 254,040–254,736/yr); hourly modal diff on sorted unique timestamps.
- **Raw wind files:** 4 cols (`datetime_beginning_utc, datetime_beginning_ept, area, wind_generation_mw`); 43,800–52,704 rows/yr (5 areas 2020–2021, 6 areas 2022–2025); 0 missing; 0 exact dup rows; hourly.
- **Raw solar files:** 4 cols (`datetime_beginning_utc, datetime_beginning_ept, area, solar_generation_mw`); 51,336–52,704 rows/yr (6 areas; 2020 OTHER partial 7,416); 0 missing; 0 exact dup rows; hourly.
- **Merged canonical:** `pjm_smart_grid_2020_2025.csv` 52,608 × 4, 2020-01-01 05:00 UTC → 2026-01-01 04:00 UTC, 0 missing, 0 dup timestamps, hourly; `pjm_smart_grid_2020.csv` 8,784 × 4, same schema.
- **Split frames (24 cols, canonical pipeline):** `train.csv` 36,825 rows with **577 warmup NaNs** (lag_1:1, lag_24:24, lag_168:168, rolling_mean_24:24, rolling_std_24:24, rolling_mean_168:168, rolling_std_168:168); `validation.csv` 7,891 rows 0 missing; `test.csv` 7,892 rows 0 missing; `pjm_features.csv` 52,608 rows with same 577-warmup pattern. All: unique hourly timestamps, 0 exact dup rows.
- **TFT frames:** `tft_frame.csv` 52,440 × 14, 0 missing, 0 dup rows; `time_idx` verified integer sequence index (0–52,439), not datetime. `pjm_tft_features.csv` 52,440 × 22, 0 missing.
- **Extended feature family (with `total_renewable_mw`, `net_load`, shares, flags):** `pjm_tabular_features.csv` 52,440 × 32 (no `season` col); `train_features.csv` 36,708 × 33; `validation_features.csv` / `test_features.csv` 7,866 × 33; all 0 missing, 0 dup rows. `net_load`/`total_renewable_mw` exist ONLY here — absent from canonical 24-col training frames.
- **Sequences:** `pjm_sequences.csv` 52,272 × 1,682 (168 steps × 10 signals + timestamp/target), 2020-01-15 05:00 → 2026-01-01 04:00 UTC, 0 missing in any column, 0 dup timestamps, hourly; exact full-row dup scan not performed (727 MB > 50 MB scan cap — stated, not assumed).
- **Weather files:** none found in `dataset/` or `data/` (`data/raw/weather/{era5,noaa}/` contain only `.gitkeep`).
