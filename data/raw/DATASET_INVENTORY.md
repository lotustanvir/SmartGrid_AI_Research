# Dataset Inventory — Phase 1A (read-only, 2026-09-24 UTC)

**Scope:** `dataset/` + `data/` inspected. No files moved, renamed, or modified. No training.
**Method:** streaming read of every CSV (header, byte size, row count, chunked missing-value scan, timestamp min/max/unique, exact full-row duplicate scan for files ≤ 50 MB). Raw-file counts cross-checked against `results/data_validation/pjm_raw_audit_2020_2025.json` (match).
**Result:** 31 CSV files inventoried — 18 raw PJM + 13 processed. **Zero weather files found anywhere** (see §4).

## 1. File map (path · format · size · rows × cols)

| # | File path | Format | Size (bytes) | Rows × Cols |
|---|---|---|---|---|
| 1 | `dataset/raw/pjm/2020/hrl_load_metered.csv` | CSV | 19,969,769 | 263,520 × 8 |
| 2 | `dataset/raw/pjm/2020/wind_gen.csv` | CSV | 2,499,027 | 43,920 × 4 |
| 3 | `dataset/raw/pjm/2020/solar_gen.csv` | CSV | 2,870,900 | 51,336 × 4 |
| 4 | `dataset/raw/pjm/2021/hrl_load_metered.csv` | CSV | 19,937,345 | 262,800 × 8 |
| 5 | `dataset/raw/pjm/2021/wind_gen.csv` | CSV | 2,491,950 | 43,800 × 4 |
| 6 | `dataset/raw/pjm/2021/solar_gen.csv` | CSV | 2,950,653 | 52,560 × 4 |
| 7 | `dataset/raw/pjm/2022/hrl_load_metered.csv` | CSV | 19,945,980 | 262,800 × 8 |
| 8 | `dataset/raw/pjm/2022/wind_gen.csv` | CSV | 2,888,694 | 50,904 × 4 |
| 9 | `dataset/raw/pjm/2022/solar_gen.csv` | CSV | 2,951,485 | 52,560 × 4 |
| 10 | `dataset/raw/pjm/2023/hrl_load_metered.csv` | CSV | 19,934,668 | 262,800 × 8 |
| 11 | `dataset/raw/pjm/2023/wind_gen.csv` | CSV | 2,978,548 | 52,560 × 4 |
| 12 | `dataset/raw/pjm/2023/solar_gen.csv` | CSV | 2,956,190 | 52,560 × 4 |
| 13 | `dataset/raw/pjm/2024/hrl_load_metered.csv` | CSV | 19,998,420 | 263,520 × 8 |
| 14 | `dataset/raw/pjm/2024/wind_gen.csv` | CSV | 2,994,851 | 52,704 × 4 |
| 15 | `dataset/raw/pjm/2024/solar_gen.csv` | CSV | 2,975,901 | 52,704 × 4 |
| 16 | `dataset/raw/pjm/2025/hrl_load_metered.csv` | CSV | 19,949,649 | 262,800 × 8 |
| 17 | `dataset/raw/pjm/2025/wind_gen.csv` | CSV | 2,988,146 | 52,560 × 4 |
| 18 | `dataset/raw/pjm/2025/solar_gen.csv` | CSV | 2,972,157 | 52,560 × 4 |
| 19 | `dataset/processed/pjm_smart_grid_2020.csv` | CSV | 424,356 | 8,784 × 4 |
| 20 | `dataset/processed/pjm_smart_grid_2020_2025.csv` | CSV | 2,418,354 | 52,608 × 4 |
| 21 | `dataset/processed/train.csv` | CSV | 10,233,651 | 36,825 × 24 |
| 22 | `dataset/processed/validation.csv` | CSV | 2,196,150 | 7,891 × 24 |
| 23 | `dataset/processed/test.csv` | CSV | 2,220,214 | 7,892 × 24 |
| 24 | `dataset/processed/tft_frame.csv` | CSV | 8,059,611 | 52,440 × 14 |
| 25 | `dataset/processed/pjm_features.csv` | CSV | 14,649,541 | 52,608 × 24 |
| 26 | `dataset/processed/pjm_tabular_features.csv` | CSV | 15,322,277 | 52,440 × 32 |
| 27 | `dataset/processed/pjm_tft_features.csv` | CSV | 7,828,847 | 52,440 × 22 |
| 28 | `dataset/processed/train_features.csv` | CSV | 10,964,836 | 36,708 × 33 |
| 29 | `dataset/processed/validation_features.csv` | CSV | 2,350,346 | 7,866 × 33 |
| 30 | `dataset/processed/test_features.csv` | CSV | 2,374,954 | 7,866 × 33 |
| 31 | `dataset/processed/pjm_sequences.csv` | CSV | 726,975,568 | 52,272 × 1,682 |

## 2. Raw PJM files (per-file detail)

**Load files (`hrl_load_metered.csv`, all years)** — columns (8): `datetime_beginning_utc, datetime_beginning_ept, nerc_region, mkt_region, zone, load_area, mw, is_verified`. Timestamp column: `datetime_beginning_utc` (format `MM/DD/YYYY HH:MM:SS AM/PM`, UTC).

| Year | Rows | Unique timestamps | Duplicate timestamps* | Range (UTC) | Missing (all cols) | Exact dup rows | Frequency |
|---|---|---|---|---|---|---|---|
| 2020 | 263,520 | 8,784 | 254,736 | 2020-01-01 05:00 → 2021-01-01 04:00 | 0 | 0 | hourly (modal diff 1h on sorted unique) |
| 2021 | 262,800 | 8,760 | 254,040 | 2021-01-01 05:00 → 2022-01-01 04:00 | 0 | 0 | hourly |
| 2022 | 262,800 | 8,760 | 254,040 | 2022-01-01 05:00 → 2023-01-01 04:00 | 0 | 0 | hourly |
| 2023 | 262,800 | 8,760 | 254,040 | 2023-01-01 05:00 → 2024-01-01 04:00 | 0 | 0 | hourly |
| 2024 | 263,520 | 8,784 | 254,736 | 2024-01-01 05:00 → 2025-01-01 04:00 | 0 | 0 | hourly |
| 2025 | 262,800 | 8,760 | 254,040 | 2025-01-01 05:00 → 2026-01-01 04:00 | 0 | 0 | hourly |

\*Duplicates are structural (30 load-areas × hourly timestamps), not data errors — matches raw audit (30 rows per timestamp).

**Wind files (`wind_gen.csv`, all years)** — columns (4): `datetime_beginning_utc, datetime_beginning_ept, area, wind_generation_mw`. Timestamp: `datetime_beginning_utc`.

| Year | Rows | Unique ts | Dup ts* | Range (UTC) | Missing | Exact dup rows | Frequency |
|---|---|---|---|---|---|---|---|
| 2020 | 43,920 | 8,784 | 35,136 | 2020-01-01 05:00 → 2021-01-01 04:00 | 0 | 0 | hourly |
| 2021 | 43,800 | 8,760 | 35,040 | 2021-01-01 05:00 → 2022-01-01 04:00 | 0 | 0 | hourly |
| 2022 | 50,904 | 8,760 | 42,144 | 2022-01-01 05:00 → 2023-01-01 04:00 | 0 | 0 | hourly |
| 2023 | 52,560 | 8,760 | 43,800 | 2023-01-01 05:00 → 2024-01-01 04:00 | 0 | 0 | hourly |
| 2024 | 52,704 | 8,784 | 43,920 | 2024-01-01 05:00 → 2025-01-01 04:00 | 0 | 0 | hourly |
| 2025 | 52,560 | 8,760 | 43,800 | 2025-01-01 05:00 → 2026-01-01 04:00 | 0 | 0 | hourly |

\*Duplicates structural (5–6 areas × hourly; 2020–2021: 5 areas; 2022–2025: 6 areas incl. OTHER).

**Solar files (`solar_gen.csv`, all years)** — columns (4): `datetime_beginning_utc, datetime_beginning_ept, area, solar_generation_mw`. Timestamp: `datetime_beginning_utc`.

| Year | Rows | Unique ts | Dup ts* | Range (UTC) | Missing | Exact dup rows | Frequency |
|---|---|---|---|---|---|---|---|
| 2020 | 51,336 | 8,784 | 42,552 | 2020-01-01 05:00 → 2021-01-01 04:00 | 0 | 0 | hourly |
| 2021 | 52,560 | 8,760 | 43,800 | 2021-01-01 05:00 → 2022-01-01 04:00 | 0 | 0 | hourly |
| 2022 | 52,560 | 8,760 | 43,800 | 2022-01-01 05:00 → 2023-01-01 04:00 | 0 | 0 | hourly |
| 2023 | 52,560 | 8,760 | 43,800 | 2023-01-01 05:00 → 2024-01-01 04:00 | 0 | 0 | hourly |
| 2024 | 52,704 | 8,784 | 43,920 | 2024-01-01 05:00 → 2025-01-01 04:00 | 0 | 0 | hourly |
| 2025 | 52,560 | 8,760 | 43,800 | 2025-01-01 05:00 → 2026-01-01 04:00 | 0 | 0 | hourly |

\*Duplicates structural (6 areas × hourly; 2020 OTHER partial: 7,416 rows). Zero missing values in all raw files as stored (negative wind/solar values exist per raw audit but are values, not missing — clipping happens downstream, unverified here).

## 3. Processed files (per-file detail)

**Merged canonical datasets** — columns: `timestamp, pjm_load_mw, wind_generation_mw, solar_generation_mw`. Timestamp: `timestamp` (`YYYY-MM-DD HH:MM:SS`, UTC, hourly).

| File | Rows | Cols | Range | Unique ts / dup ts | Missing | Exact dup rows |
|---|---|---|---|---|---|---|
| `pjm_smart_grid_2020.csv` | 8,784 | 4 | 2020-01-01 05:00 → 2021-01-01 04:00 | 8,784 / 0 | 0 | 0 |
| `pjm_smart_grid_2020_2025.csv` | 52,608 | 4 | 2020-01-01 05:00 → 2026-01-01 04:00 | 52,608 / 0 | 0 | 0 |

**Canonical split frames (24 cols)** — `timestamp, target, group_id, solar, wind, hour, day_of_week, day_of_month, month, week_of_year, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, lag_1, lag_24, lag_168, rolling_mean_24, rolling_std_24, rolling_mean_168, rolling_std_168`.

| File | Rows | Range | Missing | Exact dup rows |
|---|---|---|---|---|
| `train.csv` | 36,825 | 2020-01-01 05:00 → 2024-03-14 13:00 | **577** (warmup: lag_1:1, lag_24:24, lag_168:168, rolling_mean_24:24, rolling_std_24:24, rolling_mean_168:168, rolling_std_168:168) | 0 |
| `validation.csv` | 7,891 | 2024-03-14 14:00 → 2025-02-06 08:00 | 0 | 0 |
| `test.csv` | 7,892 | 2025-02-06 09:00 → 2026-01-01 04:00 | 0 | 0 |
| `pjm_features.csv` | 52,608 | 2020-01-01 05:00 → 2026-01-01 04:00 | 577 (same warmup pattern) | 0 |

All: unique timestamps, no duplicate timestamps, hourly modal diff.

**TFT frames** — `tft_frame.csv` (14 cols): `time_idx, group_id, electricity_demand, hour, day_of_week, solar_generation, wind_generation, lag_1, lag_24, lag_168, rolling_mean_24, rolling_std_24, rolling_mean_168, rolling_std_168`; 52,440 rows, 0 missing, 0 exact dup rows. `time_idx` is an integer sequence index (0–52,439), not a datetime — frequency inherited hourly from source ordering. `pjm_tft_features.csv` (22 cols, 52,440 rows): `time_idx, group_id, target, hour, day_of_week, month, is_weekend, hour_sin, hour_cos, month_sin, month_cos, lag_1, lag_24, lag_168, rolling_mean_24, rolling_mean_168, total_renewable_mw, net_load, high_demand_flag, peak_hour_flag, wind_generation_mw, solar_generation_mw`; 0 missing.

**Extended feature frames (33 cols)** — `timestamp, pjm_load_mw, wind_generation_mw, solar_generation_mw, hour, day_of_week, day_of_month, month, quarter, year, is_weekend, season, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, lag_1, lag_24, lag_168, rolling_mean_24, rolling_std_24, rolling_mean_168, rolling_std_168, total_renewable_mw, renewable_ratio, net_load, wind_share, solar_share, high_demand_flag, low_renewable_flag, peak_hour_flag`.

| File | Rows | Range | Missing | Exact dup rows |
|---|---|---|---|---|
| `train_features.csv` | 36,708 | 2020-01-08 05:00 → 2024-03-16 16:00 | 0 | 0 |
| `validation_features.csv` | 7,866 | 2024-03-16 17:00 → 2025-02-07 10:00 | 0 | 0 |
| `test_features.csv` | 7,866 | 2025-02-07 11:00 → 2026-01-01 04:00 | 0 | 0 |

**`pjm_tabular_features.csv`** (32 cols, 52,440 rows, 2020-01-08 05:00 → 2026-01-01 04:00): same 33-col family minus `season` (columns verified in header scan); 0 missing, 0 exact dup rows. **Note:** `net_load` and `total_renewable_mw` exist in this extended family (`*_features.csv`, `pjm_tabular_features.csv`, `pjm_tft_features.csv`) but NOT in the canonical 24-col pipeline frames — explains the SHAP phantom-feature flag (explainability outputs reference columns absent from the v1 training frames).

**`pjm_sequences.csv`** (1,682 cols, 52,272 rows, 2020-01-15 05:00 → 2026-01-01 04:00, 727 MB): `timestamp, target` + 168 flattened steps × 10 signals (`seq_{pjm_load_mw,wind_generation_mw,solar_generation_mw,hour,is_weekend,total_renewable_mw,net_load,lag_1,lag_24,lag_168}_t-168…t-1`); 0 missing in any column; 0 duplicate timestamps; hourly. Exact full-row duplicate scan not performed (file > 50 MB; chunked pandas `duplicated()` skipped for memory reasons — stated, not assumed).

## 4. Weather files

**None found.** Searched `dataset/` (all 31 files: load/wind/solar/derived only) and `data/` (`data/raw/weather/{era5,noaa}/` contain only `.gitkeep`). No temperature, humidity, wind-speed, or cloud file exists in the repository. Any weather-dependent result would have no source data — correctly, none is claimed (`configs/pjm_data.yaml` weather fields are `null`).

## 5. Cross-checks against existing audit artifacts

- Row/column counts, timestamp ranges, and duplicate-timestamp magnitudes for all 18 raw files match `results/data_validation/pjm_raw_audit_2020_2025.json`.
- Merged 52,608-row / 100%-continuity / 0-missing claims match `pjm_processed_2020_2025_report.json` and `pjm_final_metadata.json`.
- New verified facts beyond prior audits: per-file byte sizes + sha256 (`data/checksums/sha256.txt`), 0 missing in all raw files as stored, 0 exact full-row duplicates in all files ≤ 50 MB, 577 warmup NaNs localized to `train.csv`/`pjm_features.csv` lag/rolling columns, `pjm_sequences.csv` shape (52,272 × 1,682) with 0 missing, `time_idx` integer (not datetime) semantics, extended-family column lists.

## 6. Recommended placement mapping (logical only — NO files moved in Phase 1A)

| Source file (frozen, do not move) | New logical location (Phase 1+ copy/build target) |
|---|---|
| `dataset/raw/pjm/{2020..2025}/hrl_load_metered.csv` | `data/raw/pjm/load/` (per-year load drops) |
| `dataset/raw/pjm/{2020..2025}/wind_gen.csv` | `data/raw/pjm/wind/` |
| `dataset/raw/pjm/{2020..2025}/solar_gen.csv` | `data/raw/pjm/solar/` |
| *(to be acquired Phase 1)* ERA5 reanalysis | `data/raw/weather/era5/` |
| *(to be acquired Phase 1)* NOAA ISD stations | `data/raw/weather/noaa/` |
| *(to be acquired Phase 1)* holidays/alerts | `data/raw/external/` |
| `dataset/processed/pjm_smart_grid_2020_2025.csv` | `data/processed/v1/` pointer (frozen reference) |
| *(to be built Phase 1)* v2 canonical + weather | `data/processed/v2/pjm_weather_dataset.csv` |
| *(to be built Phase 1)* event flags | `data/processed/v2/events.parquet` |
| *(to be built Phase 3)* common origins | `data/processed/v2/test_origins.parquet` |
| All 31 hashes | `data/checksums/sha256.txt` (done, §7) |

## 7. Checksums

`data/checksums/sha256.txt` — sha256 of all 31 files as stored on disk (streaming hash, read-only). Re-hash after any authorized future move to detect silent corruption.
