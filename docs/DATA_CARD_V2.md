# DATA_CARD_V2 — Weather-Aware PJM Dataset (Phase 1D-C3)

**Status:** BUILDABLE AND EXTENDABLE. Pipeline complete; weather coverage grows
as the background CDS download fills 2021-2025. v1 untouched (sha-pinned).
No models trained. No evaluation code modified.

## 1. Dataset identity

- **v1 (frozen, read-only):** `dataset/processed/pjm_smart_grid_2020_2025.csv`
  (52,608 rows, 2020-01-01 05:00 UTC -> 2026-01-01 04:00 UTC, 4 cols).
  sha256 recorded in `results/data_validation/dataset_metadata_v2.json`.
- **Weather series (new):** `data/processed/pjm_weather_2020_2025.csv`
  (13 cols: timestamp, temperature_c, dewpoint_c, relative_humidity,
  wind_speed, cloud_cover, solar_radiation_wm2, heat_index_c, wet_bulb_c,
  cdh, hdh, daylight_proxy, n_cells). Hourly, monotonic, 0 duplicate
  timestamps, 0 gaps over available months.
- **v2 merged (new):** `dataset/processed/pjm_smart_grid_2020_2025_v2.csv`
  (14 cols: timestamp, pjm_load_mw, wind_generation_mw, solar_generation_mw,
  temperature, humidity, wind_speed, cloud_cover, solar_radiation,
  total_renewable, net_load, renewable_penetration, cdh, hdh).
  Row count == v1 (52,608); merge drops 0 rows by construction
  (merge_asof backward, 1h tolerance); weather NaN outside covered months.
- **Weights/coverage:** `data/processed/v2/pjm_footprint_weights.csv`
  (cell_id, lat, lon, in_footprint, w_simple, w_pop) +
  `data/processed/v2/pjm_footprint_coverage.json`.
- **Reports:** `results/q1/weather_validation_report.json` (aggregate gates +
  coverage + known limitations), `results/q1/era5_validation_<Y>_<M>.json`
  (per-month), `results/data_validation/dataset_metadata_v2.json`
  (join retention, missing %, checksums, provenance).

## 2. ERA5 source

- Product: `reanalysis-era5-single-levels` (ECMWF, via CDS API).
- Specs: `configs/era5_request.json` (5 instantaneous vars, proven pattern) +
  `configs/era5_request_v2.json` (adds `surface_solar_radiation_downwards`,
  documents the abandoned mixed 6-var request).
- Native variables pulled: `2m_temperature`, `2m_dewpoint_temperature`,
  `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_cloud_cover`
  (monthly files `era5_pjm_<Y>_<M>.nc`) + `surface_solar_radiation_downwards`
  (supplement files `era5_pjm_<Y>_<M>_ssrd.nc`, always pulled separately).
- Bounding box (CDS N/W/S/E): 48.5 / -92.0 / 33.5 / -73.5 (0.25 deg grid,
  61 x 75 cells). Source: `data/raw/external/pjm_bbox.json`.
- Download period: 2020-01-01 -> 2025-12-31 (monthly requests) + 2026-01-01
  00:00-04:00 tail (pending). As of this card: 2020 12/12 + 2021-01..03
  present; remainder downloading (see validation report `months_missing`).
- Request archives: `monthly/request_<Y>_<M>[_ssrd].json`; metadata:
  `monthly/metadata_<Y>_<M>[_ssrd].json`; checksums:
  `monthly/sha256_<Y>_<M>[_ssrd].txt`; run log:
  `monthly/batch_download_log.jsonl`. Credentials: `~/.cdsapirc` only
  (never in repo).

## 3. Variables and derivations

| Column | Derivation | Units |
|---|---|---|
| temperature_c | t2m footprint mean, K->C | degC |
| dewpoint_c | d2m footprint mean, K->C | degC |
| relative_humidity | Magnus (Alduchov-Eskridge) from footprint T/Td, clipped 0-100 | % |
| wind_speed | sqrt(u10^2+v10^2) of footprint means | m/s |
| cloud_cover | tcc footprint mean; /100 when native max>1.5 (ERA5 native = %) | 0-1 |
| solar_radiation_wm2 | max(0, ssrd(t)-ssrd(t-1h))/3600 on concatenated timeline | W/m2 |
| heat_index_c | Rothfusz (NWS, incl. low/high-RH adjustments); applied iff T>=26.7C and RH>=40, else = T | degC |
| wet_bulb_c | Stull (2011) from T/RH | degC |
| cdh / hdh | max(T-22,0) / max(18-T,0) | degC-hours |
| daylight_proxy | max(0, cos(zenith)), Spencer declination, solar time at footprint mean lat/lon; timestamp-only | 0-1 |
| total_renewable | wind + solar generation (PJM metered) | MW |
| net_load | pjm_load - total_renewable | MW |
| renewable_penetration | total_renewable / pjm_load | fraction |

## 4. Aggregation method

Footprint = ERA5 cells whose centroid falls inside
`data/raw/external/pjm_boundary.geojson` (MultiPolygon, 14 polygons;
ray-casting with bbox prefilter, stdlib only): **1340 of 4575 bbox cells**,
mean lat 38.37. Published series = boundary-masked SIMPLE mean (Method A).
Deviation from `docs/PJM_WEATHER_FOOTPRINT.md` lock (owned): Method B
(population-weighted) BLOCKED — population product never acquired
(`configs/pjm_weather_weights.yaml` TODO); `w_pop` ships NaN with the reason
in the coverage JSON; code path ready. Weights sum to 1 (asserted).
Merge: `pd.merge_asof` backward, 1h tolerance, no forward fill; unmatched
hours -> NaN and counted (`dataset_metadata_v2.json::join`).

## 5. Validation summary (current)

- 5 instantaneous vars: 0 NaN, 0 gaps, 0 physical-range violations, 0 |z|>5
  temperature outliers over available span. Gate: PASS.
- ssrd supplements: NaN exactly where supplements pending (2021-01/02 at
  build time); 0 NaN where present.
- Coverage: validation report `period_available.coverage_pct` (20.8% at
  build; grows with downloads — re-run pipeline to refresh).

## 6. Limitations (must appear in any manuscript using v2)

1. **ssrd diurnal distortion:** CDS hourly ssrd accumulations decline over
   ~19-23 UTC daily (verified Jan+Jul 2020), impossible for same-origin
   accumulations (production-stream mixing); clipped de-accumulation gives
   ~0 W/m2 evenings and an early (~15 UTC vs ~17:20 true) peak. Valid for
   day/night contrast + magnitude; NOT for precise intraday solar timing.
   `daylight_proxy` + `cloud_cover` carry unbiased diurnal shape.
2. **Simple-mean footprint:** no population weighting (see §4); rural cells
   dilute urban heat signal. Sensitivity column planned once weights land.
3. **Partial period:** weather covers only downloaded months; v2 weather
   columns are NaN elsewhere. Any model run must filter to covered span or
   state the nowcast assumption + run the 24h-lagged sensitivity
   (WEATHER_CARD §7) before operational claims.
4. **Reanalysis, not forecast:** ERA5 latency ~5 days; deployment needs NWP
   coupling (future work). Single RTO-total series masks zonal gradients.
5. **tcc units:** normalised %->fraction automatically; assumption + observed
   max recorded per build (`conversions` in validation report).

## 7. Reproducibility instructions

```text
# 1. credentials (outside repo): ~/.cdsapirc with url + key
# 2. download (idempotent; skips valid months; quarantines, never deletes):
python scripts/download_era5_batch.py --mode ssrd --years 2020
python scripts/download_era5_batch.py --mode full --years 2021 2022 2023 2024 2025
python scripts/download_era5_batch.py --mode ssrd --years 2021 2022 2023 2024 2025
# 3. validate + build (re-runnable; v1 read-only; no training):
python scripts/process_era5.py --validate-only
python scripts/process_era5.py --full
# 4. test:
python -m pytest tests/test_weather_pipeline.py -q
```

Single-downloader lock: `monthly/.batch_download.lock` (stale locks from
dead PIDs are cleared). Do NOT mix ssrd into the 5-var request (mixed
instantaneous+accumulated CDS conversion returns unreadable payloads —
2021-01..03 incident, `batch_download_log.jsonl`). Weather deps pinned in
`requirements.txt` (cdsapi/xarray/netCDF4).
