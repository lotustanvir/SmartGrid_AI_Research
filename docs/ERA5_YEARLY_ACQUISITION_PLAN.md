# ERA5 Yearly Acquisition Plan (Phase 1D-C2-C, follow-up)

**Status:** ACTIVE — supersedes the single-request approach for execution; the
request specification (`configs/era5_request.json`) is unchanged.
**Authority:** `docs/ERA5_ACQUISITION_CARD.md` §8.1 (blocked single-request attempt).

## 1. Reason for yearly split

The single 6-year request (2020–2025, 5 variables, full PJM bounding box,
hourly) was submitted via `scripts/download_era5.py` and rejected by the CDS
server. Credentials were accepted (client initialised, request submitted); the
failure is a server-side size limit, not authentication. Splitting the period
into one request per calendar year keeps every request identical in variables,
bounding box, and hourly resolution while dividing the data volume by ~6.

## 2. CDS cost limitation (exact error, 2026-09-25)

`403 Client Error: Forbidden for url:
https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-single-levels/execution`
— `cost limits exceeded. Your request is too large, please reduce your
selection.`

Implication: CDS enforces a per-request cost cap. A full 6-year hourly pull at
~0.25° over the PJM box (estimated ~4.7 GB total, ~0.8 GB/yr — estimate, not
measured) exceeds it. Yearly requests (~0.8 GB each) are the documented
fallback in `docs/ERA5_ACQUISITION_CARD.md` §8. If a yearly request still
exceeds the cap, split further by month groups — never by degrading variables,
bbox, or hourly resolution.

## 3. Yearly file structure

Landing zone: `data/raw/weather/era5/`

| Year | CDS `year` field | Output file | Request archive | Metadata | Checksum |
|---|---|---|---|---|---|
| 2020 | `["2020"]` | `era5_pjm_2020.nc` | `request_2020.json` | `metadata_2020.json` | `sha256_2020.txt` |
| 2021 | `["2021"]` | `era5_pjm_2021.nc` | `request_2021.json` | `metadata_2021.json` | `sha256_2021.txt` |
| 2022 | `["2022"]` | `era5_pjm_2022.nc` | `request_2022.json` | `metadata_2022.json` | `sha256_2022.txt` |
| 2023 | `["2023"]` | `era5_pjm_2023.nc` | `request_2023.json` | `metadata_2023.json` | `sha256_2023.txt` |
| 2024 | `["2024"]` | `era5_pjm_2024.nc` | `request_2024.json` | `metadata_2024.json` | `sha256_2024.txt` |
| 2025 | `["2025"]` | `era5_pjm_2025.nc` | `request_2025.json` | `metadata_2025.json` | `sha256_2025.txt` |

Fixed across all years (from `configs/era5_request.json`):

- Dataset: `reanalysis-era5-single-levels`, `product_type: reanalysis`
- Variables: `2m_temperature`, `2m_dewpoint_temperature`,
  `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_cloud_cover`
- Area (CDS order N,W,S,E): `[48.5, -92.0, 33.5, -73.5]`
- Months: 01–12, days: 01–31, time: 00:00–23:00 UTC hourly
- Format: `netcdf`

Per-year invocation:

```text
python scripts/download_era5.py --year 2020
python scripts/download_era5.py --year 2021
...
```

`request.json` (root 6-yr attempt record) is preserved untouched; each yearly
run archives its own `request_<year>.json` with fresh `request_hash` +
`created_date`.

## 4. Concatenation plan (after all years land)

1. Validate each yearly file independently
   (`results/q1/era5_validation_<year>.json`): NetCDF readable, variables
   t2m/d2m/u10/v10/tcc present, hourly contiguous time axis covering
   `<year>-01-01` → `<year>-12-31` 23:00 UTC, bbox matches, missing-value
   report. Leap years (2020, 2024: 8784 hrs; others: 8760 hrs) expected.
2. Concatenate along `time` with `xarray.open_mfdataset` /
   `xr.concat(..., dim="time")`, sort by time, assert contiguity
   (no gaps/overlaps at year boundaries) and uniform grids.
3. Tail extension: PJM test data ends 2026-01-01 04:00 UTC, so a final small
   request for 2026-01-01 00:00–04:00 UTC is appended before the v2 build
   (see `docs/ERA5_ACQUISITION_CARD.md` §4 and `era5_request.json`
   `time_note`).
4. Write the analysis-ready file
   `data/raw/weather/era5/era5_pjm_2020_2025.nc`, record its own checksum +
   metadata, and re-run validation over the full range before any
   aggregation into `pjm_weather_2020_2025.csv`.
5. No file is ever hand-edited or synthesised: every byte traces to a CDS
   retrieve with an archived request. Any failed year is retried as-is or
   split by months — never worked around with fabricated data.

## 5. Execution log

### 2026-09-25 — 2020 yearly request: BLOCKED

- Command: `python scripts/download_era5.py --year 2020`
- Target: `data/raw/weather/era5/era5_pjm_2020.nc` — **NOT PRESENT**
  (verified absent; no partial or fabricated files).
- Request archived: `data/raw/weather/era5/request_2020.json`
  (single-year `cds_request`, fresh `request_hash` + `created_date`).
- Exact CDS error: `403 Client Error: Forbidden` —
  `cost limits exceeded. Your request is too large, please reduce your
  selection.` A full single-year pull (12 months × 31 days × 24 hrs ×
  5 vars over this box) still exceeds the CDS per-request cost cap.
- Consequence: `metadata_2020.json`, `sha256_2020.txt`, and
  `results/q1/era5_validation_2020.json` were NOT created (nothing to
  describe, hash, or validate).
- Next step (not executed, needs approval): split further, e.g.
  `--month` support in `scripts/download_era5.py` (one month per request,
  72 requests total for 2020–2025 plus the 2026-01-01 tail), then
  concatenate yearly → full period per §4.
