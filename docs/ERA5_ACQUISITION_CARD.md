# ERA5 Acquisition Card — Request Specification (Phase 1D-C1, spec only)

**Status:** SPECIFICATION ONLY. No download performed. CDS API not called. No weather dataset created.
**Request record:** `configs/era5_request.json` (placeholders fixed at acquisition). Raw landing zone: `data/raw/weather/era5/` (manifest: `data/raw/weather/era5/README.md`).

## 1. Purpose

Acquire a reproducible, gap-free hourly atmospheric record over the PJM footprint for 2020–2025 to feed the population-weighted weather series (`pjm_weather_2020_2025.csv`), enabling the Ablation-C treatment (H3) and peak-slice diagnostics. ERA5 is the primary source; NOAA validates (WEATHER_CARD §2).

## 2. Dataset source

- **Dataset identifier:** `reanalysis-era5-single-levels` (Copernicus Climate Data Store).
- **Product type:** reanalysis — hourly single-level fields, ~0.25° × 0.25° native grid, 1940–present with ~5-day latency.
- **Provider:** Copernicus Climate Data Store (CDS); scripted pull with archived request JSON (this spec file, hashed at execution).
- **Format:** NetCDF (per-year files in landing zone).
- **Licensing:** Copernicus licence — free use with registration and attribution; exact licence text version archived alongside the request at acquisition time.

## 3. Variables

| Variable | ERA5 name | Unit | Purpose — why selected, expected impact |
|---|---|---|---|
| 2m temperature | `2m_temperature` | K → °C | Primary thermal driver (load U-curve); largest expected exogenous ablation gain, concentrated on heat/cold/peak slices |
| Relative humidity | *derived* (Magnus formula from T + dew point — NOT a native single-level variable) | % | Heat-stress modulation of cooling; summer-peak interaction term |
| 2m dew point | `2m_dewpoint_temperature` | K → °C | Humidity QA pair (dew point ≤ T assert); heat-index validity gate |
| 10m wind speed | derived `sqrt(u10²+v10²)` from `10m_u/v_component_of_wind` | m/s | Wind-chill heating; wind-regime marker for net-load framing |
| Total cloud cover | `total_cloud_cover` | 0–1 | Separates solar-driven net-load variance from thermal load |

Derivations (RH, wind speed) are computed at aggregation time with logged formulas — never presented as native fields.

## 4. Temporal coverage

Locked period **2020-01-01 → 2025-12-31**, hourly, 00:00–23:00 UTC (ERA5 native). Hourly resolution is required because the target is hourly and thermal ramps (evening cooling pickup, morning heating drop) occur within hours — daily means would erase the intraday structure the 1/6/24h horizons score. **Tail addendum:** PJM test data ends 2026-01-01 04:00 UTC (Phase 1A), so the executed request must extend through that hour; the spec records 2025-12-31 as the nominal lock with the 4-hour extension noted here and in `era5_request.json`.

## 5. Spatial strategy

| Option | Storage | Reproducibility | Processing complexity |
|---|---|---|---|
| **A. PJM bounding box (SELECTED)** | Small (~6 yrs × 5 vars over padded footprint box) | High: box coordinates versioned in request | Low: single request family, mask applied locally |
| B. Global + clip | Prohibitive (global hourly fields) | High but wasteful | Medium (heavy local clipping) |
| C. Intersecting cells only | Minimal transfer | Lower: cell set depends on local boundary version at request time | High: per-cell requests, API overhead |

**Selected: Option A.** Bounding-box pull with the centroid-in-polygon mask applied locally keeps transfer small, the request self-describing (box in versioned JSON), and the footprint definition (boundary polygon) decoupled from acquisition — so a boundary revision never requires re-download, only re-masking.

## 6. Request reproducibility

At execution: freeze box coordinates + variable list + period in `configs/era5_request.json`, record its sha256 (`request_hash`) and `created_date`, archive the CDS request ID, dataset version string, per-file checksums/sizes/variable lists (`data/raw/weather/era5/README.md` manifest). Any re-pull with the same hashed request must reproduce byte-identical inputs or the divergence is investigated, not averaged.

## 7. Limitations

Reanalysis latency (~5 days) — deployment needs NWP coupling (owned future work); 0.25° cells smooth urban heat islands (bias direction shared across years — reported, not corrected); cloud-cover skill below temperature skill (per-variable reporting); tail-extension hours must be verified present before v2 build or test coverage fails closed.

## 8. Acquisition Status (Phase 1D-C2-C)

### 8.0 Prior probe (2026-09-25, spec phase)

- **Acquisition completed: NO — blocked on CDS credentials.**
- Request archived: `data/raw/weather/era5/request.json` (sha256 `1f695c33…f07976`, created 2026-09-25T; box [48.5, −92.0, 33.5, −73.5], 5 variables, 2020–2025 + tail note). Verified against `configs/era5_request.json`: dataset, 5 variables, period, hourly, area all match.
- Tooling ready: `cdsapi 0.7.7` + `xarray` + `netCDF4` installed; probe retrieve failed cleanly with `Missing/incomplete configuration file: ~/.cdsapirc` — no partial files written, nothing to clean up.
- **To unblock (user action, ~10 min):** (1) create a CDS account and accept the ERA5 licence/terms; (2) write `%USERPROFILE%\.cdsapirc` with url + personal key; (3) re-run retrieval. Recommended invocation: yearly splits `era5_pjm_<year>.nc` (estimated ~0.8 GB/yr for 5 vars over this box, ~4.7 GB total — estimate, not measured; a single 6-yr file is impractical against CDS request limits) plus the 2026-01-01 00:00–04:00 tail.
- **Reserved locations (empty until download):** raw file(s) `data/raw/weather/era5/`; `metadata.json` (dataset, provider, CDS request ID, date, variables, bounds, range, format, size, checksum); `sha256.txt` (NetCDF + request.json + metadata.json); validation report `results/q1/era5_raw_validation.json` (checks A–F per phase spec: readability, t2m/d2m/u10/v10/tcc, 2020→2025 coverage, hourly step, bbox coverage, missing %).
- **Validation status:** NOT EXECUTED — no raw file exists; no validation artifact fabricated.
- **Checksum reference:** request-archive hash above; file checksums pending download.

### 8.1 Execution attempt (Phase 1D-C2-C, 2026-09-25)

- **Acquisition completed: NO — BLOCKED (CDS server rejected single 6-yr request).**
- Downloader: `scripts/download_era5.py` (loads `configs/era5_request.json`, `cdsapi`, no hard-coded secrets, creates output dir automatically).
- Expected file: `data/raw/weather/era5/era5_pjm_2020_2025.nc` — **NOT PRESENT** (verified absent; no fabricated or partial files).
- Request archived: `data/raw/weather/era5/request.json` (executed record with `cds_request`, fresh `request_hash` + `created_date`).
- **Exact CDS error:** `403 Client Error: Forbidden for url: https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-single-levels/execution` — `cost limits exceeded. Your request is too large, please reduce your selection.` Credentials accepted (client initialised, request submitted); failure is server-side size limit, not auth.
- **Validation result:** NOT EXECUTED — no NetCDF exists; `results/q1/era5_raw_validation.json` not created (no artifact fabricated).
- **Checksum location:** `data/raw/weather/era5/sha256.txt` NOT created (nothing to hash); `metadata.json` NOT created (no file to describe).
- **Recommended next step (not executed, needs approval):** split into yearly (or smaller) requests, e.g. `era5_pjm_<year>.nc` per year plus the 2026-01-01 00:00–04:00 tail, then concatenate locally to `era5_pjm_2020_2025.nc`; or narrow variable/day chunks per CDS cost guidance. Single 6-yr request is impractical against CDS request limits.

### 8.2 Yearly-split attempt: 2020 (2026-09-25)

- **Acquisition completed: NO — BLOCKED (CDS rejected even the single-year request).**
- Command: `python scripts/download_era5.py --year 2020` (same 5 variables, same bbox, same hourly resolution; only `year` narrowed to `["2020"]`).
- Expected file: `data/raw/weather/era5/era5_pjm_2020.nc` — **NOT PRESENT** (verified absent; no fabricated or partial files).
- Request archived: `data/raw/weather/era5/request_2020.json`.
- **Exact CDS error:** `403 Client Error: Forbidden` — `cost limits exceeded. Your request is too large, please reduce your selection.`
- **Validation result:** NOT EXECUTED — `results/q1/era5_validation_2020.json` not created.
- **Checksum/metadata:** `sha256_2020.txt` / `metadata_2020.json` NOT created (nothing to hash or describe).
- Strategy: `docs/ERA5_YEARLY_ACQUISITION_PLAN.md` (yearly files + concatenation plan). Next step if approved: monthly splits.

### 8.3 Monthly-split pilot: 2020-01 (2026-09-25)

- **Acquisition: SUCCESS (pilot month).**
- Command: `python scripts/download_era5.py --year 2020 --month 01`
  (strategy: `docs/ERA5_MONTHLY_ACQUISITION_PLAN.md`; 72 monthly requests total).
- CDS request ID: `fda37d98-07d1-47c7-91cf-4c1388b91916` (accepted → successful).
- File: `data/raw/weather/era5/monthly/era5_pjm_2020_01.nc`
  (30,167,574 bytes).
- Request archived: `monthly/request_2020_01.json`;
  metadata `monthly/metadata_2020_01.json`;
  checksum `monthly/sha256_2020_01.txt` (nc + request + metadata).
- **Validation result:** `results/q1/era5_validation_2020_01.json` — **PASS**
  (readable; t2m/d2m/u10/v10/tcc; 744 hourly steps; bbox exact; 0 NaN).
- Remaining: 71 months + 2026-01-01 tail, pending approval.
