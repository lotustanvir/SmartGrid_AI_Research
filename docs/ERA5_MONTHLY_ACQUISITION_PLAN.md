# ERA5 Monthly Acquisition Plan (Phase 1D-C2, follow-up)

**Status:** ACTIVE — supersedes yearly splits for execution; the request
specification (`configs/era5_request.json`) is unchanged.
**Authority:** `docs/ERA5_ACQUISITION_CARD.md` §§8.1–8.2; prior plans
`docs/ERA5_YEARLY_ACQUISITION_PLAN.md` (yearly attempt also blocked).

## 1. CDS cost limitation

CDS enforces a per-request cost cap and rejects oversized requests at
submission time with:

`403 Client Error: Forbidden` — `cost limits exceeded. Your request is too
large, please reduce your selection.`

History on this machine (credentials accepted in all cases; failures are
server-side size limits, not auth):

| Attempt | Request size | Result |
|---|---|---|
| Full 6-yr (2020–2025, 5 vars, PJM box, hourly) | ~6× yearly | BLOCKED |
| Yearly 2020 (12 months × 31 days × 24 hrs × 5 vars) | ~0.8 GB (estimate) | BLOCKED |

## 2. Reason monthly split selected

A single month (1 month × up-to-31 days × 24 hrs × 5 vars over the same box)
is ~1/12 of the blocked yearly request (~65 MB estimated) — small enough to
plausibly pass the cost cap while keeping variables, bounding box, and hourly
resolution identical. No degradation of science content: months tile the full
period exactly, and per-month files concatenate losslessly along `time`.
Finer splits (weekly/daily) are held in reserve if a month still fails.

## 3. Request count (72)

6 years × 12 months = **72 monthly requests** (2020-01 … 2025-12), plus one
tail request for 2026-01-01 00:00–04:00 UTC (PJM test coverage; see
`docs/ERA5_ACQUISITION_CARD.md` §4).

Invocation per month:

```text
python scripts/download_era5.py --year 2020 --month 01
```

Landing zone `data/raw/weather/era5/monthly/`:

- Data: `era5_pjm_<year>_<month>.nc` (e.g. `era5_pjm_2020_01.nc`)
- Request archive: `request_<year>_<month>.json`
- Metadata: `metadata_<year>_<month>.json`
- Checksum: `sha256_<year>_<month>.txt`

Expected month lengths (hourly steps): 744 (31-day), 720/696 (30-day),
672/624 (Feb; 2020/2024 leap → 696). Only **2020-01 is executed as a pilot**;
remaining 71 months proceed only after the pilot validates.

## 4. Merge strategy

1. Validate each month independently
   (`results/q1/era5_validation_<year>_<month>.json`): NetCDF readable,
   t2m/d2m/u10/v10/tcc present, hourly contiguous axis covering the calendar
   month exactly, bbox matches, missing-value report.
2. Concatenate months along `time` (`xarray.open_mfdataset` / `xr.concat`),
   sort, assert contiguity at month boundaries (no gaps/overlaps) and uniform
   grids; expected totals 8784 hrs (2020, 2024) / 8760 hrs (other years).
3. Append the 2026-01-01 00:00–04:00 UTC tail, then write yearly intermediates
   (`era5_pjm_<year>.nc`) and the analysis-ready
   `data/raw/weather/era5/era5_pjm_2020_2025.nc`, each with checksum +
   metadata, re-validating the full range before aggregation into
   `pjm_weather_2020_2025.csv`.

## 5. Retry strategy

- Transient failure (network/timeout, 5xx): retry the same month as-is, up
  to 3 attempts with backoff; each attempt re-archives its request record.
- Cost-limit rejection (403 too-large): do NOT retry as-is — split the month
  into two half-month (day-range) requests and concatenate locally.
- Corrupt/empty file (fails validation): delete, re-download the same month;
  never patch bytes by hand.
- Never synthesise, interpolate-as-raw, or copy another month's data. Any
  month that cannot be retrieved is reported missing, not fabricated.

## 6. Execution log

### 2026-09-25 — pilot 2020-01: SUCCESS

- Command: `python scripts/download_era5.py --year 2020 --month 01`
- CDS request ID: `fda37d98-07d1-47c7-91cf-4c1388b91916`
  (accepted → running → successful; one transient 502 retried by client).
- File: `data/raw/weather/era5/monthly/era5_pjm_2020_01.nc`
  (30,167,574 bytes, ~28.8 MB).
- Request archived: `monthly/request_2020_01.json`.
- Metadata: `monthly/metadata_2020_01.json`
  (dataset, 5 variables, year/month, bbox, size, acquisition date, request ID).
- Checksum: `monthly/sha256_2020_01.txt` (nc + request + metadata).
- Validation: `results/q1/era5_validation_2020_01.json` — **PASS**
  (readable; t2m/d2m/u10/v10/tcc present; 744 hourly steps
  2020-01-01→2020-01-31T23:00; bbox exact; 0 NaN of 17,019,000 values).
- Monthly strategy confirmed viable; remaining 71 months + tail may proceed
  on approval. Sustained rate ≈ 72 × ~30 MB ≈ 2.1 GB total (measured pilot,
  not estimated).

### 2026-09-25 — full year 2020 (Phase 1D-C2-F1): 12/12 SUCCESS

All months downloaded sequentially with
`python scripts/download_era5.py --year 2020 --month MM`; no failures, no
retries beyond client-side transient handling, nothing skipped or fabricated.

| Month | Status | File size (bytes) | Validation |
|---|---|---|---|
| 2020-01 | SUCCESS | 30,167,574 | PASS (744 hrs) |
| 2020-02 | SUCCESS | 28,369,413 | PASS (696 hrs, leap Feb) |
| 2020-03 | SUCCESS | 30,475,814 | PASS (744 hrs) |
| 2020-04 | SUCCESS | 30,160,062 | PASS (720 hrs) |
| 2020-05 | SUCCESS | 31,337,065 | PASS (744 hrs) |
| 2020-06 | SUCCESS | 30,963,531 | PASS (720 hrs) |
| 2020-07 | SUCCESS | 32,308,320 | PASS (744 hrs) |
| 2020-08 | SUCCESS | 32,094,567 | PASS (744 hrs) |
| 2020-09 | SUCCESS | 29,991,788 | PASS (720 hrs) |
| 2020-10 | SUCCESS | 30,755,570 | PASS (744 hrs) |
| 2020-11 | SUCCESS | 29,522,326 | PASS (720 hrs) |
| 2020-12 | SUCCESS | 30,745,685 | PASS (744 hrs) |

- Total 2020 storage (NetCDF): 366,891,715 bytes (~350 MB).
- Total 2020 hours: 8784 (leap year) — matches expectation.
- Every month: NetCDF readable; t2m/d2m/u10/v10/tcc present; hourly
  contiguous (3600 s); month start/end exact; bbox
  [48.5, −92.0, 33.5, −73.5] exact; 0 NaN.
- Per-month artifacts: `metadata_2020_MM.json` (dataset, variables,
  year/month, bbox, acquisition date, CDS request ID, file size),
  `sha256_2020_MM.txt` (nc + request + metadata),
  `results/q1/era5_validation_2020_MM.json` — 12/12 present.
- No merging, feature engineering, or training performed (out of scope).
