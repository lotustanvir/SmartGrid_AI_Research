# data/raw/weather/era5/ — Raw ERA5 Landing Zone (Phase 1D-C1 manifest)

**Status:** EMPTY — specification only (`configs/era5_request.json`, `docs/ERA5_ACQUISITION_CARD.md`). No downloads performed.

## Expected files (populated at acquisition, Phase 1D+)

- `request.json` — executed copy of `configs/era5_request.json` with `request_hash` + `created_date` filled.
- `metadata.json` — per-file manifest (fields below).
- Raw ERA5 files — NetCDF per year (naming: `era5_<var>_<year>.nc` or as pulled; exact names logged in `metadata.json`).

## Metadata fields (per raw file, recorded at acquisition)

- download date
- CDS request ID
- dataset version (`reanalysis-era5-single-levels` version string)
- checksum (sha256)
- file size (bytes)
- variable list (as stored)
