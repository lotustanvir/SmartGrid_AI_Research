# data/ — Canonical Q1 data tree (Phase 1 scaffolding, empty pending Phase 1)

- `raw/pjm/{load,wind,solar}/` — PJM Data Miner drops per energy type (populated Phase 1; legacy sous `dataset/raw/pjm/` stays untouched).
- `raw/weather/{era5,noaa}/` — ERA5 reanalysis (primary) / NOAA ISD fallback (Phase 1).
- `raw/external/` — holidays, alerts, misc third-party pulls with provenance notes.
- `processed/v1/` — pointer to frozen legacy (`dataset/processed/pjm_smart_grid_2020_2025.csv`; never copied).
- `processed/v2/` — `pjm_weather_dataset.csv`, `events.parquet`, `test_origins.parquet`, `dataset_metadata.json` (built Phase 1–3).
- `checksums/sha256.txt` — hashes recorded at freeze; every EXP row references them.
