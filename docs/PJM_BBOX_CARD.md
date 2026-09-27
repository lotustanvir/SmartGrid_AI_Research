# PJM Bounding Box Card (Phase 1D-C2-A)

**Status:** BOUNDARY-BLOCKED — bounding box NOT finalized; no coordinates issued. No downloads. No bbox file created. `configs/era5_request.json` left unchanged.

## 1. Boundary source (inspected, not acquired)

- Expected path: `data/raw/external/pjm_boundary.geojson` (`configs/pjm_boundary.yaml: path`).
- CRS: EPSG:4326. Format: GeoJSON preferred (Shapefile converted on ingest).
- Source: PJM-published transmission-zone boundary; fallback Census TIGER state-clip.
- **File status: DOES NOT EXIST** — `data/raw/external/` contains only `.gitkeep` (verified 2026-09-24 UTC).

## 2. CRS

Locked EPSG:4326 on acquisition (boundary spec + footprint lock agree; no reprojection ambiguity by design).

## 3. Bounding box coordinates

**Not issued.** Report: *"Boundary file required before bbox calculation."* On acquisition, this section records north/south/east/west derived as min/max longitude/latitude of the validated polygon, persisted to `data/raw/external/pjm_bbox.json` in the locked schema (`source_boundary, crs, north, south, east, west, created_date`), and copied into `configs/era5_request.json` `area` as CDS-ordered `[North, West, South, East]`.

## 4. Buffer decision (pre-decided so execution is mechanical)

- **Recommended: Option B — small buffer around the polygon bbox (one ERA5 cell, 0.25°, on all sides).**
- Why: the footprint inclusion rule is centroid-in-polygon on a 0.25° grid; an exact bbox risks clipping edge cells whose centroids fall just outside the raw min/max but whose areas overlap the territory, and CDS grid alignment can shift cell edges relative to the polygon. A 0.25° buffer guarantees the downloaded box contains every candidate cell while adding negligible transfer (~1-cell rim). Exact-bbox (Option A) saves almost nothing and risks missing edge cells — rejected.
- Execution rule: buffer applied AFTER validation (CRS EPSG:4326, geometry validity, coverage sanity), recorded in `pjm_bbox.json` alongside unbuffered values.

## 5. ERA5 usage

Buffered box [48.5, -92.0, 33.5, -73.5] written to `configs/era5_request.json` `area` (variables/dates/format untouched) — hash request → execute pull → manifest in `data/raw/weather/era5/`.

## 6. Final coordinates and validation (Phase 1D-C2-B, executed 2026-09-25)

- **Boundary source:** FALLBACK — US Census cartographic boundary `cb_2024_us_state_20m` (TIGER/GENZ2024 derived, public domain), 14 state polygons (DE/DC/IL/IN/KY/MD/MI/NJ/NC/OH/PA/TN/VA/WV) as GeoJSON MultiPolygon at `data/raw/external/pjm_boundary.geojson` (94,371 bytes). Official PJM-published polygon unavailable — limitation owned; partial-state overstatement (full IL/KY/NC/TN/MI incl. Michigan UP to 48.21°N) documented in metadata.
- **Validation result:** PASS (fallback) — file readable, 14 polygons, all rings closed, non-empty, CRS carried EPSG:4326 (source NAD83, meter-level equivalence noted). Report: `results/q1/pjm_boundary_validation.json`. sha256: `a4f288bc…5dadb88` (`data/raw/external/pjm_boundary.sha256`).
- **Final coordinates:** polygon bbox N 48.21065 / S 33.851112 / E −73.893979 / W −91.511956 (`data/raw/external/pjm_bbox.json`); buffered + grid-snapped request box N 48.5 / W −92.0 / S 33.5 / E −73.5 (0.25° rim per §4).
