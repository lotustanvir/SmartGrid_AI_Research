# Boundary Card — PJM Footprint Definition (Phase 1D-B Design)

**Status:** DESIGN ONLY — no boundary file downloaded; spec in `configs/pjm_boundary.yaml`.

## 1. PJM footprint definition

The weather footprint is the set of ERA5 ~0.25° cells whose centroids fall inside the PJM transmission-zone boundary polygon (centroid-in-polygon rule; coastal/edge partial cells included deterministically; no manual edits). PJM covers 13 states plus DC (Mid-Atlantic through Chicago); the polygon must bound the same service territory whose summed load is the forecasting target — unlike single-city proxies that overweight one microclimate.

## 2. Source

- **Selected (primary):** PJM-published transmission-zone/service-territory boundary (authoritative geography of the target itself).
- **Fallback:** US Census TIGER/Line state polygons clipped to PJM states — fully public-domain, versioned by vintage, maximally reproducible if the PJM-published file proves unversioned or ungated.
- **Options considered:** PJM-published zones (authoritative, versioning TBD at acquisition) · Census TIGER states (public domain, strong versioning, coarser than zones) · EIA utility territories (third-party compilation, licensing/attribution overhead — rejected as primary).

## 3. CRS, format, versioning, checksum

CRS EPSG:4326 throughout; GeoJSON preferred (Shapefile converted on ingest). Vintage + access date fixed at acquisition; file sha256 recorded in `configs/pjm_boundary.yaml` and verified before any weight computation; boundary file stored at `data/raw/external/pjm_boundary.geojson`.

## 4. Validation plan (executed Phase 1D+, defined here)

CRS equals EPSG:4326 (assert) · geometry validity (no self-intersections/empty geoms; repaired only with logged method or rejected) · PJM coverage (included-cell count and bounding extent sanity vs known footprint; zero-cell or out-of-range result fails closed).

## 5. Limitations

Zone boundaries evolve (transmission changes, vintage drift — pinned vintage required); centroid rule creates hard coastal edges; fallback state-clip is coarser than zones (divergence reported if fallback is used).
