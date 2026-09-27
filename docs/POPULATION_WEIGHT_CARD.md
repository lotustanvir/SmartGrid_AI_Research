# Population Weight Card (Phase 1D-B Design)

**Status:** DESIGN ONLY — no population file downloaded; spec in `configs/population_weights.yaml`.

## 1. Population source

- **Selected (primary):** Gridded Population of the World (GPW), NASA SEDAC — global gridded, versioned revisions, citable, research-redistributable license; single 2020 reference vintage across the full 2020–2025 window.
- **Fallback/sensitivity:** WorldPop unconstrained product (independent methodology cross-check).
- **Options considered:** GPW (versioned, coarse-but-stable, best reproducibility) · WorldPop (finer, modeled covariates, revision churn) · Census TIGER/ACS tables (authoritative counts but tabular, not gridded — requires areal interpolation, rejected as primary for that reason).

| Dimension | GPW (primary) | WorldPop (fallback) | Census tables |
|---|---|---|---|
| Resolution | ~30 arc-sec gridded | ~100 m modeled | Block-group tabular |
| Coverage | Global, uniform | Global, modeled gaps | US only, complete |
| License | Research-redistributable (CC-BY family; exact terms confirmed at acquisition) | Research use (terms confirmed at acquisition) | Public domain |
| Reproducibility | High (fixed revision + hash) | Medium (revision churn) | High (vintage tables) but needs interpolation |
| PJM suitability | Direct cell aggregation | Direct, finer | Indirect (areal weighting step) |

## 2. Weighting equation

```
Weather_PJM(t) = Σ_i ( w_i × weather_i(t) ),   w_i = population_i / Σ_j population_j
```

Population aggregated UP from native resolution to ERA5 0.25° cells (never downscaled); weights sum to 1 (asserted; violation fails the build). Sensitivity series: `w_i = 1/N`.

## 3. Reproducibility requirements

Product + revision + reference year (2020) + native resolution + CRS (EPSG:4326) + file hash — all recorded in `configs/population_weights.yaml` before computation; vintage frozen for the whole study; weights output `data/processed/v2/pjm_footprint_weights.csv` (cell_id, centroid, population, w_pop, w_simple) + coverage report; EXP rows cite the weights hash.

## 4. Validation plan (executed Phase 1D+, defined here)

Missing cells (any included ERA5 cell without population → quarantine + report) · aggregation check (cell populations sum vs published footprint total within tolerance) · weight sum = 1 (exact assert).

## 5. Limitations

Static 2020 vintage ignores migration/load shifts; population imperfectly proxies commercial/industrial load; gridded products smooth dense-urban peaks (both sources share this bias direction — reported, not corrected).
