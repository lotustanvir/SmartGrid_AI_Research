# PJM Weather Footprint — Methodology Lock (Phase 1D-A Design)

**Status:** DESIGN AND CONFIGURATION ONLY. No ERA5 download. No population download. No datasets modified. No features built. No models trained.
**Authority:** `docs/PHASE1B_WEATHER_DESIGN.md §3`, `docs/WEATHER_CARD.md §4`, `configs/pjm_weather_weights.yaml` (this lock's machine record).

---

## 1. Motivation

The v1 benchmark models RTO-total load without any atmospheric state, while the strongest load excursions in 2020–2025 are thermal events. A footprint weather series must therefore answer one question reproducibly: *what single hourly weather vector best represents the atmospheric conditions driving PJM-wide electricity demand?* This document locks the spatial methodology before any acquisition so the resulting series is a pre-registered construct, not a post-hoc fit.

## 2. PJM geographic representation

- **Formal definition:** the *PJM weather footprint* is the set of ERA5 native-grid cells whose centroids fall inside the PJM transmission-zone boundary polygon (boundary dataset + version/hash fixed in `configs/pjm_weather_weights.yaml` before acquisition; until then a TODO placeholder).
- **Geographic boundary source:** PJM transmission-zone boundary dataset (versioned file + sha256 recorded at acquisition; requirement, not yet selected — any published PJM service-territory polygon with documented vintage is acceptable).
- **Coordinate reference system:** EPSG:4326 (WGS 84, decimal degrees) for boundary, population, and ERA5 cell geometries alike — no reprojection ambiguity.
- **Spatial resolution:** ERA5 native ~0.25° × 0.25° single-levels grid; population product aggregated up to these cells (never downscaled below its native resolution).
- **Inclusion criteria:** centroid-in-polygon; coastal/edge partial cells included by centroid rule (documented, deterministic); no manual cell additions/removals.
- **Why it represents demand:** PJM load is the sum over the same service territory the polygon bounds; the footprint therefore covers exactly the population whose heating/cooling behavior generates the target, unlike single-city proxies (e.g. Philadelphia-only temperature) that overweight one microclimate.

## 3. Weighting comparison

**Method A — Simple grid-cell average.** Mean over included cells, equal weights.
Advantages: trivial, fully reproducible, zero auxiliary data. Disadvantages: rural cells (low load) count equally with urban corridors (high load), diluting the heat signal where cooling demand concentrates. Data requirement: boundary only. Reproducibility: maximal. Q1 suitability: acceptable only as a sensitivity baseline.

**Method B — Population-weighted average (SELECTED, primary).** Cell weights proportional to resident population.
Advantages: tracks where load lives while preserving the single-series protocol; deterministic given fixed weights; directly answers the "represents demand" requirement. Disadvantages: static weights ignore 2020–2025 migration and load-composition shifts; population ≠ load (industrial pockets underweighted). Data requirement: boundary + gridded population product (one vintage). Reproducibility: high once source/year/version are locked. Q1 suitability: strong — standard practice in energy-weather aggregation, defensible in one paragraph.

**Method C — Load-weighted average.** Cell weights proportional to metered zonal load.
Advantages: closest to true demand geography. Disadvantages: requires zonal load histories that are themselves modeled outputs in places; weights become time-varying and study-specific, harming reproducibility and inviting circularity (target-derived weights explaining the target). Data requirement: zonal load time series + allocation model. Reproducibility: low. Q1 suitability: weak as primary (circularity objection); useful only as exploratory follow-up.

## 4. Selected methodology

**Primary: Method B (population-weighted footprint mean).** It is the only option that simultaneously represents demand geography, stays reproducible from public inputs, and avoids target circularity.
**Sensitivity experiment: Method A (simple geographic mean).** Computed on the same cells and reported side-by-side in Ablation-C; divergence between A and B quantifies the weighting assumption instead of hiding it. Method C is explicitly out of scope (documented here so reviewers see the consideration, not an oversight).

## 5. Mathematical formulation

Footprint weather at hour t:

```
Weather_PJM(t) = Σ_i ( w_i × weather_i(t) )
```

with weights:

```
w_i = population_i / Σ_j population_j
```

over included ERA5 cells i, where `population_i` is the reference-vintage population intersecting cell i, and `weather_i(t)` is the cell's hourly ERA5 value. Sensitivity series uses `w_i = 1/N`. Weights sum to 1 by construction (asserted in code; violation fails the build).

## 6. Reproducibility requirements (all fixed before acquisition)

- Boundary: source name + vintage + file sha256 in `configs/pjm_weather_weights.yaml`.
- Population: product + version + spatial unit + reference year + CRS + file hash (single census vintage across the full 2020–2025 window; vintage change prohibited mid-study).
- ERA5: product (`reanalysis-era5-single-levels`), archived CDS request JSONs, pull timestamps.
- Outputs: `data/processed/v2/pjm_footprint_weights.csv` (cell_id, centroid, population, w_pop, w_simple) + `data/processed/v2/pjm_footprint_coverage.json` (cell counts, weight-concentration stats, e.g. top-decile share).
- Every EXP row referencing weather cites the weights-file hash (registry rule, `docs/EXPERIMENT_GUIDE.md`).

## 7. Limitations (carried into the manuscript)

Static population vintage; centroid rule edge effects on coastal/partial cells; single footprint series masks intra-PJM thermal gradients (zonal extension is future work); population imperfectly proxies commercial/industrial load geography (Method C trade-off owned in §3).
