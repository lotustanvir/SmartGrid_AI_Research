# WEATHER_CARD — Planned Weather Integration for PJM Load Forecasting (Phase 1B Design)

**Status:** DESIGN ONLY — no weather data acquired, no files created, no pipeline changed.
**Authority:** `docs/PROTOCOL.md` (task/leakage), `docs/STATS_PLAN.md` (ablation C vs D), `docs/DATA_CARD.md §7` (Phase 1A: zero weather files on disk), `docs/PHASE1B_WEATHER_DESIGN.md` (full strategy).

---

## 1. Weather motivation

Hour-ahead PJM load in v1 is modeled without any atmospheric state, yet heating and cooling drive the largest load excursions in the dataset (summer peaks to 320,308 MW; winter evening ramps; the 2025 record-peak year). The v1 ablation finding — adding RTO wind/solar *degrades* RMSE — is expected: at 2–3% penetration renewables carry almost no load information, while temperature carries the dominant exogenous signal and is entirely absent. Without temperature, humidity, and derived thermal-stress features, the benchmark cannot test H3 (weather necessity), cannot explain peak-slice errors, and cannot pass review at Applied Energy / IEEE TSG, where temperature-inclusive modeling is a baseline expectation. Weather integration is therefore the highest-ROI upgrade in the Q1 roadmap: it converts the strongest known omitted variable into the controlled Ablation-C treatment.

## 2. Data sources

| Source | Role | Spatial | Temporal | Access | Reproducibility |
|---|---|---|---|---|---|
| ERA5 reanalysis (ECMWF) | **Primary** | ~0.25° gridded, global, full PJM footprint coverable every hour | Hourly, 1940–present + ~5-day latency | CDS API, scripted pull, versioned requests | High: fixed product version + request JSON archived → byte-reproducible |
| NOAA ISD station observations | **Secondary / validation** | Point stations (e.g. PHL, PIT, EWR, CHI, WAS, CMH) | Hourly/sub-hourly, station-dependent gaps | Public archive, scripted pull | Medium: station dropouts and instrument changes documented per station |

Decision: primary ERA5 (complete, gap-free, gridded, reproducible footprint average); secondary NOAA (independent observational cross-check; validates that reanalysis-based gains are not a reanalysis artifact). Full comparison in `PHASE1B_WEATHER_DESIGN.md §2`.

## 3. Variables

**Mandatory observed:** 2m temperature (°C), relative humidity (%, with dew point 2m °C as cross-check pair), 10m wind speed (m/s), total cloud cover (0–1).
**Derived (deterministic from observed + timestamp):** heat index (Rothfusz, summer regime), cooling-degree-hours CDH = max(T−22,0), heating-degree-hours HDH = max(18−T,0), plus 24h rolling means of CDH/HDH, and solar-zenith clear-sky proxy (timestamp + ~40°N latitude, no weather input needed).
**Why each matters:** temperature sets heating/cooling demand (the load–temperature U-curve); humidity modulates cooling via heat stress; dew point constrains humidity QA and heat-index validity; wind speed drives wind-chill/heating and correlates with wind-generation error regimes; cloud cover separates solar-driven net-load variance from thermal load; CDH/HDH linearize the U-curve for tree/linear models (typically beating raw temperature in ablations); heat index isolates dangerous humid-heat peaks operators care about most.
Per-variable table with resolution, reason, and expected impact: `PHASE1B_WEATHER_DESIGN.md §1`.

## 4. Spatial aggregation

**Recommended: Option 2 — population-weighted footprint mean** (primary), with Option 1 (simple footprint mean) retained as a sensitivity column.
Rationale: PJM load concentrates in eastern urban corridors; a simple geographic mean overweights low-load rural grid cells, while state-level aggregation (Option 3) multiplies join complexity 10× for marginal gain at RTO-total scale. Population weighting uses fixed census weights (documented, versioned), so the series is deterministic and reproducible. Disadvantages: weights are static (ignore migration/load shifts 2020–2025 — sensitivity: compare against simple mean; divergence logged). Full trade-off: `PHASE1B_WEATHER_DESIGN.md §3`.

## 5. Temporal alignment

- **Standard:** UTC, hourly, left-labeled hour intervals matching PJM `timestamp` semantics (`YYYY-MM-DD HH:MM:SS`, hourly, verified Phase 1A).
- **Frequency:** 1h; ERA5 natively hourly (no resampling); NOAA sub-hourly aggregated to hour by mean (documented per station).
- **Daylight saving:** no DST handling required — all joins in UTC; DST enters only as a calendar flag feature (separate concern).
- **Merge:** `merge-asof` (backward, tolerance 1h) of footprint series onto PJM timestamps; every unmatched hour counted in `weather_coverage.json` (join-loss accounting replaces v1 silent inner-join semantics).
- **Missing hours:** ERA5 expected zero gaps (reanalysis); NOAA gaps tolerated and reported, never silently filled beyond the v1 cleaning contract (exogenous `time`-interpolation limit 6, inside only).

## 6. Quality control

Per-variable checks before any feature enters v2: missing percentage (gate: ERA5 0%; NOAA reported per station), impossible values (T −40…45 °C, RH 0–100%, wind ≥ 0, cloud 0–1), timestamp-gap scan, spatial-coverage log (grid cells/stations contributing per hour), outlier flags (|z| > 5 vs rolling climatology → quarantine, not deletion). Output: `results/q1/weather_validation_report.json` (schema defined Phase 1B; populated Phase 1 execution). Plan detail: `PHASE1B_WEATHER_DESIGN.md §7`.

## 7. Leakage prevention

Classification enforced in code (Phase 2+) and audited per run (`verification.json`):

| Class | Variables | Horizon use | Rule |
|---|---|---|---|
| Known future | calendar, zenith, holiday/season | may extend into horizon | deterministic from timestamp |
| Forecast-time available | weather (T/RH/wind/cloud + CDH/HDH/heat-index) | horizon values must be *forecast-time-known* | v2 baseline: contemporaneous values only with nowcast assumption stated; sensitivity rerun with 24h-lagged weather (strictly historical) |
| Historical only | lags, rolling stats, lagged weather sensitivity | past only | `shift()`-safe construction as v1 |

The nowcast-vs-lagged sensitivity is the honesty mechanism: if gains survive lagged weather, they are operationally real; if not, the paper reports nowcast-conditional gains with the NWP-coupling limitation owned.

## 8. Limitations (must appear in manuscript)

Reanalysis ≠ operational forecast (ERA5 latency ~5 days; deployment needs NWP coupling — listed future work); single footprint series masks zonal thermal gradients (zonal extension future work); static population weights; NOAA validation limited by station gaps; cloud-cover skill lower than temperature skill (reported per-variable, not averaged away).

## 9. Implementation Status

**Phase 1C: pipeline skeleton created.**

- `configs/weather.yaml` — acquisition/aggregation/validation spec (not executed).
- `src/data/weather.py` — `WeatherProcessor` stub (8 documented methods, all `NotImplementedError`).
- `src/data/weather_validation.py` — `WeatherValidator` stub (6 gate methods + report writer).
- `tests/data_tests/test_weather.py` — 4 placeholder tests (skipped, marked TODO Phase 1D).
- `docs/WEATHER_PIPELINE.md` — architecture diagram + stage contracts.

No external weather data acquired. No model impact evaluated yet. Existing datasets, pipeline, and results untouched (verified via `git status`, Phase 1C report).
