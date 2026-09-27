# Phase 1B Weather Design — Q1-Ready Weather Integration Framework (DESIGN ONLY)

**Project:** SmartGrid_AI_Research — Trustworthy Short-Term Electricity Load Forecasting using PJM 2020–2025 data
**Phase:** 1B — acquisition and integration design. No model training. No dataset modification. No downloads. No pipeline changes.
**Grounding:** `docs/DATA_CARD.md §7` (Phase 1A: 31 files scanned, zero weather files on disk, PJM timestamps UTC hourly); `docs/PROTOCOL.md §§2–7` (168h→1/6/24h task, feature-availability tiers, leakage checks); `configs/pjm_data.yaml` (weather fields `null` — the gap being closed).

---

## 1. Weather requirement analysis

### 1.1 Mandatory variable table

| Variable | Source | Resolution | Reason for inclusion | Expected impact on load forecasting |
|---|---|---|---|---|
| 2m temperature (°C) | ERA5 primary / NOAA secondary | Hourly footprint series (see §3) | Primary thermal driver of demand; load–temperature U-curve (heating below ~18 °C, cooling above ~22 °C) explains summer/winter peaks | Largest exogenous ablation gain (H3); concentrates on heat/cold/peak slices where v1 errors peak |
| Relative humidity (%) | ERA5 / NOAA | Hourly | Modulates cooling demand via heat stress; temperature–humidity interaction lifts humid-heat peaks beyond dry-heat equivalents | Moderate gain, mostly summer peaks; interaction term with temperature |
| Dew point 2m (°C) | ERA5 / NOAA | Hourly | Thermodynamic cross-check on humidity (RH derived from T + dew point); constrains heat-index validity; independent QA pair | Small direct gain; large QA value (humidity-sensor failure detection) |
| Wind speed 10m (m/s) | ERA5 / NOAA | Hourly | Wind-chill heating demand; correlates with wind-generation error regimes relevant to net-load framing | Small on load target; larger on net-load target and winter ramps |
| Total cloud cover (0–1) | ERA5 (NOAA present-weather as proxy) | Hourly | Separates solar-driven net-load variance from thermal load; cloud-front passages mark intraday solar ramps | Small on load; material on net-load/solar-ramp slices |

### 1.2 Derived variables (deterministic; no new data needed)

| Variable | Derivation | Reason | Expected impact |
|---|---|---|---|
| Heat index | Rothfusz regression on T + RH (valid regime gated by dew-point check) | Captures dangerous humid-heat peaks that drive annual maxima and operator stress events | Peak-slice gain; headline case-study variable (one heat-week figure) |
| Cooling Degree Hours (CDH) | max(T−22, 0) per hour + 24h rolling mean | Linearizes the cooling arm of the U-curve for linear/tree models that cannot learn thresholds efficiently | Typically beats raw temperature in tree-model ablations; core Ablation-C column |
| Heating Degree Hours (HDH) | max(18−T, 0) per hour + 24h rolling mean | Linearizes the heating arm; captures cold-wave demand level and duration | Winter/cold-slice gain; ramp-error reduction on cold fronts |

### 1.3 Why each variable matters for electricity demand (one paragraph)

Electricity demand follows a U-curve in temperature: mild weather (~18–22 °C) is the minimum; cooling load rises with heat and humidity (air conditioning, dehumidification, heat-stressed grids) while heating load rises with cold and wind (electric heating, longer heating hours). Humidity shifts the cooling arm upward (same temperature, more cooling work); dew point pins down that shift and validates the humidity record. Wind speed matters at both ends (wind-chill heating, and as a marker for wind-generation regimes once the target moves to net load). Cloud cover governs behind-the-meter and utility solar output, hence net load rather than gross load — small for today's RTO-total load target, decisive for the net-load framing as solar grows 6.5× across the window. Degree-hours convert the nonlinear U-curve into linear features that every model family, including the currently-winning linear baseline, can exploit — which is exactly why they are the fairest test of H3.

## 2. Weather source design

### 2.1 Comparison

| Dimension | Source A: ERA5 reanalysis | Source B: NOAA observations |
|---|---|---|
| Spatial coverage | Gridded ~0.25°, global; any PJM footprint polygon fully coverable every hour | Point stations; coverage = station network density, sparse rurally |
| Temporal resolution | Native hourly, 1940–present | Hourly/sub-hourly, station-dependent; gaps from outages/instrument changes |
| Availability | CDS API scripted pull; ~5-day latency (reanalysis, not real-time) | Public archive; near-real-time for reporting stations |
| Reproducibility | High: fixed product + archived request JSON → byte-reproducible series | Medium: station dropouts/versioning must be logged per station |
| PJM suitability | Complete gap-free footprint series; ideal for controlled ablations | Independent ground truth; ideal for validating reanalysis-based findings |

### 2.2 Recommendation

- **Primary source: ERA5 reanalysis.** Complete, gap-free, gridded, versioned, reproducible — the properties a controlled Ablation-C treatment requires. Every hour of 2020–2025 is coverable with documented provenance.
- **Secondary validation source: NOAA station observations.** Independent cross-check that gains are physical, not reanalysis artifacts: re-run key Ablation-C comparisons on NOAA-derived footprint series and report agreement/divergence.

**Decision rationale:** the paper's weather claim must survive the question "is this just ERA5-shaped?" Dual sourcing answers it by construction. ERA5-first keeps the engineering tractable (one gridded pipeline); NOAA-second bounds the validity threat without doubling the pipeline cost, since NOAA series reuse the same footprint-aggregation and alignment code paths.

## 3. PJM weather mapping strategy

PJM spans 13 states plus DC (Mid-Atlantic through Chicago); load concentrates in eastern urban corridors.

| Option | Design | Advantages | Disadvantages | Complexity |
|---|---|---|---|---|
| 1. Single PJM mean | Unweighted mean over footprint grid cells | Simplest; fully reproducible; one series | Overweights low-load rural cells; dilutes urban heat signal | Low |
| **2. Population-weighted mean (RECOMMENDED)** | Fixed census-weight mean over footprint cells | Tracks where load lives; preserves single-series protocol; deterministic | Static weights ignore 2020–2025 migration/load shifts | Low–medium |
| 3. State-level aggregation | Per-state series, multi-group modeling | Captures thermal gradients; enables zonal future work | 10× join/model complexity for RTO-total target; unjustified at current scope | High |

**Recommendation: Option 2 primary, Option 1 retained as a sensitivity column.** The sensitivity comparison (population-weighted vs simple mean) is cheap — one extra column, one extra ablation row — and pre-answers the "weights are arbitrary" reviewer objection. State-level aggregation is deferred to zonal future work and stated as such.

## 4. Data structure design

```
data/raw/weather/
  era5/          # raw ERA5 pulls (per-variable NetCDF/CSV as pulled) + request JSONs
  noaa/          # raw station pulls + station inventory (locations, coverage windows)
data/processed/v2/
  pjm_weather_2020_2025.csv   # footprint series (see columns below)
```

Expected columns of `pjm_weather_2020_2025.csv`:

```
timestamp,                  # UTC hourly, matches PJM timestamp semantics (Phase 1A)
temperature_2m,             # °C, population-weighted footprint mean (ERA5)
relative_humidity,          # %, footprint mean
dew_point,                  # 2m °C, footprint mean (humidity QA pair)
wind_speed_10m,             # m/s, footprint mean
cloud_cover,                # 0–1 total cloud cover, footprint mean
temperature_2m_simplemean,  # °C, Option-1 sensitivity column
heat_index,                 # Rothfusz, gated regime (NaN outside validity → documented)
CDH,                        # max(T-22,0)
HDH,                        # max(18-T,0)
CDH_24h, HDH_24h,           # 24h rolling means (shift-safe: past only)
noaa_temperature_2m,        # secondary-source cross-check column (where available)
coverage_flag               # contributing cells/stations count per hour
```

## 5. Timestamp alignment rule

- **Timezone standard:** UTC everywhere. PJM `timestamp` verified UTC hourly (Phase 1A); ERA5 natively UTC; NOAA station times converted to UTC at ingest with conversion logged.
- **Frequency:** 1h, left-labeled hour intervals matching PJM semantics; ERA5 natively hourly (no resampling); NOAA sub-hourly → hourly mean with per-station method log.
- **Daylight saving:** no DST shifting — all joins in UTC. DST enters only as a separate calendar flag feature (out of weather scope).
- **Missing hours:** ERA5 expected zero gaps (verified by gap scan; any gap quarantined and reported). NOAA gaps tolerated, counted per station, never silently filled beyond the v1 cleaning contract (exogenous `time`-interpolation limit 6, inside-only).
- **Leakage-safe merge:** `merge-asof` (backward direction, 1h tolerance) of footprint weather onto PJM timestamps; unmatched hours counted in `weather_coverage.json`. This replaces v1 silent inner-join semantics with accounted join loss. Forward-fill across the forecast boundary is prohibited; horizon weather handling follows §6.

## 6. Weather feature availability rule and leakage prevention table

| Class | Variables | Horizon treatment | Enforcement |
|---|---|---|---|
| Known future (deterministic) | hour, dow, holiday, season, solar-zenith | may extend into 1/6/24h horizons | derived from timestamp only; audit: no data input |
| Forecast-time available | T, RH, dew point, wind, cloud, heat index, CDH/HDH (+24h means) | v2 baseline: contemporaneous series with **stated nowcast assumption**; mandatory sensitivity: 24h-lagged (strictly historical) rerun of key Ablation-C rows | merge-asof backward-only; horizon values gated by availability flag; lagged-sensitivity reported side-by-side |
| Historical only | lags, rolling stats (load and weather-derived) | past-only construction | `shift()`-first discipline as v1 (`src/data/features.py`); test: feature timestamp < origin |

**Honesty mechanism:** if Ablation-C gains survive 24h-lagged weather, they are operationally real; if they attenuate, the paper reports nowcast-conditional gains and owns NWP coupling as future work. Either outcome is publishable; hiding the distinction is not.

## 7. Weather quality control plan and validation report

Gates executed before any weather column enters v2 (Phase 1 execution; schema defined here):

1. **Missing percentage:** ERA5 gate 0% (any gap → quarantine + report); NOAA reported per station and in aggregate.
2. **Impossible values:** T in −40…45 °C; RH 0–100%; dew point ≤ temperature; wind ≥ 0; cloud 0–1. Violations quarantined with counts.
3. **Timestamp gaps:** full-range continuity scan vs expected hourly grid 2020-01-01 05:00 UTC → 2026-01-01 04:00 UTC; every gap listed.
4. **Spatial coverage:** contributing grid cells (ERA5) / stations (NOAA) per hour; dropouts timestamped.
5. **Outliers:** |z| > 5 vs 30-day rolling climatology flagged for review (heat/cold waves must NOT be auto-deleted — cross-checked against event definitions).

Output artifact (populated in Phase 1, schema frozen here):

```
results/q1/weather_validation_report.json
{ "source": "era5|noaa", "period": {...}, "missing_pct": {...},
  "impossible_counts": {...}, "gaps": [...], "coverage": {...},
  "outlier_flags": [...], "gate": "pass|fail", "quarantined": [...] }
```

Fail-gate blocks v2 build for the affected variable; quarantine lists are append-only provenance.

## 8. Implementation plan (file responsibilities — NO code in Phase 1B)

| File (to be created Phase 1–2) | Responsibility |
|---|---|
| `src/data/weather.py` | ERA5/NOAA acquisition + footprint aggregation (population-weighted + simple-mean) + derivation (heat index, CDH/HDH, zenith) + merge-asof onto PJM timestamps + `weather_coverage.json` |
| `src/data/weather_validation.py` | §7 gates (missing/impossible/gap/coverage/outlier) + `weather_validation_report.json`; fail-gate enforcement before v2 build |
| `configs/weather.yaml` | Source endpoints, request JSONs, footprint polygon/weights version, variable list, gates thresholds — the reproducible acquisition record |
| `docs/WEATHER_CARD.md` | Variable/source/aggregation/alignment/QA/leakage/limitation reference (created Phase 1B, alongside this design) |

Build order: `configs/weather.yaml` (spec) → raw pulls → `weather_validation.py` gates → `weather.py` aggregation/merge → v2 columns → Ablation-C rows (Phase 5+).

## 9. Limitations carried into the manuscript

Reanalysis latency (~5 days) means deployment requires NWP coupling (future work); single footprint series masks zonal gradients (zonal extension future); static population weights; NOAA validation bounded by station gaps; cloud-cover skill below temperature skill (reported per-variable). The lagged-weather sensitivity (§6) quantifies the nowcast assumption instead of hiding it.
