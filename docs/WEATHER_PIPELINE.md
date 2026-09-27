# Weather Pipeline — Architecture (Phase 1C skeleton)

**Status:** skeleton only. No acquisition, no validation runs, no v2 build. Authority: `docs/PHASE1B_WEATHER_DESIGN.md`, `docs/WEATHER_CARD.md`, `configs/weather.yaml`.

## 1. Architecture diagram (text)

```
Raw weather
  (data/raw/weather/era5/ + data/raw/weather/noaa/
   + request JSONs + station inventory)
      |
      v
Weather loader
  (WeatherProcessor.load_era5 / load_noaa)
      |
      v
Validation
  (WeatherValidator.* -> results/q1/weather_validation_report.json;
   fail-gate blocks downstream use)
      |
      v
Aggregation
  (aggregate_population_weighted primary
   + aggregate_simple_mean sensitivity)
      |
      v
Feature generation
  (calculate_heat_index / CDH / HDH + 24h means + zenith)
      |
      v
PJM merge
  (merge_with_pjm: UTC hourly backward merge-asof, 1h tolerance
   + weather_coverage.json join-loss accounting)
      |
      v
v2 dataset
  (data/processed/v2/pjm_weather_2020_2025.csv + hash in data/checksums/)
```

## 2. Stage contracts

| Stage | Input | Output | Gate |
|---|---|---|---|
| Loader | raw pulls + `configs/weather.yaml` | long frames | files listed in config exist |
| Validation | long/footprint frames | `weather_validation_report.json` | all §7 gates pass for used variables |
| Aggregation | validated long frames | footprint hourly series | coverage accounted |
| Features | footprint series | + heat index/CDH/HDH/zenith | validity regimes gated (heat index) |
| Merge | footprint + PJM timestamps | merged v2 columns | backward-only, 1h tolerance, coverage JSON |
| Save | merged frame | `pjm_weather_2020_2025.csv` + sha256 | hash recorded before any EXP references it |

## 3. Leakage notes

Horizon weather is forecast-time-gated (PROTOCOL availability tiers + WEATHER_CARD §7): contemporaneous series ship with a stated nowcast assumption; the mandatory 24h-lagged sensitivity reruns in Phase 5+. Forward-fill across origins is prohibited.
