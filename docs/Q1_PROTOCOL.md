# Q1 Frozen Protocol — Phase 2.1 (`Q1_v2_LAG1_001` / features `Q1_V2_LAG1_001`)

Phase 2.1 blocker fixes applied (no Phase 3 execution; no headline
results; no model ranking). Locked values unchanged: V2 dataset + SHA,
49-feature schema + hash, 168h history, H1/H6/H24, seeds {42,123,2025},
5-fold OOF, calendar splits, common origins + hash, lagged-weather-only,
MAE/RMSE/MAPE/sMAPE/R²/MASE, 11-model set.

## 1. Dataset
Canonical: `dataset/processed/pjm_smart_grid_2020_2025_v2.csv` (V2, 14 cols).
V1 is NOT the Q1 dataset. SHA256 recorded per-run via `src/data/provenance.py`
(no invented hashes).

## 2. Splits (calendar, no ratio)
* TRAIN: `timestamp < 2024-01-01`
* VAL: `2024-01-01 <= timestamp < 2025-01-01`
* TEST: `2025-01-01 <= timestamp < 2026-01-01`
* Implementation: `src/data/splits_q1.py:split_calendar` (ordering asserts,
  duplicate-timestamp fail-closed, hourly-continuity warnings).
* 70/15/15 (`src/training/splits.py`, `configs/experiment.yaml:splits`)
  is legacy synthetic-only, NOT Q1.

## 3. History / horizons / seeds
* History 168h; horizons H ∈ {1,6,24}; seeds {42,123,2025} (stochastic only).
* Persistence / Seasonal-168 / SARIMA-lite are deterministic
  (`random_state=None` enforced; fake variation forbidden).
* Seasonal period locked 168 (`src/q1/protocol.py:SEASONAL_PERIOD`).
  SARIMA-lite enforces s=168 fail-closed (legacy s=24 raises);
  `configs/models.yaml:sarima` uses `seasonal_order [1,1,1,168]`.
* Naive multi-origin semantics (locked): single-tail `predict` is
  single-origin only. Q1 common-origin scoring MUST use origin-aligned
  `predict_at_origins` / `src/q1/evaluate.py:build_naive_preds_by_origin`:
  persistence `y_hat[t+h]=y[t]`; seasonal `y_hat[t+h]=y[t+h-168]`
  (strictly historical; `src/models/statistical/persistence.py`).

## 4. Weather information rule (LAGGED ONLY)
For origin t (last observed timestamp), features at row t use weather
with shift ≥ 1 only:
`temperature/humidity/wind_speed/cloud_cover/solar_radiation/wind_generation_mw/solar_generation_mw/total_renewable/cdh/hdh`
→ `<stem>_lag_{1,24,168}` (30 features, `src/data/features_q1.py`).
Raw contemporaneous weather columns are NEVER model inputs.
Sequence encoders use observed weather at steps s ≤ t; decoder future
weather t+1..t+H is NEVER supplied (TFT knowns = calendar only,
`src/models/tft/dataset.py:resolve_roles`, `src/q1/tft_path.py`).
No forward-fill/interpolation across the origin/horizon boundary;
cleaning interpolation (time, limit 6) applies to past gaps only.

Ambiguity log: ERA5 `solar_radiation`/downward-shortwave is taken as
observed-at-station proxy and lagged identically; no forecast-weather
product is introduced in Phase 2. If a forecast-weather source is added
later, this section must be amended (no silent rule invention).

## 5. Full V2 feature policy + target-derived analysis
* Included (lagged): load lags 1/24/168, rolling mean/std 24/168
  (shift(1)-first, strictly backward), 10 weather/renewable vars × 3 lags,
  calendar hour/dow/dom/month/woy/weekend + 6 cyclical encodings.
* `total_renewable = wind+solar` (no load) → included lagged.
* `cdh/hdh` (temperature-derived, no load) → included lagged.
* `net_load = load − total_renewable` CONTAINS target → EXCLUDED entirely
  (even lagged; redundant with load lags, risks silent leakage).
* `renewable_penetration = total/load` CONTAINS target → EXCLUDED entirely.
* One schema: `src/data/features_q1.py:canonical_feature_names()` (ordered)
  + version + `feature_schema_hash()`. No hardcoded forks allowed
  (legacy `run_phase_6b.get_model_features`, `run_fresh_*`,
  `run_tft_only` are superseded for Q1; `src/data/adapters.py:tft_ready`
  is quarantined to calendar-only knowns via the canonical role resolver
  — weather is encoder-only unknown, never decoder-known; canonical Q1
  construction is `src/q1/tft_path.py:build_q1_tft_frame`).

## 6. Origins
`src/data/origins.py:generate_origins` (168h history, horizon-in-test,
non-NaN actuals, TEST-period only, deterministic) →
actual artifact path returned by `save_origins`
(`experiments/protocol/test_origins.parquet` preferred, `.csv` fallback
when no parquet engine; registry records the exact written path).
`origins_hash()` normalizes string/datetime timestamps (locked hash
preserved). Same set to every model; `assert_same_origins` +
`src/q1/evaluate.py` fail closed on mismatch/substitution.

## 7. Models
11 Q1 (`src/training/experiment_runner.py:MODEL_REGISTRY`,
headline guard `q1_headline_models` / `run_q1_headline`):
persistence, seasonal_persistence, sarima (SARIMA-lite — transparent NumPy
least-squares approximation, NOT conventional SARIMA; statsmodels absent;
seasonal period locked 168, `src/models/statistical/sarima.py`; user-facing
label `SARIMA-lite`, internal key `sarima` retained), xgboost,
lightgbm, lstm, gru, tft, patchtst, nbeats, hybrid.
Legacy linear_regression/random_forest retained, quarantined: passing them
to `q1_headline_models` / `run_q1_headline` raises; `run_all` default
(all 13) is LEGACY/synthetic-only verification, never headline.
* Trees: direct per-horizon estimators (H1/H6/H24), `src/q1/horizons.py`
  (`build_direct_pairs`), never relabelled one-step.
* LSTM/GRU/PatchTST/N-BEATS: seq_len=168, output_size=H.
* TFT: encoder 168, pred ∈ {1,6,24} per run, single path
  (`src/q1/tft_path.py`), best-val checkpoint restored, never test-fit.
* Hybrid: TFT → chronological 5-fold OOF → residual → XGB
  (`n_oof_folds=5` code+yaml), test never in OOF, pretrained TFT test-free.

## 8. Metrics / stats
Fractions for MAPE/sMAPE (0.05==5%). MASE =
MAE/seasonal-naive-TRAIN-MAE with seasonality exactly 168 (weekly);
the denominator uses TRAINING targets only (2020–2023 boundary) —
validation/test targets MUST NOT enter. `src/q1/evaluate.py:
score_at_origins` requires `y_train` and rejects `seasonality != 168`
fail-closed; constant/short-train denominators raise
(`src/evaluation/metrics.py:mase`, Q1 default 168).
Stats on common-origin paired errors only: Wilcoxon, Friedman, Holm,
bootstrap CI (`src/evaluation/stats.py`); Diebold-Mariano raises pending
justification. Legacy `statistical_tests_v2.json` quarantined
(`results/research_analysis/UNVERIFIED_LEGACY.md`), never overwritten.

## 9. Reproducibility
Machine-generated `experiments/REGISTRY.csv` (`src/utils/registry.py`):
EXP-ID/status/git_commit/dataset-SHA/feature-hash/origin-hash/model/
horizon/seed/config-snapshot/versions/CUDA/checkpoint+hash/prediction/
metric/runtime/smoke/notes; statuses
(PLANNED/RUNNING/COMPLETED/FAILED/SUPERSEDED). Non-smoke COMPLETED Q1
records must satisfy `require_q1_complete` (fail-closed; checkpoint
required except deterministic no-checkpoint models
persistence/seasonal_persistence/sarima). Phase 2 records
SMOKE (`smoke=True`) only — excluded from headlines.
Origin artifacts: `save_origins` returns the ACTUAL written path
(parquet preferred, CSV fallback without new dependencies); the registry
records that exact path (`run_q1_protocol.py`); CSV fallback is expected
in the verified env.

## 10. Retired paths
`run_phase_6b.py` test-inclusive TFT/Hybrid path raises fail-closed
(`allow_legacy_unsafe` required) and exits on direct execution.
Canonical runner: `run_q1_protocol.py` (SMOKE/NON-RESULT only in Phase 2).
