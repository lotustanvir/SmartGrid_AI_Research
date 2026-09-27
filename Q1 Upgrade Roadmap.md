# Q1 Upgrade Roadmap — From Current PJM Benchmark to Publication-Quality Manuscript

**Project:** Short-Term Electricity Load Forecasting using PJM 2020–2025 with ML / DL / Transformer / Hybrid Models
**Target venues:** Applied Energy (IF ~11), IEEE Transactions on Smart Grid (~9–10), Energy (~9), IJEPES (~5–6)
**Audit basis:** Full repository inspection — `src/`, `configs/`, `dataset/processed/pjm_smart_grid_2020_2025.csv` (52,608 hourly rows), `results/`, `run_phase_6b.py`, `src/models/`, `src/training/`, `src/evaluation/`
**Principle:** DO NOT claim "our deep learning model is superior" unless proven. The current data proves the opposite.
**Verdict on current version:** Strong engineering pipeline (leakage-safe), irreproducible and incomparable results — **desk-reject at any Q1 journal today.**

---

## TABLE OF CONTENTS

1. [TASK 1 — Current Research: Objective, Weaknesses, Why Not Q1](#task-1--understand-current-research)
2. [TASK 2 — New Q1-Level Research Story](#task-2--create-a-new-q1-level-research-story)
3. [TASK 3 — Fix Dataset for Q1 Standard](#task-3--fix-dataset-for-q1-standard)
4. [TASK 4 — Fix Forecasting Task](#task-4--fix-forecasting-task)
5. [TASK 5 — Model Selection for Q1](#task-5--model-selection-for-q1)
6. [TASK 6 — Experiment Design](#task-6--experiment-design)
7. [TASK 7 — Metrics Required for Q1](#task-7--metrics-required-for-q1)
8. [TASK 8 — Societal Impact Analysis](#task-8--societal-impact-analysis)
9. [TASK 9 — Code Implementation Plan (File-by-File)](#task-9--code-implementation-plan)
10. [TASK 10 — Final Q1 Paper Structure](#task-10--final-q1-paper-structure)
11. [Roadmap, Reviewer Q&A, Publication Strategy](#final-roadmap--publication-strategy)

---

## TASK 1 — UNDERSTAND CURRENT RESEARCH

### 1.1 Folder structure (what actually exists)

```
configs/{default,data,pjm_data,models,experiment}.yaml  # protocol + hyperparams
src/data/{loader,schema,validation,preprocessing,features,pipeline,adapters,pjm_integration,eda}.py
src/models/base.py + baseline/{linear,rf,xgb,lgbm} + deep/{sequence,lstm,gru} + tft/{dataset,ft_model} + hybrid/tft_xgb.py
src/training/{splits,trainer,experiment_runner,ablation}.py
src/evaluation/{metrics,alignment}.py
dataset/raw/pjm/{2020..2025}/{hrl_load_metered,wind_gen,solar_gen}.csv
dataset/processed/pjm_smart_grid_2020_2025.csv  # 52,608 rows, 4 cols (verified)
results/{data_validation,eda,experiments,model_validation,research_analysis,fresh_validation*,explainability}
run_phase_6b.py / run_phase_6a.py / run_fresh_validation_aligned.py / run_tft_only.py / finalize_phase_6b.py
tests/ (12 files, 179 protocol tests claimed passing)
saved_models/ (EMPTY) | results/tables/ (EMPTY) | results/figures/ (EMPTY) | notebooks/ (EMPTY)
lightning_logs/ (700+ versions — churn, not a locked run)
```

### 1.2 Dataset (verified)

- **Source:** PJM Interconnection public Data Miner. Raw: ~1.58M load rows (30 load-areas × 22 zones) + wind + solar per year, 2020–2025.
- **Built:** `pjm_smart_grid_2020_2025.csv`, 2020-01-01 05:00 UTC → 2026-01-01 04:00 UTC, hourly, 52,608 rows, 0 gaps, columns `timestamp,pjm_load_mw,wind_generation_mw,solar_generation_mw`.
- **Target:** `pjm_load_mw`, mean ~179,950 MW, range 109,874–320,308 MW.
- **Exogenous today:** RTO wind + solar only. **No temperature, no humidity, no holidays.** `configs/pjm_data.yaml: weather.temperature=null, humidity=null`.
- **Renewable penetration:** 1.99% (2020) → 3.40% (2025); load–wind corr −0.148, load–solar +0.261. Weak signal by construction.

### 1.3 Data pipeline (strength + flaw)

`loader.py → validation.py → preprocessing.py → features.py → splits.py → adapters.py`

- **Strength (genuinely good, keep):** canonical schema (`target,timestamp,group_id`), chronological `TemporalSplit` with `TRAIN<VAL<TEST` asserts, `StandardScaler` fit-train-only, lags via `groupby.shift(lag)`, rolling via `shift(1).rolling()` — no future leakage. This is the best part of the repo and survives to Q1.
- **Flaws:**
  - `pjm_integration.py` keeps `area==RTO` only, inner-joins (silently drops hours instead of reporting loss), clips 24,245 solar negatives + handful wind negatives with no sensitivity study.
  - `adapters.py:TFT_KNOWN_MAP` references `temperature` but pipeline never provides it — dead code proving weather was planned, never delivered.
  - `pipeline.py` writes `train.csv/validation.csv/test.csv/tft_frame.csv` but `run_phase_6b.py:60-77` rebuilds TFT frames *differently* (re-indexes `time_idx` after concat) — two competing TFT constructions.

### 1.4 Feature engineering

21 features: `solar,wind + hour,dow,dom,month,woy,is_weekend + 6 cyclical sin/cos + lag_1/24/168 + rolling_mean/std_24/168`. Leakage-safe. **Missing Q1-mandatory:** temperature, humidity/wet-bulb, wind-speed, holiday/event flags, season, net-load (`load−wind−solar`), solar-zenith/daylight proxy, heating/cooling-degree-hours. SHAP JSONs reference `net_load,total_renewable_mw` that do not exist in `pjm_final_metadata.json` — phantom explainability.

### 1.5 Models (8 registered in `experiment_runner.py:42`)

LinearRegression / RandomForest(200, unbounded) / XGBoost(500,d6,no-ES) / LightGBM(500,31 leaves) / LSTM(seq24,h64) / GRU(same) / TFT(enc24,hor6,hidden32,heads2) / Hybrid TFT-XGB(OOF-2folds). All behind `BaseForecaster`. TFT uses `pytorch-forecasting` quantile loss; hybrid adds XGB on OOF residuals (`tft_xgb.py` — well-designed, honest `in_sample` warning, uncalibrated-interval labeling).

### 1.6 Training scripts

`run_phase_6b.py`: trees via `runner.run_one` on explicit splits; LSTM/GRU same; TFT fit on concatenated frame then `forecast(tft_test_df)`; hybrid reuses pretrained TFT. No HPO, single seed 42, `use_early_stopping=False` for trees. `run_fresh_validation_aligned.py`, `run_tft_only.py` are fragmentary re-runs that disagree with the headline.

### 1.7 Results (why a reviewer loses trust)

Headline `pjm_2020_2025_final/detailed_results.json`: Linear 5082/6340/0.965 **best by 2×**; Hybrid 9308/12415; TFT 9504/12677; LSTM≡GRU 9601/12807 byte-identical; XGB 9798/13069; RF/LGBM R² 0.47–0.59. Four models share R²=0.87728 to 5 decimals. `final_review.json` admits **"Deep model results estimated"** and **"Pure Python tree implementations"**. `results.csv` has `sMAPE=0` for all rows, MAPE units mixed (2.717 vs 0.0278), times 0.0/30.0 placeholders. Phase 6B-A report shows LSTM/GRU R²=−64.7 ("poor convergence — not tuned"). Fresh ablation shows OOF-hybrid R²=−2.96 (worst) contradicting headline (+5.5% over TFT). TFT fresh run scores **n_test=6** vs tabular n=7,892. Wilcoxon W fractional (e.g. 6191831.92 — mathematically impossible, must be integer), p=0.0 everywhere, no correction. PICP exactly 0.9001. `saved_models/` empty → nothing reproducible.

### 1.8 Evaluation metrics today

`metrics.py:evaluate_regression` → MAE/RMSE/safe-MAPE/sMAPE/R². Missing for Q1: MASE, skill-vs-persistence, peak/ramp errors, multi-level PICP/PINAW + calibration, DM test, latency/params/FLOPs/CO₂.

### 1.9 Current research objective (reconstructed)

> "Compare 8 forecasters on PJM 2020–2025 under leakage-safe protocol and show hybrid value."

### 1.10 Current weaknesses (reviewer-grade, blunt)

1. **No manuscript, no RQs, no lit review, no gap.** There is no paper, only logs.
2. **Headline result refutes the implied claim.** Linear wins 2×. Nothing to publish about DL superiority.
3. **Incomparable evaluation.** Horizons 1 vs 6 vs 24, samples 6 vs 7,892, encoders 24 vs hand-lags 168. The ranking is meaningless.
4. **Untuned competitors.** Trees default + no early stopping; DL 24h encoder cannot see weekly cycle; TFT config vs doc mismatch (24/32/2 vs 168/64/4).
5. **Integrity flags.** Identical numbers, "estimated" admission, fractional W, sMAPE=0, empty weights/tables.
6. **Missing physics.** No temperature in a load paper = instant reject. No holidays, no net load, no events.
7. **No generalization.** Single RTO aggregate, single split, single seed, no rolling origin, no second ISO.
8. **No deployment story.** No peak/ramp/cost/carbon/latency.

### 1.11 Why current version is not Q1-ready (one paragraph for the cover letter you cannot yet write)

Q1 in 2026 (Applied Energy / IEEE TSG) requires a stated gap, fair multi-horizon protocol, weather-aware features, tuned SOTA competitors (including persistence and modern mixers/transformers), repeated-seed statistics with proper tests, calibration, ablations, explainability from fitted models, generalization evidence, and reproducible artifacts. This repo has one of those (leakage discipline) and fails the other eight, with result artifacts a reviewer will treat as credibility problems before reaching novelty.

---

## TASK 2 — CREATE A NEW Q1-LEVEL RESEARCH STORY

### 2.1 Narrative pivot (do not claim DL superiority)

The data has spoken: **hour-ahead PJM load is dominated by autocorrelation (lag_1 R²≈0.965).** Fighting that finding loses. Owning it wins. The Q1 story is **trustworthy, renewable-aware, uncertainty-aware forecasting** — when simple wins, when complexity pays, and what operators should deploy.

**Core thesis (one sentence):**

> Under strict leakage-free evaluation on 6 years of PJM data, persistence-like linear structure dominates 1-hour-ahead load, while weather context, horizon-dependent model selection, calibrated uncertainty, and residual hybridization determine value at 6–24h, peaks, and high-solar regimes.

This reframes a "failed DL paper" into a **rigor + deployment paper** — exactly what Applied Energy and IEEE TSG currently prioritize (trustworthy AI, explainability, renewables, operation).

### 2.2 Possible paper titles (5 options, pick one)

1. **Trustworthy Short-Term Load Forecasting for Smart Grids: A Leakage-Free Benchmark of Statistical, Machine Learning, and Transformer–Hybrid Models on Six Years of PJM Data** *(Recommended — matches rigor + benchmark scope; safe for Applied Energy / IEEE TSG)*
2. **When Simple Wins: Horizon-Dependent Value of Linear, Gradient-Boosted, Recurrent, and Transformer Models for Renewable-Aware Load Forecasting in PJM 2020–2025** *(Strong narrative hook; owns the linear-wins finding)*
3. **Uncertainty-Aware Short-Term Load Forecasting with Residual Hybridization: TFT–XGBoost under Chronological, Rolling-Origin, and Extreme-Event Evaluation** *(Best if hybrid + intervals become the contribution; suits IEEE TSG)*
4. **Renewable-Aware Net-Load Forecasting for High-Solar Grids: How Weather, Calendar, and Lag Structure Interact Across 1-, 6-, and 24-Hour Horizons** *(Best if weather/net-load ablation is strongest; suits Energy / Applied Energy)*
5. **From Point Accuracy to Operational Value: Peak, Ramp, Reserve, and Carbon Implications of Load-Forecasting Choices in a 180-GW System** *(Differentiator angle; add only if §8 quantification is done — high risk, high reward)*

### 2.3 Research hypothesis (testable, falsifiable)

- **H1 (autocorrelation dominance):** At 1h-ahead, lag-based linear/persistence models match or beat all nonlinear/attention models on RMSE/MAE under equal-n, leakage-free scoring.
- **H2 (horizon dependence):** Model ranking reverses with horizon: gradient-boosted and attention models gain relative skill at 6h/24h where calendar + weather dominate over lag_1.
- **H3 (weather necessity):** Adding temperature/humidity-derived features (CDH/HDH) yields larger ablation gains than adding wind/solar alone, especially on heat/cold-wave and peak subsets.
- **H4 (hybrid selectivity):** TFT–XGBoost OOF residual correction improves over TFT alone on mid-horizons and peak subsets but not uniformly; gains are residual-variance-dependent, not universal.
- **H5 (calibration over accuracy):** Best point-RMSE model ≠ best operational model; calibrated intervals (PICP≈nominal with minimal PINAW) and peak/ramp errors reorder deployment preference.

### 2.4 Research questions

- **RQ1 (fair benchmark):** Under identical 168h-in → {1,6,24}h-out tasks with common timestamp scoring, how do 11 forecasters rank on MAE/RMSE/MAPE/sMAPE/R²/MASE/skill?
- **RQ2 (features):** What is the marginal value of calendar → weather → renewable → all-features (Ablations A–E) per horizon?
- **RQ3 (horizon):** How does ranking change from 1h to 6h to 24h, and why (lag decay vs weather dominance)?
- **RQ4 (hybrid):** When does OOF residual hybridization help, by how much, and under what residual structure?
- **RQ5 (trust):** Which models are calibrated (PICP/PINAW/curves), explainable (SHAP/attention agree on lag hierarchy?), and robust on heat/cold/peak slices?
- **RQ6 (deployment):** What is the accuracy–latency–params–cost Pareto frontier, and what should a 180-GW operator actually deploy?

### 2.5 Novel contribution statements (write these verbatim in the paper)

1. **Leakage-free 6-year PJM benchmark (2020–2025, 52,608h) with weather augmentation and net-load framing** — first open protocol in this repo family to combine RTO load + RTO wind/solar + ERA5/NOAA weather + holiday/event flags with shift-safe features, train-only scaling, and chronological + rolling-origin evaluation.
2. **Fair multi-horizon evaluation (168h → 1/6/24h) with common-timestamp scoring** — fixes the horizon/sample mismatch that invalidates most in-house comparisons; all 11 models scored on identical timestamps per horizon.
3. **Horizon-dependent ranking result** — documents persistence dominance at 1h and its decay at 6/24h with statistical rigor (Friedman + Holm-corrected Wilcoxon + Diebold-Mariano), replacing "DL wins" with an honest, actionable selection rule.
4. **OOF residual-hybrid analysis with failure modes** — TFT–XGBoost evaluated per-horizon and per-regime with residual diagnostics, showing *selective* rather than universal gains (a more credible hybrid claim).
5. **Operational trust package** — multi-level calibration (50/80/90/95%), SHAP + TFT attention agreement, peak/ramp/event-slice robustness, and accuracy–latency–cost Pareto with reserve/cost/carbon translation for a 180-GW system.

**Positioning keywords for editors:** rigorous benchmarking · trustworthy AI forecasting · renewable-aware net-load forecasting · uncertainty-aware prediction · deployment-ready smart-grid AI.

---

## TASK 3 — FIX DATASET FOR Q1 STANDARD

### 3.1 Required additions

**A. Weather (mandatory — paper is unpublishable without it)**

| Variable | Why mandatory | Source & spec |
|---|---|---|
| 2-m temperature (hourly, RTO footprint mean + population-weighted) | #1 load driver (heating/cooling); explains peaks the current model misses | **ERA5 reanalysis** (ECMWF, 0.25°, hourly) spatially averaged over PJM states; fallback **NOAA ISD station network** (e.g., PHL, PIT, CHI, WAS) inverse-distance-weighted. Keep provenance log. |
| Relative humidity / dew point → wet-bulb | Heat-index effect on cooling load; humidity–temperature interaction | ERA5 or NOAA; derive heat-index / CDH/HDH (see below) |
| Wind speed (10-m) | Cooling-tower / wind-chill / correlates with wind-gen errors | ERA5 or NOAA |
| Weather condition flag (clear/cloud/precip code) | Solar-/load regimes; reviewer expects at least cloud proxy | ERA5 total-cloud-cover or NOAA present-weather; binarize |

Derive: **cooling-degree-hours CDH = max(T−22,0), heating-degree-hours HDH = max(18−T,0)** hourly + 24h rolling means; **heat-index** (Rothfusz) for summer slice. These two derived columns typically beat raw T in ablations — include them explicitly.

**B. Calendar (mandatory, cheap)**

- `is_weekend` (exists) + **`is_holiday` (US federal + PJM-observed) + `holiday_bridge` (±1 day)** via `holidays` package — mandatory: holiday load drops 10–20%, current model has no flag.
- `season` (DJF/MAM/JJA/SON one-hot or cyclical) + `month_sin/cos` (exists) + **`dst_transition` flag** (spring/fall clock shifts break lags).
- **`extreme_event` flags** (see D): `heat_wave`, `cold_wave`, `peak_day_top1pct`. Not features for training necessarily — stratification keys for §6.5/RQ5.

**C. Renewable (upgrade from decorative to analytical)**

- Keep `wind_generation_mw`, `solar_generation_mw` (RTO) but add **`total_renewable = wind+solar`, `renewable_penetration = total/load`, `net_load = load − wind − solar`** (the operational target). Net-load forecasting is the societally relevant task under solar 6.5× growth.
- Add **`solar_zenith_proxy`** (deterministic clear-sky expectation from timestamp+latitude ~40°N: `max(sin(elevation),0)`) — separates diurnal solar physics from weather noise; costs nothing, impresses reviewers.
- Keep negative-clip but log counts + run sensitivity (clip vs drop vs keep) — closes the current audit flag.

**D. Event subsets (mandatory for RQ5; build once, reuse everywhere)**

Define deterministically in `src/data/events.py` (new):

- `normal_days`: all hours excluding below.
- `heat_waves`: ≥3 consecutive days with Tmax ≥ 35°C (95°F) footprint-mean (or PJM Hot Weather Alert days if obtainable) — expect highest RAMPS.
- `cold_waves`: ≥2 days Tmin ≤ −10°C (14°F) or PJM Cold Weather Alert.
- `peak_days`: top 1% hourly loads (≈526 hours) + top-5 peak days fully. Report per-slice MAE/RMSE/peak-error separately — operators care about these 526 hours more than the other 52,082.

### 3.2 Mandatory vs optional

- **Mandatory (reject without):** temperature, CDH/HDH, humidity/heat-index, holiday flag, season, net_load + total_renewable, event-slice keys, solar-zenith proxy.
- **Highly recommended:** wind-speed, DST flag, penetration ratio, holiday-bridge.
- **Optional (Phase 2):** price/LMP, outage data, zonal disaggregation, behind-the-meter solar estimate.

### 3.3 Implementation deltas

- New `src/data/weather.py`: ERA5 (cdsapi) / NOAA (ISD download) → hourly footprint series → merge-asof on `timestamp`, report coverage/join-loss (replaces silent inner-join).
- Extend `DataMapping` (`schema.py`): `temperature_col, humidity_col, wind_speed_col, holiday_col` + `net_load` derivation flag.
- Extend `FeatureConfig`: `weather:true, holidays:true, degree_hours:true, net_load:true, zenith:true`.
- New `dataset/processed/pjm_smart_grid_2020_2025_v2.csv` + `pjm_weather_2020_2025.csv` + `events.parquet`; freeze v1 untouched for comparability.
- Update `pjm_data.yaml` → `pjm_data_v2.yaml`; weather no longer `null`.

---

## TASK 4 — FIX FORECASTING TASK

### 4.1 The current sin (state it in the paper's limitations)

Tabular models predict 1 row; LSTM/GRU `seq_len=24/output=1`; TFT `enc=24/hor=6`; protocol `horizon=24`. TFT scored on 6 trailing points, tabular on 7,892. **Any ranking from this is void.** Reviewers at IEEE TSG check exactly this.

### 4.2 New unified task (all models, no exceptions)

- **Input:** previous **168 hours** (one full week — captures daily + weekly seasonality) of `[target + all selected features available at forecast time]`.
  - Known-future (calendar, zenith, holiday): allowed to extend into horizon.
  - Unknown (lags, rolling, weather-nowcast): strictly lagged; weather in horizon uses *forecast-time-known* values only (ERA5 actuals shifted; document as nowcast assumption or use day-ahead NWP if available).
- **Output (three fixed horizons, three leaderboards):** next **1h** (dispatch), **6h** (intraday), **24h** (day-ahead unit commitment). Direct multi-output strategy for all models (no recursive rollout to avoid error accumulation confound; state this choice).
- **Mechanics per family:**
  - Tabular (Persistence→LGBM): 168h lags/rollings flattened to one row per origin → direct 1/6/24 heads (24 separate regressors or multi-output; log choice).
  - LSTM/GRU: `seq_len=168, output_size ∈ {1,6,24}` (three trained instances per architecture).
  - TFT: `encoder_length=168, prediction_length ∈ {1,6,24}`.
  - PatchTST/N-BEATS: `context=168, horizon ∈ {1,6,24}` (same).
  - Hybrid: TFT(h) + XGB-on-OOF-residuals per horizon h.

### 4.3 Unified evaluation protocol (enforce in code, not prose)

1. **Origin set:** all test-period origins with full 168h history + full horizon truth (same `test_origins.parquet` for every model/horizon).
2. **Common-timestamp scoring:** extend `EvaluationAlignment` to `(origin, step)` keys; `report()` must show `original/predictable/aligned/dropped` per model and **fail closed** if aligned counts differ.
3. **No re-splitting inside runners:** single `TemporalSplit` + rolling origins from `splits.py`; `run_phase_6b.py`-style ad-hoc TFT re-indexing deleted.
4. **Leakage re-certification:** lags/rollings shift-safe, scalers train-only, weather-horizon discipline, holiday known-flag audit — all in `verification.json` per run.
5. **Seeds:** 3 seeds (42, 123, 2025) for stochastic models; deterministic models run once + bootstrap CIs.

Remove: horizon mismatch, unequal-n scoring, per-model re-indexing, `in_sample` hybrid from any research table (debug only).

---

## TASK 5 — MODEL SELECTION FOR Q1 (11 models + why each is *required*, not decorative)

**BASELINES (reviewers reject without these — they are the paper's immune system):**

1. **Persistence (naïve, ŷ_{t+h}=y_t)** — *Required:* the irreducible "did you beat doing nothing?" bar. Current linear ≈ persistence; making it explicit turns embarrassment into finding (H1).
2. **Seasonal persistence (ŷ_{t+h}=y_{t+h−168})** — *Required:* weekly-seasonality bar; strong at 24h, exposes whether ML learns anything beyond calendar.
3. **ARIMA/SARIMA (auto-selected, e.g. SARIMA(2,1,2)(1,1,1,24) on weekly-sampled or auto_arima)** — *Required:* classical statistical SOTA; reviewers over 40 expect it; if ML cannot beat SARIMA at 24h, that is a finding, not a failure.

**MACHINE LEARNING (tabular SOTA on lag features):**

4. **XGBoost (Optuna-tuned + early stopping)** — *Required:* de-facto tabular champion in energy literature; tests whether tuned boosting beats linear (currently untested — defaults only).
5. **LightGBM (Optuna-tuned, depth-capped, min-data-in-leaf raised)** — *Required:* faster counterpart; current unbounded leaf-wise version catastrophically overfits — tuned rerun diagnoses whether failure was methodological or fundamental.

**DEEP LEARNING (sequence inductive bias):**

6. **LSTM (seq168, 2-layer, tuned hidden/dropout)** — *Required:* canonical recurrent baseline; every load-forecasting reviewer expects it.
7. **GRU (same harness)** — *Required:* lighter recurrent control; distinguishes gating-variant effects from harness effects (currently confounded by identical outputs).

**ADVANCED (modern attention/mixer — the "SOTA" checkbox):**

8. **TFT (enc168, quantile loss, variable selection + attention)** — *Required:* incumbent attention SOTA + only native-quantile + interpretable model; anchors uncertainty (Task 7) and explainability (Task 6.7).
9. **PatchTST (NEW, patch-len ~16, stride 8, enc168)** — *Required:* 2023–2025 time-series SOTA that frequently beats TFT on energy data; without it reviewers write "why no PatchTST?" Its absence today is a gap.
10. **N-BEATS (NEW, trend+seasonality stacks)** — *Required:* strong univariate/mixer SOTA, interpretable basis expansion, cheap to train; covers the non-attention modern baseline so TFT is not a strawman.

**HYBRID (the paper's method contribution):**

11. **TFT–XGBoost OOF residual (per-horizon, K=5 folds)** — *Required:* the only methodological novelty hook; OOF walk-forward design is correct and rare; per-horizon + per-regime analysis (H4) makes a *selective-gains* claim that survives the linear-wins headline. Upgrade folds 2→5, residual features = known-future + horizon_idx + encoder stats (existing `_window_rows` kept, audited).

Deliberately excluded (state why): Informer/Autoformer (superseded by PatchTST), Prophet (weak hourly), DeepAR (heavy, low marginal value over TFT quantiles), LLM-forecasters (reproducibility concerns in 2026 power venues).

---

## TASK 6 — EXPERIMENT DESIGN (Q1-level, implementable)

### 6.1 Chronological split (fixed, citable)

- Train: 2020-01 → 2023-12 (48 mo, incl. COVID + rebound). Val: 2024-01 → 2024-12 (tuning + early stopping). Test: 2025-01 → 2026-01 (full seasonal holdout, includes record peaks). Ratios ≈ 67/16/17 — close to current 70/15/15, preserves comparability while giving a clean calendar-year test reviewers prefer over mid-month cuts (2024-03-14 / 2025-02-06 are indefensible in print).

### 6.2 Rolling-window validation (generalization proof)

- **Backtest:** 4 rolling origins (test-start shifted quarterly through 2025), each with 168h-in → 24h-out scoring on identical origins; report mean±std across origins + full-year test. Implements as `TemporalSplit.rolling_origins()` (new) reusing `EvaluationAlignment`.
- **Cross-regime:** score full test + heat/cold/peak/normal slices separately (Task 3D keys).

### 6.3 Multiple random seeds

- Stochastic (LSTM/GRU/TFT/PatchTST/N-BEATS/Hybrid-XGB): seeds `{42,123,2025}`, report mean±std + 95% bootstrap CI per metric/horizon. Deterministic (persistence/ARIMA/XGB-with-fixed-seed/LGBM): single run + bootstrap CI. Single-seed rankings are not publishable — current repo's seed-42-only is a listed limitation.

### 6.4 Hyperparameter optimization (Optuna, budgeted, logged)

- Search per model on **validation only** (never test): XGB/LGBM 50 trials (depth, lr, reg, subsample, min-child + early-stopping-rounds); LSTM/GRU 30 trials (hidden 32–128, layers 1–2, dropout 0–0.3, lr); TFT/PatchTST/N-BEATS 20 trials each (hidden/patch/heads/stacks, lr, dropout) with max-epoch + patience caps (budget: ~1 GPU-week; log wall-time per trial in `optuna_study.db` + `tuned_results_v2.json`). Freeze best config → retrain train+val → score test once. Publish search spaces + importances (replaces current `tuned_results.json` whose "params: {method: lag_1}" is not HPO).

### 6.5 Ablation studies (feature groups × horizons — the paper's empirical heart)

Fix feature sets (cumulative), run **XGBoost + TFT** (one tabular, one attention) at 1/6/24h:

- **A Historical-only:** `lag_1/24/168 + rolling_24/168` (tests H1: how far does autocorrelation go?).
- **B +Calendar:** A + hour/dow/holiday/season/cyclical/DST (tests calendar dominance at 24h).
- **C +Weather:** B − renewables + T/H/CDH/HDH/heat-index/wind-speed/zenith (tests H3; expected largest peak-slice gain).
- **D +Renewable:** B + wind/solar/total/penetration/net-load context (tests whether renewables help *load* vs *net-load* framing; expect small on load, large if target switched to net-load — report both).
- **E All:** B+C+D (tests interactions/overfit; if E < C, that is a regularization finding, not a bug — current `ablation_v2.json` shows this pattern but uninterpretable without weather).

Report ΔRMSE/MAE vs A per horizon + per event slice. This table alone justifies publication if H2/H3 hold.

### 6.6 Statistical testing (correct implementation — current one is wrong)

- **Friedman** over (origins × horizons) ranks for 11 models (df=10) — omnibus.
- **Pairwise Wilcoxon signed-rank** on per-origin errors with **Holm correction** (28→55 pairs; report adjusted p + median Δ + Cliff's δ effect size). Fix integer-W bug; never report p=0.0 (use p<1e-16 / log-p).
- **Diebold-Mariano (NEW, mandatory for forecasting venues)** per horizon for top-3 pairs ( Harvey-Leybourne-Newbold small-sample variant, h-step autocorrelation-robust). Reviewers check for DM by name.
- **Nemenyi CD diagram** for the 24h leaderboard figure.

### 6.7 Explainability (from fitted models, names reconciled)

- **SHAP:** TreeExplainer on tuned XGB per horizon (beeswarm + dependence for `lag_1, lag_168, CDH, holiday`); delete phantom `net_load` unless target is net-load.
- **TFT attention + variable selection:** `interpret()` encoder attention curves + variable importances per horizon (already implemented in `ft_model.py:330` — wire to figures, not JSON dumps).
- **Agreement test (novelty-adjacent):** rank-correlation between SHAP ranks and TFT importance ranks — "two paradigms agree lag_1≫weather≫renewables at 1h, weather rises at 24h" is a quotable finding.
- **Counterfactual slice:** show one heat-wave week: what happens when temperature permuted? (1 figure, high impact).

---

## TASK 7 — METRICS REQUIRED FOR Q1 (implement in `metrics.py` + `forecasting_metrics.py` + `uncertainty.py` + `cost.py`)

**Regression (extend existing):** MAE, RMSE, MAPE (zero-safe), sMAPE (fix =0 bug), R² (keep) + **MASE (NEW, mandatory — scale-free vs seasonal-naïve)**. Reportочувств.

**Forecasting (NEW module, reviewer-checked):**
- **Forecast skill score:** `1 − RMSE_model/RMSE_seasonalPersistence` per horizon (positive = beats weekly seasonality; linear's skill ≈0 at 1h is the H1 visualization).
- **Peak error:** MAE/RMSE on top-1% load hours + timing error of daily peak (±hours) — the operator metric.
- **Ramp error:** MAE on hour-to-hour Δ (predicted vs actual ramp) — critical for reserves.

**Uncertainty (upgrade existing PICP/PINAW):**
- **PICP + PINAW** at **50/80/90/95%** (not 90% only), per horizon, TFT/PatchTST-quantile/conformalized-XGB/conformalized-hybrid.
- **Calibration curves** (nominal vs empirical) + **Winkler score** (interval scoring rule). Current 0.9001-exact PICP must be reproduced from fitted quantiles or retracted.
- Hybrid intervals: replace uncalibrated TFT-bounds with **conformalized quantile regression (CQR)** on validation residuals — one paragraph method, large credibility gain.

**Deployment (NEW `cost.py`, for §8 + Pareto figure):**
- **Inference latency** (ms/origin, CPU + GPU), **parameters**, **FLOPs** (fvcore/ptflops or analytic), **training wall-time + GPU-hours**, **estimated CO₂e** (codecarbon). Plot accuracy–latency Pareto per horizon — the figure that gets cited by practitioners.

Fix now: sMAPE=0, MAPE fraction-vs-percent inconsistency, placeholder times, fractional Wilcoxon, single-level PICP.

---

## TASK 8 — SOCIETAL IMPACT ANALYSIS (quantify — prose without numbers is rejected)

Develop `results/operational_value/` (new) + paper Section 5.6. Baseline: 180-GW mean system, reserve price / VOLL assumptions stated explicitly (e.g., spinning-reserve $15/MW-h, peaker $120/MWh, CO₂ 0.4 t/MWh marginal — cite EIA/PJM sources, sensitivity ±50%).

Translate **per-horizon RMSE deltas** into:

- **Grid reliability:** ΔMW of required operating reserve (rule-of-thumb: 3σ of 1h-ahead error); peak-slice error → loss-of-load-probability proxy; ramp-error → regulation requirement. Example framing: "500 MW RMSE reduction ≈ X MW less spinning reserve ≈ $Y M/yr."
- **Renewable integration:** with net-load target, ΔRMSE → ΔMWh curtailment avoided during solar ramps (use zenith + penetration curve); quote curtailment % on top-solar weeks.
- **Reserve management:** regulation vs spinning split; show hybrid's 6h-gain matters more for intraday re-dispatch than 1h-accuracy.
- **Electricity cost reduction:** (ΔMAE × hours × marginal price) annual $ estimate for test year; separate peak vs off-peak (peak MWh worth 3–5×).
- **Carbon reduction:** avoided peaker MWh × marginal emission factor → tCO₂/yr; include storage-shift vignette (1 figure: better 24h forecast → X% less peaker dispatch on 3 case days).
- **Equity/resilience note (required by IEEE TSG reviewers):** aggregate-RTO gains mask zonal congestion; state that deployment must be zonal + include low-income heat-vulnerability caveat for heat-wave slices.

Every number gets a confidence interval and an assumption table. No number → cut the claim.

---

## TASK 9 — CODE IMPLEMENTATION PLAN (file-by-file)

> Convention: **Current** (what exists) → **Problem** (why insufficient) → **Required change** (what to implement). No code written here — this is the build order.

**`configs/pjm_data.yaml` → `configs/pjm_data_v2.yaml` (NEW)**
Current: RTO load/wind/solar, weather null. Problem: unpublishable feature set. Change: add `weather:{temperature,humidity,wind_speed,cloud}`, `calendar:{holidays:true,dst:true}`, `derived:{cdh_hdh:true,heat_index:true,zenith:true,net_load:true}`, paths to `pjm_smart_grid_2020_2025_v2.csv` + `pjm_weather_2020_2025.csv`.

**`configs/models.yaml`**
Current: 8 models, tiny encoders, no ES. Problem: unfair + missing SOTA. Change: `seq_len/encoder_length:168`, `output/prediction ∈ {1,6,24}` parametrized per run; add `patchtst{}, nbeats{}, arima{}, persistence{}` sections; enable `early_stopping_rounds` defaults; log Optuna spaces in `configs/hpo_spaces.yaml` (NEW).

**`configs/experiment.yaml`**
Current: single horizon 24, single split. Problem: mismatch root. Change: `horizons:[1,6,24], encoder:168, seeds:[42,123,2025], rolling_origins:4, calendar_test_year:2025`.

**`src/data/schema.py`**
Current: temp/humidity/solar/wind mapping only. Problem: no weather-speed/holiday/net-load. Change: extend `DataMapping` + `FeatureConfig(weather,holidays,degree_hours,net_load,zenith)` + `EVENT_COLS`.

**`src/data/weather.py` (NEW)**
Current: nothing. Problem: no weather. Change: ERA5/NOAA fetch → footprint aggregation → hourly `temperature,humidity,wind_speed,cloud` + QA report (`weather_coverage.json`).

**`src/data/events.py` (NEW)**
Current: nothing. Problem: no slices. Change: deterministic `heat_wave/cold_wave/peak_day/normal` flags + `events.parquet` + counts table.

**`src/data/loader.py`**
Current: single-CSV canonical load. Problem: cannot join weather/holiday. Change: multi-source `load_v2()` with merge-asof + join-loss accounting.

**`src/data/pjm_integration.py`**
Current: RTO-only, silent inner-join, undocumented clips. Problem: bias flags. Change: emit `merge_statistics` with row-loss %, clip counts + sensitivity config (`clip|sensitivity:winsorize` run), preserve v1 builder untouched.

**`src/data/preprocessing.py`**
Current: solid generic cleaner. Problem: no weather-specific QA. Change: add range checks (T −40..45°C, RH 0–100), impossible-ramp flags, `weather_qa.json`.

**`src/data/features.py`**
Current: calendar/cyclical/lags/rolling, leakage-safe. Problem: missing Q1 families. Change: add holiday/season/DST/degree-hours/heat-index/zenith/net-load/penetration builders, same shift-discipline; add `FEATURE_GROUPS = {A,B,C,D,E}` for ablations.

**`src/data/pipeline.py`**
Current: single-frame → train/val/test + TFT frame. Problem: competing TFT construction in `run_phase_6b`. Change: single `run_pipeline_v2()` emitting `train/val/test.csv + tft_frame.parquet + test_origins.parquet + events.parquet + dataset_metadata_v2.json`; delete forked logic from runners.

**`src/data/adapters.py`**
Current: tabular/sequence/TFT adapters, dead `temperature` map. Problem: horizon-agnostic, no origin windows. Change: add `origin_windows(df, encoder=168, horizons=[1,6,24])` yielding identical origin sets for all families; fix TFT known/unknown roles with weather; assert equal-n.

**`src/evaluation/metrics.py`**
Current: MAE/RMSE/MAPE/sMAPE/R². Problem: sMAPE bug path, no MASE/skill/peak/ramp. Change: fix bugs, add `mase, skill_score, peak_error, ramp_mae` + unit tests enforcing fraction-vs-percent convention.

**`src/evaluation/forecasting_metrics.py` (NEW) / `uncertainty.py` (NEW) / `cost.py` (NEW)**
Current: none (PICP one-off JSON). Problem: Q1 trust/deployment missing. Change: implement Task 7 sets + calibration curves + CQR + latency/params/FLOPs/CO₂ loggers.

**`src/evaluation/alignment.py`**
Current: `(group,time)` intersection, good. Problem: not horizon-aware; permits unequal-n tables. Change: `(origin,step,horizon)` keys, `require_equal_n=True` fail-closed, horizon-stratified `metrics_by_horizon()`.

**`src/training/splits.py`**
Current: `TemporalSplit` single split, excellent. Problem: no rolling origins / calendar-year test. Change: add `rolling_origins(n=4, step_days=90)` + `calendar_split(train_end=2023-12-31,val_end=2024-12-31)`; keep old API.

**`src/training/trainer.py` + `experiment_runner.py`**
Current: single-seed, `use_early_stopping=False`, ad-hoc TFT branch in `run_phase_6b`. Problem: unfair + unrepeatable. Change: runner takes `horizon, seed, origin_set`; enforces explicit splits for all families (no internal re-split); logs times/params; `MODEL_REGISTRY` += persistence/seasonal/arima/patchtst/nbeats.

**`src/models/baseline/*` + NEW `src/models/statistical/{persistence,arima}.py`**
Current: LR/RF/XGB/LGBM defaults. Problem: untuned, missing baselines. Change: Optuna-ready constructors + `predict_h(horizon)` multi-output; new statistical baselines (seasonal-naïve trivial, SARIMA via statsmodels/pmdarima with order logged).

**`src/models/deep/sequence.py,lstm.py,gru.py`**
Current: `seq_len=24/output=1`, internal val-fraction fork. Problem: wrong window, split fork. Change: `seq_len=168`, `output_size=h`, external val only (`val_fraction` path removed for research runs), HPO hooks, identical-vs-GRU regression test (must NOT be identical).

**`src/models/tft/{dataset,ft_model}.py`**
Current: working quantile TFT + `interpret()`. Problem: enc24/hor6, doc mismatch. Change: enc168/hor∈{1,6,24}, reconcile docs, persist checkpoints per horizon/seed, wire `interpret()` to figures.

**`src/models/advanced/{patchtst,nbeats}.py` (NEW)**
Current: absent. Problem: no modern SOTA. Change: implement via `neuralforecast` or native torch (prefer library + pinned version + seed control), same `BaseForecaster` API + quantile/CQR wrapper for PatchTST.

**`src/models/hybrid/tft_xgb.py`**
Current: good OOF-2 design. Problem: thin folds, single horizon, uncalibrated intervals. Change: `n_oof_folds=5`, per-horizon instances, residual diagnostics (`residual_report.json`), CQR intervals, `interval_diagnostics` gating in tables.

**`src/training/ablation.py`**
Current: model-variant ablation (xgb/tft/hybrid), horizon-confused. Problem: answers wrong question. Change: rewrite to feature-group A–E × {XGB,TFT} × {1,6,24} on common origins (Task 6.5); old variant logic moved to `ablation_models_legacy.py` (deprecated, not deleted).

**`src/training/hpo.py` (NEW) + `src/training/stats.py` (NEW) + `src/training/explain.py` (NEW)**
Current: none. Problem: no HPO/stats/SHAP pipeline. Change: Optuna studies (SQLite + JSON export), correct Friedman/Holm-Wilcoxon/DM + CD diagram, SHAP + attention + agreement test.

**`run_phase_6b.py` → `run_q1_benchmark.py` (NEW, old frozen)**
Current: 8-model, single-horizon, mixed TFT path. Problem: invalid comparison. Change: new runner loops horizons×seeds×origins×11 models with common origins, writes `results/q1/{h{h}}/{seed}/` + `leaderboard_{1,6,24}.csv` + `verification.json`; old script kept read-only for provenance.

**`tests/`**
Current: 12 files, protocol-heavy. Problem: no task-fairness/stats/metric tests. Change: add `test_unified_task.py` (equal-n enforcement), `test_no_leakage_weather.py`, `test_stats_correctness.py` (integer-W, Holm), `test_metrics_units.py`, `test_nonidentical_dl.py`.

**`results/` + `saved_models/`**
Current: scattered JSONs, empty weights/tables. Problem: irreproducible. Change: `results/q1/` canonical tree (leaderboards, ablations, calibration, shap/, costs, operational_value/) + `saved_models/q1/` all 33 (11×3 horizons, best seeds) weights + `environment.lock` + `RUN_MANIFEST.json`.

### Build order (dependency-safe, 4–6 months, 1 researcher + GPU)

1. Weeks 1–2: Task 4 protocol + alignment fail-closed + origin sets (unblocks everything).
2. Weeks 2–5: Task 3 weather/events/v2 dataset (long pole: ERA5 access).
3. Weeks 4–7: Task 5 new models (statistical + PatchTST/N-BEATS) + horizon parametrization.
4. Weeks 6–10: Task 6 HPO + rolling origins + 3-seed runs (GPU-bound; run 1h first, then 6/24h).
5. Weeks 10–12: Task 7 metrics/uncertainty/CQR + Task 6.7 explainability.
6. Weeks 12–14: Task 8 operational translation + Task 10 manuscript draft + artifact freeze.

---

## TASK 10 — FINAL Q1 PAPER STRUCTURE (with what goes in each section)

**Title:** Option 1 (recommended): *Trustworthy Short-Term Load Forecasting for Smart Grids: A Leakage-Free Benchmark of Statistical, Machine Learning, and Transformer–Hybrid Models on Six Years of PJM Data*

**Abstract (250 words, structured):** Context (180-GW PJM, solar 6.5×) → Gap (leakage-prone, horizon-inconsistent comparisons; no weather-aware trust package) → Method (52,608h + ERA5 weather + holidays + net-load; 168h→1/6/24h; 11 models; rolling-origin, 3 seeds, Optuna, DM/Friedman/Holm; PICP×4, SHAP/attention, peak/ramp, cost/carbon) → Results (H1–H5 outcomes with numbers + CIs; state linear-1h result plainly) → Conclusion (deployment rule + artifact release). No superiority claim beyond what CIs support.

**1. Introduction:** motivation (balancing cost, 2025 peaks, solar growth) → problem (hourly load/net-load, horizons operationally mapped) → gap (4 bullets: leakage, horizon mismatch, weather absence, trust/deployment deficit — cite 20–30) → contributions (Task 2.5, 5 items) → paper map. Figure 1: system + horizon-use schematic.

**2. Related Work:** statistical (ARIMA/SARIMA), ML (XGB/LGBM), recurrent (LSTM/GRU), attention (TFT/PatchTST), hybrids/residual, probabilistic forecasting, explainability in energy, operational-value studies. End with gap table (prior × leakage-safe? × multi-horizon? × weather? × calibration? × open artifacts?) — this table is the Q1 argument.

**3. Methodology:**
- 3.1 Dataset: PJM build (aggregation, RTO filter, join-loss %, clip audit) + ERA5/NOAA weather + holidays/events + v2 schema + descriptive stats + penetration curves.
- 3.2 Feature engineering: shift-safe lags/rollings, degree-hours/heat-index/zenith/net-load, known-vs-unknown-at-forecast-time table (reviewers check this).
- 3.3 Models: 11 with equations-lite + why-required (Task 5) + horizon parametrization + hybrid OOF diagram + CQR note.
- 3.4 Experimental protocol: unified task box (168→1/6/24), splits + rolling origins figure, HPO spaces/budget, seeds/CIs, alignment equal-n rule, leakage checklist, compute environment. This section must be reproducible from text + repo.

**4. Results:**
- 4.1 Main comparison: three leaderboards (1/6/24h) with mean±std + skill + MASE + CD diagram (RQ1/H1/H2).
- 4.2 Ablation A–E × horizons × slices (RQ2/H3) — grouped-bar figure + Δ table.
- 4.3 Hybrid diagnosis: per-horizon/per-regime gains + residual plots (RQ4/H4).
- 4.4 Uncertainty: PICP/PINAW ×4 levels + calibration curves + Winkler (RQ5/H5).
- 4.5 Explainability: SHAP beeswarm + TFT attention + rank-agreement + heat-wave counterfactual (RQ5).
- 4.6 Robustness: heat/cold/peak/normal slices + rolling-origin variance (RQ5).
- 4.7 Deployment: latency–params–accuracy Pareto + training cost/CO₂ table (RQ6).

**5. Discussion:**
- Why models behave (lag decay, weather dominance, boosting vs attention inductive bias, hybrid residual structure).
- Generalization (rolling + slice consistency; single-ISO caveat).
- Operational translation (Task 8 numbers with assumption table).
- Limitations (own them: RTO-aggregate, nowcast-weather assumption, US-only, 24h-max horizon, compute budget) — a strong limitations section is Q1-expected.

**6. Conclusion:** selection rule ("deploy X at 1h, Y at 24h, hybrid when…"), trust checklist, artifact statement (data build scripts + weights + leaderboards DOI), future (zonal, probabilistic dispatch co-opt, NWP-coupled).

**Appendices:** HPO spaces + importances, DM/σα tables, per-seed tables, clip-sensitivity, `verification.json` excerpts, compute manifest.

**Figures (10–12) / Tables (6–8):** Fig1 system; Fig2 protocol/origins; Fig3 leaderboards+skill; Fig4 ablation; Fig5 hybrid residuals; Fig6 calibration; Fig7 SHAP+attention; Fig8 peak-week case; Fig9 Pareto; Fig10 slice robustness. Tables: dataset stats; HPO spaces; 3 leaderboards (or one 3-panel); ablation Δ; stats (DM/Holm); cost/carbon.

---

## FINAL ROADMAP + PUBLICATION STRATEGY

### Current problem (one line)
Leakage-safe plumbing with incomparable, partially placeholder results and no weather — unpublishable at Q1.

### Required modifications (minimal publishable set)
Unified 168→1/6/24 equal-n protocol → weather/events/v2 data → 11-model set (persistence/SARIMA/PatchTST/N-BEATS added) → Optuna + 3 seeds + rolling origins → correct stats (DM/Friedman/Holm) → CQR calibration + SHAP/attention → peak/ramp/cost/carbon → frozen weights + q1 results tree.

### New experiments (countable)
33 trained configs (11×3 horizons, best-seed reported, 3-seed variance for stochastic) + 30 ablations (A–E × 2 models × 3 horizons) + 4 rolling origins + 4 event slices × 3 horizons + 4-level calibration + HPO (~200 trials total). Budget ≈ 1 GPU-week + 2 CPU-weeks.

### Expected contribution (if H1–H5 hold)
Aokeny: the first leakage-certified, weather-aware, multi-horizon PJM-2020–2025 trust benchmark with honest horizon-dependent selection rules and operational translation — citable as method + negative-result + deployment reference even by groups whose models beat yours later.

### Expected reviewer questions (pre-answer in text)
1. "Linear beats everything — contribution?" → H1/H2 + skill curves + 24h reversal (or honest non-reversal) + trust package; contribution is selection rigor, not DL win.
2. "Why these 11?" → Task 5 table; excluded-list paragraph.
3. "Weather nowcast leakage?" → known/unknown table + shift audit + NWP caveat + sensitivity (actuals vs lagged-24h weather).
4. "Single ISO?" → rolling-origin + slice robustness + stated US-only limitation + cross-ISO future work (do not fake it).
5. "Compute/reproducibility?" → manifest + weights DOI + seeds/CIs + pinned env.
6. "Operational numbers believable?" → assumption table + ±50% sensitivity + CI on every $/tCO₂ claim.
7. "Hybrid adds what?" → selective-gains framing with residual diagnostics; no universal claim.
8. "Why not longer horizons (week)?" → scope to operational horizons; list as future.

### Publication strategy
- **Primary:** Applied Energy (benchmark + societal value fit) or IEEE TSG (trust/operation fit). Preprint arXiv after artifact freeze (never before — current numbers must not leak).
- **Fallback:** Energy → IJEPES (same manuscript, trimmed operational section).
- **Timeline:** Protocol+v2 data (5 wks) → models+HPO+runs (5 wks) → trust/operational/manuscript (4 wks) → internal red-team review (1 wk: reproduce leaderboards from weights on clean machine) → submit.
- **Red-team gate (do not submit unless all pass):** equal-n verified per table; no identical-across-models numbers; integer-W stats with Holm; weather present; 3-seed CIs; weights load and reproduce leaderboard within tolerance; limitations section names linear-wins + single-ISO + nowcast assumption.

*End of roadmap. Next action: implement protocol fix (Task 4) first — nothing else is scor seperable until horizons and origin sets are locked.*
