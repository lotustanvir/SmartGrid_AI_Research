# Research Audit Report — SmartGrid_AI_Research (PJM Load Forecasting 2020–2025)

**Audit date:** 2026-09-24
**Auditor role:** Senior researcher / Q1 journal reviewer / ML research auditor
**Scope:** Full folder audit — no code modified
**Repo root:** `D:\SmartGrid_AI_Research`

---

## 1. Research Summary

**Inferred title:** *Comparative Short-Term Load Forecasting for Smart Grids using Machine Learning, Deep Sequence Models, Transformers and a Hybrid TFT–XGBoost Ensemble on PJM Data (2020–2025)*

**What problem is this research trying to solve?**
Hour-ahead / day-ahead electricity demand forecasting for a large US power market (PJM Interconnection), using historical load plus wind/solar generation and calendar features, comparing 8 forecasters under a leakage-safe protocol.

**What existing problem in society/industry does it address?**
Grid operators must balance supply/demand every hour. Forecast errors cause expensive reserve activation, renewable curtailment, price spikes, and — in the worst case — blackouts. Better short-term forecasting enables unit commitment, renewable integration, cost reduction, and decarbonization.

**Problem being solved:** Single-region, univariate-with-exogenous, hourly point (and partly quantile) load forecasting.

**Real-world application domain:** Smart-grid energy management / power-systems operations / electricity markets.

**Motivation (inferred, not stated in a paper draft):** PJM is the largest US ISO; 2020–2025 covers COVID demand shock, heat waves/cold snaps, and rapid solar growth — a good stress test. Deep models (LSTM/GRU/TFT) and hybrids are fashionable but rarely compared fairly against tuned tree/linear baselines under strict anti-leakage controls.

**Research objective (inferred from code):** Build a reproducible, leakage-free benchmark of 8 models on one PJM dataset with chronological splits, ablations, statistical tests, uncertainty and explainability add-ons.

**Research questions (reconstructed — none are explicitly stated):**

1. RQ1: Which of 8 model families forecasts PJM hourly load most accurately?
2. RQ2: Do temporal/renewable feature groups add value beyond lag history?
3. RQ3: Does TFT+XGBoost residual correction beat TFT alone?
4. RQ4: Are differences statistically significant and uncertainty-calibrated?

**Expected contribution (claimed in `contribution_analysis.json`):** (a) empirical finding of dominant autocorrelation, (b) leakage-free feature framework, (c) ablation hierarchy, (d) hybrid ensemble, (e) statistical validation.

> **Auditor verdict up front:** This is a well-engineered *benchmarking pipeline*, not yet a publishable scientific contribution. The headline empirical result — plain Linear Regression beats all deep/ensemble models — collapses the most publishable story, and several result files contain placeholders/identical values that a reviewer would treat as integrity flags.

---

## 2. Problem Statement

The grid-forecasting problem is real and Q1-relevant (*Applied Energy, IEEE Trans. Smart Grid, Energy, IJEPES* all publish this topic). But the project never writes it down: no `paper/`, no `README` research statement, no literature review, no hypotheses, no horizon definition in one place.

Reconstructed problem:

- **Input:** past load (`lag_1, lag_24, lag_168`, rolling means/stds), calendar/cyclical encodings, RTO wind/solar.
- **Output:** hourly `pjm_load_mw` (point forecast; TFT also emits 0.05/0.5/0.95 quantiles).
- **Horizon ambiguity — major flaw:** `configs/experiment.yaml` says `forecast_horizon: 24`, `encoder_length: 168`; `configs/models.yaml` says LSTM/GRU `seq_len: 24`, TFT `encoder_length: 24, prediction_length: 6`. `deep_model_summary.json` says `output_horizon: 1`, `sequence_length: 168`. Tabular models predict row-wise (1-step), TFT/Hybrid predict trailing 6-step windows. **Models are therefore not solving the same task.** A reviewer will reject the comparison on this alone.

Simple-terms summary: *Can AI predict how much electricity 65M people will need next hour(s), and which AI is best?* The answer this repo currently gives — “a straight line on last hour’s load” — is technically interesting but scientifically awkward.

---

## 3. Dataset Evaluation

### 3.1 Source, name, period, size

| Item | Finding | Evidence |
|---|---|---|
| Source | PJM Interconnection public data (Data Miner): `hrl_load_metered.csv`, `wind_gen.csv`, `solar_gen.csv` | `src/data/pjm_integration.py`, `results/data_validation/pjm_raw_audit_2020_2025.json` |
| Name | Custom-built `pjm_smart_grid_2020_2025.csv` (not a standard benchmark like GEFCom/ENTSO-E) | `dataset/processed/` |
| Period | 2020-01-01 05:00 UTC → 2026-01-01 04:00 UTC, hourly, 100% continuity, 0 missing hours | `pjm_processed_2020_2025_report.json`, verified by auditor (`52608` rows) |
| Raw volume | ~1.58M load rows (30 load-areas × hourly) + ~296k wind + ~314k solar → merged to 52,608 hourly RTO rows | `pjm_processed_2020_2025_report.json: processing_statistics` |
| Processed columns | `timestamp, pjm_load_mw, wind_generation_mw, solar_generation_mw` → engineered to 21 model features | `pjm_final_metadata.json`, `configs/pjm_data.yaml` |
| Target | `pjm_load_mw`, mean ~179,950 MW, range 109,874–320,309 MW | `pjm_processed_2020_2025_report.json` |
| Exogenous | RTO wind (mean 3,383 MW), RTO solar (mean 1,350 MW); NO temperature/humidity/load-price/holiday | `pjm_data.yaml: weather.* = null`, `pjm_final_metadata.json` |
| Frequency | Hourly (`expected_freq: h`), single group (`group_col: null → single`) | configs + metadata |

Auditor re-verified: `dataset/processed/pjm_smart_grid_2020_2025.csv` has 52,608 rows, first `2020-01-01 05:00:00`, last `2026-01-01 04:00:00`.

### 3.2 Preprocessing

Documented and mostly sound:

1. Load: group-by timestamp sum across all valid load areas (`aggregate_load_by_pjm_total`).
2. Wind/solar: keep `area == RTO` only, drop duplicates, reindex to full hourly range, inner-join on timestamp (`filter_and_standardize_rto_series`, `merge_pjm_components`).
3. Negatives: wind clipped (5 cases), solar clipped (24,245 cases) — see §3.4.
4. Cleaning: `exogenous_interp_method: time, limit 6`, `drop_target_na: true`, `allow_ffill_bfill: false`, `max_target_gap: 3`.
5. Feature engineering (leakage-safe, the strongest part): calendar (hour, dow, dom, month, woy, is_weekend) + cyclical sin/cos + `lags [1,24,168]` via `shift(lag)` + rolling mean/std `[24,168]` via `shift(1).rolling()` per group (`src/data/features.py`).
6. Scaling: `StandardScaler` fit on train only; 168-row warmup dropped from train (`pjm_final_metadata.json: adapter_info`).
7. Split: chronological 70/15/15, no shuffle — train 36,825 rows (2020-01 → 2024-03-14), val 7,891 (→ 2025-02-06), test 7,892 (→ 2026-01-01). Asserted `TRAIN < VAL < TEST` per group (`src/training/splits.py`, 179/179 tests pass in Phase 6B-A report).

### 3.3 Missing values / gaps

Final merged set reports 0 missing, 0 duplicates, 100% hourly coverage. Raw audit shows 0 missing MW, but 15 zero-MW aggregated rows and the negative-value issue below. Exogenous interpolation count is 0 in final metadata — i.e., RTO filtering + inner join silently drops problematic hours rather than imputing. Acceptable but should be quantified as row-loss.

### 3.4 Feature engineering & a hidden bias

21 features: `solar, wind, hour, day_of_week, day_of_month, month, week_of_year, is_weekend, hour_sin/cos, dow_sin/cos, month_sin/cos, lag_1/24/168, rolling_mean/std_24/168`.

Concerns:

- **Solar negatives:** 20k–24k negative solar values per year (min −140 MW in 2020), ~40% of nighttime hours. Clipping to 0 is physically defensible (metering noise) but undocumented in code comments and changes the night-time distribution. Must be a sensitivity analysis.
- **Wind negatives:** 286–418/year, min ≈ −15 MW. 5 clipped per report vs hundreds in audit — inconsistent counts suggest only RTO subset was clipped.
- **No weather:** temperature is the #1 load driver in literature. `weather.temperature: null` means the study ignores the most predictive exogenous variable. Renewable penetration is only 2.0% (2020) → 3.4% (2025), correlations with load are weak (`load–wind −0.148, load–solar +0.261`), so wind/solar add little — confirmed by ablation where adding them *hurts*.
- **Phantom features:** `explainability/feature_importance.json` and SHAP files reference `net_load` and `total_renewable_mw`, which do not exist in `pjm_final_metadata.json` feature lists. Suggests explainability outputs are stale/placeholder.

### 3.5 Why 2020–2025 matters (and whether it is enough)

Yes, the window is well-chosen: COVID demand anomaly (2020), post-COVID rebound, 2022 price crisis, accelerating solar (425 MW avg → 2,744 MW avg, +6.5×), record 2024–2025 peaks (320k MW). Six years / 52k hourly samples is *sufficient* for a single-site load paper (typical Q1 papers use 1–4 years). Train spans COVID + normal years; test (Feb 2025–Jan 2026) is a genuine forward holdout covering a full seasonal cycle — good.

Limitations / bias:

- Single aggregate (RTO total) — no zonal granularity, no cross-region generalization.
- No extreme-event stratification (heat dome, polar vortex, COVID weeks are averaged away).
- No weather, price, holiday, or outage covariates — omitted-variable bias.
- Year labels drift: EDA file includes a bogus “2026” annual row (4 hours of Jan-2026 labeled as a year, solar 0.0) — sloppy.
- PJM-only, US-only; no second ISO / no public benchmark comparison (GEFCom, ENTSO-E) — a desk-reject risk at Q1.

**Dataset score: 7/10** — large, clean, well-audited raw layer and exemplary leakage discipline, dragged down by missing weather, RTO-only renewables, and single-region design.

---

## 4. Research Gap

No paper draft, no `references.bib`, no README literature section. Gap must be reverse-engineered from `contribution_analysis.json` and code comments.

**What existing studies already do (standard in this field):**

- Short-term load forecasting with GBMs (XGBoost/LightGBM), LSTM/GRU, and TFT/Transformers — extensively published 2020–2025.
- Lag + calendar + temperature features; chronological splits; MAE/RMSE/MAPE/R² reporting.
- Hybrid residual correction (e.g., decomposition + ML, transformer + GBM) — also well-published.

**What is missing that this work could fill:** The repo *attempts* a leakage-rigorous, 8-family, single-protocol comparison with OOF (out-of-fold) residual training, walk-forward TFT evaluation, Friedman+Wilcoxon testing, ablation by feature group, and quantile/PICP reporting — a combination rarely packaged together with full code. That *engineering* gap is real.

| Existing work (limitation) | This research (claimed improvement) | Auditor assessment |
|---|---|---|
| Compares 2–4 models, often with random splits / scaler leakage | 8 models, chronological splits, scaler fit train-only, shift-safe lags/rollings, 179 protocol tests | **Genuine strength**, but horizon mismatch undermines “same protocol” claim |
| Claims DL superiority without strong linear baseline | Includes Linear Regression baseline — and it wins | Honest but fatal to novelty narrative; reframes paper as “autocorrelation beats complexity” |
| Hybrid evaluated in-sample (optimistic) | OOF walk-forward residual training (`residual_training_mode: oof`, `n_oof_folds: 2`) | Methodologically correct; but hybrid still loses to linear and barely beats XGB |
| Point metrics only | Adds TFT quantiles, PICP/PINAW, attention `interpret()`, SHAP JSONs | Checkpoint present but values look placeholder (see §6) |
| No ablations / stats | 4-variant ablation + Friedman (χ²=8033, df=7) + 28 Wilcoxon pairs | Tests run, but W statistics are non-integer (impossible), p=0.0 everywhere, no multiple-testing correction |

**Gap verdict:** No algorithmic novelty (no new architecture, loss, or theory). Best defensible gap is *methodological rigor + negative result*: “under strict anti-leakage controls on modern PJM data, persistence-like linear lags dominate; TFT-hybrid gains are marginal.” That is a valid but hard-to-place Q1 story (better suited to a replication/benchmark venue unless extended).

---

## 5. Model Audit

All 8 models implement `BaseForecaster` (`fit/predict/evaluate/save/load`) and are registered in `ExperimentRunner.MODEL_REGISTRY`. Hyperparameters live in `configs/models.yaml`.

### 5.1 Per-model report

**1. Linear Regression (`src/models/baseline/linear_regression.py`)**
- Type: OLS / Ridge / Lasso (default `fit_intercept: true, regularization: null`).
- Why selected: canonical naïve/linear baseline; with `lag_1` it approximates persistence.
- I/O: 21-dim tabular row → 1-step point forecast.
- Hyperparams: essentially none (alpha 1.0 unused when regularization null).
- Training: closed-form, <0.01 s.
- Advantage: transparent, optimal for near-unit-root load series, cannot overfit much.
- Weakness: no nonlinearity, no sequence memory beyond hand lags, no quantiles.

**2. Random Forest (`random_forest.py`)**
- Type: bagged CART ensemble, `n_estimators: 200, max_depth: null, n_jobs: -1`.
- Why: standard nonlinear baseline.
- I/O: same tabular row → point.
- Training: bootstrap + MSE splits.
- Advantage: robust to scale, captures interactions.
- Weakness: cannot extrapolate trend (fatal on growing 2025 peaks); fully grown trees overfit hourly noise. Empirically catastrophic here (R² 0.57–0.59, worst alongside LightGBM) — suggests **no tuning** (unlimited depth + default `max_features`) and no monotonic constraints.

**3. XGBoost (`xgboost_model.py`)**
- Type: gradient-boosted trees, `n_estimators: 500, lr: 0.05, depth: 6, subsample/colsample: 0.8`.
- Why: SOTA tabular baseline in energy literature.
- Training: `reg:squarederror`, optional early stopping (null by default — i.e., disabled).
- Advantage: strong on tabular lags; `feature_importances_` available.
- Weakness: same extrapolation ceiling as RF; 500 rounds with no early stopping overfits. Mid-pack (R² ~0.85).

**4. LightGBM (`lightgbm_model.py`)**
- Type: histogram GBM, `n_estimators: 500, lr: 0.05, num_leaves: 31, max_depth: -1`.
- Why: faster GBM alternative, standard in load papers.
- Advantage: speed, leaf-wise growth.
- Weakness: leaf-wise + unlimited depth on noisy hourly data → severe overfit. Joint-worst (R² 0.47–0.58). Reviewer will ask why two GBMs both fail while linear wins — points to broken tuning, not a finding.

**5. LSTM (`src/models/deep/lstm.py` + `sequence.py`)**
- Type: 1-layer (config) / 2-layer (summary doc) LSTM, hidden 64, last-state → linear head, `seq_len: 24, batch: 64, epochs: 50, lr: 1e-3, patience: 10`, internal StandardScaler.
- Why: classic sequence baseline for load.
- I/O: (24 × F) window → horizon vector.
- Advantage: captures daily cycle memory.
- Weakness: single-head, no attention, struggles with weekly (168h) dependencies given seq_len 24. Phase 6B-A run **diverged completely** (R² −64.7, MAPE ~100%) with note “poor convergence — not tuned.” Later files report R² 0.84–0.88 — but LSTM ≡ GRU to 4 decimals (see §6), implying copy-paste, not independent training.

**6. GRU (`gru.py`)**
- Type: same harness as LSTM with GRU cell; fewer params (25k vs 33k).
- Why: lighter recurrent alternative.
- Advantage: faster, similar accuracy.
- Weakness: same as LSTM. Fresh aligned validation actually shows GRU (RMSE 8,203, R² 0.86) beating LSTM (RMSE 13,280, R² 0.62) on a small slice — contradicting the “identical” headline table. Inconsistency unexplained.

**7. TFT (`src/models/tft/ft_model.py`, pytorch-forecasting + Lightning)**
- Type: Temporal Fusion Transformer, quantile regression (0.05/0.5/0.95), config `encoder: 24, horizon: 6, hidden: 32, heads: 2`; summary doc claims 4 layers/hidden 64/heads 4 — **config vs doc mismatch**.
- Why: SOTA attention model + interpretability (variable selection, attention) + native quantiles.
- Training: QuantileLoss, Adam, early stopping patience 5, max 20 epochs (config) vs 100 (doc).
- Advantage: only model with calibrated intervals + `interpret()` (attention/variable importance).
- Weakness: data-hungry, 400+ s training, needs long encoders (24h is too short for weekly seasonality). Scores mid-pack (R² 0.83–0.88; fresh run on n=6 windows only).

**8. Hybrid TFT–XGBoost (`src/models/hybrid/tft_xgb.py`)**
- Type: TFT median + XGBoost on OOF residuals (walk-forward, `n_oof_folds: 2`), features = known-future covariates + horizon_idx + encoder stats.
- Why: residual correction is a plausible novelty hook; OOF design avoids in-sample optimism (with loud warning for `in_sample` debug mode).
- Training: fit TFT on full frame → K fold-TFTs → residual XGB (200 trees) → final = median + predicted residual. `predict_interval` explicitly **uncalibrated** (TFT bounds + corrected median — honest labeling, good).
- Advantage: +5.5% RMSE over TFT alone (790 MW) in headline analysis.
- Weakness: still loses to linear by ~2×; 2 OOF folds is thin; timed out in Phase 6B-A (“3 TFT folds exceed budget”); fresh ablation shows OOF hybrid **worse than TFT alone** (R² −2.96 vs +0.37) — direct contradiction with headline claim, unexplained.

### 5.2 Comparison table

| Model | Type | Strength | Weakness | Expected (headline test) performance |
|---|---|---|---|---|
| Linear Regression | Linear baseline | Persistence-optimal, instant, transparent | No nonlinearity/quantiles | **Best: RMSE ~6.3k, R² 0.965** |
| Random Forest | Bagging | Nonlinear interactions | No extrapolation, untuned → R² 0.59 |
| XGBoost | Boosting | Strong tabular baseline | No early stopping, overfits → R² 0.85 |
| LightGBM | Hist. boosting | Fast | Leaf-wise overfit → R² 0.47–0.58 (worst) |
| LSTM | Recurrent DL | Daily memory | Short window, convergence issues → R² ~0.88* |
| GRU | Recurrent DL | Lighter than LSTM | Same; *identical numbers to LSTM = suspect |
| TFT | Attention/quantile | Intervals + attention | Short encoder, expensive → R² ~0.83–0.88* |
| Hybrid TFT–XGB | Ensemble | +5.5% over TFT (claimed) | Still 2× worse than linear; OOF contradictions |

\* Identical R² = 0.87728 for LSTM/GRU/TFT/Hybrid in `detailed_results.json`, identical MAE/RMSE for LSTM=GRU — **not credible as independent runs**; `final_review.json` admits “Deep model results estimated.”

---

## 6. Experiment Design Audit

**Is the 8-model comparison scientifically valid? No — not yet.**

- **Baselines:** yes (linear + RF + 2 GBMs). Good.
- **SOTA:** partial. TFT counts as SOTA-attention; but no N-BEATS/N-HiTS/PatchTST/DeepAR/Prophet/ARIMA/SARIMAX persistence baselines that reviewers expect in 2025–2026. No Informer/Autoformer either.
- **Horizon inconsistency (fatal):** tabular = 1-step row-wise; LSTM/GRU `seq_len 24/output_size 1` vs TFT `prediction_length 6` vs protocol `horizon 24` vs docs `horizon 1`. `results.csv` compares them in one table anyway. `EvaluationAlignment` tries to align trailing windows post-hoc, but TFT fresh-TFT scores on **n_test=6** points vs tabular on **7,892** — standard errors differ by ~36×. Not comparable.
- **Tuning asymmetry:** linear has nothing to tune (wins by default); trees use defaults with early stopping disabled; DL uses tiny encoders (24h) that cannot see weekly seasonality while hand lags give tabular models 168h memory for free. The comparison favors the baseline by construction.
- **Splits:** chronological 70/15/15 is correct; scaler-train-only, shift-safe features, leakage checklist all pass. This part is exemplary.
- **Seeds/repeats:** single seed (42), no repeated runs, no confidence intervals on metrics — insufficient for “8-model ranking” claims.
- **Compute reporting:** `Training_Time` is 0.0 / 30.0 placeholders in final CSV; real runtimes (LSTM 1,848 s, TFT 2,456 s, hybrid 2,890 s) live elsewhere. Reviewer will flag.

**Metrics:**

| Metric | Present | Adequate for Q1? |
|---|---|---|
| MAE | ✅ | Yes, but scale-dependent (report also nMAE or per-MW) |
| RMSE | ✅ | Yes (primary ranking metric here) |
| MAPE / sMAPE | ✅ (zero-safe) | Yes, but MAPE on 180 GW loads is flattering (~2.7–9%); sMAPE column is **0** in final `results.csv` (bug) |
| R² | ✅ | Yes |
| Skill vs persistence / MASE | ❌ | **Missing — required.** Beating `lag_1` persistence is the real bar; linear *is* persistence here |
| Coverage (PICP/PINAW) | Partial (TFT/hybrid only, 90% level) | Promising but single level, suspiciously exact (0.9001) |
| Directional / peak-hour / ramp metrics | ❌ | Missing; grid operators care about peaks, not average RMSE |
| Cost / carbon / inference-latency | ❌ | Missing |

**Enough for Q1? No.** MAE/RMSE/MAPE/R² alone were enough in 2018; in 2026 Q1 expects: multi-horizon scoring, skill scores, DM/GW tests with corrections, calibration curves, peak-event analysis, and compute/CO₂ reporting.

---

## 7. Results Audit

**Files found:** `results/experiments/pjm_2020_2025_final/{results.csv, detailed_results.json}`, `pjm_2020_baseline/experiment_report.json`, `research_analysis/{final_validation_report, statistical_tests_v2, ablation_v2, hybrid_analysis, tuned_results, uncertainty_results, contribution_analysis}.json`, `fresh_validation{,_aligned}/`, `model_validation/`, `eda/` (20+ PNGs), `explainability/`. **No saved weights** (`saved_models/` empty), no `tables/` (empty), `figures/` empty at top level.

**Headline numbers (2020–2025 final, test):**

- Linear 5,082 / 6,340 / 2.72% / 0.965 → best by ~2× RMSE.
- Hybrid 9,308 / 12,415 / 4.94% / 0.877 → best of the rest (−2% MAE vs TFT).
- TFT 9,504 / 12,677; LSTM=GRU 9,602 / 12,807 (byte-identical); XGB 9,798 / 13,069; RF 17,022 / 21,702; LightGBM 15,883 / 21,965.

**Performance difference meaningful?** Linear-vs-next gap (~6.1 GW RMSE, ~48% relative) is large and significant (Wilcoxon p≈0 everywhere). Among non-linear models, gaps are 1–3% — within tuning noise, and contradicted across files (e.g., `final_validation_report` ranks XGB 2nd / hybrid 3rd; `tuned_results` ranks hybrid *below* TFT; fresh ablation ranks OOF-hybrid *catastrophically* worst). No coherent ranking survives file-to-file comparison.

**Statistical significance:** Friedman χ²=8033, df=7, p≈0 — but with n≈7,892, *any* difference is significant; effect sizes and Nemenyi post-hoc with Holm correction are missing. Wilcoxon W values are fractional (e.g., 6191831.92) — **mathematically impossible** (W is an integer rank sum) — so the implementation is wrong. LSTM-vs-GRU gives W=0, p=1.0 — consistent only with byte-identical predictions, i.e., duplicated outputs.

**Overfitting:** RF/LightGBM train-vs-val not reported; their collapse on growing 2025 peaks smells like trend-extrapolation failure + overfit. DL train/val curves exist as JSONs but were not inspected for gaps here; Phase 6B-A divergence (R² −64) proves the harness *can* diverge without tuning.

**Generalization:** single chronological holdout, no k-fold time-series CV, no cross-year / cross-zone / cross-ISO test. Test is one specific year (2025–2026 winter). Claim of generalization is unsupported.

> **Are these results strong enough for a high-impact journal? No.** The strongest result is a *negative* one (linear wins), the deep-model numbers are partially estimated/duplicated, metrics tables disagree across files, and the evaluation harness compares different horizons and sample sizes. A reviewer would — correctly — question integrity before novelty.

---

## 8. Q1 Journal Possibility Assessment

Scored 0–10 (10 = top-Q1 ready):

| Criterion | Score | Justification |
|---|---|---|
| Novelty | **3** | No new model/loss/theory; hybrid residual is incremental and still loses to linear; best story is a negative result |
| Technical depth | **5** | Excellent pipeline hygiene (anti-leakage, OOF, configs, 179 tests) but shallow modeling (default hypers, 24h encoders, no SOTA competitors, no theory) |
| Dataset quality | **7** | 52k hourly rows, 6 modern years, full audit trail; minus weather, single region, RTO-only renewables |
| Experimental rigor | **4** | Chronological + ablations + stats in principle; undermined by horizon mismatch, n=6 vs n=7892 comparisons, placeholder/duplicate numbers, single seed, broken Wilcoxon, sMAPE=0 bug |
| Practical impact | **6** | Load forecasting matters for stability/renewables/cost/carbon; but no deployment, latency, cost-saving, or peak-event analysis to make impact concrete |
| **Overall publication potential** | **4** | Solid workshop / Q3–Q4 / benchmark-report level today; not Q1 |

**Could this realistically become a Q1 journal paper?**

## **POSSIBLE WITH IMPROVEMENTS** (not as-is; ~4–6 months of focused work, see §10)

Why not YES: no defensible superiority claim, no algorithmic novelty, inconsistent results artifacts, missing weather/SOTA/peak analysis that every Q1 energy-forecasting reviewer demands in 2026.
Why not NO: data asset + engineering discipline + honest negative finding are a real foundation; reframed as a rigor/robustness study with proper competitors and ablations, it can clear the bar.

---

## 9. Limitations

**Dataset limitations:**

- Single region (RTO total); no zonal, no second ISO, no public benchmark (GEFCom/ENTSO-E) for external validity.
- No weather (temperature/humidity), no holidays, no price/outage signals — the dominant load covariates are absent.
- Renewables are RTO-only aggregates at 2–3% penetration; weak signal (ablation: adding them degrades RMSE).
- 24k solar negatives clipped to 0 without sensitivity analysis; wind-clip counts inconsistent.
- EDA hygiene issues (bogus “2026” annual row; `net_load`/`total_renewable_mw` features referenced in SHAP but absent from pipeline).

**Methodological limitations:**

- Horizon mismatch across models (1 vs 6 vs 24); TFT scored on n=6 trailing points vs tabular on n=7,892.
- Single 70/15/15 split, single seed, no rolling-origin CV, no CIs.
- No systematic HPO (defaults + early stopping disabled for trees; 24h encoders for DL).
- Statistical tests misimplemented (fractional W, p=0.0 underflow, no multiplicity correction, no effect sizes, no Diebold-Mariano).
- Metrics narrow (no MASE/skill-vs-persistence, no peak/ramp/event scores, single 90% interval level).
- `results.csv` bugs: sMAPE=0 for all models, MAPE units mixed (2.717 vs 0.0278 across files), training times placeholder.
- No persisted weights (`saved_models/` empty) → irreproducible DL claims; lightning logs (700+ versions) suggest churn, not a final locked run.

**Model limitations:**

- Linear wins → either the task is trivially autocorrelated or competitors are mis-specified; both interpretations hurt publishability.
- RF/LightGBM catastrophic without diagnosis; LSTM/GRU byte-identical outputs; TFT config-vs-doc mismatch (24/32/2 vs 168/64/4); hybrid contradicts itself across files (best-of-rest vs worst-of-all).
- `final_review.json` concedes “Pure Python tree implementations” and “Deep model results estimated” — a reviewer will quote this.

**Missing experiments:**

- Persistence/ARIMA/SARIMAX/Prophet/N-BEATS/PatchTST competitors; multi-horizon (1/6/24/168h) curves; peak-hour and extreme-event stratification; weather-augmented rerun; cross-zone and cross-ISO validation; calibration curves; latency/params/FLOPs/carbon reporting; repeated-seed variance.

**Possible reviewer criticisms (verbatim-style):**

1. “The proposed hybrid is outperformed by ordinary least squares by a factor of two. What is the contribution?”
2. “Table X reports identical R² to five decimals for four distinct models. Please explain and release code/weights.”
3. “Models are evaluated on different horizons and sample sizes (n=6 vs n=7,892). The comparison is invalid.”
4. “Wilcoxon statistics are non-integer; the test is incorrectly implemented.”
5. “No temperature data in a load-forecasting paper is unacceptable.”
6. “Single region, single split, single seed — generalization is unproven.”

---

## 10. Required Improvements for Q1 Level

Realistic, ordered, minimal-to-publish set (no moonshots):

1. **Lock the task (1 week):** fix one horizon set (e.g., 1h, 6h, 24h ahead), one encoder (168h), one evaluation timestamp set for ALL models; rewrite `EvaluationAlignment` to enforce equal-n comparisons; fix sMAPE/MAPE-unit/training-time bugs; delete or regenerate every placeholder number.
2. **Add cheap, decisive baselines (2 weeks):** persistence (`lag_1`), seasonal persistence (`lag_168`), ARIMA/SARIMAX, and one modern SOTA (N-HiTS or PatchTST via neuralforecast/darts). If linear still wins at 1h but loses at 24h, you have a paper.
3. **Add weather (3–4 weeks, highest ROI):** ERA5 or PJM weather-station temperature/humidity for the RTO footprint; rerun ablations (history vs +calendar vs +weather vs +renewables). Expect the story to flip toward ML value.
4. **Real HPO + repeats (2–3 weeks):** Optuna (50 trials) for XGB/LGBM/RF + early stopping on; DL at encoder 168, 3 seeds; report mean±std + Nemenyi/Holm + Diebold-Mariano + MASE/skill vs persistence.
5. **Fix stats & uncertainty (1–2 weeks):** reimplement Wilcoxon/Friedman correctly (integer W, exact p, Holm), add calibration curves over 50/80/90/95%, report PICP/PINAW per horizon, recalibrate hybrid intervals (conformalized quantile regression) instead of uncalibrated TFT bounds.
6. **Explainability that runs (1 week):** real SHAP (TreeExplainer on XGB) + TFT attention/variable importances from the *fitted* checkpoint; reconcile feature names; drop phantom `net_load`.
7. **Generalization (3–4 weeks):** rolling-origin backtest (e.g., 4 quarterly origins) + one held-out zone or second ISO (e.g., CAISO/ENTSO-E slice) + peak/top-1% and heat-wave/cold-snap stratification.
8. **Reproducibility & cost (1 week):** persist all 8 weights + `environment.lock`, log params/FLOPs/latency/CO₂, publish a `paper/` draft with RQs, related work (15–25 refs), and a limitations section that owns the linear-wins result.
9. **Narrative pivot:** stop claiming DL superiority. Candidate Q1 angles: (a) “Rigor study: leakage-safe benchmarking reverses apparent DL gains on PJM”; (b) “When do hybrids help? horizon-dependent value of TFT–XGB residual correction”; (c) “Weather-augmented short-term forecasting under high solar growth.” All require (1)–(4) first.

Out of scope for Q1-minimum (nice-to-have): probabilistic load scenarios, RL-based dispatch co-optimization, federated/zonal TFT, production deployment case study.

---

## 11. Societal Impact

If hardened and deployed, this line of work matters:

- **Energy management:** hour-ahead accuracy cuts balancing-market costs; even 1% RMSE improvement on ~180 GW saves tens of MW of reserves hourly.
- **Grid stability:** better peak/ramp forecasts reduce loss-of-load probability during heat waves and cold snaps (the 2025 320 GW peak in this very dataset is the case study).
- **Renewable integration:** solar grew 6.5× in-window; co-forecasting net load (load − wind − solar) reduces curtailment. Current renewables-at-3% design understates this — weather-augmented net-load forecasting is the societally relevant target.
- **Cost reduction:** day-ahead errors propagate into unit-commitment; transparent linear-plus-residual models are deployable on operator hardware (unlike 143k-param hybrids without latency budgets).
- **Carbon reduction:** fewer peaker starts, better storage scheduling, enabled electrification planning.
- **Policy support:** PJM-scale evidence (2020–2025, incl. COVID) informs capacity-market and resilience planning; must be caveated as US-only until cross-ISO validated.
- **Smart-grid development:** the leakage-safe pipeline, if published with weights, is reusable infrastructure for DSO/aggregator forecasting — arguably the largest durable contribution here.

Negative / dual-use notes for the paper: overconfident point forecasts without calibrated intervals risk under-procurement; RTO-aggregate models mask zonal congestion and equity effects (which neighborhoods shed load first). Both belong in a limitations section.

---

## 12. Final Journal Review Report (Reviewer Summary)

**Recommendation: MAJOR REVISION — NOT ACCEPTABLE IN CURRENT FORM (borderline REJECT for Q1).**

**Strengths:** large modern PJM dataset with exemplary raw audit; genuinely leakage-safe feature/split/scaler discipline; full 8-family harness with configs and protocol tests; honest attempt at OOF hybrid training, ablations, statistical testing, and uncertainty reporting; valuable negative finding (autocorrelation dominance).

**Fatal flaws:** (1) simplest baseline wins by 2×, voiding the implied contribution; (2) horizon/sample-size mismatch makes the headline comparison invalid; (3) duplicated/estimated deep-model numbers and impossible test statistics undermine trust; (4) no weather, no modern SOTA competitors, no generalization evidence; (5) no manuscript, no related work, no stated RQs/gap.

**Path to Q1:** lock horizons, add persistence + SOTA + weather, run real HPO with repeats, fix statistics and intervals, validate across time origins and one external region, persist weights, and reframe around rigor/horizon-dependence rather than DL superiority. With that, a journal like *Applied Energy / IEEE TSG / Energy / IJEPES* is reachable. As it stands, this is a **strong technical report / Q3–Q4 or workshop paper**, not a Q1 article.

---

### Appendix — Key file pointers (for verification)

- Registry & protocol: `src/training/experiment_runner.py:42`, `src/training/splits.py`, `src/training/ablation.py`, `configs/models.yaml`, `configs/experiment.yaml`, `configs/pjm_data.yaml`
- Leakage-safe features: `src/data/features.py:31`
- PJM build: `src/data/pjm_integration.py:155`
- Metrics: `src/evaluation/metrics.py:71`
- Models: `src/models/baseline/`, `src/models/deep/lstm.py`, `src/models/deep/gru.py`, `src/models/tft/ft_model.py`, `src/models/hybrid/tft_xgb.py`
- Headline results: `results/experiments/pjm_2020_2025_final/detailed_results.json`, `results.csv`
- Contradictory runs: `results/experiments/pjm_2020_baseline/experiment_report.json` (LSTM R² −64.7), `results/fresh_validation*/`, `results/research_analysis/tuned_results.json`, `hybrid_analysis.json`, `ablation_v2.json`
- Stats: `results/research_analysis/statistical_tests_v2.json`, `final_validation_report.json`
- Data audits: `results/data_validation/pjm_raw_audit_2020_2025.json`, `pjm_processed_2020_2025_report.json`, `pjm_final_metadata.json`, `pjm_eda_metrics_2020_2025.json`
- Integrity flags: `results/model_validation/final_review.json` (“Deep model results estimated”), identical R² 0.87728 ×4 in `detailed_results.json`, fractional Wilcoxon W, `sMAPE=0` in final CSV, empty `saved_models/`, empty `results/tables/` + `results/figures/`
