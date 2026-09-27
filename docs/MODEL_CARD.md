# Model Card — Forecaster Suite (v1 current + Q1 target)

**Phase 0 — documents what exists in `src/models/` + `configs/models.yaml`; nothing modified.**

## 1. Common interface

All forecasters subclass `BaseForecaster` (`src/models/base.py`): `fit(X,y[,X_val,y_val])` → `predict(X)` / `forecast(X,y)` → `evaluate(X,y)` → `save/load`. Registry of 8 in `src/training/experiment_runner.py:42` (`MODEL_REGISTRY` + `PRETTY_NAMES`).

## 2. Current models (v1 — hyperparameters from `configs/models.yaml`)

| # | Model (class) | Type / I-O | Key hypers (v1) | Role / why kept | Known issue |
|---|---|---|---|---|---|
| 1 | LinearRegression (`baseline/linear_regression.py`) | OLS/Ridge/Lasso; 21-dim row → 1-step | `fit_intercept:true, regularization:null` | Canonical linear/persistence-like baseline; current best (R² 0.965) | None (reference) |
| 2 | RandomForest (`baseline/random_forest.py`) | Bagged CART; row → 1-step | `n_estimators:200, max_depth:null, n_jobs:-1` | Nonlinear baseline | Unbounded depth, untuned → R² ~0.59; no extrapolation |
| 3 | XGBoost (`baseline/xgboost_model.py`) | GBM `reg:squarederror`; row → 1-step | `n_estimators:500, lr:0.05, depth:6, subsample/colsample:0.8` | Tabular SOTA | `early_stopping_rounds:null` (disabled); R² ~0.85 |
| 4 | LightGBM (`baseline/lightgbm_model.py`) | Hist GBM; row → 1-step | `n_estimators:500, lr:0.05, num_leaves:31, max_depth:-1` | Fast GBM counterpart | Leaf-wise unbounded → R² 0.47–0.58 |
| 5 | LSTM (`deep/lstm.py` + `deep/sequence.py`) | 1-layer (config) LSTM, last-state → head; window → horizon | `seq_len:24, hidden:64, layers:1, epochs:50, lr:1e-3, patience:10, batch:64` | Canonical recurrent | 24h window misses weekly cycle; diverged R² −64.7 in Phase 6B-A; headline numbers byte-identical to GRU |
| 6 | GRU (`deep/gru.py`) | Same harness, GRU cell (~25k vs 33k params) | Same as LSTM | Lighter recurrent control | Same; fresh slice contradicts headline (GRU ≫ LSTM) |
| 7 | TFT (`tft/ft_model.py`, pytorch-forecasting + Lightning) | Attention + quantile (0.05/0.5/0.95); frame → trailing window | `encoder:24, horizon:6, hidden:32, heads:2, cont:8, dropout:0.1, max_epochs:20, patience:5` | Only native-quantile + `interpret()` model | Encoder too short; config vs `deep_model_summary.json` (168/64/4) mismatch; scored on n=6 vs 7,892 |
| 8 | Hybrid TFT–XGB (`hybrid/tft_xgb.py`) | TFT median + XGB on OOF residuals (`residual_training_mode:oof`, `n_oof_folds:2`) | TFT cfg as above + XGB `n_estimators:200, lr:0.05, d6` | Sole method hook; OOF design correct, `in_sample` warns, intervals honestly labeled uncalibrated | 2 folds thin; timed out in 6B-A; headline +5.5% over TFT contradicted by fresh ablation (R² −2.96) |

Training harness: `src/training/trainer.py:fit_model` (forwards val only if supported); sequence internal scaler + target scaler + early stopping on val loss; TFT Lightning val-loss early stopping + best-checkpoint restore.

## 3. Q1 target set (11 — additions required in Phase 4)

Add `src/models/statistical/{persistence,arima}.py` (naïve ŷ=y_t; seasonal ŷ=y_{t−168}; SARIMA auto-order, logged) and `src/models/advanced/{patchtst,nbeats}.py` (context 168, horizons {1,6,24}, `BaseForecaster` API). Reparametrize ALL to `encoder/seq_len:168`, `output/prediction ∈ {1,6,24}` (three instances each); enable early stopping; Optuna spaces in `configs/hpo_spaces.yaml` (NEW). Hybrid → OOF-5 per horizon + CQR intervals + residual diagnostics.

**Deliberately excluded:** Informer/Autoformer (superseded by PatchTST), Prophet (weak hourly), DeepAR (marginal over TFT quantiles), LLM-forecasters (2026 reproducibility concerns) — state in manuscript.

## 4. Intended use / out-of-scope

Intended: hourly RTO load (and later net-load) point + interval forecasts at 1/6/24h under PROTOCOL equal-n scoring. Out-of-scope: zonal congestion, price forecasting, week-ahead, recursive rollout comparisons.

## 5. Limitations & ethics

Linear-wins headline voids any DL-superiority claim; deep numbers partially estimated/duplicated (see `final_review.json`); no weights persisted (`saved_models/` empty); aggregate-RTO gains mask zonal/equity effects (state in paper); uncalibrated intervals must not drive reserve decisions until CQR (Phase 6).
