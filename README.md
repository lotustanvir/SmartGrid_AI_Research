# SmartGrid_AI_Research

Model-first smart-grid load forecasting research project.

## Strategy: MODEL-FIRST

1. Build a clean, reproducible foundation (`src/` + `configs/` + `tests/`).
2. Implement all forecasting models behind one common interface.
3. Verify models with synthetic smart-grid-like data only.
4. Connect the real dataset only after models are structurally ready.

No real dataset is connected yet. No paper claims are made from synthetic data.

## Requirements

- Python 3.11
- Install dependencies: `pip install -r requirements.txt`

## Project layout

- `src/` — new clean package (do not mix with legacy folders yet)
  - `src/utils/seed.py` — global RNG seeding
  - `src/evaluation/metrics.py` — MAE, RMSE, safe-MAPE, sMAPE, R²
  - `src/models/base.py` — `BaseForecaster` interface
- `configs/default.yaml` — external configuration (seed, paths, model slots)
- `tests/` — unit tests for foundation + model API
- Legacy (kept until migrated): `models/`, `evaluation/`, `preprocessing/`,
  `utils/`, `notebooks/`, `dataset/`, `results/`, `saved_models/`

## Commands

- Run unit tests: `python -m pytest tests/ -v`
- Fallback without pytest: `python -m unittest discover -s tests -v`
- Smoke tests (Phase 7, after approval): `python -m pytest tests/ -v -k smoke`
  with outputs under `results/smoke_tests/` (never mixed with paper results)

## Research rules (binding)

- Never fabricate accuracy values or confidence intervals.
- No superiority claims until real experiments prove them.
- Use chronological splits for time series, never random splits.
- No future leakage into lag/rolling features.
- Fit scalers on train only; never drop timestamps silently.
- No hardcoded absolute paths; no test-set use during tuning.
