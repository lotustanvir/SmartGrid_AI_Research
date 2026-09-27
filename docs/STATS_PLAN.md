# Stats Plan — Pre-registered Inference (Q1 benchmark)

**Phase 0 — analysis plan locked before any Q1 re-runs. Fixes current bugs listed in §5.**

## 1. Scope

All confirmatory claims on leaderboards (RQ1), ablations A–E (RQ2), horizon shifts (RQ3), hybrid gains (RQ4) across horizons {1,6,24} and slices (full/heat/cold/peak/normal) + 4 rolling origins. Exploratory (SHAP/attention/case studies) labeled as such.

## 2. Estimands & reporting unit

Per-horizon, per-slice: mean MAE/RMSE/MAPE/sMAPE/R²/MASE/skill + peak/ramp errors with 95% bootstrap CIs (10k resamples over origins) across 3 seeds (42,123,2025) for stochastic models; deterministic models single-run + bootstrap CI. Every table row carries model, horizon, aligned n, mean ± CI (PROTOCOL §11).

## 3. Tests (all two-sided unless noted)

1. **Friedman** over (origin × horizon) ranks for 11 models (df=10) — omnibus gate; proceed to pairwise only if rejected at α=0.05.
2. **Wilcoxon signed-rank** on paired per-origin errors for pre-specified pairs (top-3 + hybrid-vs-TFT per horizon), with **Holm step-down correction** across the family; report integer W, adjusted p (never `p=0.0`; use `p<1e-16`), median Δ + Cliff's δ effect size.
3. **Diebold-Mariano (Harvey-Leybourne-Newbold, h-step autocorrelation-robust)** per horizon for top-3 pairs — mandatory at forecasting venues; report DM stat, HLB p, loss differential (squared error).
4. **Nemenyi post-hoc CD diagram** for the 24h leaderboard figure.

## 4. Multiplicity & seeds

Holm over the full pairwise family per horizon (55 pairs if all-pairs, fewer if pre-specified — pre-specify in `experiments/REGISTRY.csv` before running). Seeds fixed (`configs/experiment.yaml`: `[42,123,2025]`); no seed-picking; HPO on validation only, test scored once per best config.

## 5. Current-bug remediation (must-fix before any Q1 number)

- Fractional Wilcoxon W (e.g. 6191831.92) → reimplement (integer rank sums; e.g. `scipy.stats.wilcoxon` + audit test `test_stats_correctness.py`).
- `p=0.0` underflow → log-p / `<1e-16` convention.
- No correction → Holm mandatory; uncorrected p reported only as supplementary.
- n=6-vs-7892 comparisons → banned by PROTOCOL §6 (fail-closed equal-n).
- Identical-across-models values (R²=0.87728 ×4; LSTM≡GRU) → regeneration gate: any duplicated leaderboard row blocks manuscript inclusion.

## 6. Calibration & intervals (reported alongside tests, not tested for "significance")

PICP/PINAW at 50/80/90/95 per horizon + calibration curves + Winkler score; hybrid via conformalized quantile regression (CQR) fit on validation. Exact-PICP 0.9001 artifact reproduced from fitted quantiles or retracted.

## 7. Outputs & traceability

`results/q1/{leaderboard_{1,6,24}.csv, stats/{friedman.json,wilcoxon_holm.csv,dm.csv,cd.png}, ablations/}`; every manuscript number traces to an `EXP-ID` registry row. Deviations from this plan logged in `experiments/REGISTRY.csv` with reason.
