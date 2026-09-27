# Forecasting Protocol

## 1. Research Objective

This study evaluates short-term electricity load forecasting models
under a unified leakage-free benchmarking framework using PJM 2020-2025 data.

## 2. Forecasting Problem Definition

Target:

pjm_load_mw

Forecast horizons:

- H1: next 1 hour
- H6: next 6 hours
- H24: next 24 hours

Input context:

Previous 168 hours

Reason:

168 hours captures:

- daily seasonality
- weekly seasonality
- load cycles

## 3. Dataset Split

Chronological split:

Training:

2020-01 to 2023-12

Validation:

2024

Testing:

2025

No random split is allowed.

## 4. Feature Availability Rule

Features are divided into:

### Historical features

Available only from past:

- previous load
- lag features
- rolling statistics

### Known future features

Allowed:

- hour
- day of week
- holiday
- season

### Forecast-dependent features

Require careful handling:

- weather
- renewable generation

No future information leakage is allowed.

## 5. Forecasting Strategy

All models use the same task.

Input:

168 hours

Output:

Direct prediction:

- 1 hour
- 6 hours
- 24 hours

No recursive forecasting comparison.

## 6. Model Evaluation Rule

All models must use:

- identical test origins
- identical timestamps
- identical evaluation samples

Any model with different sample count is rejected.

## 7. Data Leakage Prevention

Required checks:

1. Scaling fitted only on training data

2. Lag features generated using previous timestamps only

3. Rolling features shifted before calculation

4. Validation/test information never used during training

## 8. Experiment Reproducibility

Every experiment must record:

- dataset version
- feature set
- model
- hyperparameters
- seed
- metrics
- environment

## 9. Evaluation Metrics

Primary:

- MAE
- RMSE

Secondary:

- MAPE
- sMAPE
- R²
- MASE

Operational:

- peak error
- ramp error

Uncertainty:

- PICP
- PINAW

## 10. Statistical Validation

Required:

- Friedman test
- Holm corrected Wilcoxon
- Diebold-Mariano test

## 11. Reporting Rules

Every table must include:

- model name
- horizon
- number of samples
- mean performance
- confidence interval

No result can appear in manuscript without experiment ID.
