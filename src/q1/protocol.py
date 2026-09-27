"""Frozen Q1 research protocol constants (Phase 2).

Single source of truth for dataset, splits, horizons, history, seeds,
folds, and versions. Import this module instead of hardcoding values
in runners, models, or tests.

Non-negotiable principle: different models may train differently;
they MUST NOT be evaluated differently (same dataset, info boundary,
split, origins, timestamps, horizons, metrics, leakage rules).
"""

from __future__ import annotations

PROTOCOL_VERSION = "Q1_v2_LAG1_001"
FEATURE_VERSION = "Q1_V2_LAG1_001"

Q1_DATASET_PATH = "dataset/processed/pjm_smart_grid_2020_2025_v2.csv"
Q1_DATASET_VERSION = "v2"

HISTORY_LENGTH = 168
HORIZONS = (1, 6, 24)
Q1_SEEDS = (42, 123, 2025)
Q1_OOF_FOLDS = 5
SEASONAL_PERIOD = 168

TRAIN_END = "2024-01-01"  # TRAIN: < 2024-01-01
VAL_END = "2025-01-01"  # VAL: >= 2024-01-01 and < 2025-01-01
TEST_END = "2026-01-01"  # TEST: >= 2025-01-01 and < 2026-01-01

Q1_MODELS = (
    "persistence",
    "seasonal_persistence",
    "sarima",
    "xgboost",
    "lightgbm",
    "lstm",
    "gru",
    "tft",
    "patchtst",
    "nbeats",
    "hybrid",
)

LEGACY_MODELS = ("linear_regression", "random_forest")

# Raw V2 columns present in the canonical CSV (14 columns).
V2_RAW_COLUMNS = (
    "timestamp",
    "pjm_load_mw",
    "wind_generation_mw",
    "solar_generation_mw",
    "temperature",
    "humidity",
    "wind_speed",
    "cloud_cover",
    "solar_radiation",
    "total_renewable",
    "net_load",
    "renewable_penetration",
    "cdh",
    "hdh",
)

# Variables that MUST be lagged (never contemporaneous/future as inputs).
# Lag convention: feature at row t uses values strictly before t
# (shift >= 1). See src/data/features_q1.py.
WEATHER_LAG_VARS = (
    "temperature",
    "humidity",
    "wind_speed",
    "cloud_cover",
    "solar_radiation",
    "wind_generation_mw",
    "solar_generation_mw",
    "total_renewable",
    "cdh",
    "hdh",
)
WEATHER_LAGS = (1, 24, 168)

# Target-derived columns: contain load and MUST be excluded from inputs.
# net_load = load - total_renewable; renewable_penetration = total/load.
# Even lagged versions are excluded to avoid silent target-derived leakage
# and to keep the information set identical across models.
EXCLUDED_TARGET_DERIVED = ("net_load", "renewable_penetration")

ORIGIN_ARTIFACT = "experiments/protocol/test_origins.csv"
