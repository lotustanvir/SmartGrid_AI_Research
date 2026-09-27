"""Canonical Q1 V2 feature schema (Phase 2).

Dataset: ``dataset/processed/pjm_smart_grid_2020_2025_v2.csv`` (14 cols).

WEATHER INFORMATION RULE (LOCKED, primary experiment = LAGGED WEATHER ONLY):
  For a forecast origin t (last observed timestamp), allowed weather
  inputs use values strictly before-or-at the origin history, NEVER
  contemporaneous/future values of the target horizon:
    - Row t features use weather values with shift >= 1
      (weather[t-1], weather[t-24], weather[t-168]).
    - Raw contemporaneous weather columns are NOT model inputs.
    - Sequence encoders may use observed weather at steps s <= origin t;
      decoder future weather (t+1..t+H) is NEVER supplied (TFT knowns =
      calendar only).
  This is enforced by constructing ONLY ``<var>_lag_<k>`` columns and
  excluding raw weather cols from ``canonical_feature_names()``.

TARGET-DERIVED LEAKAGE ANALYSIS (explicit):
  - ``net_load = pjm_load_mw - total_renewable`` contains the target load.
    Using net_load[t] to predict load[t] (or load[t+H] with net_load[t+H])
    leaks the target. Even ``net_load[t-k]`` is a deterministic function
    of lagged load (already represented by load lags) and is EXCLUDED to
    keep the information set identical and auditable.
  - ``renewable_penetration = total_renewable / pjm_load_mw`` contains the
    target in the denominator. EXCLUDED for the same reason.
  - ``total_renewable = wind + solar`` contains NO load. Included (lagged).
  - ``cdh/hdh`` are temperature-derived (no load). Included (lagged).

One authoritative schema: ``canonical_feature_names()`` + ``Q1_FEATURE_VERSION``
+ ``feature_schema_hash()``. All models consume the same information set;
only tensor formatting may differ. Model-specific content differences are
NOT allowed unless documented.

Rolling features are strictly backward-looking: window at row t covers
t-W..t-1 (shift(1) first). No forward-fill / interpolation across the
forecast boundary is performed here; cleaning may only interpolate
exogenous gaps strictly within the past (limit 6, time method) and never
across origin/horizon.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from src.q1.protocol import (
    EXCLUDED_TARGET_DERIVED,
    FEATURE_VERSION,
    WEATHER_LAG_VARS,
    WEATHER_LAGS,
)

logger = logging.getLogger(__name__)

Q1_FEATURE_VERSION = FEATURE_VERSION

CALENDAR_COLS = ["hour", "dow", "dom", "month", "woy", "weekend"]
CYCLICAL_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
LOAD_LAG_COLS = ["lag_1", "lag_24", "lag_168"]
LOAD_ROLL_COLS = [
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
]

# Sanitised short names for weather vars (provider col -> feature stem).
WEATHER_STEMS = {
    "temperature": "temp",
    "humidity": "hum",
    "wind_speed": "ws",
    "cloud_cover": "cloud",
    "solar_radiation": "solrad",
    "wind_generation_mw": "wind",
    "solar_generation_mw": "solar",
    "total_renewable": "totren",
    "cdh": "cdh",
    "hdh": "hdh",
}


def weather_feature_names() -> list[str]:
    out: list[str] = []
    for var in WEATHER_LAG_VARS:
        stem = WEATHER_STEMS[var]
        for k in WEATHER_LAGS:
            out.append(f"{stem}_lag_{k}")
    return out


def canonical_feature_names() -> list[str]:
    """Authoritative ordered Q1 input feature list (no target, no raw weather)."""
    return (
        list(CALENDAR_COLS)
        + list(CYCLICAL_COLS)
        + list(LOAD_LAG_COLS)
        + list(LOAD_ROLL_COLS)
        + weather_feature_names()
    )


def feature_schema_hash() -> str:
    payload = Q1_FEATURE_VERSION + "\n" + "\n".join(canonical_feature_names())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_v2_columns(df: pd.DataFrame) -> None:
    need = [
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
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"V2 frame missing columns: {missing}")
    forbidden = [c for c in EXCLUDED_TARGET_DERIVED if c in canonical_feature_names()]
    if forbidden:
        raise ValueError(f"Target-derived leak in schema: {forbidden}")


def build_q1_features(raw_v2: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build canonical Q1 frame from raw V2 (leakage-safe, lagged weather).

    Returns (frame, feature_names) where frame has ``timestamp`` (datetime),
    ``target`` (float load), ``group_id='single'`` + canonical features.
    Rows keep chronological order. Warm-up NaNs remain (callers drop/report).
    """
    _require_v2_columns(raw_v2)
    work = raw_v2.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    if work["timestamp"].isna().any():
        raise ValueError("NaT timestamps in V2 input.")
    work = work.sort_values("timestamp").reset_index(drop=True)
    if work["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in V2 input.")
    work["target"] = pd.to_numeric(work["pjm_load_mw"], errors="coerce")
    work["group_id"] = "single"
    ts = work["timestamp"]
    work["hour"] = ts.dt.hour.astype(int)
    work["dow"] = ts.dt.dayofweek.astype(int)
    work["dom"] = ts.dt.day.astype(int)
    work["month"] = ts.dt.month.astype(int)
    work["woy"] = ts.dt.isocalendar().week.astype(int)
    work["weekend"] = (work["dow"] >= 5).astype(int)
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24.0)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24.0)
    work["dow_sin"] = np.sin(2 * np.pi * work["dow"] / 7.0)
    work["dow_cos"] = np.cos(2 * np.pi * work["dow"] / 7.0)
    work["month_sin"] = np.sin(2 * np.pi * (work["month"] - 1) / 12.0)
    work["month_cos"] = np.cos(2 * np.pi * (work["month"] - 1) / 12.0)
    load = work["target"]
    for lag in (1, 24, 168):
        work[f"lag_{lag}"] = load.shift(lag).to_numpy()
    for window in (24, 168):
        work[f"rolling_mean_{window}"] = (
            load.shift(1).rolling(window, min_periods=window).mean().to_numpy()
        )
        work[f"rolling_std_{window}"] = (
            load.shift(1).rolling(window, min_periods=window).std().to_numpy()
        )
    # Lagged weather: strictly past values only (shift >= 1).
    for var in WEATHER_LAG_VARS:
        stem = WEATHER_STEMS[var]
        series = pd.to_numeric(work[var], errors="coerce")
        for k in WEATHER_LAGS:
            work[f"{stem}_lag_{k}"] = series.shift(k).to_numpy()
    feats = canonical_feature_names()
    # Guard: raw contemporaneous weather must not enter as features.
    raw_weather = [
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
        "net_load",
        "renewable_penetration",
        "pjm_load_mw",
    ]
    overlap = [c for c in raw_weather if c in feats]
    if overlap:
        raise ValueError(f"Raw weather/target leak in schema: {overlap}")
    logger.info(
        "Q1 features built: %d rows, %d features (version %s)",
        len(work),
        len(feats),
        Q1_FEATURE_VERSION,
    )
    return work, feats


def drop_warmup(
    df: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Drop rows with NaN engineered features (warm-up), report counts."""
    before = len(df)
    out = df.dropna(subset=feature_names + ["target"]).reset_index(drop=True)
    info = {
        "rows_before": before,
        "rows_after": len(out),
        "warmup_rows_dropped": before - len(out),
        "feature_version": Q1_FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "n_features": len(feature_names),
    }
    return out, info
