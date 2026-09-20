"""Leakage-safe feature engineering (Phase 6A).

All features are built group-wise on chronological data:
- Lags use strictly past values (shift).
- Rolling stats use shift(1) first: window at time t covers t-W..t-1 only.
- Calendar features derive from the timestamp itself (always known).
Nothing from the future enters any row.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.schema import (
    CANONICAL_GROUP,
    CANONICAL_TARGET,
    CANONICAL_TIME,
    CYCLICAL_FEATURES,
    CALENDAR_FEATURES,
    FeatureConfig,
    lag_name,
    rolling_names,
)

logger = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame, cfg: FeatureConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Add features; return (frame, feature_names). Rows keep chrono order."""
    work = df.sort_values([CANONICAL_GROUP, CANONICAL_TIME]).reset_index(drop=True)
    made: list[str] = []

    if cfg.calendar:
        ts = work[CANONICAL_TIME]
        work["hour"] = ts.dt.hour
        work["day_of_week"] = ts.dt.dayofweek
        work["day_of_month"] = ts.dt.day
        work["month"] = ts.dt.month
        work["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        work["is_weekend"] = (work["day_of_week"] >= 5).astype(int)
        made.extend(CALENDAR_FEATURES)

    if cfg.cyclical:
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24.0)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24.0)
        work["dow_sin"] = np.sin(2 * np.pi * work["day_of_week"] / 7.0)
        work["dow_cos"] = np.cos(2 * np.pi * work["day_of_week"] / 7.0)
        work["month_sin"] = np.sin(2 * np.pi * (work["month"] - 1) / 12.0)
        work["month_cos"] = np.cos(2 * np.pi * (work["month"] - 1) / 12.0)
        made.extend(CYCLICAL_FEATURES)

    grouped = work.groupby(CANONICAL_GROUP, sort=False)[CANONICAL_TARGET]
    for lag in cfg.lags:
        if lag <= 0:
            raise ValueError(f"lags must be positive, got {lag}")
        col = lag_name(lag)
        work[col] = grouped.shift(lag).to_numpy()
        made.append(col)

    for window in cfg.rolling_windows:
        if window <= 1:
            raise ValueError(f"rolling windows must be > 1, got {window}")
        mean_col, std_col = rolling_names(window)
        # Per-group rolling over strictly-past values (shift(1) first).
        work[mean_col] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        ).to_numpy()
        work[std_col] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).std()
        ).to_numpy()
        made.extend([mean_col, std_col])

    logger.info("Engineered %d features", len(made))
    return work, made
