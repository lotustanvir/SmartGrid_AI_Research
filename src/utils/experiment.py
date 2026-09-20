"""Synthetic smart-grid-like data and experiment helpers (Phase 2B).

Everything here is for SOFTWARE VERIFICATION ONLY. Synthetic outputs must
never be treated as research results or mixed with future paper results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "electricity_demand"
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "temperature",
    "solar_generation",
    "wind_generation",
    "previous_load",
]


def generate_synthetic_grid_data(
    n_samples: int = 2160, seed: int = 42
) -> pd.DataFrame:
    """Generate time-ordered synthetic load data (trend + seasonality + noise).

    Features: hour, day_of_week, temperature, solar_generation,
    wind_generation, previous_load (lag-1 of demand, no future leakage).
    Target: electricity_demand (strictly positive).

    The first row is dropped because ``previous_load`` is undefined at t=0;
    the index is reset so the frame stays time-ordered without gaps.
    """
    if n_samples < 48:
        raise ValueError(f"n_samples must be >= 48, got {n_samples!r}")
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float)

    hour = (t % 24).astype(int)
    day_of_week = ((t // 24) % 7).astype(int)

    trend = 0.02 * t
    daily = 18.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    weekly = np.where(day_of_week >= 5, -12.0, 0.0)

    temperature = (
        20.0
        + 8.0 * np.sin(2.0 * np.pi * (hour - 15.0) / 24.0)
        + 0.004 * t
        + rng.normal(0.0, 1.0, n_samples)
    )
    solar_generation = np.maximum(
        0.0, np.sin(np.pi * (hour - 6.0) / 12.0)
    ) ** 1.5 * 30.0
    solar_generation = solar_generation * (1.0 + rng.normal(0.0, 0.05, n_samples))
    solar_generation = np.maximum(0.0, solar_generation)
    wind_generation = np.maximum(
        0.0, 12.0 + 6.0 * np.sin(2.0 * np.pi * t / 37.0) + rng.normal(0.0, 2.5, n_samples)
    )

    base = 120.0 + trend + daily + weekly
    demand = (
        base
        + 1.5 * (temperature - 20.0)
        - 0.4 * solar_generation
        - 0.3 * wind_generation
        + rng.normal(0.0, 3.0, n_samples)
    )
    demand = np.maximum(demand, 5.0)

    df = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "temperature": temperature,
            "solar_generation": solar_generation,
            "wind_generation": wind_generation,
            TARGET_COL: demand,
        }
    )
    df["previous_load"] = df[TARGET_COL].shift(1)
    df = df.iloc[1:].reset_index(drop=True)
    logger.info("Generated synthetic grid data: %s", df.shape)
    return df[FEATURE_COLS + [TARGET_COL]]


def chronological_split(
    X,
    y,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple:
    """Split time-ordered arrays chronologically (no shuffling).

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("Ratios must satisfy 0 < train < 1 and 0 <= val < 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different sample counts.")
    n = X.shape[0]
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    if i_train < 1 or i_val <= i_train or n - i_val < 1:
        raise ValueError("Not enough samples for the requested split.")
    return (
        X[:i_train],
        X[i_train:i_val],
        X[i_val:],
        y[:i_train],
        y[i_train:i_val],
        y[i_val:],
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if missing."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results_csv(
    results: pd.DataFrame, path: str | Path = "results/baseline/results.csv"
) -> Path:
    """Save the model comparison table with fixed column order."""
    cols = ["Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"]
    missing = [c for c in cols if c not in results.columns]
    if missing:
        raise ValueError(f"Results missing columns: {missing}")
    path = Path(path)
    ensure_dir(path.parent)
    results[cols].to_csv(path, index=False)
    logger.info("Saved results CSV to %s", path)
    return path
