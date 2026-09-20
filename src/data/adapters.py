"""Model-ready adapters over one engineered frame (Phase 6A).

Single preprocessing path; adapters only reshape/rename:
- tabular: (X, y, feature_cols) for LR/RF/XGB/LGBM (+ scaler fit on train).
- sequenced: (X_2d, y) passthrough for LSTM/GRU (they window internally).
- TFT: long frame (time_idx, group_id, electricity_demand, knowns...).
- Hybrid reuses the TFT frame + residual pipeline (no extra adapter).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.schema import (
    CANONICAL_GROUP,
    CANONICAL_TARGET,
    CANONICAL_TIME,
)

logger = logging.getLogger(__name__)

TFT_TARGET = "electricity_demand"
TFT_KNOWN_MAP = {
    "hour": "hour",
    "day_of_week": "day_of_week",
    "temperature": "temperature",
    "solar": "solar_generation",
    "wind": "wind_generation",
}


def _drop_feature_na(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Drop warm-up rows whose lag/rolling features are not yet defined."""
    before = len(df)
    out = df.dropna(subset=feature_cols + [CANONICAL_TARGET]).reset_index(drop=True)
    info = {"rows_before": before, "rows_after": len(out),
            "warmup_rows_dropped": before - len(out)}
    return out, info


def tabular_ready(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Numpy X/y for tree/linear models (warm-up NaNs dropped, reported)."""
    clean_df, info = _drop_feature_na(df, feature_cols)
    X = clean_df[feature_cols].to_numpy(dtype=float)
    y = clean_df[CANONICAL_TARGET].to_numpy(dtype=float)
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN remains in tabular adapter output.")
    return X, y, list(feature_cols), info


def fit_scaler(
    X_train: np.ndarray,
) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on TRAIN ONLY; return (scaler, X_train_scaled)."""
    scaler = StandardScaler().fit(X_train)
    return scaler, scaler.transform(X_train)


def sequenced_ready(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Ordered (X_2d, y) for LSTM/GRU (same NaN discipline as tabular)."""
    return tabular_ready(df, feature_cols)


def tft_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Long TFT frame: time_idx/group_id/target + known/unknown reals.

    Lags/rolling are unknown-at-forecast-time reals; calendar + weather
    are known. Warm-up NaN rows are dropped and reported.
    """
    known = [c for c in ("hour", "day_of_week", "temperature", "solar", "wind")
             if c in df.columns]
    unknown = [c for c in df.columns
               if c.startswith("lag_") or c.startswith("rolling_")]
    need = known + unknown + [CANONICAL_TARGET]
    clean_df, info = _drop_feature_na(df, [c for c in need if c != CANONICAL_TARGET])
    frames = []
    for name, g in clean_df.groupby(CANONICAL_GROUP, sort=True):
        g = g.sort_values(CANONICAL_TIME).reset_index(drop=True)
        f = pd.DataFrame(
            {
                "time_idx": np.arange(len(g)),
                "group_id": str(name),
                TFT_TARGET: g[CANONICAL_TARGET].to_numpy(dtype=float),
            }
        )
        rename = {c: TFT_KNOWN_MAP.get(c, c) for c in known + unknown}
        for src, dst in rename.items():
            f[dst] = g[src].to_numpy(dtype=float)
        if f.isna().any().any():
            raise ValueError("NaN remains in TFT adapter output.")
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    info["tft_columns"] = list(out.columns)
    info["n_groups"] = int(clean_df[CANONICAL_GROUP].nunique())
    logger.info("TFT frame: %s", out.shape)
    return out, info
