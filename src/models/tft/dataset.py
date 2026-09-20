"""TFT dataset helpers: synthetic frames + TimeSeriesDataSet wiring."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIME_IDX = "time_idx"
GROUP_ID = "group_id"
TARGET = "electricity_demand"

RICH_KNOWN_REALS = [
    "time_idx",
    "hour",
    "day_of_week",
    "temperature",
    "solar_generation",
    "wind_generation",
]
RICH_UNKNOWN_REALS = ["electricity_demand"]
RICH_STATIC_CATS = ["group_id"]


def build_synthetic_tft_frame(
    n_groups: int = 3, n_steps: int = 240, seed: int = 42
) -> pd.DataFrame:
    """Synthetic multi-series frame compatible with ``TimeSeriesDataSet``.

    Columns: time_idx, group_id, electricity_demand, temperature,
    solar_generation, wind_generation, hour, day_of_week.
    Verification only — never a research result.
    """
    if n_groups < 1 or n_steps < 48:
        raise ValueError("Need n_groups >= 1 and n_steps >= 48.")
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        t = np.arange(n_steps, dtype=float)
        hour = (t % 24).astype(int)
        dow = ((t // 24) % 7).astype(int)
        level = 100.0 + 15.0 * g
        temp = 20.0 + 8.0 * np.sin(2 * np.pi * (hour - 15.0) / 24.0)
        temp = temp + rng.normal(0.0, 1.0, n_steps)
        solar = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0)) ** 1.5 * 30.0
        wind = np.maximum(0.0, 12.0 + rng.normal(0.0, 3.0, n_steps))
        demand = (
            level
            + 0.02 * t
            + 18.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
            + np.where(dow >= 5, -12.0, 0.0)
            + 1.5 * (temp - 20.0)
            - 0.4 * solar
            - 0.3 * wind
            + rng.normal(0.0, 3.0, n_steps)
        )
        frames.append(
            pd.DataFrame(
                {
                    TIME_IDX: t.astype(int),
                    GROUP_ID: f"group_{g}",
                    TARGET: np.maximum(demand, 5.0),
                    "temperature": temp,
                    "solar_generation": np.maximum(solar, 0.0),
                    "wind_generation": wind,
                    "hour": hour,
                    "day_of_week": dow,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    logger.info("Built synthetic TFT frame: %s", df.shape)
    return df


def frame_from_arrays(X, y=None, time_start: int = 0) -> pd.DataFrame:
    """Convert 2-D time-ordered features (+ optional target) to TFT frame.

    Single ``group_id``; calendar knowns (hour/day_of_week) are derived from
    row order; passed features become time-varying unknown reals ``f0..``.
    ``time_start`` offsets ``time_idx`` so consecutive blocks (e.g. train
    then validation) keep globally increasing timestamps.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError(f"X must be 2-D with >= 3 rows, got {X.shape}")
    if np.isnan(X).any():
        raise ValueError("NaN values detected in inputs.")
    n = X.shape[0]
    idx = np.arange(time_start, time_start + n)
    data: dict[str, np.ndarray] = {
        TIME_IDX: idx,
        GROUP_ID: np.array(["single"] * n),
        "hour": idx % 24,
        "day_of_week": (idx // 24) % 7,
    }
    for j in range(X.shape[1]):
        data[f"f{j}"] = X[:, j]
    if y is None:
        data[TARGET] = np.full(n, np.nan)
    else:
        yarr = np.asarray(y, dtype=float).ravel()
        if yarr.shape[0] != n:
            raise ValueError("X and y have different lengths.")
        if np.isnan(yarr).any():
            raise ValueError("NaN values detected in y.")
        data[TARGET] = yarr
    return pd.DataFrame(data)


def coerce_frame(X: Union[pd.DataFrame, np.ndarray], y=None) -> pd.DataFrame:
    """Accept a rich TFT DataFrame or ``(X_2d, y_1d)`` arrays."""
    if isinstance(X, pd.DataFrame):
        if y is not None:
            raise ValueError("y must be None when X is a DataFrame (target in frame).")
        return X.copy()
    if y is None:
        raise ValueError("y is required when X is a numpy array.")
    return frame_from_arrays(X, y)


def validate_frame(df: pd.DataFrame, allow_nan_target: bool = False) -> pd.DataFrame:
    """Check required columns, dtypes and NaNs; return time-sorted copy."""
    required = {TIME_IDX, GROUP_ID, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Frame missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty frame.")
    df = df.copy()
    df[TIME_IDX] = pd.to_numeric(df[TIME_IDX], errors="raise").astype(int)
    check = df.drop(columns=[TARGET]) if allow_nan_target else df
    if check.isna().any().any():
        raise ValueError("NaN values detected in frame covariates.")
    if not allow_nan_target and df[TARGET].isna().any():
        raise ValueError("NaN values detected in target.")
    return df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)


def resolve_roles(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Map frame columns to (known_reals, unknown_reals, static_cats)."""
    if all(c in df.columns for c in RICH_KNOWN_REALS):
        known = [c for c in RICH_KNOWN_REALS if c in df.columns]
        unknown = [c for c in RICH_UNKNOWN_REALS if c in df.columns]
        static = [c for c in RICH_STATIC_CATS if c in df.columns]
    else:
        known = [c for c in (TIME_IDX, "hour", "day_of_week") if c in df.columns]
        unknown = [
            c
            for c in df.columns
            if c not in known + [GROUP_ID, TARGET]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        static = [GROUP_ID] if GROUP_ID in df.columns else []
    if TARGET not in unknown:
        unknown = unknown + [TARGET]
    return known, unknown, static


def split_datasets(
    df: pd.DataFrame,
    encoder_length: int,
    prediction_length: int,
    val_size: int,
    batch_size: int,
) -> dict:
    """Build training/validation TimeSeriesDataSets + loaders (chrono split)."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    if encoder_length <= 0 or prediction_length <= 0:
        raise ValueError("encoder/prediction lengths must be positive.")
    max_time = int(df[TIME_IDX].max())
    cutoff = max_time - val_size
    if cutoff <= encoder_length + prediction_length:
        raise ValueError("val_size leaves too little training history.")
    known, unknown, static = resolve_roles(df)
    train_df = df[df[TIME_IDX] <= cutoff]
    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_ID],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=static,
        time_varying_known_reals=known,
        time_varying_unknown_reals=unknown,
        target_normalizer=GroupNormalizer(groups=[GROUP_ID]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    holdout_df = df[df[TIME_IDX] > cutoff - encoder_length]
    validation = TimeSeriesDataSet.from_dataset(
        training, holdout_df, predict=False, stop_randomization=True
    )
    if len(validation) == 0:
        raise ValueError("Validation split produced no windows.")
    return {
        "training": training,
        "validation": validation,
        "train_loader": training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        ),
        "val_loader": validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        ),
    }


def fill_decoder_placeholders(df: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """Fill trailing-horizon NaN targets with the last observed value.

    The TFT decoder never consumes future targets as inputs, so the fill
    value cannot affect predictions; filled values must never be scored
    (score against separately held actuals instead). NaNs anywhere else,
    or a group with no target history at all, raise ``ValueError``.
    """
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")
    out = []
    for _, g in df.groupby(GROUP_ID, sort=True):
        g = g.sort_values(TIME_IDX).copy()
        tgt = g[TARGET].to_numpy(dtype=float).copy()
        n = len(g)
        h = min(prediction_length, n)
        is_nan = np.isnan(tgt)
        if h < n and is_nan[:-h].any():
            raise ValueError("NaN targets found outside the trailing horizon.")
        valid = tgt[~is_nan]
        if valid.size == 0:
            raise ValueError(
                "Group has no target history; pass observed targets via "
                "forecast(X, y) instead of predict(X)."
            )
        tgt[is_nan] = valid[-1]
        g[TARGET] = tgt
        out.append(g)
    return pd.concat(out, ignore_index=True)


def predict_loader(training, df: pd.DataFrame, batch_size: int, prediction_length: int):
    """One trailing window per group for inference (future may be NaN)."""
    from pytorch_forecasting import TimeSeriesDataSet

    filled = fill_decoder_placeholders(df, prediction_length)
    ds = TimeSeriesDataSet.from_dataset(
        training, filled, predict=True, stop_randomization=True
    )
    if len(ds) == 0:
        raise ValueError("Prediction frame too short for encoder + horizon.")
    return ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0), ds
