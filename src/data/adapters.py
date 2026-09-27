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
# LEGACY map (Phase 6A): retained for backward compatibility of stored
# artifacts only. Q1 MUST NOT treat weather as decoder-known; the
# canonical role resolver is calendar-only (see tft_ready below and
# src/q1/tft_path.py).
TFT_KNOWN_MAP = {
    "hour": "hour",
    "day_of_week": "day_of_week",
    "temperature": "temperature",
    "solar": "solar_generation",
    "wind": "wind_generation",
}

# Q1 canonical TFT knowns: calendar/time ONLY. Any weather column listed
# here would leak future information into the decoder and is rejected.
Q1_TFT_KNOWN_CALENDAR = ("hour", "day_of_week")


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

    Q1 SAFETY (Phase 2.1): knowns are calendar/time ONLY
    (``hour``/``day_of_week``/``time_idx``). Weather columns
    (``temperature``/``solar``/``wind`` and any lag/rolling/weather-lag
    numeric) are encoder-only unknowns — they are NEVER returned as
    decoder-known future information. Role assignment delegates to the
    canonical resolver ``src.models.tft.dataset.resolve_roles``; any
    violation raises fail-closed. For the canonical Q1 construction use
    ``src/q1/tft_path.build_q1_tft_frame`` instead.
    """
    from src.models.tft.dataset import resolve_roles

    # Quarantine: refuse frames that already carry contemporaneous raw
    # weather as if it were forecast-known. Lagged/rolling numerics pass
    # through as unknowns below.
    _raw_weather = {
        "temperature",
        "solar",
        "wind",
        "solar_generation",
        "wind_generation",
    }
    _legacy_known_request = [c for c in ("temperature", "solar", "wind") if c in df.columns]
    known = [c for c in Q1_TFT_KNOWN_CALENDAR if c in df.columns]
    unknown = [c for c in df.columns
               if c.startswith("lag_") or c.startswith("rolling_")]
    # Weather (raw or lagged) is encoder-only: never a known.
    for c in list(df.columns):
        if c in _raw_weather or c.startswith(
            ("temp_", "hum_", "ws_", "cloud", "solrad", "wind_", "solar_", "totren")
        ) or c in ("cdh", "hdh") or c.endswith(("_lag_1", "_lag_24", "_lag_168")):
            if c not in unknown and c not in known:
                unknown.append(c)
    if _legacy_known_request:
        logger.warning(
            "tft_ready: legacy weather-as-known request %s quarantined to "
            "encoder-only unknowns; use src/q1/tft_path for Q1.",
            _legacy_known_request,
        )
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
    # Delegate final role validation to the canonical resolver: knowns
    # must be calendar-only; weather must be unknown (encoder-only).
    resolved_known, resolved_unknown, _ = resolve_roles(out)
    _forbidden_known = [
        c for c in resolved_known
        if c not in ("time_idx", "hour", "day_of_week", "dow", "dom",
                     "month", "woy", "weekend")
    ]
    if _forbidden_known:
        raise ValueError(
            f"tft_ready produced non-calendar knowns {_forbidden_known}; "
            "use src/q1/tft_path.build_q1_tft_frame for Q1."
        )
    info["tft_columns"] = list(out.columns)
    info["tft_known"] = list(resolved_known)
    info["tft_unknown"] = list(resolved_unknown)
    info["n_groups"] = int(clean_df[CANONICAL_GROUP].nunique())
    logger.info("TFT frame: %s", out.shape)
    return out, info
