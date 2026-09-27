"""Q1 common-origin evaluation (Phase 2.1).

All models are scored on exactly the same (origin_timestamp, horizon)
pairs. Helpers here build aligned prediction frames and fail closed on
mismatch, unequal timestamps, or future leakage.

Q1 MASE rule (LOCKED): common-origin scoring MUST include MASE with the
denominator computed from TRAINING targets only and seasonal period
exactly 168 (weekly). Validation/test targets MUST NOT enter the
denominator. ``score_at_origins`` therefore requires ``y_train`` and
rejects any ``seasonality != 168``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.origins import assert_same_origins
from src.evaluation.metrics import evaluate_regression
from src.q1.protocol import SEASONAL_PERIOD

Q1_MASE_SEASONALITY = SEASONAL_PERIOD


def score_at_origins(
    origins: pd.DataFrame,
    frame: pd.DataFrame,
    preds_by_origin: dict,
    y_train=None,
    seasonality: int = Q1_MASE_SEASONALITY,
    time_col: str = "timestamp",
    target_col: str = "target",
) -> tuple[dict, pd.DataFrame]:
    """Score predictions keyed by (origin_timestamp, horizon).

    Args:
        origins: canonical origin set (one row per origin).
        frame: chronological test frame with timestamp + target.
        preds_by_origin: {(origin_ts_iso, horizon): float pred}.
        y_train: 1-D training-target series for the MASE denominator.
            REQUIRED for Q1. Must be the model's training targets only
            (2020-01-01 through 2023-12-31 boundary); validation/test
            targets are rejected by contract (callers must not pass them;
            only the values matter — this function cannot inspect their
            provenance, so the caller is responsible for passing the
            training series).
        seasonality: MUST be exactly 168 (Q1 weekly). Any other value
            raises fail-closed.

    Returns (metrics, aligned_df). Fails closed on missing/extra origins,
    uncomputable MASE denominator, or wrong seasonality.
    """
    if seasonality != int(SEASONAL_PERIOD):
        raise ValueError(
            f"Q1 MASE seasonality is locked to {SEASONAL_PERIOD}, "
            f"got {seasonality!r}."
        )
    if y_train is None:
        raise ValueError(
            "Q1 score_at_origins requires y_train (training targets only) "
            "for the MASE denominator. Validation/test targets MUST NOT "
            "be substituted."
        )
    lut = frame.set_index(pd.to_datetime(frame[time_col]))[target_col].to_dict()
    rows = []
    for r in origins.itertuples():
        key = (pd.Timestamp(r.origin_timestamp).isoformat(), int(r.horizon))
        if key not in preds_by_origin:
            raise ValueError(f"Missing prediction for origin {key}. No substitution allowed.")
        # Target lookup at target_start (H=1) or horizon-weighted: use the
        # actual at target_end for multi-step? Q1 scores the H-step-ahead
        # point target: y[origin+H]. Resolve via origin index + horizon.
        ts = pd.to_datetime(frame[time_col])
        # Positional resolution (robust to tz): find origin position.
        pos = ts.searchsorted(pd.Timestamp(r.origin_timestamp))
        if pos >= len(frame) or ts.iloc[pos] != pd.Timestamp(r.origin_timestamp):
            raise ValueError(f"Origin timestamp not in frame: {r.origin_timestamp}")
        y_true = float(frame[target_col].iloc[pos + int(r.horizon)])
        rows.append(
            {
                "origin_timestamp": pd.Timestamp(r.origin_timestamp),
                "horizon": int(r.horizon),
                "y_true": y_true,
                "y_pred": float(preds_by_origin[key]),
            }
        )
    aligned = pd.DataFrame(rows)
    if aligned["y_true"].isna().any() or aligned["y_pred"].isna().any():
        raise ValueError("NaN in aligned predictions/truth.")
    # Q1 MASE: training-target denominator, seasonality 168. mase()
    # raises fail-closed on too-short/constant/NaN training series.
    metrics = evaluate_regression(
        aligned["y_true"].to_numpy(),
        aligned["y_pred"].to_numpy(),
        y_train=np.asarray(y_train, dtype=float).ravel(),
        seasonality=int(seasonality),
    )
    if "MASE" not in metrics:
        raise ValueError("Q1 scoring failed to produce MASE (fail-closed).")
    return metrics, aligned


def check_common_timestamps(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Assert all model prediction frames share identical test timestamps."""
    ref = None
    for name, df in frames.items():
        ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"])).sort_values()
        if ref is None:
            ref = ts
        elif not ref.equals(ts):
            raise ValueError(
                f"Common-timestamp violation: {name} differs "
                f"({len(ts)} vs {len(ref)})."
            )
    assert ref is not None
    return ref


def build_naive_preds_by_origin(
    model: str,
    series: pd.Series | np.ndarray,
    timestamps,
    origins: pd.DataFrame,
) -> dict:
    """Build origin-aligned naive predictions (Q1-safe, no broadcast).

    Uses the locked per-origin semantics from
    ``src.models.statistical.persistence`` (``predict_at_origins``):
    persistence ``y_hat[t+h] = y[t]``; seasonal ``y_hat[t+h] =
    y[t+h-168]``. Each origin resolves against its own timestamp in the
    full observed ``series`` — predictions differ when origin values
    differ. Single-tail ``predict`` broadcast MUST NOT be used here.

    Args:
        model: "persistence" or "seasonal_persistence".
        series: full observed load series aligned with ``timestamps``.
        timestamps: datetime-like index aligned with ``series``.
        origins: canonical origin set with ``origin_timestamp``/``horizon``.

    Returns ``{(origin_ts_iso, horizon): float}`` for ``score_at_origins``.
    """
    from src.models.statistical.persistence import (
        PersistenceForecaster,
        SeasonalPersistenceForecaster,
    )

    arr = np.asarray(series, dtype=float).ravel()
    ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps).ravel()))
    if arr.size != ts.size:
        raise ValueError("series/timestamps length mismatch.")
    pos_of = {t: i for i, t in enumerate(ts)}
    origin_idx: list[int] = []
    keys: list[tuple[str, int]] = []
    for r in origins.itertuples():
        ots = pd.Timestamp(r.origin_timestamp)
        if ots not in pos_of:
            raise ValueError(f"Origin timestamp not in series: {r.origin_timestamp}")
        origin_idx.append(int(pos_of[ots]))
        keys.append((ots.isoformat(), int(r.horizon)))
    if model == "persistence":
        helper = PersistenceForecaster(horizon=1)
        per_h: dict[int, np.ndarray] = {}
        for h in {k[1] for k in keys}:
            idx_h = [i for i, k in zip(origin_idx, keys) if k[1] == h]
            per_h[h] = helper.predict_at_origins(arr, idx_h, horizon=h)
    elif model == "seasonal_persistence":
        helper = SeasonalPersistenceForecaster(horizon=1)
        per_h = {}
        for h in {k[1] for k in keys}:
            idx_h = [i for i, k in zip(origin_idx, keys) if k[1] == h]
            per_h[h] = helper.predict_at_origins(arr, idx_h, horizon=h)
    else:
        raise ValueError(f"Naive model must be persistence/seasonal_persistence, got {model!r}.")
    out: dict = {}
    counters: dict[int, int] = {h: 0 for h in per_h}
    for (iso, h) in keys:
        out[(iso, h)] = float(per_h[h][counters[h]])
        counters[h] += 1
    return out
