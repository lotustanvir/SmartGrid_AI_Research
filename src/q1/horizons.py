"""Q1 horizon helpers (Phase 2.1).

Tree multi-horizon strategy (EXPLICIT, direct):
  XGBoost/LightGBM are single-output estimators. Q1 does NOT relabel a
  one-step model as H6/H24. Instead one estimator is fitted PER horizon
  (direct H1 / H6 / H24 models) on origin-aligned pairs:
    X_origin = canonical row features at origin t (168h context via lags),
    y        = target load at t+H.
  ``build_direct_pairs`` constructs these pairs from an engineered frame;
  callers fit one estimator per horizon on the returned pairs
  (``split_direct_chrono`` gives chronological train/val/test blocks).

Sequence/Transformer models use native multi-output heads
(output_size/prediction_length = H).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.q1.protocol import HISTORY_LENGTH, HORIZONS


def build_direct_pairs(
    frame: pd.DataFrame,
    feature_names: list[str],
    horizon: int,
    target_col: str = "target",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Origin-aligned (X, y, keys) for one horizon (fail-closed).

    Row i is a valid origin only if features at i are non-NaN and the
    target at i+h exists and is non-NaN. History completeness (168h) is
    guaranteed by the warm-up drop upstream (NaN lags).
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon!r}")
    if target_col not in frame.columns:
        raise ValueError(f"Missing target {target_col!r}")
    work = frame.reset_index(drop=True)
    y_full = work[target_col].to_numpy(dtype=float)
    X_list, y_list, idx = [], [], []
    for i in range(len(work) - horizon):
        row = work.loc[i, feature_names].to_numpy(dtype=float)
        yt = y_full[i + horizon]
        if np.isnan(row).any() or np.isnan(yt):
            continue
        X_list.append(row)
        y_list.append(yt)
        idx.append(i)
    if not X_list:
        raise ValueError(f"No valid direct pairs for H={horizon}.")
    X = np.stack(X_list).astype(float)
    y = np.asarray(y_list, dtype=float)
    keys = work.loc[idx, ["timestamp"]].reset_index(drop=True) if "timestamp" in work.columns else pd.DataFrame({"row": idx})
    keys["horizon"] = int(horizon)
    keys["origin_row"] = idx
    return X, y, keys


def split_direct_chrono(
    X: np.ndarray, y: np.ndarray, train_end: int, val_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test index split for direct pairs."""
    n = len(X)
    if not (0 < train_end < val_end < n):
        raise ValueError("Invalid chrono boundaries.")
    return (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )


def history_context_ok(frame: pd.DataFrame, feature_names: list[str]) -> bool:
    """Check canonical features encode >=168h context (lags/rolling present)."""
    need = {"lag_1", "lag_24", "lag_168", "rolling_mean_168"}
    return need.issubset(set(feature_names))
