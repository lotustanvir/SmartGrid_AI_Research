"""Regression metrics with zero-safe percentage errors.

Storage convention (CANONICAL, enforced): all percentage-style metrics
(MAPE, sMAPE) are stored as FRACTIONS (0.05 == 5%), consistent with
``sklearn.metrics.mean_absolute_percentage_error``. Do NOT mix percent
(5.0) and fraction (0.05) representations.

MASE denominator (documented): seasonal-naive in-sample MAE with
seasonality m. Q1 default m=168 (weekly) for hourly load; a legacy daily
m=24 applies only to pre-Q1 synthetic verification. Q1 reports MUST pass
``seasonality=168`` explicitly (``src/q1/evaluate.py`` enforces this).
MASE = MAE / seasonal_naive_MAE. Constant targets raise (denominator 0)
— fail closed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "mae",
    "rmse",
    "safe_mape",
    "smape",
    "r2",
    "mase",
    "evaluate_regression",
    "Q1_METRIC_KEYS",
]

Q1_METRIC_KEYS = ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE")


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")
    if yt.size == 0:
        raise ValueError("Empty input arrays.")
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("NaN values detected in inputs.")
    if np.isinf(yt).any() or np.isinf(yp).any():
        raise ValueError("Inf values detected in inputs.")
    return yt, yp


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(mean_absolute_error(yt, yp))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def safe_mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """MAPE with zero-guard: ``mean(|e| / max(|y_true|, eps))``.

    Avoids ``inf``/division-by-zero when targets contain zeros.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(yt), epsilon)
    return float(np.mean(np.abs(yt - yp) / denom))


def smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE as a fraction: ``mean(2|e| / (|yt| + |yp| + eps))``."""
    yt, yp = _as_arrays(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp) + epsilon
    return float(np.mean(2.0 * np.abs(yp - yt) / denom))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination (sklearn convention)."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(r2_score(yt, yp))


def mase(
    y_true,
    y_pred,
    y_train: np.ndarray | None = None,
    seasonality: int = 168,
) -> float:
    """Mean Absolute Scaled Error (seasonal-naive denominator).

    MASE = mean(|y_true - y_pred|) / mean(|y_train[m:] - y_train[:-m]|).

    Args:
        y_true: test actuals.
        y_pred: test predictions (same shape).
        y_train: in-sample training targets for the naive denominator.
            If None, falls back to in-test seasonal naive (documented,
            NOT Q1-canonical; Q1 MUST pass train targets).
        seasonality: naive lag m (Q1 weekly = 168; legacy daily = 24
            only for pre-Q1 synthetic checks — Q1 callers pass 168
            explicitly and ``src/q1/evaluate.py`` enforces it).

    Raises:
        ValueError: on shape mismatch, NaN/Inf, too-short train, or zero
            denominator (constant train series) — fail closed.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    if seasonality <= 0:
        raise ValueError(f"seasonality must be positive, got {seasonality!r}")
    if y_train is None:
        if yt.size <= seasonality:
            raise ValueError("Not enough samples for in-test naive denominator.")
        denom = float(np.mean(np.abs(yt[seasonality:] - yt[:-seasonality])))
    else:
        tr = np.asarray(y_train, dtype=float).ravel()
        if tr.size <= seasonality:
            raise ValueError("y_train too short for seasonality.")
        if np.isnan(tr).any() or np.isinf(tr).any():
            raise ValueError("NaN/Inf in y_train.")
        denom = float(np.mean(np.abs(tr[seasonality:] - tr[:-seasonality])))
    if denom == 0.0 or not np.isfinite(denom):
        raise ValueError("MASE denominator is zero/non-finite (constant train).")
    return float(np.mean(np.abs(yt - yp)) / denom)


def evaluate_regression(
    y_true,
    y_pred,
    epsilon: float = 1e-8,
    y_train: np.ndarray | None = None,
    seasonality: int = 168,
) -> dict:
    """Compute the Q1 regression metric set.

    Returns dict with keys ``MAE``, ``RMSE``, ``MAPE`` (zero-safe fraction),
    ``sMAPE`` (fraction) and ``R2``. ``MASE`` is included ONLY when
    ``y_train`` is provided (Q1-canonical); otherwise omitted (callers must
    not compare MASE across different denominators).
    """
    yt, yp = _as_arrays(y_true, y_pred)
    out = {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": safe_mape(yt, yp, epsilon=epsilon),
        "sMAPE": smape(yt, yp, epsilon=epsilon),
        "R2": r2(yt, yp),
    }
    if y_train is not None:
        out["MASE"] = mase(yt, yp, y_train=y_train, seasonality=seasonality)
    return out
