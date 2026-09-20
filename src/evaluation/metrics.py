"""Regression metrics with zero-safe percentage errors.

All percentage-style metrics are returned as fractions (0.05 == 5%),
consistent with ``sklearn.metrics.mean_absolute_percentage_error``.
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
    "evaluate_regression",
]


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


def evaluate_regression(y_true, y_pred, epsilon: float = 1e-8) -> dict:
    """Compute the standard regression metric set.

    Returns:
        Dict with keys ``MAE``, ``RMSE``, ``MAPE`` (zero-safe),
        ``sMAPE`` and ``R2``.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    return {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": safe_mape(yt, yp, epsilon=epsilon),
        "sMAPE": smape(yt, yp, epsilon=epsilon),
        "R2": r2(yt, yp),
    }
