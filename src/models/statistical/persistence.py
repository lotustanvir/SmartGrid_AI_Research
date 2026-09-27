"""Naive persistence forecasters (Phase 2 Q1, deterministic).

Persistence: y_hat[t+h] = y[t] (last observed value carried forward).
Seasonal persistence: y_hat[t+h] = y[t+h-168] (same hour last week).

Both operate on 168h history windows. Horizon H in {1,6,24} selects
which step of the carried-forward path is returned. Deterministic:
random_state must be None (no fake stochastic variation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from src.models.base import BaseForecaster
from src.q1.protocol import HISTORY_LENGTH, SEASONAL_PERIOD


class _HistoryBase(BaseForecaster):
    _config_name = "history_base"

    def __init__(
        self,
        horizon: int = 1,
        history_length: int = HISTORY_LENGTH,
        random_state: Optional[int] = None,
    ) -> None:
        if random_state is not None:
            raise ValueError(
                f"{type(self).__name__} is deterministic; "
                f"random_state must be None, got {random_state!r}"
            )
        super().__init__(random_state=None)
        if horizon not in (1, 6, 24):
            raise ValueError(f"horizon must be one of (1,6,24), got {horizon!r}")
        if history_length <= 0:
            raise ValueError("history_length must be positive.")
        self.horizon = int(horizon)
        self.history_length = int(history_length)
        self._y_train: Optional[np.ndarray] = None

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "_HistoryBase":
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    def _store(self, y: ArrayLike) -> np.ndarray:
        arr = np.asarray(y, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("Empty y.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("NaN/Inf in y.")
        self._y_train = arr
        self._is_fitted = True
        return arr

    def fit(self, X, y, **kwargs) -> "_HistoryBase":
        self._store(y)
        return self


class PersistenceForecaster(_HistoryBase):
    """Naive persistence: repeat last observation for H steps.

    Single-origin ``predict`` broadcasts the stored tail value and is
    valid ONLY for one origin (the fitted history's last timestamp).
    For Q1 common-origin evaluation over many origins, use
    :meth:`predict_at_origins` (origin-aligned; no broadcast).
    """

    _config_name = "persistence"

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X)
        n = X.shape[0] if X.ndim >= 1 else 1
        assert self._y_train is not None
        last = float(self._y_train[-1])
        return np.full(n, last, dtype=float)

    def predict_at_origins(
        self,
        series,
        origin_idx,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Origin-aligned persistence forecasts (Q1-safe, no broadcast).

        For each origin position ``i`` in ``series`` (the load at origin
        ``t``), the H1/H6/H24 prediction is ``series[i]`` per the locked
        persistence definition ``y_hat[t+h] = y[t]``.

        Args:
            series: full 1-D load series (observed values only).
            origin_idx: integer positions of the forecast origins.
            horizon: validated against {1,6,24}; defaults to self.horizon.

        Returns one prediction per origin. Raises on out-of-bounds
        origins or NaN/Inf inputs.
        """
        h = self.horizon if horizon is None else int(horizon)
        if h not in (1, 6, 24):
            raise ValueError(f"horizon must be one of (1,6,24), got {h!r}")
        arr = np.asarray(series, dtype=float).ravel()
        idx = np.asarray(list(origin_idx), dtype=int).ravel()
        if idx.size == 0:
            raise ValueError("origin_idx must not be empty.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("NaN/Inf in series.")
        if (idx < 0).any() or (idx >= arr.size).any():
            raise ValueError("origin_idx out of series bounds.")
        return np.asarray([float(arr[i]) for i in idx], dtype=float)


class SeasonalPersistenceForecaster(_HistoryBase):
    """Seasonal naive with 168h period: y_hat[t+h] = y[t+h-168].

    Single-origin ``predict`` uses the stored history tail and is valid
    ONLY for one origin. For Q1 common-origin evaluation over many
    origins, use :meth:`predict_at_origins` (origin-aligned; no
    broadcast, no tail walking).
    """

    _config_name = "seasonal_persistence"

    def __init__(
        self,
        seasonal_period: int = SEASONAL_PERIOD,
        horizon: int = 1,
        history_length: int = HISTORY_LENGTH,
        random_state: Optional[int] = None,
    ) -> None:
        if seasonal_period != SEASONAL_PERIOD:
            raise ValueError(
                f"Q1 seasonal period is locked to {SEASONAL_PERIOD}, "
                f"got {seasonal_period!r}. Report evidence if a correction "
                "is required instead of silently changing it."
            )
        self.seasonal_period = int(seasonal_period)
        super().__init__(
            horizon=horizon, history_length=history_length, random_state=random_state
        )

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X)
        n = X.shape[0] if X.ndim >= 1 else 1
        assert self._y_train is not None
        hist = self._y_train
        # SINGLE-ORIGIN semantics: the fitted history's last timestamp is
        # the origin; step h uses hist[-(168-h+1)]. For n > 1 rows from one
        # origin (e.g. H steps), walk forward through the tail. Do NOT use
        # this for multi-origin batches — use predict_at_origins instead.
        preds = []
        for i in range(n):
            # For row-wise evaluation each test row i is one origin whose
            # seasonal source walks forward through stored history tail.
            # When history is exhausted, fall back to last seasonal value
            # (documented, deterministic, no future use).
            idx = len(hist) - self.seasonal_period + (i % self.seasonal_period)
            idx = min(max(idx, 0), len(hist) - 1)
            preds.append(float(hist[idx]))
        out = np.asarray(preds, dtype=float)
        if n == 1:
            return out
        return out

    def predict_at_origins(
        self,
        series,
        origin_idx,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Origin-aligned seasonal forecasts (Q1-safe, no broadcast).

        For each origin position ``i`` (load at origin ``t``), the
        prediction is ``series[i + h - 168]`` per the locked definition
        ``y_hat[t+h] = y[t+h-168]`` (same hour last week; strictly
        historical since ``h <= 24 < 168``).

        Raises on out-of-bounds seasonal sources (origin too early),
        NaN/Inf inputs, or invalid horizons.
        """
        h = self.horizon if horizon is None else int(horizon)
        if h not in (1, 6, 24):
            raise ValueError(f"horizon must be one of (1,6,24), got {h!r}")
        arr = np.asarray(series, dtype=float).ravel()
        idx = np.asarray(list(origin_idx), dtype=int).ravel()
        if idx.size == 0:
            raise ValueError("origin_idx must not be empty.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("NaN/Inf in series.")
        if (idx < 0).any() or (idx >= arr.size).any():
            raise ValueError("origin_idx out of series bounds.")
        src = idx + h - int(self.seasonal_period)
        if (src < 0).any():
            raise ValueError(
                "Seasonal source predates series start for at least one "
                "origin (need origin+h-168 >= 0)."
            )
        return np.asarray([float(arr[s]) for s in src], dtype=float)
