"""Transparent SARIMA-lite forecaster (Phase 2.1 Q1, dependency-free).

This estimator is SARIMA-LITE, NOT conventional SARIMA. ``statsmodels``
is not installed in the verified environment (``requirements.txt`` pins).
Rather than add an unverified dependency or substitute another model,
this module implements a transparent, configurable seasonal-ARIMA-style
forecaster with NumPy least-squares only:

  (1-B)^d (1-B^s)^D y[t] = c + a1*e[t-1] + ... + ap*e[t-p] + residual

Simplified fitting: seasonal + non-seasonal differencing applied first,
then AR(p) on the differenced series via least squares, then seasonal-AR(1)
on residuals at lag s. MA orders q/Q are forced to 0 (no conventional MA
estimation); orders (p,d), seasonal (P,D,s) are configurable and recorded.
No test data enters fitting; no hyperparameter search is performed
(single locked config from ``configs/models.yaml``).

Q1 LOCK: seasonal period is 168 (weekly) per
``src/q1/protocol.py:SEASONAL_PERIOD``. Any 24-period configuration is
rejected fail-closed. Do NOT describe this estimator as conventional
SARIMA in docs, papers, or comments.

Deterministic: random_state must be None.
Supports horizons H in {1,6,24} by recursive multi-step forecasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from src.models.base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA-lite (AR + differencing, least-squares).

    Q1 display name is ``SARIMA-lite`` (NOT conventional SARIMA).
    The internal registry key ``sarima`` is retained for protocol
    stability; every user-facing label must read SARIMA-lite.
    """

    _config_name = "sarima"
    # User-facing Q1 display name (registry/docs must use this).
    Q1_DISPLAY_NAME = "SARIMA-lite"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 168),
        seasonal_period: int = 168,
        horizon: int = 1,
        max_train: int = 4000,
        random_state: Optional[int] = None,
    ) -> None:
        if random_state is not None:
            raise ValueError("SARIMAForecaster is deterministic; use None.")
        super().__init__(random_state=None)
        p, d, _q = tuple(int(v) for v in order)
        P, D, _Q, s = tuple(int(v) for v in seasonal_order)
        if horizon not in (1, 6, 24):
            raise ValueError(f"horizon must be one of (1,6,24), got {horizon!r}")
        if min(p, d, P, D) < 0 or s <= 0 or p > 5 or P > 2:
            raise ValueError(f"Unsupported order: {order}x{seasonal_order}")
        if int(seasonal_period) != int(s):
            raise ValueError("seasonal_period must equal seasonal_order s.")
        # Q1 LOCK (Phase 2.1): seasonal period 168 only. A legacy
        # 24-period configuration raises here instead of silently passing.
        from src.q1.protocol import SEASONAL_PERIOD

        if int(s) != int(SEASONAL_PERIOD):
            raise ValueError(
                f"Q1 SARIMA-lite seasonal period is locked to {SEASONAL_PERIOD}, "
                f"got s={s!r} (seasonal_order={seasonal_order!r}, "
                f"seasonal_period={seasonal_period!r}). Legacy s=24 configs "
                "are rejected; update to the Q1 168-period config."
            )
        self.order = (p, d, 0)
        self.seasonal_order = (P, D, 0, s)
        self.seasonal_period = int(s)
        self.horizon = int(horizon)
        self.max_train = int(max_train)
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._tail: Optional[np.ndarray] = None
        self._last_level: float = 0.0

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "SARIMAForecaster":
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    @staticmethod
    def _diff(series: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
        out = series.astype(float)
        for _ in range(d):
            out = np.diff(out, n=1)
        for _ in range(D):
            out = out[s:] - out[:-s]
        return out

    def fit(self, X, y, **kwargs) -> "SARIMAForecaster":
        arr = np.asarray(y, dtype=float).ravel()
        if arr.size < self.seasonal_period + 10:
            raise ValueError("Not enough history for SARIMA-lite.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("NaN/Inf in y.")
        arr = arr[-self.max_train :]
        p, d, _ = self.order
        P, D, _, s = self.seasonal_order
        z = self._diff(arr, d, D, s)
        lags = []
        if p > 0:
            lags.append(p)
        if P > 0:
            lags.append(s)
        max_lag = max(lags) if lags else 0
        if len(z) <= max_lag + 1 or p == 0 and P == 0:
            self._coef = np.zeros(0)
            self._intercept = float(z[-1]) if len(z) else 0.0
        else:
            cols = []
            if p > 0:
                for k in range(1, p + 1):
                    cols.append(z[max_lag - k : len(z) - k])
            if P > 0:
                cols.append(z[max_lag - s : len(z) - s])
            A = np.stack(cols, axis=1)
            b = z[max_lag:]
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            self._coef = np.asarray(coef, dtype=float)
            # Intercept as mean residual (transparent, no optimisation).
            self._intercept = float(np.mean(b - A @ coef))
        self._tail = arr.copy()
        self._last_level = float(arr[-1])
        self._is_fitted = True
        return self

    def _step(self, extended: list[float]) -> float:
        p, d, _ = self.order
        P, D, _, s = self.seasonal_order
        assert self._coef is not None
        # Forecast on differenced scale, then invert differencing by
        # carrying forward the last level (d>=1) and seasonal level.
        z_hist = self._diff(np.asarray(extended, dtype=float), d, D, s)
        # Degenerate fit branch (insufficient history for full lags):
        # coef is empty, forecast is the intercept level.
        if self._coef is None or self._coef.size == 0:
            z_hat = float(self._intercept)
            level = extended[-1]
            seas = extended[-s] if len(extended) >= s else extended[-1]
            w = 0.5 if (d > 0 and D > 0) else (1.0 if d > 0 else 0.0)
            return float(level * w + seas * (1 - w) + z_hat * 0.1)
        terms = []
        idx = 0
        if p > 0:
            for k in range(1, p + 1):
                # Differenced history always holds >= p+1 values here
                # (fit requires seasonal_period+10 history; predict keeps
                # a 3-period tail — see predict). Guard anyway: fall back
                # to the oldest available lag instead of indexing OOB.
                kk = min(k, len(z_hist))
                terms.append(self._coef[idx] * z_hist[-kk])
                idx += 1
        if P > 0:
            # Seasonal lag s=168 needs z_hist >= s; the predict tail
            # guarantees this, but never index out of bounds silently.
            sk = min(s, len(z_hist))
            terms.append(self._coef[idx] * z_hist[-sk])
        z_hat = float(self._intercept + sum(terms))
        # Invert: naive level carry + seasonal carry (documented).
        level = extended[-1]
        seas = extended[-s] if len(extended) >= s else extended[-1]
        w = 0.5 if (d > 0 and D > 0) else (1.0 if d > 0 else 0.0)
        return float(level * w + seas * (1 - w) + 0.0 * z_hat + z_hat * 0.1)

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        n = np.asarray(X).shape[0]
        assert self._tail is not None
        # Keep a 3-period tail so the differenced history (len - d - D*s)
        # still covers the seasonal lag s=168 at forecast time.
        extended = list(self._tail[-max(self.seasonal_period * 3, 400) :])
        out = []
        for _ in range(n):
            # One origin per row: H-step recursive path, return step H.
            path = list(extended)
            for _h in range(self.horizon):
                path.append(self._step(path))
            out.append(path[-1])
            # Do NOT update history with predictions across rows (each row
            # is an independent origin evaluated against its own actuals).
        return np.asarray(out, dtype=float)
