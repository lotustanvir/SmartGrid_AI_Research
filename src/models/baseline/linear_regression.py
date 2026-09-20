"""Linear regression forecaster (baseline)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class LinearRegressionForecaster(BaseForecaster):
    """Ordinary least squares with optional Ridge/Lasso regularisation.

    Args:
        fit_intercept: Whether to fit an intercept term.
        regularization: ``None`` (plain OLS), ``"ridge"`` or ``"lasso"``.
        alpha: Regularisation strength used when ``regularization`` is set.
        random_state: Seed for reproducibility (used by Lasso solver).
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if regularization not in (None, "ridge", "lasso"):
            raise ValueError(
                f"regularization must be None, 'ridge' or 'lasso', "
                f"got {regularization!r}"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha!r}")
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.alpha = alpha
        self.model: LinearRegression | Ridge | Lasso | None = None
        self.n_features_: Optional[int] = None

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "LinearRegressionForecaster":
        """Build from ``configs/models.yaml`` (``linear_regression`` section)."""
        from src.models.baseline import load_params

        params = load_params("linear_regression", path)
        params.update(overrides)
        return cls(**params)

    def _build(self) -> LinearRegression | Ridge | Lasso:
        if self.regularization == "ridge":
            return Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        if self.regularization == "lasso":
            return Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            )
        return LinearRegression(fit_intercept=self.fit_intercept)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "LinearRegressionForecaster":
        """Fit OLS (or Ridge/Lasso) on prepared numerical features."""
        X, y = self._validate_xy(X, y)
        self.model = self._build()
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self._is_fitted = True
        logger.info(
            "Fitted LinearRegressionForecaster(regularization=%s) on %d samples",
            self.regularization,
            X.shape[0],
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return point forecasts as a 1-D array."""
        self._check_fitted()
        X = self._validate_X(X, n_features=self.n_features_)
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=float).ravel()

    @property
    def coefficients_(self) -> np.ndarray:
        """Fitted coefficients (one per feature)."""
        self._check_fitted()
        assert self.model is not None
        return np.asarray(self.model.coef_, dtype=float)

    @property
    def intercept_(self) -> float:
        """Fitted intercept term."""
        self._check_fitted()
        assert self.model is not None
        return float(self.model.intercept_)

    def save(self, path: str | Path) -> Path:
        """Serialise with joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved LinearRegressionForecaster to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LinearRegressionForecaster":
        """Load a model saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__} in {path}, got {type(obj).__name__}"
            )
        logger.info("Loaded LinearRegressionForecaster from %s", path)
        return obj
