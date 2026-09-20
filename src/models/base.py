"""Common forecaster interface (MODEL-FIRST, Phase 1).

Every forecasting model (tree-based, linear, or sequential) subclasses
:class:`BaseForecaster` and exposes a compatible API::

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores = model.evaluate(X_test, y_test)
    model.save(path)
    ModelClass.load(path)

Tree models and sequence models share the interface but keep their own
internal implementation. Sequence models accept prepared numerical arrays
and must not depend on any dataset-specific column layout.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.evaluation.metrics import evaluate_regression
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, random_state: Optional[int] = 42) -> None:
        if random_state is not None and (
            not isinstance(random_state, int)
            or isinstance(random_state, bool)
            or random_state < 0
        ):
            raise ValueError(
                f"random_state must be a non-negative int or None, got {random_state!r}"
            )
        self.random_state = random_state
        self._is_fitted = False
        if random_state is not None:
            set_seed(random_state)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, X, y, **kwargs) -> "BaseForecaster":
        """Fit the model. Must set ``self._is_fitted = True``."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return point forecasts as a 1-D numpy array."""

    def evaluate(self, X, y, epsilon: float = 1e-8) -> dict:
        """Score point forecasts with the shared regression metric set."""
        preds = self.predict(X)
        return evaluate_regression(y, preds, epsilon=epsilon)

    def predict_interval(self, X, quantiles=(0.05, 0.5, 0.95)) -> dict:
        """Return quantile forecasts keyed by quantile level.

        Classical point models do not implement this yet; the method exists
        so conformal-prediction wrappers can plug in later. Quantile-native
        models (e.g. TFT) override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict_interval yet."
        )

    # ------------------------------------------------------------------
    # Persistence (sklearn-style default; torch models override)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Serialise the fitted estimator via pickle."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved %s to %s", type(self).__name__, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        """Load a model previously saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__} in {path}, got {type(obj).__name__}"
            )
        logger.info("Loaded %s from %s", type(obj).__name__, path)
        return obj

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )

    def _validate_xy(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
            )
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty input arrays.")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values detected in inputs.")
        return X, y

    def _validate_X(self, X, n_features: Optional[int] = None) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.size == 0:
            raise ValueError("Empty input array.")
        if np.isnan(X).any():
            raise ValueError("NaN values detected in inputs.")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(
                f"Feature mismatch: expected {n_features}, got {X.shape[1]}"
            )
        return X

    def __repr__(self) -> str:
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{type(self).__name__}({params})"
