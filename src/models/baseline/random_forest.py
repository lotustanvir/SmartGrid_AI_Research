"""Random forest forecaster (baseline)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class RandomForestForecaster(BaseForecaster):
    """Random forest regressor behind the common forecaster interface.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (None = unlimited).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Features considered per split (None = all).
        n_jobs: Parallel jobs (-1 = all cores).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators!r}")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.model: Optional[RandomForestRegressor] = None
        self.n_features_: Optional[int] = None

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "RandomForestForecaster":
        """Build from ``configs/models.yaml`` (``random_forest`` section)."""
        from src.models.baseline import load_params

        params = load_params("random_forest", path)
        params.update(overrides)
        return cls(**params)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "RandomForestForecaster":
        """Fit the forest on prepared numerical features."""
        X, y = self._validate_xy(X, y)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self._is_fitted = True
        logger.info(
            "Fitted RandomForestForecaster(n_estimators=%d) on %d samples",
            self.n_estimators,
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
    def feature_importances_(self) -> np.ndarray:
        """Normalised impurity-based feature importances."""
        self._check_fitted()
        assert self.model is not None
        return np.asarray(self.model.feature_importances_, dtype=float)

    def save(self, path: str | Path) -> Path:
        """Serialise with joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved RandomForestForecaster to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestForecaster":
        """Load a model saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__} in {path}, got {type(obj).__name__}"
            )
        logger.info("Loaded RandomForestForecaster from %s", path)
        return obj
