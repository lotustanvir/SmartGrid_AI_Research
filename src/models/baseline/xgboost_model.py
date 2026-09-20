"""XGBoost forecaster (baseline).

Refactored from legacy ``models/xgboost_model.py``: hardcoded parameters
are now configurable (constructor + ``configs/models.yaml``), validation
data with early stopping is supported, and evaluation uses the shared
zero-safe metric set from :mod:`src.evaluation.metrics`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike
from xgboost import XGBRegressor

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """Gradient-boosted trees behind the common forecaster interface.

    Args:
        n_estimators: Maximum boosting rounds.
        learning_rate: Step size shrinkage.
        max_depth: Maximum tree depth.
        subsample: Row subsample ratio per tree.
        colsample_bytree: Column subsample ratio per tree.
        reg_alpha: L1 regularisation term.
        reg_lambda: L2 regularisation term.
        n_jobs: Parallel threads (-1 uses all cores via nthread).
        early_stopping_rounds: Stop if the validation metric does not
            improve for this many rounds. Requires validation data in
            :meth:`fit`; ``None`` disables early stopping.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators!r}")
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            raise ValueError(
                "early_stopping_rounds must be positive or None, "
                f"got {early_stopping_rounds!r}"
            )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[XGBRegressor] = None
        self.n_features_: Optional[int] = None
        self.best_iteration_: Optional[int] = None

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "XGBoostForecaster":
        """Build from ``configs/models.yaml`` (``xgboost`` section)."""
        from src.models.baseline import load_params

        params = load_params("xgboost", path)
        params.update(overrides)
        return cls(**params)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        **kwargs,
    ) -> "XGBoostForecaster":
        """Fit boosting rounds, optionally with validation + early stopping.

        Args:
            X: Training features (2-D numerical array).
            y: Training targets.
            X_val: Optional validation features for ``eval_set``.
            y_val: Optional validation targets for ``eval_set``.

        Raises:
            ValueError: If ``early_stopping_rounds`` is set but no
                validation data is provided.
        """
        X, y = self._validate_xy(X, y)

        eval_set = None
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be provided together.")
            Xv, yv = self._validate_xy(X_val, y_val)
            if Xv.shape[1] != X.shape[1]:
                raise ValueError(
                    f"Feature mismatch: train has {X.shape[1]}, "
                    f"validation has {Xv.shape[1]}"
                )
            eval_set = [(Xv, yv)]

        if self.early_stopping_rounds is not None and eval_set is None:
            raise ValueError(
                "early_stopping_rounds requires validation data "
                "(X_val and y_val)."
            )

        callbacks = None
        if self.early_stopping_rounds is not None:
            callbacks = [
                xgb.callback.EarlyStopping(rounds=self.early_stopping_rounds)
            ]

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            nthread=self.n_jobs,
            random_state=self.random_state if self.random_state is not None else 0,
            objective="reg:squarederror",
            callbacks=callbacks,
        )
        fit_kwargs: dict = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        self.model.fit(X, y, **fit_kwargs)
        self.n_features_ = X.shape[1]
        self.best_iteration_ = getattr(self.model, "best_iteration", None)
        self._is_fitted = True
        logger.info(
            "Fitted XGBoostForecaster(n_estimators=%d, early_stopping=%s) "
            "on %d samples (best_iteration=%s)",
            self.n_estimators,
            self.early_stopping_rounds,
            X.shape[0],
            self.best_iteration_,
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
        """Gain-averaged feature importances (one per feature)."""
        self._check_fitted()
        assert self.model is not None
        return np.asarray(self.model.feature_importances_, dtype=float)

    def save(self, path: str | Path) -> Path:
        """Serialise with joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved XGBoostForecaster to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostForecaster":
        """Load a model saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__} in {path}, got {type(obj).__name__}"
            )
        logger.info("Loaded XGBoostForecaster from %s", path)
        return obj
