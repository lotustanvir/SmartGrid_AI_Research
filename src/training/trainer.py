"""Single-model training utilities shared by experiment runners."""

from __future__ import annotations

import inspect
import logging
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


def supports_validation(model: BaseForecaster) -> bool:
    """Whether ``model.fit`` accepts ``(X_val, y_val)`` kwargs."""
    try:
        params = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False
    return "X_val" in params and "y_val" in params


def fit_model(
    model: BaseForecaster,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_val: Optional[ArrayLike] = None,
    y_val: Optional[ArrayLike] = None,
    use_validation: bool = True,
) -> BaseForecaster:
    """Fit a model, forwarding validation data only when supported."""
    if use_validation and X_val is not None and supports_validation(model):
        logger.info("Fitting %s with validation set", type(model).__name__)
        return model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    if use_validation and X_val is not None and not supports_validation(model):
        logger.info(
            "%s has no validation hook; fitting on train only",
            type(model).__name__,
        )
    return model.fit(X_train, y_train)


def predict_and_evaluate(
    model: BaseForecaster, X_test: ArrayLike, y_test: ArrayLike
) -> tuple[np.ndarray, dict]:
    """Predict on the held-out test set and score with shared metrics."""
    preds = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
    return preds, metrics
