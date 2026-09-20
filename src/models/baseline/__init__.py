"""Baseline forecasting models (Phase 2A).

All forecasters subclass :class:`src.models.base.BaseForecaster` and share
the ``fit / predict / evaluate / save / load`` interface.
"""

from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.baseline.lightgbm_model import LightGBMForecaster
from src.models.baseline.linear_regression import LinearRegressionForecaster
from src.models.baseline.random_forest import RandomForestForecaster
from src.models.baseline.xgboost_model import XGBoostForecaster

__all__ = [
    "LinearRegressionForecaster",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "LightGBMForecaster",
    "load_params",
    "default_config_path",
]

MODEL_NAMES = (
    "linear_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "lstm",
    "gru",
    "tft",
    "hybrid_tft_xgb",
)


def default_config_path() -> Path:
    """Absolute path to ``configs/models.yaml`` (repo-root relative)."""
    return Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def load_params(name: str, path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load constructor params for one baseline model from YAML.

    Args:
        name: One of ``linear_regression``, ``random_forest``,
            ``xgboost``, ``lightgbm``.
        path: Optional custom YAML path (defaults to configs/models.yaml).

    Raises:
        ValueError: If ``name`` is unknown.
        FileNotFoundError: If the YAML file does not exist.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Expected one of {MODEL_NAMES}")
    cfg_path = Path(path) if path is not None else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg.get(name, {}) or {})
