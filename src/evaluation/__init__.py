"""Evaluation helpers."""

from typing import Any

__all__ = [
    "EvaluationAlignment",
    "explode_windows",
    "make_sequential_keys",
    "model_frame",
    "evaluate_regression",
    "mae",
    "rmse",
    "safe_mape",
    "smape",
    "r2",
]

_LAZY: dict[str, str] = {
    "EvaluationAlignment": "src.evaluation.alignment",
    "explode_windows": "src.evaluation.alignment",
    "make_sequential_keys": "src.evaluation.alignment",
    "model_frame": "src.evaluation.alignment",
    "evaluate_regression": "src.evaluation.metrics",
    "mae": "src.evaluation.metrics",
    "rmse": "src.evaluation.metrics",
    "safe_mape": "src.evaluation.metrics",
    "smape": "src.evaluation.metrics",
    "r2": "src.evaluation.metrics",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
