"""Training package (Phase 2B baseline experiments).

Lazy exports avoid ``RuntimeWarning`` when running
``python -m src.training.experiment_runner`` (package ``__init__`` would
otherwise import the submodule before runpy executes it as ``__main__``).
"""

from typing import Any

__all__ = [
    "ExperimentRunner",
    "run_baseline_experiment",
    "fit_model",
    "predict_and_evaluate",
    "supports_validation",
    "run_ablation",
    "ABLATION_VARIANTS",
    "HORIZON_STRATEGY",
    "TemporalSplit",
    "split_from_config",
    "load_experiment_config",
]

_LAZY: dict[str, str] = {
    "ExperimentRunner": "src.training.experiment_runner",
    "run_baseline_experiment": "src.training.experiment_runner",
    "fit_model": "src.training.trainer",
    "predict_and_evaluate": "src.training.trainer",
    "supports_validation": "src.training.trainer",
    "run_ablation": "src.training.ablation",
    "ABLATION_VARIANTS": "src.training.ablation",
    "HORIZON_STRATEGY": "src.training.ablation",
    "TemporalSplit": "src.training.splits",
    "split_from_config": "src.training.splits",
    "load_experiment_config": "src.training.splits",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
