"""Unified baseline experiment runner (Phase 2B, synthetic data only).

Example:
    python -m src.training.experiment_runner --n-samples 2160
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_regression
from src.models.advanced import NBEATSForecaster, PatchTSTForecaster
from src.models.baseline import (
    LightGBMForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)
from src.models.base import BaseForecaster
from src.models.deep import GRUForecaster, LSTMForecaster
from src.models.hybrid import HybridTFTXGBoostForecaster
from src.models.statistical import (
    PersistenceForecaster,
    SARIMAForecaster,
    SeasonalPersistenceForecaster,
)
from src.models.tft import TemporalFusionTransformerForecaster
from src.training.trainer import fit_model, predict_and_evaluate
from src.utils.experiment import (
    FEATURE_COLS,
    TARGET_COL,
    chronological_split,
    ensure_dir,
    generate_synthetic_grid_data,
    save_results_csv,
)
from src.utils.seed import set_seed
from src.utils.viz import plot_model_comparison, plot_prediction_vs_actual

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {
    # Q1 headline 11 (frozen protocol order):
    "persistence": PersistenceForecaster,
    "seasonal_persistence": SeasonalPersistenceForecaster,
    "sarima": SARIMAForecaster,
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "tft": TemporalFusionTransformerForecaster,
    "patchtst": PatchTSTForecaster,
    "nbeats": NBEATSForecaster,
    "hybrid": HybridTFTXGBoostForecaster,
    # Legacy (NOT part of Q1 headline comparison):
    "linear_regression": LinearRegressionForecaster,
    "random_forest": RandomForestForecaster,
}

PRETTY_NAMES = {
    "persistence": "Persistence",
    "seasonal_persistence": "SeasonalPersistence",
    "sarima": "SARIMA-lite",
    "linear_regression": "LinearRegression",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
    "gru": "GRU",
    "tft": "TFT",
    "patchtst": "PatchTST",
    "nbeats": "N-BEATS",
    "hybrid": "HybridTFT-XGB",
}

RESULT_COLUMNS = ["Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"]

# Q1 headline guard (Phase 2.1). The locked 11-model Q1 set; legacy models
# must never enter headline results by omission or convention. ``run_all``
# below keeps its synthetic-verification default (all 13) for backward
# compatibility; Q1 headline execution MUST go through
# ``q1_headline_models`` / ``run_q1_headline`` which fail closed on legacy,
# unknown, or duplicated entries.
Q1_HEADLINE_MODELS: tuple[str, ...] = (
    "persistence",
    "seasonal_persistence",
    "sarima",
    "xgboost",
    "lightgbm",
    "lstm",
    "gru",
    "tft",
    "patchtst",
    "nbeats",
    "hybrid",
)

Q1_LEGACY_MODELS: tuple[str, ...] = ("linear_regression", "random_forest")


def q1_headline_models(models: Optional[list[str]] = None) -> list[str]:
    """Resolve the exact 11-model Q1 headline set (fail-closed).

    Args:
        models: None (default 11 in locked protocol order) or an explicit
            subset/list of Q1 headline keys.

    Raises:
        ValueError: on legacy, unknown, empty, or duplicated entries.
    """
    if models is None:
        return list(Q1_HEADLINE_MODELS)
    if not models:
        raise ValueError("Q1 headline model list must not be empty.")
    seen: set[str] = set()
    for name in models:
        if name in Q1_LEGACY_MODELS:
            raise ValueError(
                f"Legacy model {name!r} is quarantined and cannot enter "
                "Q1 headline execution. Use the locked 11-model set."
            )
        if name not in Q1_HEADLINE_MODELS:
            raise ValueError(
                f"Unknown Q1 headline model {name!r}. Expected one of "
                f"{list(Q1_HEADLINE_MODELS)}"
            )
        if name in seen:
            raise ValueError(f"Duplicated Q1 headline model {name!r}.")
        seen.add(name)
    return list(models)


class ExperimentRunner:
    """Train, evaluate and persist baseline forecasters on ordered data.

    Splits are chronological (train -> val -> test); no shuffling, so no
    future leakage by construction.
    """

    def __init__(
        self,
        output_dir: str | Path = "results/baseline",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        use_validation: bool = True,
        use_early_stopping: bool = False,
        early_stopping_rounds: int = 50,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_validation = use_validation
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        set_seed(seed)

    def build_model(self, name: str, **overrides) -> BaseForecaster:
        """Instantiate a registered model from ``configs/models.yaml``."""
        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {name!r}. Expected one of {sorted(MODEL_REGISTRY)}"
            )
        cls = MODEL_REGISTRY[name]
        params = dict(overrides)
        if self.use_early_stopping and "early_stopping_rounds" not in params:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        return cls.from_config(**params)

    def run_one(
        self,
        name: str,
        X=None,
        y=None,
        *,
        X_tr=None,
        y_tr=None,
        X_val=None,
        y_val=None,
        X_te=None,
        y_te=None,
        **overrides,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Train one model; return (result_row, y_aligned, predictions).

        If *X_tr* and *X_te* are supplied the caller's explicit splits are
        used directly.  Otherwise a chronological split is performed on
        *X*/*y* as before (backward-compatible default).
        """
        if X_tr is not None and X_te is not None:
            pass  # explicit splits supplied by caller
        else:
            X_tr, X_val, X_te, y_tr, y_val, y_te = chronological_split(
                X, y, self.train_ratio, self.val_ratio
            )
        model = self.build_model(name, **overrides)
        fit_model(
            model, X_tr, y_tr, X_val, y_val, use_validation=self.use_validation
        )
        forecast = getattr(model, "forecast", None)
        if callable(forecast):
            # Horizon models needing target history (e.g. TFT): forecast the
            # trailing horizon of the test block and score against actuals.
            preds, actuals = forecast(X_te, y_te)
            preds = np.ravel(np.asarray(preds, dtype=float))
            y_aligned = np.ravel(np.asarray(actuals, dtype=float))
            metrics = evaluate_regression(y_aligned, preds)
            row = {"Model": PRETTY_NAMES[name], **metrics}
            logger.info("%s test metrics: %s", PRETTY_NAMES[name], metrics)
            return row, y_aligned, preds
        preds, metrics = predict_and_evaluate(model, X_te, y_te)
        align = getattr(model, "align_target", None)
        y_aligned = (
            np.asarray(align(X_te, y_te), dtype=float)
            if callable(align)
            else np.asarray(y_te, dtype=float)
        )
        row = {"Model": PRETTY_NAMES[name], **metrics}
        logger.info("%s test metrics: %s", PRETTY_NAMES[name], metrics)
        return row, y_aligned, np.asarray(preds, dtype=float)

    def run_all(
        self,
        X,
        y,
        models: Optional[list[str]] = None,
        model_overrides: Optional[dict[str, dict]] = None,
    ) -> pd.DataFrame:
        """Run baselines on synthetic data, save CSV + figures, return table.

        Default (``models=None``) runs all 13 registered models for
        synthetic verification (includes legacy). This default is
        LEGACY/synthetic-only and MUST NOT be used for Q1 headline
        results — use :meth:`run_q1_headline` instead.
        """
        models = list(models) if models is not None else sorted(MODEL_REGISTRY)
        model_overrides = model_overrides or {}
        ensure_dir(self.figures_dir)

        rows: list[dict] = []
        for name in models:
            row, y_te, preds = self.run_one(
                name, X, y, **model_overrides.get(name, {})
            )
            rows.append(row)
            plot_prediction_vs_actual(
                y_te,
                preds,
                PRETTY_NAMES[name],
                self.figures_dir / f"pred_vs_actual_{name}.png",
            )

        results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        save_results_csv(results, self.output_dir / "results.csv")
        plot_model_comparison(
            results, self.figures_dir / "model_comparison.png"
        )
        return results

    def run_q1_headline(
        self,
        X,
        y,
        models: Optional[list[str]] = None,
        model_overrides: Optional[dict[str, dict]] = None,
    ) -> pd.DataFrame:
        """Run the locked Q1 headline set (synthetic-data verification only).

        Resolves via :func:`q1_headline_models` (exact 11, protocol order
        by default); legacy/unknown/duplicated entries raise. Produces no
        headline research claims; for Phase 2.1 verification plumbing only.
        """
        return self.run_all(
            X, y, models=q1_headline_models(models), model_overrides=model_overrides
        )


def run_baseline_experiment(
    n_samples: int = 2160,
    output_dir: str | Path = "results/baseline",
    seed: int = 42,
    **runner_kwargs,
) -> pd.DataFrame:
    """Generate synthetic data and run all baselines (verification only)."""
    df = generate_synthetic_grid_data(n_samples=n_samples, seed=seed)
    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET_COL].to_numpy()
    runner = ExperimentRunner(output_dir=output_dir, seed=seed, **runner_kwargs)
    return runner.run_all(X, y)


def main(argv: Optional[list[str]] = None) -> pd.DataFrame:
    """CLI entry point for the synthetic baseline experiment."""
    parser = argparse.ArgumentParser(description="Synthetic baseline experiment")
    parser.add_argument("--n-samples", type=int, default=2160)
    parser.add_argument("--output-dir", type=str, default="results/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_baseline_experiment(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        use_validation=not args.no_validation,
        use_early_stopping=args.early_stopping,
    )
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
