"""Research-safe ablation suite (Phase 5B).

Compares on the SAME aligned timestamps:

A. XGBoost only (baseline runner model)
B. TFT only
C. TFT + XGBoost, in_sample residuals (debug only, warns)
D. TFT + XGBoost, OOF residuals (research candidate)

No superiority claims are made here; outputs are metric tables +
alignment reports for later real-data experiments.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.alignment import EvaluationAlignment, make_sequential_keys
from src.training.experiment_runner import ExperimentRunner
from src.training.splits import TemporalSplit

logger = logging.getLogger(__name__)

HORIZON_STRATEGY = {
    "linear_regression": "direct single-output (row-wise point model)",
    "random_forest": "direct single-output (row-wise point model)",
    "xgboost": "direct single-output (row-wise point model)",
    "lightgbm": "direct single-output (row-wise point model)",
    "lstm": "multi-output (output_size = horizon)",
    "gru": "multi-output (output_size = horizon)",
    "tft": "multi-output direct (prediction_length = horizon)",
    "hybrid": "multi-output direct TFT + residual correction",
}

ABLATION_VARIANTS: dict[str, dict] = {
    "A_xgb_only": {"model": "xgboost", "overrides": {}},
    "B_tft_only": {"model": "tft", "overrides": {}},
    "C_hybrid_insample": {
        "model": "hybrid",
        "overrides": {"residual_training_mode": "in_sample"},
    },
    "D_hybrid_oof": {
        "model": "hybrid",
        "overrides": {"residual_training_mode": "oof"},
    },
}


def run_ablation(
    X,
    y,
    variants: Optional[dict[str, dict]] = None,
    model_overrides: Optional[dict[str, dict]] = None,
    output_dir: str = "results/ablation",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run variants, align on common timestamps, score.

    Returns ``(metrics_df, alignment_report, metadata)`` where metadata
    records horizon strategy + research-validity flags per variant.
    """
    variants = variants or dict(ABLATION_VARIANTS)
    model_overrides = model_overrides or {}
    runner = ExperimentRunner(output_dir=output_dir, seed=seed)
    split = TemporalSplit()
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    i_train, i_val = split.boundaries(len(X))
    test_start = i_val
    n_test = len(X) - i_val

    alignment = EvaluationAlignment()
    metadata: dict[str, dict] = {}
    y_test = np.asarray(y).ravel()[i_val:]
    for variant, spec in variants.items():
        name = spec["model"]
        extra = dict(model_overrides.get(name, {}))
        extra.update(spec.get("overrides", {}))
        row, y_aligned, preds = runner.run_one(name, X, y, **extra)
        pr = np.asarray(preds).ravel()
        if name in ("tft", "hybrid"):
            # Trailing-horizon windows; keys = decoder timestamps.
            horizon = len(pr)
            keys = make_sequential_keys(horizon, test_start + n_test - horizon)
            yt = np.asarray(y_aligned).ravel()
        else:
            # Row-wise models predict a (possibly history-shortened) tail.
            keys = make_sequential_keys(len(pr), test_start + n_test - len(pr))
            yt = y_test[-len(pr):]
        alignment.add_model(
            variant,
            keys,
            yt,
            pr,
            n_original=n_test,
            reason=(
                f"{name}: trailing-horizon windows"
                if name in ("tft", "hybrid")
                else f"{name}: history-limited predictable tail"
            ),
        )
        metadata[variant] = {
            "model": name,
            "horizon_strategy": HORIZON_STRATEGY[name],
            "research_valid": not (
                name == "hybrid" and extra.get("residual_training_mode") == "in_sample"
            ),
        }
    report = alignment.report()
    scores = alignment.metrics()
    metrics_df = pd.DataFrame(
        [{"Variant": v, **s} for v, s in scores.items()],
        columns=["Variant", "MAE", "RMSE", "MAPE", "sMAPE", "R2"],
    )
    return metrics_df, report, metadata
