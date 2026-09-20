"""Plotting utilities for baseline experiments (synthetic data only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_prediction_vs_actual(
    y_true,
    y_pred,
    model_name: str,
    save_path: str | Path,
    max_points: int = 500,
) -> Path:
    """Line plot of actual vs predicted demand for the last test points."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred shape mismatch.")
    n = min(len(yt), max_points)
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, yt[-n:], label="Actual", linewidth=1.2)
    ax.plot(xs, yp[-n:], label="Predicted", linewidth=1.2, alpha=0.8)
    ax.set_title(f"{model_name}: prediction vs actual (last {n} points)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Electricity demand")
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", save_path)
    return save_path


def plot_model_comparison(
    results: pd.DataFrame,
    save_path: str | Path,
    metrics: Sequence[str] = ("MAE", "RMSE", "MAPE", "sMAPE", "R2"),
) -> Path:
    """Grouped bar chart comparing baseline models across metrics."""
    metrics = [m for m in metrics if m in results.columns]
    if not metrics or "Model" not in results.columns:
        raise ValueError("Results must contain 'Model' and at least one metric.")
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    x = np.arange(len(results))
    for ax, metric in zip(axes.flat, metrics):
        ax.bar(x, results[metric].to_numpy())
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(results["Model"].tolist(), rotation=20, ha="right")
    fig.suptitle("Baseline model comparison (synthetic data)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", save_path)
    return save_path
