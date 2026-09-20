"""Exploratory EDA figures (Phase 6A).

EXPLORATORY ONLY — not final paper figures. Every figure is watermarked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.schema import CANONICAL_TARGET, CANONICAL_TIME

logger = logging.getLogger(__name__)

TAG = "EXPLORATORY — NOT FOR PUBLICATION"


def _save(fig, path: Path) -> Path:
    fig.suptitle(TAG, fontsize=8, color="red")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def generate_eda(df: pd.DataFrame, out_dir: str | Path) -> list[str]:
    """Generate EDA PNGs; return saved paths (as strings)."""
    out_dir = Path(out_dir)
    saved = []
    plot_df = df.copy()
    plot_df["hour"] = plot_df[CANONICAL_TIME].dt.hour
    plot_df["dow"] = plot_df[CANONICAL_TIME].dt.dayofweek
    plot_df["month"] = plot_df[CANONICAL_TIME].dt.month
    y = plot_df[CANONICAL_TARGET].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(plot_df[CANONICAL_TIME].to_numpy(), y, linewidth=0.6)
    ax.set_title("Load time series")
    saved.append(str(_save(fig, out_dir / "load_series.png")))

    for col, title, fname in (
        ("hour", "Hourly load profile", "hourly_profile.png"),
        ("dow", "Day-of-week load profile", "dow_profile.png"),
        ("month", "Monthly load profile", "monthly_profile.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 3))
        plot_df.groupby(col)[CANONICAL_TARGET].mean().plot(ax=ax, marker="o")
        ax.set_title(title)
        saved.append(str(_save(fig, out_dir / fname)))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(y[np.isfinite(y)], bins=50)
    ax.set_title("Target distribution")
    saved.append(str(_save(fig, out_dir / "target_dist.png")))

    fig, ax = plt.subplots(figsize=(7, 3))
    plot_df.isna().mean().plot.bar(ax=ax)
    ax.set_title("Missingness by column")
    saved.append(str(_save(fig, out_dir / "missingness.png")))

    num = plot_df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(num.corr(numeric_only=True).to_numpy(), vmin=-1, vmax=1)
        ax.set_xticks(range(num.shape[1]), num.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(num.shape[1]), num.columns, fontsize=7)
        ax.set_title("Correlation matrix")
        fig.colorbar(im, ax=ax)
        saved.append(str(_save(fig, out_dir / "correlation.png")))

    logger.info("EDA saved %d figures to %s", len(saved), out_dir)
    return saved
