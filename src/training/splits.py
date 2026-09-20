"""Chronological train/validation/test protocol (Phase 5B).

Single source of truth for temporal splits. TRAIN < VALIDATION < TEST is
asserted for every group; nothing is shuffled, so lags, rolling features,
scalers and residual training cannot leak the future by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalSplit:
    """Ratios + derived integer boundaries for one ordered series."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        for name, r in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if not 0.0 < r < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {r}")

    def boundaries(self, n: int) -> tuple[int, int]:
        """Return ``(train_end, val_end)`` row cutoffs for ``n`` rows."""
        if n < 3:
            raise ValueError(f"Need >= 3 rows, got {n}")
        i_train = int(n * self.train_ratio)
        i_val = int(n * (self.train_ratio + self.val_ratio))
        if i_train < 1 or i_val <= i_train or n - i_val < 1:
            raise ValueError("Not enough rows for the requested split.")
        return i_train, i_val

    def split_arrays(self, X, y) -> tuple:
        """Split ordered arrays -> (X_tr, X_val, X_te, y_tr, y_val, y_te)."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different sample counts.")
        i_train, i_val = self.boundaries(len(X))
        assert i_train < i_val < len(X), "Split is not chronological."
        return (
            X[:i_train], X[i_train:i_val], X[i_val:],
            y[:i_train], y[i_train:i_val], y[i_val:],
        )

    def split_frame(
        self, df: pd.DataFrame, time_col: str = "time_idx", group_col: str = "group_id"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a long frame per group; assert TRAIN < VAL < TEST everywhere."""
        for col in (time_col, group_col):
            if col not in df.columns:
                raise ValueError(f"Frame missing column {col!r}")
        trains, vals, tests = [], [], []
        for name, g in df.groupby(group_col, sort=True):
            g = g.sort_values(time_col).reset_index(drop=True)
            i_train, i_val = self.boundaries(len(g))
            tr, va, te = g.iloc[:i_train], g.iloc[i_train:i_val], g.iloc[i_val:]
            assert tr[time_col].max() < va[time_col].min(), f"Leak in {name}"
            assert va[time_col].max() < te[time_col].min(), f"Leak in {name}"
            trains.append(tr)
            vals.append(va)
            tests.append(te)
        out = (
            pd.concat(trains, ignore_index=True),
            pd.concat(vals, ignore_index=True),
            pd.concat(tests, ignore_index=True),
        )
        logger.info(
            "Temporal split: train=%d val=%d test=%d rows", *[len(p) for p in out]
        )
        return out


def default_config_path() -> Path:
    """Absolute path to ``configs/experiment.yaml`` (repo-root relative)."""
    return Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"


def load_experiment_config(path: Optional[str | Path] = None) -> dict:
    """Load the research protocol config (splits, horizon, OOF)."""
    cfg_path = Path(path) if path is not None else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_from_config(path: Optional[str | Path] = None) -> TemporalSplit:
    """Build a :class:`TemporalSplit` from ``configs/experiment.yaml``."""
    cfg = load_experiment_config(path).get("splits", {})
    return TemporalSplit(
        train_ratio=float(cfg.get("train_ratio", 0.7)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.15)),
    )
