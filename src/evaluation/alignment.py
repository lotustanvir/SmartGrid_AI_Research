"""Common evaluation index for cross-model comparison (Phase 5B).

Different models predict different row subsets (history requirements,
horizons). Fair comparison requires scoring every model on the SAME
timestamps. This module builds per-model prediction frames keyed by
``(group_id, time_idx)``, intersects them, and reports exactly which rows
were dropped and why — nothing is silently discarded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_regression

logger = logging.getLogger(__name__)

KEY_COLS = ["group_id", "time_idx"]


def make_sequential_keys(
    n: int, start: int, group: str = "single"
) -> pd.DataFrame:
    """Keys for ``n`` ordered test rows starting at global time ``start``."""
    if n <= 0:
        raise ValueError("n must be positive.")
    return pd.DataFrame(
        {"group_id": [group] * n, "time_idx": np.arange(start, start + n)}
    )


def explode_windows(
    groups: list[str],
    decoder_times: np.ndarray,
    preds: np.ndarray,
    actuals: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Flatten ``(n_windows, horizon)`` forecasts to one row per timestamp.

    Returns ``(keys, y_true_1d, preds_1d)`` for :func:`model_frame`.
    """
    groups = list(groups)
    dt = np.asarray(decoder_times)
    pr = np.asarray(preds, dtype=float)
    ac = np.asarray(actuals, dtype=float)
    if not (dt.shape == pr.shape == ac.shape and dt.ndim == 2):
        raise ValueError("decoder_times/preds/actuals must share (W, H) shape.")
    if len(groups) != dt.shape[0]:
        raise ValueError("One group label per window required.")
    keys = pd.DataFrame(
        {
            "group_id": [g for g in groups for _ in range(dt.shape[1])],
            "time_idx": dt.ravel().astype(int),
        }
    )
    return keys, ac.ravel(), pr.ravel()


def model_frame(
    keys: pd.DataFrame,
    y_true,
    preds,
    model_name: str,
) -> pd.DataFrame:
    """Assemble one model's canonical frame (keys + truth + prediction)."""
    keys = keys.reset_index(drop=True)
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(preds, dtype=float).ravel()
    if len(keys) != len(yt) or len(keys) != len(yp):
        raise ValueError(
            f"{model_name}: keys({len(keys)}) / y({len(yt)}) / "
            f"preds({len(yp)}) length mismatch."
        )
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError(f"{model_name}: NaN in aligned predictions/truth.")
    frame = keys[KEY_COLS].copy()
    frame["y_true"] = yt
    frame["prediction"] = yp
    frame["model_name"] = model_name
    return frame


class EvaluationAlignment:
    """Collect per-model frames; score on the timestamp intersection."""

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._original_counts: dict[str, int] = {}
        self._reasons: dict[str, str] = {}

    def add_model(
        self,
        model_name: str,
        keys: pd.DataFrame,
        y_true,
        preds,
        n_original: int,
        reason: str,
    ) -> None:
        """Register predictions with provenance for the drop report."""
        if model_name in self._frames:
            raise ValueError(f"Model {model_name!r} already registered.")
        self._frames[model_name] = model_frame(keys, y_true, preds, model_name)
        self._original_counts[model_name] = int(n_original)
        self._reasons[model_name] = reason

    def common_keys(self) -> pd.DataFrame:
        """Intersection of predictable timestamps across all models."""
        if not self._frames:
            raise ValueError("No models registered.")
        common = None
        for frame in self._frames.values():
            keys = frame[KEY_COLS]
            common = keys if common is None else pd.merge(common, keys, on=KEY_COLS)
        assert common is not None
        return common.sort_values(KEY_COLS).reset_index(drop=True)

    def report(self) -> dict:
        """Per-model counts: original, predictable, aligned, dropped + why."""
        common = self.common_keys()
        common_set = set(map(tuple, common[KEY_COLS].to_numpy().tolist()))
        out = {"common_aligned_count": int(len(common)), "models": {}}
        for name, frame in self._frames.items():
            predictable = len(frame)
            own = set(map(tuple, frame[KEY_COLS].to_numpy().tolist()))
            dropped = len(own - common_set)
            out["models"][name] = {
                "original_test_count": self._original_counts[name],
                "predictable_count": predictable,
                "aligned_count": int(len(common)),
                "dropped_count": int(dropped),
                "drop_reason": self._reasons[name]
                + ("; outside cross-model intersection" if dropped else ""),
            }
        logger.info("Alignment report: %s", out)
        return out

    def metrics(self, epsilon: float = 1e-8) -> dict[str, dict]:
        """Metric dict per model computed ONLY on common timestamps."""
        common = self.common_keys()
        scores = {}
        for name, frame in self._frames.items():
            merged = pd.merge(common, frame, on=KEY_COLS)
            scores[name] = evaluate_regression(
                merged["y_true"].to_numpy(),
                merged["prediction"].to_numpy(),
                epsilon,
            )
        return scores
