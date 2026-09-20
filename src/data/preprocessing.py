"""Configurable cleaning with full modification tracking (Phase 6A).

Rules:
- Exogenous gaps: configurable interpolation; ffill/bfill only if enabled.
- Target: interpolate ONLY gaps of length <= max_target_gap; longer gaps
  stay NaN and are dropped at the end (reported, never silent).
- Every removal/fill is counted in the cleaning report.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.schema import (
    CANONICAL_GROUP,
    CANONICAL_TARGET,
    CANONICAL_TIME,
    CleaningConfig,
    DataMapping,
)

logger = logging.getLogger(__name__)


def _grid_freq(df: pd.DataFrame, mapping: DataMapping) -> str:
    if mapping.expected_freq:
        return mapping.expected_freq
    diffs = []
    for _, g in df.groupby(CANONICAL_GROUP, sort=True):
        ts = g[CANONICAL_TIME].sort_values()
        if len(ts) >= 3:
            diffs.append(ts.diff().mode().iloc[0])
    if not diffs:
        raise ValueError("Cannot infer sampling frequency (too few rows).")
    return str(pd.Series(diffs).mode().iloc[0])


def clean(
    df: pd.DataFrame, mapping: DataMapping, cfg: CleaningConfig
) -> tuple[pd.DataFrame, dict]:
    """Clean canonical frame; return (cleaned, cleaning_report)."""
    rep: dict = {"rows_original": int(len(df))}
    work = df.copy()

    n_nat = int(work[CANONICAL_TIME].isna().sum())
    work = work.dropna(subset=[CANONICAL_TIME])
    rep["unparseable_timestamps_dropped"] = n_nat

    n_dup = int(work.duplicated(subset=[CANONICAL_GROUP, CANONICAL_TIME]).sum())
    work = work.drop_duplicates(subset=[CANONICAL_GROUP, CANONICAL_TIME], keep="first")
    rep["duplicates_removed"] = n_dup

    work = work.sort_values([CANONICAL_GROUP, CANONICAL_TIME]).reset_index(drop=True)
    freq = _grid_freq(work, mapping)
    rep["grid_frequency"] = freq

    missing_before = {c: int(work[c].isna().sum()) for c in work.columns}
    rep["missing_before"] = missing_before

    # Full regular grid per group -> gaps become explicit NaN rows.
    parts = []
    gap_stats = {"n_gaps": 0, "longest_gap_periods": 0, "gap_periods_total": 0}
    for name, g in work.groupby(CANONICAL_GROUP, sort=True):
        full_idx = pd.date_range(g[CANONICAL_TIME].min(), g[CANONICAL_TIME].max(), freq=freq)
        g = g.set_index(CANONICAL_TIME).reindex(full_idx)
        g[CANONICAL_GROUP] = str(name)
        runs = _gap_runs(g[CANONICAL_TARGET].isna().to_numpy())
        gap_stats["n_gaps"] += len(runs)
        if runs:
            gap_stats["longest_gap_periods"] = max(gap_stats["longest_gap_periods"], max(runs))
        gap_stats["gap_periods_total"] += int(sum(runs))
        g.index.name = CANONICAL_TIME
        parts.append(g.reset_index())
    work = pd.concat(parts, ignore_index=True).sort_values(
        [CANONICAL_GROUP, CANONICAL_TIME]
    ).reset_index(drop=True)
    rep["gap_statistics"] = gap_stats

    def _interp_group(frame: pd.DataFrame, col: str) -> np.ndarray:
        s = frame[col]
        if cfg.exogenous_interp_method == "time":
            return (
                s.set_axis(frame[CANONICAL_TIME])
                .interpolate(
                    method="time",
                    limit=cfg.exogenous_interp_limit,
                    limit_area="inside",
                )
                .to_numpy()
            )
        return s.interpolate(
            method=cfg.exogenous_interp_method,
            limit=cfg.exogenous_interp_limit,
            limit_area="inside",
        ).to_numpy()

    # Exogenous: interpolate, optional ffill/bfill.
    interp_cells = 0
    exo_cols = [c for c in mapping.exogenous() if c in work.columns]
    if cfg.exogenous_interp_method != "none":
        for name, g in work.groupby(CANONICAL_GROUP, sort=True):
            for col in exo_cols:
                before = int(g[col].isna().sum())
                work.loc[g.index, col] = _interp_group(g, col)
                interp_cells += before - int(work.loc[g.index, col].isna().sum())
    if cfg.allow_ffill_bfill:
        for col in exo_cols:
            work[col] = work.groupby(CANONICAL_GROUP, sort=False)[col].transform(
                lambda s: s.ffill().bfill()
            )
    rep["exogenous_interpolated_cells"] = int(interp_cells)
    rep["ffill_bfill_applied"] = bool(cfg.allow_ffill_bfill)

    # Target: fill ONLY interior runs of length <= max_target_gap.
    # Longer gaps stay NaN and are dropped below (reported, never silent).
    target_filled = 0
    if mapping.max_target_gap > 0:
        for name, g in work.groupby(CANONICAL_GROUP, sort=True):
            s = g[CANONICAL_TARGET].reset_index(drop=True)
            filled = s.copy()
            for start, end in _gap_spans(s.isna().to_numpy()):
                length = end - start
                if length <= mapping.max_target_gap and start > 0 and end < len(s):
                    seg = s.iloc[start - 1 : end + 1].interpolate(
                        method="linear", limit_area="inside"
                    )
                    filled.iloc[start:end] = seg.iloc[1:-1].to_numpy()
                    target_filled += length
            work.loc[g.index, CANONICAL_TARGET] = filled.to_numpy()
    rep["target_interpolated_cells"] = int(target_filled)

    dropped_target_na = 0
    if cfg.drop_target_na:
        mask = work[CANONICAL_TARGET].isna()
        dropped_target_na = int(mask.sum())
        work = work.loc[~mask].reset_index(drop=True)
    rep["dropped_rows_target_na"] = dropped_target_na
    rep["missing_after"] = {c: int(work[c].isna().sum()) for c in work.columns}
    rep["rows_final"] = int(len(work))
    logger.info("Cleaning: %d -> %d rows", rep["rows_original"], rep["rows_final"])
    return work, rep


def _gap_runs(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs."""
    return [end - start for start, end in _gap_spans(mask)]


def _gap_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index spans of consecutive True runs."""
    spans, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans
