"""Q1 calendar split (Phase 2).

Replaces the 70/15/15 ratio path for the Q1 protocol with explicit
calendar boundaries:

  TRAIN: timestamp < 2024-01-01
  VAL:   2024-01-01 <= timestamp < 2025-01-01
  TEST:  2025-01-01 <= timestamp < 2026-01-01

Hard assertions: train_end < val_start, val_end < test_start,
timestamp ordering, no duplicate timestamps, hourly continuity check.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.q1.protocol import TEST_END, TRAIN_END, VAL_END

logger = logging.getLogger(__name__)


def split_calendar(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an hourly frame on calendar boundaries (fail-closed)."""
    if time_col not in df.columns:
        raise ValueError(f"Frame missing time column {time_col!r}")
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    if work[time_col].isna().any():
        raise ValueError("NaT timestamps detected.")
    work = work.sort_values(time_col).reset_index(drop=True)
    if work[time_col].duplicated().any():
        dups = work[work[time_col].duplicated(keep=False)]
        raise ValueError(
            f"Duplicate timestamps detected: {dups[time_col].iloc[:5].tolist()}"
        )
    t0 = pd.Timestamp(train_end)
    t1 = pd.Timestamp(val_end)
    t2 = pd.Timestamp(test_end)
    tr = work[work[time_col] < t0].reset_index(drop=True)
    va = work[(work[time_col] >= t0) & (work[time_col] < t1)].reset_index(drop=True)
    te = work[(work[time_col] >= t1) & (work[time_col] < t2)].reset_index(drop=True)
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError(
            f"Calendar split produced empty block: train={len(tr)} "
            f"val={len(va)} test={len(te)}"
        )
    # Hard ordering assertions.
    assert tr[time_col].max() < va[time_col].min(), "Leak: train_end >= val_start"
    assert va[time_col].max() < te[time_col].min(), "Leak: val_end >= test_start"
    # Hourly continuity report (fail-closed on duplicates already; gaps warn).
    for name, part in (("train", tr), ("val", va), ("test", te)):
        diffs = part[time_col].diff().dropna()
        if len(diffs) and (diffs != pd.Timedelta(hours=1)).any():
            n_bad = int((diffs != pd.Timedelta(hours=1)).sum())
            logger.warning(
                "%s block has %d non-hourly steps (gaps allowed, reported).",
                name,
                n_bad,
            )
    logger.info(
        "Q1 calendar split: train=%d (%s..%s) val=%d (%s..%s) test=%d (%s..%s)",
        len(tr),
        tr[time_col].min(),
        tr[time_col].max(),
        len(va),
        va[time_col].min(),
        va[time_col].max(),
        len(te),
        te[time_col].min(),
        te[time_col].max(),
    )
    return tr, va, te
