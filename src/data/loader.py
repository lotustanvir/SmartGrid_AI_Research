"""CSV loader -> canonical frame (Phase 6A). Never shuffles."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.schema import (
    CANONICAL_GROUP,
    CANONICAL_TARGET,
    CANONICAL_TIME,
    SINGLE_GROUP,
    DataMapping,
)

logger = logging.getLogger(__name__)


def load_raw_csv(mapping: DataMapping) -> pd.DataFrame:
    """Load provider CSV and rename to canonical columns.

    Returns frame with ``timestamp`` (datetime, NaT if unparseable),
    ``group_id``, ``target`` (float) + present exogenous signals.
    Rows are sorted by (group_id, timestamp); nothing is dropped here —
    cleaning decides, with a report.
    """
    path = Path(mapping.path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in mapping.required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing mapped columns: {missing}")

    out = pd.DataFrame()
    out[CANONICAL_TIME] = pd.to_datetime(df[mapping.timestamp_col], errors="coerce")
    out[CANONICAL_TARGET] = pd.to_numeric(df[mapping.target_col], errors="coerce")
    if mapping.group_col:
        out[CANONICAL_GROUP] = df[mapping.group_col].astype(str)
    else:
        out[CANONICAL_GROUP] = SINGLE_GROUP
    for canon, provider in mapping.exogenous().items():
        out[canon] = pd.to_numeric(df[provider], errors="coerce")
    out = out.sort_values([CANONICAL_GROUP, CANONICAL_TIME]).reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(out), path)
    return out
