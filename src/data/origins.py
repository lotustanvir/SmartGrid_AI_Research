"""Canonical forecast-origin generator (Phase 2).

For each horizon H in {1,6,24}, valid test origins satisfy:
  * 168h history available (required_history_start..required_history_end),
  * all model-required input features available without future leakage
    (history fully inside the engineered frame),
  * target horizon fully inside the test evaluation period,
  * actual target observations exist (no NaN target in horizon),
  * origin/horizon inside TEST period [2025-01-01, 2026-01-01),
  * no future information in the model input (origins index history only).

Deterministic; persisted to experiments/protocol/test_origins.parquet
with an origin-set hash. All models receive the same origin set;
evaluation MUST fail closed on mismatch.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.q1.protocol import (
    FEATURE_VERSION,
    HISTORY_LENGTH,
    HORIZONS,
    ORIGIN_ARTIFACT,
    Q1_DATASET_VERSION,
)

logger = logging.getLogger(__name__)


def generate_origins(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    history: int = HISTORY_LENGTH,
    time_col: str = "timestamp",
    target_col: str = "target",
) -> pd.DataFrame:
    """Generate valid origins from a chronological test frame."""
    if time_col not in frame.columns or target_col not in frame.columns:
        raise ValueError("Frame missing timestamp/target.")
    work = frame.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values(time_col).reset_index(drop=True)
    if work[time_col].duplicated().any():
        raise ValueError("Duplicate timestamps in origin frame.")
    n = len(work)
    rows: list[dict] = []
    oid = 0
    for h in horizons:
        if h <= 0:
            raise ValueError(f"Horizon must be positive, got {h}")
        # Origin index i: history [i-history+1..i] exists, horizon
        # (i+1..i+h) exists with non-NaN targets.
        for i in range(history - 1, n - h):
            hist = work.iloc[i - history + 1 : i + 1]
            fut = work.iloc[i + 1 : i + 1 + h]
            if hist[target_col].isna().any() or fut[target_col].isna().any():
                continue
            origin_ts = work.loc[i, time_col]
            rows.append(
                {
                    "origin_id": oid,
                    "origin_timestamp": origin_ts,
                    "horizon": int(h),
                    "origin_index": int(i),
                    "target_start": fut[time_col].iloc[0],
                    "target_end": fut[time_col].iloc[-1],
                    "required_history_start": hist[time_col].iloc[0],
                    "required_history_end": origin_ts,
                    "dataset_version": Q1_DATASET_VERSION,
                    "feature_version": FEATURE_VERSION,
                }
            )
            oid += 1
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid origins generated.")
    out["origin_timestamp"] = pd.to_datetime(out["origin_timestamp"])
    out["target_start"] = pd.to_datetime(out["target_start"])
    out["target_end"] = pd.to_datetime(out["target_end"])
    out["required_history_start"] = pd.to_datetime(out["required_history_start"])
    out["required_history_end"] = pd.to_datetime(out["required_history_end"])
    return out


def origins_hash(origins: pd.DataFrame) -> str:
    """Deterministic hash of the origin set (sorted, stringified).

    Robust to timestamp representation: datetime-like or string columns
    are normalized with ``pd.to_datetime`` and canonicalized to ISO
    format before hashing, so hashing a frame reloaded via ``pd.read_csv``
    yields the same digest as the in-memory canonical frame (the locked
    origin hash is preserved).
    """
    work = origins.copy()
    for col in ("origin_timestamp", "target_start", "target_end"):
        work[col] = pd.to_datetime(work[col], errors="raise")
    key = work.sort_values(["horizon", "origin_timestamp"]).reset_index(drop=True)
    payload = "\n".join(
        f"{pd.Timestamp(r.origin_timestamp).isoformat()}|{int(r.horizon)}|"
        f"{pd.Timestamp(r.target_start).isoformat()}|"
        f"{pd.Timestamp(r.target_end).isoformat()}"
        for r in key.itertuples()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_origins(origins: pd.DataFrame, path: str | Path = ORIGIN_ARTIFACT) -> Path:
    """Persist origins and return the ACTUAL written path.

    Parquet is preferred; when no parquet engine is installed the frame
    is written to the ``.csv`` sidecar and THAT path is returned (callers
    must record the returned path in the registry, never the requested
    ``.parquet`` path when CSV was written).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        origins.to_parquet(p, index=False)
    except (ImportError, ModuleNotFoundError):
        # No pyarrow/fastparquet in the verified env; CSV is the
        # equivalent canonical machine-readable artifact (documented).
        csv_path = p.with_suffix(".csv")
        origins.to_csv(csv_path, index=False)
        logger.warning("Parquet engine missing; saved CSV fallback %s", csv_path)
        return csv_path
    logger.info("Saved %d origins to %s", len(origins), p)
    return p


def load_origins(path: str | Path = ORIGIN_ARTIFACT) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        # Accept CSV fallback sidecar.
        alt = p.with_suffix(".csv")
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"Origin artifact not found: {path}")
    if p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    for col in ("origin_timestamp", "target_start", "target_end"):
        df[col] = pd.to_datetime(df[col])
    return df


def assert_same_origins(a: pd.DataFrame, b: pd.DataFrame, label: str = "") -> None:
    """Fail closed if two origin sets differ (same horizon + timestamps)."""
    ka = (
        a.sort_values(["horizon", "origin_timestamp"])
        .reset_index(drop=True)[["horizon", "origin_timestamp"]]
        .astype(str)
    )
    kb = (
        b.sort_values(["horizon", "origin_timestamp"])
        .reset_index(drop=True)[["horizon", "origin_timestamp"]]
        .astype(str)
    )
    if len(ka) != len(kb) or not ka.equals(kb):
        raise ValueError(
            f"Origin-set mismatch {label}: {len(ka)} vs {len(kb)} rows. "
            "Models MUST share exactly the same origins."
        )
