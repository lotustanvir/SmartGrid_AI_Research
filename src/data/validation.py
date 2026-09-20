"""Schema validation with explicit issue report (Phase 6A).

Nothing is corrected here — problems are reported with severity.
``error`` issues must block the pipeline; ``warning`` issues proceed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.schema import (
    CANONICAL_GROUP,
    CANONICAL_TARGET,
    CANONICAL_TIME,
    DataMapping,
)

logger = logging.getLogger(__name__)


def _freq_label(diffs: pd.Series) -> tuple[str, int]:
    if diffs.empty:
        return "unknown", 0
    mode = diffs.mode().iloc[0]
    irregular = int((diffs != mode).sum())
    return str(mode), irregular


def validate(df: pd.DataFrame, mapping: DataMapping) -> dict:
    """Validate canonical frame; return JSON-serialisable report."""
    issues: list[dict] = []

    def add(check: str, severity: str, detail: str, count: int = 0) -> None:
        issues.append(
            {"check": check, "severity": severity, "detail": detail, "count": int(count)}
        )

    n_rows = len(df)
    groups = df[CANONICAL_GROUP].unique().tolist() if n_rows else []

    n_nat = int(df[CANONICAL_TIME].isna().sum())
    if n_nat:
        add("timestamp_parse", "error", f"{n_nat} unparseable timestamps", n_nat)

    ok = df.dropna(subset=[CANONICAL_TIME]).copy()
    dup_pairs = int(ok.duplicated(subset=[CANONICAL_GROUP, CANONICAL_TIME]).sum())
    if dup_pairs:
        add("duplicates", "warning", "duplicate group/timestamp pairs", dup_pairs)

    non_mono_groups = []
    for name, g in ok.groupby(CANONICAL_GROUP, sort=True):
        ts = g[CANONICAL_TIME].to_numpy()
        if not bool((np.diff(ts.astype("datetime64[ns]").astype(np.int64)) > 0).all()):
            non_mono_groups.append(str(name))
    if non_mono_groups:
        add("monotonic_order", "error", f"non-monotonic groups: {non_mono_groups[:5]}",
            len(non_mono_groups))

    n_missing_target = int(ok[CANONICAL_TARGET].isna().sum())
    if n_missing_target:
        add("missing_target", "warning", "missing target values", n_missing_target)
    n_inf = int(np.isinf(ok[CANONICAL_TARGET].to_numpy(dtype=float)).sum())
    if n_inf:
        add("infinite_target", "error", "infinite target values", n_inf)
    n_neg = int((ok[CANONICAL_TARGET] < 0).sum())
    if n_neg and not mapping.allow_negative_target:
        add("negative_target", "error", "negative demand values", n_neg)

    gap_total, irregular_total = 0, 0
    freq_seen: set[str] = set()
    for _, g in ok.groupby(CANONICAL_GROUP, sort=True):
        ts = g[CANONICAL_TIME].sort_values().reset_index(drop=True)
        if len(ts) < 3:
            continue
        diffs = ts.diff().dropna()
        label, irregular = _freq_label(diffs)
        freq_seen.add(label)
        irregular_total += irregular
        step = diffs.mode().iloc[0]
        expected = int((ts.iloc[-1] - ts.iloc[0]) / step) + 1
        gap_total += max(0, expected - len(ts))
    if irregular_total:
        add("irregular_intervals", "warning",
            f"non-modal sampling steps (modal freqs: {sorted(freq_seen)})",
            irregular_total)
    if gap_total:
        add("missing_timestamps", "warning", "absent timestamps on regular grid",
            gap_total)

    for canon in mapping.exogenous():
        if canon in ok.columns and not pd.api.types.is_numeric_dtype(ok[canon]):
            add("feature_dtype", "error", f"non-numeric exogenous {canon!r}", 0)

    errors = [i for i in issues if i["severity"] == "error"]
    report = {
        "rows": n_rows,
        "n_groups": len(groups),
        "groups": [str(g) for g in groups],
        "modal_frequencies": sorted(freq_seen),
        "n_issues": len(issues),
        "n_errors": len(errors),
        "passed": len(errors) == 0,
        "issues": issues,
    }
    logger.info("Validation: %d issues (%d errors)", len(issues), len(errors))
    return report
