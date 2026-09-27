"""Weather quality-control skeleton (Phase 1C).

Validates footprint weather before any column enters the v2 dataset.
SKELETON ONLY: documented stubs raising NotImplementedError. Gates and
thresholds come from configs/weather.yaml. Report schema feeds
results/q1/weather_validation_report.json. No validation executed yet
(see docs/PHASE1B_WEATHER_DESIGN.md section 7).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_PATH = Path("results/q1/weather_validation_report.json")


class WeatherValidator:
    """Gate-keeper for weather columns (stub)."""

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """Args: config_path: path to ``configs/weather.yaml`` (thresholds)."""
        # TODO(Phase 1D): load validation thresholds from configs/weather.yaml.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def check_missing(self, df: pd.DataFrame) -> dict:
        """Missing percentage per variable; ERA5 gate is 0%.

        Args: df: footprint weather frame.
        Returns: {variable: missing_pct} + gate verdict.
        """
        # TODO(Phase 1D): per-column isna().mean() vs missing_threshold.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def check_timestamp_continuity(self, df: pd.DataFrame) -> dict:
        """Gap scan vs expected UTC hourly grid 2020-01-01 05:00 → 2026-01-01 04:00.

        Args: df: footprint weather frame.
        Returns: {"gaps": [...], "coverage_pct": float}.
        """
        # TODO(Phase 1D): reindex to expected grid, list missing hours.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def check_physical_range(self, df: pd.DataFrame) -> dict:
        """Impossible-value quarantine (T/RH/dew-point/wind/cloud ranges).

        Args: df: footprint weather frame.
        Returns: {variable: violation_count} + quarantined index list.
        """
        # TODO(Phase 1D): range checks from configs/weather.yaml validation block.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def check_spatial_coverage(self, df: pd.DataFrame) -> dict:
        """Contributing cells/stations per hour (dropout accounting).

        Args: df: footprint weather frame (with coverage_flag where present).
        Returns: {"hours_below_full_coverage": int, "details": [...]}.
        """
        # TODO(Phase 1D): aggregate coverage_flag / station inventory join.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def detect_outliers(self, df: pd.DataFrame) -> dict:
        """|z| > 5 vs 30-day rolling climatology (flag, never auto-delete).

        Args: df: footprint weather frame.
        Returns: flagged (timestamp, variable, z) list for review.
        """
        # TODO(Phase 1D): rolling climatology + cross-check vs event definitions.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def generate_report(self, df: pd.DataFrame, path: str | Path = REPORT_PATH) -> Path:
        """Run all gates and write results/q1/weather_validation_report.json.

        Args: df: footprint weather frame. path: report destination.
        Returns: written path. Fail-gate blocks v2 build for failed variables.
        """
        # TODO(Phase 1D): collect check_* outputs + gate verdict → JSON.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")
