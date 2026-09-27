"""Weather acquisition/aggregation skeleton (Phase 1C).

Loads ERA5/NOAA weather, aggregates to the PJM footprint, derives thermal
features, and merges onto PJM timestamps. SKELETON ONLY: every method is a
documented stub raising NotImplementedError. No downloads, no disk writes,
no pipeline wiring until Phase 1D+ (see docs/PHASE1B_WEATHER_DESIGN.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class WeatherProcessor:
    """PJM footprint weather builder (stub).

    Config source: ``configs/weather.yaml``. All timestamps UTC hourly to
    match the PJM ``timestamp`` semantics verified in Phase 1A.
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """Args: config_path: path to ``configs/weather.yaml`` (default resolved)."""
        # TODO(Phase 1D): load configs/weather.yaml (request JSONs, weights version, gates).
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def load_era5(self) -> pd.DataFrame:
        """Load raw ERA5 pulls into a long frame.

        Returns: DataFrame with [timestamp, cell_id, <era5 variables>].
        """
        # TODO(Phase 1D): read data/raw/weather/era5/ pulls listed in configs/weather.yaml.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def load_noaa(self) -> pd.DataFrame:
        """Load NOAA station observations (validation source only).

        Returns: DataFrame with [timestamp, station_id, <noaa variables>].
        """
        # TODO(Phase 1D): read data/raw/weather/noaa/ pulls + station inventory.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def aggregate_population_weighted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate gridded/station weather to one footprint series.

        Args: df: long frame from load_era5()/load_noaa().
        Returns: hourly footprint-mean frame using fixed census weights.
        """
        # TODO(Phase 1D): apply weights_version from configs/weather.yaml.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def aggregate_simple_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unweighted footprint mean (sensitivity comparison for Option 1).

        Args: df: long frame from load_era5()/load_noaa().
        Returns: hourly simple-mean frame, same schema as population-weighted.
        """
        # TODO(Phase 1D): implement alongside aggregate_population_weighted.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def calculate_heat_index(
        self, temp_c: pd.Series, rh: pd.Series
    ) -> pd.Series:
        """Rothfusz heat index from temperature (°C) and relative humidity (%).

        Args: temp_c: 2m temperature. rh: relative humidity.
        Returns: heat-index series (NaN outside gated validity regime).
        """
        # TODO(Phase 1D): implement Rothfusz + dew-point validity gate.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def calculate_CDH(self, temp_c: pd.Series) -> pd.Series:
        """Cooling Degree Hours: max(T-22, 0) per hour.

        Args: temp_c: 2m temperature (°C).
        Returns: CDH series (float, >= 0).
        """
        # TODO(Phase 1D): elementwise max(temp_c - 22.0, 0.0).
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def calculate_HDH(self, temp_c: pd.Series) -> pd.Series:
        """Heating Degree Hours: max(18-T, 0) per hour.

        Args: temp_c: 2m temperature (°C).
        Returns: HDH series (float, >= 0).
        """
        # TODO(Phase 1D): elementwise max(18.0 - temp_c, 0.0).
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def merge_with_pjm(
        self, weather: pd.DataFrame, pjm: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge footprint weather onto PJM timestamps (leakage-safe).

        Rule: UTC hourly, backward merge-asof with 1h tolerance; unmatched
        hours counted into weather_coverage.json (never silent).
        Args: weather: footprint frame. pjm: PJM timestamp frame.
        Returns: merged frame + coverage accounting.
        """
        # TODO(Phase 1D): pd.merge_asof(backward, tolerance=1h) + coverage log.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")

    def save_weather_dataset(self, df: pd.DataFrame, path: str | Path) -> Path:
        """Persist the footprint weather dataset + record hash.

        Args: df: final weather frame. path: destination under data/processed/v2/.
        Returns: written path.
        """
        # TODO(Phase 1D): to_csv + sha256 into data/checksums/sha256.txt.
        raise NotImplementedError("Phase 1C skeleton — implemented in Phase 1D.")
