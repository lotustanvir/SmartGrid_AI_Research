"""Placeholder weather-pipeline tests (Phase 1C skeleton — all TODO, none executed).

Executed from Phase 1D onward, after configs/weather.yaml is populated and
footprint data exists. Markers: each test documents its future gate.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="TODO Phase 1D: weather pipeline not yet implemented.")


def test_weather_config_exists():
    """TODO: configs/weather.yaml exists with era5/noaa/aggregation/validation blocks."""
    assert Path("configs/weather.yaml").is_file()


def test_weather_columns_schema():
    """TODO: pjm_weather_2020_2025.csv matches WEATHER_CARD section 4 schema (14 cols)."""
    assert False, "TODO Phase 1D"


def test_timestamp_format():
    """TODO: weather timestamps UTC hourly, aligned to PJM timestamp semantics."""
    assert False, "TODO Phase 1D"


def test_temperature_range():
    """TODO: temperature_2m within [-40, 45] °C per validation gate."""
    assert False, "TODO Phase 1D"
