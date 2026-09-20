"""Dataset mapping: provider column names -> canonical frame (Phase 6A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

CANONICAL_TARGET = "target"
CANONICAL_TIME = "timestamp"
CANONICAL_GROUP = "group_id"
SINGLE_GROUP = "single"


@dataclass
class DataMapping:
    """Maps one provider CSV onto canonical names (no hardcoding)."""

    path: str
    timestamp_col: str = "timestamp"
    target_col: str = "target"
    group_col: Optional[str] = None
    temperature_col: Optional[str] = None
    humidity_col: Optional[str] = None
    solar_col: Optional[str] = None
    wind_col: Optional[str] = None
    expected_freq: Optional[str] = None  # e.g. "h"; None = infer
    allow_negative_target: bool = False
    max_target_gap: int = 3  # periods; longer target gaps are NOT filled

    def exogenous(self) -> dict[str, str]:
        """Canonical -> provider column for present exogenous signals."""
        out = {}
        for canon, col in (
            ("temperature", self.temperature_col),
            ("humidity", self.humidity_col),
            ("solar", self.solar_col),
            ("wind", self.wind_col),
        ):
            if col:
                out[canon] = col
        return out

    def required_columns(self) -> list[str]:
        """Provider columns that must exist in the CSV."""
        cols = [self.timestamp_col, self.target_col]
        if self.group_col:
            cols.append(self.group_col)
        cols.extend(self.exogenous().values())
        return cols


@dataclass
class CleaningConfig:
    """How cleaning may modify data (everything is reported)."""

    exogenous_interp_method: str = "time"  # time | linear | none
    exogenous_interp_limit: int = 6
    allow_ffill_bfill: bool = False
    drop_target_na: bool = True


@dataclass
class FeatureConfig:
    """Which feature families to build (all group-wise, leakage-safe)."""

    calendar: bool = True
    cyclical: bool = True
    lags: tuple[int, ...] = (1, 24, 168)
    rolling_windows: tuple[int, ...] = (24, 168)


def default_config_path() -> Path:
    """Absolute path to ``configs/data.yaml`` (repo-root relative)."""
    return Path(__file__).resolve().parents[2] / "configs" / "data.yaml"


def load_data_config(
    path: Optional[str | Path] = None,
) -> tuple[DataMapping, CleaningConfig, FeatureConfig, dict]:
    """Load mapping + cleaning/feature options; return (mapping, clean, feat, raw)."""
    cfg_path = Path(path) if path is not None else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Data config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get("data", {})
    weather = data.get("weather", {})
    renewable = data.get("renewable", {})
    mapping = DataMapping(
        path=data.get("path", "dataset/raw/demand.csv"),
        timestamp_col=data.get("timestamp_col", "timestamp"),
        target_col=data.get("target_col", "target"),
        group_col=data.get("group_col"),
        temperature_col=weather.get("temperature"),
        humidity_col=weather.get("humidity"),
        solar_col=renewable.get("solar"),
        wind_col=renewable.get("wind"),
        expected_freq=data.get("expected_freq"),
        allow_negative_target=bool(data.get("allow_negative_target", False)),
        max_target_gap=int(data.get("max_target_gap", 3)),
    )
    cleaning = raw.get("cleaning", {})
    clean_cfg = CleaningConfig(
        exogenous_interp_method=cleaning.get("exogenous_interp_method", "time"),
        exogenous_interp_limit=int(cleaning.get("exogenous_interp_limit", 6)),
        allow_ffill_bfill=bool(cleaning.get("allow_ffill_bfill", False)),
        drop_target_na=bool(cleaning.get("drop_target_na", True)),
    )
    features = raw.get("features", {})
    feat_cfg = FeatureConfig(
        calendar=bool(features.get("calendar", True)),
        cyclical=bool(features.get("cyclical", True)),
        lags=tuple(features.get("lags", [1, 24, 168])),
        rolling_windows=tuple(features.get("rolling_windows", [24, 168])),
    )
    return mapping, clean_cfg, feat_cfg, raw


# Canonical feature names produced by features.py (for metadata/adapters).
CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
]
CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def lag_name(lag: int) -> str:
    return f"lag_{lag}"


def rolling_names(window: int) -> tuple[str, str]:
    return f"rolling_mean_{window}", f"rolling_std_{window}"
