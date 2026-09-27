"""Canonical Q1 TFT execution path (Phase 2).

One path only: encoder_length=168, prediction_length in {1,6,24},
train/validation fit ONLY, best-validation checkpoint restored for test
prediction. Weather is NEVER a decoder-known feature (knowns = calendar
only: time_idx/hour/day_of_week + Q1 aliases). Test data never enters
fitting. Legacy test-inclusive paths (run_phase_6b.py) are retired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX, validate_frame
from src.q1.protocol import HISTORY_LENGTH, HORIZONS

# Calendar-only knowns (Q1 aliases included). Everything else numeric is
# encoder-only unknown.
Q1_KNOWN = ["time_idx", "hour", "day_of_week", "dow", "dom", "month", "woy", "weekend"]


def build_q1_tft_frame(
    feat: pd.DataFrame,
    feature_names: list[str],
    target_col: str = "target",
) -> pd.DataFrame:
    """Map canonical Q1 frame to a TFT long frame (single group).

    - time_idx: 0..n-1 chronological
    - group_id: 'single'
    - electricity_demand: target
    - knowns: calendar subset present (never weather)
    - unknowns: all other canonical features (lags/rolling/weather-lags)
    Fails closed if weather raw cols are passed as knowns.
    """
    if target_col not in feat.columns:
        raise ValueError(f"Missing target {target_col!r}")
    raw_weather = {
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
        "solar_radiation",
        "wind_generation_mw",
        "solar_generation_mw",
        "total_renewable",
        "cdh",
        "hdh",
        "net_load",
        "renewable_penetration",
        "pjm_load_mw",
    }
    if raw_weather.intersection(feature_names):
        raise ValueError(
            f"Raw weather/target leak in TFT inputs: "
            f"{sorted(raw_weather.intersection(feature_names))}"
        )
    work = feat.sort_values("timestamp").reset_index(drop=True) if "timestamp" in feat.columns else feat.reset_index(drop=True)
    known = [c for c in Q1_KNOWN if c in feature_names or c in work.columns]
    # Restrict knowns to calendar/time only (never lag/weather).
    known = [c for c in known if not c.startswith(("lag_", "rolling_", "temp_", "hum_", "ws_", "cloud", "solrad", "wind_", "solar_", "totren", "cdh", "hdh"))]
    unknown = [c for c in feature_names if c not in known]
    need = [c for c in known + unknown if c != target_col]
    clean = work.dropna(subset=need + [target_col]).reset_index(drop=True)
    out = pd.DataFrame(
        {
            TIME_IDX: np.arange(len(clean)),
            GROUP_ID: "single",
            TARGET: clean[target_col].to_numpy(dtype=float),
        }
    )
    for c in known + unknown:
        out[c] = clean[c].to_numpy(dtype=float)
    return validate_frame(out)


def make_q1_tft_config(horizon: int, seed: int = 42, **overrides) -> dict:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    cfg = {
        "encoder_length": HISTORY_LENGTH,
        "prediction_length": int(horizon),
        "random_state": int(seed),
    }
    cfg.update(overrides)
    return cfg


def assert_fit_frame(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_time_min: int | None = None,
) -> None:
    """Fail closed unless the TFT fit frame is train/val-only and ordered.

    Verifies on ``TIME_IDX`` (original frame coordinates, before any
    reindexing): validation is strictly after training, and — when
    ``test_time_min`` is given — the whole fit frame strictly precedes
    test data. Replaces the previous vacuous contiguity check.
    """
    for name, frame in (("train_df", train_df), ("val_df", val_df)):
        if TIME_IDX not in frame.columns or len(frame) == 0:
            raise ValueError(f"Q1 TFT fit frame {name!r} missing/empty {TIME_IDX!r}.")
    train_max = int(train_df[TIME_IDX].max())
    val_min = int(val_df[TIME_IDX].min())
    val_max = int(val_df[TIME_IDX].max())
    if not train_max < val_min:
        raise ValueError(
            f"Q1 TFT fit frame not chronological: train_max={train_max} "
            f"vs val_min={val_min}. Validation must be strictly after training."
        )
    if test_time_min is not None and not val_max < int(test_time_min):
        raise ValueError(
            f"Q1 TFT fit frame overlaps test data: val_max={val_max} vs "
            f"test_time_min={test_time_min}. Test data must NEVER enter fitting."
        )


def fit_q1_tft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    seed: int = 42,
    checkpoint_dir: Optional[str | Path] = None,
    test_time_min: int | None = None,
    **overrides,
):
    """Fit TFT on train+val ONLY; returns fitted forecaster (best ckpt restored)."""
    from src.models.tft import TemporalFusionTransformerForecaster

    cfg = make_q1_tft_config(horizon, seed, **overrides)
    if checkpoint_dir is not None:
        cfg["checkpoint_dir"] = str(checkpoint_dir)
    # Fail closed BEFORE concatenation/reindexing: train/val order and
    # test exclusion are verified on original TIME_IDX coordinates.
    assert_fit_frame(train_df, val_df, test_time_min=test_time_min)
    full = pd.concat([train_df, val_df], ignore_index=True)
    full[TIME_IDX] = np.arange(len(full))
    full = validate_frame(full)
    model = TemporalFusionTransformerForecaster(**cfg)
    model.fit(full)
    if model.best_checkpoint_ is None:
        raise RuntimeError("TFT fit produced no best checkpoint (validation-only).")
    return model
