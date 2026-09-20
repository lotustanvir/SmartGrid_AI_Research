from __future__ import annotations

import pandas as pd

from src.data.pjm_integration import (
    aggregate_load_by_pjm_total,
    build_pjm_dataset,
    filter_and_standardize_rto_series,
    merge_pjm_components,
)


def test_load_aggregation_correctness():
    df = pd.DataFrame(
        {
            "datetime_beginning_utc": [
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 02:00:00",
            ],
            "zone": ["A", "A", "B", "A"],
            "load_area": ["AA", "AB", "BA", "AC"],
            "mw": [100.0, 50.0, 75.0, 25.0],
        }
    )
    out, report = aggregate_load_by_pjm_total(df)
    assert report["number_of_zones"] == 2
    assert report["number_of_load_areas"] == 4
    assert report["duplicate_timestamps"] == 1
    assert report["missing_mw_values"] == 0
    assert out.loc[out["timestamp"] == pd.Timestamp("2020-01-01 00:00:00"), "pjm_load_mw"].item() == 100.0
    assert out.loc[out["timestamp"] == pd.Timestamp("2020-01-01 01:00:00"), "pjm_load_mw"].item() == 125.0


def test_rto_filtering_and_timestamp_standardization():
    df = pd.DataFrame(
        {
            "datetime_beginning_utc": [
                "2020-01-01 02:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 00:00:00",
            ],
            "area": ["RTO", "WEST", "RTO", "RTO"],
            "wind_generation_mw": [12.0, 5.0, 10.0, 8.0],
        }
    )
    out = filter_and_standardize_rto_series(df, "wind_generation_mw")
    assert list(out.columns) == ["timestamp", "wind_generation_mw"]
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].nunique() == len(out)
    assert out["wind_generation_mw"].tolist() == [8.0, 10.0, 12.0]


def test_merge_correctness_and_schema():
    load = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            "pjm_load_mw": [100.0, 120.0],
        }
    )
    wind = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            "wind_generation_mw": [10.0, 11.0],
        }
    )
    solar = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            "solar_generation_mw": [2.0, 3.0],
        }
    )
    merged, stats = merge_pjm_components(load, wind, solar)
    assert list(merged.columns) == [
        "timestamp",
        "pjm_load_mw",
        "wind_generation_mw",
        "solar_generation_mw",
    ]
    assert stats["before_merge_rows"] == 2
    assert stats["after_merge_rows"] == 2
    assert merged["pjm_load_mw"].tolist() == [100.0, 120.0]


def test_missing_value_detection_and_dataset_build(tmp_path):
    raw_dir = tmp_path / "raw" / "pjm"
    raw_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "datetime_beginning_utc": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "datetime_beginning_ept": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "nerc_region": ["RFC", "RFC"],
            "mkt_region": ["MIDATL", "WEST"],
            "zone": ["AE", "AEP"],
            "load_area": ["AECO", "AEPAPT"],
            "mw": [100.0, None],
            "is_verified": [True, True],
        }
    ).to_csv(raw_dir / "hrl_load_metered.csv", index=False)

    pd.DataFrame(
        {
            "datetime_beginning_utc": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "datetime_beginning_ept": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "area": ["RTO", "RTO"],
            "wind_generation_mw": [10.0, 11.0],
        }
    ).to_csv(raw_dir / "wind_gen.csv", index=False)

    pd.DataFrame(
        {
            "datetime_beginning_utc": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "datetime_beginning_ept": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "area": ["RTO", "RTO"],
            "solar_generation_mw": [2.0, 3.0],
        }
    ).to_csv(raw_dir / "solar_gen.csv", index=False)

    out = build_pjm_dataset(raw_dir, tmp_path / "processed" / "pjm_smart_grid_2020.csv")
    assert "timestamp" in out.columns
    assert out["pjm_load_mw"].isna().sum() == 0
    assert out["wind_generation_mw"].isna().sum() == 0
    assert out["solar_generation_mw"].isna().sum() == 0
