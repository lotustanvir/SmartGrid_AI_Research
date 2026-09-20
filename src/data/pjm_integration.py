from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)


def _hourly_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).floor("h")
    end = pd.Timestamp(end).ceil("h")
    return pd.date_range(start=start, end=end, freq="h")


def aggregate_load_by_pjm_total(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate each hourly PJM load observation across all valid load areas."""
    raw = df.copy()
    raw["datetime_beginning_utc"] = _to_datetime_utc(raw["datetime_beginning_utc"])
    raw["mw"] = pd.to_numeric(raw.get("mw"), errors="coerce")

    report = {
        "number_of_zones": int(raw["zone"].nunique(dropna=True)) if "zone" in raw.columns else 0,
        "number_of_load_areas": int(raw["load_area"].nunique(dropna=True)) if "load_area" in raw.columns else 0,
        "duplicate_timestamps": int(raw["datetime_beginning_utc"].duplicated().sum()),
        "missing_mw_values": int(raw["mw"].isna().sum()),
    }

    valid = raw.dropna(subset=["datetime_beginning_utc", "mw", "load_area"]).copy()
    hourly = (
        valid.groupby("datetime_beginning_utc", as_index=False)["mw"]
        .sum()
        .rename(columns={"datetime_beginning_utc": "timestamp", "mw": "pjm_load_mw"})
    )
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], errors="coerce")
    hourly = hourly.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return hourly, report


def filter_and_standardize_rto_series(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Keep only RTO entries, standardize timestamps, and ensure hourly continuity."""
    if feature_name not in {"wind_generation_mw", "solar_generation_mw"}:
        raise ValueError(f"Unsupported feature_name: {feature_name}")

    raw = df.copy()
    if "datetime_beginning_utc" not in raw.columns:
        raise ValueError("Raw PJM file must contain datetime_beginning_utc column")
    raw["timestamp"] = _to_datetime_utc(raw["datetime_beginning_utc"])
    raw[feature_name] = pd.to_numeric(raw[feature_name], errors="coerce")

    rto = raw[raw.get("area", pd.Series([None] * len(raw))).astype(str).str.upper().eq("RTO")].copy()
    rto = rto.dropna(subset=["timestamp", feature_name]).copy()
    rto = rto.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

    ts_all = _hourly_range(rto["timestamp"].min(), rto["timestamp"].max())
    continuity = pd.DataFrame({"timestamp": ts_all})
    merged = continuity.merge(rto[["timestamp", feature_name]], on="timestamp", how="left")
    merged = merged.rename(columns={feature_name: feature_name})

    if merged[feature_name].isna().any():
        missing_hours = int(merged[feature_name].isna().sum())
    else:
        missing_hours = 0

    out = merged[["timestamp", feature_name]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def merge_pjm_components(
    load_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    solar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge the PJM load, wind, and solar tables on timestamp and report row loss."""
    before_rows = len(load_df)
    merged = load_df.merge(wind_df, on="timestamp", how="inner")
    merge_stats = {"before_merge_rows": int(before_rows), "after_merge_rows": int(len(merged))}
    merged = merged.merge(solar_df, on="timestamp", how="inner")
    merge_stats["after_merge_rows"] = int(len(merged))
    merged = merged[["timestamp", "pjm_load_mw", "wind_generation_mw", "solar_generation_mw"]].sort_values("timestamp").reset_index(drop=True)
    return merged, merge_stats


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _generate_eda_for_pjm(df: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.suptitle("EXPLORATORY — NOT FOR PUBLICATION", fontsize=8, color="red")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=110)
        plt.close(fig)
        saved.append(str(out_dir / name))

    ts = df["timestamp"]
    load = df["pjm_load_mw"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(ts, load, linewidth=0.8)
    ax.set_title("Load time-series plot")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load (MW)")
    save(fig, "load_time_series.png")

    daily = df.assign(hour=ts.dt.hour).groupby("hour")["pjm_load_mw"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    daily.plot(ax=ax, marker="o")
    ax.set_title("Daily average load profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average load (MW)")
    save(fig, "daily_average_load_profile.png")

    wind = df["wind_generation_mw"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ts, wind, linewidth=0.8)
    ax.set_title("Wind generation profile")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Wind generation (MW)")
    save(fig, "wind_generation_profile.png")

    solar = df["solar_generation_mw"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ts, solar, linewidth=0.8)
    ax.set_title("Solar generation profile")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Solar generation (MW)")
    save(fig, "solar_generation_profile.png")

    corr = df[["pjm_load_mw", "wind_generation_mw", "solar_generation_mw"]].corr()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[1]), corr.columns, rotation=20)
    ax.set_yticks(range(corr.shape[0]), corr.columns)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax)
    save(fig, "correlation_matrix.png")

    return saved


def build_pjm_dataset(raw_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    load_raw = pd.read_csv(raw_dir / "hrl_load_metered.csv")
    wind_raw = pd.read_csv(raw_dir / "wind_gen.csv")
    solar_raw = pd.read_csv(raw_dir / "solar_gen.csv")

    load_df, load_report = aggregate_load_by_pjm_total(load_raw)
    wind_df = filter_and_standardize_rto_series(wind_raw, "wind_generation_mw")
    solar_df = filter_and_standardize_rto_series(solar_raw, "solar_generation_mw")

    if len(load_df) == 0:
        raise ValueError("No PJM load records remain after aggregation")

    before_rows = len(load_df)
    merged, merge_stats = merge_pjm_components(load_df, wind_df, solar_df)
    merge_stats["before_merge_rows"] = before_rows
    merge_stats["after_merge_rows"] = int(len(merged))

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if not merged["timestamp"].is_monotonic_increasing:
        raise ValueError("Merged timestamps are not sorted ascending")
    if merged["timestamp"].duplicated().any():
        raise ValueError("Merged dataset contains duplicate timestamps")

    missing_values = merged[["pjm_load_mw", "wind_generation_mw", "solar_generation_mw"]].isna().sum().to_dict()
    dup_count = int(merged["timestamp"].duplicated().sum())
    summary = {
        "Dataset": "PJM 2020",
        "Date range": {
            "start": merged["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": merged["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "Number of rows": int(len(merged)),
        "Sampling frequency": "H",
        "Missing values": missing_values,
        "Duplicate count": dup_count,
        "Feature statistics": {
            "load": {
                "min": float(merged["pjm_load_mw"].min()),
                "max": float(merged["pjm_load_mw"].max()),
                "mean": float(merged["pjm_load_mw"].mean()),
            },
            "wind": {
                "min": float(merged["wind_generation_mw"].min()),
                "max": float(merged["wind_generation_mw"].max()),
                "mean": float(merged["wind_generation_mw"].mean()),
            },
            "solar": {
                "min": float(merged["solar_generation_mw"].min()),
                "max": float(merged["solar_generation_mw"].max()),
                "mean": float(merged["solar_generation_mw"].mean()),
            },
        },
        "load_aggregation_report": load_report,
        "merge_statistics": merge_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    validation_path = Path("results/data_validation") / "pjm_2020_report.json"
    _write_json(validation_path, summary)

    eda_dir = Path("results/eda/pjm")
    _generate_eda_for_pjm(merged, eda_dir)

    return merged


if __name__ == "__main__":
    build_pjm_dataset(
        Path("dataset/raw/pjm"),
        Path("dataset/processed/pjm_smart_grid_2020.csv"),
    )
