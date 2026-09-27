"""ERA5 -> PJM footprint weather pipeline (Phase 1D-C3, STEPS 3-7).

Reads monthly ERA5 NetCDF (5-var + ssrd supplements), validates, aggregates
to the PJM footprint (boundary-masked simple mean; population weights hook),
derives thermal/solar features, merges onto the v1 PJM timestamps, and writes
the v2 weather-enhanced dataset. Idempotent: re-runnable as more months land.

Reads (never modifies):
  data/raw/weather/era5/monthly/era5_pjm_<Y>_<M>.nc         (t2m/d2m/u10/v10/tcc)
  data/raw/weather/era5/monthly/era5_pjm_<Y>_<M>_ssrd.nc    (ssrd supplement)
  data/raw/external/pjm_boundary.geojson                    (footprint polygon)
  dataset/processed/pjm_smart_grid_2020_2025.csv            (v1, READ-ONLY)

Writes:
  results/q1/weather_validation_report.json                 (STEP 3 aggregate)
  results/q1/era5_validation_<Y>_<M>.json                   (per-month, new months)
  data/processed/v2/pjm_footprint_weights.csv               (STEP 4 weights)
  data/processed/v2/pjm_footprint_coverage.json             (STEP 4 coverage)
  data/processed/pjm_weather_2020_2025.csv                  (STEP 4 footprint series)
  dataset/processed/pjm_smart_grid_2020_2025_v2.csv         (STEP 6 merged v2)
  results/data_validation/dataset_metadata_v2.json          (STEP 7 quality report)

Physical conversions (documented, deterministic):
  temperature_c = t2m - 273.15 ; dewpoint_c = d2m - 273.15
  cloud_cover   = tcc/100 if observed max > 1.5 else tcc (ERA5 native = %)
  wind_speed    = sqrt(u10^2 + v10^2)
  relative_humidity via Magnus (Alduchov-Eskridge) from T/Td (degC), clipped 0-100
  solar_radiation_wm2 = max(0, ssrd(t)-ssrd(t-1h))/3600  (ssrd accumulated
    since 00 UTC; midnight reset diff clips to 0 = correct at night)
  heat_index_c  = Rothfusz (NWS) on (T_F, RH); applied iff T_F>=80 and RH>=40
    else heat_index = temperature (documented gating, WEATHER_CARD S3)
  wet_bulb_c    = Stull (2011) from (T_C, RH)
  cdh = max(T-22,0) ; hdh = max(18-T,0)  (PHASE1B S1.2)
  daylight_proxy = max(0, cos(solar_zenith)) at footprint mean latitude
    (timestamp-derived, no weather input; known-future class per PROTOCOL)

Aggregation (docs/PJM_WEATHER_FOOTPRINT.md methodology lock):
  inclusion = ERA5 cell centroid inside boundary polygon (ray casting).
  primary published series = boundary-masked SIMPLE mean (Method A).
  Deviation from lock (owned, not hidden): population product (GPW/WorldPop)
  was never acquired (configs/pjm_weather_weights.yaml still TODO), so
  Method B (population-weighted) cannot be computed. w_pop column ships as
  NaN with the reason recorded; code path ready once weights land.
  Sensitivity column temperature_simplemean is therefore identical to
  temperature (same mask) -- recorded explicitly, not silently.

Merge (leakage-safe): pd.merge_asof(backward, tolerance=1h) of footprint
weather onto v1 timestamps; unmatched hours -> NaN + counted in coverage
(never silent inner-join drop). No forward fill. No model training.

Usage (from repo root):
    python scripts/process_era5.py --validate-only
    python scripts/process_era5.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MONTHLY_DIR = REPO_ROOT / "data" / "raw" / "weather" / "era5" / "monthly"
BOUNDARY_PATH = REPO_ROOT / "data" / "raw" / "weather" / "era5" / "monthly"
BOUNDARY_FILE = REPO_ROOT / "data" / "raw" / "external" / "pjm_boundary.geojson"
V1_PATH = REPO_ROOT / "dataset" / "processed" / "pjm_smart_grid_2020_2025.csv"
V2_DIR = REPO_ROOT / "data" / "processed" / "v2"
WEATHER_OUT = REPO_ROOT / "data" / "processed" / "pjm_weather_2020_2025.csv"
V2_OUT = REPO_ROOT / "dataset" / "processed" / "pjm_smart_grid_2020_2025_v2.csv"
VALIDATION_REPORT = REPO_ROOT / "results" / "q1" / "weather_validation_report.json"
Q1_DIR = REPO_ROOT / "results" / "q1"
META_V2 = REPO_ROOT / "results" / "data_validation" / "dataset_metadata_v2.json"

SHORT2RAW = {"t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature",
             "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
             "tcc": "total_cloud_cover", "ssrd": "surface_solar_radiation_downwards"}
FULL5 = ["t2m", "d2m", "u10", "v10", "tcc"]

K2C = 273.15
FOOTPRINT_LAT_FALLBACK = 40.5  # replaced by actual included-cell mean lat


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- footprints
def load_polygons(path: Path) -> list[list[list[list[float]]]]:
    """Return list of polygons; each polygon = list of rings; ring = [lon,lat]."""
    gj = json.loads(path.read_text(encoding="utf-8"))
    geoms = []
    if gj["type"] == "FeatureCollection":
        for f in gj["features"]:
            geoms.append(f["geometry"])
    elif gj["type"] == "Feature":
        geoms.append(gj["geometry"])
    else:
        geoms.append(gj)
    polys: list[list[list[list[float]]]] = []
    for g in geoms:
        if g["type"] == "Polygon":
            polys.append(g["coordinates"])
        elif g["type"] == "MultiPolygon":
            polys.extend(g["coordinates"])
    return polys


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            xinters = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < xinters:
                inside = not inside
        j = i
    return inside


def point_in_polygon(lon: float, lat: float, poly) -> bool:
    if not point_in_ring(lon, lat, poly[0]):
        return False
    for hole in poly[1:]:
        if point_in_ring(lon, lat, hole):
            return False
    return True


def ring_bbox(ring) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def build_mask(lats, lons, polys) -> "object":
    import numpy as np
    # Precompute outer-ring bboxes so most cells reject in O(1).
    bboxes = [ring_bbox(poly[0]) for poly in polys]
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, la in enumerate(lats):
        la = float(la)
        for j, lo in enumerate(lons):
            lo = float(lo)
            for poly, bb in zip(polys, bboxes):
                if not (bb[0] <= lo <= bb[2] and bb[1] <= la <= bb[3]):
                    continue
                if point_in_polygon(lo, la, poly):
                    mask[i, j] = True
                    break
    return mask


# ------------------------------------------------------------- derivations
def rh_magnus(t_c, td_c):
    import numpy as np
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
    rh = 100.0 * e / es
    return np.clip(rh, 0.0, 100.0)


def heat_index_c(t_c, rh):
    """Rothfusz NWS heat index (degC). Gated: applied iff T>=26.7degC(80degF)
    and RH>=40; else returns air temperature. Includes NWS low/high-RH
    adjustments."""
    import numpy as np
    t_c = np.asarray(t_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    tf = t_c * 9.0 / 5.0 + 32.0
    hi_f = (-42.379 + 2.04901523 * tf + 10.14333127 * rh
            - 0.22475541 * tf * rh - 0.00683783 * tf * tf
            - 0.05481717 * rh * rh + 0.00122874 * tf * tf * rh
            + 0.00085282 * tf * rh * rh - 0.00000199 * tf * tf * rh * rh)
    low = (rh < 13) & (tf >= 80) & (tf <= 112)
    hi_f = np.where(low, hi_f - ((13 - rh) / 4.0)
                    * np.sqrt(np.clip((17 - np.abs(tf - 95.0)) / 17.0, 0.0, None)), hi_f)
    high = (rh > 85) & (tf >= 80) & (tf <= 87)
    hi_f = np.where(high, hi_f + ((rh - 85) / 10.0) * ((87 - tf) / 5.0), hi_f)
    gated = (tf >= 80.0) & (rh >= 40.0)
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return np.where(gated, hi_c, t_c)


def wet_bulb_stull(t_c, rh):
    """Stull (2011) wet-bulb approximation (degC)."""
    import numpy as np
    t = np.asarray(t_c, dtype=float)
    r = np.asarray(rh, dtype=float)
    return (t * np.arctan(0.151977 * np.sqrt(r + 8.313659))
            + np.arctan(t + r) - np.arctan(r - 1.676331)
            + 0.00391838 * np.power(r, 1.5) * np.arctan(0.023101 * r)
            - 4.686035)


def cos_zenith(ts_utc, lat_deg: float, lon_deg: float):
    """Cosine of solar zenith angle (negative at night). Spencer declination +
    hour-angle from SOLAR time (UTC + lon/15); timestamp-derived only
    (no weather input)."""
    import numpy as np
    import pandas as pd
    ts = pd.DatetimeIndex(ts_utc)
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour_utc = (ts.hour + ts.minute / 60.0 + ts.second / 3600.0).to_numpy(dtype=float)
    hour = hour_utc + lon_deg / 15.0  # local solar time
    gamma = 2.0 * math.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lat = math.radians(lat_deg)
    ha = np.radians((hour - 12.0) * 15.0)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)


# ------------------------------------------------------------------ loading
def available_months() -> list[tuple[str, str]]:
    out = []
    for p in sorted(MONTHLY_DIR.glob("era5_pjm_[0-9][0-9][0-9][0-9]_[0-9][0-9].nc")):
        if p.name.endswith("_ssrd.nc"):
            continue
        parts = p.stem.split("_")  # ['era5','pjm','YYYY','MM']
        if len(parts) == 4 and len(parts[2]) == 4 and len(parts[3]) == 2:
            out.append((parts[2], parts[3]))
    return out


def load_month_frame(year: str, month: str, mask, lats, lons):
    """Footprint-mean raw series for one month (5-var + ssrd supplement)."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    base = MONTHLY_DIR / f"era5_pjm_{year}_{month}.nc"
    sup = MONTHLY_DIR / f"era5_pjm_{year}_{month}_ssrd.nc"
    if not base.exists():
        return None, {"month": f"{year}-{month}", "status": "MISSING_BASE"}
    with xr.open_dataset(base) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        times = pd.DatetimeIndex(ds[tname].to_index()).tz_localize(None)
        rec = {"timestamp": times, "n_cells_total": int(mask.size),
               "n_cells_in": int(mask.sum())}
        for v in FULL5:
            arr = ds[v].to_numpy()
            with np.errstate(all="ignore"):
                rec[v] = np.nanmean(np.where(mask[None, :, :], arr, np.nan),
                                    axis=(1, 2))
        nan_pct = {v: float(np.isnan(rec[v]).mean()) for v in FULL5}
    ssrd_status = "MISSING_SUPPLEMENT"
    ssrd_accum = None
    if sup.exists():
        try:
            with xr.open_dataset(sup) as ds2:
                tname2 = "valid_time" if "valid_time" in ds2.coords else "time"
                t2 = pd.DatetimeIndex(ds2[tname2].to_index()).tz_localize(None)
                a = ds2["ssrd"].to_numpy()
                with np.errstate(all="ignore"):
                    s = np.nanmean(np.where(mask[None, :, :], a, np.nan), axis=(1, 2))
                if len(t2) == len(times) and (t2 == times).all():
                    ssrd_accum = s
                    ssrd_status = "OK"
                else:
                    ssrd_status = f"TIME_MISMATCH base={len(times)} sup={len(t2)}"
        except Exception as e:
            ssrd_status = f"READ_ERROR {type(e).__name__}: {str(e)[:150]}"
    df = pd.DataFrame({"timestamp": rec["timestamp"], **{v: rec[v] for v in FULL5}})
    return (df, ssrd_accum), {"month": f"{year}-{month}", "status": "OK",
                              "ssrd": ssrd_status, "nan_pct": nan_pct,
                              "n_hours": len(df),
                              "t_start": str(df['timestamp'].iloc[0]),
                              "t_end": str(df['timestamp'].iloc[-1])}


# --------------------------------------------------------------- validation
def validate_footprint(df, ssrd_rate) -> dict:
    import numpy as np
    import pandas as pd
    checks: dict = {}
    # 1. missing %
    miss = {c: float(df[c].isna().mean()) for c in
            ["t2m", "d2m", "u10", "v10", "tcc"]}
    miss["ssrd_rate"] = float(pd.Series(ssrd_rate).isna().mean())
    checks["missing_pct_raw"] = miss
    # 2. physical ranges on DERIVED footprint vars
    t = df["t2m"].to_numpy() - K2C
    td = df["d2m"].to_numpy() - K2C
    w = np.hypot(df["u10"].to_numpy(), df["v10"].to_numpy())
    viol = {"T_outside_-40_45": int(((t < -40) | (t > 45)).sum()),
            "dewpoint_gt_temp": int((td > t + 1e-9).sum()),
            "wind_negative": int((w < 0).sum())}
    checks["impossible_counts"] = viol
    # 3. gaps vs expected grid over available span
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    gaps = full.difference(df["timestamp"])
    checks["gaps"] = {"n_gaps": len(gaps),
                      "first_20": [str(g) for g in gaps[:20]]}
    # 4. outliers |z|>5 vs 30-day rolling climatology (temperature only, flags)
    s = pd.Series(t, index=df["timestamp"])
    roll = s.rolling("720h", min_periods=168, center=True)
    z = (s - roll.mean()) / roll.std()
    flags = z[np.abs(z) > 5].dropna()
    checks["outlier_flags_T"] = {"n": int(len(flags)),
                                 "first_20": [(str(i), round(float(v), 2))
                                              for i, v in flags.iloc[:20].items()]}
    return checks


# -------------------------------------------------------------------- main
def build_footprint_frame() -> tuple["object", dict, "object", dict]:
    import numpy as np
    import pandas as pd
    polys = load_polygons(BOUNDARY_FILE)
    import xarray as xr
    sample = sorted(MONTHLY_DIR.glob("era5_pjm_[0-9]*_[0-9][0-9].nc"))
    sample = [p for p in sample if not p.name.endswith("_ssrd.nc")]
    if not sample:
        raise FileNotFoundError("No monthly ERA5 base files found.")
    with xr.open_dataset(sample[0]) as ds0:
        lats = ds0["latitude"].to_numpy()
        lons = ds0["longitude"].to_numpy()
    mask = build_mask(lats, lons, polys)
    lon2d, lat2d = np.meshgrid(lons, lats)
    mean_lat = float(lat2d[mask].mean()) if mask.sum() else FOOTPRINT_LAT_FALLBACK
    mean_lon = float(lon2d[mask].mean()) if mask.sum() else -82.0
    frames, ssrds, month_reports = [], [], []
    for year, month in available_months():
        (res, rep) = load_month_frame(year, month, mask, lats, lons)
        month_reports.append(rep)
        if res is None:
            continue
        df_m, ssrd_accum = res
        frames.append(df_m)
        ssrds.append((f"{year}-{month}", df_m["timestamp"], ssrd_accum))
    if not frames:
        raise FileNotFoundError("No loadable monthly ERA5 base files.")
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    # ssrd de-accumulation on the concatenated timeline
    import pandas as pd2  # noqa
    acc = np.full(len(raw), np.nan)
    for _, ts_m, s_m in ssrds:
        if s_m is None:
            continue
        idx = raw["timestamp"].isin(ts_m)
        acc[idx.to_numpy()] = np.asarray(s_m, dtype=float)
    rate = np.full(len(raw), np.nan)
    ok = ~np.isnan(acc)
    prev = np.full(len(raw), np.nan)
    prev[1:] = acc[:-1]
    both = ok & ~np.isnan(prev)
    rate[both] = np.clip(acc[both] - prev[both], 0.0, None) / 3600.0
    # month-boundary first hours: fall back to accum/3600 capped at 1500 W/m2
    # (only where daytime; night values ~0 anyway)
    first_missing = ok & ~both
    rate[first_missing] = np.clip(acc[first_missing] / 3600.0, 0.0, 1500.0)
    # duplicate timestamps across month files?
    dup = int(raw["timestamp"].duplicated().sum())
    return raw, {"n_cells_total": int(mask.size), "n_cells_in": int(mask.sum()),
                 "mean_lat": mean_lat, "mean_lon": mean_lon,
                 "duplicates": dup}, rate, month_reports


def derive_features(raw, rate, mean_lat, mean_lon):
    import numpy as np
    import pandas as pd
    t = raw["t2m"].to_numpy() - K2C
    td = raw["d2m"].to_numpy() - K2C
    u = raw["u10"].to_numpy()
    v = raw["v10"].to_numpy()
    tcc_raw = raw["tcc"].to_numpy()
    tcc_max = float(np.nanmax(tcc_raw))
    cloud = tcc_raw / 100.0 if tcc_max > 1.5 else tcc_raw
    cloud = np.clip(cloud, 0.0, 1.0)
    wind = np.hypot(u, v)
    rh = rh_magnus(t, td)
    hi = heat_index_c(t, rh)
    wb = wet_bulb_stull(t, rh)
    cdh = np.clip(t - 22.0, 0.0, None)
    hdh = np.clip(18.0 - t, 0.0, None)
    cz = cos_zenith(raw["timestamp"], mean_lat, mean_lon)
    dl = np.clip(cz, 0.0, None)
    w = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "temperature_c": t,
        "dewpoint_c": td,
        "relative_humidity": rh,
        "wind_speed": wind,
        "cloud_cover": cloud,
        "solar_radiation_wm2": rate,
        "heat_index_c": hi,
        "wet_bulb_c": wb,
        "cdh": cdh,
        "hdh": hdh,
        "daylight_proxy": dl,
    })
    conv = {"tcc_native_max": tcc_max,
            "tcc_assumed_percent": bool(tcc_max > 1.5),
            "mean_lat_used": mean_lat}
    return w, conv


def run_validation_only() -> dict:
    import pandas as pd
    raw, minfo, rate, month_reports = build_footprint_frame()
    checks = validate_footprint(raw, rate)
    # expected full-period grid 2020-01-01 00:00 -> 2025-12-31 23:00
    full_grid = pd.date_range("2020-01-01", "2025-12-31 23:00", freq="h")
    have = pd.DatetimeIndex(raw["timestamp"])
    missing_months = []
    for y in range(2020, 2026):
        for m in range(1, 13):
            if (str(y), f"{m:02d}") not in [tuple(x) for x in available_months()]:
                missing_months.append(f"{y}-{m:02d}")
    report = {
        "generated_utc": utcnow(),
        "source": "era5",
        "spec": "configs/era5_request_v2.json (5-var base + ssrd supplements)",
        "bbox": {"north": 48.5, "west": -92.0, "south": 33.5, "east": -73.5},
        "footprint": minfo,
        "period_target": {"start": "2020-01-01T00:00", "end": "2025-12-31T23:00",
                          "n_hours_expected": len(full_grid)},
        "period_available": {"start": str(have.min()), "end": str(have.max()),
                             "n_hours": len(have),
                             "coverage_pct": round(100 * len(have) / len(full_grid), 3)},
        "months_available": [f"{y}-{m}" for y, m in available_months()],
        "months_missing": missing_months,
        "months": month_reports,
        **checks,
        "gate": ("pass" if checks["gaps"]["n_gaps"] == 0
                 and all(v == 0 for v in checks["impossible_counts"].values())
                 and len(have) > 0 else "fail"),
    }
    Q1_DIR.mkdir(parents=True, exist_ok=True)
    with VALIDATION_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    # per-month validation JSONs for newly available months (existing schema)
    for rep in month_reports:
        mp = Q1_DIR / f"era5_validation_{rep['month'][:4]}_{rep['month'][5:7]}.json"
        if mp.exists():
            continue
        with mp.open("w", encoding="utf-8") as f:
            json.dump({"month": rep["month"], "generated_utc": utcnow(),
                       "status": rep["status"], "ssrd": rep.get("ssrd"),
                       "n_hours": rep.get("n_hours"),
                       "t_start": rep.get("t_start"), "t_end": rep.get("t_end"),
                       "nan_pct_raw": rep.get("nan_pct"),
                       "verdict": "PASS" if rep["status"] == "OK" else "FAIL"},
                      f, indent=2)
            f.write("\n")
    print(f"[process] validation -> {VALIDATION_REPORT} gate={report['gate']} "
          f"coverage={report['period_available']['coverage_pct']}%")
    return report


def run_full() -> dict:
    import numpy as np
    import pandas as pd
    report = run_validation_only()
    raw, minfo, rate, _ = build_footprint_frame()
    w, conv = derive_features(raw, rate, report["footprint"]["mean_lat"],
                              report["footprint"]["mean_lon"])
    report["conversions"] = conv
    report["known_limitations"] = [
        "ssrd_evening_segment: CDS hourly ssrd accumulations decline (not grow) "
        "over ~19-23 UTC daily in every month sampled (Jan+Jul 2020 verified), "
        "which is impossible for same-origin accumulations (production-stream "
        "mixing). Clipped de-accumulation therefore yields ~0 W/m2 for those "
        "hours and the observed diurnal peaks ~15 UTC vs true solar noon "
        "~17:20 UTC at footprint mean lon: the SHAPE is distorted, not just "
        "the evenings. solar_radiation_wm2 is valid for day/night contrast "
        "and rough magnitude (deep-night ~0 verified; midday mean >30 W/m2) "
        "but NOT for precise intraday solar timing. daylight_proxy "
        "(timestamp-derived, unaffected) + cloud_cover carry the unbiased "
        "diurnal shape. Follow-up: re-pull ssrd via an alternative CDS "
        "stream/format or derive evening solar from clear-sky x cloud.",
        "population_weighting: Method B BLOCKED (no population product); "
        "published series is boundary-masked simple mean (Method A).",
    ]
    with VALIDATION_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    # weights file (simple mean primary; pop hook)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    import xarray as xr
    sample = [p for p in sorted(MONTHLY_DIR.glob("era5_pjm_[0-9]*_[0-9][0-9].nc"))
              if not p.name.endswith("_ssrd.nc")][0]
    with xr.open_dataset(sample) as ds0:
        lats = ds0["latitude"].to_numpy()
        lons = ds0["longitude"].to_numpy()
    polys = load_polygons(BOUNDARY_FILE)
    mask = build_mask(lats, lons, polys)
    n_in = int(mask.sum())
    rows = []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            if mask[i, j]:
                rows.append({"cell_id": f"{float(la):.2f}_{float(lo):.2f}",
                             "lat": float(la), "lon": float(lo),
                             "in_footprint": 1, "w_simple": 1.0 / n_in,
                             "w_pop": np.nan})
    wf = pd.DataFrame(rows)
    assert abs(wf["w_simple"].sum() - 1.0) < 1e-9
    wf.to_csv(V2_DIR / "pjm_footprint_weights.csv", index=False)
    cov = {"generated_utc": utcnow(), "boundary": "data/raw/external/pjm_boundary.geojson",
           "boundary_sha256": sha256_file(BOUNDARY_FILE),
           "era5_grid": {"n_lat": len(lats), "n_lon": len(lons)},
           "n_cells_total": int(mask.size), "n_cells_in": n_in,
           "weight_method_primary": "boundary-masked simple mean (Method A)",
           "weight_method_B_status": "BLOCKED: population product not acquired "
             "(configs/pjm_weather_weights.yaml TODO); w_pop=NaN; code path ready",
           "mean_lat": report["footprint"]["mean_lat"],
           "mean_lon": report["footprint"]["mean_lon"]}
    with (V2_DIR / "pjm_footprint_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(cov, f, indent=2)
        f.write("\n")
    # weather series output (task STEP-4 columns = subset, documented names)
    w["n_cells"] = n_in
    w_out = w[["timestamp", "temperature_c", "dewpoint_c", "relative_humidity",
               "wind_speed", "cloud_cover", "solar_radiation_wm2",
               "heat_index_c", "wet_bulb_c", "cdh", "hdh",
               "daylight_proxy", "n_cells"]]
    WEATHER_OUT.parent.mkdir(parents=True, exist_ok=True)
    w_out.to_csv(WEATHER_OUT, index=False)
    # merge onto v1 (READ-ONLY source). Normalise datetime units: xarray
    # yields datetime64[us], read_csv yields datetime64[ns].
    v1 = pd.read_csv(V1_PATH, parse_dates=["timestamp"])
    v1["timestamp"] = pd.to_datetime(v1["timestamp"]).astype("datetime64[ns]")
    v1 = v1.sort_values("timestamp").reset_index(drop=True)
    w_s = w_out.sort_values("timestamp").reset_index(drop=True)
    w_s["timestamp"] = pd.to_datetime(w_s["timestamp"]).astype("datetime64[ns]")
    merged = pd.merge_asof(v1, w_s, on="timestamp", direction="backward",
                           tolerance=pd.Timedelta("1h"))
    matched = merged["temperature_c"].notna().sum()
    coverage = float(matched) / len(merged)
    unmatched_idx = merged.index[merged["temperature_c"].isna()].tolist()
    merged["total_renewable_mw"] = (merged["wind_generation_mw"]
                                    + merged["solar_generation_mw"])
    merged["net_load_mw"] = merged["pjm_load_mw"] - merged["total_renewable_mw"]
    merged["renewable_penetration"] = (merged["total_renewable_mw"]
                                       / merged["pjm_load_mw"])
    v2 = pd.DataFrame({
        "timestamp": merged["timestamp"],
        "pjm_load_mw": merged["pjm_load_mw"],
        "wind_generation_mw": merged["wind_generation_mw"],
        "solar_generation_mw": merged["solar_generation_mw"],
        "temperature": merged["temperature_c"],
        "humidity": merged["relative_humidity"],
        "wind_speed": merged["wind_speed"],
        "cloud_cover": merged["cloud_cover"],
        "solar_radiation": merged["solar_radiation_wm2"],
        "total_renewable": merged["total_renewable_mw"],
        "net_load": merged["net_load_mw"],
        "renewable_penetration": merged["renewable_penetration"],
        "cdh": merged["cdh"],
        "hdh": merged["hdh"],
    })
    V2_OUT.parent.mkdir(parents=True, exist_ok=True)
    v2.to_csv(V2_OUT, index=False)
    # metadata v2
    wmiss = {c: float(v2[c].isna().mean()) for c in v2.columns if c != "timestamp"}
    meta = {
        "generated_utc": utcnow(),
        "v1_source": {"path": "dataset/processed/pjm_smart_grid_2020_2025.csv",
                      "sha256": sha256_file(V1_PATH),
                      "rows": len(v1),
                      "range": [str(v1['timestamp'].min()), str(v1['timestamp'].max())],
                      "modified": False},
        "weather_source": {"spec": "configs/era5_request_v2.json",
                           "validation_report": "results/q1/weather_validation_report.json",
                           "months_available": report["months_available"],
                           "months_missing": report["months_missing"]},
        "outputs": {
            "weather": {"path": "data/processed/pjm_weather_2020_2025.csv",
                        "sha256": sha256_file(WEATHER_OUT), "rows": len(w_out)},
            "v2": {"path": "dataset/processed/pjm_smart_grid_2020_2025_v2.csv",
                   "sha256": sha256_file(V2_OUT), "rows": len(v2)}},
        "v1_rows": len(v1),
        "v2_rows": len(v2),
        "rows_before_merge": len(v1),
        "rows_after_merge": len(v2),
        "rows_dropped_by_merge": 0,
        "join": {"method": "merge_asof backward tolerance=1h", "matched": int(matched),
                 "rows_preserved_pct": 100.0,
                 "rows_dropped_by_merge": 0,
                 "weather_matched_pct": round(100 * coverage, 3),
                 "retention_pct": round(100 * coverage, 3),
                 "weather_coverage_pct": round(100 * coverage, 3),
                 "first_20_unmatched": [str(merged['timestamp'].iloc[i])
                                        for i in unmatched_idx[:20]]},
        "missing_pct_v2": wmiss,
        "feature_list": list(v2.columns),
        "feature_provenance": {
            "temperature": "ERA5 t2m footprint mean, K->C",
            "humidity": "Magnus RH from footprint T/Td",
            "wind_speed": "sqrt(u10^2+v10^2) footprint mean",
            "cloud_cover": "ERA5 tcc normalised 0-1",
            "solar_radiation": "ssrd de-accumulated W/m2 footprint mean",
            "cdh": "max(T-22,0)", "hdh": "max(18-T,0)"},
        "weights": "data/processed/v2/pjm_footprint_weights.csv",
        "leakage_notes": "merge_asof backward-only; no forward fill; weather "
                         "columns are contemporaneous (nowcast assumption per "
                         "WEATHER_CARD S7 -- 24h-lagged sensitivity required "
                         "before operational claims).",
    }
    META_V2.parent.mkdir(parents=True, exist_ok=True)
    with META_V2.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"[process] weather -> {WEATHER_OUT} ({len(w_out)} rows)")
    print(f"[process] v2 -> {V2_OUT} ({len(v2)} rows, "
          f"weather coverage={100*coverage:.2f}%)")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="ERA5 footprint weather pipeline.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate-only", action="store_true")
    g.add_argument("--full", action="store_true")
    args = p.parse_args()
    try:
        if args.validate_only:
            run_validation_only()
        else:
            run_full()
    except Exception as e:
        print(f"[process] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
