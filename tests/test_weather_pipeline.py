"""Weather pipeline tests (Phase 1D-C3, STEP 9).

Covers: weather file existence+schema, timestamp continuity, no duplicate
timestamps, merge row preservation (no silent drops), feature correctness
(CDH/HDH/RH/solar/daylight), backward-only merge (no future leakage),
physical ranges. Repo-relative paths (no hardcoded drives). Real-data tests:
skipped with a clear reason if pipeline outputs are absent.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WEATHER = REPO / "data" / "processed" / "pjm_weather_2020_2025.csv"
V1 = REPO / "dataset" / "processed" / "pjm_smart_grid_2020_2025.csv"
V2 = REPO / "dataset" / "processed" / "pjm_smart_grid_2020_2025_v2.csv"
REPORT = REPO / "results" / "q1" / "weather_validation_report.json"
META_V2 = REPO / "results" / "data_validation" / "dataset_metadata_v2.json"

WEATHER_COLS = ["timestamp", "temperature_c", "dewpoint_c", "relative_humidity",
                "wind_speed", "cloud_cover", "solar_radiation_wm2",
                "heat_index_c", "wet_bulb_c", "cdh", "hdh",
                "daylight_proxy", "n_cells"]
V2_COLS = ["timestamp", "pjm_load_mw", "wind_generation_mw",
           "solar_generation_mw", "temperature", "humidity", "wind_speed",
           "cloud_cover", "solar_radiation", "total_renewable", "net_load",
           "renewable_penetration", "cdh", "hdh"]


def needs_outputs(test):
    def wrapper(self, *a, **k):
        missing = [str(p) for p in (WEATHER, V1, V2) if not p.exists()]
        if missing:
            self.skipTest(f"pipeline outputs absent: {missing}")
        return test(self, *a, **k)
    return wrapper


class TestWeatherFile(unittest.TestCase):
    @needs_outputs
    def test_exists_and_schema(self):
        w = pd.read_csv(WEATHER, nrows=5)
        self.assertEqual(list(w.columns), WEATHER_COLS)

    @needs_outputs
    def test_timestamp_continuity_no_duplicates(self):
        w = pd.read_csv(WEATHER, parse_dates=["timestamp"])
        self.assertEqual(w["timestamp"].duplicated().sum(), 0)
        self.assertTrue(w["timestamp"].is_monotonic_increasing)
        full = pd.date_range(w["timestamp"].min(), w["timestamp"].max(), freq="h")
        self.assertEqual(len(full), len(w),
                         f"gaps: {len(full) - len(w)} missing hours in weather series")

    @needs_outputs
    def test_physical_ranges(self):
        w = pd.read_csv(WEATHER)
        self.assertTrue(((w["temperature_c"] >= -40) & (w["temperature_c"] <= 45)).all())
        self.assertTrue(((w["relative_humidity"] >= 0) & (w["relative_humidity"] <= 100)).all())
        self.assertTrue((w["wind_speed"] >= 0).all())
        self.assertTrue(((w["cloud_cover"] >= 0) & (w["cloud_cover"] <= 1)).all())
        # solar may be NaN where the ssrd supplement is pending (documented
        # coverage gap); non-null values must be non-negative.
        sol = w["solar_radiation_wm2"].dropna()
        self.assertTrue(len(sol) > 0)
        self.assertTrue((sol >= 0).all())
        self.assertTrue((w["dewpoint_c"] <= w["temperature_c"] + 1e-9).all())
        self.assertTrue((w["cdh"] >= 0).all() and (w["hdh"] >= 0).all())
        self.assertTrue(((w["daylight_proxy"] >= 0) & (w["daylight_proxy"] <= 1)).all())


class TestFeatureGeneration(unittest.TestCase):
    @needs_outputs
    def test_degree_hours_definition(self):
        # tolerance 1e-4: footprint means derive from float32 ERA5 grids,
        # so CSV round-trips carry ~1e-6 float32 noise.
        w = pd.read_csv(WEATHER)
        t = w["temperature_c"]
        self.assertTrue(((w["cdh"] - (t - 22.0).clip(lower=0)).abs() < 1e-4).all())
        self.assertTrue(((w["hdh"] - (18.0 - t).clip(lower=0)).abs() < 1e-4).all())

    @needs_outputs
    def test_heat_index_gating(self):
        w = pd.read_csv(WEATHER)
        cool = w[w["temperature_c"] < 20.0]
        self.assertTrue(len(cool) > 0)
        self.assertTrue(((cool["heat_index_c"] - cool["temperature_c"]).abs() < 1e-4).all())

    @needs_outputs
    def test_solar_day_night_contrast(self):
        # Deep-night 02-06 UTC is dark across the whole footprint year-round
        # (lon -92..-73.5 -> 21:00-01:00 local at latest). Midday must carry
        # real signal with peak inside daylight hours. Qualitative only: the
        # CDS ssrd stream-mixing artifact distorts the exact diurnal shape
        # (documented limitation, DATA_CARD_V2), so no proxy-correlation
        # assert here.
        w = pd.read_csv(WEATHER, parse_dates=["timestamp"])
        deep = w[(w["timestamp"].dt.hour.between(2, 6))
                 & w["solar_radiation_wm2"].notna()]
        self.assertTrue(len(deep) > 0)
        self.assertTrue((deep["solar_radiation_wm2"] < 5.0).all())
        mid = w[(w["timestamp"].dt.hour.between(12, 16))
                & w["solar_radiation_wm2"].notna()]
        self.assertTrue(len(mid) > 100)
        self.assertGreater(float(mid["solar_radiation_wm2"].mean()), 30.0)
        peak_hour = (w[w["solar_radiation_wm2"].notna()]
                     .groupby(w["timestamp"].dt.hour)["solar_radiation_wm2"]
                     .mean().idxmax())
        self.assertTrue(11 <= int(peak_hour) <= 18)


class TestMerge(unittest.TestCase):
    @needs_outputs
    def test_no_silent_row_drops(self):
        v1 = pd.read_csv(V1, parse_dates=["timestamp"])
        v2 = pd.read_csv(V2, parse_dates=["timestamp"])
        self.assertEqual(len(v2), len(v1))
        self.assertTrue((v2["timestamp"].astype(str) == v1["timestamp"].astype(str)).all())
        self.assertTrue((v2["pjm_load_mw"] == v1["pjm_load_mw"]).all())

    @needs_outputs
    def test_v2_schema_and_derived(self):
        v2 = pd.read_csv(V2)
        self.assertEqual(list(v2.columns), V2_COLS)
        self.assertTrue((abs(v2["total_renewable"]
                             - v2["wind_generation_mw"] - v2["solar_generation_mw"]) < 1e-6).all())
        self.assertTrue((abs(v2["net_load"] - v2["pjm_load_mw"]
                             + v2["total_renewable"]) < 1e-6).all())

    @needs_outputs
    def test_no_future_leakage(self):
        """Every matched weather value must come from its own hour or earlier
        (merge_asof backward). Re-check on a sample: weather timestamp <= v2
        timestamp for matched rows."""
        v1 = pd.read_csv(V1, parse_dates=["timestamp"])
        w = pd.read_csv(WEATHER, parse_dates=["timestamp"])
        v2 = pd.read_csv(V2, parse_dates=["timestamp"])
        sample = v2.dropna(subset=["temperature"]).iloc[::5000]
        self.assertTrue(len(sample) > 0)
        widx = w.set_index("timestamp")["temperature_c"]
        for _, r in sample.iterrows():
            ts = r["timestamp"]
            # backward-asof value at ts equals weather at ts (exact hours exist)
            self.assertIn(ts, widx.index)
            self.assertAlmostEqual(r["temperature"], widx.loc[ts], places=9)

    @needs_outputs
    def test_v1_untouched(self):
        meta = json.loads(META_V2.read_text(encoding="utf-8"))
        import hashlib
        h = hashlib.sha256()
        with V1.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), meta["v1_source"]["sha256"])
        self.assertFalse(meta["v1_source"]["modified"])


class TestReports(unittest.TestCase):
    def test_validation_report_gate(self):
        if not REPORT.exists():
            self.skipTest("weather_validation_report.json absent")
        r = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertIn(r.get("gate"), ("pass", "fail"))
        self.assertIn("months_missing", r)
        self.assertIn("coverage_pct", r["period_available"])

    def test_metadata_v2_integrity(self):
        if not META_V2.exists():
            self.skipTest("dataset_metadata_v2.json absent")
        m = json.loads(META_V2.read_text(encoding="utf-8"))
        self.assertEqual(m["rows_dropped_by_merge"], 0)
        self.assertEqual(m["v1_rows"], m["v2_rows"])
        self.assertIn("join", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
