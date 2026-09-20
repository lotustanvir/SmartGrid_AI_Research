"""Tests for the Phase 6A real-data pipeline (small fixture CSVs)."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.adapters import fit_scaler, sequenced_ready, tabular_ready, tft_ready
from src.data.features import add_features
from src.data.loader import load_raw_csv
from src.data.preprocessing import clean
from src.data.schema import (
    CleaningConfig,
    DataMapping,
    FeatureConfig,
    load_data_config,
)
from src.data.validation import validate
from src.training.splits import TemporalSplit

PROVIDER_COLS = {
    "ts": "timestamp",
    "load_mw": "target",
    "site": "group",
    "temp_c": "temperature",
    "sol": "solar",
    "wnd": "wind",
}


def make_mapping(path, **over):
    kw = {
        "path": path,
        "timestamp_col": "ts",
        "target_col": "load_mw",
        "group_col": "site",
        "temperature_col": "temp_c",
        "solar_col": "sol",
        "wind_col": "wnd",
        "expected_freq": "h",
    }
    kw.update(over)
    return DataMapping(**kw)


def make_fixture(path, n_hours=24 * 14, groups=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2024-01-01")
    for g in groups:
        base = 100.0 + (10.0 if g == "B" else 0.0)
        for h in range(n_hours):
            ts = t0 + pd.Timedelta(hours=h)
            rows.append(
                {
                    "ts": ts,
                    "load_mw": base + 15 * np.sin(2 * np.pi * (h % 24) / 24)
                    + rng.normal(0, 1),
                    "site": g,
                    "temp_c": 20 + 5 * np.sin(2 * np.pi * (h % 24) / 24),
                    "sol": max(0.0, np.sin(np.pi * ((h % 24) - 6) / 12)) * 20,
                    "wnd": 8 + rng.normal(0, 1),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return t0


class TestLoaderSchema(unittest.TestCase):
    def test_custom_mapping_loads(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.csv")
            make_fixture(p)
            df = load_raw_csv(make_mapping(p))
        self.assertEqual(
            list(df.columns),
            ["timestamp", "target", "group_id", "temperature", "solar", "wind"],
        )
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertEqual(sorted(df["group_id"].unique()), ["A", "B"])

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.csv")
            make_fixture(p)
            with self.assertRaises(ValueError):
                load_raw_csv(make_mapping(p, wind_col="nope"))

    def test_data_yaml_loads(self):
        mapping, clean_cfg, feat_cfg, _ = load_data_config()
        self.assertEqual(mapping.timestamp_col, "datetime")
        self.assertIsNone(mapping.group_col)
        self.assertEqual(feat_cfg.lags, (1, 24, 168))


class TestValidation(unittest.TestCase):
    def _frame(self, d, **kw):
        p = os.path.join(d, "data.csv")
        make_fixture(p, **kw)
        return load_raw_csv(make_mapping(p))

    def test_clean_passes(self):
        with tempfile.TemporaryDirectory() as d:
            rep = validate(self._frame(d), make_mapping("x"))
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["n_groups"], 2)

    def test_duplicates_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            df = self._frame(d)
            df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
            rep = validate(df, make_mapping("x"))
        self.assertTrue(any(i["check"] == "duplicates" for i in rep["issues"]))

    def test_gap_detected(self):
        with tempfile.TemporaryDirectory() as d:
            df = self._frame(d)
            df = df.drop(df[(df["group_id"] == "A")].index[10:15]).reset_index(drop=True)
            rep = validate(df, make_mapping("x"))
        gap = [i for i in rep["issues"] if i["check"] == "missing_timestamps"]
        self.assertTrue(gap and gap[0]["count"] >= 5)

    def test_negative_target_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            df = self._frame(d)
            df.loc[0, "target"] = -5.0
            rep = validate(df, make_mapping("x"))
        self.assertFalse(rep["passed"])
        self.assertTrue(any(i["check"] == "negative_target" for i in rep["issues"]))

    def test_missing_target_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            df = self._frame(d)
            df.loc[5, "target"] = np.nan
            rep = validate(df, make_mapping("x"))
        self.assertTrue(any(i["check"] == "missing_target" for i in rep["issues"]))


class TestCleaningFeatures(unittest.TestCase):
    def _cleaned(self, d, n_hours=24 * 14):
        p = os.path.join(d, "data.csv")
        make_fixture(p, n_hours=n_hours)
        df = load_raw_csv(make_mapping(p))
        return clean(df, make_mapping(p), CleaningConfig())

    def test_exogenous_interp_tracked(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.csv")
            make_fixture(p)
            df = load_raw_csv(make_mapping(p))
            idx = (df["group_id"] == "A").to_numpy().nonzero()[0][50:53]
            df.loc[idx, "temperature"] = np.nan
            out, rep = clean(df, make_mapping(p), CleaningConfig())
        self.assertFalse(out["temperature"].isna().any())
        self.assertGreater(rep["exogenous_interpolated_cells"], 0)

    def test_long_target_gap_dropped_not_filled(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.csv")
            make_fixture(p)
            df = load_raw_csv(make_mapping(p))
            b_idx = (df["group_id"] == "B").to_numpy().nonzero()[0][50:60]
            df.loc[b_idx, "target"] = np.nan
            a_idx = (df["group_id"] == "A").to_numpy().nonzero()[0][50:52]
            df.loc[a_idx, "target"] = np.nan
            out, rep = clean(df, make_mapping(p), CleaningConfig())
        self.assertEqual(rep["dropped_rows_target_na"], 10)
        self.assertEqual(rep["target_interpolated_cells"], 2)
        self.assertFalse(out["target"].isna().any())

    def test_lag_correctness(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = self._cleaned(d)
            feat, _ = add_features(out, FeatureConfig(lags=(1, 24), rolling_windows=()))
        for _, g in feat.groupby("group_id"):
            g = g.reset_index(drop=True)
            np.testing.assert_allclose(
                g["lag_1"].to_numpy()[1:], g["target"].to_numpy()[:-1]
            )
            np.testing.assert_allclose(
                g["lag_24"].to_numpy()[24:], g["target"].to_numpy()[:-24]
            )

    def test_rolling_uses_past_only(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = self._cleaned(d, n_hours=24 * 5)
            feat, _ = add_features(out, FeatureConfig(lags=(), rolling_windows=(24,)))
        g = feat[feat["group_id"] == "A"].reset_index(drop=True)
        y = g["target"].to_numpy()
        # rolling_mean_24[t] == mean(y[t-24..t-1])  (shift(1) semantics)
        self.assertAlmostEqual(g["rolling_mean_24"].iloc[30], y[6:30].mean(), places=9)
        self.assertTrue(g["rolling_mean_24"].iloc[:24].isna().all())

    def test_grouped_rolling_no_crosstalk(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = self._cleaned(d, n_hours=24 * 5)
            feat, _ = add_features(out, FeatureConfig(lags=(), rolling_windows=(24,)))
        b = feat[feat["group_id"] == "B"].reset_index(drop=True)
        yb = b["target"].to_numpy()
        self.assertAlmostEqual(b["rolling_mean_24"].iloc[30], yb[6:30].mean(), places=9)


class TestSplitsAdapters(unittest.TestCase):
    def _feat(self, d):
        p = os.path.join(d, "data.csv")
        make_fixture(p)
        df = load_raw_csv(make_mapping(p))
        out, _ = clean(df, make_mapping(p), CleaningConfig())
        feat, names = add_features(out, FeatureConfig())
        return feat, names

    def test_chronological_split(self):
        with tempfile.TemporaryDirectory() as d:
            feat, _ = self._feat(d)
            tr, va, te = TemporalSplit().split_frame(
                feat, time_col="timestamp", group_col="group_id"
            )
        self.assertLess(tr["timestamp"].max(), va["timestamp"].min())
        self.assertLess(va["timestamp"].max(), te["timestamp"].min())

    def test_scaler_fit_train_only(self):
        with tempfile.TemporaryDirectory() as d:
            feat, names = self._feat(d)
            tr, _, _ = TemporalSplit().split_frame(
                feat, time_col="timestamp", group_col="group_id"
            )
            X_tr, _, cols, _ = tabular_ready(tr, names)
            scaler, _ = fit_scaler(X_tr)
        np.testing.assert_allclose(scaler.mean_, X_tr.mean(axis=0))
        self.assertEqual(list(cols), names)

    def test_tabular_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            feat, names = self._feat(d)
            X, y, cols, info = tabular_ready(feat, names)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertGreater(info["warmup_rows_dropped"], 0)
        self.assertFalse(np.isnan(X).any())

    def test_sequence_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            feat, names = self._feat(d)
            X, y, _, _ = sequenced_ready(feat, names)
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_tft_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            feat, _ = self._feat(d)
            frame, info = tft_ready(feat)
        for col in ("time_idx", "group_id", "electricity_demand"):
            self.assertIn(col, frame.columns)
        self.assertEqual(info["n_groups"], 2)
        self.assertFalse(frame.isna().any().any())
        g = frame[frame["group_id"] == "A"].reset_index(drop=True)
        self.assertEqual(g["time_idx"].tolist(), list(range(len(g))))


class TestPipelineMode(unittest.TestCase):
    def test_waiting_without_csv(self):
        from src.data.pipeline import run_pipeline

        out = run_pipeline()
        self.assertEqual(out["status"], "waiting_for_dataset")

    def test_pipeline_end_to_end_mapped(self):
        from src.data.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "demand.csv")
            make_fixture(p)
            cfg = os.path.join(d, "data.yaml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(
                    "data:\n"
                    f"  path: {p!r}\n"
                    "  timestamp_col: ts\n"
                    "  target_col: load_mw\n"
                    "  group_col: site\n"
                    "  expected_freq: h\n"
                    "  weather: {temperature: temp_c}\n"
                    "  renewable: {solar: sol, wind: wnd}\n"
                )
            out = run_pipeline(
                data_config=cfg,
                validation_dir=os.path.join(d, "val"),
                processed_dir=os.path.join(d, "proc"),
                eda_dir=os.path.join(d, "eda"),
            )
            self.assertEqual(out["status"], "ready")
            md = out["metadata"]
            self.assertEqual(md["n_groups"], 2)
            self.assertTrue(all(c["passed"] for c in md["leakage_checks"]))
            self.assertLess(
                md["train_period"]["end"], md["validation_period"]["start"]
            )
            self.assertTrue(os.path.isfile(os.path.join(d, "val", "report.json")))
            self.assertTrue(os.path.isfile(os.path.join(d, "val", "dataset_metadata.json")))
            self.assertGreaterEqual(len(out["eda_files"]), 6)


if __name__ == "__main__":
    unittest.main()
