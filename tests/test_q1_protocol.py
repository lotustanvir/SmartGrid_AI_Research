"""Q1 Phase 2 protocol tests (fail-closed, no research results)."""

import unittest

import numpy as np
import pandas as pd


def _synth_v2(n=600, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    load = 100 + 10 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24) + rng.normal(0, 1, n)
    wind = np.abs(rng.normal(5, 1, n))
    solar = np.maximum(0, np.sin(np.pi * ((np.arange(n) % 24) - 6) / 12)) * 8
    temp = 15 + 5 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24) + rng.normal(0, 0.5, n)
    tot = wind + solar
    return pd.DataFrame(
        {
            "timestamp": ts,
            "pjm_load_mw": load,
            "wind_generation_mw": wind,
            "solar_generation_mw": solar,
            "temperature": temp,
            "humidity": 60 + rng.normal(0, 2, n),
            "wind_speed": 3 + rng.normal(0, 0.5, n),
            "cloud_cover": 0.4 + rng.normal(0, 0.1, n),
            "solar_radiation": solar * 10,
            "total_renewable": tot,
            "net_load": load - tot,
            "renewable_penetration": tot / load,
            "cdh": np.maximum(0, temp - 18),
            "hdh": np.maximum(0, 18 - temp),
        }
    )


class TestQ1Registry(unittest.TestCase):
    def test_eleven_registered(self):
        from src.training.experiment_runner import MODEL_REGISTRY

        from src.q1.protocol import Q1_MODELS

        for m in Q1_MODELS:
            self.assertIn(m, MODEL_REGISTRY, f"missing {m}")

    def test_q1_constants(self):
        from src.q1 import protocol as P

        self.assertEqual(P.HISTORY_LENGTH, 168)
        self.assertEqual(tuple(P.HORIZONS), (1, 6, 24))
        self.assertEqual(tuple(P.Q1_SEEDS), (42, 123, 2025))
        self.assertEqual(P.Q1_OOF_FOLDS, 5)
        self.assertEqual(P.Q1_DATASET_PATH, "dataset/processed/pjm_smart_grid_2020_2025_v2.csv")


class TestQ1Features(unittest.TestCase):
    def test_schema_canonical(self):
        from src.data.features_q1 import (
            Q1_FEATURE_VERSION,
            canonical_feature_names,
            feature_schema_hash,
        )

        feats = canonical_feature_names()
        self.assertNotIn("net_load", feats)
        self.assertNotIn("renewable_penetration", feats)
        self.assertNotIn("temperature", feats)  # raw weather excluded
        self.assertIn("lag_168", feats)
        self.assertIn("rolling_mean_168", feats)
        self.assertIn("temp_lag_1", feats)
        self.assertIn("temp_lag_168", feats)
        h1 = feature_schema_hash()
        self.assertEqual(h1, feature_schema_hash())  # deterministic
        self.assertEqual(len(h1), 64)

    def test_build_lagged(self):
        from src.data.features_q1 import build_q1_features, canonical_feature_names

        raw = _synth_v2(400)
        feat, names = build_q1_features(raw)
        self.assertEqual(names, canonical_feature_names())
        # Weather lag correctness: temp_lag_1[t] == temperature[t-1].
        row = feat.iloc[50]
        self.assertAlmostEqual(row["temp_lag_1"], raw["temperature"].iloc[49])
        self.assertAlmostEqual(row["lag_24"], raw["pjm_load_mw"].iloc[26])
        # Rolling backward: rolling_mean_24[t] == mean(load[t-24..t-1]).
        expect = raw["pjm_load_mw"].iloc[26:50].mean()
        self.assertAlmostEqual(feat["rolling_mean_24"].iloc[50], expect)

    def test_target_derived_excluded(self):
        from src.data.features_q1 import build_q1_features

        raw = _synth_v2(300)
        feat, names = build_q1_features(raw)
        for bad in ("net_load", "renewable_penetration", "pjm_load_mw"):
            self.assertNotIn(bad, names)


class TestCalendarSplit(unittest.TestCase):
    def test_boundaries(self):
        from src.data.splits_q1 import split_calendar

        ts = pd.date_range("2023-12-30", "2025-01-03", freq="h")
        df = pd.DataFrame({"timestamp": ts, "v": range(len(ts))})
        tr, va, te = split_calendar(df)
        self.assertTrue(tr["timestamp"].max() < va["timestamp"].min())
        self.assertTrue(va["timestamp"].max() < te["timestamp"].min())
        self.assertTrue((tr["timestamp"] < "2024-01-01").all())
        self.assertTrue(((va["timestamp"] >= "2024-01-01") & (va["timestamp"] < "2025-01-01")).all())
        self.assertTrue(((te["timestamp"] >= "2025-01-01") & (te["timestamp"] < "2026-01-01")).all())

    def test_duplicates_fail(self):
        from src.data.splits_q1 import split_calendar

        ts = list(pd.date_range("2024-06-01", periods=10, freq="h"))
        ts[5] = ts[4]
        with self.assertRaises(ValueError):
            split_calendar(pd.DataFrame({"timestamp": ts}))


class TestOrigins(unittest.TestCase):
    def test_generate_and_hash(self):
        from src.data.features_q1 import build_q1_features, drop_warmup
        from src.data.origins import assert_same_origins, generate_origins, origins_hash

        raw = _synth_v2(500)
        feat, names = build_q1_features(raw)
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=(1, 6, 24), history=168)
        self.assertTrue(set(org["horizon"]) == {1, 6, 24})
        h = origins_hash(org)
        self.assertEqual(h, origins_hash(org))
        assert_same_origins(org, org.copy())
        bad = org.iloc[1:].reset_index(drop=True)
        with self.assertRaises(ValueError):
            assert_same_origins(org, bad)


class TestMissingModels(unittest.TestCase):
    def test_persistence_rules(self):
        from src.models.statistical import PersistenceForecaster, SeasonalPersistenceForecaster

        y = np.arange(200, dtype=float) + 100
        X = np.zeros((3, 2))
        p = PersistenceForecaster(horizon=1).fit(X, y)
        np.testing.assert_allclose(p.predict(X), np.full(3, y[-1]))
        s = SeasonalPersistenceForecaster(horizon=1).fit(X, y)
        out = s.predict(np.zeros((2, 2)))
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all(np.isfinite(out)))
        with self.assertRaises(ValueError):
            PersistenceForecaster(random_state=42)
        with self.assertRaises(ValueError):
            SeasonalPersistenceForecaster(seasonal_period=24)
        with self.assertRaises(ValueError):
            PersistenceForecaster(horizon=5)

    def test_sarima_shapes(self):
        from src.models.statistical import SARIMAForecaster

        rng = np.random.default_rng(0)
        y = 100 + np.sin(np.arange(600) / 24 * 2 * np.pi) + rng.normal(0, 0.5, 600)
        X = np.zeros((len(y), 2))
        for h in (1, 6, 24):
            m = SARIMAForecaster(horizon=h).fit(X, y)
            pr = m.predict(np.zeros((4, 2)))
            self.assertEqual(pr.shape, (4,))
            self.assertTrue(np.all(np.isfinite(pr)))

    def test_patchtst_nbeats_shapes(self):
        from src.models.advanced import NBEATSForecaster, PatchTSTForecaster

        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 5))
        y = rng.normal(size=300)
        for cls in (PatchTSTForecaster, NBEATSForecaster):
            for h in (1, 6, 24):
                m = cls(seq_len=24, output_size=h, epochs=1, patience=2, batch_size=32)
                m.fit(X, y)
                pr = m.predict(X)
                self.assertTrue(np.all(np.isfinite(np.asarray(pr).ravel())))

    def test_tree_direct_pairs(self):
        from src.data.features_q1 import build_q1_features, drop_warmup
        from src.q1.horizons import build_direct_pairs

        raw = _synth_v2(400)
        feat, names = build_q1_features(raw)
        clean, _ = drop_warmup(feat, names)
        for h in (1, 6, 24):
            X, y, keys = build_direct_pairs(clean, names, h)
            self.assertEqual(X.shape[0], len(y))
            self.assertTrue((keys["horizon"] == h).all())
            # Pair correctness: y[k] == target[origin_row + h].
            self.assertAlmostEqual(y[0], clean["target"].iloc[int(keys["origin_row"].iloc[0]) + h])


class TestMetricsStats(unittest.TestCase):
    def test_mase(self):
        from src.evaluation.metrics import evaluate_regression, mase

        yt = np.array([10.0, 12, 11, 13, 12, 14])
        yp = np.array([10.5, 11.5, 11, 12.5, 12.5, 13.5])
        tr = np.array([9.0, 11, 10, 12, 11, 13, 12, 14, 13, 15] * 5, dtype=float)
        v = mase(yt, yp, y_train=tr, seasonality=2)
        self.assertTrue(np.isfinite(v) and v > 0)
        out = evaluate_regression(yt, yp, y_train=tr, seasonality=2)
        self.assertIn("MASE", out)
        out2 = evaluate_regression(yt, yp)
        self.assertNotIn("MASE", out2)  # omitted without train (documented)
        with self.assertRaises(ValueError):
            mase(yt, yp, y_train=np.ones(50), seasonality=2)  # constant

    def test_stats_framework(self):
        from src.evaluation import stats as S

        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 0.5, 60)
        b = rng.normal(1.2, 0.5, 60)
        w = S.wilcoxon_signed_rank(a, b)
        self.assertIn("p_value", w)
        f = S.friedman_test(a, b, rng.normal(1.1, 0.5, 60))
        self.assertIn("statistic", f)
        h = S.holm_correction([0.01, 0.04, 0.3])
        self.assertEqual(len(h["adjusted"]), 3)
        c = S.bootstrap_mean_diff_ci(a, b, n_boot=50, seed=0)
        self.assertLessEqual(c["ci_low"], c["ci_high"])
        with self.assertRaises(NotImplementedError):
            S.diebold_mariano(a, b)


class TestProvenanceRegistry(unittest.TestCase):
    def test_hashes_and_registry(self):
        import tempfile
        from pathlib import Path

        from src.data.features_q1 import canonical_feature_names, feature_schema_hash
        from src.data.provenance import feature_schema_hash as ph, sha256_text
        from src.utils.registry import record_experiment

        self.assertEqual(len(sha256_text("x")), 64)
        self.assertEqual(
            feature_schema_hash(),
            ph(canonical_feature_names(), "Q1_V2_LAG1_001"),
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "REG.csv"
            row = record_experiment(
                exp_id="Q1_SMOKE_unittest",
                status="COMPLETED",
                model="SMOKE",
                horizon=1,
                seed=42,
                smoke=True,
                path=p,
            )
            self.assertEqual(row["smoke"], "True")
            with self.assertRaises(ValueError):
                record_experiment(
                    exp_id="bad", status="DONE", model="x", horizon=1, seed=1, path=p
                )


if __name__ == "__main__":
    unittest.main()
