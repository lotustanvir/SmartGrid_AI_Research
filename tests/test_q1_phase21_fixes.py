"""Phase 2.1 blocker-fix regression tests (synthetic only, no research results).

Covers FIX 1-10: SARIMA-lite 168 lock, Q1 MASE scoring, headline guard,
registry completeness, real artifact path, TFT quarantine, multi-origin
naive semantics, 168 defaults, origins_hash hardening, docs.
No full-dataset training. No headline metrics. No model ranking.
"""

import unittest

import numpy as np
import pandas as pd


def _synth_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    load = 100 + 10 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24) + rng.normal(0, 1, n)
    return ts, load


class TestFix1SarimaLite168(unittest.TestCase):
    def test_defaults_are_168(self):
        from src.models.statistical import SARIMAForecaster

        m = SARIMAForecaster()
        self.assertEqual(m.seasonal_period, 168)
        # MA orders q/Q are forced to 0 (documented approximation).
        self.assertEqual(m.seasonal_order, (1, 1, 0, 168))
        self.assertEqual(m.Q1_DISPLAY_NAME, "SARIMA-lite")

    def test_legacy_24_rejected(self):
        from src.models.statistical import SARIMAForecaster

        with self.assertRaises(ValueError):
            SARIMAForecaster(seasonal_order=(1, 1, 1, 24), seasonal_period=24)
        with self.assertRaises(ValueError):
            SARIMAForecaster(seasonal_order=(1, 1, 1, 24), seasonal_period=24,
                             order=(1, 1, 1))

    def test_from_config_is_168(self):
        from src.models.statistical import SARIMAForecaster

        m = SARIMAForecaster.from_config()
        self.assertEqual(m.seasonal_period, 168)
        self.assertEqual(m.seasonal_order[3], 168)

    def test_yaml_has_no_24(self):
        from pathlib import Path

        text = Path("configs/models.yaml").read_text(encoding="utf-8")
        sarima_block = text.split("sarima:")[1].split("linear_regression:")[0]
        # No 24-period seasonal configuration may remain.
        self.assertNotIn("seasonal_period: 24", sarima_block)
        self.assertNotIn("[1, 1, 1, 24]", sarima_block)
        self.assertIn("168", sarima_block)

    def test_pretty_name_is_lite(self):
        from src.training.experiment_runner import PRETTY_NAMES

        self.assertEqual(PRETTY_NAMES["sarima"], "SARIMA-lite")

    def test_fit_predict_shapes_168(self):
        from src.models.statistical import SARIMAForecaster

        rng = np.random.default_rng(0)
        y = 100 + np.sin(np.arange(600) / 24 * 2 * np.pi) + rng.normal(0, 0.5, 600)
        X = np.zeros((len(y), 2))
        for h in (1, 6, 24):
            m = SARIMAForecaster(horizon=h).fit(X, y)
            pr = m.predict(np.zeros((4, 2)))
            self.assertEqual(pr.shape, (4,))
            self.assertTrue(np.all(np.isfinite(pr)))


class TestFix2MaseInScoring(unittest.TestCase):
    def _frame(self, n=400):
        ts, load = _synth_series(n)
        frame = pd.DataFrame({"timestamp": ts, "target": load})
        origins = pd.DataFrame({
            "origin_timestamp": [ts[200], ts[210], ts[220]],
            "horizon": [1, 1, 1],
        })
        # Keys are (origin_ts_iso, horizon); values are noisy forecasts.
        preds = {(pd.Timestamp(t).isoformat(), 1): float(load[i + 1] + 0.5)
                 for t, i in [(ts[200], 200), (ts[210], 210), (ts[220], 220)]}
        return frame, origins, preds, load

    def test_mase_present_default_168(self):
        from src.q1 import evaluate as E

        self.assertEqual(E.Q1_MASE_SEASONALITY, 168)
        frame, origins, preds, load = self._frame()
        metrics, _ = E.score_at_origins(origins, frame, preds, y_train=load[:180])
        self.assertIn("MASE", metrics)
        for k in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE"):
            self.assertIn(k, metrics)

    def test_missing_ytrain_raises(self):
        from src.q1 import evaluate as E

        frame, origins, preds, _ = self._frame()
        with self.assertRaises(ValueError):
            E.score_at_origins(origins, frame, preds)

    def test_wrong_seasonality_raises(self):
        from src.q1 import evaluate as E

        frame, origins, preds, load = self._frame()
        with self.assertRaises(ValueError):
            E.score_at_origins(origins, frame, preds, y_train=load[:180],
                               seasonality=24)

    def test_val_test_do_not_change_denominator(self):
        from src.q1 import evaluate as E

        frame, origins, preds, load = self._frame()
        m1, _ = E.score_at_origins(origins, frame, preds, y_train=load[:180])
        # Perturb validation/test-region values; denominator uses y_train only.
        rng = np.random.default_rng(9)
        frame2 = frame.copy()
        frame2.loc[300:, "target"] = frame2.loc[300:, "target"] + rng.normal(0, 50, len(frame2) - 300)
        preds2 = {(k[0], k[1]): v + 0.0 for k, v in preds.items()}
        m2, _ = E.score_at_origins(origins, frame2, preds2, y_train=load[:180])
        self.assertAlmostEqual(m1["MASE"], m2["MASE"])

    def test_insufficient_train_raises(self):
        from src.q1 import evaluate as E

        frame, origins, preds, _ = self._frame()
        with self.assertRaises(ValueError):
            E.score_at_origins(origins, frame, preds, y_train=np.ones(100))


class TestFix3HeadlineGuard(unittest.TestCase):
    def test_exact_eleven_default(self):
        from src.q1.protocol import Q1_MODELS
        from src.training.experiment_runner import Q1_HEADLINE_MODELS, q1_headline_models

        self.assertEqual(list(q1_headline_models()), list(Q1_MODELS))
        self.assertEqual(len(q1_headline_models()), 11)
        self.assertEqual(list(Q1_HEADLINE_MODELS), [
            "persistence", "seasonal_persistence", "sarima", "xgboost",
            "lightgbm", "lstm", "gru", "tft", "patchtst", "nbeats", "hybrid",
        ])

    def test_legacy_rejected(self):
        from src.training.experiment_runner import q1_headline_models

        for legacy in ("linear_regression", "random_forest"):
            with self.assertRaises(ValueError):
                q1_headline_models(["xgboost", legacy])

    def test_unknown_rejected(self):
        from src.training.experiment_runner import q1_headline_models

        with self.assertRaises(ValueError):
            q1_headline_models(["xgboost", "not_a_model"])
        with self.assertRaises(ValueError):
            q1_headline_models([])

    def test_duplicate_rejected(self):
        from src.training.experiment_runner import q1_headline_models

        with self.assertRaises(ValueError):
            q1_headline_models(["xgboost", "xgboost"])

    def test_subset_ok(self):
        from src.training.experiment_runner import q1_headline_models

        self.assertEqual(q1_headline_models(["tft", "xgboost"]), ["tft", "xgboost"])


class TestFix4Registry(unittest.TestCase):
    def _complete(self):
        return {
            "exp_id": "Q1_H1_xgboost_s42",
            "status": "COMPLETED",
            "git_commit": "abc1234",
            "dataset_version": "v2",
            "dataset_sha256": "f" * 64,
            "features": "Q1_V2_LAG1_001",
            "feature_schema_hash": "e" * 64,
            "origin_set_hash": "5" * 64,
            "model": "xgboost",
            "horizon": 1,
            "seed": 42,
            "config_snapshot": "configs/models.yaml",
            "predictions": "preds/xgb_h1.csv",
            "metrics": '{"MAE": 1.0}',
            "checkpoint": "ckpt/xgb_h1.bin",
            "smoke": "False",
        }

    def test_git_commit_stored(self):
        import tempfile
        from pathlib import Path

        from src.utils.registry import HEADER, record_experiment

        self.assertIn("git_commit", HEADER)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "REG.csv"
            row = record_experiment(exp_id="e1", status="COMPLETED", model="SMOKE",
                                    horizon=0, seed=42, git_commit="deadbee",
                                    smoke=True, path=p)
            self.assertEqual(row["git_commit"], "deadbee")

    def test_complete_passes(self):
        from src.utils.registry import require_q1_complete

        rec = self._complete()
        self.assertEqual(require_q1_complete(rec), rec)

    def test_missing_field_fails(self):
        from src.utils.registry import require_q1_complete

        for field in ("git_commit", "dataset_sha256", "feature_schema_hash",
                      "origin_set_hash", "predictions", "metrics"):
            rec = self._complete()
            rec[field] = ""
            with self.assertRaises(ValueError):
                require_q1_complete(rec)

    def test_smoke_cannot_be_headline(self):
        from src.utils.registry import require_q1_complete

        rec = self._complete()
        rec["smoke"] = "True"
        with self.assertRaises(ValueError):
            require_q1_complete(rec)

    def test_deterministic_no_checkpoint_ok(self):
        from src.utils.registry import require_q1_complete

        for model in ("persistence", "seasonal_persistence", "sarima"):
            rec = self._complete()
            rec["model"] = model
            rec["seed"] = "det"  # Phase 3: deterministic runs record "det", never a fake seed
            rec["checkpoint"] = ""
            require_q1_complete(rec)
        # A numeric seed on a deterministic model is fake variation: reject.
        with self.assertRaises(ValueError):
            require_q1_complete({**self._complete(), "model": "sarima", "seed": 42, "checkpoint": ""})

    def test_stochastic_needs_checkpoint(self):
        from src.utils.registry import require_q1_complete

        rec = self._complete()
        rec["model"] = "lstm"
        rec["checkpoint"] = ""
        with self.assertRaises(ValueError):
            require_q1_complete(rec)


class TestFix5ArtifactPath(unittest.TestCase):
    def test_save_returns_existing_path(self):
        import tempfile
        from pathlib import Path

        from src.data.origins import generate_origins, save_origins
        from tests.test_q1_protocol import _synth_v2

        from src.data.features_q1 import build_q1_features, drop_warmup

        feat, names = build_q1_features(_synth_v2(500))
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=(1,), history=168)
        with tempfile.TemporaryDirectory() as d:
            p = save_origins(org, Path(d) / "test_origins.parquet")
            self.assertTrue(Path(str(p)).is_file())
            # No parquet engine in verified env -> CSV sidecar recorded.
            self.assertEqual(Path(str(p)).suffix, ".csv")

    def test_recorded_path_exists(self):
        import tempfile
        from pathlib import Path

        from src.data.origins import generate_origins, save_origins
        from src.utils.registry import record_experiment
        from tests.test_q1_protocol import _synth_v2

        from src.data.features_q1 import build_q1_features, drop_warmup

        feat, names = build_q1_features(_synth_v2(500))
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=(1,), history=168)
        with tempfile.TemporaryDirectory() as d:
            actual = str(save_origins(org, Path(d) / "o.parquet"))
            rp = Path(d) / "REG.csv"
            row = record_experiment(exp_id="e2", status="COMPLETED", model="SMOKE",
                                    horizon=0, seed=42, git_commit="c",
                                    predictions=actual, smoke=True, path=rp)
            self.assertTrue(Path(row["predictions"]).is_file())


class TestFix6TFTQuarantine(unittest.TestCase):
    def test_roles_calendar_only(self):
        from src.models.tft.dataset import resolve_roles

        df = pd.DataFrame({
            "time_idx": range(10),
            "group_id": ["single"] * 10,
            "electricity_demand": np.arange(10, dtype=float),
            "hour": range(10),
            "day_of_week": [0] * 10,
            "temperature": np.arange(10, dtype=float),
            "solar_generation": np.arange(10, dtype=float),
            "wind_generation": np.arange(10, dtype=float),
            "temp_lag_1": np.arange(10, dtype=float),
            "lag_1": np.arange(10, dtype=float),
        })
        known, unknown, _ = resolve_roles(df)
        for w in ("temperature", "solar_generation", "wind_generation",
                  "temp_lag_1", "lag_1"):
            self.assertNotIn(w, known)
            self.assertIn(w, unknown)
        self.assertIn("hour", known)

    def test_q1_frame_rejects_raw_weather(self):
        from src.q1.tft_path import build_q1_tft_frame

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="h"),
            "target": np.arange(20, dtype=float),
            "hour": range(20),
        })
        with self.assertRaises(ValueError):
            build_q1_tft_frame(df, ["hour", "temperature"])

    def test_legacy_adapter_weather_unknown(self):
        from src.data.adapters import tft_ready

        n = 60
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "group_id": ["single"] * n,
            "target": np.arange(n, dtype=float) + 100,
            "hour": [i % 24 for i in range(n)],
            "day_of_week": [(i // 24) % 7 for i in range(n)],
            "temperature": np.arange(n, dtype=float),
            "solar": np.arange(n, dtype=float),
            "wind": np.arange(n, dtype=float),
            "lag_1": np.arange(n, dtype=float),
            "rolling_mean_24": np.arange(n, dtype=float),
        })
        frame, info = tft_ready(df)
        for w in ("temperature", "solar_generation", "wind_generation"):
            self.assertNotIn(w, info["tft_known"])
        self.assertIn("hour", info["tft_known"])


class TestFix7MultiOriginNaive(unittest.TestCase):
    def test_persistence_differs_per_origin(self):
        from src.models.statistical import PersistenceForecaster

        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        m = PersistenceForecaster(horizon=6)
        out = m.predict_at_origins(series, [1, 2, 4], horizon=6)
        np.testing.assert_allclose(out, [20.0, 30.0, 50.0])

    def test_no_broadcast(self):
        from src.models.statistical import PersistenceForecaster

        series = np.arange(100, dtype=float)
        m = PersistenceForecaster(horizon=1).fit(np.zeros((10, 1)), series[:50])
        single = m.predict(np.zeros((3, 1)))
        # Single-tail broadcast repeats one value...
        self.assertTrue((single == single[0]).all())
        # ...while origin-aligned predictions follow each origin.
        aligned = m.predict_at_origins(series, [10, 20, 30])
        self.assertFalse((aligned == aligned[0]).all())
        np.testing.assert_allclose(aligned, [10.0, 20.0, 30.0])

    def test_seasonal_exact_source(self):
        from src.models.statistical import SeasonalPersistenceForecaster

        series = np.arange(300, dtype=float)
        m = SeasonalPersistenceForecaster(horizon=6)
        out = m.predict_at_origins(series, [200, 210, 220], horizon=6)
        # source = origin + 6 - 168
        np.testing.assert_allclose(out, [38.0, 48.0, 58.0])

    def test_seasonal_too_early_raises(self):
        from src.models.statistical import SeasonalPersistenceForecaster

        m = SeasonalPersistenceForecaster(horizon=24)
        with self.assertRaises(ValueError):
            m.predict_at_origins(np.arange(300, dtype=float), [100], horizon=24)

    def test_scorer_helper_aligns(self):
        from src.q1.evaluate import build_naive_preds_by_origin

        ts, load = _synth_series(500)
        origins = pd.DataFrame({
            "origin_timestamp": [ts[200], ts[210], ts[220]],
            "horizon": [1, 6, 24],
        })
        d = build_naive_preds_by_origin("persistence", load, ts, origins)
        self.assertEqual(len(d), 3)
        self.assertAlmostEqual(d[(ts[200].isoformat(), 1)], load[200])
        d2 = build_naive_preds_by_origin("seasonal_persistence", load, ts, origins)
        self.assertAlmostEqual(d2[(ts[220].isoformat(), 24)], load[220 + 24 - 168])


class TestFix8Defaults168(unittest.TestCase):
    def test_tft_default_168(self):
        from src.models.tft import TemporalFusionTransformerForecaster

        self.assertEqual(TemporalFusionTransformerForecaster().encoder_length, 168)

    def test_sequence_default_168(self):
        from src.models.deep import GRUForecaster, LSTMForecaster

        self.assertEqual(LSTMForecaster().seq_len, 168)
        self.assertEqual(GRUForecaster().seq_len, 168)

    def test_metrics_default_168(self):
        import inspect

        from src.evaluation.metrics import evaluate_regression, mase

        self.assertEqual(inspect.signature(mase).parameters["seasonality"].default, 168)
        self.assertEqual(
            inspect.signature(evaluate_regression).parameters["seasonality"].default, 168)

    def test_no_bundle_mention(self):
        import src.q1.horizons as H

        self.assertFalse(hasattr(H, "DirectHorizonBundle"))
        src = open(__import__("src.q1.horizons", fromlist=["__file__"]).__file__,
                   encoding="utf-8").read()
        self.assertNotIn("DirectHorizonBundle", src)

    def test_ratio_keys_marked_legacy(self):
        from pathlib import Path

        text = Path("configs/experiment.yaml").read_text(encoding="utf-8")
        self.assertIn("LEGACY", text)


class TestFix9OriginsHash(unittest.TestCase):
    def test_csv_roundtrip_same_hash(self):
        import tempfile
        from pathlib import Path

        from src.data.features_q1 import build_q1_features, drop_warmup
        from src.data.origins import generate_origins, origins_hash
        from tests.test_q1_protocol import _synth_v2

        feat, names = build_q1_features(_synth_v2(500))
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=(1, 6), history=168)
        h1 = origins_hash(org)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.csv"
            org.to_csv(p, index=False)
            reloaded = pd.read_csv(p)
            # String timestamps must not raise and must match.
            self.assertEqual(origins_hash(reloaded), h1)

    def test_locked_canonical_hash(self):
        from src.data.origins import origins_hash

        df = pd.read_csv("experiments/protocol/test_origins.csv")
        self.assertEqual(
            origins_hash(df),
            "573de597b4ac4a4f7f2db79ff57045a0308ec38e013a329f16ae8017b19fe2ce",
        )


class TestFix10Docs(unittest.TestCase):
    def test_protocol_docs(self):
        from pathlib import Path

        text = Path("docs/Q1_PROTOCOL.md").read_text(encoding="utf-8")
        for needle in (
            "SARIMA-lite",
            "168",
            "TRAINING targets only",
            "q1_headline_models",
            "require_q1_complete",
            "src/q1/tft_path",
            "predict_at_origins",
            "exact written path",
        ):
            self.assertIn(needle, text)
        self.assertNotIn("conventional SARIMA", text.replace("NOT conventional SARIMA", ""))


if __name__ == "__main__":
    unittest.main()
