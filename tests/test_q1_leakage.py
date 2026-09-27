"""Q1 leakage / protocol fail-closed tests (Phase 2, 15 items)."""

import unittest

import numpy as np
import pandas as pd


def _v2(n=500, seed=0):
    from tests.test_q1_protocol import _synth_v2

    return _synth_v2(n, seed)


class TestQ1Leakage(unittest.TestCase):
    def test_01_chronological_split(self):
        from src.data.splits_q1 import split_calendar

        ts = pd.date_range("2023-06-01", periods=24 * 800, freq="h")
        tr, va, te = split_calendar(pd.DataFrame({"timestamp": ts}))
        self.assertLess(tr["timestamp"].max(), va["timestamp"].min())
        self.assertLess(va["timestamp"].max(), te["timestamp"].min())

    def test_02_no_train_test_overlap(self):
        from src.data.features_q1 import build_q1_features, drop_warmup
        from src.data.splits_q1 import split_calendar

        raw = _v2(20000)
        raw["timestamp"] = pd.date_range("2023-06-01", periods=len(raw), freq="h")
        feat, names = build_q1_features(raw)
        clean, _ = drop_warmup(feat, names)
        tr, va, te = split_calendar(clean)
        self.assertEqual(
            len(set(tr["timestamp"]).intersection(set(te["timestamp"]))), 0
        )

    def test_03_scaler_train_only(self):
        from src.data.adapters import fit_scaler

        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, (100, 3))
        Xte = rng.normal(5, 1, (20, 3))
        scaler, _ = fit_scaler(Xtr)
        np.testing.assert_allclose(scaler.mean_, Xtr.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, np.vstack([Xtr, Xte]).mean(axis=0)))

    def test_04_lag_correctness(self):
        from src.data.features_q1 import build_q1_features

        raw = _v2(300)
        feat, _ = build_q1_features(raw)
        self.assertAlmostEqual(feat["lag_1"].iloc[100], raw["pjm_load_mw"].iloc[99])

    def test_05_rolling_correctness(self):
        from src.data.features_q1 import build_q1_features

        raw = _v2(300)
        feat, _ = build_q1_features(raw)
        expect = raw["pjm_load_mw"].iloc[76:100].mean()
        self.assertAlmostEqual(feat["rolling_mean_24"].iloc[100], expect)

    def test_06_weather_lag_correctness(self):
        from src.data.features_q1 import build_q1_features

        raw = _v2(300)
        feat, _ = build_q1_features(raw)
        self.assertAlmostEqual(feat["ws_lag_24"].iloc[100], raw["wind_speed"].iloc[76])

    def test_07_no_future_weather_cols(self):
        from src.data.features_q1 import canonical_feature_names

        feats = canonical_feature_names()
        for raw_col in ("temperature", "humidity", "wind_speed", "cloud_cover",
                        "solar_radiation", "wind_generation_mw", "solar_generation_mw"):
            self.assertNotIn(raw_col, feats)

    def test_08_no_target_derived(self):
        from src.data.features_q1 import canonical_feature_names

        feats = canonical_feature_names()
        self.assertNotIn("net_load", feats)
        self.assertNotIn("renewable_penetration", feats)

    def test_09_origin_validity(self):
        from src.data.features_q1 import build_q1_features, drop_warmup
        from src.data.origins import generate_origins

        feat, names = build_q1_features(_v2(500))
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=(6,), history=168)
        # Every origin has full history + horizon inside frame.
        self.assertTrue((org["horizon"] == 6).all())
        self.assertTrue((org["required_history_end"] < org["target_start"]).all())

    def test_10_equal_origin_sets(self):
        from src.data.origins import assert_same_origins
        from src.data.features_q1 import build_q1_features, drop_warmup
        from src.data.origins import generate_origins

        feat, names = build_q1_features(_v2(500))
        clean, _ = drop_warmup(feat, names)
        a = generate_origins(clean, horizons=(1,), history=168)
        with self.assertRaises(ValueError):
            assert_same_origins(a, a.iloc[:-1])

    def test_11_tft_roles_calendar_only(self):
        from src.models.tft.dataset import resolve_roles

        df = pd.DataFrame(
            {
                "time_idx": range(10),
                "group_id": ["single"] * 10,
                "electricity_demand": np.arange(10, dtype=float),
                "hour": range(10),
                "day_of_week": [0] * 10,
                "temp_lag_1": np.arange(10, dtype=float),
                "lag_1": np.arange(10, dtype=float),
            }
        )
        known, unknown, _ = resolve_roles(df)
        self.assertNotIn("temp_lag_1", known)
        self.assertNotIn("lag_1", known)
        self.assertIn("temp_lag_1", unknown)

    def test_12_hybrid_oof_folds_default5(self):
        from src.models.hybrid import HybridTFTXGBoostForecaster

        m = HybridTFTXGBoostForecaster(
            tft_config={"encoder_length": 12, "prediction_length": 3,
                        "hidden_size": 8, "attention_head_size": 2,
                        "hidden_continuous_size": 4, "max_epochs": 1,
                        "batch_size": 16},
            xgb_config={"n_estimators": 5},
        )
        self.assertEqual(m.n_oof_folds, 5)
        # Chronology helper on synthetic times.
        times = np.arange(120)
        bounds = m._oof_fold_bounds(times)
        self.assertEqual(len(bounds), 5)
        for te1, bs1, be1 in bounds:
            self.assertLess(te1, bs1)
            self.assertLessEqual(bs1, be1)
        tes = [b[0] for b in bounds]
        self.assertEqual(tes, sorted(tes))

    def test_13_oof_no_test_timestamps(self):
        # Structural: OOF bounds tile within the fit frame; test exclusion
        # is enforced by callers fitting on train/val only (retired runner
        # raises fail-closed). Here verify bounds never exceed max fit time.
        from src.models.hybrid import HybridTFTXGBoostForecaster

        m = HybridTFTXGBoostForecaster(
            tft_config={"encoder_length": 12, "prediction_length": 3,
                        "hidden_size": 8, "attention_head_size": 2,
                        "hidden_continuous_size": 4, "max_epochs": 1,
                        "batch_size": 16},
            xgb_config={"n_estimators": 5},
        )
        times = np.arange(100)
        for _, _, be in m._oof_fold_bounds(times):
            self.assertLessEqual(be, 99)

    def test_14_checkpoint_from_validation(self):
        from src.models.tft import TemporalFusionTransformerForecaster

        m = TemporalFusionTransformerForecaster(
            encoder_length=12, prediction_length=3, hidden_size=8,
            attention_head_size=2, hidden_continuous_size=4,
            max_epochs=1, batch_size=16,
        )
        # Trainer wiring guarantees val-loss checkpointing.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            t = m._trainer(Path(d))
            cbs = {type(c).__name__ for c in t.callbacks}
            self.assertIn("EarlyStopping", cbs)
            self.assertIn("ModelCheckpoint", cbs)

    def test_15_common_timestamps(self):
        from src.q1.evaluate import check_common_timestamps

        a = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=5, freq="h")})
        b = a.copy()
        check_common_timestamps({"m1": a, "m2": b})
        c = a.iloc[1:].reset_index(drop=True)
        with self.assertRaises(ValueError):
            check_common_timestamps({"m1": a, "m2": c})

    def test_legacy_runner_blocked(self):
        import run_phase_6b as R

        with self.assertRaises(RuntimeError):
            R.run_all_experiments()


if __name__ == "__main__":
    unittest.main()
