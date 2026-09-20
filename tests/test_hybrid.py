"""Tests for src.models.hybrid (Phase 5, synthetic data only)."""

import os
import tempfile
import unittest

import numpy as np

from src.models.hybrid import HybridTFTXGBoostForecaster
from src.models.tft import build_synthetic_tft_frame

EXPECTED_KEYS = {"MAE", "RMSE", "MAPE", "sMAPE", "R2"}
TINY_TFT = {
    "encoder_length": 12,
    "prediction_length": 3,
    "hidden_size": 8,
    "attention_head_size": 2,
    "hidden_continuous_size": 4,
    "max_epochs": 1,
    "patience": 3,
    "batch_size": 16,
}
TINY_XGB = {"n_estimators": 10}


def tiny_frame():
    return build_synthetic_tft_frame(n_groups=2, n_steps=120, seed=11)


class TestHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = tiny_frame()
        cls.model = HybridTFTXGBoostForecaster(
            tft_config=dict(TINY_TFT), xgb_config=dict(TINY_XGB)
        ).fit(cls.df)

    def test_init_defaults(self):
        m = HybridTFTXGBoostForecaster()
        self.assertEqual(m._tft.encoder_length, 24)
        self.assertEqual(m.tft_config, {"random_state": 42})
        self.assertFalse(m._is_fitted)

    def test_from_config(self):
        m = HybridTFTXGBoostForecaster.from_config()
        self.assertEqual(m.tft_config["encoder_length"], 24)
        self.assertIn("n_estimators", m.xgb_config)

    def test_fit_pipeline(self):
        self.assertTrue(self.model._is_fitted)
        self.assertTrue(self.model._tft._is_fitted)
        self.assertTrue(self.model._xgb._is_fitted)
        self.assertGreater(self.model.n_residual_samples_, 0)
        self.assertTrue(self.model.residual_feature_names_)
        self.assertIn("mean_abs_residual_before", self.model.residual_stats_)

    def test_residual_definition(self):
        before = self.model.residual_stats_["mean_abs_residual_before"]
        after = self.model.residual_stats_["mean_abs_residual_after"]
        self.assertGreater(before, 0.0)
        self.assertGreaterEqual(before, 0.0)
        self.assertLessEqual(after, before + 1e-9)

    def test_predict_shape(self):
        preds = self.model.predict(self.df)
        self.assertEqual(preds.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_correction_applied(self):
        tft_med = self.model._tft.predict(self.df)
        hyb = self.model.predict(self.df)
        self.assertFalse(np.allclose(tft_med, hyb))

    def test_forecast(self):
        preds, actuals = self.model.forecast(self.df)
        self.assertEqual(preds.shape, actuals.shape)
        self.assertEqual(preds.shape, (2, 3))

    def test_evaluate(self):
        out = self.model.evaluate(self.df)
        self.assertEqual(set(out), EXPECTED_KEYS)
        for v in out.values():
            self.assertTrue(np.isfinite(v), f"non-finite metric: {out}")

    def test_predict_interval(self):
        out = self.model.predict_interval(self.df)
        self.assertEqual(
            set(out), {"base_tft_lower", "hybrid_median", "base_tft_upper"}
        )
        np.testing.assert_allclose(
            out["hybrid_median"], self.model.predict(self.df), rtol=1e-6
        )
        diag = self.model.interval_diagnostics(self.df)
        self.assertEqual(
            set(diag), {"n_points", "n_violations", "violation_idx", "clipped"}
        )
        self.assertFalse(diag["clipped"])

    def test_ablation(self):
        comp = self.model.ablation(self.df)
        self.assertEqual(set(comp["tft"]), EXPECTED_KEYS)
        self.assertEqual(set(comp["hybrid"]), EXPECTED_KEYS)
        self.assertIn("mean_abs_residual_before", comp)
        self.assertIn("mean_abs_residual_after", comp)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hybrid.pt")
            self.model.save(p)
            m2 = HybridTFTXGBoostForecaster.load(p)
        np.testing.assert_allclose(
            self.model.predict(self.df), m2.predict(self.df), rtol=1e-5
        )
        self.assertEqual(
            m2.residual_feature_names_, self.model.residual_feature_names_
        )

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            HybridTFTXGBoostForecaster.load("does_not_exist_xyz.pt")

    def test_array_fit_forecast(self):
        df = self.df[self.df["group_id"] == "group_0"].reset_index(drop=True)
        X = df[["temperature", "solar_generation", "wind_generation"]].to_numpy()
        y = df["electricity_demand"].to_numpy()
        m = HybridTFTXGBoostForecaster(
            tft_config=dict(TINY_TFT), xgb_config=dict(TINY_XGB)
        ).fit(X, y)
        preds, actuals = m.forecast(X, y)
        self.assertEqual(preds.shape, (1, 3))
        out = m.evaluate(X, y)
        self.assertEqual(set(out), EXPECTED_KEYS)


if __name__ == "__main__":
    unittest.main()
