"""Tests for src.models.tft (Phase 4A, synthetic data only)."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.models.tft import (
    TemporalFusionTransformerForecaster,
    build_synthetic_tft_frame,
    frame_from_arrays,
    validate_frame,
)

EXPECTED_KEYS = {"MAE", "RMSE", "MAPE", "sMAPE", "R2"}
TINY_KW = {
    "encoder_length": 12,
    "prediction_length": 3,
    "hidden_size": 8,
    "attention_head_size": 2,
    "hidden_continuous_size": 4,
    "max_epochs": 1,
    "patience": 3,
    "batch_size": 16,
}


def tiny_frame():
    return build_synthetic_tft_frame(n_groups=2, n_steps=120, seed=7)


class TestTFTDataset(unittest.TestCase):
    def test_frame_columns(self):
        df = tiny_frame()
        self.assertEqual(
            list(df.columns),
            [
                "time_idx",
                "group_id",
                "electricity_demand",
                "temperature",
                "solar_generation",
                "wind_generation",
                "hour",
                "day_of_week",
            ],
        )
        self.assertFalse(df.isna().any().any())
        self.assertEqual(df["group_id"].nunique(), 2)

    def test_frame_from_arrays(self):
        X = np.random.RandomState(0).rand(60, 4)
        y = np.random.RandomState(1).rand(60) + 10
        df = frame_from_arrays(X, y)
        self.assertEqual(len(df), 60)
        self.assertIn("electricity_demand", df.columns)
        self.assertFalse(df.drop(columns=["electricity_demand"]).isna().any().any())

    def test_validate_rejects_missing_column(self):
        df = tiny_frame().drop(columns=["group_id"])
        with self.assertRaises(ValueError):
            validate_frame(df)

    def test_validate_rejects_nan(self):
        df = tiny_frame()
        df.loc[0, "temperature"] = np.nan
        with self.assertRaises(ValueError):
            validate_frame(df)


class TestTFTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = tiny_frame()
        cls.model = TemporalFusionTransformerForecaster(**TINY_KW).fit(cls.df)

    def test_init_defaults(self):
        m = TemporalFusionTransformerForecaster()
        self.assertEqual((m.encoder_length, m.prediction_length), (24, 6))
        self.assertEqual(m.quantiles, (0.05, 0.5, 0.95))
        self.assertFalse(m._is_fitted)

    def test_from_config(self):
        m = TemporalFusionTransformerForecaster.from_config()
        self.assertEqual((m.encoder_length, m.prediction_length), (24, 6))

    def test_bad_config_rejected(self):
        with self.assertRaises(ValueError):
            TemporalFusionTransformerForecaster(quantiles=(0.1, 0.9))
        with self.assertRaises(ValueError):
            TemporalFusionTransformerForecaster(hidden_size=7, attention_head_size=2)

    def test_one_epoch_fit(self):
        self.assertTrue(self.model._is_fitted)
        self.assertEqual(self.model.n_epochs_, 1)
        self.assertTrue(np.isfinite(self.model.best_val_loss_))

    def test_forward_predict_shape(self):
        preds = self.model.predict(self.df)
        self.assertEqual(preds.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_interval(self):
        out = self.model.predict_interval(self.df)
        self.assertEqual(set(out), {0.05, 0.5, 0.95})
        for arr in out.values():
            self.assertEqual(arr.shape, (2, 3))

    def test_evaluate(self):
        out = self.model.evaluate(self.df)
        self.assertEqual(set(out), EXPECTED_KEYS)
        for v in out.values():
            self.assertTrue(np.isfinite(v), f"non-finite metric: {out}")

    def test_interpret(self):
        interp = self.model.interpret(self.df)
        self.assertEqual(
            set(interp),
            {"attention", "encoder_variables", "decoder_variables", "static_variables"},
        )
        self.assertEqual(interp["attention"].shape, (12,))
        self.assertTrue(interp["encoder_variables"])
        self.assertTrue(interp["decoder_variables"])

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tft.pt")
            self.model.save(p)
            m2 = TemporalFusionTransformerForecaster.load(p)
        np.testing.assert_allclose(
            self.model.predict(self.df), m2.predict(self.df), rtol=1e-5
        )

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            TemporalFusionTransformerForecaster.load("does_not_exist_xyz.pt")

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TemporalFusionTransformerForecaster(**TINY_KW).predict(self.df)

    def test_array_inputs(self):
        df = self.df[self.df["group_id"] == "group_0"].reset_index(drop=True)
        X = df[["temperature", "solar_generation", "wind_generation"]].to_numpy()
        y = df["electricity_demand"].to_numpy()
        m = TemporalFusionTransformerForecaster(**TINY_KW).fit(X, y)
        preds, actuals = m.forecast(X, y)
        self.assertEqual(preds.shape, (1, 3))
        self.assertEqual(actuals.shape, (1, 3))
        out = m.evaluate(X, y)
        self.assertEqual(set(out), EXPECTED_KEYS)

    def test_predict_without_history_raises(self):
        X = np.random.RandomState(0).rand(60, 3)
        with self.assertRaises(ValueError):
            self.model.predict(X)


if __name__ == "__main__":
    unittest.main()
