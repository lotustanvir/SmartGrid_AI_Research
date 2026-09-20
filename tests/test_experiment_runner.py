"""Tests for the Phase 2B experiment runner (synthetic data only)."""

import unittest

import numpy as np
import pandas as pd

from src.models.baseline import (
    LightGBMForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)
from src.training.experiment_runner import ExperimentRunner, MODEL_REGISTRY
from src.training.trainer import fit_model, supports_validation
from src.utils.experiment import (
    FEATURE_COLS,
    TARGET_COL,
    chronological_split,
    generate_synthetic_grid_data,
)

SMALL_OVERRIDES = {
    "random_forest": {"n_estimators": 10},
    "xgboost": {"n_estimators": 20},
    "lightgbm": {"n_estimators": 20},
    "lstm": {"hidden_size": 8, "epochs": 2, "seq_len": 12},
    "gru": {"hidden_size": 8, "epochs": 2, "seq_len": 12},
    "tft": {
        "encoder_length": 12,
        "prediction_length": 3,
        "hidden_size": 8,
        "attention_head_size": 2,
        "hidden_continuous_size": 4,
        "max_epochs": 1,
        "batch_size": 16,
    },
    "hybrid": {
        "tft_config": {
            "encoder_length": 12,
            "prediction_length": 3,
            "hidden_size": 8,
            "attention_head_size": 2,
            "hidden_continuous_size": 4,
            "max_epochs": 1,
            "batch_size": 16,
        },
        "xgb_config": {"n_estimators": 10},
    },
}


class TestSyntheticData(unittest.TestCase):
    def test_columns_and_order(self):
        df = generate_synthetic_grid_data(n_samples=300, seed=1)
        self.assertEqual(list(df.columns), FEATURE_COLS + [TARGET_COL])
        self.assertEqual(len(df), 299)  # first row dropped (lag undefined)
        self.assertFalse(df.isna().any().any())
        self.assertTrue((df[TARGET_COL] > 0).all())

    def test_lag_has_no_leakage(self):
        df = generate_synthetic_grid_data(n_samples=300, seed=1)
        demand = df[TARGET_COL].to_numpy()
        prev = df["previous_load"].to_numpy()
        np.testing.assert_allclose(prev[1:], demand[:-1])

    def test_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            generate_synthetic_grid_data(n_samples=10)

    def test_chronological_split_order(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        X_tr, X_val, X_te, _, _, _ = chronological_split(X, y, 0.7, 0.15)
        self.assertTrue(X_tr.max() < X_val.min() < X_te.min())
        self.assertEqual(len(X_tr) + len(X_val) + len(X_te), 100)

    def test_chronological_split_bad_ratios(self):
        with self.assertRaises(ValueError):
            chronological_split(np.zeros((10, 2)), np.zeros(10), 0.8, 0.2)


class TestTrainerHelpers(unittest.TestCase):
    def test_supports_validation(self):
        self.assertTrue(supports_validation(XGBoostForecaster()))
        self.assertTrue(supports_validation(LightGBMForecaster()))
        self.assertFalse(supports_validation(LinearRegressionForecaster()))
        self.assertFalse(supports_validation(RandomForestForecaster()))

    def test_fit_model_without_validation(self):
        X = np.random.RandomState(0).rand(40, 3)
        y = X[:, 0] * 2 + 1
        m = fit_model(LinearRegressionForecaster(), X, y)
        self.assertEqual(m.predict(X).shape, (40,))


class TestExperimentRunner(unittest.TestCase):
    def _data(self):
        df = generate_synthetic_grid_data(n_samples=400, seed=42)
        return df[FEATURE_COLS].to_numpy(), df[TARGET_COL].to_numpy()

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            ExperimentRunner(output_dir="tmp").build_model("not_a_model")

    def test_registry_covers_baselines(self):
        self.assertEqual(
            set(MODEL_REGISTRY),
            {
                "linear_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
                "lstm",
                "gru",
                "tft",
                "hybrid",
            },
        )

    def test_all_models_execute(self):
        import tempfile

        X, y = self._data()
        with tempfile.TemporaryDirectory() as d:
            runner = ExperimentRunner(output_dir=d)
            results = runner.run_all(X, y, model_overrides=SMALL_OVERRIDES)
        self.assertEqual(list(results.columns), ["Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"])
        self.assertEqual(len(results), 8)
        for col in ["MAE", "RMSE", "MAPE", "sMAPE", "R2"]:
            self.assertTrue(np.all(np.isfinite(results[col].to_numpy())), col)

    def test_csv_saved(self):
        import os
        import tempfile

        X, y = self._data()
        with tempfile.TemporaryDirectory() as d:
            ExperimentRunner(output_dir=d).run_all(
                X, y, models=["linear_regression"], model_overrides=SMALL_OVERRIDES
            )
            csv = os.path.join(d, "results.csv")
            self.assertTrue(os.path.isfile(csv))
            df = pd.read_csv(csv)
            self.assertEqual(list(df.columns), ["Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"])
            self.assertEqual(df.iloc[0]["Model"], "LinearRegression")

    def test_plots_generated(self):
        import os
        import tempfile

        X, y = self._data()
        with tempfile.TemporaryDirectory() as d:
            ExperimentRunner(output_dir=d).run_all(
                X, y, models=["linear_regression", "random_forest"],
                model_overrides=SMALL_OVERRIDES,
            )
            figs = os.path.join(d, "figures")
            self.assertTrue(os.path.isfile(os.path.join(figs, "model_comparison.png")))
            self.assertTrue(os.path.isfile(os.path.join(figs, "pred_vs_actual_linear_regression.png")))
            self.assertTrue(os.path.isfile(os.path.join(figs, "pred_vs_actual_random_forest.png")))

    def test_run_one_row(self):
        X, y = self._data()
        runner = ExperimentRunner(output_dir="tmp_no_write")
        row, y_te, preds = runner.run_one("linear_regression", X, y)
        self.assertEqual(set(row), {"Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"})
        self.assertEqual(preds.shape, y_te.shape)


if __name__ == "__main__":
    unittest.main()
