"""Tests for src.evaluation.metrics."""

import unittest

import numpy as np

from src.evaluation.metrics import (
    evaluate_regression,
    mae,
    rmse,
    r2,
    safe_mape,
    smape,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([10.0, 20.0, 30.0, 40.0])
        self.y_pred = np.array([11.0, 19.0, 30.0, 44.0])

    def test_metric_keys(self):
        out = evaluate_regression(self.y_true, self.y_pred)
        self.assertEqual(set(out), {"MAE", "RMSE", "MAPE", "sMAPE", "R2"})

    def test_perfect_prediction(self):
        out = evaluate_regression(self.y_true, self.y_true)
        self.assertAlmostEqual(out["MAE"], 0.0)
        self.assertAlmostEqual(out["RMSE"], 0.0)
        self.assertAlmostEqual(out["MAPE"], 0.0)
        self.assertAlmostEqual(out["sMAPE"], 0.0)
        self.assertAlmostEqual(out["R2"], 1.0)

    def test_zero_targets_no_inf(self):
        yt = np.array([0.0, 0.0, 5.0, 10.0])
        yp = np.array([0.5, 0.0, 5.0, 8.0])
        self.assertTrue(np.isfinite(safe_mape(yt, yp)))
        self.assertTrue(np.isfinite(smape(yt, yp)))
        out = evaluate_regression(yt, yp)
        for v in out.values():
            self.assertTrue(np.isfinite(v))

    def test_mae_rmse_r2_values(self):
        self.assertAlmostEqual(mae(self.y_true, self.y_pred), 1.5)
        self.assertAlmostEqual(rmse(self.y_true, self.y_pred), np.sqrt(4.5))
        self.assertGreater(r2(self.y_true, self.y_pred), 0.95)

    def test_nan_rejected(self):
        bad = self.y_true.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_regression(bad, self.y_pred)
        with self.assertRaises(ValueError):
            evaluate_regression(self.y_true, bad)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_regression([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_regression([], [])


if __name__ == "__main__":
    unittest.main()
