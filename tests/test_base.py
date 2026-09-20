"""Tests for src.models.base.BaseForecaster."""

import os
import tempfile
import unittest

import numpy as np

from src.models.base import BaseForecaster


class DummyForecaster(BaseForecaster):
    """Minimal mean-predictor used to exercise the base interface."""

    def fit(self, X, y, **kwargs):
        X, y = self._validate_xy(X, y)
        self.mean_ = float(np.mean(y))
        self.n_features_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        X = self._validate_X(X, n_features=self.n_features_)
        return np.full(X.shape[0], self.mean_)


class TestBaseForecaster(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(20, 3)
        self.y = rng.rand(20) * 10

    def test_fit_predict_shape(self):
        m = DummyForecaster(random_state=42)
        m.fit(self.X, self.y)
        preds = m.predict(self.X)
        self.assertEqual(preds.shape, (20,))

    def test_evaluate_keys(self):
        m = DummyForecaster(random_state=42).fit(self.X, self.y)
        out = m.evaluate(self.X, self.y)
        self.assertEqual(set(out), {"MAE", "RMSE", "MAPE", "sMAPE", "R2"})

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DummyForecaster().predict(self.X)

    def test_save_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DummyForecaster().save(os.path.join(tempfile.gettempdir(), "x.pkl"))

    def test_save_load_roundtrip(self):
        m = DummyForecaster(random_state=42).fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model.pkl")
            m.save(p)
            m2 = DummyForecaster.load(p)
        np.testing.assert_array_equal(m.predict(self.X), m2.predict(self.X))

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            DummyForecaster.load("does_not_exist_xyz.pkl")

    def test_predict_interval_not_implemented(self):
        m = DummyForecaster(random_state=42).fit(self.X, self.y)
        with self.assertRaises(NotImplementedError):
            m.predict_interval(self.X)

    def test_nan_rejected(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            DummyForecaster().fit(X, self.y)

    def test_feature_mismatch_rejected(self):
        m = DummyForecaster(random_state=42).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            m.predict(np.zeros((5, 9)))

    def test_bad_random_state_rejected(self):
        with self.assertRaises(ValueError):
            DummyForecaster(random_state=-1)

    def test_deterministic_seed(self):
        a = DummyForecaster(random_state=1).fit(self.X, self.y).predict(self.X)
        b = DummyForecaster(random_state=1).fit(self.X, self.y).predict(self.X)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
