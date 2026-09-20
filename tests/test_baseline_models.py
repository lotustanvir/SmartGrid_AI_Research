"""Tests for src.models.baseline (Phase 2A).

Synthetic data only (sklearn make_regression) for software verification.
No real dataset is connected; nothing here is a research result.
"""

import os
import tempfile
import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.models.baseline import (
    LightGBMForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    load_params,
)

EXPECTED_KEYS = {"MAE", "RMSE", "MAPE", "sMAPE", "R2"}


def make_data():
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=7,
        noise=5.0,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, random_state=42)


class BaselineMixin:
    model_cls = None
    model_kwargs: dict = {}

    def _split(self):
        return make_data()

    def _make(self, **overrides):
        kw = dict(self.model_kwargs)
        kw.update(overrides)
        return self.model_cls(**kw)

    def test_init_defaults(self):
        m = self.model_cls()
        self.assertEqual(m.random_state, 42)
        self.assertFalse(m._is_fitted)

    def test_from_config(self):
        m = self.model_cls.from_config()
        self.assertEqual(m.random_state, 42)

    def test_load_params_keys(self):
        name = {
            LinearRegressionForecaster: "linear_regression",
            RandomForestForecaster: "random_forest",
            XGBoostForecaster: "xgboost",
            LightGBMForecaster: "lightgbm",
        }[self.model_cls]
        params = load_params(name)
        self.assertIn("random_state", params)

    def test_fit_predict_shape(self):
        X_tr, X_te, y_tr, y_te = self._split()
        m = self._make().fit(X_tr, y_tr)
        preds = m.predict(X_te)
        self.assertEqual(preds.shape, (X_te.shape[0],))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_evaluate(self):
        X_tr, X_te, y_tr, y_te = self._split()
        m = self._make().fit(X_tr, y_tr)
        out = m.evaluate(X_te, y_te)
        self.assertEqual(set(out), EXPECTED_KEYS)
        for v in out.values():
            self.assertTrue(np.isfinite(v), f"non-finite metric: {out}")

    def test_save_load_roundtrip(self):
        X_tr, X_te, y_tr, _ = self._split()
        m = self._make().fit(X_tr, y_tr)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model.pkl")
            m.save(p)
            m2 = self.model_cls.load(p)
        np.testing.assert_allclose(m.predict(X_te), m2.predict(X_te), rtol=1e-10)

    def test_nan_rejected(self):
        X_tr, _, y_tr, _ = self._split()
        X_bad = X_tr.copy()
        X_bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self._make().fit(X_bad, y_tr)

    def test_feature_mismatch_rejected(self):
        X_tr, _, y_tr, _ = self._split()
        m = self._make().fit(X_tr, y_tr)
        with self.assertRaises(ValueError):
            m.predict(np.zeros((4, X_tr.shape[1] + 3)))


class TestLinearRegression(BaselineMixin, unittest.TestCase):
    model_cls = LinearRegressionForecaster

    def test_regularization_variants(self):
        X_tr, X_te, y_tr, _ = self._split()
        for reg in (None, "ridge", "lasso"):
            with self.subTest(regularization=reg):
                m = LinearRegressionForecaster(regularization=reg).fit(X_tr, y_tr)
                self.assertEqual(m.predict(X_te).shape, (X_te.shape[0],))
                self.assertEqual(m.coefficients_.shape, (X_tr.shape[1],))

    def test_bad_regularization_rejected(self):
        with self.assertRaises(ValueError):
            LinearRegressionForecaster(regularization="elastic")


class TestRandomForest(BaselineMixin, unittest.TestCase):
    model_cls = RandomForestForecaster
    model_kwargs = {"n_estimators": 10}

    def test_feature_importances(self):
        X_tr, _, y_tr, _ = self._split()
        m = self._make().fit(X_tr, y_tr)
        imp = m.feature_importances_
        self.assertEqual(imp.shape, (X_tr.shape[1],))
        self.assertAlmostEqual(float(np.sum(imp)), 1.0, places=5)


class TestXGBoost(BaselineMixin, unittest.TestCase):
    model_cls = XGBoostForecaster
    model_kwargs = {"n_estimators": 50}

    def test_feature_importances(self):
        X_tr, _, y_tr, _ = self._split()
        m = self._make().fit(X_tr, y_tr)
        self.assertEqual(m.feature_importances_.shape, (X_tr.shape[1],))

    def test_early_stopping_with_val(self):
        X_tr, X_te, y_tr, y_te = self._split()
        m = self._make(early_stopping_rounds=5).fit(X_tr, y_tr, X_te, y_te)
        self.assertIsNotNone(m.best_iteration_)
        self.assertEqual(m.predict(X_te).shape, (X_te.shape[0],))

    def test_early_stopping_without_val_raises(self):
        X_tr, _, y_tr, _ = self._split()
        with self.assertRaises(ValueError):
            self._make(early_stopping_rounds=5).fit(X_tr, y_tr)


class TestLightGBM(BaselineMixin, unittest.TestCase):
    model_cls = LightGBMForecaster
    model_kwargs = {"n_estimators": 50}

    def test_feature_importances(self):
        X_tr, _, y_tr, _ = self._split()
        m = self._make().fit(X_tr, y_tr)
        self.assertEqual(m.feature_importances_.shape, (X_tr.shape[1],))

    def test_early_stopping_with_val(self):
        X_tr, X_te, y_tr, y_te = self._split()
        m = self._make(early_stopping_rounds=5).fit(X_tr, y_tr, X_te, y_te)
        self.assertIsNotNone(m.best_iteration_)
        self.assertEqual(m.predict(X_te).shape, (X_te.shape[0],))

    def test_early_stopping_without_val_raises(self):
        X_tr, _, y_tr, _ = self._split()
        with self.assertRaises(ValueError):
            self._make(early_stopping_rounds=5).fit(X_tr, y_tr)


if __name__ == "__main__":
    unittest.main()
