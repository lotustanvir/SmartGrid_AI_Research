"""LightGBM validation-pipeline regression tests (Phase 1 critical fix).

Guards the eval_set API mismatch: validation must flow through the
canonical ``eval_set=[(X_val, y_val)]`` sklearn API so that
early stopping tracks the real chronological validation metric,
and validation data must never alter the fitted trees except
through early stopping (no leakage).
"""

import unittest

import numpy as np

from src.models.baseline.lightgbm_model import LightGBMForecaster
from src.training.experiment_runner import MODEL_REGISTRY, ExperimentRunner
from src.training.trainer import supports_validation

TINY_KW = {"n_estimators": 50, "n_jobs": 1}


def make_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 2.0 * X[:, 0] - X[:, 1] + rng.randn(n) * 0.1
    return X, y


class TestLightGBMValidationAPI(unittest.TestCase):
    def test_eval_set_tracked_with_early_stopping(self):
        X, y = make_data()
        m = LightGBMForecaster(**TINY_KW, early_stopping_rounds=5).fit(
            X[:200], y[:200], X[200:250], y[200:250]
        )
        evals = m.model.evals_result_
        self.assertIn("valid_0", evals)
        self.assertIn("l2", evals["valid_0"])
        scores = list(evals["valid_0"]["l2"])
        self.assertLessEqual(len(scores), TINY_KW["n_estimators"])
        # best_iteration_ (1-indexed) must point at the minimum val score.
        self.assertEqual(m.best_iteration_, int(np.argmin(scores)) + 1)
        preds = m.predict(X[250:])
        self.assertEqual(preds.shape, (50,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_validation_does_not_alter_trees_without_early_stopping(self):
        # Without early stopping the eval set is tracked but must not
        # change the fitted model: identical predictions with/without it.
        X, y = make_data()
        a = LightGBMForecaster(**TINY_KW).fit(X[:200], y[:200], X[200:250], y[200:250])
        b = LightGBMForecaster(**TINY_KW).fit(X[:200], y[:200])
        self.assertIn("valid_0", a.model.evals_result_)
        np.testing.assert_allclose(a.predict(X[250:]), b.predict(X[250:]), rtol=1e-9)

    def test_early_stopping_without_val_raises(self):
        X, y = make_data()
        with self.assertRaises(ValueError):
            LightGBMForecaster(**TINY_KW, early_stopping_rounds=5).fit(X, y)

    def test_half_validation_raises(self):
        X, y = make_data()
        with self.assertRaises(ValueError):
            LightGBMForecaster(**TINY_KW).fit(X, y, X[:50], None)

    def test_feature_mismatch_raises(self):
        X, y = make_data()
        with self.assertRaises(ValueError):
            LightGBMForecaster(**TINY_KW).fit(
                X, y, np.zeros((50, X.shape[1] + 1)), y[:50]
            )

    def test_registry_and_runner_hook_preserved(self):
        self.assertIs(MODEL_REGISTRY["lightgbm"], LightGBMForecaster)
        runner = ExperimentRunner(output_dir="results/test_tmp")
        model = runner.build_model("lightgbm", **TINY_KW)
        self.assertIsInstance(model, LightGBMForecaster)
        self.assertTrue(supports_validation(model))


if __name__ == "__main__":
    unittest.main()
