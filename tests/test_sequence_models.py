"""Tests for src.models.deep (Phase 3A, synthetic data only)."""

import os
import tempfile
import unittest

import numpy as np
import torch

from src.models.deep import GRUForecaster, LSTMForecaster
from src.training.trainer import supports_validation
from src.utils.experiment import FEATURE_COLS, TARGET_COL, generate_synthetic_grid_data

EXPECTED_KEYS = {"MAE", "RMSE", "MAPE", "sMAPE", "R2"}
SMALL_KW = {
    "hidden_size": 8,
    "num_layers": 1,
    "seq_len": 12,
    "batch_size": 32,
    "epochs": 3,
    "patience": 10,
}


def make_data(n=300):
    df = generate_synthetic_grid_data(n_samples=n, seed=42)
    return df[FEATURE_COLS].to_numpy(), df[TARGET_COL].to_numpy()


class SequenceMixin:
    model_cls = None

    def _make(self, **overrides):
        kw = dict(SMALL_KW)
        kw.update(overrides)
        return self.model_cls(**kw)

    def test_init_defaults(self):
        m = self.model_cls()
        self.assertEqual((m.hidden_size, m.seq_len, m.output_size), (64, 24, 1))
        self.assertEqual(m.random_state, 42)
        self.assertFalse(m._is_fitted)
        self.assertIn(str(m.device), ("cpu", "cuda"))

    def test_from_config(self):
        m = self.model_cls.from_config()
        self.assertEqual((m.hidden_size, m.seq_len), (64, 24))

    def test_forward_pass(self):
        m = self._make()
        net = m.build_module(input_size=6)
        net.eval()
        with torch.no_grad():
            out = net(torch.rand(4, 12, 6))
        self.assertEqual(tuple(out.shape), (4, 1))

    def test_forward_pass_multistep(self):
        m = self._make(output_size=3)
        net = m.build_module(input_size=6)
        net.eval()
        with torch.no_grad():
            out = net(torch.rand(4, 12, 6))
        self.assertEqual(tuple(out.shape), (4, 3))

    def test_fit_predict_2d(self):
        X, y = make_data()
        m = self._make().fit(X, y)
        preds = m.predict(X)
        self.assertEqual(preds.shape, (len(X) - 12 - 1 + 1,))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(len(m.history_["train_loss"]) >= 1)
        self.assertTrue(np.isfinite(m.best_val_loss_))

    def test_fit_predict_3d(self):
        X, y = make_data()
        Xw = np.stack([X[i : i + 12] for i in range(50)])
        yw = y[12 : 12 + 50]
        m = self._make().fit(Xw, yw)
        preds = m.predict(Xw)
        self.assertEqual(preds.shape, (50,))

    def test_evaluate(self):
        X, y = make_data()
        m = self._make().fit(X, y)
        out = m.evaluate(X, y)
        self.assertEqual(set(out), EXPECTED_KEYS)
        for v in out.values():
            self.assertTrue(np.isfinite(v), f"non-finite metric: {out}")

    def test_save_load_roundtrip(self):
        X, _ = make_data()
        m = self._make().fit(X, make_data()[1])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ckpt.pt")
            m.save(p)
            m2 = self.model_cls.load(p)
        np.testing.assert_allclose(m.predict(X), m2.predict(X), rtol=1e-5)

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.model_cls.load("does_not_exist_xyz.pt")

    def test_early_stopping_triggers(self):
        X, y = make_data()
        m = self._make(epochs=20, patience=1, min_delta=1e12).fit(X, y)
        self.assertEqual(m.n_epochs_, 2)

    def test_explicit_validation(self):
        X, y = make_data()
        m = self._make(epochs=2).fit(X[:200], y[:200], X[200:250], y[200:250])
        self.assertTrue(m._is_fitted)
        self.assertTrue(supports_validation(m))

    def test_nan_rejected(self):
        X, y = make_data()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self._make().fit(X, y)

    def test_seq_too_long_raises(self):
        X, y = make_data(n=60)
        with self.assertRaises(ValueError):
            self._make(seq_len=100).fit(X, y)

    def test_bad_params_rejected(self):
        with self.assertRaises(ValueError):
            self.model_cls(hidden_size=0)
        with self.assertRaises(ValueError):
            self.model_cls(dropout=1.5)


class TestLSTM(SequenceMixin, unittest.TestCase):
    model_cls = LSTMForecaster


class TestGRU(SequenceMixin, unittest.TestCase):
    model_cls = GRUForecaster


if __name__ == "__main__":
    unittest.main()
