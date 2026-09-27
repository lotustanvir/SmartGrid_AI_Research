"""Leakage-audit tests (Phase 1 critical fixes).

Proves, with failing-before/passing-after assertions:
1. LSTM/GRU feature scaler is fit on TRAIN windows only (internal and
   explicit validation paths); target scaler likewise.
2. TFT weather covariates are encoder-only (unknown) reals; only calendar
   covariates are known into the future. Validation windows start after
   the training cutoff (no test leakage by construction).
"""

import unittest

import numpy as np

from src.models.deep import GRUForecaster, LSTMForecaster
from src.models.tft.dataset import (
    GROUP_ID,
    TARGET,
    TIME_IDX,
    build_synthetic_tft_frame,
    resolve_roles,
    split_datasets,
)
from src.models.deep.sequence import build_windows
from src.utils.experiment import chronological_split

SHIFT = 50.0
SMALL_KW = {
    "hidden_size": 8,
    "num_layers": 1,
    "seq_len": 12,
    "batch_size": 32,
    "epochs": 2,
    "patience": 10,
}


def shifted_data(n=240, n_feat=3, seed=0):
    """Features with a large distribution shift in the tail (val region)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    X[int(n * 0.7):] += SHIFT
    y = rng.randn(n) + 10.0
    return X, y


class TestSequenceScalerTrainOnly(unittest.TestCase):
    CLASSES = (LSTMForecaster, GRUForecaster)

    def test_feature_scaler_ignores_validation_internal_split(self):
        for cls in self.CLASSES:
            with self.subTest(model=cls.__name__):
                X, y = shifted_data()
                m = cls(**SMALL_KW).fit(X, y)
                Xw, yw = build_windows(X, y, m.seq_len, m.output_size)
                n_val = max(1, int(len(Xw) * m.val_fraction))
                expected = Xw[:-n_val].reshape(-1, X.shape[1]).mean(axis=0)
                full = Xw.reshape(-1, X.shape[1]).mean(axis=0)
                # Scaler must match train-window stats, not full-data stats.
                np.testing.assert_allclose(m.scaler_.mean_, expected, rtol=1e-6)
                self.assertFalse(
                    np.allclose(m.scaler_.mean_, full, rtol=1e-3),
                    f"{cls.__name__}: scaler matches full-data mean (leakage)",
                )

    def test_target_scaler_ignores_validation(self):
        for cls in self.CLASSES:
            with self.subTest(model=cls.__name__):
                X, y = shifted_data()
                m = cls(**SMALL_KW).fit(X, y)
                _, yw = build_windows(X, y, m.seq_len, m.output_size)
                n_val = max(1, int(len(yw) * m.val_fraction))
                expected = yw[:-n_val].mean()
                np.testing.assert_allclose(
                    m.target_scaler_.mean_, [expected], rtol=1e-6
                )

    def test_explicit_validation_path_unchanged(self):
        for cls in self.CLASSES:
            with self.subTest(model=cls.__name__):
                X, y = shifted_data()
                X_tr, X_va, _, y_tr, y_va, _ = chronological_split(X, y, 0.7, 0.15)
                m = cls(**SMALL_KW).fit(X_tr, y_tr, X_va, y_va)
                Xw_tr, _ = build_windows(X_tr, y_tr, m.seq_len, m.output_size)
                expected = Xw_tr.reshape(-1, X.shape[1]).mean(axis=0)
                np.testing.assert_allclose(m.scaler_.mean_, expected, rtol=1e-6)


class TestTFTRolesNoFutureWeather(unittest.TestCase):
    def test_weather_is_unknown_calendar_is_known(self):
        df = build_synthetic_tft_frame(n_groups=2, n_steps=120, seed=7)
        known, unknown, _ = resolve_roles(df)
        self.assertEqual(known, ["time_idx", "hour", "day_of_week"])
        self.assertIn(TARGET, unknown)
        for col in ("temperature", "solar_generation", "wind_generation"):
            self.assertIn(col, unknown, f"{col} must stay a model input")
            self.assertNotIn(col, known, f"{col} must not be decoder-known")

    def test_split_datasets_encoder_only_weather(self):
        df = build_synthetic_tft_frame(n_groups=2, n_steps=120, seed=7)
        parts = split_datasets(
            df, encoder_length=12, prediction_length=3,
            val_size=12, batch_size=16,
        )
        training = parts["training"]
        known_reals = list(training.time_varying_known_reals)
        unknown_reals = list(training.time_varying_unknown_reals)
        for col in ("temperature", "solar_generation", "wind_generation"):
            self.assertIn(col, unknown_reals)
            self.assertNotIn(col, known_reals)
        self.assertIn(TARGET, unknown_reals)
        self.assertGreater(len(parts["validation"]), 0)

    def test_validation_after_training_cutoff(self):
        import torch

        df = build_synthetic_tft_frame(n_groups=2, n_steps=120, seed=7)
        max_time = int(df[TIME_IDX].max())
        val_size = 12
        cutoff = max_time - val_size
        parts = split_datasets(
            df, encoder_length=12, prediction_length=3,
            val_size=val_size, batch_size=256,
        )
        # Every validation prediction target must lie strictly after the
        # training cutoff (no test/val target leaks into training).
        mins = []
        with torch.no_grad():
            for batch in parts["val_loader"]:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                mins.append(int(np.asarray(x["decoder_time_idx"].cpu()).min()))
        self.assertTrue(mins, "validation loader produced no batches")
        self.assertGreater(min(mins), cutoff)
        self.assertEqual(df[GROUP_ID].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
