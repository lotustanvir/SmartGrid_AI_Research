"""Tests for Phase 5B research protocol (synthetic data only)."""

import unittest
import warnings

import numpy as np
import pandas as pd

from src.evaluation.alignment import (
    EvaluationAlignment,
    explode_windows,
    make_sequential_keys,
)
from src.models.hybrid import HybridTFTXGBoostForecaster
from src.training.splits import (
    TemporalSplit,
    load_experiment_config,
    split_from_config,
)
from src.utils.experiment import FEATURE_COLS, TARGET_COL, generate_synthetic_grid_data

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


def tiny_arrays(n=400):
    df = generate_synthetic_grid_data(n_samples=n, seed=42)
    return df[FEATURE_COLS].to_numpy(), df[TARGET_COL].to_numpy()


class TestTemporalSplit(unittest.TestCase):
    def test_ratios_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            TemporalSplit(0.8, 0.15, 0.15)

    def test_chronological_arrays(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        X_tr, X_va, X_te, _, _, _ = TemporalSplit().split_arrays(X, y)
        self.assertTrue(X_tr.max() < X_va.min() < X_te.min())
        self.assertEqual(len(X_tr) + len(X_va) + len(X_te), 100)

    def test_frame_split_per_group(self):
        df = pd.DataFrame(
            {
                "group_id": ["a"] * 50 + ["b"] * 50,
                "time_idx": list(range(50)) * 2,
                "v": np.random.RandomState(0).rand(100),
            }
        )
        tr, va, te = TemporalSplit(0.6, 0.2, 0.2).split_frame(df)
        self.assertEqual((len(tr), len(va), len(te)), (60, 20, 20))
        for name, g in df.groupby("group_id"):
            gtr = tr[tr["group_id"] == name]
            gva = va[va["group_id"] == name]
            gte = te[te["group_id"] == name]
            self.assertLess(gtr["time_idx"].max(), gva["time_idx"].min())
            self.assertLess(gva["time_idx"].max(), gte["time_idx"].min())

    def test_config_loads(self):
        cfg = load_experiment_config()
        self.assertIn("splits", cfg)
        self.assertIn("experiment", cfg)
        self.assertIn("oof", cfg)
        split = split_from_config()
        self.assertAlmostEqual(
            split.train_ratio + split.val_ratio + split.test_ratio, 1.0
        )


class TestAlignment(unittest.TestCase):
    def test_identical_timestamps(self):
        al = EvaluationAlignment()
        k1 = make_sequential_keys(10, 70)
        k2 = make_sequential_keys(6, 74)  # shorter history, same tail
        al.add_model("m1", k1, np.arange(10.0), np.arange(10.0), 10, "full")
        al.add_model("m2", k2, np.arange(74.0, 80.0), np.arange(74.0, 80.0), 10, "tail")
        common = al.common_keys()
        self.assertEqual(list(common["time_idx"]), [74, 75, 76, 77, 78, 79])
        rep = al.report()
        self.assertEqual(rep["common_aligned_count"], 6)
        self.assertEqual(rep["models"]["m1"]["dropped_count"], 4)
        self.assertEqual(rep["models"]["m2"]["dropped_count"], 0)
        scores = al.metrics()
        self.assertAlmostEqual(scores["m1"]["R2"], 1.0)

    def test_explode_windows(self):
        keys, yt, pr = explode_windows(
            ["a", "b"],
            np.array([[10, 11], [20, 21]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.5, 2.5], [3.5, 4.5]]),
        )
        self.assertEqual(list(keys["time_idx"]), [10, 11, 20, 21])
        self.assertEqual(list(keys["group_id"]), ["a", "a", "b", "b"])
        np.testing.assert_array_equal(pr, [1.0, 2.0, 3.0, 4.0])

    def test_duplicate_rejected(self):
        al = EvaluationAlignment()
        k = make_sequential_keys(4, 0)
        al.add_model("m", k, np.zeros(4), np.zeros(4), 4, "r")
        with self.assertRaises(ValueError):
            al.add_model("m", k, np.zeros(4), np.zeros(4), 4, "r")

    def test_nan_rejected(self):
        al = EvaluationAlignment()
        k = make_sequential_keys(4, 0)
        bad = np.array([1.0, np.nan, 2.0, 3.0])
        with self.assertRaises(ValueError):
            al.add_model("m", k, np.zeros(4), bad, 4, "r")


class TestOOFProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = tiny_arrays()
        split = TemporalSplit()
        i_tr, i_va = split.boundaries(len(X))
        cls.X_tr, cls.y_tr = X[:i_tr], y[:i_tr]
        cls.test_start = i_va
        cls.model = HybridTFTXGBoostForecaster(
            tft_config=dict(TINY_TFT),
            xgb_config=dict(TINY_XGB),
            n_oof_folds=2,
        ).fit(cls.X_tr, cls.y_tr)

    def test_default_is_oof(self):
        self.assertEqual(HybridTFTXGBoostForecaster().residual_training_mode, "oof")
        self.assertEqual(self.model.residual_training_mode, "oof")

    def test_folds_chronological_no_overlap(self):
        folds = self.model.oof_folds_
        self.assertEqual(len(folds), 2)
        for f in folds:
            self.assertLess(f["train_end_time"], f["predict_start"])
            self.assertLessEqual(f["predict_start"], f["predict_end"])
        self.assertLess(folds[0]["predict_end"], folds[1]["predict_start"])

    def test_residual_count_matches_folds(self):
        total = sum(f["n_samples"] for f in self.model.oof_folds_)
        self.assertEqual(total, self.model.n_residual_samples_)

    def test_residual_learner_never_sees_test(self):
        lo, hi = self.model.residual_period_
        self.assertLess(hi, self.test_start)
        self.assertGreaterEqual(lo, 0)

    def test_insample_warns_loudly(self):
        X, y = tiny_arrays(n=200)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            HybridTFTXGBoostForecaster(
                tft_config=dict(TINY_TFT),
                xgb_config=dict(TINY_XGB),
                residual_training_mode="in_sample",
            ).fit(X[:120], y[:120])
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in rec))

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            HybridTFTXGBoostForecaster(residual_training_mode="kfold")


class TestHorizons(unittest.TestCase):
    def test_one_step_horizon(self):
        X, y = tiny_arrays(n=300)
        m = HybridTFTXGBoostForecaster(
            tft_config={**TINY_TFT, "prediction_length": 1},
            xgb_config=dict(TINY_XGB),
            n_oof_folds=2,
        ).fit(X, y)
        preds, actuals = m.forecast(X, y)
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(preds.shape, actuals.shape)

    def test_24_step_horizon(self):
        X, y = tiny_arrays(n=500)
        cfg = {**TINY_TFT, "encoder_length": 24, "prediction_length": 24}
        m = HybridTFTXGBoostForecaster(
            tft_config=cfg, xgb_config=dict(TINY_XGB), n_oof_folds=2
        ).fit(X, y)
        preds, actuals = m.forecast(X, y)
        self.assertEqual(preds.shape[1], 24)
        out = m.evaluate(X, y)
        self.assertEqual(set(out), {"MAE", "RMSE", "MAPE", "sMAPE", "R2"})


class TestUncertaintyNaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.models.tft import build_synthetic_tft_frame

        cls.df = build_synthetic_tft_frame(n_groups=2, n_steps=120, seed=11)
        cls.model = HybridTFTXGBoostForecaster(
            tft_config=dict(TINY_TFT), xgb_config=dict(TINY_XGB), n_oof_folds=2
        ).fit(cls.df)

    def test_explicit_keys(self):
        out = self.model.predict_interval(self.df)
        self.assertEqual(
            set(out), {"base_tft_lower", "hybrid_median", "base_tft_upper"}
        )

    def test_bounds_consistency_with_report(self):
        out = self.model.predict_interval(self.df)
        diag = self.model.interval_diagnostics(self.df)
        lo = np.ravel(out["base_tft_lower"])
        med = np.ravel(out["hybrid_median"])
        hi = np.ravel(out["base_tft_upper"])
        expect = [int(i) for i in np.where((med < lo) | (med > hi))[0]]
        self.assertEqual(diag["violation_idx"], expect)
        self.assertEqual(diag["n_violations"], len(expect))
        self.assertEqual(diag["n_points"], len(med))

    def test_no_silent_clip_by_default(self):
        out = self.model.predict_interval(self.df)
        np.testing.assert_allclose(
            np.ravel(out["hybrid_median"]),
            np.ravel(self.model.predict(self.df)),
        )


class TestAblationSuite(unittest.TestCase):
    def test_suite_runs_aligned(self):
        from src.training.ablation import ABLATION_VARIANTS, run_ablation

        X, y = tiny_arrays(n=400)
        tiny = {
            "encoder_length": 12,
            "prediction_length": 3,
            "hidden_size": 8,
            "attention_head_size": 2,
            "hidden_continuous_size": 4,
            "max_epochs": 1,
            "batch_size": 16,
        }
        overrides = {
            "xgboost": {"n_estimators": 10},
            "tft": dict(tiny),
            "hybrid": {
                "tft_config": dict(tiny),
                "xgb_config": {"n_estimators": 10},
                "n_oof_folds": 2,
            },
        }
        metrics_df, report, metadata = run_ablation(
            X, y, model_overrides=overrides, output_dir="tmp_ablation"
        )
        self.assertEqual(set(metrics_df["Variant"]), set(ABLATION_VARIANTS))
        self.assertGreater(report["common_aligned_count"], 0)
        for v in metrics_df["Variant"]:
            self.assertEqual(report["models"][v]["aligned_count"], report["common_aligned_count"])
        self.assertFalse(metadata["C_hybrid_insample"]["research_valid"])
        self.assertTrue(metadata["D_hybrid_oof"]["research_valid"])
        self.assertIn("multi-output", metadata["D_hybrid_oof"]["horizon_strategy"])


if __name__ == "__main__":
    unittest.main()
