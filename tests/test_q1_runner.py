"""Locked Q1 runner tests (Phase 3).

Synthetic/in-memory or tiny fixtures ONLY. No real-data model training,
no tuning, no headline results, no ranking. The single real-data test
runs SAFE preflight (metadata/features/origins only, temp outputs).
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features_q1 import (
    build_q1_features,
    canonical_feature_names,
    drop_warmup,
)
from src.data.origins import generate_origins, origins_hash
from src.q1 import runner as R
from src.q1.protocol import HORIZONS, Q1_MODELS, Q1_SEEDS


def _synth_v2(n=600, seed=0):
    from tests.test_q1_protocol import _synth_v2 as _base

    return _base(n, seed)


class TestManifest(unittest.TestCase):
    def test_exact_11_protocol_order(self):
        specs = R.build_manifest()
        models = [s.model for s in specs]
        self.assertEqual(sorted(set(models)), sorted(Q1_MODELS))
        # Protocol order, model-major: persistence block first (H1/H6/H24).
        self.assertEqual(models[:3], ["persistence"] * 3)
        self.assertEqual([s.horizon for s in specs[:3]], [1, 6, 24])
        self.assertEqual(
            models,
            [m for m in Q1_MODELS for _ in range(3 if m in R.DETERMINISTIC_MODELS else 9)],
        )

    def test_full_size_81(self):
        specs = R.build_manifest()
        self.assertEqual(len(specs), R.FULL_MANIFEST_SIZE)
        self.assertEqual(R.FULL_MANIFEST_SIZE, 81)
        R.validate_manifest_size(specs, expect_full=True)
        self.assertEqual(len({s.exp_id for s in specs}), 81)

    def test_seed_multiplicity(self):
        specs = R.build_manifest()
        det = [s for s in specs if s.model in R.DETERMINISTIC_MODELS]
        sto = [s for s in specs if s.model not in R.DETERMINISTIC_MODELS]
        self.assertEqual(len(det), 9)
        self.assertEqual(len(sto), 72)
        self.assertTrue(all(s.seed is None for s in det))
        self.assertEqual({s.seed for s in sto}, set(Q1_SEEDS))
        for h in HORIZONS:
            per_h = [s for s in sto if s.horizon == h]
            self.assertEqual(len(per_h), 24)  # 8 models x 3 seeds

    def test_exp_id_deterministic(self):
        self.assertEqual(R.exp_id_for("tft", 24, 42), "Q1_tft_H24_S42")
        self.assertEqual(R.exp_id_for("persistence", 1, None), "Q1_persistence_H1_Sdet")
        twice = [s.exp_id for s in R.build_manifest()]
        self.assertEqual(twice, [s.exp_id for s in R.build_manifest()])

    def test_legacy_rejected(self):
        with self.assertRaises(ValueError):
            R.build_manifest(models=["xgboost", "linear_regression"])
        with self.assertRaises(ValueError):
            R.build_manifest(models=["nope"])
        with self.assertRaises(ValueError):
            R.build_manifest(models=["tft", "tft"])
        with self.assertRaises(ValueError):
            R.build_manifest(models=[])

    def test_horizon_seed_validation(self):
        with self.assertRaises(ValueError):
            R.build_manifest(horizons=[1, 2])
        with self.assertRaises(ValueError):
            R.build_manifest(horizons=[1, 1])
        with self.assertRaises(ValueError):
            R.build_manifest(seeds=[42, 7])
        with self.assertRaises(ValueError):
            R.build_manifest(seeds=[42, 42])
        partial = R.build_manifest(models=["tft"], horizons=[1], seeds=[42])
        self.assertEqual(len(partial), 1)
        with self.assertRaises(ValueError):
            R.validate_manifest_size(partial, expect_full=True)
        R.validate_manifest_size(partial, expect_full=False)


class TestGates(unittest.TestCase):
    def test_feature_enforcement(self):
        feat, names = build_q1_features(_synth_v2())
        clean, _ = drop_warmup(feat, names)
        R.assert_canonical_frame(clean, names)
        # Exact 49-column extraction in locked order for model input.
        X = R.canonical_matrix(clean, names)
        self.assertEqual(X.shape[1], 49)
        self.assertEqual(list(clean[names].columns), list(canonical_feature_names()))
        tampered = clean.drop(columns=[names[0]])
        with self.assertRaises(ValueError):
            R.assert_canonical_frame(tampered, names)
        with self.assertRaises(ValueError):
            R.canonical_matrix(tampered, names)
        with self.assertRaises(ValueError):
            R.assert_canonical_frame(clean, names[::-1])

    def test_dataset_sha_enforcement(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "other.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                R.load_and_validate_dataset(str(p))

    def test_origin_hash_enforcement(self):
        feat, names = build_q1_features(_synth_v2())
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=tuple(HORIZONS))
        h0 = origins_hash(org)
        tampered = org.copy()
        tampered.loc[tampered.index[0], "origin_timestamp"] = pd.Timestamp("2030-01-01")
        self.assertNotEqual(origins_hash(tampered), h0)

    def test_missing_artifact_no_generate(self):
        import tempfile

        feat, names = build_q1_features(_synth_v2())
        clean, _ = drop_warmup(feat, names)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                R.ensure_origins(clean, str(Path(d) / "missing.parquet"), allow_generate=False)

    def test_test_exclusion(self):
        R.assert_no_test_overlap(pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), label="t")
        with self.assertRaises(ValueError):
            R.assert_no_test_overlap(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), label="t")
        with self.assertRaises(ValueError):
            R.assert_no_test_overlap(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-01-01"), label="t")

    def test_tft_fit_frame_gate(self):
        from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX
        from src.q1.tft_path import assert_fit_frame

        base = pd.DataFrame(
            {
                TIME_IDX: np.arange(300),
                GROUP_ID: "single",
                TARGET: np.linspace(100, 120, 300),
                "hour": np.tile(np.arange(24), 13)[:300],
                "day_of_week": np.tile(np.arange(7), 43)[:300],
            }
        )
        assert_fit_frame(base.iloc[:200], base.iloc[200:250], test_time_min=250)
        with self.assertRaises(ValueError):
            assert_fit_frame(base.iloc[200:250], base.iloc[:200])
        with self.assertRaises(ValueError):
            assert_fit_frame(base.iloc[:200], base.iloc[200:250], test_time_min=240)

    def test_hybrid_defaults(self):
        from src.models.hybrid import HybridTFTXGBoostForecaster

        m = HybridTFTXGBoostForecaster()
        self.assertEqual(m.n_oof_folds, 5)
        self.assertEqual(m.residual_training_mode, "oof")

    def test_seed_rules(self):
        R.apply_spec_seed(R.Q1ExperimentSpec(model="sarima", horizon=6, seed=None))
        R.apply_spec_seed(R.Q1ExperimentSpec(model="gru", horizon=1, seed=2025))
        with self.assertRaises(ValueError):
            R.apply_spec_seed(R.Q1ExperimentSpec(model="gru", horizon=1, seed=7))
        with self.assertRaises(ValueError):
            R.apply_spec_seed(R.Q1ExperimentSpec(model="persistence", horizon=1, seed=42))


class TestAlignmentScoring(unittest.TestCase):
    def _setup(self):
        feat, names = build_q1_features(_synth_v2())
        clean, _ = drop_warmup(feat, names)
        org = generate_origins(clean, horizons=(1,))
        return clean, names, org

    def test_validate_ok_and_rejections(self):
        clean, _, org = self._setup()
        series = clean["target"].to_numpy(dtype=float)
        ts = clean["timestamp"].to_numpy()
        from src.q1.evaluate import build_naive_preds_by_origin as _b

        preds = _b("persistence", series, ts, org)
        rows = [
            {
                "exp_id": "Q1_persistence_H1_Sdet",
                "model": "persistence",
                "horizon": 1,
                "seed": "det",
                "origin_timestamp_iso": k[0],
                "target_timestamp": pd.Timestamp(k[0]) + pd.Timedelta(hours=1),
                "y_pred": v,
            }
            for k, v in preds.items()
        ]
        R.validate_pred_rows(org, rows)
        for mutate in ("missing", "duplicate", "wrong_horizon", "outside"):
            bad = [dict(r) for r in rows]
            if mutate == "missing":
                bad = bad[1:]
            elif mutate == "duplicate":
                bad = bad + [dict(bad[0])]
            elif mutate == "wrong_horizon":
                bad[0] = dict(bad[0], horizon=6)
            else:
                bad[0] = dict(bad[0], origin_timestamp_iso="2030-01-01T00:00:00")
            with self.assertRaises(ValueError, msg=mutate):
                R.validate_pred_rows(org, bad)

    def test_mase_wiring_train_only(self):
        from src.q1.evaluate import score_at_origins

        clean, _, _ = self._setup()
        y = clean["target"].to_numpy(dtype=float)
        ts = clean["timestamp"]
        origins = pd.DataFrame(
            {"origin_timestamp": [ts.iloc[200], ts.iloc[210]], "horizon": [1, 1]}
        )
        preds = {
            (pd.Timestamp(ts.iloc[200]).isoformat(), 1): float(y[201]),
            (pd.Timestamp(ts.iloc[210]).isoformat(), 1): float(y[211]),
        }
        m, _ = score_at_origins(origins, clean, preds, y_train=y[:180], seasonality=168)
        self.assertEqual(
            set(m), {"MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE"}
        )
        with self.assertRaises(ValueError):
            score_at_origins(origins, clean, preds, y_train=None)
        with self.assertRaises(ValueError):
            score_at_origins(origins, clean, preds, y_train=y[:180], seasonality=24)


class TestRegistryGate(unittest.TestCase):
    def _rec(self, **over):
        rec = {
            "exp_id": "Q1_xgboost_H6_S123",
            "status": "COMPLETED",
            "git_commit": "abc1234",
            "dataset_version": "v2",
            "dataset_sha256": "f" * 64,
            "features": "Q1_V2_LAG1_001",
            "feature_schema_hash": "e" * 64,
            "origin_set_hash": "5" * 64,
            "model": "xgboost",
            "horizon": 6,
            "seed": 123,
            "config_snapshot": "c",
            "predictions": "p",
            "metrics": "{}",
            "checkpoint": "c",
            "smoke": "False",
        }
        rec.update(over)
        return rec

    def test_complete_and_deterministic(self):
        from src.utils.registry import require_q1_complete

        require_q1_complete(self._rec())
        det = self._rec(model="persistence", horizon=1, seed="det", checkpoint="")
        require_q1_complete(det)
        with self.assertRaises(ValueError):
            require_q1_complete(self._rec(model="persistence", horizon=1, seed=42, checkpoint=""))
        with self.assertRaises(ValueError):
            require_q1_complete(self._rec(seed=7))
        with self.assertRaises(ValueError):
            require_q1_complete(self._rec(smoke="True"))
        with self.assertRaises(ValueError):
            require_q1_complete(self._rec(checkpoint=""))
        with self.assertRaises(ValueError):
            require_q1_complete(self._rec(model="linear_regression"))

    def test_validate_run_set(self):
        specs = R.build_manifest(models=["persistence"], horizons=[1], seeds=[42])
        recs = [
            {
                **self._rec(exp_id=s.exp_id, model=s.model, horizon=s.horizon, seed="det", checkpoint=""),
            }
            for s in specs
        ]
        R.validate_run_set(recs, specs)
        with self.assertRaises(ValueError):
            R.validate_run_set(recs[1:], specs)

    def test_stale_artifact_check(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            stale = Path(d) / "REG.csv"
            with open(stale, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["predictions"])
                w.writeheader()
                w.writerow({"predictions": str(Path(d) / "missing.parquet")})
            with self.assertRaises(ValueError):
                R._check_stale_artifact(stale)
            fixed = Path(d) / "REG2.csv"
            with open(fixed, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["predictions"])
                w.writeheader()
                w.writerow({"predictions": str(stale)})
            R._check_stale_artifact(fixed)


class TestBuildCombinedTftFrame(unittest.TestCase):
    def _make_data(self, n_per_split=400):
        synth = _synth_v2(n=n_per_split * 3 + 200)
        feat, names = build_q1_features(synth)
        clean, _ = drop_warmup(feat, names)
        tr = clean.iloc[:n_per_split].reset_index(drop=True)
        va = clean.iloc[n_per_split : n_per_split * 2].reset_index(drop=True)
        te = clean.iloc[n_per_split * 2 :].reset_index(drop=True)
        full = pd.concat([tr, va, te], ignore_index=True)
        return R.Q1Data(
            raw=synth,
            frame=clean,
            feature_names=names,
            tr=tr,
            va=va,
            te=te,
            full=full,
            origins=pd.DataFrame(),
            origin_artifact="test.csv",
            y_train=tr["target"].to_numpy(dtype=float),
            provenance={},
        )

    def test_accepts_canonical_calendar(self):
        data = self._make_data()
        tft, train_tft, val_tft, test_min = R.build_combined_tft_frame(data)
        self.assertIn("dow", tft.columns)
        self.assertNotIn("day_of_week", tft.columns)
        self.assertEqual(len(tft), len(data.full))
        self.assertIn("time_idx", tft.columns)
        self.assertIn("electricity_demand", tft.columns)

    def test_rejects_non_canonical_column(self):
        data = self._make_data()
        data.tr = data.tr.copy()
        data.va = data.va.copy()
        data.te = data.te.copy()
        data.full = data.full.copy()
        for df in (data.tr, data.va, data.te, data.full):
            df["day_of_week"] = 0
        with self.assertRaises(ValueError) as ctx:
            R.build_combined_tft_frame(data)
        self.assertIn("day_of_week", str(ctx.exception))


class TestPreflightRealData(unittest.TestCase):
    def test_preflight_passes_temp_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "preflight.json"
            report = R.preflight(output_path=out)
            self.assertEqual(report["preflight"], "PASS")
            self.assertTrue(out.is_file())
            self.assertIn("NOT headline results", report["note"])
            self.assertEqual(report["manifest_size"], 81)


if __name__ == "__main__":
    unittest.main()
