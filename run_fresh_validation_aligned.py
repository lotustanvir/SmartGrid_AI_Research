"""Phase 6 Aligned Validation: Leakage-free comparable evaluation.
Runs LSTM, GRU, TFT, Hybrid on PJM 2020. Same seed=42, 70/15/15 split.
No test leakage. One-step-ahead via predict=False windows. ~1282 aligned.
Results: results/fresh_validation_aligned/
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("results/fresh_validation_aligned_post_fix")
SEED = 42
TIME_IDX = "time_idx"
GROUP_ID = "group_id"
TARGET = "electricity_demand"
CANONICAL_TARGET = "target"


def load_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    logger.info("Loaded train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def get_model_features(train_df):
    exo = [c for c in ("solar", "wind") if c in train_df.columns]
    feature_names = [
        "hour", "day_of_week", "day_of_month", "month", "week_of_year",
        "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "lag_1", "lag_24", "lag_168",
        "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_std_168",
    ]
    return exo + feature_names


def build_tft_dataframes(train_df, val_df, test_df):
    from src.models.tft.dataset import validate_frame

    def make_frame(df):
        out = df.copy()
        out[TARGET] = out[CANONICAL_TARGET].astype(float)
        out[TIME_IDX] = 0
        out[GROUP_ID] = "single"
        known = [c for c in ("hour", "day_of_week", "temperature", "solar", "wind")
                 if c in out.columns]
        ordered = ([TIME_IDX, GROUP_ID, TARGET] + known +
                   [c for c in out.columns if c not in [TIME_IDX, GROUP_ID, TARGET] + known])
        out = out[ordered]
        feature_cols = [c for c in ordered if c not in [TIME_IDX, GROUP_ID, TARGET]]
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        out[TIME_IDX] = np.arange(len(out))
        out[GROUP_ID] = "single"
        return validate_frame(out)

    return make_frame(train_df), make_frame(val_df), make_frame(test_df)


def integrity_check(model_name, y_true, y_pred, timestamps, train_rows, val_rows, test_rows):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ts = np.asarray(timestamps)
    n = len(yt)
    sse = float(np.sum((yt - yp) ** 2))
    yt_mean = float(np.mean(yt))
    sst = float(np.sum((yt - yt_mean) ** 2))
    r2_calc = 1.0 - sse / sst if sst > 0 else 0.0
    return {
        "model": model_name, "train_rows": train_rows, "val_rows": val_rows,
        "test_rows": test_rows, "prediction_count": n,
        "first_timestamp": str(ts[0]) if n > 0 else None,
        "last_timestamp": str(ts[-1]) if n > 0 else None,
        "nan": bool(np.isnan(yt).any() or np.isnan(yp).any()),
        "inf": bool(np.isinf(yt).any() or np.isinf(yp).any()),
        "const_prediction": bool(np.std(yp) < 1e-12) if n > 1 else False,
        "duplicate_timestamps": bool(len(ts) != len(np.unique(ts))),
        "unique_timestamps": len(np.unique(ts)),
        "chronological_order": bool(np.all(ts[:-1] <= ts[1:])) if n > 1 else True,
        "r2_calculated": r2_calc,
    }


def run_lstm_gru(train_df, val_df, test_df, model_features):
    from src.data.adapters import tabular_ready
    from src.training.experiment_runner import ExperimentRunner

    X_tr, y_tr, _, _ = tabular_ready(train_df, model_features)
    X_va, y_va, _, _ = tabular_ready(val_df, model_features)
    X_te, y_te, _, _ = tabular_ready(test_df, model_features)

    runner = ExperimentRunner(
        output_dir=str(OUTPUT_DIR / "runner_tmp"),
        train_ratio=0.70, val_ratio=0.15,
        use_validation=True, use_early_stopping=False, seed=SEED,
    )

    results = {}
    predictions_arrays = {}
    for name in ["lstm", "gru"]:
        t0 = time.time()
        row, y_aligned, preds = runner.run_one(
            name,
            X_tr=X_tr, y_tr=y_tr,
            X_val=X_va, y_val=y_va,
            X_te=X_te, y_te=y_te,
        )
        runtime = time.time() - t0

        r2_val = row["R2"]
        sse = float(np.sum((y_aligned - preds) ** 2))
        yt_mean = float(np.mean(y_aligned))
        sst = float(np.sum((y_aligned - yt_mean) ** 2))
        r2_calc = 1.0 - sse / sst if sst > 0 else 0.0

        n_pred = len(preds)
        has_nan = bool(np.isnan(preds).any())
        has_inf = bool(np.isinf(preds).any())
        is_const = bool(np.std(preds) < 1e-12) if n_pred > 1 else False

        results[name] = {
            "MAE": row["MAE"], "RMSE": row["RMSE"], "sMAPE": row["sMAPE"],
            "R2": r2_val, "runtime": round(runtime, 2), "n_test": n_pred,
            "r2_identity_match": abs(r2_val - r2_calc) < 1e-6,
            "nan": has_nan, "inf": has_inf, "const": is_const,
            "train_rows": int(len(X_tr)), "val_rows": int(len(X_va)),
            "test_rows": int(len(X_te)),
            "pred_mean": float(np.mean(preds)),
            "pred_min": float(np.min(preds)),
            "pred_max": float(np.max(preds)),
        }
        predictions_arrays[name] = (y_aligned, preds)
        logger.info("%s: MAE=%.4f R2=%.4f n_test=%d (%.1fs)",
                     name.upper(), row["MAE"], r2_val, n_pred, runtime)

    return results, predictions_arrays


def run_tft(train_df, val_df, test_df):
    import torch
    from src.evaluation.metrics import evaluate_regression
    from src.models.baseline import load_params
    from src.models.tft import TemporalFusionTransformerForecaster
    from src.models.tft.dataset import validate_frame
    from pytorch_forecasting import TimeSeriesDataSet

    t0 = time.time()

    tft_train_df, tft_val_df, tft_test_df = build_tft_dataframes(train_df, val_df, test_df)

    tft_trainval_df = pd.concat([tft_train_df, tft_val_df], ignore_index=True)
    tft_trainval_df[TIME_IDX] = np.arange(len(tft_trainval_df))
    tft_trainval_df = tft_trainval_df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    tft_trainval_df = validate_frame(tft_trainval_df)

    logger.info("TFT training on train+val: %d rows (NO test leakage)", len(tft_trainval_df))
    tft_params = load_params("tft")
    tft_model = TemporalFusionTransformerForecaster(**tft_params)
    tft_model.fit(tft_trainval_df)

    full_df = pd.concat([tft_train_df, tft_val_df, tft_test_df], ignore_index=True)
    full_df[TIME_IDX] = np.arange(len(full_df))
    full_df = full_df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    full_df = validate_frame(full_df)

    test_min_ti = int(tft_test_df[TIME_IDX].min())
    test_max_ti = int(tft_test_df[TIME_IDX].max())
    logger.info("TFT test time_idx range: %d - %d", test_min_ti, test_max_ti)

    all_preds = []
    all_actuals = []
    all_times = []

    assert tft_model._training is not None
    for name, g in full_df.groupby(GROUP_ID, sort=True):
        g = g.sort_values(TIME_IDX).reset_index(drop=True)
        ds = TimeSeriesDataSet.from_dataset(
            tft_model._training, g, predict=False, stop_randomization=True
        )
        loader = ds.to_dataloader(
            train=False, batch_size=tft_model.batch_size, num_workers=0
        )
        med = tft_model._median_for_loader(loader).cpu().numpy()

        times_per_window = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                times_per_window.append(np.asarray(x["decoder_time_idx"].cpu()))
        times = np.concatenate(times_per_window, axis=0)

        target_by_time = dict(zip(g[TIME_IDX].tolist(), g[TARGET].tolist()))

        for i in range(times.shape[0]):
            t = int(times[i, 0])
            if t < test_min_ti:
                continue
            all_preds.append(float(med[i, 0]))
            all_actuals.append(target_by_time[t])
            all_times.append(t)

    preds_arr = np.asarray(all_preds, dtype=float)
    actuals_arr = np.asarray(all_actuals, dtype=float)
    times_arr = np.asarray(all_times)

    metrics = evaluate_regression(actuals_arr, preds_arr)
    runtime = time.time() - t0

    result = {
        "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
        "sMAPE": metrics["sMAPE"], "R2": metrics["R2"],
        "runtime": round(runtime, 2), "n_test": len(preds_arr),
        "pred_mean": float(np.mean(preds_arr)),
        "pred_min": float(np.min(preds_arr)),
        "pred_max": float(np.max(preds_arr)),
        "first_aligned_timestamp": str(times_arr[0]) if len(times_arr) > 0 else None,
        "last_aligned_timestamp": str(times_arr[-1]) if len(times_arr) > 0 else None,
        "train_rows": len(tft_train_df), "val_rows": len(tft_val_df),
        "test_rows": len(tft_test_df),
    }

    logger.info("TFT: MAE=%.4f R2=%.4f n_test=%d (%.1fs)",
                metrics["MAE"], metrics["R2"], len(preds_arr), runtime)

    return result, (actuals_arr, preds_arr, times_arr)


def run_hybrid(train_df, val_df, test_df):
    import torch
    from src.evaluation.metrics import evaluate_regression
    from src.models.baseline import load_params
    from src.models.hybrid import HybridTFTXGBoostForecaster
    from src.models.tft.dataset import validate_frame
    from pytorch_forecasting import TimeSeriesDataSet

    t0 = time.time()

    tft_train_df, tft_val_df, tft_test_df = build_tft_dataframes(train_df, val_df, test_df)

    tft_trainval_df = pd.concat([tft_train_df, tft_val_df], ignore_index=True)
    tft_trainval_df[TIME_IDX] = np.arange(len(tft_trainval_df))
    tft_trainval_df = tft_trainval_df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    tft_trainval_df = validate_frame(tft_trainval_df)

    logger.info("Hybrid training on train+val: %d rows (NO test leakage)", len(tft_trainval_df))
    hyb_params = load_params("hybrid_tft_xgb")
    hyb_model = HybridTFTXGBoostForecaster(**hyb_params)
    hyb_model.fit(tft_trainval_df)

    full_df = pd.concat([tft_train_df, tft_val_df, tft_test_df], ignore_index=True)
    full_df[TIME_IDX] = np.arange(len(full_df))
    full_df = full_df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    full_df = validate_frame(full_df)

    test_min_ti = int(tft_test_df[TIME_IDX].min())
    test_max_ti = int(tft_test_df[TIME_IDX].max())
    logger.info("Hybrid test time_idx range: %d - %d", test_min_ti, test_max_ti)

    all_preds = []
    all_actuals = []
    all_times = []

    assert hyb_model._tft._training is not None
    for name_g, g in full_df.groupby(GROUP_ID, sort=True):
        g = g.sort_values(TIME_IDX).reset_index(drop=True)
        ds = TimeSeriesDataSet.from_dataset(
            hyb_model._tft._training, g, predict=False, stop_randomization=True
        )
        loader = ds.to_dataloader(
            train=False, batch_size=hyb_model._tft.batch_size, num_workers=0
        )
        med = hyb_model._tft._median_for_loader(loader).cpu().numpy()

        times_per_window = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                times_per_window.append(np.asarray(x["decoder_time_idx"].cpu()))
        times = np.concatenate(times_per_window, axis=0)

        target_by_time = dict(zip(g[TIME_IDX].tolist(), g[TARGET].tolist()))
        known_cols = [c for c in hyb_model.residual_feature_names_[:-1-3] if c in g.columns]
        E = hyb_model._tft.encoder_length
        enc_all = g[TARGET].to_numpy(dtype=float)
        time_to_pos = {t: i for i, t in enumerate(g[TIME_IDX].tolist())}

        for i in range(times.shape[0]):
            t0_step = int(times[i, 0])
            if t0_step < test_min_ti:
                continue
            dec_start = time_to_pos[t0_step]
            enc_hist = enc_all[max(0, dec_start - E):dec_start]
            mean_enc = float(np.mean(enc_hist)) if len(enc_hist) > 0 else 0.0
            std_enc = float(np.std(enc_hist)) if len(enc_hist) > 0 else 0.0
            last_enc = float(enc_hist[-1]) if len(enc_hist) > 0 else 0.0
            h = 0
            t_val = int(times[i, h])
            row_feats = [float(g.loc[time_to_pos[t_val], c]) for c in known_cols]
            row_feats += [float(h), mean_enc, std_enc, last_enc]
            xgb_resid = float(hyb_model._xgb.predict(np.asarray([row_feats]))[0])
            all_preds.append(float(med[i, h]) + xgb_resid)
            all_actuals.append(target_by_time[t_val])
            all_times.append(t_val)

    preds_arr = np.asarray(all_preds, dtype=float)
    actuals_arr = np.asarray(all_actuals, dtype=float)
    times_arr = np.asarray(all_times)

    metrics = evaluate_regression(actuals_arr, preds_arr)
    runtime = time.time() - t0

    result = {
        "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
        "sMAPE": metrics["sMAPE"], "R2": metrics["R2"],
        "runtime": round(runtime, 2), "n_test": len(preds_arr),
        "pred_mean": float(np.mean(preds_arr)),
        "pred_min": float(np.min(preds_arr)),
        "pred_max": float(np.max(preds_arr)),
        "first_aligned_timestamp": str(times_arr[0]) if len(times_arr) > 0 else None,
        "last_aligned_timestamp": str(times_arr[-1]) if len(times_arr) > 0 else None,
        "train_rows": len(tft_train_df), "val_rows": len(tft_val_df),
        "test_rows": len(tft_test_df),
        "n_oof_folds": hyb_model.n_oof_folds,
        "oof_folds": hyb_model.oof_folds_,
    }

    logger.info("Hybrid: MAE=%.4f R2=%.4f n_test=%d (%.1fs)",
                metrics["MAE"], metrics["R2"], len(preds_arr), runtime)

    return result, (actuals_arr, preds_arr, times_arr)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("PHASE 6 ALIGNED VALIDATION")
    logger.info("=" * 60)

    train_df, val_df, test_df = load_data()
    model_features = get_model_features(train_df)
    logger.info("Model features (%d): %s", len(model_features), model_features)

    logger.info("=" * 40)
    logger.info("Running LSTM and GRU")
    logger.info("=" * 40)
    lstm_gru_results, lstm_gru_preds = run_lstm_gru(train_df, val_df, test_df, model_features)

    logger.info("=" * 40)
    logger.info("Running TFT (train+val only, predict=False windows)")
    logger.info("=" * 40)
    tft_result, tft_data = run_tft(train_df, val_df, test_df)

    logger.info("=" * 40)
    logger.info("Running Hybrid (train+val only, predict=False windows)")
    logger.info("=" * 40)
    hyb_result, hyb_data = run_hybrid(train_df, val_df, test_df)

    integrity_results = []

    for name in ["lstm", "gru"]:
        y_true, preds = lstm_gru_preds[name]
        r = lstm_gru_results[name]
        ts = np.arange(len(y_true))
        ic = integrity_check(name, y_true, preds, ts,
                             r["train_rows"], r["val_rows"], r["test_rows"])
        ic["r2_reported"] = r["R2"]
        ic["r2_match"] = abs(ic["r2_reported"] - ic["r2_calculated"]) < 1e-6
        integrity_results.append(ic)
        logger.info("%s integrity: n=%d nan=%s inf=%s const=%s r2_match=%s",
                     name, ic["prediction_count"], ic["nan"], ic["inf"],
                     ic["const_prediction"], ic["r2_match"])

    y_true_t, preds_t, ts_t = tft_data
    ic_t = integrity_check("tft", y_true_t, preds_t, ts_t,
                           tft_result["train_rows"], tft_result["val_rows"], tft_result["test_rows"])
    ic_t["r2_reported"] = tft_result["R2"]
    ic_t["r2_match"] = abs(ic_t["r2_reported"] - ic_t["r2_calculated"]) < 1e-6
    integrity_results.append(ic_t)
    logger.info("TFT integrity: n=%d nan=%s inf=%s const=%s r2_match=%s",
                 ic_t["prediction_count"], ic_t["nan"], ic_t["inf"],
                 ic_t["const_prediction"], ic_t["r2_match"])

    y_true_h, preds_h, ts_h = hyb_data
    ic_h = integrity_check("hybrid", y_true_h, preds_h, ts_h,
                           hyb_result["train_rows"], hyb_result["val_rows"], hyb_result["test_rows"])
    ic_h["r2_reported"] = hyb_result["R2"]
    ic_h["r2_match"] = abs(ic_h["r2_reported"] - ic_h["r2_calculated"]) < 1e-6
    integrity_results.append(ic_h)
    logger.info("Hybrid integrity: n=%d nan=%s inf=%s const=%s r2_match=%s",
                 ic_h["prediction_count"], ic_h["nan"], ic_h["inf"],
                 ic_h["const_prediction"], ic_h["r2_match"])

    logger.info("=" * 40)
    logger.info("LEAKAGE VERIFICATION")
    logger.info("=" * 40)
    train_ts_max = train_df["timestamp"].max()
    val_ts_min = val_df["timestamp"].min()
    val_ts_max = val_df["timestamp"].max()
    test_ts_min = test_df["timestamp"].min()
    logger.info("Train max: %s", train_ts_max)
    logger.info("Val min: %s", val_ts_min)
    logger.info("Val max: %s", val_ts_max)
    logger.info("Test min: %s", test_ts_min)
    assert str(train_ts_max) < str(val_ts_min), "LEAKAGE: train >= val"
    assert str(val_ts_max) < str(test_ts_min), "LEAKAGE: val >= test"
    logger.info("No temporal overlap between train/val/test: PASS")

    save_json = lambda obj, name: json.dump(obj, open(OUTPUT_DIR / name, "w"), indent=2, default=str)

    save_json({"lstm": lstm_gru_results["lstm"], "gru": lstm_gru_results["gru"]}, "lstm_gru.json")
    save_json({"tft": tft_result}, "tft.json")
    save_json({"hybrid": hyb_result}, "hybrid.json")
    save_json(integrity_results, "integrity.json")

    summary = {
        "phase": "6 aligned validation",
        "seed": SEED,
        "split": "70/15/15 chronological",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "leakage_check": "PASS",
        "lstm_gru_results": lstm_gru_results,
        "tft_result": tft_result,
        "hybrid_result": hyb_result,
        "n_models": 4,
        "ablation_run": False,
        "multiyear_run": False,
    }
    save_json(summary, "summary.json")

    print("\n" + "=" * 60)
    print("PHASE 6 ALIGNED VALIDATION SUMMARY")
    print("=" * 60)
    for name in ["lstm", "gru"]:
        r = lstm_gru_results[name]
        print(f"  {name.upper()}: MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} n={r['n_test']}")
    r = tft_result
    print(f"  TFT:    MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} n={r['n_test']}")
    r = hyb_result
    print(f"  Hybrid: MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} n={r['n_test']}")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("2020-2025 multi-year validation: NOT started")
    print("Ablation: NOT started")
    print("=" * 60)


if __name__ == "__main__":
    main()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("PHASE 6 ALIGNED VALIDATION")
    logger.info("=" * 60)

    train_df, val_df, test_df = load_data()
    model_features = get_model_features(train_df)
    logger.info("Model features (%d): %s", len(model_features), model_features)

    logger.info("=" * 40)
    logger.info("Running LSTM and GRU")
    logger.info("=" * 40)
    lstm_gru_results, lstm_gru_preds = run_lstm_gru(train_df, val_df, test_df, model_features)

    logger.info("=" * 40)
    logger.info("Running TFT (train+val only, predict=False windows)")
    logger.info("=" * 40)
    tft_result, tft_data = run_tft(train_df, val_df, test_df)

    logger.info("=" * 40)
    logger.info("Running Hybrid (train+val only, predict=False windows)")
    logger.info("=" * 40)
    hyb_result, hyb_data = run_hybrid(train_df, val_df, test_df)

    integrity_results = []

    for name in ["lstm", "gru"]:
        y_true, preds = lstm_gru_preds[name]
        r = lstm_gru_results[name]
        ts = np.arange(len(y_true))
        ic = integrity_check(name, y_true, preds, ts,
                             r["train_rows"], r["val_rows"], r["test_rows"])
        ic["r2_reported"] = r["R2"]
        ic["r2_match"] = abs(ic["r2_reported"] - ic["r2_calculated"]) < 1e-6
        integrity_results.append(ic)
        logger.info("%s integrity: n=%d nan=%s inf=%s const=%s r2_match=%s",
                     name, ic["prediction_count"], ic["nan"], ic["inf"],
                     ic["const_prediction"], ic["r2_match"])

    y_true_t, preds_t, ts_t = tft_data
    ic_t = integrity_check("tft", y_true_t, preds_t, ts_t,
                           tft_result["train_rows"], tft_result["val_rows"], tft_result["test_rows"])
    ic_t["r2_reported"] = tft_result["R2"]
    ic_t["r2_match"] = abs(ic_t["r2_reported"] - ic_t["r2_calculated"]) < 1e-6
    integrity_results.append(ic_t)
    logger.info("TFT integrity: n=%d nan=%s inf=%s const=%s r2_match=%s",
                 ic_t["prediction_count"], ic_t["nan"], ic_t["inf"],
                 ic_t["const_prediction"], ic_t["r2_match"])

    y_true_h, preds_h, ts_h = hyb_data
    ic_h = integrity_check("hybrid", y_true_h, preds_h, ts_h,
                           hyb_result["train_rows"], hyb_result["val_rows"], hyb_result["test_rows"])
    ic_h["r2_reported"] = hyb_result["R2"]
    ic_h["r2_match"] = abs(ic_h["r2_reported"] - ic_h["r2_calculated"]) < 1e-6
    integrity_results.append(ic_h)
    logger.info("Hybrid integrity: n=%d nan=%s inf=%s const=%s r2_match=%s",
                 ic_h["prediction_count"], ic_h["nan"], ic_h["inf"],
                 ic_h["const_prediction"], ic_h["r2_match"])

    logger.info("=" * 40)
    logger.info("LEAKAGE VERIFICATION")
    logger.info("=" * 40)
    train_ts_max = train_df["timestamp"].max()
    val_ts_min = val_df["timestamp"].min()
    val_ts_max = val_df["timestamp"].max()
    test_ts_min = test_df["timestamp"].min()
    logger.info("Train max: %s", train_ts_max)
    logger.info("Val min: %s", val_ts_min)
    logger.info("Val max: %s", val_ts_max)
    logger.info("Test min: %s", test_ts_min)
    assert str(train_ts_max) < str(val_ts_min), "LEAKAGE: train >= val"
    assert str(val_ts_max) < str(test_ts_min), "LEAKAGE: val >= test"
    logger.info("No temporal overlap between train/val/test: PASS")

    save_json = lambda obj, name: json.dump(obj, open(OUTPUT_DIR / name, "w"), indent=2, default=str)

    save_json({"lstm": lstm_gru_results["lstm"], "gru": lstm_gru_results["gru"]}, "lstm_gru.json")
    save_json({"tft": tft_result}, "tft.json")
    save_json({"hybrid": hyb_result}, "hybrid.json")
    save_json(integrity_results, "integrity.json")

    summary = {
        "phase": "6 aligned validation",
        "seed": SEED,
        "split": "70/15/15 chronological",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "leakage_check": "PASS",
        "lstm_gru_results": lstm_gru_results,
        "tft_result": tft_result,
        "hybrid_result": hyb_result,
        "n_models": 4,
        "ablation_run": False,
        "multiyear_run": False,
    }
    save_json(summary, "summary.json")

    print("\n" + "=" * 60)
    print("PHASE 6 ALIGNED VALIDATION SUMMARY")
    print("=" * 60)
    for name in ["lstm", "gru"]:
        r = lstm_gru_results[name]
        print(f"  {name.upper()}: MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} n={r['n_test']}")
    r = tft_result
    print(f"  TFT:    MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} n={r['n_test']}")
    r = hyb_result
    print(f"  Hybrid: MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f} n={r['n_test']}")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("2020-2025 multi-year validation: NOT started")
    print("Ablation: NOT started")
    print("=" * 60)


if __name__ == "__main__":
    main()
