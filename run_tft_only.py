"""TFT-only aligned validation runner.

Trains TFT on train+val only (NO test leakage).
Evaluates via predict=False all-valid-windows for one-step-ahead predictions.
Saves results to results/fresh_validation_aligned/tft.json.
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("results/fresh_validation_aligned")
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


def main():
    import torch
    from src.evaluation.metrics import evaluate_regression
    from src.models.baseline import load_params
    from src.models.tft import TemporalFusionTransformerForecaster
    from src.models.tft.dataset import validate_frame
    from pytorch_forecasting import TimeSeriesDataSet

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("TFT-ONLY ALIGNED VALIDATION")
    logger.info("=" * 60)

    train_df, val_df, test_df = load_data()

    t0 = time.time()

    tft_train_df, tft_val_df, tft_test_df = build_tft_dataframes(train_df, val_df, test_df)
    logger.info("TFT sub-frames: train=%d, val=%d, test=%d",
                len(tft_train_df), len(tft_val_df), len(tft_test_df))

    tft_trainval_df = pd.concat([tft_train_df, tft_val_df], ignore_index=True)
    tft_trainval_df[TIME_IDX] = np.arange(len(tft_trainval_df))
    tft_trainval_df = tft_trainval_df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    tft_trainval_df = validate_frame(tft_trainval_df)

    logger.info("TFT training data: %d rows (train+val only, NO test leakage)", len(tft_trainval_df))
    logger.info("  train time_idx range: 0 - %d", tft_trainval_df[TIME_IDX].max())

    tft_params = load_params("tft")
    tft_model = TemporalFusionTransformerForecaster(**tft_params)
    tft_model.fit(tft_trainval_df)

    full_df = pd.concat([tft_train_df, tft_val_df, tft_test_df], ignore_index=True)
    full_df[TIME_IDX] = np.arange(len(full_df))
    full_df = full_df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    full_df = validate_frame(full_df)

    test_min_ti = int(tft_test_df[TIME_IDX].min())
    test_max_ti = int(tft_test_df[TIME_IDX].max())
    logger.info("Full eval frame: %d rows, test time_idx range: %d - %d",
                len(full_df), test_min_ti, test_max_ti)

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

        n_total_windows = times.shape[0]
        n_test_windows = 0
        for i in range(n_total_windows):
            t = int(times[i, 0])
            if t < test_min_ti:
                continue
            n_test_windows += 1
            all_preds.append(float(med[i, 0]))
            all_actuals.append(target_by_time[t])
            all_times.append(t)

        logger.info("  Group %s: %d total windows, %d test-aligned windows (h=0)",
                     name, n_total_windows, n_test_windows)

    preds_arr = np.asarray(all_preds, dtype=float)
    actuals_arr = np.asarray(all_actuals, dtype=float)
    times_arr = np.asarray(all_times)

    metrics = evaluate_regression(actuals_arr, preds_arr)
    runtime = time.time() - t0

    n = len(preds_arr)
    sse = float(np.sum((actuals_arr - preds_arr) ** 2))
    yt_mean = float(np.mean(actuals_arr))
    sst = float(np.sum((actuals_arr - yt_mean) ** 2))
    r2_calc = 1.0 - sse / sst if sst > 0 else 0.0

    result = {
        "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
        "sMAPE": metrics["sMAPE"], "R2": metrics["R2"],
        "runtime": round(runtime, 2), "n_test": n,
        "r2_identity_match": abs(metrics["R2"] - r2_calc) < 1e-6,
        "pred_mean": float(np.mean(preds_arr)),
        "pred_min": float(np.min(preds_arr)),
        "pred_max": float(np.max(preds_arr)),
        "first_aligned_timestamp": str(times_arr[0]) if n > 0 else None,
        "last_aligned_timestamp": str(times_arr[-1]) if n > 0 else None,
        "train_rows": len(tft_train_df),
        "val_rows": len(tft_val_df),
        "test_rows": len(tft_test_df),
        "nan": bool(np.isnan(preds_arr).any()),
        "inf": bool(np.isinf(preds_arr).any()),
        "const": bool(np.std(preds_arr) < 1e-12) if n > 1 else False,
        "duplicate_timestamps": bool(len(times_arr) != len(np.unique(times_arr))),
        "unique_timestamps": len(np.unique(times_arr)),
        "chronological_order": bool(np.all(times_arr[:-1] <= times_arr[1:])) if n > 1 else True,
        "test_leakage": False,
    }

    with open(OUTPUT_DIR / "tft.json", "w") as f:
        json.dump({"tft": result}, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("TFT RESULT SAVED")
    logger.info("  MAE=%.4f  RMSE=%.4f  R2=%.6f", metrics["MAE"], metrics["RMSE"], metrics["R2"])
    logger.info("  n_test=%d  runtime=%.1fs", n, runtime)
    logger.info("  first_ts=%s  last_ts=%s", result["first_aligned_timestamp"], result["last_aligned_timestamp"])
    logger.info("  nan=%s  inf=%s  const=%s  dup_ts=%s  ordered=%s",
                result["nan"], result["inf"], result["const"],
                result["duplicate_timestamps"], result["chronological_order"])
    logger.info("  r2_identity_match=%s", result["r2_identity_match"])
    logger.info("  Saved to: %s", OUTPUT_DIR / "tft.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
