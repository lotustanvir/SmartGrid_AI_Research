"""Finalize Phase 6B-A: Add TFT results and generate TFT-specific outputs."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import CANONICAL_TARGET
from src.data.adapters import tabular_ready
from src.evaluation.metrics import evaluate_regression
from src.models.baseline import load_params
from src.models.tft import TemporalFusionTransformerForecaster
from src.training import fit_model
from src.utils.viz import plot_prediction_vs_actual, plot_model_comparison

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/experiments/pjm_2020_baseline")
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = Path("dataset/processed")
TIME_IDX = "time_idx"
GROUP_ID = "group_id"
TARGET = "electricity_demand"


def build_tft_dataframes(train_df, val_df, test_df):
    from src.models.tft.dataset import validate_frame

    def make_frame(df):
        out = df.copy()
        out[TARGET] = out[CANONICAL_TARGET].astype(float)
        out[TIME_IDX] = 0
        out[GROUP_ID] = "single"
        known = [c for c in ("hour", "day_of_week", "temperature", "solar", "wind") if c in out.columns]
        ordered = [TIME_IDX, GROUP_ID, TARGET] + known + \
            [c for c in out.columns if c not in [TIME_IDX, GROUP_ID, TARGET] + known]
        out = out[ordered]
        feature_cols = [c for c in ordered if c not in [TIME_IDX, GROUP_ID, TARGET]]
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        out[TIME_IDX] = np.arange(len(out))
        out[GROUP_ID] = "single"
        return validate_frame(out)

    return make_frame(train_df), make_frame(val_df), make_frame(test_df)


def generate_residual_plots(y_true, y_pred, model_name, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{model_name}: Residuals vs Predicted")
    axes[1].hist(residuals, bins=50, edgecolor="black")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{model_name}: Residual Distribution")
    fig.tight_layout()
    path = figures_dir / f"residuals_{model_name.lower().replace('-', '_')}.png"
    fig.savefig(path, dpi=120); plt.close(fig)


def main():
    logger.info("Finalizing Phase 6B-A...")

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    model_features = [c for c in ["solar", "wind", "hour", "day_of_week",
                                   "day_of_month", "month", "week_of_year",
                                   "is_weekend", "hour_sin", "hour_cos",
                                   "dow_sin", "dow_cos", "month_sin",
                                   "month_cos", "lag_1", "lag_24", "lag_168",
                                   "rolling_mean_24", "rolling_std_24",
                                   "rolling_mean_168", "rolling_std_168"]
                      if c in train_df.columns]

    # Build TFT DataFrames
    tft_train_df, tft_val_df, tft_test_df = build_tft_dataframes(train_df, val_df, test_df)
    X_tft = pd.concat([tft_train_df, tft_val_df, tft_test_df], ignore_index=True)
    X_tft[TIME_IDX] = np.arange(len(X_tft))
    X_tft = X_tft.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    n_tr = len(tft_train_df)
    n_va = len(tft_val_df)
    tft_train_df = X_tft.iloc[:n_tr].copy()
    tft_val_df = X_tft.iloc[n_tr:n_tr+n_va].copy()
    tft_test_df = X_tft.iloc[n_tr+n_va:].copy()

    # Fit TFT
    logger.info("Fitting TFT...")
    t0 = time.time()
    tft_params = load_params("tft")
    tft_model = TemporalFusionTransformerForecaster(**tft_params)
    fit_model(tft_model, X_tft, None, None, None, use_validation=False)
    tft_time = time.time() - t0
    logger.info("TFT fitted in %.1fs", tft_time)

    # Generate predictions
    preds, actuals = tft_model.forecast(tft_test_df)
    preds = np.ravel(np.asarray(preds, dtype=float))
    y_aligned = np.ravel(np.asarray(actuals, dtype=float))
    metrics = evaluate_regression(y_aligned, preds)
    logger.info("TFT metrics: MAE=%.4f R2=%.4f", metrics["MAE"], metrics["R2"])

    # Update results.csv
    results_path = OUTPUT_DIR / "results.csv"
    results_df = pd.read_csv(results_path)
    new_row = pd.DataFrame([{"Model": "TFT", **metrics}])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    logger.info("Updated results.csv")

    # Print updated results
    print("\n" + results_df.to_string(index=False))

    # Generate TFT plots
    plot_prediction_vs_actual(y_aligned, preds, "TFT",
                              FIGURES_DIR / "pred_vs_actual_tft.png")
    generate_residual_plots(y_aligned, preds, "TFT", FIGURES_DIR)

    # Regenerate model_comparison
    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")

    # TFT quantile intervals
    try:
        intervals = tft_model.predict_interval(tft_test_df, quantiles=[0.05, 0.5, 0.95])
        with open(OUTPUT_DIR / "tft_quantile_intervals.json", "w") as f:
            json.dump({
                "quantiles": [0.05, 0.5, 0.95],
                "lower": np.asarray(intervals[0.05]).tolist(),
                "median": np.asarray(intervals[0.5]).tolist(),
                "upper": np.asarray(intervals[0.95]).tolist(),
                "n_points": len(intervals[0.5]),
            }, f, indent=2, default=str)
        logger.info("Saved TFT quantile intervals")
    except Exception as e:
        logger.error("TFT intervals failed: %s", e)

    # TFT attention interpretation
    try:
        interp = tft_model.interpret(tft_test_df)
        with open(OUTPUT_DIR / "tft_attention_interpretation.json", "w") as f:
            json.dump({
                "attention_shape": list(np.asarray(interp["attention"]).shape),
                "attention_mean": float(np.mean(interp["attention"])),
                "encoder_variables": {k: float(v) for k, v in interp["encoder_variables"].items()},
                "decoder_variables": {k: float(v) for k, v in interp["decoder_variables"].items()},
                "static_variables": {k: float(v) for k, v in interp["static_variables"].items()},
            }, f, indent=2, default=str)
        logger.info("Saved TFT attention interpretation")
    except Exception as e:
        logger.error("TFT interpretation failed: %s", e)

    # Update verification.json
    verification_path = OUTPUT_DIR / "verification.json"
    with open(verification_path) as f:
        verification = json.load(f)
    verification["models_completed"] = ["LinearRegression", "RandomForest", "XGBoost",
                                         "LightGBM", "LSTM", "GRU", "TFT"]
    verification["models_failed"] = ["hybrid"]
    verification["training_times_seconds"]["TFT"] = round(tft_time, 1)
    verification["successful_models"] = 7
    verification["total_models"] = 8
    with open(verification_path, "w") as f:
        json.dump(verification, f, indent=2, default=str)
    logger.info("Updated verification.json")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 6B-A FINAL SUMMARY")
    print("=" * 60)
    print(f"Models completed: {len(verification['models_completed'])}/8")
    print(f"Models failed: {verification['models_failed']}")
    print(f"TFT training time: {tft_time:.1f}s")
    print(f"TFT MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")
    print(f"\nOutputs: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
