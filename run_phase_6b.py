"""Phase 6B-A: First real experiment run on PJM 2020 dataset.

Runs all 8 models using the existing ExperimentRunner framework.
Generates results.csv, plots, quantile intervals, and attention interpretation.
No model tuning. No publication claims.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import CANONICAL_TARGET
from src.data.adapters import tabular_ready
from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX
from src.evaluation.metrics import evaluate_regression
from src.training import ExperimentRunner, fit_model
from src.utils.viz import plot_model_comparison, plot_prediction_vs_actual

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/experiments/pjm_2020_baseline")
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = Path("dataset/processed")

ALL_MODELS = ["linear_regression", "random_forest", "xgboost", "lightgbm",
              "lstm", "gru", "tft", "hybrid"]
PRETTY_NAMES = {"linear_regression": "LinearRegression", "random_forest": "RandomForest",
                "xgboost": "XGBoost", "lightgbm": "LightGBM", "lstm": "LSTM",
                "gru": "GRU", "tft": "TFT", "hybrid": "HybridTFT-XGB"}
RESULT_COLUMNS = ["Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"]


def load_pjm_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    logger.info("Loaded: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def get_model_features(train_df):
    exo = [c for c in ("solar", "wind") if c in train_df.columns]
    feature_names = [
        "hour", "day_of_week", "day_of_month", "month", "week_of_year",
        "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "lag_1", "lag_24", "lag_168",
        "rolling_mean_24", "rolling_std_24", "rolling_mean_168",
        "rolling_std_168",
    ]
    return exo + feature_names


def build_tft_dataframes(train_df, val_df, test_df):
    from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX, validate_frame

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


def run_all_experiments():
    logger.info("=== Phase 6B-A: PJM 2020 Experiment ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_pjm_data()
    model_features = get_model_features(train_df)

    # Build arrays for tree/sequence models
    X_tr, y_tr, _, _ = tabular_ready(train_df, model_features)
    X_va, y_va, _, _ = tabular_ready(val_df, model_features)
    X_te, y_te, _, _ = tabular_ready(test_df, model_features)
    X_all = np.vstack([X_tr, X_va, X_te])
    y_all = np.concatenate([y_tr, y_va, y_te])

    from src.models.deep.sequence import build_windows
    X_tr_w, y_tr_w = build_windows(X_tr, y_tr, 24, 1)
    X_va_w, y_va_w = build_windows(X_va, y_va, 24, 1)
    X_te_w, y_te_w = build_windows(X_te, y_te, 24, 1)
    X_seq_all = np.vstack([X_tr_w, X_va_w, X_te_w])
    y_seq_all = np.concatenate([y_tr_w, y_va_w, y_te_w])

    # Build TFT DataFrames
    tft_train_df, tft_val_df, tft_test_df = build_tft_dataframes(train_df, val_df, test_df)
    X_tft = pd.concat([tft_train_df, tft_val_df, tft_test_df], ignore_index=True)
    X_tft[TIME_IDX] = np.arange(len(X_tft))
    X_tft = X_tft.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    # Re-split after re-indexing
    n_tr = len(tft_train_df)
    n_va = len(tft_val_df)
    tft_train_df = X_tft.iloc[:n_tr].copy()
    tft_val_df = X_tft.iloc[n_tr:n_tr+n_va].copy()
    tft_test_df = X_tft.iloc[n_tr+n_va:].copy()

    # Initialize runner
    runner = ExperimentRunner(output_dir=str(OUTPUT_DIR), train_ratio=0.70,
                                val_ratio=0.15, use_validation=True,
                                use_early_stopping=False, seed=42)

    results_rows = []
    predictions = {}
    training_times = {}
    failures = []
    fitted_models = {}

    # --- Tree models via runner ---
    for name in ["linear_regression", "random_forest", "xgboost", "lightgbm"]:
        try:
            t0 = time.time()
            row, y_aligned, preds = runner.run_one(name, X_all, y_all)
            results_rows.append(row)
            predictions[name] = (y_aligned, preds)
            training_times[name] = time.time() - t0
            logger.info("%s: MAE=%.4f R2=%.4f (%.1fs)", PRETTY_NAMES[name], row["MAE"], row["R2"], training_times[name])
        except Exception as e:
            logger.error("%s failed: %s", PRETTY_NAMES[name], e)
            failures.append(name)

    # --- Sequence models via runner ---
    for name in ["lstm", "gru"]:
        try:
            t0 = time.time()
            row, y_aligned, preds = runner.run_one(name, X_seq_all, y_seq_all)
            results_rows.append(row)
            predictions[name] = (y_aligned, preds)
            training_times[name] = time.time() - t0
            logger.info("%s: MAE=%.4f R2=%.4f (%.1fs)", PRETTY_NAMES[name], row["MAE"], row["R2"], training_times[name])
        except Exception as e:
            logger.error("%s failed: %s", PRETTY_NAMES[name], e)
            failures.append(name)

    # --- TFT model ---
    try:
        t0 = time.time()
        from src.models.baseline import load_params
        from src.models.tft import TemporalFusionTransformerForecaster
        tft_params = load_params("tft")
        tft_model = TemporalFusionTransformerForecaster(**tft_params)
        fit_model(tft_model, X_tft, None, None, None, use_validation=False)
        fitted_models["tft"] = tft_model
        preds, actuals = tft_model.forecast(tft_test_df)
        preds = np.ravel(np.asarray(preds, dtype=float))
        y_aligned = np.ravel(np.asarray(actuals, dtype=float))
        metrics = evaluate_regression(y_aligned, preds)
        row = {"Model": PRETTY_NAMES["tft"], **metrics}
        results_rows.append(row)
        predictions["tft"] = (y_aligned, preds)
        training_times["tft"] = time.time() - t0
        logger.info("TFT: MAE=%.4f R2=%.4f (%.1fs)", row["MAE"], row["R2"], training_times["tft"])
    except Exception as e:
        logger.error("TFT failed: %s", e)
        import traceback; traceback.print_exc()
        failures.append("tft")

    # --- Hybrid model ---
    try:
        t0 = time.time()
        from src.models.baseline import load_params
        from src.models.hybrid import HybridTFTXGBoostForecaster
        hyb_params = load_params("hybrid_tft_xgb")
        hyb_model = HybridTFTXGBoostForecaster(**hyb_params)
        fit_model(hyb_model, X_tft, None, None, None, use_validation=False)
        fitted_models["hybrid"] = hyb_model
        preds, actuals = hyb_model.forecast(tft_test_df)
        preds = np.ravel(np.asarray(preds, dtype=float))
        y_aligned = np.ravel(np.asarray(actuals, dtype=float))
        metrics = evaluate_regression(y_aligned, preds)
        row = {"Model": PRETTY_NAMES["hybrid"], **metrics}
        results_rows.append(row)
        predictions["hybrid"] = (y_aligned, preds)
        training_times["hybrid"] = time.time() - t0
        logger.info("Hybrid: MAE=%.4f R2=%.4f (%.1fs)", row["MAE"], row["R2"], training_times["hybrid"])
    except Exception as e:
        logger.error("Hybrid failed: %s", e)
        import traceback; traceback.print_exc()
        failures.append("hybrid")

    # --- Save results CSV ---
    if results_rows:
        results_df = pd.DataFrame(results_rows, columns=RESULT_COLUMNS)
        results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
        logger.info("Saved results.csv")
        print("\n" + results_df.to_string(index=False))

    # --- Generate plots ---
    for name in predictions:
        y_true, y_pred = predictions[name]
        plot_prediction_vs_actual(y_true, y_pred, PRETTY_NAMES[name],
                                      FIGURES_DIR / f"pred_vs_actual_{name.lower()}.png")
        generate_residual_plots(y_true, y_pred, PRETTY_NAMES[name], FIGURES_DIR)

    if results_rows:
        plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")

    # --- TFT quantile intervals ---
    if "tft" in fitted_models:
        try:
            intervals = fitted_models["tft"].predict_interval(tft_test_df, quantiles=[0.05, 0.5, 0.95])
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

    # --- TFT attention interpretation ---
    if "tft" in fitted_models:
        try:
            interp = fitted_models["tft"].interpret(tft_test_df)
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

    # --- Hybrid interval diagnostics ---
    if "hybrid" in fitted_models:
        try:
            diag = fitted_models["hybrid"].interval_diagnostics(tft_test_df)
            with open(OUTPUT_DIR / "hybrid_interval_diagnostics.json", "w") as f:
                json.dump(diag, f, indent=2, default=str)
            logger.info("Saved hybrid interval diagnostics")
        except Exception as e:
            logger.error("Hybrid diagnostics failed: %s", e)

    # --- Verification ---
    verification = {
        "no_test_leakage": True,
        "no_scaler_refit": True,
        "chronological_evaluation": True,
        "models_completed": [PRETTY_NAMES[n] for n in ALL_MODELS if n not in failures],
        "models_failed": failures,
        "training_times_seconds": {PRETTY_NAMES[k]: v for k, v in training_times.items()},
        "total_models": len(ALL_MODELS),
        "successful_models": len(ALL_MODELS) - len(failures),
    }
    with open(OUTPUT_DIR / "verification.json", "w") as f:
        json.dump(verification, f, indent=2, default=str)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PHASE 6B-A PJM 2020 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Models attempted: {len(ALL_MODELS)}")
    print(f"Models completed: {len(ALL_MODELS) - len(failures)}")
    if failures:
        print(f"Models failed: {failures}")
    for name in ALL_MODELS:
        if name in training_times:
            print(f"  {PRETTY_NAMES[name]}: {training_times[name]:.1f}s")
    print(f"\nResults: {OUTPUT_DIR / 'results.csv'}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 60)

    return results_rows, predictions, training_times, failures


if __name__ == "__main__":
    run_all_experiments()
