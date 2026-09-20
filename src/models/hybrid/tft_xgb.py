"""Hybrid TFT + XGBoost residual correction (Phase 5B: research-hardened).

Pipeline:

1. Fit TFT on the training frame (refit on train+val when given).
2. Generate OUT-OF-FOLD TFT predictions via walk-forward expanding windows
   (default ``residual_training_mode="oof"``).
3. residual_oof = y_true_oof - y_tft_oof per (window, horizon step).
4. Fit XGBoost ONLY on OOF residuals, using forecast-time-known features.
5. Final forecast = TFT median + predicted residual.

``in_sample`` mode exists for debugging/smoke tests only and emits a
loud warning: its results are NOT valid for research evaluation.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import evaluate_regression
from src.models.base import BaseForecaster
from src.models.baseline import XGBoostForecaster
from src.models.tft import TemporalFusionTransformerForecaster
from src.models.tft.dataset import (
    GROUP_ID,
    TARGET,
    TIME_IDX,
    coerce_frame,
    resolve_roles,
    validate_frame,
)

logger = logging.getLogger(__name__)

STAT_NAMES = ["enc_target_mean", "enc_target_std", "enc_target_last"]


class HybridTFTXGBoostForecaster(BaseForecaster):
    """TFT baseline forecast + XGBoost residual correction.

    Args:
        tft_config: Kwargs for :class:`TemporalFusionTransformerForecaster`.
        xgb_config: Kwargs for :class:`XGBoostForecaster` (residual model).
        residual_training_mode: ``"oof"`` (walk-forward, research-safe,
            default) or ``"in_sample"`` (debug only, warns loudly).
        n_oof_folds: Walk-forward folds for OOF residual generation.
        clip_to_bounds: If True, clip the corrected median into
            ``[base_tft_lower, base_tft_upper]``. Default False (report only).
        random_state: Seed used as default for both sub-models.
    """

    _config_name = "hybrid_tft_xgb"

    def __init__(
        self,
        tft_config: Optional[dict] = None,
        xgb_config: Optional[dict] = None,
        residual_training_mode: str = "oof",
        n_oof_folds: int = 3,
        clip_to_bounds: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if residual_training_mode not in ("oof", "in_sample"):
            raise ValueError(
                f"residual_training_mode must be 'oof' or 'in_sample', "
                f"got {residual_training_mode!r}"
            )
        if n_oof_folds < 2:
            raise ValueError(f"n_oof_folds must be >= 2, got {n_oof_folds!r}")
        tft_config = dict(tft_config or {})
        xgb_config = dict(xgb_config or {})
        tft_config.setdefault("random_state", random_state)
        xgb_config.setdefault("random_state", random_state)
        self.tft_config = tft_config
        self.xgb_config = xgb_config
        self.residual_training_mode = residual_training_mode
        self.n_oof_folds = n_oof_folds
        self.clip_to_bounds = clip_to_bounds
        self._tft = TemporalFusionTransformerForecaster(**tft_config)
        self._xgb: Optional[XGBoostForecaster] = None
        self.residual_feature_names_: Optional[list[str]] = None
        self.n_residual_samples_: int = 0
        self.residual_stats_: dict = {}
        self.oof_folds_: list[dict] = []
        self.residual_period_: Optional[tuple[int, int]] = None
        self._config = {
            "tft_config": dict(tft_config),
            "xgb_config": dict(xgb_config),
            "residual_training_mode": residual_training_mode,
            "n_oof_folds": n_oof_folds,
            "clip_to_bounds": clip_to_bounds,
            "random_state": random_state,
        }

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "HybridTFTXGBoostForecaster":
        """Build from ``configs/models.yaml`` (``hybrid_tft_xgb`` section)."""
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    # ------------------------------------------------------------------
    # Residual feature engineering (forecast-time-known info only)
    # ------------------------------------------------------------------
    def _encoder_stats(self, history: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.mean(history)),
            float(np.std(history)),
            float(history[-1]),
        )

    def _window_rows(
        self, df: pd.DataFrame, known: list[str], horizon: int
    ) -> np.ndarray:
        """Residual feature rows for trailing windows per group."""
        E = self._tft.encoder_length
        feats = []
        for name, g in df.groupby(GROUP_ID, sort=True):
            g = g.sort_values("time_idx").reset_index(drop=True)
            if len(g) < E + horizon:
                raise ValueError(
                    f"Group {name!r} too short ({len(g)}) for "
                    f"encoder={E} + horizon={horizon}."
                )
            enc_targets = g[TARGET].to_numpy(dtype=float)[:-horizon]
            mean, std, last = self._encoder_stats(enc_targets[-E:])
            for h in range(horizon):
                row = g.iloc[len(g) - horizon + h]
                feats.append(
                    [float(row[c]) for c in known] + [float(h), mean, std, last]
                )
        return np.asarray(feats, dtype=float)

    def _residual_rows(
        self,
        tft: TemporalFusionTransformerForecaster,
        df: pd.DataFrame,
        known: list[str],
        cutoff_time: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run a fitted TFT over frame windows; return (X_res, resid, groups, times).

        With ``cutoff_time`` set, only windows whose decoder starts strictly
        after it are kept (out-of-fold guarantee for walk-forward folds).
        """
        from pytorch_forecasting import TimeSeriesDataSet

        E = tft.encoder_length
        H = tft.prediction_length
        assert tft._training is not None
        X_parts, r_parts, g_parts, t_parts = [], [], [], []
        for name, g in df.groupby(GROUP_ID, sort=True):
            g = g.sort_values("time_idx").reset_index(drop=True)
            ds = TimeSeriesDataSet.from_dataset(
                tft._training, g, predict=False, stop_randomization=True
            )
            loader = ds.to_dataloader(
                train=False, batch_size=tft.batch_size, num_workers=0
            )
            med = tft._median_for_loader(loader).cpu().numpy()
            times_per_window = []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    times_per_window.append(
                        np.asarray(x["decoder_time_idx"].cpu())
                    )
            times = np.concatenate(times_per_window, axis=0)
            assert times.shape[0] == med.shape[0], "Window order mismatch."
            target_by_time = dict(zip(g["time_idx"].tolist(), g[TARGET].tolist()))
            enc_all = g[TARGET].to_numpy(dtype=float)
            time_to_pos = {t: i for i, t in enumerate(g["time_idx"].tolist())}
            for i in range(times.shape[0]):
                if cutoff_time is not None and int(times[i, 0]) <= cutoff_time:
                    continue
                dec_start = time_to_pos[int(times[i, 0])]
                enc_hist = enc_all[max(0, dec_start - E) : dec_start]
                mean, std, last = self._encoder_stats(enc_hist)
                for h in range(med.shape[1]):
                    t = int(times[i, h])
                    row_feats = [float(g.loc[time_to_pos[t], c]) for c in known]
                    row_feats += [float(h), mean, std, last]
                    X_parts.append(row_feats)
                    r_parts.append(target_by_time[t] - float(med[i, h]))
                    g_parts.append(name)
                    t_parts.append(t)
        return (
            np.asarray(X_parts, dtype=float),
            np.asarray(r_parts, dtype=float),
            np.asarray(g_parts),
            np.asarray(t_parts),
        )

    def _oof_fold_bounds(self, times: np.ndarray) -> list[tuple[int, int, int]]:
        """Walk-forward (train_end, block_start, block_end) per fold.

        The training span is split so fold ``k`` trains on everything up to
        ``train_end`` and predicts the next contiguous block; blocks tile
        the OOF region without overlap. No shuffle anywhere.
        """
        E = self._tft.encoder_length
        H = self._tft.prediction_length
        val_size = self._tft.val_size
        uniq = np.unique(np.asarray(times, dtype=int))
        # First fold must itself hold encoder + horizon + TFT val split.
        # split_datasets needs cutoff > E + H with cutoff = S-1-val_size.
        min_train = E + H + val_size + H + 2
        if len(uniq) < min_train + self.n_oof_folds * H:
            raise ValueError(
                f"Need >= {min_train + self.n_oof_folds * H} distinct times "
                f"for {self.n_oof_folds} OOF folds, got {len(uniq)}."
            )
        oof_start = min_train
        oof_len = len(uniq) - oof_start
        bounds = []
        for k in range(self.n_oof_folds):
            b0 = oof_start + (oof_len * k) // self.n_oof_folds
            b1 = oof_start + (oof_len * (k + 1)) // self.n_oof_folds
            train_end = int(uniq[b0 - 1])
            block_end = int(uniq[b1 - 1])
            assert train_end < int(uniq[b0]), "Fold is not chronological."
            bounds.append((train_end, int(uniq[b0]), block_end))
        for (te1, _, be1), (te2, bs2, _) in zip(bounds, bounds[1:]):
            assert be1 < bs2 and te1 < te2, "OOF blocks must not overlap."
        return bounds

    def _fit_oof_residuals(
        self, df: pd.DataFrame, known: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Walk-forward OOF residuals; final TFT already fitted on full data."""
        self.oof_folds_ = []
        X_all, r_all, t_all = [], [], []
        for k, (train_end, block_start, block_end) in enumerate(
            self._oof_fold_bounds(df["time_idx"].to_numpy())
        ):
            fold_tft = TemporalFusionTransformerForecaster(
                **{**self.tft_config, "random_state": self._tft.random_state}
            )
            fold_df = df[df["time_idx"] <= train_end]
            fold_tft.fit(fold_df)
            frame = df[df["time_idx"] <= block_end]
            Xk, rk, _, tk = self._residual_rows(fold_tft, frame, known, train_end)
            if Xk.shape[0] == 0:
                raise ValueError(f"OOF fold {k} produced no residual samples.")
            assert tk.min() > train_end, "Fold predictions leak into training."
            assert tk.max() <= block_end, "Fold predictions exceed block."
            self.oof_folds_.append(
                {
                    "fold": k,
                    "train_end_time": train_end,
                    "predict_start": int(tk.min()),
                    "predict_end": int(tk.max()),
                    "n_samples": int(Xk.shape[0]),
                }
            )
            X_all.append(Xk)
            r_all.append(rk)
            t_all.append(tk)
        X_res = np.concatenate(X_all, axis=0)
        resid = np.concatenate(r_all, axis=0)
        times = np.concatenate(t_all, axis=0)
        self.residual_period_ = (int(times.min()), int(times.max()))
        logger.info("OOF folds: %s", self.oof_folds_)
        return X_res, resid

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val=None,
        **kwargs,
    ) -> "HybridTFTXGBoostForecaster":
        """Fit TFT (+ optional train+val refit), then XGBoost on residuals.

        ``residual_training_mode="oof"`` (default): walk-forward OOF TFT
        predictions over the ``(X, y)`` frame only; the residual learner
        never sees validation/test targets. ``"in_sample"`` warns loudly
        and exists for debugging only.
        """
        df = validate_frame(coerce_frame(X, y))
        full = df
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be provided together.")
            if isinstance(X_val, pd.DataFrame):
                vdf = validate_frame(coerce_frame(X_val, y_val))
            else:
                # Array blocks restart at 0; offset past the training block
                # so train/val/test timestamps stay globally increasing.
                from src.models.tft.dataset import frame_from_arrays

                vdf = validate_frame(
                    frame_from_arrays(
                        X_val, y_val, time_start=int(df[TIME_IDX].max()) + 1
                    )
                )
            assert vdf["time_idx"].min() > df["time_idx"].max(), (
                "Validation must be strictly after training."
            )
            full = pd.concat([df, vdf], ignore_index=True)
        self._tft = TemporalFusionTransformerForecaster(
            **{**self.tft_config, "random_state": self._tft.random_state}
        )
        self._tft.fit(full)
        known, _, _ = resolve_roles(df)
        H = self._tft.prediction_length
        self.oof_folds_ = []
        if self.residual_training_mode == "in_sample":
            warnings.warn(
                "in_sample residuals are optimistic and NOT valid for "
                "research evaluation (debugging only).",
                UserWarning,
                stacklevel=2,
            )
            logger.warning("in_sample residual mode: results not research-valid.")
            X_res, resid, _, _ = self._residual_rows(self._tft, df, known)
        else:
            X_res, resid = self._fit_oof_residuals(df, known)
        if X_res.shape[0] == 0:
            raise ValueError("No residual training samples generated.")
        self.residual_feature_names_ = list(known) + ["horizon_idx"] + STAT_NAMES
        self.n_residual_samples_ = int(X_res.shape[0])
        self._xgb = XGBoostForecaster(**self.xgb_config)
        self._xgb.fit(X_res, resid)
        resid_pred = self._xgb.predict(X_res)
        self.residual_stats_ = {
            "mode": self.residual_training_mode,
            "mean_abs_residual_before": float(np.mean(np.abs(resid))),
            "mean_abs_residual_after": float(np.mean(np.abs(resid - resid_pred))),
            "n_windows_samples": self.n_residual_samples_,
            "horizon": int(H),
        }
        self._is_fitted = True
        logger.info(
            "Fitted hybrid (%s): TFT + XGB on %d residual samples "
            "(|resid| %.4f -> %.4f)",
            self.residual_training_mode,
            self.n_residual_samples_,
            self.residual_stats_["mean_abs_residual_before"],
            self.residual_stats_["mean_abs_residual_after"],
        )
        return self

    def _correct(self, medians_2d: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Add XGBoost residual correction to TFT median windows."""
        self._check_fitted()
        assert self._xgb is not None and self.residual_feature_names_ is not None
        known = self.residual_feature_names_[: -1 - len(STAT_NAMES)]
        H = self._tft.prediction_length
        X_res = self._window_rows(df, known, H)
        if X_res.shape[0] != medians_2d.shape[0] * H:
            raise ValueError("Residual feature / prediction shape mismatch.")
        corr = self._xgb.predict(X_res).reshape(medians_2d.shape)
        return np.asarray(medians_2d, dtype=float) + np.asarray(corr, dtype=float)

    def _as_2d(self, med) -> np.ndarray:
        med = np.asarray(med, dtype=float)
        if med.ndim == 1:
            med = med.reshape(1, -1)
        return med

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Final forecast = TFT median + residual correction."""
        df = X if isinstance(X, pd.DataFrame) else None
        if df is None:
            raise ValueError(
                "Hybrid predict needs a DataFrame (or use forecast(X, y)); "
                "plain arrays lack target history."
            )
        med = self._as_2d(self._tft.predict(df))
        final = self._correct(med, validate_frame(df, allow_nan_target=True))
        if self._tft.prediction_length == 1:
            return np.asarray(final).ravel()
        return np.asarray(final)

    def forecast(
        self, X: Union[pd.DataFrame, np.ndarray], y=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trailing-horizon final forecasts + actuals per group."""
        self._check_fitted()
        df = validate_frame(coerce_frame(X, y))
        med, acts = self._tft.forecast(df)
        final = self._correct(self._as_2d(med), df)
        if self._tft.prediction_length == 1:
            return np.asarray(final).ravel(), np.asarray(acts).ravel()
        return np.asarray(final), np.asarray(acts)

    def evaluate(
        self, X: Union[pd.DataFrame, np.ndarray], y=None, epsilon: float = 1e-8
    ) -> dict:
        """Score final forecasts (see :meth:`forecast`)."""
        preds, actuals = self.forecast(X, y)
        return evaluate_regression(
            np.ravel(np.asarray(actuals, dtype=float)),
            np.ravel(np.asarray(preds, dtype=float)),
            epsilon,
        )

    def predict_interval(
        self, X: Union[pd.DataFrame, np.ndarray], quantiles=None
    ) -> dict:
        """Uncalibrated hybrid intervals (NOT calibrated — see below).

        Keys are explicit: ``base_tft_lower`` / ``hybrid_median`` /
        ``base_tft_upper``. Bounds are TFT-native; only the median carries
        the residual correction. Do not describe these as calibrated
        hybrid intervals until recalibration is implemented.
        """
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("predict_interval needs a DataFrame input.")
        want = [float(q) for q in (quantiles or self._tft.quantiles)]
        lo, _, hi = (
            min(self._tft.quantiles),
            0.5,
            max(self._tft.quantiles),
        )
        raw = self._tft.predict_interval(X, quantiles=[lo, 0.5, hi])
        med = self._as_2d(self.predict(X))
        out = {
            "base_tft_lower": np.asarray(raw[lo]),
            "hybrid_median": np.asarray(med),
            "base_tft_upper": np.asarray(raw[hi]),
        }
        if self.clip_to_bounds:
            out["hybrid_median"] = np.minimum(
                np.maximum(out["hybrid_median"], out["base_tft_lower"]),
                out["base_tft_upper"],
            )
        if any(q not in (lo, 0.5, hi) for q in want):
            raise ValueError(
                "Only TFT-native bound/median levels supported pre-recalibration."
            )
        return out

    def interval_diagnostics(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> dict:
        """Check lower <= median <= upper; report (never silently clip)."""
        out = self.predict_interval(X)
        lo, med, hi = (
            np.ravel(out["base_tft_lower"]),
            np.ravel(out["hybrid_median"]),
            np.ravel(out["base_tft_upper"]),
        )
        bad = np.where((med < lo) | (med > hi))[0]
        return {
            "n_points": int(len(med)),
            "n_violations": int(len(bad)),
            "violation_idx": [int(i) for i in bad],
            "clipped": bool(self.clip_to_bounds),
        }

    def ablation(
        self, X: Union[pd.DataFrame, np.ndarray], y=None, epsilon: float = 1e-8
    ) -> dict:
        """Compare TFT-only vs hybrid on the same trailing horizons.

        Returns metric dicts for ``tft`` and ``hybrid`` plus residual
        magnitudes. ("XGBoost only" comes from the baseline runner;
        no superiority is claimed here.)
        """
        self._check_fitted()
        df = validate_frame(coerce_frame(X, y))
        tft_med, acts = self._tft.forecast(df)
        hyb, _ = self.forecast(df)
        a = np.ravel(np.asarray(acts, dtype=float))
        t = np.ravel(np.asarray(tft_med, dtype=float))
        h = np.ravel(np.asarray(hyb, dtype=float))
        return {
            "tft": evaluate_regression(a, t, epsilon),
            "hybrid": evaluate_regression(a, h, epsilon),
            "mean_abs_residual_before": float(np.mean(np.abs(a - t))),
            "mean_abs_residual_after": float(np.mean(np.abs(a - h))),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save TFT payload + pickled residual XGBoost + config."""
        self._check_fitted()
        assert self._xgb is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "forecaster": type(self).__name__,
                "config": dict(self._config),
                "tft": self._tft._to_payload(),
                "xgb": pickle.dumps(self._xgb),
                "residual_feature_names": self.residual_feature_names_,
                "residual_stats": dict(self.residual_stats_),
                "n_residual_samples": int(self.n_residual_samples_),
                "oof_folds": list(self.oof_folds_),
                "residual_period": self.residual_period_,
            },
            path,
        )
        logger.info("Saved %s to %s", type(self).__name__, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HybridTFTXGBoostForecaster":
        """Load a checkpoint saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("forecaster") != cls.__name__:
            raise TypeError(
                f"Expected {cls.__name__} in {path}, "
                f"got {payload.get('forecaster')}"
            )
        obj = cls(**payload["config"])
        obj._tft = TemporalFusionTransformerForecaster._from_payload(payload["tft"])
        obj._xgb = pickle.loads(payload["xgb"])
        obj.residual_feature_names_ = payload["residual_feature_names"]
        obj.residual_stats_ = payload["residual_stats"]
        obj.n_residual_samples_ = payload["n_residual_samples"]
        obj.oof_folds_ = payload.get("oof_folds", [])
        obj.residual_period_ = payload.get("residual_period")
        obj._is_fitted = True
        logger.info("Loaded %s from %s", cls.__name__, path)
        return obj
