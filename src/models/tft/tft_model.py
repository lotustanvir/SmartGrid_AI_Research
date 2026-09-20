"""Temporal Fusion Transformer forecaster (quantile regression)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import evaluate_regression
from src.models.base import BaseForecaster
from src.models.tft.dataset import (
    GROUP_ID,
    TARGET,
    coerce_frame,
    frame_from_arrays,
    predict_loader,
    split_datasets,
    validate_frame,
)

logger = logging.getLogger(__name__)


class TemporalFusionTransformerForecaster(BaseForecaster):
    """TFT behind the common forecaster interface.

    ``fit``/``predict``/``evaluate`` accept either a rich TFT DataFrame
    (time_idx, group_id, electricity_demand + covariates) or ``(X_2d, y)``
    arrays (converted to a single-group frame; keeps ExperimentRunner
    compatibility).

    Point forecasts are the 0.5 quantile; :meth:`predict_interval` exposes
    all fitted quantiles for uncertainty-ready downstream use.

    Args:
        encoder_length: Look-back window.
        prediction_length: Forecast horizon.
        hidden_size: TFT hidden state size.
        attention_head_size: Multi-head attention heads.
        hidden_continuous_size: Continuous variable embedding size.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Maximum Lightning epochs.
        patience: Early-stopping patience on ``val_loss``.
        gradient_clip_val: Gradient clipping value.
        quantiles: Fitted quantile levels (must include 0.5).
        val_size: Trailing time steps held out for validation
            (default: ``4 * prediction_length``).
        checkpoint_dir: Directory for best-checkpoint files (temp dir if None).
        verbose: Enable Lightning progress/model summary.
        random_state: Seed for reproducibility.
    """

    _config_name = "tft"

    def __init__(
        self,
        encoder_length: int = 24,
        prediction_length: int = 6,
        hidden_size: int = 32,
        attention_head_size: int = 2,
        hidden_continuous_size: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 20,
        patience: int = 5,
        gradient_clip_val: float = 0.1,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        val_size: Optional[int] = None,
        checkpoint_dir: Optional[str | Path] = None,
        verbose: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if encoder_length <= 0 or prediction_length <= 0:
            raise ValueError("encoder/prediction lengths must be positive.")
        if hidden_size <= 0 or attention_head_size <= 0:
            raise ValueError("hidden/attention sizes must be positive.")
        if hidden_size % attention_head_size != 0:
            raise ValueError("hidden_size must be divisible by attention_head_size.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        quantiles = tuple(float(q) for q in quantiles)
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include 0.5 (point forecast).")
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantiles must be in (0, 1), got {quantiles!r}")
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_val = gradient_clip_val
        self.quantiles = quantiles
        self.val_size = val_size or 4 * prediction_length
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.verbose = verbose
        self._tft = None
        self._training = None
        self.best_val_loss_: Optional[float] = None
        self.n_epochs_: int = 0
        self.best_checkpoint_: Optional[Path] = None
        self._config = {
            "encoder_length": encoder_length,
            "prediction_length": prediction_length,
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "hidden_continuous_size": hidden_continuous_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "gradient_clip_val": gradient_clip_val,
            "quantiles": list(quantiles),
            "val_size": self.val_size,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "verbose": verbose,
            "random_state": random_state,
        }

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "TemporalFusionTransformerForecaster":
        """Build from ``configs/models.yaml`` (``tft`` section)."""
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _tft_kwargs(self) -> dict:
        from pytorch_forecasting.metrics import QuantileLoss

        return {
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "hidden_continuous_size": self.hidden_continuous_size,
            "dropout": self.dropout,
            "output_size": len(self.quantiles),
            "loss": QuantileLoss(quantiles=list(self.quantiles)),
            "learning_rate": self.learning_rate,
            "log_interval": -1,
        }

    def _trainer(self, ckpt_dir: Path):
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

        return Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1,
            enable_model_summary=self.verbose,
            enable_progress_bar=self.verbose,
            logger=False,
            gradient_clip_val=self.gradient_clip_val,
            default_root_dir=str(ckpt_dir),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=self.patience, mode="min"
                ),
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    filename="tft-best",
                ),
            ],
        )

    def _raw_predict(self, df: pd.DataFrame):
        """Run the TFT and return the raw quantile Output + actuals tensor."""
        self._check_fitted()
        assert self._tft is not None and self._training is not None
        df = validate_frame(df, allow_nan_target=True)
        loader, _ = predict_loader(
            self._training, df, self.batch_size, self.prediction_length
        )
        raw = self._tft.predict(loader, mode="raw", return_x=False)
        preds = torch.as_tensor(np.asarray(raw.prediction, dtype=np.float32))
        actuals = torch.cat([y for _, (y, _) in iter(loader)], dim=0)
        return preds, actuals

    def _median(self, preds: torch.Tensor) -> np.ndarray:
        q50 = list(self.quantiles).index(0.5)
        med = preds[..., q50].cpu().numpy()
        if self.prediction_length == 1:
            return np.asarray(med).ravel()
        return np.asarray(med)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y=None, **kwargs
    ) -> "TemporalFusionTransformerForecaster":
        """Train with Lightning (val-loss early stopping + best checkpoint)."""
        from lightning.pytorch import seed_everything
        from pytorch_forecasting import TemporalFusionTransformer

        df = validate_frame(coerce_frame(X, y))
        if self.random_state is not None:
            seed_everything(self.random_state, workers=True)
        parts = split_datasets(
            df,
            encoder_length=self.encoder_length,
            prediction_length=self.prediction_length,
            val_size=self.val_size,
            batch_size=self.batch_size,
        )
        self._training = parts["training"]
        self._tft = TemporalFusionTransformer.from_dataset(
            self._training, **self._tft_kwargs()
        )
        ckpt_dir = self.checkpoint_dir or Path(tempfile.mkdtemp(prefix="tft-ckpt-"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer = self._trainer(ckpt_dir)
        logger.info(
            "Fitting TFT (encoder=%d, horizon=%d) on %d windows",
            self.encoder_length,
            self.prediction_length,
            len(self._training),
        )
        trainer.fit(self._tft, parts["train_loader"], parts["val_loader"])
        val_loss = trainer.callback_metrics.get("val_loss")
        self.best_val_loss_ = (
            float(val_loss.cpu().item()) if val_loss is not None else None
        )
        self.n_epochs_ = int(trainer.current_epoch)
        ckpts = sorted(ckpt_dir.glob("tft-best*.ckpt"))
        if ckpts:
            self.best_checkpoint_ = ckpts[0]
            restored = TemporalFusionTransformer.load_from_checkpoint(ckpts[0])
            self._tft.load_state_dict(restored.state_dict())
        self._tft.eval()
        self._is_fitted = True
        logger.info(
            "Fitted TFT in %d epochs (best val loss=%s)",
            self.n_epochs_,
            self.best_val_loss_,
        )
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Median forecasts for the trailing window of each group.

        DataFrames may carry NaN targets in the trailing horizon (unknown
        future). Plain arrays without target history cannot be encoded —
        use :meth:`forecast` with ``y`` instead.
        """
        df = X if isinstance(X, pd.DataFrame) else frame_from_arrays(X)
        preds, _ = self._raw_predict(df)
        return self._median(preds)

    def forecast(
        self, X: Union[pd.DataFrame, np.ndarray], y=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast the trailing horizon per group; return (preds, actuals).

        The last ``prediction_length`` targets per group are masked as the
        unknown future and scored against the held actuals. Shapes are
        ``(n_groups, horizon)`` (1-D when the horizon is 1).
        """
        self._check_fitted()
        df = validate_frame(coerce_frame(X, y))
        masked = df.copy()
        idx = masked.groupby(GROUP_ID, sort=True).tail(self.prediction_length).index
        masked.loc[idx, TARGET] = np.nan
        preds, _ = self._raw_predict(masked)
        med = self._median(preds)
        acts = np.stack(
            [
                g[TARGET].to_numpy(dtype=float)[-self.prediction_length :]
                for _, g in df.groupby(GROUP_ID, sort=True)
            ]
        )
        if self.prediction_length == 1:
            acts = np.asarray(acts).ravel()
        return med, np.asarray(acts)

    def predict_interval(
        self, X: Union[pd.DataFrame, np.ndarray], quantiles=None
    ) -> dict:
        """Quantile forecasts keyed by level, each (n, horizon)."""
        want = [float(q) for q in (quantiles or self.quantiles)]
        missing = [q for q in want if q not in self.quantiles]
        if missing:
            raise ValueError(f"Quantiles not fitted: {missing}")
        df = X if isinstance(X, pd.DataFrame) else frame_from_arrays(X)
        preds, _ = self._raw_predict(df)
        arr = preds.cpu().numpy()
        out = {}
        for q in want:
            idx = list(self.quantiles).index(q)
            out[q] = np.asarray(arr[..., idx])
        return out

    def evaluate(
        self, X: Union[pd.DataFrame, np.ndarray], y=None, epsilon: float = 1e-8
    ) -> dict:
        """Score trailing-horizon median forecasts (see :meth:`forecast`)."""
        preds, actuals = self.forecast(X, y)
        return evaluate_regression(
            np.ravel(np.asarray(actuals, dtype=float)),
            np.ravel(np.asarray(preds, dtype=float)),
            epsilon,
        )

    # ------------------------------------------------------------------
    # Explainability (real fitted-model outputs only)
    # ------------------------------------------------------------------
    def interpret(self, X: Union[pd.DataFrame, np.ndarray]) -> dict:
        """Attention weights + variable importances from the fitted TFT."""
        self._check_fitted()
        assert self._tft is not None and self._training is not None
        df = X if isinstance(X, pd.DataFrame) else frame_from_arrays(X)
        df = validate_frame(df, allow_nan_target=True)
        loader, _ = predict_loader(
            self._training, df, self.batch_size, self.prediction_length
        )
        raw = self._tft.predict(loader, mode="raw", return_x=False)
        interp = self._tft.interpret_output(raw, reduction="sum")
        return {
            "attention": np.asarray(interp["attention"], dtype=float),
            "encoder_variables": self._named_importance(
                interp["encoder_variables"], self._tft.encoder_variables
            ),
            "decoder_variables": self._named_importance(
                interp["decoder_variables"], self._tft.decoder_variables
            ),
            "static_variables": self._named_importance(
                interp["static_variables"], self._tft.static_variables
            ),
        }

    @staticmethod
    def _named_importance(values, names: list[str]) -> dict:
        """Map an importance tensor onto variable names (fallback: v0..)."""
        arr = np.asarray(values, dtype=float).ravel()
        if len(arr) != len(names):
            names = [f"v{i}" for i in range(len(arr))]
        return {name: float(v) for name, v in zip(names, arr)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _to_payload(self) -> dict:
        """Serialisable checkpoint dict (also reused by hybrid models)."""
        self._check_fitted()
        assert self._tft is not None
        return {
            "forecaster": type(self).__name__,
            "config": dict(self._config),
            "state_dict": {k: v.cpu() for k, v in self._tft.state_dict().items()},
            "training_dataset": self._training,
        }

    @classmethod
    def _from_payload(cls, payload: dict) -> "TemporalFusionTransformerForecaster":
        """Rebuild a forecaster from :meth:`_to_payload` output."""
        from pytorch_forecasting import TemporalFusionTransformer

        if payload.get("forecaster") != cls.__name__:
            raise TypeError(
                f"Expected {cls.__name__} payload, got {payload.get('forecaster')}"
            )
        obj = cls(**payload["config"])
        obj._training = payload["training_dataset"]
        obj._tft = TemporalFusionTransformer.from_dataset(
            obj._training, **obj._tft_kwargs()
        )
        obj._tft.load_state_dict(payload["state_dict"])
        obj._tft.eval()
        obj._is_fitted = True
        return obj

    def _median_for_loader(self, loader) -> torch.Tensor:
        """Median quantile predictions for a forecasting dataloader."""
        self._check_fitted()
        assert self._tft is not None
        raw = self._tft.predict(loader, mode="raw", return_x=False)
        q = torch.as_tensor(np.asarray(raw.prediction, dtype=np.float32))
        return q[..., list(self.quantiles).index(0.5)]

    def save(self, path: str | Path) -> Path:
        """Save checkpoint (weights + training dataset + config)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._to_payload(), path)
        logger.info("Saved %s checkpoint to %s", type(self).__name__, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TemporalFusionTransformerForecaster":
        """Load a checkpoint saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls._from_payload(payload)
        logger.info("Loaded %s checkpoint from %s", cls.__name__, path)
        return obj
