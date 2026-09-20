"""Shared sequence-learning plumbing for deep forecasters (Phase 3A).

Forecasters accept either prepared 3-D sequences ``(n, seq_len, features)``
or 2-D time-ordered frames ``(time, features)`` (windowed internally with
``seq_len`` / ``output_size`` as horizon). They never depend on any
dataset-specific column layout.
"""

from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import evaluate_regression
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


def detect_device() -> torch.device:
    """Return CUDA device when available, else CPU (and log the choice)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Deep models using device: %s", device)
    return device


class SequenceDataset(Dataset):
    """Torch dataset wrapping ``(n, seq_len, features)`` + targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (n, seq_len, features), got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}"
            )
        if X.size == 0:
            raise ValueError("Empty input arrays.")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values detected in inputs.")
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y)
        if y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_windows(
    X: ArrayLike, y: ArrayLike, seq_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows from time-ordered 2-D features + 1-D target.

    Window ``i`` uses ``X[i:i+seq_len]`` to predict
    ``y[i+seq_len:i+seq_len+horizon]`` (no future leakage by construction).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D for windowing, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different lengths.")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values detected in inputs.")
    n_win = X.shape[0] - seq_len - horizon + 1
    if n_win <= 0:
        raise ValueError(
            f"Not enough rows ({X.shape[0]}) for seq_len={seq_len} "
            f"+ horizon={horizon}."
        )
    Xw = np.stack([X[i : i + seq_len] for i in range(n_win)])
    yw = np.stack([y[i + seq_len : i + seq_len + horizon] for i in range(n_win)])
    if horizon == 1:
        yw = yw[:, 0]
    return Xw, yw


class SequenceForecasterBase(BaseForecaster):
    """Base class for recurrent forecasters (LSTM / GRU).

    Args:
        input_size: Features per time step (None = inferred on fit).
        hidden_size: Recurrent hidden dimension.
        num_layers: Stacked recurrent layers.
        dropout: Dropout between recurrent layers (0 disables).
        output_size: Forecast horizon (steps predicted per window).
        seq_len: Look-back window for 2-D inputs.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        lr: Adam learning rate.
        patience: Early-stopping patience on validation loss (epochs).
        min_delta: Minimum val-loss improvement to reset patience.
        val_fraction: Chronological tail fraction used for validation when
            no explicit validation set is passed to :meth:`fit`.
        scale: Standardise features (scaler fit on train only).
        verbose: Log per-epoch losses.
        random_state: Seed for reproducibility.
    """

    _config_name = "sequence"

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
        seq_len: int = 24,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        min_delta: float = 0.0,
        val_fraction: float = 0.15,
        scale: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if output_size <= 0 or seq_len <= 0 or batch_size <= 0 or epochs <= 0:
            raise ValueError("output_size, seq_len, batch_size, epochs > 0 required.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.val_fraction = val_fraction
        self.scale = scale
        self.verbose = verbose
        self.device = detect_device()
        self._torch_model: Optional[nn.Module] = None
        self.scaler_: Optional[StandardScaler] = None
        self.target_scaler_: Optional[StandardScaler] = None
        self.input_size_: Optional[int] = None
        self.history_: dict = {"train_loss": [], "val_loss": []}
        self.n_epochs_: int = 0
        self.best_val_loss_: Optional[float] = None
        self._config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "output_size": output_size,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "min_delta": min_delta,
            "val_fraction": val_fraction,
            "scale": scale,
            "verbose": verbose,
            "random_state": random_state,
        }

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "SequenceForecasterBase":
        """Build from ``configs/models.yaml`` (section ``cls._config_name``)."""
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    @abstractmethod
    def build_module(self, input_size: int) -> nn.Module:
        """Build the recurrent network mapping (B, seq, F) -> (B, horizon)."""

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def _to_windows(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            yarr = np.asarray(y, dtype=float)
            if yarr.shape[0] != X.shape[0]:
                raise ValueError("X and y have different sample counts.")
            if np.isnan(X).any() or np.isnan(yarr).any():
                raise ValueError("NaN values detected in inputs.")
            if X.shape[1] != self.seq_len:
                raise ValueError(
                    f"Expected seq_len={self.seq_len}, got {X.shape[1]}"
                )
            if yarr.ndim == 2 and yarr.shape[1] != self.output_size:
                raise ValueError(
                    f"Expected horizon={self.output_size}, got {yarr.shape[1]}"
                )
            return X, yarr
        if X.ndim == 2:
            return build_windows(X, y, self.seq_len, self.output_size)
        raise ValueError(f"X must be 2-D or 3-D, got shape {X.shape}")

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler().fit(X.reshape(-1, X.shape[2]))
        return self._scale_apply(X)

    def _scale_apply(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler_ is not None
        shape = X.shape
        return self.scaler_.transform(X.reshape(-1, shape[2])).reshape(shape)

    def _target_scale_fit(self, y: np.ndarray) -> np.ndarray:
        """Fit target scaler on training y and return scaled y."""
        y_flat = y.reshape(-1, 1) if y.ndim == 1 else y.reshape(-1, y.shape[-1])
        self.target_scaler_ = StandardScaler().fit(y_flat)
        return self._target_scale_apply(y)

    def _target_scale_apply(self, y: np.ndarray) -> np.ndarray:
        """Transform y using the fitted target scaler."""
        assert self.target_scaler_ is not None
        orig_shape = y.shape
        y_flat = y.reshape(-1, 1) if y.ndim == 1 else y.reshape(-1, y.shape[-1])
        scaled = self.target_scaler_.transform(y_flat)
        return scaled.reshape(orig_shape)

    def _target_inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse-transform predictions from scaled space back to original."""
        if self.target_scaler_ is None:
            return y
        orig_shape = y.shape
        y_flat = y.reshape(-1, 1) if y.ndim == 1 else y.reshape(-1, y.shape[-1])
        inv = self.target_scaler_.inverse_transform(y_flat)
        return inv.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        **kwargs,
    ) -> "SequenceForecasterBase":
        """Train with validation loss + early stopping (best weights kept)."""
        Xw, yw = self._to_windows(X, y)
        n_feat = Xw.shape[2]
        if self.input_size is not None and self.input_size != n_feat:
            raise ValueError(
                f"input_size={self.input_size} but data has {n_feat} features."
            )
        self.input_size_ = n_feat
        if self.scale:
            Xw = self._scale_fit(Xw)

        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be provided together.")
            Xvw, yvw = self._to_windows(X_val, y_val)
            if Xvw.shape[2] != n_feat:
                raise ValueError("Train/validation feature mismatch.")
            if self.scale:
                Xvw = self._scale_apply(Xvw)
            Xtr, ytr = Xw, yw
        else:
            n_val = max(1, int(len(Xw) * self.val_fraction))
            if len(Xw) - n_val < 1:
                raise ValueError("Not enough windows for train/val split.")
            Xtr, ytr = Xw[:-n_val], yw[:-n_val]  # chronological tail = val
            Xvw, yvw = Xw[-n_val:], yw[-n_val:]

        ytr_scaled = self._target_scale_fit(ytr)
        yvw_scaled = self._target_scale_apply(yvw)

        if np.isnan(ytr_scaled).any() or np.isinf(ytr_scaled).any():
            raise ValueError("NaN/Inf in scaled training target.")
        if np.isnan(yvw_scaled).any() or np.isinf(yvw_scaled).any():
            raise ValueError("NaN/Inf in scaled validation target.")

        self._torch_model = self.build_module(n_feat).to(self.device)
        optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(
            SequenceDataset(Xtr, ytr_scaled),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            SequenceDataset(Xvw, yvw_scaled),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        best = float("inf")
        best_state: Optional[dict] = None
        bad = 0
        self.history_ = {"train_loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            self._torch_model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self._torch_model(xb)
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    logger.warning("NaN/Inf in model output at epoch %d, skipping batch", epoch + 1)
                    continue
                loss = loss_fn(pred, yb)
                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss at epoch %d, skipping batch", epoch + 1)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._torch_model.parameters(), max_norm=1.0
                )
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= max(1, len(train_loader.dataset))

            self._torch_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += loss_fn(self._torch_model(xb), yb).item() * len(xb)
            val_loss /= len(val_loader.dataset)

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            if self.verbose:
                logger.info("epoch %d: train=%.4f val=%.4f", epoch + 1, train_loss, val_loss)

            if val_loss < best - self.min_delta:
                best = val_loss
                best_state = copy.deepcopy(self._torch_model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        assert best_state is not None
        self._torch_model.load_state_dict(best_state)
        self._torch_model.eval()
        self.best_val_loss_ = best
        self.n_epochs_ = len(self.history_["train_loss"])
        self._is_fitted = True
        logger.info(
            "Fitted %s in %d epochs (best val loss=%.4f)",
            type(self).__name__,
            self.n_epochs_,
            best,
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Forecast per window; 1-D for horizon 1, else ``(n, horizon)``."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            if X.shape[1] != self.seq_len:
                raise ValueError(f"Expected seq_len={self.seq_len}, got {X.shape[1]}")
            if X.shape[2] != self.input_size_:
                raise ValueError(
                    f"Feature mismatch: expected {self.input_size_}, got {X.shape[2]}"
                )
            Xw = X
        elif X.ndim == 2:
            if X.shape[1] != self.input_size_:
                raise ValueError(
                    f"Feature mismatch: expected {self.input_size_}, got {X.shape[1]}"
                )
            n_win = X.shape[0] - self.seq_len - self.output_size + 1
            if n_win <= 0:
                raise ValueError("Not enough rows to build one window.")
            Xw = np.stack([X[i : i + self.seq_len] for i in range(n_win)])
        else:
            raise ValueError(f"X must be 2-D or 3-D, got shape {X.shape}")
        if np.isnan(Xw).any():
            raise ValueError("NaN values detected in inputs.")
        if self.scale:
            Xw = self._scale_apply(Xw)

        assert self._torch_model is not None
        loader = DataLoader(
            torch.as_tensor(Xw.astype(np.float32)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for xb in loader:
                outs.append(
                    self._torch_model(xb.to(self.device)).cpu().numpy()
                )
        preds = np.concatenate(outs, axis=0)
        preds = self._target_inverse_transform(preds)
        if self.output_size == 1:
            return preds.ravel()
        return preds

    def align_target(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Return the target values corresponding to :meth:`predict` output.

        With 2-D time-ordered input the first ``seq_len + horizon - 1`` rows
        cannot be forecast (history requirement); the tail is returned.
        With 3-D input ``y`` is returned as-is.
        """
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 2:
            _, y_aligned = build_windows(Xa, y, self.seq_len, self.output_size)
            y_aligned = np.asarray(y_aligned, dtype=float)
            return y_aligned.ravel() if self.output_size == 1 else y_aligned
        return np.asarray(y, dtype=float)

    def evaluate(self, X: ArrayLike, y: ArrayLike, epsilon: float = 1e-8) -> dict:
        """Score windows; 2-D inputs align ``y`` to the predictable tail."""
        preds = self.predict(X)
        return evaluate_regression(self.align_target(X, y), preds, epsilon=epsilon)

    # ------------------------------------------------------------------
    # Persistence (torch checkpoint incl. scaler + config)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save best-weights checkpoint (model + scaler + config)."""
        self._check_fitted()
        assert self._torch_model is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "forecaster": type(self).__name__,
                "config": self._config,
                "state_dict": {
                    k: v.cpu() for k, v in self._torch_model.state_dict().items()
                },
                "scaler": self.scaler_,
                "target_scaler": self.target_scaler_,
                "input_size": self.input_size_,
                "history": self.history_,
                "best_val_loss": self.best_val_loss_,
            },
            path,
        )
        logger.info("Saved %s checkpoint to %s", type(self).__name__, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SequenceForecasterBase":
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
        obj.input_size_ = payload["input_size"]
        obj.scaler_ = payload["scaler"]
        obj.target_scaler_ = payload.get("target_scaler")
        obj.history_ = payload.get("history", {"train_loss": [], "val_loss": []})
        obj.best_val_loss_ = payload.get("best_val_loss")
        obj._torch_model = obj.build_module(obj.input_size_).to(obj.device)
        obj._torch_model.load_state_dict(payload["state_dict"])
        obj._torch_model.eval()
        obj._is_fitted = True
        logger.info("Loaded %s checkpoint from %s", cls.__name__, path)
        return obj
