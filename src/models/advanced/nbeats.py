"""N-BEATS forecaster (Phase 2 Q1, torch-native).

Transparent reproducible N-BEATS-style implementation: a stack of fully
connected blocks operating on the flattened 168-step history, each block
producing a backcast (subtracted from the residual stream) and a partial
forecast (summed). No external dependency beyond torch. Q1: seq_len=168,
output_size in {1,6,24}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.models.deep.sequence import SequenceForecasterBase


class _NBEATSBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, out_len: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden, in_dim)
        self.forecast = nn.Linear(hidden, out_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.backcast(h), self.forecast(h)


class _NBEATSNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        output_size: int,
        hidden_size: int = 128,
        n_blocks: int = 3,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        in_dim = seq_len * n_features
        self.blocks = nn.ModuleList(
            [_NBEATSBlock(in_dim, hidden_size, n_layers, output_size) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.reshape(x.size(0), -1)
        forecast = 0.0
        for blk in self.blocks:
            backcast, part = blk(residual)
            residual = residual - backcast
            forecast = forecast + part
        assert isinstance(forecast, torch.Tensor)
        return forecast


class NBEATSForecaster(SequenceForecasterBase):
    """N-BEATS behind the common sequence interface."""

    _config_name = "nbeats"

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        output_size: int = 1,
        seq_len: int = 168,
        n_blocks: int = 3,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        min_delta: float = 0.0,
        val_fraction: float = 0.15,
        scale: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = 42,
        n_layers: Optional[int] = None,
    ) -> None:
        if n_layers is not None:
            num_layers = int(n_layers)
        self.n_blocks = n_blocks
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            seq_len=seq_len,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            min_delta=min_delta,
            val_fraction=val_fraction,
            scale=scale,
            verbose=verbose,
            random_state=random_state,
        )
        self._config.update({"n_blocks": n_blocks})

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "NBEATSForecaster":
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    def build_module(self, n_features: int) -> nn.Module:
        return _NBEATSNet(
            n_features=n_features,
            seq_len=self.seq_len,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            n_blocks=self.n_blocks,
            n_layers=self.num_layers,
        )
