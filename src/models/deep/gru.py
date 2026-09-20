"""GRU forecaster (deep sequence model)."""

from __future__ import annotations

import logging

import torch.nn as nn

from src.models.deep.sequence import SequenceForecasterBase

logger = logging.getLogger(__name__)


class _GRUNet(nn.Module):
    """Single-head GRU: last hidden state -> horizon outputs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class GRUForecaster(SequenceForecasterBase):
    """GRU behind the common forecaster interface (see base for args)."""

    _config_name = "gru"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.dropout > 0.0 and self.num_layers == 1:
            logger.warning("dropout has no effect with num_layers=1; ignored.")

    def build_module(self, input_size: int) -> nn.Module:
        """Build the GRU network for ``input_size`` features per step."""
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size!r}")
        return _GRUNet(
            input_size, self.hidden_size, self.num_layers, self.dropout, self.output_size
        )
