"""PatchTST forecaster (Phase 2 Q1, torch-native).

Transparent patch-based Transformer: non-overlapping (or strided) patches
of the 168-step history are linearly embedded, processed by a
TransformerEncoder, and mapped to H outputs. No external time-series
dependency beyond torch (already pinned). Q1: seq_len=168,
output_size in {1,6,24}, seeds {42,123,2025}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.models.deep.sequence import SequenceForecasterBase


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        output_size: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0 or patch_len > seq_len:
            raise ValueError("Invalid patch_len/stride.")
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * self.n_patches, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> patches (B, P, patch_len*F)
        patches = [
            x[:, i : i + self.patch_len, :].reshape(x.size(0), -1)
            for i in range(0, x.size(1) - self.patch_len + 1, self.stride)
        ]
        p = torch.stack(patches, dim=1)
        h = self.embed(p)
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h.reshape(h.size(0), -1))


class PatchTSTForecaster(SequenceForecasterBase):
    """PatchTST behind the common sequence interface."""

    _config_name = "patchtst"

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        seq_len: int = 168,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
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
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
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
        self._config.update(
            {
                "patch_len": patch_len,
                "stride": stride,
                "d_model": d_model,
                "n_heads": n_heads,
            }
        )

    @classmethod
    def from_config(
        cls, path: Optional[str | Path] = None, **overrides
    ) -> "PatchTSTForecaster":
        from src.models.baseline import load_params

        params = load_params(cls._config_name, path)
        params.update(overrides)
        return cls(**params)

    def build_module(self, n_features: int) -> nn.Module:
        return _PatchTSTNet(
            n_features=n_features,
            seq_len=self.seq_len,
            output_size=self.output_size,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            dropout=self.dropout,
        )
