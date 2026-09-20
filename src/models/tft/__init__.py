"""Temporal Fusion Transformer package (Phase 4A)."""

from src.models.tft.dataset import (
    build_synthetic_tft_frame,
    coerce_frame,
    fill_decoder_placeholders,
    frame_from_arrays,
    resolve_roles,
    split_datasets,
    validate_frame,
)
from src.models.tft.tft_model import TemporalFusionTransformerForecaster

__all__ = [
    "TemporalFusionTransformerForecaster",
    "build_synthetic_tft_frame",
    "fill_decoder_placeholders",
    "frame_from_arrays",
    "coerce_frame",
    "resolve_roles",
    "split_datasets",
    "validate_frame",
]
