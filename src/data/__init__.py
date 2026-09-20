"""Real-data pipeline package (Phase 6A).

CSV -> validation -> cleaning -> features -> chrono splits -> adapters.
No model training here. Never shuffles, never fits on val/test.
"""

from src.data.adapters import sequenced_ready, tabular_ready, tft_ready
from src.data.features import add_features
from src.data.loader import load_raw_csv
from src.data.preprocessing import clean
from src.data.schema import DataMapping, load_data_config
from src.data.validation import validate

__all__ = [
    "DataMapping",
    "load_data_config",
    "load_raw_csv",
    "validate",
    "clean",
    "add_features",
    "tabular_ready",
    "sequenced_ready",
    "tft_ready",
]
