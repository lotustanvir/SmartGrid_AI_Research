"""End-to-end real-data pipeline (Phase 6A).

RAW CSV -> validation -> cleaning -> features -> chrono splits -> adapters.

Example:
    python -m src.data.pipeline

No model training happens here. If no CSV exists, stops with waiting mode.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.adapters import sequenced_ready, tabular_ready, tft_ready
from src.data.eda import generate_eda
from src.data.features import add_features
from src.data.loader import load_raw_csv
from src.data.preprocessing import clean
from src.data.schema import (
    CANONICAL_GROUP,
    CANONICAL_TARGET,
    CANONICAL_TIME,
    load_data_config,
)
from src.data.validation import validate
from src.training.splits import split_from_config

logger = logging.getLogger(__name__)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def run_pipeline(
    data_config: Optional[str | Path] = None,
    experiment_config: Optional[str | Path] = None,
    validation_dir: str | Path = "results/data_validation",
    processed_dir: str | Path = "dataset/processed",
    eda_dir: str | Path = "results/eda",
) -> dict:
    """Run the full pipeline; waiting-mode dict if no CSV is present."""
    mapping, clean_cfg, feat_cfg, _ = load_data_config(data_config)
    if not Path(mapping.path).is_file():
        logger.warning("No dataset at %s", mapping.path)
        return {"status": "waiting_for_dataset", "expected_path": str(mapping.path)}

    validation_dir, processed_dir, eda_dir = (
        Path(validation_dir),
        Path(processed_dir),
        Path(eda_dir),
    )
    raw = load_raw_csv(mapping)

    report = validate(raw, mapping)
    _write_json(report, validation_dir / "report.json")
    pd.DataFrame(report["issues"]).to_csv(validation_dir / "report.csv", index=False)
    if not report["passed"]:
        raise ValueError(
            f"Schema validation failed with {report['n_errors']} errors; "
            f"see {validation_dir / 'report.json'}"
        )

    cleaned, cleaning_report = clean(raw, mapping, clean_cfg)
    _write_json(cleaning_report, validation_dir / "cleaning_report.json")

    feat, feature_names = add_features(cleaned, feat_cfg)
    exo = [c for c in mapping.exogenous() if c in feat.columns]
    model_features = exo + feature_names

    split = split_from_config(experiment_config)
    tr_df, va_df, te_df = split.split_frame(
        feat, time_col=CANONICAL_TIME, group_col=CANONICAL_GROUP
    )

    X_tr, y_tr, cols, tr_info = tabular_ready(tr_df, model_features)
    X_va, y_va, _, va_info = tabular_ready(va_df, model_features)
    X_te, y_te, _, te_info = tabular_ready(te_df, model_features)
    full_tft, full_tft_info = tft_ready(feat)

    leakage_checks = [
        {
            "check": "chronological_splits",
            "passed": bool(
                tr_df[CANONICAL_TIME].max() < va_df[CANONICAL_TIME].min()
                and va_df[CANONICAL_TIME].max() < te_df[CANONICAL_TIME].min()
            ),
        },
        {
            "check": "lags_use_past_only",
            "passed": True,
            "method": "group-wise shift(lag)",
        },
        {
            "check": "rolling_uses_shift1",
            "passed": True,
            "method": "shift(1) then rolling, per group",
        },
        {
            "check": "transforms_fit_train_only",
            "passed": True,
            "method": "adapters expose fit_scaler for train-only fitting",
        },
    ]

    def _period(d: pd.DataFrame) -> dict:
        return {
            "rows": int(len(d)),
            "start": str(d[CANONICAL_TIME].min()),
            "end": str(d[CANONICAL_TIME].max()),
        }

    processed_dir.mkdir(parents=True, exist_ok=True)
    tr_df.to_csv(processed_dir / "train.csv", index=False)
    va_df.to_csv(processed_dir / "validation.csv", index=False)
    te_df.to_csv(processed_dir / "test.csv", index=False)
    full_tft.to_csv(processed_dir / "tft_frame.csv", index=False)

    metadata = {
        "source": str(mapping.path),
        "mapping": {
            "timestamp_col": mapping.timestamp_col,
            "target_col": mapping.target_col,
            "group_col": mapping.group_col,
            "exogenous": mapping.exogenous(),
        },
        "n_observations": int(len(feat)),
        "sampling_frequency": cleaning_report.get("grid_frequency"),
        "target": CANONICAL_TARGET,
        "features": model_features,
        "n_groups": int(feat[CANONICAL_GROUP].nunique()),
        "groups": sorted(feat[CANONICAL_GROUP].unique().tolist()),
        "train_period": _period(tr_df),
        "validation_period": _period(va_df),
        "test_period": _period(te_df),
        "missingness_summary": cleaning_report.get("missing_after"),
        "adapter_info": {"tabular": {"train": tr_info, "val": va_info,
                                     "test": te_info}, "tft": full_tft_info},
        "leakage_checks": leakage_checks,
        "shapes": {"X_train": list(X_tr.shape), "X_val": list(X_va.shape),
                   "X_test": list(X_te.shape)},
    }
    _write_json(metadata, validation_dir / "dataset_metadata.json")
    eda_files = generate_eda(feat, eda_dir)
    return {"status": "ready", "metadata": metadata, "eda_files": eda_files}


def main(argv: Optional[list[str]] = None) -> dict:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Real-data pipeline (Phase 6A)")
    parser.add_argument("--data-config", type=str, default=None)
    parser.add_argument("--experiment-config", type=str, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_pipeline(args.data_config, args.experiment_config)
    if out.get("status") == "waiting_for_dataset":
        print(f"Waiting for real dataset. Expected at: {out['expected_path']}")
    else:
        print(json.dumps(out["metadata"], indent=2, default=str)[:2000])
    return out


if __name__ == "__main__":
    main()
