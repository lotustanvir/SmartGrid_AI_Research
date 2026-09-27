"""Canonical Q1 protocol runner (Phase 2, SMOKE / NON-RESULT ONLY).

Builds the Q1 pipeline on the canonical V2 dataset WITHOUT running the
full 11-model x H1/H6/H24 x 3-seed experiment:

  V2 CSV -> calendar split -> canonical lagged-weather features ->
  origin registry (H=1/6/24) -> provenance + registry SMOKE record.

Produces NO headline metrics, ranks NO models, declares NO best model.
Full Q1 experiments are BLOCKED until re-audit passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.data.features_q1 import (
    Q1_FEATURE_VERSION,
    build_q1_features,
    canonical_feature_names,
    drop_warmup,
    feature_schema_hash,
)
from src.data.origins import generate_origins, origins_hash, save_origins
from src.data.provenance import (
    cuda_info,
    feature_schema_hash as _unused,  # noqa: F401 (explicit re-export guard)
    git_commit,
    package_versions,
    python_version,
    sha256_file,
)
from src.data.splits_q1 import split_calendar
from src.q1.protocol import (
    HORIZONS,
    ORIGIN_ARTIFACT,
    PROTOCOL_VERSION,
    Q1_DATASET_PATH,
    Q1_DATASET_VERSION,
    Q1_OOF_FOLDS,
    Q1_SEEDS,
)
from src.utils.registry import record_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("q1_protocol")


def run_q1_smoke(
    n_rows: int | None = None,
    origin_artifact: str | Path = ORIGIN_ARTIFACT,
    registry_path: str | Path = "experiments/REGISTRY.csv",
) -> dict:
    """SMOKE pipeline: provenance + features + splits + origins + registry."""
    t0 = time.time()
    raw = pd.read_csv(Q1_DATASET_PATH)
    if n_rows is not None:
        raw = raw.iloc[: int(n_rows)].reset_index(drop=True)
    dataset_sha = sha256_file(Q1_DATASET_PATH)
    feat, feature_names = build_q1_features(raw)
    assert feature_names == canonical_feature_names(), "Schema drift."
    clean, info = drop_warmup(feat, feature_names)
    tr, va, te = split_calendar(clean)
    origins = generate_origins(te, horizons=tuple(HORIZONS))
    o_hash = origins_hash(origins)
    # Fix 5: store the ACTUAL written artifact path (parquet preferred,
    # CSV fallback when no parquet engine). Never record a path that was
    # not written.
    actual_artifact = str(save_origins(origins, origin_artifact))
    commit = git_commit(".")
    exp_id = f"Q1_SMOKE_{PROTOCOL_VERSION}_nonresult"
    record_experiment(
        exp_id=exp_id,
        status="COMPLETED",
        git_commit=commit,
        model="SMOKE",
        horizon=0,
        seed=Q1_SEEDS[0],
        dataset_version=Q1_DATASET_VERSION,
        dataset_sha256=dataset_sha,
        features=Q1_FEATURE_VERSION,
        feature_schema_hash=feature_schema_hash(),
        origin_set_hash=o_hash,
        config_snapshot="configs/pjm_data_v2.yaml;configs/experiment.yaml;configs/models.yaml",
        python_version=python_version(),
        packages=json.dumps(
            package_versions(["numpy", "pandas", "sklearn", "torch", "xgboost", "lightgbm"])
        ),
        cuda=json.dumps(cuda_info()),
        checkpoint="",
        checkpoint_sha256="",
        predictions=actual_artifact,
        metrics="",
        runtime_s=time.time() - t0,
        smoke=True,
        notes=(
            f"SMOKE/NON-RESULT protocol dry-run {PROTOCOL_VERSION}; "
            f"rows={len(raw)} origins={len(origins)} oof_folds={Q1_OOF_FOLDS}; "
            "NOT headline results; models NOT ranked."
        ),
        path=Path(registry_path),
    )
    out = {
        "status": "SMOKE_COMPLETED_NON_RESULT",
        "protocol": PROTOCOL_VERSION,
        "dataset_sha256": dataset_sha,
        "feature_version": Q1_FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "origin_set_hash": o_hash,
        "n_origins": int(len(origins)),
        "origins_per_horizon": {
            int(h): int((origins["horizon"] == h).sum()) for h in HORIZONS
        },
        "split_rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "warmup": info,
        "git_commit": commit,
        "runtime_s": round(time.time() - t0, 2),
    }
    logger.info("Q1 SMOKE (NON-RESULT): %s", json.dumps(out, default=str)[:1500])
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Q1 protocol SMOKE (non-result).")
    p.add_argument("--n-rows", type=int, default=None)
    p.add_argument("--origin-artifact", type=str, default=ORIGIN_ARTIFACT)
    p.add_argument("--registry", type=str, default="experiments/REGISTRY.csv")
    args = p.parse_args(argv)
    return run_q1_smoke(
        n_rows=args.n_rows,
        origin_artifact=args.origin_artifact,
        registry_path=args.registry,
    )


if __name__ == "__main__":
    main()
