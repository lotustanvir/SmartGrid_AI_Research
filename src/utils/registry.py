"""Enforced experiment registry (Phase 2.1).

``experiments/REGISTRY.csv`` is machine-generated, never hand-edited.
Each record carries EXP-ID, status, git commit, dataset+SHA, feature
version+hash, origin-set hash, model, horizon, seed, config snapshot path,
environment, checkpoint/prediction/metric paths+hashes, runtime, status.

Statuses: PLANNED | RUNNING | COMPLETED | FAILED | SUPERSEDED.
SMOKE records must set smoke=True and are excluded from headline results.
Use :func:`require_q1_complete` to fail closed on incomplete non-smoke
COMPLETED Q1 records before Phase 3 headline use.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path("experiments/REGISTRY.csv")
HEADER = [
    "exp_id",
    "status",
    "git_commit",
    "dataset_version",
    "dataset_sha256",
    "features",
    "feature_schema_hash",
    "origin_set_hash",
    "model",
    "horizon",
    "seed",
    "config_snapshot",
    "python_version",
    "packages",
    "cuda",
    "checkpoint",
    "checkpoint_sha256",
    "predictions",
    "metrics",
    "runtime_s",
    "smoke",
    "notes",
]

STATUSES = ("PLANNED", "RUNNING", "COMPLETED", "FAILED", "SUPERSEDED")

# Deterministic Q1 models that legitimately produce no checkpoint artifact.
# Every other non-smoke COMPLETED Q1 record must carry checkpoint info.
Q1_NO_CHECKPOINT_MODELS = frozenset(
    {"persistence", "seasonal_persistence", "sarima", "SMOKE"}
)


def _ensure_header(path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
        return
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    # Migrate registries written before the git_commit column existed
    # (Phase 2.0 header without git_commit): preserve rows, fill "".
    if first.split(",") != HEADER:
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in HEADER})


def require_q1_complete(record: dict) -> dict:
    """Fail closed unless a non-smoke COMPLETED Q1 record is complete.

    Requires: valid EXP-ID, git commit, dataset SHA, feature version+hash,
    origin hash, Q1 headline model, horizon in {1,6,24}, seed (locked-list
    integer for stochastic models, "det" marker for deterministic models),
    config snapshot, prediction artifact, metrics content, checkpoint info
    (except models in ``Q1_NO_CHECKPOINT_MODELS``), status COMPLETED,
    smoke == False.

    Smoke records (``smoke`` true) must set smoke=True and are exempt
    from result-artifact requirements, but they can never satisfy this
    validator (headline use requires smoke == False).

    Raises ValueError on the first missing/invalid field. Returns the
    record unchanged when complete.
    """
    from src.q1.protocol import HORIZONS, Q1_MODELS

    def _truthy(value) -> bool:
        return value is not None and str(value).strip() not in ("", "None")

    smoke = str(record.get("smoke", "")).strip().lower() in ("true", "1", "yes")
    if smoke:
        raise ValueError("Smoke records cannot satisfy Q1 headline completeness.")
    if str(record.get("status", "")).strip() != "COMPLETED":
        raise ValueError(
            f"Q1 headline record must have status COMPLETED, "
            f"got {record.get('status')!r}."
        )
    for field in (
        "exp_id",
        "git_commit",
        "dataset_sha256",
        "features",
        "feature_schema_hash",
        "origin_set_hash",
        "model",
        "horizon",
        "seed",
        "config_snapshot",
        "predictions",
        "metrics",
    ):
        if not _truthy(record.get(field)):
            raise ValueError(f"Q1 headline record missing required field {field!r}.")
    if str(record.get("dataset_version", "")).strip() not in ("v2",):
        raise ValueError(
            f"Q1 headline dataset_version must be 'v2', "
            f"got {record.get('dataset_version')!r}."
        )
    if record.get("model") not in tuple(Q1_MODELS):
        raise ValueError(
            f"Q1 headline model {record.get('model')!r} not in locked set "
            f"{list(Q1_MODELS)}."
        )
    try:
        horizon = int(record.get("horizon"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"Q1 headline horizon invalid: {record.get('horizon')!r}.")
    if horizon not in tuple(HORIZONS):
        raise ValueError(f"Q1 headline horizon must be one of {tuple(HORIZONS)}.")
    # Seed rule (Phase 3): deterministic models carry the non-numeric
    # marker "det" (never a fake numeric seed); stochastic models carry
    # an integer from the locked seed list.
    from src.q1.protocol import Q1_SEEDS

    if record.get("model") in ("persistence", "seasonal_persistence", "sarima"):
        if str(record.get("seed")).strip() != "det":
            raise ValueError(
                "Q1 deterministic models must record seed 'det' "
                f"(no fake seed variation), got {record.get('seed')!r}."
            )
    else:
        try:
            seed_int = int(record.get("seed"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(f"Q1 headline seed invalid: {record.get('seed')!r}.")
        if seed_int not in tuple(Q1_SEEDS):
            raise ValueError(
                f"Q1 headline seed must be one of {tuple(Q1_SEEDS)}, "
                f"got {seed_int!r}."
            )
    if record.get("model") not in Q1_NO_CHECKPOINT_MODELS and not _truthy(
        record.get("checkpoint")
    ):
        raise ValueError(
            f"Q1 headline model {record.get('model')!r} must carry checkpoint "
            "information (deterministic no-checkpoint models are limited to "
            f"{sorted(Q1_NO_CHECKPOINT_MODELS)})."
        )
    return record


def record_experiment(
    exp_id: str,
    status: str,
    model: str,
    horizon: int,
    seed: int | str,
    git_commit: str = "",
    dataset_version: str = "v2",
    dataset_sha256: str = "",
    features: str = "Q1_V2_LAG1_001",
    feature_schema_hash: str = "",
    origin_set_hash: str = "",
    config_snapshot: str = "",
    python_version: str = "",
    packages: str = "",
    cuda: str = "",
    checkpoint: str = "",
    checkpoint_sha256: str = "",
    predictions: str = "",
    metrics: str = "",
    runtime_s: float = 0.0,
    smoke: bool = False,
    notes: str = "",
    path: Path = REGISTRY_PATH,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    _ensure_header(path)
    # Seed is stored verbatim: integer seeds for stochastic runs, the
    # non-numeric marker "det" for deterministic runs (never fabricated).
    try:
        seed_stored: int | str = int(seed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        seed_stored = str(seed)
    row = {
        "exp_id": exp_id,
        "status": status,
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "features": features,
        "feature_schema_hash": feature_schema_hash,
        "origin_set_hash": origin_set_hash,
        "model": model,
        "horizon": int(horizon),
        "seed": seed_stored,
        "config_snapshot": config_snapshot,
        "python_version": python_version,
        "packages": packages,
        "cuda": cuda,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "predictions": predictions,
        "metrics": metrics,
        "runtime_s": round(float(runtime_s), 3),
        "smoke": str(bool(smoke)),
        "notes": notes,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HEADER).writerow(row)
    return row
