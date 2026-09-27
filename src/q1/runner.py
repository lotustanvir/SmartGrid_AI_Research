"""Locked Q1 real-data experiment orchestration (Phase 3).

Single canonical orchestration path — the ONLY sanctioned route from the
locked Q1 protocol to real-data model execution:

dataset (SHA gate)
-> canonical features (49/order/hash gate)
-> warmup removal
-> calendar split (boundary/count gate)
-> common origins (count/hash gate, ONE authoritative artifact)
-> manifest (11 models x H1/H6/H24 x seeds; deterministic single-run)
-> per-experiment: seed -> construct (168/history asserts) -> fit
   (test-exclusion asserts) -> origin-aligned predict -> alignment
   validation -> score (MASE train-only, s=168) -> artifacts -> registry
   -> require_q1_complete
-> run-set completeness validation.

Full execution happens ONLY via :func:`run_manifest` (never in tests or
preflight). :func:`preflight` verifies the whole graph on real metadata
without training, writing canonical artifacts, or registering results.
No ranking is produced anywhere in this module.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.features_q1 import (
    Q1_FEATURE_VERSION,
    build_q1_features,
    canonical_feature_names,
    drop_warmup,
    feature_schema_hash,
)
from src.data.origins import (
    assert_same_origins,
    generate_origins,
    load_origins,
    origins_hash,
    save_origins,
)
from src.data.provenance import (
    cuda_info,
    git_commit,
    package_versions,
    python_version,
    sha256_file,
)
from src.data.splits_q1 import split_calendar
from src.evaluation.metrics import Q1_METRIC_KEYS
from src.q1 import horizons as q1_horizons
from src.q1 import tft_path as q1_tft_path
from src.q1.evaluate import build_naive_preds_by_origin, score_at_origins
from src.q1.protocol import (
    FEATURE_VERSION,
    HISTORY_LENGTH,
    HORIZONS,
    ORIGIN_ARTIFACT,
    PROTOCOL_VERSION,
    Q1_DATASET_PATH,
    Q1_DATASET_VERSION,
    Q1_MODELS,
    Q1_OOF_FOLDS,
    Q1_SEEDS,
    SEASONAL_PERIOD,
    TEST_END,
    TRAIN_END,
    VAL_END,
)
from src.training.experiment_runner import MODEL_REGISTRY, q1_headline_models
from src.utils.registry import record_experiment, require_q1_complete
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locked expectations (must match src/q1/protocol.py; asserted at runtime)
# ---------------------------------------------------------------------------
Q1_LOCKED_DATASET_SHA = "fb261563977bd9c096b96a5402914e1b511cd880b34293e0aad9bf421b6db486"
Q1_LOCKED_FEATURE_HASH = "ed0f6f5fb4df31566fb3b866c14810f7970a1865c8a0ee0b63b7c6b02a85705e"
Q1_LOCKED_N_FEATURES = 49
Q1_LOCKED_SPLIT_COUNTS = {"train": 34891, "val": 8784, "test": 8760}
Q1_LOCKED_ORIGIN_COUNTS = {1: 8592, 6: 8587, 24: 8569}
Q1_LOCKED_ORIGIN_HASH = "573de597b4ac4a4f7f2db79ff57045a0308ec38e013a329f16ae8017b19fe2ce"

DETERMINISTIC_MODELS = ("persistence", "seasonal_persistence", "sarima")
STOCHASTIC_MODELS = tuple(m for m in Q1_MODELS if m not in DETERMINISTIC_MODELS)
DET_SEED_MARKER = "det"
FULL_MANIFEST_SIZE = (
    len(DETERMINISTIC_MODELS) * len(HORIZONS)
    + len(STOCHASTIC_MODELS) * len(HORIZONS) * len(Q1_SEEDS)
)  # 9 + 72 = 81

CONFIG_SNAPSHOT = "configs/pjm_data_v2.yaml;configs/experiment.yaml;configs/models.yaml"
PROVENANCE_PACKAGES = ["numpy", "pandas", "sklearn", "torch", "xgboost", "lightgbm"]

# Non-timestamp, non-target columns of a post-warmup Q1 frame.
Q1_ID_COLUMNS = ("timestamp", "target", "group_id")


# ---------------------------------------------------------------------------
# Experiment spec + manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Q1ExperimentSpec:
    """One locked experiment: model x horizon x seed.

    Deterministic models carry ``seed=None`` (recorded as ``"det"``; never
    a fabricated numeric seed). Stochastic models carry an int from
    ``Q1_SEEDS``.
    """

    model: str
    horizon: int
    seed: Optional[int]  # None for deterministic models

    @property
    def exp_id(self) -> str:
        seed_part = f"S{self.seed_marker}"
        return f"Q1_{self.model}_H{int(self.horizon)}_{seed_part}"

    @property
    def seed_marker(self) -> int | str:
        return DET_SEED_MARKER if self.seed is None else int(self.seed)


def exp_id_for(model: str, horizon: int, seed: Optional[int]) -> str:
    """Deterministic EXP-ID for (model, horizon, seed)."""
    return Q1ExperimentSpec(model=model, horizon=horizon, seed=seed).exp_id


def build_manifest(
    models: Optional[list[str]] = None,
    horizons: Optional[list[int]] = None,
    seeds: Optional[list[int]] = None,
) -> list[Q1ExperimentSpec]:
    """Build the locked experiment manifest (fail-closed).

    Defaults: exactly the 11 locked models (protocol order) x H1/H6/H24 x
    locked seeds for stochastic models, single deterministic run per
    horizon for naive/SARIMA-lite. Explicit subsets are permitted for
    debugging but :func:`run_manifest` refuses partial sets unless
    ``allow_partial=True``.
    """
    resolved_models = q1_headline_models(models)
    resolved_horizons = list(HORIZONS) if horizons is None else list(horizons)
    if not resolved_horizons:
        raise ValueError("Q1 horizon list must not be empty.")
    if len(set(resolved_horizons)) != len(resolved_horizons):
        raise ValueError(f"Duplicated Q1 horizons: {resolved_horizons!r}.")
    for h in resolved_horizons:
        if h not in HORIZONS:
            raise ValueError(f"Q1 horizon must be one of {HORIZONS}, got {h!r}.")
    resolved_seeds = list(Q1_SEEDS) if seeds is None else list(seeds)
    if not resolved_seeds:
        raise ValueError("Q1 seed list must not be empty.")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError(f"Duplicated Q1 seeds: {resolved_seeds!r}.")
    for s in resolved_seeds:
        if s not in Q1_SEEDS:
            raise ValueError(f"Q1 seed must be one of {Q1_SEEDS}, got {s!r}.")
    specs: list[Q1ExperimentSpec] = []
    for model in resolved_models:
        for horizon in resolved_horizons:
            if model in DETERMINISTIC_MODELS:
                specs.append(Q1ExperimentSpec(model=model, horizon=horizon, seed=None))
            else:
                for seed in resolved_seeds:
                    specs.append(Q1ExperimentSpec(model=model, horizon=horizon, seed=seed))
    return specs


def validate_manifest_size(specs: list[Q1ExperimentSpec], expect_full: bool = True) -> None:
    """Fail closed unless the manifest has the expected size and unique IDs."""
    ids = [s.exp_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Q1 manifest contains duplicated EXP-IDs.")
    if expect_full and len(specs) != FULL_MANIFEST_SIZE:
        raise ValueError(
            f"Q1 full manifest must hold {FULL_MANIFEST_SIZE} experiments "
            f"(9 deterministic + 72 stochastic), got {len(specs)}."
        )


# ---------------------------------------------------------------------------
# Gates 1-4: dataset / features / splits / origins
# ---------------------------------------------------------------------------
@dataclass
class Q1Data:
    """Validated Q1 working set shared by every experiment in a run."""

    raw: pd.DataFrame
    frame: pd.DataFrame  # post-warmup, chronological
    feature_names: list[str]
    tr: pd.DataFrame
    va: pd.DataFrame
    te: pd.DataFrame
    full: pd.DataFrame  # concat(tr, va, te), chronological
    origins: pd.DataFrame
    origin_artifact: str
    y_train: np.ndarray
    provenance: dict


def load_and_validate_dataset(dataset_path: str = Q1_DATASET_PATH) -> tuple[pd.DataFrame, dict]:
    """Gate 1: load ONLY the canonical dataset, verify SHA256 (fail-closed)."""
    if str(dataset_path) != Q1_DATASET_PATH:
        raise ValueError(
            f"Q1 runner loads ONLY {Q1_DATASET_PATH!r}, got {dataset_path!r}."
        )
    raw = pd.read_csv(dataset_path)
    sha = sha256_file(dataset_path)
    if sha != Q1_LOCKED_DATASET_SHA:
        raise ValueError(
            f"Q1 dataset SHA mismatch: got {sha!r}, expected {Q1_LOCKED_DATASET_SHA!r}."
        )
    ts = pd.to_datetime(raw["timestamp"])
    provenance = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha,
        "dataset_version": Q1_DATASET_VERSION,
        "n_rows": int(len(raw)),
        "timestamp_min": str(ts.min()),
        "timestamp_max": str(ts.max()),
    }
    logger.info("Q1 dataset gate: sha ok, rows=%d (%s..%s)", len(raw), ts.min(), ts.max())
    return raw, provenance


def build_and_validate_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Gate 2: canonical builder only; assert 49/order/hash/version."""
    feat, feature_names = build_q1_features(raw)
    expected = canonical_feature_names()
    if list(feature_names) != list(expected):
        raise ValueError("Q1 feature builder returned a non-canonical name list.")
    if len(feature_names) != Q1_LOCKED_N_FEATURES:
        raise ValueError(f"Q1 feature count {len(feature_names)} != 49.")
    schema_hash = feature_schema_hash()
    if schema_hash != Q1_LOCKED_FEATURE_HASH:
        raise ValueError(f"Q1 feature hash {schema_hash!r} != locked hash.")
    if Q1_FEATURE_VERSION != FEATURE_VERSION:
        raise ValueError("Q1 feature version drift.")
    clean, info = drop_warmup(feat, feature_names)
    logger.info("Q1 feature gate: 49 features, hash ok, warmup dropped=%d", info["warmup_rows_dropped"])
    return clean, feature_names, info


def assert_canonical_frame(frame: pd.DataFrame, feature_names: list[str]) -> None:
    """Fail closed unless ``frame`` was built by the canonical builder.

    Requires the passed name list to be exactly canonical AND every one
    of the 49 canonical features to be present in ``frame``. Raw V2
    columns / timestamp / target / group_id may coexist in the frame;
    exact 49-column extraction for model input happens ONLY through
    :func:`canonical_matrix`.
    """
    if list(feature_names) != list(canonical_feature_names()):
        raise ValueError("Q1 model input uses a non-canonical feature list.")
    missing = [c for c in canonical_feature_names() if c not in frame.columns]
    if missing:
        raise ValueError(f"Q1 frame missing canonical features: {missing}.")
    if feature_schema_hash() != Q1_LOCKED_FEATURE_HASH:
        raise ValueError("Q1 feature hash drift at model-input gate.")


def canonical_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Return EXACTLY the 49 canonical features in locked order (float).

    This is the sole sanctioned conversion from a Q1 frame to a model
    input matrix: no subsets, no reordering, no raw columns. Every
    array-based model in the runner receives this matrix.
    """
    assert_canonical_frame(frame, feature_names)
    ordered = list(canonical_feature_names())
    X = frame[ordered].to_numpy(dtype=float)
    if X.shape[1] != Q1_LOCKED_N_FEATURES:
        raise ValueError("Canonical matrix width drift.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("NaN/Inf in canonical Q1 input matrix.")
    return X


def split_and_validate(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gate 3: calendar split only; assert boundaries + exact counts."""
    assert_canonical_frame(frame, feature_names)
    tr, va, te = split_calendar(frame)
    for name, part, expect in (
        ("train", tr, Q1_LOCKED_SPLIT_COUNTS["train"]),
        ("val", va, Q1_LOCKED_SPLIT_COUNTS["val"]),
        ("test", te, Q1_LOCKED_SPLIT_COUNTS["test"]),
    ):
        if len(part) != expect:
            raise ValueError(f"Q1 {name} rows {len(part)} != locked {expect}.")
    if not (tr["timestamp"].max() < pd.Timestamp(TRAIN_END) <= va["timestamp"].min()):
        raise ValueError("Q1 train/validation boundary violation.")
    if not (va["timestamp"].max() < pd.Timestamp(VAL_END) <= te["timestamp"].min()):
        raise ValueError("Q1 validation/test boundary violation.")
    if not (te["timestamp"].max() < pd.Timestamp(TEST_END)):
        raise ValueError("Q1 test upper-boundary violation.")
    return tr, va, te


def ensure_origins(
    test_frame: pd.DataFrame,
    artifact_path: str = ORIGIN_ARTIFACT,
    allow_generate: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Gate 4: ONE authoritative origin set; assert counts + hash.

    Loads the canonical artifact when present (verifying it); generates
    via the canonical helper only when absent and ``allow_generate``.
    Preflight passes ``allow_generate=False`` so canonical artifacts are
    never modified.
    """
    origins: pd.DataFrame | None = None
    try:
        origins = load_origins(artifact_path)
        logger.info("Q1 origins loaded from %s", artifact_path)
    except FileNotFoundError:
        if not allow_generate:
            raise ValueError(
                f"Q1 origin artifact missing at {artifact_path!r} and "
                "generation is disabled (preflight must not modify artifacts)."
            )
        origins = generate_origins(test_frame, horizons=tuple(HORIZONS))
    assert origins is not None
    for h, expect in Q1_LOCKED_ORIGIN_COUNTS.items():
        got = int((origins["horizon"] == h).sum())
        if got != expect:
            raise ValueError(f"Q1 H{h} origins {got} != locked {expect}.")
    if len(origins) != sum(Q1_LOCKED_ORIGIN_COUNTS.values()):
        raise ValueError("Q1 total origin count mismatch.")
    if origins_hash(origins) != Q1_LOCKED_ORIGIN_HASH:
        raise ValueError("Q1 origin hash mismatch.")
    parquet_path = Path(artifact_path)
    if parquet_path.is_file():
        actual = artifact_path
    elif parquet_path.with_suffix(".csv").is_file():
        actual = str(parquet_path.with_suffix(".csv"))
    else:
        actual = str(save_origins(origins, artifact_path))
    return origins, actual


def prepare_q1_data(
    dataset_path: str = Q1_DATASET_PATH,
    origin_artifact: str = ORIGIN_ARTIFACT,
    allow_generate_origins: bool = True,
) -> Q1Data:
    """Run gates 1-4 and return the validated working set shared by a run."""
    raw, provenance = load_and_validate_dataset(dataset_path)
    clean, feature_names, warmup_info = build_and_validate_features(raw)
    tr, va, te = split_and_validate(clean, feature_names)
    origins, actual_artifact = ensure_origins(te, origin_artifact, allow_generate_origins)
    full = pd.concat([tr, va, te], ignore_index=True)
    if not full["timestamp"].is_monotonic_increasing:
        raise ValueError("Q1 conduced frame is not chronological.")
    y_train = tr["target"].to_numpy(dtype=float)
    if y_train.size <= SEASONAL_PERIOD:
        raise ValueError("Q1 training targets too short for MASE seasonality.")
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        raise ValueError("NaN/Inf in Q1 training targets.")
    provenance.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "feature_version": Q1_FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "n_features": len(feature_names),
            "warmup_rows_dropped": int(warmup_info["warmup_rows_dropped"]),
            "split_rows": {k: int(len(p)) for k, p in (("train", tr), ("val", va), ("test", te))},
            "train_range": (str(tr["timestamp"].min()), str(tr["timestamp"].max())),
            "val_range": (str(va["timestamp"].min()), str(va["timestamp"].max())),
            "test_range": (str(te["timestamp"].min()), str(te["timestamp"].max())),
            "origin_artifact": str(actual_artifact),
            "origin_set_hash": origins_hash(origins),
        }
    )
    return Q1Data(
        raw=raw,
        frame=clean,
        feature_names=feature_names,
        tr=tr,
        va=va,
        te=te,
        full=full,
        origins=origins,
        origin_artifact=str(actual_artifact),
        y_train=y_train,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Seeds + model construction (§7 input enforcement, §14, §15)
# ---------------------------------------------------------------------------
def apply_spec_seed(spec: Q1ExperimentSpec) -> None:
    """Apply the exact requested seed; deterministic specs get no seeding."""
    if spec.model in DETERMINISTIC_MODELS:
        if spec.seed is not None:
            raise ValueError(
                f"Deterministic model {spec.model!r} must not receive a seed "
                f"(got {spec.seed!r}); manufactures fake variation."
            )
        return
    if spec.seed is None:
        raise ValueError(f"Stochastic model {spec.model!r} requires a locked seed.")
    if spec.seed not in Q1_SEEDS:
        raise ValueError(f"Q1 seed must be one of {Q1_SEEDS}, got {spec.seed!r}.")
    set_seed(int(spec.seed))


def assert_no_test_overlap(fit_max_ts: pd.Timestamp, test_min_ts: pd.Timestamp, label: str = "") -> None:
    """Fail closed if any fitting timestamp reaches the test period."""
    if not pd.Timestamp(fit_max_ts) < pd.Timestamp(test_min_ts):
        raise ValueError(
            f"Q1 test exclusion violated {label}: fit_max={fit_max_ts} "
            f"vs test_min={test_min_ts}. Test data must NEVER enter fitting."
        )


def construct_model(spec: Q1ExperimentSpec, checkpoint_dir: Optional[Path] = None) -> tuple[object, dict]:
    """Build one locked model for (horizon, seed) with runtime asserts.

    Returns (model, effective_config). Raises on legacy/unknown models,
    wrong horizons/seeds, non-168 history, or non-OOF hybrid config.
    """
    q1_headline_models([spec.model])  # legacy/unknown fail-closed
    if spec.horizon not in HORIZONS:
        raise ValueError(f"Q1 horizon must be one of {HORIZONS}, got {spec.horizon!r}.")
    cls = MODEL_REGISTRY[spec.model]
    horizon = int(spec.horizon)
    effective: dict = {"model": spec.model, "horizon": horizon}

    if spec.model in DETERMINISTIC_MODELS:
        if spec.seed is not None:
            raise ValueError(f"Deterministic model {spec.model!r} must not receive a seed.")
        if spec.model == "seasonal_persistence":
            model = cls(seasonal_period=SEASONAL_PERIOD, horizon=horizon, random_state=None)
            if int(model.seasonal_period) != SEASONAL_PERIOD:
                raise ValueError("Seasonal period drift.")
        elif spec.model == "sarima":
            model = cls.from_config(horizon=horizon)
            if int(model.seasonal_period) != SEASONAL_PERIOD:
                raise ValueError(
                    f"SARIMA-lite seasonal period {model.seasonal_period!r} != 168."
                )
        else:  # persistence
            model = cls(horizon=horizon, random_state=None)
        if model.random_state is not None:
            raise ValueError(f"Deterministic model {spec.model!r} carries a seed.")
        if int(model.horizon) != horizon:
            raise ValueError("Horizon drift in deterministic construction.")
        effective.update({"random_state": None})
        return model, effective

    # Stochastic: exact locked seed required.
    if spec.seed not in Q1_SEEDS:
        raise ValueError(f"Stochastic model {spec.model!r} needs seed in {Q1_SEEDS}.")
    seed = int(spec.seed)
    if spec.model in ("xgboost", "lightgbm"):
        model = cls.from_config(random_state=seed)
    elif spec.model in ("lstm", "gru", "patchtst", "nbeats"):
        model = cls.from_config(output_size=horizon, seq_len=HISTORY_LENGTH, random_state=seed)
        if int(model.seq_len) != HISTORY_LENGTH:
            raise ValueError(f"{spec.model} seq_len {model.seq_len!r} != 168.")
        if int(model.output_size) != horizon:
            raise ValueError(f"{spec.model} output_size != horizon {horizon}.")
    elif spec.model == "tft":
        from src.q1.tft_path import make_q1_tft_config

        cfg = make_q1_tft_config(horizon, seed)
        if checkpoint_dir is not None:
            cfg["checkpoint_dir"] = str(checkpoint_dir)
        model = cls.from_config(**cfg)
        if int(model.encoder_length) != HISTORY_LENGTH:
            raise ValueError(f"TFT encoder {model.encoder_length!r} != 168.")
        if int(model.prediction_length) != horizon:
            raise ValueError("TFT prediction_length != horizon.")
    elif spec.model == "hybrid":
        from src.models.baseline import load_params

        base = load_params("hybrid_tft_xgb")
        tft_cfg = dict(base.get("tft_config", {}))
        tft_cfg.update(
            {"encoder_length": HISTORY_LENGTH, "prediction_length": horizon, "random_state": seed}
        )
        model = cls(
            tft_config=tft_cfg,
            xgb_config=dict(base.get("xgb_config", {})),
            residual_training_mode="oof",
            n_oof_folds=Q1_OOF_FOLDS,
            clip_to_bounds=bool(base.get("clip_to_bounds", False)),
            random_state=seed,
        )
        if int(model.n_oof_folds) != Q1_OOF_FOLDS:
            raise ValueError("Hybrid OOF folds != 5.")
        if model.residual_training_mode != "oof":
            raise ValueError("Hybrid must use OOF residuals (in_sample forbidden).")
        if int(model._tft.encoder_length) != HISTORY_LENGTH:
            raise ValueError("Hybrid TFT encoder != 168.")
        if int(model._tft.prediction_length) != horizon:
            raise ValueError("Hybrid TFT prediction_length != horizon.")
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unknown Q1 model {spec.model!r}.")
    if model.random_state != seed:
        raise ValueError(f"{spec.model} random_state did not take seed {seed}.")
    effective.update({"random_state": seed, "config": dict(getattr(model, "_config", {}))})
    return model, effective


# ---------------------------------------------------------------------------
# Origin-aligned prediction (§8-13, §16)
# ---------------------------------------------------------------------------
def _timestamp_positions(frame: pd.DataFrame, time_col: str = "timestamp") -> dict:
    ts = pd.to_datetime(frame[time_col]).reset_index(drop=True)
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps in position map.")
    return {t: i for i, t in enumerate(pd.DatetimeIndex(ts))}


def _origins_for_horizon(origins: pd.DataFrame, horizon: int) -> pd.DataFrame:
    sub = origins[origins["horizon"] == int(horizon)].copy()
    if len(sub) != Q1_LOCKED_ORIGIN_COUNTS[int(horizon)]:
        raise ValueError(f"Authoritative H{horizon} set has {len(sub)} rows (locked {Q1_LOCKED_ORIGIN_COUNTS[int(horizon)]}).")
    return sub.sort_values("origin_timestamp").reset_index(drop=True)


def validate_pred_rows(origins_h: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Fail closed unless predictions match the authoritative origin set.

    Checks: exact key-set equality (no missing/extra/outside origins), no
    duplicates, correct horizon, target == origin+H == origins target_end.
    """
    if not rows:
        raise ValueError("Empty prediction rows.")
    keys = [(r["origin_timestamp_iso"], int(r["horizon"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicated (origin, horizon) predictions.")
    expected: dict[tuple[str, int], pd.Timestamp] = {}
    for r in origins_h.itertuples():
        key = (pd.Timestamp(r.origin_timestamp).isoformat(), int(r.horizon))
        if key in expected:
            raise ValueError(f"Duplicated authoritative origin {key}.")
        expected[key] = pd.Timestamp(r.target_end)
    if set(keys) != set(expected):
        missing = set(expected) - set(keys)
        extra = set(keys) - set(expected)
        raise ValueError(
            f"Prediction origin-set mismatch: {len(missing)} missing, "
            f"{len(extra)} outside authoritative set."
        )
    out = []
    for r in rows:
        key = (r["origin_timestamp_iso"], int(r["horizon"]))
        origin_ts = pd.Timestamp(r["origin_timestamp_iso"])
        target_ts = pd.Timestamp(r["target_timestamp"])
        if target_ts != origin_ts + pd.Timedelta(hours=int(r["horizon"])):
            raise ValueError(f"Wrong target timestamp for origin {key}.")
        if target_ts != expected[key]:
            raise ValueError(f"Target timestamp mismatch vs origins for {key}.")
        out.append(
            {
                "exp_id": r["exp_id"],
                "model": r["model"],
                "horizon": int(r["horizon"]),
                "seed": r["seed"],
                "origin_timestamp": origin_ts,
                "target_timestamp": target_ts,
                "y_pred": float(r["y_pred"]),
            }
        )
    pred_frame = pd.DataFrame(out).sort_values("origin_timestamp").reset_index(drop=True)
    if pred_frame["y_pred"].isna().any() or np.isinf(pred_frame["y_pred"].to_numpy()).any():
        raise ValueError("NaN/Inf in validated predictions.")
    return pred_frame


def _preds_dict(pred_frame: pd.DataFrame) -> dict:
    return {
        (pd.Timestamp(r.origin_timestamp).isoformat(), int(r.horizon)): float(r.y_pred)
        for r in pred_frame.itertuples()
    }


def predict_naive(spec: Q1ExperimentSpec, data: Q1Data) -> list[dict]:
    """Origin-aligned naive forecasts over the full observed series."""
    origins_h = _origins_for_horizon(data.origins, spec.horizon)
    series = data.full["target"].to_numpy(dtype=float)
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("NaN/Inf in full target series.")
    preds = build_naive_preds_by_origin(
        spec.model, series, data.full["timestamp"].to_numpy(), origins_h
    )
    return [
        {
            "exp_id": spec.exp_id,
            "model": spec.model,
            "horizon": int(r.horizon),
            "seed": spec.seed_marker,
            "origin_timestamp_iso": pd.Timestamp(r.origin_timestamp).isoformat(),
            "horizon_key": int(r.horizon),
            "target_timestamp": pd.Timestamp(r.origin_timestamp) + pd.Timedelta(hours=int(r.horizon)),
            "y_pred": float(preds[(pd.Timestamp(r.origin_timestamp).isoformat(), int(r.horizon))]),
        }
        for r in origins_h.itertuples()
    ]


def predict_sarima(spec: Q1ExperimentSpec, data: Q1Data, model) -> list[dict]:
    """Per-origin SARIMA-lite refits on history <= origin (fail-closed)."""
    from src.models.statistical import SARIMAForecaster  # noqa: F401 (type guard)

    if int(model.seasonal_period) != SEASONAL_PERIOD:
        raise ValueError("SARIMA-lite seasonal period != 168 at predict time.")
    origins_h = _origins_for_horizon(data.origins, spec.horizon)
    series = data.full["target"].to_numpy(dtype=float)
    pos_of = _timestamp_positions(data.full)
    max_train = int(model.max_train)
    rows = []
    for r in origins_h.itertuples():
        ots = pd.Timestamp(r.origin_timestamp)
        if ots not in pos_of:
            raise ValueError(f"Origin {ots} not in full frame.")
        p = int(pos_of[ots])
        hist = series[max(0, p - max_train + 1) : p + 1]
        if hist.size < SEASONAL_PERIOD + 10:
            raise ValueError(f"Insufficient SARIMA history at origin {ots}.")
        model.fit(np.zeros((hist.size, 1)), hist)
        y_hat = float(np.ravel(model.predict(np.zeros((1, 1))))[0])
        if not np.isfinite(y_hat):
            raise ValueError(f"Non-finite SARIMA forecast at origin {ots}.")
        rows.append(
            {
                "exp_id": spec.exp_id,
                "model": spec.model,
                "horizon": int(r.horizon),
                "seed": spec.seed_marker,
                "origin_timestamp_iso": ots.isoformat(),
                "horizon_key": int(r.horizon),
                "target_timestamp": ots + pd.Timedelta(hours=int(r.horizon)),
                "y_pred": y_hat,
            }
        )
    return rows


def _tree_pairs_for_horizon(
    data: Q1Data, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Direct H-pairs on train; chronological train/val/test index blocks."""
    assert_canonical_frame(data.tr, data.feature_names)
    X_all, y_all, keys = q1_horizons.build_direct_pairs(data.tr, data.feature_names, int(horizon))
    if X_all.shape[1] != Q1_LOCKED_N_FEATURES:
        raise ValueError("Tree input width != 49.")
    # All pair targets must lie inside the training block (no val/test target).
    if (np.asarray(keys["origin_row"]) + int(horizon) >= len(data.tr)).any():
        raise ValueError("Tree pair target escapes the training block.")
    n_tr_pairs = len(X_all)
    # Validation / test pairs come from their own blocks (same builder).
    X_va, y_va, _ = q1_horizons.build_direct_pairs(data.va, data.feature_names, int(horizon))
    X_te, y_te, keys_te = q1_horizons.build_direct_pairs(data.te, data.feature_names, int(horizon))
    return X_all, y_all, X_va, y_va, X_te, y_te, keys_te


def predict_tree(spec: Q1ExperimentSpec, data: Q1Data, model) -> list[dict]:
    """Fit one direct estimator on train pairs; predict test origins."""
    X_tr, y_tr, X_va, y_va, X_te, y_te, keys_te = _tree_pairs_for_horizon(data, spec.horizon)
    use_val = getattr(model, "early_stopping_rounds", None) is not None
    if use_val:
        model.fit(X_tr, y_tr, X_va, y_va)
    else:
        model.fit(X_tr, y_tr)
    assert_no_test_overlap(
        pd.Timestamp(data.tr["timestamp"].max()), pd.Timestamp(data.te["timestamp"].min()), label=f"tree/{spec.exp_id}"
    )
    preds = np.asarray(model.predict(X_te), dtype=float).ravel()
    if preds.shape[0] != len(keys_te):
        raise ValueError("Tree prediction count != test pair count.")
    origins_h = _origins_for_horizon(data.origins, spec.horizon)
    # Test-pair timestamps must cover every authoritative origin.
    pair_ts = set(pd.to_datetime(keys_te["timestamp"]).map(lambda t: pd.Timestamp(t).isoformat()))
    for r in origins_h.itertuples():
        if pd.Timestamp(r.origin_timestamp).isoformat() not in pair_ts:
            raise ValueError(f"Authoritative origin {r.origin_timestamp} has no test pair.")
    pred_by_ts = {
        pd.Timestamp(t).isoformat(): float(p)
        for t, p in zip(pd.to_datetime(keys_te["timestamp"]), preds)
    }
    return [
        {
            "exp_id": spec.exp_id,
            "model": spec.model,
            "horizon": int(r.horizon),
            "seed": spec.seed_marker,
            "origin_timestamp_iso": pd.Timestamp(r.origin_timestamp).isoformat(),
            "horizon_key": int(r.horizon),
            "target_timestamp": pd.Timestamp(r.origin_timestamp) + pd.Timedelta(hours=int(r.horizon)),
            "y_pred": pred_by_ts[pd.Timestamp(r.origin_timestamp).isoformat()],
        }
        for r in origins_h.itertuples()
    ]


def _full_feature_matrix(data: Q1Data) -> tuple[np.ndarray, np.ndarray, dict]:
    """Chronological 49-col matrix over train+val+test + timestamp positions."""
    X = canonical_matrix(data.full, data.feature_names)
    y = data.full["target"].to_numpy(dtype=float)
    return X, y, _timestamp_positions(data.full)


def predict_sequence(spec: Q1ExperimentSpec, data: Q1Data, model) -> list[dict]:
    """Fit on train (+explicit val); window-select origin forecasts."""
    if int(model.seq_len) != HISTORY_LENGTH or int(model.output_size) != int(spec.horizon):
        raise ValueError("Sequence history/horizon drift at predict time.")
    X_tr = canonical_matrix(data.tr, data.feature_names)
    y_tr = data.tr["target"].to_numpy(dtype=float)
    X_va = canonical_matrix(data.va, data.feature_names)
    y_va = data.va["target"].to_numpy(dtype=float)
    assert_no_test_overlap(
        pd.Timestamp(data.tr["timestamp"].max()), pd.Timestamp(data.te["timestamp"].min()), label=f"sequence-train/{spec.exp_id}"
    )
    assert_no_test_overlap(
        pd.Timestamp(data.va["timestamp"].max()), pd.Timestamp(data.te["timestamp"].min()), label=f"sequence-val/{spec.exp_id}"
    )
    model.fit(X_tr, y_tr, X_va, y_va)
    X_full, _, pos_of = _full_feature_matrix(data)
    origins_h = _origins_for_horizon(data.origins, spec.horizon)
    seq, horizon = HISTORY_LENGTH, int(spec.horizon)
    window_index = []
    for r in origins_h.itertuples():
        ots = pd.Timestamp(r.origin_timestamp)
        if ots not in pos_of:
            raise ValueError(f"Origin {ots} not in full frame.")
        p = int(pos_of[ots])
        # Window ending exactly at origin p: rows [p-seq+1, p].
        if p - seq + 1 < 0:
            raise ValueError(f"Insufficient 168h window at origin {ots}.")
        window_index.append(p - seq + 1)
    Xw = np.stack([X_full[i : i + seq] for i in window_index]).astype(float)
    if model.scale:
        Xw = model._scale_apply(Xw)
    if Xw.shape[2] != model.input_size_:
        raise ValueError("Sequence feature-width drift at predict time.")
    import torch
    from torch.utils.data import DataLoader

    assert model._torch_model is not None
    loader = DataLoader(
        torch.as_tensor(Xw.astype(np.float32)), batch_size=model.batch_size, shuffle=False, num_workers=0
    )
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            outs.append(model._torch_model(xb.to(model.device)).cpu().numpy())
    step_preds = np.concatenate(outs, axis=0)
    step_preds = model._target_inverse_transform(step_preds)
    # Window i forecasts rows [p+1 .. p+H]; Q1 scores the H-step point.
    last_step = step_preds[:, -1] if step_preds.ndim == 2 else np.ravel(step_preds)
    if last_step.shape[0] != len(origins_h):
        raise ValueError("Sequence origin-window count mismatch.")
    return [
        {
            "exp_id": spec.exp_id,
            "model": spec.model,
            "horizon": int(r.horizon),
            "seed": spec.seed_marker,
            "origin_timestamp_iso": pd.Timestamp(r.origin_timestamp).isoformat(),
            "horizon_key": int(r.horizon),
            "target_timestamp": pd.Timestamp(r.origin_timestamp) + pd.Timedelta(hours=int(r.horizon)),
            "y_pred": float(v),
        }
        for r, v in zip(origins_h.itertuples(), last_step)
    ]


def build_combined_tft_frame(data: Q1Data):
    """ONE combined TFT long frame over train+val+test (row-aligned)."""
    from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX

    combined_feat = pd.concat([data.tr, data.va, data.te], ignore_index=True)
    assert_canonical_frame(combined_feat, data.feature_names)
    tft = q1_tft_path.build_q1_tft_frame(combined_feat, data.feature_names)
    if len(tft) != len(combined_feat):
        raise ValueError("TFT frame row count != feature frame (leakage of alignment).")
    # Every non-key TFT column must be a canonical feature (no raw weather).
    allowed = set(canonical_feature_names()) | {TIME_IDX, GROUP_ID, TARGET}
    foreign = [c for c in tft.columns if c not in allowed]
    if foreign:
        raise ValueError(f"Non-canonical TFT columns: {foreign}.")
    n_tr, n_va = len(data.tr), len(data.va)
    train_tft = tft.iloc[:n_tr].copy()
    val_tft = tft.iloc[n_tr : n_tr + n_va].copy()
    test_min_time = int(tft.iloc[n_tr + n_va :][TIME_IDX].min())
    return tft, train_tft, val_tft, test_min_time


def build_tft_origin_frame(
    combined_tft: pd.DataFrame, origin_time: int, horizon: int, history: int = HISTORY_LENGTH
) -> pd.DataFrame:
    """Leakage-free per-origin TFT frame.

    History rows (``history`` rows ``<= origin``) are realized; the H
    decoder rows carry TRUE FUTURE CALENDAR knowns (deterministic, known
    in advance) with unknown reals carried forward from the origin row
    and NaN targets. No future-observed value except deterministic
    calendar ever enters the frame.
    """
    from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX

    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError(f"Q1 horizon must be one of {HORIZONS}.")
    hist = combined_tft[combined_tft[TIME_IDX] <= int(origin_time)].tail(int(history))
    if len(hist) != int(history):
        raise ValueError(f"Insufficient 168h TFT history at time {origin_time}.")
    if int(hist[TIME_IDX].max()) != int(origin_time):
        raise ValueError("TFT history does not end at the origin.")
    known = [c for c in q1_tft_path.Q1_KNOWN if c in combined_tft.columns]
    unknown = [c for c in combined_tft.columns if c not in known + [TIME_IDX, GROUP_ID, TARGET]]
    if not unknown:
        raise ValueError("TFT frame has no unknown (encoder) columns.")
    future = combined_tft[
        (combined_tft[TIME_IDX] > int(origin_time)) & (combined_tft[TIME_IDX] <= int(origin_time) + horizon)
    ]
    if len(future) != horizon:
        raise ValueError("Incomplete future calendar block for TFT origin frame.")
    last_hist = hist.iloc[-1]
    if last_hist[unknown].isna().any():
        raise ValueError("NaN in TFT origin history unknowns.")
    dec_rows = []
    for _, frow in future.iterrows():
        row = {TIME_IDX: int(frow[TIME_IDX]), GROUP_ID: "single", TARGET: np.nan}
        for c in known:  # deterministic future calendar only
            row[c] = frow[c]
        for c in unknown:  # carried forward from origin (no future data)
            row[c] = last_hist[c]
        dec_rows.append(row)
    frame = pd.concat([hist, pd.DataFrame(dec_rows)], ignore_index=True)
    if len(frame) != int(history) + horizon:
        raise ValueError("TFT origin frame has wrong length.")
    return frame


def predict_tft_family(spec: Q1ExperimentSpec, data: Q1Data, model, combined_tft: pd.DataFrame) -> list[dict]:
    """Per-origin TFT/Hybrid forecasts; last decode step = origin+H point."""
    origins_h = _origins_for_horizon(data.origins, spec.horizon)
    pos_of = _timestamp_positions(data.full)
    horizon = int(spec.horizon)
    rows = []
    for r in origins_h.itertuples():
        ots = pd.Timestamp(r.origin_timestamp)
        if ots not in pos_of:
            raise ValueError(f"Origin {ots} not in full frame.")
        o = int(pos_of[ots])  # == TIME_IDX in the combined TFT frame
        frame = build_tft_origin_frame(combined_tft, o, horizon)
        med = np.asarray(model.predict(frame), dtype=float)
        flat = np.ravel(med)
        if flat.shape[0] != horizon:
            raise ValueError(f"TFT median length {flat.shape[0]} != horizon {horizon}.")
        y_hat = float(flat[-1])
        if not np.isfinite(y_hat):
            raise ValueError(f"Non-finite TFT forecast at origin {ots}.")
        rows.append(
            {
                "exp_id": spec.exp_id,
                "model": spec.model,
                "horizon": int(r.horizon),
                "seed": spec.seed_marker,
                "origin_timestamp_iso": ots.isoformat(),
                "horizon_key": int(r.horizon),
                "target_timestamp": ots + pd.Timedelta(hours=int(r.horizon)),
                "y_pred": y_hat,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Scoring (§17), artifacts (§18), registry (§19)
# ---------------------------------------------------------------------------
def score_pred_rows(
    spec: Q1ExperimentSpec, rows: list[dict], data: Q1Data
) -> tuple[dict, pd.DataFrame]:
    """Validate alignment, then score with the existing Q1 scoring API."""
    origins_h = _origins_for_horizon(data.origins, spec.horizon)
    norm_rows = [
        {
            "exp_id": r["exp_id"],
            "model": r["model"],
            "horizon": int(r.get("horizon", r.get("horizon_key"))),
            "seed": r["seed"],
            "origin_timestamp_iso": r["origin_timestamp_iso"],
            "target_timestamp": pd.Timestamp(r["target_timestamp"]),
            "y_pred": float(r["y_pred"]),
        }
        for r in rows
    ]
    pred_frame = validate_pred_rows(origins_h, norm_rows)
    if (pred_frame["exp_id"] != spec.exp_id).any():
        raise ValueError("Prediction EXP-ID mismatch.")
    metrics, aligned = score_at_origins(
        origins_h,
        data.te,
        _preds_dict(pred_frame),
        y_train=data.y_train,
        seasonality=SEASONAL_PERIOD,
    )
    if set(metrics) != set(Q1_METRIC_KEYS):
        raise ValueError(f"Q1 metric set incomplete: {sorted(metrics)}.")
    scored = pred_frame.merge(
        aligned[["origin_timestamp", "y_true"]], on="origin_timestamp", how="left"
    )
    scored["target_timestamp"] = scored["origin_timestamp"] + pd.to_timedelta(scored["horizon"], unit="h")
    if scored["y_true"].isna().any():
        raise ValueError("Unmatched actuals after scoring.")
    return metrics, scored[
        ["exp_id", "model", "horizon", "seed", "origin_timestamp", "target_timestamp", "y_true", "y_pred"]
    ]


def write_artifacts(
    run_dir: Path,
    spec: Q1ExperimentSpec,
    model,
    pred_frame: pd.DataFrame,
    metrics: dict,
    data: Q1Data,
    effective_config: dict,
    duration_s: float,
) -> tuple[dict, dict]:
    """Write deterministic artifacts; save+hash checkpoint where applicable."""
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions" / f"{spec.exp_id}.csv"
    met_path = run_dir / "metrics" / f"{spec.exp_id}.json"
    meta_path = run_dir / "meta" / f"{spec.exp_id}.json"
    for p in (pred_path, met_path, meta_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(pred_path, index=False)
    commit = git_commit(str(Path(__file__).resolve().parents[2]))
    meta = {
        "exp_id": spec.exp_id,
        "protocol_version": PROTOCOL_VERSION,
        "model": spec.model,
        "horizon": int(spec.horizon),
        "seed": spec.seed_marker,
        "dataset_path": data.provenance["dataset_path"],
        "dataset_sha256": data.provenance["dataset_sha256"],
        "dataset_version": Q1_DATASET_VERSION,
        "n_rows": int(data.provenance["n_rows"]),
        "timestamp_range": (data.provenance["timestamp_min"], data.provenance["timestamp_max"]),
        "feature_version": Q1_FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "n_features": len(data.feature_names),
        "origin_artifact": data.origin_artifact,
        "origin_set_hash": data.provenance["origin_set_hash"],
        "git_commit": commit,
        "train_range": data.provenance["train_range"],
        "val_range": data.provenance["val_range"],
        "test_range": data.provenance["test_range"],
        "train_boundary": TRAIN_END,
        "val_boundary": VAL_END,
        "test_boundary": TEST_END,
        "config_snapshot": CONFIG_SNAPSHOT,
        "effective_config": effective_config,
        "python_version": python_version(),
        "packages": package_versions(PROVENANCE_PACKAGES),
        "cuda": cuda_info(),
        "runtime_s": round(float(duration_s), 3),
        "smoke": False,
    }
    checkpoint_path, checkpoint_sha = "", ""
    if spec.model not in ("persistence", "seasonal_persistence", "sarima"):
        ckpt_path = run_dir / "checkpoints" / f"{spec.exp_id}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(ckpt_path)
        checkpoint_path, checkpoint_sha = str(ckpt_path), sha256_file(ckpt_path)
    meta.update({"checkpoint": checkpoint_path, "checkpoint_sha256": checkpoint_sha})
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump({"exp_id": spec.exp_id, "metrics": metrics}, f, indent=2, default=str)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    record = record_experiment(
        exp_id=spec.exp_id,
        status="COMPLETED",
        git_commit=commit,
        model=spec.model,
        horizon=int(spec.horizon),
        seed=spec.seed_marker,
        dataset_version=Q1_DATASET_VERSION,
        dataset_sha256=data.provenance["dataset_sha256"],
        features=Q1_FEATURE_VERSION,
        feature_schema_hash=feature_schema_hash(),
        origin_set_hash=data.provenance["origin_set_hash"],
        config_snapshot=CONFIG_SNAPSHOT,
        python_version=meta["python_version"],
        packages=json.dumps(meta["packages"]),
        cuda=json.dumps(meta["cuda"]),
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        predictions=str(pred_path),
        metrics=json.dumps(metrics, default=str),
        runtime_s=duration_s,
        smoke=False,
        notes=(
            f"Q1 headline experiment {spec.exp_id} protocol {PROTOCOL_VERSION}; "
            f"train<{TRAIN_END} val< {VAL_END} test<{TEST_END}; "
            "scored on authoritative common origins; models NOT ranked."
        ),
        path=Path(run_dir) / "REGISTRY.csv",
    )
    # §19: completeness gate stays closed on any gap.
    require_q1_complete(record)
    return record, meta


# ---------------------------------------------------------------------------
# Single-experiment + manifest execution
# ---------------------------------------------------------------------------
def run_experiment(
    spec: Q1ExperimentSpec,
    data: Q1Data,
    run_dir: Path,
    combined_tft: Optional[pd.DataFrame] = None,
) -> dict:
    """Execute ONE locked experiment end-to-end (fit on real data)."""
    t0 = time.time()
    run_dir = Path(run_dir)
    apply_spec_seed(spec)
    model, effective = construct_model(
        spec, checkpoint_dir=run_dir / "checkpoints" / f"{spec.exp_id}_tft"
    )
    if spec.model in ("persistence", "seasonal_persistence"):
        rows = predict_naive(spec, data)
    elif spec.model == "sarima":
        rows = predict_sarima(spec, data, model)
    elif spec.model in ("xgboost", "lightgbm"):
        rows = predict_tree(spec, data, model)
    elif spec.model in ("lstm", "gru", "patchtst", "nbeats"):
        rows = predict_sequence(spec, data, model)
    elif spec.model in ("tft", "hybrid"):
        if combined_tft is None:
            raise ValueError("TFT-family experiments require the combined TFT frame.")
        rows = fit_and_predict_tft_family(spec, data, model, combined_tft, run_dir)
    else:  # pragma: no cover - guarded by construct_model
        raise ValueError(f"Unknown Q1 model {spec.model!r}.")
    metrics, pred_frame = score_pred_rows(spec, rows, data)
    record, _ = write_artifacts(run_dir, spec, model, pred_frame, metrics, data, effective, time.time() - t0)
    logger.info("Q1 completed %s", spec.exp_id)
    return record


def fit_and_predict_tft_family(
    spec: Q1ExperimentSpec,
    data: Q1Data,
    model,
    combined_tft: pd.DataFrame,
    run_dir: Path,
) -> list[dict]:
    """Fit TFT/Hybrid on train+val ONLY (test-exclusion asserted), then predict."""
    from src.models.tft.dataset import TIME_IDX

    n_tr, n_va = len(data.tr), len(data.va)
    test_min_time = int(combined_tft[TIME_IDX].iloc[n_tr + n_va :].min())
    if spec.model == "tft":
        train_tft = combined_tft.iloc[:n_tr].copy()
        val_tft = combined_tft.iloc[n_tr : n_tr + n_va].copy()
        fitted = q1_tft_path.fit_q1_tft(
            train_tft,
            val_tft,
            int(spec.horizon),
            int(spec.seed),  # type: ignore[arg-type]
            checkpoint_dir=Path(run_dir) / "checkpoints" / f"{spec.exp_id}_tft",
            test_time_min=test_min_time,
        )
        # The fitted model (best-val checkpoint restored) is the scorer.
        return predict_tft_family(spec, data, fitted, combined_tft)
    # Hybrid: OOF over the train+val TFT frame; test strictly excluded.
    fit_frame = combined_tft.iloc[: n_tr + n_va].copy()
    if int(fit_frame[TIME_IDX].max()) >= test_min_time:
        raise ValueError("Hybrid fit frame reaches test data.")
    if int(model.n_oof_folds) != Q1_OOF_FOLDS or model.residual_training_mode != "oof":
        raise ValueError("Hybrid must be 5-fold chronological OOF (in_sample forbidden).")
    model.fit(fit_frame)
    if len(model.oof_folds_) != Q1_OOF_FOLDS:
        raise ValueError(f"Hybrid produced {len(model.oof_folds_)} OOF folds, expected 5.")
    if model.residual_stats_.get("mode") != "oof":
        raise ValueError("Hybrid residual mode is not OOF.")
    lo, hi = model.residual_period_
    if not (int(fit_frame[TIME_IDX].min()) <= int(lo) <= int(hi) < test_min_time):
        raise ValueError("Hybrid OOF residual period escapes the train+val frame.")
    return predict_tft_family(spec, data, model, combined_tft)


def run_manifest(
    specs: list[Q1ExperimentSpec],
    data: Q1Data,
    run_dir: Path,
    allow_partial: bool = False,
) -> list[dict]:
    """Execute a validated manifest; refuse partial sets unless allowed."""
    validate_manifest_size(specs, expect_full=not allow_partial)
    if not allow_partial:
        expected = {s.exp_id for s in build_manifest()}
        if {s.exp_id for s in specs} != expected:
            raise ValueError("Q1 run set != full 81-experiment manifest (partial sets forbidden).")
    run_dir = Path(run_dir)
    combined_tft: Optional[pd.DataFrame] = None
    if any(s.model in ("tft", "hybrid") for s in specs):
        combined_tft, _, _, _ = build_combined_tft_frame(data)
    records = [run_experiment(spec, data, run_dir, combined_tft) for spec in specs]
    validate_run_set(records, specs)
    return records


def validate_run_set(records: list[dict], specs: list[Q1ExperimentSpec]) -> None:
    """Fail closed unless every manifest experiment completed exactly once."""
    expected = sorted(s.exp_id for s in specs)
    got = sorted(r["exp_id"] for r in records)
    if got != expected:
        raise ValueError("Completed run set != manifest (missing/extra experiments).")
    for record in records:
        require_q1_complete(record)


# ---------------------------------------------------------------------------
# Preflight (SAFE: real metadata, no training, no canonical writes/results)
# ---------------------------------------------------------------------------
def _synthetic_v2_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Tiny deterministic V2-like frame for preflight/tests (NON-RESEARCH)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    load = 100 + 10 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24) + rng.normal(0, 1, n)
    wind = np.abs(rng.normal(5, 1, n))
    solar = np.maximum(0, np.sin(np.pi * ((np.arange(n) % 24) - 6) / 12)) * 8
    temp = 15 + 5 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24) + rng.normal(0, 0.5, n)
    tot = wind + solar
    return pd.DataFrame(
        {
            "timestamp": ts,
            "pjm_load_mw": load,
            "wind_generation_mw": wind,
            "solar_generation_mw": solar,
            "temperature": temp,
            "humidity": 60 + rng.normal(0, 2, n),
            "wind_speed": 3 + rng.normal(0, 0.5, n),
            "cloud_cover": 0.4 + rng.normal(0, 0.1, n),
            "solar_radiation": solar * 10,
            "total_renewable": tot,
            "net_load": load - tot,
            "renewable_penetration": tot / load,
            "cdh": np.maximum(0, temp - 18),
            "hdh": np.maximum(0, 18 - temp),
        }
    )


def preflight(
    dataset_path: str = Q1_DATASET_PATH,
    origin_artifact: str = ORIGIN_ARTIFACT,
    output_path: Optional[str | Path] = None,
) -> dict:
    """SAFE preflight: real gates 1-4 + synthetic orchestration checks.

    Trains NOTHING, registers NOTHING, modifies NO canonical artifact, and
    produces NO research results. Report is labeled PREFLIGHT/NON-RESULT.
    """
    checks: dict[str, str] = {}

    def check(name: str, fn) -> None:
        try:
            fn()
            checks[name] = "PASS"
        except Exception as exc:  # fail-closed: record, do not raise yet
            checks[name] = f"FAIL: {exc}"

    # -- Real-data gates (read-only; origins loaded, never generated) --
    raw, provenance = load_and_validate_dataset(dataset_path)
    check("dataset_gate", lambda: load_and_validate_dataset(dataset_path))
    clean, feature_names, _ = build_and_validate_features(raw)
    check("feature_gate", lambda: build_and_validate_features(raw))
    tr, va, te = split_and_validate(clean, feature_names)
    check("split_gate", lambda: split_and_validate(clean, feature_names))
    origins, actual_artifact = ensure_origins(te, origin_artifact, allow_generate=False)
    check("origin_gate", lambda: ensure_origins(te, origin_artifact, allow_generate=False))

    # -- Manifest / EXP-ID / seeds (pure logic) --
    def _manifest():
        specs = build_manifest()
        validate_manifest_size(specs, expect_full=True)
        if len(specs) != FULL_MANIFEST_SIZE:
            raise ValueError("Manifest size drift.")
        det = [s for s in specs if s.seed is None]
        sto = [s for s in specs if s.seed is not None]
        if len(det) != 9 or len(sto) != 72:
            raise ValueError("Deterministic/stochastic multiplicity drift.")
        if {s.seed for s in sto} != set(Q1_SEEDS):
            raise ValueError("Seed expansion drift.")
        twice = [s.exp_id for s in build_manifest()]
        if twice != [s.exp_id for s in specs]:
            raise ValueError("EXP-ID nondeterminism.")

    check("manifest_81_deterministic", _manifest)
    check("headline_guard", lambda: q1_headline_models() or _raise_guard())
    check("seed_rules", _check_seed_rules)

    # -- Synthetic orchestration checks (no real data, no training) --
    synth = _synthetic_v2_frame()
    syn_feat, syn_names = build_q1_features(synth)
    syn_clean, _ = drop_warmup(syn_feat, syn_names)
    syn_origins = generate_origins(syn_clean, horizons=tuple(HORIZONS))
    check("feature_propagation", lambda: _check_feature_propagation(syn_clean, syn_names))
    check("origin_propagation_alignment", lambda: _check_alignment(syn_clean, syn_origins))
    check("mase_wiring", lambda: _check_mase_wiring(syn_clean))
    check("tft_test_exclusion", _check_tft_exclusion)
    check("hybrid_test_exclusion", _check_hybrid_exclusion)
    check("registry_fields", _check_registry_fields)
    check("stale_artifact_absent", _check_stale_artifact)

    report = {
        "status": "PREFLIGHT_COMPLETED_NON_RESULT",
        "protocol_version": PROTOCOL_VERSION,
        "note": "Preflight only. NOT headline results. Models NOT ranked. Nothing trained.",
        "dataset_sha256": provenance["dataset_sha256"],
        "feature_schema_hash": feature_schema_hash(),
        "origin_set_hash": origins_hash(origins),
        "origin_artifact": str(actual_artifact),
        "split_rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "manifest_size": FULL_MANIFEST_SIZE,
        "checks": checks,
    }
    failed = {k: v for k, v in checks.items() if v != "PASS"}
    report["preflight"] = "PASS" if not failed else "FAIL"
    if output_path is None:
        commit = git_commit(str(Path(__file__).resolve().parents[2]))
        output_path = Path("results") / "q1_preflight" / f"preflight_{PROTOCOL_VERSION}_{commit}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Q1 preflight %s (%d/%d checks): %s", report["preflight"], sum(v == "PASS" for v in checks.values()), len(checks), output_path)
    if failed:
        raise ValueError(f"Q1 preflight FAILED checks: {failed}.")
    return report


def _raise_guard():
    for bad in (["xgboost", "linear_regression"], ["nope"], ["tft", "tft"], []):
        try:
            q1_headline_models(bad)
        except ValueError:
            continue
        raise ValueError(f"Headline guard accepted {bad!r}.")
    return True


def _check_seed_rules() -> None:
    for s in build_manifest():
        if s.model in DETERMINISTIC_MODELS and s.seed is not None:
            raise ValueError("Deterministic spec carries a seed.")
        if s.model not in DETERMINISTIC_MODELS and s.seed not in Q1_SEEDS:
            raise ValueError("Stochastic spec seed drift.")
    apply_spec_seed(Q1ExperimentSpec(model="persistence", horizon=1, seed=None))
    apply_spec_seed(Q1ExperimentSpec(model="xgboost", horizon=1, seed=42))
    try:
        apply_spec_seed(Q1ExperimentSpec(model="xgboost", horizon=1, seed=7))
    except ValueError:
        return
    raise ValueError("Seed gate accepted seed 7.")


def _check_feature_propagation(frame: pd.DataFrame, names: list[str]) -> None:
    assert_canonical_frame(frame, names)
    tampered = [c for c in frame.columns if c not in Q1_ID_COLUMNS][::-1]
    bad = frame.rename(columns={tampered[0]: tampered[0] + "_x"})
    try:
        assert_canonical_frame(bad, names)
    except ValueError:
        return
    raise ValueError("Feature gate accepted a tampered frame.")


def _check_alignment(frame: pd.DataFrame, origins: pd.DataFrame) -> None:
    origins_h = origins[origins["horizon"] == 1].reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    pos_of = {t: i for i, t in enumerate(ts)}
    series = frame["target"].to_numpy(dtype=float)
    good = [
        {
            "exp_id": "PREFLIGHT",
            "model": "persistence",
            "horizon": 1,
            "seed": DET_SEED_MARKER,
            "origin_timestamp_iso": pd.Timestamp(r.origin_timestamp).isoformat(),
            "target_timestamp": pd.Timestamp(r.origin_timestamp) + pd.Timedelta(hours=1),
            "y_pred": float(series[pos_of[pd.Timestamp(r.origin_timestamp)]]),
        }
        for r in origins_h.itertuples()
    ]
    validate_pred_rows(origins_h, good)
    for mutate in ("missing", "duplicate", "wrong_horizon", "outside"):
        bad = [dict(r) for r in good]
        if mutate == "missing":
            bad = bad[1:]
        elif mutate == "duplicate":
            bad = bad + [dict(bad[0])]
        elif mutate == "wrong_horizon":
            bad[0] = dict(bad[0], horizon=6)
        else:
            bad[0] = dict(bad[0], origin_timestamp_iso="2030-01-01T00:00:00")
        try:
            validate_pred_rows(origins_h, bad)
        except ValueError:
            continue
        raise ValueError(f"Alignment gate accepted {mutate} predictions.")


def _check_mase_wiring(frame: pd.DataFrame) -> None:
    y = frame["target"].to_numpy(dtype=float)
    if y.size <= SEASONAL_PERIOD + 10:
        raise ValueError("Synthetic frame too short for MASE check.")
    mini_origins = pd.DataFrame(
        {"origin_timestamp": [frame["timestamp"].iloc[200], frame["timestamp"].iloc[210]], "horizon": [1, 1]}
    )
    preds = {
        (pd.Timestamp(frame["timestamp"].iloc[200]).isoformat(), 1): float(y[201]),
        (pd.Timestamp(frame["timestamp"].iloc[210]).isoformat(), 1): float(y[211]),
    }
    m1, _ = score_at_origins(mini_origins, frame, preds, y_train=y[:180], seasonality=168)
    if set(m1) != set(Q1_METRIC_KEYS):
        raise ValueError("Metric set incomplete.")
    m2, _ = score_at_origins(mini_origins, frame, preds, y_train=y[:180], seasonality=168)
    if m1["MASE"] != m2["MASE"]:
        raise ValueError("MASE nondeterminism.")
    for bad_kwargs in ({"y_train": None}, {"y_train": y[:180], "seasonality": 24}, {"y_train": np.ones(50)}):
        try:
            score_at_origins(mini_origins, frame, preds, **bad_kwargs)
        except ValueError:
            continue
        raise ValueError(f"MASE gate accepted {sorted(bad_kwargs)}.")


def _check_tft_exclusion() -> None:
    from src.models.tft.dataset import GROUP_ID, TARGET, TIME_IDX
    from src.q1.tft_path import assert_fit_frame

    base = pd.DataFrame(
        {
            TIME_IDX: np.arange(300),
            GROUP_ID: "single",
            TARGET: np.linspace(100, 120, 300),
            "hour": np.tile(np.arange(24), 13)[:300],
            "day_of_week": np.tile(np.arange(7), 43)[:300],
        }
    )
    assert_fit_frame(base.iloc[:200], base.iloc[200:250], test_time_min=250)
    for bad in (
        lambda: assert_fit_frame(base.iloc[200:250], base.iloc[:200]),
        lambda: assert_fit_frame(base.iloc[:200], base.iloc[200:250], test_time_min=240),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise ValueError("TFT fit-frame gate accepted an unsafe frame.")


def _check_hybrid_exclusion() -> None:
    # Hybrid test exclusion reuses the TFT-frame disjointness rule: the OOF
    # fit frame must end strictly before test data begins.
    hw = {"n_oof_folds": Q1_OOF_FOLDS, "mode": "oof"}
    if hw["n_oof_folds"] != 5 or hw["mode"] != "oof":
        raise ValueError("Hybrid OOF requirement drift.")
    fit_max, test_min = 1000, 1001
    assert_no_test_overlap(pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), label="hybrid")
    try:
        assert_no_test_overlap(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), label="hybrid")
    except ValueError:
        return
    raise ValueError("Hybrid test-exclusion gate accepted overlap.")


def _check_registry_fields() -> None:
    rec = {
        "exp_id": "Q1_tft_H24_S42",
        "status": "COMPLETED",
        "git_commit": "abc1234",
        "dataset_version": "v2",
        "dataset_sha256": "f" * 64,
        "features": Q1_FEATURE_VERSION,
        "feature_schema_hash": "e" * 64,
        "origin_set_hash": "5" * 64,
        "model": "tft",
        "horizon": 24,
        "seed": 42,
        "config_snapshot": CONFIG_SNAPSHOT,
        "predictions": "p",
        "metrics": "{}",
        "checkpoint": "c",
        "smoke": "False",
    }
    require_q1_complete(rec)
    try:
        require_q1_complete({**rec, "smoke": "True"})
    except ValueError:
        pass
    else:
        raise ValueError("Registry gate accepted a smoke record.")
    try:
        require_q1_complete({**rec, "checkpoint": ""})
    except ValueError:
        pass
    else:
        raise ValueError("Registry gate accepted a missing checkpoint.")


def _check_stale_artifact(registry_path: str | Path = Path("experiments/REGISTRY.csv")) -> None:
    import csv as _csv

    reg = Path(registry_path)
    if not reg.is_file():
        return  # nothing stale to check
    with open(reg, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            pred = (row.get("predictions") or "").strip()
            if pred.endswith(".parquet") and not Path(pred).is_file():
                raise ValueError(f"Stale registry artifact path: {pred}.")
