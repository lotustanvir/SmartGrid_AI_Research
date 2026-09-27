# Project Structure — Q1-Ready Repository Map (Phase 1 preparation)

**Principle:** new Q1 scaffolding alongside frozen legacy. Nothing existing was moved, edited, or deleted to create this tree (verified via `git status`: new `??` paths only).

## Root

- `README.md` — science statement + reproduce-in-3-commands (to be rewritten Phase 8; current 44-line model-first note preserved).
- `requirements.txt` — current loose deps (kept; locked by `environment.yml` in Phase 5).
- `environment.yml` — **placeholder**: pinned env locked in Phase 5 (reproducibility gate).
- `CITATION.cff` — **placeholder**: authors/DOI completed at artifact freeze (Phase 8).
- `research_audit_report.md`, `Q1 Upgrade Roadmap.md`, `Q1_Phase0_Research_Strategy.md` — prior audit/strategy, read-only provenance.

## configs/

Locked research protocol + search spaces. Existing (`default/data/pjm_data/models/experiment.yaml`) untouched.
**Added placeholders:** `pjm_data_v1.yaml` (frozen-legacy mirror of current mapping), `pjm_data_v2.yaml` (weather/calendar/derived spec, dataset not yet built), `hpo_spaces.yaml` (Optuna ranges consumed Phase 5; no HPO executed).

## data/ (NEW canonical tree; legacy `dataset/` frozen in place)

- `raw/pjm/{load,wind,solar}/` — PJM drops per energy type (Phase 1; legacy `dataset/raw/pjm/` untouched).
- `raw/weather/{era5,noaa}/` — ERA5 primary / NOAA fallback pulls (Phase 1).
- `raw/external/` — holidays, alerts, third-party pulls + provenance notes.
- `processed/v1/` — pointer to frozen `dataset/processed/pjm_smart_grid_2020_2025.csv` (never copied).
- `processed/v2/` — `pjm_weather_dataset.csv`, `events.parquet`, `test_origins.parquet`, `dataset_metadata.json` (built Phases 1–3).
- `checksums/sha256.txt` — hashes at freeze; every EXP row references them.
- `README.md` — mapping note above.

## docs/

Publication-control documents (authority order: PROTOCOL → STATS_PLAN → DATA/MODEL cards → guides).
`PROTOCOL.md` (unified 168h→1/6/24h task, equal-n rule, leakage checks), `DATA_CARD.md` (v1 reality + v2 spec), `MODEL_CARD.md` (8 current + 11 target), `STATS_PLAN.md` (Friedman/Holm-Wilcoxon/DM, bug remediation), `EXPERIMENT_GUIDE.md` (EXP lifecycle), `PROJECT_STRUCTURE.md` (this file).

## src/

Existing code (`data/`, `models/`, `training/`, `evaluation/`, `utils/`) untouched.
**Added folder:** `explainability/` (empty; `shap_analysis.py`, `attention_analysis.py` land in Phase 7).
Target model/train/eval files in the tree (`persistence.py`, `patchtst.py`, `hpo.py`, `statistics.py`, …) are **not created in Phase 1** — folders are ready, code arrives in Phases 4–7.

## experiments/ (NEW)

- `EXP_TEMPLATE/{config.yaml,metrics.json,verification.json,README.md}` — copy-per-run template (placeholders).
- `REGISTRY.csv` — append-only header; one row per EXP-ID; manuscript numbers trace here (PROTOCOL §11).

## results/ (existing frozen + new)

- Existing `results/{data_validation,eda,experiments,…}/` — frozen provenance, untouched.
- `results/q1/{leaderboards,ablation,statistics,calibration,explainability,operational}/` — canonical Q1 outputs (Phases 5–7).
- `results/legacy/README.md` — pointer note (no copies moved).

## saved_models/ (empty, preserved) + `saved_models/q1/` (NEW, empty)

33 weights (11 models × 3 horizons) land in Phase 5 with `RUN_MANIFEST.json`. Current emptiness is an explicit reproducibility gap, not filler.

## figures/ + tables/ (NEW top-level; empty)

Numbered exports for the manuscript (Phase 8). Existing scattered EDA PNGs stay under `results/eda/`; `results/figures/` + `results/tables/` subfolders preserved as-is.

## paper/ (NEW, empty scaffolding)

`manuscript/` (draft Phase 8), `references.bib` (placeholder; entries traced to EXP-IDs), `figures/` (exports), `response_to_reviewers/` (template Phase 8).

## tests/ (existing flat suite preserved) + NEW subfolders

`tests/data_tests/`, `model_tests/`, `evaluation_tests/` (empty; task-fairness/leakage/stats/units tests land Phases 2–6). Existing 12 test files untouched.

## Legacy top-level folders (preserved, out of Q1 tree)

`dataset/`, `evaluation/`, `preprocessing/`, `utils/`, `notebooks/`, `lightning_logs/`, `venv/`, `__pycache__/` — untouched; consolidation (if any) decided after Phase 8, never by moving data.
