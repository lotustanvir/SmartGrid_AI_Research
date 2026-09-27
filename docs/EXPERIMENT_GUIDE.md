# Experiment Guide — How to Run and Record an Experiment (Phase 1 READY, no runs yet)

**Authority:** `docs/PROTOCOL.md` (task), `docs/STATS_PLAN.md` (inference), `docs/DATA_CARD.md` (data), `docs/MODEL_CARD.md` (models).
**Scope:** Phases 1–7. No experiment may start until its EXP folder + registry row exist.

## 1. Experiment ID format

```
EXP<NNN>_<family>_<focus>_h<H>_s<seed>
```

Families: `baseline | ml | dl | adv | hybrid | ablation | uncertainty | explain | operational`.
Horizons: `h1 | h6 | h24 | hAll`. Seeds: `s42 | s123 | s2025 | sDet`.
Example: `EXP032_ablation_weather_C_h6_s42`. IDs are never reused.

## 2. Lifecycle

1. Copy `experiments/EXP_TEMPLATE/` → `experiments/<EXP-ID>/`.
2. Fill `config.yaml` (dataset version + hash, feature group, model + git tag, hypers/trial, seed/env, splits + origin-set hash).
3. Append one row to `experiments/REGISTRY.csv` with status `planned`.
4. Run (Phases 5–7 only). Write `metrics.json` + `verification.json` + figures.
5. Flip registry status to `done` (or `failed` with reason). `superseded` rows are retained, never deleted.
6. Link manuscript tables/figures to EXP-IDs (`paper_table` column). **No result enters the manuscript without an EXP-ID (PROTOCOL §11).**

## 3. Per-experiment record (mandatory)

Dataset version · feature set (+hash) · model (+code tag) · hyperparameters (or Optuna trial-ID) · seed + environment · splits/origins (+hash) · full metrics per horizon/slice · result paths · leakage check (5/5) · cost (wall-time, params, CO₂e) · status.

## 4. Rules

- HPO on validation only; test scored once per best config.
- Equal-n origins enforced fail-closed (PROTOCOL §6); unequal-n runs are rejected, not averaged.
- `in_sample` hybrid mode is debug-only and banned from research tables.
- Failed runs stay in the registry with logs — deletion is forbidden (provenance).
