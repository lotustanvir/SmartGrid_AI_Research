# Q1 Phase 0 Research Strategy — PJM Short-Term Load Forecasting (2020–2025)

**Repository:** `D:\SmartGrid_AI_Research`
**Phase:** 0 ONLY — restructuring, organization, publication preparation
**Constraints obeyed:** No code modified. No model trained. No results changed.
**Reviewer lens:** IEEE Transactions on Smart Grid / Applied Energy
**Date (UTC):** 2026-09-24
**Prior artifacts (untouched):** `research_audit_report.md`, `Q1 Upgrade Roadmap.md` — Phase 0 builds on them, replaces none.

---

## 1. Current Research Understanding

### 1.1 What is this research actually trying to solve?

**One-sentence answer:** Hour-ahead (nominally day-ahead) electricity demand forecasting for the PJM Interconnection footprint, by comparing 8 forecasters on one hourly dataset covering 2020–2025.

**Operational mapping:** PJM (~65M people, ~180 GW mean load in this dataset) clears day-ahead and balancing markets on forecasts. An hourly error of a few GW triggers reserve activation, peaker starts, price spikes, or curtailment. The societal problem is real; the repository's framing of it is not yet written anywhere (no `paper/`, no RQs, no gap paragraph).

**Reconstructed technical task (as-coded, not as-documented):**

- Input: hand-built lags (`lag_1/24/168`), rolling means/stds (`24/168`), calendar + cyclical encodings, RTO wind/solar.
- Target: `pjm_load_mw` (RTO-total, hourly).
- Output: nominally 1-step point forecast; TFT/Hybrid actually emit trailing 6-step windows; protocol YAML says horizon 24. **Three different tasks share one leaderboard.**

### 1.2 Inspected reality (evidence, not impression)

| Layer | What exists | Verdict |
|---|---|---|
| Configs | `configs/{default,data,pjm_data,models,experiment}.yaml` — mapping, cleaning, features, 8-model hypers, 70/15/15 split | Good skeleton; horizons/encoders contradictory (24 vs 6 vs 168) |
| Data code | `src/data/{loader,schema,validation,preprocessing,features,pipeline,adapters,pjm_integration,eda}.py` | Best part: canonical schema, chronological `TemporalSplit` with TRAIN<VAL<TEST asserts, shift-safe lags (`shift(lag)`), rolling via `shift(1).rolling()`, scaler fit-train-only |
| Models | `src/models/base.py` + `baseline/{linear,rf,xgb,lgbm}` + `deep/{sequence,lstm,gru}` + `tft/{dataset,ft_model}` + `hybrid/tft_xgb.py`; registry of 8 in `experiment_runner.py:42` | Common API real; OOF hybrid design thoughtful with honest `in_sample` warning; TFT `interpret()` implemented |
| Training | `src/training/{splits,trainer,experiment_runner,ablation}.py`; `run_phase_6b.py`, `run_phase_6a.py`, `run_fresh_validation_aligned.py`, `run_tft_only.py` | Single seed 42, `use_early_stopping=False` for trees, ad-hoc TFT re-indexing fork in `run_phase_6b.py:60-77` competing with `pipeline.py` |
| Evaluation | `src/evaluation/{metrics,alignment}.py` — MAE/RMSE/safe-MAPE/sMAPE/R² + `(group,time)` intersection logic | Metrics correct in isolation; alignment not horizon-aware; permits unequal-n tables |
| Dataset | `dataset/raw/pjm/{2020..2025}/` (18 CSVs) → `dataset/processed/pjm_smart_grid_2020_2025.csv` — **52,608 hourly rows**, 2020-01-01 05:00 UTC → 2026-01-01 04:00 UTC, 0 gaps (auditor re-verified); target mean ~179,950 MW; wind 3,383 / solar 1,350 MW mean | Large, modern, well-audited raw layer (`pjm_raw_audit_2020_2025.json`, 1,653 lines); **no temperature/humidity/holiday** (`weather.*: null`); renewables RTO-only at 2.0%→3.4% penetration |
| Features | 21 cols: solar, wind + 6 calendar + 6 cyclical + 3 lags + 4 rolling stats | Leakage-safe but Q1-incomplete (no CDH/HDH, heat-index, holiday, season, DST, zenith, net-load) |
| Results | `results/{data_validation,eda,experiments,model_validation,research_analysis,fresh_validation*,explainability}`; 20+ EDA PNGs; `detailed_results.json`, `final_validation_report.json`, `statistical_tests_v2.json`, `ablation_v2.json`, `hybrid_analysis.json`, `tuned_results.json`, `uncertainty_results.json`, `contribution_analysis.json` | Internally contradictory (see §1.4); `saved_models/` **empty**, `results/tables/` **empty**, top `results/figures/` **empty**, `notebooks/` **empty**, `lightning_logs/` 700+ versions (churn) |
| Tests/docs | `tests/` 12 files; `README.md` 44 lines (model-first strategy + research rules, no science statement) | Protocol tests claimed 179/179; no paper draft, no `references.bib`, no RQs |

### 1.3 Current methodology (as-executed)

1. Aggregate load across 30 load-areas per timestamp; keep `area==RTO` wind/solar; inner-join; clip negatives (solar 24,245 cells, wind handful — undocumented sensitivity).
2. Engineer 21 features (leakage-safe); chronological 70/15/15 split (train → 2024-03-14, val → 2025-02-06, test → 2026-01-01); StandardScaler train-only; 168 warmup rows dropped.
3. Train 8 models (trees row-wise 1-step; LSTM/GRU `seq_len=24`; TFT `enc=24/hor=6`) with default hypers, seed 42.
4. Score headline table on mismatched horizons/samples; run 4-variant model-ablation, Friedman + 28 Wilcoxons, SHAP-style JSONs, single-level PICP.

### 1.4 Current contribution claim (from `contribution_analysis.json`)

Claims: (1) autocorrelation empirical finding (R²=0.965 lag features), (2) leakage-free framework, (3) ablation hierarchy Historical>Temporal>Renewable, (4) hybrid +5.54% over TFT, (5) 8-model statistical validation (χ²=8033, df=7). Positioning note honestly states *"not positioned as deep learning superiority claim"* — the one self-aware line in the results tree.

### 1.5 Current limitations (reviewer-grade — problems before solutions)

1. **No paper object.** No title, abstract, RQs, related work, gap, or claims document. Nothing to review except logs.
2. **Headline refutes itself.** Linear Regression (MAE 5,082 / RMSE 6,340 / R² 0.965) beats Hybrid/TFT/LSTM by ~2×. RF (R² 0.59) and LightGBM (R² 0.47–0.58) collapse — untuned defaults, not a finding.
3. **Invalid comparison.** Horizons 1 vs 6 vs 24; TFT fresh run n=6 vs tabular n=7,892; LSTM≡GRU byte-identical MAE/RMSE; 4 models share R²=0.87728; `final_review.json` admits *"Deep model results estimated"* + *"Pure Python tree implementations"*.
4. **Broken artifacts.** `results.csv`: sMAPE=0 all rows, MAPE units mixed (2.717 vs 0.0278), times 0.0/30.0 placeholders; Phase 6B-A LSTM/GRU R²=−64.7 ("not tuned"); fresh OOF-hybrid R²=−2.96 contradicts headline +5.54%; Wilcoxon W fractional (e.g. 6191831.92 — must be integer); PICP exactly 0.9001; SHAP refs phantom `net_load`/`total_renewable_mw` absent from pipeline; bogus "2026" EDA annual row.
5. **Missing physics.** No temperature in a load paper = desk reject. No humidity, holiday, DST, season, degree-hours, zenith, net-load. Weather adapter key (`adapters.py:TFT_KNOWN_MAP[temperature]`) is dead code.
6. **No generalization.** Single RTO aggregate, single split with mid-month cuts, single seed, no rolling origin, no second ISO/benchmark (GEFCom/ENTSO-E), no HPO (trees no early stopping; DL 24h encoder blind to weekly cycle; TFT config 24/32/2 vs doc 168/64/4).
7. **No trust/deployment.** Single 90% interval, uncalibrated hybrid bounds (honestly labeled, still unpublishable), SHAP from non-fitted values, no peak/ramp/MASE/skill/DM, no latency/params/FLOPs/CO₂, no cost/carbon translation, no weights to reproduce anything.

**Phase 0 conclusion:** Preserve the leakage-safe pipeline skeleton and raw-data audit trail. Freeze everything else as provenance. Nothing currently in `results/` can appear in a Q1 manuscript without regeneration under a locked protocol.

---

## 2. New Q1 Research Direction

### 2.1 Guiding constraint

> The project shall not assume deep learning superiority. The data currently proves the opposite. The publishable thesis is **trustworthy, fair, renewable-aware, uncertainty-aware, deployable forecasting** — when simple wins, when complexity pays, and what operators should actually run.

### 2.2 New research title options (5 — pick ONE at Phase 8, not before)

1. **Trustworthy Short-Term Load Forecasting for Smart Grids: A Leakage-Free Benchmark of Statistical, Machine Learning, and Transformer–Hybrid Models on Six Years of PJM Data** *(Recommended — rigor + benchmark scope; fits Applied Energy / IEEE TSG)*
2. **When Simple Wins: Horizon-Dependent Value of Linear, Gradient-Boosted, Recurrent, and Transformer Models for Renewable-Aware Load Forecasting in PJM 2020–2025** *(Owns the linear-wins finding; strong hook)*
3. **Uncertainty-Aware Short-Term Load Forecasting with Residual Hybridization: TFT–XGBoost under Chronological, Rolling-Origin, and Extreme-Event Evaluation** *(Hybrid + intervals as contribution; IEEE TSG-leaning)*
4. **Renewable-Aware Net-Load Forecasting for High-Solar Grids: How Weather, Calendar, and Lag Structure Interact Across 1-, 6-, and 24-Hour Horizons** *(Weather/net-load as heart; Energy / Applied Energy-leaning)*
5. **From Point Accuracy to Operational Value: Peak, Ramp, Reserve, and Carbon Implications of Load-Forecasting Choices in a 180-GW System** *(Differentiator; attempt only if §7 operational quantification is completed)*

### 2.3 Main research hypothesis (falsifiable, 5 sub-hypotheses)

- **H1 — Autocorrelation dominance (1h):** Lag-based linear/persistence matches or beats all nonlinear/attention models at 1h-ahead under equal-n leakage-free scoring.
- **H2 — Horizon dependence:** Rankings reverse with horizon; boosted/attention models gain relative skill at 6h/24h as calendar + weather dominate lag_1.
- **H3 — Weather necessity:** Temperature-derived features (CDH/HDH/heat-index) deliver larger ablation gains than wind/solar alone, concentrated on heat/cold/peak slices.
- **H4 — Selective hybridization:** OOF TFT–XGBoost residual correction improves over TFT on mid-horizons and peak regimes conditional on residual structure — not universally.
- **H5 — Calibration over accuracy:** Best point-RMSE ≠ best operational model; calibrated intervals + peak/ramp errors reorder deployment preference.

### 2.4 Research questions (RQ1–RQ6 — each maps to a results subsection)

- **RQ1 (fair benchmark):** Under identical 168h-in → {1,6,24}h-out tasks with common-timestamp scoring, how do 11 forecasters rank on MAE/RMSE/MAPE/sMAPE/R²/MASE/skill?
- **RQ2 (features):** Marginal value of Historical → +Calendar → +Weather → +Renewable → All (Ablations A–E) per horizon?
- **RQ3 (horizon):** How and why does ranking change 1h→6h→24h (lag decay vs weather dominance)?
- **RQ4 (hybrid):** When does OOF residual hybridization help, by how much, under what residual structure?
- **RQ5 (trust):** Which models are calibrated (multi-level PICP/PINAW/curves), explainable (SHAP↔attention agreement), and robust on heat/cold/peak/normal slices and rolling origins?
- **RQ6 (deployment):** What is the accuracy–latency–params–cost Pareto, and what should a 180-GW operator deploy per horizon?

### 2.5 Expected scientific contributions (preview — full framework in §4)

Dataset (weather-augmented 6-yr PJM + net-load + events) · Methodological (unified multi-horizon equal-n protocol + OOF-hybrid discipline) · Experimental (horizon-dependent ranking with proper stats) · Explainability (SHAP + attention agreement + counterfactual) · Operational (peak/ramp/calibration/cost-carbon translation). No new-algorithm claim — the contribution is rigor, honesty, and deployability.

**Direction keywords for editors:** trustworthy AI forecasting · fair benchmarking · renewable-aware forecasting · uncertainty-aware prediction · practical smart-grid deployment.

---

## 3. Research Gap + Paper Story (Old vs New)

### 3.1 Old story — what the original research was trying to claim

> "We trained 8 ML/DL models on PJM 2020–2025 and the hybrid/deep models add value."

Why it fails: the numbers say linear wins 2×; the protocol compares different tasks; the SOTA set lacks persistence/SARIMA/PatchTST/N-BEATS; weather is absent; statistics are misimplemented; artifacts are irreproducible. A reviewer stops at Table 1.

### 3.2 New story — what the Q1 paper should claim

> "Hour-ahead PJM load is so autocorrelated that persistence-like linear structure wins — and that is precisely why the field needs a leakage-free, weather-aware, multi-horizon, calibrated, operationally-grounded benchmark: we provide it, show exactly where complexity starts paying (6–24h, weather-driven peaks, high-solar net-load), when residual hybridization helps, and what to deploy."

### 3.3 Why the new story is scientifically stronger

1. **It survives its own results.** H1 absorbs the linear-wins headline instead of being refuted by it.
2. **It fixes a real methodological deficit** (leakage + horizon mismatch + missing weather + single-split/single-seed + broken stats) rather than adding another "my model wins" table.
3. **It matches 2026 editorial demand** (trustworthy AI, explainability, uncertainty, renewables, operation) at Applied Energy / IEEE TSG.
4. **It is falsifiable per horizon/slice** (H2–H5), so negative sub-results remain publishable.
5. **It yields a citable artifact** (protocol + leaderboards + weights) even for groups whose later models beat these numbers.

### 3.4 Research gap analysis

**Existing limitations (as evidenced by this repo's own state + standard field practice):**

1. Leakage-prone or undocumented evaluation (random splits, scaler refit, unshifted rollings, weather-nowcast leakage) inflates reported DL gains.
2. Horizon- and sample-inconsistent comparisons (1-step tabular vs multi-step attention, trailing-window vs full-test scoring) make rankings meaningless.
3. Weather- and event-blind load modeling (no temperature/holiday/degree-hours; no heat/cold/peak stratification) hides where errors actually cost money and reliability.
4. Point-metric-only reporting without calibration, explainability agreement, or deployment costing leaves operators unable to choose.

**How this research will address them:**

1. Certified leakage-free pipeline (shift-safe features, train-only scaling, known/unknown-at-forecast-time audit, `verification.json` per run).
2. Unified 168h→1/6/24h task with common-origin equal-n scoring enforced in code (fail-closed).
3. ERA5/NOAA weather + holidays + degree-hours + net-load + deterministic event slices, with ablation A–E isolating each group's marginal value.
4. Full trust package: multi-level calibration (50/80/90/95 + curves + Winkler via CQR), SHAP + TFT-attention agreement, peak/ramp/MASE/skill/DM, latency–params–CO₂ Pareto, and reserve/$/carbon translation.

**Gap paragraph (draft for Introduction — refine at Phase 8):**

> *Despite rapid adoption of gradient-boosted, recurrent, and transformer forecasters for short-term load forecasting, the literature lacks a leakage-certified, weather-aware, multi-horizon benchmark on contemporary high-solar data that scores all competitors on identical forecast origins with calibrated uncertainty, regime-stratified robustness, and operational cost translation. Existing comparisons mix horizons and sample sets, omit temperature and holiday structure, report point metrics on single splits and seeds with inadequate statistical testing, and rarely release reproducible artifacts — leaving grid operators without a trustworthy model-selection rule. Using six years (52,608 hours) of PJM load, wind, and solar data augmented with reanalysis weather and event stratification, this study closes that gap with a unified 168-hour-in → 1/6/24-hour-out protocol over eleven statistical, machine-learning, recurrent, attention, and residual-hybrid forecasters under chronological and rolling-origin evaluation with repeated seeds, principled hyperparameter optimization, multiplicity-corrected testing, and a calibration–explainability–deployment package.*

---

## 4. Contribution Framework (5 points — realistic, no invented algorithms)

**Contribution 1 — Dataset contribution.**
Weather-augmented 6-year PJM benchmark (52,608h RTO load + wind/solar + ERA5/NOAA temperature/humidity/wind/cloud + federal-holiday/DST/season flags + derived CDH/HDH/heat-index/solar-zenith + `net_load`/`penetration` framing + deterministic heat/cold/peak/normal event keys). Versioned v1→v2 with join-loss and clip-sensitivity accounting. *Evidence: `pjm_smart_grid_2020_2025_v2.csv`, `pjm_weather_2020_2025.csv`, `events.parquet`, `dataset_metadata_v2.json`.*

**Contribution 2 — Methodological contribution.**
Unified fair-task protocol: 168h input → {1,6,24}h direct multi-output for all 11 models; common-origin equal-n scoring enforced fail-closed in `EvaluationAlignment`; known-vs-unknown-at-forecast-time weather discipline; OOF-5 residual-hybrid procedure with residual diagnostics; conformalized (CQR) hybrid intervals. *Evidence: locked `test_origins.parquet`, protocol box in §3.4, `verification.json` per run.*

**Contribution 3 — Experimental contribution.**
Horizon-dependent ranking over 11 forecasters (persistence → SARIMA → XGB/LGBM → LSTM/GRU → TFT/PatchTST/N-BEATS → hybrid) with Optuna HPO on validation only, 3 seeds + bootstrap CIs, 4 rolling origins, feature-group ablations A–E per horizon/slice, and correct inference (Friedman + Holm-Wilcoxon + Diebold-Mariano + Nemenyi CD). *Evidence: three leaderboards + ablation Δ-table + CD diagram.*

**Contribution 4 — Explainability + uncertainty contribution.**
Fitted-model SHAP (XGB) + TFT attention/variable-importance with rank-agreement test and heat-wave counterfactual; multi-level calibration (PICP/PINAW ×4 + curves + Winkler). Replaces current phantom-SHAP and single exact-PICP artifacts. *Evidence: `shap/`, calibration figures, agreement statistic.*

**Contribution 5 — Operational impact contribution.**
Peak/top-1%/ramp/MASE/skill reporting plus accuracy–latency–params–FLOPs–CO₂ Pareto and reserve-capacity / $ / tCO₂ translation for a 180-GW system with stated assumptions and sensitivity. Turns accuracy deltas into deployment rules per horizon. *Evidence: `operational_value/` + assumption table + Pareto figure.*

---

## 5. Repository Structure (Q1-ready — what stays, what is added, what gets documented)

### 5.1 Target structure (Phase 0 design — create in Phase 1+, not now)

```
SmartGrid_AI_Research/
  README.md                  # science statement + reproduce-in-3-commands (REWRITE)
  CITATION.cff / LICENSE    # NEW — required for artifact citation
  environment.lock           # NEW — pinned env (replaces loose requirements.txt only)
  configs/
    pjm_data_v2.yaml         # NEW (v1 frozen); models.yaml (enc168, 11 models); experiment.yaml (hor/seeds/origins); hpo_spaces.yaml NEW
  data/                      # RENAME from dataset/ (keep dataset/ as read-only legacy link)
    raw/ raw_checksums.sha256 NEW
    processed/pjm_smart_grid_2020_2025_v2.csv + pjm_weather_2020_2025.csv + events.parquet + test_origins.parquet NEW
  src/                       # keep; ADD data/{weather,events}.py; models/{statistical,advanced}/; training/{hpo,stats,explain}.py; evaluation/{forecasting_metrics,uncertainty,cost}.py
  experiments/               # NEW — one folder per EXP-ID (replaces ad-hoc run_*.py outputs)
  results/q1/                # NEW canonical tree (leaderboards, ablations, calibration, shap, costs, operational_value); legacy results/ FROZEN read-only
  saved_models/q1/           # NEW — 33 weights + RUN_MANIFEST.json (currently empty — flag, not filler)
  paper/                     # NEW — manuscript/ + figures/ + tables/ + references.bib + response_to_reviewers/
  figures/ tables/           # NEW top-level (exported, numbered for paper; replaces scattered EDA dumps)
  docs/                      # NEW — PROTOCOL.md, DATA_CARD.md, MODEL_CARD.md, ABLATION_SPEC.md, STATS_PLAN.md
  tests/                     # keep + NEW task-fairness/stats/units tests
  notebooks/                 # keep empty OR delete (currently empty — decide at Phase 1; no dead folders in Q1 repos)
```

### 5.2 Which current folders remain / added / documented

- **Remain (as-is or extended):** `configs/`, `src/` (extended, never rewritten wholesale), `dataset/` (frozen legacy), `tests/` (extended), `results/` (frozen; new work under `results/q1/`), `utils/` logic merged into `src/` by Phase 2.
- **Added:** `data/` (canonical v2), `experiments/`, `results/q1/`, `saved_models/q1/`, `paper/`, `figures/`, `tables/`, `docs/`, plus per-model `src/models/{statistical,advanced}/` and `src/training/{hpo,stats,explain}.py`, `src/evaluation/{forecasting_metrics,uncertainty,cost}.py`, `src/data/{weather,events}.py`.
- **Needs documentation (each gets a `docs/` card in Phase 1):** data provenance + join-loss + clip policy (`DATA_CARD.md`); locked task definition (`PROTOCOL.md`); per-model cards incl. excluded-model rationale (`MODEL_CARD.md`); ablation/stats pre-registration (`ABLATION_SPEC.md`, `STATS_PLAN.md`); run manifest template (see §6).

---

## 6. Experiment Management Plan (professional tracking — design now, enforce from Phase 1)

### 6.1 Experiment ID format

```
EXP<NNN>_<family>_<focus>_h<H>_s<seed>
```

- `NNN`: zero-padded sequence (001…999), never reused.
- `family`: `baseline | ml | dl | adv | hybrid | ablation | uncertainty | explain | operational`.
- `focus`: short slug, e.g. `persistence`, `xgb_tuned`, `tft_h6`, `weather_ablation_C`, `cqr_calibration`, `peak_slice`.
- `h<H>`: horizon `h1 | h6 | h24 | hAll`.
- `s<seed>`: `s42 | s123 | s2025 | sDet` (deterministic).

Examples: `EXP001_baseline_persistence_hAll_sDet`, `EXP014_ml_xgb_tuned_h24_s42`, `EXP032_ablation_weather_C_h6_s42`, `EXP047_hybrid_tftxgb_oof5_h24_s123`, `EXP058_uncertainty_cqr_h24_sDet`.

### 6.2 Per-experiment record (mandatory fields — no EXP folder merges without them)

| Field | Content | Example |
|---|---|---|
| EXP-ID | per §6.1 | `EXP032_ablation_weather_C_h6_s42` |
| Dataset version | exact CSV + hash | `pjm_smart_grid_2020_2025_v2.csv (sha256:…)` + weather file |
| Features | group code + list hash | `C = B+weather (14 cols, sha:…)` |
| Model | class + code tag | `TFT(enc168/hor6, git:abc123)` |
| Hyperparameters | full dict or Optuna trial-ID | `trial 17/20, hidden 64, heads 4, lr 3e-4` |
| Random seed | 42/123/2025/det + env | `123, torch 2.14, cuda/no-cuda` |
| Splits/origins | train/val/test + origin-set hash | `cal-years 20-23/24/25 + origins v3 (n=…)` |
| Metrics | full set per horizon/slice | `MAE/RMSE/MAPE/sMAPE/R²/MASE/skill/peak/ramp/PICP…` |
| Result location | canonical paths | `experiments/EXP032_…/{metrics.json,verification.json,figures/}` + leaderboard row |
| Leakage check | pass/fail + report | `verification.json: 5/5 pass` |
| Cost | wall-time, params, CO₂e | `412s, 142k params, 0.8 kgCO₂e` |
| Status | `planned/running/done/failed/superseded` | `done` (superseded never deleted) |

Template: `experiments/_TEMPLATE/{README.md, config.yaml, metrics.json, verification.json, environment.lock}` (create Phase 1). Failed/superseded EXPs are retained with reason — deletion is forbidden (provenance).

### 6.3 Registry

Single `experiments/REGISTRY.csv` (append-only): one row per EXP-ID with §6.2 columns + `paper_table` flag (which EXPs feed which manuscript table/figure). The manuscript's every number traces to a registry row — this is what makes reviewer reproduction demands answerable.

---

## 7. Paper Requirement Checklist (gates — do not submit unless ALL checked)

**Dataset:**
- [ ] v2 CSV + weather + events + origins frozen with hashes; v1 preserved for comparability
- [ ] Join-loss %, clip counts + sensitivity, weather coverage/QA logged
- [ ] DATA_CARD.md (sources ERA5/NOAA/holidays, aggregation, RTO-only caveat, license)

**Methodology:**
- [ ] PROTOCOL.md: 168h→1/6/24h box, known/unknown table, direct strategy stated
- [ ] Leakage checklist 5/5 per run (`verification.json`); no `in_sample` hybrid in any research table
- [ ] Calendar-year splits + 4 rolling origins + equal-n proof per table

**Models:**
- [ ] 11 models per §5-rationale (+ excluded-list paragraph); MODEL_CARD.md each
- [ ] Horizon-parametrized (enc168), Optuna spaces/budget logged, best-config retrain train+val → test once
- [ ] 33 weights persisted + RUN_MANIFEST (currently 0/33 — explicit gap)

**Evaluation:**
- [ ] MAE/RMSE/MAPE/sMAPE/R²/MASE + skill + peak/top-1% + peak-timing + ramp per horizon/slice; sMAPE/unit/time bugs fixed
- [ ] 3 leaderboards mean±std + bootstrap CIs; 3 seeds stochastic; CD diagram

**Statistics:**
- [ ] Correct Friedman (df=10) + Holm-Wilcoxon (integer W, adj-p, Cliff δ) + Diebold-Mariano (Harvey); no p=0.0; STATS_PLAN.md pre-registered

**Explainability:**
- [ ] Fitted SHAP + TFT attention + rank-agreement + heat-week counterfactual; phantom features removed

**Uncertainty:**
- [ ] PICP/PINAW @50/80/90/95 + calibration curves + Winkler; CQR hybrid; exact-0.9001 artifact reproduced or retracted

**Reproducibility:**
- [ ] `environment.lock`, seeds, registry→table traceability, weights DOI, clean-machine reproduction of leaderboards within tolerance

**Figures (10–12):** system+horizons; protocol/origins; leaderboards+skill; ablation Δ; hybrid residuals; calibration; SHAP+attention; peak-week case; Pareto; slice robustness
**Tables (6–8):** dataset stats; HPO spaces; 3-panel leaderboard; ablation Δ; DM/Holm; cost/carbon assumptions
**Supplementary:** per-seed tables, DM full matrix, clip-sensitivity, verification excerpts, compute manifest, `response_to_reviewers/` template

---

## 8. Complete Upgrade Roadmap (Phase 0 → Paper)

| Phase | Objective | Files affected | Expected output (gate) |
|---|---|---|---|
| **0 — Research restructuring (NOW)** | Freeze science scope, IDs, checklists before touching code | This strategy doc only (+ prior audits) | `Q1_Phase0_Research_Strategy.md` approved; no code/results touched |
| **1 — Dataset upgrade** | ERA5/NOAA weather + holidays/DST/season + zenith/CDH/net-load + event keys; v2 freeze with hashes + QA | `configs/pjm_data_v2.yaml`; `src/data/{weather,events,loader,schema}.py`; `docs/DATA_CARD.md`; `data/processed/*v2*` | v2 CSVs + `dataset_metadata_v2.json` + coverage/join-loss/clip reports |
| **2 — Feature engineering** | Leakage-safe weather/holiday/degree/zenith/net-load builders + FEATURE_GROUPS A–E | `src/data/{features,pipeline,adapters,preprocessing}.py`; `docs/PROTOCOL.md` (features part) | Group-A–E feature sets + unit tests (shift/scaler/holiday audit) |
| **3 — Forecasting task redesign** | Lock 168→1/6/24 equal-n origins; fail-closed alignment; kill forked TFT path | `configs/experiment.yaml`; `src/{training/splits,evaluation/alignment,data/adapters}.py`; `experiments/_TEMPLATE/` | `test_origins.parquet` + equal-n enforcement test passing |
| **4 — Model upgrade** | Add persistence/seasonal/SARIMA/PatchTST/N-BEATS; horizon-parametrize all; OOF-5 hybrid | `configs/{models,hpo_spaces}.yaml`; `src/models/**`; `docs/MODEL_CARD.md` | 11 `BaseForecaster` APIs + smoke test per horizon |
| **5 — Training optimization** | Optuna (val-only) + 3-seed + rolling-origin runs; best→retrain→test-once | `src/training/{hpo,experiment_runner,trainer}.py`; `experiments/EXP*/` | Registry rows + `tuned_results_v2.json` + weights (0→33) |
| **6 — Evaluation** | Full metric set + correct stats + calibration/CQR + cost logging; fix all known bugs | `src/evaluation/*.py`; `src/training/stats.py`; `results/q1/` | 3 leaderboards + DM/Holm/CD + calibration + Pareto inputs |
| **7 — Explainability** | Fitted SHAP + attention + agreement + counterfactual + slice robustness | `src/training/explain.py`; `results/q1/{shap,calibration}/`; figures | SHAP/attention figures + agreement stat + case figure |
| **8 — Paper writing** | Manuscript + figures/tables + supplements + red-team reproduction | `paper/`; `figures/`; `tables/`; `docs/` | Submittable draft + DOI artifacts + reviewer-response template |

**Resourcing (planning figure):** ~14–16 weeks, 1 researcher + 1 GPU (≈1 GPU-week HPO/runs + 2 CPU-weeks); long pole = ERA5 access (Phase 1). **First action after Phase 0 sign-off:** Phase 1 weather acquisition + Phase 3 origin-lock in parallel (protocol must precede any training).

**Red-team submit gate (all or no-submit):** equal-n per table · no cross-model identical numbers · integer-W Holm stats · weather present · 3-seed CIs · weights reproduce leaderboards · limitations own linear-win + single-ISO + nowcast assumption.

*End of Phase 0 strategy. No implementation beyond this document is authorized under Phase 0.*
