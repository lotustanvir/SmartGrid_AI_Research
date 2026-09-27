# QUARANTINED — UNVERIFIED LEGACY STATISTICAL ARTIFACTS (Phase 2)

The following files MUST NOT be treated as Q1 evidence:

* `statistical_tests_v2.json` (fractional Wilcoxon W, p=0.0 everywhere,
  byte-identical duplicates, no Holm/effect/DM — mathematically invalid)
* `final_validation_report.json:statistical_tests` (identical numbers)
* `uncertainty_results.json` (single PICP 0.9001 placeholder)
* `tuned_results.*`, `ablation_v2.*`, `hybrid_analysis.json`,
  `contribution_analysis.json`, `verification_report.json`

Reasons: obsolete protocols, single-horizon/mismatched-n scoring,
synthetic verification, suspicious estimates, possible leakage
(run_phase_6b.py test-inclusive TFT/Hybrid path).

Phase 2 does NOT overwrite these files. The Q1 framework is
`src/evaluation/stats.py` (paired Wilcoxon/Friedman/Holm/bootstrap;
Diebold-Mariano quarantined pending justification) and must run on
common-origin paired errors only.
