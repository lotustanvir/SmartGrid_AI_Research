"""Q1 paired statistical framework (Phase 2).

Implemented ONLY for use on common-origin paired errors. Required:
  * paired error samples at common origins,
  * Wilcoxon signed-rank (paired, two-sided),
  * Friedman test for multi-model repeated comparisons,
  * Holm step-down correction for multiple pairwise comparisons,
  * bootstrap confidence intervals (mean error difference).

Diebold-Mariano is NOT implemented here: its assumptions (covariance-
stationary loss differentials, non-nested models, large-T asymptotics)
are not verified for this protocol. A ``diebold_mariano`` stub raises
with justification required instead of silently returning p-values.

Legacy artifact ``results/research_analysis/statistical_tests_v2.json``
is QUARANTINED (see results/research_analysis/UNVERIFIED_LEGACY.md) and
must NOT be treated as evidence. This module never overwrites it.

Uses scipy only (already installed); no new dependencies.
"""

from __future__ import annotations

import numpy as np


def _as_paired(a, b) -> tuple[np.ndarray, np.ndarray]:
    ea = np.asarray(a, dtype=float).ravel()
    eb = np.asarray(b, dtype=float).ravel()
    if ea.shape != eb.shape:
        raise ValueError(f"Paired shape mismatch: {ea.shape} vs {eb.shape}")
    if ea.size < 10:
        raise ValueError("Need >= 10 paired samples.")
    if np.isnan(ea).any() or np.isnan(eb).any() or np.isinf(ea).any() or np.isinf(eb).any():
        raise ValueError("NaN/Inf in paired errors.")
    return ea, eb


def wilcoxon_signed_rank(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank on paired errors (scipy)."""
    from scipy.stats import wilcoxon

    ea, eb = _as_paired(a, b)
    try:
        res = wilcoxon(ea, eb, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        return {"statistic": 0.0, "p_value": 1.0, "note": f"degenerate: {e}"}
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def friedman_test(*groups) -> dict:
    """Friedman chi-square across >=3 models (repeated measures)."""
    from scipy.stats import friedmanchisquare

    if len(groups) < 3:
        raise ValueError("Friedman needs >= 3 groups.")
    arrs = [np.asarray(g, dtype=float).ravel() for g in groups]
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError("Friedman groups must share sample count (common origins).")
    stat, p = friedmanchisquare(*arrs)
    return {"statistic": float(stat), "p_value": float(p), "n_models": len(arrs), "n": int(n)}


def holm_correction(p_values: list[float], alpha: float = 0.05) -> dict:
    """Holm step-down family-wise correction (fail-closed on bad input)."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("Empty p-value list.")
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be in [0,1].")
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, float(p[idx] * (m - rank)))
    # Enforce monotonicity (step-down).
    sorted_adj = np.maximum.accumulate(adj[order])
    for rank, idx in enumerate(order):
        adj[idx] = float(sorted_adj[rank])
    return {
        "adjusted": [float(v) for v in adj],
        "reject": [bool(v < alpha) for v in adj],
        "alpha": float(alpha),
    }


def bootstrap_mean_diff_ci(
    a, b, n_boot: int = 2000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Bootstrap CI for mean(a-b) with paired resampling."""
    ea, eb = _as_paired(a, b)
    if not (0 < ci < 1):
        raise ValueError("ci must be in (0,1).")
    rng = np.random.default_rng(seed)
    diff = ea - eb
    boots = np.array(
        [rng.choice(diff, size=diff.size, replace=True).mean() for _ in range(n_boot)]
    )
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, lo_q)),
        "ci_high": float(np.percentile(boots, hi_q)),
        "ci": float(ci),
        "n_boot": int(n_boot),
    }


def diebold_mariano(*args, **kwargs) -> dict:  # noqa: ARG001
    """NOT implemented: raises until assumptions are justified in protocol."""
    raise NotImplementedError(
        "Diebold-Mariano is quarantined: covariance-stationary loss "
        "differentials, non-nested models, and HAC bandwidth must be "
        "justified in the protocol before use. See docs/STATS_PLAN."
    )
