"""
Bootstrap confidence intervals and non-inferiority testing.

Reproduces the statistical analyses reported in the paper:
  - KG-paired bootstrap (B=10,000)
  - KG-family grouped bootstrap (8 families)
  - Bottom-k average ∆MRR
  - One-sided non-inferiority t-test (δ=0.02)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    deltas: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Non-parametric bootstrap CI on the mean of `deltas`.

    Parameters
    ----------
    deltas : 1-D array of per-KG ∆MRR values (CAMP-KG − ULTRA-Real).
    B      : Number of bootstrap resamples.
    alpha  : Two-sided CI level (default 0.05 → 95% CI).

    Returns
    -------
    dict with 'mean', 'lower', 'upper' (percentile CI),
    and 'one_sided_lower' (one-sided 95% lower bound).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(deltas)
    boot_means = np.array([
        deltas[rng.integers(0, n, n)].mean() for _ in range(B)
    ])
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    one_sided = float(np.percentile(boot_means, 100 * alpha))   # one-sided lower
    return {
        "mean":             float(deltas.mean()),
        "lower":            lower,
        "upper":            upper,
        "one_sided_lower":  one_sided,
        "B":                B,
    }


def bootstrap_ci_grouped(
    deltas: np.ndarray,
    family_ids: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    KG-family grouped bootstrap to account for within-family dependence.

    Parameters
    ----------
    deltas     : 1-D array of per-KG ∆MRR, length N.
    family_ids : 1-D array of integer family labels, length N.
    """
    if rng is None:
        rng = np.random.default_rng()

    families = np.unique(family_ids)
    n_fam = len(families)

    fam_deltas = {f: deltas[family_ids == f] for f in families}

    boot_means = []
    for _ in range(B):
        sel = rng.choice(families, size=n_fam, replace=True)
        sample = np.concatenate([fam_deltas[f] for f in sel])
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    one_sided = float(np.percentile(boot_means, 100 * alpha))
    return {
        "mean":            float(deltas.mean()),
        "lower":           lower,
        "upper":           upper,
        "one_sided_lower": one_sided,
        "B":               B,
        "n_families":      n_fam,
    }


# ---------------------------------------------------------------------------
# Non-inferiority test
# ---------------------------------------------------------------------------

def noninferiority_test(
    camp_mrrs: np.ndarray,
    ultra_mrrs: np.ndarray,
    delta: float = 0.02,
    alpha: float = 0.05,
) -> Dict:
    """
    Paired one-sided t-test for non-inferiority.

    H0: μ_CAMP - μ_ULTRA ≤ -δ
    H1: μ_CAMP - μ_ULTRA > -δ  (non-inferior)

    Parameters
    ----------
    camp_mrrs  : array of per-seed mean MRR for CAMP-KG (length S).
    ultra_mrrs : array of per-seed mean MRR for ULTRA-Real (length S).
    delta      : non-inferiority margin (default 0.02).
    alpha      : significance level (default 0.05).

    Returns dict with 'mean_diff', 'p_value', 'reject_h0',
    't_stat', 'df', 'margin'.
    """
    diffs = camp_mrrs - ultra_mrrs
    n     = len(diffs)
    mean  = float(diffs.mean())
    se    = float(diffs.std(ddof=1) / np.sqrt(n))

    # t-statistic for H0: μ_diff = -delta
    t_stat = (mean + delta) / se if se > 0 else float("inf")
    df     = n - 1
    p_val  = float(1 - stats.t.cdf(t_stat, df=df))

    ci_half = stats.t.ppf(1 - alpha, df=df) * se
    lower_bound = mean - ci_half

    return {
        "mean_diff":    mean,
        "std_diff":     float(diffs.std(ddof=1)),
        "t_stat":       t_stat,
        "df":           df,
        "p_value":      p_val,
        "reject_h0":    p_val < alpha,
        "margin":       delta,
        "ci_lower":     lower_bound,
    }


# ---------------------------------------------------------------------------
# Worst-k analysis (bottom-k by ascending ULTRA-Real MRR)
# ---------------------------------------------------------------------------

def bottom_k_deltas(
    kg_names: List[str],
    ultra_mrrs: np.ndarray,
    camp_mrrs:  np.ndarray,
) -> Dict[int, float]:
    """
    For k = 1, 2, …, N: average ∆MRR over the k KGs with lowest ULTRA MRR.

    Returns dict k → mean ∆MRR.
    """
    order = np.argsort(ultra_mrrs)
    deltas = camp_mrrs - ultra_mrrs
    result = {}
    for k in range(1, len(order) + 1):
        result[k] = float(deltas[order[:k]].mean())
    return result
